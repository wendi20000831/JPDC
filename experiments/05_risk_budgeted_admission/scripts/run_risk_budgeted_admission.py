#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_LEGACY_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "04_job_risk_graph"
    / "scripts"
    / "run_learned_risk_baseline.py"
)
LEGACY_SCRIPT = Path(
    os.environ.get("SAFEHARVEST_LEARNED_SCRIPT", str(DEFAULT_LEGACY_SCRIPT))
)


def load_legacy_module() -> Any:
    spec = importlib.util.spec_from_file_location("safeharvest_learned", LEGACY_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load legacy module from {LEGACY_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


legacy = load_legacy_module()


@dataclass(frozen=True)
class BudgetPolicy:
    name: str
    average_risk_budget: float


@dataclass(frozen=True)
class OrderingPolicy:
    name: str


DEFAULT_BUDGETS = [
    BudgetPolicy("B005_avg_risk", 0.005),
    BudgetPolicy("B010_avg_risk", 0.010),
    BudgetPolicy("B020_avg_risk", 0.020),
    BudgetPolicy("B030_avg_risk", 0.030),
    BudgetPolicy("B050_avg_risk", 0.050),
]

DEFAULT_ORDERINGS = [
    OrderingPolicy("risk_first"),
    OrderingPolicy("utility_per_risk"),
    OrderingPolicy("balanced"),
]

DURATION_GROUP_COLS = [
    "candidate_class",
    "priority_bucket",
    "instance_bucket",
    "cpu_bucket",
    "memory_bucket",
]


def parse_float_list(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_str_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def temporal_train_calibration_split(
    train: pd.DataFrame, calibration_fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.05 <= calibration_fraction <= 0.50:
        raise ValueError("--calibration-fraction must be between 0.05 and 0.50")
    ordered = train.sort_values("schedule_win5", kind="mergesort")
    cutoff_idx = max(1, int(len(ordered) * (1.0 - calibration_fraction)))
    model_train = ordered.iloc[:cutoff_idx].copy()
    calibration = ordered.iloc[cutoff_idx:].copy()
    if calibration.empty or model_train.empty:
        raise ValueError("temporal calibration split produced an empty partition")
    return model_train, calibration


def wilson_upper_bound(successes: int, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return 1.0
    phat = successes / n
    denom = 1.0 + z * z / n
    center = phat + z * z / (2.0 * n)
    margin = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n)
    return min(1.0, (center + margin) / denom)


def make_prediction_edges(pred: pd.Series, bins: int) -> np.ndarray:
    values = pred.to_numpy(dtype=float)
    if len(values) == 0:
        return np.array([-np.inf, np.inf])
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(values, quantiles))
    if len(edges) < 3:
        edges = np.array([float(np.min(values)), float(np.max(values))])
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def calibrate_risk_bins(
    calibration: pd.DataFrame, bins: int
) -> tuple[np.ndarray, pd.DataFrame]:
    edges = make_prediction_edges(calibration["predicted_bad_rate"], bins)
    tmp = calibration.copy()
    tmp["risk_bin"] = pd.cut(
        tmp["predicted_bad_rate"],
        bins=edges,
        labels=False,
        include_lowest=True,
    )
    rows = []
    for bin_id, group in tmp.groupby("risk_bin", observed=True):
        n = int(len(group))
        bad = int(group["observed_bad_collection"].sum())
        mean_pred = float(group["predicted_bad_rate"].mean()) if n else 0.0
        observed = bad / n if n else 0.0
        upper = wilson_upper_bound(bad, n)
        rows.append(
            {
                "risk_bin": int(bin_id),
                "pred_lower": float(edges[int(bin_id)]),
                "pred_upper": float(edges[int(bin_id) + 1]),
                "calibration_jobs": n,
                "observed_bad_collections": bad,
                "observed_bad_collection_rate": observed,
                "mean_predicted_bad_rate": mean_pred,
                "wilson_upper_bad_rate": upper,
                "risk_charge": max(mean_pred, upper),
            }
        )
    table = pd.DataFrame(rows).sort_values("risk_bin").reset_index(drop=True)
    return edges, table


def attach_risk_charge(
    df: pd.DataFrame, edges: np.ndarray, calibration_table: pd.DataFrame
) -> pd.DataFrame:
    out = df.copy()
    out["risk_bin"] = pd.cut(
        out["predicted_bad_rate"],
        bins=edges,
        labels=False,
        include_lowest=True,
    )
    charge_map = calibration_table.set_index("risk_bin")["risk_charge"].to_dict()
    fallback = float(calibration_table["risk_charge"].max())
    out["calibrated_risk_charge"] = (
        out["risk_bin"].map(charge_map).fillna(fallback).clip(lower=0.0, upper=1.0)
    )
    out["utility_per_risk"] = out["cpu_window_demand"] / (
        out["calibrated_risk_charge"] + 1e-6
    )
    return out


def duration_quantile_from_mode(duration_mode: str) -> float | None:
    if duration_mode == "observed":
        return None
    if duration_mode == "calibrated_p90":
        return 0.90
    if duration_mode == "calibrated_p95":
        return 0.95
    raise ValueError(f"unknown duration mode: {duration_mode}")


def attach_duration_charge(
    test: pd.DataFrame,
    history: pd.DataFrame,
    duration_mode: str,
    min_group_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = test.copy()
    quantile = duration_quantile_from_mode(duration_mode)
    if quantile is None:
        out["placement_runtime_windows"] = out["runtime_windows"].clip(lower=1).astype(int)
        out["duration_charge_source"] = "observed"
        out["duration_charge_group_count"] = pd.NA
        out["duration_miss"] = 0
        out["duration_overrun_windows"] = 0
        return out, pd.DataFrame()

    if min_group_size < 1:
        raise ValueError("--duration-min-group-size must be positive")

    fallback = int(math.ceil(history["runtime_windows"].clip(lower=1).quantile(quantile)))
    grouped = (
        history.groupby(DURATION_GROUP_COLS, observed=True)["runtime_windows"]
        .agg(
            duration_group_count="size",
            placement_runtime_windows=lambda s: int(
                math.ceil(s.clip(lower=1).quantile(quantile))
            ),
        )
        .reset_index()
    )
    grouped["duration_mode"] = duration_mode
    grouped.loc[
        grouped["duration_group_count"] < min_group_size, "placement_runtime_windows"
    ] = pd.NA

    out = out.merge(grouped, on=DURATION_GROUP_COLS, how="left")
    out["duration_charge_source"] = np.where(
        out["placement_runtime_windows"].notna(), "group_quantile", "global_quantile"
    )
    out["placement_runtime_windows"] = (
        out["placement_runtime_windows"].fillna(fallback).clip(lower=1).astype(int)
    )
    out["duration_charge_group_count"] = out["duration_group_count"]
    out["duration_miss"] = (
        out["runtime_windows"].astype(int) > out["placement_runtime_windows"].astype(int)
    ).astype(int)
    out["duration_overrun_windows"] = (
        out["runtime_windows"].astype(int) - out["placement_runtime_windows"].astype(int)
    ).clip(lower=0)

    duration_table = grouped.copy()
    duration_table["global_fallback_runtime_windows"] = fallback
    duration_table["duration_quantile"] = quantile
    duration_table["duration_min_group_size"] = min_group_size
    return out, duration_table


def sort_for_ordering(df: pd.DataFrame, ordering: str) -> pd.DataFrame:
    if ordering == "risk_first":
        return df.sort_values(
            [
                "schedule_win5",
                "calibrated_risk_charge",
                "class_priority",
                "cpu_window_demand",
            ],
            ascending=[True, True, True, False],
            kind="mergesort",
        )
    if ordering == "utility_per_risk":
        return df.sort_values(
            [
                "schedule_win5",
                "utility_per_risk",
                "calibrated_risk_charge",
                "class_priority",
            ],
            ascending=[True, False, True, True],
            kind="mergesort",
        )
    if ordering == "balanced":
        tmp = df.copy()
        tmp["balanced_score"] = tmp["utility_per_risk"] / (
            1.0 + 10.0 * tmp["calibrated_risk_charge"]
        )
        return tmp.sort_values(
            [
                "schedule_win5",
                "balanced_score",
                "calibrated_risk_charge",
                "class_priority",
            ],
            ascending=[True, False, True, True],
            kind="mergesort",
        )
    raise ValueError(f"unknown ordering policy: {ordering}")


def run_budgeted_replay(
    test: pd.DataFrame,
    intervals: pd.DataFrame,
    budgets: list[BudgetPolicy],
    orderings: list[OrderingPolicy],
    per_job_risk_cap: float,
    write_samples: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    min_win = int(test["schedule_win5"].min())
    max_win = int(test["schedule_win5"].max())
    active_index = legacy.build_active_index(intervals, min_win, max_win)
    starts = intervals["start_win5"].astype(int).to_list()
    ends = intervals["end_win5"].astype(int).to_list()
    caps = intervals["capacity_cpu"].astype(float).to_list()
    mem_caps = intervals["capacity_memory"].astype(float).to_list()

    summary_rows = []
    reason_rows = []
    placement_rows = []

    for ordering in orderings:
        ordered = sort_for_ordering(test, ordering.name).reset_index(drop=True)
        for budget in budgets:
            used_cpu: dict[int, list[float]] = {}
            used_mem: dict[int, list[float]] = {}
            reasons: Counter = Counter()
            machines: set[int] = set()
            day_risk_spent: dict[int, float] = defaultdict(float)
            day_admitted: dict[int, int] = defaultdict(int)

            placed_jobs = 0
            placed_cpu = 0.0
            placed_mem = 0.0
            placed_bad_instances = 0
            placed_bad_collections = 0
            placed_duration_misses = 0
            placed_duration_overrun = 0
            placed_runtime_reserved = 0
            placed_runtime_observed = 0
            risk_spent = 0.0
            risk_cap_rejects = 0
            budget_rejects = 0

            for job in ordered.itertuples(index=False):
                risk_charge = float(job.calibrated_risk_charge)
                if risk_charge > per_job_risk_cap:
                    reasons["per_job_risk_cap"] += 1
                    risk_cap_rejects += 1
                    continue

                day = int((int(job.schedule_win5) - min_win) // 288)
                next_count = day_admitted[day] + 1
                next_spent = day_risk_spent[day] + risk_charge
                if next_spent > budget.average_risk_budget * next_count:
                    reasons["risk_budget_exhausted"] += 1
                    budget_rejects += 1
                    continue

                actual_runtime = max(1, int(job.runtime_windows))
                runtime = max(1, int(job.placement_runtime_windows))
                placed, interval_id, reason = legacy.try_place_exact(
                    int(job.schedule_win5),
                    runtime,
                    float(job.total_cpu_request),
                    float(job.total_memory_request),
                    active_index,
                    starts,
                    ends,
                    caps,
                    mem_caps,
                    used_cpu,
                    used_mem,
                )

                if not placed:
                    reasons[reason] += 1
                    continue

                placed_jobs += 1
                placed_cpu += float(job.cpu_window_demand)
                placed_mem += float(job.memory_window_demand)
                placed_bad_instances += int(job.bad_terminal_instances)
                placed_bad_collections += int(job.observed_bad_collection)
                placed_duration_misses += int(job.duration_miss)
                placed_duration_overrun += int(job.duration_overrun_windows)
                placed_runtime_reserved += int(runtime)
                placed_runtime_observed += int(actual_runtime)
                risk_spent += risk_charge
                day_risk_spent[day] = next_spent
                day_admitted[day] = next_count
                if interval_id is not None:
                    machines.add(interval_id)
                if write_samples and len(placement_rows) < 10000:
                    placement_rows.append(
                        {
                            "ordering_policy": ordering.name,
                            "budget_policy": budget.name,
                            "collection_id": int(job.collection_id),
                            "schedule_win5": int(job.schedule_win5),
                            "observed_runtime_windows": int(actual_runtime),
                            "placement_runtime_windows": int(runtime),
                            "duration_miss": int(job.duration_miss),
                            "predicted_bad_rate": float(job.predicted_bad_rate),
                            "calibrated_risk_charge": risk_charge,
                            "cpu_window_demand": float(job.cpu_window_demand),
                            "memory_window_demand": float(job.memory_window_demand),
                            "bad_terminal_instances": int(job.bad_terminal_instances),
                            "observed_bad_collection": int(job.observed_bad_collection),
                            "interval_id": int(interval_id) if interval_id is not None else -1,
                        }
                    )

            demanded_cpu = float(ordered["cpu_window_demand"].sum())
            demanded_mem = float(ordered["memory_window_demand"].sum())
            summary_rows.append(
                {
                    "ordering_policy": ordering.name,
                    "budget_policy": budget.name,
                    "average_risk_budget": budget.average_risk_budget,
                    "per_job_risk_cap": per_job_risk_cap,
                    "candidate_jobs": int(len(ordered)),
                    "risk_cap_rejects": int(risk_cap_rejects),
                    "risk_budget_rejects": int(budget_rejects),
                    "placed_jobs": int(placed_jobs),
                    "placement_rate_over_candidates": placed_jobs / len(ordered)
                    if len(ordered)
                    else 0.0,
                    "demanded_cpu_window": demanded_cpu,
                    "placed_cpu_window": placed_cpu,
                    "placed_cpu_window_share": placed_cpu / demanded_cpu
                    if demanded_cpu
                    else 0.0,
                    "demanded_memory_window": demanded_mem,
                    "placed_memory_window": placed_mem,
                    "placed_memory_window_share": placed_mem / demanded_mem
                    if demanded_mem
                    else 0.0,
                    "risk_spent": risk_spent,
                    "expected_bad_collection_share": risk_spent / placed_jobs
                    if placed_jobs
                    else 0.0,
                    "realized_bad_collections": int(placed_bad_collections),
                    "realized_bad_collection_share": placed_bad_collections / placed_jobs
                    if placed_jobs
                    else 0.0,
                    "placed_bad_terminal_instances": int(placed_bad_instances),
                    "duration_miss_jobs": int(placed_duration_misses),
                    "duration_miss_share": placed_duration_misses / placed_jobs
                    if placed_jobs
                    else 0.0,
                    "duration_overrun_windows": int(placed_duration_overrun),
                    "mean_observed_runtime_windows": placed_runtime_observed / placed_jobs
                    if placed_jobs
                    else 0.0,
                    "mean_reserved_runtime_windows": placed_runtime_reserved / placed_jobs
                    if placed_jobs
                    else 0.0,
                    "machines_used": int(len(machines)),
                    "active_days": int(len(day_admitted)),
                }
            )
            for reason, count in reasons.items():
                reason_rows.append(
                    {
                        "ordering_policy": ordering.name,
                        "budget_policy": budget.name,
                        "average_risk_budget": budget.average_risk_budget,
                        "reason": reason,
                        "jobs": int(count),
                    }
                )

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(reason_rows),
        pd.DataFrame(placement_rows),
    )


def run(
    expanded_candidates: Path,
    intervals_path: Path,
    output_dir: Path,
    risk_budgets: list[float],
    ordering_names: list[str],
    calibration_fraction: float,
    calibration_bins: int,
    per_job_risk_cap: float,
    drop_graph_future: bool,
    duration_mode: str,
    duration_min_group_size: int,
    write_samples: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    train, test = legacy.load_and_prepare(expanded_candidates)
    model_train, calibration = temporal_train_calibration_split(
        train, calibration_fraction
    )

    pipe = legacy.fit_model(model_train, drop_graph_future=drop_graph_future)
    categorical, numeric = legacy.feature_lists(drop_graph_future)
    features = categorical + numeric

    calibration["predicted_bad_rate"] = pipe.predict_proba(calibration[features])[:, 1]
    test["predicted_bad_rate"] = pipe.predict_proba(test[features])[:, 1]

    edges, calibration_table = calibrate_risk_bins(calibration, calibration_bins)
    calibration_table.to_csv(output_dir / "risk_budget_calibration_bins.csv", index=False)

    charged_test = attach_risk_charge(test, edges, calibration_table)
    charged_test, duration_table = attach_duration_charge(
        charged_test,
        train,
        duration_mode=duration_mode,
        min_group_size=duration_min_group_size,
    )
    if not duration_table.empty:
        duration_table.to_csv(output_dir / "duration_charge_table.csv", index=False)
    intervals = legacy.load_intervals(intervals_path)

    budgets = [BudgetPolicy(f"B{int(b * 1000):03d}_avg_risk", b) for b in risk_budgets]
    orderings = [OrderingPolicy(name) for name in ordering_names]
    summary, reasons, placements = run_budgeted_replay(
        charged_test,
        intervals,
        budgets,
        orderings,
        per_job_risk_cap=per_job_risk_cap,
        write_samples=write_samples,
    )
    summary.to_csv(output_dir / "risk_budgeted_admission_summary.csv", index=False)
    reasons.to_csv(output_dir / "risk_budgeted_rejection_reasons.csv", index=False)
    if write_samples:
        placements.to_csv(output_dir / "risk_budgeted_placements_sample.csv", index=False)

    metadata = pd.DataFrame(
        [
            {
                "expanded_candidates": str(expanded_candidates),
                "intervals": str(intervals_path),
                "train_rows": int(len(train)),
                "model_train_rows": int(len(model_train)),
                "calibration_rows": int(len(calibration)),
                "test_rows": int(len(test)),
                "drop_graph_future": bool(drop_graph_future),
                "calibration_fraction": calibration_fraction,
                "calibration_bins": calibration_bins,
                "per_job_risk_cap": per_job_risk_cap,
                "duration_mode": duration_mode,
                "duration_min_group_size": duration_min_group_size,
            }
        ]
    )
    metadata.to_csv(output_dir / "risk_budgeted_run_metadata.csv", index=False)
    print(f"wrote risk-budgeted admission outputs to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expanded-candidates", type=Path, required=True)
    parser.add_argument("--intervals", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--risk-budgets",
        default="0.005,0.010,0.020,0.030,0.050",
        help="Comma-separated average admitted-risk budgets.",
    )
    parser.add_argument(
        "--orderings",
        default="risk_first,utility_per_risk,balanced",
        help="Comma-separated ordering policies.",
    )
    parser.add_argument("--calibration-fraction", type=float, default=0.20)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--per-job-risk-cap", type=float, default=0.10)
    parser.add_argument(
        "--duration-mode",
        choices=["observed", "calibrated_p90", "calibrated_p95"],
        default="observed",
        help=(
            "Use observed test runtime for offline replay, or reserve a "
            "pre-test calibrated duration quantile for online-feasible replay."
        ),
    )
    parser.add_argument("--duration-min-group-size", type=int, default=50)
    parser.add_argument(
        "--no-graph-future-features",
        action="store_true",
        help="Drop out-degree-derived graph features and keep admission-time features.",
    )
    parser.add_argument(
        "--write-samples",
        action="store_true",
        help="Write up to 10k placement samples for diagnostics.",
    )
    args = parser.parse_args()

    run(
        expanded_candidates=args.expanded_candidates,
        intervals_path=args.intervals,
        output_dir=args.output,
        risk_budgets=parse_float_list(args.risk_budgets),
        ordering_names=parse_str_list(args.orderings),
        calibration_fraction=args.calibration_fraction,
        calibration_bins=args.calibration_bins,
        per_job_risk_cap=args.per_job_risk_cap,
        drop_graph_future=args.no_graph_future_features,
        duration_mode=args.duration_mode,
        duration_min_group_size=args.duration_min_group_size,
        write_samples=args.write_samples,
    )


if __name__ == "__main__":
    main()
