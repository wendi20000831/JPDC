#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TEST_START_WIN5 = 6480
TEST_END_WIN5 = 8784

LOW_CLASS = {
    "batch_scheduler",
    "best_effort_queueable",
    "low_priority_flexible",
    "medium_flexibility",
}

CLASS_PRIORITY = {
    "best_effort_queueable": 0,
    "batch_scheduler": 1,
    "low_priority_flexible": 2,
    "medium_flexibility": 3,
}


@dataclass(frozen=True)
class LearnedFrontier:
    name: str
    risk_threshold: float


FRONTIERS = [
    LearnedFrontier("L0_learned_p005", 0.005),
    LearnedFrontier("L1_learned_p010", 0.010),
    LearnedFrontier("L2_learned_p030", 0.030),
    LearnedFrontier("L3_learned_p050", 0.050),
    LearnedFrontier("L4_learned_p100", 0.100),
]


def bucketize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["priority_bucket"] = pd.cut(
        out["priority"].fillna(-1),
        bins=[-np.inf, 99, 199, 299, np.inf],
        labels=["p000_099", "p100_199", "p200_299", "p300_plus"],
    ).astype(str)
    out["instance_bucket"] = pd.cut(
        out["instances"].fillna(0),
        bins=[-np.inf, 1, 5, 20, np.inf],
        labels=["i001", "i002_005", "i006_020", "i021_plus"],
    ).astype(str)
    out["cpu_bucket"] = pd.cut(
        out["total_cpu_request"].fillna(0.0),
        bins=[-np.inf, 0.10, 0.25, 0.50, np.inf],
        labels=["cpu_000_010", "cpu_010_025", "cpu_025_050", "cpu_050_plus"],
    ).astype(str)
    out["memory_bucket"] = pd.cut(
        out["total_memory_request"].fillna(0.0),
        bins=[-np.inf, 0.10, 0.25, 0.50, np.inf],
        labels=["mem_000_010", "mem_010_025", "mem_025_050", "mem_050_plus"],
    ).astype(str)
    out["max_cpu_bucket"] = pd.cut(
        out["max_instance_cpu_request"].fillna(0.0),
        bins=[-np.inf, 0.05, 0.10, 0.25, np.inf],
        labels=["mcpu_000_005", "mcpu_005_010", "mcpu_010_025", "mcpu_025_plus"],
    ).astype(str)
    out["max_memory_bucket"] = pd.cut(
        out["max_instance_memory_request"].fillna(0.0),
        bins=[-np.inf, 0.05, 0.10, 0.25, np.inf],
        labels=["mmem_000_005", "mmem_005_010", "mmem_010_025", "mmem_025_plus"],
    ).astype(str)
    degree = out["incoming_dependency_edges"].fillna(0) + out[
        "outgoing_dependent_edges"
    ].fillna(0)
    out["degree_bucket"] = pd.cut(
        degree,
        bins=[-np.inf, 0, 2, 10, np.inf],
        labels=["deg_0", "deg_1_2", "deg_3_10", "deg_11_plus"],
    ).astype(str)
    return out


def load_and_prepare(expanded_candidates: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        "collection_id",
        "candidate_class",
        "priority",
        "scheduling_class",
        "scheduler",
        "collection_terminal_outcome",
        "dependency_bucket",
        "dependency_graph_role",
        "incoming_dependency_edges",
        "outgoing_dependent_edges",
        "schedule_win5",
        "runtime_windows",
        "total_cpu_request",
        "total_memory_request",
        "max_instance_cpu_request",
        "max_instance_memory_request",
        "p50_instance_cpu_request",
        "p90_instance_cpu_request",
        "p50_instance_memory_request",
        "p90_instance_memory_request",
        "instances",
        "bad_terminal_instances",
        "cpu_window_demand",
        "memory_window_demand",
    ]
    df = pd.read_parquet(expanded_candidates, columns=columns)
    valid = (
        df["schedule_win5"].notna()
        & df["runtime_windows"].notna()
        & (df["runtime_windows"] > 0)
        & df["total_cpu_request"].notna()
        & (df["total_cpu_request"] > 0)
        & df["total_memory_request"].notna()
        & df["candidate_class"].isin(LOW_CLASS)
    )
    df = bucketize(df[valid].copy())
    df["bad_terminal_instances"] = df["bad_terminal_instances"].fillna(0).astype(int)
    df["observed_bad_collection"] = (
        df["collection_terminal_outcome"].isin(["EVICT", "FAIL", "KILL", "LOST"])
        | (df["bad_terminal_instances"] > 0)
    ).astype(int)
    df["runtime_windows"] = df["runtime_windows"].clip(lower=1).astype(int)
    df["instances"] = df["instances"].fillna(0).astype(int)
    df["class_priority"] = df["candidate_class"].map(CLASS_PRIORITY).fillna(99)

    train = df[df["schedule_win5"] < TEST_START_WIN5].copy()
    test = df[
        (df["schedule_win5"] >= TEST_START_WIN5)
        & (df["schedule_win5"] <= TEST_END_WIN5)
    ].copy()
    return train, test


def feature_lists(drop_graph_future: bool = False) -> tuple[list[str], list[str]]:
    """Return (categorical, numeric) feature lists.

    When drop_graph_future is True, removes features that depend on the
    complete observed dependency graph (out-degree), keeping only
    admission-time-visible signals (in-degree, parent edges).
    """
    categorical = [
        "candidate_class",
        "scheduling_class",
        "scheduler",
        "dependency_bucket",
        "dependency_graph_role",
        "priority_bucket",
        "instance_bucket",
        "cpu_bucket",
        "memory_bucket",
        "max_cpu_bucket",
        "max_memory_bucket",
        "degree_bucket",
    ]
    numeric = [
        "priority",
        "incoming_dependency_edges",
        "outgoing_dependent_edges",
        "instances",
        "total_cpu_request",
        "total_memory_request",
        "max_instance_cpu_request",
        "max_instance_memory_request",
        "p50_instance_cpu_request",
        "p90_instance_cpu_request",
        "p50_instance_memory_request",
        "p90_instance_memory_request",
    ]
    if drop_graph_future:
        # dependency_bucket and dependency_graph_role both derive from the
        # full observed graph (in+out degree); degree_bucket is in+out;
        # outgoing_dependent_edges is the raw out-degree count.
        future_categorical = {"dependency_bucket", "dependency_graph_role", "degree_bucket"}
        future_numeric = {"outgoing_dependent_edges"}
        categorical = [c for c in categorical if c not in future_categorical]
        numeric = [n for n in numeric if n not in future_numeric]
    return categorical, numeric


def fit_model(train: pd.DataFrame, drop_graph_future: bool = False) -> Pipeline:
    categorical, numeric = feature_lists(drop_graph_future)

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=True)

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", encoder, categorical),
            ("num", StandardScaler(), numeric),
        ],
        remainder="drop",
    )
    model = LogisticRegression(
        solver="saga",
        max_iter=600,
        C=1.0,
        n_jobs=-1,
        random_state=17,
    )
    pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
    pipe.fit(train[categorical + numeric], train["observed_bad_collection"])
    return pipe


def load_intervals(path: Path) -> pd.DataFrame:
    intervals = pd.read_csv(path)
    intervals = intervals.reset_index(drop=True)
    intervals["interval_id"] = intervals.index
    intervals["capacity_cpu"] = intervals["cpu_cap_mean"] * intervals["min_room_frac"]
    intervals["capacity_memory"] = intervals["min_memory_room_max_usage"].clip(lower=0.0)
    return intervals


def build_active_index(intervals: pd.DataFrame, min_win: int, max_win: int) -> dict[int, list[int]]:
    active: dict[int, list[int]] = {win: [] for win in range(min_win, max_win + 1)}
    starts = intervals["start_win5"].astype(int).to_list()
    ends = intervals["end_win5"].astype(int).to_list()
    caps = intervals["capacity_cpu"].astype(float).to_list()
    durations = intervals["duration_windows"].astype(int).to_list()
    for interval_id, (start, end, cap) in enumerate(zip(starts, ends, caps)):
        if cap <= 0:
            continue
        lo = max(start, min_win)
        hi = min(end, max_win)
        for win in range(lo, hi + 1):
            active[win].append(interval_id)
    for interval_ids in active.values():
        interval_ids.sort(key=lambda idx: (caps[idx], durations[idx], ends[idx]))
    return active


def try_place_exact(
    start: int,
    runtime: int,
    cpu: float,
    memory: float,
    active_index: dict[int, list[int]],
    interval_starts: list[int],
    interval_ends: list[int],
    interval_caps: list[float],
    interval_memory_caps: list[float],
    used_cpu_by_interval: dict[int, list[float]],
    used_memory_by_interval: dict[int, list[float]],
) -> tuple[bool, int | None, str]:
    end = start + runtime - 1
    interval_ids = active_index.get(start)
    if not interval_ids:
        return False, None, "no_active_interval"

    saw_duration_fit = False
    saw_cpu_fit = False
    saw_memory_fit = False
    saw_cpu_exhausted = False
    saw_memory_exhausted = False

    for interval_id in interval_ids:
        if interval_ends[interval_id] < end:
            continue
        saw_duration_fit = True

        cap = interval_caps[interval_id]
        if cpu > cap:
            continue
        saw_cpu_fit = True

        mem_cap = interval_memory_caps[interval_id]
        if memory > mem_cap:
            continue
        saw_memory_fit = True

        offset = start - interval_starts[interval_id]
        hi = offset + runtime
        used_cpu = used_cpu_by_interval.get(interval_id)
        if used_cpu is None:
            used_cpu = [0.0] * (interval_ends[interval_id] - interval_starts[interval_id] + 1)
            used_memory = [0.0] * len(used_cpu)
            used_cpu_by_interval[interval_id] = used_cpu
            used_memory_by_interval[interval_id] = used_memory
        else:
            used_memory = used_memory_by_interval[interval_id]

        fits = True
        cpu_fits = True
        memory_fits = True
        for pos in range(offset, hi):
            if used_cpu[pos] + cpu > cap:
                fits = False
                cpu_fits = False
                break
            if used_memory[pos] + memory > mem_cap:
                fits = False
                memory_fits = False
                break
        if not fits:
            if not cpu_fits:
                saw_cpu_exhausted = True
            elif not memory_fits:
                saw_memory_exhausted = True
            continue

        for pos in range(offset, hi):
            used_cpu[pos] += cpu
            used_memory[pos] += memory
        return True, interval_id, "placed"

    if not saw_duration_fit:
        return False, None, "no_duration_fit"
    if not saw_cpu_fit:
        return False, None, "cpu_too_large_for_machine_gap"
    if not saw_memory_fit:
        return False, None, "memory_too_large_for_machine_gap"
    if saw_memory_exhausted:
        return False, None, "machine_memory_exhausted"
    if saw_cpu_exhausted:
        return False, None, "machine_cpu_exhausted"
    return False, None, "no_feasible_interval"


def run_matching(test: pd.DataFrame, intervals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    min_win = int(test["schedule_win5"].min())
    max_win = int(test["schedule_win5"].max())
    active_index = build_active_index(intervals, min_win, max_win)
    starts = intervals["start_win5"].astype(int).to_list()
    ends = intervals["end_win5"].astype(int).to_list()
    caps = intervals["capacity_cpu"].astype(float).to_list()
    mem_caps = intervals["capacity_memory"].astype(float).to_list()

    summary_rows = []
    reason_rows = []
    for frontier in FRONTIERS:
        eligible = test[test["predicted_bad_rate"] <= frontier.risk_threshold].copy()
        eligible = eligible.sort_values(
            [
                "schedule_win5",
                "predicted_bad_rate",
                "class_priority",
                "cpu_window_demand",
            ],
            kind="mergesort",
        ).reset_index(drop=True)

        used_cpu: dict[int, list[float]] = {}
        used_mem: dict[int, list[float]] = {}
        reasons: Counter = Counter()
        machines: set[int] = set()
        placed_jobs = 0
        placed_cpu = 0.0
        placed_mem = 0.0
        placed_bad = 0

        for job in eligible.itertuples(index=False):
            runtime = max(1, int(job.runtime_windows))
            placed, interval_id, reason = try_place_exact(
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
            if placed:
                placed_jobs += 1
                placed_cpu += float(job.cpu_window_demand)
                placed_mem += float(job.memory_window_demand)
                placed_bad += int(job.bad_terminal_instances)
                if interval_id is not None:
                    machines.add(interval_id)
            else:
                reasons[reason] += 1

        demanded_cpu = float(eligible["cpu_window_demand"].sum())
        demanded_mem = float(eligible["memory_window_demand"].sum())
        eligible_bad = int(eligible["bad_terminal_instances"].sum())
        summary_rows.append(
            {
                "frontier_policy": frontier.name,
                "risk_threshold": frontier.risk_threshold,
                "eligible_jobs": int(len(eligible)),
                "placed_jobs": int(placed_jobs),
                "placement_rate": placed_jobs / len(eligible) if len(eligible) else 0.0,
                "demanded_cpu_window": demanded_cpu,
                "placed_cpu_window": placed_cpu,
                "placed_cpu_window_share": placed_cpu / demanded_cpu if demanded_cpu else 0.0,
                "demanded_memory_window": demanded_mem,
                "placed_memory_window": placed_mem,
                "placed_memory_window_share": placed_mem / demanded_mem if demanded_mem else 0.0,
                "eligible_bad_terminal_instances": eligible_bad,
                "placed_bad_terminal_instances": int(placed_bad),
                "eligible_collection_bad_share": float(eligible["observed_bad_collection"].mean())
                if len(eligible)
                else 0.0,
                "mean_predicted_bad_rate": float(eligible["predicted_bad_rate"].mean())
                if len(eligible)
                else 0.0,
                "machines_used": int(len(machines)),
            }
        )
        for reason, count in reasons.items():
            reason_rows.append(
                {
                    "frontier_policy": frontier.name,
                    "risk_threshold": frontier.risk_threshold,
                    "reason": reason,
                    "jobs": int(count),
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(reason_rows)


def calibration_bins(test: pd.DataFrame) -> pd.DataFrame:
    bins = [-np.inf, 0.005, 0.010, 0.030, 0.050, 0.100, 0.150, np.inf]
    labels = [
        "00_<=0.005",
        "01_0.005_0.010",
        "02_0.010_0.030",
        "03_0.030_0.050",
        "04_0.050_0.100",
        "05_0.100_0.150",
        "06_>0.150",
    ]
    tmp = test.copy()
    tmp["predicted_bad_rate_bin"] = pd.cut(
        tmp["predicted_bad_rate"],
        bins=bins,
        labels=labels,
    ).astype(str)
    grouped = tmp.groupby("predicted_bad_rate_bin", observed=False)
    return grouped.agg(
        test_window_collections=("collection_id", "count"),
        observed_bad_collection_share=("observed_bad_collection", "mean"),
        instance_bad_terminal_share=("bad_terminal_instances", lambda x: x.sum()),
        mean_predicted_bad_rate=("predicted_bad_rate", "mean"),
        cpu_window_demand=("cpu_window_demand", "sum"),
    ).reset_index()


def run(expanded_candidates: Path, intervals_path: Path, output_dir: Path,
        drop_graph_future: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    train, test = load_and_prepare(expanded_candidates)
    pipe = fit_model(train, drop_graph_future=drop_graph_future)

    categorical, numeric = feature_lists(drop_graph_future)
    features = categorical + numeric
    test["predicted_bad_rate"] = pipe.predict_proba(test[features])[:, 1]
    train_pred = pipe.predict_proba(train[features])[:, 1]
    test_pred = test["predicted_bad_rate"].to_numpy()

    metrics = pd.DataFrame(
        [
            {
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "train_bad_share": float(train["observed_bad_collection"].mean()),
                "test_bad_share": float(test["observed_bad_collection"].mean()),
                "train_roc_auc": float(roc_auc_score(train["observed_bad_collection"], train_pred)),
                "test_roc_auc": float(roc_auc_score(test["observed_bad_collection"], test_pred)),
                "train_average_precision": float(
                    average_precision_score(train["observed_bad_collection"], train_pred)
                ),
                "test_average_precision": float(
                    average_precision_score(test["observed_bad_collection"], test_pred)
                ),
                "train_brier": float(brier_score_loss(train["observed_bad_collection"], train_pred)),
                "test_brier": float(brier_score_loss(test["observed_bad_collection"], test_pred)),
            }
        ]
    )
    metrics.to_csv(output_dir / "learned_risk_model_metrics.csv", index=False)
    calibration_bins(test).to_csv(output_dir / "learned_risk_calibration_bins.csv", index=False)

    intervals = load_intervals(intervals_path)
    matching, reasons = run_matching(test, intervals)
    matching.to_csv(output_dir / "learned_risk_matching_summary.csv", index=False)
    reasons.to_csv(output_dir / "learned_risk_unplaced_reasons.csv", index=False)
    print(f"wrote learned risk baseline outputs to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expanded-candidates", type=Path, required=True)
    parser.add_argument("--intervals", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--no-graph-future-features",
        action="store_true",
        help="Drop dependency_bucket, dependency_graph_role, degree_bucket, "
             "and outgoing_dependent_edges (out-degree is future information "
             "at admission time); keeps incoming_dependency_edges only.",
    )
    args = parser.parse_args()
    run(args.expanded_candidates, args.intervals, args.output,
        drop_graph_future=args.no_graph_future_features)


if __name__ == "__main__":
    main()
