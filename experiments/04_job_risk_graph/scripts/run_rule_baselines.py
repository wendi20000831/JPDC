#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import duckdb
import pandas as pd

from run_frontier_matching import (
    CLASS_PRIORITY,
    PLACEMENT_POLICIES,
    PlacementPolicy,
    build_active_index,
    load_intervals,
    try_place_exact,
)


DATA_ROOT = Path(
    "/media/jiang/0bfc75f7-326d-2d45-b017-7100903796b6/google-cluster-data-2019"
)
DEFAULT_SAFEHARVEST_ROOT = DATA_ROOT / "_analysis" / "safeharvest"
DEFAULT_EXPANDED_CANDIDATES = (
    DEFAULT_SAFEHARVEST_ROOT / "04_job_risk_graph" / "expanded_candidate_table.parquet"
)
DEFAULT_CPU_OPPORTUNITY_DIR = DEFAULT_SAFEHARVEST_ROOT / "00_safe_harvestability"
DEFAULT_MEMORY_OPPORTUNITY_DIR = DEFAULT_SAFEHARVEST_ROOT / "03_opportunity_matching"
DEFAULT_OUTPUT = DEFAULT_SAFEHARVEST_ROOT / "04_job_risk_graph" / "baselines"

LOW_CLASS = {
    "batch_scheduler",
    "best_effort_queueable",
    "low_priority_flexible",
}


@dataclass(frozen=True)
class RuleBaseline:
    name: str
    label: str
    mask: Callable[[pd.DataFrame], pd.Series]


def small_job_mask(df: pd.DataFrame) -> pd.Series:
    return (
        df["instances"].between(1, 5, inclusive="both")
        & (df["runtime_windows"] <= 48)
        & (df["total_cpu_request"] <= 0.25)
        & (df["total_memory_request"] <= 0.25)
    )


BASELINES = [
    RuleBaseline(
        "all_workload_blind",
        "All candidate work",
        lambda df: pd.Series(True, index=df.index),
    ),
    RuleBaseline(
        "low_class_only",
        "Low-class only",
        lambda df: df["candidate_class"].isin(LOW_CLASS),
    ),
    RuleBaseline(
        "small_only",
        "Small-job only",
        small_job_mask,
    ),
    RuleBaseline(
        "independent_only",
        "Independent-only",
        lambda df: df["dependency_graph_role"] == "independent_leaf",
    ),
    RuleBaseline(
        "finished_only",
        "FINISH-only",
        lambda df: df["collection_terminal_outcome"] == "FINISH",
    ),
    RuleBaseline(
        "finished_small",
        "FINISH + small",
        lambda df: (df["collection_terminal_outcome"] == "FINISH") & small_job_mask(df),
    ),
    RuleBaseline(
        "finished_small_independent",
        "FINISH + small + independent",
        lambda df: (
            (df["collection_terminal_outcome"] == "FINISH")
            & small_job_mask(df)
            & (df["dependency_graph_role"] == "independent_leaf")
        ),
    ),
    RuleBaseline(
        "safeharvest_oracle_p3",
        "SafeHarvest oracle P3",
        lambda df: df["risk_tier_order"] <= 3,
    ),
]


def selected_policies(names: list[str] | None) -> list[PlacementPolicy]:
    if not names:
        return PLACEMENT_POLICIES
    known = {policy.name: policy for policy in PLACEMENT_POLICIES}
    return [known[name] for name in names]


def load_jobs(expanded_candidates: Path, output_dir: Path) -> pd.DataFrame:
    out = output_dir / "rule_baseline_jobs.csv"
    con = duckdb.connect()
    con.execute(
        f"""
        COPY (
          SELECT
            collection_id,
            risk_tier,
            risk_tier_family,
            risk_tier_order,
            candidate_class,
            priority,
            scheduling_class,
            scheduler,
            collection_terminal_outcome,
            dependency_graph_role,
            incoming_dependency_edges,
            outgoing_dependent_edges,
            schedule_win5,
            runtime_windows,
            collection_runtime_sec,
            total_cpu_request,
            total_memory_request,
            instances,
            COALESCE(bad_terminal_instances, 0) AS bad_terminal_instances,
            harvest_risk_score,
            cpu_window_demand,
            memory_window_demand
          FROM read_parquet('{expanded_candidates}')
          WHERE schedule_win5 BETWEEN 6480 AND 8784
            AND risk_tier != 'X_excluded'
            AND total_cpu_request IS NOT NULL
            AND total_cpu_request > 0
            AND total_memory_request IS NOT NULL
            AND runtime_windows IS NOT NULL
            AND runtime_windows > 0
          ORDER BY schedule_win5, risk_tier_order, harvest_risk_score, cpu_window_demand
        ) TO '{out}' (HEADER, DELIMITER ',')
        """
    )
    jobs = pd.read_csv(out)
    jobs["runtime_windows"] = jobs["runtime_windows"].clip(lower=1).astype(int)
    jobs["instances"] = jobs["instances"].fillna(0).astype(int)
    jobs["bad_terminal_instances"] = jobs["bad_terminal_instances"].fillna(0).astype(int)
    jobs["class_priority"] = jobs["candidate_class"].map(CLASS_PRIORITY).fillna(99)
    return jobs


def run_one_baseline(
    jobs: pd.DataFrame,
    intervals: pd.DataFrame,
    baseline: RuleBaseline,
    opportunity_class: str,
    policy: PlacementPolicy,
) -> tuple[dict[str, object], Counter]:
    eligible = jobs[baseline.mask(jobs)].copy()
    eligible = eligible.sort_values(
        [
            "schedule_win5",
            "risk_tier_order",
            "class_priority",
            "harvest_risk_score",
            "cpu_window_demand",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    min_win = int(jobs["schedule_win5"].min())
    max_win = int(jobs["schedule_win5"].max())
    active_index = build_active_index(intervals, min_win, max_win)
    interval_starts = intervals["start_win5"].astype(int).to_list()
    interval_ends = intervals["end_win5"].astype(int).to_list()
    interval_caps = intervals["capacity_cpu"].astype(float).to_list()
    interval_memory_caps = None
    if policy.memory_source:
        interval_memory_caps = intervals["capacity_memory"].astype(float).to_list()

    used_cpu_by_interval: dict[int, list[float]] = {}
    used_memory_by_interval: dict[int, list[float]] | None = {} if policy.memory_source else None
    reasons: Counter = Counter()

    demanded_cpu_work = float(eligible["cpu_window_demand"].sum())
    demanded_memory_work = float(eligible["memory_window_demand"].sum())
    demanded_bad_terminal_instances = int(eligible["bad_terminal_instances"].sum())
    placed_jobs = 0
    placed_cpu_work = 0.0
    placed_memory_work = 0.0
    placed_bad_terminal_instances = 0

    for job in eligible.itertuples(index=False):
        runtime = max(1, int(job.runtime_windows))
        start = int(job.schedule_win5)
        cpu = float(job.total_cpu_request)
        memory = float(job.total_memory_request)
        placed, _, reason = try_place_exact(
            start,
            runtime,
            cpu,
            memory,
            active_index,
            interval_starts,
            interval_ends,
            interval_caps,
            interval_memory_caps,
            used_cpu_by_interval,
            used_memory_by_interval,
        )
        cpu_work = cpu * runtime
        memory_work = memory * runtime
        if placed:
            placed_jobs += 1
            placed_cpu_work += cpu_work
            placed_memory_work += memory_work
            placed_bad_terminal_instances += int(job.bad_terminal_instances)
        else:
            reasons[reason] += 1

    summary = {
        "baseline_policy": baseline.name,
        "baseline_label": baseline.label,
        "opportunity_class": opportunity_class,
        "placement_policy": policy.name,
        "eligible_jobs": int(len(eligible)),
        "placed_jobs": int(placed_jobs),
        "placement_rate": placed_jobs / len(eligible) if len(eligible) else 0.0,
        "demanded_cpu_window": demanded_cpu_work,
        "placed_cpu_window": placed_cpu_work,
        "placed_work_share": placed_cpu_work / demanded_cpu_work if demanded_cpu_work else 0.0,
        "demanded_memory_window": demanded_memory_work,
        "placed_memory_window": placed_memory_work,
        "placed_memory_work_share": placed_memory_work / demanded_memory_work
        if demanded_memory_work
        else 0.0,
        "demanded_bad_terminal_instances": demanded_bad_terminal_instances,
        "placed_bad_terminal_instances": int(placed_bad_terminal_instances),
        "placed_bad_inst_share": placed_bad_terminal_instances / placed_jobs
        if placed_jobs
        else 0.0,
        "used_machines": int(intervals.loc[list(used_cpu_by_interval.keys()), "machine_id"].nunique())
        if used_cpu_by_interval
        else 0,
        "unplaced_jobs": int(len(eligible) - placed_jobs),
    }
    return summary, reasons


def run(
    expanded_candidates: Path,
    cpu_opportunity_dir: Path,
    memory_opportunity_dir: Path,
    output_dir: Path,
    opportunity_classes: list[str],
    placement_policies: list[str] | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = load_jobs(expanded_candidates, output_dir)
    policies = selected_policies(placement_policies)

    summaries: list[dict[str, object]] = []
    reason_rows: list[dict[str, object]] = []

    for opportunity_class in opportunity_classes:
        for policy in policies:
            intervals = load_intervals(
                cpu_opportunity_dir,
                memory_opportunity_dir,
                opportunity_class,
                policy,
            )
            for baseline in BASELINES:
                summary, reasons = run_one_baseline(
                    jobs,
                    intervals,
                    baseline,
                    opportunity_class,
                    policy,
                )
                summaries.append(summary)
                for reason, count in reasons.items():
                    reason_rows.append(
                        {
                            "baseline_policy": baseline.name,
                            "baseline_label": baseline.label,
                            "opportunity_class": opportunity_class,
                            "placement_policy": policy.name,
                            "reason": reason,
                            "jobs": count,
                        }
                    )

    pd.DataFrame(summaries).to_csv(
        output_dir / "rule_baseline_matching_summary.csv", index=False
    )
    pd.DataFrame(reason_rows).to_csv(
        output_dir / "rule_baseline_unplaced_reasons.csv", index=False
    )
    print(f"wrote rule baseline outputs to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expanded-candidates", type=Path, default=DEFAULT_EXPANDED_CANDIDATES)
    parser.add_argument("--cpu-opportunity-dir", type=Path, default=DEFAULT_CPU_OPPORTUNITY_DIR)
    parser.add_argument("--memory-opportunity-dir", type=Path, default=DEFAULT_MEMORY_OPPORTUNITY_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--opportunity-classes", nargs="+", default=["10pct_1h"])
    parser.add_argument(
        "--placement-policies",
        nargs="+",
        default=["frontier_min_cpu_mem_max_exact"],
    )
    args = parser.parse_args()
    run(
        args.expanded_candidates,
        args.cpu_opportunity_dir,
        args.memory_opportunity_dir,
        args.output,
        args.opportunity_classes,
        args.placement_policies,
    )


if __name__ == "__main__":
    main()
