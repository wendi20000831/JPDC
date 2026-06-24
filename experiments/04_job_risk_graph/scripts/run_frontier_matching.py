#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd


DATA_ROOT = Path(
    "/media/jiang/0bfc75f7-326d-2d45-b017-7100903796b6/google-cluster-data-2019"
)
DEFAULT_SAFEHARVEST_ROOT = DATA_ROOT / "_analysis" / "safeharvest"
DEFAULT_EXPANDED_CANDIDATES = (
    DEFAULT_SAFEHARVEST_ROOT / "04_job_risk_graph" / "expanded_candidate_table.parquet"
)
DEFAULT_CPU_OPPORTUNITY_DIR = DEFAULT_SAFEHARVEST_ROOT / "00_safe_harvestability"
DEFAULT_MEMORY_OPPORTUNITY_DIR = DEFAULT_SAFEHARVEST_ROOT / "03_opportunity_matching"
DEFAULT_OUTPUT = DEFAULT_SAFEHARVEST_ROOT / "04_job_risk_graph"

CLASS_PRIORITY = {
    "best_effort_queueable": 0,
    "batch_scheduler": 1,
    "low_priority_flexible": 2,
    "medium_flexibility": 3,
}

FRONTIERS = {
    "P0_strict_seed": lambda df: df["risk_tier_order"] <= 0,
    "P1_plus_clean_small_independent": lambda df: df["risk_tier_order"] <= 1,
    "P2_plus_clean_leaf_dependency": lambda df: df["risk_tier_order"] <= 2,
    "P3_plus_clean_medium_independent": lambda df: df["risk_tier_order"] <= 3,
    "P4_plus_retry_signal": lambda df: (df["risk_tier_order"] <= 3)
    | (df["risk_tier"] == "R1_retry_tolerant_signal"),
}


@dataclass(frozen=True)
class PlacementPolicy:
    name: str
    capacity_source: str
    memory_source: str | None = None


PLACEMENT_POLICIES = [
    PlacementPolicy("frontier_min_cpu_exact", "min_room_frac"),
    PlacementPolicy(
        "frontier_min_cpu_mem_avg_exact",
        "min_room_frac",
        memory_source="min_memory_room_avg_usage",
    ),
    PlacementPolicy(
        "frontier_min_cpu_mem_max_exact",
        "min_room_frac",
        memory_source="min_memory_room_max_usage",
    ),
]


def load_jobs(expanded_candidates: Path, output_dir: Path) -> pd.DataFrame:
    out = output_dir / "frontier_matching_jobs.csv"
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
            dependency_graph_role,
            schedule_win5,
            runtime_windows,
            collection_runtime_sec,
            total_cpu_request,
            total_memory_request,
            instances,
            bad_terminal_instances,
            harvest_risk_score,
            cpu_window_demand,
            memory_window_demand
          FROM read_parquet('{expanded_candidates}')
          WHERE schedule_win5 BETWEEN 6480 AND 8784
            AND risk_tier != 'X_excluded'
            AND total_cpu_request IS NOT NULL
            AND total_cpu_request > 0
            AND runtime_windows IS NOT NULL
            AND runtime_windows > 0
          ORDER BY schedule_win5, risk_tier_order, harvest_risk_score, cpu_window_demand
        ) TO '{out}' (HEADER, DELIMITER ',')
        """
    )
    jobs = pd.read_csv(out)
    jobs["runtime_windows"] = jobs["runtime_windows"].clip(lower=1).astype(int)
    jobs["class_priority"] = jobs["candidate_class"].map(CLASS_PRIORITY).fillna(99)
    return jobs


def load_intervals(
    cpu_opportunity_dir: Path,
    memory_opportunity_dir: Path,
    opportunity_class: str,
    policy: PlacementPolicy,
) -> pd.DataFrame:
    if policy.memory_source:
        path = memory_opportunity_dir / f"memory_enriched_opportunity_intervals_{opportunity_class}.csv"
    else:
        path = cpu_opportunity_dir / f"resource_opportunity_intervals_{opportunity_class}.csv"
    intervals = pd.read_csv(path)
    intervals = intervals.reset_index(drop=True)
    intervals["interval_id"] = intervals.index
    intervals["capacity_cpu"] = intervals["cpu_cap_mean"] * intervals[policy.capacity_source]
    intervals["capacity_cpu_window"] = intervals["capacity_cpu"] * intervals["duration_windows"]
    if policy.memory_source:
        intervals["capacity_memory"] = intervals[policy.memory_source].clip(lower=0.0)
        intervals["capacity_memory_window"] = (
            intervals["capacity_memory"] * intervals["duration_windows"]
        )
    return intervals


def build_active_index(
    intervals: pd.DataFrame,
    min_win: int,
    max_win: int,
) -> dict[int, list[int]]:
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
    interval_memory_caps: list[float] | None,
    used_cpu_by_interval: dict[int, list[float]],
    used_memory_by_interval: dict[int, list[float]] | None,
) -> tuple[bool, int | None, str]:
    end = start + runtime - 1
    interval_ids = active_index.get(start)
    if not interval_ids:
        return False, None, "no_active_interval"

    saw_duration_fit = False
    saw_cpu_fit = False
    saw_memory_fit = interval_memory_caps is None
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

        mem_cap = None
        if interval_memory_caps is not None:
            mem_cap = interval_memory_caps[interval_id]
            if memory > mem_cap:
                continue
            saw_memory_fit = True

        offset = start - interval_starts[interval_id]
        hi = offset + runtime
        used_cpu = used_cpu_by_interval.get(interval_id)
        if used_cpu is None:
            used_cpu = [0.0] * (interval_ends[interval_id] - interval_starts[interval_id] + 1)
            used_cpu_by_interval[interval_id] = used_cpu
            used_memory = None
            if interval_memory_caps is not None and used_memory_by_interval is not None:
                used_memory = [0.0] * len(used_cpu)
                used_memory_by_interval[interval_id] = used_memory
            for pos in range(offset, hi):
                used_cpu[pos] += cpu
                if used_memory is not None:
                    used_memory[pos] += memory
            return True, interval_id, "placed"

        used_memory = None
        if interval_memory_caps is not None and used_memory_by_interval is not None:
            used_memory = used_memory_by_interval[interval_id]

        fits = True
        cpu_fits = True
        memory_fits = True
        for pos in range(offset, hi):
            if used_cpu[pos] + cpu > cap:
                fits = False
                cpu_fits = False
                break
            if used_memory is not None and mem_cap is not None and used_memory[pos] + memory > mem_cap:
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
            if used_memory is not None:
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


def run_one(
    jobs: pd.DataFrame,
    intervals: pd.DataFrame,
    frontier_name: str,
    opportunity_class: str,
    policy: PlacementPolicy,
) -> tuple[dict[str, object], list[dict[str, object]], Counter]:
    mask = FRONTIERS[frontier_name](jobs)
    eligible = jobs[mask].copy()
    eligible = eligible.sort_values(
        ["schedule_win5", "risk_tier_order", "class_priority", "harvest_risk_score", "cpu_window_demand"],
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
    samples: list[dict[str, object]] = []

    demanded_cpu_work = float(eligible["cpu_window_demand"].sum())
    demanded_memory_work = float(eligible["memory_window_demand"].sum())
    placed_jobs = 0
    placed_cpu_work = 0.0
    placed_memory_work = 0.0
    placed_bad_terminal_instances = 0

    for job in eligible.itertuples(index=False):
        runtime = max(1, int(job.runtime_windows))
        start = int(job.schedule_win5)
        cpu = float(job.total_cpu_request)
        memory = float(job.total_memory_request)
        placed, interval_id, reason = try_place_exact(
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

        if len(samples) < 20000:
            samples.append(
                {
                    "frontier_policy": frontier_name,
                    "opportunity_class": opportunity_class,
                    "placement_policy": policy.name,
                    "collection_id": int(job.collection_id),
                    "risk_tier": job.risk_tier,
                    "candidate_class": job.candidate_class,
                    "schedule_win5": start,
                    "runtime_windows": runtime,
                    "total_cpu_request": cpu,
                    "total_memory_request": memory,
                    "placed": int(placed),
                    "interval_id": interval_id if interval_id is not None else "",
                    "unplaced_reason": "" if placed else reason,
                }
            )

    summary = {
        "frontier_policy": frontier_name,
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
        "placed_memory_work_share": placed_memory_work / demanded_memory_work if demanded_memory_work else 0.0,
        "placed_bad_terminal_instances": int(placed_bad_terminal_instances),
        "used_machines": int(intervals.loc[list(used_cpu_by_interval.keys()), "machine_id"].nunique())
        if used_cpu_by_interval
        else 0,
        "unplaced_jobs": int(len(eligible) - placed_jobs),
    }
    return summary, samples, reasons


def run(
    expanded_candidates: Path,
    cpu_opportunity_dir: Path,
    memory_opportunity_dir: Path,
    output_dir: Path,
    opportunity_classes: list[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = load_jobs(expanded_candidates, output_dir)

    summaries: list[dict[str, object]] = []
    samples: list[dict[str, object]] = []
    reason_rows: list[dict[str, object]] = []

    for opportunity_class in opportunity_classes:
        for policy in PLACEMENT_POLICIES:
            intervals = load_intervals(
                cpu_opportunity_dir,
                memory_opportunity_dir,
                opportunity_class,
                policy,
            )
            for frontier_name in FRONTIERS:
                summary, policy_samples, reasons = run_one(
                    jobs,
                    intervals,
                    frontier_name,
                    opportunity_class,
                    policy,
                )
                summaries.append(summary)
                samples.extend(policy_samples[:1000])
                for reason, count in reasons.items():
                    reason_rows.append(
                        {
                            "frontier_policy": frontier_name,
                            "opportunity_class": opportunity_class,
                            "placement_policy": policy.name,
                            "reason": reason,
                            "jobs": count,
                        }
                    )

    pd.DataFrame(summaries).to_csv(output_dir / "frontier_matching_summary.csv", index=False)
    pd.DataFrame(reason_rows).to_csv(output_dir / "frontier_matching_unplaced_reasons.csv", index=False)
    pd.DataFrame(samples).to_csv(output_dir / "frontier_matching_sample.csv", index=False)
    print(f"wrote frontier matching outputs to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expanded-candidates", type=Path, default=DEFAULT_EXPANDED_CANDIDATES)
    parser.add_argument("--cpu-opportunity-dir", type=Path, default=DEFAULT_CPU_OPPORTUNITY_DIR)
    parser.add_argument("--memory-opportunity-dir", type=Path, default=DEFAULT_MEMORY_OPPORTUNITY_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--opportunity-classes",
        nargs="+",
        default=["10pct_1h", "20pct_1h", "10pct_4h", "20pct_4h"],
    )
    args = parser.parse_args()
    run(
        args.expanded_candidates,
        args.cpu_opportunity_dir,
        args.memory_opportunity_dir,
        args.output,
        args.opportunity_classes,
    )


if __name__ == "__main__":
    main()
