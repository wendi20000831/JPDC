#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd


DATA_ROOT = Path(
    "/media/jiang/0bfc75f7-326d-2d45-b017-7100903796b6/google-cluster-data-2019"
)
DEFAULT_SAFEHARVEST_ROOT = DATA_ROOT / "_analysis" / "safeharvest"
DEFAULT_OPPORTUNITY_DIR = DEFAULT_SAFEHARVEST_ROOT / "00_safe_harvestability"
DEFAULT_RISK_TABLE = (
    DEFAULT_SAFEHARVEST_ROOT
    / "01_candidate_workload_risk"
    / "candidate_workload_risk_table.parquet"
)
DEFAULT_DEMAND_TABLE = (
    DEFAULT_SAFEHARVEST_ROOT
    / "02_candidate_resource_demand"
    / "candidate_resource_demand_table.parquet"
)
DEFAULT_OUTPUT = DEFAULT_SAFEHARVEST_ROOT / "03_opportunity_matching"
DEFAULT_TEMP = Path("/mnt/clusterdata0/duckdb_tmp")

CLASS_PRIORITY = {
    "best_effort_queueable": 0,
    "batch_scheduler": 1,
    "low_priority_flexible": 2,
}


@dataclass(frozen=True)
class Policy:
    name: str
    capacity_source: str
    require_single_instance: bool = False
    require_clean_instances: bool = False
    max_total_cpu_request: float | None = None
    priority_order: bool = False


POLICIES = [
    Policy("avg_room_replay", "avg_room_frac"),
    Policy("p10_room_replay", "p10_room_frac"),
    Policy("min_room_replay", "min_room_frac"),
    Policy(
        "safeharvest_v1",
        "min_room_frac",
        require_single_instance=True,
        require_clean_instances=True,
        max_total_cpu_request=0.10,
        priority_order=True,
    ),
]


def build_candidate_jobs(
    risk_table: Path,
    demand_table: Path,
    output_dir: Path,
    temp_dir: Path,
    threads: int,
    memory_limit: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "matching_candidate_jobs.csv"

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={threads}")
    con.execute(f"PRAGMA memory_limit='{memory_limit}'")
    con.execute("PRAGMA temp_directory=?", [str(temp_dir)])
    con.execute(
        f"""
        COPY (
          SELECT
            r.collection_id,
            r.candidate_class,
            CAST(FLOOR(r.schedule_time / 300000000.0) AS BIGINT) AS schedule_win5,
            CAST(CEIL(d.collection_runtime_sec / 300.0) AS BIGINT) AS runtime_windows,
            d.collection_runtime_sec,
            d.collection_queue_sec,
            d.total_cpu_request,
            d.total_memory_request,
            d.instances,
            d.finished_instances,
            d.bad_terminal_instances,
            d.no_terminal_instances,
            d.multi_machine_instances,
            CAST(d.harvest_risk_score AS DOUBLE) AS harvest_risk_score
          FROM read_parquet('{risk_table}') AS r
          INNER JOIN read_parquet('{demand_table}') AS d
            ON r.collection_id = d.collection_id
          WHERE r.safeharvest_seed_candidate = 1
            AND r.schedule_time IS NOT NULL
            AND d.collection_runtime_sec IS NOT NULL
            AND d.collection_runtime_sec > 0
            AND d.total_cpu_request IS NOT NULL
            AND d.total_cpu_request > 0
            AND FLOOR(r.schedule_time / 300000000.0) BETWEEN 6480 AND 8784
        ) TO '{out}' (HEADER, DELIMITER ',')
        """
    )
    return out


def load_intervals(opportunity_dir: Path, opportunity_class: str, capacity_source: str) -> pd.DataFrame:
    path = opportunity_dir / f"resource_opportunity_intervals_{opportunity_class}.csv"
    intervals = pd.read_csv(path)
    intervals = intervals.reset_index(drop=True)
    intervals["interval_id"] = intervals.index
    intervals["capacity_cpu"] = intervals["cpu_cap_mean"] * intervals[capacity_source]
    intervals["capacity_cpu_window"] = intervals["capacity_cpu"] * intervals["duration_windows"]
    return intervals


def filter_jobs(jobs: pd.DataFrame, policy: Policy) -> tuple[pd.DataFrame, Counter]:
    reasons: Counter = Counter()
    eligible = jobs.copy()

    if policy.require_single_instance:
        mask = eligible["instances"] == 1
        reasons["not_single_instance"] += int((~mask).sum())
        eligible = eligible[mask]

    if policy.require_clean_instances:
        mask = eligible["bad_terminal_instances"] == 0
        reasons["bad_instance_terminal"] += int((~mask).sum())
        eligible = eligible[mask]

    if policy.max_total_cpu_request is not None:
        mask = eligible["total_cpu_request"] <= policy.max_total_cpu_request
        reasons["too_large_for_policy"] += int((~mask).sum())
        eligible = eligible[mask]

    eligible = eligible.copy()
    eligible["class_priority"] = eligible["candidate_class"].map(CLASS_PRIORITY).fillna(99)
    eligible["job_work_cpu_window"] = eligible["total_cpu_request"] * eligible["runtime_windows"]

    sort_cols = ["schedule_win5"]
    if policy.priority_order:
        sort_cols.extend(["class_priority", "harvest_risk_score", "job_work_cpu_window"])
    else:
        sort_cols.extend(["job_work_cpu_window", "harvest_risk_score"])
    eligible = eligible.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    return eligible, reasons


def build_window_capacity(
    intervals: pd.DataFrame, min_win: int, max_win: int
) -> tuple[dict[int, float], dict[int, float]]:
    total_capacity: defaultdict[int, float] = defaultdict(float)
    max_machine_capacity: defaultdict[int, float] = defaultdict(float)

    for row in intervals[
        ["start_win5", "end_win5", "capacity_cpu"]
    ].itertuples(index=False):
        capacity = float(row.capacity_cpu)
        if capacity <= 0:
            continue
        start = max(int(row.start_win5), min_win)
        end = min(int(row.end_win5), max_win)
        for win in range(start, end + 1):
            total_capacity[win] += capacity
            if capacity > max_machine_capacity[win]:
                max_machine_capacity[win] = capacity
    return dict(total_capacity), dict(max_machine_capacity)


def try_place_job_aggregate(
    job,
    total_capacity: dict[int, float],
    max_machine_capacity: dict[int, float],
    used_by_window: defaultdict[int, float],
) -> tuple[bool, str]:
    start = int(job.schedule_win5)
    runtime = max(1, int(job.runtime_windows))
    end = start + runtime - 1
    cpu = float(job.total_cpu_request)

    for win in range(start, end + 1):
        if total_capacity.get(win, 0.0) <= 0:
            return False, "no_active_capacity"
        if max_machine_capacity.get(win, 0.0) < cpu:
            return False, "cpu_too_large_for_any_gap"

    for win in range(start, end + 1):
        if used_by_window[win] + cpu > total_capacity.get(win, 0.0):
            return False, "capacity_exhausted"

    for win in range(start, end + 1):
        used_by_window[win] += cpu
    return True, "placed"


def simulate_policy(
    jobs: pd.DataFrame,
    opportunity_dir: Path,
    opportunity_class: str,
    policy: Policy,
) -> tuple[dict[str, float | int | str], pd.DataFrame, Counter]:
    intervals = load_intervals(opportunity_dir, opportunity_class, policy.capacity_source)
    eligible, prefilter_reasons = filter_jobs(jobs, policy)

    min_win = int(jobs["schedule_win5"].min())
    max_win = int(jobs["schedule_win5"].max())
    total_capacity, max_machine_capacity = build_window_capacity(intervals, min_win, max_win)
    used_by_window: defaultdict[int, float] = defaultdict(float)

    placement_rows = []
    unplaced_reasons = Counter(prefilter_reasons)
    placed_count = 0
    placed_work = 0.0
    demanded_work = float(eligible["job_work_cpu_window"].sum()) if len(eligible) else 0.0
    placed_bad_instances = 0.0

    for job in eligible.itertuples(index=False):
        placed, reason = try_place_job_aggregate(
            job, total_capacity, max_machine_capacity, used_by_window
        )
        if placed:
            placed_count += 1
            work = float(job.total_cpu_request) * int(job.runtime_windows)
            placed_work += work
            placed_bad_instances += float(job.bad_terminal_instances)
            placement_rows.append(
                {
                    "collection_id": int(job.collection_id),
                    "candidate_class": job.candidate_class,
                    "opportunity_class": opportunity_class,
                    "policy": policy.name,
                    "schedule_win5": int(job.schedule_win5),
                    "runtime_windows": int(job.runtime_windows),
                    "total_cpu_request": float(job.total_cpu_request),
                    "job_work_cpu_window": work,
                    "bad_terminal_instances": float(job.bad_terminal_instances),
                    "harvest_risk_score": float(job.harvest_risk_score),
                }
            )
        else:
            unplaced_reasons[reason] += 1

    total_capacity_window = float(intervals["capacity_cpu_window"].sum())
    summary = {
        "opportunity_class": opportunity_class,
        "policy": policy.name,
        "capacity_source": policy.capacity_source,
        "input_jobs": int(len(jobs)),
        "eligible_jobs": int(len(eligible)),
        "placed_jobs": int(placed_count),
        "placement_rate": placed_count / len(eligible) if len(eligible) else 0.0,
        "demanded_cpu_window": demanded_work,
        "placed_cpu_window": placed_work,
        "placed_work_share": placed_work / demanded_work if demanded_work else 0.0,
        "opportunity_cpu_window": total_capacity_window,
        "used_opportunity_share": placed_work / total_capacity_window
        if total_capacity_window
        else 0.0,
        "placed_bad_terminal_instances": placed_bad_instances,
        "used_peak_window_cpu": max(used_by_window.values()) if used_by_window else 0.0,
        "available_peak_window_cpu": max(total_capacity.values()) if total_capacity else 0.0,
        "intervals": int(len(intervals)),
        "machines": int(intervals["machine_id"].nunique()),
    }
    return summary, pd.DataFrame(placement_rows), unplaced_reasons


def run(
    opportunity_dir: Path,
    risk_table: Path,
    demand_table: Path,
    output_dir: Path,
    temp_dir: Path,
    threads: int,
    memory_limit: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs_path = build_candidate_jobs(
        risk_table, demand_table, output_dir, temp_dir, threads, memory_limit
    )
    jobs = pd.read_csv(jobs_path)
    jobs["runtime_windows"] = jobs["runtime_windows"].clip(lower=1)

    summaries = []
    by_class_rows = []
    reason_rows = []
    sampled_placements = []

    for opportunity_class in ["10pct_1h", "20pct_1h", "10pct_4h", "20pct_4h"]:
        for policy in POLICIES:
            summary, placements, reasons = simulate_policy(
                jobs, opportunity_dir, opportunity_class, policy
            )
            summaries.append(summary)
            for reason, count in reasons.items():
                reason_rows.append(
                    {
                        "opportunity_class": opportunity_class,
                        "policy": policy.name,
                        "reason": reason,
                        "jobs": int(count),
                    }
                )
            if len(placements):
                class_summary = (
                    placements.groupby("candidate_class", as_index=False)
                    .agg(
                        placed_jobs=("collection_id", "count"),
                        placed_cpu_window=("job_work_cpu_window", "sum"),
                        placed_bad_terminal_instances=("bad_terminal_instances", "sum"),
                    )
                    .assign(
                        opportunity_class=opportunity_class,
                        policy=policy.name,
                    )
                )
                by_class_rows.append(class_summary)
                sampled_placements.append(placements.head(5000))

    pd.DataFrame(summaries).to_csv(
        output_dir / "matching_policy_summary.csv", index=False
    )
    if by_class_rows:
        pd.concat(by_class_rows, ignore_index=True).to_csv(
            output_dir / "matching_policy_by_class.csv", index=False
        )
    pd.DataFrame(reason_rows).to_csv(
        output_dir / "matching_unplaced_reasons.csv", index=False
    )
    if sampled_placements:
        pd.concat(sampled_placements, ignore_index=True).to_csv(
            output_dir / "matching_placement_sample.csv", index=False
        )
    print(f"wrote opportunity matching outputs to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opportunity-dir", type=Path, default=DEFAULT_OPPORTUNITY_DIR)
    parser.add_argument("--risk-table", type=Path, default=DEFAULT_RISK_TABLE)
    parser.add_argument("--demand-table", type=Path, default=DEFAULT_DEMAND_TABLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--temp-dir", type=Path, default=DEFAULT_TEMP)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--memory-limit", default="64GB")
    args = parser.parse_args()
    run(
        args.opportunity_dir,
        args.risk_table,
        args.demand_table,
        args.output,
        args.temp_dir,
        args.threads,
        args.memory_limit,
    )


if __name__ == "__main__":
    main()
