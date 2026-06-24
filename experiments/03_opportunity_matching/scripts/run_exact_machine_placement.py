#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DATA_ROOT = Path(
    "/media/jiang/0bfc75f7-326d-2d45-b017-7100903796b6/google-cluster-data-2019"
)
DEFAULT_SAFEHARVEST_ROOT = DATA_ROOT / "_analysis" / "safeharvest"
DEFAULT_OPPORTUNITY_DIR = DEFAULT_SAFEHARVEST_ROOT / "00_safe_harvestability"
DEFAULT_INPUT_JOBS = DEFAULT_SAFEHARVEST_ROOT / "03_opportunity_matching" / "matching_candidate_jobs.csv"
DEFAULT_OUTPUT = DEFAULT_SAFEHARVEST_ROOT / "03_opportunity_matching"

CLASS_PRIORITY = {
    "best_effort_queueable": 0,
    "batch_scheduler": 1,
    "low_priority_flexible": 2,
}


@dataclass(frozen=True)
class Policy:
    name: str
    capacity_source: str
    memory_source: str | None = None
    require_single_instance: bool = False
    require_clean_instances: bool = False
    max_total_cpu_request: float | None = None
    max_total_memory_request: float | None = None
    priority_order: bool = False
    max_delay_windows: int = 0


POLICIES = [
    Policy("avg_room_exact", "avg_room_frac"),
    Policy("p10_room_exact", "p10_room_frac"),
    Policy("min_room_exact", "min_room_frac"),
    Policy(
        "min_room_cpu_mem_avg_exact",
        "min_room_frac",
        memory_source="min_memory_room_avg_usage",
    ),
    Policy(
        "safeharvest_v1_exact",
        "min_room_frac",
        require_single_instance=True,
        require_clean_instances=True,
        max_total_cpu_request=0.10,
        priority_order=True,
    ),
    Policy(
        "safeharvest_v1_shift_1h_exact",
        "min_room_frac",
        require_single_instance=True,
        require_clean_instances=True,
        max_total_cpu_request=0.10,
        priority_order=True,
        max_delay_windows=12,
    ),
    Policy(
        "safeharvest_v1_cpu_mem_avg_exact",
        "min_room_frac",
        memory_source="min_memory_room_avg_usage",
        require_single_instance=True,
        require_clean_instances=True,
        max_total_cpu_request=0.10,
        priority_order=True,
    ),
    Policy(
        "safeharvest_v1_cpu_mem_max_exact",
        "min_room_frac",
        memory_source="min_memory_room_max_usage",
        require_single_instance=True,
        require_clean_instances=True,
        max_total_cpu_request=0.10,
        priority_order=True,
    ),
]


def load_intervals(
    opportunity_dir: Path,
    output_dir: Path,
    opportunity_class: str,
    policy: Policy,
) -> pd.DataFrame:
    if policy.memory_source:
        path = output_dir / f"memory_enriched_opportunity_intervals_{opportunity_class}.csv"
    else:
        path = opportunity_dir / f"resource_opportunity_intervals_{opportunity_class}.csv"
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

    if policy.max_total_memory_request is not None:
        mask = eligible["total_memory_request"] <= policy.max_total_memory_request
        reasons["too_memory_large_for_policy"] += int((~mask).sum())
        eligible = eligible[mask]

    eligible = eligible.copy()
    eligible["class_priority"] = eligible["candidate_class"].map(CLASS_PRIORITY).fillna(99)
    eligible["job_work_cpu_window"] = eligible["total_cpu_request"] * eligible["runtime_windows"]
    sort_cols = ["schedule_win5"]
    if policy.priority_order:
        sort_cols.extend(["class_priority", "harvest_risk_score", "job_work_cpu_window"])
    else:
        sort_cols.extend(["job_work_cpu_window", "harvest_risk_score"])
    return eligible.sort_values(sort_cols, kind="mergesort").reset_index(drop=True), reasons


def build_active_index(
    intervals: pd.DataFrame, min_win: int, max_win: int
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

    for win, interval_ids in active.items():
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
        used_cpu = used_cpu_by_interval.get(interval_id)
        hi = offset + runtime
        if used_cpu is None:
            used_cpu = [0.0] * (interval_ends[interval_id] - interval_starts[interval_id] + 1)
            used_cpu_by_interval[interval_id] = used_cpu
            used_memory = None
            if interval_memory_caps is not None and used_memory_by_interval is not None:
                used_memory = [0.0] * (
                    interval_ends[interval_id] - interval_starts[interval_id] + 1
                )
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
                cpu_fits = False
                fits = False
                break
            if used_memory is not None and mem_cap is not None and used_memory[pos] + memory > mem_cap:
                memory_fits = False
                fits = False
                break
        if not fits:
            if not cpu_fits:
                saw_cpu_exhausted = True
            elif not memory_fits:
                saw_memory_exhausted = True
            else:
                saw_cpu_exhausted = True
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
        return False, None, "machine_capacity_exhausted"
    return False, None, "machine_capacity_exhausted"


def simulate_policy(
    jobs: pd.DataFrame,
    opportunity_dir: Path,
    output_dir: Path,
    opportunity_class: str,
    policy: Policy,
) -> tuple[dict[str, float | int | str], pd.DataFrame, Counter]:
    intervals = load_intervals(opportunity_dir, output_dir, opportunity_class, policy)
    eligible, reasons = filter_jobs(jobs, policy)

    min_win = int(jobs["schedule_win5"].min())
    max_win = int(jobs["schedule_win5"].max())
    active_index = build_active_index(intervals, min_win, max_win)

    interval_starts = intervals["start_win5"].astype(int).to_list()
    interval_ends = intervals["end_win5"].astype(int).to_list()
    interval_caps = intervals["capacity_cpu"].astype(float).to_list()
    interval_memory_caps = (
        intervals["capacity_memory"].astype(float).to_list()
        if policy.memory_source
        else None
    )
    interval_machines = intervals["machine_id"].astype(int).to_list()
    used_cpu_by_interval: dict[int, list[float]] = {}
    used_memory_by_interval: dict[int, list[float]] | None = {} if policy.memory_source else None

    placed_rows = []
    placed_count = 0
    placed_work = 0.0
    placed_bad_instances = 0.0
    shifted_jobs = 0
    total_delay_windows = 0
    demanded_work = float(eligible["job_work_cpu_window"].sum()) if len(eligible) else 0.0
    demanded_memory_work = (
        float((eligible["total_memory_request"] * eligible["runtime_windows"]).sum())
        if len(eligible)
        else 0.0
    )
    placed_memory_work = 0.0

    for job in eligible.itertuples(index=False):
        runtime = max(1, int(job.runtime_windows))
        original_start = int(job.schedule_win5)
        cpu = float(job.total_cpu_request)
        memory = float(job.total_memory_request)
        placed = False
        interval_id = None
        reason = "no_shifted_fit"
        placed_start = original_start
        delay_windows = 0
        for delay in range(policy.max_delay_windows + 1):
            start = original_start + delay
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
            if placed:
                placed_start = start
                delay_windows = delay
                break
        if not placed:
            if policy.max_delay_windows > 0:
                reasons["no_shifted_fit"] += 1
            else:
                reasons[reason] += 1
            continue

        work = cpu * runtime
        placed_count += 1
        placed_work += work
        placed_memory_work += memory * runtime
        placed_bad_instances += float(job.bad_terminal_instances)
        if delay_windows > 0:
            shifted_jobs += 1
            total_delay_windows += delay_windows
        placed_rows.append(
            {
                "collection_id": int(job.collection_id),
                "candidate_class": job.candidate_class,
                "opportunity_class": opportunity_class,
                "policy": policy.name,
                "machine_id": interval_machines[int(interval_id)],
                "interval_id": int(interval_id),
                "schedule_win5": original_start,
                "placed_start_win5": placed_start,
                "delay_windows": delay_windows,
                "runtime_windows": runtime,
                "total_cpu_request": cpu,
                "total_memory_request": float(job.total_memory_request),
                "job_work_cpu_window": work,
                "bad_terminal_instances": float(job.bad_terminal_instances),
                "harvest_risk_score": float(job.harvest_risk_score),
            }
        )

    used_peak_interval_cpu = 0.0
    used_cpu_window = 0.0
    for interval_id, used in used_cpu_by_interval.items():
        if used:
            used_peak_interval_cpu = max(used_peak_interval_cpu, max(used))
            used_cpu_window += sum(used)
    used_peak_interval_memory = 0.0
    if used_memory_by_interval is not None:
        for used in used_memory_by_interval.values():
            if used:
                used_peak_interval_memory = max(used_peak_interval_memory, max(used))

    total_capacity_window = float(intervals["capacity_cpu_window"].sum())
    total_memory_capacity_window = (
        float(intervals["capacity_memory_window"].sum()) if policy.memory_source else 0.0
    )
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
        "demanded_memory_window": demanded_memory_work,
        "placed_memory_window": placed_memory_work,
        "placed_memory_work_share": placed_memory_work / demanded_memory_work
        if demanded_memory_work
        else 0.0,
        "opportunity_cpu_window": total_capacity_window,
        "used_opportunity_share": placed_work / total_capacity_window if total_capacity_window else 0.0,
        "opportunity_memory_window": total_memory_capacity_window,
        "used_memory_opportunity_share": placed_memory_work / total_memory_capacity_window
        if total_memory_capacity_window
        else 0.0,
        "placed_bad_terminal_instances": placed_bad_instances,
        "shifted_jobs": int(shifted_jobs),
        "avg_delay_minutes": (total_delay_windows * 5.0 / shifted_jobs) if shifted_jobs else 0.0,
        "used_intervals": int(len(used_cpu_by_interval)),
        "used_machines": int(pd.Series([interval_machines[i] for i in used_cpu_by_interval]).nunique())
        if used_cpu_by_interval
        else 0,
        "used_peak_interval_cpu": used_peak_interval_cpu,
        "used_peak_interval_memory": used_peak_interval_memory,
        "intervals": int(len(intervals)),
        "machines": int(intervals["machine_id"].nunique()),
    }
    return summary, pd.DataFrame(placed_rows), reasons


def run(input_jobs: Path, opportunity_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = pd.read_csv(input_jobs)
    jobs["runtime_windows"] = jobs["runtime_windows"].clip(lower=1).astype(int)

    summaries = []
    reason_rows = []
    by_class_rows = []
    placement_samples = []

    for opportunity_class in ["10pct_1h", "20pct_1h", "10pct_4h", "20pct_4h"]:
        for policy in POLICIES:
            summary, placements, reasons = simulate_policy(
                jobs, opportunity_dir, output_dir, opportunity_class, policy
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
                by_class_rows.append(
                    placements.groupby("candidate_class", as_index=False)
                    .agg(
                        placed_jobs=("collection_id", "count"),
                        placed_cpu_window=("job_work_cpu_window", "sum"),
                        placed_bad_terminal_instances=("bad_terminal_instances", "sum"),
                        used_machines=("machine_id", "nunique"),
                    )
                    .assign(opportunity_class=opportunity_class, policy=policy.name)
                )
                placement_samples.append(placements.head(5000))

    pd.DataFrame(summaries).to_csv(output_dir / "exact_machine_policy_summary.csv", index=False)
    pd.DataFrame(reason_rows).to_csv(output_dir / "exact_machine_unplaced_reasons.csv", index=False)
    if by_class_rows:
        pd.concat(by_class_rows, ignore_index=True).to_csv(
            output_dir / "exact_machine_policy_by_class.csv", index=False
        )
    if placement_samples:
        pd.concat(placement_samples, ignore_index=True).to_csv(
            output_dir / "exact_machine_placement_sample.csv", index=False
        )
    print(f"wrote exact machine placement outputs to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jobs", type=Path, default=DEFAULT_INPUT_JOBS)
    parser.add_argument("--opportunity-dir", type=Path, default=DEFAULT_OPPORTUNITY_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.input_jobs, args.opportunity_dir, args.output)


if __name__ == "__main__":
    main()
