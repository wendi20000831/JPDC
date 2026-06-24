#!/usr/bin/env python3
"""Per-day placement breakdown for four headline frontiers.

For oracle P3 (S0+S1+S2+S3), operational OP3 (OP0+OP1+OP2+OP3), learned-full
L2 (predicted bad rate <= 0.03 with all features), and learned-indeg L2
(predicted bad rate <= 0.03 with out-degree-derived features removed), this
script:

  1. loads the expanded candidate table and the 10pct_1h memory-enriched
     opportunity intervals;
  2. assembles each frontier's eligible set;
  3. runs the same exact CPU+memory-maximum replay used elsewhere;
  4. aggregates per-job placement results into per-day metrics across
     test-window days 23-30.

Output: a single CSV with one row per (frontier, day) and aggregate
metrics (eligible_jobs, placed_jobs, placed_cpu_window, placed_bad).
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TEST_START_WIN5 = 6480
TEST_END_WIN5 = 8784
WINDOWS_PER_DAY = 288
START_DAY = 23  # day index for the first test-window day

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


def load_candidates(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        "collection_id", "candidate_class", "priority", "scheduling_class",
        "scheduler", "collection_terminal_outcome", "dependency_bucket",
        "dependency_graph_role", "incoming_dependency_edges",
        "outgoing_dependent_edges", "schedule_win5", "runtime_windows",
        "total_cpu_request", "total_memory_request",
        "max_instance_cpu_request", "max_instance_memory_request",
        "p50_instance_cpu_request", "p90_instance_cpu_request",
        "p50_instance_memory_request", "p90_instance_memory_request",
        "instances", "bad_terminal_instances", "cpu_window_demand",
        "memory_window_demand", "risk_tier", "risk_tier_family",
    ]
    df = pd.read_parquet(path, columns=columns)
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


def fit_learned(train: pd.DataFrame, drop_graph_future: bool) -> Pipeline:
    categorical = [
        "candidate_class", "scheduling_class", "scheduler",
        "dependency_bucket", "dependency_graph_role",
        "priority_bucket", "instance_bucket", "cpu_bucket", "memory_bucket",
        "max_cpu_bucket", "max_memory_bucket", "degree_bucket",
    ]
    numeric = [
        "priority", "incoming_dependency_edges", "outgoing_dependent_edges",
        "instances", "total_cpu_request", "total_memory_request",
        "max_instance_cpu_request", "max_instance_memory_request",
        "p50_instance_cpu_request", "p90_instance_cpu_request",
        "p50_instance_memory_request", "p90_instance_memory_request",
    ]
    if drop_graph_future:
        categorical = [c for c in categorical
                       if c not in {"dependency_bucket", "dependency_graph_role", "degree_bucket"}]
        numeric = [n for n in numeric if n != "outgoing_dependent_edges"]
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=True)
    preprocessor = ColumnTransformer(
        transformers=[("cat", encoder, categorical),
                      ("num", StandardScaler(), numeric)],
        remainder="drop",
    )
    model = LogisticRegression(solver="saga", max_iter=600, C=1.0, n_jobs=-1, random_state=17)
    pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
    pipe.fit(train[categorical + numeric], train["observed_bad_collection"])
    return pipe, categorical + numeric


def load_intervals(path: Path) -> pd.DataFrame:
    intervals = pd.read_csv(path)
    intervals = intervals.reset_index(drop=True)
    intervals["interval_id"] = intervals.index
    intervals["capacity_cpu"] = intervals["cpu_cap_mean"] * intervals["min_room_frac"]
    intervals["capacity_memory"] = intervals["min_memory_room_max_usage"].clip(lower=0.0)
    return intervals


def build_active_index(intervals: pd.DataFrame, min_win: int, max_win: int) -> dict[int, list[int]]:
    starts = intervals["start_win5"].astype(int).to_list()
    ends = intervals["end_win5"].astype(int).to_list()
    active: dict[int, list[int]] = {}
    for i, (s, e) in enumerate(zip(starts, ends)):
        if e < min_win or s > max_win:
            continue
        s_clip = max(s, min_win)
        e_clip = min(e, max_win)
        for w in range(s_clip, e_clip + 1):
            active.setdefault(w, []).append(i)
    return active


def try_place(schedule_win5, runtime, cpu, memory, active_index, starts, ends, caps, mem_caps, used_cpu, used_mem):
    last_win = schedule_win5 + runtime - 1
    if schedule_win5 not in active_index:
        return False
    candidates = active_index[schedule_win5]
    for interval_id in candidates:
        s_int = starts[interval_id]
        e_int = ends[interval_id]
        if last_win > e_int:
            continue
        if cpu > caps[interval_id] or memory > mem_caps[interval_id]:
            continue
        offset = schedule_win5 - s_int
        hi = offset + runtime
        duration_total = e_int - s_int + 1
        u_cpu = used_cpu.get(interval_id)
        if u_cpu is None:
            u_cpu = [0.0] * duration_total
            used_cpu[interval_id] = u_cpu
        u_mem = used_mem.get(interval_id)
        if u_mem is None:
            u_mem = [0.0] * duration_total
            used_mem[interval_id] = u_mem
        ok_cpu = all(u_cpu[p] + cpu <= caps[interval_id] for p in range(offset, hi))
        ok_mem = all(u_mem[p] + memory <= mem_caps[interval_id] for p in range(offset, hi))
        if not ok_cpu or not ok_mem:
            continue
        for p in range(offset, hi):
            u_cpu[p] += cpu
            u_mem[p] += memory
        return True
    return False


def run_match_per_day(eligible: pd.DataFrame, intervals: pd.DataFrame,
                      sort_keys: list[str], frontier_name: str) -> pd.DataFrame:
    if len(eligible) == 0:
        return pd.DataFrame()
    eligible = eligible.sort_values(sort_keys, kind="mergesort").reset_index(drop=True)
    min_win = int(eligible["schedule_win5"].min())
    max_win = int(eligible["schedule_win5"].max())
    active_index = build_active_index(intervals, min_win, max_win)
    starts = intervals["start_win5"].astype(int).to_list()
    ends = intervals["end_win5"].astype(int).to_list()
    caps = intervals["capacity_cpu"].astype(float).to_list()
    mem_caps = intervals["capacity_memory"].astype(float).to_list()
    used_cpu: dict[int, list[float]] = {}
    used_mem: dict[int, list[float]] = {}

    placements = []
    for job in eligible.itertuples(index=False):
        runtime = max(1, int(job.runtime_windows))
        placed = try_place(int(job.schedule_win5), runtime,
                           float(job.total_cpu_request),
                           float(job.total_memory_request),
                           active_index, starts, ends, caps, mem_caps,
                           used_cpu, used_mem)
        day = START_DAY + (int(job.schedule_win5) - TEST_START_WIN5) // WINDOWS_PER_DAY
        placements.append({
            "day": day,
            "placed": int(placed),
            "cpu_window_demand": float(job.cpu_window_demand),
            "memory_window_demand": float(job.memory_window_demand),
            "bad_terminal_instances": int(job.bad_terminal_instances),
        })
    pj = pd.DataFrame(placements)
    agg = pj.groupby("day").agg(
        eligible_jobs=("placed", "size"),
        placed_jobs=("placed", "sum"),
        eligible_cpu_window=("cpu_window_demand", "sum"),
        placed_cpu_window=("cpu_window_demand", lambda x: x[pj.loc[x.index, "placed"] == 1].sum()),
        eligible_memory_window=("memory_window_demand", "sum"),
        placed_memory_window=("memory_window_demand", lambda x: x[pj.loc[x.index, "placed"] == 1].sum()),
        eligible_bad_inst=("bad_terminal_instances", "sum"),
        placed_bad_inst=("bad_terminal_instances", lambda x: x[pj.loc[x.index, "placed"] == 1].sum()),
    ).reset_index()
    agg["frontier"] = frontier_name
    return agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expanded-candidates", type=Path, required=True)
    parser.add_argument("--intervals", type=Path, required=True)
    parser.add_argument("--op-frontier-jobs", type=Path, required=True,
                        help="operational_frontier_jobs.csv from prior OP3 run")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    print("loading candidates...", flush=True)
    train, test = load_candidates(args.expanded_candidates)
    print(f"  train={len(train)}, test={len(test)}", flush=True)

    print("training learned (full features)...", flush=True)
    pipe_full, feats_full = fit_learned(train, drop_graph_future=False)
    test["pred_full"] = pipe_full.predict_proba(test[feats_full])[:, 1]

    print("training learned (in-degree only)...", flush=True)
    pipe_indeg, feats_indeg = fit_learned(train, drop_graph_future=True)
    test["pred_indeg"] = pipe_indeg.predict_proba(test[feats_indeg])[:, 1]

    print("loading intervals...", flush=True)
    intervals = load_intervals(args.intervals)

    print("loading operational frontier collection ids...", flush=True)
    op_jobs = pd.read_csv(args.op_frontier_jobs)
    # OP3 = OP0 ∪ OP1 ∪ OP2 ∪ OP3 (cumulative, excludes OP4 retry band)
    op3_tiers = {"OP0_visible_seed", "OP1_visible_small_independent",
                 "OP2_visible_small_dependency", "OP3_visible_medium_independent"}
    op3_ids = set(op_jobs[op_jobs["risk_tier"].isin(op3_tiers)]["collection_id"].tolist())
    print(f"  OP3 admitted: {len(op3_ids)} collection ids", flush=True)

    sort_keys = ["schedule_win5", "class_priority", "cpu_window_demand"]
    frontiers = []

    print("running oracle P3...", flush=True)
    oracle_p3 = test[test["risk_tier"].isin(
        ["S0_strict_seed", "S1_clean_small_independent",
         "S2_clean_small_leaf_dependency", "S3_clean_medium_independent"])]
    frontiers.append(run_match_per_day(oracle_p3, intervals,
                                       sort_keys, "oracle_P3"))

    print("running OP3...", flush=True)
    op3 = test[test["collection_id"].isin(op3_ids)]
    frontiers.append(run_match_per_day(op3, intervals,
                                       sort_keys, "operational_OP3"))

    print("running learned L2 (full)...", flush=True)
    l2_full = test[test["pred_full"] <= 0.03]
    frontiers.append(run_match_per_day(l2_full, intervals,
                                       sort_keys, "learned_L2_full"))

    print("running learned L2 (in-degree only)...", flush=True)
    l2_indeg = test[test["pred_indeg"] <= 0.03]
    frontiers.append(run_match_per_day(l2_indeg, intervals,
                                       sort_keys, "learned_L2_indeg"))

    out = pd.concat(frontiers, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"wrote {args.output}", flush=True)
    print(out.groupby("frontier")[["placed_jobs", "placed_cpu_window",
                                    "placed_bad_inst"]].sum())


if __name__ == "__main__":
    main()
