#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import duckdb
import pandas as pd

from run_frontier_matching import (
    CLASS_PRIORITY,
    PLACEMENT_POLICIES,
    load_intervals,
    run_one,
)


DATA_ROOT = Path(
    "/media/jiang/0bfc75f7-326d-2d45-b017-7100903796b6/google-cluster-data-2019"
)
DEFAULT_SAFEHARVEST_ROOT = DATA_ROOT / "_analysis" / "safeharvest"
DEFAULT_EXPANDED_CANDIDATES = (
    DEFAULT_SAFEHARVEST_ROOT / "04_job_risk_graph" / "expanded_candidate_table.parquet"
)
DEFAULT_RISK_TABLE = (
    DEFAULT_SAFEHARVEST_ROOT
    / "01_candidate_workload_risk"
    / "candidate_workload_risk_table.parquet"
)
DEFAULT_CPU_OPPORTUNITY_DIR = DEFAULT_SAFEHARVEST_ROOT / "00_safe_harvestability"
DEFAULT_MEMORY_OPPORTUNITY_DIR = DEFAULT_SAFEHARVEST_ROOT / "03_opportunity_matching"
DEFAULT_OUTPUT = DEFAULT_SAFEHARVEST_ROOT / "04_job_risk_graph" / "operational"
DEFAULT_TEMP = Path("/mnt/clusterdata0/duckdb_tmp")

TEST_START_WIN5 = 6480
TEST_END_WIN5 = 8784


def _copy_sql(con: duckdb.DuckDBPyConnection, query: str, path: Path) -> None:
    con.execute(f"COPY ({query}) TO '{path}' (HEADER, DELIMITER ',')")


def _selected_policies(names: list[str] | None):
    if not names:
        return PLACEMENT_POLICIES
    known = {p.name: p for p in PLACEMENT_POLICIES}
    selected = []
    for name in names:
        if name not in known:
            raise ValueError(f"unknown placement policy: {name}")
        selected.append(known[name])
    return selected


def build_operational_jobs(
    expanded_candidates: Path,
    risk_table: Path,
    output_dir: Path,
    temp_dir: Path,
    threads: int,
    memory_limit: str,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={threads}")
    con.execute(f"PRAGMA memory_limit='{memory_limit}'")
    con.execute("PRAGMA temp_directory=?", [str(temp_dir)])

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE base AS
        SELECT
          e.*,
          r.dependency_bucket,
          r.has_parent_dependency,
          r.has_start_after_dependency,
          r.dependency_count,
          CASE
            WHEN e.priority < 100 THEN 'p000_099'
            WHEN e.priority < 200 THEN 'p100_199'
            WHEN e.priority < 300 THEN 'p200_299'
            ELSE 'p300_plus'
          END AS priority_bucket,
          CASE
            WHEN COALESCE(e.instances, 0) <= 1 THEN 'i001'
            WHEN COALESCE(e.instances, 0) <= 5 THEN 'i002_005'
            WHEN COALESCE(e.instances, 0) <= 20 THEN 'i006_020'
            ELSE 'i021_plus'
          END AS instance_bucket,
          CASE
            WHEN COALESCE(e.total_cpu_request, 0.0) <= 0.10 THEN 'cpu_000_010'
            WHEN COALESCE(e.total_cpu_request, 0.0) <= 0.25 THEN 'cpu_010_025'
            WHEN COALESCE(e.total_cpu_request, 0.0) <= 0.50 THEN 'cpu_025_050'
            ELSE 'cpu_050_plus'
          END AS cpu_bucket,
          CASE
            WHEN COALESCE(e.total_memory_request, 0.0) <= 0.10 THEN 'mem_000_010'
            WHEN COALESCE(e.total_memory_request, 0.0) <= 0.25 THEN 'mem_010_025'
            WHEN COALESCE(e.total_memory_request, 0.0) <= 0.50 THEN 'mem_025_050'
            ELSE 'mem_050_plus'
          END AS memory_bucket,
          CASE
            WHEN e.collection_terminal_outcome IN ('EVICT','FAIL','KILL','LOST')
              OR COALESCE(e.bad_terminal_instances, 0) > 0
            THEN 1.0 ELSE 0.0
          END AS observed_bad_collection
        FROM read_parquet('{expanded_candidates}') AS e
        LEFT JOIN read_parquet('{risk_table}') AS r
          ON e.collection_id = r.collection_id
        WHERE e.schedule_win5 IS NOT NULL
          AND e.runtime_windows IS NOT NULL
          AND e.runtime_windows > 0
          AND e.total_cpu_request IS NOT NULL
          AND e.total_cpu_request > 0
          AND e.total_memory_request IS NOT NULL
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE train AS
        SELECT *
        FROM base
        WHERE schedule_win5 < {TEST_START_WIN5}
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE group_stats AS
        SELECT
          candidate_class,
          scheduling_class,
          scheduler,
          priority_bucket,
          dependency_bucket,
          instance_bucket,
          cpu_bucket,
          memory_bucket,
          COUNT(*) AS n_train,
          AVG(observed_bad_collection) AS bad_rate_train
        FROM train
        GROUP BY
          candidate_class,
          scheduling_class,
          scheduler,
          priority_bucket,
          dependency_bucket,
          instance_bucket,
          cpu_bucket,
          memory_bucket
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE coarse_stats AS
        SELECT
          candidate_class,
          scheduling_class,
          scheduler,
          priority_bucket,
          dependency_bucket,
          COUNT(*) AS n_train_coarse,
          AVG(observed_bad_collection) AS bad_rate_train_coarse
        FROM train
        GROUP BY
          candidate_class,
          scheduling_class,
          scheduler,
          priority_bucket,
          dependency_bucket
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE class_stats AS
        SELECT
          candidate_class,
          COUNT(*) AS n_train_class,
          AVG(observed_bad_collection) AS bad_rate_train_class
        FROM train
        GROUP BY candidate_class
        """
    )

    global_bad = con.execute("SELECT AVG(observed_bad_collection) FROM train").fetchone()[0]

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE scored AS
        SELECT
          b.*,
          COALESCE(g.n_train, 0) AS n_train_exact_group,
          COALESCE(c.n_train_coarse, 0) AS n_train_coarse_group,
          COALESCE(s.n_train_class, 0) AS n_train_class_group,
          CASE
            WHEN COALESCE(g.n_train, 0) >= 50 THEN g.bad_rate_train
            WHEN COALESCE(c.n_train_coarse, 0) >= 100 THEN c.bad_rate_train_coarse
            WHEN COALESCE(s.n_train_class, 0) >= 100 THEN s.bad_rate_train_class
            ELSE {global_bad}
          END AS predicted_bad_rate,
          CASE
            WHEN b.candidate_class IN (
              'batch_scheduler',
              'best_effort_queueable',
              'low_priority_flexible',
              'medium_flexibility'
            )
             AND b.dependency_bucket = 'independent'
             AND COALESCE(b.instances, 0) = 1
             AND COALESCE(b.total_cpu_request, 0.0) <= 0.10
             AND COALESCE(b.total_memory_request, 0.0) <= 0.10
             AND (
               CASE
                 WHEN COALESCE(g.n_train, 0) >= 50 THEN g.bad_rate_train
                 WHEN COALESCE(c.n_train_coarse, 0) >= 100 THEN c.bad_rate_train_coarse
                 WHEN COALESCE(s.n_train_class, 0) >= 100 THEN s.bad_rate_train_class
                 ELSE {global_bad}
               END
             ) <= 0.010
              THEN 'O0_visible_seed'
            WHEN b.candidate_class IN (
              'batch_scheduler',
              'best_effort_queueable',
              'low_priority_flexible',
              'medium_flexibility'
            )
             AND b.dependency_bucket = 'independent'
             AND COALESCE(b.instances, 0) BETWEEN 1 AND 5
             AND COALESCE(b.total_cpu_request, 0.0) <= 0.25
             AND COALESCE(b.total_memory_request, 0.0) <= 0.25
             AND (
               CASE
                 WHEN COALESCE(g.n_train, 0) >= 50 THEN g.bad_rate_train
                 WHEN COALESCE(c.n_train_coarse, 0) >= 100 THEN c.bad_rate_train_coarse
                 WHEN COALESCE(s.n_train_class, 0) >= 100 THEN s.bad_rate_train_class
                 ELSE {global_bad}
               END
             ) <= 0.030
              THEN 'O1_visible_small_independent'
            WHEN b.candidate_class IN (
              'batch_scheduler',
              'best_effort_queueable',
              'low_priority_flexible',
              'medium_flexibility'
            )
             AND b.dependency_bucket = 'light_dependency'
             AND COALESCE(b.instances, 0) BETWEEN 1 AND 5
             AND COALESCE(b.total_cpu_request, 0.0) <= 0.25
             AND COALESCE(b.total_memory_request, 0.0) <= 0.25
             AND (
               CASE
                 WHEN COALESCE(g.n_train, 0) >= 50 THEN g.bad_rate_train
                 WHEN COALESCE(c.n_train_coarse, 0) >= 100 THEN c.bad_rate_train_coarse
                 WHEN COALESCE(s.n_train_class, 0) >= 100 THEN s.bad_rate_train_class
                 ELSE {global_bad}
               END
             ) <= 0.030
              THEN 'O2_visible_small_dependency'
            WHEN b.candidate_class IN (
              'batch_scheduler',
              'best_effort_queueable',
              'low_priority_flexible',
              'medium_flexibility'
            )
             AND b.dependency_bucket = 'independent'
             AND COALESCE(b.instances, 0) BETWEEN 1 AND 20
             AND COALESCE(b.total_cpu_request, 0.0) <= 0.50
             AND COALESCE(b.total_memory_request, 0.0) <= 0.50
             AND (
               CASE
                 WHEN COALESCE(g.n_train, 0) >= 50 THEN g.bad_rate_train
                 WHEN COALESCE(c.n_train_coarse, 0) >= 100 THEN c.bad_rate_train_coarse
                 WHEN COALESCE(s.n_train_class, 0) >= 100 THEN s.bad_rate_train_class
                 ELSE {global_bad}
               END
             ) <= 0.050
              THEN 'O3_visible_medium_independent'
            WHEN b.candidate_class IN (
              'batch_scheduler',
              'best_effort_queueable',
              'low_priority_flexible',
              'medium_flexibility'
            )
             AND COALESCE(b.instances, 0) BETWEEN 1 AND 20
             AND COALESCE(b.total_cpu_request, 0.0) <= 0.50
             AND COALESCE(b.total_memory_request, 0.0) <= 0.50
             AND (
               CASE
                 WHEN COALESCE(g.n_train, 0) >= 50 THEN g.bad_rate_train
                 WHEN COALESCE(c.n_train_coarse, 0) >= 100 THEN c.bad_rate_train_coarse
                 WHEN COALESCE(s.n_train_class, 0) >= 100 THEN s.bad_rate_train_class
                 ELSE {global_bad}
               END
             ) <= 0.150
              THEN 'R1_retry_tolerant_signal'
            ELSE 'X_excluded'
          END AS operational_risk_tier
        FROM base AS b
        LEFT JOIN group_stats AS g
          ON b.candidate_class = g.candidate_class
         AND b.scheduling_class = g.scheduling_class
         AND b.scheduler = g.scheduler
         AND b.priority_bucket = g.priority_bucket
         AND b.dependency_bucket = g.dependency_bucket
         AND b.instance_bucket = g.instance_bucket
         AND b.cpu_bucket = g.cpu_bucket
         AND b.memory_bucket = g.memory_bucket
        LEFT JOIN coarse_stats AS c
          ON b.candidate_class = c.candidate_class
         AND b.scheduling_class = c.scheduling_class
         AND b.scheduler = c.scheduler
         AND b.priority_bucket = c.priority_bucket
         AND b.dependency_bucket = c.dependency_bucket
        LEFT JOIN class_stats AS s
          ON b.candidate_class = s.candidate_class
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE operational_jobs AS
        SELECT
          collection_id,
          operational_risk_tier AS risk_tier,
          CASE
            WHEN operational_risk_tier IN (
              'O0_visible_seed',
              'O1_visible_small_independent',
              'O2_visible_small_dependency',
              'O3_visible_medium_independent'
            ) THEN 'operational_predicted_clean'
            WHEN operational_risk_tier = 'R1_retry_tolerant_signal'
              THEN 'operational_retry_signal'
            ELSE 'excluded'
          END AS risk_tier_family,
          CASE operational_risk_tier
            WHEN 'O0_visible_seed' THEN 0
            WHEN 'O1_visible_small_independent' THEN 1
            WHEN 'O2_visible_small_dependency' THEN 2
            WHEN 'O3_visible_medium_independent' THEN 3
            WHEN 'R1_retry_tolerant_signal' THEN 90
            ELSE 99
          END AS risk_tier_order,
          candidate_class,
          dependency_graph_role,
          schedule_win5,
          runtime_windows,
          collection_runtime_sec,
          total_cpu_request,
          total_memory_request,
          instances,
          bad_terminal_instances,
          predicted_bad_rate AS harvest_risk_score,
          cpu_window_demand,
          memory_window_demand,
          collection_terminal_outcome,
          observed_bad_collection,
          predicted_bad_rate,
          n_train_exact_group,
          n_train_coarse_group,
          n_train_class_group,
          dependency_bucket
        FROM scored
        WHERE schedule_win5 BETWEEN 6480 AND 8784
          AND operational_risk_tier != 'X_excluded'
        """
    )

    _copy_sql(
        con,
        """
        SELECT
          risk_tier,
          risk_tier_order,
          risk_tier_family,
          COUNT(*) AS test_window_collections,
          SUM(instances) AS instances,
          SUM(cpu_window_demand) AS cpu_window_demand,
          SUM(memory_window_demand) AS memory_window_demand,
          SUM(bad_terminal_instances) AS bad_terminal_instances,
          SUM(bad_terminal_instances) / NULLIF(SUM(instances), 0) AS instance_bad_terminal_share,
          AVG(observed_bad_collection) AS collection_bad_terminal_share,
          AVG(predicted_bad_rate) AS mean_predicted_bad_rate,
          APPROX_QUANTILE(predicted_bad_rate, 0.50) AS p50_predicted_bad_rate,
          APPROX_QUANTILE(predicted_bad_rate, 0.90) AS p90_predicted_bad_rate,
          APPROX_QUANTILE(collection_runtime_sec, 0.50) AS p50_runtime_sec,
          APPROX_QUANTILE(collection_runtime_sec, 0.90) AS p90_runtime_sec,
          APPROX_QUANTILE(total_cpu_request, 0.90) AS p90_total_cpu_request
        FROM operational_jobs
        GROUP BY risk_tier, risk_tier_order, risk_tier_family
        ORDER BY risk_tier_order
        """,
        output_dir / "operational_risk_tier_summary.csv",
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE frontier_policies AS
        SELECT * FROM (
          VALUES
            ('P0_operational_seed', 0, 0),
            ('P1_plus_visible_small_independent', 1, 0),
            ('P2_plus_visible_small_dependency', 2, 0),
            ('P3_plus_visible_medium_independent', 3, 0),
            ('P4_plus_visible_retry_signal', 3, 1)
        ) AS t(frontier_policy, max_safe_tier_order, include_retry_signal)
        """
    )

    _copy_sql(
        con,
        """
        SELECT
          p.frontier_policy,
          COUNT(*) AS test_window_collections,
          SUM(e.instances) AS instances,
          SUM(e.cpu_window_demand) AS cpu_window_demand,
          SUM(e.memory_window_demand) AS memory_window_demand,
          SUM(e.bad_terminal_instances) AS bad_terminal_instances,
          SUM(e.bad_terminal_instances) / NULLIF(SUM(e.instances), 0) AS instance_bad_terminal_share,
          AVG(e.observed_bad_collection) AS collection_bad_terminal_share,
          AVG(e.predicted_bad_rate) AS mean_predicted_bad_rate,
          APPROX_QUANTILE(e.predicted_bad_rate, 0.90) AS p90_predicted_bad_rate,
          APPROX_QUANTILE(e.collection_runtime_sec, 0.50) AS p50_runtime_sec,
          APPROX_QUANTILE(e.collection_runtime_sec, 0.90) AS p90_runtime_sec,
          APPROX_QUANTILE(e.total_cpu_request, 0.90) AS p90_total_cpu_request
        FROM frontier_policies AS p
        INNER JOIN operational_jobs AS e
          ON (
            e.risk_tier_order <= p.max_safe_tier_order
            OR (
              p.include_retry_signal = 1
              AND e.risk_tier = 'R1_retry_tolerant_signal'
            )
          )
        GROUP BY p.frontier_policy
        ORDER BY p.frontier_policy
        """,
        output_dir / "operational_frontier_summary.csv",
    )

    _copy_sql(
        con,
        """
        SELECT
          CASE
            WHEN predicted_bad_rate <= 0.005 THEN '00_<=0.005'
            WHEN predicted_bad_rate <= 0.010 THEN '01_0.005_0.010'
            WHEN predicted_bad_rate <= 0.030 THEN '02_0.010_0.030'
            WHEN predicted_bad_rate <= 0.050 THEN '03_0.030_0.050'
            WHEN predicted_bad_rate <= 0.100 THEN '04_0.050_0.100'
            WHEN predicted_bad_rate <= 0.150 THEN '05_0.100_0.150'
            ELSE '06_>0.150'
          END AS predicted_bad_rate_bin,
          COUNT(*) AS test_window_collections,
          AVG(observed_bad_collection) AS observed_bad_collection_share,
          SUM(bad_terminal_instances) / NULLIF(SUM(instances), 0) AS instance_bad_terminal_share,
          AVG(predicted_bad_rate) AS mean_predicted_bad_rate,
          SUM(cpu_window_demand) AS cpu_window_demand
        FROM scored
        WHERE schedule_win5 BETWEEN 6480 AND 8784
        GROUP BY predicted_bad_rate_bin
        ORDER BY predicted_bad_rate_bin
        """,
        output_dir / "operational_predictor_bins.csv",
    )

    job_path = output_dir / "operational_frontier_jobs.csv"
    _copy_sql(
        con,
        """
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
        FROM operational_jobs
        ORDER BY schedule_win5, risk_tier_order, harvest_risk_score, cpu_window_demand
        """,
        job_path,
    )

    jobs = pd.read_csv(job_path)
    jobs["runtime_windows"] = jobs["runtime_windows"].clip(lower=1).astype(int)
    jobs["class_priority"] = jobs["candidate_class"].map(CLASS_PRIORITY).fillna(99)
    return jobs


def run(
    expanded_candidates: Path,
    risk_table: Path,
    cpu_opportunity_dir: Path,
    memory_opportunity_dir: Path,
    output_dir: Path,
    temp_dir: Path,
    opportunity_classes: list[str],
    placement_policy_names: list[str] | None,
    threads: int,
    memory_limit: str,
) -> None:
    jobs = build_operational_jobs(
        expanded_candidates,
        risk_table,
        output_dir,
        temp_dir,
        threads,
        memory_limit,
    )
    placement_policies = _selected_policies(placement_policy_names)

    summaries: list[dict[str, object]] = []
    samples: list[dict[str, object]] = []
    reason_rows: list[dict[str, object]] = []
    interval_cache = {}

    for opportunity_class in opportunity_classes:
        for policy in placement_policies:
            cache_key = (opportunity_class, policy.name)
            if cache_key not in interval_cache:
                interval_cache[cache_key] = load_intervals(
                    cpu_opportunity_dir,
                    memory_opportunity_dir,
                    opportunity_class,
                    policy,
                )
            intervals = interval_cache[cache_key]
            for frontier_name in [
                "P0_strict_seed",
                "P1_plus_clean_small_independent",
                "P2_plus_clean_leaf_dependency",
                "P3_plus_clean_medium_independent",
                "P4_plus_retry_signal",
            ]:
                summary, policy_samples, reasons = run_one(
                    jobs,
                    intervals,
                    frontier_name,
                    opportunity_class,
                    policy,
                )
                summary["frontier_policy"] = summary["frontier_policy"].replace(
                    "strict_seed", "operational_seed"
                )
                summary["frontier_policy"] = summary["frontier_policy"].replace(
                    "clean_small_independent", "visible_small_independent"
                )
                summary["frontier_policy"] = summary["frontier_policy"].replace(
                    "clean_leaf_dependency", "visible_small_dependency"
                )
                summary["frontier_policy"] = summary["frontier_policy"].replace(
                    "clean_medium_independent", "visible_medium_independent"
                )
                summary["frontier_policy"] = summary["frontier_policy"].replace(
                    "retry_signal", "visible_retry_signal"
                )
                summaries.append(summary)
                samples.extend(policy_samples[:500])
                for reason, count in Counter(reasons).items():
                    reason_rows.append(
                        {
                            "frontier_policy": summary["frontier_policy"],
                            "opportunity_class": opportunity_class,
                            "placement_policy": policy.name,
                            "reason": reason,
                            "jobs": count,
                        }
                    )

    pd.DataFrame(summaries).to_csv(output_dir / "operational_matching_summary.csv", index=False)
    pd.DataFrame(reason_rows).to_csv(
        output_dir / "operational_matching_unplaced_reasons.csv",
        index=False,
    )
    pd.DataFrame(samples).to_csv(output_dir / "operational_matching_sample.csv", index=False)
    print(f"wrote operational frontier outputs to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expanded-candidates", type=Path, default=DEFAULT_EXPANDED_CANDIDATES)
    parser.add_argument("--risk-table", type=Path, default=DEFAULT_RISK_TABLE)
    parser.add_argument("--cpu-opportunity-dir", type=Path, default=DEFAULT_CPU_OPPORTUNITY_DIR)
    parser.add_argument("--memory-opportunity-dir", type=Path, default=DEFAULT_MEMORY_OPPORTUNITY_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--temp-dir", type=Path, default=DEFAULT_TEMP)
    parser.add_argument(
        "--opportunity-classes",
        nargs="+",
        default=["10pct_1h"],
    )
    parser.add_argument("--placement-policies", nargs="+", default=["frontier_min_cpu_mem_max_exact"])
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--memory-limit", default="64GB")
    args = parser.parse_args()
    run(
        args.expanded_candidates,
        args.risk_table,
        args.cpu_opportunity_dir,
        args.memory_opportunity_dir,
        args.output,
        args.temp_dir,
        args.opportunity_classes,
        args.placement_policies,
        args.threads,
        args.memory_limit,
    )


if __name__ == "__main__":
    main()
