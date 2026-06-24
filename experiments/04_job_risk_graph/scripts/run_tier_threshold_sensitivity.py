#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

from risk_threshold_variants import (
    risk_tier_family_case,
    risk_tier_order_case,
    variant_union_sql,
)


DATA_ROOT = Path(
    "/media/jiang/0bfc75f7-326d-2d45-b017-7100903796b6/google-cluster-data-2019"
)
DEFAULT_EXPANDED_CANDIDATES = (
    DATA_ROOT / "_analysis" / "safeharvest" / "04_job_risk_graph" / "expanded_candidate_table.parquet"
)
DEFAULT_OUTPUT = DATA_ROOT / "_analysis" / "safeharvest" / "04_job_risk_graph" / "sensitivity"
DEFAULT_TEMP = Path("/mnt/clusterdata0/duckdb_tmp")


def _copy_sql(con: duckdb.DuckDBPyConnection, query: str, path: Path) -> None:
    con.execute(f"COPY ({query}) TO '{path}' (HEADER, DELIMITER ',')")


def run(
    expanded_candidates: Path,
    output_dir: Path,
    temp_dir: Path,
    variants: list[str] | None,
    threads: int,
    memory_limit: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={threads}")
    con.execute(f"PRAGMA memory_limit='{memory_limit}'")
    con.execute("PRAGMA temp_directory=?", [str(temp_dir)])

    tier_order = risk_tier_order_case("sensitivity_risk_tier")
    tier_family = risk_tier_family_case("sensitivity_risk_tier")
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE scored AS
        WITH unioned AS (
          {variant_union_sql(str(expanded_candidates), variants)}
        )
        SELECT
          *,
          {tier_order} AS sensitivity_risk_tier_order,
          {tier_family} AS sensitivity_risk_tier_family
        FROM unioned
        """
    )

    _copy_sql(
        con,
        """
        SELECT
          threshold_variant,
          sensitivity_risk_tier AS risk_tier,
          sensitivity_risk_tier_order AS risk_tier_order,
          sensitivity_risk_tier_family AS risk_tier_family,
          COUNT(*) AS collections,
          SUM(CASE WHEN schedule_win5 BETWEEN 6480 AND 8784 THEN 1 ELSE 0 END) AS test_window_collections,
          SUM(instances) AS instances,
          SUM(finished_instances) AS finished_instances,
          SUM(bad_terminal_instances) AS bad_terminal_instances,
          SUM(no_terminal_instances) AS no_terminal_instances,
          SUM(cpu_window_demand) AS cpu_window_demand,
          SUM(memory_window_demand) AS memory_window_demand,
          AVG(CASE WHEN collection_terminal_outcome = 'FINISH' THEN 1.0 ELSE 0.0 END) AS collection_finish_share,
          AVG(CASE WHEN collection_terminal_outcome IN ('EVICT','FAIL','KILL','LOST') THEN 1.0 ELSE 0.0 END) AS collection_bad_terminal_share,
          SUM(bad_terminal_instances) / NULLIF(SUM(instances), 0) AS instance_bad_terminal_share,
          APPROX_QUANTILE(collection_runtime_sec, 0.50) AS p50_runtime_sec,
          APPROX_QUANTILE(collection_runtime_sec, 0.90) AS p90_runtime_sec,
          APPROX_QUANTILE(total_cpu_request, 0.50) AS p50_total_cpu_request,
          APPROX_QUANTILE(total_cpu_request, 0.90) AS p90_total_cpu_request,
          APPROX_QUANTILE(total_memory_request, 0.50) AS p50_total_memory_request,
          APPROX_QUANTILE(total_memory_request, 0.90) AS p90_total_memory_request
        FROM scored
        GROUP BY
          threshold_variant,
          threshold_variant_order,
          sensitivity_risk_tier,
          sensitivity_risk_tier_order,
          sensitivity_risk_tier_family
        ORDER BY threshold_variant_order, sensitivity_risk_tier_order, collections DESC
        """,
        output_dir / "risk_tier_threshold_sensitivity.csv",
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE frontier_policies AS
        SELECT * FROM (
          VALUES
            ('P0_strict_seed', 0, 0),
            ('P1_plus_clean_small_independent', 1, 0),
            ('P2_plus_clean_leaf_dependency', 2, 0),
            ('P3_plus_clean_medium_independent', 3, 0),
            ('P4_plus_retry_signal', 3, 1)
        ) AS t(frontier_policy, max_safe_tier_order, include_retry_signal)
        """
    )

    _copy_sql(
        con,
        """
        SELECT
          e.threshold_variant,
          p.frontier_policy,
          COUNT(*) AS collections,
          SUM(CASE WHEN e.schedule_win5 BETWEEN 6480 AND 8784 THEN 1 ELSE 0 END) AS test_window_collections,
          SUM(e.instances) AS instances,
          SUM(e.cpu_window_demand) AS cpu_window_demand,
          SUM(e.memory_window_demand) AS memory_window_demand,
          SUM(e.bad_terminal_instances) AS bad_terminal_instances,
          SUM(e.bad_terminal_instances) / NULLIF(SUM(e.instances), 0) AS instance_bad_terminal_share,
          AVG(CASE WHEN e.collection_terminal_outcome = 'FINISH' THEN 1.0 ELSE 0.0 END) AS collection_finish_share,
          AVG(CASE WHEN e.collection_terminal_outcome IN ('EVICT','FAIL','KILL','LOST') THEN 1.0 ELSE 0.0 END) AS collection_bad_terminal_share,
          APPROX_QUANTILE(e.collection_runtime_sec, 0.50) AS p50_runtime_sec,
          APPROX_QUANTILE(e.collection_runtime_sec, 0.90) AS p90_runtime_sec,
          APPROX_QUANTILE(e.total_cpu_request, 0.90) AS p90_total_cpu_request,
          APPROX_QUANTILE(e.total_memory_request, 0.90) AS p90_total_memory_request
        FROM scored AS e
        INNER JOIN frontier_policies AS p
          ON (
            e.sensitivity_risk_tier_order <= p.max_safe_tier_order
            OR (
              p.include_retry_signal = 1
              AND e.sensitivity_risk_tier = 'R1_retry_tolerant_signal'
            )
          )
        GROUP BY e.threshold_variant, e.threshold_variant_order, p.frontier_policy
        ORDER BY e.threshold_variant_order, p.frontier_policy
        """,
        output_dir / "frontier_threshold_sensitivity.csv",
    )

    _copy_sql(
        con,
        """
        SELECT
          threshold_variant,
          risk_tier AS baseline_risk_tier,
          sensitivity_risk_tier AS sensitivity_risk_tier,
          COUNT(*) AS collections,
          SUM(CASE WHEN schedule_win5 BETWEEN 6480 AND 8784 THEN 1 ELSE 0 END) AS test_window_collections,
          SUM(cpu_window_demand) AS cpu_window_demand,
          SUM(memory_window_demand) AS memory_window_demand,
          SUM(bad_terminal_instances) AS bad_terminal_instances,
          SUM(bad_terminal_instances) / NULLIF(SUM(instances), 0) AS instance_bad_terminal_share
        FROM scored
        GROUP BY threshold_variant, threshold_variant_order, risk_tier, sensitivity_risk_tier
        ORDER BY threshold_variant_order, baseline_risk_tier, sensitivity_risk_tier
        """,
        output_dir / "risk_threshold_shift_matrix.csv",
    )

    _copy_sql(
        con,
        """
        SELECT
          threshold_variant,
          sensitivity_risk_tier AS risk_tier,
          candidate_class,
          COUNT(*) AS collections,
          SUM(CASE WHEN schedule_win5 BETWEEN 6480 AND 8784 THEN 1 ELSE 0 END) AS test_window_collections,
          SUM(cpu_window_demand) AS cpu_window_demand,
          SUM(memory_window_demand) AS memory_window_demand,
          SUM(bad_terminal_instances) AS bad_terminal_instances,
          SUM(bad_terminal_instances) / NULLIF(SUM(instances), 0) AS instance_bad_terminal_share
        FROM scored
        GROUP BY threshold_variant, threshold_variant_order, sensitivity_risk_tier, candidate_class
        ORDER BY threshold_variant_order, MIN(sensitivity_risk_tier_order), risk_tier, collections DESC
        """,
        output_dir / "risk_threshold_sensitivity_by_class.csv",
    )

    print(f"wrote risk-tier threshold sensitivity outputs to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expanded-candidates", type=Path, default=DEFAULT_EXPANDED_CANDIDATES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--temp-dir", type=Path, default=DEFAULT_TEMP)
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--memory-limit", default="64GB")
    args = parser.parse_args()
    run(
        args.expanded_candidates,
        args.output,
        args.temp_dir,
        args.variants,
        args.threads,
        args.memory_limit,
    )


if __name__ == "__main__":
    main()
