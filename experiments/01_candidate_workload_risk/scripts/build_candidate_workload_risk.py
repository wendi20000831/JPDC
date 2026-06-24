#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


DATA_ROOT = Path(
    "/media/jiang/0bfc75f7-326d-2d45-b017-7100903796b6/google-cluster-data-2019"
)
DEFAULT_INPUT = DATA_ROOT / "cell_a" / "collection_events-*.parquet.gz"
DEFAULT_OUTPUT = DATA_ROOT / "_analysis" / "safeharvest" / "01_candidate_workload_risk"
DEFAULT_TEMP = Path("/mnt/clusterdata0/duckdb_tmp")


def run(input_glob: str, output_dir: Path, temp_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute("PRAGMA threads=16")
    con.execute("PRAGMA memory_limit='64GB'")
    con.execute("PRAGMA temp_directory=?", [str(temp_dir)])

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE lifecycle AS
        WITH per_collection AS (
          SELECT
            collection_id,
            arg_min(collection_type, time) AS collection_type,
            arg_min(priority, time) AS priority,
            arg_min(scheduling_class, time) AS scheduling_class,
            arg_min(scheduler, time) AS scheduler,
            arg_min(user, time) AS user_id,
            arg_min(collection_logical_name, time) AS collection_logical_name,
            arg_min(parent_collection_id, time) AS parent_collection_id,
            max(array_length(start_after_collection_ids)) AS dependency_count,
            arg_min(max_per_machine, time) AS max_per_machine,
            arg_min(max_per_switch, time) AS max_per_switch,
            arg_min(vertical_scaling, time) AS vertical_scaling,
            MIN(time) AS first_event_time,
            MAX(time) AS last_event_time,
            MIN(time) FILTER (WHERE type = 0) AS submit_time,
            MIN(time) FILTER (WHERE type = 1) AS queue_time,
            MIN(time) FILTER (WHERE type = 2) AS enable_time,
            MIN(time) FILTER (WHERE type = 3) AS schedule_time,
            MIN(time) FILTER (WHERE type IN (4, 5, 6, 7, 8)) AS terminal_time,
            arg_min(type, time) FILTER (WHERE type IN (4, 5, 6, 7, 8)) AS terminal_type,
            COUNT(*) AS event_count
          FROM read_parquet('{input_glob}')
          WHERE collection_type = 0
          GROUP BY collection_id
        )
        SELECT
          *,
          CASE terminal_type
            WHEN 4 THEN 'EVICT'
            WHEN 5 THEN 'FAIL'
            WHEN 6 THEN 'FINISH'
            WHEN 7 THEN 'KILL'
            WHEN 8 THEN 'LOST'
            ELSE 'NO_TERMINAL'
          END AS terminal_outcome,
          CASE
            WHEN scheduler = 1 THEN 'batch_scheduler'
            WHEN priority < 100 AND scheduling_class <= 1 THEN 'low_priority_flexible'
            WHEN priority < 200 AND scheduling_class <= 1 THEN 'best_effort_queueable'
            WHEN priority < 300 AND scheduling_class <= 2 THEN 'medium_flexibility'
            ELSE 'latency_sensitive_or_high_priority'
          END AS candidate_class,
          CASE
            WHEN schedule_time IS NOT NULL AND submit_time IS NOT NULL
              THEN (schedule_time - submit_time) / 1000000.0
          END AS submit_to_schedule_sec,
          CASE
            WHEN schedule_time IS NOT NULL AND enable_time IS NOT NULL
              THEN (schedule_time - enable_time) / 1000000.0
          END AS enable_to_schedule_sec,
          CASE
            WHEN terminal_time IS NOT NULL AND schedule_time IS NOT NULL
             AND terminal_time >= schedule_time
              THEN (terminal_time - schedule_time) / 1000000.0
          END AS runtime_sec,
          CASE
            WHEN parent_collection_id IS NOT NULL AND parent_collection_id != 0 THEN 1
            ELSE 0
          END AS has_parent_dependency,
          CASE
            WHEN COALESCE(dependency_count, 0) > 0 THEN 1
            ELSE 0
          END AS has_start_after_dependency
        FROM per_collection
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE risk_table AS
        SELECT
          *,
          CASE
            WHEN runtime_sec IS NULL THEN 'unknown'
            WHEN runtime_sec <= 300 THEN '00_<=5m'
            WHEN runtime_sec <= 1800 THEN '01_5m_30m'
            WHEN runtime_sec <= 3600 THEN '02_30m_1h'
            WHEN runtime_sec <= 14400 THEN '03_1h_4h'
            WHEN runtime_sec <= 28800 THEN '04_4h_8h'
            ELSE '05_>8h'
          END AS runtime_bucket,
          CASE
            WHEN COALESCE(dependency_count, 0) = 0
             AND COALESCE(has_parent_dependency, 0) = 0 THEN 'independent'
            WHEN COALESCE(dependency_count, 0) <= 2 THEN 'light_dependency'
            ELSE 'heavy_dependency'
          END AS dependency_bucket,
          CASE
            WHEN terminal_type = 6 THEN 0.0
            WHEN terminal_type IN (4, 7, 8) THEN 1.0
            WHEN terminal_type = 5 THEN 1.0
            ELSE 0.5
          END AS terminal_risk_score,
          CASE
            WHEN runtime_sec IS NULL THEN 0.5
            WHEN runtime_sec <= 14400 THEN 0.0
            WHEN runtime_sec <= 28800 THEN 0.4
            ELSE 0.8
          END AS runtime_risk_score,
          CASE
            WHEN COALESCE(dependency_count, 0) = 0
             AND COALESCE(has_parent_dependency, 0) = 0 THEN 0.0
            WHEN COALESCE(dependency_count, 0) <= 2 THEN 0.3
            ELSE 0.7
          END AS dependency_risk_score,
          CASE
            WHEN scheduling_class <= 1 AND priority < 200 THEN 0.0
            WHEN scheduling_class <= 2 AND priority < 300 THEN 0.3
            ELSE 0.8
          END AS priority_sensitivity_score
        FROM lifecycle
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE scored AS
        SELECT
          *,
          (
            0.35 * terminal_risk_score
            + 0.25 * runtime_risk_score
            + 0.20 * dependency_risk_score
            + 0.20 * priority_sensitivity_score
          ) AS harvest_risk_score,
          CASE
            WHEN terminal_outcome = 'FINISH'
             AND runtime_sec IS NOT NULL
             AND runtime_sec <= 14400
             AND COALESCE(dependency_count, 0) = 0
             AND has_parent_dependency = 0
             AND (
               scheduler = 1
               OR (priority < 200 AND scheduling_class <= 1)
             )
            THEN 1 ELSE 0
          END AS safeharvest_seed_candidate
        FROM risk_table
        """
    )

    con.execute(
        f"""
        COPY scored TO '{output_dir / "candidate_workload_risk_table.parquet"}'
          (FORMAT 'parquet', COMPRESSION 'zstd')
        """
    )

    con.execute(
        f"""
        COPY (
          SELECT
            candidate_class,
            COUNT(*) AS collections,
            SUM(safeharvest_seed_candidate) AS seed_candidates,
            AVG(safeharvest_seed_candidate) AS seed_candidate_share,
            AVG(harvest_risk_score) AS avg_harvest_risk_score,
            APPROX_QUANTILE(harvest_risk_score, 0.50) AS p50_harvest_risk_score,
            APPROX_QUANTILE(harvest_risk_score, 0.90) AS p90_harvest_risk_score,
            AVG(CASE WHEN terminal_outcome = 'FINISH' THEN 1.0 ELSE 0.0 END) AS finish_share,
            AVG(CASE WHEN terminal_outcome IN ('EVICT','FAIL','KILL','LOST') THEN 1.0 ELSE 0.0 END) AS bad_terminal_share,
            AVG(CASE WHEN terminal_outcome = 'NO_TERMINAL' THEN 1.0 ELSE 0.0 END) AS no_terminal_share,
            AVG(CASE WHEN dependency_bucket = 'independent' THEN 1.0 ELSE 0.0 END) AS independent_share,
            APPROX_QUANTILE(runtime_sec, 0.50) AS p50_runtime_sec,
            APPROX_QUANTILE(runtime_sec, 0.90) AS p90_runtime_sec,
            APPROX_QUANTILE(runtime_sec, 0.99) AS p99_runtime_sec,
            APPROX_QUANTILE(submit_to_schedule_sec, 0.50) AS p50_queue_sec,
            APPROX_QUANTILE(submit_to_schedule_sec, 0.90) AS p90_queue_sec,
            APPROX_QUANTILE(submit_to_schedule_sec, 0.99) AS p99_queue_sec
          FROM scored
          GROUP BY candidate_class
          ORDER BY seed_candidates DESC, collections DESC
        ) TO '{output_dir / "candidate_class_summary.csv"}' (HEADER, DELIMITER ',')
        """
    )

    con.execute(
        f"""
        COPY (
          SELECT
            candidate_class,
            runtime_bucket,
            dependency_bucket,
            COUNT(*) AS collections,
            SUM(safeharvest_seed_candidate) AS seed_candidates,
            AVG(harvest_risk_score) AS avg_harvest_risk_score,
            AVG(CASE WHEN terminal_outcome = 'FINISH' THEN 1.0 ELSE 0.0 END) AS finish_share,
            AVG(CASE WHEN terminal_outcome IN ('EVICT','FAIL','KILL','LOST') THEN 1.0 ELSE 0.0 END) AS bad_terminal_share,
            APPROX_QUANTILE(runtime_sec, 0.50) AS p50_runtime_sec,
            APPROX_QUANTILE(runtime_sec, 0.90) AS p90_runtime_sec
          FROM scored
          GROUP BY candidate_class, runtime_bucket, dependency_bucket
          ORDER BY seed_candidates DESC, collections DESC
        ) TO '{output_dir / "candidate_risk_matrix.csv"}' (HEADER, DELIMITER ',')
        """
    )

    con.execute(
        f"""
        COPY (
          SELECT
            candidate_class,
            terminal_outcome,
            COUNT(*) AS collections,
            AVG(harvest_risk_score) AS avg_harvest_risk_score,
            APPROX_QUANTILE(runtime_sec, 0.50) AS p50_runtime_sec,
            APPROX_QUANTILE(runtime_sec, 0.90) AS p90_runtime_sec
          FROM scored
          GROUP BY candidate_class, terminal_outcome
          ORDER BY candidate_class, collections DESC
        ) TO '{output_dir / "candidate_terminal_outcome_matrix.csv"}' (HEADER, DELIMITER ',')
        """
    )

    con.execute(
        f"""
        COPY (
          SELECT
            candidate_class,
            dependency_bucket,
            COUNT(*) AS collections,
            SUM(safeharvest_seed_candidate) AS seed_candidates,
            AVG(harvest_risk_score) AS avg_harvest_risk_score,
            AVG(CASE WHEN terminal_outcome = 'FINISH' THEN 1.0 ELSE 0.0 END) AS finish_share,
            APPROX_QUANTILE(runtime_sec, 0.50) AS p50_runtime_sec,
            APPROX_QUANTILE(runtime_sec, 0.90) AS p90_runtime_sec
          FROM scored
          GROUP BY candidate_class, dependency_bucket
          ORDER BY candidate_class, dependency_bucket
        ) TO '{output_dir / "candidate_dependency_summary.csv"}' (HEADER, DELIMITER ',')
        """
    )

    con.execute(
        f"""
        COPY (
          SELECT
            COUNT(*) AS collections,
            SUM(safeharvest_seed_candidate) AS seed_candidates,
            AVG(safeharvest_seed_candidate) AS seed_candidate_share,
            SUM(CASE WHEN safeharvest_seed_candidate = 1 AND runtime_sec <= 1800 THEN 1 ELSE 0 END) AS fit_30m,
            SUM(CASE WHEN safeharvest_seed_candidate = 1 AND runtime_sec <= 3600 THEN 1 ELSE 0 END) AS fit_1h,
            SUM(CASE WHEN safeharvest_seed_candidate = 1 AND runtime_sec <= 14400 THEN 1 ELSE 0 END) AS fit_4h,
            APPROX_QUANTILE(CASE WHEN safeharvest_seed_candidate = 1 THEN runtime_sec END, 0.50) AS seed_p50_runtime_sec,
            APPROX_QUANTILE(CASE WHEN safeharvest_seed_candidate = 1 THEN runtime_sec END, 0.90) AS seed_p90_runtime_sec,
            APPROX_QUANTILE(CASE WHEN safeharvest_seed_candidate = 1 THEN runtime_sec END, 0.99) AS seed_p99_runtime_sec,
            APPROX_QUANTILE(CASE WHEN safeharvest_seed_candidate = 1 THEN submit_to_schedule_sec END, 0.50) AS seed_p50_queue_sec,
            APPROX_QUANTILE(CASE WHEN safeharvest_seed_candidate = 1 THEN submit_to_schedule_sec END, 0.90) AS seed_p90_queue_sec,
            APPROX_QUANTILE(CASE WHEN safeharvest_seed_candidate = 1 THEN submit_to_schedule_sec END, 0.99) AS seed_p99_queue_sec
          FROM scored
        ) TO '{output_dir / "safeharvest_candidate_summary.csv"}' (HEADER, DELIMITER ',')
        """
    )

    con.execute(
        f"""
        COPY (
          SELECT
            candidate_class,
            runtime_bucket,
            COUNT(*) AS seed_candidates,
            APPROX_QUANTILE(runtime_sec, 0.50) AS p50_runtime_sec,
            APPROX_QUANTILE(runtime_sec, 0.90) AS p90_runtime_sec,
            APPROX_QUANTILE(submit_to_schedule_sec, 0.50) AS p50_queue_sec,
            APPROX_QUANTILE(submit_to_schedule_sec, 0.90) AS p90_queue_sec
          FROM scored
          WHERE safeharvest_seed_candidate = 1
          GROUP BY candidate_class, runtime_bucket
          ORDER BY candidate_class, runtime_bucket
        ) TO '{output_dir / "candidate_runtime_fit_summary.csv"}' (HEADER, DELIMITER ',')
        """
    )

    con.execute(
        f"""
        COPY (
          SELECT
            collection_id,
            candidate_class,
            priority,
            scheduling_class,
            scheduler,
            terminal_outcome,
            runtime_bucket,
            dependency_bucket,
            submit_to_schedule_sec,
            runtime_sec,
            harvest_risk_score,
            safeharvest_seed_candidate
          FROM scored
          WHERE candidate_class IN ('batch_scheduler', 'low_priority_flexible', 'best_effort_queueable')
          ORDER BY safeharvest_seed_candidate DESC, harvest_risk_score ASC, runtime_sec ASC
          LIMIT 20000
        ) TO '{output_dir / "candidate_workload_sample.csv"}' (HEADER, DELIMITER ',')
        """
    )

    print(f"wrote candidate workload risk outputs to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--temp-dir", type=Path, default=DEFAULT_TEMP)
    args = parser.parse_args()
    run(args.input_glob, args.output, args.temp_dir)


if __name__ == "__main__":
    main()
