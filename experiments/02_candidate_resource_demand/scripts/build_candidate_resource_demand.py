#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


DATA_ROOT = Path(
    "/media/jiang/0bfc75f7-326d-2d45-b017-7100903796b6/google-cluster-data-2019"
)
DEFAULT_INSTANCE_EVENTS = DATA_ROOT / "cell_a" / "instance_events-*.parquet.gz"
DEFAULT_CANDIDATES = (
    DATA_ROOT
    / "_analysis"
    / "safeharvest"
    / "01_candidate_workload_risk"
    / "candidate_workload_risk_table.parquet"
)
DEFAULT_OUTPUT = DATA_ROOT / "_analysis" / "safeharvest" / "02_candidate_resource_demand"
DEFAULT_TEMP = Path("/mnt/clusterdata0/duckdb_tmp")


def run(
    instance_glob: str,
    candidate_table: Path,
    output_dir: Path,
    temp_dir: Path,
    threads: int,
    memory_limit: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={threads}")
    con.execute(f"PRAGMA memory_limit='{memory_limit}'")
    con.execute("PRAGMA temp_directory=?", [str(temp_dir)])

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE seed_candidates AS
        SELECT
          collection_id,
          candidate_class,
          priority,
          scheduling_class,
          scheduler,
          terminal_outcome AS collection_terminal_outcome,
          runtime_bucket AS collection_runtime_bucket,
          dependency_bucket,
          submit_to_schedule_sec AS collection_queue_sec,
          runtime_sec AS collection_runtime_sec,
          harvest_risk_score
        FROM read_parquet('{candidate_table}')
        WHERE safeharvest_seed_candidate = 1
        """
    )

    con.execute("CREATE INDEX seed_candidate_collection_idx ON seed_candidates(collection_id)")

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE instance_lifecycle AS
        WITH filtered AS (
          SELECT
            e.collection_id,
            e.instance_index,
            e.time,
            e.type,
            e.machine_id,
            e.resource_request.cpus AS request_cpus,
            e.resource_request.memory AS request_memory
          FROM read_parquet('{instance_glob}') AS e
          INNER JOIN seed_candidates AS c
            ON e.collection_id = c.collection_id
        ),
        per_instance AS (
          SELECT
            collection_id,
            instance_index,
            arg_min(request_cpus, time) FILTER (WHERE request_cpus IS NOT NULL) AS first_request_cpus,
            arg_min(request_memory, time) FILTER (WHERE request_memory IS NOT NULL) AS first_request_memory,
            max(request_cpus) AS max_request_cpus,
            max(request_memory) AS max_request_memory,
            min(time) AS first_event_time,
            max(time) AS last_event_time,
            min(time) FILTER (WHERE type = 0) AS submit_time,
            min(time) FILTER (WHERE type = 3) AS schedule_time,
            min(time) FILTER (WHERE type IN (4, 5, 6, 7, 8)) AS terminal_time,
            arg_min(type, time) FILTER (WHERE type IN (4, 5, 6, 7, 8)) AS terminal_type,
            count(*) AS instance_event_count,
            count(DISTINCT machine_id) FILTER (WHERE machine_id IS NOT NULL) AS touched_machines
          FROM filtered
          GROUP BY collection_id, instance_index
        )
        SELECT
          *,
          COALESCE(max_request_cpus, first_request_cpus, 0.0) AS request_cpus,
          COALESCE(max_request_memory, first_request_memory, 0.0) AS request_memory,
          CASE terminal_type
            WHEN 4 THEN 'EVICT'
            WHEN 5 THEN 'FAIL'
            WHEN 6 THEN 'FINISH'
            WHEN 7 THEN 'KILL'
            WHEN 8 THEN 'LOST'
            ELSE 'NO_TERMINAL'
          END AS instance_terminal_outcome,
          CASE
            WHEN terminal_time IS NOT NULL AND schedule_time IS NOT NULL
             AND terminal_time >= schedule_time
              THEN (terminal_time - schedule_time) / 1000000.0
          END AS instance_runtime_sec
        FROM per_instance
        """
    )

    con.execute(
        f"""
        COPY (
          SELECT
            c.collection_id,
            c.candidate_class,
            c.priority,
            c.scheduling_class,
            c.scheduler,
            c.collection_terminal_outcome,
            c.collection_runtime_bucket,
            c.dependency_bucket,
            c.collection_queue_sec,
            c.collection_runtime_sec,
            c.harvest_risk_score,
            count(i.instance_index) AS instances,
            sum(i.request_cpus) AS total_cpu_request,
            sum(i.request_memory) AS total_memory_request,
            max(i.request_cpus) AS max_instance_cpu_request,
            max(i.request_memory) AS max_instance_memory_request,
            approx_quantile(i.request_cpus, 0.50) AS p50_instance_cpu_request,
            approx_quantile(i.request_cpus, 0.90) AS p90_instance_cpu_request,
            approx_quantile(i.request_memory, 0.50) AS p50_instance_memory_request,
            approx_quantile(i.request_memory, 0.90) AS p90_instance_memory_request,
            sum(CASE WHEN i.instance_terminal_outcome = 'FINISH' THEN 1 ELSE 0 END) AS finished_instances,
            sum(CASE WHEN i.instance_terminal_outcome IN ('EVICT','FAIL','KILL','LOST') THEN 1 ELSE 0 END) AS bad_terminal_instances,
            sum(CASE WHEN i.instance_terminal_outcome = 'NO_TERMINAL' THEN 1 ELSE 0 END) AS no_terminal_instances,
            approx_quantile(i.instance_runtime_sec, 0.50) AS p50_instance_runtime_sec,
            approx_quantile(i.instance_runtime_sec, 0.90) AS p90_instance_runtime_sec,
            sum(CASE WHEN i.touched_machines > 1 THEN 1 ELSE 0 END) AS multi_machine_instances
          FROM seed_candidates AS c
          LEFT JOIN instance_lifecycle AS i
            ON c.collection_id = i.collection_id
          GROUP BY
            c.collection_id,
            c.candidate_class,
            c.priority,
            c.scheduling_class,
            c.scheduler,
            c.collection_terminal_outcome,
            c.collection_runtime_bucket,
            c.dependency_bucket,
            c.collection_queue_sec,
            c.collection_runtime_sec,
            c.harvest_risk_score
        ) TO '{output_dir / "candidate_resource_demand_table.parquet"}'
          (FORMAT 'parquet', COMPRESSION 'zstd')
        """
    )

    demand_table = output_dir / "candidate_resource_demand_table.parquet"

    con.execute(
        f"""
        COPY (
          SELECT
            count(*) AS collections,
            sum(instances) AS instances,
            approx_quantile(instances, 0.50) AS p50_instances,
            approx_quantile(instances, 0.90) AS p90_instances,
            approx_quantile(instances, 0.99) AS p99_instances,
            approx_quantile(total_cpu_request, 0.50) AS p50_total_cpu_request,
            approx_quantile(total_cpu_request, 0.90) AS p90_total_cpu_request,
            approx_quantile(total_cpu_request, 0.99) AS p99_total_cpu_request,
            approx_quantile(total_memory_request, 0.50) AS p50_total_memory_request,
            approx_quantile(total_memory_request, 0.90) AS p90_total_memory_request,
            approx_quantile(total_memory_request, 0.99) AS p99_total_memory_request,
            sum(CASE WHEN total_cpu_request <= 0.10 THEN 1 ELSE 0 END) AS fit_cpu_0p10,
            sum(CASE WHEN total_cpu_request <= 0.25 THEN 1 ELSE 0 END) AS fit_cpu_0p25,
            sum(CASE WHEN total_cpu_request <= 0.50 THEN 1 ELSE 0 END) AS fit_cpu_0p50,
            sum(CASE WHEN total_cpu_request <= 1.00 THEN 1 ELSE 0 END) AS fit_cpu_1p00,
            sum(finished_instances) / nullif(sum(instances), 0) AS instance_finish_share,
            sum(bad_terminal_instances) / nullif(sum(instances), 0) AS instance_bad_terminal_share,
            sum(no_terminal_instances) / nullif(sum(instances), 0) AS instance_no_terminal_share
          FROM read_parquet('{demand_table}')
        ) TO '{output_dir / "resource_demand_overall.csv"}' (HEADER, DELIMITER ',')
        """
    )

    con.execute(
        f"""
        COPY (
          SELECT
            candidate_class,
            count(*) AS collections,
            sum(instances) AS instances,
            approx_quantile(instances, 0.50) AS p50_instances,
            approx_quantile(instances, 0.90) AS p90_instances,
            approx_quantile(total_cpu_request, 0.50) AS p50_total_cpu_request,
            approx_quantile(total_cpu_request, 0.90) AS p90_total_cpu_request,
            approx_quantile(total_cpu_request, 0.99) AS p99_total_cpu_request,
            approx_quantile(total_memory_request, 0.50) AS p50_total_memory_request,
            approx_quantile(total_memory_request, 0.90) AS p90_total_memory_request,
            sum(CASE WHEN total_cpu_request <= 0.10 THEN 1 ELSE 0 END) AS fit_cpu_0p10,
            sum(CASE WHEN total_cpu_request <= 0.25 THEN 1 ELSE 0 END) AS fit_cpu_0p25,
            sum(CASE WHEN total_cpu_request <= 0.50 THEN 1 ELSE 0 END) AS fit_cpu_0p50,
            sum(CASE WHEN total_cpu_request <= 1.00 THEN 1 ELSE 0 END) AS fit_cpu_1p00,
            sum(finished_instances) / nullif(sum(instances), 0) AS instance_finish_share,
            sum(bad_terminal_instances) / nullif(sum(instances), 0) AS instance_bad_terminal_share
          FROM read_parquet('{demand_table}')
          GROUP BY candidate_class
          ORDER BY collections DESC
        ) TO '{output_dir / "resource_demand_by_class.csv"}' (HEADER, DELIMITER ',')
        """
    )

    con.execute(
        f"""
        COPY (
          SELECT
            CASE
              WHEN total_cpu_request IS NULL THEN 'unknown'
              WHEN total_cpu_request <= 0.05 THEN '00_<=0.05'
              WHEN total_cpu_request <= 0.10 THEN '01_0.05_0.10'
              WHEN total_cpu_request <= 0.25 THEN '02_0.10_0.25'
              WHEN total_cpu_request <= 0.50 THEN '03_0.25_0.50'
              WHEN total_cpu_request <= 1.00 THEN '04_0.50_1.00'
              WHEN total_cpu_request <= 2.00 THEN '05_1.00_2.00'
              ELSE '06_>2.00'
            END AS total_cpu_request_bucket,
            candidate_class,
            count(*) AS collections,
            approx_quantile(collection_runtime_sec, 0.50) AS p50_collection_runtime_sec,
            approx_quantile(collection_runtime_sec, 0.90) AS p90_collection_runtime_sec,
            approx_quantile(instances, 0.50) AS p50_instances
          FROM read_parquet('{demand_table}')
          GROUP BY total_cpu_request_bucket, candidate_class
          ORDER BY total_cpu_request_bucket, collections DESC
        ) TO '{output_dir / "resource_demand_buckets.csv"}' (HEADER, DELIMITER ',')
        """
    )

    con.execute(
        f"""
        COPY (
          SELECT
            CASE
              WHEN instances IS NULL THEN 'unknown'
              WHEN instances <= 1 THEN '00_1'
              WHEN instances <= 5 THEN '01_2_5'
              WHEN instances <= 20 THEN '02_6_20'
              WHEN instances <= 100 THEN '03_21_100'
              WHEN instances <= 1000 THEN '04_101_1000'
              ELSE '05_>1000'
            END AS instance_count_bucket,
            candidate_class,
            count(*) AS collections,
            approx_quantile(total_cpu_request, 0.50) AS p50_total_cpu_request,
            approx_quantile(total_cpu_request, 0.90) AS p90_total_cpu_request,
            approx_quantile(collection_runtime_sec, 0.50) AS p50_collection_runtime_sec
          FROM read_parquet('{demand_table}')
          GROUP BY instance_count_bucket, candidate_class
          ORDER BY instance_count_bucket, collections DESC
        ) TO '{output_dir / "instance_count_buckets.csv"}' (HEADER, DELIMITER ',')
        """
    )

    con.execute(
        f"""
        COPY (
          SELECT *
          FROM read_parquet('{demand_table}')
          ORDER BY total_cpu_request ASC NULLS LAST, collection_runtime_sec ASC
          LIMIT 20000
        ) TO '{output_dir / "candidate_resource_sample.csv"}' (HEADER, DELIMITER ',')
        """
    )

    print(f"wrote candidate resource demand outputs to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance-glob", default=str(DEFAULT_INSTANCE_EVENTS))
    parser.add_argument("--candidate-table", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--temp-dir", type=Path, default=DEFAULT_TEMP)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--memory-limit", default="64GB")
    args = parser.parse_args()
    run(
        args.instance_glob,
        args.candidate_table,
        args.output,
        args.temp_dir,
        args.threads,
        args.memory_limit,
    )


if __name__ == "__main__":
    main()
