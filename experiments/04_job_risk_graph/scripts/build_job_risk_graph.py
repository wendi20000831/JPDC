#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


DATA_ROOT = Path(
    "/media/jiang/0bfc75f7-326d-2d45-b017-7100903796b6/google-cluster-data-2019"
)
DEFAULT_COLLECTION_EVENTS = DATA_ROOT / "cell_a" / "collection_events-*.parquet.gz"
DEFAULT_INSTANCE_EVENTS = DATA_ROOT / "cell_a" / "instance_events-*.parquet.gz"
DEFAULT_SAFEHARVEST_ROOT = DATA_ROOT / "_analysis" / "safeharvest"
DEFAULT_RISK_TABLE = (
    DEFAULT_SAFEHARVEST_ROOT
    / "01_candidate_workload_risk"
    / "candidate_workload_risk_table.parquet"
)
DEFAULT_OUTPUT = DEFAULT_SAFEHARVEST_ROOT / "04_job_risk_graph"
DEFAULT_TEMP = Path("/mnt/clusterdata0/duckdb_tmp")


def _configure(con: duckdb.DuckDBPyConnection, threads: int, memory_limit: str, temp_dir: Path) -> None:
    con.execute(f"PRAGMA threads={threads}")
    con.execute(f"PRAGMA memory_limit='{memory_limit}'")
    con.execute("PRAGMA temp_directory=?", [str(temp_dir)])


def run(
    collection_glob: str,
    instance_glob: str,
    risk_table: Path,
    output_dir: Path,
    temp_dir: Path,
    threads: int,
    memory_limit: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    _configure(con, threads, memory_limit, temp_dir)

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE base_nodes AS
        SELECT
          *,
          CAST(FLOOR(schedule_time / 300000000.0) AS BIGINT) AS schedule_win5,
          CAST(GREATEST(1, CEIL(runtime_sec / 300.0)) AS BIGINT) AS runtime_windows
        FROM read_parquet('{risk_table}')
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE parent_edges AS
        SELECT
          parent_collection_id AS src_collection_id,
          collection_id AS dst_collection_id,
          'parent' AS edge_type
        FROM base_nodes
        WHERE parent_collection_id IS NOT NULL
          AND parent_collection_id != 0
          AND collection_id IS NOT NULL
          AND parent_collection_id != collection_id
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE start_after_edges AS
        WITH first_deps AS (
          SELECT
            collection_id,
            start_after_collection_ids AS deps
          FROM read_parquet('{collection_glob}')
          WHERE collection_type = 0
            AND start_after_collection_ids IS NOT NULL
            AND array_length(start_after_collection_ids) > 0
          QUALIFY row_number() OVER (PARTITION BY collection_id ORDER BY time) = 1
        )
        SELECT
          dep_id AS src_collection_id,
          collection_id AS dst_collection_id,
          'start_after' AS edge_type
        FROM first_deps, UNNEST(deps) AS u(dep_id)
        WHERE dep_id IS NOT NULL
          AND dep_id != 0
          AND collection_id IS NOT NULL
          AND dep_id != collection_id
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE graph_edges AS
        SELECT DISTINCT * FROM parent_edges
        UNION
        SELECT DISTINCT * FROM start_after_edges
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE graph_degrees AS
        WITH incoming AS (
          SELECT
            dst_collection_id AS collection_id,
            COUNT(*) AS incoming_dependency_edges,
            SUM(CASE WHEN edge_type = 'parent' THEN 1 ELSE 0 END) AS incoming_parent_edges,
            SUM(CASE WHEN edge_type = 'start_after' THEN 1 ELSE 0 END) AS incoming_start_after_edges
          FROM graph_edges
          GROUP BY dst_collection_id
        ),
        outgoing AS (
          SELECT
            src_collection_id AS collection_id,
            COUNT(*) AS outgoing_dependent_edges,
            SUM(CASE WHEN edge_type = 'parent' THEN 1 ELSE 0 END) AS outgoing_parent_edges,
            SUM(CASE WHEN edge_type = 'start_after' THEN 1 ELSE 0 END) AS outgoing_start_after_edges
          FROM graph_edges
          GROUP BY src_collection_id
        )
        SELECT
          COALESCE(i.collection_id, o.collection_id) AS collection_id,
          COALESCE(i.incoming_dependency_edges, 0) AS incoming_dependency_edges,
          COALESCE(i.incoming_parent_edges, 0) AS incoming_parent_edges,
          COALESCE(i.incoming_start_after_edges, 0) AS incoming_start_after_edges,
          COALESCE(o.outgoing_dependent_edges, 0) AS outgoing_dependent_edges,
          COALESCE(o.outgoing_parent_edges, 0) AS outgoing_parent_edges,
          COALESCE(o.outgoing_start_after_edges, 0) AS outgoing_start_after_edges
        FROM incoming AS i
        FULL OUTER JOIN outgoing AS o
          ON i.collection_id = o.collection_id
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE graph_nodes AS
        SELECT
          b.*,
          COALESCE(g.incoming_dependency_edges, 0) AS incoming_dependency_edges,
          COALESCE(g.incoming_parent_edges, 0) AS incoming_parent_edges,
          COALESCE(g.incoming_start_after_edges, 0) AS incoming_start_after_edges,
          COALESCE(g.outgoing_dependent_edges, 0) AS outgoing_dependent_edges,
          COALESCE(g.outgoing_parent_edges, 0) AS outgoing_parent_edges,
          COALESCE(g.outgoing_start_after_edges, 0) AS outgoing_start_after_edges,
          CASE
            WHEN COALESCE(g.incoming_dependency_edges, 0) = 0
             AND COALESCE(g.outgoing_dependent_edges, 0) = 0 THEN 'independent_leaf'
            WHEN COALESCE(g.incoming_dependency_edges, 0) = 0 THEN 'upstream_prerequisite'
            WHEN COALESCE(g.outgoing_dependent_edges, 0) = 0 THEN 'dependent_leaf'
            ELSE 'internal_dependency_chain'
          END AS dependency_graph_role
        FROM base_nodes AS b
        LEFT JOIN graph_degrees AS g
          ON b.collection_id = g.collection_id
        """
    )

    con.execute(
        f"""
        COPY graph_edges TO '{output_dir / "job_risk_graph_edges.parquet"}'
          (FORMAT 'parquet', COMPRESSION 'zstd')
        """
    )

    con.execute(
        f"""
        COPY graph_nodes TO '{output_dir / "job_risk_graph_nodes.parquet"}'
          (FORMAT 'parquet', COMPRESSION 'zstd')
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE expansion_frontier_keys AS
        SELECT
          collection_id
        FROM graph_nodes
        WHERE collection_id IS NOT NULL
          AND candidate_class != 'latency_sensitive_or_high_priority'
          AND schedule_time IS NOT NULL
          AND runtime_sec IS NOT NULL
          AND runtime_sec > 0
          AND runtime_sec <= 28800
          AND (
            safeharvest_seed_candidate = 1
            OR (
              terminal_outcome = 'FINISH'
              AND runtime_sec <= 14400
              AND (
                dependency_graph_role = 'independent_leaf'
                OR (
                  incoming_dependency_edges <= 2
                  AND outgoing_dependent_edges = 0
                )
              )
            )
            OR (
              terminal_outcome = 'FINISH'
              AND runtime_sec <= 28800
              AND dependency_graph_role = 'independent_leaf'
            )
            OR (
              terminal_outcome IN ('EVICT', 'KILL', 'LOST')
              AND runtime_sec <= 3600
              AND dependency_graph_role = 'independent_leaf'
              AND candidate_class IN (
                'batch_scheduler',
                'best_effort_queueable',
                'low_priority_flexible'
              )
            )
          )
        """
    )
    con.execute("CREATE INDEX expansion_frontier_idx ON expansion_frontier_keys(collection_id)")

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
          INNER JOIN expansion_frontier_keys AS f
            ON e.collection_id = f.collection_id
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
        """
        CREATE OR REPLACE TEMP TABLE instance_demand AS
        SELECT
          collection_id,
          count(instance_index) AS instances,
          sum(request_cpus) AS total_cpu_request,
          sum(request_memory) AS total_memory_request,
          max(request_cpus) AS max_instance_cpu_request,
          max(request_memory) AS max_instance_memory_request,
          approx_quantile(request_cpus, 0.50) AS p50_instance_cpu_request,
          approx_quantile(request_cpus, 0.90) AS p90_instance_cpu_request,
          approx_quantile(request_memory, 0.50) AS p50_instance_memory_request,
          approx_quantile(request_memory, 0.90) AS p90_instance_memory_request,
          sum(CASE WHEN instance_terminal_outcome = 'FINISH' THEN 1 ELSE 0 END) AS finished_instances,
          sum(CASE WHEN instance_terminal_outcome IN ('EVICT','FAIL','KILL','LOST') THEN 1 ELSE 0 END) AS bad_terminal_instances,
          sum(CASE WHEN instance_terminal_outcome = 'NO_TERMINAL' THEN 1 ELSE 0 END) AS no_terminal_instances,
          approx_quantile(instance_runtime_sec, 0.50) AS p50_instance_runtime_sec,
          approx_quantile(instance_runtime_sec, 0.90) AS p90_instance_runtime_sec,
          sum(CASE WHEN touched_machines > 1 THEN 1 ELSE 0 END) AS multi_machine_instances
        FROM instance_lifecycle
        GROUP BY collection_id
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE expanded_candidates AS
        SELECT
          g.collection_id,
          g.candidate_class,
          g.priority,
          g.scheduling_class,
          g.scheduler,
          g.terminal_outcome AS collection_terminal_outcome,
          g.runtime_bucket AS collection_runtime_bucket,
          g.dependency_bucket,
          g.dependency_graph_role,
          g.incoming_dependency_edges,
          g.outgoing_dependent_edges,
          g.submit_to_schedule_sec AS collection_queue_sec,
          g.runtime_sec AS collection_runtime_sec,
          g.runtime_windows,
          g.schedule_win5,
          CAST(g.harvest_risk_score AS DOUBLE) AS harvest_risk_score,
          g.safeharvest_seed_candidate,
          COALESCE(d.instances, 0) AS instances,
          COALESCE(d.total_cpu_request, 0.0) AS total_cpu_request,
          COALESCE(d.total_memory_request, 0.0) AS total_memory_request,
          COALESCE(d.max_instance_cpu_request, 0.0) AS max_instance_cpu_request,
          COALESCE(d.max_instance_memory_request, 0.0) AS max_instance_memory_request,
          d.p50_instance_cpu_request,
          d.p90_instance_cpu_request,
          d.p50_instance_memory_request,
          d.p90_instance_memory_request,
          COALESCE(d.finished_instances, 0) AS finished_instances,
          COALESCE(d.bad_terminal_instances, 0) AS bad_terminal_instances,
          COALESCE(d.no_terminal_instances, 0) AS no_terminal_instances,
          d.p50_instance_runtime_sec,
          d.p90_instance_runtime_sec,
          COALESCE(d.multi_machine_instances, 0) AS multi_machine_instances,
          CASE
            WHEN g.safeharvest_seed_candidate = 1
             AND COALESCE(d.instances, 0) = 1
             AND COALESCE(d.bad_terminal_instances, 0) = 0
             AND COALESCE(d.total_cpu_request, 0.0) > 0
             AND COALESCE(d.total_cpu_request, 0.0) <= 0.10
             AND COALESCE(d.total_memory_request, 0.0) <= 0.10
              THEN 'S0_strict_seed'
            WHEN g.terminal_outcome = 'FINISH'
             AND g.runtime_sec <= 14400
             AND g.dependency_graph_role = 'independent_leaf'
             AND COALESCE(d.instances, 0) BETWEEN 1 AND 5
             AND COALESCE(d.bad_terminal_instances, 0) = 0
             AND COALESCE(d.total_cpu_request, 0.0) > 0
             AND COALESCE(d.total_cpu_request, 0.0) <= 0.25
             AND COALESCE(d.total_memory_request, 0.0) <= 0.25
             AND g.candidate_class IN (
               'batch_scheduler',
               'best_effort_queueable',
               'low_priority_flexible',
               'medium_flexibility'
             )
              THEN 'S1_clean_small_independent'
            WHEN g.terminal_outcome = 'FINISH'
             AND g.runtime_sec <= 14400
             AND g.incoming_dependency_edges <= 2
             AND g.outgoing_dependent_edges = 0
             AND COALESCE(d.instances, 0) BETWEEN 1 AND 5
             AND COALESCE(d.bad_terminal_instances, 0) = 0
             AND COALESCE(d.total_cpu_request, 0.0) > 0
             AND COALESCE(d.total_cpu_request, 0.0) <= 0.25
             AND COALESCE(d.total_memory_request, 0.0) <= 0.25
             AND g.candidate_class IN (
               'batch_scheduler',
               'best_effort_queueable',
               'low_priority_flexible',
               'medium_flexibility'
             )
              THEN 'S2_clean_small_leaf_dependency'
            WHEN g.terminal_outcome = 'FINISH'
             AND g.runtime_sec <= 28800
             AND g.dependency_graph_role = 'independent_leaf'
             AND COALESCE(d.instances, 0) BETWEEN 1 AND 20
             AND COALESCE(d.bad_terminal_instances, 0) = 0
             AND COALESCE(d.total_cpu_request, 0.0) > 0
             AND COALESCE(d.total_cpu_request, 0.0) <= 0.50
             AND COALESCE(d.total_memory_request, 0.0) <= 0.50
             AND g.candidate_class IN (
               'batch_scheduler',
               'best_effort_queueable',
               'low_priority_flexible',
               'medium_flexibility'
             )
              THEN 'S3_clean_medium_independent'
            WHEN g.terminal_outcome IN ('EVICT', 'KILL', 'LOST')
             AND g.runtime_sec <= 3600
             AND g.dependency_graph_role = 'independent_leaf'
             AND COALESCE(d.instances, 0) BETWEEN 1 AND 5
             AND COALESCE(d.total_cpu_request, 0.0) > 0
             AND COALESCE(d.total_cpu_request, 0.0) <= 0.25
             AND COALESCE(d.total_memory_request, 0.0) <= 0.25
             AND g.candidate_class IN (
               'batch_scheduler',
               'best_effort_queueable',
               'low_priority_flexible'
             )
              THEN 'R1_retry_tolerant_signal'
            ELSE 'X_excluded'
          END AS risk_tier
        FROM graph_nodes AS g
        INNER JOIN expansion_frontier_keys AS f
          ON g.collection_id = f.collection_id
        LEFT JOIN instance_demand AS d
          ON g.collection_id = d.collection_id
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE expanded_candidates_scored AS
        SELECT
          *,
          CASE risk_tier
            WHEN 'S0_strict_seed' THEN 0
            WHEN 'S1_clean_small_independent' THEN 1
            WHEN 'S2_clean_small_leaf_dependency' THEN 2
            WHEN 'S3_clean_medium_independent' THEN 3
            WHEN 'R1_retry_tolerant_signal' THEN 90
            ELSE 99
          END AS risk_tier_order,
          CASE
            WHEN risk_tier IN (
              'S0_strict_seed',
              'S1_clean_small_independent',
              'S2_clean_small_leaf_dependency',
              'S3_clean_medium_independent'
            )
              THEN 'safe_expansion'
            WHEN risk_tier = 'R1_retry_tolerant_signal'
              THEN 'retry_signal_only'
            ELSE 'excluded'
          END AS risk_tier_family,
          total_cpu_request * runtime_windows AS cpu_window_demand,
          total_memory_request * runtime_windows AS memory_window_demand
        FROM expanded_candidates
        """
    )

    con.execute(
        f"""
        COPY expanded_candidates_scored TO '{output_dir / "expanded_candidate_table.parquet"}'
          (FORMAT 'parquet', COMPRESSION 'zstd')
        """
    )

    con.execute(
        f"""
        COPY (
          WITH edge_counts AS (
            SELECT
              COUNT(*) AS edges,
              SUM(CASE WHEN edge_type = 'parent' THEN 1 ELSE 0 END) AS parent_edges,
              SUM(CASE WHEN edge_type = 'start_after' THEN 1 ELSE 0 END) AS start_after_edges
            FROM graph_edges
          )
          SELECT
            COUNT(*) AS collections,
            (SELECT edges FROM edge_counts) AS edges,
            (SELECT parent_edges FROM edge_counts) AS parent_edges,
            (SELECT start_after_edges FROM edge_counts) AS start_after_edges,
            SUM(CASE WHEN incoming_dependency_edges = 0 THEN 1 ELSE 0 END) AS no_incoming_dependency_collections,
            SUM(CASE WHEN outgoing_dependent_edges = 0 THEN 1 ELSE 0 END) AS no_outgoing_dependent_collections,
            SUM(CASE WHEN dependency_graph_role = 'independent_leaf' THEN 1 ELSE 0 END) AS independent_leaf_collections,
            AVG(incoming_dependency_edges) AS avg_incoming_dependency_edges,
            AVG(outgoing_dependent_edges) AS avg_outgoing_dependent_edges,
            APPROX_QUANTILE(incoming_dependency_edges, 0.90) AS p90_incoming_dependency_edges,
            APPROX_QUANTILE(outgoing_dependent_edges, 0.90) AS p90_outgoing_dependent_edges,
            APPROX_QUANTILE(outgoing_dependent_edges, 0.99) AS p99_outgoing_dependent_edges
          FROM graph_nodes
        ) TO '{output_dir / "graph_structure_summary.csv"}' (HEADER, DELIMITER ',')
        """
    )

    con.execute(
        f"""
        COPY (
          SELECT
            candidate_class,
            dependency_graph_role,
            COUNT(*) AS collections,
            AVG(CASE WHEN terminal_outcome = 'FINISH' THEN 1.0 ELSE 0.0 END) AS finish_share,
            AVG(CASE WHEN terminal_outcome IN ('EVICT','FAIL','KILL','LOST') THEN 1.0 ELSE 0.0 END) AS bad_terminal_share,
            APPROX_QUANTILE(runtime_sec, 0.50) AS p50_runtime_sec,
            APPROX_QUANTILE(runtime_sec, 0.90) AS p90_runtime_sec
          FROM graph_nodes
          GROUP BY candidate_class, dependency_graph_role
          ORDER BY candidate_class, collections DESC
        ) TO '{output_dir / "dependency_role_summary.csv"}' (HEADER, DELIMITER ',')
        """
    )

    con.execute(
        f"""
        COPY (
          SELECT
            risk_tier,
            risk_tier_order,
            risk_tier_family,
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
          FROM expanded_candidates_scored
          GROUP BY risk_tier, risk_tier_order, risk_tier_family
          ORDER BY risk_tier_order, collections DESC
        ) TO '{output_dir / "risk_tier_summary.csv"}' (HEADER, DELIMITER ',')
        """
    )

    con.execute(
        f"""
        COPY (
          SELECT
            risk_tier,
            candidate_class,
            COUNT(*) AS collections,
            SUM(CASE WHEN schedule_win5 BETWEEN 6480 AND 8784 THEN 1 ELSE 0 END) AS test_window_collections,
            SUM(cpu_window_demand) AS cpu_window_demand,
            SUM(instances) AS instances,
            SUM(bad_terminal_instances) AS bad_terminal_instances,
            SUM(bad_terminal_instances) / NULLIF(SUM(instances), 0) AS instance_bad_terminal_share,
            APPROX_QUANTILE(collection_runtime_sec, 0.50) AS p50_runtime_sec,
            APPROX_QUANTILE(total_cpu_request, 0.90) AS p90_total_cpu_request
          FROM expanded_candidates_scored
          GROUP BY risk_tier, candidate_class
          ORDER BY
            MIN(risk_tier_order),
            risk_tier,
            collections DESC
        ) TO '{output_dir / "risk_tier_by_class.csv"}' (HEADER, DELIMITER ',')
        """
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

    con.execute(
        f"""
        COPY (
          SELECT
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
          FROM frontier_policies AS p
          INNER JOIN expanded_candidates_scored AS e
            ON (
              e.risk_tier_order <= p.max_safe_tier_order
              OR (
                p.include_retry_signal = 1
                AND e.risk_tier = 'R1_retry_tolerant_signal'
              )
            )
          GROUP BY p.frontier_policy
          ORDER BY p.frontier_policy
        ) TO '{output_dir / "expansion_frontier_summary.csv"}' (HEADER, DELIMITER ',')
        """
    )

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
            incoming_dependency_edges,
            outgoing_dependent_edges,
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
          FROM expanded_candidates_scored
          WHERE schedule_win5 BETWEEN 6480 AND 8784
            AND risk_tier != 'X_excluded'
            AND total_cpu_request > 0
          ORDER BY risk_tier_order, schedule_win5, harvest_risk_score, cpu_window_demand
          LIMIT 100000
        ) TO '{output_dir / "test_window_expanded_candidate_sample.csv"}' (HEADER, DELIMITER ',')
        """
    )

    print(f"wrote job risk graph outputs to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection-events", default=str(DEFAULT_COLLECTION_EVENTS))
    parser.add_argument("--instance-events", default=str(DEFAULT_INSTANCE_EVENTS))
    parser.add_argument("--risk-table", type=Path, default=DEFAULT_RISK_TABLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--temp-dir", type=Path, default=DEFAULT_TEMP)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--memory-limit", default="64GB")
    args = parser.parse_args()
    run(
        args.collection_events,
        args.instance_events,
        args.risk_table,
        args.output,
        args.temp_dir,
        args.threads,
        args.memory_limit,
    )


if __name__ == "__main__":
    main()
