#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


DATA_ROOT = Path(
    "/media/jiang/0bfc75f7-326d-2d45-b017-7100903796b6/google-cluster-data-2019"
)
DEFAULT_INSTANCE_USAGE = DATA_ROOT / "cell_a" / "instance_usage-*.parquet.gz"
DEFAULT_MACHINE_EVENTS = DATA_ROOT / "cell_a" / "machine_events-000000000000.parquet.gz"
DEFAULT_OPPORTUNITY_DIR = DATA_ROOT / "_analysis" / "safeharvest" / "00_safe_harvestability"
DEFAULT_OUTPUT = DATA_ROOT / "_analysis" / "safeharvest" / "03_opportunity_matching"
DEFAULT_TEMP = Path("/mnt/clusterdata0/duckdb_tmp")

WIN_US = 300_000_000
MIN_WIN5 = 6480
MAX_WIN5 = 8784


def run(
    instance_usage_glob: str,
    machine_events: Path,
    opportunity_dir: Path,
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
        """
        CREATE OR REPLACE TEMP TABLE machine_caps AS
        SELECT
          machine_id,
          ARG_MAX(capacity.memory, time) AS memory_cap,
          ARG_MAX(capacity.cpus, time) AS cpu_cap_from_events
        FROM read_parquet(?)
        WHERE capacity.memory IS NOT NULL
        GROUP BY machine_id
        """,
        [str(machine_events)],
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE memory_usage AS
        SELECT
          machine_id,
          CAST(FLOOR(start_time / {WIN_US}.0) AS BIGINT) AS win5,
          SUM(COALESCE(average_usage.memory, 0.0)) AS avg_memory_used,
          SUM(COALESCE(maximum_usage.memory, average_usage.memory, 0.0)) AS max_memory_used,
          COUNT(*) AS usage_rows
        FROM read_parquet('{instance_usage_glob}')
        WHERE start_time >= {MIN_WIN5 * WIN_US}
          AND start_time < {(MAX_WIN5 + 1) * WIN_US}
          AND machine_id IS NOT NULL
        GROUP BY machine_id, win5
        """
    )

    for tag in ["10pct_1h", "20pct_1h", "10pct_4h", "20pct_4h"]:
        interval_path = opportunity_dir / f"resource_opportunity_intervals_{tag}.csv"
        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE intervals AS
            SELECT
              ROW_NUMBER() OVER () - 1 AS memory_interval_id,
              *
            FROM read_csv_auto('{interval_path}')
            """
        )

        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE memory_metrics AS
            WITH expanded AS (
              SELECT
                i.memory_interval_id,
                g.win5,
                c.memory_cap,
                COALESCE(u.avg_memory_used, 0.0) AS avg_memory_used,
                COALESCE(u.max_memory_used, 0.0) AS max_memory_used
              FROM intervals AS i
              INNER JOIN machine_caps AS c
                ON i.machine_id = c.machine_id
              CROSS JOIN UNNEST(
                range(CAST(i.start_win5 AS BIGINT), CAST(i.end_win5 AS BIGINT) + 1)
              ) AS g(win5)
              LEFT JOIN memory_usage AS u
                ON u.machine_id = i.machine_id
               AND u.win5 = g.win5
            ),
            scored AS (
              SELECT
                *,
                GREATEST(memory_cap - avg_memory_used, 0.0) AS memory_room_avg_usage,
                GREATEST(memory_cap - max_memory_used, 0.0) AS memory_room_max_usage,
                GREATEST(memory_cap - avg_memory_used, 0.0) / NULLIF(memory_cap, 0.0)
                  AS memory_room_frac_avg_usage,
                GREATEST(memory_cap - max_memory_used, 0.0) / NULLIF(memory_cap, 0.0)
                  AS memory_room_frac_max_usage,
                CASE WHEN avg_memory_used > memory_cap THEN 1 ELSE 0 END AS avg_memory_violation,
                CASE WHEN max_memory_used > memory_cap THEN 1 ELSE 0 END AS max_memory_violation
              FROM expanded
            )
            SELECT
              memory_interval_id,
              AVG(memory_cap) AS memory_cap_mean,
              MIN(memory_room_avg_usage) AS min_memory_room_avg_usage,
              APPROX_QUANTILE(memory_room_avg_usage, 0.10) AS p10_memory_room_avg_usage,
              AVG(memory_room_avg_usage) AS avg_memory_room_avg_usage,
              MIN(memory_room_max_usage) AS min_memory_room_max_usage,
              APPROX_QUANTILE(memory_room_max_usage, 0.10) AS p10_memory_room_max_usage,
              AVG(memory_room_max_usage) AS avg_memory_room_max_usage,
              MIN(memory_room_frac_avg_usage) AS min_memory_room_frac_avg_usage,
              APPROX_QUANTILE(memory_room_frac_avg_usage, 0.10) AS p10_memory_room_frac_avg_usage,
              AVG(memory_room_frac_avg_usage) AS avg_memory_room_frac_avg_usage,
              MIN(memory_room_frac_max_usage) AS min_memory_room_frac_max_usage,
              APPROX_QUANTILE(memory_room_frac_max_usage, 0.10) AS p10_memory_room_frac_max_usage,
              AVG(memory_room_frac_max_usage) AS avg_memory_room_frac_max_usage,
              SUM(avg_memory_violation) AS avg_memory_violation_windows,
              SUM(max_memory_violation) AS max_memory_violation_windows
            FROM scored
            GROUP BY memory_interval_id
            """
        )

        con.execute(
            f"""
            COPY (
              SELECT
                i.*,
                m.memory_cap_mean,
                m.min_memory_room_avg_usage,
                m.p10_memory_room_avg_usage,
                m.avg_memory_room_avg_usage,
                m.min_memory_room_max_usage,
                m.p10_memory_room_max_usage,
                m.avg_memory_room_max_usage,
                m.min_memory_room_frac_avg_usage,
                m.p10_memory_room_frac_avg_usage,
                m.avg_memory_room_frac_avg_usage,
                m.min_memory_room_frac_max_usage,
                m.p10_memory_room_frac_max_usage,
                m.avg_memory_room_frac_max_usage,
                m.avg_memory_violation_windows,
                m.max_memory_violation_windows
              FROM intervals AS i
              INNER JOIN memory_metrics AS m
                ON i.memory_interval_id = m.memory_interval_id
              ORDER BY i.duration_windows DESC, i.avg_room_frac DESC
            ) TO '{output_dir / f"memory_enriched_opportunity_intervals_{tag}.csv"}'
              (HEADER, DELIMITER ',')
            """
        )

        con.execute(
            f"""
            COPY (
              SELECT
                '{tag}' AS opportunity_class,
                COUNT(*) AS intervals,
                COUNT(DISTINCT machine_id) AS machines,
                SUM(duration_windows) AS total_windows,
                APPROX_QUANTILE(min_memory_room_frac_avg_usage, 0.50)
                  AS p50_min_memory_room_frac_avg_usage,
                APPROX_QUANTILE(min_memory_room_frac_avg_usage, 0.10)
                  AS p10_min_memory_room_frac_avg_usage,
                APPROX_QUANTILE(min_memory_room_avg_usage, 0.50)
                  AS p50_min_memory_room_avg_usage,
                APPROX_QUANTILE(min_memory_room_avg_usage, 0.10)
                  AS p10_min_memory_room_avg_usage,
                SUM(avg_memory_violation_windows) AS avg_memory_violation_windows,
                SUM(max_memory_violation_windows) AS max_memory_violation_windows
              FROM read_csv_auto('{output_dir / f"memory_enriched_opportunity_intervals_{tag}.csv"}')
            ) TO '{output_dir / f"memory_enriched_opportunity_summary_{tag}.csv"}'
              (HEADER, DELIMITER ',')
            """
        )

    con.execute(
        f"""
        COPY (
          SELECT * FROM read_csv_auto('{output_dir}/memory_enriched_opportunity_summary_*.csv')
        ) TO '{output_dir / "memory_enriched_opportunity_summary.csv"}'
          (HEADER, DELIMITER ',')
        """
    )
    print(f"wrote memory-enriched opportunity intervals to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance-usage-glob", default=str(DEFAULT_INSTANCE_USAGE))
    parser.add_argument("--machine-events", type=Path, default=DEFAULT_MACHINE_EVENTS)
    parser.add_argument("--opportunity-dir", type=Path, default=DEFAULT_OPPORTUNITY_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--temp-dir", type=Path, default=DEFAULT_TEMP)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--memory-limit", default="64GB")
    args = parser.parse_args()
    run(
        args.instance_usage_glob,
        args.machine_events,
        args.opportunity_dir,
        args.output,
        args.temp_dir,
        args.threads,
        args.memory_limit,
    )


if __name__ == "__main__":
    main()
