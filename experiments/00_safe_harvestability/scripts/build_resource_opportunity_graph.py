#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


DEFAULT_INPUT = Path("/home/jiang/crust/results/08_machine_window_budgets.parquet")
DEFAULT_OUTPUT = Path(
    "/media/jiang/0bfc75f7-326d-2d45-b017-7100903796b6/"
    "google-cluster-data-2019/_analysis/safeharvest/00_safe_harvestability"
)
DEFAULT_TEMP = Path("/mnt/clusterdata0/duckdb_tmp")


def run(input_path: Path, output_dir: Path, temp_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute("PRAGMA threads=16")
    con.execute("PRAGMA memory_limit='64GB'")
    con.execute("PRAGMA temp_directory=?", [str(temp_dir)])

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE windows AS
        SELECT
          machine_id,
          win5,
          day_bucket,
          cpu_cap,
          n_tasks,
          true_agg,
          res_q3 AS reservation,
          GREATEST(cpu_cap - res_q3, 0.0) AS room,
          GREATEST(cpu_cap - res_q3, 0.0) / cpu_cap AS room_frac,
          CASE WHEN true_agg > res_q3 THEN 1.0 ELSE 0.0 END AS violation
        FROM read_parquet(?)
        WHERE cpu_cap > 0
          AND true_agg IS NOT NULL
          AND res_q3 IS NOT NULL
        """,
        [str(input_path)],
    )

    for threshold, min_windows, tag in [
        (0.10, 12, "10pct_1h"),
        (0.10, 48, "10pct_4h"),
        (0.20, 12, "20pct_1h"),
        (0.20, 48, "20pct_4h"),
    ]:
        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE qualified AS
            SELECT
              *,
              win5 - ROW_NUMBER() OVER (PARTITION BY machine_id ORDER BY win5) AS run_key
            FROM windows
            WHERE room_frac >= ?
            """,
            [threshold],
        )
        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE intervals AS
            SELECT
              machine_id,
              MIN(win5) AS start_win5,
              MAX(win5) AS end_win5,
              COUNT(*) AS duration_windows,
              COUNT(*) * 5 AS duration_minutes,
              MIN(day_bucket) AS start_day,
              MAX(day_bucket) AS end_day,
              AVG(cpu_cap) AS cpu_cap_mean,
              MIN(room_frac) AS min_room_frac,
              AVG(room_frac) AS avg_room_frac,
              APPROX_QUANTILE(room_frac, 0.10) AS p10_room_frac,
              SUM(room) AS total_room_cpu_window,
              AVG(n_tasks) AS avg_tasks,
              SUM(violation) AS violation_windows,
              AVG(violation) AS violation_window_share
            FROM qualified
            GROUP BY machine_id, run_key
            HAVING COUNT(*) >= ?
            """,
            [min_windows],
        )
        con.execute(
            f"""
            COPY (
              SELECT
                '{tag}' AS opportunity_class,
                {threshold} AS min_room_threshold,
                {min_windows} AS min_duration_windows,
                *
              FROM intervals
              ORDER BY duration_windows DESC, avg_room_frac DESC
            ) TO '{output_dir / f"resource_opportunity_intervals_{tag}.csv"}'
              (HEADER, DELIMITER ',')
            """
        )
        con.execute(
            f"""
            COPY (
              SELECT
                '{tag}' AS opportunity_class,
                {threshold} AS min_room_threshold,
                {min_windows} AS min_duration_windows,
                COUNT(*) AS intervals,
                COUNT(DISTINCT machine_id) AS machines,
                SUM(duration_windows) AS total_windows,
                SUM(total_room_cpu_window) AS total_room_cpu_window,
                AVG(duration_minutes) AS avg_duration_minutes,
                APPROX_QUANTILE(duration_minutes, 0.50) AS p50_duration_minutes,
                APPROX_QUANTILE(duration_minutes, 0.90) AS p90_duration_minutes,
                APPROX_QUANTILE(avg_room_frac, 0.50) AS p50_avg_room_frac,
                APPROX_QUANTILE(avg_room_frac, 0.90) AS p90_avg_room_frac,
                SUM(violation_windows) AS violation_windows,
                SUM(violation_windows) / NULLIF(SUM(duration_windows), 0) AS violation_window_share,
                AVG(violation_window_share) AS avg_interval_violation_window_share
              FROM intervals
            ) TO '{output_dir / f"resource_opportunity_summary_{tag}.csv"}'
              (HEADER, DELIMITER ',')
            """
        )

    con.execute(
        f"""
        COPY (
          SELECT * FROM read_csv_auto('{output_dir}/resource_opportunity_summary_*.csv')
        ) TO '{output_dir / "resource_opportunity_summary.csv"}' (HEADER, DELIMITER ',')
        """
    )
    print(f"wrote Resource Opportunity Graph slices to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--temp-dir", type=Path, default=DEFAULT_TEMP)
    args = parser.parse_args()
    run(args.input, args.output, args.temp_dir)


if __name__ == "__main__":
    main()
