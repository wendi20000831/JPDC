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


POLICY_SQL = """
    SELECT 'raw_actual' AS policy, machine_id, win5, day_bucket, cpu_cap, n_tasks,
           true_agg, true_agg AS reservation
    FROM base
    UNION ALL
    SELECT 'request_visible' AS policy, machine_id, win5, day_bucket, cpu_cap, n_tasks,
           true_agg, res_p0 AS reservation
    FROM base
    UNION ALL
    SELECT 'autopilot_style' AS policy, machine_id, win5, day_bucket, cpu_cap, n_tasks,
           true_agg, res_p5 AS reservation
    FROM base
    UNION ALL
    SELECT 'risk_bounded_q95' AS policy, machine_id, win5, day_bucket, cpu_cap, n_tasks,
           true_agg, res_q3 AS reservation
    FROM base
"""


def run(input_path: Path, output_dir: Path, temp_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute("PRAGMA threads=16")
    con.execute("PRAGMA memory_limit='64GB'")
    con.execute("PRAGMA temp_directory=?", [str(temp_dir)])

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE base AS
        SELECT *
        FROM read_parquet(?)
        WHERE cpu_cap > 0
          AND true_agg IS NOT NULL
          AND res_p0 IS NOT NULL
          AND res_p5 IS NOT NULL
          AND res_q3 IS NOT NULL
        """,
        [str(input_path)],
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE policy_windows AS
        SELECT
          policy,
          machine_id,
          win5,
          day_bucket,
          cpu_cap,
          n_tasks,
          true_agg,
          reservation,
          GREATEST(cpu_cap - reservation, 0.0) AS room,
          GREATEST(cpu_cap - reservation, 0.0) / cpu_cap AS room_frac,
          reservation / cpu_cap AS reservation_frac,
          true_agg / cpu_cap AS true_load_frac,
          CASE WHEN true_agg > reservation THEN 1.0 ELSE 0.0 END AS violation,
          GREATEST(true_agg - reservation, 0.0) / cpu_cap AS overload_frac
        FROM ({POLICY_SQL})
        """
    )

    con.execute(
        f"""
        COPY (
          SELECT
            policy,
            COUNT(*) AS machine_windows,
            COUNT(DISTINCT machine_id) AS machines,
            MIN(day_bucket) AS min_day,
            MAX(day_bucket) AS max_day,
            SUM(cpu_cap) AS total_capacity_window,
            SUM(room) AS total_room,
            SUM(room) / SUM(cpu_cap) AS room_capacity_fraction,
            AVG(room_frac) AS mean_room_fraction,
            APPROX_QUANTILE(room_frac, 0.10) AS p10_room_fraction,
            APPROX_QUANTILE(room_frac, 0.50) AS p50_room_fraction,
            APPROX_QUANTILE(room_frac, 0.90) AS p90_room_fraction,
            APPROX_QUANTILE(room_frac, 0.99) AS p99_room_fraction,
            AVG(CASE WHEN room > 0 THEN 1.0 ELSE 0.0 END) AS nonzero_room_window_share,
            AVG(CASE WHEN reservation >= cpu_cap THEN 1.0 ELSE 0.0 END) AS no_room_window_share,
            AVG(reservation_frac) AS mean_reservation_fraction,
            APPROX_QUANTILE(reservation_frac, 0.50) AS p50_reservation_fraction,
            APPROX_QUANTILE(reservation_frac, 0.95) AS p95_reservation_fraction,
            AVG(true_load_frac) AS mean_true_load_fraction,
            AVG(violation) AS violation_rate,
            SUM(GREATEST(true_agg - reservation, 0.0)) / SUM(cpu_cap) AS overload_mass_fraction
          FROM policy_windows
          GROUP BY policy
          ORDER BY
            CASE policy
              WHEN 'raw_actual' THEN 1
              WHEN 'request_visible' THEN 2
              WHEN 'autopilot_style' THEN 3
              WHEN 'risk_bounded_q95' THEN 4
              ELSE 9
            END
        ) TO '{output_dir / "harvestability_funnel.csv"}' (HEADER, DELIMITER ',')
        """
    )

    con.execute(
        f"""
        COPY (
          SELECT
            policy,
            day_bucket,
            COUNT(*) AS machine_windows,
            SUM(cpu_cap) AS total_capacity_window,
            SUM(room) / SUM(cpu_cap) AS room_capacity_fraction,
            AVG(room_frac) AS mean_room_fraction,
            AVG(CASE WHEN room > 0 THEN 1.0 ELSE 0.0 END) AS nonzero_room_window_share,
            AVG(CASE WHEN reservation >= cpu_cap THEN 1.0 ELSE 0.0 END) AS no_room_window_share,
            AVG(violation) AS violation_rate,
            SUM(GREATEST(true_agg - reservation, 0.0)) / SUM(cpu_cap) AS overload_mass_fraction
          FROM policy_windows
          GROUP BY policy, day_bucket
          ORDER BY policy, day_bucket
        ) TO '{output_dir / "harvestability_by_day.csv"}' (HEADER, DELIMITER ',')
        """
    )

    con.execute(
        f"""
        COPY (
          SELECT
            policy,
            APPROX_QUANTILE(room_frac, 0.01) AS p01_room,
            APPROX_QUANTILE(room_frac, 0.05) AS p05_room,
            APPROX_QUANTILE(room_frac, 0.10) AS p10_room,
            APPROX_QUANTILE(room_frac, 0.25) AS p25_room,
            APPROX_QUANTILE(room_frac, 0.50) AS p50_room,
            APPROX_QUANTILE(room_frac, 0.75) AS p75_room,
            APPROX_QUANTILE(room_frac, 0.90) AS p90_room,
            APPROX_QUANTILE(room_frac, 0.95) AS p95_room,
            APPROX_QUANTILE(room_frac, 0.99) AS p99_room,
            APPROX_QUANTILE(reservation_frac, 0.50) AS p50_reservation,
            APPROX_QUANTILE(reservation_frac, 0.90) AS p90_reservation,
            APPROX_QUANTILE(reservation_frac, 0.95) AS p95_reservation,
            APPROX_QUANTILE(reservation_frac, 0.99) AS p99_reservation
          FROM policy_windows
          GROUP BY policy
          ORDER BY policy
        ) TO '{output_dir / "policy_window_quantiles.csv"}' (HEADER, DELIMITER ',')
        """
    )

    thresholds = [0.05, 0.10, 0.20]
    durations = [1, 6, 12, 48]  # 5 min, 30 min, 1 h, 4 h
    total_cap = con.execute("SELECT SUM(cpu_cap) FROM base").fetchone()[0]
    total_windows = con.execute("SELECT COUNT(*) FROM base").fetchone()[0]

    rows: list[tuple] = []
    for policy in ["raw_actual", "request_visible", "autopilot_style", "risk_bounded_q95"]:
        for threshold in thresholds:
            con.execute(
                """
                CREATE OR REPLACE TEMP TABLE qualified AS
                SELECT
                  machine_id,
                  win5,
                  cpu_cap,
                  room,
                  win5 - ROW_NUMBER() OVER (PARTITION BY machine_id ORDER BY win5) AS run_key
                FROM policy_windows
                WHERE policy = ?
                  AND room_frac >= ?
                """,
                [policy, threshold],
            )
            con.execute(
                """
                CREATE OR REPLACE TEMP TABLE runs AS
                SELECT
                  machine_id,
                  run_key,
                  COUNT(*) AS run_windows,
                  SUM(room) AS run_room,
                  SUM(cpu_cap) AS run_capacity,
                  MIN(win5) AS start_win5,
                  MAX(win5) AS end_win5
                FROM qualified
                GROUP BY machine_id, run_key
                """
            )
            for duration in durations:
                stable = con.execute(
                    """
                    SELECT
                      COUNT(*) AS stable_runs,
                      COALESCE(SUM(run_windows), 0) AS stable_windows,
                      COALESCE(SUM(run_room), 0.0) AS stable_room,
                      COALESCE(SUM(run_capacity), 0.0) AS stable_capacity,
                      COALESCE(AVG(run_windows), 0.0) AS avg_run_windows,
                      COALESCE(APPROX_QUANTILE(run_windows, 0.50), 0) AS p50_run_windows,
                      COALESCE(APPROX_QUANTILE(run_windows, 0.90), 0) AS p90_run_windows
                    FROM runs
                    WHERE run_windows >= ?
                    """,
                    [duration],
                ).fetchone()
                rows.append(
                    (
                        policy,
                        threshold,
                        duration,
                        duration * 5,
                        stable[0],
                        stable[1],
                        stable[2],
                        stable[3],
                        stable[4],
                        stable[5],
                        stable[6],
                        stable[1] / total_windows,
                        stable[2] / total_cap,
                    )
                )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE stability_matrix (
          policy VARCHAR,
          room_threshold_fraction DOUBLE,
          min_duration_windows INTEGER,
          min_duration_minutes INTEGER,
          stable_runs BIGINT,
          stable_windows DOUBLE,
          stable_room DOUBLE,
          stable_capacity DOUBLE,
          avg_run_windows DOUBLE,
          p50_run_windows DOUBLE,
          p90_run_windows DOUBLE,
          stable_window_share DOUBLE,
          stable_room_capacity_fraction DOUBLE
        )
        """
    )
    con.executemany(
        "INSERT INTO stability_matrix VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    con.execute(
        f"""
        COPY (
          SELECT *
          FROM stability_matrix
          ORDER BY
            CASE policy
              WHEN 'raw_actual' THEN 1
              WHEN 'request_visible' THEN 2
              WHEN 'autopilot_style' THEN 3
              WHEN 'risk_bounded_q95' THEN 4
              ELSE 9
            END,
            room_threshold_fraction,
            min_duration_windows
        ) TO '{output_dir / "stability_matrix.csv"}' (HEADER, DELIMITER ',')
        """
    )

    con.execute(
        f"""
        COPY (
          SELECT
            COUNT(*) AS machine_windows,
            COUNT(DISTINCT machine_id) AS machines,
            MIN(win5) AS min_win5,
            MAX(win5) AS max_win5,
            MIN(day_bucket) AS min_day,
            MAX(day_bucket) AS max_day,
            SUM(cpu_cap) AS total_capacity_window,
            AVG(n_tasks) AS avg_tasks_per_machine_window,
            APPROX_QUANTILE(n_tasks, 0.50) AS p50_tasks_per_machine_window,
            APPROX_QUANTILE(n_tasks, 0.95) AS p95_tasks_per_machine_window
          FROM base
        ) TO '{output_dir / "input_summary.csv"}' (HEADER, DELIMITER ',')
        """
    )

    print(f"wrote SafeHarvest harvestability profile to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--temp-dir", type=Path, default=DEFAULT_TEMP)
    args = parser.parse_args()
    run(args.input, args.output, args.temp_dir)


if __name__ == "__main__":
    main()
