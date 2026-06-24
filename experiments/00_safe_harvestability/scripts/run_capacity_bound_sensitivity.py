#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


DEFAULT_INPUT = Path("/home/jiang/crust/results/08_machine_window_budgets.parquet")
DEFAULT_OUTPUT = Path(
    "/media/jiang/0bfc75f7-326d-2d45-b017-7100903796b6/"
    "google-cluster-data-2019/_analysis/safeharvest/00_safe_harvestability/sensitivity"
)
DEFAULT_TEMP = Path("/mnt/clusterdata0/duckdb_tmp")

POLICIES = [
    {
        "name": "raw_actual",
        "source": "true_agg",
        "expr": "true_agg",
        "order": 0,
    },
    {
        "name": "request_visible",
        "source": "res_p0",
        "expr": "res_p0",
        "order": 1,
    },
    {
        "name": "autopilot_style",
        "source": "res_p5",
        "expr": "res_p5",
        "order": 2,
    },
    {
        "name": "risk_q95_guard_0p90",
        "source": "0.90 * res_q3",
        "expr": "0.90 * res_q3",
        "order": 3,
    },
    {
        "name": "risk_q95_guard_1p00",
        "source": "res_q3",
        "expr": "res_q3",
        "order": 4,
    },
    {
        "name": "risk_q95_guard_1p10",
        "source": "1.10 * res_q3",
        "expr": "1.10 * res_q3",
        "order": 5,
    },
    {
        "name": "risk_q99_guard",
        "source": "res_q3b",
        "expr": "res_q3b",
        "order": 6,
    },
]


def _copy_sql(con: duckdb.DuckDBPyConnection, query: str, path: Path) -> None:
    con.execute(f"COPY ({query}) TO '{path}' (HEADER, DELIMITER ',')")


def _policy_union_sql() -> str:
    parts = []
    for policy in POLICIES:
        parts.append(
            f"""
            SELECT
              '{policy["name"]}' AS policy,
              '{policy["source"]}' AS reservation_source,
              {policy["order"]} AS policy_order,
              machine_id,
              win5,
              day_bucket,
              cpu_cap,
              n_tasks,
              true_agg,
              {policy["expr"]} AS reservation
            FROM base
            """
        )
    return "\nUNION ALL\n".join(parts)


def run(input_path: Path, output_dir: Path, temp_dir: Path, threads: int, memory_limit: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={threads}")
    con.execute(f"PRAGMA memory_limit='{memory_limit}'")
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
          AND res_q3b IS NOT NULL
        """,
        [str(input_path)],
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE policy_windows AS
        SELECT
          policy,
          reservation_source,
          policy_order,
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
        FROM ({_policy_union_sql()})
        WHERE reservation IS NOT NULL
        """
    )

    _copy_sql(
        con,
        """
        SELECT
          policy,
          reservation_source,
          COUNT(*) AS machine_windows,
          COUNT(DISTINCT machine_id) AS machines,
          MIN(win5) AS min_win5,
          MAX(win5) AS max_win5,
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
        GROUP BY policy, reservation_source, policy_order
        ORDER BY policy_order
        """,
        output_dir / "capacity_bound_sensitivity.csv",
    )

    _copy_sql(
        con,
        """
        SELECT
          policy,
          reservation_source,
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
        GROUP BY policy, reservation_source, policy_order, day_bucket
        ORDER BY policy_order, day_bucket
        """,
        output_dir / "capacity_bound_daily_sensitivity.csv",
    )

    _copy_sql(
        con,
        """
        SELECT
          policy,
          reservation_source,
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
        GROUP BY policy, reservation_source, policy_order
        ORDER BY policy_order
        """,
        output_dir / "capacity_bound_quantiles.csv",
    )

    thresholds = [0.05, 0.10, 0.20]
    durations = [1, 6, 12, 48]
    total_cap = con.execute("SELECT SUM(cpu_cap) FROM base").fetchone()[0]
    total_windows = con.execute("SELECT COUNT(*) FROM base").fetchone()[0]

    rows: list[tuple] = []
    for policy in POLICIES:
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
                [policy["name"], threshold],
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
                        policy["name"],
                        policy["source"],
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
        CREATE OR REPLACE TEMP TABLE stability (
          policy VARCHAR,
          reservation_source VARCHAR,
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
        "INSERT INTO stability VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    _copy_sql(
        con,
        """
        SELECT *
        FROM stability
        ORDER BY
          CASE policy
            WHEN 'raw_actual' THEN 0
            WHEN 'request_visible' THEN 1
            WHEN 'autopilot_style' THEN 2
            WHEN 'risk_q95_guard_0p90' THEN 3
            WHEN 'risk_q95_guard_1p00' THEN 4
            WHEN 'risk_q95_guard_1p10' THEN 5
            WHEN 'risk_q99_guard' THEN 6
            ELSE 99
          END,
          room_threshold_fraction,
          min_duration_windows
        """,
        output_dir / "capacity_bound_stability_sensitivity.csv",
    )

    _copy_sql(
        con,
        """
        SELECT
          COUNT(*) AS machine_windows,
          COUNT(DISTINCT machine_id) AS machines,
          MIN(win5) AS min_win5,
          MAX(win5) AS max_win5,
          SUM(cpu_cap) AS total_capacity_window,
          AVG(n_tasks) AS avg_tasks_per_machine_window
        FROM base
        """,
        output_dir / "capacity_bound_input_summary.csv",
    )
    print(f"wrote capacity bound sensitivity outputs to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--temp-dir", type=Path, default=DEFAULT_TEMP)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--memory-limit", default="64GB")
    args = parser.parse_args()
    run(args.input, args.output, args.temp_dir, args.threads, args.memory_limit)


if __name__ == "__main__":
    main()
