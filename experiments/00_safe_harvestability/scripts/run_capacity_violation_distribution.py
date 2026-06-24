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


def run(input_path: Path, output_dir: Path, temp_dir: Path, threads: int, memory_limit: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={threads}")
    con.execute(f"PRAGMA memory_limit='{memory_limit}'")
    con.execute("PRAGMA temp_directory=?", [str(temp_dir)])

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE policy_windows AS
        SELECT 'request_visible' AS policy, machine_id, true_agg, res_p0 AS reservation
        FROM read_parquet(?)
        WHERE true_agg IS NOT NULL AND res_p0 IS NOT NULL
        UNION ALL
        SELECT 'autopilot_style' AS policy, machine_id, true_agg, res_p5 AS reservation
        FROM read_parquet(?)
        WHERE true_agg IS NOT NULL AND res_p5 IS NOT NULL
        UNION ALL
        SELECT 'risk_q95_guard_1p00' AS policy, machine_id, true_agg, res_q3 AS reservation
        FROM read_parquet(?)
        WHERE true_agg IS NOT NULL AND res_q3 IS NOT NULL
        UNION ALL
        SELECT 'risk_q99_guard' AS policy, machine_id, true_agg, res_q3b AS reservation
        FROM read_parquet(?)
        WHERE true_agg IS NOT NULL AND res_q3b IS NOT NULL
        """,
        [str(input_path), str(input_path), str(input_path), str(input_path)],
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE machine_rates AS
        SELECT
          policy,
          machine_id,
          COUNT(*) AS windows,
          AVG(CASE WHEN true_agg > reservation THEN 1.0 ELSE 0.0 END) AS violation_rate,
          SUM(GREATEST(true_agg - reservation, 0.0)) / NULLIF(SUM(reservation), 0.0)
            AS overload_to_reserved_mass
        FROM policy_windows
        GROUP BY policy, machine_id
        """
    )

    con.execute(
        f"""
        COPY (
          SELECT
            policy,
            COUNT(*) AS machines,
            AVG(violation_rate) AS mean_machine_violation_rate,
            APPROX_QUANTILE(violation_rate, 0.50) AS p50_machine_violation_rate,
            APPROX_QUANTILE(violation_rate, 0.90) AS p90_machine_violation_rate,
            APPROX_QUANTILE(violation_rate, 0.95) AS p95_machine_violation_rate,
            APPROX_QUANTILE(violation_rate, 0.99) AS p99_machine_violation_rate,
            MAX(violation_rate) AS max_machine_violation_rate,
            AVG(overload_to_reserved_mass) AS mean_overload_to_reserved_mass,
            APPROX_QUANTILE(overload_to_reserved_mass, 0.99) AS p99_overload_to_reserved_mass,
            MAX(overload_to_reserved_mass) AS max_overload_to_reserved_mass
          FROM machine_rates
          GROUP BY policy
          ORDER BY
            CASE policy
              WHEN 'request_visible' THEN 1
              WHEN 'autopilot_style' THEN 2
              WHEN 'risk_q95_guard_1p00' THEN 3
              WHEN 'risk_q99_guard' THEN 4
              ELSE 99
            END
        ) TO '{output_dir / "capacity_bound_machine_violation_distribution.csv"}'
          (HEADER, DELIMITER ',')
        """
    )
    print(f"wrote per-machine violation distribution to {output_dir}")


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
