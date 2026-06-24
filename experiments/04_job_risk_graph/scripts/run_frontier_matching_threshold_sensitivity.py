#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import duckdb
import pandas as pd

from risk_threshold_variants import (
    risk_tier_family_case,
    risk_tier_order_case,
    variant_by_name,
    variant_union_sql,
)
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
DEFAULT_CPU_OPPORTUNITY_DIR = DEFAULT_SAFEHARVEST_ROOT / "00_safe_harvestability"
DEFAULT_MEMORY_OPPORTUNITY_DIR = DEFAULT_SAFEHARVEST_ROOT / "03_opportunity_matching"
DEFAULT_OUTPUT = DEFAULT_SAFEHARVEST_ROOT / "04_job_risk_graph" / "sensitivity"
DEFAULT_TEMP = Path("/mnt/clusterdata0/duckdb_tmp")


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


def load_variant_jobs(
    expanded_candidates: Path,
    output_dir: Path,
    variant_name: str,
    temp_dir: Path,
    threads: int,
    memory_limit: str,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"frontier_matching_jobs_{variant_name}.csv"
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={threads}")
    con.execute(f"PRAGMA memory_limit='{memory_limit}'")
    con.execute("PRAGMA temp_directory=?", [str(temp_dir)])

    tier_order = risk_tier_order_case("sensitivity_risk_tier")
    tier_family = risk_tier_family_case("sensitivity_risk_tier")
    con.execute(
        f"""
        COPY (
          WITH unioned AS (
            {variant_union_sql(str(expanded_candidates), [variant_name])}
          ),
          scored AS (
            SELECT
              *,
              {tier_order} AS sensitivity_risk_tier_order,
              {tier_family} AS sensitivity_risk_tier_family
            FROM unioned
          )
          SELECT
            collection_id,
            sensitivity_risk_tier AS risk_tier,
            sensitivity_risk_tier_family AS risk_tier_family,
            sensitivity_risk_tier_order AS risk_tier_order,
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
          FROM scored
          WHERE schedule_win5 BETWEEN 6480 AND 8784
            AND sensitivity_risk_tier != 'X_excluded'
            AND total_cpu_request IS NOT NULL
            AND total_cpu_request > 0
            AND runtime_windows IS NOT NULL
            AND runtime_windows > 0
          ORDER BY schedule_win5, sensitivity_risk_tier_order, harvest_risk_score, cpu_window_demand
        ) TO '{out}' (HEADER, DELIMITER ',')
        """
    )
    jobs = pd.read_csv(out)
    jobs["runtime_windows"] = jobs["runtime_windows"].clip(lower=1).astype(int)
    jobs["class_priority"] = jobs["candidate_class"].map(CLASS_PRIORITY).fillna(99)
    return jobs


def run(
    expanded_candidates: Path,
    cpu_opportunity_dir: Path,
    memory_opportunity_dir: Path,
    output_dir: Path,
    temp_dir: Path,
    variants: list[str] | None,
    opportunity_classes: list[str],
    placement_policy_names: list[str] | None,
    threads: int,
    memory_limit: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    selected_variants = variant_by_name(variants)
    placement_policies = _selected_policies(placement_policy_names)

    summaries: list[dict[str, object]] = []
    samples: list[dict[str, object]] = []
    reason_rows: list[dict[str, object]] = []

    interval_cache = {}
    for variant in selected_variants:
        variant_name = str(variant["name"])
        jobs = load_variant_jobs(
            expanded_candidates,
            output_dir,
            variant_name,
            temp_dir,
            threads,
            memory_limit,
        )

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
                    summary["threshold_variant"] = variant_name
                    summaries.append(summary)
                    for sample in policy_samples[:500]:
                        sample["threshold_variant"] = variant_name
                        samples.append(sample)
                    reasons = Counter(reasons)
                    for reason, count in reasons.items():
                        reason_rows.append(
                            {
                                "threshold_variant": variant_name,
                                "frontier_policy": frontier_name,
                                "opportunity_class": opportunity_class,
                                "placement_policy": policy.name,
                                "reason": reason,
                                "jobs": count,
                            }
                        )

    pd.DataFrame(summaries).to_csv(
        output_dir / "frontier_matching_threshold_sensitivity_summary.csv",
        index=False,
    )
    pd.DataFrame(reason_rows).to_csv(
        output_dir / "frontier_matching_threshold_sensitivity_unplaced_reasons.csv",
        index=False,
    )
    pd.DataFrame(samples).to_csv(
        output_dir / "frontier_matching_threshold_sensitivity_sample.csv",
        index=False,
    )
    print(f"wrote frontier-matching threshold sensitivity outputs to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expanded-candidates", type=Path, default=DEFAULT_EXPANDED_CANDIDATES)
    parser.add_argument("--cpu-opportunity-dir", type=Path, default=DEFAULT_CPU_OPPORTUNITY_DIR)
    parser.add_argument("--memory-opportunity-dir", type=Path, default=DEFAULT_MEMORY_OPPORTUNITY_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--temp-dir", type=Path, default=DEFAULT_TEMP)
    parser.add_argument("--variants", nargs="+", default=["conservative", "baseline", "relaxed"])
    parser.add_argument(
        "--opportunity-classes",
        nargs="+",
        default=["10pct_1h", "20pct_1h", "10pct_4h", "20pct_4h"],
    )
    parser.add_argument("--placement-policies", nargs="+", default=None)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--memory-limit", default="64GB")
    args = parser.parse_args()
    run(
        args.expanded_candidates,
        args.cpu_opportunity_dir,
        args.memory_opportunity_dir,
        args.output,
        args.temp_dir,
        args.variants,
        args.opportunity_classes,
        args.placement_policies,
        args.threads,
        args.memory_limit,
    )


if __name__ == "__main__":
    main()
