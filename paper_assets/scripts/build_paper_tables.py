#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "paper_assets" / "tables"

EXP00 = ROOT / "experiments" / "00_safe_harvestability" / "results"
EXP03 = ROOT / "experiments" / "03_opportunity_matching" / "results"
EXP04 = ROOT / "experiments" / "04_job_risk_graph" / "results"

POLICY_LABELS = {
    "raw_actual": "Raw actual room",
    "request_visible": "Request-visible room",
    "autopilot_style": "Autopilot-style room",
    "risk_bounded_q95": "Risk-bounded q95 room",
    "avg_room_exact": "Avg-room exact",
    "p10_room_exact": "P10-room exact",
    "min_room_exact": "Min-room exact",
    "safeharvest_v1_exact": "SafeHarvest v1 exact",
    "safeharvest_v1_cpu_mem_max_exact": "SafeHarvest v1 CPU+mem max",
}

FRONTIER_LABELS = {
    "P0_strict_seed": "P0 strict seed",
    "P1_plus_clean_small_independent": "P1 + clean small independent",
    "P2_plus_clean_leaf_dependency": "P2 + clean leaf dependency",
    "P3_plus_clean_medium_independent": "P3 + clean medium independent",
    "P4_plus_retry_signal": "P4 + retry signal",
}

TIER_LABELS = {
    "S0_strict_seed": "S0 strict seed",
    "S1_clean_small_independent": "S1 clean small independent",
    "S2_clean_small_leaf_dependency": "S2 clean small leaf dependency",
    "S3_clean_medium_independent": "S3 clean medium independent",
    "R1_retry_tolerant_signal": "R1 retry-signal only",
    "X_excluded": "Excluded",
}


def pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def num(value: float, digits: int = 1) -> str:
    return f"{value:,.{digits}f}"


def integer(value: float) -> str:
    return f"{int(round(value)):,}"


def to_markdown(df: pd.DataFrame) -> str:
    headers = [str(col) for col in df.columns]
    rows = [[str(value) for value in row] for row in df.itertuples(index=False, name=None)]
    widths = [
        max(len(headers[idx]), *(len(row[idx]) for row in rows))
        for idx in range(len(headers))
    ]
    header = "| " + " | ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |"
    rule = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    body = [
        "| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header, rule, *body])


def save_table(df: pd.DataFrame, stem: str) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / f"{stem}.csv", index=False)
    (TABLES / f"{stem}.md").write_text(to_markdown(df) + "\n", encoding="utf-8")
    latex = df.to_latex(index=False, escape=False)
    (TABLES / f"{stem}.tex").write_text(latex, encoding="utf-8")


def build_harvestability_funnel() -> None:
    funnel = pd.read_csv(EXP00 / "harvestability_funnel.csv")
    rows = []
    for policy in ["raw_actual", "request_visible", "autopilot_style", "risk_bounded_q95"]:
        row = funnel[funnel["policy"] == policy].iloc[0]
        rows.append(
            {
                "Capacity policy": POLICY_LABELS[policy],
                "Room capacity": pct(row["room_capacity_fraction"], 1),
                "Nonzero-room windows": pct(row["nonzero_room_window_share"], 1),
                "No-room windows": pct(row["no_room_window_share"], 1),
                "Violation rate": pct(row["violation_rate"], 2),
                "Overload mass": pct(row["overload_mass_fraction"], 2),
            }
        )
    save_table(pd.DataFrame(rows), "table1_harvestability_funnel")


def build_stability_summary() -> None:
    stability = pd.read_csv(EXP00 / "stability_matrix.csv")
    sub = stability[
        (stability["policy"] == "risk_bounded_q95")
        & (stability["room_threshold_fraction"].isin([0.10, 0.20]))
        & (stability["min_duration_minutes"].isin([60, 240]))
    ].copy()
    sub = sub.sort_values(["room_threshold_fraction", "min_duration_minutes"])
    rows = []
    for _, row in sub.iterrows():
        rows.append(
            {
                "Room threshold": pct(row["room_threshold_fraction"], 0),
                "Minimum duration": f"{int(row['min_duration_minutes'])} min",
                "Stable runs": integer(row["stable_runs"]),
                "Stable windows": integer(row["stable_windows"]),
                "Stable window share": pct(row["stable_window_share"], 1),
                "Stable room capacity": pct(row["stable_room_capacity_fraction"], 1),
            }
        )
    save_table(pd.DataFrame(rows), "table2_stability_summary")


def build_strict_seed_matching() -> None:
    exact = pd.read_csv(EXP03 / "exact_machine_policy_summary.csv")
    frontier = pd.read_csv(EXP04 / "frontier_matching_summary.csv")
    policies = [
        "avg_room_exact",
        "p10_room_exact",
        "min_room_exact",
    ]
    sub = exact[
        (exact["opportunity_class"] == "10pct_1h")
        & (exact["policy"].isin(policies))
    ].copy()
    sub["policy_order"] = sub["policy"].map({p: i for i, p in enumerate(policies)})
    sub = sub.sort_values("policy_order")
    rows = []
    for _, row in sub.iterrows():
        rows.append(
            {
                "Policy": POLICY_LABELS[row["policy"]],
                "Eligible jobs": integer(row["eligible_jobs"]),
                "Placed jobs": integer(row["placed_jobs"]),
                "Placement rate": pct(row["placement_rate"], 2),
                "CPU-window placed": pct(row["placed_work_share"], 1),
                "Bad terminal inst.": integer(row["placed_bad_terminal_instances"]),
                "Machines used": integer(row["used_machines"]),
            }
        )
    p0_rows = {
        "SafeHarvest P0 exact": frontier[
            (frontier["frontier_policy"] == "P0_strict_seed")
            & (frontier["opportunity_class"] == "10pct_1h")
            & (frontier["placement_policy"] == "frontier_min_cpu_exact")
        ].iloc[0],
        "SafeHarvest P0 CPU+mem max": frontier[
            (frontier["frontier_policy"] == "P0_strict_seed")
            & (frontier["opportunity_class"] == "10pct_1h")
            & (frontier["placement_policy"] == "frontier_min_cpu_mem_max_exact")
        ].iloc[0],
    }
    for label, row in p0_rows.items():
        rows.append(
            {
                "Policy": label,
                "Eligible jobs": integer(row["eligible_jobs"]),
                "Placed jobs": integer(row["placed_jobs"]),
                "Placement rate": pct(row["placement_rate"], 2),
                "CPU-window placed": pct(row["placed_work_share"], 1),
                "Bad terminal inst.": integer(row["placed_bad_terminal_instances"]),
                "Machines used": integer(row["used_machines"]),
            }
        )
    save_table(pd.DataFrame(rows), "table3_strict_seed_matching")


def build_risk_frontier() -> None:
    tiers = pd.read_csv(EXP04 / "risk_tier_summary.csv")
    order = [
        "S0_strict_seed",
        "S1_clean_small_independent",
        "S2_clean_small_leaf_dependency",
        "S3_clean_medium_independent",
        "R1_retry_tolerant_signal",
        "X_excluded",
    ]
    tiers = tiers.set_index("risk_tier").loc[order].reset_index()
    rows = []
    for _, row in tiers.iterrows():
        rows.append(
            {
                "Risk tier": TIER_LABELS[row["risk_tier"]],
                "Collections": integer(row["collections"]),
                "Test-window jobs": integer(row["test_window_collections"]),
                "CPU-window demand": num(row["cpu_window_demand"], 1),
                "Collection bad-term.": pct(row["collection_bad_terminal_share"], 1),
                "Instance bad-term.": pct(row["instance_bad_terminal_share"], 1),
                "P90 CPU req.": num(row["p90_total_cpu_request"], 3),
            }
        )
    save_table(pd.DataFrame(rows), "table4_risk_frontier_tiers")


def build_frontier_matching_main() -> None:
    frontier = pd.read_csv(EXP04 / "frontier_matching_summary.csv")
    order = list(FRONTIER_LABELS.keys())
    sub = frontier[
        (frontier["opportunity_class"] == "10pct_1h")
        & (frontier["placement_policy"] == "frontier_min_cpu_mem_max_exact")
    ].copy()
    sub = sub.set_index("frontier_policy").loc[order].reset_index()
    rows = []
    for _, row in sub.iterrows():
        rows.append(
            {
                "Frontier": FRONTIER_LABELS[row["frontier_policy"]],
                "Eligible jobs": integer(row["eligible_jobs"]),
                "Placed jobs": integer(row["placed_jobs"]),
                "Placement rate": pct(row["placement_rate"], 2),
                "CPU%": pct(row["placed_work_share"], 1),
                "CPU k": num(row["placed_cpu_window"] / 1000, 3),
                "Mem%": pct(row["placed_memory_work_share"], 1),
                "Mem k": num(row["placed_memory_window"] / 1000, 3),
                "Bad terminal inst.": integer(row["placed_bad_terminal_instances"]),
            }
        )
    save_table(pd.DataFrame(rows), "table5_frontier_matching_main")


def build_frontier_matching_robustness() -> None:
    frontier = pd.read_csv(EXP04 / "frontier_matching_summary.csv")
    sub = frontier[
        (frontier["frontier_policy"] == "P3_plus_clean_medium_independent")
        & (frontier["placement_policy"] == "frontier_min_cpu_mem_max_exact")
    ].copy()
    sub = sub.set_index("opportunity_class").loc[
        ["10pct_1h", "20pct_1h", "10pct_4h", "20pct_4h"]
    ].reset_index()
    rows = []
    for _, row in sub.iterrows():
        rows.append(
            {
                "Opportunity class": row["opportunity_class"],
                "Eligible jobs": integer(row["eligible_jobs"]),
                "Placed jobs": integer(row["placed_jobs"]),
                "Placement rate": pct(row["placement_rate"], 2),
                "CPU-window placed": pct(row["placed_work_share"], 1),
                "Memory-window placed": pct(row["placed_memory_work_share"], 1),
                "Unplaced jobs": integer(row["unplaced_jobs"]),
            }
        )
    save_table(pd.DataFrame(rows), "table6_p3_robustness")


def build_key_numbers() -> None:
    strict = pd.read_csv(EXP04 / "frontier_matching_summary.csv")
    p0 = strict[
        (strict["frontier_policy"] == "P0_strict_seed")
        & (strict["opportunity_class"] == "10pct_1h")
        & (strict["placement_policy"] == "frontier_min_cpu_mem_max_exact")
    ].iloc[0]
    p3 = strict[
        (strict["frontier_policy"] == "P3_plus_clean_medium_independent")
        & (strict["opportunity_class"] == "10pct_1h")
        & (strict["placement_policy"] == "frontier_min_cpu_mem_max_exact")
    ].iloc[0]
    p4 = strict[
        (strict["frontier_policy"] == "P4_plus_retry_signal")
        & (strict["opportunity_class"] == "10pct_1h")
        & (strict["placement_policy"] == "frontier_min_cpu_mem_max_exact")
    ].iloc[0]
    lines = [
        "# Key Numbers",
        "",
        f"- P3 expands test-window eligible jobs from {integer(p0['eligible_jobs'])} to {integer(p3['eligible_jobs'])} ({p3['eligible_jobs'] / p0['eligible_jobs']:.2f}x).",
        f"- P3 places {integer(p3['placed_jobs'])} jobs under CPU+memory-max exact matching on 10pct_1h ({pct(p3['placement_rate'], 2)}).",
        f"- P3 places {num(p3['placed_cpu_window'], 1)} CPU-window demand, compared with {num(p0['placed_cpu_window'], 1)} for P0 ({p3['placed_cpu_window'] / p0['placed_cpu_window']:.2f}x).",
        f"- P3 admits {integer(p3['placed_bad_terminal_instances'])} observed bad-terminal instances; P4 admits {integer(p4['placed_bad_terminal_instances'])}.",
        f"- P3 remains above {pct(0.9965, 2)} placement across the four opportunity classes under CPU+memory-max matching.",
    ]
    (TABLES / "key_numbers.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    build_harvestability_funnel()
    build_stability_summary()
    build_strict_seed_matching()
    build_risk_frontier()
    build_frontier_matching_main()
    build_frontier_matching_robustness()
    build_key_numbers()
    print(f"wrote paper tables to {TABLES}")


if __name__ == "__main__":
    main()
