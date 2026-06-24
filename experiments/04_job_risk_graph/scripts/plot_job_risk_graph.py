#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("HOME", str(ROOT / ".home"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

TIER_ORDER = [
    "S0_strict_seed",
    "S1_clean_small_independent",
    "S2_clean_small_leaf_dependency",
    "S3_clean_medium_independent",
    "R1_retry_tolerant_signal",
]
TIER_LABELS = {
    "S0_strict_seed": "Strict\nseed",
    "S1_clean_small_independent": "Clean small\nindependent",
    "S2_clean_small_leaf_dependency": "Clean small\nleaf-dep",
    "S3_clean_medium_independent": "Clean medium\nindependent",
    "R1_retry_tolerant_signal": "Retry signal\nonly",
}
TIER_COLORS = {
    "S0_strict_seed": "#16a34a",
    "S1_clean_small_independent": "#2563eb",
    "S2_clean_small_leaf_dependency": "#0891b2",
    "S3_clean_medium_independent": "#f59e0b",
    "R1_retry_tolerant_signal": "#dc2626",
}

POLICY_LABELS = {
    "P0_strict_seed": "P0\nstrict",
    "P1_plus_clean_small_independent": "P1\n+indep",
    "P2_plus_clean_leaf_dependency": "P2\n+leaf-dep",
    "P3_plus_clean_medium_independent": "P3\n+medium",
    "P4_plus_retry_signal": "P4\n+retry",
}

ROLE_ORDER = [
    "independent_leaf",
    "dependent_leaf",
    "upstream_prerequisite",
    "internal_dependency_chain",
]
ROLE_LABELS = {
    "independent_leaf": "Independent leaf",
    "dependent_leaf": "Dependent leaf",
    "upstream_prerequisite": "Upstream prerequisite",
    "internal_dependency_chain": "Internal chain",
}
ROLE_COLORS = {
    "independent_leaf": "#16a34a",
    "dependent_leaf": "#2563eb",
    "upstream_prerequisite": "#f59e0b",
    "internal_dependency_chain": "#64748b",
}

CLASS_ORDER = [
    "batch_scheduler",
    "best_effort_queueable",
    "low_priority_flexible",
    "medium_flexibility",
    "latency_sensitive_or_high_priority",
]
CLASS_LABELS = {
    "batch_scheduler": "Batch",
    "best_effort_queueable": "Best-effort",
    "low_priority_flexible": "Low-priority",
    "medium_flexibility": "Medium",
    "latency_sensitive_or_high_priority": "Latency/high",
}


def _save(fig: plt.Figure, name: str) -> None:
    for ext in ["png", "pdf"]:
        fig.savefig(FIGURES / f"{name}.{ext}", dpi=220)
    plt.close(fig)


def plot_risk_tier_funnel() -> None:
    df = pd.read_csv(RESULTS / "risk_tier_summary.csv")
    df = df[df["risk_tier"].isin(TIER_ORDER)].set_index("risk_tier").reindex(TIER_ORDER)
    x = range(len(df))

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), constrained_layout=True)
    axes[0].bar(
        x,
        df["test_window_collections"] / 1000.0,
        color=[TIER_COLORS[t] for t in df.index],
        alpha=0.88,
    )
    axes[0].set_ylabel("Test-window collections (thousand)")
    axes[0].set_title("Candidate volume")

    axes[1].bar(
        x,
        df["cpu_window_demand"] / 1000.0,
        color=[TIER_COLORS[t] for t in df.index],
        alpha=0.88,
    )
    axes[1].set_ylabel("CPU-window demand (thousand)")
    axes[1].set_title("Demand volume")

    for ax in axes:
        ax.set_xticks(list(x))
        ax.set_xticklabels([TIER_LABELS[t] for t in df.index], fontsize=8)
        ax.grid(axis="y", color="#e5e7eb")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Job Risk Graph expansion tiers", fontsize=13)
    _save(fig, "risk_tier_funnel")


def plot_expansion_frontier() -> None:
    df = pd.read_csv(RESULTS / "expansion_frontier_summary.csv")
    df = df.set_index("frontier_policy").reindex(POLICY_LABELS.keys()).reset_index()
    x = range(len(df))

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0), constrained_layout=True)
    axes[0].plot(x, df["test_window_collections"] / 1000.0, marker="o", color="#2563eb")
    axes[0].set_ylabel("Test-window collections (thousand)")
    axes[0].set_title("Cumulative harvest pool")

    axes[1].plot(x, df["cpu_window_demand"] / 1000.0, marker="o", color="#16a34a", label="CPU-window")
    axes[1].set_ylabel("CPU-window demand (thousand)")
    axes[1].set_title("Cumulative demand")
    ax2 = axes[1].twinx()
    ax2.plot(
        x,
        df["instance_bad_terminal_share"].fillna(0.0) * 100.0,
        marker="s",
        color="#dc2626",
        label="Bad terminal",
    )
    ax2.set_ylabel("Bad terminal instances (%)")

    for ax in axes:
        ax.set_xticks(list(x))
        ax.set_xticklabels([POLICY_LABELS[p] for p in df["frontier_policy"]], fontsize=8)
        ax.grid(axis="y", color="#e5e7eb")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    fig.suptitle("Controlled expansion frontier", fontsize=13)
    _save(fig, "expansion_frontier")


def plot_dependency_profile() -> None:
    df = pd.read_csv(RESULTS / "dependency_role_summary.csv")
    pivot = (
        df.pivot_table(
            index="candidate_class",
            columns="dependency_graph_role",
            values="collections",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(CLASS_ORDER)
        .reindex(columns=ROLE_ORDER, fill_value=0)
    )
    share = pivot.div(pivot.sum(axis=1), axis=0) * 100.0

    fig, ax = plt.subplots(figsize=(9.4, 4.2), constrained_layout=True)
    left = pd.Series(0.0, index=share.index)
    labels = [CLASS_LABELS[c] for c in share.index]
    for role in ROLE_ORDER:
        values = share[role]
        ax.barh(
            labels,
            values,
            left=left,
            color=ROLE_COLORS[role],
            alpha=0.88,
            label=ROLE_LABELS[role],
        )
        left += values

    ax.set_xlabel("Collections (%)")
    ax.set_title("Dependency-graph role by collection class")
    ax.grid(axis="x", color="#e5e7eb")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(ncol=2, fontsize=8, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.30))
    _save(fig, "dependency_graph_profile")


def plot_frontier_matching() -> None:
    path = RESULTS / "frontier_matching_summary.csv"
    if not path.exists():
        return

    df = pd.read_csv(path)
    mem_max = df[df["placement_policy"] == "frontier_min_cpu_mem_max_exact"].copy()
    frontier_order = list(POLICY_LABELS.keys())
    front = (
        mem_max[mem_max["opportunity_class"] == "10pct_1h"]
        .set_index("frontier_policy")
        .reindex(frontier_order)
        .reset_index()
    )
    p3 = (
        mem_max[mem_max["frontier_policy"] == "P3_plus_clean_medium_independent"]
        .set_index("opportunity_class")
        .reindex(["10pct_1h", "20pct_1h", "10pct_4h", "20pct_4h"])
        .reset_index()
    )

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.0), constrained_layout=True)
    x = range(len(front))
    axes[0].bar(x, front["eligible_jobs"] / 1000.0, color="#2563eb", alpha=0.88)
    axes[0].set_title("Expanded pool")
    axes[0].set_ylabel("Eligible jobs (thousand)")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels([POLICY_LABELS[p] for p in front["frontier_policy"]], fontsize=8)

    axes[1].bar(x, front["placement_rate"] * 100.0, color="#16a34a", alpha=0.88)
    axes[1].set_ylim(99.0, 100.05)
    axes[1].set_title("10pct_1h placement")
    axes[1].set_ylabel("Jobs placed (%)")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels([POLICY_LABELS[p] for p in front["frontier_policy"]], fontsize=8)

    x2 = range(len(p3))
    axes[2].bar(x2, p3["placed_work_share"] * 100.0, color="#f59e0b", alpha=0.88)
    axes[2].set_ylim(93.5, 100.2)
    axes[2].set_title("P3 CPU work admitted")
    axes[2].set_ylabel("Demanded CPU-window placed (%)")
    axes[2].set_xticks(list(x2))
    axes[2].set_xticklabels(p3["opportunity_class"], fontsize=8)

    for ax in axes:
        ax.grid(axis="y", color="#e5e7eb")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Exact CPU+memory-max matching for graph-aware frontiers", fontsize=13)
    _save(fig, "frontier_matching_summary")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_risk_tier_funnel()
    plot_expansion_frontier()
    plot_dependency_profile()
    plot_frontier_matching()
    print(f"wrote figures to {FIGURES}")


if __name__ == "__main__":
    main()
