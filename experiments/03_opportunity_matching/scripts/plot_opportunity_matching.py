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

POLICY_ORDER = ["avg_room_replay", "p10_room_replay", "min_room_replay", "safeharvest_v1"]
POLICY_LABELS = {
    "avg_room_replay": "Avg-room\nreplay",
    "p10_room_replay": "P10-room\nreplay",
    "min_room_replay": "Min-room\nreplay",
    "safeharvest_v1": "SafeHarvest\nv1",
}
POLICY_COLORS = {
    "avg_room_replay": "#64748b",
    "p10_room_replay": "#f59e0b",
    "min_room_replay": "#2563eb",
    "safeharvest_v1": "#16a34a",
}


def _save(fig: plt.Figure, name: str) -> None:
    for ext in ["png", "pdf"]:
        fig.savefig(FIGURES / f"{name}.{ext}", dpi=220)
    plt.close(fig)


def plot_policy_summary() -> None:
    df = pd.read_csv(RESULTS / "matching_policy_summary.csv")
    df = df[df["opportunity_class"] == "10pct_1h"].set_index("policy").loc[POLICY_ORDER]
    labels = [POLICY_LABELS[p] for p in df.index]
    colors = [POLICY_COLORS[p] for p in df.index]
    x = range(len(df))

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.0), constrained_layout=True)

    axes[0].bar(x, df["placement_rate"] * 100.0, color=colors, alpha=0.88)
    axes[0].set_ylim(98.8, 100.1)
    axes[0].set_ylabel("Eligible jobs placed (%)")
    axes[0].set_title("Placement rate")

    axes[1].bar(x, df["placed_work_share"] * 100.0, color=colors, alpha=0.88)
    axes[1].set_ylabel("Demanded CPU-window placed (%)")
    axes[1].set_title("Work admitted")

    axes[2].bar(x, df["placed_bad_terminal_instances"], color=colors, alpha=0.88)
    axes[2].set_ylabel("Bad terminal instances")
    axes[2].set_title("Residual workload risk")

    for ax in axes:
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, fontsize=8)
        ax.grid(axis="y", color="#e5e7eb")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Aggregate replay on 10pct_1h stable opportunity windows", fontsize=13)
    _save(fig, "matching_policy_summary")


def plot_opportunity_classes() -> None:
    df = pd.read_csv(RESULTS / "matching_policy_summary.csv")
    safe = df[df["policy"] == "safeharvest_v1"].copy()
    x = range(len(safe))

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8), constrained_layout=True)
    axes[0].bar(x, safe["placement_rate"] * 100.0, color="#16a34a", alpha=0.88)
    axes[0].set_ylim(98.8, 100.1)
    axes[0].set_ylabel("Eligible jobs placed (%)")
    axes[0].set_title("SafeHarvest v1 placement")

    axes[1].bar(x, safe["used_opportunity_share"] * 100.0, color="#2563eb", alpha=0.88)
    axes[1].set_ylabel("Opportunity CPU-window used (%)")
    axes[1].set_title("Capacity used")

    for ax in axes:
        ax.set_xticks(list(x))
        ax.set_xticklabels(safe["opportunity_class"], fontsize=8)
        ax.grid(axis="y", color="#e5e7eb")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("SafeHarvest demand is much smaller than stable opportunity supply", fontsize=13)
    _save(fig, "safeharvest_by_opportunity_class")


def plot_exact_machine_summary() -> None:
    df = pd.read_csv(RESULTS / "exact_machine_policy_summary.csv")
    order = ["avg_room_exact", "p10_room_exact", "min_room_exact", "safeharvest_v1_exact"]
    labels = {
        "avg_room_exact": "Avg-room\nexact",
        "p10_room_exact": "P10-room\nexact",
        "min_room_exact": "Min-room\nexact",
        "safeharvest_v1_exact": "SafeHarvest\nexact",
    }
    colors = {
        "avg_room_exact": "#64748b",
        "p10_room_exact": "#f59e0b",
        "min_room_exact": "#2563eb",
        "safeharvest_v1_exact": "#16a34a",
    }
    df = df[df["opportunity_class"] == "10pct_1h"].set_index("policy").loc[order]
    x = range(len(df))

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.0), constrained_layout=True)
    axes = axes.ravel()

    axes[0].bar(x, df["placement_rate"] * 100.0, color=[colors[p] for p in df.index], alpha=0.88)
    axes[0].set_ylim(98.8, 100.1)
    axes[0].set_ylabel("Eligible jobs placed (%)")
    axes[0].set_title("Placement rate")

    axes[1].bar(x, df["placed_work_share"] * 100.0, color=[colors[p] for p in df.index], alpha=0.88)
    axes[1].set_ylabel("Demanded CPU-window placed (%)")
    axes[1].set_title("Work admitted")

    axes[2].bar(x, df["placed_bad_terminal_instances"], color=[colors[p] for p in df.index], alpha=0.88)
    axes[2].set_ylabel("Bad terminal instances")
    axes[2].set_title("Residual workload risk")

    axes[3].bar(x, df["used_machines"], color=[colors[p] for p in df.index], alpha=0.88)
    axes[3].set_ylabel("Machines used")
    axes[3].set_title("Placement footprint")

    for ax in axes:
        ax.set_xticks(list(x))
        ax.set_xticklabels([labels[p] for p in df.index], fontsize=8)
        ax.grid(axis="y", color="#e5e7eb")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Exact per-machine placement on 10pct_1h windows", fontsize=13)
    _save(fig, "exact_machine_policy_summary")


def plot_exact_safeharvest_classes() -> None:
    df = pd.read_csv(RESULTS / "exact_machine_policy_summary.csv")
    safe = df[df["policy"] == "safeharvest_v1_exact"].copy()
    x = range(len(safe))

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8), constrained_layout=True)
    axes[0].bar(x, safe["placement_rate"] * 100.0, color="#16a34a", alpha=0.88)
    axes[0].set_ylim(98.8, 100.1)
    axes[0].set_ylabel("Eligible jobs placed (%)")
    axes[0].set_title("Exact SafeHarvest placement")

    axes[1].bar(x, safe["used_machines"], color="#2563eb", alpha=0.88)
    axes[1].set_ylabel("Machines used")
    axes[1].set_title("Placement footprint")

    for ax in axes:
        ax.set_xticks(list(x))
        ax.set_xticklabels(safe["opportunity_class"], fontsize=8)
        ax.grid(axis="y", color="#e5e7eb")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("SafeHarvest remains stable under exact machine placement", fontsize=13)
    _save(fig, "exact_safeharvest_by_opportunity_class")


def plot_memory_aware_safeharvest() -> None:
    df = pd.read_csv(RESULTS / "exact_machine_policy_summary.csv")
    policies = [
        "safeharvest_v1_exact",
        "safeharvest_v1_cpu_mem_avg_exact",
        "safeharvest_v1_cpu_mem_max_exact",
    ]
    labels = {
        "safeharvest_v1_exact": "CPU only",
        "safeharvest_v1_cpu_mem_avg_exact": "CPU+memory\navg usage",
        "safeharvest_v1_cpu_mem_max_exact": "CPU+memory\nmax usage",
    }
    colors = {
        "safeharvest_v1_exact": "#16a34a",
        "safeharvest_v1_cpu_mem_avg_exact": "#2563eb",
        "safeharvest_v1_cpu_mem_max_exact": "#f59e0b",
    }
    opportunity_order = ["10pct_1h", "20pct_1h", "10pct_4h", "20pct_4h"]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0), constrained_layout=True)
    width = 0.24
    x = list(range(len(opportunity_order)))

    for idx, policy in enumerate(policies):
        sub = (
            df[df["policy"] == policy]
            .set_index("opportunity_class")
            .loc[opportunity_order]
        )
        offset = (idx - 1) * width
        axes[0].bar(
            [v + offset for v in x],
            sub["placement_rate"] * 100.0,
            width=width,
            color=colors[policy],
            alpha=0.88,
            label=labels[policy],
        )
        axes[1].bar(
            [v + offset for v in x],
            sub["placed_memory_work_share"] * 100.0,
            width=width,
            color=colors[policy],
            alpha=0.88,
            label=labels[policy],
        )

    axes[0].set_ylim(99.4, 100.05)
    axes[0].set_ylabel("Eligible jobs placed (%)")
    axes[0].set_title("Placement rate")
    axes[1].set_ylim(98.5, 100.05)
    axes[1].set_ylabel("Demanded memory-window placed (%)")
    axes[1].set_title("Memory work admitted")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(opportunity_order, fontsize=8)
        ax.grid(axis="y", color="#e5e7eb")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[1].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("Memory-aware exact SafeHarvest placement", fontsize=13)
    _save(fig, "memory_aware_safeharvest_summary")


def plot_memory_room_profile() -> None:
    df = pd.read_csv(RESULTS / "memory_enriched_opportunity_summary.csv")
    order = ["10pct_1h", "20pct_1h", "10pct_4h", "20pct_4h"]
    df = df.set_index("opportunity_class").loc[order].reset_index()
    x = range(len(df))

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8), constrained_layout=True)
    axes[0].bar(
        x,
        df["p50_min_memory_room_avg_usage"],
        color="#2563eb",
        alpha=0.88,
        label="p50",
    )
    axes[0].scatter(
        list(x),
        df["p10_min_memory_room_avg_usage"],
        color="#111827",
        s=24,
        label="p10",
    )
    axes[0].set_ylabel("Normalized memory room")
    axes[0].set_title("Minimum memory room per interval")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].bar(x, df["avg_memory_violation_windows"] / 1000.0, color="#dc2626", alpha=0.82)
    axes[1].set_ylabel("Windows (thousand)")
    axes[1].set_title("Avg-usage memory over-cap windows")

    for ax in axes:
        ax.set_xticks(list(x))
        ax.set_xticklabels(order, fontsize=8)
        ax.grid(axis="y", color="#e5e7eb")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Memory headroom is much tighter than CPU headroom", fontsize=13)
    _save(fig, "memory_room_profile")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_policy_summary()
    plot_opportunity_classes()
    plot_exact_machine_summary()
    plot_exact_safeharvest_classes()
    plot_memory_aware_safeharvest()
    plot_memory_room_profile()
    print(f"wrote figures to {FIGURES}")


if __name__ == "__main__":
    main()
