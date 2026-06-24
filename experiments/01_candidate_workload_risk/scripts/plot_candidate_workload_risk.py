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

CLASS_ORDER = [
    "batch_scheduler",
    "best_effort_queueable",
    "low_priority_flexible",
    "medium_flexibility",
    "latency_sensitive_or_high_priority",
]

CLASS_LABELS = {
    "batch_scheduler": "Batch\nscheduler",
    "best_effort_queueable": "Best-effort\nqueueable",
    "low_priority_flexible": "Low-priority\nflexible",
    "medium_flexibility": "Medium\nflexibility",
    "latency_sensitive_or_high_priority": "Latency/high\npriority",
}

CLASS_COLORS = {
    "batch_scheduler": "#2563eb",
    "best_effort_queueable": "#16a34a",
    "low_priority_flexible": "#f59e0b",
    "medium_flexibility": "#64748b",
    "latency_sensitive_or_high_priority": "#dc2626",
}

OUTCOME_ORDER = ["FINISH", "KILL", "FAIL", "EVICT", "LOST", "NO_TERMINAL"]
OUTCOME_COLORS = {
    "FINISH": "#16a34a",
    "KILL": "#f97316",
    "FAIL": "#dc2626",
    "EVICT": "#7c3aed",
    "LOST": "#111827",
    "NO_TERMINAL": "#94a3b8",
}

RUNTIME_ORDER = ["00_<=5m", "01_5m_30m", "02_30m_1h", "03_1h_4h"]
RUNTIME_LABELS = {
    "00_<=5m": "<=5m",
    "01_5m_30m": "5-30m",
    "02_30m_1h": "30m-1h",
    "03_1h_4h": "1-4h",
}
RUNTIME_COLORS = {
    "00_<=5m": "#22c55e",
    "01_5m_30m": "#06b6d4",
    "02_30m_1h": "#f59e0b",
    "03_1h_4h": "#ef4444",
}


def _save(fig: plt.Figure, name: str) -> None:
    for ext in ["png", "pdf"]:
        fig.savefig(FIGURES / f"{name}.{ext}", dpi=220)
    plt.close(fig)


def plot_candidate_classes() -> None:
    df = pd.read_csv(RESULTS / "candidate_class_summary.csv")
    df = df.set_index("candidate_class").loc[CLASS_ORDER].reset_index()
    labels = [CLASS_LABELS[value] for value in df["candidate_class"]]
    colors = [CLASS_COLORS[value] for value in df["candidate_class"]]
    x = range(len(df))

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.9), constrained_layout=True)

    axes[0].bar(x, df["collections"] / 1_000_000.0, color=colors, alpha=0.86)
    axes[0].set_title("Collection population")
    axes[0].set_ylabel("Collections (million)")

    axes[1].bar(x, df["seed_candidates"] / 1_000.0, color=colors, alpha=0.86)
    axes[1].set_title("SafeHarvest seed candidates")
    axes[1].set_ylabel("Seed candidates (thousand)")

    for ax in axes:
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, fontsize=8)
        ax.grid(axis="y", color="#e5e7eb")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Candidate workload filtering by collection class", fontsize=13)
    _save(fig, "candidate_class_filter")


def plot_terminal_outcomes() -> None:
    df = pd.read_csv(RESULTS / "candidate_terminal_outcome_matrix.csv")
    pivot = (
        df.pivot_table(
            index="candidate_class",
            columns="terminal_outcome",
            values="collections",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(CLASS_ORDER)
        .reindex(columns=OUTCOME_ORDER, fill_value=0)
    )
    share = pivot.div(pivot.sum(axis=1), axis=0) * 100.0

    fig, ax = plt.subplots(figsize=(8.8, 4.2), constrained_layout=True)
    left = pd.Series(0.0, index=share.index)
    for outcome in OUTCOME_ORDER:
        values = share[outcome]
        ax.barh(
            [CLASS_LABELS[value].replace("\n", " ") for value in share.index],
            values,
            left=left,
            color=OUTCOME_COLORS[outcome],
            alpha=0.88,
            label=outcome.replace("_", " "),
        )
        left += values

    ax.set_xlabel("Collections (%)")
    ax.set_title("Terminal outcome mix by candidate class")
    ax.grid(axis="x", color="#e5e7eb")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(ncol=3, fontsize=8, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.28))
    _save(fig, "candidate_terminal_outcomes")


def plot_runtime_fit() -> None:
    df = pd.read_csv(RESULTS / "candidate_runtime_fit_summary.csv")
    pivot = (
        df.pivot_table(
            index="candidate_class",
            columns="runtime_bucket",
            values="seed_candidates",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(["batch_scheduler", "best_effort_queueable", "low_priority_flexible"])
        .reindex(columns=RUNTIME_ORDER, fill_value=0)
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), constrained_layout=True)

    bottom = pd.Series(0.0, index=pivot.index)
    for bucket in RUNTIME_ORDER:
        values = pivot[bucket] / 1000.0
        axes[0].bar(
            [CLASS_LABELS[value] for value in pivot.index],
            values,
            bottom=bottom / 1000.0,
            color=RUNTIME_COLORS[bucket],
            alpha=0.88,
            label=RUNTIME_LABELS[bucket],
        )
        bottom += pivot[bucket]

    share = pivot.div(pivot.sum(axis=1), axis=0) * 100.0
    bottom_share = pd.Series(0.0, index=share.index)
    for bucket in RUNTIME_ORDER:
        values = share[bucket]
        axes[1].bar(
            [CLASS_LABELS[value] for value in share.index],
            values,
            bottom=bottom_share,
            color=RUNTIME_COLORS[bucket],
            alpha=0.88,
            label=RUNTIME_LABELS[bucket],
        )
        bottom_share += values

    axes[0].set_ylabel("Seed candidates (thousand)")
    axes[0].set_title("Absolute volume")
    axes[1].set_ylabel("Seed candidates (%)")
    axes[1].set_title("Runtime fit")

    for ax in axes:
        ax.grid(axis="y", color="#e5e7eb")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", labelsize=8)
    axes[1].legend(
        ncol=4,
        fontsize=8,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.28),
    )

    fig.suptitle("SafeHarvest seed candidates are mostly short jobs", fontsize=13)
    _save(fig, "candidate_runtime_fit")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_candidate_classes()
    plot_terminal_outcomes()
    plot_runtime_fit()
    print(f"wrote figures to {FIGURES}")


if __name__ == "__main__":
    main()
