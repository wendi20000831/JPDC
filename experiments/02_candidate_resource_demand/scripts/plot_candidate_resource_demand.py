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

CLASS_ORDER = ["batch_scheduler", "best_effort_queueable", "low_priority_flexible"]
CLASS_LABELS = {
    "batch_scheduler": "Batch scheduler",
    "best_effort_queueable": "Best-effort queueable",
    "low_priority_flexible": "Low-priority flexible",
}
CLASS_COLORS = {
    "batch_scheduler": "#2563eb",
    "best_effort_queueable": "#16a34a",
    "low_priority_flexible": "#f59e0b",
}

CPU_BUCKET_LABELS = {
    "00_<=0.05": "<=0.05",
    "01_0.05_0.10": "0.05-0.10",
    "02_0.10_0.25": "0.10-0.25",
    "03_0.25_0.50": "0.25-0.50",
    "04_0.50_1.00": "0.50-1.00",
    "05_1.00_2.00": "1.00-2.00",
    "06_>2.00": ">2.00",
}

INSTANCE_BUCKET_LABELS = {
    "00_1": "1",
    "01_2_5": "2-5",
    "02_6_20": "6-20",
    "03_21_100": "21-100",
    "04_101_1000": "101-1000",
    "05_>1000": ">1000",
}


def _save(fig: plt.Figure, name: str) -> None:
    for ext in ["png", "pdf"]:
        fig.savefig(FIGURES / f"{name}.{ext}", dpi=220)
    plt.close(fig)


def _plot_stacked_bucket(
    ax: plt.Axes,
    df: pd.DataFrame,
    bucket_col: str,
    label_map: dict[str, str],
    title: str,
) -> None:
    pivot = (
        df.pivot_table(
            index=bucket_col,
            columns="candidate_class",
            values="collections",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(label_map.keys())
        .reindex(columns=CLASS_ORDER, fill_value=0)
    )

    bottom = pd.Series(0.0, index=pivot.index)
    for klass in CLASS_ORDER:
        ax.bar(
            [label_map[value] for value in pivot.index],
            pivot[klass] / 1000.0,
            bottom=bottom / 1000.0,
            color=CLASS_COLORS[klass],
            alpha=0.88,
            label=CLASS_LABELS[klass],
        )
        bottom += pivot[klass]

    ax.set_title(title)
    ax.set_ylabel("Collections (thousand)")
    ax.grid(axis="y", color="#e5e7eb")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=8, rotation=20)


def plot_bucket_profile() -> None:
    cpu_df = pd.read_csv(RESULTS / "resource_demand_buckets.csv")
    instance_df = pd.read_csv(RESULTS / "instance_count_buckets.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4), constrained_layout=True)
    _plot_stacked_bucket(
        axes[0],
        cpu_df,
        "total_cpu_request_bucket",
        CPU_BUCKET_LABELS,
        "Total CPU request",
    )
    _plot_stacked_bucket(
        axes[1],
        instance_df,
        "instance_count_bucket",
        INSTANCE_BUCKET_LABELS,
        "Instance count",
    )
    axes[1].legend(ncol=1, fontsize=8, frameon=False, loc="upper right")
    fig.suptitle("SafeHarvest candidates are small enough for fine-grained gaps", fontsize=13)
    _save(fig, "candidate_resource_bucket_profile")


def plot_class_quantiles() -> None:
    df = pd.read_csv(RESULTS / "resource_demand_by_class.csv")
    df = df.set_index("candidate_class").loc[CLASS_ORDER].reset_index()
    labels = [CLASS_LABELS[value] for value in df["candidate_class"]]
    colors = [CLASS_COLORS[value] for value in df["candidate_class"]]
    x = range(len(df))

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1), constrained_layout=True)

    axes[0].bar(x, df["p90_total_cpu_request"], color=colors, alpha=0.88)
    axes[0].scatter(x, df["p99_total_cpu_request"], color="#111827", s=24, label="p99")
    axes[0].set_title("Total CPU request")
    axes[0].set_ylabel("Normalized CPU request")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].bar(x, df["instance_finish_share"] * 100.0, color=colors, alpha=0.88)
    axes[1].set_ylim(80, 100)
    axes[1].set_title("Instance finish share")
    axes[1].set_ylabel("Instances finished (%)")

    for ax in axes:
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, fontsize=8)
        ax.grid(axis="y", color="#e5e7eb")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Resource size and residual instance risk by class", fontsize=13)
    _save(fig, "candidate_resource_by_class")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_bucket_profile()
    plot_class_quantiles()
    print(f"wrote figures to {FIGURES}")


if __name__ == "__main__":
    main()
