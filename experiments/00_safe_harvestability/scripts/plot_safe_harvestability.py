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

POLICY_LABELS = {
    "raw_actual": "Raw\nactual",
    "request_visible": "Request\nvisible",
    "autopilot_style": "Autopilot\nstyle",
    "risk_bounded_q95": "Risk\nbounded",
}

COLORS = {
    "raw_actual": "#64748b",
    "request_visible": "#ef4444",
    "autopilot_style": "#f59e0b",
    "risk_bounded_q95": "#16a34a",
}

ORDER = ["raw_actual", "request_visible", "autopilot_style", "risk_bounded_q95"]


def plot_funnel() -> None:
    df = pd.read_csv(RESULTS / "harvestability_funnel.csv").set_index("policy").loc[ORDER]
    x = range(len(df))
    labels = [POLICY_LABELS[p] for p in df.index]
    colors = [COLORS[p] for p in df.index]

    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.8), constrained_layout=True)

    axes[0].bar(x, df["room_capacity_fraction"] * 100.0, color=colors, alpha=0.88)
    axes[0].set_ylabel("Room / capacity-window (%)")
    axes[0].set_title("Available room")

    axes[1].bar(x, df["violation_rate"] * 100.0, color=colors, alpha=0.88)
    axes[1].set_ylabel("Violation rate (%)")
    axes[1].set_title("Unsafe if harvested")

    axes[2].bar(x, df["no_room_window_share"] * 100.0, color=colors, alpha=0.88)
    axes[2].set_ylabel("No-room windows (%)")
    axes[2].set_title("Windows with no headroom")

    for ax in axes:
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, fontsize=8)
        ax.grid(axis="y", color="#e5e7eb")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("SafeHarvest capacity funnel on cell_a test window", fontsize=13)
    for ext in ["png", "pdf"]:
        fig.savefig(FIGURES / f"safeharvest_funnel.{ext}", dpi=220)
    plt.close(fig)


def plot_stability() -> None:
    df = pd.read_csv(RESULTS / "stability_matrix.csv")
    df = df[df["policy"] == "risk_bounded_q95"].copy()
    df["duration_label"] = df["min_duration_minutes"].map(
        {5: "5m", 30: "30m", 60: "1h", 240: "4h"}
    )
    pivot = df.pivot(
        index="room_threshold_fraction",
        columns="duration_label",
        values="stable_room_capacity_fraction",
    )[["5m", "30m", "1h", "4h"]]

    fig, ax = plt.subplots(figsize=(6.4, 3.8), constrained_layout=True)
    image = ax.imshow(pivot.to_numpy() * 100.0, cmap="YlGn", vmin=0, vmax=40)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v:.0%}" for v in pivot.index])
    ax.set_xlabel("Minimum contiguous duration")
    ax.set_ylabel("Minimum room threshold")
    ax.set_title("Risk-bounded stable harvestable room")
    for i, threshold in enumerate(pivot.index):
        for j, duration in enumerate(pivot.columns):
            value = pivot.loc[threshold, duration] * 100.0
            ax.text(j, i, f"{value:.1f}%", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, label="Stable room / total capacity-window (%)")
    for ext in ["png", "pdf"]:
        fig.savefig(FIGURES / f"risk_bounded_stability_heatmap.{ext}", dpi=220)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_funnel()
    plot_stability()
    print(f"wrote figures to {FIGURES}")


if __name__ == "__main__":
    main()
