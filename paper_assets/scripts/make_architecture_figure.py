#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "paper_assets" / "figures"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "paper_assets" / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "paper_assets" / ".cache"))
os.environ.setdefault("HOME", str(ROOT / "paper_assets" / ".home"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def add_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    body: str,
    face: str,
    edge: str,
) -> None:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.018",
        linewidth=1.2,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(box)
    x, y = xy
    ax.text(
        x + width / 2,
        y + height - 0.09,
        title,
        ha="center",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        color="#111827",
    )
    ax.text(
        x + width / 2,
        y + height / 2 - 0.05,
        body,
        ha="center",
        va="center",
        fontsize=8.2,
        color="#374151",
        linespacing=1.35,
    )


def add_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.25,
            color="#4b5563",
            shrinkA=5,
            shrinkB=5,
        )
    )


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12.6, 5.7), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    colors = {
        "input": ("#eff6ff", "#2563eb"),
        "resource": ("#ecfdf5", "#16a34a"),
        "risk": ("#fff7ed", "#f59e0b"),
        "admit": ("#f5f3ff", "#7c3aed"),
        "eval": ("#f8fafc", "#64748b"),
    }

    boxes = {
        "trace": (0.04, 0.58, 0.19, 0.26),
        "capacity": (0.30, 0.58, 0.20, 0.26),
        "resource_graph": (0.57, 0.58, 0.19, 0.26),
        "workload": (0.04, 0.16, 0.19, 0.26),
        "risk_graph": (0.30, 0.16, 0.20, 0.26),
        "admission": (0.57, 0.16, 0.19, 0.26),
        "replay": (0.81, 0.37, 0.15, 0.26),
    }

    add_box(
        ax,
        boxes["trace"][:2],
        boxes["trace"][2],
        boxes["trace"][3],
        "Cluster trace",
        "machine events\ninstance usage\ncollection lifecycle",
        *colors["input"],
    )
    add_box(
        ax,
        boxes["capacity"][:2],
        boxes["capacity"][2],
        boxes["capacity"][3],
        "Risk-bounded capacity",
        "reserve calibrated demand\nmeasure overload risk\nfilter unstable room",
        *colors["resource"],
    )
    add_box(
        ax,
        boxes["resource_graph"][:2],
        boxes["resource_graph"][2],
        boxes["resource_graph"][3],
        "Resource Opportunity Graph",
        "machine intervals\nCPU headroom\nmemory headroom\nduration fit",
        *colors["resource"],
    )
    add_box(
        ax,
        boxes["workload"][:2],
        boxes["workload"][2],
        boxes["workload"][3],
        "Candidate workloads",
        "runtime\nresource request\nterminal outcome\npriority",
        *colors["input"],
    )
    add_box(
        ax,
        boxes["risk_graph"][:2],
        boxes["risk_graph"][2],
        boxes["risk_graph"][3],
        "Job Risk Graph",
        "dependency role\nclean leaf tiers\nretry signal split\nrisk frontier",
        *colors["risk"],
    )
    add_box(
        ax,
        boxes["admission"][:2],
        boxes["admission"][2],
        boxes["admission"][3],
        "SafeHarvest admission",
        "P0 strict seed\nP1-P3 clean expansion\nP4 diagnostic only",
        *colors["admit"],
    )
    add_box(
        ax,
        boxes["replay"][:2],
        boxes["replay"][2],
        boxes["replay"][3],
        "Exact replay",
        "per-machine placement\nCPU + memory guards\nrisk accounting",
        *colors["eval"],
    )

    add_arrow(ax, (0.23, 0.71), (0.30, 0.71))
    add_arrow(ax, (0.50, 0.71), (0.57, 0.71))
    add_arrow(ax, (0.23, 0.29), (0.30, 0.29))
    add_arrow(ax, (0.50, 0.29), (0.57, 0.29))
    add_arrow(ax, (0.76, 0.71), (0.81, 0.51))
    add_arrow(ax, (0.76, 0.29), (0.81, 0.49))

    ax.text(
        0.50,
        0.94,
        "SafeHarvest converts apparent idle capacity into auditable, failure-aware harvestable work",
        ha="center",
        va="center",
        fontsize=13.5,
        fontweight="bold",
        color="#111827",
    )
    ax.text(
        0.50,
        0.895,
        "The key separation is between stable resource opportunity and workload risk.",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#4b5563",
    )

    for ext in ["png", "pdf"]:
        fig.savefig(FIGURES / f"safeharvest_architecture.{ext}", dpi=240)
    plt.close(fig)
    print(f"wrote architecture figure to {FIGURES}")


if __name__ == "__main__":
    main()
