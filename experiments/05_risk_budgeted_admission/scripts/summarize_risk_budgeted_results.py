#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
EXP05_DIR = SCRIPT_DIR.parent
EXPERIMENTS_DIR = EXP05_DIR.parent
EXP04_DIR = EXPERIMENTS_DIR / "04_job_risk_graph"

RESULTS_DIR = EXP05_DIR / "results"
ANALYSIS_DIR = EXP05_DIR / "analysis"
FIGURES_DIR = EXP05_DIR / "figures"
TABLES_DIR = EXP05_DIR / "tables"


def ensure_dirs() -> None:
    for directory in (ANALYSIS_DIR, FIGURES_DIR, TABLES_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def cpu_k(value: float) -> float:
    return float(value) / 1000.0


def bad_per_k_cpu(bad_instances: float, placed_cpu_window: float) -> float:
    if placed_cpu_window <= 0:
        return 0.0
    return float(bad_instances) / cpu_k(placed_cpu_window)


def add_row(
    rows: list[dict],
    *,
    method: str,
    family: str,
    feature_set: str,
    policy: str,
    risk_control: str,
    placed_jobs: int,
    placed_cpu_window: float,
    placed_memory_window: float | None,
    bad_terminal_instances: int,
    machines_used: int | None,
    expected_bad_share: float | None = None,
    realized_bad_share: float | None = None,
    duration_miss_share: float | None = None,
    duration_overrun_windows: float | None = None,
    mean_reserved_runtime_windows: float | None = None,
    mean_observed_runtime_windows: float | None = None,
    source: str,
    note: str = "",
) -> None:
    rows.append(
        {
            "method": method,
            "family": family,
            "feature_set": feature_set,
            "policy": policy,
            "risk_control": risk_control,
            "placed_jobs": int(placed_jobs),
            "placed_cpu_window": float(placed_cpu_window),
            "placed_cpu_window_k": cpu_k(float(placed_cpu_window)),
            "placed_memory_window": (
                float(placed_memory_window) if placed_memory_window is not None else pd.NA
            ),
            "placed_memory_window_k": (
                cpu_k(float(placed_memory_window))
                if placed_memory_window is not None
                else pd.NA
            ),
            "bad_terminal_instances": int(bad_terminal_instances),
            "bad_inst_per_k_cpu": bad_per_k_cpu(
                float(bad_terminal_instances), float(placed_cpu_window)
            ),
            "machines_used": int(machines_used) if machines_used is not None else pd.NA,
            "expected_bad_collection_share": expected_bad_share,
            "realized_bad_collection_share": realized_bad_share,
            "duration_miss_share": duration_miss_share,
            "duration_overrun_windows": duration_overrun_windows,
            "mean_reserved_runtime_windows": mean_reserved_runtime_windows,
            "mean_observed_runtime_windows": mean_observed_runtime_windows,
            "source": source,
            "note": note,
        }
    )


def load_oracle_and_operational(rows: list[dict]) -> None:
    frontier = pd.read_csv(EXP04_DIR / "results" / "frontier_matching_summary.csv")
    oracle = frontier[
        (frontier["frontier_policy"] == "P3_plus_clean_medium_independent")
        & (frontier["opportunity_class"] == "10pct_1h")
        & (frontier["placement_policy"] == "frontier_min_cpu_mem_max_exact")
    ].iloc[0]
    add_row(
        rows,
        method="Oracle P3",
        family="oracle",
        feature_set="post-hoc terminal outcome",
        policy="P3 clean medium independent",
        risk_control="observed-clean by construction",
        placed_jobs=oracle["placed_jobs"],
        placed_cpu_window=oracle["placed_cpu_window"],
        placed_memory_window=oracle["placed_memory_window"],
        bad_terminal_instances=oracle["placed_bad_terminal_instances"],
        machines_used=oracle["used_machines"],
        source="04_job_risk_graph/results/frontier_matching_summary.csv",
        note="Upper-bound frontier using future terminal outcomes.",
    )

    operational = pd.read_csv(
        EXP04_DIR / "results" / "operational" / "operational_matching_summary.csv"
    )
    op3 = operational[
        operational["frontier_policy"] == "P3_plus_visible_medium_independent"
    ].iloc[0]
    add_row(
        rows,
        method="OP3",
        family="hand-crafted",
        feature_set="admission-time visible",
        policy="visible medium independent",
        risk_control="rule frontier",
        placed_jobs=op3["placed_jobs"],
        placed_cpu_window=op3["placed_cpu_window"],
        placed_memory_window=op3["placed_memory_window"],
        bad_terminal_instances=op3["placed_bad_terminal_instances"],
        machines_used=op3["used_machines"],
        source="04_job_risk_graph/results/operational/operational_matching_summary.csv",
    )


def load_learned(rows: list[dict], result_dir: Path, feature_set: str) -> None:
    path = result_dir / "learned_risk_matching_summary.csv"
    if not path.exists():
        return
    learned = pd.read_csv(path)
    for threshold in (0.03, 0.05):
        row = learned[learned["risk_threshold"].round(6) == round(threshold, 6)].iloc[0]
        add_row(
            rows,
            method=f"Learned <= {threshold:.2f}",
            family="fixed-threshold learned",
            feature_set=feature_set,
            policy=row["frontier_policy"],
            risk_control=f"predicted risk <= {threshold:.2f}",
            placed_jobs=row["placed_jobs"],
            placed_cpu_window=row["placed_cpu_window"],
            placed_memory_window=row["placed_memory_window"],
            bad_terminal_instances=row["placed_bad_terminal_instances"],
            machines_used=row["machines_used"],
            expected_bad_share=row["mean_predicted_bad_rate"],
            source=str(path.relative_to(EXP05_DIR.parent.parent)),
        )


def load_budgeted(rows: list[dict], result_dir: Path, feature_set: str) -> None:
    path = result_dir / "risk_budgeted_admission_summary.csv"
    if not path.exists():
        return
    budgeted = pd.read_csv(path)
    for _, row in budgeted.iterrows():
        beta = float(row["average_risk_budget"])
        add_row(
            rows,
            method=f"Risk-budgeted beta={beta:.3f}",
            family="risk-budgeted controller",
            feature_set=feature_set,
            policy=str(row["ordering_policy"]),
            risk_control=f"daily average risk <= {beta:.3f}",
            placed_jobs=row["placed_jobs"],
            placed_cpu_window=row["placed_cpu_window"],
            placed_memory_window=row["placed_memory_window"],
            bad_terminal_instances=row["placed_bad_terminal_instances"],
            machines_used=row["machines_used"],
            expected_bad_share=row["expected_bad_collection_share"],
            realized_bad_share=row["realized_bad_collection_share"],
            duration_miss_share=row["duration_miss_share"]
            if "duration_miss_share" in row.index
            else None,
            duration_overrun_windows=row["duration_overrun_windows"]
            if "duration_overrun_windows" in row.index
            else None,
            mean_reserved_runtime_windows=row["mean_reserved_runtime_windows"]
            if "mean_reserved_runtime_windows" in row.index
            else None,
            mean_observed_runtime_windows=row["mean_observed_runtime_windows"]
            if "mean_observed_runtime_windows" in row.index
            else None,
            source=str(path.relative_to(EXP05_DIR.parent.parent)),
        )


def build_comparison() -> pd.DataFrame:
    rows: list[dict] = []
    load_oracle_and_operational(rows)
    load_learned(rows, EXP04_DIR / "results" / "learned", "dependency-augmented")
    load_learned(
        rows, EXP04_DIR / "results" / "learned_indegree_only", "in-degree only"
    )
    load_budgeted(rows, RESULTS_DIR / "indegree_only", "in-degree only")
    load_budgeted(
        rows,
        RESULTS_DIR / "indegree_only_p95_duration",
        "in-degree only P95-duration",
    )
    load_budgeted(rows, RESULTS_DIR / "dependency_augmented", "dependency-augmented")
    load_budgeted(
        rows,
        RESULTS_DIR / "dependency_augmented_p95_duration",
        "dependency-augmented P95-duration",
    )
    return pd.DataFrame(rows)


def write_latex_table(df: pd.DataFrame) -> None:
    selected = df[
        (df["method"].isin(["Oracle P3", "OP3"]))
        | (
            (df["family"] == "fixed-threshold learned")
            & (df["method"].isin(["Learned <= 0.03", "Learned <= 0.05"]))
        )
        | (
            (df["family"] == "risk-budgeted controller")
            & (
                (
                    (df["feature_set"] == "dependency-augmented")
                    & (df["policy"] == "risk_first")
                    & (
                        df["risk_control"].isin(
                            [
                                "daily average risk <= 0.005",
                                "daily average risk <= 0.010",
                            ]
                        )
                    )
                )
                | (
                    (df["feature_set"] == "dependency-augmented P95-duration")
                    & (df["policy"] == "risk_first")
                    & (
                        df["risk_control"].isin(
                            [
                                "daily average risk <= 0.005",
                                "daily average risk <= 0.010",
                            ]
                        )
                    )
                )
                | (
                    (df["feature_set"] == "in-degree only P95-duration")
                    & (df["policy"] == "risk_first")
                    & (
                        df["risk_control"].isin(
                            [
                                "daily average risk <= 0.005",
                                "daily average risk <= 0.020",
                            ]
                        )
                    )
                )
                | (
                    (df["feature_set"] == "in-degree only")
                    & (df["policy"] == "risk_first")
                    & (
                        df["risk_control"].isin(
                            [
                                "daily average risk <= 0.005",
                                "daily average risk <= 0.020",
                            ]
                        )
                    )
                )
            )
        )
    ].copy()

    selected["CPU-win k"] = selected["placed_cpu_window_k"].map(lambda x: f"{x:.3f}")
    selected["Bad/kCPU"] = selected["bad_inst_per_k_cpu"].map(lambda x: f"{x:.1f}")
    selected["Bad inst."] = selected["bad_terminal_instances"].map(lambda x: f"{x:,}")
    selected["Jobs"] = selected["placed_jobs"].map(lambda x: f"{x:,}")
    def method_label(row: pd.Series) -> str:
        feature_map = {
            "dependency-augmented": "dep.-aug.",
            "dependency-augmented P95-duration": "dep.-aug. P95 dur.",
            "in-degree only P95-duration": "in-degree P95 dur.",
        }
        feature = feature_map.get(row["feature_set"], row["feature_set"])
        if row["method"] == "Oracle P3":
            return "Oracle P3"
        if row["method"] == "OP3":
            return "OP3"
        if row["family"] == "fixed-threshold learned":
            threshold = row["risk_control"].split("<=")[-1].strip()
            return f"Learned $\\hat{{\\rho}}\\leq{threshold}$ ({feature})"
        if row["family"] == "risk-budgeted controller":
            beta = row["risk_control"].split("<=")[-1].strip()
            return f"Risk-budgeted $\\beta={beta}$ ({feature})"
        return str(row["method"]).replace("_", "-")

    def policy_label(row: pd.Series) -> str:
        if row["family"] == "fixed-threshold learned":
            return "fixed threshold"
        if row["policy"] == "risk_first":
            return "risk-first"
        return str(row["policy"]).replace("_", "-")

    selected["Method"] = selected.apply(method_label, axis=1)
    selected["Policy"] = selected.apply(policy_label, axis=1)

    table = selected[["Method", "Policy", "Jobs", "CPU-win k", "Bad inst.", "Bad/kCPU"]]
    lines = [
        "\\begin{tabular}{@{}llrrrr@{}}",
        "\\toprule",
        "Method & Policy & Jobs & CPU-win k & Bad inst. & Bad/kCPU \\\\",
        "\\midrule",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"{row['Method']} & {row['Policy']} & {row['Jobs']} & "
            f"{row['CPU-win k']} & {row['Bad inst.']} & {row['Bad/kCPU']} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (TABLES_DIR / "table_risk_budgeted_comparison.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def plot_pareto(df: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 10,
            "legend.fontsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(7.0, 3.8))

    colors = {
        "in-degree only": "#2166ac",
        "in-degree only P95-duration": "#67a9cf",
        "dependency-augmented": "#b2182b",
        "dependency-augmented P95-duration": "#d6604d",
    }
    markers = {
        "risk_first": "o",
        "utility_per_risk": "s",
        "balanced": "^",
    }

    budgeted = df[
        (df["family"] == "risk-budgeted controller") & (df["policy"] == "risk_first")
    ].copy()
    for (feature_set, policy), group in budgeted.groupby(["feature_set", "policy"]):
        group = group.sort_values("bad_terminal_instances")
        ax.plot(
            group["bad_terminal_instances"],
            group["placed_cpu_window_k"],
            marker=markers.get(policy, "o"),
            linewidth=2.0,
            markersize=5.5,
            color=colors.get(feature_set, "#444444"),
            alpha=0.95,
            label=f"Risk-budgeted {feature_set}",
        )

    baseline_styles = [
        ("OP3", "X", "#4d4d4d"),
        ("Learned <= 0.03", "D", "#7b3294"),
        ("Learned <= 0.05", "P", "#008837"),
        ("Oracle P3", "*", "#d95f02"),
    ]
    for method, marker, color in baseline_styles:
        points = df[df["method"] == method]
        for _, row in points.iterrows():
            label = method
            if method.startswith("Learned"):
                label = f"{method}, {row['feature_set']}"
            ax.scatter(
                row["bad_terminal_instances"],
                row["placed_cpu_window_k"],
                marker=marker,
                s=80 if marker != "*" else 150,
                color=color,
                edgecolor="white",
                linewidth=0.8,
                label=label,
                zorder=5,
            )

    ax.set_xlabel("Observed bad-terminal instances placed")
    ax.set_ylabel("Placed CPU-window demand (k)")
    ax.grid(True, axis="both", linewidth=0.5, alpha=0.25)

    handles, labels = ax.get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels):
        unique.setdefault(label, handle)
    ax.legend(
        unique.values(),
        unique.keys(),
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
    )
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES_DIR / f"risk_budgeted_pareto.{suffix}", dpi=300)
    plt.close(fig)


def write_findings(df: pd.DataFrame) -> None:
    lines = [
        "# Risk-Budgeted Admission Findings",
        "",
        "Generated from Experiment 05 outputs.",
        "",
    ]

    for feature_set in (
        "in-degree only",
        "in-degree only P95-duration",
        "dependency-augmented",
        "dependency-augmented P95-duration",
    ):
        subset = df[
            (df["family"] == "risk-budgeted controller")
            & (df["feature_set"] == feature_set)
            & (df["policy"] == "risk_first")
        ].sort_values("bad_terminal_instances")
        if subset.empty:
            continue
        lines.append(f"## {feature_set} risk-first")
        lines.append("")
        for _, row in subset.iterrows():
            duration_note = ""
            if pd.notna(row.get("duration_miss_share", pd.NA)):
                duration_note = (
                    f", duration miss {100.0 * row['duration_miss_share']:.1f}%"
                )
            lines.append(
                "- "
                f"{row['risk_control']}: {int(row['placed_jobs']):,} jobs, "
                f"{row['placed_cpu_window_k']:.3f}k CPU-window, "
                f"{int(row['bad_terminal_instances']):,} bad-terminal instances, "
                f"{row['bad_inst_per_k_cpu']:.1f} bad/kCPU"
                f"{duration_note}."
            )
        lines.append("")

    op3 = df[df["method"] == "OP3"].iloc[0]
    dep_low = df[
        (df["family"] == "risk-budgeted controller")
        & (df["feature_set"] == "dependency-augmented")
        & (df["policy"] == "risk_first")
        & (df["risk_control"] == "daily average risk <= 0.005")
    ]
    lines.append("## Current interpretation")
    lines.append("")
    if not dep_low.empty:
        row = dep_low.iloc[0]
        cpu_gain = (row["placed_cpu_window_k"] / op3["placed_cpu_window_k"] - 1.0) * 100.0
        bad_drop = (1.0 - row["bad_terminal_instances"] / op3["bad_terminal_instances"]) * 100.0
        lines.append(
            f"- The dependency-augmented risk-first controller at 0.5% budget places "
            f"{row['placed_cpu_window_k']:.3f}k CPU-window, {cpu_gain:.1f}% more than "
            f"OP3, while placing {int(row['bad_terminal_instances']):,} bad-terminal "
            f"instances, {bad_drop:.1f}% fewer than OP3."
        )
    lines.append(
        f"- OP3 places {int(op3['placed_jobs']):,} jobs and "
        f"{op3['placed_cpu_window_k']:.3f}k CPU-window with "
        f"{int(op3['bad_terminal_instances']):,} observed bad-terminal instances."
    )
    lines.append(
        "- The new controller should be presented as a risk-budgeted admission "
        "algorithm, not as a guarantee of zero bad outcomes."
    )
    lines.append(
        "- Oracle P3 remains an upper-bound analysis because it uses post-hoc "
        "terminal outcomes."
    )
    lines.append("")
    (ANALYSIS_DIR / "risk_budgeted_findings.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    ensure_dirs()
    comparison = build_comparison()
    comparison.to_csv(ANALYSIS_DIR / "risk_budgeted_comparison_summary.csv", index=False)
    write_latex_table(comparison)
    plot_pareto(comparison)
    write_findings(comparison)
    print(f"Wrote {ANALYSIS_DIR / 'risk_budgeted_comparison_summary.csv'}")
    print(f"Wrote {TABLES_DIR / 'table_risk_budgeted_comparison.tex'}")
    print(f"Wrote {FIGURES_DIR / 'risk_budgeted_pareto.png'}")


if __name__ == "__main__":
    main()
