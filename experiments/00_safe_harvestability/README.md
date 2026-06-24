# Experiment 00: Safe Harvestability Funnel

Goal:

> Quantify how much machine-level idle capacity remains after request accounting, calibrated risk reservation, and temporal stability filters.

Input on workstation:

```text
/home/jiang/crust/results/08_machine_window_budgets.parquet
```

Outputs:

- `results/harvestability_funnel.csv`
- `results/stability_matrix.csv`
- `results/policy_window_quantiles.csv`
- `preliminary_findings.md`

This experiment is intentionally CPU-only and machine-window level. It is a fast first pass to decide whether the SafeHarvest story is empirically strong before doing heavier lifecycle/dependency analysis.
