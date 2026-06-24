# Experiment 01: Candidate Workload Risk

Goal:

> Identify which collections are plausible SafeHarvest candidates and quantify their runtime, dependency, and terminal-outcome risk.

Input:

```text
/media/jiang/0bfc75f7-326d-2d45-b017-7100903796b6/google-cluster-data-2019/cell_a/collection_events-*.parquet.gz
```

Remote full table:

```text
/media/jiang/0bfc75f7-326d-2d45-b017-7100903796b6/google-cluster-data-2019/_analysis/safeharvest/01_candidate_workload_risk/candidate_workload_risk_table.parquet
```

Local outputs:

- `results/candidate_class_summary.csv`
- `results/candidate_runtime_fit_summary.csv`
- `results/candidate_terminal_outcome_matrix.csv`
- `results/candidate_dependency_summary.csv`
- `results/safeharvest_candidate_summary.csv`
- `figures/candidate_class_filter.png`
- `figures/candidate_terminal_outcomes.png`
- `figures/candidate_runtime_fit.png`
- `preliminary_findings.md`

This is collection-level only. Resource request and instance-level lifecycle are deliberately deferred to the next pass.
