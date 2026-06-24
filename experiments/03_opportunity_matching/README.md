# Experiment 03: Opportunity Matching

Goal:

> Replay SafeHarvest seed candidates against stable per-machine resource opportunity intervals and compare progressively more conservative placement policies.

Inputs:

```text
/media/jiang/0bfc75f7-326d-2d45-b017-7100903796b6/google-cluster-data-2019/_analysis/safeharvest/00_safe_harvestability/resource_opportunity_intervals_*.csv
/media/jiang/0bfc75f7-326d-2d45-b017-7100903796b6/google-cluster-data-2019/_analysis/safeharvest/01_candidate_workload_risk/candidate_workload_risk_table.parquet
/media/jiang/0bfc75f7-326d-2d45-b017-7100903796b6/google-cluster-data-2019/_analysis/safeharvest/02_candidate_resource_demand/candidate_resource_demand_table.parquet
```

Local outputs:

- `results/matching_policy_summary.csv`
- `results/matching_policy_by_class.csv`
- `results/matching_unplaced_reasons.csv`
- `results/matching_candidate_jobs.csv`
- `results/exact_machine_policy_summary.csv`
- `results/exact_machine_policy_by_class.csv`
- `results/exact_machine_unplaced_reasons.csv`
- `results/memory_enriched_opportunity_summary.csv`
- `figures/matching_policy_summary.png`
- `figures/safeharvest_by_opportunity_class.png`
- `figures/exact_machine_policy_summary.png`
- `figures/exact_safeharvest_by_opportunity_class.png`
- `figures/memory_aware_safeharvest_summary.png`
- `figures/memory_room_profile.png`
- `preliminary_findings.md`

This is the first end-to-end SafeHarvest simulation. It includes an aggregate replay upper bound, exact per-machine placement with interval-level residual CPU accounting, and a memory-aware variant using memory headroom derived from `instance_usage`.
