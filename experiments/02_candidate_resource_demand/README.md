# Experiment 02: Candidate Resource Demand

Goal:

> Join SafeHarvest seed collections with instance-level events to estimate candidate job size, CPU/memory request, and instance lifecycle risk.

Inputs:

```text
/media/jiang/0bfc75f7-326d-2d45-b017-7100903796b6/google-cluster-data-2019/_analysis/safeharvest/01_candidate_workload_risk/candidate_workload_risk_table.parquet
/media/jiang/0bfc75f7-326d-2d45-b017-7100903796b6/google-cluster-data-2019/cell_a/instance_events-*.parquet.gz
```

Planned remote full table:

```text
/media/jiang/0bfc75f7-326d-2d45-b017-7100903796b6/google-cluster-data-2019/_analysis/safeharvest/02_candidate_resource_demand/candidate_resource_demand_table.parquet
```

Expected local outputs:

- `results/resource_demand_overall.csv`
- `results/resource_demand_by_class.csv`
- `results/resource_demand_buckets.csv`
- `results/instance_count_buckets.csv`
- `results/candidate_resource_sample.csv`
- `figures/candidate_resource_bucket_profile.png`
- `figures/candidate_resource_by_class.png`
- `preliminary_findings.md`

This experiment is the bridge between workload risk filtering and placement/matching simulation.
