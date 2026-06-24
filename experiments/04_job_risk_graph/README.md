# Experiment 04: Job Risk Graph

This experiment expands SafeHarvest beyond a strict seed candidate pool by building
a graph-aware workload risk model.

## Inputs

- `01_candidate_workload_risk/candidate_workload_risk_table.parquet`
- `cell_a/collection_events-*.parquet.gz`
- `cell_a/instance_events-*.parquet.gz`

## Method

The experiment constructs a collection-level dependency graph:

- `parent` edges from `parent_collection_id`
- `start_after` edges from `start_after_collection_ids`

Each collection is assigned a dependency role:

- `independent_leaf`: no incoming dependency and no downstream dependent
- `dependent_leaf`: has dependencies but no downstream dependent
- `upstream_prerequisite`: no dependency but has downstream dependents
- `internal_dependency_chain`: both depends on others and has dependents

The graph is joined with runtime, terminal outcome, priority/scheduling class,
and instance-level resource demand. The resulting frontier separates candidates
into risk tiers:

- `S0_strict_seed`: the original conservative SafeHarvest seed pool
- `S1_clean_small_independent`: clean, small, independent jobs
- `S2_clean_small_leaf_dependency`: clean, small jobs with light dependencies and no dependents
- `S3_clean_medium_independent`: larger but still clean independent jobs
- `R1_retry_tolerant_signal`: short, flexible non-finish jobs; diagnostic only, not a safe default
- `X_excluded`: outside the controlled expansion frontier

## Outputs

Large remote tables:

- `job_risk_graph_nodes.parquet`
- `job_risk_graph_edges.parquet`
- `expanded_candidate_table.parquet`

Local summary outputs:

- `graph_structure_summary.csv`
- `dependency_role_summary.csv`
- `risk_tier_summary.csv`
- `risk_tier_by_class.csv`
- `expansion_frontier_summary.csv`
- `test_window_expanded_candidate_sample.csv`
- `frontier_matching_jobs.csv`
- `frontier_matching_summary.csv`
- `frontier_matching_unplaced_reasons.csv`
- `frontier_matching_sample.csv`

Figures:

- `risk_tier_funnel.png`
- `expansion_frontier.png`
- `dependency_graph_profile.png`
- `frontier_matching_summary.png`

## Research Purpose

The goal is to test whether SafeHarvest can expand beyond a very conservative
seed pool while preserving a clear safety boundary. The paper-relevant question is:

> How much additional harvestable demand appears when the workload filter becomes
> graph-aware rather than simply strict?

The matching pass then asks whether the expanded demand pool remains feasible
under concrete per-machine CPU and memory constraints.
