#!/usr/bin/env bash
# cell_b full SafeHarvest pipeline (workload-side external validity).
# Mirrors the cell_a pipeline with cell_b inputs and an isolated output tree.
set -euo pipefail

CTRL=/media/jiang/0bfc75f7-326d-2d45-b017-7100903796b6/google-cluster-data-2019/_control/venv/bin/python
DATA=/media/jiang/0bfc75f7-326d-2d45-b017-7100903796b6/google-cluster-data-2019
CB=$DATA/cell_b
OUT=$DATA/_analysis/safeharvest_cellb
SCR=$HOME/safeharvest_runs/scripts
TMP=/mnt/clusterdata0/duckdb_tmp
export SAFEHARVEST_LEARNED_SCRIPT=$SCR/run_learned_risk_baseline.py

mkdir -p "$OUT"/{00_safe_harvestability,01_candidate_workload_risk,02_candidate_resource_demand,03_opportunity_matching,04_job_risk_graph,risk_budgeted}
mkdir -p "$OUT"/04_job_risk_graph/operational "$TMP"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
log()   { echo "[$(stamp)] $*"; }

log "STEP 0: map cell_b aggregates -> machine_window_budgets (sum_q95 -> res_q3)"
$CTRL - <<PY
import duckdb
con = duckdb.connect()
con.execute("PRAGMA threads=16")
con.execute(f"PRAGMA temp_directory='$TMP'")
con.execute("""
COPY (
  SELECT machine_id, win5, day_bucket, cpu_cap, n_tasks, true_agg,
         sum_q95 AS res_q3
  FROM read_parquet('/home/jiang/crust/results/cell_b/aggregates.parquet')
  WHERE cpu_cap > 0 AND true_agg IS NOT NULL AND sum_q95 IS NOT NULL
) TO '$OUT/cellb_machine_window_budgets.parquet' (FORMAT parquet)
""")
print("budgets rows:", con.execute("SELECT COUNT(*) FROM read_parquet('$OUT/cellb_machine_window_budgets.parquet')").fetchone()[0])
PY
log "STEP 0 DONE"

log "STEP 1/00: resource opportunity intervals"
$CTRL $SCR/build_resource_opportunity_graph.py \
  --input "$OUT/cellb_machine_window_budgets.parquet" \
  --output "$OUT/00_safe_harvestability" --temp-dir "$TMP"
log "STEP 1/00 DONE"

log "STEP 2/01: candidate workload risk table"
$CTRL $SCR/build_candidate_workload_risk.py \
  --input-glob "$CB/collection_events-*.parquet.gz" \
  --output "$OUT/01_candidate_workload_risk" --temp-dir "$TMP"
log "STEP 2/01 DONE"

log "STEP 3/03: memory-enriched opportunity intervals (reads instance_usage, slow)"
$CTRL $SCR/build_memory_enriched_opportunities.py \
  --instance-usage-glob "$CB/instance_usage-*.parquet.gz" \
  --machine-events "$CB/machine_events-000000000000.parquet.gz" \
  --opportunity-dir "$OUT/00_safe_harvestability" \
  --output "$OUT/03_opportunity_matching" --temp-dir "$TMP" \
  --threads 16 --memory-limit 64GB
log "STEP 3/03 DONE"

log "STEP 4/04: job risk graph + expanded candidate table"
$CTRL $SCR/build_job_risk_graph.py \
  --collection-events "$CB/collection_events-*.parquet.gz" \
  --instance-events "$CB/instance_events-*.parquet.gz" \
  --risk-table "$OUT/01_candidate_workload_risk/candidate_workload_risk_table.parquet" \
  --output "$OUT/04_job_risk_graph" --temp-dir "$TMP" \
  --threads 16 --memory-limit 64GB
log "STEP 4/04 DONE"

EXP=$OUT/04_job_risk_graph/expanded_candidate_table.parquet
RISK=$OUT/01_candidate_workload_risk/candidate_workload_risk_table.parquet
MEMINT=$OUT/03_opportunity_matching/memory_enriched_opportunity_intervals_10pct_1h.csv

log "STEP 5/oracle: frontier matching (oracle P0-P4)"
$CTRL $SCR/run_frontier_matching.py \
  --expanded-candidates "$EXP" \
  --cpu-opportunity-dir "$OUT/00_safe_harvestability" \
  --memory-opportunity-dir "$OUT/03_opportunity_matching" \
  --output "$OUT/04_job_risk_graph/frontier_matching"
log "STEP 5/oracle DONE"

log "STEP 6/operational: OP0-OP4"
$CTRL $SCR/run_operational_frontier.py \
  --expanded-candidates "$EXP" --risk-table "$RISK" \
  --cpu-opportunity-dir "$OUT/00_safe_harvestability" \
  --memory-opportunity-dir "$OUT/03_opportunity_matching" \
  --output "$OUT/04_job_risk_graph/operational"
log "STEP 6/operational DONE"

log "STEP 7/risk-budgeted: dependency-augmented + in-degree-only"
$CTRL $SCR/run_risk_budgeted_admission.py \
  --expanded-candidates "$EXP" --intervals "$MEMINT" \
  --output "$OUT/risk_budgeted/dependency_augmented" \
  --risk-budgets 0.005,0.010,0.020 --orderings risk_first
$CTRL $SCR/run_risk_budgeted_admission.py \
  --expanded-candidates "$EXP" --intervals "$MEMINT" \
  --output "$OUT/risk_budgeted/indegree_only" \
  --risk-budgets 0.005,0.010,0.020 --orderings risk_first \
  --no-graph-future-features
log "STEP 7/risk-budgeted DONE"

log "ALL CELL_B STEPS DONE"
