# SafeHarvest — Reproduction Code

Code for *"SafeHarvest: Risk-Calibrated Admission-Time Control for Harvesting
Stranded Cloud Cluster Resources."*

This archive contains the **source code** (Python + shell) and per-stage
documentation that produce the paper's tables, figures, and numbers. Raw trace
data and large regenerable intermediates are **not** included (see *Data* below).

## Layout

```
experiments/                     # the SafeHarvest pipeline, run in numbered order
  00_safe_harvestability/        # resource side: opportunity intervals + capacity bounds (q95/q99 split-conformal)
  01_candidate_workload_risk/    # workload side: terminal outcomes, bad-terminal labels, risk signals
  02_candidate_resource_demand/  # per-instance CPU/memory demand table
  03_opportunity_matching/       # opportunity<->candidate matching + exact per-machine CPU+memory placement replay
  04_job_risk_graph/             # dependency graph, oracle/operational frontiers, learned + in-degree-only risk, ablations, per-day bootstrap
  05_risk_budgeted_admission/    # the risk-ledger controller (risk-budgeted admission) + result summaries
  run_cellb_pipeline.sh          # re-runs the workload-side pipeline on a second cell (cross-cell validation)
paper_assets/                    # figure + table builders for the manuscript (build_paper_tables.py, make_architecture_figure.py)
jpdc_artifacts/                  # block-bootstrap CI script (bootstrap_gap_ci.py) + the small per-day CSVs it consumes
```

Each stage directory has its own `README.md` documenting its scripts, inputs,
and outputs. Within a stage, `build_*`/`run_*` scripts compute results into
`results/` and `plot_*` scripts render figures.

## Pipeline order

Run the stages in numeric order; each consumes the previous stage's outputs:

1. `00_safe_harvestability` — mine stable per-machine CPU/memory opportunity
   intervals and calibrate the risk-bounded capacity bounds.
2. `01_candidate_workload_risk` — derive collection/instance terminal outcomes
   and the bad-terminal risk signals.
3. `02_candidate_resource_demand` — build the per-instance resource-demand table.
4. `03_opportunity_matching` — match opportunities to candidates and run the
   exact per-machine CPU+memory placement replay.
5. `04_job_risk_graph` — build the dependency-annotated Job Risk Catalog;
   produce the oracle (P0–P4) and operational (OP0–OP4) frontiers, the learned
   and in-degree-only risk models, the rule baselines, the threshold/τ
   sensitivities, and the per-day bootstrap inputs.
6. `05_risk_budgeted_admission` — run the calibrated risk-ledger controller at
   the reported budgets and summarize the risk-budgeted comparison.

Cross-cell validation: `experiments/run_cellb_pipeline.sh` re-runs the
workload-side pipeline on a second cell.

Paper outputs: `paper_assets/scripts/build_paper_tables.py` regenerates the
LaTeX tables; `jpdc_artifacts/bootstrap_gap_ci.py` computes the controller-vs-OP3
gap confidence intervals from the per-day CSVs in the same folder.

## Data

All experiments run on the public **Google ClusterData2019** trace (Borg), days
23–30 of cell `a` for the main evaluation and a second cell for cross-cell
validation. The trace is available from Google (BigQuery / Cloud Storage) and is
**not redistributed here**.

To keep the archive lean, large regenerable intermediates (the multi-megabyte
`results/*.csv` and `*.parquet` files, e.g. the opportunity-interval and
placement-sample tables) are **excluded** — they are reproduced by running the
stage scripts above. The small derived CSVs needed for the bootstrap CI are kept
under `jpdc_artifacts/`.

## Environment

Python 3 with the standard scientific stack: `numpy`, `pandas`, `scipy`,
`scikit-learn` (logistic risk model), and `matplotlib` (figures). Trace ingestion
uses the Google ClusterData2019 schema; configure the trace path at the top of
the stage-0/1 build scripts before running.
