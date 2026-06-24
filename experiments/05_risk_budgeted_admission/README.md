# Experiment 05: Risk-Budgeted Admission

This experiment adds an online-feasible controller to SafeHarvest. Instead of
selecting jobs by a fixed risk threshold, the controller maintains a risk ledger
and admits a job only when calibrated risk and exact CPU+memory placement both
fit.

## Inputs

Required:

- `expanded_candidate_table.parquet` from Experiment 04.
- Memory-enriched opportunity intervals, for example
  `experiments/03_opportunity_matching/results/memory_enriched_opportunity_summary_10pct_1h.csv`.

The full expanded candidate table is large and may live on the workstation
rather than in the local submission package.

## Main Script

```sh
python3 scripts/run_risk_budgeted_admission.py \
  --expanded-candidates /path/to/expanded_candidate_table.parquet \
  --intervals ../03_opportunity_matching/results/memory_enriched_opportunity_summary_10pct_1h.csv \
  --output results/indegree_only \
  --no-graph-future-features
```

Run without `--no-graph-future-features` to obtain the dependency-augmented
upper operational variant.

For a more online-feasible duration guard, replace trace-observed test runtime
with a pre-test historical duration quantile:

```sh
python3 scripts/run_risk_budgeted_admission.py \
  --expanded-candidates /path/to/expanded_candidate_table.parquet \
  --intervals ../03_opportunity_matching/results/memory_enriched_opportunity_summary_10pct_1h.csv \
  --output results/dependency_augmented_p95_duration \
  --orderings risk_first \
  --risk-budgets 0.005,0.010 \
  --duration-mode calibrated_p95 \
  --write-samples
```

## Method

1. Train a logistic bad-terminal risk model on pre-test collections.
2. Hold out the last part of the pre-test period for calibration.
3. Convert predicted risks into conservative bin-wise risk charges using a
   Wilson upper confidence bound.
4. Choose the placement duration. The default `observed` mode preserves the
   previous offline exact-replay protocol; `calibrated_p90` and
   `calibrated_p95` reserve a pre-test historical duration quantile for each
   candidate class/resource-demand group.
5. Replay test-window jobs in schedule order.
6. Admit and place a job only when:
   - calibrated per-job risk does not exceed a cap;
   - the daily average risk ledger remains within budget;
   - exact per-machine CPU+memory placement succeeds.

## Outputs

- `risk_budget_calibration_bins.csv`
- `risk_budgeted_admission_summary.csv`
- `risk_budgeted_rejection_reasons.csv`
- `duration_charge_table.csv` when a calibrated duration mode is used
- optionally `risk_budgeted_placements_sample.csv`

## Post-Processing

After at least one result directory is available, generate the comparison table
and Pareto figure:

```sh
python3 scripts/summarize_risk_budgeted_results.py
```

This writes:

- `analysis/risk_budgeted_comparison_summary.csv`
- `analysis/risk_budgeted_findings.md`
- `tables/table_risk_budgeted_comparison.tex`
- `figures/risk_budgeted_pareto.png`
- `figures/risk_budgeted_pareto.pdf`

## Research Purpose

This experiment turns SafeHarvest into an admission-control algorithm
with calibrated risk accounting, rather than only a retrospective trace catalog
and replay analysis.

## Current Status

As of June 6, 2026, both the strictly admission-time in-degree-only run and the
dependency-augmented run are complete.

For the in-degree-only risk-first controller, average risk budgets of 0.5%, 1%,
and 2% place 76,187, 91,275, and 104,500 jobs, respectively, with 2.568k,
3.361k, and 4.595k CPU-window demand. The corresponding observed bad-terminal
instances are 440, 1,007, and 1,979.

For the dependency-augmented risk-first controller, the 0.5% budget places
90,114 jobs and 3.791k CPU-window demand with 589 observed bad-terminal
instances. This is the current strongest low-risk result: it places 6.1% more
CPU-window than OP3 while placing 27.4% fewer observed bad-terminal instances.
At budgets of 1% and above, the current run saturates at the per-job risk cap,
placing 108,086 jobs and 4.732k CPU-window demand with 929 observed
bad-terminal instances.

For the dependency-augmented P95-duration guard, the 0.5% and 1% budgets place
90,093 and 108,027 jobs with 3.793k and 4.754k CPU-window demand. Observed
bad-terminal instances are 592 and 936, respectively. This shows that replacing
trace-observed test runtime with a pre-test historical P95 duration guard does
not materially reduce utility in this window, although 17.3% and 16.1% of placed
jobs still run longer than the guarded duration and must be reported as
duration-overrun residual risk.

For the fully admission-visible in-degree-only P95-duration guard, budgets of
0.5%, 1%, and 2% place 76,212, 91,270, and 104,456 jobs with 2.596k, 3.385k,
and 4.619k CPU-window demand. Observed bad-terminal instances are 448, 1,013,
and 1,987; duration miss rates are 19.3%, 17.5%, and 15.9%. This is the most
defensible online-feasible variant because it removes out-degree-derived graph
features and avoids trace-observed test runtime for placement reservation.

These results should be written as a controllable utility/risk frontier rather
than a zero-risk guarantee.
