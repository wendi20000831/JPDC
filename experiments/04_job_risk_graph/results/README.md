# Test-window per-day breakdown

`per_day_breakdown.csv` covers test-window days 23–30 of cell_a (eight
days, the manuscript's main reporting period). A day-31 fragment from
the original run (≤40 jobs per frontier, corresponding to the trailing
edge of test_win5 = 8784) has been removed because it is incomplete and
not part of any aggregate reported in the manuscript.

Columns: day, eligible_jobs, placed_jobs, eligible_cpu_window,
placed_cpu_window, eligible_memory_window, placed_memory_window,
eligible_bad_inst, placed_bad_inst, frontier.

Each frontier row appears 8 times (one per day):
- oracle_P3: retrospective oracle frontier (S0+S1+S2+S3 tiers, FINISH labels)
- operational_OP3: dependency-augmented operational proxy (O0+O1+O2+O3 tiers)
- learned_L2_full: visible-feature logistic risk model, ρ̂ ≤ 0.03, full features (incl. out-degree)
- learned_L2_indeg: same logistic model, ρ̂ ≤ 0.03, in-degree-only ablation

See manuscript Table 12 (per-day variability) for the summary.

