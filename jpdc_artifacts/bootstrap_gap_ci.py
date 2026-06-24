#!/usr/bin/env python3
"""Paired per-day block-bootstrap CI for the controller(beta=0.005)-vs-OP3 gap.
Usage: python3 _bootstrap_ci.py <controller_per_day.csv> <per_day_breakdown.csv>
controller csv cols: day, placed_jobs, cpu, bad_inst
breakdown csv cols : day,...,placed_cpu_window,...,placed_bad_inst,frontier (filter frontier==operational_OP3)
"""
import sys, csv
import numpy as np

ctrl_path, brk_path = sys.argv[1], sys.argv[2]

ctrl = {}
for r in csv.DictReader(open(ctrl_path)):
    ctrl[int(float(r['day']))] = (float(r['cpu']), float(r['bad_inst']), int(float(r['placed_jobs'])))

op3 = {}
for r in csv.DictReader(open(brk_path)):
    if r['frontier'] == 'operational_OP3':
        # day buckets in breakdown are 23..30; controller days are absolute win//288 -> remap to 0..7 order
        op3[int(r['day'])] = (float(r['placed_cpu_window']), float(r['placed_bad_inst']))

# align by sorted day order (both have 8 days)
cdays = sorted(ctrl); odays = sorted(op3)
assert len(cdays) == len(odays) == 8, (cdays, odays)
ctrl_cpu = np.array([ctrl[d][0] for d in cdays]); ctrl_bad = np.array([ctrl[d][1] for d in cdays])
op3_cpu  = np.array([op3[d][0] for d in odays]);  op3_bad  = np.array([op3[d][1] for d in odays])

def gaps(idx):
    cc, cb = ctrl_cpu[idx].sum(), ctrl_bad[idx].sum()
    oc, ob = op3_cpu[idx].sum(),  op3_bad[idx].sum()
    return 100*(cc-oc)/oc, 100*(cb-ob)/ob

full = np.arange(8)
pt_cpu, pt_bad = gaps(full)
print(f"point: dCPU% = {pt_cpu:+.2f}  dBAD% = {pt_bad:+.2f}")
print(f"controller totals: cpu={ctrl_cpu.sum()/1000:.3f}k bad={int(ctrl_bad.sum())} jobs={sum(ctrl[d][2] for d in cdays)}")
print(f"OP3 totals:        cpu={op3_cpu.sum()/1000:.3f}k bad={int(op3_bad.sum())}")

rng = np.random.default_rng(0)
B = 20000
bc = np.empty(B); bb = np.empty(B)
for i in range(B):
    idx = rng.integers(0, 8, size=8)   # resample 8 days with replacement
    bc[i], bb[i] = gaps(idx)
lo_c, hi_c = np.percentile(bc, [2.5, 97.5])
lo_b, hi_b = np.percentile(bb, [2.5, 97.5])
print(f"95% CI dCPU%: [{lo_c:+.1f}, {hi_c:+.1f}]")
print(f"95% CI dBAD%: [{lo_b:+.1f}, {hi_b:+.1f}]")
