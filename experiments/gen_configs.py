#!/usr/bin/env python
"""Generate cheap-screen config lists (JSONL). One JSON object per line = one task.
Capacity/GLU use experiments/run_experiment.py; missingness/temporal/monthly use
experiments/exp_main.py (set RUNNER accordingly in the sbatch)."""
import json
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
os.makedirs(OUT, exist_ok=True)
YEARS = [2014, 2016, 2018]   # hard / weak / strong
SEEDS = 3


def write(name, rows):
    with open(os.path.join(OUT, name), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{name:28s} {len(rows):3d} tasks")


# EXP-003 capacity + VoC (run_experiment.py)
caps = [("base", 32, 64, 1), ("L2", 32, 64, 2), ("L3", 32, 64, 3),
        ("d64", 64, 128, 1), ("d64L2", 64, 128, 2)]
write("capacity.jsonl", [
    dict(model="mse", year=y, n_seeds=SEEDS, d_model=dm, d_ff=df, n_layers=nl,
         ffn_kind="gelu", outdir=f"output/exp/cap/{tag}_{y}")
    for (tag, dm, df, nl) in caps for y in YEARS])

# EXP-004a GLU sophistication (run_experiment.py)
write("sophistication_glu.jsonl", [
    dict(model="mse", year=y, n_seeds=SEEDS, ffn_kind="glu",
         outdir=f"output/exp/glu/glu_{y}") for y in YEARS])

# EXP-004b missingness (exp_main.py) -- includes a matched baseline through the SAME driver
write("missingness.jsonl", [
    dict(model="mse", year=y, n_seeds=SEEDS, missingness=mb,
         outdir=f"output/exp/miss/{'miss' if mb else 'base'}_{y}")
    for mb in (False, True) for y in YEARS])

# EXP-005 temporal macro-GRU (exp_main.py) -- baseline + L=12 + L=24
rows = [dict(model="mse", year=y, n_seeds=SEEDS, outdir=f"output/exp/temporal/base_{y}") for y in YEARS]
for L in (12, 24):
    rows += [dict(model="mse", year=y, n_seeds=SEEDS, macro_temporal="gru",
                  macro_lookback=L, outdir=f"output/exp/temporal/gru{L}_{y}") for y in YEARS]
write("temporal.jsonl", rows)

# EXP-006 monthly refit (exp_main.py) -- one year only
write("monthly.jsonl", [dict(model="mse", year=2018, n_seeds=SEEDS, monthly=True,
                             outdir="output/exp/monthly/mse_2018")])
