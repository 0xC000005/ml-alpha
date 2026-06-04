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

# ---------------------------------------------------------------------------
# Roadmap (wf_2f4b6eee) -- MSRR transformer enhancements. RUNNER=exp_main_msrr.py.
# 5 seeds (dispersion is binding); judge on the L1-normalized SDF Sharpe.
# ---------------------------------------------------------------------------
SEEDS5 = 5

# B-11 rank-standardization A/B (Tier 1, the one survivor). 4 arms break the confound:
#   base=A0 pooled-z control / a1monthz=A1 / a2rank=A2 (candidate) / a3rankgauss=A3 hybrid.
RANK_YEARS = [2014, 2016, 2018]
rank_arms = [("base", "pooled_z"), ("a1monthz", "month_z"),
             ("a2rank", "rank"), ("a3rankgauss", "rank_gauss")]
write("rank_ab.jsonl", [
    dict(model="msrr", year=y, n_seeds=SEEDS5, feat_scaler=fs, combiner="l1norm",
         outdir=f"output/exp/rank/{tag}_{y}")
    for (tag, fs) in rank_arms for y in RANK_YEARS])

# B-01 depth on MSRR (Tier 2, GATED on Tier-1 + honest metric). base=K1/L2=K2/L3=K3,
# pooled-z held fixed to isolate depth; 2018/2019 = raw star years (test norm survival).
DEPTH_YEARS = [2014, 2018, 2019]
depth_arms = [("base", 1), ("L2", 2), ("L3", 3)]
write("msrr_depth.jsonl", [
    dict(model="msrr", year=y, n_seeds=SEEDS5, n_layers=nl, combiner="l1norm",
         outdir=f"output/exp/msrr_depth/{tag}_{y}")
    for (tag, nl) in depth_arms for y in DEPTH_YEARS])

# EXP-009: full confirmation of the rank screen winner (a2rank PASSED EXP-007).
# 8 years × 10 seeds (production standard) over the ~±0.4 96-month Sharpe floor.
# base = comparison denominator (must run in the same conditions); a1monthz = robustness
# comparator (uniquely survived 2016 in the screen). Judge on L1 SDF Sharpe.
CONFIRM_YEARS = list(range(2012, 2020))   # 2012..2019
confirm_arms = [("base", "pooled_z"), ("a1monthz", "month_z"), ("a2rank", "rank")]
write("confirm_rank.jsonl", [
    dict(model="msrr", year=y, n_seeds=10, feat_scaler=fs, combiner="l1norm",
         outdir=f"output/exp/confirm_rank/{tag}_{y}")
    for (tag, fs) in confirm_arms for y in CONFIRM_YEARS])
