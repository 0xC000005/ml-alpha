#!/usr/bin/env python
"""Tabulate a cheap-screen: config (tag) × year for Sharpe and IC, with deltas vs
the 'base' config. Usage: collect_screen.py <screen_dir> [summary_csv_name]
e.g. collect_screen.py output/exp/cap transformer_summary.csv
     collect_screen.py output/exp/miss exp_summary.csv
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

screen_dir = sys.argv[1]
summary_name = sys.argv[2] if len(sys.argv) > 2 else "transformer_summary.csv"

rows = []
for d in sorted(glob.glob(os.path.join(screen_dir, "*"))):
    f = os.path.join(d, "metrics", summary_name)
    if not os.path.isfile(f):
        continue
    m = re.match(r"(.+)_(\d{4})$", os.path.basename(d))
    if not m:
        continue
    df = pd.read_csv(f)
    r = df.iloc[0].to_dict()
    r["tag"], r["year"] = m.group(1), int(m.group(2))
    rows.append(r)

if not rows:
    print(f"no results yet under {screen_dir}/*/metrics/{summary_name}", file=sys.stderr)
    sys.exit(0)

D = pd.DataFrame(rows)
metric = "sdf_sharpe" if "sdf_sharpe" in D.columns else "sharpe_ls_annual"
piv = D.pivot_table(index="tag", columns="year", values=metric)
print(f"=== {metric} by config × year ({len(D)} tasks present) ===")
print(piv.round(2).to_string())
piv = piv.assign(mean=piv.mean(axis=1))
print("\nrow means:"); print(piv["mean"].round(3).to_string())

if "base" in piv.index:
    delta = piv.drop(columns="mean").sub(piv.drop(columns="mean").loc["base"])
    print(f"\n=== Δ {metric} vs base (positive = better) ===")
    print(delta.round(2).to_string())
    wins = (delta > 0).sum(axis=1)
    print("\nyears-beating-base (of "
          f"{delta.shape[1]}):"); print(wins.to_string())
    print("\nVERDICT: promote a config to a full run only if it beats base in "
          ">= 2/3 years AND its row-mean is clearly higher (judge dispersion too).")

if "mean_ic" in D.columns:
    pic = D.pivot_table(index="tag", columns="year", values="mean_ic")
    print(f"\n=== mean_ic by config × year ===")
    print(pic.round(3).to_string())
