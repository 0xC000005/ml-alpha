#!/usr/bin/env python
"""Collect the per-(model,year) summary CSVs from the reproduction array and
compare the aggregates to the README/report numbers.

Reads:  output/repro/{mse,msrr}_{year}/metrics/{transformer,msrr_transformer}_summary.csv
Prints: per-year tables + README comparison with deltas. Writes a combined CSV.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

REPO = "/scratch/maxzhang/ml-alpha"
ROOT = os.path.join(REPO, "output", "repro")

# README / report targets (10-seed ensemble, RTX 4080).
README = {
    "mse_2012_2019":  {"oos_r2_pct": -0.08, "mean_ic": 0.021, "mean_ls_ret_pct": 1.53,
                       "sharpe_ls_annual": 2.16, "positive_years": "8/8"},
    "mse_2016_2019":  {"sharpe_ls_annual": 1.81},
    "msrr_2016_2019": {"sdf_sharpe": 2.05, "best": "2016 (+3.03)", "worst": "2018 (+0.82)"},
}


def load(model, summary_name):
    rows = []
    for d in sorted(glob.glob(os.path.join(ROOT, f"{model}_*"))):
        p = os.path.join(d, "metrics", summary_name)
        if os.path.exists(p):
            rows.append(pd.read_csv(p))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values("test_year").reset_index(drop=True)


def agg(df, years, col):
    sub = df[df.test_year.isin(years)]
    return float(sub[col].mean()) if len(sub) else float("nan")


def line(label, got, want):
    d = got - want
    flag = "OK" if abs(d) <= max(0.15 * abs(want), 0.20) else "CHECK"
    return f"  {label:24s} got={got:+.3f}  README={want:+.3f}  Δ={d:+.3f}  [{flag}]"


mse = load("mse", "transformer_summary.csv")
msrr = load("msrr", "msrr_transformer_summary.csv")

print("=" * 64)
print(f"MSE Transformer — {len(mse)}/8 years present")
print("=" * 64)
if len(mse):
    print(mse[["test_year", "oos_r2_pct", "mean_ic", "mean_ls_ret_pct",
               "sharpe_ls_annual"]].to_string(index=False))
    yrs = list(range(2012, 2020))
    pos = int((mse[mse.test_year.isin(yrs)].sharpe_ls_annual > 0).sum())
    tot = int(mse.test_year.isin(yrs).sum())
    print("\n  -- 2012-2019 vs README --")
    print(line("avg OOS R2 %", agg(mse, yrs, "oos_r2_pct"), README["mse_2012_2019"]["oos_r2_pct"]))
    print(line("avg IC", agg(mse, yrs, "mean_ic"), README["mse_2012_2019"]["mean_ic"]))
    print(line("avg L/S %/mo", agg(mse, yrs, "mean_ls_ret_pct"), README["mse_2012_2019"]["mean_ls_ret_pct"]))
    print(line("avg Sharpe (L/S)", agg(mse, yrs, "sharpe_ls_annual"), README["mse_2012_2019"]["sharpe_ls_annual"]))
    print(f"  {'positive years':24s} got={pos}/{tot}  README=8/8")
    print("\n  -- 2016-2019 subset vs README --")
    print(line("avg Sharpe (L/S)", agg(mse, range(2016, 2020), "sharpe_ls_annual"),
               README["mse_2016_2019"]["sharpe_ls_annual"]))

print("\n" + "=" * 64)
print(f"MSRR Transformer — {len(msrr)}/8 years present")
print("=" * 64)
if len(msrr):
    print(msrr[["test_year", "sdf_sharpe", "sharpe_ls_annual", "sdf_mean_ret",
                "sdf_std_ret"]].to_string(index=False))
    print("\n  -- 2016-2019 vs README (SDF) --")
    print(line("avg SDF Sharpe", agg(msrr, range(2016, 2020), "sdf_sharpe"),
               README["msrr_2016_2019"]["sdf_sharpe"]))
    sub = msrr[msrr.test_year.isin(range(2016, 2020))]
    if len(sub):
        b = sub.loc[sub.sdf_sharpe.idxmax()]
        w = sub.loc[sub.sdf_sharpe.idxmin()]
        print(f"  best={int(b.test_year)} (+{b.sdf_sharpe:.2f})  worst={int(w.test_year)} (+{w.sdf_sharpe:.2f})"
              f"   README: best {README['msrr_2016_2019']['best']}, worst {README['msrr_2016_2019']['worst']}")
    print(f"\n  (2012-2015 MSRR is an extension beyond the README — new numbers)")

if len(mse) or len(msrr):
    out = os.path.join(ROOT, "repro_combined.csv")
    pd.concat([mse.assign(model="mse"), msrr.assign(model="msrr")],
              ignore_index=True).to_csv(out, index=False)
    print(f"\nwrote {out}")
else:
    print("No summary CSVs found yet.", file=sys.stderr)
