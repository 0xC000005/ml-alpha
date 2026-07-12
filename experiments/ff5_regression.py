#!/usr/bin/env python
"""Regress the base MSRR SDF portfolio's monthly return on the Fama-French factors to
isolate market beta + residual alpha. CAPM (market only) and FF5. OLS with Newey-West
(HAC) standard errors, computed by hand (no statsmodels dependency).

Inputs: output/exp/confirm_rank/base_monthly_returns.csv (month, port_ret — decimal,
excess) from portfolio_analysis.py, and the FF5 monthly CSV path as argv[1].
Our port_ret is already an EXCESS return (target = next-month excess), so it is directly
comparable to Mkt-RF and the long-short factors — no RF adjustment on the LHS.
"""
import re
import sys

import numpy as np
import pandas as pd

FF_CSV = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ff5/ff5.csv"
PORT = "output/exp/confirm_rank/base_monthly_returns.csv"

rows = []
with open(FF_CSV) as f:
    for line in f:
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 7 and re.fullmatch(r"\d{6}", p[0]):
            try:
                rows.append([int(p[0])] + [float(x) for x in p[1:7]])
            except ValueError:
                pass
fac = pd.DataFrame(rows, columns=["month", "MktRF", "SMB", "HML", "RMW", "CMA", "RF"])
for c in ["MktRF", "SMB", "HML", "RMW", "CMA", "RF"]:
    fac[c] /= 100.0

port = pd.read_csv(PORT)
df = port.merge(fac, on="month", how="left")
miss = int(df["MktRF"].isna().sum())
print(f"portfolio months: {len(port)};  matched to factors: {len(df)-miss};  "
      f"unmatched: {miss}  (window {df.month.min()}-{df.month.max()})")
df = df.dropna(subset=["MktRF"])
y = df.port_ret.values


def reg(cols, name, L=6):
    X = np.column_stack([np.ones(len(df))] + [df[c].values for c in cols])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    XtXi = np.linalg.inv(X.T @ X)
    Xe = X * e[:, None]
    S = Xe.T @ Xe
    for l in range(1, L + 1):
        w = 1 - l / (L + 1)
        G = Xe[l:].T @ Xe[:-l]
        S += w * (G + G.T)
    se = np.sqrt(np.diag(XtXi @ S @ XtXi))
    t = beta / se
    r2 = 1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)
    print(f"\n== {name} ==   (n={len(df)}, R²={r2:.2f}, Newey-West L={L})")
    print(f"  alpha (annualized):  {beta[0]*12*100:+6.2f}% / yr     t = {t[0]:+.2f}")
    for i, c in enumerate(cols, 1):
        star = " *" if abs(t[i]) > 1.96 else ""
        print(f"  beta[{c:5s}]:        {beta[i]:+7.3f}        t = {t[i]:+.2f}{star}")


reg(["MktRF"], "CAPM (market only)")
reg(["MktRF", "SMB", "HML", "RMW", "CMA"], "Fama-French 5-factor")
print(f"\nraw corr(portfolio, Mkt-RF) = {np.corrcoef(y, df.MktRF.values)[0,1]:+.2f}")
print(f"portfolio: mean {y.mean()*12*100:+.2f}%/yr, vol {y.std(ddof=1)*np.sqrt(12)*100:.1f}%/yr, "
      f"Sharpe {y.mean()/y.std(ddof=1)*np.sqrt(12):.2f}")
