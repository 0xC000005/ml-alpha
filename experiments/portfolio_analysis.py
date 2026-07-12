#!/usr/bin/env python
"""Reconstruct the base MSRR SDF portfolio (2012-2019, 10-seed L1 ensemble) from saved
models and run the free "is this portfolio real & practical" battery — local, no
cluster, no training (the EXP-002 reanalysis pattern):
  - sanity : per-year + pooled SDF Sharpe vs EXP-009 (base ≈ 1.81)
  - turnover: within-year monthly + refit-boundary + year-over-year sign stability
  - sparsity: SDF Sharpe vs top-K positions (post-hoc truncation, no retrain)
Saves reconstructed (month,permno,weight,ret) to npz and the monthly SDF return series to
csv (for the FF5 regression step).
"""
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_nn import (Config, load_returns, load_universe, load_signals, load_macro,
                      load_sector_mapping, build_long_panel, build_industry_dummies)
from train_transformer import MonthGroupedData
from experiments.exp_transformer import ExpTransformer
from experiments.feature_scalers import make_scaler
from experiments.msrr_combine import l1norm_per_month, sdf_sharpe

YEARS = list(range(2012, 2020))
ARM, FEAT, N_SEEDS, TRAIN_START = "base", "pooled_z", 10, 1975
logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("pa")
cfg = Config()

# ---- load full panel ONCE, slice per year ----
returns = load_returns(cfg.data_dir, TRAIN_START, max(YEARS), log)
universe = load_universe(cfg.data_dir, TRAIN_START, max(YEARS), log)
signals = load_signals(cfg.data_dir, cfg.signal_names, TRAIN_START, max(YEARS), log)
macro, rfree = load_macro(cfg.macro_file, TRAIN_START, max(YEARS), log)
p2s, sic2 = load_sector_mapping(cfg.sector_file, log)
sf, mf, y, mid, pid = build_long_panel(universe, returns, signals, macro, rfree,
                                       cfg.signal_names, cfg.macro_names, log)
del returns, universe, signals, macro, rfree

allmon, allper, allw, allr = [], [], [], []
per_year = {}
for YEAR in YEARS:
    train_end_m = (YEAR - 2) * 100 + 12
    ts, te = YEAR * 100 + 1, YEAR * 100 + 11
    tr = mid <= train_end_m
    tm = (mid >= ts) & (mid <= te)
    scaler = make_scaler(FEAT, clip_std=5.0)
    scaler.fit(sf[tr], mf[tr])
    s, m = scaler.transform(sf[tm].copy(), mf[tm].copy(), mid[tm])
    ind = build_industry_dummies(pid[tm], p2s, sic2)
    td = MonthGroupedData(s, m, ind, y[tm], mid[tm], pid[tm])
    seed_preds = []
    for seed in range(N_SEEDS):
        model = ExpTransformer(n_signals=95, n_industries=74, n_macro=8, d_model=32,
                               n_heads=4, d_ff=64, dropout=0.10, n_layers=1)
        model.load_state_dict(torch.load(
            f"output/exp/confirm_rank/{ARM}_{YEAR}/models/MSRR_year{YEAR}_seed{seed}.pt",
            map_location="cpu"))
        model.eval()
        preds = []
        with torch.no_grad():
            for mo in td.months:
                stk = torch.from_numpy(td.stock_dict[mo]).unsqueeze(0)
                mac = torch.from_numpy(td.macro_dict[mo])
                idd = torch.from_numpy(td.ind_dict[mo]).unsqueeze(0)
                preds.append(model(stk, mac, idd).numpy())
        seed_preds.append(np.concatenate(preds))
    months = np.concatenate([np.full(len(td.target_dict[mo]), mo) for mo in td.months])
    rets = np.concatenate([td.target_dict[mo] for mo in td.months])
    pers = np.concatenate([td.permno_dict[mo] for mo in td.months]).astype(int)
    normed = np.stack([l1norm_per_month(sp, months) for sp in seed_preds])
    ens = l1norm_per_month(normed.mean(0), months)   # final L1 → gross 1 per month
    sh, _ = sdf_sharpe(ens, rets, months, l1_final=True)
    per_year[YEAR] = sh
    allmon.append(months); allper.append(pers); allw.append(ens); allr.append(rets)
    print(f"  {YEAR}: reconstructed Sharpe {sh:.3f}")

allmon = np.concatenate(allmon); allper = np.concatenate(allper)
allw = np.concatenate(allw); allr = np.concatenate(allr)
os.makedirs("output/exp/confirm_rank", exist_ok=True)
np.savez("output/exp/confirm_rank/reconstructed_base.npz",
         months=allmon, permnos=allper, weights=allw, rets=allr)

umon = np.unique(allmon)
mret = np.array([np.dot(allw[allmon == mo], allr[allmon == mo]) for mo in umon])
pooled = mret.mean() / mret.std(ddof=1) * np.sqrt(12)
pd.DataFrame({"month": umon, "port_ret": mret}).to_csv(
    "output/exp/confirm_rank/base_monthly_returns.csv", index=False)
print(f"\nSANITY: per-year mean Sharpe {np.mean(list(per_year.values())):.3f} "
      f"(EXP-009 base = 1.81);  pooled {len(umon)}-month Sharpe {pooled:.3f}")

# ---------- TURNOVER ----------
mw = {mo: dict(zip(allper[allmon == mo], allw[allmon == mo])) for mo in umon}


def turnover(a, b):
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


within, boundary = [], []
for i in range(1, len(umon)):
    prev, cur = umon[i - 1], umon[i]
    to = turnover(mw[prev], mw[cur])
    (within if cur // 100 == prev // 100 else boundary).append(to)
print(f"\nTURNOVER (one-way, gross=1):")
print(f"  within-year monthly: mean {np.mean(within):.2f}  "
      f"(= {100*np.mean(within):.0f}% of the book traded / month)")
print(f"  refit-boundary (Nov→Jan, new model): mean {np.mean(boundary):.2f}")

# year-over-year sign stability (mean weight per stock per year)
yw = {Y: pd.DataFrame({"p": allper[allmon // 100 == Y], "w": allw[allmon // 100 == Y]})
         .groupby("p").w.mean() for Y in YEARS}
print(f"\nYEAR-OVER-YEAR sign stability (mean weight per stock, consecutive years):")
for i in range(1, len(YEARS)):
    a, b = yw[YEARS[i - 1]], yw[YEARS[i]]
    common = a.index.intersection(b.index)
    wa, wb = a[common].values, b[common].values
    corr = np.corrcoef(wa, wb)[0, 1]
    flip = np.mean(np.sign(wa) != np.sign(wb))
    print(f"  {YEARS[i-1]}→{YEARS[i]}: {len(common)} common stocks, "
          f"weight corr {corr:+.2f}, sign-flip {100*flip:.0f}%")

# ---------- SPARSITY (post-hoc top-K) ----------
print(f"\nSPARSITY (keep top-K |w| per month, renormalize, recompute Sharpe):")
print(f"  {'K':>6} {'Sharpe':>8} {'% of full':>10}")
full = None
for K in [None, 2000, 1000, 500, 200, 100, 50, 20, 10]:
    w2 = allw.copy()
    if K is not None:
        for mo in umon:
            idx = np.where(allmon == mo)[0]
            a = np.abs(allw[idx])
            if len(idx) > K:
                cut = np.partition(a, -K)[-K]
                w2[idx[a < cut]] = 0.0
        w2 = l1norm_per_month(w2, allmon)
    sh, _ = sdf_sharpe(w2, allr, allmon, l1_final=True)
    if K is None:
        full = sh
    print(f"  {'full' if K is None else K:>6} {sh:>8.3f} "
          f"{('--' if K is None else f'{100*sh/full:.0f}%'):>10}")
