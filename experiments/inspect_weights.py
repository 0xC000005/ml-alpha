#!/usr/bin/env python
"""Reconstruct MSRR portfolio weights from saved per-seed models and gauge
CONCENTRATION (no training, no cluster — the EXP-002 reanalysis pattern, run locally).

Question: does the MSRR SDF portfolio spread its weight over the whole ~N-stock
cross-section each month, or concentrate it in a few names? For each test month it
reports: universe size N, the EFFECTIVE number of positions  N_eff = 1/Σ pᵢ²  with
pᵢ=|wᵢ|/Σ|w| (N_eff≈N ⇒ fully diffuse; N_eff≪N ⇒ concentrated), the top-10/50/100
gross-weight share, how many names hold 50%/90% of gross, long/short counts,
net/gross, and the RAW per-seed gross leverage Σ|w| (the magnitude the L1 metric
normalizes away). Sanity-checks the reconstructed L1 SDF Sharpe against the saved CSV.

Usage: python experiments/inspect_weights.py [YEAR] [ARM] [N_SEEDS]
  ARM ∈ base|a1monthz|a2rank|a3rankgauss  (default base = pooled-z = production input)
"""
import logging
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_nn import (Config, load_returns, load_universe, load_signals, load_macro,
                      load_sector_mapping, build_long_panel, build_industry_dummies)
from train_transformer import MonthGroupedData
from experiments.exp_transformer import ExpTransformer
from experiments.feature_scalers import make_scaler
from experiments.msrr_combine import l1norm_per_month, sdf_sharpe

YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 2019
ARM = sys.argv[2] if len(sys.argv) > 2 else "base"
N_SEEDS = int(sys.argv[3]) if len(sys.argv) > 3 else 10
FEAT = {"base": "pooled_z", "a1monthz": "month_z", "a2rank": "rank",
        "a3rankgauss": "rank_gauss"}[ARM]
MODELS = f"output/exp/confirm_rank/{ARM}_{YEAR}/models"
TRAIN_START = 1975

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("inspect")
cfg = Config()

returns = load_returns(cfg.data_dir, TRAIN_START, YEAR, log)
universe = load_universe(cfg.data_dir, TRAIN_START, YEAR, log)
signals = load_signals(cfg.data_dir, cfg.signal_names, TRAIN_START, YEAR, log)
macro, rfree = load_macro(cfg.macro_file, TRAIN_START, YEAR, log)
p2s, sic2 = load_sector_mapping(cfg.sector_file, log)
sf, mf, y, mid, pid = build_long_panel(universe, returns, signals, macro, rfree,
                                       cfg.signal_names, cfg.macro_names, log)

train_end_m = (YEAR - 2) * 100 + 12
test_start_m, test_end_m = YEAR * 100 + 1, YEAR * 100 + 11
train_mask = mid <= train_end_m
test_mask = (mid >= test_start_m) & (mid <= test_end_m)

scaler = make_scaler(FEAT, clip_std=5.0)
scaler.fit(sf[train_mask], mf[train_mask])
s, m = scaler.transform(sf[test_mask].copy(), mf[test_mask].copy(), mid[test_mask])
ind = build_industry_dummies(pid[test_mask], p2s, sic2)
test_d = MonthGroupedData(s, m, ind, y[test_mask], mid[test_mask], pid[test_mask])

# per-seed inference (CPU fp32, no autocast)
seed_preds = []
for seed in range(N_SEEDS):
    model = ExpTransformer(n_signals=95, n_industries=74, n_macro=8, d_model=32,
                           n_heads=4, d_ff=64, dropout=0.10, n_layers=1)
    sd = torch.load(os.path.join(MODELS, f"MSRR_year{YEAR}_seed{seed}.pt"),
                    map_location="cpu")
    model.load_state_dict(sd)
    model.eval()
    preds = []
    with torch.no_grad():
        for mo in test_d.months:
            stk = torch.from_numpy(test_d.stock_dict[mo]).unsqueeze(0)
            mac = torch.from_numpy(test_d.macro_dict[mo])
            idd = torch.from_numpy(test_d.ind_dict[mo]).unsqueeze(0)
            preds.append(model(stk, mac, idd).numpy())
    seed_preds.append(np.concatenate(preds))

test_months = np.concatenate([np.full(len(test_d.target_dict[mo]), mo)
                              for mo in test_d.months])
test_targets = np.concatenate([test_d.target_dict[mo] for mo in test_d.months])

# L1 equal-vote ensemble (the deployed portfolio)
normed = np.stack([l1norm_per_month(sp, test_months) for sp in seed_preds])
ens = normed.mean(0)
sh, _ = sdf_sharpe(ens, test_targets, test_months, l1_final=True)
print(f"\n=== {ARM} {YEAR} ({FEAT}, {N_SEEDS} seeds) — reconstructed L1 SDF Sharpe = "
      f"{sh:.3f}  (sanity-check vs saved CSV) ===")


def conc(w):
    a = np.abs(w)
    g = a.sum()
    if g == 0:
        return None
    p = a / g
    order = np.sort(p)[::-1]
    csum = np.cumsum(order)
    return dict(
        N=len(w), Neff=1.0 / np.sum(p ** 2),
        top10=order[:10].sum(), top50=order[:50].sum(), top100=order[:100].sum(),
        n50=int(np.searchsorted(csum, 0.50) + 1), n90=int(np.searchsorted(csum, 0.90) + 1),
        nlong=int((w > 0).sum()), nshort=int((w < 0).sum()), netgross=w.sum() / g)


wfinal = l1norm_per_month(ens, test_months)
rows = [conc(wfinal[test_months == mo]) for mo in np.unique(test_months)]
seed0 = seed_preds[0]
raw_g = [np.abs(seed0[test_months == mo]).sum() for mo in np.unique(test_months)]


def col(k):
    return np.array([r[k] for r in rows], float)


print(f"  universe N / month:        mean {col('N').mean():.0f}   "
      f"(min {col('N').min():.0f}, max {col('N').max():.0f})")
print(f"  N_eff (effective names):   mean {col('Neff').mean():.0f}   "
      f"(min {col('Neff').min():.0f}, max {col('Neff').max():.0f})   "
      f"= {100 * col('Neff').mean() / col('N').mean():.1f}% of the universe")
print(f"  top-10  gross-weight share: mean {100 * col('top10').mean():.1f}%")
print(f"  top-50  gross-weight share: mean {100 * col('top50').mean():.1f}%")
print(f"  top-100 gross-weight share: mean {100 * col('top100').mean():.1f}%")
print(f"  names holding 50% of gross: mean {col('n50').mean():.0f}")
print(f"  names holding 90% of gross: mean {col('n90').mean():.0f}")
print(f"  long / short counts:        mean {col('nlong').mean():.0f} long / "
      f"{col('nshort').mean():.0f} short")
print(f"  net / gross exposure:       mean {col('netgross').mean():+.3f}  "
      f"(0 = dollar-neutral)")
print(f"  RAW per-seed gross Σ|w|:    mean {np.mean(raw_g):.1f}  "
      f"(leverage scale the model emits, before L1 normalization)")
