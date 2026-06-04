#!/usr/bin/env python
"""Experimental MSRR driver — the MSRR counterpart to exp_main.py, needed because the
feature-scaler hook (B-11 rank-standardization) and the honest L1 ensemble combiner
(Tier 0) cannot be reached by monkeypatching the production main()'s nested build_split.
It REUSES the frozen MSRR training loop (train_model_msrr), data pipeline, and model
(swapping in the configurable ExpTransformer so n_layers/ffn_kind are available for the
Tier-2 depth screen). Production scripts stay frozen.

One config = one test year (the cheap-screen array maps one JSONL line → one task).

Config JSON (one object), e.g. the rank A/B arms:
  {"model":"msrr","year":2016,"n_seeds":5,"feat_scaler":"rank","outdir":"output/exp/rank/a2rank_2016"}
  feat_scaler ∈ {pooled_z (A0/control), month_z (A1), rank (A2), rank_gauss (A3)}
  combiner    ∈ {l1norm (default headline), raw, trimmed}   # raw + l1 are ALWAYS logged
Optional model knobs: d_model,d_ff,n_heads,n_layers,ffn_kind,dropout,lr,max_epochs,
                      patience,weight_decay,ridge_lambda,train_start.

Usage:
  python exp_main_msrr.py '<json>'          # run it (needs data + GPU)
  python exp_main_msrr.py '<json>' --dry     # build model + scaler only, no data/training
"""
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_transformer_msrr as MSRR
from train_nn import (Config, load_returns, load_universe, load_signals, load_macro,
                      load_sector_mapping, build_long_panel, build_industry_dummies,
                      compute_oos_metrics)
from train_transformer import MonthGroupedData, evaluate
from experiments.exp_transformer import ExpTransformer
from experiments.feature_scalers import make_scaler
from experiments.msrr_combine import both_sharpes, sdf_sharpe, combine_seeds

cfgj = json.loads(sys.argv[1])
DRY = "--dry" in sys.argv[2:]
FEAT = cfgj.get("feat_scaler", "pooled_z")      # A0..A3
COMBINER = cfgj.get("combiner", "l1norm")        # headline selector (raw+l1 always logged)
N_LAYERS = int(cfgj.get("n_layers", 1))
FFN = cfgj.get("ffn_kind", "gelu")
YEAR = int(cfgj["year"])
N_SEEDS = int(cfgj.get("n_seeds", 5))
OUTDIR = cfgj["outdir"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("exp_main_msrr")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- model: ExpTransformer with the experiment knobs (bit-identical at n_layers=1) ---
def _factory(**kw):
    return ExpTransformer(**kw, n_layers=N_LAYERS, ffn_kind=FFN)
MSRR.CrossSectionalTransformer = _factory

cfg = MSRR.MSRRConfig()
cfg.output_dir = OUTDIR
cfg.n_seeds = N_SEEDS
for fld in ("d_model", "d_ff", "n_heads", "dropout", "lr", "max_epochs",
            "patience", "weight_decay", "ridge_lambda"):
    if fld in cfgj:
        setattr(cfg, fld, cfgj[fld])
TRAIN_START = int(cfgj.get("train_start", cfg.train_start))

print(f"[exp_msrr] year={YEAR} seeds={N_SEEDS} feat={FEAT} combiner={COMBINER} "
      f"n_layers={N_LAYERS} ffn={FFN} d_model={cfg.d_model} d_ff={cfg.d_ff} "
      f"-> {OUTDIR}", flush=True)

if DRY:
    m = _factory(n_signals=cfg.n_signals, n_industries=cfg.n_industries,
                 n_macro=cfg.n_macro, d_model=cfg.d_model, n_heads=cfg.n_heads,
                 d_ff=cfg.d_ff, dropout=cfg.dropout)
    nparam = sum(p.numel() for p in m.parameters())
    sc = make_scaler(FEAT, clip_std=cfg.clip_std)
    print(f"[exp_msrr] DRY: built {type(m).__name__} {nparam:,} params + "
          f"{type(sc).__name__}({FEAT}) (no data, no training).")
    sys.exit(0)

for sub in ("logs", "models", "predictions", "metrics", "features"):
    os.makedirs(os.path.join(OUTDIR, sub), exist_ok=True)

# --- data ---
returns = load_returns(cfg.data_dir, TRAIN_START, YEAR, log)
universe = load_universe(cfg.data_dir, TRAIN_START, YEAR, log)
signals = load_signals(cfg.data_dir, cfg.signal_names, TRAIN_START, YEAR, log)
macro, rfree = load_macro(cfg.macro_file, TRAIN_START, YEAR, log)
p2s, sic2 = load_sector_mapping(cfg.sector_file, log)
sf, mf, y, mid, pid = build_long_panel(universe, returns, signals, macro, rfree,
                                       cfg.signal_names, cfg.macro_names, log)
del returns, universe, signals, macro, rfree

# --- expanding-window split (val_years=1); test = Jan..Nov (data-edge convention) ---
val_end_year = YEAR - 1
val_start_year = YEAR - cfg.val_years
train_end_year = val_start_year - 1
train_end_m = train_end_year * 100 + 12
val_start_m = val_start_year * 100 + 1
val_end_m = val_end_year * 100 + 12
test_start_m = YEAR * 100 + 1
test_end_m = YEAR * 100 + 11

train_mask = mid <= train_end_m
val_mask = (mid >= val_start_m) & (mid <= val_end_m)
test_mask = (mid >= test_start_m) & (mid <= test_end_m)
log.info(f"train<= {train_end_year}-12 ({train_mask.sum():,}), "
         f"val {val_start_year} ({val_mask.sum():,}), "
         f"test {YEAR} Jan-Nov ({test_mask.sum():,})")

scaler = make_scaler(FEAT, clip_std=cfg.clip_std)
scaler.fit(sf[train_mask], mf[train_mask])


def build_split(mask):
    s, m = scaler.transform(sf[mask].copy(), mf[mask].copy(), mid[mask])
    ind = build_industry_dummies(pid[mask], p2s, sic2)
    return MonthGroupedData(s, m, ind, y[mask], mid[mask], pid[mask])


train_d, val_d, test_d = build_split(train_mask), build_split(val_mask), build_split(test_mask)

# --- train n_seeds (save per-seed models so the L1 reanalysis is zero-compute later) ---
seed_preds, seed_epochs = [], []
for seed in range(cfg.n_seeds):
    res = MSRR.train_model_msrr(train_d, val_d, YEAR, seed, cfg, dev, log)
    preds = evaluate(res["model"], test_d, dev)[0]
    seed_preds.append(preds)
    seed_epochs.append(res["best_epoch"])
    torch.save(res["model"].state_dict(),
               os.path.join(OUTDIR, "models", f"MSRR_year{YEAR}_seed{seed}.pt"))
    del res
    if dev.type == "cuda":
        torch.cuda.empty_cache()

# --- combine + honest metric (raw AND L1 always reported) ---
test_targets = np.concatenate([test_d.target_dict[m] for m in test_d.months])
test_months = np.concatenate([np.full(len(test_d.target_dict[m]), m, dtype=np.int32)
                              for m in test_d.months])
sh = both_sharpes(seed_preds, test_targets, test_months)

if COMBINER == "raw":
    headline, ens = sh["sdf_sharpe_raw"], sh["raw_ensemble"]
elif COMBINER == "trimmed":
    ens = combine_seeds(seed_preds, test_months, "trimmed")
    headline, _ = sdf_sharpe(ens, test_targets, test_months, l1_final=True)
else:  # l1norm (default)
    headline, ens = sh["sdf_sharpe_l1"], sh["l1_ensemble"]

dec = compute_oos_metrics(ens, test_targets, test_months, log)
log.info(f"  SDF Sharpe: raw={sh['sdf_sharpe_raw']:.3f}  "
         f"l1={sh['sdf_sharpe_l1']:.3f}  headline({COMBINER})={headline:.3f}")

row = {
    "test_year": YEAR,
    "feat_scaler": FEAT,
    "combiner": COMBINER,
    "sdf_sharpe": headline,                 # collect_screen.py reads this
    "sdf_sharpe_raw": sh["sdf_sharpe_raw"],
    "sdf_sharpe_l1": sh["sdf_sharpe_l1"],
    "sdf_mean_ret_l1": sh["sdf_mean_ret_l1"],
    "sdf_std_ret_l1": sh["sdf_std_ret_l1"],
    "mean_ic": dec.get("mean_ic", np.nan),
    "sharpe_ls_annual": dec.get("sharpe_ls_annual", np.nan),
    "n_months": len(test_d.months),
    "n_obs": len(test_targets),
    "avg_epochs": float(np.mean(seed_epochs)),
}
out = os.path.join(OUTDIR, "metrics", "msrr_transformer_summary.csv")
pd.DataFrame([row]).to_csv(out, index=False)
log.info(f"wrote {out}")
print(f"[exp_main_msrr] DONE {OUTDIR} sdf_sharpe={headline:.3f} "
      f"(raw={sh['sdf_sharpe_raw']:.3f} l1={sh['sdf_sharpe_l1']:.3f})", flush=True)
