#!/usr/bin/env python
"""Experimental MSE driver for the data/loop hooks that main() cannot express via
pure monkeypatch: missingness indicators (B-06), macro-temporal GRU (B-02), and
monthly refit (B-04). It REUSES the production data pipeline, train_model, evaluate
and metrics — only the orchestration (which features go in, which periods to refit)
is new. Production scripts stay frozen.

Config JSON (one object), e.g.:
  {"model":"mse","year":2018,"n_seeds":3,"missingness":true,"outdir":"output/exp/miss/2018"}
  {"model":"mse","year":2018,"n_seeds":3,"macro_temporal":"gru","macro_lookback":12,"outdir":...}
  {"model":"mse","year":2018,"n_seeds":3,"monthly":true,"outdir":...}
Optional model knobs: d_model,d_ff,n_heads,n_layers,ffn_kind,dropout,weight_decay,max_epochs,patience.
"""
import json
import logging
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_transformer as TT
from train_nn import (Config, load_returns, load_universe, load_signals, load_macro,
                      load_sector_mapping, build_long_panel, compute_oos_metrics)
from train_transformer import (TransformerConfig, TransformerFeatureScaler,
                               MonthGroupedData, build_industry_dummies, evaluate)
from experiments.exp_transformer import ExpTransformer
from experiments.manifest import write_manifest

cfgj = json.loads(sys.argv[1])
MISS = bool(cfgj.get("missingness", False))
MACRO_T = cfgj.get("macro_temporal", "none")
LOOKBACK = int(cfgj.get("macro_lookback", 12))
MONTHLY = bool(cfgj.get("monthly", False))
N_LAYERS = int(cfgj.get("n_layers", 1))
FFN = cfgj.get("ffn_kind", "gelu")
YEAR = int(cfgj["year"])
N_SEEDS = int(cfgj.get("n_seeds", 3))
OUTDIR = cfgj["outdir"]
for sub in ("logs", "models", "predictions", "metrics", "features"):
    os.makedirs(os.path.join(OUTDIR, sub), exist_ok=True)
write_manifest(OUTDIR, cfgj)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("exp_main")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base = Config()
TRAIN_START = int(cfgj.get("train_start", base.train_start))  # earlier = less history (faster / rolling)
returns = load_returns(base.data_dir, TRAIN_START, YEAR, log)
universe = load_universe(base.data_dir, TRAIN_START, YEAR, log)
signals = load_signals(base.data_dir, base.signal_names, TRAIN_START, YEAR, log)
macro, rfree = load_macro(base.macro_file, TRAIN_START, YEAR, log)
p2s, sic2 = load_sector_mapping(base.sector_file, log)
sf, mf, y, mid, pid = build_long_panel(universe, returns, signals, macro, rfree,
                                       base.signal_names, base.macro_names, log)

# --- missingness mask over heavily-missing signals (computed on RAW features) ---
n_extra = 0
miss_cols = []
if MISS:
    frac_missing = np.isnan(sf).mean(axis=0)
    miss_cols = np.where(frac_missing > 0.40)[0]
    n_extra = len(miss_cols)
    log.info(f"missingness: {n_extra} signals with >40% missing -> mask features")

# --- per-month trailing macro windows for the temporal GRU ---
macro_win = None
if MACRO_T == "gru":
    months_sorted = sorted(np.unique(mid).tolist())
    per_month_macro = {m: mf[mid == m][0] for m in months_sorted}  # (8,)
    idx = {m: i for i, m in enumerate(months_sorted)}
    seq = np.stack([per_month_macro[m] for m in months_sorted])    # (M, 8)

    def macro_window(m):
        i = idx[m]
        lo = max(0, i - LOOKBACK + 1)
        w = seq[lo:i + 1]
        if len(w) < LOOKBACK:                      # left-pad with earliest row
            w = np.concatenate([np.repeat(w[:1], LOOKBACK - len(w), axis=0), w])
        return w.astype(np.float32)                # (LOOKBACK, 8)
    macro_win = macro_window

# --- model: ExpTransformer with the experiment knobs ---
def _factory(**kw):
    return ExpTransformer(**kw, n_layers=N_LAYERS, ffn_kind=FFN,
                          n_extra=n_extra, macro_temporal=MACRO_T)
TT.CrossSectionalTransformer = _factory

cfg = TransformerConfig()
cfg.output_dir = OUTDIR
cfg.n_seeds = N_SEEDS
for fld in ("d_model", "d_ff", "n_heads", "dropout", "weight_decay", "max_epochs", "patience"):
    if fld in cfgj:
        setattr(cfg, fld, cfgj[fld])


def build_split(mask):
    scaler = build_split.scaler
    s, m = scaler.transform(sf[mask].copy(), mf[mask].copy())
    ind = build_industry_dummies(pid[mask], p2s, sic2)
    if MISS:
        msk = np.isfinite(sf[mask][:, miss_cols]).astype(np.float32)
        ind = np.concatenate([ind, msk], axis=1)            # append UNSCALED
    data = MonthGroupedData(s, m, ind, y[mask], mid[mask], pid[mask])
    if MACRO_T == "gru":                                     # overwrite macro with windows
        for mo in data.months:
            data.macro_dict[mo] = macro_window(mo)[None, ...]  # (1, L, 8)
    return data


def run_period(test_start, test_end):
    """Train n_seeds on the expanding window, return (ensemble_preds, targets, month_ids)."""
    # compute month windows in yyyymm space
    ts_y, ts_m = divmod(test_start, 100)
    val_end_m = test_start - 1 if ts_m > 1 else (ts_y - 1) * 100 + 12
    val_start_m = val_end_m  # 1-month validation (val_years=1 convention)
    train_end_m = val_start_m - 1 if (val_start_m % 100) > 1 else (val_start_m // 100 - 1) * 100 + 12

    train_mask = mid <= train_end_m
    val_mask = (mid >= val_start_m) & (mid <= val_end_m)
    test_mask = (mid >= test_start) & (mid <= test_end)
    if test_mask.sum() == 0 or val_mask.sum() == 0:
        return None

    build_split.scaler = TransformerFeatureScaler(clip_std=base.clip_std)
    build_split.scaler.fit(sf[train_mask], mf[train_mask])
    train_d, val_d, test_d = build_split(train_mask), build_split(val_mask), build_split(test_mask)

    seed_preds = []
    for seed in range(cfg.n_seeds):
        res = TT.train_model(train_d, val_d, YEAR, seed, cfg, dev, log)
        preds, _, _, _ = evaluate(res["model"], test_d, dev)
        seed_preds.append(preds)
        del res
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    ens = np.mean(seed_preds, axis=0)
    tt = np.concatenate([test_d.target_dict[m] for m in test_d.months])
    tm = np.concatenate([np.full(len(test_d.target_dict[m]), m, dtype=np.int32) for m in test_d.months])
    tp = np.concatenate([test_d.permno_dict[m] for m in test_d.months])
    return ens, tt, tm, tp


# --- periods: monthly refit (each month of YEAR, separate model) or yearly (one model) ---
# In BOTH cases the metric is the annualized Sharpe over the YEAR's 11 monthly L/S
# returns, so monthly-vs-yearly is apples-to-apples (only the refit cadence differs).
rows = []
ens = tt = tm = tp = None
if MONTHLY:
    P, T, M, Q = [], [], [], []
    for mm in range(1, 12):  # Jan..Nov (Dec dropped: no next-month return at the data edge)
        ym = YEAR * 100 + mm
        out = run_period(ym, ym)
        if out:
            P.append(out[0]); T.append(out[1]); M.append(out[2]); Q.append(out[3])
            log.info(f"  refit month {ym}: predicted {len(out[1])} stocks")
    if P:
        ens, tt, tm, tp = (np.concatenate(P), np.concatenate(T),
                           np.concatenate(M), np.concatenate(Q))
else:
    out = run_period(YEAR * 100 + 1, YEAR * 100 + 11)
    if out:
        ens, tt, tm, tp = out

import pandas as pd

if ens is not None:
    metrics = compute_oos_metrics(ens, tt, tm, log)  # Sharpe over the full 11-month series
    metrics["test_year"] = YEAR
    rows.append(metrics)

    # Per-stock predictions. Without these the pooled and VALUE-weighted Sharpes that GKX
    # actually reports cannot be recovered after the run -- only re-derived by retraining.
    # gkx_report.py consumes this file. (month_id is the FEATURE month t; the target is
    # realized at t+1, so a market-cap join on month_id is the formation-date weight.)
    pdir = os.path.join(OUTDIR, "predictions", "predictions.parquet")
    pd.DataFrame({"permno": tp.astype(np.int64), "month_id": tm.astype(np.int64),
                  "pred": ens.astype(np.float64), "target": tt.astype(np.float64),
                  }).to_parquet(pdir, index=False)
    log.info(f"wrote {pdir} ({len(tp)} stock-months)")

out = os.path.join(OUTDIR, "metrics", "exp_summary.csv")
pd.DataFrame(rows).to_csv(out, index=False)
log.info(f"wrote {out} ({len(rows)} rows)")
print(f"[exp_main] DONE {cfgj.get('outdir')} rows={len(rows)}", flush=True)
