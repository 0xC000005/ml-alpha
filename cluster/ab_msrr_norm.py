#!/usr/bin/env python
"""A/B test the MSRR ensemble-averaging fix WITHOUT retraining.

The MSRR path averages RAW per-seed weight vectors (train_transformer_msrr.py:431
`np.mean(seed_preds)`). Because MSRR's Sharpe is scale-invariant, each seed's
weights converge at an arbitrary magnitude, so the raw mean is dominated by the
largest-scale seed. This reloads the 80 saved per-seed models, reconstructs each
seed's test predictions, and compares:
  (A) raw mean  (current behaviour)   vs   (B) L1-normalized-per-month mean (fix)

Sanity check: (A) must reproduce the saved summary's sdf_sharpe — confirming the
preprocessing reload is exact, so (B) is trustworthy. No training.

Run on a GPU (evaluate() uses autocast('cuda')):
  srun --account=def-cglee --nodes=1 --gpus-per-node=1 --cpus-per-task=24 --time=0:20:00 \
       bash -lc 'cd /scratch/maxzhang/ml-alpha && source activate_cluster.sh && python cluster/ab_msrr_norm.py'
"""
import os
import sys
import logging
import numpy as np
import torch
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_nn import (load_returns, load_universe, load_signals, load_macro,
                      load_sector_mapping, build_long_panel)
from train_transformer import (CrossSectionalTransformer, MonthGroupedData,
                               TransformerFeatureScaler, evaluate, build_industry_dummies)
from train_transformer_msrr import MSRRConfig

log = logging.getLogger("ab")
logging.basicConfig(level=logging.ERROR)
assert torch.cuda.is_available(), "run on a GPU (evaluate uses autocast cuda)"
dev = torch.device("cuda")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg = MSRRConfig()
YEARS = list(range(2012, 2020))


def sdf_sharpe(w, r, months):
    rets = np.array([np.dot(w[months == m], r[months == m]) for m in np.unique(months)])
    mu, sd = rets.mean(), rets.std(ddof=1)
    return mu / sd * np.sqrt(12) if sd > 0 else 0.0


def l1norm_per_month(w, months):
    out = w.astype(np.float64).copy()
    for m in np.unique(months):
        mask = months == m
        s = np.abs(out[mask]).sum()
        if s > 0:
            out[mask] /= s
    return out


def saved_sdf(year):
    p = f"{REPO}/output/repro/msrr_{year}/metrics/msrr_transformer_summary.csv"
    if os.path.exists(p):
        return float(pd.read_csv(p).iloc[0]["sdf_sharpe"])
    return float("nan")


print(f"device={dev} ({torch.cuda.get_device_name(0)})", flush=True)
print("loading data 1975-2019 ...", flush=True)
end = max(YEARS)
returns = load_returns(cfg.data_dir, cfg.train_start, end, log)
universe = load_universe(cfg.data_dir, cfg.train_start, end, log)
signals = load_signals(cfg.data_dir, cfg.signal_names, cfg.train_start, end, log)
macro, rfree = load_macro(cfg.macro_file, cfg.train_start, end, log)
p2s, sic2 = load_sector_mapping(cfg.sector_file, log)
sf, mf, y, mid, pid = build_long_panel(universe, returns, signals, macro, rfree,
                                       cfg.signal_names, cfg.macro_names, log)

hdr = f"{'year':>5} {'saved':>7} {'raw(A)':>7} {'norm(B)':>8} {'seedΔ':>14} {'scaleratio':>11}"
print("\n" + hdr)
print("-" * len(hdr))
rows = []
for Y in YEARS:
    val_end, val_start = Y - 1, Y - cfg.val_years
    train_end = val_start - 1
    trm = mid <= train_end * 100 + 12
    # original per-year tasks loaded data only through year Y, so the test
    # year's December was dropped (no next-month return) -> 11 months (Jan-Nov).
    tm = (mid >= Y * 100 + 1) & (mid <= Y * 100 + 11)
    scaler = TransformerFeatureScaler(clip_std=cfg.clip_std)
    scaler.fit(sf[trm], mf[trm])
    s_t, m_t = scaler.transform(sf[tm].copy(), mf[tm].copy())
    ind_t = build_industry_dummies(pid[tm], p2s, sic2)
    test_data = MonthGroupedData(s_t, m_t, ind_t, y[tm], mid[tm], pid[tm])

    seed_preds, seed_scale, seed_sdf = [], [], []
    tgts = mths = None
    for s in range(cfg.n_seeds):
        mp = f"{REPO}/output/repro/msrr_{Y}/models/MSRR_year{Y}_seed{s}.pt"
        if not os.path.exists(mp):
            continue
        model = CrossSectionalTransformer(
            n_signals=cfg.n_signals, n_industries=cfg.n_industries, n_macro=cfg.n_macro,
            d_model=cfg.d_model, n_heads=cfg.n_heads, d_ff=cfg.d_ff, dropout=cfg.dropout).to(dev)
        model.load_state_dict(torch.load(mp, map_location=dev, weights_only=True))
        model.eval()
        preds, tgts, mths, _ = evaluate(model, test_data, dev)
        seed_preds.append(preds)
        seed_scale.append(float(np.abs(preds).mean()))
        seed_sdf.append(sdf_sharpe(preds, tgts, mths))

    sp = np.array(seed_preds)
    raw = sdf_sharpe(sp.mean(axis=0), tgts, mths)
    norm_sp = np.array([l1norm_per_month(p, mths) for p in sp])
    norm = sdf_sharpe(norm_sp.mean(axis=0), tgts, mths)
    ss = np.array(seed_sdf)
    sc = np.array(seed_scale)
    ratio = sc.max() / max(sc.min(), 1e-12)
    print(f"{Y:>5} {saved_sdf(Y):>7.2f} {raw:>7.2f} {norm:>8.2f} "
          f"[{ss.min():>5.1f},{ss.max():>5.1f}] {ratio:>11.1f}", flush=True)
    rows.append((Y, raw, norm))

r = np.array([x[1] for x in rows])
n = np.array([x[2] for x in rows])
print(f"\nAVG 2012-2019 : raw(A)={r.mean():+.3f}  norm(B)={n.mean():+.3f}")
print(f"AVG 2016-2019 : raw(A)={r[4:].mean():+.3f}  norm(B)={n[4:].mean():+.3f}   (README +2.05)")
print("\nseedΔ = [min,max] per-seed SDF Sharpe (dispersion); scaleratio = max/min seed weight magnitude")
print("If raw(A) ~= saved, the reload is exact and norm(B) is a valid A/B.")
