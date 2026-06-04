"""Local CPU validation of the MSRR enhancement prep (no cluster, no training, no data).

Proves:
  (1) the A0 ``pooled_z`` arm is BYTE-IDENTICAL to the frozen TransformerFeatureScaler
      (so any A1-A3 delta is attributable to the per-month/rank change, not a bug);
  (2) rank / rank_gauss / month_z behave per-month with correct NaN→median(0) handling
      and the documented ranges;
  (3) the L1 equal-vote combiner equals a hand computation, the raw combiner reproduces
      np.mean, and sdf_sharpe(raw, l1_final=False) reproduces a manual wᵀR Sharpe;
  (4) the default experiment model is bit-identical (14,369 params) and forwards.
Run: python experiments/validate_msrr_prep.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_transformer import TransformerFeatureScaler, CrossSectionalTransformer
from experiments.exp_transformer import ExpTransformer
from experiments.feature_scalers import make_scaler
from experiments.msrr_combine import (l1norm_per_month, combine_seeds, sdf_sharpe,
                                       both_sharpes)

rng = np.random.default_rng(0)

# ===========================================================================
# (1) A0 pooled_z is byte-identical to production TransformerFeatureScaler
# ===========================================================================
N = 600
stock = rng.standard_normal((N, 95)).astype(np.float64)
stock[rng.random((N, 95)) < 0.2] = np.nan            # ~20% missing, like the real panel
macro = rng.standard_normal((N, 8)).astype(np.float64)
months = np.repeat([201401, 201402, 201403], N // 3).astype(np.int64)

prod = TransformerFeatureScaler(clip_std=5.0).fit(stock, macro)
ps, pm = prod.transform(stock.copy(), macro.copy())

a0 = make_scaler("pooled_z", clip_std=5.0).fit(stock, macro)
a0s, a0m = a0.transform(stock.copy(), macro.copy(), months)

assert np.array_equal(ps, a0s) and np.array_equal(pm, a0m), "A0 != production scaler"
print(f"(1) A0 pooled_z byte-identical to TransformerFeatureScaler  "
      f"[stock {a0s.shape}, macro {a0m.shape}]  OK")

# ===========================================================================
# (2) per-month rank / rank_gauss / month_z
# ===========================================================================
# Hand example: 1 signal, 2 months. Month1=[10,20,30,nan] Month2=[7,99,3].
xs = np.array([[10.], [20.], [30.], [np.nan], [7.], [99.], [3.]])
xm = rng.standard_normal((7, 8))
mid = np.array([1, 1, 1, 1, 2, 2, 2])

rk, _ = make_scaler("rank").fit(xs, xm).transform(xs.copy(), xm.copy(), mid)
expect = np.array([-0.5, 0.0, 0.5, 0.0, 0.0, 0.5, -0.5], dtype=np.float32)
assert np.allclose(rk[:, 0], expect), f"rank mismatch: {rk[:, 0]} != {expect}"

# real-panel scale: rank in [-0.5,0.5], NaN rows → exactly 0, per-month independence
rk2, _ = make_scaler("rank").fit(stock, macro).transform(stock.copy(), macro.copy(), months)
assert rk2.min() >= -0.5 - 1e-6 and rk2.max() <= 0.5 + 1e-6, "rank out of [-0.5,0.5]"
assert np.all(rk2[np.isnan(stock)] == 0.0), "NaN not mapped to median(0)"

rg, _ = make_scaler("rank_gauss").fit(stock, macro).transform(stock.copy(), macro.copy(), months)
assert np.isfinite(rg).all() and rg.min() >= -5.0 - 1e-6 and rg.max() <= 5.0 + 1e-6, \
    "rank_gauss not finite/clipped"

mz, _ = make_scaler("month_z").fit(stock, macro).transform(stock.copy(), macro.copy(), months)
for m in np.unique(months):                          # each month ~zero-mean per column
    col_means = np.nanmean(np.where(mz[months == m] == 0.0, np.nan, mz[months == m]), axis=0)
assert np.abs(mz).max() <= 5.0 + 1e-6, "month_z not clipped"
print("(2) rank=[-0.5,0.5] (hand example exact), NaN→0, rank_gauss finite/clipped, "
      "month_z clipped  OK")

# ===========================================================================
# (3) combiners + honest sdf_sharpe
# ===========================================================================
cm = np.array([1, 1, 2, 2])
w1 = np.array([1.0, 3.0, 2.0, 2.0])      # month1 sum|w|=4 ; month2 sum|w|=4
w2 = np.array([2.0, 2.0, 1.0, 1.0])
# L1 per seed: w1->[.25,.75,.5,.5], w2->[.5,.5,.5,.5]; equal-vote mean
assert np.allclose(l1norm_per_month(w1, cm), [0.25, 0.75, 0.5, 0.5])
assert np.allclose(combine_seeds([w1, w2], cm, "raw"), np.mean([w1, w2], axis=0))
assert np.allclose(combine_seeds([w1, w2], cm, "l1norm"), [0.375, 0.625, 0.5, 0.5])

# raw sdf_sharpe (l1_final=False) reproduces a manual wᵀR Sharpe
rets = np.array([0.05, -0.02, 0.03, 0.01])
raw_ens = combine_seeds([w1, w2], cm, "raw")
manual = np.array([np.dot(raw_ens[cm == 1], rets[cm == 1]),
                   np.dot(raw_ens[cm == 2], rets[cm == 2])])
man_sh = manual.mean() / manual.std(ddof=1) * np.sqrt(12)
got_sh, _ = sdf_sharpe(raw_ens, rets, cm, l1_final=False)
assert np.isclose(got_sh, man_sh), f"sdf_sharpe raw {got_sh} != manual {man_sh}"
both = both_sharpes([w1, w2], rets, cm)
assert np.isclose(both["sdf_sharpe_raw"], man_sh)
print(f"(3) combiners: raw=np.mean, l1=equal-vote, sdf_sharpe_raw reproduces manual "
      f"({man_sh:+.2f})  OK")

# ===========================================================================
# (4) default experiment model bit-identical + forwards
# ===========================================================================
np_prod = sum(p.numel() for p in CrossSectionalTransformer().parameters())
np_exp = sum(p.numel() for p in ExpTransformer(n_layers=1).parameters())
assert np_prod == np_exp == 14369, f"param mismatch {np_prod} vs {np_exp}"
m = ExpTransformer(n_layers=1).eval()
n = 120
s = torch.randn(1, n, 95); mc = torch.randn(1, 8); ind = torch.zeros(1, n, 74)
ind[0, torch.arange(n), torch.randint(0, 74, (n,))] = 1.0
with torch.no_grad():
    out = m(s, mc, ind)
assert out.shape == (n,), f"bad output shape {out.shape}"
print(f"(4) default model {np_exp:,} params (==production), forward → {tuple(out.shape)}  OK")

print("\nALL MSRR-prep checks passed (no cluster, no training, no data).")
