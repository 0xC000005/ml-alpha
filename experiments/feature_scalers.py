"""Configurable per-month feature scalers for the rank-standardization A/B (B-11).

The frozen production scaler (`TransformerFeatureScaler`) standardizes each signal
with a POOLED z-score (mean/std over the whole train window) + 5σ clip. The paper's
AIPM (w33351 §4.1) instead CROSS-SECTIONALLY RANK-standardizes each characteristic
PER MONTH to [-0.5, 0.5], imputing missing → the cross-sectional median (0). We diverge
on three axes at once (pooled→monthly, mean→median, z→rank), so the screen breaks the
confound into four arms:

  A0 ``pooled_z``    = the production `TransformerFeatureScaler` EXACTLY (the control).
  A1 ``month_z``     = per-month cross-sectional z-score (this month's own mean/std),
                       NaN→0, clip. Isolates per-month-vs-pooled standardization.
  A2 ``rank``        = per-month cross-sectional rank → [-0.5, 0.5], NaN→0 (= median).
                       The paper's exact input contract. The candidate.
  A3 ``rank_gauss``  = per-month rank mapped through the inverse-normal CDF (van der
                       Waerden / Gaussianized rank): robust like a rank but with
                       informative, unbounded-then-clipped tails. The "retains tail
                       magnitude" arm — load-bearing because a PURE rank destroys tail
                       info, and if the SDF signal lives in the extremes pure rank can
                       cancel or reverse the conditioning gain.

All arms share ONE macro path: the 8 macros are z-scored on the TRAIN window only
(reusing `FeatureScaler` stats) and clipped — macro is a single shared row per month,
there is nothing cross-sectional to rank. CRITICAL: these scalers transform ONLY the
95 signals. The 74 industry dummies are 0/1 and are appended UNSCALED downstream
(never passed through here).

Uniform interface (so the experiment driver's `build_split` is arm-agnostic):
    scaler = make_scaler(kind, clip_std)
    scaler.fit(stock_features, macro_features)            # macro stats (+ A0 stock stats)
    stock_scaled, macro_scaled = scaler.transform(stock, macro, month_ids)
"""
import warnings

import numpy as np
from scipy.stats import norm, rankdata

from train_nn import FeatureScaler
from train_transformer import TransformerFeatureScaler


# ---------------------------------------------------------------------------
# Per-month stock transforms (operate on ONE month's (n_stocks, n_signals) block)
# ---------------------------------------------------------------------------

def _zscore_month(x: np.ndarray, clip_std: float) -> np.ndarray:
    """Per-month cross-sectional z-score; NaN→0 (= mean); clip to ±clip_std."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns
        mu = np.nanmean(x, axis=0)
        sd = np.nanstd(x, axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    z = (x - mu) / sd
    np.nan_to_num(z, copy=False, nan=0.0)
    np.clip(z, -clip_std, clip_std, out=z)
    return z.astype(np.float32)


def _rank_month(x: np.ndarray, clip_std: float, gauss: bool) -> np.ndarray:
    """Per-month cross-sectional rank.

    gauss=False → linear rank to [-0.5, 0.5] via (rank-1)/(n_valid-1) - 0.5 (A2).
    gauss=True  → inverse-normal of the rank percentile Φ⁻¹(rank/(n_valid+1)),
                  clipped to ±clip_std (A3, van der Waerden scores).
    NaN / a column with <2 valid obs → 0 (the cross-sectional median).
    Ranks are computed independently per signal over that signal's non-missing
    cross-section, so different missing counts per column are handled correctly.
    """
    out = np.zeros((x.shape[0], x.shape[1]), dtype=np.float32)
    for j in range(x.shape[1]):
        col = x[:, j]
        valid = np.isfinite(col)
        nv = int(valid.sum())
        if nv < 2:
            continue  # leave 0 (median)
        r = rankdata(col[valid], method="average")  # 1..nv, ties averaged
        if gauss:
            u = r / (nv + 1.0)                       # in (0,1), never 0/1 → finite ppf
            z = norm.ppf(u).astype(np.float32)
            np.clip(z, -clip_std, clip_std, out=z)
            out[valid, j] = z
        else:
            out[valid, j] = ((r - 1.0) / (nv - 1.0) - 0.5).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Scalers (uniform .fit / .transform(stock, macro, month_ids))
# ---------------------------------------------------------------------------

class _PooledZAdapter:
    """A0: the production `TransformerFeatureScaler`, behind the 3-arg signature
    (month_ids is accepted and ignored). Guarantees the control arm is byte-identical
    to production, so any A1–A3 delta is attributable to the per-month/rank change."""

    kind = "pooled_z"

    def __init__(self, clip_std: float = 5.0):
        self._inner = TransformerFeatureScaler(clip_std=clip_std)

    def fit(self, stock_features: np.ndarray, macro_features: np.ndarray):
        self._inner.fit(stock_features, macro_features)
        return self

    def transform(self, stock_features: np.ndarray, macro_features: np.ndarray,
                  month_ids: np.ndarray = None):
        return self._inner.transform(stock_features, macro_features)


class PerMonthScaler:
    """A1/A2/A3: per-month cross-sectional stock transform + train-window macro z."""

    def __init__(self, kind: str, clip_std: float = 5.0):
        if kind not in ("month_z", "rank", "rank_gauss"):
            raise ValueError(f"unknown per-month kind {kind!r}")
        self.kind = kind
        self.clip_std = clip_std
        self._macro_stats = FeatureScaler(clip_std=clip_std)  # macro mean/std only

    def fit(self, stock_features: np.ndarray, macro_features: np.ndarray):
        # FeatureScaler.fit computes stock + macro stats cheaply (nanmean/nanstd);
        # we only consume the macro stats — stock stats are unused (no train snoop).
        self._macro_stats.fit(stock_features, macro_features)
        return self

    def _macro(self, macro_features: np.ndarray) -> np.ndarray:
        m = (macro_features - self._macro_stats.macro_mean_) / self._macro_stats.macro_std_
        np.nan_to_num(m, copy=False, nan=0.0)
        np.clip(m, -self.clip_std, self.clip_std, out=m)
        return m.astype(np.float32)

    def transform(self, stock_features: np.ndarray, macro_features: np.ndarray,
                  month_ids: np.ndarray):
        if month_ids is None:
            raise ValueError(f"{self.kind} requires month_ids for per-month grouping")
        out = np.zeros((stock_features.shape[0], stock_features.shape[1]),
                       dtype=np.float32)
        gauss = self.kind == "rank_gauss"
        for mo in np.unique(month_ids):
            idx = np.where(month_ids == mo)[0]
            xs = stock_features[idx]
            if self.kind == "month_z":
                out[idx] = _zscore_month(xs, self.clip_std)
            else:
                out[idx] = _rank_month(xs, self.clip_std, gauss=gauss)
        return out, self._macro(macro_features)


def make_scaler(kind: str, clip_std: float = 5.0):
    """Factory. kind ∈ {pooled_z (A0), month_z (A1), rank (A2), rank_gauss (A3)}."""
    if kind in ("pooled_z", "a0", "base"):
        return _PooledZAdapter(clip_std=clip_std)
    return PerMonthScaler(kind, clip_std=clip_std)
