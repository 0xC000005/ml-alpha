"""MSRR ensemble combiners + the honest SDF-Sharpe metric (Tier 0 of the
enhancement roadmap; see RESEARCH_LOG.md L-02 / EXP-002).

Per-seed MSRR portfolio weights are SCALE-INVARIANT (Sharpe is unchanged if you
multiply a seed's weights by a constant), so the production ensemble
``np.mean(seed_preds, axis=0)`` (train_transformer_msrr.py:431) is a MAGNITUDE-weighted
average — dominated by whichever seed happened to train to the largest scale, giving an
effective ensemble size far below n_seeds. That makes a measured "win" potentially just
a lucky-large-seed artifact.

The honest combiner L1-normalizes each seed's weights PER MONTH (‖w‖₁ = 1) before
averaging — an equal-vote ensemble — and L1-normalizes the final ensemble per month so
every month contributes a unit-gross-leverage portfolio return (fixed leverage). This
matches `cluster/ab_msrr_norm.py::l1norm_per_month` exactly, so the A/B reanalysis over
saved per-seed models stays consistent.

Use this module's `sdf_sharpe(..., l1_final=False)` on the RAW mean to reproduce the
saved/production `sdf_sharpe`, and `combine_seeds(..., 'l1norm')` + `sdf_sharpe(...,
l1_final=True)` for the honest number. Always report BOTH: any gain that exists only in
the raw column is a scale artifact and must be rejected.
"""
from typing import List, Sequence, Tuple

import numpy as np
from scipy.stats import trim_mean


def l1norm_per_month(weights: np.ndarray, month_ids: np.ndarray) -> np.ndarray:
    """Scale each month's weights to ‖w‖₁ = 1 (a month with all-zero weights is left
    untouched). Mirrors cluster/ab_msrr_norm.py exactly."""
    out = np.asarray(weights, dtype=np.float64).copy()
    for m in np.unique(month_ids):
        mask = month_ids == m
        s = np.abs(out[mask]).sum()
        if s > 0:
            out[mask] = out[mask] / s
    return out


def combine_seeds(seed_preds: Sequence[np.ndarray], month_ids: np.ndarray,
                  kind: str = "l1norm", trim: float = 0.1) -> np.ndarray:
    """Combine a list of per-seed weight vectors (each aligned to ``month_ids``).

    kind:
      raw      → plain mean across seeds (reproduces production; magnitude-weighted).
      l1norm   → L1-normalize each seed per month, then mean (equal-vote). RECOMMENDED.
      trimmed  → L1-normalize each seed per month, then per-element trimmed mean across
                 seeds (drops the most extreme ``trim`` fraction each side — robust to a
                 single rogue seed even after equal-voting).
    """
    arr = np.asarray(seed_preds, dtype=np.float64)  # (S, n)
    if kind == "raw":
        return arr.mean(axis=0)
    normed = np.stack([l1norm_per_month(arr[s], month_ids) for s in range(arr.shape[0])])
    if kind == "l1norm":
        return normed.mean(axis=0)
    if kind == "trimmed":
        return trim_mean(normed, trim, axis=0)
    raise ValueError(f"unknown combiner kind {kind!r}")


def sdf_sharpe(weights: np.ndarray, returns: np.ndarray, month_ids: np.ndarray,
               l1_final: bool = True) -> Tuple[float, np.ndarray]:
    """Annualized SDF-portfolio Sharpe from weights `w` and excess returns `R`.

    Monthly SDF return is wᵀR; Sharpe = mean/std·√12 (ddof=1), matching
    `compute_sdf_portfolio_metrics`. With l1_final=True the ensemble is L1-normalized
    per month first (fixed unit gross leverage) — the honest convention. With
    l1_final=False it reproduces the production raw wᵀR.
    """
    w = l1norm_per_month(weights, month_ids) if l1_final else np.asarray(weights, float)
    months = np.unique(month_ids)
    rets = np.array([float(np.dot(w[month_ids == m], returns[month_ids == m]))
                     for m in months])
    mean = rets.mean()
    std = rets.std(ddof=1) if len(rets) > 1 else 1.0
    sharpe = (mean / std * np.sqrt(12)) if std > 0 else 0.0
    return float(sharpe), rets


def both_sharpes(seed_preds: List[np.ndarray], returns: np.ndarray,
                 month_ids: np.ndarray) -> dict:
    """Convenience: report raw (production-reproducing) AND L1 equal-vote SDF Sharpe."""
    raw_ens = combine_seeds(seed_preds, month_ids, "raw")
    l1_ens = combine_seeds(seed_preds, month_ids, "l1norm")
    sdf_raw, _ = sdf_sharpe(raw_ens, returns, month_ids, l1_final=False)
    sdf_l1, l1_rets = sdf_sharpe(l1_ens, returns, month_ids, l1_final=True)
    return {
        "sdf_sharpe_raw": sdf_raw,
        "sdf_sharpe_l1": sdf_l1,
        "sdf_mean_ret_l1": float(l1_rets.mean()),
        "sdf_std_ret_l1": float(l1_rets.std(ddof=1) if len(l1_rets) > 1 else 0.0),
        "raw_ensemble": raw_ens,
        "l1_ensemble": l1_ens,
    }
