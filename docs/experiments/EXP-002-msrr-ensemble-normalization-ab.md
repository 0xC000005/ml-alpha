---
id: EXP-002
status: done
---

# EXP-002 — MSRR ensemble weight-normalization A/B (no retraining)
- **Date:** 2026-06-03 · **Status:** ✅ done (tested-negative as a Sharpe win) · **Script:** `cluster/ab_msrr_norm.py`
- **Hypothesis (from brainstorm):** `train_transformer_msrr.py:431` averages RAW per-seed weights; since MSRR Sharpe is scale-invariant, the largest-magnitude seed dominates the mean → L1-normalizing each seed's per-month weights before averaging should *improve* the ensemble.
- **Setup:** reload the 80 saved per-seed MSRR models, reconstruct each seed's test predictions (no training), compare raw-mean (A) vs L1-normalized-mean (B). Sanity: raw(A) must equal the saved `sdf_sharpe`. Fixed test window to Jan–Nov (11 mo) to match the original per-year data loads — after which raw(A) reproduced saved EXACTLY for all 8 years.
- **Result:** normalization **LOWERS** avg SDF Sharpe (2016–19: **3.13 → 2.06**; 2012–19: 2.14 → 1.78). It rescues blow-up years (2014 −1.13→+0.60, 2017 0.66→1.97) but caps luck-driven star years (2018 5.31→0.68, 2019 4.30→3.07).
- **Conclusion:** the fix is *principled-correct* (raw mean weights seeds by an arbitrary, signal-free magnitude) but is **not a Sharpe win** — it's a robustness trade. The hot +3.13 reproduction was partly **seed-scale luck**; the normalized **+2.06 ≈ the published +2.05** is the more honest/robust estimate.
- **Next:** decision pending on adopting normalization + reporting both (B-05). Key lesson → L-02.

