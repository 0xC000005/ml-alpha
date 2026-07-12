---
id: EXP-009
status: done
---

# EXP-009 — Full confirmation of rank-standardization (B-11), 8yr × 10 seeds, MSRR
- **Date:** 2026-06-03/04 · **Status:** ❌ **done — REVERSES the screen; rank NOT adopted** · **Array:** 565895 (24 tasks = base/a1monthz/a2rank × 2012–2019 × **10 seeds**, %2; driver `exp_main_msrr.py`) · **Metric:** L1 SDF Sharpe.
- **Result (8yr mean L1 SDF Sharpe):** **base 1.81 > a2rank 1.42 > a1monthz 0.46.** a2rank beats base in only **3/8** years; a1monthz 2/8. **mean IC tied** (base ≈ a2rank ≈ 0.014; a1monthz 0.006). So **base (current pooled-z) is best** — rank-standardization does not help and is slightly worse.
- **Why the screen lied:** the 3-year screen (EXP-007, **5 seeds**) caught base on unlucky low draws in exactly 2014/16/18 — at 10 seeds those base numbers jump (base 2014 L1: **1.07 @5seeds → 3.18 @10seeds**; base 2016: −1.24 → +1.31), which **erased** a2rank's apparent +0.92 edge. The honest L1 ensemble is *still* so noisy that adding 5 seeds swings a single-year Sharpe by >2 points.
- **Conclusion:** **keep pooled-z; do NOT adopt rank-standardization** (B-11 ❌). a1monthz's screen "robustness" was also a small-sample fluke (catastrophic 2012 −2.96, 2018 −1.14). Confirmation discipline (L-01) caught a false positive — working as intended. → L-08.
- **Next:** pivot to a lever with a *larger* expected effect than the noise floor — the gated MSRR depth ladder (EXP-008) or paper-regime training (Tier 3). Re-screen only with ≥10 seeds.
- **⚠️ SUPERSEDED IN PART by EXP-010 (2026-07-11):** the metric this entry is judged on (mean-of-annual-Sharpes) is biased, and the base-vs-a2rank gap is **not statistically significant** (paired t=1.09). The *decision* stands — do not adopt rank — but the stated reason ("base is best / rank is slightly worse") is over-read. The honest conclusion is **no detectable difference**. See EXP-010, L-09.

---

