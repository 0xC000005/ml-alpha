---
id: EXP-003
status: done
---

# EXP-003 — Capacity / Virtue-of-Complexity cheap screen (MSE transformer proxy)
- **Date:** 2026-06-03 · **Status:** ✅ done (inconclusive — *tested-weak*) · **Array:** 564717 (`capacity.jsonl`, 5 configs × 3 screen years = 15 tasks, %2)
- **Hypothesis (B-01):** more capacity — depth (`n_layers`) and/or width (`d_model`) — raises OOS Sharpe in this regime.
- **Setup:** MSE cross-sectional transformer, screen years 2014/2016/2018; configs base(K=1,d32)/L2(K=2)/L3(K=3)/d64(d_model=64)/d64L2(d64,K=2); metric `sharpe_ls_annual` (decile L/S).
- **Result (mean over 3 yrs):** base **2.02**; L2 **2.47** (win concentrated in 2014 +1.69, flat/worse 2016/2018); L3 **1.95** (<base; 2016 collapse −0.89); d64 **1.11** (worst, 0/3 — width hurts every year); d64L2 **2.16** (2/3 but 2016 craters −1.20).
- **Conclusion:** **no config clears the promote bar** (≥2/3 yrs AND clearly higher mean w/ controlled dispersion). Year-to-year swings (2–3 Sharpe) **dwarf** every config effect → *noise-limited at this scale*. One robust signal: **widening `d_model` alone HURTS** (matches the paper's kernel-limit — capacity must come from depth, not a fatter embedding). Depth is noisy-positive at best and **non-monotone** (L3<base).
- **Next:** re-test depth on the **MSRR** model under the **L1-normalized honest metric** with a strict monotone kill-bar (Roadmap Tier 2); do **not** widen `d_model`.

