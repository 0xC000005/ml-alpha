---
id: EXP-006
status: planned-gated
---

# EXP-006 — Monthly refit

- **EXP-006 — Monthly refit** (idea 5): `monthly.jsonl` (exp_main, 1 yr=2018). Prep ✅. ~3–4 GPU-h. **P5a power note (CORRECTED 2026-07-11, EXP-010 D-2 — the original was wrong):** SE(**annualized** Sharpe) = **√12** · √((1+½·SR_m²)/T), T in months, SR_m = *monthly* Sharpe. Over 1 year (T≈11) **SE ≈ 1.11**; over 8 yrs (T≈88) **SE ≈ 0.39**. ~~0.30 / 0.11~~ — those were the *monthly* SEs, understating noise by √12 ≈ 3.46×. A realistic monthly-vs-yearly gain (≤~0.3 Sharpe) is therefore **not merely near-undetectable but far below the 1-year floor**; a single-year probe can tell you essentially nothing. Also note **monthly refit does NOT create 12× OOS observations** — it fits 12× more models over the *same* calendar returns. This remains the lowest-priority direction.
