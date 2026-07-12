---
id: EXP-011
status: planned-gated
---

# EXP-011 — Delisting sensitivity (PLANNED, GATED)
- **Question:** does D-4 move the number at all? The delisted stocks were never in the panel, so the model never scored them and they carry **no weight** — the bias cannot be measured by reweighting. It must be measured by **rebuilding the panel with them kept**.
- **Design:** rebuild `build_long_panel` keeping universe-at-*t* stocks with a NaN *t+1* return, imputing that return two ways — **(a) −30%** (the CRSP performance-delist convention) and **(b) 0%** (the merger convention). GKX ships no delisting codes, so the two arms **bracket** the truth. Retrain MSRR, 10 seeds × 8 years, compare pooled Sharpe against base.
- **Kill-bar (written before the run):** if the pooled Sharpe under the harsh (−30%) arm moves by **< 0.15**, declare the bias immaterial, record it, and stop. Only if it moves more is a CRSP delist-file merge worth the effort.
- **Cost:** ~1 full MSRR confirmation run (2 arms × 8 yr × 10 seeds). **Gated** — do not submit without go-ahead.

---

