---
id: L-09
status: accepted
supersedes: null
---

# L-09 — The metric was the bug -- audit the statistic, not just the model

- **L-09 (the metric was the bug — audit the statistic, not just the model; EXP-010):** Every error this project has actually made was **statistical, not modelling**, and each one read as perfectly reasonable code. Four, all verified 2026-07-11: (1) **never average annual Sharpes** — `collect_screen.py` row-means them; that is a mean of ratios, biased up (1.81 vs pooled 1.41). Pool the monthly return series. (2) **Annualize the SE** — `√12·√((1+½·SR_m²)/T)`; one year → **±1.11**, eight years → **±0.39**. A single-year Sharpe is nearly pure noise, and this log understated that by 3.46×. (3) **Pair every A/B and quote a t-stat** — EXP-009 base−a2rank: mean diff **+0.391**, sd 1.016, **t = 1.09** over 8 years → **no detectable difference**. It was logged as a "reversal"; that is over-reading noise in the *opposite* direction from EXP-007's false positive. Below |t|≈2.4 you have found **nothing**, and "slightly worse" / "trends toward" are forbidden phrasings. (4) **Seeds are not observations** — they cut estimator variance, they add zero economic sample size; only months do. Corollary to L-06: the binding constraint really is OOS months, and an effect below ~0.5 Sharpe **cannot be resolved on these windows at all** — so the prior for any such result is "this is noise." Enforcement: `stats-gatekeeper` agent + a PreToolUse hook (`.claude/hooks/research-log-checklist.sh`) that injects this checklist into the model's context on every `RESEARCH_LOG.md` write. It **auto-allows** the write — the log is the record of autonomous work and must never require a human click to update; the discipline binds the model, not the user.
