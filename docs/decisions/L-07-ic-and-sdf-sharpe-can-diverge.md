---
id: L-07
status: accepted
supersedes: null
---

# L-07 — IC and SDF Sharpe can diverge -- judge MSRR on its objective

- **L-07 (IC and SDF Sharpe can diverge — judge MSRR on its objective):** In EXP-007 per-month rank-standardization raised the MSRR **portfolio** L1 Sharpe (+0.92 mean vs base) while leaving **IC unchanged/slightly worse** (base 0.012 ≥ rank 0.010). An MSRR model outputs portfolio WEIGHTS, not a ranking — a better-conditioned input can improve the realized portfolio (weight structure, faster/cleaner convergence) without improving rank-correlation. So judge MSRR changes on the **L1 SDF Sharpe** (the optimized objective), use IC as a secondary stability check, and never assume the two move together. (For the MSE transformer the reverse holds — there IC is the stable signal; pick the metric that matches the loss.)
