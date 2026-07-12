---
id: L-06
status: accepted
supersedes: null
---

# L-06 — Period-matching and the 1-survivor verdict

- **L-06 (period-matching + the 1-survivor verdict):** The paper's ~4 vs our ~2 is **~80% regime + scale, not data.** The 4.57 headline is *full-sample 1968–2022*; every model halves after ~2002, and their best **modern** transformer is **3.37** while their **linear BSV** baseline is **2.03** — exactly where our reproduction sits. So period-matched we already equal their linear modern-era baseline; closing to ~3.4 is a capacity+apparatus problem (deeper stack, monthly refit, CV'd ridge, rank-normed inputs), and their own decomposition says **depth does most of the work, cross-asset attention is the smaller increment.** When 20 enhancement candidates were put through adversarial scrutiny against our ±0.4–0.5 window-noise floor, **only per-month rank-standardization survived as `robust=true`** — most "obvious" knobs (width, GLU, extra heads, missingness, macro interactions, turnover penalty, LR schedules) have plausible effects *smaller than the sampling SE on 48–96 OOS months*, so they literally cannot be resolved on our windows. Corollary: the binding constraint is **statistical power (OOS months), not ideas** — which is why monthly refit (≈12× the OOS observations) is the highest-leverage *regime* change even though its per-estimate effect is itself subfloor.
