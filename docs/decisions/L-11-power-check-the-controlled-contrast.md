---
id: L-11
status: accepted
supersedes: null
---

# L-11 — Power-check the controlled contrast, not the headline one

- **L-11 (power-check the *controlled* contrast, not the headline one — 2026-07-11):** planning EXP-012, I read KKM's nonlinear attention effect as **+0.7** (transformer 4.57 vs DKKM 3.87) and concluded the 635-month rolling window made it comfortably testable. Wrong. That contrast moves **depth *and* attention** and identifies neither. The **controlled** one — transformer 4.57 vs the matched MLP 4.31 — is **+0.26**, and with ρ≈0.76 between the two arms (KKM Fig. 3) the 80%-power MDE at T=635 is **≈0.27**. The effect *is* the noise floor. Two compounding errors made it look safe: (i) I quoted a 5%-significance threshold (×1.96) as if it were a minimum detectable effect (×2.80 for 80% power), and (ii) I assumed ρ=0.9 — true for BSV-vs-linear-attention, **not** for MLP-vs-transformer. Had this run, ~1,000 GPU-hours would have bought a coin-flip on the paper's central claim: **EXP-007's failure mode at 50× the cost.** Caught by Codex, not by me. **Rule: before costing any A/B, name the two arms that differ in exactly one thing, get ρ between them, and use the 80%-power MDE — not the 5% threshold, and not the effect from a contrast that moves two knobs.**

  **Coda — and then we were both wrong in the other direction.** Having "established" that
  Stage 2 was underpowered, I nearly cancelled it. But KKM's model-comparison statistic is
  **not** a Sharpe difference: it is the **spanning regression** of eq. (26),
  `R_A = α + β·R_B + ε` with both legs rescaled to 15% annualized vol (their "t = 6.8"). That
  test conditions on `R_B` and only needs to detect the residual mean — at ρ=0.76 the residual
  vol is 65% of total — giving **t(α) ≈ 6.0** at our Sharpe levels over 635 months (≈4.0 with a
  1.5× HAC haircut) where the Δ-Sharpe MDE said "hopeless". Same data, same effect, opposite
  verdict, purely from choice of estimator. **Corollary: a power calculation is only as good as
  the test it powers. Before declaring an experiment un-runnable, check which statistic the
  paper actually uses — the difference between "coin-flip" and "t = 6" here was reading eq. (26).** (Codex also caught that the KKM nonlinear models are *unpenalized*, that its linear benchmark is BSV rather than an attention-disabled transformer, that our per-seed L1 normalization and early stopping are repo inventions absent from the paper, and that the NYSE-percentile nano filter **cannot be built from the parquets on disk** — they carry market cap but no exchange code.)
