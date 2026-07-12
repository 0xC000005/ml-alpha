# Postmortem — misidentified the KKM controlled attention effect (+0.7 claimed, +0.26 actual)

Date: 2026-07-11 · Related: L-11, EXP-012

## Summary / impact
While planning EXP-012 (KKM replication), the attention effect used to size the
statistical power of Stage 2 was read as **+0.7** (transformer 4.57 vs DKKM 3.87).
That contrast moves depth *and* attention simultaneously and identifies neither. The
controlled comparison — transformer 4.57 vs the matched MLP 4.31 — is **+0.26**, an
effect roughly a third the size, sitting almost exactly at the 80%-power minimum
detectable effect for the planned sample. Had the plan proceeded on the wrong number,
~1,000 GPU-hours would have bought a near coin-flip on the paper's central claim.
Caught before any cluster time was spent, by an independent Codex review — not by the
original planning pass.

## Timeline
- 2026-07-11: EXP-012 planning identifies KKM's model ladder and reads the "+0.7"
  contrast as the effect to power Stage 2 against; also uses a 5%-significance
  threshold (x1.96) where an 80%-power minimum detectable effect (x2.80) was needed,
  and assumes rho=0.9 between compared arms (true for BSV-vs-linear-attention, not for
  MLP-vs-transformer, where KKM Fig. 3 gives rho~=0.76).
- Same day: `verify-with-codex` skill invoked on the EXP-012 plan before logging it.
  Codex identifies the correct controlled contrast (+0.26) and the two compounding
  power-sizing errors.
- Same day: further review finds both the original plan AND Codex's correction were
  powering the wrong *statistic* — KKM's actual model-comparison test is the eq. (26)
  spanning-regression alpha, not a Sharpe difference, which is far more powerful at the
  same sample size (t~=6.0 vs the Delta-Sharpe MDE's "hopeless" verdict). See L-11 Coda.

## Contributing factors
- KKM's paper reports multiple pairwise Sharpe contrasts in one table; picking the
  wrong pair (transformer vs DKKM instead of transformer vs MLP) is an easy
  cherry-pick error when skimming for "the attention number."
- Significance threshold and power threshold use the same distributional building
  block (a z-multiplier) and are easy to conflate without deriving both explicitly.
- rho was assumed from a different, more-correlated pair of arms in the same figure
  rather than looked up for the specific pair being compared.

## Lessons learned
- Before costing any A/B, name the two arms that differ in exactly one thing (L-11).
- Get rho between exactly those two arms, not a different pair from the same source.
- Use the 80%-power MDE (x2.80), not the 5% significance threshold (x1.96).
- Before declaring an experiment underpowered, check which statistic the target paper
  actually uses for its comparison — a Delta-Sharpe power analysis can say "hopeless"
  while the paper's own test (here, eq. 26's spanning alpha) says "detectable," because
  conditioning on the benchmark arm removes shared variance a raw difference doesn't.

## Action items
- [x] L-11 records the rule and the Coda records the eq. (26) correction.
- [x] EXP-012's kill-bars were rewritten to require the alpha-regression t-stat (HAC
  errors), not a Delta-Sharpe threshold.
- [ ] When planning future replication power analyses, default to searching the source
  paper for its own model-comparison statistic before deriving one from scratch.
