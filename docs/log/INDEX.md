# Research log — index

Append-only narrative journal. **Newest month first; within a month, newest entry on
top.** Pull numbers from `output/exp/` + `experiments/manifest.py`; record load-bearing
decisions as `docs/decisions/L-NN-*.md`. Pre-2026-07-12 history:
`docs/log/archive/RESEARCH_LOG-2026-06-02-to-2026-07-12.md`.

## Months
- [2026-07](2026-07.md) — migrated to this layered docs/ system (see the 2026-07-12 entry).

## Entry conventions
- Header: `## YYYY-MM-DD — subject`
- Write fast and messy — capture, not polish. Never rewrite a past entry; supersede with
  a newer one.

## Experiments (EXP-NNN)

| ID | Status | Title | Verdict |
|----|--------|-------|---------|
| [EXP-001](../experiments/EXP-001-reproduce-readme-transformer.md) | done | Reproduce README Transformer results | MSE L/S Sharpe 2.84, MSRR raw SDF 3.13 — signal faithful, Sharpe ran hot (seed luck) |
| [EXP-002](../experiments/EXP-002-msrr-ensemble-normalization-ab.md) | done | MSRR ensemble L1-normalization A/B | L1 is the honest combiner; adopted as metric |
| [EXP-003](../experiments/EXP-003-capacity-voc-screen.md) | done | Capacity / Virtue-of-Complexity screen | depth non-monotone, width hurts, noise-limited |
| [EXP-004](../experiments/EXP-004-sophistication.md) | planned-gated | Sophistication (GLU FFN + missingness) | prepped, not run |
| [EXP-005](../experiments/EXP-005-temporal.md) | planned-gated | Temporal (macro-state GRU) | prepped, not run |
| [EXP-006](../experiments/EXP-006-monthly-refit.md) | planned-gated | Monthly refit | prepped, not run |
| [EXP-007](../experiments/EXP-007-rank-standardize-ab-screen.md) | done | Rank-standardize A/B screen | a2rank +0.92 vs base — passed the screen (later overturned) |
| [EXP-008](../experiments/EXP-008-msrr-depth-ladder.md) | planned-gated | MSRR depth ladder K in {1,2,3} | prepped, not run |
| [EXP-009](../experiments/EXP-009-rank-standardize-confirmation.md) | done | Rank-standardize confirmation | rank rejected, but gap not significant (paired t=1.09) |
| [EXP-010](../experiments/EXP-010-pipeline-audit.md) | done | Pipeline audit | 4 defects found: mean-of-ratios, missed sqrt(12), FF5 off-by-one-month, universe look-ahead |
| [EXP-011](../experiments/EXP-011-delisting-sensitivity.md) | planned-gated | Delisting sensitivity | kill-bar defined, not run |
| [EXP-012](../experiments/EXP-012-kkm-replication.md) | planned-gated | KKM (w33351) replication plan | protocol gap analysis + 4-rung ladder + power correction, not run |

## Decisions (L-NN)

| ID | Title |
|----|-------|
| [L-01](../decisions/L-01-judge-across-seed-year-distribution.md) | Judge across the seed x year distribution |
| [L-02](../decisions/L-02-msrr-scale-invariance.md) | MSRR weight magnitude carries no signal |
| [L-03](../decisions/L-03-test-dont-dismiss-on-priors.md) | Test ideas; don't dismiss them on priors |
| [L-04](../decisions/L-04-check-saved-artifacts-first.md) | Check saved artifacts before retraining |
| [L-05](../decisions/L-05-attention-scaling-is-not-the-bottleneck.md) | Attention is not the memory bottleneck at this scale |
| [L-06](../decisions/L-06-period-matching-and-the-1-survivor-verdict.md) | Period-matching and the 1-survivor verdict |
| [L-07](../decisions/L-07-ic-and-sdf-sharpe-can-diverge.md) | IC and SDF Sharpe can diverge |
| [L-08](../decisions/L-08-screens-can-be-false-positives.md) | Screens can be false positives |
| [L-09](../decisions/L-09-the-metric-was-the-bug.md) | The metric was the bug |
| [L-10](../decisions/L-10-use-an-adversarial-second-model.md) | Use an adversarial second model |
| [L-11](../decisions/L-11-power-check-the-controlled-contrast.md) | Power-check the controlled contrast, not the headline one |

## Postmortems
- [2026-07-11 — mean-of-ratios Sharpe bug](../postmortems/2026-07-11-mean-of-ratios-sharpe-bug.md)
- [2026-07-11 — attention-effect misidentification](../postmortems/2026-07-11-attention-effect-misidentification.md)
