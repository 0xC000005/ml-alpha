# Postmortem — the mean-of-ratios Sharpe bug (headline 1.81 vs pooled 1.41)

Date: 2026-07-11 (discovered) · Related: L-09, EXP-010

## Summary / impact
`experiments/collect_screen.py` computed the project's headline MSRR Sharpe as a
row-mean of per-year annualized Sharpe ratios. That statistic is a mean of ratios, not
the Sharpe of the pooled return series, and is biased upward. The reported "1.81" headline
that appeared throughout the log and in prior status updates should have read pooled
**1.41** — a 0.40 Sharpe overstatement carried for weeks before being caught.

## Timeline
- Multiple sessions prior to 2026-07-11: 1.81 quoted as the honest MSRR baseline in
  status updates and comparisons.
- 2026-07-11: EXP-010 pipeline audit (triggered by a scheduled Codex cross-check, see
  L-10) recomputes the statistic directly from pooled monthly returns instead of
  reading the summary CSV; finds the discrepancy.

## Contributing factors
- `collect_screen.py:36-41` averages a column of annual Sharpes without documenting
  that this is a biased estimator of the pooled Sharpe.
- No prior step in the pipeline computed or displayed the pooled monthly return series
  directly, so there was nothing to cross-check the mean-of-annuals against.
- The bias is not obvious from reading the code — averaging a metric that is itself
  already a ratio "looks like" a reasonable summary statistic.

## Lessons learned
- Never average annual Sharpe ratios. Always compute the Sharpe of the pooled monthly
  return series (see L-09 item 1).
- A statistic with no independent cross-check can drift for an arbitrarily long time.

## Action items
- [x] `stats-gatekeeper` agent's error catalogue item 1 now names this bug explicitly
  and requires reconstructing the number from artifacts rather than accepting a quoted
  figure.
- [x] `.claude/hooks/research-log-checklist.sh` injects "POOLED, not averaged" as
  checklist item 1 on every research-log write.
- [ ] Consider adding an automated check in `collect_screen.py` itself that prints both
  the mean-of-annuals and the pooled figure side by side, so the gap is visible at
  computation time rather than only on audit.
