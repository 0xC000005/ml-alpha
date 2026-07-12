#!/usr/bin/env bash
# PreToolUse(Edit|Write) on docs/log/, docs/experiments/, docs/decisions/ — statistical checklist injector.
#
# NOT a permission gate. The research log is where autonomous work is recorded; making a
# human approve every write would defeat the point of running unattended. So this
# auto-allows the write and injects the checklist into the model's context instead, as a
# system reminder. The discipline is enforced on the model, not on the user.
set -uo pipefail

f=$(jq -r '.tool_input.file_path // empty')
case "$f" in
  */docs/log/*.md|*/docs/experiments/*.md|*/docs/decisions/*.md) ;;
  *) exit 0 ;;
esac

read -r -d '' CHECKLIST <<'EOF' || true
Writing to the research log (docs/log, docs/experiments, or docs/decisions) — a number is about to become a finding. Self-check all seven
before you write (EXP-010 / L-09). Do NOT ask the user about these; just get them right.

  1. POOLED, not averaged. Quote the Sharpe of the pooled monthly return series, never
     collect_screen.py's row-mean of annual Sharpes (biased up: 1.81 vs 1.41).
  2. NOISE FLOOR. SE(annualized Sharpe) = sqrt(12)*sqrt((1+0.5*SR_m^2)/T).
     T=11 (one year) -> ~1.11.   T=88 (eight years) -> ~0.39.
  3. PAIRED t-stat for any A/B claim, on (year, seed). Below |t|=2.4 the only honest
     wording is "no detectable difference" -- never "slightly worse" or "trends toward".
  4. SEEDS are not observations. Only months add economic sample size.
  5. DATE ALIGNMENT. month_ids are the FEATURE month t; the return lands at t+1. Any
     factor/macro merge must shift first (ff5_regression.py currently does not).
  6. PROVENANCE. Cite the git SHA / config / manifest.json behind the number.
  7. TRIAL COUNT. 2012-2019 is development data now, not out-of-sample. Say how many
     specifications have been screened on it.

If a claim cannot pass these, write down what IS established and what is not. An honest
"no detectable difference" is a valid, valuable log entry.
EOF

jq -cn --arg c "$CHECKLIST" '{
  hookSpecificOutput: {
    hookEventName: "PreToolUse",
    permissionDecision: "allow",
    permissionDecisionReason: "Research-log write (docs/log, docs/experiments, or docs/decisions) — auto-allowed; statistical checklist injected.",
    additionalContext: $c
  }
}'
