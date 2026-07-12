#!/usr/bin/env bash
# PreToolUse(Bash) — cluster submission gate.
#
# The cluster skill's binding rule #1 is "never submit sbatch/srun without an explicit
# go-ahead". That was prose an LLM could forget under context pressure. Now it is a hook.
set -uo pipefail

cmd=$(jq -r '.tool_input.command // empty')
printf '%s' "$cmd" | grep -Eq '(^|[;&|[:space:]])(sbatch|srun)([[:space:]]|$)' || exit 0

# Attaching to an ALREADY-RUNNING job allocates nothing — it is monitoring, not submission.
# On Trillium `srun --jobid=$J --overlap nvidia-smi` is the *only* way to see GPU
# utilization (ssh into a compute node is blocked by pam_slurm_adopt), so the monitoring
# path must not be gated or cluster-monitor stalls on every poll.
if printf '%s' "$cmd" | grep -Eq -- '--overlap|--jobid='; then
  printf '%s' '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"allow","permissionDecisionReason":"srun attach to an existing job (monitoring) — allocates no new resources."}}'
  exit 0
fi

read -r -d '' REASON <<'EOF' || true
Cluster job submission. This always requires an explicit human go-ahead (cluster skill,
binding rule #1). Before approving, confirm:

  - Has the screen been power-checked? An arm that cannot resolve its own effect size
    against the +/-0.4 (8yr) or +/-1.1 (1yr) noise floor is wasted GPU time.
  - Does the control arm run at the SAME seed count, in the same conditions? Its noise is
    the thing being tested against (learning L-08).
  - >= 10 seeds? A 5-seed screen produced this repo's known false positive.
  - Is the kill-bar written down BEFORE the run, not chosen after seeing the numbers?
EOF

jq -cn --arg r "$REASON" '{
  hookSpecificOutput: {
    hookEventName: "PreToolUse",
    permissionDecision: "ask",
    permissionDecisionReason: $r
  }
}'
