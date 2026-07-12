#!/usr/bin/env bash
# PreToolUse(Edit) on an EXISTING docs/experiments/EXP-*.md or docs/decisions/L-*.md —
# these are meant to be immutable once their verdict/status is set. Not a permission
# gate (never blocks autonomous work) -- just a reminder, same philosophy as
# research-log-checklist.sh.
set -uo pipefail

f=$(jq -r '.tool_input.file_path // empty')
case "$f" in
  */docs/experiments/EXP-*.md|*/docs/decisions/L-*.md) ;;
  *) exit 0 ;;
esac

jq -cn '{
  hookSpecificOutput: {
    hookEventName: "PreToolUse",
    permissionDecision: "allow",
    permissionDecisionReason: "Editing a closed record — auto-allowed.",
    additionalContext: "This file is meant to be immutable once its verdict/status is set (docs/log/INDEX.md convention). If this is a substantive correction rather than a typo/status-field fix, prefer creating a new superseding file (docs/decisions/L-NN with supersedes: <old id>) instead of editing this one in place."
  }
}'
