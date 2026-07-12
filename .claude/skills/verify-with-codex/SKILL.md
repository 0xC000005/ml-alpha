---
name: verify-with-codex
description: Get an independent second opinion from OpenAI Codex (gpt-5.6-sol) by calling the Codex MCP server directly. Use before committing GPU time to an experiment, before writing a result into RESEARCH_LOG.md, when a number looks too good, or when the user says "verify this", "second opinion", "sanity check", "check with codex". Project-scoped: overrides the paste-based global skill.
---

# Verify with Codex (live MCP)

Codex reads this repo with its own eyes and reasons independently. It has already caught
real defects here that a Claude-only review missed — the mean-of-annual-Sharpes bug, the
√12 annualization error, and the off-by-one month in the factor regression. Treat it as a
genuine adversary, not a rubber stamp.

The global `~/.claude/skills/verify-with-codex` skill generates a prompt for you to
paste into a separate terminal. **In this repo, don't do that** — call the MCP server and
get the answer in-session.

## How to call it

1. Load the tool (schemas are deferred):

   ```
   ToolSearch: select:mcp__codex__codex
   ```

2. Call `mcp__codex__codex` with:

   | param | value |
   |---|---|
   | `model` | `gpt-5.6-sol` (frontier). `gpt-5.6-terra` for routine checks, `gpt-5.6-luna` for cheap ones. There is **no** `-ultra` suffix — it 400s. |
   | `sandbox` | `read-only` (it should verify, not edit) |
   | `cwd` | `/home/max/Documents/ml-alpha` |
   | `config` | `{"model_reasoning_effort": "high"}` |
   | `prompt` | the verification request (below) |

3. Continue a thread with `mcp__codex__codex-reply` + `threadId` rather than re-sending
   context.

## Writing the prompt

Codex has filesystem access — **point it at files, don't paste them**. That is the whole
token saving. A good request has five parts:

1. **Role + repo.** "You are a senior quant-finance researcher reviewing /home/max/Documents/ml-alpha."
2. **The claim to attack**, stated precisely, with the file:line that produced it.
3. **Adversarial framing.** "Do not agree with me. Refute this if you can."
4. **Where to look.** Name the specific files — `experiments/*.py`, `RESEARCH_LOG.md`,
   the frozen `train_*.py`. Tell it to recompute from artifacts under `output/exp/`.
5. **What would change your mind.** Force a falsifiable answer.

Always include this repo's standing context, because it is what makes the review sharp:

> Known failure mode: EXP-007 (rank-standardization) passed as a 5-seed screen; EXP-009
> (10 seeds, 8 years) reversed it. The OOS Sharpe noise floor is ≈±0.4 over 88 months and
> ≈±1.1 over a single test year. Seeds are not economic observations. The headline Sharpe
> is a mean of annual Sharpes, which is biased upward.

## When to use it

- **Before** any `sbatch` — a screen that cannot resolve its own effect size is wasted GPU time.
- **Before** a number enters `RESEARCH_LOG.md` — pair with the `stats-gatekeeper` agent.
- Whenever a result improves on the baseline by less than ~0.5 Sharpe. That is inside the
  noise floor, and the prior should be "this is nothing."

## Reporting back

Do not launder Codex's answer into your own voice. Say what Codex claimed, then **verify
its concrete claims yourself** against the code before repeating them as fact — it is
confident, specific, and occasionally wrong. Where you and Codex disagree, show both and
say which you believe and why.
