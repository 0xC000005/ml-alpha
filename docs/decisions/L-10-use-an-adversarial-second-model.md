---
id: L-10
status: accepted
supersedes: null
---

# L-10 — Use an adversarial second model

- **L-10 (use an adversarial second model — Codex earned its keep):** D-1, D-2 and D-3 (EXP-010) were found by an independent **Codex gpt-5.6-sol** review reading the same code; a Claude-only multi-agent audit had just walked past all three. Different model, different blind spots. Every claim it made was then re-verified by hand — it was also confidently wrong about some things, so **verify, don't relay**. Use the `verify-with-codex` skill (calls `mcp__codex__codex` live) before spending GPU time and before any number enters this file.
