# Research-log restructure — design spec

Status: approved (pending user sign-off on this file) · Author: Claude Code + Max Zhang · 2026-07-12

## 1. Problem

`RESEARCH_LOG.md` is a single 369-line file with fixed sections (Backlog, Experiment
Log, Decisions & Learnings, Status snapshot, ...). New content doesn't append to the
*file*, it inserts into the matching *section* — and two of those sections (Status
snapshot, Decisions & Learnings) are explicitly meant to be **rewritten in place** as
understanding changes. That's how the 1.81→1.41 pooling correction happened: by editing
old prose, not by appending new prose. The result is a file that cannot be trusted to
show "every decision in order" just by reading top to bottom, and where a stale number
(the 1.81 headline) can sit uncorrected in one section while a correction lives in
another.

## 2. Reference model: implied-vol-backfill

`~/Documents/implied-vol-backfill` solves this with a convention a prior Claude Code
session bootstrapped in its first commit (`1c27ec7`) — **not** a superpowers-skill
default (verified: the installed superpowers plugin has no reference to `docs/log`,
`docs/decisions`, or `docs/postmortems` anywhere; it only defines `docs/superpowers/`
for its own spec/plan artifacts). The pattern:

- `docs/log/YYYY-MM.md` — append-only narrative diary, newest entry on top, one file per
  month. Never edited after the fact; a correction is a new entry.
- `docs/decisions/NNNN-slug.md` — one immutable file per load-bearing decision (ADR
  style). A correction is a **new** numbered file that supersedes the old one; the old
  file gets a one-line "superseded by NNNN" banner but its body is untouched.
- `docs/postmortems/YYYY-MM-DD-slug.md` — one file per costly surprise, written after
  the fact (Summary/Timeline/Contributing factors/Lessons/Action items).
- `docs/log/INDEX.md` — a small, purely mechanical index: month links + a decision
  table. This is the file you actually open to review everything in order. It only ever
  grows by appending a row.
- `runs/<timestamp>-<slug>-<hash>/` — dated artifact folders (ml-alpha's analog is
  already `output/exp/**` + `experiments/manifest.py`; no change needed there).
- `CLAUDE.md` itself carries one small **"Current state"** section explicitly labeled
  as mutable state ("overwrite as things change... not history"), plus a "Where things
  live" pointer table and a session-start/session-end protocol.

The key structural idea: separate **state** (allowed to be rewritten — small, lives in
CLAUDE.md), **history** (never rewritten — diary + decisions + postmortems), and
**static reference** (edited only when a fact changes, not as a periodic snapshot).
ml-alpha's current file conflates all three.

## 3. Layout for ml-alpha

```
docs/
  log/
    INDEX.md                          — months list + EXP-NNN table + L-NN table + postmortems list
    2026-07.md                        — current month, narrative, newest entry on top, append-only
    archive/
      RESEARCH_LOG-2026-06-02-to-2026-07-12.md   — frozen, read-only copy of the pre-migration file
  experiments/
    EXP-NNN-slug.md                   — one file per experiment (existing fields: hypothesis, git SHA,
                                         config/command, data, metrics, verdict — no format change)
  decisions/
    L-NN-slug.md                      — one file per distilled decision (Status/Context/Decision/
                                         Evidence/Consequences); corrections supersede, never edit
  postmortems/
    README.md                         — convention (ported from implied-vol-backfill)
    YYYY-MM-DD-slug.md                — one file per costly surprise
  BACKLOG.md                          — B-NN table + the 2026-06-03 Enhancement Roadmap as its
                                         prioritization rationale. Living state — mutation in place is
                                         fine here, it's a todo list, not history.
  REFERENCE.md                        — compute discipline, environment/data card, conventions,
                                         glossary. Edited only when a fact changes.
```

`CLAUDE.md` gains a short **"Current state"** section (replacing "Status snapshot"),
explicitly marked as mutable, plus a "Where things live" table and a session-start/
session-end protocol, mirroring implied-vol-backfill's `CLAUDE.md` almost exactly.

## 4. Content map (current file → destination)

| Current section (lines) | Destination |
|---|---|
| Title/intro/maintainer (1–20) | Folded into `docs/log/INDEX.md` header + `CLAUDE.md` |
| Contents ToC (21) | Dropped — INDEX.md replaces this function |
| Compute discipline (36–47) | `docs/REFERENCE.md` |
| Status snapshot (48–58) | Retired as prose; facts already live in the L-NN files below. Replaced by `CLAUDE.md`'s new "Current state" |
| Environment, Data & Reproducibility (59–75) | `docs/REFERENCE.md` |
| Backlog table (76–100) | `docs/BACKLOG.md` |
| Experiment summary table (101–115) | EXP-NNN table in `docs/log/INDEX.md` |
| EXP-001, 002, 003, 007, 009, 010, 011, 012 (117–292) | One file each: `docs/experiments/EXP-NNN-slug.md`, verbatim |
| Enhancement Roadmap (293–302) | Folded into `docs/BACKLOG.md` as prioritization rationale |
| Planned experiments prose — EXP-004/005/006/008 (305–319) | Given real stub files for consistency: `docs/experiments/EXP-004-sophistication.md` etc., status PLANNED/GATED |
| L-01 … L-11 + Coda (321–346) | One file each: `docs/decisions/L-NN-slug.md`, verbatim |
| Conventions (349–358) | `docs/REFERENCE.md` |
| Glossary (359–369) | `docs/REFERENCE.md` |

New: `docs/postmortems/2026-07-11-mean-of-ratios-sharpe-bug.md` and
`docs/postmortems/2026-07-11-attention-effect-misidentification.md`, written fresh
(not migrated — these events are already described inside L-09/L-11/L-10 but don't
have their own postmortem-shaped writeup).

## 5. Migration method (correctness is the point)

These are exactly the numbers `stats-gatekeeper` exists to protect — a hand-retyped
migration risks silently changing a digit in a 369-line file full of dense tables and
Sharpe ratios. So the split is **mechanical, not transcribed**:

1. A short Python script parses `RESEARCH_LOG.md` by its existing heading/bullet
   boundaries (`### EXP-NNN ...` through the next `##`/`###`; `- **L-NN (...):**`
   through its trailing sub-bullets) and writes each block **verbatim** into its new
   file, adding only a thin hand-authored frontmatter (status, cross-links).
2. Verification pass before anything is archived: every EXP-NNN / L-NN identifier and
   every numeric token in the original file must appear in the concatenation of the new
   files. Spot-check the densest entries (L-09, L-11, EXP-012, EXP-010) by eye.
3. Only after verification passes: `git mv RESEARCH_LOG.md
   docs/log/archive/RESEARCH_LOG-2026-06-02-to-2026-07-12.md`, stamped "ARCHIVED,
   read-only" at the top. This is a safety net, not part of the active system — the new
   files are what gets read and written going forward.

## 6. Automation changes

- `.claude/hooks/research-log-checklist.sh` currently matches the literal filename
  `RESEARCH_LOG.md`. Repoint to match `docs/log/**/*.md`, `docs/experiments/EXP-*.md`,
  `docs/decisions/L-*.md` (the three places "a number becomes a finding") — not
  `BACKLOG.md` or `REFERENCE.md`.
- New light-touch hook: editing an *existing* `docs/experiments/EXP-*.md` or
  `docs/decisions/L-*.md` still auto-allows (never blocks autonomous work) but injects
  a reminder — "you're editing a closed record; if this is a substantive correction,
  prefer a new superseding file." Same non-blocking philosophy as the existing
  checklist hook.
- `CLAUDE.md`'s "How research work is done here" section and the `stats-gatekeeper`
  agent description both currently say "before it enters RESEARCH_LOG.md" — repoint to
  the new locations.

## 7. Non-goals

- No change to `output/exp/**` or `experiments/manifest.py` — that system already plays
  the role of implied-vol-backfill's `runs/`.
- No renumbering of EXP-NNN / L-NN / B-NN identifiers — already cross-referenced in
  hooks, agent descriptions, and memory files; renumbering is pure churn.
- Not migrating `output/` artifacts or `experiments/` code — this spec is docs-only.
