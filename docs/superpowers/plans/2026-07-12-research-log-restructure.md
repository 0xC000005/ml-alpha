# Research-Log Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split `RESEARCH_LOG.md` (369 lines, single file) into the layered
`docs/log/` + `docs/experiments/` + `docs/decisions/` + `docs/postmortems/` +
`docs/BACKLOG.md` + `docs/REFERENCE.md` structure from
`docs/superpowers/specs/2026-07-12-research-log-restructure-design.md`, fully
migrating every existing EXP/L entry, then archive the original.

**Architecture:** Every extraction is a `sed -n 'START,ENDp'` byte-for-byte copy of an
already-identified line range (no retyping, no summarization) into a new file with a
small hand-authored frontmatter/heading. A final coverage pass proves every substantive
line of the original landed somewhere or was deliberately not migrated (and says why).

**Tech Stack:** Bash (`sed`, `diff`, `git mv`), Markdown. No new dependencies.

## Global Constraints

- Every extraction must be **byte-identical** to its source range (verified by `diff`)
  — never retype numbers or prose from RESEARCH_LOG.md.
- Do not touch `train_nn.py`, `train_transformer.py`, `train_transformer_msrr.py` (frozen,
  hook-enforced).
- Do not run `git commit` — this project's standing convention is nothing is committed
  without an explicit user ask (per this session; skip the "Commit" step template
  normally used in this skill).
- Do not run `sbatch`/`srun` — not applicable to this plan, no cluster work involved.
- EXP-NNN / L-NN / B-NN identifiers are never renumbered.

---

## Migration manifest (source line ranges → destination)

All ranges are 1-indexed, inclusive, against `RESEARCH_LOG.md` as it exists at the start
of this plan (369 lines). Re-verify line numbers with
`grep -n "^#\{2,4\} " RESEARCH_LOG.md` before extracting if the file has changed.

| Range | Content | Destination |
|---|---|---|
| 1–35 | Title, intro, how-to-use, Contents ToC | NOT verbatim-migrated — paraphrased into `docs/log/INDEX.md` header + `CLAUDE.md` (Task 1) |
| 36–47 | Compute discipline | `docs/REFERENCE.md` §1 (Task 2) |
| 48–58 | Status snapshot | NOT verbatim-migrated — retired; content is stale by design (see spec §4). Durable facts (harness design-doc pointer) fold into `docs/REFERENCE.md`; current facts get freshly authored into `CLAUDE.md` "Current state" (Task 13) |
| 59–75 | Environment, Data & Reproducibility | `docs/REFERENCE.md` §2 (Task 2) |
| 76–100 | Backlog table + prose | `docs/BACKLOG.md` §1 (Task 3) |
| 101–115 | Experiment summary table | `docs/log/INDEX.md` EXP table, split to one row per EXP-NNN (Task 9) |
| 117–118 | "## Experiment Log" heading | structural, dropped |
| 119–126 | EXP-001 | `docs/experiments/EXP-001-reproduce-readme-transformer.md` (Task 4) |
| 127–134 | EXP-002 | `docs/experiments/EXP-002-msrr-ensemble-normalization-ab.md` (Task 4) |
| 135–142 | EXP-003 | `docs/experiments/EXP-003-capacity-voc-screen.md` (Task 4) |
| 143–150 | EXP-007 | `docs/experiments/EXP-007-rank-standardize-ab-screen.md` (Task 5) |
| 151–160 | EXP-009 | `docs/experiments/EXP-009-rank-standardize-confirmation.md` (Task 5) |
| 161–172 | EXP-010 | `docs/experiments/EXP-010-pipeline-audit.md` (Task 5) |
| 173–180 | EXP-011 | `docs/experiments/EXP-011-delisting-sensitivity.md` (Task 6) |
| 181–292 | EXP-012 | `docs/experiments/EXP-012-kkm-replication.md` (Task 6) |
| 293–304 | Enhancement Roadmap | `docs/BACKLOG.md` §2 (Task 3) |
| 305–320 | Planned experiments prose (EXP-004/5/6/8) | Split into 4 stub files (Task 7) |
| 321–322 | "## Decisions & Learnings" heading | structural, dropped |
| 323 | L-01 | `docs/decisions/L-01-judge-across-seed-year-distribution.md` (Task 8) |
| 324 | L-02 | `docs/decisions/L-02-msrr-scale-invariance.md` (Task 8) |
| 325 | L-03 | `docs/decisions/L-03-test-dont-dismiss-on-priors.md` (Task 8) |
| 326 | L-04 | `docs/decisions/L-04-check-saved-artifacts-first.md` (Task 8) |
| 327 | L-05 | `docs/decisions/L-05-attention-scaling-is-not-the-bottleneck.md` (Task 8) |
| 328 | L-06 | `docs/decisions/L-06-period-matching-and-the-1-survivor-verdict.md` (Task 8) |
| 329 | L-07 | `docs/decisions/L-07-ic-and-sdf-sharpe-can-diverge.md` (Task 8) |
| 330 | L-08 | `docs/decisions/L-08-screens-can-be-false-positives.md` (Task 8) |
| 331 | L-09 | `docs/decisions/L-09-the-metric-was-the-bug.md` (Task 8) |
| 332–343 | L-11 + Coda | `docs/decisions/L-11-power-check-the-controlled-contrast.md` (Task 8) |
| 344 | blank | dropped |
| 345 | L-10 | `docs/decisions/L-10-use-an-adversarial-second-model.md` (Task 8) |
| 346–348 | blank / `---` | dropped |
| 349–358 | Conventions | `docs/REFERENCE.md` §3 (Task 2) |
| 359–369 | Glossary | `docs/REFERENCE.md` §4 (Task 2) |

Note L-10 physically follows L-11 in the source file (out-of-numeric-order) — extract
each by its own line number regardless of file position; the new INDEX.md (Task 9)
restores numeric order.

---

## Task 1: Scaffold directories, postmortems README, changeover diary entry

**Files:**
- Create: `docs/log/2026-07.md`
- Create: `docs/log/archive/.gitkeep` (placeholder until Task 11 populates it)
- Create: `docs/postmortems/README.md`

**Interfaces:**
- Produces: the directory tree `docs/log/`, `docs/log/archive/`, `docs/experiments/`,
  `docs/decisions/`, `docs/postmortems/` that every later task writes into.

- [ ] **Step 1: Create the directories**

```bash
mkdir -p docs/log/archive docs/experiments docs/decisions docs/postmortems
```

- [ ] **Step 2: Write the postmortems README (ported convention)**

Write `docs/postmortems/README.md`:

```markdown
# postmortems/

Blameless mini-postmortems, written **after a costly surprise** — a statistical bug that
shipped a wrong headline number, a false-positive screen that got confirmed too late, a
misidentified effect, or a result that didn't reproduce.

One file per incident: `YYYY-MM-DD-slug.md`, with:
- **Summary / impact** — what went wrong and what it affected.
- **Timeline** — timestamped sequence of what happened.
- **Contributing factors** — factual, evidence-backed (not blame).
- **Lessons learned** — what we now know.
- **Action items** — concrete follow-up (e.g. promote a check into a hook, add a rule to
  `stats-gatekeeper`).

The goal is to turn each surprise into a durable lesson instead of a forgotten scar.
```

- [ ] **Step 3: Write the changeover diary entry**

Write `docs/log/2026-07.md`:

```markdown
# Research log — 2026-07

Newest entry on top.

## 2026-07-12 — Migrated to the layered docs/ research-log system

**What.** Replaced the single `RESEARCH_LOG.md` with `docs/log/` (this file — narrative,
append-only, newest entry on top) + `docs/experiments/EXP-NNN-*.md` (one file per
experiment) + `docs/decisions/L-NN-*.md` (one file per distilled finding, ADR-style,
supersede-don't-edit) + `docs/postmortems/` (costly-surprise writeups) +
`docs/BACKLOG.md` (living B-NN backlog) + `docs/REFERENCE.md` (static reference).

**Why.** The old file mixed history (Experiment Log, Decisions & Learnings) with living
state (Status snapshot) and static reference (Compute discipline, Environment card,
Conventions, Glossary) in one file with fixed sections. New content had to be inserted
mid-document rather than appended, and "Decisions & Learnings" / "Status snapshot" were
explicitly meant to be rewritten in place — which is how a stale number (the 1.81
mean-of-ratios headline) could sit uncorrected while the correction lived elsewhere.
Modeled on `~/Documents/implied-vol-backfill`'s `docs/log` + `docs/decisions` +
`docs/postmortems` split (project-specific convention there, not a superpowers-skill
default — verified against the installed plugin).

**What moved.** All of EXP-001…012 (plus retroactive stub files for EXP-004/5/6/8, which
previously only existed as prose) and L-01…L-11 were extracted byte-for-byte (via `sed`
line ranges, never retyped) into their own files; see
`docs/superpowers/specs/2026-07-12-research-log-restructure-design.md` for the full
content map and `docs/superpowers/plans/2026-07-12-research-log-restructure.md` for the
line-by-line manifest. The original file is preserved read-only at
`docs/log/archive/RESEARCH_LOG-2026-06-02-to-2026-07-12.md`.

**Decision.** Going forward: diary entries append to the top of the current month's
`docs/log/YYYY-MM.md`; a new EXP gets its stub file created *before* it runs; a
correction to an L-NN is a new L-NN that supersedes it, never an edit to the old one.

**Next.** EXP-013 onward uses this structure.
```

- [ ] **Step 4: Verify**

```bash
test -d docs/log/archive && test -d docs/experiments && test -d docs/decisions && test -d docs/postmortems && echo OK
```
Expected: `OK`

(No commit — see Global Constraints.)

---

## Task 2: Extract `docs/REFERENCE.md`

**Files:**
- Create: `docs/REFERENCE.md`

**Interfaces:**
- Consumes: `RESEARCH_LOG.md` lines 36–47, 59–75, 349–358, 359–369 (see manifest).

- [ ] **Step 1: Extract the four sections verbatim**

```bash
{
  echo "# ml-alpha — Reference"
  echo
  echo "Static facts: compute discipline, environment/data card, conventions, glossary."
  echo "Edited only when a fact changes, never as a periodic snapshot. History lives in"
  echo "\`docs/log/\`, \`docs/experiments/\`, \`docs/decisions/\`."
  echo
  sed -n '36,47p' RESEARCH_LOG.md
  echo
  sed -n '59,75p' RESEARCH_LOG.md
  echo
  sed -n '349,358p' RESEARCH_LOG.md
  echo
  sed -n '359,369p' RESEARCH_LOG.md
} > docs/REFERENCE.md
```

- [ ] **Step 2: Verify byte-fidelity of each extracted block**

```bash
diff <(sed -n '36,47p' RESEARCH_LOG.md) <(sed -n '36,47p' RESEARCH_LOG.md)  # sanity: diff tool works
for range in "36,47" "59,75" "349,358" "359,369"; do
  start=${range%,*}; end=${range#*,}
  sed -n "${range}p" RESEARCH_LOG.md > /tmp/ref_src.txt
  grep -F -f /tmp/ref_src.txt docs/REFERENCE.md > /dev/null && echo "OK $range" || echo "MISSING $range"
done
```
Expected: `OK 36,47`, `OK 59,75`, `OK 349,358`, `OK 359,369`

- [ ] **Step 3: Read the result and confirm headings render sensibly**

Read `docs/REFERENCE.md` and confirm the four sections (Compute discipline, Environment/
Data & Reproducibility, Conventions, Glossary) are present with their original `##`
headings intact (sed captured the heading lines themselves, so no re-heading needed).

---

## Task 3: Extract `docs/BACKLOG.md`

**Files:**
- Create: `docs/BACKLOG.md`

**Interfaces:**
- Consumes: `RESEARCH_LOG.md` lines 76–100 (Backlog), 293–304 (Enhancement Roadmap).

- [ ] **Step 1: Extract both sections verbatim, roadmap first (it's the rationale for the table that follows)**

```bash
{
  echo "# ml-alpha — Backlog"
  echo
  echo "Living state — prioritized hypotheses (B-NN). Unlike \`docs/log/\` and"
  echo "\`docs/decisions/\`, this file is meant to be mutated in place (status checkboxes),"
  echo "it is a todo list, not history."
  echo
  sed -n '293,304p' RESEARCH_LOG.md
  echo
  sed -n '76,100p' RESEARCH_LOG.md
} > docs/BACKLOG.md
```

- [ ] **Step 2: Verify**

```bash
grep -c "B-0" docs/BACKLOG.md   # expect >= 6 (B-00..B-09, B-11)
grep -c "Roadmap" docs/BACKLOG.md  # expect >= 1
```
Expected: both counts > 0

---

## Task 4: Extract EXP-001, EXP-002, EXP-003

**Files:**
- Create: `docs/experiments/EXP-001-reproduce-readme-transformer.md`
- Create: `docs/experiments/EXP-002-msrr-ensemble-normalization-ab.md`
- Create: `docs/experiments/EXP-003-capacity-voc-screen.md`

**Interfaces:**
- Consumes: `RESEARCH_LOG.md` lines 119–126, 127–134, 135–142.
- Produces: the EXP file frontmatter pattern reused by Tasks 5–7:
  ```
  ---
  id: EXP-NNN
  status: done | planned-gated
  ---

  # EXP-NNN — <title from the original ### heading, verbatim>

  <verbatim body>
  ```

- [ ] **Step 1: Extract each, converting the `###` heading to `#` and prepending frontmatter**

```bash
extract_exp() {
  local start=$1 end=$2 id=$3 status=$4 file=$5
  { echo "---"; echo "id: $id"; echo "status: $status"; echo "---"; echo
    sed -n "${start},${end}p" RESEARCH_LOG.md | sed '1s/^### /# /'
  } > "$file"
}
extract_exp 119 126 EXP-001 done docs/experiments/EXP-001-reproduce-readme-transformer.md
extract_exp 127 134 EXP-002 done docs/experiments/EXP-002-msrr-ensemble-normalization-ab.md
extract_exp 135 142 EXP-003 done docs/experiments/EXP-003-capacity-voc-screen.md
```

- [ ] **Step 2: Verify byte-fidelity (body, excluding the added frontmatter and heading-level change)**

```bash
for spec in "119 126 EXP-001-reproduce-readme-transformer" "127 134 EXP-002-msrr-ensemble-normalization-ab" "135 142 EXP-003-capacity-voc-screen"; do
  set -- $spec
  diff <(sed -n "$1,$2p" RESEARCH_LOG.md | sed '1s/^### /# /') <(tail -n +6 "docs/experiments/$3.md") && echo "OK $3"
done
```
Expected: `OK EXP-001-reproduce-readme-transformer`, `OK EXP-002-msrr-ensemble-normalization-ab`, `OK EXP-003-capacity-voc-screen` (no diff output before each)

---

## Task 5: Extract EXP-007, EXP-009, EXP-010

**Files:**
- Create: `docs/experiments/EXP-007-rank-standardize-ab-screen.md`
- Create: `docs/experiments/EXP-009-rank-standardize-confirmation.md`
- Create: `docs/experiments/EXP-010-pipeline-audit.md`

**Interfaces:**
- Consumes: `RESEARCH_LOG.md` lines 143–150, 151–160, 161–172.
- Consumes: `extract_exp()` shell function defined in Task 4 (redefine if running this
  task in a fresh shell/session).

- [ ] **Step 1: Extract**

```bash
extract_exp 143 150 EXP-007 done docs/experiments/EXP-007-rank-standardize-ab-screen.md
extract_exp 151 160 EXP-009 done docs/experiments/EXP-009-rank-standardize-confirmation.md
extract_exp 161 172 EXP-010 done docs/experiments/EXP-010-pipeline-audit.md
```

- [ ] **Step 2: Verify**

```bash
for spec in "143 150 EXP-007-rank-standardize-ab-screen" "151 160 EXP-009-rank-standardize-confirmation" "161 172 EXP-010-pipeline-audit"; do
  set -- $spec
  diff <(sed -n "$1,$2p" RESEARCH_LOG.md | sed '1s/^### /# /') <(tail -n +6 "docs/experiments/$3.md") && echo "OK $3"
done
```
Expected: three `OK` lines, no diff output

---

## Task 6: Extract EXP-011, EXP-012 (planned/gated)

**Files:**
- Create: `docs/experiments/EXP-011-delisting-sensitivity.md`
- Create: `docs/experiments/EXP-012-kkm-replication.md`

**Interfaces:**
- Consumes: `RESEARCH_LOG.md` lines 173–180, 181–292.

- [ ] **Step 1: Extract (status `planned-gated`, not `done`)**

```bash
extract_exp 173 180 EXP-011 planned-gated docs/experiments/EXP-011-delisting-sensitivity.md
extract_exp 181 292 EXP-012 planned-gated docs/experiments/EXP-012-kkm-replication.md
```

- [ ] **Step 2: Verify**

```bash
diff <(sed -n '173,180p' RESEARCH_LOG.md | sed '1s/^### /# /') <(tail -n +6 docs/experiments/EXP-011-delisting-sensitivity.md) && echo "OK EXP-011"
diff <(sed -n '181,292p' RESEARCH_LOG.md | sed '1s/^### /# /') <(tail -n +6 docs/experiments/EXP-012-kkm-replication.md) && echo "OK EXP-012"
wc -l docs/experiments/EXP-012-kkm-replication.md   # sanity: should be ~118 lines (112 body + 6 frontmatter)
```
Expected: `OK EXP-011`, `OK EXP-012`, line count ~118

---

## Task 7: Create retroactive EXP-004/005/006/008 stub files

**Files:**
- Create: `docs/experiments/EXP-004-sophistication.md`
- Create: `docs/experiments/EXP-005-temporal.md`
- Create: `docs/experiments/EXP-006-monthly-refit.md`
- Create: `docs/experiments/EXP-008-msrr-depth-ladder.md`

**Interfaces:**
- Consumes: `RESEARCH_LOG.md` lines 305–320 (Planned experiments prose — each of these
  four is one bullet inside that range, previously never given its own file).

Unlike Tasks 4–6, these four are NOT one contiguous range each — they're four bullets
sharing a 16-line block. Extract each bullet by its own start marker.

- [ ] **Step 1: Find each bullet's exact line range**

```bash
grep -n "EXP-004\|EXP-005\|EXP-006\|EXP-008" RESEARCH_LOG.md
```
Use this output to confirm the bullet start lines within 305–320 before extracting (the
manifest above assumes EXP-004 starts the block; re-grep if the file has since changed,
since this task runs after Tasks 1-6 which don't touch these lines, so line numbers
should be stable — but always re-verify before a sed extraction).

- [ ] **Step 2: Extract each bullet verbatim (adjust exact line numbers to match Step 1's grep output)**

```bash
extract_exp_stub() {
  local start=$1 end=$2 id=$3 title=$4 file=$5
  { echo "---"; echo "id: $id"; echo "status: planned-gated"; echo "---"; echo
    echo "# $id — $title"; echo
    sed -n "${start},${end}p" RESEARCH_LOG.md
  } > "$file"
}
# Line numbers below are from the manifest (Task-0 grep); confirm against Step 1 output first.
extract_exp_stub 308 308 EXP-004 "Sophistication (GLU FFN + missingness indicators)" docs/experiments/EXP-004-sophistication.md
extract_exp_stub 310 310 EXP-005 "Temporal (macro-state GRU)" docs/experiments/EXP-005-temporal.md
extract_exp_stub 311 311 EXP-006 "Monthly refit" docs/experiments/EXP-006-monthly-refit.md
extract_exp_stub 318 318 EXP-008 "MSRR depth ladder K in {1,2,3}" docs/experiments/EXP-008-msrr-depth-ladder.md
```

Note: lines 312–317 are EXP-007's prep-detail bullet, re-describing an experiment that
already has its own file from Task 5 (extracted from its `### EXP-007` block at lines
143–150). Do not create a second EXP-007 file — instead, append lines 312–317 to the
*existing* `docs/experiments/EXP-007-rank-standardize-ab-screen.md` as a "## Prep notes"
section, since it's genuinely additional detail about that same experiment that the
Task-5 extraction didn't capture:

```bash
{ echo; echo "## Prep notes (from the former Planned-experiments block)"; echo
  sed -n '312,317p' RESEARCH_LOG.md
} >> docs/experiments/EXP-007-rank-standardize-ab-screen.md
```

- [ ] **Step 3: Verify each file has real content (not empty) and references the correct EXP-NNN**

```bash
for f in docs/experiments/EXP-004-sophistication.md docs/experiments/EXP-005-temporal.md docs/experiments/EXP-006-monthly-refit.md docs/experiments/EXP-008-msrr-depth-ladder.md; do
  test -s "$f" && grep -q "$(basename "$f" | grep -o 'EXP-[0-9]*')" "$f" && echo "OK $f"
done
```
Expected: four `OK` lines

- [ ] **Step 4: Manually read the 305–320 block once more and confirm nothing else in it (the shared preamble sentence, the "Total cheap-screen budget" closing line) was silently dropped**

Read lines 305–307 and 319 of `RESEARCH_LOG.md` (the shared preamble and closing budget
line, not owned by any single EXP-NNN) and fold them into `docs/BACKLOG.md` §2
(Enhancement Roadmap section from Task 3) as a short "Planned-experiments budget" note,
since they're backlog-level context, not any one experiment's content.

---

## Task 8: Extract L-01 through L-11

**Files:**
- Create: `docs/decisions/L-01-judge-across-seed-year-distribution.md`
- Create: `docs/decisions/L-02-msrr-scale-invariance.md`
- Create: `docs/decisions/L-03-test-dont-dismiss-on-priors.md`
- Create: `docs/decisions/L-04-check-saved-artifacts-first.md`
- Create: `docs/decisions/L-05-attention-scaling-is-not-the-bottleneck.md`
- Create: `docs/decisions/L-06-period-matching-and-the-1-survivor-verdict.md`
- Create: `docs/decisions/L-07-ic-and-sdf-sharpe-can-diverge.md`
- Create: `docs/decisions/L-08-screens-can-be-false-positives.md`
- Create: `docs/decisions/L-09-the-metric-was-the-bug.md`
- Create: `docs/decisions/L-10-use-an-adversarial-second-model.md`
- Create: `docs/decisions/L-11-power-check-the-controlled-contrast.md`

**Interfaces:**
- Consumes: `RESEARCH_LOG.md` lines 323, 324, 325, 326, 327, 328, 329, 330, 331,
  332–343 (L-11 + Coda), 345 (L-10).
- Produces: the L-NN file frontmatter pattern:
  ```
  ---
  id: L-NN
  status: accepted | superseded
  supersedes: null | L-MM
  ---

  # L-NN — <short title>

  <verbatim bullet text, "- **L-NN (...):**" prefix kept as-is>
  ```

- [ ] **Step 1: Extract each (all currently `status: accepted`, none supersede another yet)**

```bash
extract_decision() {
  local start=$1 end=$2 id=$3 title=$4 file=$5
  { echo "---"; echo "id: $id"; echo "status: accepted"; echo "supersedes: null"; echo "---"; echo
    echo "# $id — $title"; echo
    sed -n "${start},${end}p" RESEARCH_LOG.md
  } > "$file"
}
extract_decision 323 323 L-01 "Judge across the seed x year distribution" docs/decisions/L-01-judge-across-seed-year-distribution.md
extract_decision 324 324 L-02 "MSRR weight magnitude carries no signal (scale invariance)" docs/decisions/L-02-msrr-scale-invariance.md
extract_decision 325 325 L-03 "Test ideas; don't dismiss them on priors" docs/decisions/L-03-test-dont-dismiss-on-priors.md
extract_decision 326 326 L-04 "Check saved artifacts before retraining" docs/decisions/L-04-check-saved-artifacts-first.md
extract_decision 327 327 L-05 "Attention is not the memory bottleneck at this scale" docs/decisions/L-05-attention-scaling-is-not-the-bottleneck.md
extract_decision 328 328 L-06 "Period-matching and the 1-survivor verdict" docs/decisions/L-06-period-matching-and-the-1-survivor-verdict.md
extract_decision 329 329 L-07 "IC and SDF Sharpe can diverge -- judge MSRR on its objective" docs/decisions/L-07-ic-and-sdf-sharpe-can-diverge.md
extract_decision 330 330 L-08 "Screens can be false positives -- confirm with full seeds x years" docs/decisions/L-08-screens-can-be-false-positives.md
extract_decision 331 331 L-09 "The metric was the bug -- audit the statistic, not just the model" docs/decisions/L-09-the-metric-was-the-bug.md
extract_decision 345 345 L-10 "Use an adversarial second model" docs/decisions/L-10-use-an-adversarial-second-model.md
extract_decision 332 343 L-11 "Power-check the controlled contrast, not the headline one" docs/decisions/L-11-power-check-the-controlled-contrast.md
```

- [ ] **Step 2: Verify byte-fidelity of each body**

```bash
declare -A ranges=( [L-01]="323 323" [L-02]="324 324" [L-03]="325 325" [L-04]="326 326" [L-05]="327 327" [L-06]="328 328" [L-07]="329 329" [L-08]="330 330" [L-09]="331 331" [L-10]="345 345" [L-11]="332 343" )
for f in docs/decisions/L-*.md; do
  id=$(grep "^id:" "$f" | awk '{print $2}')
  read -r start end <<< "${ranges[$id]}"
  diff <(sed -n "${start},${end}p" RESEARCH_LOG.md) <(tail -n +9 "$f") && echo "OK $id"
done
```
Expected: 11 `OK` lines, no diff output

---

## Task 9: Build `docs/log/INDEX.md`

**Files:**
- Create: `docs/log/INDEX.md`

**Interfaces:**
- Consumes: the file list from Tasks 4–8 (all `docs/experiments/*.md`,
  `docs/decisions/*.md`), plus `RESEARCH_LOG.md` lines 101–115 (Experiment summary
  table) as the source of one-line descriptions.

This is the one hand-authored file — short (under 40 lines) and low numeric-density, so
hand-authoring from the already-extracted files is lower-risk than scripting it, but
every one-line description must be copied from the Task-4–7 files or the original
summary table, not freshly paraphrased.

- [ ] **Step 1: Read `RESEARCH_LOG.md` lines 101–115 and all files in `docs/experiments/`
  and `docs/decisions/` to source each row's one-line description**

- [ ] **Step 2: Write `docs/log/INDEX.md`**

```markdown
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
```

- [ ] **Step 3: Verify every link target exists**

```bash
grep -oE '\]\(\.\./[a-z]+/[A-Za-z0-9._-]+\.md\)' docs/log/INDEX.md | tr -d '](' | tr -d ')' | while read -r rel; do
  test -f "docs/log/$rel" && echo "OK $rel" || echo "BROKEN $rel"
done
```
Expected: every line says `OK` (note: postmortem links won't resolve until Task 12 —
re-run this check after Task 12 completes, not a blocker for this task)

---

## Task 10: Coverage verification — prove nothing was lost

**Files:**
- None created; this is a read-only audit task.

**Interfaces:**
- Consumes: `RESEARCH_LOG.md` (original, still present at repo root) and every file
  created in Tasks 2–9.

- [ ] **Step 1: Confirm every EXP-NNN and L-NN token in the original appears in the new tree**

```bash
grep -oE 'EXP-[0-9]{3}' RESEARCH_LOG.md | sort -u > /tmp/orig_exp_ids.txt
grep -orhE 'EXP-[0-9]{3}' docs/experiments/ docs/log/ docs/decisions/ docs/BACKLOG.md docs/REFERENCE.md | sort -u > /tmp/new_exp_ids.txt
comm -23 /tmp/orig_exp_ids.txt /tmp/new_exp_ids.txt
```
Expected: no output (every EXP-NNN from the original appears somewhere in the new tree)

```bash
grep -oE 'L-[0-9]{2}' RESEARCH_LOG.md | sort -u > /tmp/orig_l_ids.txt
grep -orhE 'L-[0-9]{2}' docs/decisions/ docs/log/ | sort -u > /tmp/new_l_ids.txt
comm -23 /tmp/orig_l_ids.txt /tmp/new_l_ids.txt
```
Expected: no output

- [ ] **Step 2: Confirm every numeric token (Sharpe ratios, t-stats, dollar/percent figures)
  in the migrated ranges survives**

```bash
# Every line that was migrated verbatim (all ranges in the manifest except the
# intentionally-dropped ones: 1-35, 48-58, 117-118, 321-322, 344, 346-348)
for range in 36,47 59,75 76,100 101,115 293,320 349,358 359,369 119,292 323,331 332,343 345,345; do
  sed -n "${range}p" RESEARCH_LOG.md
done | grep -oE '[-+]?[0-9]+\.[0-9]+' | sort -u > /tmp/orig_numbers.txt
cat docs/REFERENCE.md docs/BACKLOG.md docs/experiments/*.md docs/decisions/*.md docs/log/*.md docs/log/archive/*.md 2>/dev/null \
  | grep -oE '[-+]?[0-9]+\.[0-9]+' | sort -u > /tmp/new_numbers.txt
comm -23 /tmp/orig_numbers.txt /tmp/new_numbers.txt
```
Expected: no output (every decimal *value* from the migrated ranges appears somewhere in
the new tree). Use `sort -u` on BOTH sides, not plain `sort` — `comm -23` on
non-deduplicated input does a multiset (occurrence-count) subtraction, not a set
membership test, so a value that's simply quoted a different number of times across the
old vs. new layout (e.g. redundant between a summary-table row and its full writeup)
produces spurious "missing" output even though the value is genuinely present. Sweep
`docs/log/*.md` and `docs/log/archive/*.md` too — the summary-table range (101-115) is
hand-authored into `docs/log/INDEX.md` (Task 9), not sed-extracted, so it's the one range
that doesn't otherwise land in `docs/experiments/`/`docs/decisions/`.

- [ ] **Step 3: If either check produces output, stop and investigate before Task 11**

A non-empty result means something was missed — go back to the manifest, find which
range covers the missing ID/number, and add the missing extraction before proceeding.
Do not archive the original (Task 11) until both checks in this task are clean.

---

## Task 11: Archive the original

**Files:**
- Modify (rename): `RESEARCH_LOG.md` → `docs/log/archive/RESEARCH_LOG-2026-06-02-to-2026-07-12.md`

**Interfaces:**
- Depends on: Task 10 passing cleanly.

- [ ] **Step 1: Move the file**

```bash
git mv RESEARCH_LOG.md docs/log/archive/RESEARCH_LOG-2026-06-02-to-2026-07-12.md
```

- [ ] **Step 2: Stamp it read-only/archived at the top**

Edit the first line of `docs/log/archive/RESEARCH_LOG-2026-06-02-to-2026-07-12.md`,
inserting before the existing `# ml-alpha — Research Log` title:

```markdown
> **ARCHIVED 2026-07-12, read-only.** Fully migrated into `docs/log/`,
> `docs/experiments/`, `docs/decisions/`, `docs/BACKLOG.md`, `docs/REFERENCE.md` — see
> `docs/log/INDEX.md` for the current index and
> `docs/superpowers/plans/2026-07-12-research-log-restructure.md` for the migration
> manifest. Kept as a byte-for-byte safety net; not part of the active system.

```

- [ ] **Step 3: Verify**

```bash
test -f RESEARCH_LOG.md && echo "FAIL: original still at repo root" || echo "OK: moved"
test -f docs/log/archive/RESEARCH_LOG-2026-06-02-to-2026-07-12.md && echo "OK: archive exists"
head -5 docs/log/archive/RESEARCH_LOG-2026-06-02-to-2026-07-12.md
```
Expected: `OK: moved`, `OK: archive exists`, banner visible in head output

(No commit — see Global Constraints. `git mv` stages the rename; leave it staged or
unstage with `git restore --staged` to match how prior work in this session was left.)

---

## Task 12: Backfill 2 postmortems

**Files:**
- Create: `docs/postmortems/2026-07-11-mean-of-ratios-sharpe-bug.md`
- Create: `docs/postmortems/2026-07-11-attention-effect-misidentification.md`

**Interfaces:**
- Consumes: `docs/decisions/L-09-the-metric-was-the-bug.md` (Task 8 output) and
  `docs/decisions/L-11-power-check-the-controlled-contrast.md` (Task 8 output) as the
  factual source — these postmortems restructure already-established facts into the
  postmortem template, they do not introduce new claims.

- [ ] **Step 1: Write the mean-of-ratios postmortem**

Write `docs/postmortems/2026-07-11-mean-of-ratios-sharpe-bug.md`:

```markdown
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
```

- [ ] **Step 2: Write the attention-effect postmortem**

Write `docs/postmortems/2026-07-11-attention-effect-misidentification.md`:

```markdown
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
```

- [ ] **Step 3: Verify**

```bash
test -s docs/postmortems/2026-07-11-mean-of-ratios-sharpe-bug.md && echo OK1
test -s docs/postmortems/2026-07-11-attention-effect-misidentification.md && echo OK2
```
Expected: `OK1`, `OK2`

- [ ] **Step 4: Re-run Task 9 Step 3's link check** (now that these files exist)

```bash
grep -oE '\]\(\.\./[a-z]+/[A-Za-z0-9._-]+\.md\)' docs/log/INDEX.md | tr -d '](' | tr -d ')' | while read -r rel; do
  test -f "docs/log/$rel" && echo "OK $rel" || echo "BROKEN $rel"
done
```
Expected: every line says `OK`

---

## Task 13: Update `CLAUDE.md`

**Files:**
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: nothing external; content is freshly authored to reflect current status
  (per spec §3, "Current state" is living, not migrated verbatim from the retired
  Status Snapshot).

- [ ] **Step 1: Replace the "How research work is done here" RESEARCH_LOG.md references**

In `CLAUDE.md`, find:
```
is committed. `RESEARCH_LOG.md` is the lab notebook and the only place a number becomes a
"finding".
```
Replace with:
```
is committed. `docs/log/` (narrative) + `docs/experiments/EXP-NNN-*.md` (per-experiment
writeups) + `docs/decisions/L-NN-*.md` (distilled findings) are the lab notebook — a
number becomes a "finding" only in one of these three places, never edited in place once
written (a correction is a new entry/file, never a rewrite — see `docs/log/INDEX.md`).
Pre-2026-07-12 history: `docs/log/archive/RESEARCH_LOG-2026-06-02-to-2026-07-12.md`.
```

- [ ] **Step 2: Update the agents table row and downhill-handoff sentence**

Find:
```
| `stats-gatekeeper` | **opus** | **any** number, comparison, or conclusion before it enters `RESEARCH_LOG.md`. |
```
Replace with:
```
| `stats-gatekeeper` | **opus** | **any** number, comparison, or conclusion before it enters `docs/experiments/` or `docs/decisions/`. |
```

Find (the downhill-handoff sentence, wording may vary slightly — match on
`→ \`RESEARCH_LOG.md\`` at the end):
```
numbers?) → `stats-gatekeeper` (do they mean anything?) → `RESEARCH_LOG.md`. The two haiku
```
Replace with:
```
numbers?) → `stats-gatekeeper` (do they mean anything?) → `docs/experiments/`/`docs/decisions/`. The two haiku
```

- [ ] **Step 3: Add a "Current state" section and a "Where things live" table**

Insert after the existing project-overview paragraph near the top of `CLAUDE.md` (before
the "## Commands" section):

```markdown
## Current state (2026-07-12 — living, overwrite as things change)

- **MSRR pooled baseline:** SDF Sharpe **1.41** over 88 OOS months (L1 combiner; the
  1.81 figure quoted before 2026-07-11 was a mean-of-ratios bug — see
  `docs/postmortems/2026-07-11-mean-of-ratios-sharpe-bug.md`). Still subject to the
  universe look-ahead (EXP-010 D-4, unresolved) and a +-0.39 noise floor over 88 months.
- **MSE transformer:** decile L/S Sharpe reproduces at ~2.0-2.8 (EXP-001), not yet
  recomputed under the pooled convention.
- **Replication target is KKM/AIPM (w33351), not GKX** — see
  `docs/experiments/EXP-012-kkm-replication.md`. Not yet attempted; protocol gap
  analysis + 4-rung ladder plan is written and gated pending go-ahead.
- **Binding constraint:** statistical power (OOS months), not model ideas — see L-06,
  L-09.

## Where things live
- Narrative log: `docs/log/` (newest month on top) + `docs/log/INDEX.md`
- Experiments (immutable once verdict is in): `docs/experiments/EXP-NNN-*.md`
- Decisions (immutable, ADR-style — supersede, don't edit): `docs/decisions/L-NN-*.md`
- Postmortems (after a costly surprise): `docs/postmortems/`
- Backlog (living, mutate in place): `docs/BACKLOG.md`
- Static reference (compute discipline, env/data card, conventions, glossary):
  `docs/REFERENCE.md`
- Pre-2026-07-12 history: `docs/log/archive/RESEARCH_LOG-2026-06-02-to-2026-07-12.md`
```

- [ ] **Step 4: Verify**

```bash
grep -c "docs/experiments\|docs/decisions" CLAUDE.md   # expect several hits now
grep -c "RESEARCH_LOG.md" CLAUDE.md   # expect only the archive-path mentions, not bare references
```

---

## Task 14: Update hooks

**Files:**
- Modify: `.claude/hooks/research-log-checklist.sh`
- Create: `.claude/hooks/closed-record-reminder.sh`
- Modify: `.claude/settings.json`

**Interfaces:**
- Consumes: existing hook wiring in `.claude/settings.json` (`PreToolUse` → `Edit|Write`
  matcher, two hooks currently chained: frozen-script guard, research-log-checklist).

- [ ] **Step 1: Repoint `research-log-checklist.sh`'s filename match**

In `.claude/hooks/research-log-checklist.sh`, find:
```bash
f=$(jq -r '.tool_input.file_path // empty')
[ "${f##*/}" = "RESEARCH_LOG.md" ] || exit 0
```
Replace with:
```bash
f=$(jq -r '.tool_input.file_path // empty')
case "$f" in
  */docs/log/*.md|*/docs/experiments/*.md|*/docs/decisions/*.md) ;;
  *) exit 0 ;;
esac
```

- [ ] **Step 2: Update the hook's `statusMessage`/reason text in `.claude/settings.json`
  and the checklist's own trailing sentence to no longer say "RESEARCH_LOG.md write"**

In `.claude/hooks/research-log-checklist.sh`, find:
```
"Research log write — auto-allowed; statistical checklist injected."
```
Replace with:
```
"Research-log write (docs/log, docs/experiments, or docs/decisions) — auto-allowed; statistical checklist injected."
```

- [ ] **Step 3: Write the new non-blocking "closed record" reminder hook**

Write `.claude/hooks/closed-record-reminder.sh`:

```bash
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
```

- [ ] **Step 4: Make it executable and wire it into `.claude/settings.json`**

```bash
chmod +x .claude/hooks/closed-record-reminder.sh
```

In `.claude/settings.json`, inside the existing `"matcher": "Edit|Write"` hooks array
(alongside the frozen-script guard and research-log-checklist entries), add:
```json
          {
            "type": "command",
            "command": "\"$CLAUDE_PROJECT_DIR/.claude/hooks/closed-record-reminder.sh\"",
            "statusMessage": "Checking closed-record convention"
          }
```

- [ ] **Step 5: Verify the hook fires correctly (dry test, no real edit)**

```bash
echo '{"tool_input":{"file_path":"/home/max/Documents/ml-alpha/docs/experiments/EXP-001-reproduce-readme-transformer.md"}}' | .claude/hooks/closed-record-reminder.sh
echo '{"tool_input":{"file_path":"/home/max/Documents/ml-alpha/docs/BACKLOG.md"}}' | .claude/hooks/closed-record-reminder.sh   # expect no output (exit 0 before the jq)
```
Note: the path must be absolute (or otherwise contain a `/` immediately before `docs`)
for the hook's `*/docs/experiments/EXP-*.md` case pattern to match — this matches how
Claude Code's actual `tool_input.file_path` is always absolute, but a bare relative path
like `docs/experiments/EXP-001-....md` (no leading `/`) will silently not match and is
not a valid test of the hook.
Expected: first command prints the `permissionDecision: allow` JSON with the reminder;
second command prints nothing.

---

## Task 15: Update `stats-gatekeeper` agent description

**Files:**
- Modify: `.claude/agents/stats-gatekeeper.md`

**Interfaces:**
- None (standalone agent definition file).

- [ ] **Step 1: Update the frontmatter `description` field**

Find:
```
description: MUST be used before any numerical result, comparison, or conclusion is written into RESEARCH_LOG.md, a report, or a paper.
```
Replace with:
```
description: MUST be used before any numerical result, comparison, or conclusion is written into docs/experiments/, docs/decisions/, a report, or a paper.
```

- [ ] **Step 2: Update the procedure step that names the destination file**

Find:
```
5. Write the exact sentence that may be entered into `RESEARCH_LOG.md`, including the
   uncertainty. If the answer is NOT DETECTABLE, the sentence must say so — not "slightly
   worse", not "trends toward".
```
Replace with:
```
5. Write the exact sentence that may be entered into `docs/experiments/EXP-NNN-*.md` or
   `docs/decisions/L-NN-*.md`, including the uncertainty. If the answer is NOT
   DETECTABLE, the sentence must say so — not "slightly worse", not "trends toward".
```

- [ ] **Step 3: Verify**

```bash
grep -n "RESEARCH_LOG.md" .claude/agents/stats-gatekeeper.md
```
Expected: no output (all references updated)

---

## Final check (run after all 15 tasks)

```bash
test -f RESEARCH_LOG.md && echo "FAIL: original still present" || echo "OK"
ls docs/experiments/ | wc -l    # expect 12 (EXP-001..012)
ls docs/decisions/ | wc -l      # expect 11 (L-01..L-11)
ls docs/postmortems/ | wc -l    # expect 3 (README + 2 postmortems)
test -f docs/BACKLOG.md && test -f docs/REFERENCE.md && test -f docs/log/INDEX.md && echo OK
grep -rn "RESEARCH_LOG.md" CLAUDE.md .claude/hooks/ .claude/agents/ | grep -v archive
```
Expected: `OK`, `12`, `11`, `3`, `OK`, and the final grep produces no output (every
remaining reference to `RESEARCH_LOG.md` outside the archive path has been repointed).
