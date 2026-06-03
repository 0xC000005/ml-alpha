# ml-alpha — Research Log

A hypothesis-driven lab notebook for improving the cross-sectional / MSRR
return-prediction models. Format follows standard ML experiment-tracking practice
(record hypothesis · code version · config · data · command · metrics · conclusion;
append-only; one hypothesis per experiment so results are comparable and reproducible).

**How to use this file**
- Pick an idea from the **Backlog**, give it the next `EXP-NNN`, and add a stub to the **Experiment Log** *before* running.
- An experiment must record: git commit, exact config/command, where outputs live, and a verdict against a stated prediction.
- Append; do not rewrite history. Promote distilled findings to **Decisions & Learnings**.

---

## ⚠️ Compute discipline (read before launching anything)

The Trillium harness makes runs cheap, which makes waste easy. Rules:

1. **Screen cheap before committing.** New idea → first run a *minimal* probe: 2–3 representative years (e.g. 2014 = a hard year, 2018/2019 = strong years), **3 seeds**, one or two configs. Only promote to a full 8-year × 10-seed run if the cheap screen shows real, consistent signal.
2. **Judge across the seed × year distribution, not a point estimate.** Per-year Sharpe swings −1 → +5 and per-seed dispersion is large; a single good number is almost always noise. Report min/median/max across seeds and all years.
3. **Reuse artifacts; avoid retraining.** Many questions can be answered by re-analyzing *saved* per-seed models/predictions (see EXP-002) — no GPU training at all.
4. **1 GPU = quarter node, throttle the array.** `--gpus-per-node=1`, no `--mem`/`--partition`, submit arrays with `%2` (≤2 H100s held). Never request whole nodes.
5. **Don't run what you can't measure.** If an effect is smaller than the noise floor over the available years, the experiment can't conclude — redesign or skip.

---

## Status snapshot (2026-06-03)

- **Models:** Cross-Sectional Transformer (MSE) + MSRR Transformer (Kelly 2025), ~14K params, ONE pre-norm block (d_model=32, n_heads=4, d_ff=64, **n_layers is dead config**). Expanding-window yearly refit, monthly predictions, multi-seed ensemble. Shared pipeline in `train_nn.py`.
- **Reproduced on Trillium (EXP-001):** MSE avg L/S Sharpe **+2.84** (README 2.16); MSRR avg SDF Sharpe **+3.13** on 2016–19 (README 2.05). Both ran *hot* vs published; MSRR per-year is high-variance.
- **Infra:** validated clone→data→offline-venv→`cluster/` harness→collect pipeline. See `COMPUTE_CANADA.md` (gitignored) + `docs/superpowers/specs/2026-06-03-transformer-repro-harness-design.md`.

---

## Backlog (prioritized hypotheses)

Priority reflects *robust expected impact ÷ cost*, re-assessed honestly (an earlier
adversarial brainstorm was biased against capacity/architecture ideas — see
Decisions L-03). Status: 🔲 todo · 🔬 in progress · ✅ done · ❌ tested-negative.

| ID | Direction | Hypothesis | Key evidence | Cost | Status |
|----|-----------|-----------|--------------|------|--------|
| B-01 | **Capacity / Virtue of Complexity** (bigger/deeper NN) | More parameters (depth via `n_layers`, width via d_model/d_ff) raise OOS Sharpe in *this* return-prediction regime | Kelly–Malamud–Zhou, "Virtue of Complexity in Return Prediction," *JF* 2024 (complex > simple for equity returns); counter: GKX 2020 found shallow FFNs best | med | 🔲 (blocked by B-00) |
| B-00 | **Fix `n_layers` dead config** | The model ignores `n_layers` and hardcodes 1 block; stacking it (byte-identical at n_layers=1) is the prerequisite to test depth | Prior code audit | trivial | 🔲 |
| B-02 | **Temporal dimension** | Adding memory across months (per-stock sequence / temporal attention / macro-regime conditioning) captures characteristic *dynamics* the snapshot model can't | Chen–Pelger–Zhu (macro recurrence); IPCA time-varying loadings; FactorVAE | large | 🔲 |
| B-03 | **More sophisticated architecture** | Better attention for the ~5000-stock set (set-transformer / inducing points), macro gating (FiLM/GLU), characteristic embeddings, missingness-aware inputs | transformer asset-pricing literature | med–large | 🔲 |
| B-04 | **Monthly / more-frequent refit** | Retraining monthly (vs yearly) on fresher data raises OOS Sharpe | mixed; caveat: ~12× compute and likely within the noise floor over few years | high | 🔲 (low prio) |
| B-05 | **MSRR ensemble combiner** | Robust aggregation (trimmed-mean / val-loss-screened over normalized weights) tames the heavy-tailed seed distribution | EXP-002 showed normalization changes the estimate materially | trivial | 🔲 |
| B-06 | **Missing-data handling** | 26% NaN→0 erases size/quality info; per-row missingness indicators and/or GKX rank-normalized features add robust IC | GKX rank-norm preprocessing | small | 🔲 |
| B-07 | **Transaction-cost realism** | Reported Sharpe ignores costs/turnover (weights imply ~200× gross leverage) → likely overstated; need a cost model before trusting any "win" | — | med | 🔲 (credibility) |
| B-08 | **Selection metric** | Early-stopping on a *deployed, scale-invariant* metric (val IC for MSE; scale-free for MSRR) beats stopping on raw val loss on a ~0% R² panel | val_ic already computed but unused (`train_transformer.py:347`) | small | 🔲 |
| B-09 | **IPCA conditional-linear baseline** | A parsimonious Kelly-style yardstick to discipline every "bigger/temporal helps" claim (diagnostic, not a Sharpe lever) | Kelly–Pruitt–Su IPCA | med | 🔲 |

**First move:** B-00 → B-01 (fix `n_layers`, then a *cheap-screened* depth/width sweep) — this directly tests the "bigger NN" hypothesis with the strongest literature support, and the result decides it.

---

## Experiment Log (append-only)

### EXP-001 — Reproduce README Transformer results on Trillium
- **Date:** 2026-06-02/03 · **Status:** ✅ done · **Commit:** `cluster-repro-harness` (29f5740)
- **Hypothesis:** the committed code reproduces the README's MSE (2012–19) and MSRR (2016–19) numbers (statistical match, not bit-exact; RTX 4080 → H100).
- **Setup:** `cluster/repro.sbatch` array `--array=0-15%2` (MSE 2012–19 ×5 seeds, MSRR 2012–19 ×10 seeds), 1 H100/quarter-node each, config injected via `cluster/run_year.py` (no edits to training scripts). Outputs: `output/repro/*/metrics/*summary.csv`, combined `output/repro/repro_combined.csv`.
- **Result:** 16/16 COMPLETED, ~9h50m wall, ≤2 GPUs, ~19 GPU-h. MSE avg L/S Sharpe **+2.84** (IC +0.021 exact, 8/8 positive); MSRR avg SDF Sharpe **+3.13** on 2016–19 (all positive). Both ran higher than published (MSE 2.16 / MSRR 2.05).
- **Conclusion:** signal reproduces faithfully (IC, direction); Sharpe runs hot — flagged for investigation (→ EXP-002). MSRR fails on early years (2012/2014 negative), high per-year variance.
- **Next:** investigate the hot Sharpe; brainstorm improvements.

### EXP-002 — MSRR ensemble weight-normalization A/B (no retraining)
- **Date:** 2026-06-03 · **Status:** ✅ done (tested-negative as a Sharpe win) · **Script:** `cluster/ab_msrr_norm.py`
- **Hypothesis (from brainstorm):** `train_transformer_msrr.py:431` averages RAW per-seed weights; since MSRR Sharpe is scale-invariant, the largest-magnitude seed dominates the mean → L1-normalizing each seed's per-month weights before averaging should *improve* the ensemble.
- **Setup:** reload the 80 saved per-seed MSRR models, reconstruct each seed's test predictions (no training), compare raw-mean (A) vs L1-normalized-mean (B). Sanity: raw(A) must equal the saved `sdf_sharpe`. Fixed test window to Jan–Nov (11 mo) to match the original per-year data loads — after which raw(A) reproduced saved EXACTLY for all 8 years.
- **Result:** normalization **LOWERS** avg SDF Sharpe (2016–19: **3.13 → 2.06**; 2012–19: 2.14 → 1.78). It rescues blow-up years (2014 −1.13→+0.60, 2017 0.66→1.97) but caps luck-driven star years (2018 5.31→0.68, 2019 4.30→3.07).
- **Conclusion:** the fix is *principled-correct* (raw mean weights seeds by an arbitrary, signal-free magnitude) but is **not a Sharpe win** — it's a robustness trade. The hot +3.13 reproduction was partly **seed-scale luck**; the normalized **+2.06 ≈ the published +2.05** is the more honest/robust estimate.
- **Next:** decision pending on adopting normalization + reporting both (B-05). Key lesson → L-02.

---

### Planned experiments (GATED — full plan: `docs/superpowers/plans/2026-06-03-model-improvement-experiments.md`)
Prep (code + local CPU validation) done where noted; **no cluster job runs without explicit go-ahead.**
**All prep code is implemented and LOCALLY VALIDATED (CPU/local GPU, no cluster).** Runners: `run_experiment.py` (capacity/GLU, via main()) and `exp_main.py` (missingness/temporal/monthly). Configs: `gen_configs.py` → `experiments/configs/*.jsonl`. Submit harness: `sweep.sbatch` (generic array). Default model is bit-identical to frozen.
- **EXP-003 — Capacity + Virtue of Complexity** (ideas 1,2): `n_layers∈{1,2,3} × d_model∈{32,64}` (+ `weight_decay` ridge knob). Prep ✅. `configs/capacity.jsonl` = 15 tasks. ~7–10 GPU-h. Held: `sbatch --array=0-14%2 --export=ALL,CONFIG_FILE=experiments/configs/capacity.jsonl experiments/sweep.sbatch`.
- **EXP-004 — Sophistication** (idea 4): GLU FFN (`sophistication_glu.jsonl`, run_experiment) + missingness indicators (`missingness.jsonl`, exp_main, with matched baseline). Prep ✅. ~3–4 GPU-h.
- **EXP-005 — Temporal** (idea 3): macro-state GRU over trailing 12/24 mo (`temporal.jsonl`, exp_main). Prep ✅. ~8–12 GPU-h. Bigger bet.
- **EXP-006 — Monthly refit** (idea 5): `monthly.jsonl` (exp_main, 1 yr=2018). Prep ✅. ~3–4 GPU-h. **P5a power note:** SE(annualized Sharpe) ≈ √((1+½·SR²)/T); over 1 year (T≈11 mo) SE≈0.30, over 8 yrs (T≈88) SE≈0.11 — a realistic monthly-vs-yearly gain (≤~0.3 Sharpe) is **near-undetectable**, so the 1-year run is only a *sanity probe*; do NOT scale to all years unless it shows a large, consistent effect (unlikely). This is the lowest-priority direction.
- Total cheap-screen budget ≈ **20–30 GPU-h, ≤2 GPUs at a time**. Full runs only for screens that pass. **Every `sbatch` is GATED behind explicit go-ahead.**

## Decisions & Learnings

- **L-01 (methodology):** Judge every change across the **full seed × year distribution**. A single hot number (e.g. MSRR +3.13) can be seed luck. This is the most important discipline in this project.
- **L-02 (MSRR scale):** MSRR weight *magnitude* carries no signal (scale-invariant Sharpe). Raw ensemble averaging therefore weights seeds by noise; this inflates Sharpe via lucky large-scale seeds. Verified in EXP-002.
- **L-03 (don't trust armchair dismissals):** An adversarial brainstorm (workflow `wk32jnjtd`) was prompted to treat overfitting as the enemy and predictably labeled capacity/temporal ideas "traps." But (a) its top pick failed when tested (EXP-002), and (b) Kelly et al.'s *Virtue of Complexity* (JF 2024) — same author lineage as the MSRR loss — argues bigger *helps* return prediction. **Test ideas; don't dismiss them on priors.**
- **L-04 (efficiency):** Some questions need zero training — EXP-002 answered a real question by re-analyzing saved models. Always check whether saved artifacts suffice first.

---

## Conventions

- **Code version:** every experiment records its git commit / branch. Config injected via `cluster/run_year.py` style wrappers; do not edit the training scripts to parameterize a run unless the change is the experiment.
- **Outputs:** `output/<exp>/...` on scratch (cluster) and pulled to local `output/` (gitignored). Summary CSVs + a one-line verdict are the durable record; large model/prediction files stay on scratch.
- **Run via the harness:** `sbatch --array=...%2 cluster/repro.sbatch` (parameterized), or `srun ... bash -lc 'source activate_cluster.sh && python ...'` for one-offs. See `cluster/` and the design doc.
- **Monitoring:** `sq`; `srun --jobid=$J --overlap nvidia-smi` (NOT `--gres=none` on Trillium); `seff <jid>` post-run.
- **Naming:** experiments `EXP-NNN`; backlog ideas `B-NN`; learnings `L-NN`.
