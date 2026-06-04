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
- **Capacity screen done (EXP-003):** depth noisy-positive but **non-monotone** (L3<base), width (d_model) **HURTS** every year, nothing clears the promote bar — *noise-limited, not capacity-limited, at this scale*.
- **Paper read (w33351 AIPM):** our ~2.0 sits at their **post-2002 BSV/linear** level (2.03); their **4.57 headline is full-sample** (every model ~halves post-2002 — their best modern transformer = 3.37). Gap is **scale + regime, not data**. Period-matched, we already equal their linear modern-era baseline. → **Enhancement Roadmap** below.

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
| B-06 | **Missing-data handling** | 26% NaN→0 erases size/quality info; per-row missingness indicators and/or GKX rank-normalized features add robust IC | GKX rank-norm preprocessing | small | ❌ (missingness mask: subfloor, near-0 corr) |
| B-11 | **Per-month rank-standardize** (paper input contract) | Replace pooled z-score+5σ-clip with per-month **cross-sectional rank** to [−0.5,0.5], missing→cross-sectional median (0); closes a 3-axis fidelity gap with the AIPM input | w33351 §4.1; **only candidate `robust=true`** in enhancement-design workflow (`wf_2f4b6eee`) | small | ❌ **REJECTED — passed 5-seed screen (EXP-007) but FAILED 10-seed/8yr confirmation (EXP-009); base ≥ rank** |
| B-07 | **Transaction-cost realism** | Reported Sharpe ignores costs/turnover (weights imply ~200× gross leverage) → likely overstated; need a cost model before trusting any "win" | — | med | 🔲 (credibility) |
| B-08 | **Selection metric** | Early-stopping on a *deployed, scale-invariant* metric (val IC for MSE; scale-free for MSRR) beats stopping on raw val loss on a ~0% R² panel | val_ic already computed but unused (`train_transformer.py:347`) | small | 🔲 |
| B-09 | **IPCA conditional-linear baseline** | A parsimonious Kelly-style yardstick to discipline every "bigger/temporal helps" claim (diagnostic, not a Sharpe lever) | Kelly–Pruitt–Su IPCA | med | 🔲 |

**First move (revised 2026-06-04 — after EXP-003/007/009):**
B-00/B-01 **done-and-weak** (EXP-003). B-05 (L1 honest metric) **adopted + proven essential**. B-11 (rank-standardize) **REJECTED** — passed a 5-seed screen but failed the 10-seed/8yr confirmation (EXP-009); base ≥ rank (L-08). **Sobering takeaway:** the two cheapest, best-motivated input/preprocessing levers (rank-norm, per-month-z) do **not** beat base over 8 years — the binding constraint is the **noise floor**, not the input. Remaining options, all needing an effect *larger* than the floor to be worth it: the gated MSRR **depth ladder** (EXP-008, screen with ≥10 seeds now) and **paper-regime training** (Tier 3: monthly refit ⇒ ~12× OOS months ⇒ a tighter floor, the one move that attacks the *power* problem directly). See **Enhancement Roadmap** below.

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

### EXP-003 — Capacity / Virtue-of-Complexity cheap screen (MSE transformer proxy)
- **Date:** 2026-06-03 · **Status:** ✅ done (inconclusive — *tested-weak*) · **Array:** 564717 (`capacity.jsonl`, 5 configs × 3 screen years = 15 tasks, %2)
- **Hypothesis (B-01):** more capacity — depth (`n_layers`) and/or width (`d_model`) — raises OOS Sharpe in this regime.
- **Setup:** MSE cross-sectional transformer, screen years 2014/2016/2018; configs base(K=1,d32)/L2(K=2)/L3(K=3)/d64(d_model=64)/d64L2(d64,K=2); metric `sharpe_ls_annual` (decile L/S).
- **Result (mean over 3 yrs):** base **2.02**; L2 **2.47** (win concentrated in 2014 +1.69, flat/worse 2016/2018); L3 **1.95** (<base; 2016 collapse −0.89); d64 **1.11** (worst, 0/3 — width hurts every year); d64L2 **2.16** (2/3 but 2016 craters −1.20).
- **Conclusion:** **no config clears the promote bar** (≥2/3 yrs AND clearly higher mean w/ controlled dispersion). Year-to-year swings (2–3 Sharpe) **dwarf** every config effect → *noise-limited at this scale*. One robust signal: **widening `d_model` alone HURTS** (matches the paper's kernel-limit — capacity must come from depth, not a fatter embedding). Depth is noisy-positive at best and **non-monotone** (L3<base).
- **Next:** re-test depth on the **MSRR** model under the **L1-normalized honest metric** with a strict monotone kill-bar (Roadmap Tier 2); do **not** widen `d_model`.

### EXP-007 — Rank-standardize A/B (B-11, Roadmap Tier 1), cheap screen, MSRR
- **Date:** 2026-06-03 · **Status:** ✅ done (**a2rank PASSES the screen → promote, with caveats**) · **Array:** 565492 (12 tasks, %2; driver `experiments/exp_main_msrr.py`) · **Metric:** L1 honest SDF Sharpe (raw + L1 both logged).
- **Setup:** 4 arms (base=A0 pooled-z / a1monthz=A1 / a2rank=A2 / a3rankgauss=A3) × 2014/16/18 × 5 seeds. Bar: A2 mean Δ-vs-base ≥ +0.5 AND ≥2/3 yrs positive AND A2-gap > A1-gap.
- **Result (L1 SDF Sharpe, mean over 3 yrs):** base **−0.02**, a1monthz **+0.34**, a2rank **+0.91**, a3rankgauss **+0.14**. a2rank Δ-vs-base **+0.92** (≥0.5 ✓); beats base 2/3 (2014 +1.35, 2018 +1.47; 2016 tie −0.05) ✓; rank-gap 0.92 ≫ monthz-gap 0.35 ✓ → **clears the bar**. Win **survives the honest L1** combiner (a2rank raw mean 0.81 ≈ L1 0.91 → *not* lucky-seed) and a2rank converges **faster** (fewer epochs; base 2018 ran 231 ep near the 300 cap).
- **Caveats:** (1) **IC flat-to-worse** — base mean IC 0.012 ≥ a2rank 0.010; the win is in the **portfolio** Sharpe, not the ranking (→ L-07). (2) Mean driven by 2 good years; **2016 is negative for all rank arms** (hard year). (3) 3 single-year windows are thin (SE ~±0.5–1.0/yr). (4) **a3rankgauss (tail-restore) does NOT help** → A3 hybrid killed; pure rank ≥ rank_gauss. (5) **a1monthz uniquely positive 3/3** (0.61/0.18/0.22) — the robustness arm; survives 2016 where everything else fails.
- **Conclusion:** promote **a2rank** to a full 8yr×10-seed confirmation (gated) vs **base** (+ a1monthz as the robustness comparator); judge on L1 SDF Sharpe over the ~±0.4 96-month floor, watch IC. Per-seed `.pt` saved on scratch (`output/exp/rank/*/models`) — Tier-0 artifact blocker now resolved.
- **Next:** EXP-009 full confirmation (GATED).

### EXP-009 — Full confirmation of rank-standardization (B-11), 8yr × 10 seeds, MSRR
- **Date:** 2026-06-03/04 · **Status:** ❌ **done — REVERSES the screen; rank NOT adopted** · **Array:** 565895 (24 tasks = base/a1monthz/a2rank × 2012–2019 × **10 seeds**, %2; driver `exp_main_msrr.py`) · **Metric:** L1 SDF Sharpe.
- **Result (8yr mean L1 SDF Sharpe):** **base 1.81 > a2rank 1.42 > a1monthz 0.46.** a2rank beats base in only **3/8** years; a1monthz 2/8. **mean IC tied** (base ≈ a2rank ≈ 0.014; a1monthz 0.006). So **base (current pooled-z) is best** — rank-standardization does not help and is slightly worse.
- **Why the screen lied:** the 3-year screen (EXP-007, **5 seeds**) caught base on unlucky low draws in exactly 2014/16/18 — at 10 seeds those base numbers jump (base 2014 L1: **1.07 @5seeds → 3.18 @10seeds**; base 2016: −1.24 → +1.31), which **erased** a2rank's apparent +0.92 edge. The honest L1 ensemble is *still* so noisy that adding 5 seeds swings a single-year Sharpe by >2 points.
- **Conclusion:** **keep pooled-z; do NOT adopt rank-standardization** (B-11 ❌). a1monthz's screen "robustness" was also a small-sample fluke (catastrophic 2012 −2.96, 2018 −1.14). Confirmation discipline (L-01) caught a false positive — working as intended. → L-08.
- **Next:** pivot to a lever with a *larger* expected effect than the noise floor — the gated MSRR depth ladder (EXP-008) or paper-regime training (Tier 3). Re-screen only with ≥10 seeds.

---

### Enhancement Roadmap (2026-06-03, design-only — workflow `wf_2f4b6eee`, 27 agents, 20 candidates → 1 survivor)
Goal: **robustly** higher OOS Sharpe for the MSRR transformer (not the lucky raw-ensemble number). 5 lenses proposed → deduped to 20 → adversarial scrutiny vs the ±0.4–0.5 window-noise floor. **Exactly one candidate survived as `robust=true`.**

- **Tier 0 — measurement infra (do FIRST, not a Sharpe claim):** adopt the **L1 equal-vote combiner** (normalize each seed's per-month weights to ‖w‖₁=1 before averaging; also L1 the final ensemble) as the *sole* combiner + comparison denominator. Honest point estimate is flat-to-lower but dispersion shrinks 30–50%. Gate on EXACTNESS (raw reproduces saved `sdf_sharpe` ≤0.02) + VARIANCE-REDUCTION, **not** a Sharpe bar. ⚠️ **Artifact blocker:** local `output/repro/msrr_*/` has only metrics CSVs — **no per-seed `.pt`**; either pull from Trillium (gated) or regenerate via the Tier-1 retrain.
- **Tier 1 — the one survivor (`robust=true`, medium impact):** **per-month cross-sectional rank-standardize the 95 signals to [−0.5,0.5]** (missing→median 0), replacing pooled-z+5σ. Closes a 3-axis divergence from the AIPM input contract (pooled→monthly, mean→median, z→rank). Cheap, no params, reversible. **Footgun:** transform signals only, never the 74 industry dummies. Mechanism verified live: LayerNorm normalizes per-token across `d_model`, *not* per-feature across stocks, so each char's cross-sectional dispersion leaks into attention keys/queries. **Screen:** 4 arms (A0 pooled-z control / A1 per-month-z / A2 rank / A3 winsorized-rank) × 2014/16/18 × 5 seeds, %2, L1-metric. **Bar:** A2 mean Δ-vs-A0 ≥ +0.5 AND ≥2/3 yrs positive AND A2-gap > A1-gap (else the win is just per-month standardization, not rank); if A2 fails but A3 clears, promote the hybrid.
- **Tier 2 — one capacity bet, conditional:** **MSRR depth ladder K∈{2,3}** (fixed d_model=32/d_ff=64; do **not** widen). Judge on L1-normalized per-year **median**, raw-vs-norm side by side. **PROMOTE only if monotone + normalized + multi-year:** median beats base ≥2/3 yrs (incl. 2014), ≥+0.5 in ≥2 yrs, K=3≥K=2. **KILL** on raw-only, 2014-only, or L3<L2 (the MSE-proxy pathology). Companions (paper-scaled init, QK-norm/RMSNorm/LayerScale) only if depth first shows a real trend.
- **Tier 3 — needs building, only if Tier 1–2 stall:** **paper-regime training** — build `exp_main_msrr.py` (current `exp_main.py` is MSE-only), screen the cheaper **rolling-window length** (60/120-mo vs expanding) *before* the expensive **monthly/quarterly refit**; co-tune head-ridge CV with depth. This attacks the modern-regime gap directly (regime-local data) but every verdict is subfloor on a single year — full-run only.
- **Discarded (inside the noise floor / confounded):** FFN-width d_f (≤+0.4k params), GLU/SwiGLU (not param-matched — capacity confound), single-head/temperature, vol-target combiner, scale-invariant early-stop, ridge-λ CV (no-op at lr 7.5e-5), missingness mask (~0 corr, width-in-a-bad-regime), macro×char FiLM, signal-set hygiene, turnover penalty (no-op on scale-invariant Sharpe — answer from saved artifacts, not a new loss), cosine-LR, longer val window.
- **Sequenced plan:** ① freeze L1 metric → ② build `RankFeatureScaler` + MSRR routing → ③ cheap-screen the 4-arm rank A/B (this retrain also regenerates the per-seed `.pt` Tier 0 needs) → ④ cheap-screen MSRR depth K∈{2,3} → ⑤ (cond.) init/stability → ⑥ promote ≤1 survivor to a full 8yr×10-seed run = **first legitimate cluster trigger, STOP for go-ahead** → ⑦ (if stall) Tier-3 driver. Judge every A/B on the L1 seed×year distribution (median+IQR), never raw mean, never one window.

---

### Planned experiments (GATED — full plan: `docs/superpowers/plans/2026-06-03-model-improvement-experiments.md`)
Prep (code + local CPU validation) done where noted; **no cluster job runs without explicit go-ahead.**
**All prep code is implemented and LOCALLY VALIDATED (CPU/local GPU, no cluster).** Runners: `run_experiment.py` (capacity/GLU, via main()) and `exp_main.py` (missingness/temporal/monthly). Configs: `gen_configs.py` → `experiments/configs/*.jsonl`. Submit harness: `sweep.sbatch` (generic array). Default model is bit-identical to frozen.
- **EXP-003 — Capacity + Virtue of Complexity** (ideas 1,2): ✅ **DONE** (array 564717) — see Experiment Log. Verdict: depth non-monotone, width hurts, noise-limited. Depth to be re-screened on MSRR under the honest metric (Roadmap Tier 2).
- **EXP-004 — Sophistication** (idea 4): GLU FFN (`sophistication_glu.jsonl`, run_experiment) + missingness indicators (`missingness.jsonl`, exp_main, with matched baseline). Prep ✅. ~3–4 GPU-h.
- **EXP-005 — Temporal** (idea 3): macro-state GRU over trailing 12/24 mo (`temporal.jsonl`, exp_main). Prep ✅. ~8–12 GPU-h. Bigger bet.
- **EXP-006 — Monthly refit** (idea 5): `monthly.jsonl` (exp_main, 1 yr=2018). Prep ✅. ~3–4 GPU-h. **P5a power note:** SE(annualized Sharpe) ≈ √((1+½·SR²)/T); over 1 year (T≈11 mo) SE≈0.30, over 8 yrs (T≈88) SE≈0.11 — a realistic monthly-vs-yearly gain (≤~0.3 Sharpe) is **near-undetectable**, so the 1-year run is only a *sanity probe*; do NOT scale to all years unless it shows a large, consistent effect (unlikely). This is the lowest-priority direction.
- **EXP-007 — Rank-standardize A/B (B-11, Roadmap Tier 1)** — the one `robust=true` lever. Prep ✅ **this session** (CPU-validated, no cluster). New code, all in `experiments/`, production frozen:
  - `feature_scalers.py` — `make_scaler(kind)`: A0 `pooled_z` (byte-identical to `TransformerFeatureScaler`, proven), A1 `month_z`, A2 `rank`→[−0.5,0.5], A3 `rank_gauss` (van der Waerden). Transforms the 95 signals only; dummies stay unscaled.
  - `msrr_combine.py` — L1 equal-vote combiner + honest `sdf_sharpe` (raw reproduces production; **B-05 wired**). Matches `cluster/ab_msrr_norm.py`.
  - `exp_main_msrr.py` — MSRR driver (reuses frozen `train_model_msrr`/`evaluate`), configurable scaler+combiner, saves per-seed `.pt`, logs raw+L1 Sharpe. `--dry` builds model+scaler with no data.
  - `validate_msrr_prep.py` — all 4 checks pass. `configs/rank_ab.jsonl` = **12 tasks** (4 arms × 2014/16/18 × 5 seeds).
  - **Held command:** `sbatch --array=0-11%2 --export=ALL,CONFIG_FILE=experiments/configs/rank_ab.jsonl,RUNNER=experiments/exp_main_msrr.py experiments/sweep.sbatch`. **Bar:** A2 mean Δ-vs-base ≥ +0.5 AND ≥2/3 yrs positive AND A2-gap > A1-gap (else promote A3 if it clears, else shelve). Collect: `collect_screen.py output/exp/rank msrr_transformer_summary.csv`.
- **EXP-008 — MSRR depth ladder K∈{1,2,3} (B-01, Roadmap Tier 2, GATED on EXP-007 + honest metric)** — Prep ✅ (`n_layers` already plumbed). `configs/msrr_depth.jsonl` = **9 tasks** (base/L2/L3 × 2014/18/19 × 5 seeds, pooled-z held fixed). Strict monotone kill-bar: median beats base ≥2/3 incl. 2014, ≥+0.5 in ≥2 yrs, K=3≥K=2; raw-only/2014-only/L3<L2 ⇒ KILL.
- Total cheap-screen budget ≈ **20–30 GPU-h, ≤2 GPUs at a time**. Full runs only for screens that pass. **Every `sbatch` is GATED behind explicit go-ahead.**

## Decisions & Learnings

- **L-01 (methodology):** Judge every change across the **full seed × year distribution**. A single hot number (e.g. MSRR +3.13) can be seed luck. This is the most important discipline in this project.
- **L-02 (MSRR scale):** MSRR weight *magnitude* carries no signal (scale-invariant Sharpe). Raw ensemble averaging therefore weights seeds by noise; this inflates Sharpe via lucky large-scale seeds. Verified in EXP-002.
- **L-03 (don't trust armchair dismissals):** An adversarial brainstorm (workflow `wk32jnjtd`) was prompted to treat overfitting as the enemy and predictably labeled capacity/temporal ideas "traps." But (a) its top pick failed when tested (EXP-002), and (b) Kelly et al.'s *Virtue of Complexity* (JF 2024) — same author lineage as the MSRR loss — argues bigger *helps* return prediction. **Test ideas; don't dismiss them on priors.**
- **L-04 (efficiency):** Some questions need zero training — EXP-002 answered a real question by re-analyzing saved models. Always check whether saved artifacts suffice first.
- **L-08 (screens can be FALSE POSITIVES — confirm with full seeds × years):** EXP-007 (5 seeds, 3 hand-picked years) showed rank-standardization +0.92 over base; EXP-009 (10 seeds, 8 years) **reversed it** — base 1.81 ≥ a2rank 1.42, rank wins only 3/8 years, IC tied. The screen caught *base* on unlucky 5-seed draws in exactly those 3 years (base 2014 L1: 1.07@5 → 3.18@10 seeds). The L1 SDF Sharpe is so noisy that **even a 10-seed ensemble over one year swings >2 points**, so a 5-seed/3-year screen has ~no power to resolve a ≤0.5 effect. Rules going forward: (1) screen with **≥10 seeds**; (2) judge the **row-mean over many years**, never a 3-year cherry-pick; (3) treat a single-arm screen "win" as a *hypothesis*, never a finding, until full confirmation; (4) the base/control MUST run at the same seed count in the same conditions (its noise is the thing you're testing against). Confirmation discipline (L-01) is what caught this — keep doing it.
- **L-07 (IC and SDF Sharpe can diverge — judge MSRR on its objective):** In EXP-007 per-month rank-standardization raised the MSRR **portfolio** L1 Sharpe (+0.92 mean vs base) while leaving **IC unchanged/slightly worse** (base 0.012 ≥ rank 0.010). An MSRR model outputs portfolio WEIGHTS, not a ranking — a better-conditioned input can improve the realized portfolio (weight structure, faster/cleaner convergence) without improving rank-correlation. So judge MSRR changes on the **L1 SDF Sharpe** (the optimized objective), use IC as a secondary stability check, and never assume the two move together. (For the MSE transformer the reverse holds — there IC is the stable signal; pick the metric that matches the loss.)
- **L-06 (period-matching + the 1-survivor verdict):** The paper's ~4 vs our ~2 is **~80% regime + scale, not data.** The 4.57 headline is *full-sample 1968–2022*; every model halves after ~2002, and their best **modern** transformer is **3.37** while their **linear BSV** baseline is **2.03** — exactly where our reproduction sits. So period-matched we already equal their linear modern-era baseline; closing to ~3.4 is a capacity+apparatus problem (deeper stack, monthly refit, CV'd ridge, rank-normed inputs), and their own decomposition says **depth does most of the work, cross-asset attention is the smaller increment.** When 20 enhancement candidates were put through adversarial scrutiny against our ±0.4–0.5 window-noise floor, **only per-month rank-standardization survived as `robust=true`** — most "obvious" knobs (width, GLU, extra heads, missingness, macro interactions, turnover penalty, LR schedules) have plausible effects *smaller than the sampling SE on 48–96 OOS months*, so they literally cannot be resolved on our windows. Corollary: the binding constraint is **statistical power (OOS months), not ideas** — which is why monthly refit (≈12× the OOS observations) is the highest-leverage *regime* change even though its per-estimate effect is itself subfloor.
- **L-05 (attention scaling, workflow wqq713yie):** The cross-section is a permutation-invariant SET (no positional encoding anywhere) and at N≈5000 it is **NOT attention-bound** (block fwd+bwd ≈1.4 GB; the 16–25 GB is gradients/AMP/optimizer/rolling-window data). So **KV-cache** (an autoregressive-decoding optimization — this is a single-pass non-causal encoder, nothing to cache) and **sequence-sparse** patterns (Longformer/BigBird sliding-window/strided — assume an ordering the set lacks; would mask arbitrary stocks and destroy signal-bearing cross-stock edges) are **misapplied here**. The long-context curse was tamed by **exact FlashAttention**, not approximation. Free exact win TAKEN: `need_weights=False` at the MHA calls → drops the discarded (1,N,N) tensor + dispatches the fused SDPA/FlashAttention kernel (applied in `experiments/exp_transformer.py`; outputs match to ~2.6e-7). The real O(N²) cliff is the **TEMPORAL extension**: naive tokens = stock×month → (N·T)² ≈ 29 GB fp16 scores at T=24 (~576×). **Design it out** — per-stock temporal encoder (GRU/TCN) collapsing each stock's trailing window to ONE token so token count stays N (cheapest; macro-GRU B-02 is the wired first step), or exact axial/factorized attention; never build the flat (N·T)² map then approximate. ISAB inducing points (B-03) only matter past ~10–15k stocks/month; Linformer is invalid (its fixed projection breaks when N varies monthly). Approximate attention must beat exact on the OOS seed×year distribution before adoption (L-01).

---

## Conventions

- **Code version:** every experiment records its git commit / branch. Config injected via `cluster/run_year.py` style wrappers; do not edit the training scripts to parameterize a run unless the change is the experiment.
- **Outputs:** `output/<exp>/...` on scratch (cluster) and pulled to local `output/` (gitignored). Summary CSVs + a one-line verdict are the durable record; large model/prediction files stay on scratch.
- **Run via the harness:** `sbatch --array=...%2 cluster/repro.sbatch` (parameterized), or `srun ... bash -lc 'source activate_cluster.sh && python ...'` for one-offs. See `cluster/` and the design doc.
- **Monitoring:** `sq`; `srun --jobid=$J --overlap nvidia-smi` (NOT `--gres=none` on Trillium); `seff <jid>` post-run.
- **Naming:** experiments `EXP-NNN`; backlog ideas `B-NN`; learnings `L-NN`.
