# ml-alpha — Backlog

Living state — prioritized hypotheses (B-NN). Unlike `docs/log/` and
`docs/decisions/`, this file is meant to be mutated in place (status checkboxes),
it is a todo list, not history.

### Enhancement Roadmap (2026-06-03, design-only — workflow `wf_2f4b6eee`, 27 agents, 20 candidates → 1 survivor)
Goal: **robustly** higher OOS Sharpe for the MSRR transformer (not the lucky raw-ensemble number). 5 lenses proposed → deduped to 20 → adversarial scrutiny vs the ±0.4–0.5 window-noise floor. **Exactly one candidate survived as `robust=true`.**

- **Tier 0 — measurement infra (do FIRST, not a Sharpe claim):** adopt the **L1 equal-vote combiner** (normalize each seed's per-month weights to ‖w‖₁=1 before averaging; also L1 the final ensemble) as the *sole* combiner + comparison denominator. Honest point estimate is flat-to-lower but dispersion shrinks 30–50%. Gate on EXACTNESS (raw reproduces saved `sdf_sharpe` ≤0.02) + VARIANCE-REDUCTION, **not** a Sharpe bar. ⚠️ **Artifact blocker:** local `output/repro/msrr_*/` has only metrics CSVs — **no per-seed `.pt`**; either pull from Trillium (gated) or regenerate via the Tier-1 retrain.
- **Tier 1 — the one survivor (`robust=true`, medium impact):** **per-month cross-sectional rank-standardize the 95 signals to [−0.5,0.5]** (missing→median 0), replacing pooled-z+5σ. Closes a 3-axis divergence from the AIPM input contract (pooled→monthly, mean→median, z→rank). Cheap, no params, reversible. **Footgun:** transform signals only, never the 74 industry dummies. Mechanism verified live: LayerNorm normalizes per-token across `d_model`, *not* per-feature across stocks, so each char's cross-sectional dispersion leaks into attention keys/queries. **Screen:** 4 arms (A0 pooled-z control / A1 per-month-z / A2 rank / A3 winsorized-rank) × 2014/16/18 × 5 seeds, %2, L1-metric. **Bar:** A2 mean Δ-vs-A0 ≥ +0.5 AND ≥2/3 yrs positive AND A2-gap > A1-gap (else the win is just per-month standardization, not rank); if A2 fails but A3 clears, promote the hybrid.
- **Tier 2 — one capacity bet, conditional:** **MSRR depth ladder K∈{2,3}** (fixed d_model=32/d_ff=64; do **not** widen). Judge on L1-normalized per-year **median**, raw-vs-norm side by side. **PROMOTE only if monotone + normalized + multi-year:** median beats base ≥2/3 yrs (incl. 2014), ≥+0.5 in ≥2 yrs, K=3≥K=2. **KILL** on raw-only, 2014-only, or L3<L2 (the MSE-proxy pathology). Companions (paper-scaled init, QK-norm/RMSNorm/LayerScale) only if depth first shows a real trend.
- **Tier 3 — needs building, only if Tier 1–2 stall:** **paper-regime training** — build `exp_main_msrr.py` (current `exp_main.py` is MSE-only), screen the cheaper **rolling-window length** (60/120-mo vs expanding) *before* the expensive **monthly/quarterly refit**; co-tune head-ridge CV with depth. This attacks the modern-regime gap directly (regime-local data) but every verdict is subfloor on a single year — full-run only.
- **Discarded (inside the noise floor / confounded):** FFN-width d_f (≤+0.4k params), GLU/SwiGLU (not param-matched — capacity confound), single-head/temperature, vol-target combiner, scale-invariant early-stop, ridge-λ CV (no-op at lr 7.5e-5), missingness mask (~0 corr, width-in-a-bad-regime), macro×char FiLM, signal-set hygiene, turnover penalty (no-op on scale-invariant Sharpe — answer from saved artifacts, not a new loss), cosine-LR, longer val window.
- **Sequenced plan:** ① freeze L1 metric → ② build `RankFeatureScaler` + MSRR routing → ③ cheap-screen the 4-arm rank A/B (this retrain also regenerates the per-seed `.pt` Tier 0 needs) → ④ cheap-screen MSRR depth K∈{2,3} → ⑤ (cond.) init/stability → ⑥ promote ≤1 survivor to a full 8yr×10-seed run = **first legitimate cluster trigger, STOP for go-ahead** → ⑦ (if stall) Tier-3 driver. Judge every A/B on the L1 seed×year distribution (median+IQR), never raw mean, never one window.

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


## Planned-experiments budget

### Planned experiments (GATED — full plan: `docs/superpowers/plans/2026-06-03-model-improvement-experiments.md`)
Prep (code + local CPU validation) done where noted; **no cluster job runs without explicit go-ahead.**
**All prep code is implemented and LOCALLY VALIDATED (CPU/local GPU, no cluster).** Runners: `run_experiment.py` (capacity/GLU, via main()) and `exp_main.py` (missingness/temporal/monthly). Configs: `gen_configs.py` → `experiments/configs/*.jsonl`. Submit harness: `sweep.sbatch` (generic array). Default model is bit-identical to frozen.

- Total cheap-screen budget ≈ **20–30 GPU-h, ≤2 GPUs at a time**. Full runs only for screens that pass. **Every `sbatch` is GATED behind explicit go-ahead.**
