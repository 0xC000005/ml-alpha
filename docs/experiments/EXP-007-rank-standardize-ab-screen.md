---
id: EXP-007
status: done
---

# EXP-007 — Rank-standardize A/B (B-11, Roadmap Tier 1), cheap screen, MSRR
- **Date:** 2026-06-03 · **Status:** ✅ done (**a2rank PASSES the screen → promote, with caveats**) · **Array:** 565492 (12 tasks, %2; driver `experiments/exp_main_msrr.py`) · **Metric:** L1 honest SDF Sharpe (raw + L1 both logged).
- **Setup:** 4 arms (base=A0 pooled-z / a1monthz=A1 / a2rank=A2 / a3rankgauss=A3) × 2014/16/18 × 5 seeds. Bar: A2 mean Δ-vs-base ≥ +0.5 AND ≥2/3 yrs positive AND A2-gap > A1-gap.
- **Result (L1 SDF Sharpe, mean over 3 yrs):** base **−0.02**, a1monthz **+0.34**, a2rank **+0.91**, a3rankgauss **+0.14**. a2rank Δ-vs-base **+0.92** (≥0.5 ✓); beats base 2/3 (2014 +1.35, 2018 +1.47; 2016 tie −0.05) ✓; rank-gap 0.92 ≫ monthz-gap 0.35 ✓ → **clears the bar**. Win **survives the honest L1** combiner (a2rank raw mean 0.81 ≈ L1 0.91 → *not* lucky-seed) and a2rank converges **faster** (fewer epochs; base 2018 ran 231 ep near the 300 cap).
- **Caveats:** (1) **IC flat-to-worse** — base mean IC 0.012 ≥ a2rank 0.010; the win is in the **portfolio** Sharpe, not the ranking (→ L-07). (2) Mean driven by 2 good years; **2016 is negative for all rank arms** (hard year). (3) 3 single-year windows are thin (SE ~±0.5–1.0/yr). (4) **a3rankgauss (tail-restore) does NOT help** → A3 hybrid killed; pure rank ≥ rank_gauss. (5) **a1monthz uniquely positive 3/3** (0.61/0.18/0.22) — the robustness arm; survives 2016 where everything else fails.
- **Conclusion:** promote **a2rank** to a full 8yr×10-seed confirmation (gated) vs **base** (+ a1monthz as the robustness comparator); judge on L1 SDF Sharpe over the ~±0.4 96-month floor, watch IC. Per-seed `.pt` saved on scratch (`output/exp/rank/*/models`) — Tier-0 artifact blocker now resolved.
- **Next:** EXP-009 full confirmation (GATED).


## Prep notes (from the former Planned-experiments block)

- **EXP-007 — Rank-standardize A/B (B-11, Roadmap Tier 1)** — the one `robust=true` lever. Prep ✅ **this session** (CPU-validated, no cluster). New code, all in `experiments/`, production frozen:
  - `feature_scalers.py` — `make_scaler(kind)`: A0 `pooled_z` (byte-identical to `TransformerFeatureScaler`, proven), A1 `month_z`, A2 `rank`→[−0.5,0.5], A3 `rank_gauss` (van der Waerden). Transforms the 95 signals only; dummies stay unscaled.
  - `msrr_combine.py` — L1 equal-vote combiner + honest `sdf_sharpe` (raw reproduces production; **B-05 wired**). Matches `cluster/ab_msrr_norm.py`.
  - `exp_main_msrr.py` — MSRR driver (reuses frozen `train_model_msrr`/`evaluate`), configurable scaler+combiner, saves per-seed `.pt`, logs raw+L1 Sharpe. `--dry` builds model+scaler with no data.
  - `validate_msrr_prep.py` — all 4 checks pass. `configs/rank_ab.jsonl` = **12 tasks** (4 arms × 2014/16/18 × 5 seeds).
  - **Held command:** `sbatch --array=0-11%2 --export=ALL,CONFIG_FILE=experiments/configs/rank_ab.jsonl,RUNNER=experiments/exp_main_msrr.py experiments/sweep.sbatch`. **Bar:** A2 mean Δ-vs-base ≥ +0.5 AND ≥2/3 yrs positive AND A2-gap > A1-gap (else promote A3 if it clears, else shelve). Collect: `collect_screen.py output/exp/rank msrr_transformer_summary.csv`.
