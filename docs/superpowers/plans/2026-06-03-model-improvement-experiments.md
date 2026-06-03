# Model-Improvement Experiments — Implementation Plan

> **For agentic workers:** execute task-by-task; compute steps are GATED — do not submit any cluster job until the user gives an explicit go-ahead. Prep steps (code + local CPU validation) may proceed freely.

**Goal:** Test whether bigger/deeper, virtue-of-complexity, temporal, more-sophisticated, and monthly-refit variants beat the frozen baseline — efficiently, with cheap screens before any full run.

**Architecture:** All experiment code lives in `experiments/` on branch `experiments`; the production scripts (`train_transformer.py`, `train_transformer_msrr.py`, `train_nn.py`) stay frozen. A configurable `ExpTransformer` + a monkeypatching runner inject any variant into the unmodified training loop. Each direction is a config, not a fork.

**Tech stack:** PyTorch 2.12 / H100 (Trillium, `cluster/` harness), local CPU (RTX 3070 Ti) for validation.

---

## ✅ Prep already done & locally validated (zero cluster)

| File | What | Validated |
|---|---|---|
| `experiments/exp_transformer.py` | configurable model: `n_layers` (depth), `d_model/d_ff/n_heads` (width), `ffn_kind` gelu/glu | `n_layers=1,gelu` is **bit-identical** to frozen model (Δ=0, 14,369 params); deeper/wider/glu scale + fwd/bwd OK |
| `experiments/run_experiment.py` | inject any JSON config into the frozen `main()` via monkeypatch (model + hyperparams), `--dry` builds-only | dry-built 14,369 / 112,065 / 70,401-param variants correctly |
| `experiments/validate_local.py` | the CPU validation above | passes |

This unblocks **Directions 1, 2, and the GLU half of 4** with no further model code. Remaining prep (Directions 3, 5, missingness, VoC-ridge) is specified below as code.

---

## Compute discipline (binding)
- **Cheap screen first:** few configs × **3 representative years** (2014 hard, 2016 weak, 2018 strong) × **3 seeds**. Promote to a full 8-yr × 5–10-seed run ONLY on a consistent, across-the-distribution signal.
- **Judge across seed × year**, never a point estimate.
- **1 GPU/quarter-node, `--array ...%2`.** Reuse saved artifacts where possible.
- Every job is **GATED** behind explicit user go-ahead.

Timing anchors from the reproduction (per seed, H100): MSE ≈ 6 min, MSRR ≈ 9 min at the base size. Bigger models scale ~linearly in `n_layers` and ~quadratically-ish in `d_model` (attention over ~5000 tokens). Estimates below use ≈2× for `d64` and ≈3× for `L3`.

---

## Phase 1 — Capacity + Virtue of Complexity (your ideas 1 & 2)  ·  status: READY

**Hypothesis:** depth/width raise robust OOS Sharpe/IC (Kelly–Malamud–Zhou, *Virtue of Complexity*, JF 2024). VoC specifically predicts gains persist as params grow **with ridge shrinkage**.

**Code:** none needed for width/depth (done). VoC ridge = add weight decay to the optimizer. The MSE path uses plain Adam (`train_transformer.py` train loop); add an optional `weight_decay` already supported by `torch.optim.AdamW`. To stay non-invasive, expose `weight_decay` via the config monkeypatch and (one small addition to `run_experiment.py`) swap Adam→AdamW with that decay. *Prep task P1.*

**Cheap screen (MSE first — cheaper, better-behaved):**
- Configs (5): `L1d32` (baseline), `L2d32`, `L3d32`, `L1d64`, `L2d64`.
- Years 2014/2016/2018, seeds 3 → **15 tasks**.
- Est: baseline ~18 min/task; L3/d64 up to ~70 min/task → **~7–10 GPU-h** total, `%2`.
- Held command: `sbatch --array=0-14%2 experiments/sweep_capacity.sbatch` (sbatch + config list = prep task P2).

**Success criterion:** a config whose **median across 3 seeds beats baseline in ≥2 of 3 years** AND doesn't blow up dispersion → promote to 8-yr × 5-seed full run (~12–20 GPU-h). Otherwise: capacity doesn't help here (a real, publishable negative result), and we stop.

---

## Phase 2 — More sophisticated architecture (your idea 4)  ·  GLU ready; missingness needs prep

**2a. GLU/SwiGLU FFN** — implemented (`ffn_kind="glu"`). Screen as one extra config in the Phase-1 grid (cheap).

**2b. Missingness-aware inputs (B-06)** — the 26% NaN→0 impute erases size/quality info. *Prep task P3:* add per-signal "was-missing" indicators.
- Data side (non-invasive): compute the mask in the runner before scaling. The cleanest hook is a small `experiments/missingness.py` that, given the raw `stock_features`, returns a `(N, k)` 0/1 mask for the ~15 signals with >40% missingness, concatenated UNSCALED onto the stock input (exactly like industry dummies). Bump `ExpTransformer` stock_proj `in_dim` by `k`.
- Requires the runner to intercept the panel before `build_split` (a small wrapper around `build_industry_dummies`-style concatenation). Code sketch in Appendix A.
- Screen: baseline vs +missingness, 3 yrs × 3 seeds (MSE) → **9 tasks, ~3–4 GPU-h**.

**Success:** robust IC gain across the distribution.

---

## Phase 3 — Temporal dimension (your idea 3)  ·  needs prep (data + model)

**Hypothesis:** modeling how state evolves over time (rather than per-month snapshots) adds signal. Start with the **lowest-overfit-risk** variant per the literature (Chen–Pelger–Zhu put recurrence on *macro*, not per-stock characteristics): a small GRU over the trailing `L`=12–24 months of the 8 macro series producing an economic-state vector that replaces the additive macro input.

*Prep task P4 (data):* the panel currently exposes one macro row per month. Add `experiments/macro_window.py` to build, per test-month `t`, the `(L, 8)` trailing macro matrix from the already-loaded `macro` frame (no new data; just windowing). Feed it through a `nn.GRU(8, d_model)` whose last hidden state replaces `macro_proj` output. Model hook: add `macro_temporal="gru"` to `ExpTransformer` (a few lines; the GRU consumes a `(1, L, 8)` seq). Code sketch in Appendix B.
- Shared across stocks → few hundred extra params → low cross-sectional overfit.

**Cheap screen:** baseline vs macro-GRU(L=12) vs macro-GRU(L=24), 3 yrs × 3 seeds → **9 tasks, ~4–6 GPU-h**. Test on **MSRR too** here (macro regime is most plausibly an MSRR-timing signal), +9 tasks.

**Success:** robust Sharpe/IC gain. **Explicitly a bigger bet** — if flat (likely per the literature on this panel), record the negative and stop before the heavier per-stock-sequence variant.

---

## Phase 4 — Monthly refit (your idea 5)  ·  needs prep (loop) + honest power check

**Hypothesis:** retraining every month (vs yearly) on fresher data raises OOS Sharpe.

**Reality check (do this FIRST, no compute):** rebalancing is already monthly; this is 12× the *training*. Over the available years the per-year Sharpe SE (~0.33) likely exceeds any monthly-refit effect → **may be statistically undetectable**. *Prep task P5a:* a 1-page power note in the log; only proceed if a minimal test is justified.

**Minimal test (if approved):** *Prep task P5b:* `experiments/run_monthly.py` — same monkeypatch, but the refit loop steps **monthly within ONE test year** instead of yearly (retrain on 1975→month-2, predict month). Restrict to **1 year (2018) × 3 seeds, baseline size** to bound cost. Code sketch in Appendix C.
- Est: 11 refits × ~18 min (3 seeds, but each refit trains on ~all history) ≈ **~3–4 GPU-h for one year**. Compare its 2018 Sharpe to the yearly-refit 2018 Sharpe (already have it).

**Success:** a clear, across-seed Sharpe gain on 2018 that exceeds the noise band. If marginal, do NOT scale to all years (the cost/benefit fails the discipline rule).

---

## Sequencing & total cheap-screen budget (all GATED)

| Phase | Direction(s) | Prep status | Cheap-screen GPU-h | Gate |
|---|---|---|---|---|
| 1 | capacity + VoC (1,2) | ready (P1,P2 small) | ~7–10 | go-ahead |
| 2 | sophistication (4) | GLU ready; P3 | ~3–4 | go-ahead |
| 3 | temporal (3) | P4 | ~8–12 | go-ahead |
| 4 | monthly refit (5) | P5 + power note | ~3–4 (1 yr) | go-ahead |

**Total cheap-screen ≈ 20–30 GPU-h, ≤2 GPUs held at a time.** Full runs only for screens that pass, costed then. Recommended order: **1 → 2 → 3 → 4** (cheapest/highest-evidence first; each phase's result informs the next).

## Remaining prep tasks (no compute) — to finish before go-ahead
- **P1** AdamW+weight_decay knob in `run_experiment.py` (VoC ridge). *(small)*
- **P2** `experiments/sweep_capacity.sbatch` + config list (array index → config). *(small)*
- **P3** `experiments/missingness.py` + runner hook + model `in_dim` bump; CPU-validate. *(small)*
- **P4** `experiments/macro_window.py` + `ExpTransformer(macro_temporal="gru")` + runner hook; CPU-validate. *(medium)*
- **P5** power note (P5a) + `experiments/run_monthly.py` (P5b); CPU dry-validate. *(medium)*
- Log stubs `EXP-003…007` in `RESEARCH_LOG.md` (one per phase) before any run.

## Self-review
- **Coverage:** all 5 user directions have a phase. ✔
- **Compute safety:** every run gated; cheap-screen-first; ≤2 GPUs; reuse artifacts. ✔
- **Non-invasive:** production scripts frozen; all changes in `experiments/`. ✔
- **Honesty:** negative results are valid outcomes and explicitly allowed to stop a phase. ✔

## Appendix — code sketches for remaining prep
**A. Missingness:** before `nan_to_num`, `mask = np.isfinite(stock_features[:, cols]).astype(np.float32)`; concat to the model input UNSCALED alongside industry dummies; `ExpTransformer` `stock_proj = Linear(n_signals + n_industries + k, d_model)`.
**B. Macro-GRU:** `self.macro_gru = nn.GRU(n_macro, d_model, batch_first=True)`; `_, h = self.macro_gru(macro_seq); macro_embed = h[-1]`; replaces `self.macro_proj(macro)`. Runner builds `macro_seq=(1,L,8)` from trailing months.
**C. Monthly refit:** outer loop `for m in test_months:` with `train_mask = month_ids <= (m shifted back 2 months)`, `val = previous month`, `test = m`; reuse `build_split`, `train_model`, `evaluate`; ensemble seeds per month; write monthly rows.
