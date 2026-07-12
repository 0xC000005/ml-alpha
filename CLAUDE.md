# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Machine-learning models for cross-sectional monthly stock-return prediction, in the
Gu–Kelly–Xiu (2020) tradition. Three training scripts share one data pipeline and
differ only in model and loss. See `README.md` for results tables, the GKX/Kelly
citations, and the data-sourcing instructions; this file covers architecture and
how to work in the code.

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

## Commands

There is **no build, no lint, and no test suite** — these are research scripts run directly.

```bash
python train_nn.py              # FFN (NN5), MSE loss
python train_transformer.py     # Cross-sectional Transformer, MSE loss
python train_transformer_msrr.py # Cross-sectional Transformer, MSRR loss
```

A CUDA GPU is assumed (AMP `GradScaler("cuda")`, ~16GB VRAM). On CPU the scripts
run but log a warning and are very slow.

**Configuration is in code, not flags.** Each script has a `@dataclass` config at the
top — `Config` (train_nn.py:30), `TransformerConfig` (train_transformer.py:44),
`MSRRConfig` (train_transformer_msrr.py:58). Edit fields there to change test years,
hyperparameters, the validation window, seed count, etc. There is no argument parser.

**Entry-point gotcha:** `train_nn.py`'s `if __name__ == "__main__"` calls
`run_experiments()` (train_nn.py:1166), **not** `main()`. `run_experiments()` overrides
`test_years` to `2001–2009` and runs the single `MSE_noind_1yr` experiment from its
`experiments` list (its docstring still says "8 experiments" — stale; the grid was
trimmed to one). `main()` (train_nn.py:800) is a separate, full-config driver kept for
reference. The two Transformer scripts run `main()` directly.

To run a single year, set `test_years=[YYYY]` in the config; to change the experiment in
`run_experiments`, edit its `experiments` list.

## Module dependency hierarchy (the key structural fact)

`train_nn.py` is the foundation. The other two scripts import its data pipeline rather
than reimplementing it, so the dependency graph is layered, not three parallel copies:

- **`train_nn.py`** — owns the entire data pipeline (`load_returns`, `load_universe`,
  `load_signals`, `load_macro`, `load_sector_mapping`, `build_long_panel`,
  `build_industry_dummies`, `FeatureScaler`), shared utilities (`setup_logging`,
  `set_seed`, `compute_cross_sectional_ic`, `compute_oos_metrics`), the `Config`
  dataclass, and the FFN model `GKXNet`.
- **`train_transformer.py`** — `from train_nn import (...)` for the whole data pipeline +
  `Config`. Adds the Transformer-specific pieces: `CrossSectionalTransformer`,
  `MonthGroupedData` (keeps observations grouped by month for per-month attention),
  and `TransformerFeatureScaler`.
- **`train_transformer_msrr.py`** — imports the data pipeline from `train_nn` **and**
  `CrossSectionalTransformer` / `MonthGroupedData` / `TransformerFeatureScaler` from
  `train_transformer`. It is the same model and data as the MSE Transformer; only the
  **loss, train/eval loop, and portfolio metric** differ.

Implication: changes to data loading, feature construction, or `Config.signal_names` /
`macro_names` in `train_nn.py` propagate to all three scripts. The two Transformer
scripts must be run from the repo root so these imports resolve.

## Data pipeline

External data is **not** in the repo (gitignored). Paths are hardcoded as `Config`
defaults (train_nn.py:32-34) and assume the current working directory is the repo root:
`ml_alpha_data/gkx_full/` (per-signal `signal_*.parquet`, `returns.parquet`,
`universe.parquet`), `ml_alpha_data/gkx_full/sector_mapping.csv` (PERMNO → SIC 2-digit),
and `ml_alpha_data/welch_goyal_2024.xlsx` (macro predictors). See `README.md` and
`MSE_ind_1yr_report.md` §10 for the expected layout.

`build_long_panel` (train_nn.py:255) converts wide monthly frames into long-format
NumPy arrays: `stock_features (N, 95)`, `macro_features (N, 8)`, `targets (N,)`,
`month_ids (N,)` as `yyyymm` integers, `permno_ids (N,)`. The **target is next month's
excess return** — signals at month *t* predict returns at *t+1* (train_nn.py:300), the
universe is taken at *t*, and rows with NaN next-month returns are dropped.

Feature scaling is **fit on the training split only** then applied to val/test (no
look-ahead). Both scalers standardize, impute NaN→0 (= the mean post-standardization),
and clip to ±`clip_std` (default 5σ). They then diverge:

- **FFN (`FeatureScaler`, train_nn.py:381)** materializes an 863-dim block:
  95 signals + 8 macro + **760 signal×macro interactions** (95×8, computed in chunks to
  cap memory). The 74 industry dummies are built separately (`build_industry_dummies`)
  and concatenated **unscaled** after scaling, and are **optional** — gated by each
  experiment's `use_dummies`. With dummies on, `Config.n_features = 937`; the active
  `run_experiments` config (`MSE_noind_1yr`) runs **without** them → 863 dims. (The 8
  macros are themselves *derived* in `load_macro` from raw Welch–Goyal columns:
  dp, ep, bm, ntis, tbl, tms, dfy, svar.)
- **Transformer (`TransformerFeatureScaler`)** does *not* form interactions. Each stock's
  per-month vector is 95 signals + 74 industry dummies = **169 dims**, plus the 8 macro
  values shared across the month; attention learns cross-stock interactions instead.

## Models

- **`GKXNet`** (train_nn.py:474) — feedforward MLP, `NN5` = layers `(32,16,8,4,2)` →
  1 output, MSE loss. Each hidden layer is `Linear → BatchNorm1d → ReLU → Dropout`
  (Kaiming-normal init). The other NN1–NN4 depths are commented out in
  `Config.architectures`. A per-architecture-per-year L1 penalty is grid-searched over
  `l1_lambdas` via validation loss in `main()` only — `run_experiments()` skips it.
- **`CrossSectionalTransformer`** (train_transformer.py:163) — processes **all stocks in
  one month together** via self-attention (the month is the sequence, stocks are tokens).
  Pre-norm block: `stock_proj(169→d_model) + macro_proj(8→d_model)` (additive) →
  LayerNorm → MultiheadSelfAttention → residual → LayerNorm → FFN(d_model→d_ff→d_model)
  → residual → LayerNorm → Linear(d_model→1). Defaults `d_model=32, n_heads=4,
  n_layers=1, d_ff=64` (~14K params). Xavier-normal init.
- The MSRR script reuses `CrossSectionalTransformer` unchanged; see below.

## Rolling-window training & ensembling

All three use an **expanding-window, one-year-out** scheme: train from `train_start`
(1975) up to a validation window, validate on the most recent `val_years` (10 for the
FFN MSE configs, 1 for MSRR), then test on a single held-out year; repeat per year in
`test_years`. Each (year) model is an **ensemble of `n_seeds=10`** independently-seeded
fits, averaged. Early stopping watches the validation metric (`patience`, only after
`min_epochs`). Data is loaded **once** up front and reused across all years/experiments;
raw frames are freed with `gc.collect()` after `build_long_panel`.

## MSRR specifics (train_transformer_msrr.py)

The loss directly optimizes the Sharpe ratio of an SDF portfolio instead of prediction
accuracy (Kelly et al. 2025):

- **Loss** (`msrr_loss_month`, :104): per month, `(1 − wᵀR)²` where `w` = raw model
  outputs used **directly as portfolio weights** and `R` = excess returns. Averaged
  over months. The SDF portfolio return is `wᵀR`; Sharpe is scale-invariant in `w`.
- **`compute_sdf_portfolio_metrics`** (:197): evaluation uses the raw weights as-is
  (no decile sorting), monthly return `wᵀR`, annualized Sharpe `mean/std·√12`.
- **Split optimizer** (:250): AdamW with **no weight decay on the Transformer body** and
  `ridge_lambda` (1e-3) weight decay on the **output head only** — the ridge-on-head
  convention from the paper. Also uses gradient accumulation (`grad_accum_steps=4`) and
  gradient-norm clipping (`max_grad_norm=1.0`). Early stopping is on validation MSRR loss.

## Outputs

Scripts create `output/{logs,models,predictions,metrics,features}/`. Each writes a
summary CSV under `output/metrics/` (e.g. `experiments_summary.csv`,
`transformer_summary.csv`), plus per-year predictions, saved models, and the pickled
fitted scaler. `output/` is gitignored. Logs are timestamped and stream to both file and
stdout (`setup_logging`).

## How research work is done here (read before writing any experiment)

`train_nn.py`, `train_transformer.py`, and `train_transformer_msrr.py` are **frozen** — a
PreToolUse hook denies edits to them. All new work goes in `experiments/`, which imports
the frozen scripts rather than modifying them. Drivers take a one-line JSON config
(`experiments/configs/*.jsonl`, generated by `gen_configs.py`); `sweep.sbatch` runs them
as a SLURM array; `collect_screen.py` tabulates the results.

Every run writes `manifest.json` into its output dir (`experiments/manifest.py`): git SHA,
dirty flag, config, seeds, environment, and SHA-256 of every data input. **A run whose
manifest says `dirty: true` is not reproducible** — the code that ran is not the code that
is committed. `docs/log/` (narrative) + `docs/experiments/EXP-NNN-*.md` (per-experiment
writeups) + `docs/decisions/L-NN-*.md` (distilled findings) are the lab notebook — a
number becomes a "finding" only in one of these three places, never edited in place once
written (a correction is a new entry/file, never a rewrite — see `docs/log/INDEX.md`).
Pre-2026-07-12 history: `docs/log/archive/RESEARCH_LOG-2026-06-02-to-2026-07-12.md`.

**Never run `sbatch`/`srun` without an explicit human go-ahead.** A hook enforces this.

## The statistical error catalogue (this repo has made every one of these)

The models here are small; the noise is enormous. Every real mistake in this project has
been a *statistical* one that read as perfectly reasonable code. Before you compute,
quote, or believe any performance number, check all seven:

1. **Never average annual Sharpe ratios.** `collect_screen.py` takes a row-mean across
   years. That is not the portfolio's Sharpe and it is biased upward — the headline
   "1.81" is such a mean; the pooled 88-month series gives ≈**1.41**. Report the Sharpe of
   the **pooled monthly return series** (`portfolio_analysis.py` already prints both).
2. **Seeds are not observations.** Ensemble seeds cut estimator variance; they add zero
   economic sample size. Only months do.
3. **Annualize the standard error.** SE(annualized Sharpe) = `√12 · √((1 + ½·SR_m²)/T)`,
   T in months. One test year (T=11) → **SE ≈ 1.11**. Eight years (T=88) → **SE ≈ 0.39**.
   A single-year Sharpe is nearly pure noise. (`RESEARCH_LOG.md` once used the *monthly*
   SE and understated the noise by √12 ≈ 3.46×.)
4. **`month_ids` are the FEATURE month `t`; the return is realized at `t+1`.** Any merge
   against an external monthly series must shift first. `experiments/ff5_regression.py`
   currently does **not**, so its alpha and factor t-stats are invalid until fixed.
5. **The universe conditions on the future.** train_nn.py:305-315 takes the universe at
   `t`, then drops stocks whose `t+1` return is NaN. If NaN correlates with delisting, the
   worst outcomes are silently removed. Unresolved — do not trust a level Sharpe until it is.
6. **January is never evaluated.** Drivers test Jan–Nov features → Feb–Dec returns.
7. **Pair your comparisons.** A/B claims need a paired t-stat on (year, seed). Worked
   example: EXP-009 base vs a2rank → mean diff +0.391, sd 1.016, **t = 1.09** over 8 years.
   That is *not significant*: the honest conclusion is **"no detectable difference"**, not
   "base wins". Below |t| ≈ 2.4 you have found nothing, and saying otherwise is how EXP-007
   became a false positive in the first place.

Corollary: an improvement smaller than ~0.5 Sharpe **cannot be resolved** on these windows.
The prior for any such result is "this is noise."

## Agents & model tiering

Route work to the cheapest model that can do it safely — but statistics is never the cheap
tier. Custom agents live in `.claude/agents/`:

| agent | model | use for |
|---|---|---|
| `cluster-monitor` | haiku | watching a sweep in flight — queue state, GPU utilization, failures, completion. Never submits, never interprets. |
| `results-triager` | haiku | collecting/tabulating finished sweeps. Reports numbers, never judges them. |
| `experiment-implementer` | sonnet | writing experiment code from a settled plan. Never decides research questions. |
| `stats-gatekeeper` | **opus** | **any** number, comparison, or conclusion before it enters `docs/experiments/` or `docs/decisions/`. |

The handoff runs downhill: `cluster-monitor` (is it done?) → `results-triager` (what are the
numbers?) → `stats-gatekeeper` (do they mean anything?) → `docs/experiments/`/`docs/decisions/`. The two haiku
agents are structurally forbidden from judging results, because that is where cheap models
silently go wrong on this repo (L-09).

For an independent second opinion, use the `verify-with-codex` skill — it calls
`mcp__codex__codex` (gpt-5.6-sol) live against this repo. Codex found the mean-of-ratios
bug, the √12 error, and the off-by-one month that a Claude-only review missed. Use it
before spending GPU time and before recording a result.

## Conventions & gotchas

- Run Transformer scripts from the **repo root** — they `import` from `train_nn` /
  `train_transformer` as sibling modules.
- All file paths in the configs are **relative** and assume CWD = repo root.
- The data is not redistributable (gitignored); the scripts will fail at load time until
  `gkx_full/`, `welch_goyal_2024.xlsx`, and `sector_mapping.csv` are present.
- `Config.n_industries = 74` (SIC 2-digit) is assumed consistent across loaded data;
  industry dummy construction depends on `sector_mapping.csv` coverage.
- Markdown reports (`*_report.md`, `gkx_paper_nn_settings.md`, `nn_all_parameters.md`)
  are the research write-ups / documented hyperparameter settings, not executable config.
