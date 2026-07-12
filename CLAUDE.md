# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Machine-learning models for cross-sectional monthly stock-return prediction, in the
Gu–Kelly–Xiu (2020) tradition. Three training scripts share one data pipeline and
differ only in model and loss. See `README.md` for results tables, the GKX/Kelly
citations, and the data-sourcing instructions; this file covers architecture and
how to work in the code.

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
defaults and assume the current working directory is the repo root:
`gkx_full/` (per-signal `signal_*.parquet`, returns, universe), `gkx_full/sector_mapping.csv`
(PERMNO → SIC 2-digit), and `welch_goyal_2024.xlsx` (macro predictors). See
`README.md` and `MSE_ind_1yr_report.md` §10 for the expected layout.

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
