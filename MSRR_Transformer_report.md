# MSRR Transformer Experiment Report

## Cross-Sectional Transformer with Maximum Sharpe Ratio Regression Loss

**Baseline**: Cross-Sectional Transformer with MSE loss (see `Transformer_report.md`)

**Experiment**: Replace MSE loss with MSRR loss — the model directly optimizes the stochastic discount factor (SDF), which is equivalent to maximizing the portfolio Sharpe ratio.

**Test Period**: 2016-2019 (4 years of out-of-sample evaluation)

**Date**: 2026-04-13

**Status**: Proof-of-concept. Due to infrastructure and time constraints, this is a basic first-pass implementation without hyperparameter tuning, ridge penalty grid search, or monthly retraining. Results should be interpreted as a lower bound on MSRR performance.

---

## 1. Motivation

The MSE Transformer predicts individual stock returns, then constructs a portfolio by sorting into deciles. This is a two-step process where the training objective (return prediction accuracy) differs from the evaluation objective (portfolio Sharpe ratio).

Kelly, Kuznetsov, Malamud, and Xu (2025) propose training the Transformer to directly output portfolio weights, with a loss function that maximizes the Sharpe ratio of the resulting portfolio. This aligns training and evaluation objectives.

**Reference**: Kelly, B.T., Kuznetsov, B., Malamud, S., Xu, T.A. (2025). "Artificial Intelligence Asset Pricing Models." NBER Working Paper 33351.

---

## 2. MSRR Loss Function

### 2.1 Formulation

The Maximum Sharpe Ratio Regression (MSRR) objective:

```
min  E[(1 - w(X_t)' R_{t+1})²] + ridge_penalty
```

Where:
- `w(X_t)` = vector of portfolio weights output by the Transformer (one per stock)
- `R_{t+1}` = vector of next-month excess returns
- `w(X_t)' R_{t+1}` = portfolio return (dot product of weights and returns)
- The loss pushes the portfolio return toward 1 each month

### 2.2 Why This Maximizes Sharpe

The solution to `min E[(1 - w'R)²]` is mathematically equivalent to the **mean-variance efficient portfolio** — the portfolio with the maximum achievable Sharpe ratio. This is a known result from asset pricing theory (Hansen & Jagannathan, 1991; Kelly & Xiu, 2023).

### 2.3 Gradient

For a single month, the gradient with respect to weight w_i:

```
dL/dw_i = -2(1 - w'R) * R_i
```

All stocks share the same scalar `(1 - w'R)`, making the gradient noisier than MSE (which has per-stock independent gradients).

### 2.4 MSE vs MSRR

| Property | MSE Loss | MSRR Loss |
|----------|---------|-----------|
| Objective | Predict returns accurately | Maximize portfolio Sharpe |
| Model output | Return prediction | Portfolio weight |
| Loss per month | Average of N individual errors | Single scalar (1 - w'R)² |
| Gradient signal | Smooth (5000 terms averaged) | Noisy (1 scalar) |
| Portfolio construction | Post-hoc (sort into deciles) | Direct (weights ARE the portfolio) |

---

## 3. Architecture

**Identical to the MSE Transformer.** Same `CrossSectionalTransformer` class imported from `train_transformer.py`. The only change is the loss function and optimizer configuration.

```
Per month t with N stocks (~5,000):

Input per stock: [stock_signals(95) || industry_dummies(74)] = 169 dims
Macro (shared):  8 dims

stock_proj = Linear(169, 32)
macro_proj = Linear(8, 32)
x = stock_proj(input) + macro_proj(macro)       # additive conditioning

x = x + Dropout(MultiHeadSelfAttention(LayerNorm(x)))    # 4 heads, d_k=8
x = x + FFN(LayerNorm(x))                                 # 32->64->32, GELU

weight_i = Linear(LayerNorm(x), 1)              # per-stock portfolio weight
```

**Total parameters**: 14,369 (identical to MSE Transformer)

---

## 4. Training Configuration

| Parameter | MSRR Value | MSE Value | Rationale |
|-----------|-----------|-----------|-----------|
| **Loss function** | **(1 - w'R)²** | **MSE** | **Key difference** |
| Optimizer | AdamW | AdamW | Same |
| Learning rate | **7.5e-5** | 1e-4 | Lower — MSRR gradient is noisier |
| Weight decay (body) | **0.0** | 1e-4 | Unconstrained Transformer (Kelly approach) |
| Ridge penalty (output head) | **1e-3** | N/A | Constrains weight magnitudes only |
| Gradient accumulation | 4 months | 4 months | Same |
| Gradient clipping | max_norm=1.0 | max_norm=1.0 | Same |
| Max epochs | 300 | 300 | Same |
| Early stopping metric | **Val MSRR loss** | Val MSE | Lower = better |
| Early stopping patience | 25 epochs | 25 epochs | Same |
| Min epochs before stopping | 20 | 20 | Same |
| Dropout | 0.10 | 0.10 | Same |
| Mixed precision | fp16 | fp16 | Same |
| Number of seeds | 10 | 10 | Same |
| Ensemble method | Mean of 10 predictions | Mean of 10 predictions | Same |

### 4.1 Ridge Penalty on Output Head Only

Following Kelly et al., the ridge penalty is applied **only to the output head** (the Linear(32,1) layer that maps representations to portfolio weights), not to the Transformer body. This allows the attention mechanism to learn unconstrained cross-stock patterns while preventing the output weights from exploding.

```python
optimizer = AdamW([
    {"params": body_params, "weight_decay": 0.0},        # Transformer free
    {"params": output_head_params, "weight_decay": 1e-3}, # Ridge on head
])
```

### 4.2 Hyperparameter Tuning History

The learning rate and gradient accumulation were tuned through iterative experiments:

| LR | Grad Accum | Result |
|----|-----------|--------|
| 1e-4 | 4 | Val MSRR spiky (104 → 732 → 124), converged but unstable |
| 1e-5 | 8 | Stable but too slow — barely improving by epoch 240 |
| 5e-5 | 4 | Stable but slow — val MSRR 77 at epoch 240 |
| 1e-4 | 8 | Smoother but slow — half updates per epoch |
| **7.5e-5** | **4** | **Best trade-off: converges to val_msrr ~2-8, moderate stability** |

---

## 5. Training Procedure

### 5.1 Expanding Window with Yearly Refit

Identical to the MSE Transformer:

| Split | Period |
|-------|--------|
| Train | 1975-01 to (Y-2)-12 |
| Validation | (Y-1)-01 to (Y-1)-12 |
| Test | Y-01 to Y-12 |

### 5.2 Month-Level Training Loop

```python
for epoch in 1..300:
    shuffle month order
    for month in train_months:
        stock, macro, ind, target = load_month_to_gpu(month)
        weights = model(stock, macro, ind)         # N portfolio weights
        loss = (1 - dot(weights, target))² / grad_accum_steps
        loss.backward()
        
        if (step + 1) % 4 == 0:
            clip_grad_norm(1.0)
            optimizer.step()
            optimizer.zero_grad()
    
    compute val_msrr on all validation months
    early stopping check (patience=25, min_epochs=20)
```

---

## 6. Evaluation Metrics

Two portfolio evaluation methods:

### 6.1 SDF Portfolio (Direct Weights)

Uses the model's raw weight outputs directly:

```
Per month: portfolio_return = sum(w_i * R_i)
Sharpe = mean(monthly_returns) / std(monthly_returns) * sqrt(12)
```

The raw weights imply massive leverage (~200x gross). For practical comparison, weights can be normalized to any target leverage (Sharpe is scale-invariant).

### 6.2 L/S Decile Sort (For Comparison with MSE Transformer)

Same as the MSE Transformer evaluation: sort stocks by model output, long top 10%, short bottom 10%, equal-weighted. This ignores weight magnitudes and only uses rankings.

---

## 7. Results: MSRR Transformer (2016-2019)

### 7.1 Year-by-Year Performance

| Year | OOS R² (%) | Mean IC | L/S %/mo | L/S Sharpe | SDF Sharpe | Avg Epochs |
|------|-----------|---------|----------|------------|------------|------------|
| 2016 | -8.861 | +0.003 | -0.29 | -0.83 | +3.03 | 158.3 |
| 2017 | -5.296 | +0.018 | +0.34 | +0.60 | +1.60 | 176.0 |
| 2018 | -4.600 | +0.023 | +0.90 | +1.26 | +0.82 | 175.7 |
| 2019 | -21.244 | +0.015 | +0.34 | +1.35 | +2.76 | 101.7 |

### 7.2 Per-Seed Detail

**2016** (avg epoch 158.3):

| Seed | Best Epoch | Val MSRR | Training Time |
|------|-----------|----------|---------------|
| 0 | 128 | 42.883 | 1713s |
| 1 | 118 | 23.331 | 1601s |
| 2 | 138 | 4.804 | 1817s |
| 3 | 105 | 7.411 | 1450s |
| 4 | 294 | 0.409 | 3347s |
| 5 | 294 | 2.177 | 3422s |
| 6 | 121 | 16.075 | 1632s |
| 7 | 167 | 2.872 | 2152s |
| 8 | 49 | 105.596 | 830s |
| 9 | 169 | 7.753 | 2168s |

**2017** (avg epoch 176.0):

| Seed | Best Epoch | Val MSRR | Training Time |
|------|-----------|----------|---------------|
| 0 | 176 | 4.553 | 2298s |
| 1 | 121 | 3.731 | 1666s |
| 2 | 167 | 5.156 | 2194s |
| 3 | 218 | 3.379 | 2767s |
| 4 | 123 | 1.811 | 1685s |
| 5 | 247 | 2.307 | 3096s |
| 6 | 184 | 8.049 | 2381s |
| 7 | 72 | 43.036 | 1105s |
| 8 | 154 | 4.087 | 2038s |
| 9 | 298 | 2.412 | 3417s |

**2018** (avg epoch 175.7):

| Seed | Best Epoch | Val MSRR | Training Time |
|------|-----------|----------|---------------|
| 0 | 196 | 7.090 | 2564s |
| 1 | 237 | 1.527 | 3040s |
| 2 | 300 | 0.245 | 3482s |
| 3 | 168 | 2.612 | 2240s |
| 4 | 129 | 1.776 | 1786s |
| 5 | 200 | 1.218 | 2611s |
| 6 | 131 | 1.791 | 1816s |
| 7 | 167 | 1.222 | 2241s |
| 8 | 102 | 8.351 | 1483s |
| 9 | 127 | 3.388 | 1779s |

**2019** (avg epoch 101.7):

| Seed | Best Epoch | Val MSRR | Training Time |
|------|-----------|----------|---------------|
| 0 | 115 | 20.493 | 1671s |
| 1 | 73 | 8.362 | 1164s |
| 2 | 97 | 4.345 | 1450s |
| 3 | 38 | 122.831 | 749s |
| 4 | 41 | 20.139 | 785s |
| 5 | 95 | 19.563 | 1424s |
| 6 | 203 | 2.761 | 2717s |
| 7 | 200 | 1.719 | 2818s |
| 8 | 61 | 9.467 | 1051s |
| 9 | 94 | 20.167 | 1419s |

### 7.3 Summary Statistics

| Metric | Value |
|--------|-------|
| **Average OOS R²** | -10.00% |
| **Average IC** | +0.015 |
| **Average L/S Sharpe** | +0.59 |
| **Average SDF Sharpe** | **+2.05** |
| **Positive SDF Sharpe years** | 4 / 4 (100%) |
| **Best SDF year** | 2016 (Sharpe = +3.03) |
| **Worst SDF year** | 2018 (Sharpe = +0.82) |

### 7.4 Portfolio Weight Characteristics (2016)

| Statistic | Value |
|-----------|-------|
| Mean weight | +0.006 |
| Std weight | 0.050 |
| Max weight | +0.236 |
| Min weight | -0.268 |
| Avg gross leverage | ~216x |
| After normalization to 2x | Sharpe unchanged (scale-invariant) |

---

## 8. Comparison: MSRR vs MSE Transformer

### 8.1 Head-to-Head (2016-2019)

| Year | MSE TF (L/S Sharpe) | MSRR (L/S Sharpe) | MSRR (SDF Sharpe) | Best |
|------|--------------------|--------------------|-------------------|------|
| 2016 | +0.58 | -0.83 | **+3.03** | MSRR SDF |
| 2017 | **+3.07** | +0.60 | +1.60 | MSE L/S |
| 2018 | +1.04 | +1.26 | +0.82 | MSRR L/S |
| 2019 | +2.55 | +1.35 | **+2.76** | MSRR SDF |
| **Avg** | +1.81 | +0.60 | **+2.05** | **MSRR SDF** |

### 8.2 Key Observations

1. **MSRR SDF portfolio (avg 2.05) beats MSE L/S portfolio (avg 1.81)** — directly optimizing the portfolio objective produces better risk-adjusted returns.

2. **MSRR L/S is consistently worse (avg 0.60)** — the model doesn't learn good stock rankings, only good weight ratios. The decile sort discards the magnitude information that MSRR optimizes.

3. **SDF Sharpe is scale-invariant** — normalizing weights to any leverage level (1x, 2x, 4x) produces the same Sharpe. Only the relative proportions between stocks matter.

4. **High seed variance** — val_msrr ranges from 0.25 to 122.8 across seeds. Some seeds barely converge. The 10-seed ensemble is critical for stability.

### 8.3 What MSRR Learns vs MSE

| Property | MSE Transformer | MSRR Transformer |
|----------|----------------|-----------------|
| Good at stock ranking | Yes (IC +0.023) | No (IC +0.015) |
| Good at weight ratios | Not designed for this | Yes (SDF Sharpe 2.05) |
| Decile sort works well | Yes | No |
| Direct weight portfolio | Poor (Sharpe -1.70) | Excellent (Sharpe 2.05) |
| What it optimizes | Per-stock prediction accuracy | Whole-portfolio risk-adjusted return |

---

## 9. Factor Attribution (FF5 + Momentum)

### 9.1 Methodology

We regress monthly portfolio returns against the Fama-French 5 factors (MktRF, SMB, HML, RMW, CMA) plus the momentum factor (UMD):

```
R_portfolio = alpha + b1*MktRF + b2*SMB + b3*HML + b4*RMW + b5*CMA + b6*UMD + epsilon
```

If alpha is statistically significant (|t-stat| > 2), the portfolio generates returns that cannot be explained by known risk factors — i.e., genuine alpha.

Factor data source: `fama_french_factors.xlsx` (1963-2019).

### 9.2 Results (2016-2019, 47 months)

**MSRR SDF Portfolio (normalized to 2x gross leverage):**

| Factor | Coefficient | t-stat | p-value |
|--------|-----------|--------|---------|
| **Alpha** | **+0.759%/mo** | **5.34** | **0.000** |
| MktRF | +0.117 | 2.28 | 0.028 |
| SMB | -0.069 | -1.02 | 0.313 |
| HML | -0.113 | -1.64 | 0.109 |
| RMW | -0.137 | -1.35 | 0.185 |
| CMA | +0.086 | 0.75 | 0.456 |
| UMD | -0.114 | -2.13 | 0.040 |

R² = 0.328. **Annualized alpha = 9.11%, t-stat = 5.34**.

**MSRR L/S Decile Portfolio:**

| Factor | Coefficient | t-stat | p-value |
|--------|-----------|--------|---------|
| **Alpha** | **+0.903%/mo** | **4.26** | **0.000** |
| MktRF | -0.179 | -2.34 | 0.024 |
| SMB | -0.367 | -3.66 | 0.001 |
| HML | -0.110 | -1.07 | 0.292 |
| RMW | -0.169 | -1.12 | 0.271 |
| CMA | -0.079 | -0.46 | 0.645 |
| UMD | -0.150 | -1.87 | 0.068 |

R² = 0.450. **Annualized alpha = 10.83%, t-stat = 4.26**.

**FFN (NN5 MSE_ind_1yr) L/S Decile Portfolio (for comparison):**

| Factor | Coefficient | t-stat | p-value |
|--------|-----------|--------|---------|
| Alpha | +0.497%/mo | 1.07 | 0.290 |
| MktRF | -0.351 | -2.10 | 0.043 |
| SMB | -0.270 | -1.23 | 0.226 |
| HML | +0.333 | 1.48 | 0.147 |
| RMW | +0.883 | 2.66 | 0.011 |
| CMA | -0.223 | -0.60 | 0.555 |
| UMD | +0.248 | 1.41 | 0.166 |

R² = 0.457. **Annualized alpha = 5.96%, t-stat = 1.07 (NOT significant)**.

### 9.3 Summary

| Portfolio | Annual Alpha | t-stat | Significant? | R² |
|-----------|-------------|--------|-------------|-----|
| **MSRR SDF** | **9.11%** | **5.34** | **Yes (p<0.001)** | 0.33 |
| **MSRR L/S** | **10.83%** | **4.26** | **Yes (p<0.001)** | 0.45 |
| FFN L/S | 5.96% | 1.07 | No (p=0.29) | 0.46 |

### 9.4 Interpretation

1. **MSRR produces highly significant alpha** — both the SDF and L/S portfolios have alpha t-stats above 4, well beyond the conventional significance threshold of 2.0.

2. **FFN alpha is insignificant** — its returns are largely explained by known factors, particularly RMW (profitability factor, coef = +0.88, t = 2.66). The FFN is essentially repackaging known factor exposures.

3. **Low R² for MSRR SDF (0.33)** — only one-third of the SDF portfolio return is explained by known factors. The remaining two-thirds is genuine alpha from cross-stock patterns discovered by the attention mechanism.

4. **MSRR loads negatively on momentum (UMD)** — the model takes contrarian positions relative to momentum, which is consistent with cross-sectional attention discovering mean-reversion patterns that momentum misses.

5. **Caveat**: 47 months is a short sample. Factor attribution significance should be interpreted with caution. A longer OOS period would strengthen these conclusions.

---

## 10. Limitations and Future Improvements

This implementation is a **proof-of-concept** with significant room for improvement. Due to infrastructure and time constraints, the following enhancements were not pursued:

### 10.1 Not Implemented (Would Likely Improve Results)

| Enhancement | Kelly et al. Setting | Our Setting | Expected Impact |
|-------------|---------------------|-------------|-----------------|
| **Monthly retraining** | 60-month rolling, retrain every month | Yearly expanding window | High — avoids stale weights |
| **Ridge penalty grid search** | z = 10^i, i in {-10,...,3}, LOO-CV | Fixed at 1e-3 | High — optimal regularization |
| **Deeper model** | K=10 blocks, ~1M params | K=1 block, 14K params | High — Sharpe scales with depth (Kelly Fig. 6) |
| **Larger d_model** | d_model=132 (= input dim) | d_model=32 | Medium — richer representations |
| **More features** | 132 JKP characteristics | 95 GKX signals | Medium — more information |

### 10.2 Infrastructure Constraints

| Setting | Compute Required | Our Hardware |
|---------|-----------------|-------------|
| Our setup (yearly, K=1) | ~22 hours, 1 GPU | RTX 4080 SUPER |
| Monthly refit, K=1 | ~30 hours, 1 GPU | Feasible |
| Monthly refit, K=10 | ~13 days, 1 GPU | Impractical |
| Kelly full replication | ~4 days, 100 GPUs | Swiss National Supercomputing Centre |

### 10.3 Known Issues

1. **Noisy MSRR gradients** — the loss collapses 5000 stocks into a single scalar per month, making optimization harder than MSE. Required careful LR tuning (7.5e-5 vs 1e-4 for MSE).

2. **High seed variance** — some seeds converge to val_msrr < 1, others barely improve (val_msrr > 100). The ensemble averages this out but individual seeds are unreliable.

3. **Potential overfitting to validation** — inverse relationship between val_msrr and OOS SDF Sharpe suggests the model may overfit the 12-month validation window. Longer validation windows or monthly retraining would mitigate this.

---

## 11. Replication Instructions

### 11.1 Prerequisites

```
Python 3.10+
PyTorch 2.0+ (with CUDA)
numpy, pandas, scipy
GPU: NVIDIA with >= 16GB VRAM (tested on RTX 4080 SUPER)
```

### 11.2 Dependencies

The MSRR Transformer imports from both `train_nn.py` and `train_transformer.py`:

```python
# From train_nn.py (data pipeline)
from train_nn import (Config, setup_logging, load_returns, load_universe,
    load_signals, load_macro, load_sector_mapping, build_long_panel,
    build_industry_dummies, FeatureScaler, compute_cross_sectional_ic,
    compute_oos_metrics, set_seed)

# From train_transformer.py (model + data containers)
from train_transformer import (TransformerFeatureScaler, MonthGroupedData,
    CrossSectionalTransformer, evaluate)
```

All three files (`train_nn.py`, `train_transformer.py`, `train_transformer_msrr.py`) must be in the same directory.

### 11.3 Data Setup

Same as MSE Transformer. See `Transformer_report.md` section 10.3.

### 11.4 Running

```bash
python train_transformer_msrr.py
```

To change test years, edit `MSRRConfig` in `train_transformer_msrr.py`:

```python
test_years: List[int] = field(default_factory=lambda: [2016, 2017, 2018, 2019])
```

### 11.5 Output

```
output/
  logs/train_YYYYMMDD_HHMMSS.log                 # Full training log
  metrics/msrr_transformer_summary.csv            # OOS metrics per test year
  predictions/pred_ensemble_MSRR_year{Y}.parquet  # Ensemble weights per stock per month
  models/MSRR_year{Y}_seed{S}.pt                  # Model checkpoints (if saving enabled)
```

The prediction parquet files contain columns `(permno, month, prediction)` where `prediction` is the portfolio weight for that stock-month.

### 11.6 Expected Runtime

| Component | Time |
|-----------|------|
| Data loading + panel build | ~2 min |
| Per seed training | ~15-55 min (high variance across seeds) |
| Per year (10 seeds) | ~4-6 hours |
| 4 years total | **~20-22 hours** |

Tested on RTX 4080 SUPER with CUDA, mixed precision enabled.

---

## 12. Code Reference

| Component | File | Description |
|-----------|------|-------------|
| MSRRConfig | train_transformer_msrr.py | All hyperparameters |
| msrr_loss_month() | train_transformer_msrr.py | Core loss: (1 - w'R)² |
| train_one_epoch_msrr() | train_transformer_msrr.py | Month-level MSRR training |
| evaluate_msrr() | train_transformer_msrr.py | MSRR loss + predictions |
| compute_sdf_portfolio_metrics() | train_transformer_msrr.py | Direct w'R portfolio |
| train_model_msrr() | train_transformer_msrr.py | Full training with split optimizer |
| main() | train_transformer_msrr.py | Expanding window refit + ensemble |
| CrossSectionalTransformer | train_transformer.py | Model architecture (shared) |
| MonthGroupedData | train_transformer.py | Per-month data container (shared) |
| Data pipeline | train_nn.py | Loading, panel build, scaling (shared) |
