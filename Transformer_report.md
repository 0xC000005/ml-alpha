# Cross-Sectional Transformer Experiment Report

## Transformer for Asset Pricing via Cross-Sectional Self-Attention

**Baseline**: GKX (2020) NN5 feedforward network (MSE_ind_1yr config)

**Experiment**: Cross-Sectional Transformer — all stocks attend to each other within each month

**Test Period**: 2012-2019 (8 years of out-of-sample evaluation)

**Date**: 2026-04-12

---

## 1. Pipeline Overview

```
Raw Data (parquet/xlsx)                          [reused from train_nn.py]
    |
    v
build_long_panel()          One row per (month, stock): signals, macro, excess returns
    |
    v
TransformerFeatureScaler.fit()    Compute mean/std from training set only
    |
    v
TransformerFeatureScaler.transform()   Standardize -> impute NaN to 0 -> clip +/-5 std
    |                                  Returns (stock_scaled, macro_scaled) SEPARATELY
    |                                  ** No interaction terms computed **
    v
build_industry_dummies()    Append 74 SIC-2 one-hot columns
    |
    v
MonthGroupedData            Group by month, store on CPU, transfer one month to GPU per forward pass
    |
    v
train_model() x10           Train 10 seeds, MSE early stopping, gradient accumulation
    |
    v
Ensemble (mean of 10)       Average 10 seed predictions
    |
    v
compute_oos_metrics()       OOS R-squared, IC, long-short Sharpe
```

**Key difference from FFN pipeline**: No interaction terms (760 signal x macro cross-products). The Transformer's self-attention learns implicit cross-stock interactions instead.

---

## 2. Data

### 2.1 Sources

Identical to FFN (MSE_ind_1yr). See `MSE_ind_1yr_report.md` for full data documentation.

| File | Description |
|------|-------------|
| `gkx_full/signal_*.parquet` | 95 stock-level characteristics |
| `gkx_full/returns.parquet` | Monthly stock returns |
| `gkx_full/universe.parquet` | Stock universe membership |
| `gkx_full/sector_mapping.csv` | PERMNO -> SIC 2-digit mapping |
| `welch_goyal_2024.xlsx` | Welch-Goyal macroeconomic predictors |

### 2.2 Target Variable

**Excess return**: raw monthly stock return minus the risk-free rate.

Signal alignment: characteristics at month *t* predict excess return at month *t+1*.

---

## 3. Feature Engineering

### 3.1 Feature Construction

| Component | Count | Description |
|-----------|-------|-------------|
| Stock signals | 95 | Raw characteristics |
| Industry dummies | 74 | SIC 2-digit one-hot |
| **Stock input total** | **169** | Concatenated per stock |
| Macro predictors | 8 | Welch-Goyal derived (added via projection, not concatenated) |

**vs FFN (937 features)**: The Transformer uses **169 per-stock features + 8 macro** = 177 total unique features. The FFN uses 937 (including 760 hand-crafted interaction terms). The Transformer has **12x fewer input features**.

### 3.2 TransformerFeatureScaler Pipeline

Applied per refit year. Scaler fitted on **training data only**.

1. **Standardize**: z-score each feature using training-set `nanmean` and `nanstd`
2. **Impute**: replace remaining NaN with 0
3. **Clip**: truncate at +/- 5 standard deviations
4. **No interactions**: stock and macro features returned separately

### 3.3 Macro Conditioning

Macro variables are projected to d_model dimensions and **added** to each stock's embedding (broadcast). This is more parameter-efficient than concatenation since macro is identical for all stocks in a month.

---

## 4. Model Architecture

### 4.1 CrossSectionalTransformer

```
Per month t with N stocks (~5,000):

Input per stock: [stock_signals(95) || industry_dummies(74)] = 169 dims
Macro (shared):  8 dims

stock_proj = Linear(169, 32)                    # project to d_model
macro_proj = Linear(8, 32)                      # project macro
x = stock_proj(input) + macro_proj(macro)       # additive conditioning (broadcast)

# Pre-norm Transformer block (1 layer)
x = x + Dropout(MultiHeadSelfAttention(LayerNorm(x)))    # 4 heads, d_k=8
x = x + FFN(LayerNorm(x))                                 # 32->64->32, GELU, Dropout

# Output
pred = Linear(LayerNorm(x), 1)                  # per-stock scalar prediction
```

### 4.2 Design Choices

| Choice | Rationale |
|--------|-----------|
| **d_model=32** | Small for low-SNR task. Matches FFN first hidden dim. |
| **1 layer** | Minimal architecture. More layers risk overfitting noise. |
| **4 heads** (d_k=8) | Allows diverse attention patterns while keeping model small. |
| **d_ff=64** | 2x expansion ratio. Standard but minimal. |
| **Pre-norm** (LayerNorm before attention/FFN) | More stable training, standard since GPT-2. |
| **LayerNorm** (not BatchNorm) | Appropriate for variable-size sets (N varies by month). |
| **No positional encoding** | Stocks have no natural ordering within a month. |
| **Additive macro** | Macro is identical for all stocks; add rather than concatenate. |
| **Xavier init** | Balanced for attention layers (vs He/Kaiming for ReLU). |
| **GELU activation** | Standard for Transformers (vs ReLU for FFNs). |

### 4.3 Why Cross-Sectional Attention

The FFN scores each stock independently. The Transformer's self-attention lets every stock "see" every other stock in the same month, enabling:

- **Relative pricing**: "this stock is cheap vs sector peers"
- **Crowding detection**: "value is crowded this month, discount signal"
- **Implicit interactions**: replaces 760 hand-crafted signal x macro features

The attention matrix is N x N per head (~5000 x 5000 = 25M elements), making this the dominant computation and the reason for ~90% GPU utilization (vs ~30% for the small FFN).

### 4.4 Parameter Count

| Component | Parameters |
|-----------|-----------|
| stock_proj (169->32) | 5,440 |
| macro_proj (8->32) | 288 |
| LayerNorm x3 | 192 |
| MultiHeadAttention (d=32, h=4) | 4,256 |
| FFN (32->64->32) | 4,192 |
| Output head (32->1) | 33 |
| **Total** | **14,369** |

**vs FFN (NN5): ~30,853 parameters**. The Transformer has **less than half** the parameters.

---

## 5. Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Optimizer | AdamW | Decoupled weight decay, standard for Transformers |
| Learning rate | 1e-4 | Lower than FFN (1e-3). Transformers prefer smaller LR. |
| Weight decay | 1e-4 | L2 regularization via AdamW |
| Gradient accumulation | 4 months | Each "batch" = 4 months of cross-sections. Reduces gradient noise. |
| Gradient clipping | max_norm=1.0 | Stabilizes training with variable-N months. |
| Max epochs | 300 | Same ceiling as FFN |
| Early stopping metric | Validation MSE | Lower = better. Same as FFN (MSE_ind_1yr). |
| Early stopping patience | 25 epochs | Same as FFN |
| Min epochs before stopping | 20 | Same as FFN |
| Dropout | 0.10 | Higher than FFN (0.05). Attention overfits easier. |
| Feature clipping | +/- 5 std | Same as FFN |
| Mixed precision | fp16 via torch.amp.autocast('cuda') + GradScaler | Same as FFN |
| Number of seeds | 10 | Same as FFN |
| Ensemble method | Arithmetic mean of 10 seed predictions | Same as FFN |

---

## 6. Training Procedure

### 6.1 Expanding Window with Yearly Refit

Identical to FFN. For each test year Y:

| Split | Period | Description |
|-------|--------|-------------|
| Train | 1975-01 to (Y-2)-12 | All data before validation window |
| Validation | (Y-1)-01 to (Y-1)-12 | 1-year rolling window |
| Test | Y-01 to Y-12 | Out-of-sample evaluation |

### 6.2 Month-Level Training Loop

Unlike the FFN (which shuffles individual stock observations into mini-batches of 10,000), the Transformer processes **one month at a time** — all ~5,000 stocks simultaneously through self-attention.

```python
for epoch in 1..300:
    shuffle month order
    for month in train_months:
        stock, macro, ind, target = load_month_to_gpu(month)
        preds = model(stock, macro, ind)           # all N stocks at once
        loss = MSE(preds, target) / grad_accum_steps
        loss.backward()
        
        if (step + 1) % 4 == 0:                    # accumulate 4 months
            clip_grad_norm(1.0)
            optimizer.step()
            optimizer.zero_grad()
    
    compute val_mse on all validation months
    early stopping check (patience=25, min_epochs=20)
```

### 6.3 Memory Optimization

| Strategy | Description |
|----------|-------------|
| CPU storage | MonthGroupedData stores all months on CPU as numpy arrays |
| Per-month GPU transfer | One month (~5MB) transferred to GPU per forward pass |
| No padding | Each month processed independently (variable N) |
| Peak GPU | < 100 MB total (vs ~12 GB for FFN with interaction matrix) |

### 6.4 Ensemble

Same as FFN: 10 seeds per test year, ensemble = mean of 10 predictions.

---

## 7. Evaluation Metrics

Identical to FFN. See `MSE_ind_1yr_report.md` for full metric definitions.

| Metric | Formula |
|--------|---------|
| OOS R² | 1 - SSE/SST, where SST = sum(actual²) (not mean-adjusted, per GKX) |
| IC | Monthly Spearman rank correlation, averaged |
| L/S Sharpe | Top/bottom decile equal-weighted, annualized |

---

## 8. Results: Transformer (2012-2019)

### 8.1 Year-by-Year Performance

| Year | OOS R² (%) | Mean IC | L/S %/mo | Sharpe | Avg Epochs |
|------|-----------|---------|----------|--------|------------|
| 2012 | +0.462 | +0.017 | +2.17 | +3.33 | 66.6 |
| 2013 | -0.160 | +0.037 | +1.88 | +4.13 | 62.0 |
| 2014 | -2.994 | -0.044 | +0.80 | +1.09 | 62.6 |
| 2015 | +0.129 | +0.063 | +1.99 | +1.52 | 74.2 |
| 2016 | +0.600 | +0.015 | +0.64 | +0.58 | 56.9 |
| 2017 | +0.677 | +0.026 | +1.66 | +3.07 | 59.7 |
| 2018 | -0.070 | +0.012 | +1.18 | +1.04 | 62.9 |
| 2019 | +0.688 | +0.038 | +1.93 | +2.55 | 97.2 |

### 8.2 Per-Seed Detail

**2012** (avg epoch 66.6):

| Seed | Best Epoch | Val IC | Training Time |
|------|-----------|--------|---------------|
| 0 | 74 | +0.055 | 1018s |
| 1 | 86 | +0.022 | 1141s |
| 2 | 66 | -0.018 | 936s |
| 3 | 67 | +0.026 | 946s |
| 4 | 85 | +0.002 | 1131s |
| 5 | 69 | +0.059 | 966s |
| 6 | 49 | +0.007 | 761s |
| 7 | 45 | +0.026 | 720s |
| 8 | 80 | +0.027 | 1080s |
| 9 | 45 | -0.016 | 720s |

**2013** (avg epoch 62.0):

| Seed | Best Epoch | Val IC | Training Time |
|------|-----------|--------|---------------|
| 0 | 86 | +0.030 | 1165s |
| 1 | 42 | +0.018 | 704s |
| 2 | 59 | +0.030 | 883s |
| 3 | 47 | +0.028 | 757s |
| 4 | 76 | +0.001 | 1061s |
| 5 | 87 | +0.016 | 1177s |
| 6 | 45 | +0.003 | 736s |
| 7 | 77 | +0.028 | 1071s |
| 8 | 63 | +0.018 | 925s |
| 9 | 38 | -0.010 | 662s |

**2014** (avg epoch 62.6):

| Seed | Best Epoch | Val IC | Training Time |
|------|-----------|--------|---------------|
| 0 | 82 | +0.028 | 1147s |
| 1 | 64 | +0.001 | 954s |
| 2 | 56 | +0.027 | 868s |
| 3 | 52 | +0.019 | 825s |
| 4 | 76 | +0.023 | 1082s |
| 5 | 84 | +0.019 | 1168s |
| 6 | 51 | +0.005 | 814s |
| 7 | 84 | +0.013 | 1168s |
| 8 | 37 | +0.010 | 664s |
| 9 | 40 | +0.041 | 697s |

**2015** (avg epoch 74.2):

| Seed | Best Epoch | Val IC | Training Time |
|------|-----------|--------|---------------|
| 0 | 101 | +0.020 | 1376s |
| 1 | 41 | +0.015 | 721s |
| 2 | 53 | +0.022 | 853s |
| 3 | 64 | +0.006 | 973s |
| 4 | 179 | -0.013 | 2230s |
| 5 | 54 | +0.030 | 863s |
| 6 | 43 | +0.027 | 743s |
| 7 | 103 | +0.017 | 1398s |
| 8 | 70 | +0.026 | 1038s |
| 9 | 34 | +0.043 | 645s |

**2016** (avg epoch 56.9):

| Seed | Best Epoch | Val IC | Training Time |
|------|-----------|--------|---------------|
| 0 | 58 | +0.011 | 937s |
| 1 | 51 | +0.047 | 851s |
| 2 | 39 | +0.068 | 724s |
| 3 | 32 | +0.016 | 658s |
| 4 | 42 | +0.021 | 753s |
| 5 | 86 | +0.078 | 1243s |
| 6 | 96 | +0.076 | 1379s |
| 7 | 45 | +0.051 | 791s |
| 8 | 30 | +0.028 | 619s |
| 9 | 90 | +0.093 | 1370s |

**2017** (avg epoch 59.7):

| Seed | Best Epoch | Val IC | Training Time |
|------|-----------|--------|---------------|
| 0 | 67 | -0.000 | 1063s |
| 1 | 45 | +0.040 | 814s |
| 2 | 60 | +0.009 | 987s |
| 3 | 51 | +0.021 | 873s |
| 4 | 48 | +0.011 | 834s |
| 5 | 55 | +0.012 | 907s |
| 6 | 65 | +0.011 | 1021s |
| 7 | 62 | +0.000 | 987s |
| 8 | 91 | -0.012 | 1315s |
| 9 | 53 | +0.021 | 885s |

**2018** (avg epoch 62.9):

| Seed | Best Epoch | Val IC | Training Time |
|------|-----------|--------|---------------|
| 0 | 91 | +0.007 | 1340s |
| 1 | 58 | +0.045 | 959s |
| 2 | 59 | +0.014 | 971s |
| 3 | 75 | +0.023 | 1155s |
| 4 | 66 | +0.008 | 1052s |
| 5 | 79 | +0.010 | 1218s |
| 6 | 55 | +0.046 | 938s |
| 7 | 60 | +0.041 | 996s |
| 8 | 48 | -0.002 | 860s |
| 9 | 38 | +0.013 | 739s |

**2019** (avg epoch 97.2):

| Seed | Best Epoch | Val IC | Training Time |
|------|-----------|--------|---------------|
| 0 | 69 | -0.008 | 1108s |
| 1 | 32 | +0.024 | 671s |
| 2 | 256 | +0.061 | 3307s |
| 3 | 205 | +0.032 | 2709s |
| 4 | 68 | +0.013 | 1095s |
| 5 | 53 | -0.005 | 919s |
| 6 | 154 | +0.031 | 2108s |
| 7 | 28 | -0.020 | 624s |
| 8 | 46 | +0.004 | 836s |
| 9 | 61 | +0.010 | 1014s |

### 8.3 Summary Statistics

| Metric | Value |
|--------|-------|
| **Average OOS R²** | -0.08% |
| **Average IC** | +0.021 |
| **Average L/S Return** | +1.53%/mo |
| **Average Annualized Sharpe** | +2.16 |
| **Positive Sharpe years** | 8 / 8 (100%) |
| **Worst year** | 2016 (Sharpe = +0.58) |
| **Best year** | 2013 (Sharpe = +4.13) |

---

## 9. Comparison: Transformer vs FFN (MSE_ind_1yr)

### 9.1 Head-to-Head (2012-2019)

| Year | Model | OOS R² (%) | Mean IC | L/S %/mo | Sharpe | Winner |
|------|-------|-----------|---------|----------|--------|--------|
| 2012 | TF | +0.462 | +0.017 | **+2.17** | **+3.33** | **TF** |
| 2012 | FFN | **+0.796** | **+0.031** | +0.94 | +1.94 | |
| 2013 | TF | -0.160 | +0.037 | +1.88 | **+4.13** | **TF** |
| 2013 | FFN | **+1.772** | **+0.052** | **+2.19** | +3.96 | |
| 2014 | TF | -2.994 | -0.044 | **+0.80** | **+1.09** | **TF** |
| 2014 | FFN | **-1.549** | **-0.018** | +0.26 | +0.25 | |
| 2015 | TF | +0.129 | **+0.063** | **+1.99** | +1.52 | |
| 2015 | FFN | **-0.168** | +0.045 | +1.79 | **+1.93** | **FFN** |
| 2016 | TF | **+0.600** | +0.015 | +0.64 | **+0.58** | **TF** |
| 2016 | FFN | -1.486 | **+0.026** | +0.60 | +0.44 | |
| 2017 | TF | +0.677 | **+0.026** | **+1.66** | **+3.07** | **TF** |
| 2017 | FFN | **+0.701** | -0.003 | +0.72 | +2.44 | |
| 2018 | TF | -0.070 | +0.012 | +1.18 | +1.04 | |
| 2018 | FFN | **+0.066** | **+0.044** | **+1.69** | **+2.32** | **FFN** |
| 2019 | TF | **+0.688** | **+0.038** | **+1.93** | **+2.55** | **TF** |
| 2019 | FFN | -4.021 | -0.018 | -0.30 | -0.21 | |

### 9.2 Average Comparison (8 years)

| Metric | Transformer | FFN (NN5) | Improvement |
|--------|------------|-----------|-------------|
| **Avg OOS R²** | -0.08% | -0.55% | +0.47 pp |
| **Avg IC** | +0.021 | +0.014 | +0.007 |
| **Avg L/S Return** | +1.53%/mo | +0.91%/mo | +0.62 pp |
| **Avg Sharpe** | +2.16 | +1.63 | +0.53 |
| **Sharpe Std** | 1.30 | 1.37 | Lower variance |
| **Positive Sharpe years** | 8/8 (100%) | 7/8 (88%) | |
| **Sharpe winner** | 6/8 years | 2/8 years | |

### 9.3 Architecture Comparison

| Property | Transformer | FFN (NN5) |
|----------|------------|-----------|
| Parameters | 14,369 | ~30,853 |
| Input features | 169 + 8 macro | 937 |
| Interaction terms | 0 (learned via attention) | 760 (hand-crafted) |
| Hidden layers | 1 Transformer block | 5 fully-connected |
| Activation | GELU | ReLU |
| Normalization | LayerNorm (pre-norm) | BatchNorm |
| Optimizer | AdamW (lr=1e-4) | Adam (lr=1e-3) |
| Dropout | 0.10 | 0.05 |
| GPU utilization | ~90% | ~30% |
| GPU memory | < 100 MB | ~12 GB |
| Training time per seed | ~15 min | ~2 min |

### 9.4 Key Takeaways

1. **Transformer wins 6 out of 8 years** on Sharpe ratio, losing only 2015 and 2018.
2. **100% positive Sharpe** across all 8 years — the FFN posted a negative year (2019: -0.21).
3. **2019 is the most dramatic difference**: FFN collapsed (Sharpe -0.21) while Transformer thrived (+2.55).
4. **Higher average L/S returns** (+1.53%/mo vs +0.91%/mo) with lower Sharpe variance (std 1.30 vs 1.37).
5. **Transformer achieves this with fewer parameters and no hand-crafted interactions**, suggesting the attention mechanism learns useful cross-stock patterns that hand-crafted signal x macro interactions miss.
6. **FFN wins on R² and IC more often**, but the Transformer wins on what matters for portfolio construction: L/S Sharpe.

---

## 10. Replication Instructions

### 10.1 Prerequisites

```
Python 3.10+
PyTorch 2.0+ (with CUDA)
numpy, pandas, scipy
GPU: NVIDIA with >= 16GB VRAM (tested on RTX 4080 SUPER)
```

### 10.2 Dependencies

The Transformer imports data pipeline functions from `train_nn.py`:

```python
from train_nn import (
    Config, setup_logging, load_returns, load_universe, load_signals,
    load_macro, load_sector_mapping, build_long_panel, build_industry_dummies,
    FeatureScaler, compute_cross_sectional_ic, compute_oos_metrics, set_seed,
)
```

Both `train_nn.py` and `train_transformer.py` must be in the same directory.

### 10.3 Data Setup

Same as FFN. Place the following in the project root:

```
gkx_full/
  signal_absacc.parquet
  signal_acc.parquet
  ... (95 signal files)
  returns.parquet
  universe.parquet
  sector_mapping.csv
welch_goyal_2024.xlsx
```

### 10.4 Running

Edit `TransformerConfig` in `train_transformer.py` to set test years:

```python
test_years: List[int] = field(default_factory=lambda: [2012, 2013, 2014, 2015])
```

Then run:

```bash
python train_transformer.py
```

### 10.5 Output

```
output/
  logs/train_YYYYMMDD_HHMMSS.log             # Full training log with per-epoch metrics
  metrics/transformer_summary.csv             # OOS metrics per test year
```

Note: The current implementation does not save model checkpoints or per-seed predictions to disk. Add `torch.save()` calls if needed.

### 10.6 Expected Runtime

| Component | Time |
|-----------|------|
| Data loading + panel build | ~2 min |
| Per seed training | ~10-20 min (varies by early stop epoch) |
| Per year (10 seeds) | ~2-3 hours |
| 4 years (one run) | **~10-12 hours** |
| 8 years (two runs) | **~20-24 hours** |

Tested on RTX 4080 SUPER with CUDA, mixed precision enabled.

---

## 11. Code Reference

| Component | File | Lines |
|-----------|------|-------|
| TransformerConfig | train_transformer.py | 44-82 |
| TransformerFeatureScaler | train_transformer.py | 89-109 |
| MonthGroupedData | train_transformer.py | 116-157 |
| CrossSectionalTransformer | train_transformer.py | 163-240 |
| train_one_epoch() | train_transformer.py | 247-286 |
| evaluate() | train_transformer.py | 289-306 |
| train_model() | train_transformer.py | 309-381 |
| main() | train_transformer.py | 388-548 |

### Reused from train_nn.py

| Function | Purpose |
|----------|---------|
| load_returns, load_universe, load_signals, load_macro | Data loading |
| load_sector_mapping | Industry code mapping |
| build_long_panel | Panel construction |
| build_industry_dummies | SIC 2-digit one-hot encoding |
| FeatureScaler | Mean/std computation (wrapped by TransformerFeatureScaler) |
| compute_cross_sectional_ic | Monthly Spearman IC |
| compute_oos_metrics | R², IC, L/S Sharpe |
| set_seed, setup_logging | Reproducibility and logging |
