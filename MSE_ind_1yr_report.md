# MSE_ind_1yr Experiment Report

## Replication of GKX (2020) Neural Network for Asset Pricing

**Paper**: Gu, Kelly, Xiu (2020) "Empirical Asset Pricing via Machine Learning", *Review of Financial Studies*

**Experiment**: MSE_ind_1yr — MSE-based early stopping, with industry dummies, 1-year validation window

**Test Period**: 2001-2019 (19 years of out-of-sample evaluation)

**Date**: 2026-04-10

---

## 1. Pipeline Overview

```
Raw Data (parquet/xlsx)
    |
    v
build_long_panel()          One row per (month, stock): signals, macro, excess returns
    |
    v
FeatureScaler.fit()         Compute mean/std from training set only
    |
    v
FeatureScaler.transform()   Standardize -> impute NaN to 0 -> clip +/-5 std -> compute interactions
    |
    v
build_industry_dummies()    Append 74 SIC-2 one-hot columns
    |
    v
GPUData (to CUDA)           Transfer to GPU, free CPU arrays
    |
    v
train_model_exp() x10       Train 10 seeds, MSE early stopping
    |
    v
Ensemble (mean of 10)       Average 10 seed predictions
    |
    v
compute_oos_metrics()       OOS R², IC, long-short Sharpe
```

---

## 2. Data

### 2.1 Sources

| File | Description |
|------|-------------|
| `gkx_full/signal_*.parquet` | 95 stock-level characteristics, wide format (months x stocks) |
| `gkx_full/returns.parquet` | Monthly stock returns, wide format |
| `gkx_full/universe.parquet` | Boolean mask for stock universe membership |
| `gkx_full/sector_mapping.csv` | PERMNO -> SIC 2-digit industry code mapping |
| `welch_goyal_2024.xlsx` | Welch-Goyal macroeconomic predictors (Monthly sheet) |

### 2.2 Stock Characteristics (95 signals)

```
absacc   acc      aeavol   age      agr      baspread beta     betasq
bm       bm_ia    cash     cashdebt cashpr   cfp      cfp_ia   chatoia
chcsho   chempia  chinv    chmom    chpmia   chtx     cinvest  convind
currat   depr     divi     divo     dolvol   dy       ear      egr
ep       gma      grcapx   grltnoa  herf     hire     idiovol  ill
indmom   invest   lev      lgr      maxret   mom12m   mom1m    mom36m
mom6m    ms       mve0     mve_ia   mvel1    nincr    operprof orgcap
pchcapx_ia        pchcurrat         pchdepr  pchgm_pchsale     pchquick
pchsale_pchinvt   pchsale_pchrect   pchsale_pchxsga  pchsaleinv
pctacc   pricedelay        ps       quick    rd       rd_mve   rd_sale
realestate        retvol   roaq     roavol   roeq     roic     rsup
salecash saleinv  salerec  secured  securedind        sgr      sin
sp       std_dolvol        std_turn stdacc   stdcf    tang     tb
turn     zerotrade
```

### 2.3 Macroeconomic Predictors (8 variables)

Derived from Welch-Goyal dataset:

| Variable | Formula |
|----------|---------|
| `dp` | log(D12) - log(Index) |
| `ep` | log(E12) - log(Index) |
| `bm` | b/m (book-to-market, as-is) |
| `ntis` | Net equity issuance (as-is) |
| `tbl` | Treasury bill rate (as-is) |
| `tms` | lty - tbl (term spread) |
| `dfy` | BAA - AAA (default yield spread) |
| `svar` | Stock variance (as-is) |

### 2.4 Target Variable

**Excess return**: raw monthly stock return minus the risk-free rate (Rfree derived from tbl in Welch-Goyal).

Signal alignment: characteristics at month *t* predict excess return at month *t+1*.

### 2.5 Industry Dummies

74 one-hot columns from SIC 2-digit codes. PERMNOs are mapped via `sector_mapping.csv`. Stocks with unknown SIC get all-zero dummy rows.

---

## 3. Feature Engineering

### 3.1 Feature Construction

| Component | Count | Description |
|-----------|-------|-------------|
| Stock signals | 95 | Raw characteristics |
| Macro predictors | 8 | Welch-Goyal derived |
| Interactions | 760 | 95 signals x 8 macro (cross-products) |
| Industry dummies | 74 | SIC 2-digit one-hot |
| **Total** | **937** | |

### 3.2 FeatureScaler Pipeline

Applied per refit year. Scaler fitted on **training data only**.

1. **Standardize**: z-score each feature using training-set `nanmean` and `nanstd`
2. **Impute**: replace remaining NaN with 0 (= training-set mean after standardization)
3. **Clip**: truncate at +/- 5 standard deviations
4. **Interactions**: compute 95 x 8 = 760 cross-products of standardized stock and macro features (done in 200K-row chunks)

---

## 4. Model Architecture

**NN5**: 5 hidden layers, geometric pyramid.

```
Input (937) -> Linear(937, 32) -> BatchNorm -> ReLU -> Dropout(0.05)
           -> Linear(32, 16)   -> BatchNorm -> ReLU -> Dropout(0.05)
           -> Linear(16, 8)    -> BatchNorm -> ReLU -> Dropout(0.05)
           -> Linear(8, 4)     -> BatchNorm -> ReLU -> Dropout(0.05)
           -> Linear(4, 2)     -> BatchNorm -> ReLU -> Dropout(0.05)
           -> Linear(2, 1)     (no activation)
```

**Weight initialization**: Kaiming normal (He) for ReLU, zero biases.

**Total parameters**: ~30,853 (varies slightly with input dimension)

---

## 5. Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Beta1, Beta2 | 0.9, 0.999 (PyTorch defaults) |
| Epsilon | 1e-8 |
| Batch size | 10,000 |
| Max epochs | 300 |
| Early stopping metric | **Validation MSE** (lower = better) |
| Early stopping patience | 25 epochs |
| Min epochs before stopping | 20 |
| Dropout | 0.05 |
| Feature clipping | +/- 5 std |
| Mixed precision | fp16 via torch.amp.autocast('cuda') + GradScaler |
| L1 penalty | None (0.0) |
| Loss function | MSE (mean squared error) |
| Number of seeds | 10 |
| Ensemble method | Simple arithmetic mean of 10 seed predictions |
| CUDA deterministic | True |
| cuDNN benchmark | False |

---

## 6. Training Procedure

### 6.1 Expanding Window with Yearly Refit

For each test year Y:

| Split | Period | Description |
|-------|--------|-------------|
| Train | 1975-01 to (Y-2)-12 | All data before validation window |
| Validation | (Y-1)-01 to (Y-1)-12 | 1-year rolling window |
| Test | Y-01 to Y-12 | Out-of-sample evaluation |

Example for test_year = 2016:

| Split | Period | Approximate Obs |
|-------|--------|-----------------|
| Train | 1975-01 to 2014-12 | ~2.9M |
| Val | 2015-01 to 2015-12 | ~67K |
| Test | 2016-01 to 2016-12 | ~68K |

### 6.2 Early Stopping (MSE)

```
best_val_loss = infinity
patience_counter = 0

for epoch in 1..300:
    train one epoch (mini-batch SGD with Adam)
    compute val_mse on full validation set

    if epoch >= 20:                    # min_epochs guard
        if val_mse < best_val_loss:
            best_val_loss = val_mse
            save model state
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 25:
                break

return best model state
```

### 6.3 Ensemble

For each (test_year, architecture):
- Train 10 models with seeds 0-9
- Each seed: `torch.manual_seed(seed)`, `np.random.seed(seed)`, `torch.cuda.manual_seed_all(seed)`
- Generate test predictions from each seed's best model
- Final prediction = mean of 10 seed predictions

### 6.4 Memory Optimization

Data splits are processed serially to limit peak CPU RAM:
1. Build train features -> transfer to GPU -> free CPU array
2. Build val features -> transfer to GPU -> free CPU array
3. Build test features -> transfer to GPU -> free CPU array

---

## 7. Evaluation Metrics

### 7.1 OOS R-squared

```
R² = 1 - SSE / SST
SSE = sum((actual - predicted)^2)
SST = sum(actual^2)           # NOT mean-adjusted, per GKX convention
```

Pooled across all stocks and months within each test year.

### 7.2 Cross-Sectional IC

Monthly Spearman rank correlation between predicted and actual excess returns. Computed independently for each month, then averaged.

### 7.3 Long-Short Portfolio

Per month:
1. Sort stocks by predicted excess return
2. Long = top decile (10%), Short = bottom decile (10%)
3. Equal-weighted returns within each decile
4. L/S return = mean(long returns) - mean(short returns)
5. Annualized Sharpe = (mean monthly L/S) / (std monthly L/S) * sqrt(12)

Minimum 20 stocks per month, minimum 2 stocks per decile.

---

## 8. Results: MSE_ind_1yr (2001-2019)

### 8.1 Year-by-Year Performance

| Year | OOS R² (%) | Mean IC | L/S %/mo | Sharpe |
|------|-----------|---------|----------|--------|
| 2001 | -0.120 | +0.002 | +1.09 | +0.82 |
| 2002 | -0.592 | -0.003 | +1.47 | +0.79 |
| 2003 | +1.016 | -0.003 | +0.72 | +0.69 |
| 2004 | +0.217 | -0.021 | +0.62 | +1.02 |
| 2005 | +0.624 | -0.010 | +0.03 | +0.07 |
| 2006 | +0.181 | +0.009 | +1.02 | +1.80 |
| 2007 | -1.537 | +0.008 | +0.47 | +1.41 |
| 2008 | -1.987 | +0.021 | -0.63 | -0.70 |
| 2009 | -0.786 | -0.020 | -0.02 | -0.02 |
| 2010 | +1.416 | -0.029 | +0.01 | +0.02 |
| 2011 | +0.772 | -0.001 | +1.25 | +1.58 |
| 2012 | +0.796 | +0.031 | +0.94 | +1.94 |
| 2013 | +1.772 | +0.052 | +2.19 | +3.96 |
| 2014 | -1.549 | -0.018 | +0.26 | +0.25 |
| 2015 | -0.168 | +0.045 | +1.79 | +1.93 |
| 2016 | -1.486 | +0.026 | +0.60 | +0.44 |
| 2017 | +0.701 | -0.003 | +0.72 | +2.44 |
| 2018 | +0.066 | +0.044 | +1.69 | +2.32 |
| 2019 | -4.021 | -0.018 | -0.30 | -0.21 |

### 8.2 Summary Statistics

| Metric | Value |
|--------|-------|
| **Average OOS R²** | -0.25% |
| **Average IC** | +0.006 |
| **Average L/S Return** | +0.73%/mo |
| **Average Annualized Sharpe** | +1.07 |
| **Sharpe Std (across years)** | 1.05 |
| **Positive Sharpe years** | 15 / 19 (79%) |
| **Worst year** | 2008 (Sharpe = -0.70) |
| **Best year** | 2013 (Sharpe = +3.96) |

---

## 9. Comparison: Paper vs Our Hyperparameters

From Internet Appendix Table A.5:

| Parameter | GKX Paper | Ours | Match? |
|-----------|-----------|------|--------|
| Architecture | NN1-NN5 | NN5 only | Partial |
| Hidden dims (NN5) | (32,16,8,4,2) | (32,16,8,4,2) | Yes |
| Activation | ReLU | ReLU | Yes |
| Batch normalization | Yes | Yes | Yes |
| Dropout | Not mentioned | 0.05 | Unknown |
| Optimizer | Adam | Adam | Yes |
| Learning rate | {0.001, 0.01} grid | 0.001 only | Partial |
| Batch size | 10,000 | 10,000 | Yes |
| Max epochs | 100 | 300 | No |
| Patience | 5 | 25 | No |
| Early stopping metric | Validation MSE | Validation MSE | Yes |
| L1 penalty | lambda in (1e-5, 1e-3) | None | No |
| Ensemble seeds | 10 | 10 | Yes |
| Feature scaling | Rank to [-1,1] per month | StandardScaler + clip +/-5 | No |
| Missing imputation | Monthly cross-sectional median | Training-set mean (0) | No |
| Validation window | 12 years (fixed, rolling) | 1 year | No |
| Training start | 1957 | 1975 | No |
| Industry dummies | 74 (SIC 2-digit) | 74 (SIC 2-digit) | Yes |
| Total input features | 920 | 937 | Close |

### Key Differences

1. **Patience**: Paper uses 5, we use 25. Paper stops much more aggressively.
2. **Max epochs**: Paper uses 100, we use 300.
3. **LR grid**: Paper searches {0.001, 0.01}, we fix at 0.001.
4. **L1 penalty**: Paper tunes lambda via validation grid search, we omit it.
5. **Feature scaling**: Paper ranks to [-1,1] per month (robust to outliers), we standardize + clip.
6. **Validation window**: Paper uses 12 years, we use 1 year.
7. **Training start**: Paper starts from 1957, we start from 1975.

---

## 10. Replication Instructions

### 10.1 Prerequisites

```
Python 3.10+
PyTorch 2.0+ (with CUDA)
numpy, pandas, scipy
GPU: NVIDIA with >= 16GB VRAM (tested on RTX 4080 SUPER)
```

### 10.2 Data Setup

Place the following in the project root:
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

All parquet files are wide format: PeriodIndex (monthly) x stock columns.

### 10.3 Running

To replicate MSE_ind_1yr for 2001-2019:

In `train_nn.py`, set the experiment config in `run_experiments()`:

```python
experiments = [
    {"name": "MSE_ind_1yr", "stop_metric": "mse", "use_dummies": True, "val_years": 1},
]
config = Config(test_years=list(range(2001, 2020)))
```

Then run:

```bash
python train_nn.py
```

### 10.4 Output

```
output/
  logs/train_YYYYMMDD_HHMMSS.log       # Full training log with per-epoch metrics
  models/NN5_year{Y}_seed{S}.pt        # Best model checkpoint per seed
  predictions/pred_ensemble_NN5_year{Y}.parquet  # Ensemble predictions
  metrics/experiments_summary.csv       # All OOS metrics per year
  features/scaler_year{Y}.pkl          # Fitted scaler per refit year
```

### 10.5 Expected Runtime

- ~2 min per seed (RTX 4080 SUPER with mixed precision)
- 19 years x 10 seeds = 190 models
- Total: ~25 minutes

---

## 11. Code Reference

| Component | File | Lines |
|-----------|------|-------|
| Config | train_nn.py | 30-99 |
| build_long_panel() | train_nn.py | 255-374 |
| FeatureScaler | train_nn.py | 381-447 |
| GKXNet model | train_nn.py | 474-500 |
| compute_oos_metrics() | train_nn.py | 724-782 |
| train_model_exp() | train_nn.py | 1076-1163 |
| run_experiments() | train_nn.py | 1166-1375 |
