# GKX (2020) Neural Network Settings — Paper Reference

All neural network settings from "Empirical Asset Pricing via Machine Learning" (Gu, Kelly, Xiu 2020, Review of Financial Studies).

## Architecture

| Setting | Paper Value |
|---------|------------|
| Architectures | NN1(32), NN2(32,16), NN3(32,16,8), NN4(32,16,8,4), NN5(32,16,8,4,2) |
| Layer sizing rule | Geometric pyramid (Masters 1993) |
| Connectivity | Fully connected (all units receive input from all units in layer below) |
| Activation | ReLU at all hidden nodes |
| Output activation | None (linear) |

## Features (920 total)

| Component | Count | Details |
|-----------|-------|---------|
| Stock characteristics | 94 | Cross-sectionally ranked each period, mapped to [-1, 1] |
| Macro predictors | 8 | dp, ep, bm, ntis, tbl, tms, dfy, svar |
| Interaction structure | 846 | 94 characteristics x (8 macro + 1 constant) = 94 x 9 |
| Industry dummies | 74 | SIC 2-digit codes |
| **Total input dim** | **920** | z_{i,t} = x_t tensor-product c_{i,t}, plus industry dummies |

### Feature preprocessing
- All 94 characteristics are **cross-sectionally ranked** each period and mapped to [-1, 1] interval (following Kelly, Pruitt, Su 2019 and Freyberger, Neuhierl, Weber 2020)
- Missing characteristics replaced with **cross-sectional median** each month for each stock
- 8 macro variables derived from Welch & Goyal (2008):
  - dp = log(D12) - log(Index)
  - ep = log(E12) - log(Index)
  - bm = book-to-market (as-is)
  - ntis = net equity expansion (as-is)
  - tbl = Treasury-bill rate (as-is)
  - tms = lty - tbl (term spread)
  - dfy = BAA - AAA (default spread)
  - svar = stock variance (as-is)

### Interaction detail
The interaction z_{i,t} = x_t tensor-product c_{i,t} means each of the 94 characteristics is multiplied by each of the 9 macro terms (8 macro + 1 constant). The constant interaction recovers the raw characteristic, so the 846 features = 94 raw characteristics + 752 characteristic-macro interactions.

## Regularization (5 techniques combined)

1. **L1 penalty (l1)** on weight parameters — tuned via validation sample
2. **Learning rate shrinkage** — Adam optimizer (Kingma & Ba 2014), described in Internet Appendix Algorithm 5 / Section B.3
3. **Early stopping** — terminate training when validation sample prediction errors begin to increase (Internet Appendix Algorithm 6 / Section B.3)
4. **Batch normalization** — cross-sectionally demeans and variance-standardizes hidden unit inputs within each mini-batch (Ioffe & Szegedy 2015)
5. **Ensemble** — average predictions from multiple random seed initializations to reduce variance from stochastic optimization (Hansen & Salamon 1990; Dietterich 2000)

## Optimization

| Setting | Paper Value | Notes |
|---------|------------|-------|
| Optimizer | Adam (Kingma & Ba 2014) | Called "learning rate shrinkage" in paper |
| Learning rate | 0.001 (Adam default) | Paper does not override; Adam default from Kingma & Ba 2014 |
| Beta1 | 0.9 (Adam default) | Exponential decay rate for 1st moment |
| Beta2 | 0.999 (Adam default) | Exponential decay rate for 2nd moment |
| Epsilon | 1e-8 (Adam default) | Numerical stability |
| Training method | SGD with mini-batches | Random subsetting adds regularization (footnote 23) |
| Loss function | Penalized l2: MSE + lambda * L1(weights) | L1 = sum of absolute weight values |
| Batch size | Not specified in main text | Paper says "mini-batches" |

### L1 Regularization (Weight Penalty)

The objective function is:

    L(theta) = (1/N) * sum((r - r_hat)^2) + lambda * sum(|theta|)

- lambda (L1 penalty strength) is **tuned via validation** — they search over a grid of lambda values and select the one with lowest validation error
- The exact grid of lambda values is in **Internet Appendix B.3** (not included in this PDF)
- L1 encourages sparsity: neurons connect to fewer inputs (footnote 24)
- L1's penalty component shrinks weights toward zero ("weight decay")
- Early stopping and L1 are shown to be equivalent in certain circumstances (Bishop 1995; Goodfellow et al. 2016)

### Early Stopping

- Terminate training when validation sample prediction errors begin to increase
- By ending the search early, parameters are shrunk toward the initial guess (near zero)
- Acts as a substitute for L2 regularization at lower computational cost (footnote 24)
- Used together with L1 in the paper
- Exact patience / stopping criteria in **Internet Appendix Algorithm 6** (not included in this PDF)

## Data Splits & Refit

| Split | Period | Size |
|-------|--------|------|
| Initial training | 1957-1974 | 18 years |
| Validation | 1975-1986 | 12 years (fixed size, rolls forward) |
| Out-of-sample test | 1987-2016 | 30 years |
| Refit frequency | Yearly | Expand training +1 year, roll validation +1 year |

- No cross-validation (to maintain temporal ordering of data)
- Each refit: training sample grows by 1 year, validation window stays 12 years but shifts forward

## Target & Evaluation

| Setting | Paper Value |
|---------|------------|
| Target | Individual excess stock returns (monthly total return - T-bill rate) |
| Universe | All NYSE, AMEX, NASDAQ stocks (including <$5, codes >10/11, financials) |
| Avg stocks/month | ~6,200 |
| Total stocks | ~30,000 |
| OOS R² formula | R²_oos = 1 - sum(r - r_hat)^2 / sum(r^2) |
| R² denominator | Sum of squared excess returns (**not demeaned**) — benchmarks against zero forecast |

### Why not demean the denominator?
The paper argues that predicting future excess stock returns with historical mean **underperforms** a naive forecast of zero (the historical mean is too noisy). Using the non-demeaned denominator avoids artificially lowering the bar for "good" forecasting. If demeaned, all methods' R² would rise by ~3 percentage points.

## Paper Results (Table 1)

### Monthly OOS R² (percentage, 1987-2016)

| Model | All Stocks | Top 1,000 | Bottom 1,000 |
|-------|-----------|-----------|--------------|
| NN1 | 0.33 | 0.49 | 0.38 |
| NN2 | 0.39 | 0.62 | 0.46 |
| **NN3** | **0.40** | **0.70** | 0.45 |
| NN4 | 0.39 | 0.67 | 0.47 |
| NN5 | 0.36 | 0.64 | 0.42 |

- NN3 is the best overall performer
- Deep networks (NN4, NN5) do NOT improve over NN3 for monthly returns
- Neural networks are the best performing nonlinear method overall
- Large stocks (top 1,000 by market cap) show especially strong predictability (R² 0.52-0.70%)

### Comparison with other methods (All stocks)

| Method | R²_oos |
|--------|--------|
| OLS (all 920 features) | -3.46% |
| OLS-3 (size, bm, mom) | 0.16% |
| PLS | 0.27% |
| PCR | 0.26% |
| Elastic Net | 0.11% |
| GLM + Huber | 0.19% |
| Random Forest | 0.33% |
| GBRT + Huber | 0.34% |
| **NN3** | **0.40%** |

## What's NOT in the Main Paper (in Internet Appendix B.3, separate document)

The following specifics are referenced as Internet Appendix B.3 Algorithms 5-6, which is a separate document not included in this PDF:

- Exact learning rate value (likely Adam default 0.001)
- L1 penalty grid values searched over
- Early stopping patience (number of epochs to wait)
- Number of ensemble seeds
- Exact mini-batch size
- Any learning rate schedule or decay
- Initialization scheme details (though Kaiming/He init is standard for ReLU)

## Differences: Our Implementation vs Paper

| Setting | Paper | Our Implementation |
|---------|-------|--------------------|
| Stock characteristics | 94 | 95 (including mvel1) |
| Industry dummies | 74 (SIC 2-digit) | Not included |
| Total features | 920 | 863 |
| Feature scaling | Cross-sectional rank to [-1, 1] | StandardScaler (nanmean/nanstd), clip +/-5 std |
| Missing imputation | Cross-sectional median | 0 after scaling (= feature mean) |
| L1 penalty | Tuned via validation | Not used (collapsed predictions in prior run) |
| Ensemble seeds | Not specified in main text | 10 |
| Training start | 1957 | 1975 |
| Validation size | 12 years (rolling) | 1 year (year before test) |
| Test period | 1987-2016 (30 years) | 2016-2019 (4 years) |
| Dropout | Not explicitly mentioned | 0.05 |
| Learning rate | Adam (default) | 0.001 |
| Batch size | Mini-batch (unspecified) | 10,000 |
| Early stopping metric | Validation prediction error | Validation cross-sectional IC |
