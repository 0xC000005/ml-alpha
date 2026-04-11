# Complete NN Parameter List: Paper vs Our Implementation

Every tunable parameter for a feedforward neural network, with GKX (2020) paper settings and our current settings.

## Architecture

| # | Parameter | GKX Paper | Ours | Notes |
|---|-----------|-----------|------|-------|
| 1 | Number of hidden layers | 1-5 (test all) | 1-5 (test all) | Match |
| 2 | Neurons per layer | NN1(32), NN2(32,16), NN3(32,16,8), NN4(32,16,8,4), NN5(32,16,8,4,2) | Same | Match |
| 3 | Layer sizing pattern | Geometric pyramid (Masters 1993) | Geometric pyramid | Match |
| 4 | Activation function | ReLU | ReLU | Match |
| 5 | Output activation | Linear (none) | Linear (none) | Match |
| 6 | Skip/residual connections | No | No | Match |
| 7 | Batch normalization | Yes, every hidden layer | Yes, every hidden layer | Match |
| 8 | Layer normalization | No | No | Match |
| 9 | Dropout rate | Not mentioned in main text | 0.05 | Paper may use 0; detail in Internet Appendix |
| 10 | Dropout type | Not mentioned | Standard | — |
| 11 | Connectivity | Fully connected | Fully connected | Match |

## Optimization

| # | Parameter | GKX Paper | Ours | Notes |
|---|-----------|-----------|------|-------|
| 12 | Optimizer | Adam (Kingma & Ba 2014) | Adam | Match |
| 13 | Learning rate | 0.001 (Adam default, not overridden) | 0.001 | Match (assumed) |
| 14 | Beta1 | 0.9 (Adam default) | 0.9 (PyTorch default) | Match (assumed) |
| 15 | Beta2 | 0.999 (Adam default) | 0.999 (PyTorch default) | Match (assumed) |
| 16 | Epsilon | 1e-8 (Adam default) | 1e-8 (PyTorch default) | Match (assumed) |
| 17 | Weight decay (L2 in optimizer) | Not mentioned | 0 | — |
| 18 | Momentum (SGD only) | N/A (uses Adam) | N/A | — |
| 19 | LR scheduler | Not mentioned | None | Paper may use one; detail in Internet Appendix |
| 20 | LR scheduler type | Not mentioned | N/A | Options: StepLR, CosineAnnealing, ReduceOnPlateau, OneCycleLR, warmup+decay |
| 21 | LR scheduler params | Not mentioned | N/A | e.g., step_size, gamma, T_max, patience, min_lr |
| 22 | Batch size | Not mentioned ("mini-batches") | 10,000 | Paper detail in Internet Appendix |
| 23 | Max epochs | Not mentioned | 300 | Paper detail in Internet Appendix |
| 24 | Gradient clipping | Not mentioned | None | Options: max_norm=1.0/5.0, clip_value |
| 25 | Gradient accumulation steps | Not mentioned | 1 (none) | — |
| 26 | Mixed precision | Not mentioned (likely fp32) | fp16 (torch.amp.autocast) | We use AMP for speed |

## Regularization

| # | Parameter | GKX Paper | Ours | Notes |
|---|-----------|-----------|------|-------|
| 27 | L1 penalty (on weights) | Yes, lambda tuned via validation grid search | None (0.0) | **Difference** — we dropped L1 because it collapsed predictions at lambda=0.001 |
| 28 | L1 lambda values | Grid in Internet Appendix B.3 | N/A | Do not know exact grid |
| 29 | L2 penalty / weight decay | Not mentioned | None (0.0) | — |
| 30 | Dropout rate | Not mentioned in main text | 0.05 | **Difference** — paper doesn't mention dropout explicitly |
| 31 | Early stopping | Yes | Yes | Match |
| 32 | Early stopping patience | Not mentioned (Internet Appendix) | 25 epochs | Do not know paper's value |
| 33 | Early stopping metric | Validation prediction error (MSE) | Validation cross-sectional IC | **Difference** — paper uses val MSE, we use val IC |
| 34 | Early stopping min_delta | Not mentioned | 0 (any improvement counts) | — |
| 35 | Label smoothing | No | No | Match |
| 36 | Data augmentation / noise injection | No | No | Match |
| 37 | Stochastic depth | No | No | Match |

## Weight Initialization

| # | Parameter | GKX Paper | Ours | Notes |
|---|-----------|-----------|------|-------|
| 38 | Init method | Not mentioned (Internet Appendix) | Kaiming/He normal (for ReLU) | Standard choice for ReLU |
| 39 | Init gain/scale | Not mentioned | Default (sqrt(2) for ReLU) | — |
| 40 | Bias init | Not mentioned | Zeros (PyTorch default) | — |

## Loss Function

| # | Parameter | GKX Paper | Ours | Notes |
|---|-----------|-----------|------|-------|
| 41 | Loss type | MSE (l2) + L1 penalty | MSE only | **Difference** — we don't add L1 penalty to loss |
| 42 | Huber delta | N/A (NN uses MSE, not Huber) | N/A | Paper uses Huber only for OLS/ENet/GLM/GBRT |
| 43 | Sample weighting | Not mentioned (likely equal) | Equal | Match (assumed) |

## Features / Data

| # | Parameter | GKX Paper | Ours | Notes |
|---|-----------|-----------|------|-------|
| 44 | Stock characteristics | 94 | 95 | **Difference** — we include mvel1, paper uses 94 |
| 45 | Macro predictors | 8 (dp, ep, bm, ntis, tbl, tms, dfy, svar) | 8 (same) | Match |
| 46 | Interaction features | 94 x 9 = 846 (with constant) | 95 x 8 = 760 (no constant) | **Difference** — paper includes constant interaction (= raw signal); we don't include constant but have raw signals separately |
| 47 | Industry dummies | 74 (SIC 2-digit) | None | **Difference** — we don't use industry dummies |
| 48 | Total input dim | 920 | 863 | **Difference** |
| 49 | Feature scaling | Cross-sectional rank to [-1, 1] each month | StandardScaler (nanmean/nanstd over training set) | **Difference** — paper ranks per month, we standardize over full training period |
| 50 | Missing imputation | Cross-sectional median per month | 0 after scaling (= feature mean) | **Difference** — paper uses monthly median, we use training-set mean |
| 51 | Outlier clipping | Not mentioned | +/- 5 std after scaling | **Difference** — we clip, paper doesn't mention (rank to [-1,1] naturally bounds) |
| 52 | Feature selection | All 920 | All 863 | Both use all features |
| 53 | Target variable | Excess return (return - Rfree) | Excess return (return - Rfree) | Match |
| 54 | Target transform | None (raw excess return) | None (raw excess return) | Match |

## Training Procedure

| # | Parameter | GKX Paper | Ours | Notes |
|---|-----------|-----------|------|-------|
| 55 | Training start year | 1957 | 1975 | **Difference** — paper has 18 more years of training data |
| 56 | Train/val split method | Chronological, expanding window | Chronological, expanding window | Match |
| 57 | Validation window size | 12 years (fixed, rolling) | 1 year (year before test) | **Difference** — paper uses 12-year val, we use 1-year val |
| 58 | Test period | 1987-2016 (30 years) | 2016-2019 (4 years) | **Difference** — paper tests 30 years, we test 4 years |
| 59 | Refit frequency | Yearly | Yearly | Match |
| 60 | Signal-return alignment | Signal_t predicts return_{t+1} | Signal_t predicts return_{t+1} | Match |

## Ensemble

| # | Parameter | GKX Paper | Ours | Notes |
|---|-----------|-----------|------|-------|
| 61 | Number of seeds | Not mentioned in main text | 10 | Paper detail in Internet Appendix |
| 62 | Aggregation method | Mean (average predictions) | Mean | Match |
| 63 | Model selection | All seeds averaged | All seeds averaged | Match |

## Hardware / Reproducibility

| # | Parameter | GKX Paper | Ours | Notes |
|---|-----------|-----------|------|-------|
| 64 | Random seed | Not mentioned | 0-9 (10 seeds) | — |
| 65 | CUDA deterministic | Not mentioned | Yes (cudnn.deterministic=True) | — |
| 66 | CUDA benchmark | Not mentioned | Disabled (cudnn.benchmark=False) | — |
| 67 | NumPy seed | Not mentioned | Set per run | — |
| 68 | DataLoader workers | Not mentioned | 0 (GPU-resident data) | — |
| 69 | Pin memory | Not mentioned | N/A (data on GPU) | — |
| 70 | Mixed precision | Not mentioned | fp16 via torch.amp | — |

## Evaluation Metrics

| # | Parameter | GKX Paper | Ours | Notes |
|---|-----------|-----------|------|-------|
| 71 | OOS R² formula | 1 - sum(r-r_hat)^2 / sum(r^2) | Same | Match — denominator NOT demeaned |
| 72 | Cross-sectional IC | Not reported as primary metric | Monthly Spearman rank IC | We report IC, paper uses R² |
| 73 | Long-short portfolio | Decile 10 - Decile 1 (value-weighted) | Decile 10 - Decile 1 (equal-weighted) | **Difference** — paper value-weights, we equal-weight |
| 74 | Portfolio Sharpe ratio | Reported | Reported | Match |
| 75 | Diebold-Mariano test | Yes (modified for panel) | No | **Difference** — we don't do DM tests |

---

## Summary of Key Differences

| Parameter | Paper | Ours | Impact |
|-----------|-------|------|--------|
| L1 penalty | Tuned via validation | None | Paper has extra regularization |
| Feature scaling | Rank to [-1,1] per month | StandardScaler + clip ±5 | Rank is more robust to outliers |
| Missing imputation | Monthly cross-sectional median | Training-set mean (0 after scaling) | Paper is more adaptive |
| Industry dummies | 74 features | Not used | 57 fewer features |
| Validation window | 12 years | 1 year | Paper has more stable val signal |
| Training start | 1957 | 1975 | Paper has 18 more years |
| Test period | 30 years | 4 years | Paper tests over many regimes |
| Early stopping metric | Val MSE | Val cross-sectional IC | Different optimization target |
| Dropout | Not mentioned | 0.05 | We add extra regularization |
| Portfolio weighting | Value-weighted | Equal-weighted | Different economic interpretation |

Total parameters: **75**
Matching: ~45
Different: ~15
Unknown (Internet Appendix): ~15
