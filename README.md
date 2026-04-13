# ml-alpha

Machine learning models for cross-sectional stock return prediction. Replicates and extends the feedforward neural network benchmark from Gu, Kelly, and Xiu (2020), and introduces a cross-sectional Transformer that learns stock-stock interactions via self-attention. Also experiments with the MSRR loss function from Kelly et al. (2025) that directly optimizes portfolio Sharpe ratio.

## Models

| Model | Parameters | Features | Loss | Notes |
|-------|-----------|----------|------|-------|
| **NN5 (FFN)** | ~30K | 937 (95 signals + 8 macro + 760 interactions + 74 dummies) | MSE | Replication of GKX (2020) |
| **Cross-Sectional Transformer** | ~14K | 169 per stock + 8 macro | MSE | Self-attention across stocks |
| **MSRR Transformer** | ~14K | 169 per stock + 8 macro | MSRR | Same architecture, portfolio-optimized loss |

## Results

### MSE Models (2012-2019, L/S decile Sharpe)

| Model | Avg OOS R² | Avg IC | Avg L/S %/mo | Avg Sharpe | Positive years |
|-------|-----------|--------|-------------|------------|----------------|
| NN5 (FFN) | -0.55% | +0.014 | +0.91% | +1.63 | 7/8 |
| Transformer (MSE) | **-0.08%** | **+0.021** | **+1.53%** | **+2.16** | **8/8** |

### MSE vs MSRR Transformer (2016-2019)

| Model | Portfolio | Avg Sharpe | Best Year | Worst Year |
|-------|-----------|------------|-----------|------------|
| Transformer (MSE) | L/S decile sort | +1.81 | 2017 (+3.07) | 2016 (+0.58) |
| Transformer (MSRR) | SDF (direct weights) | **+2.05** | 2016 (+3.03) | 2018 (+0.82) |

The MSRR loss directly optimizes portfolio Sharpe ratio instead of return prediction accuracy. Same architecture, same data — only the loss function differs. The SDF portfolio uses model outputs as portfolio weights directly (scale-invariant Sharpe).

Full 19-year results (2001-2019) for the FFN are in `MSE_ind_1yr_report.md` (average Sharpe +1.07).

## Reports

- **`MSE_ind_1yr_report.md`** — Full replication report for the NN5 FFN (2001-2019)
- **`Transformer_report.md`** — Cross-sectional Transformer report (2012-2019) with head-to-head comparison
- **`MSRR_Transformer_report.md`** — MSRR loss experiment (2016-2019), proof-of-concept

## Code

- **`train_nn.py`** — Data pipeline, FFN training, 8-experiment grid configurations
- **`train_transformer.py`** — Cross-sectional Transformer with MSE loss
- **`train_transformer_msrr.py`** — Cross-sectional Transformer with MSRR loss (Kelly et al. 2025)

## Data (Not Included)

The data files are **not** included in this repository due to licensing restrictions. You need to obtain them separately:

1. **Stock characteristics (95 signals)** and **returns** — derived from CRSP/Compustat via the 94 signals defined in Green, Hand, and Zhang (2017) and extended in GKX (2020). Available with a WRDS subscription. The processed signal files should be placed in `gkx_full/`.

2. **Sector mapping** (`gkx_full/sector_mapping.csv`) — SIC 2-digit industry codes per PERMNO, from CRSP.

3. **Welch-Goyal macroeconomic predictors** (`welch_goyal_2024.xlsx`) — publicly available from Amit Goyal's website: http://www.hec.unil.ch/agoyal/

The expected data layout is documented in `MSE_ind_1yr_report.md` section 10.2.

## Requirements

```
Python 3.10+
PyTorch 2.0+ (CUDA)
numpy, pandas, scipy, openpyxl
NVIDIA GPU with ≥16GB VRAM (tested on RTX 4080 SUPER)
```

## Running

```bash
# FFN (NN5), MSE_ind_1yr configuration
python train_nn.py

# Cross-sectional Transformer (MSE loss)
python train_transformer.py

# Cross-sectional Transformer (MSRR loss)
python train_transformer_msrr.py
```

Edit `Config` / `TransformerConfig` / `MSRRConfig` at the top of each file to change test years, hyperparameters, etc.

## Citation

This work builds on:

> Gu, S., Kelly, B., & Xiu, D. (2020). **Empirical Asset Pricing via Machine Learning**. *The Review of Financial Studies*, 33(5), 2223-2273. https://doi.org/10.1093/rfs/hhaa009

> Kelly, B.T., Kuznetsov, B., Malamud, S., & Xu, T.A. (2025). **Artificial Intelligence Asset Pricing Models**. NBER Working Paper 33351.

If you use this code, please cite the original papers.

## Disclaimer

This repository is for research and educational purposes only. It is **not** investment advice. Past performance does not guarantee future results. The authors of this repository are not affiliated with Gu, Kelly, or Xiu.
