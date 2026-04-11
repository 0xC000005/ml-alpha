# ml-alpha

Machine learning models for cross-sectional stock return prediction. Replicates and extends the feedforward neural network benchmark from Gu, Kelly, and Xiu (2020), and introduces a cross-sectional Transformer that learns stock-stock interactions via self-attention.

## Models

| Model | Parameters | Features | Notes |
|-------|-----------|----------|-------|
| **NN5 (FFN)** | ~30K | 937 (95 signals + 8 macro + 760 hand-crafted interactions + 74 industry dummies) | Replication of GKX (2020) NN5 architecture |
| **Cross-Sectional Transformer** | ~14K | 169 per stock + 8 macro (no hand-crafted interactions) | Self-attention across all stocks in a month |

## Results (2016-2019 out-of-sample)

| Model | Avg OOS R² | Avg IC | Avg Sharpe | Positive years |
|-------|-----------|--------|------------|----------------|
| NN5 (FFN) | -1.18% | +0.012 | +1.25 | 3/4 |
| Transformer | **+0.47%** | **+0.023** | **+1.81** | **4/4** |

The Transformer wins on every average metric despite having **half the parameters** and **no hand-crafted interaction features**.

Full 19-year results (2001-2019) for the FFN are in `MSE_ind_1yr_report.md` (average Sharpe +1.07).

## Reports

- **`MSE_ind_1yr_report.md`** — Full replication report for the NN5 FFN (2001-2019)
- **`Transformer_report.md`** — Cross-sectional Transformer report (2016-2019) with head-to-head comparison

## Code

- **`train_nn.py`** — Data pipeline, FFN training, 8-experiment grid configurations
- **`train_transformer.py`** — Cross-sectional Transformer, reuses data pipeline from `train_nn.py`

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

# Cross-sectional Transformer
python train_transformer.py
```

Edit `Config` / `TransformerConfig` at the top of each file to change test years, hyperparameters, etc.

## Citation

This work builds on:

> Gu, S., Kelly, B., & Xiu, D. (2020). **Empirical Asset Pricing via Machine Learning**. *The Review of Financial Studies*, 33(5), 2223-2273. https://doi.org/10.1093/rfs/hhaa009

If you use this code, please cite the original GKX paper.

## Disclaimer

This repository is for research and educational purposes only. It is **not** investment advice. Past performance does not guarantee future results. The authors of this repository are not affiliated with Gu, Kelly, or Xiu.
