"""
Cross-Sectional Transformer with MSRR Loss for Asset Pricing.

Same architecture as train_transformer.py, but instead of predicting returns
with MSE loss, the model outputs portfolio weights and is trained with
Maximum Sharpe Ratio Regression (MSRR) loss:

    L = E[(1 - w(X_t)' R_{t+1})²]

This directly optimizes the stochastic discount factor (SDF), which is
equivalent to finding the mean-variance efficient portfolio.

Reference: Kelly, Kuznetsov, Malamud, Xu (2025) "Artificial Intelligence
Asset Pricing Models", NBER Working Paper 33351.
"""

import os
import gc
import csv
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr

from train_nn import (
    Config,
    setup_logging,
    load_returns,
    load_universe,
    load_signals,
    load_macro,
    load_sector_mapping,
    build_long_panel,
    build_industry_dummies,
    FeatureScaler,
    compute_cross_sectional_ic,
    compute_oos_metrics,
    set_seed,
)

from train_transformer import (
    TransformerFeatureScaler,
    MonthGroupedData,
    CrossSectionalTransformer,
    evaluate,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MSRRConfig:
    # Data
    data_dir: str = "ml_alpha_data/gkx_full"
    macro_file: str = "ml_alpha_data/welch_goyal_2024.xlsx"
    sector_file: str = "ml_alpha_data/gkx_full/sector_mapping.csv"
    output_dir: str = "output"

    # Architecture (same as MSE transformer)
    d_model: int = 32
    n_heads: int = 4
    n_layers: int = 1
    d_ff: int = 64
    dropout: float = 0.10

    # Features
    n_signals: int = 95
    n_macro: int = 8
    n_industries: int = 74

    # Training
    lr: float = 7.5e-5
    weight_decay: float = 0.0   # no decay on Transformer body
    ridge_lambda: float = 1e-3  # ridge penalty on output head only (Kelly et al. approach)
    grad_accum_steps: int = 4
    max_grad_norm: float = 1.0
    max_epochs: int = 300
    patience: int = 25
    min_epochs: int = 20
    n_seeds: int = 10
    clip_std: float = 5.0

    # Time periods
    train_start: int = 1975
    val_years: int = 1
    test_years: List[int] = field(default_factory=lambda: [2016, 2017, 2018, 2019])

    # Signal/macro names
    signal_names: List[str] = field(default_factory=lambda: Config().signal_names)
    macro_names: List[str] = field(default_factory=lambda: Config().macro_names)


# ---------------------------------------------------------------------------
# MSRR Loss
# ---------------------------------------------------------------------------

def msrr_loss_month(weights: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    """MSRR loss for a single month.

    Args:
        weights: (N,) portfolio weights from model
        returns: (N,) excess returns for that month

    Returns:
        scalar loss: (1 - w'R)²
    """
    port_return = torch.dot(weights, returns)
    return (1.0 - port_return) ** 2


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch_msrr(model: nn.Module, data: MonthGroupedData,
                         optimizer: torch.optim.Optimizer,
                         amp_scaler: torch.amp.GradScaler,
                         config: MSRRConfig,
                         device: torch.device) -> float:
    """Train one epoch with MSRR loss."""
    model.train()
    total_loss = 0.0
    n_months = len(data)

    month_order = np.random.permutation(data.months)
    optimizer.zero_grad(set_to_none=True)

    for step, month_id in enumerate(month_order):
        stock, macro, ind, target = data.get_month(int(month_id), device)

        with torch.amp.autocast("cuda"):
            weights = model(stock, macro, ind)  # (N,) — portfolio weights
            loss = msrr_loss_month(weights, target) / config.grad_accum_steps

        amp_scaler.scale(loss).backward()
        total_loss += loss.item() * config.grad_accum_steps

        if (step + 1) % config.grad_accum_steps == 0:
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            amp_scaler.step(optimizer)
            amp_scaler.update()
            optimizer.zero_grad(set_to_none=True)

    if n_months % config.grad_accum_steps != 0:
        amp_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        amp_scaler.step(optimizer)
        amp_scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return total_loss / n_months


@torch.no_grad()
def evaluate_msrr(model: nn.Module, data: MonthGroupedData,
                  device: torch.device) -> Tuple[float, np.ndarray, np.ndarray,
                                                   np.ndarray, np.ndarray]:
    """Evaluate MSRR loss and collect predictions for OOS metrics.

    Returns:
        avg_msrr_loss: average MSRR loss across months
        all_preds: flat array of model outputs (weights/scores)
        all_targets: flat array of actual returns
        all_months: flat array of month IDs
        all_permnos: flat array of permno IDs
    """
    model.eval()
    total_loss = 0.0
    all_preds, all_targets, all_months, all_permnos = [], [], [], []

    for month_id in data.months:
        stock, macro, ind, target = data.get_month(month_id, device)
        with torch.amp.autocast("cuda"):
            weights = model(stock, macro, ind)

        loss = msrr_loss_month(weights.float(), target)
        total_loss += loss.item()

        all_preds.append(weights.float().cpu().numpy())
        all_targets.append(target.cpu().numpy())
        all_months.append(np.full(len(target), month_id, dtype=np.int32))
        all_permnos.append(data.permno_dict[month_id])

    avg_loss = total_loss / len(data)
    return (avg_loss, np.concatenate(all_preds), np.concatenate(all_targets),
            np.concatenate(all_months), np.concatenate(all_permnos))


def compute_sdf_portfolio_metrics(preds: np.ndarray, targets: np.ndarray,
                                  month_ids: np.ndarray,
                                  logger: logging.Logger) -> Dict:
    """Compute SDF portfolio metrics using raw model weights.

    The model outputs are used directly as portfolio weights (not sorted
    into deciles). The SDF portfolio return each month is w'R.
    """
    unique_months = sorted(np.unique(month_ids))
    monthly_returns = []

    for m in unique_months:
        mask = month_ids == m
        w = preds[mask]
        r = targets[mask]
        port_ret = np.dot(w, r)
        monthly_returns.append(port_ret)

    monthly_returns = np.array(monthly_returns)
    mean_ret = np.mean(monthly_returns)
    std_ret = np.std(monthly_returns, ddof=1) if len(monthly_returns) > 1 else 1.0
    sharpe = mean_ret / std_ret * np.sqrt(12) if std_ret > 0 else 0.0

    logger.info(f"    SDF Portfolio: mean={mean_ret:.6f}/mo, "
                f"std={std_ret:.6f}, Sharpe={sharpe:.2f}")

    return {
        "sdf_mean_ret": mean_ret,
        "sdf_std_ret": std_ret,
        "sdf_sharpe": sharpe,
        "sdf_monthly_returns": monthly_returns,
    }


def train_model_msrr(train_data: MonthGroupedData, val_data: MonthGroupedData,
                     test_year: int, seed: int,
                     config: MSRRConfig, device: torch.device,
                     logger: logging.Logger) -> Dict:
    """Full training with early stopping on validation MSRR loss."""
    set_seed(seed)

    model = CrossSectionalTransformer(
        n_signals=config.n_signals, n_industries=config.n_industries,
        n_macro=config.n_macro, d_model=config.d_model,
        n_heads=config.n_heads, d_ff=config.d_ff, dropout=config.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    head_params = sum(p.numel() for p in model.output_head.parameters())
    logger.info(f"  [MSRR|year{test_year}|seed{seed}] params={n_params:,} "
                f"(head={head_params:,}), "
                f"train={train_data.n_obs:,}, val={val_data.n_obs:,}")

    # Split optimizer: no weight decay on Transformer body, ridge on output head only
    body_params = [p for n, p in model.named_parameters()
                   if not n.startswith("output_head")]
    head_params_list = list(model.output_head.parameters())
    optimizer = torch.optim.AdamW([
        {"params": body_params, "weight_decay": 0.0},
        {"params": head_params_list, "weight_decay": config.ridge_lambda},
    ], lr=config.lr)
    amp_scaler = torch.amp.GradScaler("cuda")

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0

    t0 = time.time()
    for epoch in range(1, config.max_epochs + 1):
        t_ep = time.time()

        train_loss = train_one_epoch_msrr(
            model, train_data, optimizer, amp_scaler, config, device)

        # Validation — MSRR loss
        val_loss, val_preds, val_targets, val_months, _ = evaluate_msrr(
            model, val_data, device)
        val_ic = compute_cross_sectional_ic(val_preds, val_targets, val_months)

        elapsed = time.time() - t_ep
        if epoch <= 5 or epoch % 10 == 0 or epoch == config.max_epochs:
            logger.debug(f"    Epoch {epoch:3d}: trn={train_loss:.6f}, "
                         f"val_msrr={val_loss:.6f}, val_ic={val_ic:.4f}, "
                         f"pat={patience_counter}/{config.patience}, {elapsed:.1f}s")

        # Early stopping on MSRR loss (lower is better)
        if epoch >= config.min_epochs:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    logger.info(f"    Early stopped epoch {epoch}, best={best_epoch}")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    total_time = time.time() - t0
    logger.info(f"    seed{seed}: ep={best_epoch}, val_msrr={best_val_loss:.6f}, "
                f"{total_time:.1f}s")

    return {
        "model": model,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    config = MSRRConfig()
    logger = setup_logging(config.output_dir)

    logger.info("=" * 70)
    logger.info("Cross-Sectional Transformer with MSRR Loss")
    logger.info(f"  d_model={config.d_model}, heads={config.n_heads}, "
                f"layers={config.n_layers}, d_ff={config.d_ff}")
    logger.info(f"  lr={config.lr}, wd={config.weight_decay}, "
                f"dropout={config.dropout}, grad_accum={config.grad_accum_steps}")
    logger.info(f"  Loss: MSRR  L = E[(1 - w'R)^2]")
    logger.info(f"  test_years={config.test_years}")
    logger.info("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    for subdir in ["logs", "models", "predictions", "metrics", "features"]:
        os.makedirs(os.path.join(config.output_dir, subdir), exist_ok=True)

    # --- Load data (reuse from train_nn) ---
    end_year = max(config.test_years)
    logger.info("Loading data...")
    returns = load_returns(config.data_dir, config.train_start, end_year, logger)
    universe = load_universe(config.data_dir, config.train_start, end_year, logger)
    signals = load_signals(config.data_dir, config.signal_names,
                           config.train_start, end_year, logger)
    macro, rfree = load_macro(config.macro_file, config.train_start, end_year, logger)
    permno_to_sic2, sic2_codes = load_sector_mapping(config.sector_file, logger)

    logger.info("Building long panel...")
    stock_features, macro_features, targets, month_ids, permno_ids = \
        build_long_panel(universe, returns, signals, macro, rfree,
                         config.signal_names, config.macro_names, logger)

    del returns, universe, signals, macro, rfree
    gc.collect()

    logger.info(f"Panel: {len(targets):,} obs, "
                f"{len(np.unique(month_ids))} months, "
                f"{stock_features.shape[1]} signals")

    # --- Results ---
    all_results = []
    csv_path = os.path.join(config.output_dir, "metrics", "msrr_transformer_summary.csv")
    csv_fields = ["test_year", "oos_r2_pct", "mean_ic", "std_ic",
                  "mean_ls_ret_pct", "std_ls_ret_pct", "sharpe_ls_annual",
                  "sdf_sharpe", "sdf_mean_ret", "sdf_std_ret",
                  "n_months", "n_obs", "avg_epochs"]

    # --- Yearly expanding-window refit ---
    for test_year in config.test_years:
        logger.info(f"\n{'='*60}")
        logger.info(f"Test year: {test_year}")
        logger.info(f"{'='*60}")

        val_end_year = test_year - 1
        val_start_year = test_year - config.val_years
        train_end_year = val_start_year - 1

        train_end_month = train_end_year * 100 + 12
        val_start_month = val_start_year * 100 + 1
        val_end_month = val_end_year * 100 + 12
        test_start_month = test_year * 100 + 1
        test_end_month = test_year * 100 + 12

        train_mask = month_ids <= train_end_month
        val_mask = (month_ids >= val_start_month) & (month_ids <= val_end_month)
        test_mask = (month_ids >= test_start_month) & (month_ids <= test_end_month)

        logger.info(f"  Train: <={train_end_year}-12 ({train_mask.sum():,})")
        logger.info(f"  Val:   {val_start_year}-01..{val_end_year}-12 ({val_mask.sum():,})")
        logger.info(f"  Test:  {test_year} ({test_mask.sum():,})")

        scaler = TransformerFeatureScaler(clip_std=config.clip_std)
        scaler.fit(stock_features[train_mask], macro_features[train_mask])

        def build_split(mask):
            s, m = scaler.transform(stock_features[mask].copy(),
                                    macro_features[mask].copy())
            ind = build_industry_dummies(permno_ids[mask], permno_to_sic2, sic2_codes)
            data = MonthGroupedData(s, m, ind, targets[mask],
                                    month_ids[mask], permno_ids[mask])
            del s, m, ind
            return data

        train_data = build_split(train_mask)
        val_data = build_split(val_mask)
        test_data = build_split(test_mask)

        logger.info(f"  Train months: {len(train_data)}, "
                     f"Val months: {len(val_data)}, "
                     f"Test months: {len(test_data)}")

        # Train 10 seeds
        seed_preds = []
        seed_epochs = []
        for seed in range(config.n_seeds):
            result = train_model_msrr(train_data, val_data, test_year, seed,
                                      config, device, logger)

            # Evaluate on test set
            preds, _, _, _ = evaluate(result["model"], test_data, device)
            seed_preds.append(preds)
            seed_epochs.append(result["best_epoch"])

            # Save model weights
            model_path = os.path.join(config.output_dir, "models",
                                      f"MSRR_year{test_year}_seed{seed}.pt")
            torch.save(result["model"].state_dict(), model_path)

            del result
            torch.cuda.empty_cache()

        # Ensemble
        ensemble_preds = np.mean(seed_preds, axis=0)
        test_targets = np.concatenate([test_data.target_dict[m]
                                       for m in test_data.months])
        test_months = np.concatenate([np.full(len(test_data.target_dict[m]), m,
                                              dtype=np.int32)
                                      for m in test_data.months])

        # Save ensemble predictions
        test_permnos = np.concatenate([test_data.permno_dict[m]
                                       for m in test_data.months])
        import pandas as pd
        pred_df = pd.DataFrame({
            "permno": test_permnos, "month": test_months,
            "prediction": ensemble_preds,
        })
        pred_path = os.path.join(config.output_dir, "predictions",
                                 f"pred_ensemble_MSRR_year{test_year}.parquet")
        pred_df.to_parquet(pred_path, index=False)

        logger.info(f"  ENSEMBLE ({config.n_seeds} seeds):")

        # Standard decile-sort metrics (for comparison with MSE transformer)
        metrics = compute_oos_metrics(ensemble_preds, test_targets,
                                      test_months, logger)

        # SDF portfolio metrics (direct w'R)
        sdf_metrics = compute_sdf_portfolio_metrics(
            ensemble_preds, test_targets, test_months, logger)

        metrics.update(sdf_metrics)
        metrics["test_year"] = test_year
        metrics["avg_epochs"] = np.mean(seed_epochs)
        all_results.append(metrics)

        # Save CSV
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            for r in all_results:
                writer.writerow({k: r.get(k, "") for k in csv_fields})

        del train_data, val_data, test_data, seed_preds
        gc.collect()
        torch.cuda.empty_cache()

    # --- Summary ---
    logger.info("\n" + "=" * 70)
    logger.info("MSRR TRANSFORMER RESULTS SUMMARY")
    logger.info("=" * 70)
    for r in all_results:
        logger.info(f"  {r['test_year']}: R²={r['oos_r2_pct']:+.4f}%, "
                     f"IC={r['mean_ic']:.4f}, "
                     f"L/S Sharpe={r['sharpe_ls_annual']:.2f}, "
                     f"SDF Sharpe={r['sdf_sharpe']:.2f}, "
                     f"epochs={r['avg_epochs']:.0f}")

    r2s = [r["oos_r2_pct"] for r in all_results]
    ics = [r["mean_ic"] for r in all_results]
    ls_shs = [r["sharpe_ls_annual"] for r in all_results]
    sdf_shs = [r["sdf_sharpe"] for r in all_results]
    logger.info(f"  AVG:  R²={np.mean(r2s):+.4f}%, IC={np.mean(ics):.4f}, "
                f"L/S Sharpe={np.mean(ls_shs):.2f}, SDF Sharpe={np.mean(sdf_shs):.2f}")

    logger.info(f"\nSaved to {csv_path}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
