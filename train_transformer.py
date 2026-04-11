"""
Cross-Sectional Transformer for Asset Pricing.

Processes all stocks within each month via self-attention, letting
the model learn stock-stock interactions instead of hand-crafting them.

Imports data pipeline from train_nn.py.
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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TransformerConfig:
    # Data (reuse paths from Config)
    data_dir: str = "gkx_full"
    macro_file: str = "welch_goyal_2024.xlsx"
    sector_file: str = "gkx_full/sector_mapping.csv"
    output_dir: str = "output"

    # Architecture
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
    lr: float = 1e-4
    weight_decay: float = 1e-4
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

    # Signal/macro names (reuse from Config)
    signal_names: List[str] = field(default_factory=lambda: Config().signal_names)
    macro_names: List[str] = field(default_factory=lambda: Config().macro_names)


# ---------------------------------------------------------------------------
# Feature Scaling (no interactions)
# ---------------------------------------------------------------------------

class TransformerFeatureScaler:
    """Scale stock/macro features separately. No interaction computation."""

    def __init__(self, clip_std: float = 5.0):
        self.clip_std = clip_std
        self._inner = FeatureScaler(clip_std=clip_std)

    def fit(self, stock_features: np.ndarray, macro_features: np.ndarray):
        self._inner.fit(stock_features, macro_features)
        return self

    def transform(self, stock_features: np.ndarray,
                  macro_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (stock_scaled, macro_scaled) separately — no interactions."""
        stock = (stock_features - self._inner.stock_mean_) / self._inner.stock_std_
        macro = (macro_features - self._inner.macro_mean_) / self._inner.macro_std_
        np.nan_to_num(stock, copy=False, nan=0.0)
        np.nan_to_num(macro, copy=False, nan=0.0)
        np.clip(stock, -self.clip_std, self.clip_std, out=stock)
        np.clip(macro, -self.clip_std, self.clip_std, out=macro)
        return stock.astype(np.float32), macro.astype(np.float32)


# ---------------------------------------------------------------------------
# Month-Grouped Data Container
# ---------------------------------------------------------------------------

class MonthGroupedData:
    """Organizes panel data into per-month groups for Transformer batching.

    Stores data on CPU as numpy arrays. Transfers one month to GPU per call.
    """

    def __init__(self, stock_features: np.ndarray, macro_features: np.ndarray,
                 industry_dummies: np.ndarray, targets: np.ndarray,
                 month_ids: np.ndarray, permno_ids: np.ndarray):
        unique_months = np.unique(month_ids)
        self.months = sorted(unique_months.tolist())

        self.stock_dict: Dict[int, np.ndarray] = {}
        self.macro_dict: Dict[int, np.ndarray] = {}
        self.ind_dict: Dict[int, np.ndarray] = {}
        self.target_dict: Dict[int, np.ndarray] = {}
        self.permno_dict: Dict[int, np.ndarray] = {}

        for m in self.months:
            mask = month_ids == m
            self.stock_dict[m] = stock_features[mask]
            # Macro is same for all stocks in a month — store just one row
            self.macro_dict[m] = macro_features[mask][0:1]
            self.ind_dict[m] = industry_dummies[mask]
            self.target_dict[m] = targets[mask]
            self.permno_dict[m] = permno_ids[mask]

    def __len__(self):
        return len(self.months)

    def get_month(self, month_id: int, device: torch.device):
        """Transfer one month's data to GPU. Returns unsqueezed batch dim."""
        stock = torch.from_numpy(self.stock_dict[month_id]).unsqueeze(0).to(device)
        macro = torch.from_numpy(self.macro_dict[month_id]).to(device)  # (1, 8)
        ind = torch.from_numpy(self.ind_dict[month_id]).unsqueeze(0).to(device)
        target = torch.from_numpy(self.target_dict[month_id]).to(device)
        return stock, macro, ind, target

    @property
    def n_obs(self):
        return sum(len(self.target_dict[m]) for m in self.months)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CrossSectionalTransformer(nn.Module):
    """Transformer that processes all stocks in a month via self-attention.

    Architecture:
        stock_proj(169->d_model) + macro_proj(8->d_model)  [additive]
        -> LayerNorm -> MultiHeadSelfAttention -> residual
        -> LayerNorm -> FFN(d_model->d_ff->d_model) -> residual
        -> LayerNorm -> Linear(d_model->1)
    """

    def __init__(self, n_signals: int = 95, n_industries: int = 74,
                 n_macro: int = 8, d_model: int = 32, n_heads: int = 4,
                 d_ff: int = 64, dropout: float = 0.10):
        super().__init__()
        self.stock_proj = nn.Linear(n_signals + n_industries, d_model)
        self.macro_proj = nn.Linear(n_macro, d_model)

        # Pre-norm Transformer block
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        # Output head
        self.norm_out = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, stock_features: torch.Tensor,
                macro_features: torch.Tensor,
                industry_dummies: torch.Tensor) -> torch.Tensor:
        """
        Args:
            stock_features: (1, N, n_signals)
            macro_features: (1, n_macro)
            industry_dummies: (1, N, n_industries)
        Returns:
            predictions: (N,)
        """
        # Concatenate stock signals + industry dummies
        x = torch.cat([stock_features, industry_dummies], dim=-1)  # (1, N, 169)
        x = self.stock_proj(x)  # (1, N, d_model)

        # Additive macro conditioning (broadcast across stocks)
        macro_embed = self.macro_proj(macro_features)  # (1, d_model)
        x = x + macro_embed.unsqueeze(1)  # (1, N, d_model)

        # Pre-norm self-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_out)

        # Pre-norm FFN with residual
        x_norm = self.norm2(x)
        x = x + self.ffn(x_norm)

        # Output
        x = self.norm_out(x)
        preds = self.output_head(x).squeeze(-1).squeeze(0)  # (N,)
        return preds


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model: nn.Module, data: MonthGroupedData,
                    optimizer: torch.optim.Optimizer,
                    criterion: nn.Module,
                    amp_scaler: torch.amp.GradScaler,
                    config: TransformerConfig,
                    device: torch.device) -> float:
    """Train one epoch: iterate over months with gradient accumulation."""
    model.train()
    total_loss = 0.0
    n_months = len(data)

    month_order = np.random.permutation(data.months)
    optimizer.zero_grad(set_to_none=True)

    for step, month_id in enumerate(month_order):
        stock, macro, ind, target = data.get_month(int(month_id), device)

        with torch.amp.autocast("cuda"):
            preds = model(stock, macro, ind)
            loss = criterion(preds, target) / config.grad_accum_steps

        amp_scaler.scale(loss).backward()
        total_loss += loss.item() * config.grad_accum_steps

        if (step + 1) % config.grad_accum_steps == 0:
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            amp_scaler.step(optimizer)
            amp_scaler.update()
            optimizer.zero_grad(set_to_none=True)

    # Handle remaining steps
    if n_months % config.grad_accum_steps != 0:
        amp_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        amp_scaler.step(optimizer)
        amp_scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return total_loss / n_months


@torch.no_grad()
def evaluate(model: nn.Module, data: MonthGroupedData,
             device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Predict all months, return flat arrays."""
    model.eval()
    all_preds, all_targets, all_months, all_permnos = [], [], [], []

    for month_id in data.months:
        stock, macro, ind, target = data.get_month(month_id, device)
        with torch.amp.autocast("cuda"):
            preds = model(stock, macro, ind)
        all_preds.append(preds.float().cpu().numpy())
        all_targets.append(target.cpu().numpy())
        all_months.append(np.full(len(target), month_id, dtype=np.int32))
        all_permnos.append(data.permno_dict[month_id])

    return (np.concatenate(all_preds), np.concatenate(all_targets),
            np.concatenate(all_months), np.concatenate(all_permnos))


def train_model(train_data: MonthGroupedData, val_data: MonthGroupedData,
                test_year: int, seed: int,
                config: TransformerConfig, device: torch.device,
                logger: logging.Logger) -> Dict:
    """Full training with early stopping on validation MSE."""
    set_seed(seed)

    model = CrossSectionalTransformer(
        n_signals=config.n_signals, n_industries=config.n_industries,
        n_macro=config.n_macro, d_model=config.d_model,
        n_heads=config.n_heads, d_ff=config.d_ff, dropout=config.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  [TF|year{test_year}|seed{seed}] params={n_params:,}, "
                f"train={train_data.n_obs:,}, val={val_data.n_obs:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr,
                                  weight_decay=config.weight_decay)
    criterion = nn.MSELoss()
    amp_scaler = torch.amp.GradScaler("cuda")

    best_val_loss = float("inf")
    best_val_ic = -float("inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0

    t0 = time.time()
    for epoch in range(1, config.max_epochs + 1):
        t_ep = time.time()

        train_loss = train_one_epoch(
            model, train_data, optimizer, criterion, amp_scaler, config, device)

        # Validation
        val_preds, val_targets, val_months, _ = evaluate(model, val_data, device)
        val_mse = float(np.mean((val_targets - val_preds) ** 2))
        val_ic = compute_cross_sectional_ic(val_preds, val_targets, val_months)

        elapsed = time.time() - t_ep
        if epoch <= 5 or epoch % 10 == 0 or epoch == config.max_epochs:
            logger.debug(f"    Epoch {epoch:3d}: trn={train_loss:.6f}, "
                         f"val_mse={val_mse:.6f}, val_ic={val_ic:.4f}, "
                         f"pat={patience_counter}/{config.patience}, {elapsed:.1f}s")

        # Early stopping (MSE, with min_epochs)
        if epoch >= config.min_epochs:
            if val_mse < best_val_loss:
                best_val_loss = val_mse
                best_val_ic = val_ic
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
    logger.info(f"    seed{seed}: ep={best_epoch}, ic={best_val_ic:.4f}, {total_time:.1f}s")

    return {
        "model": model,
        "best_epoch": best_epoch,
        "best_val_ic": best_val_ic,
        "best_val_loss": best_val_loss,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    config = TransformerConfig()
    logger = setup_logging(config.output_dir)

    logger.info("=" * 70)
    logger.info("Cross-Sectional Transformer for Asset Pricing")
    logger.info(f"  d_model={config.d_model}, heads={config.n_heads}, "
                f"layers={config.n_layers}, d_ff={config.d_ff}")
    logger.info(f"  lr={config.lr}, wd={config.weight_decay}, "
                f"dropout={config.dropout}, grad_accum={config.grad_accum_steps}")
    logger.info(f"  test_years={config.test_years}")
    logger.info("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    for subdir in ["logs", "models", "predictions", "metrics", "features"]:
        os.makedirs(os.path.join(config.output_dir, subdir), exist_ok=True)

    # --- Load data (reuse from train_nn) ---
    end_year = max(config.test_years)
    base_cfg = Config()

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

    # Free raw data
    del returns, universe, signals, macro, rfree
    gc.collect()

    logger.info(f"Panel: {len(targets):,} obs, "
                f"{len(np.unique(month_ids))} months, "
                f"{stock_features.shape[1]} signals")

    # --- Results collector ---
    all_results = []
    csv_path = os.path.join(config.output_dir, "metrics", "transformer_summary.csv")
    csv_fields = ["test_year", "oos_r2_pct", "mean_ic", "std_ic",
                  "mean_ls_ret_pct", "std_ls_ret_pct", "sharpe_ls_annual",
                  "n_months", "n_obs", "avg_epochs"]

    # --- Yearly expanding-window refit ---
    for test_year in config.test_years:
        logger.info(f"\n{'='*60}")
        logger.info(f"Test year: {test_year}")
        logger.info(f"{'='*60}")

        # Split
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

        # Scale features (no interactions)
        scaler = TransformerFeatureScaler(clip_std=config.clip_std)
        scaler.fit(stock_features[train_mask], macro_features[train_mask])

        # Build MonthGroupedData for each split
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
            result = train_model(train_data, val_data, test_year, seed,
                                 config, device, logger)
            preds, _, _, _ = evaluate(result["model"], test_data, device)
            seed_preds.append(preds)
            seed_epochs.append(result["best_epoch"])

            # Free model
            del result
            torch.cuda.empty_cache()

        # Ensemble
        ensemble_preds = np.mean(seed_preds, axis=0)
        test_targets = np.concatenate([test_data.target_dict[m]
                                       for m in test_data.months])
        test_months = np.concatenate([np.full(len(test_data.target_dict[m]), m,
                                              dtype=np.int32)
                                      for m in test_data.months])

        logger.info(f"  ENSEMBLE ({config.n_seeds} seeds):")
        metrics = compute_oos_metrics(ensemble_preds, test_targets,
                                      test_months, logger)
        metrics["test_year"] = test_year
        metrics["avg_epochs"] = np.mean(seed_epochs)
        all_results.append(metrics)

        # Save intermediate CSV
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            for r in all_results:
                writer.writerow({k: r[k] for k in csv_fields})

        # Cleanup
        del train_data, val_data, test_data, seed_preds
        gc.collect()
        torch.cuda.empty_cache()

    # --- Summary ---
    logger.info("\n" + "=" * 70)
    logger.info("TRANSFORMER RESULTS SUMMARY")
    logger.info("=" * 70)
    for r in all_results:
        logger.info(f"  {r['test_year']}: R²={r['oos_r2_pct']:+.4f}%, "
                     f"IC={r['mean_ic']:.4f}, Sharpe={r['sharpe_ls_annual']:.2f}, "
                     f"epochs={r['avg_epochs']:.0f}")

    r2s = [r["oos_r2_pct"] for r in all_results]
    ics = [r["mean_ic"] for r in all_results]
    shs = [r["sharpe_ls_annual"] for r in all_results]
    logger.info(f"  AVG:  R²={np.mean(r2s):+.4f}%, IC={np.mean(ics):.4f}, "
                f"Sharpe={np.mean(shs):.2f}")

    logger.info(f"\nSaved to {csv_path}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
