"""
Replicate the Neural Network results from:
"Empirical Asset Pricing via Machine Learning" (Gu, Kelly, Xiu 2020)

GPU-accelerated training with expanding window yearly refits,
interaction features, regime features, and detailed logging.
"""

import os
import gc
import json
import pickle
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    data_dir: str = "gkx_full"
    macro_file: str = "welch_goyal_2024.xlsx"
    sector_file: str = "gkx_full/sector_mapping.csv"
    output_dir: str = "output"

    # 95 stock-level signals (all signal_*.parquet files)
    signal_names: List[str] = field(default_factory=lambda: [
        "absacc", "acc", "aeavol", "age", "agr", "baspread", "beta", "betasq",
        "bm", "bm_ia", "cash", "cashdebt", "cashpr", "cfp", "cfp_ia",
        "chatoia", "chcsho", "chempia", "chinv", "chmom", "chpmia", "chtx",
        "cinvest", "convind", "currat", "depr", "divi", "divo", "dolvol", "dy",
        "ear", "egr", "ep", "gma", "grcapx", "grltnoa", "herf", "hire",
        "idiovol", "ill", "indmom", "invest", "lev", "lgr", "maxret",
        "mom12m", "mom1m", "mom36m", "mom6m", "ms", "mve0", "mve_ia", "mvel1",
        "nincr", "operprof", "orgcap", "pchcapx_ia", "pchcurrat", "pchdepr",
        "pchgm_pchsale", "pchquick", "pchsale_pchinvt", "pchsale_pchrect",
        "pchsale_pchxsga", "pchsaleinv", "pctacc", "pricedelay", "ps",
        "quick", "rd", "rd_mve", "rd_sale", "realestate", "retvol", "roaq",
        "roavol", "roeq", "roic", "rsup", "salecash", "saleinv", "salerec",
        "secured", "securedind", "sgr", "sin", "sp", "std_dolvol", "std_turn",
        "stdacc", "stdcf", "tang", "tb", "turn", "zerotrade",
    ])

    # 8 macro predictors (derived from Welch-Goyal)
    macro_names: List[str] = field(default_factory=lambda: [
        "dp", "ep", "bm", "ntis", "tbl", "tms", "dfy", "svar",
    ])

    # NN architectures from the paper
    architectures: Dict[str, Tuple[int, ...]] = field(default_factory=lambda: {
        # "NN1": (32,),
        # "NN2": (32, 16),
        # "NN3": (32, 16, 8),
        # "NN4": (32, 16, 8, 4),
        "NN5": (32, 16, 8, 4, 2),
    })

    # Training
    lr: float = 0.001
    batch_size: int = 10_000
    max_epochs: int = 300
    patience: int = 25
    min_epochs: int = 20  # early stopping only checked after this many epochs
    dropout: float = 0.05
    clip_std: float = 5.0
    n_seeds: int = 10

    # L1 penalty grid (tuned per architecture per year via validation loss)
    l1_lambdas: List[float] = field(default_factory=lambda: [0.0, 1e-5, 5e-5, 1e-4])

    # Time periods
    train_start: int = 1975
    val_years: int = 10  # rolling validation window size in years
    test_years: List[int] = field(default_factory=lambda: [2016, 2017, 2018, 2019])

    @property
    def n_signals(self) -> int:
        return len(self.signal_names)

    @property
    def n_macro(self) -> int:
        return len(self.macro_names)

    n_industries: int = 74  # SIC 2-digit codes

    @property
    def n_features(self) -> int:
        return self.n_signals + self.n_macro + self.n_signals * self.n_macro + self.n_industries


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(output_dir: str) -> logging.Logger:
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_{timestamp}.log")

    logger = logging.getLogger("gkx_nn")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.info(f"Log file: {log_file}")
    return logger


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_returns(data_dir: str, start_year: int, end_year: int,
                 logger: logging.Logger) -> pd.DataFrame:
    """Load returns parquet, filter to date range."""
    path = os.path.join(data_dir, "returns.parquet")
    logger.info(f"Loading returns from {path}")
    df = pd.read_parquet(path)
    mask = (df.index.year >= start_year) & (df.index.year <= end_year)
    df = df.loc[mask].astype(np.float32)
    logger.info(f"Returns shape after filter: {df.shape}")
    return df


def load_universe(data_dir: str, start_year: int, end_year: int,
                  logger: logging.Logger) -> pd.DataFrame:
    """Load universe parquet, filter to date range."""
    path = os.path.join(data_dir, "universe.parquet")
    logger.info(f"Loading universe from {path}")
    df = pd.read_parquet(path)
    mask = (df.index.year >= start_year) & (df.index.year <= end_year)
    df = df.loc[mask]
    logger.info(f"Universe shape after filter: {df.shape}")
    return df


def load_signals(data_dir: str, signal_names: List[str],
                 start_year: int, end_year: int,
                 logger: logging.Logger) -> Dict[str, pd.DataFrame]:
    """Load each signal parquet one at a time, filter dates."""
    signals = {}
    for i, name in enumerate(signal_names):
        path = os.path.join(data_dir, f"signal_{name}.parquet")
        df = pd.read_parquet(path)
        mask = (df.index.year >= start_year) & (df.index.year <= end_year)
        df = df.loc[mask].astype(np.float32)
        nan_pct = df.isna().sum().sum() / df.size * 100
        signals[name] = df
        if (i + 1) % 20 == 0 or i == 0:
            logger.info(f"  Loaded signal {i+1}/{len(signal_names)}: {name} "
                        f"shape={df.shape}, NaN={nan_pct:.1f}%")
    logger.info(f"All {len(signal_names)} signals loaded")
    return signals


def load_macro(macro_file: str, start_year: int, end_year: int,
               logger: logging.Logger) -> Tuple[pd.DataFrame, pd.Series]:
    """Load Welch-Goyal macro predictors and derive the 8 macro variables + Rfree.

    Returns:
        macro_df: DataFrame with 8 macro variables indexed by PeriodIndex
        rfree: Series of monthly risk-free rates indexed by PeriodIndex
    """
    logger.info(f"Loading macro from {macro_file}")
    df = pd.read_excel(macro_file, sheet_name="Monthly")

    # Convert yyyymm to PeriodIndex
    df["period"] = pd.PeriodIndex(df["yyyymm"].astype(int).astype(str),
                                   freq="M")
    df = df.set_index("period")
    mask = (df.index.year >= start_year) & (df.index.year <= end_year)
    df = df.loc[mask]

    # Derive the 8 macro variables
    macro = pd.DataFrame(index=df.index)
    macro["dp"] = np.log(df["D12"]).astype(np.float32) - np.log(df["Index"]).astype(np.float32)
    macro["ep"] = np.log(df["E12"]).astype(np.float32) - np.log(df["Index"]).astype(np.float32)
    macro["bm"] = df["b/m"].astype(np.float32)
    macro["ntis"] = df["ntis"].astype(np.float32)
    macro["tbl"] = df["tbl"].astype(np.float32)
    macro["tms"] = (df["lty"] - df["tbl"]).astype(np.float32)
    macro["dfy"] = (df["BAA"] - df["AAA"]).astype(np.float32)
    macro["svar"] = df["svar"].astype(np.float32)

    rfree = df["Rfree"].astype(np.float32)

    logger.info(f"Macro shape: {macro.shape}, Rfree months: {len(rfree)}")
    logger.info(f"Macro NaN check: {macro.isna().sum().sum()}")
    return macro, rfree


def load_sector_mapping(sector_file: str,
                        logger: logging.Logger) -> Tuple[Dict[int, int], List[int]]:
    """Load PERMNO -> SIC 2-digit mapping and return lookup dict + sorted unique codes.

    Returns:
        permno_to_sic2: dict mapping permno (int) -> sic2 code (int)
        sic2_codes: sorted list of all 74 unique SIC 2-digit codes
    """
    logger.info(f"Loading sector mapping from {sector_file}")
    df = pd.read_csv(sector_file)
    permno_to_sic2 = dict(zip(df["permno"].astype(int), df["sic2"].astype(int)))
    sic2_codes = sorted(df["sic2"].unique().tolist())
    logger.info(f"Sector mapping: {len(permno_to_sic2)} PERMNOs, "
                f"{len(sic2_codes)} unique SIC2 codes")
    return permno_to_sic2, sic2_codes


def build_industry_dummies(permno_ids: np.ndarray,
                           permno_to_sic2: Dict[int, int],
                           sic2_codes: List[int]) -> np.ndarray:
    """One-hot encode industry dummies from PERMNO IDs (vectorized).

    Returns:
        dummies: (N, n_industries) float32 array
    """
    sic2_to_idx = {code: i for i, code in enumerate(sic2_codes)}
    n = len(permno_ids)
    n_ind = len(sic2_codes)

    # Vectorized: map all permnos to sic2 indices at once
    sic2_arr = np.array([permno_to_sic2.get(int(p), -1) for p in permno_ids])
    idx_arr = np.array([sic2_to_idx.get(s, -1) for s in sic2_arr])

    dummies = np.zeros((n, n_ind), dtype=np.float32)
    valid = idx_arr >= 0
    dummies[np.where(valid)[0], idx_arr[valid]] = 1.0
    return dummies


def build_long_panel(
    universe: pd.DataFrame,
    returns: pd.DataFrame,
    signals: Dict[str, pd.DataFrame],
    macro: pd.DataFrame,
    rfree: pd.Series,
    signal_names: List[str],
    macro_names: List[str],
    logger: logging.Logger,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert wide-format data to long-format panel arrays.

    Uses vectorized reindex for fast column alignment instead of per-stock lookups.

    Returns:
        stock_features: (N, n_signals) float32
        macro_features: (N, n_macro) float32
        targets: (N,) float32 -- excess returns
        month_ids: (N,) int32 -- integer encoding of period (yyyymm)
        permno_ids: (N,) int32
    """
    logger.info("Building long-format panel...")

    # Use union of periods available in returns, universe, and macro.
    # Signals may be missing for some periods — we fill with NaN.
    common_periods = universe.index.intersection(returns.index)
    common_periods = common_periods.intersection(macro.index)
    common_periods = common_periods.sort_values()
    logger.info(f"Available periods: {len(common_periods)} months "
                f"({common_periods[0]} to {common_periods[-1]})")

    # Pre-allocate lists
    stock_feat_chunks = []
    macro_feat_chunks = []
    target_chunks = []
    month_id_chunks = []
    permno_id_chunks = []
    total_obs = 0

    # Build mapping from period to next period for return alignment
    all_ret_periods = returns.index.sort_values()
    next_period_map = {all_ret_periods[i]: all_ret_periods[i + 1]
                       for i in range(len(all_ret_periods) - 1)}

    for i, period in enumerate(common_periods):
        # Target: NEXT month's return (signals at t predict returns at t+1)
        next_period = next_period_map.get(period)
        if next_period is None or next_period not in returns.index:
            continue

        # Get valid stocks for this month (universe at t)
        univ_row = universe.loc[period]
        valid_permnos = univ_row.index[univ_row.values]

        # Get NEXT month's returns for valid stocks
        ret_row = returns.loc[next_period, valid_permnos]

        # Drop stocks with NaN returns
        valid_mask = ret_row.notna()
        valid_permnos = valid_permnos[valid_mask.values]
        if len(valid_permnos) == 0:
            continue

        # Excess returns = next-month raw return - next-month risk-free rate
        if next_period in rfree.index:
            rf = rfree.loc[next_period]
        else:
            rf = rfree.loc[period]
        excess_ret = (ret_row[valid_permnos].values - rf).astype(np.float32)

        # Stock signals — vectorized via reindex (fills missing cols with NaN)
        n_stocks = len(valid_permnos)
        stock_feat = np.empty((n_stocks, len(signal_names)), dtype=np.float32)
        for j, name in enumerate(signal_names):
            sig = signals[name]
            if period in sig.index:
                # reindex aligns columns to valid_permnos, NaN for missing
                stock_feat[:, j] = sig.loc[period].reindex(valid_permnos).values.astype(np.float32)
            else:
                stock_feat[:, j] = np.nan

        # Macro features (same for all stocks in this month)
        macro_row = macro.loc[period].values.astype(np.float32)
        macro_feat = np.tile(macro_row, (n_stocks, 1))

        # Month and PERMNO IDs
        month_id = int(str(period).replace("-", ""))  # e.g. "1975-01" -> 197501
        month_ids_arr = np.full(n_stocks, month_id, dtype=np.int32)
        permno_ids_arr = valid_permnos.astype(np.int32).values

        stock_feat_chunks.append(stock_feat)
        macro_feat_chunks.append(macro_feat)
        target_chunks.append(excess_ret)
        month_id_chunks.append(month_ids_arr)
        permno_id_chunks.append(permno_ids_arr)
        total_obs += n_stocks

        if (i + 1) % 60 == 0:
            logger.info(f"  Processed {i+1}/{len(common_periods)} months, "
                        f"{total_obs:,} total obs so far")

    # Concatenate
    stock_features = np.concatenate(stock_feat_chunks, axis=0)
    macro_features = np.concatenate(macro_feat_chunks, axis=0)
    targets = np.concatenate(target_chunks, axis=0)
    month_ids = np.concatenate(month_id_chunks, axis=0)
    permno_ids = np.concatenate(permno_id_chunks, axis=0)

    logger.info(f"Long panel built: {total_obs:,} observations")
    logger.info(f"  stock_features: {stock_features.shape}, "
                f"NaN={np.isnan(stock_features).sum() / stock_features.size * 100:.1f}%")
    logger.info(f"  macro_features: {macro_features.shape}")
    logger.info(f"  targets: {targets.shape}, "
                f"mean={np.nanmean(targets):.6f}, std={np.nanstd(targets):.6f}")
    logger.info(f"  month range: {month_ids.min()} to {month_ids.max()}")
    logger.info(f"  Memory: stock={stock_features.nbytes/1e9:.2f}GB, "
                f"macro={macro_features.nbytes/1e9:.2f}GB, "
                f"targets={targets.nbytes/1e6:.1f}MB")

    return stock_features, macro_features, targets, month_ids, permno_ids


# ---------------------------------------------------------------------------
# Feature Scaling
# ---------------------------------------------------------------------------

class FeatureScaler:
    """StandardScaler with NaN-aware fit, NaN imputation, and clipping."""

    def __init__(self, clip_std: float = 5.0):
        self.clip_std = clip_std
        self.stock_mean_: Optional[np.ndarray] = None
        self.stock_std_: Optional[np.ndarray] = None
        self.macro_mean_: Optional[np.ndarray] = None
        self.macro_std_: Optional[np.ndarray] = None

    def fit(self, stock_features: np.ndarray, macro_features: np.ndarray):
        """Compute mean/std from training data only."""
        self.stock_mean_ = np.nanmean(stock_features, axis=0).astype(np.float32)
        self.stock_std_ = np.nanstd(stock_features, axis=0).astype(np.float32)
        # Avoid division by zero
        self.stock_std_[self.stock_std_ < 1e-8] = 1.0

        self.macro_mean_ = np.nanmean(macro_features, axis=0).astype(np.float32)
        self.macro_std_ = np.nanstd(macro_features, axis=0).astype(np.float32)
        self.macro_std_[self.macro_std_ < 1e-8] = 1.0
        return self

    def transform(self, stock_features: np.ndarray,
                  macro_features: np.ndarray) -> np.ndarray:
        """Standardize, impute NaN->0, clip, and pre-compute interactions.

        Returns:
            features: (N, n_signals + n_macro + n_signals*n_macro) float32
        """
        stock = (stock_features - self.stock_mean_) / self.stock_std_
        macro = (macro_features - self.macro_mean_) / self.macro_std_

        # Impute NaN with 0 (equivalent to mean after standardization)
        np.nan_to_num(stock, copy=False, nan=0.0)
        np.nan_to_num(macro, copy=False, nan=0.0)

        # Clip extreme values
        np.clip(stock, -self.clip_std, self.clip_std, out=stock)
        np.clip(macro, -self.clip_std, self.clip_std, out=macro)

        # Pre-compute interaction features in chunks to limit peak memory
        n = stock.shape[0]
        n_sig = stock.shape[1]
        n_mac = macro.shape[1]
        features = np.empty((n, n_sig + n_mac + n_sig * n_mac), dtype=np.float32)
        features[:, :n_sig] = stock
        features[:, n_sig:n_sig + n_mac] = macro

        chunk_size = 200_000
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            s = stock[start:end]   # (chunk, n_sig)
            m = macro[start:end]   # (chunk, n_mac)
            features[start:end, n_sig + n_mac:] = (
                s[:, :, None] * m[:, None, :]
            ).reshape(end - start, -1)

        return features

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "FeatureScaler":
        with open(path, "rb") as f:
            return pickle.load(f)


# ---------------------------------------------------------------------------
# GPU-Resident Data Container
# ---------------------------------------------------------------------------

class GPUData:
    """Holds features and targets on GPU for fast batching."""

    def __init__(self, features: np.ndarray, targets: np.ndarray,
                 month_ids: np.ndarray, permno_ids: np.ndarray,
                 device: torch.device):
        self.features = torch.from_numpy(features).to(device)  # (N, n_features)
        self.targets = torch.from_numpy(targets).to(device)    # (N,)
        self.month_ids = month_ids    # keep on CPU for grouping
        self.permno_ids = permno_ids
        self.n = len(targets)

    def __len__(self):
        return self.n


# ---------------------------------------------------------------------------
# Model Definition
# ---------------------------------------------------------------------------

class GKXNet(nn.Module):
    """Neural network following GKX (2020) architecture."""

    def __init__(self, input_dim: int, hidden_dims: Tuple[int, ...],
                 dropout: float = 0.05):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Training Utilities
# ---------------------------------------------------------------------------

def compute_cross_sectional_ic(predictions: np.ndarray, targets: np.ndarray,
                                month_ids: np.ndarray) -> float:
    """Compute mean monthly cross-sectional Spearman IC."""
    unique_months = np.unique(month_ids)
    ics = []
    for m in unique_months:
        mask = month_ids == m
        pred_m = predictions[mask]
        tgt_m = targets[mask]
        if len(pred_m) < 10:
            continue
        # Check for constant predictions
        if np.std(pred_m) < 1e-10 or np.std(tgt_m) < 1e-10:
            continue
        ic, _ = spearmanr(pred_m, tgt_m)
        if not np.isnan(ic):
            ics.append(ic)
    return np.mean(ics) if len(ics) > 0 else 0.0


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model: nn.Module, data: GPUData,
                    optimizer: torch.optim.Optimizer,
                    criterion: nn.Module,
                    scaler: torch.amp.GradScaler,
                    batch_size: int,
                    l1_lambda: float = 0.0) -> float:
    """Train one epoch with mixed precision on GPU-resident data."""
    model.train()
    total_loss = 0.0

    # Shuffle indices on GPU
    perm = torch.randperm(data.n, device=data.features.device)
    n_batches = data.n // batch_size

    for i in range(n_batches):
        idx = perm[i * batch_size:(i + 1) * batch_size]
        features = data.features[idx]
        targets = data.targets[idx]

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda"):
            predictions = model(features)
            loss = criterion(predictions, targets)

            # L1 penalty on all weights (not biases)
            if l1_lambda > 0:
                l1_norm = sum(p.abs().sum() for name, p in model.named_parameters()
                              if "weight" in name)
                loss = loss + l1_lambda * l1_norm

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model: nn.Module, data: GPUData,
             batch_size: int) -> np.ndarray:
    """Evaluate model on GPU-resident data, return predictions as numpy."""
    model.eval()
    all_preds = []

    for i in range(0, data.n, batch_size):
        features = data.features[i:i + batch_size]
        with torch.amp.autocast("cuda"):
            preds = model(features)
        all_preds.append(preds.float().cpu().numpy())

    return np.concatenate(all_preds)


def train_model(
    arch_name: str,
    hidden_dims: Tuple[int, ...],
    train_data: GPUData,
    val_data: GPUData,
    test_year: int,
    seed: int,
    config: Config,
    device: torch.device,
    logger: logging.Logger,
    l1_lambda: float = 0.0,
) -> Dict:
    """Full training loop with early stopping on validation MSE."""

    set_seed(seed)

    model = GKXNet(config.n_features, hidden_dims, config.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.MSELoss()
    amp_scaler = torch.amp.GradScaler("cuda")

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  [{arch_name}|year{test_year}|seed{seed}] "
                f"params={n_params:,}, train={len(train_data):,}, "
                f"val={len(val_data):,}")

    # Training history
    history = []
    best_val_loss = np.inf
    best_val_ic = -np.inf
    best_epoch = 0
    best_state = None
    patience_counter = 0

    for epoch in range(1, config.max_epochs + 1):
        t0 = time.time()

        # Train
        train_loss = train_one_epoch(model, train_data, optimizer,
                                     criterion, amp_scaler, config.batch_size,
                                     l1_lambda=l1_lambda)

        # Validate
        val_preds = evaluate(model, val_data, config.batch_size)
        val_targets = val_data.targets.cpu().numpy()
        val_loss = float(np.mean((val_preds - val_targets) ** 2))
        val_ic = compute_cross_sectional_ic(val_preds, val_targets,
                                            val_data.month_ids)

        elapsed = time.time() - t0
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_ic": val_ic,
            "elapsed_sec": elapsed,
        })

        # Early stopping on val MSE (lower is better), only after min_epochs
        if epoch >= config.min_epochs and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_ic = val_ic
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        elif epoch >= config.min_epochs:
            patience_counter += 1

        if epoch % 25 == 0 or epoch == 1 or patience_counter == 0:
            logger.debug(f"    Epoch {epoch:3d}: train_loss={train_loss:.6f}, "
                         f"val_loss={val_loss:.6f}, val_ic={val_ic:.4f}, "
                         f"best_loss={best_val_loss:.6f}@{best_epoch}, "
                         f"patience={patience_counter}/{config.patience}, "
                         f"{elapsed:.1f}s")

        if patience_counter >= config.patience:
            logger.info(f"    Early stopped at epoch {epoch}, "
                        f"best_epoch={best_epoch}, best_val_loss={best_val_loss:.6f}")
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)

    # Save model checkpoint
    model_dir = os.path.join(config.output_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir,
                              f"{arch_name}_year{test_year}_seed{seed}.pt")
    torch.save({
        "state_dict": best_state or model.state_dict(),
        "arch_name": arch_name,
        "hidden_dims": hidden_dims,
        "config": {
            "n_features": config.n_features,
            "dropout": config.dropout,
            "lr": config.lr,
            "batch_size": config.batch_size,
        },
        "best_epoch": best_epoch,
        "best_val_ic": best_val_ic,
        "seed": seed,
        "test_year": test_year,
    }, model_path)

    # Save epoch metrics
    metrics_dir = os.path.join(config.output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir,
                                f"epoch_metrics_{arch_name}_year{test_year}_seed{seed}.csv")
    pd.DataFrame(history).to_csv(metrics_path, index=False)

    return {
        "model": model,
        "best_epoch": best_epoch,
        "best_val_ic": best_val_ic,
        "best_val_loss": best_val_loss,
        "history": history,
    }


# ---------------------------------------------------------------------------
# Prediction and Evaluation
# ---------------------------------------------------------------------------

def predict_data(model: nn.Module, data: GPUData,
                  batch_size: int) -> np.ndarray:
    """Get predictions for GPU-resident data."""
    return evaluate(model, data, batch_size)


def compute_oos_metrics(predictions: np.ndarray, targets: np.ndarray,
                        month_ids: np.ndarray,
                        logger: logging.Logger) -> Dict:
    """Compute OOS R-squared, monthly IC, and long-short returns."""

    # OOS R-squared (pooled)
    sse = np.sum((targets - predictions) ** 2)
    sst = np.sum(targets ** 2)  # NOT mean-adjusted, per GKX
    oos_r2 = 1.0 - sse / sst

    # Monthly cross-sectional IC
    unique_months = np.unique(month_ids)
    monthly_ics = []
    monthly_ls_returns = []

    for m in unique_months:
        mask = month_ids == m
        pred_m = predictions[mask]
        tgt_m = targets[mask]

        if len(pred_m) < 20:
            continue

        # IC
        if np.std(pred_m) > 1e-10 and np.std(tgt_m) > 1e-10:
            ic, _ = spearmanr(pred_m, tgt_m)
            if not np.isnan(ic):
                monthly_ics.append(ic)

        # Long-short decile returns
        n = len(pred_m)
        decile_size = n // 10
        if decile_size < 2:
            continue
        sorted_idx = np.argsort(pred_m)
        long_ret = np.mean(tgt_m[sorted_idx[-decile_size:]])
        short_ret = np.mean(tgt_m[sorted_idx[:decile_size]])
        monthly_ls_returns.append(long_ret - short_ret)

    mean_ic = np.mean(monthly_ics) if monthly_ics else 0.0
    std_ic = np.std(monthly_ics) if monthly_ics else 0.0
    mean_ls = np.mean(monthly_ls_returns) if monthly_ls_returns else 0.0
    std_ls = np.std(monthly_ls_returns) if monthly_ls_returns else 0.0
    sharpe_ls = (mean_ls / std_ls * np.sqrt(12)) if std_ls > 1e-10 else 0.0

    metrics = {
        "oos_r2_pct": oos_r2 * 100,
        "mean_ic": mean_ic,
        "std_ic": std_ic,
        "mean_ls_ret_pct": mean_ls * 100,
        "std_ls_ret_pct": std_ls * 100,
        "sharpe_ls_annual": sharpe_ls,
        "n_months": len(unique_months),
        "n_obs": len(predictions),
    }

    logger.info(f"    OOS R²={oos_r2*100:.4f}%, IC={mean_ic:.4f}±{std_ic:.4f}, "
                f"L/S={mean_ls*100:.2f}%/mo, Sharpe={sharpe_ls:.2f}")
    return metrics


def save_predictions(predictions: np.ndarray, permno_ids: np.ndarray,
                     month_ids: np.ndarray, path: str):
    """Save predictions as parquet."""
    df = pd.DataFrame({
        "permno": permno_ids,
        "month": month_ids,
        "prediction": predictions,
    })
    df.to_parquet(path, index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    config = Config()
    logger = setup_logging(config.output_dir)

    logger.info("=" * 70)
    logger.info("GKX (2020) Neural Network Replication")
    logger.info("=" * 70)
    logger.info(f"Config: {config}")

    # Check GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("No GPU available, using CPU!")

    # Create output directories
    for subdir in ["logs", "models", "predictions", "metrics", "features"]:
        os.makedirs(os.path.join(config.output_dir, subdir), exist_ok=True)

    # -----------------------------------------------------------------------
    # Load all data
    # -----------------------------------------------------------------------
    end_year = max(config.test_years)
    start_year = config.train_start

    logger.info(f"Loading data for {start_year}-{end_year}...")
    returns = load_returns(config.data_dir, start_year, end_year, logger)
    universe = load_universe(config.data_dir, start_year, end_year, logger)
    signals = load_signals(config.data_dir, config.signal_names,
                           start_year, end_year, logger)
    macro, rfree = load_macro(config.macro_file, start_year, end_year, logger)
    permno_to_sic2, sic2_codes = load_sector_mapping(config.sector_file, logger)

    # Build long panel
    stock_features, macro_features, targets, month_ids, permno_ids = \
        build_long_panel(universe, returns, signals, macro, rfree,
                         config.signal_names, config.macro_names, logger)

    # Save feature names
    industry_names = [f"ind_sic2_{code}" for code in sic2_codes]
    feature_names = (config.signal_names + config.macro_names +
                     [f"{s}_x_{m}" for s in config.signal_names
                      for m in config.macro_names] +
                     industry_names)
    feat_path = os.path.join(config.output_dir, "features", "feature_names.json")
    with open(feat_path, "w") as f:
        json.dump(feature_names, f, indent=2)
    logger.info(f"Total features: {len(feature_names)} "
                f"({config.n_signals} signals + {config.n_macro} macro + "
                f"{config.n_signals * config.n_macro} interactions + "
                f"{len(sic2_codes)} industry dummies)")

    # Free raw dataframes (keep permno_to_sic2, sic2_codes for industry dummies)
    del returns, universe, signals, macro, rfree
    gc.collect()
    logger.info("Raw data freed from memory")

    # -----------------------------------------------------------------------
    # Yearly refit loop
    # -----------------------------------------------------------------------
    all_results = []

    for test_year in config.test_years:
        logger.info("=" * 70)
        logger.info(f"TEST YEAR: {test_year}")
        logger.info("=" * 70)

        val_end_year = test_year - 1
        val_start_year = test_year - config.val_years
        train_end_year = val_start_year - 1

        # Split by month_id
        train_end_month = train_end_year * 100 + 12
        val_start_month = val_start_year * 100 + 1
        val_end_month = val_end_year * 100 + 12
        test_start_month = test_year * 100 + 1
        test_end_month = test_year * 100 + 12

        train_mask = month_ids <= train_end_month
        val_mask = (month_ids >= val_start_month) & (month_ids <= val_end_month)
        test_mask = (month_ids >= test_start_month) & (month_ids <= test_end_month)

        logger.info(f"  Train: {start_year}-01 to {train_end_year}-12 "
                    f"({train_mask.sum():,} obs)")
        logger.info(f"  Val:   {val_start_year}-01 to {val_end_year}-12 "
                    f"({val_mask.sum():,} obs)")
        logger.info(f"  Test:  {test_year}-01 to {test_year}-12 "
                    f"({test_mask.sum():,} obs)")

        # Fit scaler on training data only
        scaler = FeatureScaler(clip_std=config.clip_std)
        scaler.fit(stock_features[train_mask], macro_features[train_mask])

        # Transform all splits (pre-computes interactions)
        logger.info("  Pre-computing features with interactions...")
        train_feat = scaler.transform(
            stock_features[train_mask].copy(), macro_features[train_mask].copy())
        val_feat = scaler.transform(
            stock_features[val_mask].copy(), macro_features[val_mask].copy())
        test_feat = scaler.transform(
            stock_features[test_mask].copy(), macro_features[test_mask].copy())

        # Append industry dummies (not scaled — binary 0/1)
        logger.info("  Building industry dummies...")
        train_ind = build_industry_dummies(permno_ids[train_mask], permno_to_sic2, sic2_codes)
        val_ind = build_industry_dummies(permno_ids[val_mask], permno_to_sic2, sic2_codes)
        test_ind = build_industry_dummies(permno_ids[test_mask], permno_to_sic2, sic2_codes)
        train_feat = np.concatenate([train_feat, train_ind], axis=1)
        val_feat = np.concatenate([val_feat, val_ind], axis=1)
        test_feat = np.concatenate([test_feat, test_ind], axis=1)
        del train_ind, val_ind, test_ind

        logger.info(f"  Features pre-computed: train={train_feat.shape}, "
                    f"val={val_feat.shape}, test={test_feat.shape}, "
                    f"mem={train_feat.nbytes/1e9:.2f}GB")

        # Save scaler
        scaler_path = os.path.join(config.output_dir, "features",
                                   f"scaler_year{test_year}.pkl")
        scaler.save(scaler_path)
        logger.info(f"  Scaler saved to {scaler_path}")

        # Move data to GPU
        logger.info("  Moving data to GPU...")
        train_data = GPUData(train_feat, targets[train_mask],
                             month_ids[train_mask], permno_ids[train_mask], device)
        val_data = GPUData(val_feat, targets[val_mask],
                           month_ids[val_mask], permno_ids[val_mask], device)
        test_data = GPUData(test_feat, targets[test_mask],
                            month_ids[test_mask], permno_ids[test_mask], device)

        # Free CPU arrays (data now on GPU)
        del train_feat, val_feat, test_feat
        gc.collect()
        logger.info(f"  GPU memory: {torch.cuda.memory_allocated()/1e9:.2f}GB allocated")

        # -------------------------------------------------------------------
        # Train each architecture
        # -------------------------------------------------------------------
        for arch_name, hidden_dims in config.architectures.items():
            logger.info(f"\n  Architecture: {arch_name} {hidden_dims}")

            # --- L1 grid search: train 1 seed per lambda, pick best val loss ---
            # Reject lambdas that collapse predictions (val IC ≈ 0)
            best_l1 = 0.0
            if len(config.l1_lambdas) > 1:
                logger.info(f"  L1 grid search over {config.l1_lambdas}...")
                grid_results = []
                for lam in config.l1_lambdas:
                    result = train_model(
                        arch_name, hidden_dims, train_data, val_data,
                        test_year, seed=0, config=config, device=device,
                        logger=logger, l1_lambda=lam)
                    collapsed = abs(result["best_val_ic"]) < 0.001
                    grid_results.append((lam, result["best_val_loss"],
                                         result["best_val_ic"], collapsed))
                    logger.info(f"    L1={lam:.1e}: best_val_loss={result['best_val_loss']:.6f}, "
                                f"val_ic={result['best_val_ic']:.4f}, "
                                f"best_epoch={result['best_epoch']}"
                                f"{' [COLLAPSED]' if collapsed else ''}")
                    del result["model"]
                    torch.cuda.empty_cache()

                # Pick best val loss among non-collapsed models
                valid = [(lam, loss) for lam, loss, ic, col in grid_results if not col]
                if valid:
                    best_l1 = min(valid, key=lambda x: x[1])[0]
                else:
                    logger.warning("  All L1 values collapsed! Using lambda=0.0")
                    best_l1 = 0.0
                logger.info(f"  Best L1 lambda: {best_l1:.1e}")

            # --- Train all seeds with best L1 ---
            seed_test_preds = []
            pred_dir = os.path.join(config.output_dir, "predictions")

            for seed in range(config.n_seeds):
                t0 = time.time()

                # Train
                result = train_model(
                    arch_name, hidden_dims, train_data, val_data,
                    test_year, seed, config, device, logger,
                    l1_lambda=best_l1)

                # Predict on test set
                test_preds = predict_data(
                    result["model"], test_data, config.batch_size)

                # Save per-seed predictions
                pred_path = os.path.join(
                    pred_dir,
                    f"pred_{arch_name}_year{test_year}_seed{seed}.parquet")
                save_predictions(test_preds, permno_ids[test_mask],
                                 month_ids[test_mask], pred_path)

                seed_test_preds.append(test_preds)

                elapsed = time.time() - t0
                logger.info(f"    seed{seed} done: best_epoch={result['best_epoch']}, "
                            f"val_ic={result['best_val_ic']:.4f}, "
                            f"l1={best_l1:.1e}, {elapsed:.1f}s")

                # Free GPU memory
                del result["model"]
                torch.cuda.empty_cache()

            # Ensemble: mean of seed predictions
            ensemble_preds = np.mean(seed_test_preds, axis=0)

            # Save ensemble predictions
            ens_path = os.path.join(
                pred_dir,
                f"pred_ensemble_{arch_name}_year{test_year}.parquet")
            save_predictions(ensemble_preds, permno_ids[test_mask],
                             month_ids[test_mask], ens_path)

            # Compute OOS metrics for ensemble
            logger.info(f"  {arch_name} year{test_year} ENSEMBLE metrics:")
            metrics = compute_oos_metrics(
                ensemble_preds, targets[test_mask],
                month_ids[test_mask], logger)
            metrics["arch"] = arch_name
            metrics["test_year"] = test_year
            metrics["l1_lambda"] = best_l1
            all_results.append(metrics)

            gc.collect()
            torch.cuda.empty_cache()

        # Free GPU data
        del train_data, val_data, test_data
        gc.collect()
        torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("FINAL OOS SUMMARY")
    logger.info("=" * 70)

    summary_df = pd.DataFrame(all_results)
    summary_path = os.path.join(config.output_dir, "metrics", "oos_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    # Print summary table
    pivot_r2 = summary_df.pivot(index="arch", columns="test_year",
                                values="oos_r2_pct")
    pivot_ic = summary_df.pivot(index="arch", columns="test_year",
                                values="mean_ic")

    logger.info("\nOOS R² (%) by architecture and year:")
    logger.info(f"\n{pivot_r2.to_string()}")
    logger.info(f"\nMean OOS R² across years:")
    logger.info(f"\n{pivot_r2.mean(axis=1).to_string()}")

    logger.info("\nMean IC by architecture and year:")
    logger.info(f"\n{pivot_ic.to_string()}")

    # Overall averages
    for arch in config.architectures:
        arch_df = summary_df[summary_df["arch"] == arch]
        logger.info(f"\n{arch} average: R²={arch_df['oos_r2_pct'].mean():.4f}%, "
                    f"IC={arch_df['mean_ic'].mean():.4f}, "
                    f"Sharpe={arch_df['sharpe_ls_annual'].mean():.2f}")

    logger.info(f"\nResults saved to {summary_path}")
    logger.info("Done!")


# ---------------------------------------------------------------------------
# Experiment Grid: 2 (IC/MSE) × 2 (dummies/no) × 2 (10yr/1yr) = 8 runs
# ---------------------------------------------------------------------------

def train_model_exp(
    arch_name: str,
    hidden_dims: Tuple[int, ...],
    train_data: GPUData,
    val_data: GPUData,
    test_year: int,
    seed: int,
    n_features: int,
    stop_metric: str,
    config: Config,
    device: torch.device,
    logger: logging.Logger,
) -> Dict:
    """Training loop with configurable early stopping metric.

    Args:
        n_features: actual input dim (varies with/without dummies)
        stop_metric: "ic" (higher=better) or "mse" (lower=better)
    """
    set_seed(seed)

    model = GKXNet(n_features, hidden_dims, config.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.MSELoss()
    amp_scaler = torch.amp.GradScaler("cuda")

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  [{arch_name}|year{test_year}|seed{seed}] "
                f"params={n_params:,}, train={len(train_data):,}, "
                f"val={len(val_data):,}")

    best_val_loss = np.inf
    best_val_ic = -np.inf
    best_epoch = 0
    best_state = None
    patience_counter = 0

    for epoch in range(1, config.max_epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_data, optimizer,
                                     criterion, amp_scaler, config.batch_size)

        val_preds = evaluate(model, val_data, config.batch_size)
        val_targets = val_data.targets.cpu().numpy()
        val_loss = float(np.mean((val_preds - val_targets) ** 2))
        val_ic = compute_cross_sectional_ic(val_preds, val_targets,
                                            val_data.month_ids)

        elapsed = time.time() - t0

        # Early stopping: IC (higher=better) or MSE (lower=better)
        if stop_metric == "ic":
            improved = val_ic > best_val_ic
        else:
            improved = val_loss < best_val_loss

        if improved:
            best_val_loss = val_loss
            best_val_ic = val_ic
            best_epoch = epoch
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 25 == 0 or epoch == 1 or patience_counter == 0:
            logger.debug(f"    Epoch {epoch:3d}: trn={train_loss:.6f}, "
                         f"val_mse={val_loss:.6f}, val_ic={val_ic:.4f}, "
                         f"pat={patience_counter}/{config.patience}, "
                         f"{elapsed:.1f}s")

        if patience_counter >= config.patience:
            logger.info(f"    Early stopped epoch {epoch}, "
                        f"best={best_epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)

    return {
        "model": model,
        "best_epoch": best_epoch,
        "best_val_ic": best_val_ic,
        "best_val_loss": best_val_loss,
    }


def run_experiments():
    """Run 8 experiments: 2 (IC/MSE) x 2 (dummies/no) x 2 (10yr/1yr val).

    Loads data once, then iterates over all combinations.
    Saves intermediate results after each experiment so partial runs are safe.
    Each experiment trains NN5 x 10 years x 10 seeds = 100 models.
    """
    experiments = [
        {"name": "MSE_noind_1yr",  "stop_metric": "mse", "use_dummies": False, "val_years": 1},
    ]

    config = Config(test_years=list(range(2001, 2010)))
    logger = setup_logging(config.output_dir)

    logger.info("=" * 70)
    logger.info("GKX (2020) NN — 8-Experiment Grid")
    logger.info("  Factors: stop_metric(IC/MSE) x dummies(yes/no) x val(10yr/1yr)")
    logger.info("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("No GPU — this will be very slow!")

    for subdir in ["logs", "models", "predictions", "metrics", "features"]:
        os.makedirs(os.path.join(config.output_dir, subdir), exist_ok=True)

    # ------------------------------------------------------------------
    # Load all data ONCE
    # ------------------------------------------------------------------
    end_year = max(config.test_years)
    start_year = config.train_start

    logger.info(f"Loading data for {start_year}-{end_year}...")
    returns = load_returns(config.data_dir, start_year, end_year, logger)
    universe = load_universe(config.data_dir, start_year, end_year, logger)
    signals = load_signals(config.data_dir, config.signal_names,
                           start_year, end_year, logger)
    macro, rfree = load_macro(config.macro_file, start_year, end_year, logger)
    permno_to_sic2, sic2_codes = load_sector_mapping(config.sector_file, logger)

    stock_features, macro_features, targets, month_ids, permno_ids = \
        build_long_panel(universe, returns, signals, macro, rfree,
                         config.signal_names, config.macro_names, logger)

    del returns, universe, signals, macro, rfree
    gc.collect()
    logger.info("Raw data freed from memory")

    # ------------------------------------------------------------------
    # Run experiments
    # ------------------------------------------------------------------
    all_results = []
    summary_path = os.path.join(config.output_dir, "metrics",
                                "experiments_summary.csv")
    exp_start = time.time()

    for exp_idx, exp in enumerate(experiments):
        exp_name = exp["name"]
        stop_metric = exp["stop_metric"]
        use_dummies = exp["use_dummies"]
        val_years = exp["val_years"]

        logger.info(f"\n{'='*70}")
        logger.info(f"EXPERIMENT {exp_idx+1}/8: {exp_name}")
        logger.info(f"  stop={stop_metric}, dummies={use_dummies}, "
                    f"val_years={val_years}")
        logger.info("=" * 70)

        try:
            for test_year in config.test_years:
                logger.info(f"\n  --- {exp_name} | test_year={test_year} ---")

                # Data split
                val_end_year = test_year - 1
                val_start_year = test_year - val_years
                train_end_year = val_start_year - 1

                train_end_month = train_end_year * 100 + 12
                val_start_month = val_start_year * 100 + 1
                val_end_month = val_end_year * 100 + 12
                test_start_month = test_year * 100 + 1
                test_end_month = test_year * 100 + 12

                train_mask = month_ids <= train_end_month
                val_mask = ((month_ids >= val_start_month)
                            & (month_ids <= val_end_month))
                test_mask = ((month_ids >= test_start_month)
                             & (month_ids <= test_end_month))

                logger.info(f"  Train: ≤{train_end_year}-12 "
                            f"({train_mask.sum():,})")
                logger.info(f"  Val:   {val_start_year}-01…{val_end_year}-12 "
                            f"({val_mask.sum():,})")
                logger.info(f"  Test:  {test_year} ({test_mask.sum():,})")

                # Scale — process one split at a time to cap peak RAM
                # (avoids having all 3 feat arrays + concat spike together)
                scaler = FeatureScaler(clip_std=config.clip_std)
                scaler.fit(stock_features[train_mask],
                           macro_features[train_mask])

                # --- train split → GPU, then free CPU ---
                feat = scaler.transform(
                    stock_features[train_mask].copy(),
                    macro_features[train_mask].copy())
                if use_dummies:
                    ind = build_industry_dummies(
                        permno_ids[train_mask], permno_to_sic2, sic2_codes)
                    feat = np.concatenate([feat, ind], axis=1)
                    del ind
                n_feat = feat.shape[1]
                logger.info(f"  n_features={n_feat}, "
                            f"train_mem={feat.nbytes/1e9:.2f}GB")
                train_data = GPUData(
                    feat, targets[train_mask],
                    month_ids[train_mask], permno_ids[train_mask], device)
                del feat; gc.collect()

                # --- val split → GPU, then free CPU ---
                feat = scaler.transform(
                    stock_features[val_mask].copy(),
                    macro_features[val_mask].copy())
                if use_dummies:
                    ind = build_industry_dummies(
                        permno_ids[val_mask], permno_to_sic2, sic2_codes)
                    feat = np.concatenate([feat, ind], axis=1)
                    del ind
                val_data = GPUData(
                    feat, targets[val_mask],
                    month_ids[val_mask], permno_ids[val_mask], device)
                del feat; gc.collect()

                # --- test split → GPU, then free CPU ---
                feat = scaler.transform(
                    stock_features[test_mask].copy(),
                    macro_features[test_mask].copy())
                if use_dummies:
                    ind = build_industry_dummies(
                        permno_ids[test_mask], permno_to_sic2, sic2_codes)
                    feat = np.concatenate([feat, ind], axis=1)
                    del ind
                test_data = GPUData(
                    feat, targets[test_mask],
                    month_ids[test_mask], permno_ids[test_mask], device)
                del feat; gc.collect()

                logger.info(f"  GPU mem: "
                            f"{torch.cuda.memory_allocated()/1e9:.2f}GB")

                # Train each architecture
                for arch_name, hidden_dims in config.architectures.items():
                    logger.info(f"  {arch_name} {hidden_dims}")

                    seed_preds = []
                    seed_epochs = []

                    for seed in range(config.n_seeds):
                        t0 = time.time()

                        result = train_model_exp(
                            arch_name, hidden_dims,
                            train_data, val_data,
                            test_year, seed,
                            n_feat, stop_metric,
                            config, device, logger)

                        preds = evaluate(result["model"], test_data,
                                         config.batch_size)
                        seed_preds.append(preds)
                        seed_epochs.append(result["best_epoch"])

                        elapsed = time.time() - t0
                        logger.info(
                            f"    seed{seed}: ep={result['best_epoch']}, "
                            f"ic={result['best_val_ic']:.4f}, "
                            f"{elapsed:.1f}s")

                        del result["model"]
                        torch.cuda.empty_cache()

                    # Ensemble
                    ensemble = np.mean(seed_preds, axis=0)

                    # Save ensemble predictions
                    pred_dir = os.path.join(config.output_dir, "predictions")
                    ens_path = os.path.join(
                        pred_dir,
                        f"pred_{exp_name}_{arch_name}_year{test_year}.parquet")
                    save_predictions(ensemble, permno_ids[test_mask],
                                     month_ids[test_mask], ens_path)

                    # OOS metrics
                    logger.info(
                        f"  {exp_name} {arch_name} {test_year} ENSEMBLE:")
                    metrics = compute_oos_metrics(
                        ensemble, targets[test_mask],
                        month_ids[test_mask], logger)
                    metrics["experiment"] = exp_name
                    metrics["stop_metric"] = stop_metric
                    metrics["use_dummies"] = use_dummies
                    metrics["val_years"] = val_years
                    metrics["arch"] = arch_name
                    metrics["test_year"] = test_year
                    metrics["avg_epochs"] = float(np.mean(seed_epochs))
                    all_results.append(metrics)

                    gc.collect()
                    torch.cuda.empty_cache()

                # Free GPU
                del train_data, val_data, test_data
                gc.collect()
                torch.cuda.empty_cache()

        except Exception:
            logger.exception(f"EXPERIMENT {exp_name} FAILED — skipping")
            continue

        # Save after each experiment so partial runs are not lost
        pd.DataFrame(all_results).to_csv(summary_path, index=False)
        elapsed_so_far = (time.time() - exp_start) / 3600
        logger.info(f"  {exp_name} done. {len(all_results)} rows saved. "
                    f"Elapsed: {elapsed_so_far:.1f}h")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    total_hours = (time.time() - exp_start) / 3600
    logger.info(f"\n{'='*70}")
    logger.info(f"ALL 8 EXPERIMENTS COMPLETE ({total_hours:.1f}h)")
    logger.info("=" * 70)

    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(summary_path, index=False)

    # Comparison table
    for exp_name in summary_df["experiment"].unique():
        edf = summary_df[summary_df["experiment"] == exp_name]
        logger.info(
            f"  {exp_name:20s}  "
            f"R²={edf['oos_r2_pct'].mean():+.4f}%  "
            f"IC={edf['mean_ic'].mean():.4f}  "
            f"Sharpe={edf['sharpe_ls_annual'].mean():.2f}  "
            f"epochs={edf['avg_epochs'].mean():.0f}")

    logger.info(f"\nSaved to {summary_path}")
    logger.info("Done!")


if __name__ == "__main__":
    run_experiments()
