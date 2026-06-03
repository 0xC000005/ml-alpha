"""Configurable experimental Transformer for the ml-alpha improvement experiments.

The frozen production model (train_transformer.CrossSectionalTransformer) hardcodes
a SINGLE pre-norm block and ignores n_layers. This module re-expresses the same
architecture as a stack of N identical blocks plus optional sophistication knobs,
WITHOUT touching the production scripts. At the default config
(n_layers=1, ffn_kind="gelu") it is computationally identical to the production
model (same param count = 14,369 at d_model=32/d_ff=64; verified in
experiments/validate_local.py), so capacity ablations are apples-to-apples.

Knobs implemented + locally validated here:
  - n_layers     : B-01 depth (capacity / virtue-of-complexity)
  - d_model/d_ff/n_heads : width (already plumbed via TransformerConfig; recorded here)
  - ffn_kind     : "gelu" (default) | "glu" (SwiGLU-style gated FFN) -- B-03 sophistication

Knobs that additionally require a DATA-pipeline change (designed in the plan, added
later): per-signal missingness indicators (B-06) and macro-temporal GRU (B-02).
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ExpModelConfig:
    n_signals: int = 95
    n_industries: int = 74
    n_macro: int = 8
    d_model: int = 32
    n_heads: int = 4
    d_ff: int = 64
    n_layers: int = 1            # B-01: depth (was dead config in production)
    dropout: float = 0.10
    ffn_kind: str = "gelu"       # "gelu" (default) | "glu" (B-03)


class _GLUFFN(nn.Module):
    """SwiGLU-style gated FFN: out(silu(W x) * V x)."""

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.w = nn.Linear(d_model, d_ff)
        self.v = nn.Linear(d_model, d_ff)
        self.out = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.out(self.drop(F.silu(self.w(x)) * self.v(x))))


def _build_ffn(d_model: int, d_ff: int, dropout: float, kind: str) -> nn.Module:
    if kind == "gelu":
        return nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
    if kind == "glu":
        return _GLUFFN(d_model, d_ff, dropout)
    raise ValueError(f"unknown ffn_kind {kind!r}")


class _Block(nn.Module):
    """One pre-norm Transformer block (identical to the production block)."""

    def __init__(self, d_model, n_heads, d_ff, dropout, ffn_kind):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = _build_ffn(d_model, d_ff, dropout, ffn_kind)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_out)
        x = x + self.ffn(self.norm2(x))
        return x


class ExpTransformer(nn.Module):
    """Configurable cross-sectional Transformer (drop-in for CrossSectionalTransformer)."""

    def __init__(self, n_signals: int = 95, n_industries: int = 74, n_macro: int = 8,
                 d_model: int = 32, n_heads: int = 4, d_ff: int = 64, dropout: float = 0.10,
                 n_layers: int = 1, ffn_kind: str = "gelu",
                 n_extra: int = 0, macro_temporal: str = "none"):
        super().__init__()
        # n_extra = width of extra UNSCALED per-stock features appended to industry
        # dummies (e.g. missingness mask, B-06). macro_temporal: "none" (additive,
        # default) | "gru" (GRU over a trailing macro window, B-02).
        self.macro_temporal = macro_temporal
        self.stock_proj = nn.Linear(n_signals + n_industries + n_extra, d_model)
        if macro_temporal == "gru":
            self.macro_gru = nn.GRU(n_macro, d_model, batch_first=True)
        else:
            self.macro_proj = nn.Linear(n_macro, d_model)
        self.blocks = nn.ModuleList([
            _Block(d_model, n_heads, d_ff, dropout, ffn_kind) for _ in range(n_layers)])
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

    def forward(self, stock_features, macro_features, industry_dummies):
        # industry_dummies already includes any appended missingness mask (width
        # n_industries + n_extra). macro_features is (1, n_macro) for additive, or
        # (1, L, n_macro) for the GRU temporal variant.
        x = torch.cat([stock_features, industry_dummies], dim=-1)
        x = self.stock_proj(x)
        if self.macro_temporal == "gru":
            _, h = self.macro_gru(macro_features)   # h: (1, 1, d_model)
            macro_embed = h[-1]                      # (1, d_model)
        else:
            macro_embed = self.macro_proj(macro_features)  # (1, d_model)
        x = x + macro_embed.unsqueeze(1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm_out(x)
        return self.output_head(x).squeeze(-1).squeeze(0)
