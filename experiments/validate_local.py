"""Local CPU validation of ExpTransformer (no cluster, no training).

Proves: (1) default ExpTransformer(n_layers=1, gelu) has the SAME param count as the
frozen CrossSectionalTransformer; (2) it is computationally IDENTICAL to it given the
same weights (key-remap) -> capacity ablations are apples-to-apples; (3) deeper/wider
and GLU variants forward+backward correctly with the expected param scaling.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_transformer import CrossSectionalTransformer
from experiments.exp_transformer import ExpTransformer


def npar(m):
    return sum(p.numel() for p in m.parameters())


torch.manual_seed(0)
prod = CrossSectionalTransformer()
exp1 = ExpTransformer(n_layers=1)
print(f"prod CrossSectionalTransformer params : {npar(prod):,}")
print(f"ExpTransformer(n_layers=1)      params : {npar(exp1):,}")
assert npar(prod) == npar(exp1), "param count mismatch at n_layers=1"

# Faithfulness: remap prod weights (norm1/self_attn/norm2/ffn -> blocks.0.*) and compare.
exp_sd = exp1.state_dict()
remap = {}
for k, v in prod.state_dict().items():
    nk = ("blocks.0." + k) if k.startswith(("norm1", "self_attn", "norm2", "ffn")) else k
    assert nk in exp_sd, f"unmapped key {nk}"
    remap[nk] = v
assert set(remap) == set(exp_sd), f"key mismatch: {set(exp_sd) ^ set(remap)}"
exp1.load_state_dict(remap)

prod.eval(); exp1.eval()
N = 200
stock = torch.randn(1, N, 95)
macro = torch.randn(1, 8)
ind = torch.zeros(1, N, 74)
ind[0, torch.arange(N), torch.randint(0, 74, (N,))] = 1.0
with torch.no_grad():
    a, b = prod(stock, macro, ind), exp1(stock, macro, ind)
maxdiff = (a - b).abs().max().item()
print(f"max|prod - exp(n_layers=1)| = {maxdiff:.2e}")
assert maxdiff < 1e-5, "ExpTransformer(n_layers=1) is NOT faithful to production"
print("FAITHFUL: ExpTransformer(n_layers=1, gelu) == CrossSectionalTransformer\n")

print("variant forward/backward + param scaling:")
variants = [
    dict(n_layers=2), dict(n_layers=3),
    dict(d_model=64, d_ff=128), dict(d_model=64, n_layers=3, d_ff=128),
    dict(d_model=128, n_layers=4, d_ff=256, n_heads=8), dict(ffn_kind="glu"),
]
for cfg in variants:
    m = ExpTransformer(**cfg)
    out = m(stock, macro, ind)
    (out ** 2).mean().backward()
    print(f"  {str(cfg):45s} params={npar(m):>8,}  out={tuple(out.shape)}  bwd-ok")
print("\nOK: configurable model validated locally (no cluster used).")
