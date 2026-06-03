#!/usr/bin/env python
"""Run ONE experiment config through the unmodified train_transformer /
train_transformer_msrr main(), swapping in the configurable ExpTransformer and
overriding hyperparameters via monkeypatch (same non-invasive pattern as
cluster/run_year.py). The production scripts are never edited.

Config is a JSON object, e.g.:
  {"model":"mse","year":2018,"n_seeds":3,"d_model":64,"d_ff":128,"n_layers":2,
   "ffn_kind":"gelu","outdir":"output/exp/cap_d64L2_2018"}

Usage:
  python run_experiment.py '<json>'         # run it
  python run_experiment.py '<json>' --dry    # build the model only, print params (no training)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.exp_transformer import ExpTransformer

if len(sys.argv) < 2:
    sys.exit("usage: run_experiment.py '<json-config>' [--dry]")
cfgj = json.loads(sys.argv[1])
dry = "--dry" in sys.argv[2:]

model_kind = cfgj["model"]                # "mse" | "msrr"
n_layers = int(cfgj.get("n_layers", 1))
ffn_kind = cfgj.get("ffn_kind", "gelu")

if model_kind == "mse":
    import train_transformer as M
    cfg_cls_name = "TransformerConfig"
elif model_kind == "msrr":
    import train_transformer_msrr as M
    cfg_cls_name = "MSRRConfig"
else:
    sys.exit(f"unknown model {model_kind!r}")


def _model_factory(**kw):
    """train_model() calls CrossSectionalTransformer(n_signals=.., d_model=.., ...);
    we forward those and add the experiment-only knobs (n_layers, ffn_kind)."""
    return ExpTransformer(**kw, n_layers=n_layers, ffn_kind=ffn_kind)


# swap the model class the training loop resolves at call time
M.CrossSectionalTransformer = _model_factory

# override the config (width via d_model/d_ff/n_heads flow into the factory call)
_Orig = getattr(M, cfg_cls_name)


def _cfg_factory(*a, **k):
    c = _Orig(*a, **k)
    c.test_years = [int(cfgj["year"])]
    c.n_seeds = int(cfgj.get("n_seeds", 3))
    c.output_dir = cfgj["outdir"]
    for fld in ("d_model", "d_ff", "n_heads", "dropout", "lr", "max_epochs",
                "patience", "weight_decay", "ridge_lambda"):  # ridge = VoC knob (P1)
        if fld in cfgj:
            setattr(c, fld, cfgj[fld])
    return c


setattr(M, cfg_cls_name, _cfg_factory)

print(f"[exp] model={model_kind} year={cfgj['year']} seeds={cfgj.get('n_seeds',3)} "
      f"d_model={cfgj.get('d_model',32)} d_ff={cfgj.get('d_ff',64)} "
      f"n_layers={n_layers} ffn={ffn_kind} -> {cfgj['outdir']}", flush=True)

if dry:
    m = _model_factory(n_signals=95, n_industries=74, n_macro=8,
                       d_model=int(cfgj.get("d_model", 32)), n_heads=int(cfgj.get("n_heads", 4)),
                       d_ff=int(cfgj.get("d_ff", 64)), dropout=float(cfgj.get("dropout", 0.10)))
    nparam = sum(p.numel() for p in m.parameters())
    print(f"[exp] DRY: built {type(m).__name__} with {nparam:,} params (no training).")
    sys.exit(0)

os.makedirs(cfgj["outdir"], exist_ok=True)
M.main()
print(f"[exp] DONE {model_kind} {cfgj['year']}", flush=True)
