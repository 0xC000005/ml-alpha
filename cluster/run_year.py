#!/usr/bin/env python
"""Leanest reproduction wrapper.

Runs ONE (model, test_year) with N seeds through the UNMODIFIED
train_transformer / train_transformer_msrr `main()`, by rebinding the config
class in that module so `main()`'s internal `TransformerConfig()` /
`MSRRConfig()` picks up our test_years=[year], n_seeds, and per-task output_dir.
The training scripts themselves are never edited.

Usage:
    python run_year.py <mse|msrr> <year> <n_seeds> <output_dir>
"""
import os
import sys
import time

# this file lives in cluster/, but the training modules are in the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if len(sys.argv) != 5:
    sys.exit("usage: run_year.py <mse|msrr> <year> <n_seeds> <output_dir>")

model, year, n_seeds, outdir = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
os.makedirs(outdir, exist_ok=True)

if model == "mse":
    import train_transformer as M
    cfg_name = "TransformerConfig"
elif model == "msrr":
    import train_transformer_msrr as M
    cfg_name = "MSRRConfig"
else:
    sys.exit(f"unknown model '{model}' (expected mse|msrr)")

_Orig = getattr(M, cfg_name)


def _factory(*args, **kwargs):
    c = _Orig(*args, **kwargs)
    c.test_years = [year]
    c.n_seeds = n_seeds
    c.output_dir = outdir
    return c


setattr(M, cfg_name, _factory)  # main() resolves the name from module globals at call time

print(f"[run_year] model={model} year={year} n_seeds={n_seeds} outdir={outdir}", flush=True)
t0 = time.time()
M.main()
print(f"[run_year] DONE {model} {year} ({n_seeds} seeds) in {time.time()-t0:.1f}s", flush=True)
