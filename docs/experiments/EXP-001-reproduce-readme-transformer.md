---
id: EXP-001
status: done
---

# EXP-001 — Reproduce README Transformer results on Trillium
- **Date:** 2026-06-02/03 · **Status:** ✅ done · **Commit:** `cluster-repro-harness` (29f5740)
- **Hypothesis:** the committed code reproduces the README's MSE (2012–19) and MSRR (2016–19) numbers (statistical match, not bit-exact; RTX 4080 → H100).
- **Setup:** `cluster/repro.sbatch` array `--array=0-15%2` (MSE 2012–19 ×5 seeds, MSRR 2012–19 ×10 seeds), 1 H100/quarter-node each, config injected via `cluster/run_year.py` (no edits to training scripts). Outputs: `output/repro/*/metrics/*summary.csv`, combined `output/repro/repro_combined.csv`.
- **Result:** 16/16 COMPLETED, ~9h50m wall, ≤2 GPUs, ~19 GPU-h. MSE avg L/S Sharpe **+2.84** (IC +0.021 exact, 8/8 positive); MSRR avg SDF Sharpe **+3.13** on 2016–19 (all positive). Both ran higher than published (MSE 2.16 / MSRR 2.05).
- **Conclusion:** signal reproduces faithfully (IC, direction); Sharpe runs hot — flagged for investigation (→ EXP-002). MSRR fails on early years (2012/2014 negative), high per-year variance.
- **Next:** investigate the hot Sharpe; brainstorm improvements.

