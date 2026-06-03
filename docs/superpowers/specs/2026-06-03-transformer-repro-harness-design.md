# Transformer README-reproduction harness — design (2026-06-03)

## Goal
Reproduce the README's Transformer results on Trillium, **without editing the
training scripts**, minimizing GPU occupation.

## Scope
- **MSE Transformer** (`train_transformer.py`), test years **2012–2019**, **5 seeds**.
- **MSRR Transformer** (`train_transformer_msrr.py`), test years **2012–2019**, **10 seeds**
  (report documents extreme seed variance → full ensemble needed; 2016–2019 is the
  README range, 2012–2015 is a new extension).
- Reproduction = **statistical match** (signs, positive-year count, Sharpe/IC within
  tolerance), not bitwise — README was RTX 4080, we run H100 + AMP nondeterminism.

## Mechanism (no script edits)
`cluster/run_year.py <model> <year> <n_seeds> <outdir>` imports the training module
and **rebinds its config class** so the unmodified `main()` constructs a config with
`test_years=[year]`, `n_seeds`, `output_dir=outdir`. `main()` still does the
expanding-window refit, the per-year seed ensembling, and the summary-CSV write.

## Execution (leanest)
- `cluster/repro.sbatch`: 16-task array, index→ (model,year,seeds). 1 H100/quarter-node
  per task, `--cpus-per-task=24`, no `--mem`/`--partition`, `--time=08:00:00`, output to scratch.
- Submit `sbatch --array=0-15%2` → **≤2 H100s held at once** (the occupation dial = `%K`).
- Seeds run sequentially inside a task (load amortized over seeds; no extra data staging).
- `SEEDS_OVERRIDE` env supports a quick 1-seed calibration task.

## Collect / validate
`cluster/collect.py` concatenates the per-task `*_summary.csv`, computes the README
aggregates (MSE 2012–2019 + 2016–2019 subset; MSRR SDF 2016–2019), and prints deltas
vs the README targets.

## Plan
1. rsync `cluster/` to `/scratch/maxzhang/ml-alpha`.
2. Calibrate: one 1-seed MSE task → measure load + per-seed time on H100, confirm CSV
   format + plausible Sharpe, size `--time`.
3. Launch `--array=0-15%2`; manage (squeue/sacct) + monitor GPU (`srun --jobid --overlap nvidia-smi`).
4. Collect + compare to README.

## Caveats
- Trillium 24h walltime cap; `%2` → long wall-clock (hours), accepted for min occupation.
- MSRR per-seed time is high-variance (report: 15–55 min/seed on RTX 4080).
