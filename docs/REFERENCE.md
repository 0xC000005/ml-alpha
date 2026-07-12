# ml-alpha — Reference

Static facts: compute discipline, environment/data card, conventions, glossary.
Edited only when a fact changes, never as a periodic snapshot. History lives in
`docs/log/`, `docs/experiments/`, `docs/decisions/`.

## ⚠️ Compute discipline (read before launching anything)

The Trillium harness makes runs cheap, which makes waste easy. Rules:

1. **Screen cheap before committing.** New idea → first run a *minimal* probe: 2–3 representative years (e.g. 2014 = a hard year, 2018/2019 = strong years), **3 seeds**, one or two configs. Only promote to a full 8-year × 10-seed run if the cheap screen shows real, consistent signal.
2. **Judge across the seed × year distribution, not a point estimate.** Per-year Sharpe swings −1 → +5 and per-seed dispersion is large; a single good number is almost always noise. Report min/median/max across seeds and all years.
3. **Reuse artifacts; avoid retraining.** Many questions can be answered by re-analyzing *saved* per-seed models/predictions (see EXP-002) — no GPU training at all.
4. **1 GPU = quarter node, throttle the array.** `--gpus-per-node=1`, no `--mem`/`--partition`, submit arrays with `%2` (≤2 H100s held). Never request whole nodes.
5. **Don't run what you can't measure.** If an effect is smaller than the noise floor over the available years, the experiment can't conclude — redesign or skip.

---


## Environment, Data & Reproducibility

*(PLOS Rules 6–7: record how every result was produced, under version control.)*

- **Hardware.** Cluster: **Trillium** (SciNet / DRAC), NVIDIA **H100 80GB**; 1 GPU = quarter-node (24 cores, ~188 GiB RAM); arrays throttled `%2`. Local dev: single RTX 4080 (~16 GB).
- **Software.** Python **3.12**; **PyTorch 2.12 (cu130)**, AMP (`GradScaler("cuda")`); numpy / pandas / scipy / openpyxl. **pyarrow comes from the `arrow/24.0.0` module, not pip** (cluster). Local env is uv-managed (`pyproject.toml` + `uv.lock` + `.python-version`); cluster env via `source activate_cluster.sh`.
- **Data card** (not redistributable; gitignored under `ml_alpha_data/`):
  - `gkx_full/` — per-signal `signal_*.parquet`, `returns.parquet`, `universe.parquet` (Gu–Kelly–Xiu 2020 lineage).
  - `welch_goyal_2024.xlsx` — 8 macro predictors derived in `load_macro`: dp, ep, bm, ntis, tbl, tms, dfy, svar.
  - `sector_mapping.csv` — PERMNO → SIC 2-digit (74 industries).
  - Panel (`build_long_panel`, train_nn.py): **95 signals + 8 macro + 74 industry dummies**; **target = next-month excess return** (signals at *t* predict *t+1*); ~26% signal missingness → 0 post-standardization; rows with NaN next-month return dropped. Period used here: **1975–2019**.
- **Reproducibility.** Every run now auto-writes **`manifest.json`** into its output dir (`experiments/manifest.py`, wired into `exp_main.py` / `exp_main_msrr.py` / `run_experiment.py`): **git SHA + dirty flag**, resolved **config**, seeds, torch/CUDA/GPU, hostname, Slurm job+array id, and **SHA-256 of every data input** (the 95 signal parquets collapse to one order-independent digest). ⚠️ **A run whose manifest says `dirty: true` is not reproducible** — the code that ran is not the code that is committed. ⚠️ **Runs predating 2026-07-11 have NO manifest** (git commit was recorded for only 1 of 5 completed EXPs), so the existing `output/exp/` artifacts cannot be fully reconstructed; provenance is only guaranteed from EXP-011 onward. Models are **n-seed ensembles** (`set_seed(0..n-1)`). Determinism caveat: AMP + CUDA are **not bit-exact across GPUs** → judge **statistically across the seed × year distribution** (L-01), never a single run.
- **Data provenance.** SHA-256 of the current snapshot is pinned in each `manifest.json` (cached at `.data_hash.json`, gitignored). Verified 2026-07-11: `returns.parquet` `00ad1cea…`, `universe.parquet` `31d268e2…`, signals `031f72df…` (95 files). ⚠️ **`returns.parquet` contains no delisting returns** and at least one impossible value (min return **−1.437**) — see EXP-010 D-4.
- **Reproduce an EXP:** regenerate configs (`experiments/gen_configs.py`) → `rsync -a experiments/ trillium-gpu:/scratch/maxzhang/ml-alpha/experiments/` → `sbatch --array=0-N%2 --export=ALL,CONFIG_FILE=…,RUNNER=… experiments/sweep.sbatch` → `python experiments/collect_screen.py <dir> <summary.csv>`.

---


## Conventions

- **Code version:** every experiment records its git commit / branch. Config injected via `cluster/run_year.py` style wrappers; do not edit the training scripts to parameterize a run unless the change is the experiment.
- **Outputs:** `output/<exp>/...` on scratch (cluster) and pulled to local `output/` (gitignored). Summary CSVs + a one-line verdict are the durable record; large model/prediction files stay on scratch.
- **Run via the harness:** `sbatch --array=...%2 cluster/repro.sbatch` (parameterized), or `srun ... bash -lc 'source activate_cluster.sh && python ...'` for one-offs. See `cluster/` and the design doc.
- **Monitoring:** `sq`; `srun --jobid=$J --overlap nvidia-smi` (NOT `--gres=none` on Trillium); `seff <jid>` post-run.
- **Naming:** experiments `EXP-NNN`; backlog ideas `B-NN`; learnings `L-NN`.

---


## Glossary

- **MSRR** — Maximum Sharpe Ratio Regression (Kelly–Xiu): loss `E[(1 − wᵀR)²]`; the model outputs portfolio **weights** `w` directly (the SDF), so training maximizes the portfolio Sharpe rather than predicting returns.
- **SDF / SDF Sharpe** — stochastic discount factor `M = 1 − wᵀR`; the SDF portfolio's monthly return is `wᵀR` and its annualized Sharpe is `mean/std·√12` — the MSRR evaluation metric.
- **L1 (honest) combiner** — ensemble rule that L1-normalizes each seed's per-month weights to ‖w‖₁=1 *before* averaging (equal-vote), then L1-normalizes the final ensemble. Removes the scale-invariance bias of raw `np.mean` (L-02); the comparison denominator for all MSRR A/Bs.
- **IC** — Information Coefficient: cross-sectional correlation of predictions with next-month returns. The stable signal metric for the **MSE** model; only a secondary check for MSRR (L-07).
- **OOS** — out-of-sample (the held-out test year, unseen in train/val).
- **Noise floor** — irreducible sampling SE of an annualized Sharpe ≈ `√((1 + ½·SR²)/T)`: ~±0.5 over 48 months, ~±0.4 over 96. Effects smaller than this can't be resolved on our windows (L-08).
- **Expanding vs rolling window** — expanding = train from 1975 up to *t*−1 (our default); rolling = a fixed trailing window (the paper uses 60 months). The **refit cadence** (yearly vs monthly) is an orthogonal knob.
- **Paper model ladder (w33351)** — BSV (linear, own-asset) → DKKM (shallow NN, own-asset) → MLP (deep NN, own-asset) → **Transformer** (deep + cross-asset attention). Our MSRR transformer is a toy instance of the last rung.
- **Screen vs confirmation** — *screen* = cheap probe (few seeds × few years) to form a hypothesis; *confirmation* = full 8yr × 10-seed run that can overturn it (EXP-007 → EXP-009; L-08).
