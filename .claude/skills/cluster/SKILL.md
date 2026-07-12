---
name: cluster
description: >-
  Run, monitor, stage, and collect Slurm GPU jobs on the Trillium (Compute Canada
  / Digital Research Alliance) cluster for this repo. Use this WHENEVER a task
  touches the cluster — Trillium, sbatch/srun, "train on the cluster", GPU jobs,
  monitoring a job or GPU utilization, staging code/data to scratch, an experiment
  sweep, or anything under /scratch/maxzhang/ml-alpha — even if "Trillium" or
  "Slurm" isn't said explicitly. It encodes this repo's NON-OBVIOUS, hard-won
  conventions (Trillium rejects --gres, ssh-into-node is blocked, pyarrow is a
  module, etc.); prefer it over guessing cluster commands from general Slurm
  knowledge, which will be wrong here.
---

# Trillium (Compute Canada) cluster workflow

Trillium is SciNet's all-H100 GPU cluster (4× H100 80GB / node). Its Slurm has a
custom submission wrapper and PAM policies that differ from generic Slurm and from
other Alliance clusters — the conventions below were validated empirically in this
repo; trust them over defaults.

## ⚠️ Compute discipline (binding — read first)
- **Never submit `sbatch`/`srun` without an explicit user go-ahead.** Treat launching
  cluster compute as a gated action.
- **Cheap-screen first:** screen new ideas on 2–3 representative years × 3 seeds before
  any full run. Judge results across the **seed × year distribution**, not a point estimate.
- **Throttle arrays to `%2`** (≤ 2 GPUs held at once). Reuse saved artifacts instead of
  retraining when a question allows it.

## 1. Connect (SSH ControlMaster — one Duo, reused)
- Use the alias **`trillium-gpu`** (host key trusted; the plain `trillium` alias is NOT
  in known_hosts and will fail host-key verification).
- Check the master: `ssh -O check trillium-gpu`. If alive, every command multiplexes over
  it with no re-auth. If dead, opening a new connection triggers a **Duo push** the user
  must approve (device 3) — tell them before connecting.
- Run remote commands: `ssh trillium-gpu '<cmd>'`, or pipe a script to avoid quoting hell:
  `ssh trillium-gpu 'bash -l -s' < /tmp/script.sh` (use `bash -l` so `module` works).

## 2. Repo + environment on the cluster
- Repo lives at **`/scratch/maxzhang/ml-alpha`** (scratch auto-purges ~60 days since last
  access; data is in `ml_alpha_data/`, gitignored).
- Activate the env with **`source activate_cluster.sh`** — it does
  `module load python/3.12 gcc arrow/24.0.0` then activates `venv/`.
  **pyarrow comes from the `arrow` MODULE, not pip** (pip's pyarrow is a dummy wheel that
  aborts the install); the module must be loaded *before* the venv.

## 3. Submit GPU jobs (Trillium-specific Slurm syntax)
- **Use `--gpus-per-node=1`, NOT `--gres=gpu:...`** — Trillium's wrapper rejects `--gres`
  ("option not recognized") and `--test-only`.
- **1 GPU = a quarter node** (24 cores, ~188 GiB). `--mem` is **ignored** (you auto-get
  188 GiB) — don't pass it. You may request 1 GPU or a multiple of 4 (not 2/3).
- **Don't specify `--partition` for production jobs** — the scheduler auto-routes to
  `compute`. Use `--partition=debug` only for quick tests (2h cap, fast scheduling).
- **Output must be written to scratch** (`$HOME`/`$PROJECT` are read-only on compute nodes).
- Account: **`def-cglee`** (or `rrg-cglee`). Canonical interactive run (streams output):
  ```bash
  srun --account=def-cglee --nodes=1 --gpus-per-node=1 --cpus-per-task=24 --time=0:15:00 \
       bash -lc 'cd /scratch/maxzhang/ml-alpha && source activate_cluster.sh && python <script>'
  ```

## 4. Job arrays (the experiment harness)
- Generic runner: `experiments/sweep.sbatch` (1 H100/quarter-node per task). Submit with a
  config list + throttle:
  ```bash
  # mkdir the --output dir BEFORE submit; sbatch needs it to exist
  ssh trillium-gpu 'mkdir -p /scratch/maxzhang/ml-alpha/output/exp/logs'
  sbatch --array=0-14%2 --export=ALL,CONFIG_FILE=experiments/configs/capacity.jsonl experiments/sweep.sbatch
  # exp_main.py-based phases also pass ,RUNNER=experiments/exp_main.py
  ```
- `%2` = at most 2 tasks (GPUs) run at once; the rest pend with reason `JobArrayTaskLimit`
  (that's correct, not an error). Raise live: `scontrol update JobId=<id> ArrayTaskThrottle=4`.

## 5. Monitor
- Queue: **`sq`** (Alliance shortcut for your `squeue`). Don't poll faster than ~60s.
- **GPU utilization — `srun --jobid=$J --overlap nvidia-smi`** (or `--pty nvtop`).
  - **Do NOT add `--gres=none`** — on Trillium that hides the GPU (`nvidia-smi` → "No devices").
  - **`ssh`-ing into the compute node is BLOCKED** here (`pam_slurm_adopt`) — use `srun --overlap`.
  - **Array tasks:** `--jobid=562608_0` fails (it parses the pending array meta-job). Get the
    task's real numeric JobId first:
    ```bash
    J=$(scontrol show job 562608_0 | grep -oP 'JobId=\K[0-9]+' | head -1); srun --jobid=$J --overlap nvidia-smi
    ```
- Live training log: `tail -f output/exp/logs/<jobname>_*.out` (per-epoch is DEBUG-suppressed;
  expect quiet gaps — `nvtop` confirms it's working).
- Finished: `seff <jid>` (CPU/RAM efficiency); `sacct -X -u $USER --name=<name> --format=JobID%16,State,Elapsed,ExitCode,AllocTRES%30`.

## 6. Stage code/data
- `rsync -av <files> trillium-gpu:/scratch/maxzhang/ml-alpha/` (uses the SSH master).
  Production scripts (`train_*.py`) are frozen — experiment code goes in `experiments/`.

## 7. Collect + compare
- `python experiments/collect_screen.py <screen_dir> <summary_csv>` → config × year table with
  Δ-vs-`base` (e.g. `output/exp/cap transformer_summary.csv`, or `output/exp/miss exp_summary.csv`).
- A pass = a config beats `base` in ≥ 2/3 screen years AND a higher row-mean, judged with
  dispersion in mind — only then promote to a full 8-year run.

## Long-running jobs
A sweep can run hours. Drive it with a background watcher that polls `sacct` until all tasks
finish, then runs `collect_screen.py` — rather than holding a foreground command. Report at
meaningful checkpoints (first task done = health check; all done = the comparison).
