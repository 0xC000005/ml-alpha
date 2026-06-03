#!/bin/bash
# Transformer README reproduction — leanest profile.
# 16-task array = (2 models x 8 years). Index 0-7 = MSE (5 seeds), 8-15 = MSRR (10 seeds).
# 1 H100 (quarter node) per task; no --mem / no --partition (Trillium auto-routes).
# Submit with a concurrency cap, e.g.:  sbatch --array=0-15%2 cluster/repro.sbatch
#SBATCH --job-name=tfrepro
#SBATCH --account=def-cglee
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/maxzhang/ml-alpha/output/repro/logs/%x_%A_%a.out

set -uo pipefail
REPO=/scratch/maxzhang/ml-alpha
cd "$REPO"

YEARS=(2012 2013 2014 2015 2016 2017 2018 2019)
i=$SLURM_ARRAY_TASK_ID
if [ "$i" -lt 8 ]; then
  MODEL=mse;  YEAR=${YEARS[$i]};        SEEDS=5
else
  MODEL=msrr; YEAR=${YEARS[$((i-8))]};  SEEDS=10
fi
# optional override for calibration: --export=ALL,SEEDS_OVERRIDE=1
SEEDS=${SEEDS_OVERRIDE:-$SEEDS}

OUT="$REPO/output/repro/${MODEL}_${YEAR}"
mkdir -p "$OUT"
echo "[task $i] model=$MODEL year=$YEAR seeds=$SEEDS node=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?} $(date)"

source "$REPO/activate_cluster.sh"
python "$REPO/cluster/run_year.py" "$MODEL" "$YEAR" "$SEEDS" "$OUT"; rc=$?
echo "[task $i] EXIT=$rc $(date)"
exit $rc
