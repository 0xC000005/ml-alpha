#!/bin/bash
# Generic cheap-screen array runner. Submit with a config file + runner + throttle, e.g.:
#   sbatch --array=0-14%2 --export=ALL,CONFIG_FILE=experiments/configs/capacity.jsonl \
#          experiments/sweep.sbatch
#   sbatch --array=0-5%2  --export=ALL,CONFIG_FILE=experiments/configs/missingness.jsonl,RUNNER=experiments/exp_main.py \
#          experiments/sweep.sbatch
# One H100 / quarter-node per task; no --mem / no --partition (Trillium auto-routes).
#SBATCH --job-name=mlexp
#SBATCH --account=def-cglee
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/maxzhang/ml-alpha/output/exp/logs/%x_%A_%a.out

set -uo pipefail
REPO=/scratch/maxzhang/ml-alpha
cd "$REPO"
: "${CONFIG_FILE:?set CONFIG_FILE=experiments/configs/<file>.jsonl}"
RUNNER="${RUNNER:-experiments/run_experiment.py}"

i=$SLURM_ARRAY_TASK_ID
CFG=$(sed -n "$((i + 1))p" "$CONFIG_FILE")
[ -z "$CFG" ] && { echo "no config at line $((i + 1)) of $CONFIG_FILE"; exit 1; }
mkdir -p output/exp/logs
echo "[exp $i] runner=$RUNNER node=$(hostname) cfg=$CFG"

source "$REPO/activate_cluster.sh"
python "$RUNNER" "$CFG"; rc=$?
echo "[exp $i] EXIT=$rc"
exit $rc
