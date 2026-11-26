#!/bin/bash
#SBATCH --job-name=sgp
#SBATCH --partition=LocalQ
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.log

# Runs one experiment as SHARDS processes on the same GPU (one process leaves most of an
# H100 idle), then writes its table. Submit from the repository root.
#   sbatch --job-name=table1 scripts/run.slurm experiment_table_1.py
#   SHARDS=4 sbatch --job-name=table2-mvit scripts/run.slurm experiment_table_2.py --model mobilevit
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
EXPERIMENT=${1:?"usage: sbatch scripts/run.slurm EXPERIMENT.py [run options]"}
shift
SHARDS=${SHARDS:-3}
echo "Job $SLURM_JOB_ID on $(hostname), $(date): $EXPERIMENT $* ($SHARDS shards)"

# skip a GPU of the allocation that needs a reset ("not in ready state")
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mapfile -t STATE < <(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader)
IFS=, read -ra DEVS <<< "${CUDA_VISIBLE_DEVICES:-}"
GOOD=""
for i in "${!STATE[@]}"; do
  if ! grep -qi "not in ready state" <<< "${STATE[$i]}"; then GOOD=$i; break; fi
done
[ -n "$GOOD" ] || { echo "allocated GPU is not in ready state, aborting"; exit 1; }
[ "${#STATE[@]}" -gt 1 ] && export CUDA_VISIBLE_DEVICES=${DEVS[$GOOD]:-$GOOD}

pids=()
for ((i = 0; i < SHARDS; i++)); do
  uv run python -W ignore "$EXPERIMENT" run --shard "$i/$SHARDS" "$@" > "logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_s$i.log" 2>&1 &
  pids+=($!)
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
[ $status -eq 0 ] || { echo "a shard failed, see logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_s*.log"; exit 1; }
uv run python -W ignore "$EXPERIMENT" table || true  # fig_2 has no table
echo "Finished $(date)"
