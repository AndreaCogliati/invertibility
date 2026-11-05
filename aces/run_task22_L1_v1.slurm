#!/bin/bash
#SBATCH --job-name=muq-task22-L1-v1
#SBATCH --output=aces/logs/%x-%j.out
#SBATCH --error=aces/logs/%x-%j.err
#SBATCH --time=04:00:00          # expected ~1 h (see aces/TASK22_ACES.md); pvc limit is 2-00:00:00
#SBATCH --partition=pvc          # Intel Data Center GPU Max (PVC) nodes on ACES
#SBATCH --gres=gpu:pvc:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --signal=B:TERM@600      # safety net: checkpoint latest.pt and exit 3 ten minutes before the walltime
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=andrea.cogliati@rochester.edu

# TASK-22 (tasks/TASK-22.md, configs/task22.yaml arm L1, aces/TASK22_ACES.md): translator trained FROM
# SCRATCH on MuQ conformer block 1 output (hidden_states[1]) for 200,000 steps, no early stopping; then
# best.pt, final.pt (200k) and the milestones at 10k/20k/50k/60k/100k are evaluated in the same job.
#   mkdir -p aces/logs && sbatch aces/run_task22_L1_v1.slurm
# If the job is ever cut short (exit 3), resubmitting this file resumes from latest.pt.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
RUN_NAME=task22_L1_v1
export UV_PROJECT="$SLURM_SUBMIT_DIR/envs/xpu"
export HF_HOME="${HF_HOME:-/scratch/group/p.ele260080.000/hf_cache}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
WANDB_MODE_ARG="${WANDB_MODE_ARG:-online}"   # online via WebProxy; failures fall back without stopping training

module load WebProxy
module load uv
uv sync --frozen
echo "=== job $SLURM_JOB_ID | $(hostname) | $SLURM_JOB_PARTITION | time limit $(squeue -h -j "$SLURM_JOB_ID" -o %l) | $RUN_NAME ==="
echo "commit $(git rev-parse HEAD) dirty=$(test -n "$(git status --porcelain)" && echo yes || echo no)"
# Fail fast (and say why) if the allocated PVC GPU is not visible to PyTorch on this node.
echo "node $(hostname) | SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset} GPU_DEVICE_ORDINAL=${GPU_DEVICE_ORDINAL:-unset} ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-unset} ONEAPI_DEVICE_SELECTOR=${ONEAPI_DEVICE_SELECTOR:-unset}"
if ! uv run python -c "import sys, torch; n = torch.xpu.device_count(); print('torch', torch.__version__, 'xpu_device_count', n); sys.exit(0 if n > 0 else 1)"; then
    ls -l /dev/dri 2>&1 || true
    xpu-smi discovery 2>&1 | head -20 || true
    echo "ERROR: no XPU visible on $(hostname); resubmit excluding it: sbatch --exclude=$(hostname) aces/run_task22_L1_v1.slurm" >&2
    exit 1
fi

if [ -f "artifacts/task22/$RUN_NAME/report.json" ]; then
    echo "ERROR: $RUN_NAME already finished; bump the version for a new run" >&2
    exit 1
fi
uv run python scripts/build_task20_manifests.py > "aces/logs/task22-manifests-$SLURM_JOB_ID.json"
test "$(grep -c '"unchanged"' "aces/logs/task22-manifests-$SLURM_JOB_ID.json")" -eq 3 \
    || { echo "ERROR: TASK-20 manifests differ from the committed ones" >&2; exit 1; }
uv run python scripts/task22_verify_layer.py --device xpu --out "artifacts/task22/layer_verification_$SLURM_JOB_ID" | tail -2

uv run python -u scripts/train_task18_heldout.py --config configs/task22.yaml --arm L1 --device xpu \
    --run-name "$RUN_NAME" --resume auto --wandb-mode "$WANDB_MODE_ARG" &
PID=$!
trap 'echo "TERM received: asking training to checkpoint"; kill -TERM $PID' TERM
set +e
wait $PID; STATUS=$?
[ $STATUS -gt 128 ] && { wait $PID; STATUS=$?; }
set -e
echo "=== exit $STATUS (0 = trained 200,000 steps and evaluated; 3 = checkpointed before the walltime, resubmit) ==="
sacct -j "$SLURM_JOB_ID" --format=JobID,JobName,Partition,AllocTRES%60,Elapsed,State || true
exit $STATUS
