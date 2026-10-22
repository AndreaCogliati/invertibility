#!/bin/bash
#SBATCH --job-name=muq-task20-N512-v1
#SBATCH --output=aces/logs/%x-%j.out
#SBATCH --error=aces/logs/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=pvc          # Intel Data Center GPU Max (PVC) nodes on ACES
#SBATCH --gres=gpu:pvc:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G                # ~13.4k crops: features + waveforms + targets ~6 GB in RAM, plus the cache write
#SBATCH --signal=B:TERM@300      # 5 min before the limit: checkpoint and exit cleanly
#SBATCH --mail-type=ALL
#SBATCH --mail-user=andrea.cogliati@rochester.edu

# TASK-20 (tasks/TASK-20.md, configs/task20.yaml, aces/TASK20_ACES.md): one N=512 data-scaling run,
# TASK-19 recipe A, early stopping on primary val (every 250 steps, patience 8, max 20,000 steps).
# Submit only when instructed, after aces/task20_smoke.sh passed:
#   mkdir -p aces/logs && sbatch aces/run_task20_N512_v1.slurm
# Run name fixed (task20_N512_v1): resubmitting resumes from latest.pt, including the early-stopping
# state; a finished run is never overwritten.
# Estimated cost: ~25-40 min on 1 PVC (~4-6 billing-hours if billing stays 8 at 64G; check sacct).

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

ARM=N512
RUN_NAME=task20_${ARM}_v1
export UV_PROJECT="$SLURM_SUBMIT_DIR/envs/xpu"
export HF_HOME="${HF_HOME:-/scratch/group/p.ele260080.000/hf_cache}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
WANDB_MODE_ARG="${WANDB_MODE_ARG:-online}"   # online via WebProxy; WANDB_MODE_ARG=offline if the proxy fails

module load WebProxy
module load uv
uv sync --frozen

echo "=== job $SLURM_JOB_ID | $(hostname) | partition $SLURM_JOB_PARTITION | cpus $SLURM_CPUS_PER_TASK | mem ${SLURM_MEM_PER_NODE:-?} MB | gres gpu:pvc:1 | $RUN_NAME ==="
echo "commit $(git rev-parse HEAD) dirty=$(test -n "$(git status --porcelain)" && echo yes || echo no)"
uv run python -c "import torch; assert torch.xpu.is_available(), 'XPU not available'; print('torch', torch.__version__, torch.xpu.get_device_name(0))"

test -f third_party/bigvgan_v2_24khz_100band_256x/bigvgan_generator.pt || { echo "ERROR: BigVGAN missing" >&2; exit 1; }
uv run python scripts/build_task20_manifests.py --verify-audio > "aces/logs/task20-manifests-$SLURM_JOB_ID.json"
test "$(grep -c '"unchanged"' "aces/logs/task20-manifests-$SLURM_JOB_ID.json")" -eq 3 \
    || { echo "ERROR: TASK-20 manifests differ from the committed ones" >&2; exit 1; }
if [ -f "artifacts/task20/$RUN_NAME/report.json" ]; then
    echo "ERROR: $RUN_NAME already finished; bump the version for a new run" >&2
    exit 1
fi

uv run python -u scripts/train_task18_heldout.py --config configs/task20.yaml --arm "$ARM" --device xpu \
    --run-name "$RUN_NAME" --resume auto --wandb-mode "$WANDB_MODE_ARG" &
PID=$!
trap 'echo "TERM received: asking the run to checkpoint"; kill -TERM $PID' TERM
set +e
wait $PID; STATUS=$?
[ $STATUS -gt 128 ] && { wait $PID; STATUS=$?; }
set -e

echo "=== resources used (sacct) ==="
sacct -j "$SLURM_JOB_ID" --format=JobID,JobName,Partition,AllocTRES%60,Elapsed,MaxRSS,State || true
case $STATUS in
    0) echo "=== finished: artifacts/task20/$RUN_NAME/ ===" ;;
    3) echo "=== interrupted after checkpointing: resubmit to resume ===" ;;
esac
exit $STATUS
