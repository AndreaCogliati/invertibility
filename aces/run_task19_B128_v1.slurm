#!/bin/bash
#SBATCH --job-name=muq-task19-B128-v1
#SBATCH --output=aces/logs/%x-%j.out
#SBATCH --error=aces/logs/%x-%j.err
#SBATCH --time=03:00:00
#SBATCH --partition=pvc          # Intel Data Center GPU Max (PVC) nodes on ACES
#SBATCH --gres=gpu:pvc:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --signal=B:TERM@300      # 5 min before the limit: checkpoint and exit cleanly
#SBATCH --mail-type=ALL
#SBATCH --mail-user=andrea.cogliati@rochester.edu

# TASK-19 arm B128 (tasks/TASK-19.md, configs/task19.yaml): train on 128 tracks, recipe B (warmup 1k + cosine to 1e-5).
# Instructions: aces/TASK19_ACES.md. Submit only when instructed, after aces/task19_smoke.sh passed:
#   mkdir -p aces/logs && sbatch aces/run_task19_B128_v1.slurm
# Run name fixed (task19_B128_v1): resubmitting resumes; a finished run is never overwritten.
# Estimated cost: ~0.55 h on 1 PVC (~4.3 billing-hours at billing=8, +/-50%); see aces/TASK19_ACES.md.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

ARM=B128
RUN_NAME=task19_${ARM}_v1
export UV_PROJECT="$SLURM_SUBMIT_DIR/envs/xpu"
export HF_HOME="${HF_HOME:-/scratch/group/p.ele260080.000/hf_cache}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
WANDB_MODE_ARG="${WANDB_MODE_ARG:-online}"   # online via WebProxy; set WANDB_MODE_ARG=offline if the proxy fails

module load WebProxy
module load uv
uv sync --frozen

echo "=== job $SLURM_JOB_ID | $(hostname) | partition $SLURM_JOB_PARTITION | cpus $SLURM_CPUS_PER_TASK | mem ${SLURM_MEM_PER_NODE:-?} MB | gres gpu:pvc:1 | arm $ARM ==="
echo "commit $(git rev-parse HEAD) dirty=$(test -n "$(git status --porcelain)" && echo yes || echo no)"
uv run python -c "import torch; assert torch.xpu.is_available(), 'XPU not available'; print('torch', torch.__version__, torch.xpu.get_device_name(0))"

test -f third_party/bigvgan_v2_24khz_100band_256x/bigvgan_generator.pt || { echo "ERROR: BigVGAN missing" >&2; exit 1; }
:
uv run python scripts/build_task19_manifests.py --verify-audio > "aces/logs/task19-manifests-$SLURM_JOB_ID.json"
grep -q '"data/manifests/task19_crops.csv": "unchanged"' "aces/logs/task19-manifests-$SLURM_JOB_ID.json" \
    || { echo "ERROR: TASK-19 manifests differ from the committed ones" >&2; exit 1; }
if [ -f "artifacts/task19/$RUN_NAME/report.json" ]; then
    echo "ERROR: $RUN_NAME already finished; bump the version for a new run" >&2
    exit 1
fi

uv run python -u scripts/train_task18_heldout.py --config configs/task19.yaml --arm "$ARM" --device xpu \
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
    0) echo "=== finished: artifacts/task19/$RUN_NAME/ ===" ;;
    3) echo "=== interrupted after checkpointing: resubmit to resume ===" ;;
esac
exit $STATUS
