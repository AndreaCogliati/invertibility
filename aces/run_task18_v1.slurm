#!/bin/bash
#SBATCH --job-name=muq-task18-heldout-v1
#SBATCH --output=aces/logs/%x-%j.out
#SBATCH --error=aces/logs/%x-%j.err
#SBATCH --time=03:00:00
#SBATCH --partition=pvc          # Intel Data Center GPU Max (PVC) nodes on ACES
#SBATCH --gres=gpu:pvc:1
#SBATCH --ntasks=1               # one Python process at a time (required on ACES)
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --signal=B:TERM@300      # 5 min before the time limit: checkpoint and exit cleanly
#SBATCH --mail-type=ALL
#SBATCH --mail-user=andrea.cogliati@rochester.edu

# TASK-18 v1 (tasks/TASK-18.md, configs/task18_heldout.yaml): 32/8/8 held-out generalization of the
# corrected MuQ -> BigVGAN-mel -> BigVGAN pipeline on one PVC. Full instructions: aces/TASK18_ACES.md.
#
# Submit only after aces/task18_smoke.sh has passed, from the repository root:
#   mkdir -p aces/logs && sbatch aces/run_task18_v1.slurm
#
# Run name is fixed (task18_heldout_v1): resubmitting the same file resumes from
# runs/task18_heldout/task18_heldout_v1/checkpoints/latest.pt (same W&B run id). A finished run
# (report.json present) is never overwritten -- bump the version for a new run.
#
# Expected cost: on the M4 Max (MPS), MuQ features for the 896 unique crops took 12 s and 500
# training steps ~20 s (3.7 GB RSS), i.e. ~15 min for 20,000 steps, plus a few minutes of
# evaluation (decoding ~1,150 crops, CPU contour metrics). PVC speed is not yet measured -- the
# smoke test's timings give a first estimate. 3 h is a wide margin; if it runs out, resubmit (it resumes).

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

RUN_NAME=task18_heldout_v1
export UV_PROJECT="$SLURM_SUBMIT_DIR/envs/xpu"
export HF_HOME="${HF_HOME:-/scratch/group/p.ele260080.000/hf_cache}"
# export HF_HUB_OFFLINE=1          # MuQ weights come from the login-node cache
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

module load WebProxy
module load uv
uv sync --frozen

echo "=== job $SLURM_JOB_ID | $(hostname) | partition $SLURM_JOB_PARTITION | cpus $SLURM_CPUS_PER_TASK | mem ${SLURM_MEM_PER_NODE:-?} MB | gres gpu:pvc:1 ==="
echo "commit $(git rev-parse HEAD) dirty=$(test -n "$(git status --porcelain)" && echo yes || echo no)"
uv run python -c "import torch; assert torch.xpu.is_available(), 'XPU not available'; print('torch', torch.__version__, torch.xpu.get_device_name(0))"

# Preconditions (all prepared on a login node; compute nodes have no network).
test -f third_party/bigvgan_v2_24khz_100band_256x/bigvgan_generator.pt || { echo "ERROR: BigVGAN missing -- run scripts/fetch_bigvgan.py on a login node" >&2; exit 1; }
uv run python scripts/build_task18_manifests.py --verify-audio > "aces/logs/task18-manifests-$SLURM_JOB_ID.json"
grep -q '"data/manifests/task18_crops.csv": "unchanged"' "aces/logs/task18-manifests-$SLURM_JOB_ID.json" \
    || { echo "ERROR: TASK-18 manifests differ from the committed ones" >&2; exit 1; }
if [ -f "artifacts/task18_heldout/$RUN_NAME/report.json" ]; then
    echo "ERROR: $RUN_NAME already finished; bump the version for a new run (outputs are never overwritten)" >&2
    exit 1
fi

# Run in the background so SLURM's TERM (time limit/preemption) reaches Python, which saves
# latest.pt at the end of the current step and exits 3.
uv run python -u scripts/train_task18_heldout.py --device xpu --run-name "$RUN_NAME" --resume auto --wandb-mode offline &
PID=$!
trap 'echo "TERM received: asking training to checkpoint"; kill -TERM $PID' TERM
set +e
wait $PID; STATUS=$?
[ $STATUS -gt 128 ] && { wait $PID; STATUS=$?; }   # wait was interrupted by the trap
set -e

echo "=== resources used (sacct) ==="
sacct -j "$SLURM_JOB_ID" --format=JobID,JobName,Partition,AllocTRES%60,Elapsed,MaxRSS,State || true
if [ $STATUS -eq 3 ]; then
    echo "=== interrupted after checkpointing: resubmit 'sbatch aces/run_task18_v1.slurm' to resume ==="
elif [ $STATUS -eq 0 ]; then
    echo "=== finished: artifacts/task18_heldout/$RUN_NAME/ (report.json, TASK-18_RESULTS.md, listening/) ==="
    echo "Upload W&B from a login node: uv run wandb sync \$(ls -d wandb/offline-run-* | sort)"
fi
exit $STATUS
