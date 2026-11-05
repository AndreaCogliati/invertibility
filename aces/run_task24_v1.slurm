#!/bin/bash
#SBATCH --job-name=muq-task24-v1
#SBATCH --output=aces/logs/%x-%j.out
#SBATCH --error=aces/logs/%x-%j.err
#SBATCH --time=06:00:00          # expected ~1.5 h (aces/TASK24_ACES.md; re-measured by smoke step 4); pvc limit 2-00:00:00
#SBATCH --partition=pvc          # Intel Data Center GPU Max (PVC) nodes on ACES
#SBATCH --gres=gpu:pvc:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G                # the TASK-22 first-block cache (~7 GB) is memory-mapped during prepare
#SBATCH --signal=B:TERM@600      # safety net: latest.pt of the running arm, exit 3, ten minutes before the walltime
#SBATCH --dependency=singleton   # never two jobs of this name at once (one writer per run); arms also hold LOCK files
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=andrea.cogliati@rochester.edu

# TASK-24 (tasks/TASK-24.md, configs/task24.yaml, aces/TASK24_ACES.md): fixed-crop decoder fitting ladder.
# prepare (manifest verify, 272 crops from data/cache/task22, anchor == TASK-23 pair bit for bit, cross-check,
# init = TASK-23 B step 0) -> arms N = 1, 4, 16, 64, 256 (200,000 batch-1 updates each) -> report.
#   mkdir -p aces/logs && sbatch --exclude=ac079 aces/run_task24_v1.slurm
# Exit 3 (walltime/TERM): resubmit this file; finished arms are skipped and the running arm resumes from latest.pt.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
RUN_NAME=task24_v1
export UV_PROJECT="$SLURM_SUBMIT_DIR/envs/xpu"
export HF_HOME="${HF_HOME:-/scratch/group/p.ele260080.000/hf_cache}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

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
    echo "ERROR: no XPU visible on $(hostname); resubmit excluding it: sbatch --exclude=$(hostname) aces/run_task24_v1.slurm" >&2
    exit 1
fi

if [ -f "artifacts/task24/$RUN_NAME/report.json" ]; then
    echo "ERROR: $RUN_NAME already finished; bump the version for a new run" >&2
    exit 1
fi
ls data/cache/task22/task18_v1_L1_*.pt runs/task23/task23_v1/pair.pt runs/task23/task23_v1/B/checkpoints/milestone_000000.pt > /dev/null \
    || { echo "ERROR: TASK-22 cache or TASK-23 anchor/init files missing" >&2; exit 1; }

uv run python -u scripts/task24_ladder.py all --config configs/task24.yaml --device xpu --run-name "$RUN_NAME" &
PID=$!
trap 'echo "TERM received: asking the ladder to checkpoint"; kill -TERM $PID' TERM
set +e
wait $PID; STATUS=$?
[ $STATUS -gt 128 ] && { wait $PID; STATUS=$?; }
set -e
echo "=== exit $STATUS (0 = all arms finished and reported; 3 = checkpointed before the walltime, resubmit) ==="
sacct -j "$SLURM_JOB_ID" --format=JobID,JobName,Partition,AllocTRES%60,Elapsed,State || true
exit $STATUS
