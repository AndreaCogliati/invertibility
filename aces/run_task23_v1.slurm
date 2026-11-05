#!/bin/bash
#SBATCH --job-name=muq-task23-v1
#SBATCH --output=aces/logs/%x-%j.out
#SBATCH --error=aces/logs/%x-%j.err
#SBATCH --time=02:00:00          # expected ~35 min (see aces/TASK23_ACES.md); pvc limit is 2-00:00:00
#SBATCH --partition=pvc          # Intel Data Center GPU Max (PVC) nodes on ACES
#SBATCH --gres=gpu:pvc:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G                # the TASK-22 first-block cache (~7 GB) is memory-mapped, not copied
#SBATCH --signal=B:TERM@300      # safety net: latest.pt and exit 3 five minutes before the walltime
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=andrea.cogliati@rochester.edu

# TASK-23 (tasks/TASK-23.md, configs/task23.yaml, aces/TASK23_ACES.md): one-crop decoder-overfit
# diagnostic. Pair from TASK-22's own first-block feature cache (data/cache/task22, written by job
# 2261697), then arms A (free tensor), B (TASK-22 recipe, batch 1), C (B's init, regularizers off),
# C_seed1 only if C misses the criterion; plots, BigVGAN audio, report.
#   mkdir -p aces/logs && sbatch --exclude=ac079 aces/run_task23_v1.slurm
# If the job is ever cut short (exit 3), resubmitting this file resumes from each arm's latest.pt.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
RUN_NAME=task23_v1
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
    echo "ERROR: no XPU visible on $(hostname); resubmit excluding it: sbatch --exclude=$(hostname) aces/run_task23_v1.slurm" >&2
    exit 1
fi

if [ -f "artifacts/task23/$RUN_NAME/report.json" ]; then
    echo "ERROR: $RUN_NAME already finished; bump the version for a new run" >&2
    exit 1
fi
ls data/cache/task22/task18_v1_L1_*.pt > /dev/null || { echo "ERROR: TASK-22 first-block cache missing in data/cache/task22" >&2; exit 1; }

uv run python -u scripts/task23_overfit.py --config configs/task23.yaml --device xpu --run-name "$RUN_NAME" &
PID=$!
trap 'echo "TERM received: asking the diagnostic to checkpoint"; kill -TERM $PID' TERM
set +e
wait $PID; STATUS=$?
[ $STATUS -gt 128 ] && { wait $PID; STATUS=$?; }
set -e
echo "=== exit $STATUS (0 = all arms finished and reported; 3 = checkpointed before the walltime, resubmit) ==="
sacct -j "$SLURM_JOB_ID" --format=JobID,JobName,Partition,AllocTRES%60,Elapsed,State || true
exit $STATUS
