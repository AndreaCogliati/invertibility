#!/bin/bash
#SBATCH --job-name=muq-task25-v1
#SBATCH --output=aces/logs/%x-%j.out
#SBATCH --error=aces/logs/%x-%j.err
#SBATCH --time=06:00:00          # expected ~1.5-2 h sequential on PVC (aces/TASK25_ACES.md); pvc limit 2-00:00:00
#SBATCH --partition=pvc
#SBATCH --gres=gpu:pvc:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --signal=B:TERM@600      # latest.pt of the running arm, exit 3, ten minutes before the walltime
#SBATCH --dependency=singleton   # one writer per run; each arm also holds a LOCK file
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=andrea.cogliati@rochester.edu

# TASK-25 (tasks/TASK-25.md, configs/task25.yaml, aces/TASK25_ACES.md): arms A-F on TASK-24's N=256 pool,
# 819,200 presentations each, then the report. W&B online (offline fallback). Only needed if the run is
# executed on ACES instead of locally.
#   mkdir -p aces/logs && sbatch --exclude=ac079 aces/run_task25_v1.slurm
# Exit 3: resubmit; finished arms are skipped and the running arm resumes.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
RUN_NAME=task25_v1
export UV_PROJECT="$SLURM_SUBMIT_DIR/envs/xpu"
export HF_HOME="${HF_HOME:-/scratch/group/p.ele260080.000/hf_cache}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
module load WebProxy
module load uv
uv sync --frozen
echo "=== job $SLURM_JOB_ID | $(hostname) | time limit $(squeue -h -j "$SLURM_JOB_ID" -o %l) | $RUN_NAME | commit $(git rev-parse HEAD) dirty=$(test -n "$(git status --porcelain)" && echo yes || echo no) ==="
if ! uv run python -c "import sys, torch; n = torch.xpu.device_count(); print('xpu_device_count', n); sys.exit(0 if n > 0 else 1)"; then
    echo "ERROR: no XPU visible on $(hostname); resubmit with --exclude=$(hostname)" >&2
    exit 1
fi
[ -f "artifacts/task25/$RUN_NAME/report.json" ] && { echo "ERROR: $RUN_NAME already finished" >&2; exit 1; }
ls runs/task24/task24_v1/ladder_pairs.pt runs/task24/task24_v1/init_state.pt runs/task23/task23_v1/pair.pt > /dev/null

uv run python -u scripts/task25_ablation.py all --config configs/task25.yaml --device xpu --run-name "$RUN_NAME" --wandb-mode online &
PID=$!
trap 'echo "TERM received: asking training to checkpoint"; kill -TERM $PID' TERM
set +e
wait $PID; STATUS=$?
[ $STATUS -gt 128 ] && { wait $PID; STATUS=$?; }
set -e
echo "=== exit $STATUS (0 = all arms and report; 3 = checkpointed, resubmit) ==="
sacct -j "$SLURM_JOB_ID" --format=JobID,JobName,Partition,AllocTRES%60,Elapsed,State || true
exit $STATUS
