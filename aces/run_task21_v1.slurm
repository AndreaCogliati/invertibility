#!/bin/bash
#SBATCH --job-name=muq-task21-N512ext-v1
#SBATCH --output=aces/logs/%x-%j.out
#SBATCH --error=aces/logs/%x-%j.err
#SBATCH --time=48:00:00          # the pvc partition limit (2-00:00:00, verified with sinfo 2026-10-23)
#SBATCH --partition=pvc          # Intel Data Center GPU Max (PVC) nodes on ACES
#SBATCH --gres=gpu:pvc:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --signal=B:TERM@600      # safety net: 10 min before the walltime, checkpoint latest.pt and exit 3
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=andrea.cogliati@rochester.edu

# TASK-21 (tasks/TASK-21.md, configs/task21.yaml, aces/TASK21_ACES.md): one job that continues TASK-20's
# last checkpoint (step 4,250) with no early stopping for a 47.0 h runtime budget, then evaluates the
# overall val-selected best.pt and final.pt in the same job (~20 min). Expected end ~47.4 h.
#   mkdir -p aces/logs && sbatch aces/run_task21_v1.slurm
# If the job is ever cut short, resubmitting this file resumes from latest.pt and only uses what is
# left of the 47 h budget. If only the evaluation must be rerun: sbatch aces/run_task21_eval.slurm.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
RUN_NAME=task21_N512ext_v1
export UV_PROJECT="$SLURM_SUBMIT_DIR/envs/xpu"
export HF_HOME="${HF_HOME:-/scratch/group/p.ele260080.000/hf_cache}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
WANDB_MODE_ARG="${WANDB_MODE_ARG:-online}"   # online via WebProxy; failures fall back without stopping training

module load WebProxy
module load uv
uv sync --frozen
echo "=== job $SLURM_JOB_ID | $(hostname) | $SLURM_JOB_PARTITION | time limit $(squeue -h -j "$SLURM_JOB_ID" -o %l) | $RUN_NAME ==="
echo "commit $(git rev-parse HEAD) dirty=$(test -n "$(git status --porcelain)" && echo yes || echo no)"

if [ -f "artifacts/task21/$RUN_NAME/report.json" ]; then
    echo "ERROR: $RUN_NAME already finished; bump the version for a new run" >&2
    exit 1
fi
test -f runs/task20/task20_N512_v1/checkpoints/latest.pt || { echo "ERROR: TASK-20 latest.pt missing" >&2; exit 1; }
uv run python scripts/build_task20_manifests.py > "aces/logs/task21-manifests-$SLURM_JOB_ID.json"
test "$(grep -c '"unchanged"' "aces/logs/task21-manifests-$SLURM_JOB_ID.json")" -eq 3 \
    || { echo "ERROR: TASK-20 manifests differ from the committed ones" >&2; exit 1; }

uv run python -u scripts/train_task18_heldout.py --config configs/task21.yaml --arm N512ext --device xpu \
    --run-name "$RUN_NAME" --resume auto --wandb-mode "$WANDB_MODE_ARG" &
PID=$!
trap 'echo "TERM received: asking training to checkpoint"; kill -TERM $PID' TERM
set +e
wait $PID; STATUS=$?
[ $STATUS -gt 128 ] && { wait $PID; STATUS=$?; }
set -e
echo "=== exit $STATUS (0 = trained to the budget and evaluated; 3 = checkpointed before the walltime, resubmit) ==="
sacct -j "$SLURM_JOB_ID" --format=JobID,JobName,Partition,AllocTRES%60,Elapsed,State || true
exit $STATUS
