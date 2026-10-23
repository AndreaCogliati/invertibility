#!/bin/bash
#SBATCH --job-name=muq-task21-N512ext-v1
#SBATCH --output=aces/logs/%x-%j.out
#SBATCH --error=aces/logs/%x-%j.err
#SBATCH --partition=pvc          # Intel Data Center GPU Max (PVC) nodes on ACES
#SBATCH --gres=gpu:pvc:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --signal=B:TERM@600      # 10 min before the walltime: checkpoint latest.pt and exit 3
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=andrea.cogliati@rochester.edu
# --time is set by aces/submit_task21_chain.sh (verify the partition limit first; see aces/TASK21_ACES.md).

# TASK-21 training segment (tasks/TASK-21.md, configs/task21.yaml): continue TASK-20's last
# checkpoint for 48 h of cumulative job runtime, no early stopping. Segments run strictly one after
# another (afterany dependencies), all writing runs/task21/task21_N512ext_v1. A segment that starts
# after the budget is spent exits immediately. Evaluation is a separate job (run_task21_eval.slurm).

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
RUN_NAME=task21_N512ext_v1
LATEST=runs/task21/$RUN_NAME/checkpoints/latest.pt
export UV_PROJECT="$SLURM_SUBMIT_DIR/envs/xpu"
export HF_HOME="${HF_HOME:-/scratch/group/p.ele260080.000/hf_cache}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
WANDB_MODE_ARG="${WANDB_MODE_ARG:-online}"

module load WebProxy
module load uv
uv sync --frozen
echo "=== segment job $SLURM_JOB_ID | $(hostname) | $SLURM_JOB_PARTITION | time limit $(squeue -h -j "$SLURM_JOB_ID" -o %l) | $RUN_NAME ==="
echo "commit $(git rev-parse HEAD) dirty=$(test -n "$(git status --porcelain)" && echo yes || echo no)"

if [ -f "$LATEST" ] && uv run python -c "
import sys; from muq_inversion.device import load_portable
sys.exit(0 if load_portable('$LATEST')['train_state'].get('budget_reached') else 1)"; then
    echo "=== runtime budget already reached: nothing to do in this segment ==="
    exit 0
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
echo "=== segment exit $STATUS (0 = budget reached, 3 = checkpointed before walltime) ==="
sacct -j "$SLURM_JOB_ID" --format=JobID,JobName,Partition,AllocTRES%60,Elapsed,State || true
[ $STATUS -eq 3 ] && exit 0   # a clean pre-walltime checkpoint is success for a segment
exit $STATUS
