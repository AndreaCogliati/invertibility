#!/bin/bash
#SBATCH --job-name=muq-task21-eval-v1
#SBATCH --output=aces/logs/%x-%j.out
#SBATCH --error=aces/logs/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=pvc
#SBATCH --gres=gpu:pvc:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=andrea.cogliati@rochester.edu
# TASK-21 evaluation fallback (normally evaluation runs inside run_task21_v1.slurm): the overall val-selected best.pt and final.pt of task21_N512ext_v1, with the
# TASK-20 metrics and controls on identical crops. Submit after the last training segment finished
# with the budget reached (aces/TASK21_ACES.md).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
CK=runs/task21/task21_N512ext_v1/checkpoints
export UV_PROJECT="$SLURM_SUBMIT_DIR/envs/xpu"
export HF_HOME="${HF_HOME:-/scratch/group/p.ele260080.000/hf_cache}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
module load WebProxy
module load uv
uv sync --frozen
echo "=== eval job $SLURM_JOB_ID | commit $(git rev-parse HEAD) dirty=$(test -n "$(git status --porcelain)" && echo yes || echo no) ==="
uv run python -c "
import sys; from muq_inversion.device import load_portable
st = load_portable('$CK/final.pt')['train_state']
print('final step', st['step'], 'best step', st['best'], 'budget reached', st.get('budget_reached'))
sys.exit(0 if st.get('budget_reached') else 1)" || { echo "ERROR: training has not reached its runtime budget" >&2; exit 1; }
uv run python -u scripts/train_task18_heldout.py --config configs/task21.yaml --arm N512ext --device xpu \
    --evaluate-checkpoints "$CK" --run-name task21_N512ext_v1_eval --wandb-mode "${WANDB_MODE_ARG:-online}"
sacct -j "$SLURM_JOB_ID" --format=JobID,JobName,Partition,AllocTRES%60,Elapsed,State || true
