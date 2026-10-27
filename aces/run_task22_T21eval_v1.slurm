#!/bin/bash
#SBATCH --job-name=muq-task22-T21eval-v1
#SBATCH --output=aces/logs/%x-%j.out
#SBATCH --error=aces/logs/%x-%j.err
#SBATCH --time=02:00:00          # expected ~35 min
#SBATCH --partition=pvc
#SBATCH --gres=gpu:pvc:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=andrea.cogliati@rochester.edu

# TASK-22 baseline (configs/task22.yaml arm T21eval): evaluation only -- no training -- of TASK-21's
# last_hidden_state checkpoints at the matched steps (milestones 10k/20k/50k/100k/200k), plus TASK-21's
# best.pt (step 60,000 = its verified best within 200,000 steps) and final.pt, re-evaluated as a
# reproducibility check of TASK-21's own evaluation. Uses the existing last_hidden_state feature cache
# (data/cache/task18, a cache hit). Independent of the L1 job; can run before, after, or alongside it.
#   mkdir -p aces/logs && sbatch aces/run_task22_T21eval_v1.slurm

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
RUN_NAME=task22_T21eval_v1
export UV_PROJECT="$SLURM_SUBMIT_DIR/envs/xpu"
export HF_HOME="${HF_HOME:-/scratch/group/p.ele260080.000/hf_cache}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

module load WebProxy
module load uv
uv sync --frozen
echo "=== job $SLURM_JOB_ID | $(hostname) | $RUN_NAME ==="
echo "commit $(git rev-parse HEAD) dirty=$(test -n "$(git status --porcelain)" && echo yes || echo no)"
if [ -f "artifacts/task22/$RUN_NAME/report.json" ]; then
    echo "ERROR: $RUN_NAME already finished; bump the version for a new run" >&2
    exit 1
fi
ls -l runs/task21/task21_N512ext_v1/checkpoints/{best,final}.pt runs/task21/task21_N512ext_v1/checkpoints/milestone_0000{1,2,5}0000.pt \
    runs/task21/task21_N512ext_v1/checkpoints/milestone_000{1,2}00000.pt || echo "WARNING: a TASK-21 checkpoint is missing (it will be reported unavailable)"
uv run python scripts/build_task20_manifests.py > "aces/logs/task22-manifests-$SLURM_JOB_ID.json"
test "$(grep -c '"unchanged"' "aces/logs/task22-manifests-$SLURM_JOB_ID.json")" -eq 3 \
    || { echo "ERROR: TASK-20 manifests differ from the committed ones" >&2; exit 1; }

uv run python -u scripts/train_task18_heldout.py --config configs/task22.yaml --arm T21eval --device xpu \
    --run-name "$RUN_NAME" --wandb-mode "${WANDB_MODE_ARG:-online}"
sacct -j "$SLURM_JOB_ID" --format=JobID,JobName,Partition,AllocTRES%60,Elapsed,State || true
