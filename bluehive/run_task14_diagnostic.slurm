#!/bin/bash
#SBATCH --job-name=muq-gate5-diagnostic
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=00:15:00
#SBATCH --partition=preempt          # e.g. "gpu" -- check `sinfo -o "%P %G"` on BlueHive for your allocation's GPU partitions
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=andrea.cogliati@rochester.edu

# TASK-14 (tasks/TASK-14.md): OPTIONAL bounded diagnostic run on BlueHive itself, using
# configs/gate5_smoke.yaml (20 steps, 4 train tracks, 4+4 held-out crops -- same bound as the
# local M4 Max diagnostic this task already ran and verified, including a genuine partial-train-
# then-resume continuation test). This exercises the actual CUDA path (imports, CUDA determinism
# settings, manifest loading, MuQ/EnCodec extraction, forward/backward, checkpoint write/load,
# W&B init, argument parsing) on real BlueHive hardware before committing to the ~16-hour full
# run. NOT scientific evidence -- outputs land under runs/task14_gate5_smoke and
# artifacts/task14_gate5_smoke, clearly separate from the full run's task14_gate5_full_gtzan
# directories, and must never be reported as Gate 5 evidence.
#
# This file was NOT submitted by this task (this environment has no BlueHive SSH/sbatch access,
# same limitation as TASK-03 -- see bluehive/README.md) and is optional: TASK-13's own BlueHive
# submission already validated the CUDA determinism fix this script also relies on
# (`_enable_cuda_determinism`, duplicated from scripts/train_gate4_encodec_wandb.py). Submit this
# only if you want an extra CUDA-specific check before the full job; otherwise go straight to
# bluehive/run_task14.slurm once explicitly instructed to.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

echo "=== job $SLURM_JOB_ID on $(hostname), GPU(s): $CUDA_VISIBLE_DEVICES ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

module load cuda 2>/dev/null || true
uv sync --frozen

if [ ! -f "$HOME/.netrc" ] && [ -z "${WANDB_API_KEY:-}" ]; then
    echo "ERROR: no W&B credentials found (~/.netrc or \$WANDB_API_KEY). Run 'uv run wandb login' first." >&2
    exit 1
fi

test -f data/manifests/gtzan_manifest_smoke.csv || { echo "ERROR: data/manifests/gtzan_manifest_smoke.csv missing" >&2; exit 1; }
test -f data/manifests/gate5_heldout_crops.csv || { echo "ERROR: data/manifests/gate5_heldout_crops.csv missing -- run scripts/build_gate5_manifests.py first" >&2; exit 1; }
# The smoke config only needs a handful of real tracks (train_track_limit: 4) to be present --
# not the full 999-track GTZAN copy the full run requires.
test -f data/raw/gtzan/genres/blues/blues.00000.wav || { echo "ERROR: data/raw/gtzan/ not populated -- see bluehive/README.md" >&2; exit 1; }

uv run python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('CUDA OK:', torch.cuda.get_device_name(0))"

uv run python -u scripts/train_gate5_full_gtzan_wandb.py \
    --config configs/gate5_smoke.yaml \
    --wandb-mode online

echo "=== job $SLURM_JOB_ID finished (DIAGNOSTIC -- not scientific evidence) ==="
