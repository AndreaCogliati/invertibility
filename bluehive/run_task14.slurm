#!/bin/bash
#SBATCH --job-name=muq-gate5-full-gtzan
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=16:00:00
#SBATCH --partition=preempt          # e.g. "gpu" -- check `sinfo -o "%P %G"` on BlueHive for your allocation's GPU partitions
#SBATCH --gres=gpu:1                 # 1 GPU is enough -- see resource justification below
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=andrea.cogliati@rochester.edu

# TASK-14 (tasks/TASK-14.md): Gate 5 full-GTZAN pilot -- the FULL scientific run
# (configs/gate5_full_gtzan.yaml, 300,000 steps over the full 792-track training split). This is
# NOT a diagnostic; per TASK-14's explicit "do not submit the full BlueHive job unless explicitly
# instructed" constraint, this file must only be submitted (`sbatch bluehive/run_task14.slurm`)
# after the user explicitly says to. CHANGE_ME above (--partition, --mail-user) must match your
# actual BlueHive allocation -- see bluehive/README.md for how to find them.
#
# Resource choices, justified (fill in the real numbers from your job's sacct/seff output once
# you've run it -- CLAUDE.md's "record all runtime environment information" requirement):
#   - 1 GPU: the trained translator is 377,344 parameters; MuQ-large and EnCodec are both frozen,
#     inference-mode-only (no gradient storage, no optimizer state for them) -- this job is not
#     GPU-memory-bound. Any single modern GPU (A40/A100/similar) is comfortably sufficient.
#   - 32GB system memory: no large precomputed cache is used (training crops are sampled with
#     random start offsets across the full training split, encoded on the fly -- a fixed cache
#     of every possible crop position is not feasible); the 414-crop held-out set IS precomputed
#     once at startup and held in memory, which is small (414 x ~50 MuQ frames x 1024ch +
#     150 EnCodec frames x 128ch, well under 1GB).
#   - 16-hour wall-clock budget: extrapolated from this task's own local bounded diagnostic
#     (configs/gate5_smoke.yaml, 20 steps including on-the-fly MuQ+EnCodec encoding every step,
#     measured 2.32s total = ~116ms/step on an M4 Max/MPS) -- 300,000 steps x ~116ms/step =~9.7
#     hours. 16 hours is a documented safety margin (~1.65x), not an expected duration -- BlueHive
#     GPU throughput for this per-step-encoding-heavy workload has NOT been measured yet (unlike
#     TASK-13's cached-feature-free pipeline, which was directly comparable across MPS/CUDA); this
#     estimate should be corrected with real BlueHive numbers after the first submission, and the
#     step budget in configs/gate5_full_gtzan.yaml revisited if the true per-step cost differs
#     substantially. Do not extend the step budget merely because a result hasn't appeared yet --
#     per this task's own explicit instruction.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

echo "=== job $SLURM_JOB_ID on $(hostname), GPU(s): $CUDA_VISIBLE_DEVICES ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# Astral uv only -- CLAUDE.md rule 4/5.
module load cuda 2>/dev/null || true   # if your BlueHive site uses environment modules for CUDA
uv sync --frozen

# Required once per allocation/session -- do NOT commit the resulting API key.
if [ ! -f "$HOME/.netrc" ] && [ -z "${WANDB_API_KEY:-}" ]; then
    echo "ERROR: no W&B credentials found (~/.netrc or \$WANDB_API_KEY). Run 'uv run wandb login' first." >&2
    exit 1
fi

# Data must already be present -- this job does not fetch or prepare GTZAN (CLAUDE.md's "no
# hidden downloads during training" rule). Unlike TASK-13 (12 tracks), the full Gate 5 run needs
# ALL 999 valid GTZAN tracks (792 train / 94 val / 113 test) plus both manifests. See
# bluehive/README.md for how to copy data/raw/gtzan/ onto BlueHive.
test -f data/manifests/gtzan_manifest_smoke.csv || { echo "ERROR: data/manifests/gtzan_manifest_smoke.csv missing" >&2; exit 1; }
test -f data/manifests/gate5_heldout_crops.csv || { echo "ERROR: data/manifests/gate5_heldout_crops.csv missing -- run scripts/build_gate5_manifests.py first" >&2; exit 1; }
TRACK_COUNT=$(find data/raw/gtzan/genres -name '*.wav' 2>/dev/null | wc -l)
if [ "$TRACK_COUNT" -lt 999 ]; then
    echo "ERROR: found only $TRACK_COUNT GTZAN .wav files under data/raw/gtzan/genres, expected >= 999 -- see bluehive/README.md" >&2
    exit 1
fi

# MuQ (OpenMuQ/MuQ-large-msd-iter) and EnCodec (facebook/encodec_24khz) are downloaded via
# huggingface_hub on first use -- compute nodes may not have outbound network access. If this
# job fails at model instantiation with a network error, either run the training script once on
# a login node first to populate the shared HF cache ($HF_HOME, default ~/.cache/huggingface), or
# set HF_HUB_OFFLINE=1 here after doing so.

# CUDA must actually be visible before training -- fail fast rather than silently falling back
# to CPU (CLAUDE.md rule 10).
uv run python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('CUDA OK:', torch.cuda.get_device_name(0))"

# --resume auto: if runs/task14_gate5_full_gtzan/checkpoints/latest.pt already exists (e.g. this
# job was preempted and resubmitted -- see bluehive/resubmit_task14.sh), continues from there
# with the sampler/optimizer/RNG state intact, appending to the same run lineage rather than
# starting a new one. Verified locally: a checkpoint saved partway through
# configs/gate5_smoke.yaml's tiny run, reloaded into a fresh process and resumed to completion,
# reproduces the exact same final metrics as an uninterrupted run of the same length.
uv run python -u scripts/train_gate5_full_gtzan_wandb.py \
    --config configs/gate5_full_gtzan.yaml \
    --wandb-mode online \
    --resume auto

echo "=== job $SLURM_JOB_ID finished ==="
