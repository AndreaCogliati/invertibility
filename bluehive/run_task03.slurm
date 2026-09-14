#!/bin/bash
#SBATCH --job-name=muq-spec-dense-wandb
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=CHANGE_ME        # e.g. "gpu" -- check `sinfo -o "%P %G"` on BlueHive for your allocation's GPU partitions
#SBATCH --account=CHANGE_ME          # your BlueHive allocation/account (check `sacctmgr show assoc user=$USER`)
#SBATCH --gres=gpu:1                 # 1 GPU is enough for this model (15.9M params) -- see bluehive/README.md
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=CHANGE_ME@rochester.edu

# TASK-03 (tasks/TASK-03.md): exact reproduction of TASK-02's dense-coverage spectrogram-to-DAC
# experiment, on BlueHive, with full W&B instrumentation. Every CHANGE_ME above must be filled in
# for your actual BlueHive allocation before submitting -- see bluehive/README.md.
#
# Resource choices, justified (fill in the real numbers from your job's sacct/seff output once
# you've run it once -- CLAUDE.md's "record all runtime environment information" requirement):
#   - 1 GPU: this model is 15.9M parameters (~63MB fp32) and the training loop processes one
#     2-second crop per step (no batching) -- it's not GPU-memory-bound. A100/V100/similar are
#     all comfortably sufficient; the dominant cost is 450000 sequential small steps, not
#     per-step FLOPs, so more GPUs would not meaningfully speed this up without batching changes
#     this task explicitly forbids ("do not redesign the experiment").
#   - 64GB system memory: the DAC+mel cache alone is 10.36GB (loaded fully into memory/GPU);
#     leaves comfortable headroom for the frozen DAC model, W&B buffering, and OS overhead.
#   - 6-hour wall-clock budget: the M4 Max reference run (TASK-02) took 4677s (78 min) training +
#     3805s (63 min) full-split evaluation = ~2.4 hours total; BlueHive's GPU should not be
#     slower, so 6 hours is a safety margin, not an expected duration -- update this comment with
#     the actual measured BlueHive wall time once you have it.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

echo "=== job $SLURM_JOB_ID on $(hostname), GPU(s): $CUDA_VISIBLE_DEVICES ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# Astral uv only -- CLAUDE.md rule 4/5. `uv sync` resolves the CUDA-enabled torch wheel
# automatically on Linux (the same `torch>=2.14.0` pin as pyproject.toml -- no separate CUDA
# extra index needed for the standard PyPI wheels as of this project's pinned version).
module load cuda 2>/dev/null || true   # if your BlueHive site uses environment modules for CUDA
uv sync --frozen

# Required once per allocation/session -- do NOT commit the resulting API key. Either run
# `uv run wandb login` interactively once and let it cache to ~/.netrc on the login node (shared
# home directory), or export WANDB_API_KEY as a BlueHive-managed secret/env var before sbatch.
if [ ! -f "$HOME/.netrc" ] && [ -z "${WANDB_API_KEY:-}" ]; then
    echo "ERROR: no W&B credentials found (~/.netrc or \$WANDB_API_KEY). Run 'uv run wandb login' first." >&2
    exit 1
fi

# Data must already be present -- this job does not fetch or prepare GTZAN (CLAUDE.md's "no
# hidden downloads during training" rule). See bluehive/README.md for how to get
# data/raw/gtzan/, data/manifests/dense_crops.csv, and data/cache/dense/mel_dac_cache.pt onto
# BlueHive before submitting.
test -f data/manifests/dense_crops.csv || { echo "ERROR: data/manifests/dense_crops.csv missing -- see bluehive/README.md" >&2; exit 1; }
test -f data/cache/dense/mel_dac_cache.pt || { echo "ERROR: data/cache/dense/mel_dac_cache.pt missing -- see bluehive/README.md" >&2; exit 1; }

# The real TASK-02 reproduction: default --steps (450000), online W&B logging.
uv run python -u scripts/train_spectrogram_to_dac_wandb.py \
    --wandb-mode online \
    --job-id "$SLURM_JOB_ID"

echo "=== job $SLURM_JOB_ID finished ==="
