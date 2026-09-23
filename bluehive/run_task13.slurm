#!/bin/bash
#SBATCH --job-name=muq-gate4-encodec-revalidation
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=preempt          # e.g. "gpu" -- check `sinfo -o "%P %G"` on BlueHive for your allocation's GPU partitions
#SBATCH --gres=gpu:1                 # 1 GPU is enough -- see resource justification below
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=andrea.cogliati@rochester.edu

# TASK-13 (tasks/TASK-13.md): Gate 4 revalidation on the merged (post-experiment/muq-layer-sweep)
# architecture, on BlueHive, with full W&B instrumentation. Reproduces the M4 Max run recorded in
# artifacts/task13_gate4_encodec_revalidation/TASK-13_RESULTS.md -- same manifest, architecture,
# loss, optimizer, and step budget; only the device changes (MPS -> CUDA). CHANGE_ME above
# (--partition, --mail-user) must match your actual BlueHive allocation before submitting -- see
# bluehive/README.md for how to find them.
#
# Resource choices, justified (fill in the real numbers from your job's sacct/seff output once
# you've run it once -- CLAUDE.md's "record all runtime environment information" requirement):
#   - 1 GPU: the trained translator is 377,344 parameters; MuQ-large and EnCodec are both frozen,
#     inference-mode-only (no gradient storage, no optimizer state) -- this job is not
#     GPU-memory-bound. Any single modern GPU (A100/V100/similar) is comfortably sufficient; the
#     dominant cost is ~40,000 sequential single-example steps (20k main run + 10k pre-checkpoint
#     + 10k resumed), not per-step FLOPs, so more GPUs would not meaningfully speed this up.
#   - 32GB system memory: no large precomputed cache is loaded here (unlike TASK-03) -- this job
#     reads raw GTZAN audio directly and encodes it with MuQ/EnCodec on the fly for only 20 crops
#     total (Experiment B). 32GB leaves comfortable headroom for both frozen models, W&B
#     buffering, and OS overhead.
#   - 1-hour wall-clock budget: the M4 Max reference run (TASK-13) completed the full protocol --
#     20k-step main training + resume-equivalence check (pre/post halves) + four-way conditioning-
#     control evaluation + listening-artifact export -- in well under 10 minutes. BlueHive's GPU
#     should not be slower; 1 hour is a safety margin (mainly to absorb first-run model-download
#     time), not an expected duration -- update this comment with the actual measured BlueHive
#     wall time once you have it.

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
# hidden downloads during training" rule). gate4_crops.csv is tracked in git (already present
# after cloning); the 12 raw GTZAN tracks it references (Experiment B) must be copied over --
# see bluehive/README.md. Unlike TASK-03, no precomputed feature cache is needed -- this script
# encodes raw audio with MuQ/EnCodec directly.
test -f data/manifests/gate4_crops.csv || { echo "ERROR: data/manifests/gate4_crops.csv missing" >&2; exit 1; }
for f in \
    data/raw/gtzan/genres/classical/classical.0000{1,2,3}.wav \
    data/raw/gtzan/genres/disco/disco.0000{1,2,3}.wav \
    data/raw/gtzan/genres/pop/pop.0000{1,2,3}.wav \
    data/raw/gtzan/genres/rock/rock.0000{1,2,3}.wav
do
    test -f "$f" || { echo "ERROR: $f missing -- see bluehive/README.md (Experiment B's 12 tracks must be present)" >&2; exit 1; }
done

# MuQ (OpenMuQ/MuQ-large-msd-iter) and EnCodec (facebook/encodec_24khz) are downloaded via
# huggingface_hub on first use -- compute nodes may not have outbound network access (see
# bluehive/README.md). If this job fails at model instantiation with a network error, either run
# the training script once on a login node first to populate the shared HF cache
# ($HF_HOME, default ~/.cache/huggingface), or set HF_HUB_OFFLINE=1 here after doing so.

# CUDA must actually be visible before training -- fail fast rather than silently falling back
# to CPU (CLAUDE.md rule 10; select_device(requested="auto", allow_cpu_fallback=False) in the
# training script already enforces this, this is a faster, clearer pre-check).
uv run python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('CUDA OK:', torch.cuda.get_device_name(0))"

# The real TASK-13 reproduction: default --steps (20000), online W&B logging.
uv run python -u scripts/train_gate4_encodec_wandb.py \
    --wandb-mode online

echo "=== job $SLURM_JOB_ID finished ==="
