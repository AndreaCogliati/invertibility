#!/bin/bash
#SBATCH --job-name=muq-task15-bigvgan
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=pvc          # e.g. "gpu" -- check `sinfo -o "%P %G"` on BlueHive for your allocation's GPU partitions
#SBATCH --gres=gpu:pvc:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=andrea.cogliati@rochester.edu

# TASK-15 (tasks/TASK-15.md): MuQ -> BigVGAN-v2 mel -> frozen BigVGAN-v2, small-N memorization.
# Runs the whole task in order, exactly as on the M4 Max:
#   1. BigVGAN positive control on the 8 crops   (scripts/task15_bigvgan_oracle.py)
#   2. N=2 memorization                          (scripts/train_task15_bigvgan_memorization.py --n 2)
#   3. N=8 memorization, only if N=2 meets its pre-registered criteria (TASK-15's decision rule)
# Config: configs/task15_bigvgan_memorization.yaml (unchanged). W&B group task15-bigvgan-memorization.
#
# Resources: 1 GPU. Both frozen models (MuQ-large, BigVGAN-v2 112M params) run inference only;
# the trained translator has 1.27M params and trains on 2 or 8 fixed crops whose MuQ features and
# target mels are computed once. On the M4 Max the whole task took a few minutes (training 38 s for
# N=2, 69 s for N=8). One hour leaves room for model loading, the BigVGAN fetch if needed, and a
# slow node. Only 8 GTZAN tracks are needed, not the full dataset.
#
# Expect results close to, but not bit-identical with, the MPS runs: this script does not enable
# CUDA deterministic algorithms, and TASK-15 has no resume-equivalence check that would need them.
# Adam + L1 at a constant LR can spike near the minimum (the MPS N=2 run spiked at its final
# step); the training script evaluates the best checkpoint as well as the final one for this reason.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

echo "=== job $SLURM_JOB_ID on $(hostname), GPU(s): ${CUDA_VISIBLE_DEVICES:-none} ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# Astral uv only -- CLAUDE.md rules 4/5. TASK-15 adds no dependencies.
module load cuda 2>/dev/null || true
uv sync --frozen

if [ ! -f "$HOME/.netrc" ] && [ -z "${WANDB_API_KEY:-}" ]; then
    echo "ERROR: no W&B credentials found (~/.netrc or \$WANDB_API_KEY). Run 'uv run wandb login' first." >&2
    exit 1
fi

# Never overwrite a previous run's outputs (CLAUDE.md rule 8). The scripts write to fixed
# directories, so move or rename these before resubmitting (including after a preemption).
for d in artifacts/task15_bigvgan_memorization runs/task15_bigvgan_memorization; do
    if [ -e "$d" ]; then
        echo "ERROR: $d already exists -- move it aside before rerunning TASK-15 (outputs are never overwritten)" >&2
        exit 1
    fi
done

# Data: both manifests plus the 8 raw tracks they reference (N=2 is a subset of N=8).
test -f configs/task15_bigvgan_memorization.yaml || { echo "ERROR: configs/task15_bigvgan_memorization.yaml missing" >&2; exit 1; }
for m in data/manifests/gate3_scale_2.csv data/manifests/gate3_scale_8.csv; do
    test -f "$m" || { echo "ERROR: $m missing" >&2; exit 1; }
done
tail -n +2 data/manifests/gate3_scale_8.csv | cut -d, -f4 | while read -r wav; do
    test -f "$wav" || { echo "ERROR: $wav missing -- copy data/raw/gtzan/ onto BlueHive (see bluehive/README.md)" >&2; exit 1; }
done

# BigVGAN-v2 code + weights: an explicit preparation step, not a hidden download during training.
# Best done once on a login node beforehand (`uv run python scripts/fetch_bigvgan.py`), since
# compute nodes may lack outbound network access. If it's missing, fetch it here explicitly.
if [ ! -f third_party/bigvgan_v2_24khz_100band_256x/bigvgan_generator.pt ]; then
    echo "--- fetching BigVGAN-v2 (pinned revision) ---"
    uv run python scripts/fetch_bigvgan.py
fi
cat third_party/bigvgan_v2_24khz_100band_256x/REVISION

# MuQ (OpenMuQ/MuQ-large-msd-iter) is loaded through huggingface_hub; if the compute node has no
# network, run any TASK-15 script once on a login node to populate $HF_HOME, or set HF_HUB_OFFLINE=1.

# No silent CPU fallback (CLAUDE.md rule 10): --device cuda fails if CUDA is unavailable. The bounded
# smoke test exercises every pipeline stage on the GPU before the real runs start.
uv run python -u scripts/task15_smoke_test.py --device cuda

echo "--- step 1/3: BigVGAN oracle positive control ---"
uv run python -u scripts/task15_bigvgan_oracle.py --device cuda

echo "--- step 2/3: N=2 memorization ---"
uv run python -u scripts/train_task15_bigvgan_memorization.py --n 2 --device cuda --wandb-mode online

# TASK-15 decision rule: run N=8 only if N=2 memorizes. The pre-registered numerical criteria are
# read from the N=2 report (best checkpoint); listening remains a separate, manual check.
N2_PASSED=$(uv run python -c "import json; print(json.load(open('artifacts/task15_bigvgan_memorization/n2/report.json'))['best']['summary']['memorization_criteria_passed'])")
if [ "$N2_PASSED" != "True" ]; then
    echo "N=2 did not meet the pre-registered memorization criteria -- per TASK-15, stopping before N=8 to diagnose."
    echo "=== job $SLURM_JOB_ID finished (N=8 skipped) ==="
    exit 0
fi

echo "--- step 3/3: N=8 memorization ---"
uv run python -u scripts/train_task15_bigvgan_memorization.py --n 8 --device cuda --wandb-mode online

echo "=== job $SLURM_JOB_ID finished ==="
echo "Outputs: artifacts/task15_bigvgan_memorization/{oracle,n2,n8}/ (WAVs, mel plots, report.json), runs/task15_bigvgan_memorization/"
