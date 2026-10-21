#!/bin/bash
#SBATCH --job-name=muq-task15-bigvgan
#SBATCH --output=aces/logs/%x-%j.out
#SBATCH --error=aces/logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=pvc          # Intel Data Center GPU Max (PVC) nodes on ACES
#SBATCH --gres=gpu:pvc:1
#SBATCH --ntasks=1               # one Python process at a time (required on ACES)
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=andrea.cogliati@rochester.edu

# TASK-15 (tasks/TASK-15.md): MuQ -> BigVGAN-v2 mel -> frozen BigVGAN-v2, small-N memorization,
# on one Intel XPU (ACES PVC). Runs the whole task in order, exactly as on the M4 Max:
#   1. BigVGAN positive control on the 8 crops   (scripts/task15_bigvgan_oracle.py)
#   2. N=2 memorization                          (scripts/train_task15_bigvgan_memorization.py --n 2)
#   3. N=8 memorization, only if N=2 meets its pre-registered criteria (TASK-15's decision rule)
# Config: configs/task15_bigvgan_memorization.yaml (unchanged). W&B group task15-bigvgan-memorization.
#
# Submit from the repository root (`sbatch aces/run_task15.slurm`); create aces/logs/ first, since
# SLURM does not create the output directory.
#
# ACES compute nodes have no outbound network. Before submitting, on a login node:
#   export UV_PROJECT=$PWD/envs/xpu HF_HOME=<same cache as HF_HOME below>
#   uv sync --frozen
#   uv run python scripts/fetch_bigvgan.py
#   uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('OpenMuQ/MuQ-large-msd-iter')"
# W&B runs offline here; afterwards, on a login node: uv run wandb sync wandb/offline-run-*
#
# Resources: 1 GPU. Both frozen models (MuQ-large, BigVGAN-v2 112M params) run inference only;
# the trained translator has 1.27M params and trains on 2 or 8 fixed crops whose MuQ features and
# target mels are computed once. On the M4 Max the whole task took a few minutes (training 38 s for
# N=2, 69 s for N=8). One hour leaves room for model loading and a slow node. Only 8 GTZAN tracks
# are needed, not the full dataset. Less than 32G of memory killed the smoke test while loading
# the models.
#
# Expect results close to, but not bit-identical with, the MPS runs: kernels and reduction order
# differ across backends, so compare by TASK-15's pass criteria, not exact losses.
# Adam + L1 at a constant LR can spike near the minimum (the MPS N=2 run spiked at its final
# step); the training script evaluates the best checkpoint as well as the final one for this reason.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

# Intel XPU environment (envs/xpu: torch/torchaudio/torchvision +xpu builds), not the canonical one.
export UV_PROJECT="$SLURM_SUBMIT_DIR/envs/xpu"
# MuQ weights come from the cache filled on the login node; never try the network.
export HF_HOME="${HF_HOME:-/scratch/group/p.ele260080.000/hf_cache}"
export HF_HUB_OFFLINE=1

module load uv
uv sync --frozen

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
    test -f "$wav" || { echo "ERROR: $wav missing -- copy data/raw/gtzan/ onto ACES first" >&2; exit 1; }
done

# BigVGAN-v2 code + weights: an explicit preparation step on a login node (no network here).
if [ ! -f third_party/bigvgan_v2_24khz_100band_256x/bigvgan_generator.pt ]; then
    echo "ERROR: BigVGAN-v2 missing -- run 'uv run python scripts/fetch_bigvgan.py' on a login node first" >&2
    exit 1
fi
cat third_party/bigvgan_v2_24khz_100band_256x/REVISION

# No silent CPU fallback (CLAUDE.md rule 10): --device xpu fails if XPU is unavailable. The bounded
# smoke test exercises every pipeline stage on the GPU before the real runs start.
uv run python -u scripts/task15_smoke_test.py --device xpu

echo "--- step 1/3: BigVGAN oracle positive control ---"
uv run python -u scripts/task15_bigvgan_oracle.py --device xpu

echo "--- step 2/3: N=2 memorization ---"
uv run python -u scripts/train_task15_bigvgan_memorization.py --n 2 --device xpu --wandb-mode offline

# TASK-15 decision rule: run N=8 only if N=2 memorizes. The pre-registered numerical criteria are
# read from the N=2 report (best checkpoint); listening remains a separate, manual check.
N2_PASSED=$(uv run python -c "import json; print(json.load(open('artifacts/task15_bigvgan_memorization/n2/report.json'))['best']['summary']['memorization_criteria_passed'])")
if [ "$N2_PASSED" != "True" ]; then
    echo "N=2 did not meet the pre-registered memorization criteria -- per TASK-15, stopping before N=8 to diagnose."
    echo "=== job $SLURM_JOB_ID finished (N=8 skipped) ==="
    exit 0
fi

echo "--- step 3/3: N=8 memorization ---"
uv run python -u scripts/train_task15_bigvgan_memorization.py --n 8 --device xpu --wandb-mode offline

echo "=== job $SLURM_JOB_ID finished ==="
echo "Outputs: artifacts/task15_bigvgan_memorization/{oracle,n2,n8}/ (WAVs, mel plots, report.json), runs/task15_bigvgan_memorization/"
echo "W&B: sync from a login node with 'uv run wandb sync wandb/offline-run-*'"
