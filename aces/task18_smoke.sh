#!/bin/bash
# TASK-18 bounded smoke test on one ACES PVC (Intel XPU). Run INTERACTIVELY on a compute node, from
# the repository root, inside an allocation -- never on a login node:
#
#   srun --partition=pvc --gres=gpu:pvc:1 --ntasks=1 --cpus-per-task=8 --mem=32G --time=01:00:00 --pty bash
#   bash aces/task18_smoke.sh
#
# Bounded: no full training. Steps (each must pass; the script stops at the first failure):
#   1. backend smoke test (scripts/task15_smoke_test.py --device xpu): every pipeline stage on XPU
#   2. manifest integrity: the committed TASK-18 manifests rebuild byte-identically, audio checksums match
#   3. TASK-18 code path on a tiny subset (configs/task18_smoke.yaml, 40 steps): features, training,
#      all controls/metrics, listening export, outcome rule, report
#   4. checkpoint/resume equivalence on XPU: two uninterrupted runs (noise floor) vs one run
#      interrupted at step 20 and resumed, compared at step 40
# Writes artifacts/task18_smoke/aces_<jobid>/ACES_SMOKE_REPORT.md. Send that file back.

set -euo pipefail
cd "$(dirname "$0")/.."

export UV_PROJECT="$PWD/envs/xpu"
export HF_HOME="${HF_HOME:-/scratch/group/p.ele260080.000/hf_cache}"
export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled

TAG="aces_${SLURM_JOB_ID:-nojob}_$(date -u +%Y%m%dT%H%M%SZ)"
OUT="artifacts/task18_smoke/$TAG"
mkdir -p "$OUT"
LOG="$OUT/smoke.log"
exec > >(tee -a "$LOG") 2>&1

if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "ERROR: not inside a SLURM allocation -- run this on a compute node (srun ... --pty bash), not a login node" >&2
    exit 1
fi

echo "=== TASK-18 ACES smoke $TAG ==="
echo "host $(hostname) | job $SLURM_JOB_ID | partition ${SLURM_JOB_PARTITION:-?} | cpus ${SLURM_CPUS_PER_TASK:-?} | mem ${SLURM_MEM_PER_NODE:-?}"
echo "git $(git rev-parse HEAD 2>/dev/null || echo unknown) dirty=$(test -n "$(git status --porcelain 2>/dev/null)" && echo yes || echo no)"
uv sync --frozen
uv run python -c "import torch; print('torch', torch.__version__, '| xpu', torch.xpu.is_available(), torch.xpu.get_device_name(0) if torch.xpu.is_available() else '')"

RUN() { uv run python -u scripts/train_task18_heldout.py --config configs/task18_smoke.yaml --device xpu --wandb-mode disabled "$@"; }

echo "--- 1/4 backend smoke test ---"
uv run python -u scripts/task15_smoke_test.py --device xpu

echo "--- 2/4 manifest integrity ---"
uv run python scripts/build_task18_manifests.py --verify-audio | tee "$OUT/manifests.json"
grep -q '"data/manifests/task18_tracks.csv": "unchanged"' "$OUT/manifests.json"
grep -q '"data/manifests/task18_crops.csv": "unchanged"' "$OUT/manifests.json"

echo "--- 3/4 TASK-18 code path (tiny subset, 40 steps) ---"
RUN --run-name "${TAG}_full"

echo "--- 4/4 checkpoint/resume equivalence ---"
RUN --run-name "${TAG}_ref1" --stop-after-step 40 || [ $? -eq 3 ]
RUN --run-name "${TAG}_ref2" --stop-after-step 40 || [ $? -eq 3 ]
RUN --run-name "${TAG}_resumed" --stop-after-step 20 || [ $? -eq 3 ]
RUN --run-name "${TAG}_resumed" --stop-after-step 40 || [ $? -eq 3 ]
C=runs/task18_smoke
uv run python scripts/task18_compare_checkpoints.py "$C/${TAG}_ref1/checkpoints/latest.pt" \
    "$C/${TAG}_resumed/checkpoints/latest.pt" --noise-reference "$C/${TAG}_ref2/checkpoints/latest.pt" \
    | tee "$OUT/resume_check.txt"

{
    echo "# TASK-18 ACES smoke report ($TAG)"
    echo
    echo "- Job $SLURM_JOB_ID on $(hostname), partition ${SLURM_JOB_PARTITION:-?}, ${SLURM_CPUS_PER_TASK:-?} CPUs, ${SLURM_MEM_PER_NODE:-?} MB"
    echo "- Commit $(git rev-parse HEAD 2>/dev/null || echo unknown) (dirty: $(test -n "$(git status --porcelain 2>/dev/null)" && echo yes || echo no))"
    echo "- Backend smoke: $(ls -td artifacts/backend_smoke/xpu_* | head -1)/report.json"
    echo "- Code-path smoke report: artifacts/task18_smoke/${TAG}_full/TASK-18_RESULTS.md (40 steps on a tiny subset: NOT a result)"
    echo
    echo '## Resume equivalence'
    echo
    echo '```'
    cat "$OUT/resume_check.txt"
    echo '```'
    echo
    echo "All steps passed. Full log: $LOG"
} > "$OUT/ACES_SMOKE_REPORT.md"
echo "=== SMOKE PASSED -> $OUT/ACES_SMOKE_REPORT.md ==="
