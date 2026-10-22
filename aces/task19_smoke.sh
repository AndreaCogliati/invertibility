#!/bin/bash
# TASK-19 bounded smoke test on one ACES PVC. Run INTERACTIVELY on a compute node from the
# repository root, inside an allocation -- never on a login node:
#
#   srun --partition=pvc --gres=gpu:pvc:1 --ntasks=1 --cpus-per-task=8 --mem=32G --time=01:30:00 --pty bash
#   module load WebProxy uv && bash aces/task19_smoke.sh
#
# Bounded: no full training. Steps (stops at the first failure):
#   1. backend smoke (scripts/task15_smoke_test.py --device xpu)
#   2. manifests: TASK-18 and TASK-19 manifests rebuild unchanged; 192 WAV checksums; fault report
#   3. all four arms on a tiny subset (configs/task19_smoke.yaml, 40 steps); A32 evaluates the REAL
#      TASK-18 checkpoints (runs/task18_heldout/task18_heldout_v1/checkpoints) on the subset
#   4. cross-arm analysis + blinded listening package on the smoke outputs
#   5. resume equivalence for recipe B (LR schedule) on XPU: 2 uninterrupted + 1 interrupted/resumed
# Writes artifacts/task19_smoke/aces_<job>_<time>/ACES_SMOKE_REPORT.md. Send that file back.

set -euo pipefail
cd "$(dirname "$0")/.."

export UV_PROJECT="$PWD/envs/xpu"
export HF_HOME="${HF_HOME:-/scratch/group/p.ele260080.000/hf_cache}"
export WANDB_MODE=disabled

if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "ERROR: not inside a SLURM allocation -- run on a compute node (srun ... --pty bash)" >&2
    exit 1
fi
TAG="aces_${SLURM_JOB_ID}_$(date -u +%Y%m%dT%H%M%SZ)"
OUT="artifacts/task19_smoke/$TAG"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/smoke.log") 2>&1
T0=$(date +%s)
stamp() { echo "[$(( $(date +%s) - T0 )) s] $*"; }

echo "=== TASK-19 ACES smoke $TAG ==="
echo "host $(hostname) | job $SLURM_JOB_ID | partition ${SLURM_JOB_PARTITION:-?} | cpus ${SLURM_CPUS_PER_TASK:-?} | mem ${SLURM_MEM_PER_NODE:-?}"
echo "git $(git rev-parse HEAD 2>/dev/null || echo unknown) dirty=$(test -n "$(git status --porcelain 2>/dev/null)" && echo yes || echo no)"
uv sync --frozen
uv run python -c "import torch; print('torch', torch.__version__, '| xpu', torch.xpu.is_available(), torch.xpu.get_device_name(0))"
CK=runs/task18_heldout/task18_heldout_v1/checkpoints
test -f "$CK/best.pt" -a -f "$CK/final.pt" || { echo "ERROR: $CK/{best,final}.pt missing (needed for A32)" >&2; exit 1; }

stamp "--- 1/5 backend smoke ---"
uv run python -u scripts/task15_smoke_test.py --device xpu

stamp "--- 2/5 manifests ---"
uv run python scripts/build_task18_manifests.py | grep -c '"unchanged"' | grep -q 2
uv run python scripts/build_task19_manifests.py --verify-audio | tee "$OUT/manifests.json"
test "$(grep -c '"unchanged"' "$OUT/manifests.json")" -eq 3

stamp "--- 3/5 four arms (tiny subset) ---"
RUN() { uv run python -u scripts/train_task18_heldout.py --config configs/task19_smoke.yaml --device xpu --wandb-mode disabled "$@"; }
RUN --arm A32 --run-name "${TAG}_A32"
for ARM in A128 B32 B128; do stamp "arm $ARM"; RUN --arm "$ARM" --run-name "${TAG}_$ARM"; done

stamp "--- 4/5 analysis + blinded package ---"
D=artifacts/task19_smoke
ARMS=(--arm "A32=$D/${TAG}_A32" --arm "A128=$D/${TAG}_A128" --arm "B32=$D/${TAG}_B32" --arm "B128=$D/${TAG}_B128")
uv run python scripts/task19_analyze.py "${ARMS[@]}" --out "$OUT/analysis" > /dev/null
uv run python scripts/task19_listening_package.py "${ARMS[@]}" --out "$OUT/listening_blind"

stamp "--- 5/5 resume equivalence (recipe B) ---"
RUN --arm B32 --run-name "${TAG}_ref1" --stop-after-step 40 || [ $? -eq 3 ]
RUN --arm B32 --run-name "${TAG}_ref2" --stop-after-step 40 || [ $? -eq 3 ]
RUN --arm B32 --run-name "${TAG}_resumed" --stop-after-step 20 || [ $? -eq 3 ]
RUN --arm B32 --run-name "${TAG}_resumed" --stop-after-step 40 || [ $? -eq 3 ]
C=runs/task19_smoke
uv run python scripts/task18_compare_checkpoints.py "$C/${TAG}_ref1/checkpoints/latest.pt" "$C/${TAG}_resumed/checkpoints/latest.pt" \
    --noise-reference "$C/${TAG}_ref2/checkpoints/latest.pt" | tee "$OUT/resume_check.txt"

{
    echo "# TASK-19 ACES smoke report ($TAG)"
    echo
    echo "- Job $SLURM_JOB_ID on $(hostname); commit $(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "- Total wall time: $(( $(date +%s) - T0 )) s (stage times in smoke.log)"
    echo "- Arm outputs: $D/${TAG}_{A32,A128,B32,B128}/ (40 steps on a tiny subset: NOT results)"
    echo
    echo '## A32 re-evaluation sanity and smoke analysis'
    echo
    grep -E "outcome|re-evaluation" "$OUT/analysis/TASK-19_ANALYSIS.md"
    echo
    echo '## Resume equivalence (recipe B)'
    echo
    echo '```'
    cat "$OUT/resume_check.txt"
    echo '```'
} > "$OUT/ACES_SMOKE_REPORT.md"
stamp "=== SMOKE PASSED -> $OUT/ACES_SMOKE_REPORT.md ==="
