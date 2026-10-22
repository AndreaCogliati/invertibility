#!/bin/bash
# TASK-20 bounded smoke test on one ACES PVC. Run INTERACTIVELY on a compute node from the repo root:
#   srun --partition=pvc --gres=gpu:pvc:1 --ntasks=1 --cpus-per-task=8 --mem=32G --time=01:00:00 --pty bash
#   module load WebProxy uv && bash aces/task20_smoke.sh
# Steps (stops at the first failure):
#   1. manifests: TASK-18/19/20 rebuild unchanged; 576 WAV checksums
#   2. N512 code path on a tiny subset (configs/task20_smoke.yaml): training with early stopping
#      (patience 2 checks of 10 steps), evaluation, listening export
#   3. resume equivalence WITH early stopping on XPU: two uninterrupted runs (noise floor) vs one
#      interrupted at step 30 and resumed; both must stop at the same step with the same best step
# Writes artifacts/task20_smoke/aces_<job>_<time>/ACES_SMOKE_REPORT.md. Send that file back.
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_PROJECT="$PWD/envs/xpu"
export HF_HOME="${HF_HOME:-/scratch/group/p.ele260080.000/hf_cache}"
export WANDB_MODE=disabled
[ -n "${SLURM_JOB_ID:-}" ] || { echo "ERROR: run inside a SLURM allocation (srun ... --pty bash)" >&2; exit 1; }
TAG="aces_${SLURM_JOB_ID}_$(date -u +%Y%m%dT%H%M%SZ)"
OUT="artifacts/task20_smoke/$TAG"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/smoke.log") 2>&1
T0=$(date +%s); stamp() { echo "[$(( $(date +%s) - T0 )) s] $*"; }
echo "=== TASK-20 ACES smoke $TAG | $(hostname) | commit $(git rev-parse HEAD) dirty=$(test -n "$(git status --porcelain)" && echo yes || echo no) ==="
uv sync --frozen

stamp "--- 1/3 manifests ---"
uv run python scripts/build_task18_manifests.py | grep -c '"unchanged"' | grep -q 2
uv run python scripts/build_task19_manifests.py | grep -c '"unchanged"' | grep -q 3
uv run python scripts/build_task20_manifests.py --verify-audio | tee "$OUT/manifests.json"
test "$(grep -c '"unchanged"' "$OUT/manifests.json")" -eq 3

RUN() { uv run python -u scripts/train_task18_heldout.py --config configs/task20_smoke.yaml --arm N512 --device xpu --wandb-mode disabled "$@"; }
stamp "--- 2/3 N512 code path ---"
RUN --run-name "${TAG}_full"

stamp "--- 3/3 resume equivalence with early stopping ---"
RUN --run-name "${TAG}_ref1" || [ $? -eq 3 ]
RUN --run-name "${TAG}_ref2" || [ $? -eq 3 ]
RUN --run-name "${TAG}_resumed" --stop-after-step 30 || [ $? -eq 3 ]
RUN --run-name "${TAG}_resumed" || [ $? -eq 3 ]
C=runs/task20_smoke
uv run python scripts/task18_compare_checkpoints.py "$C/${TAG}_ref1/checkpoints/final.pt" "$C/${TAG}_resumed/checkpoints/final.pt" \
    --noise-reference "$C/${TAG}_ref2/checkpoints/final.pt" | tee "$OUT/resume_check.txt"
uv run python - "$C/${TAG}_ref1" "$C/${TAG}_resumed" <<'PY' | tee -a "$OUT/resume_check.txt"
import sys
from muq_inversion.device import load_portable
a, b = (load_portable(f"{d}/checkpoints/final.pt")["train_state"] for d in sys.argv[1:])
print("early stopping ref:", a["early_stopping"], "best", a["best"]["step"])
print("early stopping resumed:", b["early_stopping"], "best", b["best"]["step"])
ok = a["early_stopping"]["stopped_at_step"] == b["early_stopping"]["stopped_at_step"] and a["best"]["step"] == b["best"]["step"]
print("EARLY-STOPPING STATE PRESERVED:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
{
    echo "# TASK-20 ACES smoke report ($TAG)"; echo
    echo "- Job $SLURM_JOB_ID on $(hostname); commit $(git rev-parse HEAD); $(( $(date +%s) - T0 )) s"
    echo "- Code path: artifacts/task20_smoke/${TAG}_full/ (tiny subset: NOT a result)"; echo
    echo '```'; cat "$OUT/resume_check.txt"; echo '```'
} > "$OUT/ACES_SMOKE_REPORT.md"
stamp "=== SMOKE PASSED -> $OUT/ACES_SMOKE_REPORT.md ==="
