#!/bin/bash
# TASK-21 bounded smoke test (continuation, budget, graceful shutdown, resume, retention, evaluation).
# On ACES, interactively on a PVC node from the repo root:
#   srun --partition=pvc --gres=gpu:pvc:1 --ntasks=1 --cpus-per-task=8 --mem=32G --time=01:00:00 --pty bash
#   module load WebProxy uv && bash aces/task21_smoke.sh
# Locally (no SLURM):  DEVICE=cpu LOCAL=1 OMP_NUM_THREADS=1 bash aces/task21_smoke.sh
# Writes only under runs/task20_smoke, runs/task21_smoke, artifacts/task21_smoke (never TASK-20/21 runs).
# Steps (stops at the first failure):
#   0. parent: a TASK-20 smoke run stopped at step 70, before its early stop at 80 (stands in for TASK-20's last checkpoint)
#   1. exact continuation: continue to step 150 uninterrupted vs stop at 120 + resume to 150 (+ a
#      second uninterrupted run as the noise floor on accelerators); continuation check vs the parent
#   2. optimizer state and global step survive: Adam step count == global step after each resume
#   3. graceful shutdown: SIGTERM mid-run -> latest.pt + exit 3; resume -> runs to the runtime budget,
#      writes final.pt, exits 0 (train_only); recovery files bounded
#   4. separate evaluation job on the continuation's checkpoints
set -euo pipefail
cd "$(dirname "$0")/.."
DEVICE="${DEVICE:-xpu}"
if [ -z "${LOCAL:-}" ]; then
    [ -n "${SLURM_JOB_ID:-}" ] || { echo "ERROR: run inside a SLURM allocation (or set LOCAL=1 off-cluster)" >&2; exit 1; }
    export UV_PROJECT="$PWD/envs/xpu"
fi
export WANDB_MODE=disabled
TAG="t21_${SLURM_JOB_ID:-local}_$(date -u +%Y%m%dT%H%M%SZ)"
OUT="artifacts/task21_smoke/$TAG"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/smoke.log") 2>&1
T0=$(date +%s); stamp() { echo "[$(( $(date +%s) - T0 )) s] $*"; }
echo "=== TASK-21 smoke $TAG on $DEVICE | commit $(git rev-parse HEAD) dirty=$(test -n "$(git status --porcelain)" && echo yes || echo no) ==="
uv run python scripts/build_task20_manifests.py | grep -c '"unchanged"' | grep -q 3

T20() { uv run python -u scripts/train_task18_heldout.py --config configs/task20_smoke.yaml --arm N512 --device "$DEVICE" --wandb-mode disabled "$@"; }
T21() { uv run python -u scripts/train_task18_heldout.py --config configs/task21_smoke.yaml --arm N512ext --device "$DEVICE" --wandb-mode disabled "$@"; }
OK3() { local s=0; "$@" || s=$?; [ $s -eq 3 ] || { echo "expected exit 3, got $s" >&2; exit 1; }; }
PARENT=runs/task20_smoke/${TAG}_parent/checkpoints
C=runs/task21_smoke

stamp "--- 0/4 parent (TASK-20 smoke, stop at step 70) ---"
OK3 T20 --run-name "${TAG}_parent" --stop-after-step 70

stamp "--- 1/4 exact continuation ---"
OK3 T21 --init-from "$PARENT" --run-name "${TAG}_ref1" --stop-after-step 150
OK3 T21 --init-from "$PARENT" --run-name "${TAG}_ref2" --stop-after-step 150
OK3 T21 --init-from "$PARENT" --run-name "${TAG}_resumed" --stop-after-step 120
OK3 T21 --init-from "$PARENT" --run-name "${TAG}_resumed" --stop-after-step 150
uv run python scripts/task18_compare_checkpoints.py "$C/${TAG}_ref1/checkpoints/latest.pt" "$C/${TAG}_resumed/checkpoints/latest.pt" \
    --noise-reference "$C/${TAG}_ref2/checkpoints/latest.pt" | tee "$OUT/continuation_check.txt"

stamp "--- 2/4 optimizer state and step survive ---"
uv run python scripts/task21_check_run.py "$PARENT/latest.pt" "$C/${TAG}_resumed" | tee "$OUT/state_check.txt"

stamp "--- 3/4 graceful shutdown and runtime budget ---"
T21 --init-from "$PARENT" --run-name "${TAG}_term" &
PID=$!
for _ in $(seq 1 600); do [ -f "$C/${TAG}_term/checkpoints/latest.pt" ] && break; sleep 1; done
sleep 5
pkill -TERM -f "run-name ${TAG}_term" || true
S=0; wait $PID || S=$?
[ $S -eq 3 ] || { echo "graceful shutdown: expected exit 3, got $S" >&2; exit 1; }
T21 --init-from "$PARENT" --run-name "${TAG}_term"      # resumes; trains to the runtime budget; exit 0
uv run python scripts/task21_check_run.py "$PARENT/latest.pt" "$C/${TAG}_term" --expect-final | tee -a "$OUT/state_check.txt"

stamp "--- 4/4 evaluation job ---"
T21 --evaluate-checkpoints "$C/${TAG}_term/checkpoints" --run-name "${TAG}_term_eval"
test -f "artifacts/task21_smoke/${TAG}_term_eval/report.json"

{
    echo "# TASK-21 smoke report ($TAG, $DEVICE)"; echo
    echo "- commit $(git rev-parse HEAD); $(( $(date +%s) - T0 )) s"; echo
    echo '```'; cat "$OUT/continuation_check.txt" "$OUT/state_check.txt"; echo '```'
} > "$OUT/SMOKE_REPORT.md"
stamp "=== SMOKE PASSED -> $OUT/SMOKE_REPORT.md ==="
