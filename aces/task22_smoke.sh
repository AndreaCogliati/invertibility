#!/bin/bash
# TASK-22 bounded smoke test (first-block features, cache separation, resume, milestone + baseline
# evaluation, analysis, listening package). On ACES, interactively on a PVC node from the repo root:
#   srun --partition=pvc --gres=gpu:pvc:1 --ntasks=1 --cpus-per-task=8 --mem=32G --time=01:00:00 --pty bash
#   module load WebProxy uv && bash aces/task22_smoke.sh
# Locally (no SLURM):  DEVICE=cpu LOCAL=1 bash aces/task22_smoke.sh
# Writes only under runs/task22_smoke, artifacts/task22_smoke, data/cache/task22_smoke (and, for the
# last_hidden_state stand-in, a small smoke-subset cache file in data/cache/task18 -- a new file,
# never an existing one). Never touches TASK-21/22 runs. Steps (stops at the first failure):
#   0. manifests unchanged; layer verification on real crops (hook-checked hidden-state indices)
#   1. resume equivalence: L1 to step 150 uninterrupted (x2, the second = noise floor) vs stop at 90 + resume
#   2. full L1 smoke run (200 steps, from scratch) with best/final/milestone evaluation
#   3. last_hidden_state smoke run (stands in for TASK-21) + evaluation-only baseline job on its checkpoints
#      (one requested milestone does not exist: must be reported unavailable)
#   4. cache separation, layer identity in the report, matched-step analysis, listening package
set -euo pipefail
cd "$(dirname "$0")/.."
DEVICE="${DEVICE:-xpu}"
if [ -z "${LOCAL:-}" ]; then
    [ -n "${SLURM_JOB_ID:-}" ] || { echo "ERROR: run inside a SLURM allocation (or set LOCAL=1 off-cluster)" >&2; exit 1; }
    export UV_PROJECT="$PWD/envs/xpu"
fi
export WANDB_MODE=disabled
TAG="t22_${SLURM_JOB_ID:-local}_$(date -u +%Y%m%dT%H%M%SZ)"
OUT="artifacts/task22_smoke/$TAG"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/smoke.log") 2>&1
T0=$(date +%s); stamp() { echo "[$(( $(date +%s) - T0 )) s] $*"; }
echo "=== TASK-22 smoke $TAG on $DEVICE | commit $(git rev-parse HEAD) dirty=$(test -n "$(git status --porcelain)" && echo yes || echo no) ==="

RUN() { local a=$1; shift; uv run python -u scripts/train_task18_heldout.py --config configs/task22_smoke.yaml --arm "$a" --device "$DEVICE" --wandb-mode disabled "$@"; }
OK3() { local s=0; "$@" || s=$?; [ $s -eq 3 ] || { echo "expected exit 3, got $s" >&2; exit 1; }; }
R=runs/task22_smoke; A=artifacts/task22_smoke

stamp "--- 0/4 manifests and layer verification ---"
uv run python scripts/build_task20_manifests.py | grep -c '"unchanged"' | grep -q 3
uv run python scripts/task22_verify_layer.py --device "$DEVICE" --out "$OUT/layer_verification" | tail -3

stamp "--- 1/4 resume equivalence (L1, from scratch) ---"
OK3 RUN L1 --run-name "${TAG}_ref1" --stop-after-step 150
OK3 RUN L1 --run-name "${TAG}_ref2" --stop-after-step 150
OK3 RUN L1 --run-name "${TAG}_resumed" --stop-after-step 90
OK3 RUN L1 --run-name "${TAG}_resumed" --stop-after-step 150
uv run python scripts/task18_compare_checkpoints.py "$R/${TAG}_ref1/checkpoints/latest.pt" "$R/${TAG}_resumed/checkpoints/latest.pt" \
    --noise-reference "$R/${TAG}_ref2/checkpoints/latest.pt" | tee "$OUT/resume_check.txt"

stamp "--- 2/4 full L1 smoke run ---"
RUN L1 --run-name "${TAG}_L1"

stamp "--- 3/4 last_hidden_state stand-in + baseline evaluation-only job ---"
RUN L12 --run-name "${TAG}_L12"
RUN T21eval --evaluate-checkpoints "$R/${TAG}_L12/checkpoints" --run-name "${TAG}_T21eval"

stamp "--- 4/4 cache separation, layer identity, analysis, listening ---"
uv run python - "$A/${TAG}_L1" "$A/${TAG}_L12" "$A/${TAG}_T21eval" <<'EOF' | tee "$OUT/cache_check.txt"
import json, sys, torch
l1, l12, ev = (json.load(open(f"{d}/report.json")) for d in sys.argv[1:])
c1, c12 = l1["resolved_config"]["feature_cache"], l12["resolved_config"]["feature_cache"]
assert c1.startswith("data/cache/task22_smoke/task18_v1_L1_") and c12.startswith("data/cache/task18/task18_v1_") and "_L1_" not in c12, (c1, c12)
f1, f12 = (torch.load(p, map_location="cpu", weights_only=False) for p in (c1, c12))
assert f1["keys"] == f12["keys"], "different crops"
assert f1["key_parts"]["hidden_state_index"] == 1 and "muq_layer" not in f12["key_parts"]
assert torch.equal(f1["tensors"]["mel"], f12["tensors"]["mel"]) and torch.equal(f1["tensors"]["wave"], f12["tensors"]["wave"])
assert f1["tensors"]["muq"].shape == f12["tensors"]["muq"].shape and not torch.equal(f1["tensors"]["muq"], f12["tensors"]["muq"])
assert torch.isfinite(f1["tensors"]["muq"]).all()
mf = l1["resolved_config"]["muq_feature"]
assert mf["hidden_state_index"] == 1 and mf["index_check"]["index_is_block_output"] and mf["index_check"]["index0_is_block1_input"]
assert l1["extra_checkpoints"]["step_60"]["available"] and l1["extra_checkpoints"]["step_100"]["available"]
assert not ev["extra_checkpoints"]["step_150"]["available"] and ev["extra_checkpoints"]["step_60"]["available"]
assert l1["final_step"] == 200 and (l1.get("early_stopping") is None)
print("cache separation OK:", c1, "vs", c12, "| same crops/mels/waves, different features | layer", mf["name"],
      "| index check", mf["index_check"], "| milestones OK, missing milestone reported unavailable")
EOF
uv run python scripts/task22_analyze.py --t22 "$A/${TAG}_L1" --t21eval "$A/${TAG}_T21eval" --t21 "$A/${TAG}_L12" \
    --window 200 --matched 60 100 150 200 --bootstrap 200 --out "$OUT/analysis" | head -12
uv run python scripts/task22_listening_package.py --l1 "$A/${TAG}_L1" --l12 "$A/${TAG}_L12" --window 200 --out "$OUT/listening_blind"

{
    echo "# TASK-22 smoke report ($TAG, $DEVICE)"; echo
    echo "- commit $(git rev-parse HEAD); $(( $(date +%s) - T0 )) s"; echo
    echo '```'; cat "$OUT/resume_check.txt" "$OUT/cache_check.txt"; echo '```'
} > "$OUT/SMOKE_REPORT.md"
stamp "=== SMOKE PASSED -> $OUT/SMOKE_REPORT.md ==="
