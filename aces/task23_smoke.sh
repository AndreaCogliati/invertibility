#!/bin/bash
# TASK-23 bounded smoke test. On ACES, interactively on a PVC node from the repo root:
#   srun --partition=pvc --gres=gpu:pvc:1 --ntasks=1 --cpus-per-task=8 --mem=32G --time=00:45:00 --exclude=ac079 --pty bash
#   module load WebProxy uv && bash aces/task23_smoke.sh
# Locally (no SLURM, no TASK-22 cache: step 3 is skipped):  DEVICE=mps LOCAL=1 bash aces/task23_smoke.sh
# Writes only under runs/task23_smoke and artifacts/task23_smoke. Steps (stops at the first failure):
#   0. unit tests
#   1. resume equivalence, bit-exact: CPU with deterministic algorithms; B stopped at 150 and resumed must
#      equal C (same init, same config, uninterrupted) at every report point and in its final weights
#   2. accelerator smoke (features recomputed): stop/resume, all arms incl. the extension and C_seed1
#      branches, plots, audio, report; B-C spread = the backend's run-to-run nondeterminism
#   3. TASK-22 cache path on the accelerator (ACES only): the real pair from data/cache/task22, fresh-vs-cache check
set -euo pipefail
cd "$(dirname "$0")/.."
DEVICE="${DEVICE:-xpu}"
if [ -z "${LOCAL:-}" ]; then
    [ -n "${SLURM_JOB_ID:-}" ] || { echo "ERROR: run inside a SLURM allocation (or set LOCAL=1 off-cluster)" >&2; exit 1; }
    export UV_PROJECT="$PWD/envs/xpu"
    export HF_HOME="${HF_HOME:-/scratch/group/p.ele260080.000/hf_cache}"
fi
TAG="t23_${SLURM_JOB_ID:-local}_$(date -u +%Y%m%dT%H%M%SZ)"
OUT="artifacts/task23_smoke/$TAG"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/smoke.log") 2>&1
T0=$(date +%s); stamp() { echo "[$(( $(date +%s) - T0 )) s] $*"; }
echo "=== TASK-23 smoke $TAG on $DEVICE | commit $(git rev-parse HEAD) dirty=$(test -n "$(git status --porcelain)" && echo yes || echo no) ==="
RUN() { uv run python -u scripts/task23_overfit.py "$@"; }
OK3() { local s=0; "$@" || s=$?; [ $s -eq 3 ] || { echo "expected exit 3, got $s" >&2; exit 1; }; }
CHECK() {  # report.json of run $1: print the key facts; $2 = require B == C bit-exact
    uv run python - "artifacts/task23_smoke/$1/report.json" "$2" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
for k, s in r["summaries"].items():
    print(f"  {k}: steps {s['steps_run']}, initial {s['initial']['eval_l1']:.4f}, best {s['best']['eval_l1']:.5f} @ {s['best']['step']}, "
          f"criterion {s['criterion_met']}, extension {s['extension']}, train/eval max diff {s['max_train_eval_abs_diff']:.1e}, "
          f"{s['steps_run'] / max(s['runtime_seconds'], 1e-9):.0f} updates/s")
print("  nondeterminism:", r["nondeterminism"])
print("  interpretation:", r["interpretation"]["rule"])
assert r["b_c_same_init"], "B and C must start from identical weights"
if sys.argv[2] == "exact":
    assert r["b_c_identical"] and r["nondeterminism"]["max_abs_eval_l1_difference"] == 0.0, "resume is not bit-exact"
    print("  RESUME CHECK PASS: B (stopped at 150, resumed) == C (uninterrupted), bit-exact")
EOF
}

stamp "--- 0/3 unit tests ---"
uv run pytest -q tests/test_task23.py

stamp "--- 1/3 bit-exact resume (CPU, deterministic algorithms) ---"
OK3 RUN --config configs/task23_smoke.yaml --device cpu --deterministic --run-name "${TAG}_cpu" --stop-after B:150
RUN --config configs/task23_smoke.yaml --device cpu --deterministic --run-name "${TAG}_cpu"
CHECK "${TAG}_cpu" exact

stamp "--- 2/3 accelerator smoke ($DEVICE) ---"
OK3 RUN --config configs/task23_smoke.yaml --device "$DEVICE" --run-name "${TAG}_${DEVICE}" --stop-after B:150
RUN --config configs/task23_smoke.yaml --device "$DEVICE" --run-name "${TAG}_${DEVICE}"
CHECK "${TAG}_${DEVICE}" spread
ls "artifacts/task23_smoke/${TAG}_${DEVICE}"/{A,B,C,C_seed1}/{learning_curves,progression,best_spectrogram,best_zoom_transient,best_zoom_harmonic}.png \
   "artifacts/task23_smoke/${TAG}_${DEVICE}"/audio/{original,ground_truth_mel_bigvgan}.wav > /dev/null
uv run python - "$DEVICE" "$OUT" <<'EOF' || echo "  (deterministic algorithms not available on this backend: recorded, not required)"
import subprocess, sys
dev, out = sys.argv[1:]
cfg = f"{out}/det_probe.yaml"
open(cfg, "w").write("base_config: configs/task23_smoke.yaml\narms:\n  B: {kind: decoder, init: {seed: recipe}, steps: 20, disable_regularizers: false}\n")
r = subprocess.run(["uv", "run", "python", "scripts/task23_overfit.py", "--config", cfg, "--device", dev, "--deterministic",
                    "--run-name", out.split("/")[-1] + "_detprobe"], capture_output=True, text=True)
print("  deterministic algorithms on", dev, ":", "supported" if r.returncode == 0 else "NOT supported: " + r.stderr.strip().splitlines()[-1][:200])
sys.exit(r.returncode)
EOF

if [ -z "${LOCAL:-}" ]; then
    stamp "--- 3/3 TASK-22 cache path on $DEVICE (the real pair) ---"
    cat > "$OUT/cache_probe.yaml" <<'EOF'
base_config: configs/task23_smoke.yaml
pair: {crop_manifest: data/manifests/task20_crops.csv, track_manifest: data/manifests/task20_tracks.csv,
       crop_id: "rock.00009:train_pool:pool:18", feature_source: task22_cache, task22_cache_dir: data/cache/task22,
       feature_match_atol: 0.1, zoom_frames: 48, harmonic_max_bin: 64}
arms:
  A: {kind: free_tensor, learning_rate: 0.01, init_noise_std: 0.01, init_seed: 0, steps: 1000, stop_at_criterion: true}
  B: {kind: decoder, init: {seed: recipe}, steps: 100, disable_regularizers: false}
EOF
    RUN --config "$OUT/cache_probe.yaml" --device "$DEVICE" --run-name "${TAG}_cache"
    uv run python - "artifacts/task23_smoke/${TAG}_cache/pair/pair_meta.json" <<'EOF'
import json, sys
f = json.load(open(sys.argv[1]))["features"]
print("  cache:", f["task22_cache"], "| index check:", f["index_check"])
print("  fresh vs TASK-22 cache max |diff|:", f["fresh_vs_task22_cache_max_abs_diff"], "| mel equal:", f["task22_cache_mel_equal_recomputed"],
      "| wave equal:", f["task22_cache_wave_equal_recomputed"])
assert f["task22_cache_mel_equal_recomputed"] and f["task22_cache_wave_equal_recomputed"]
EOF
fi
stamp "=== TASK-23 smoke PASSED ($OUT) ==="
