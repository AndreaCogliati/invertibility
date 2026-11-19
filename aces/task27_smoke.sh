#!/bin/bash
# TASK-27 bounded smoke test. On ACES, on a PVC node from the repo root:
#   srun --partition=pvc --gres=gpu:pvc:1 --ntasks=1 --cpus-per-task=8 --mem=32G --time=00:45:00 --exclude=ac079 --pty bash
#   module load WebProxy uv && bash aces/task27_smoke.sh
# Locally (stand-in start checkpoint, clearly labeled): DEVICE=mps LOCAL=1 bash aces/task27_smoke.sh
# Needs runs/task24/task24_v1/ladder_pairs.pt, runs/task23/task23_v1/pair.pt and the TASK-25 C final checkpoint
# (runs/task25/task25_v1/C/checkpoints/final.pt on ACES). Writes only under runs/task27_smoke and artifacts/task27_smoke;
# runs/task27_resume, artifacts/task27_resume; W&B disabled. Steps:
#   0. unit tests
#   1. bit-exact resume (CPU, deterministic; configs/task27_resume*.yaml, 64 updates): A and B stopped mid-run and resumed == uninterrupted
#   2. both arms on the accelerator (4,096 presentations), calibration, report, listening package; measured throughput
set -euo pipefail
cd "$(dirname "$0")/.."
DEVICE="${DEVICE:-xpu}"
CONFIG=configs/task27_smoke.yaml
RCONFIG=configs/task27_resume.yaml
if [ -z "${LOCAL:-}" ]; then
    [ -n "${SLURM_JOB_ID:-}" ] || { echo "ERROR: run inside a SLURM allocation (or set LOCAL=1 off-cluster)" >&2; exit 1; }
    export UV_PROJECT="$PWD/envs/xpu"
    export HF_HUB_OFFLINE=1
else
    CONFIG=configs/task27_smoke_local.yaml
    RCONFIG=configs/task27_resume_local.yaml
fi
TAG="t27_${SLURM_JOB_ID:-local}_$(date -u +%Y%m%dT%H%M%SZ)"
OUT="artifacts/task27_smoke/$TAG"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/smoke.log") 2>&1
T0=$(date +%s); stamp() { echo "[$(( $(date +%s) - T0 )) s] $*"; }
echo "=== TASK-27 smoke $TAG on $DEVICE ($CONFIG) | commit $(git rev-parse HEAD) dirty=$(test -n "$(git status --porcelain)" && echo yes || echo no) ==="
R() { uv run python -u scripts/task27_adversarial.py "$@" --config "$CONFIG" --wandb-mode disabled; }
RR() { uv run python -u scripts/task27_adversarial.py "$@" --config "$RCONFIG" --wandb-mode disabled; }
OK3() { local s=0; "$@" || s=$?; [ $s -eq 3 ] || { echo "expected exit 3, got $s" >&2; exit 1; }; }
ls runs/task24/task24_v1/ladder_pairs.pt runs/task23/task23_v1/pair.pt > /dev/null

stamp "--- 0/2 unit tests ---"
uv run pytest -q tests/test_task27.py

stamp "--- 1/2 bit-exact resume (CPU, deterministic; A and B) ---"
for RUN in cpu_ref cpu_res; do
    RR prepare --device cpu --run-name "${TAG}_$RUN" >> "$OUT/cpu.log" 2>&1
    RR calibrate --device cpu --deterministic --run-name "${TAG}_$RUN" >> "$OUT/cpu.log" 2>&1
done
RR train --device cpu --deterministic --run-name "${TAG}_cpu_ref" >> "$OUT/cpu.log" 2>&1
OK3 RR train --device cpu --deterministic --run-name "${TAG}_cpu_res" --stop-after A:30 >> "$OUT/cpu.log" 2>&1
OK3 RR train --device cpu --deterministic --run-name "${TAG}_cpu_res" --stop-after B:40 >> "$OUT/cpu.log" 2>&1
RR train --device cpu --deterministic --run-name "${TAG}_cpu_res" >> "$OUT/cpu.log" 2>&1
uv run python - "runs/task27_resume/${TAG}_cpu_ref" "runs/task27_resume/${TAG}_cpu_res" <<'PY'
import json, sys
a, b = sys.argv[1:]
T = ("wall_seconds", "updates_per_second", "presentations_per_second")
clean = lambda r: {k: clean(v) if isinstance(v, dict) else v for k, v in r.items() if k not in T}
assert json.load(open(f"{a}/calibration.json"))["lambda"] == json.load(open(f"{b}/calibration.json"))["lambda"], "calibration"
for arm in ("A", "B"):
    sa, sb = (json.load(open(f"{d}/{arm}/summary.json")) for d in (a, b))
    for k in ("final_sha256", "discriminator_final_sha256", "order_sha256", "presentation_counts_min_max"):
        assert sa[k] == sb[k], (arm, k)
    for f in ("fit_eval", "monitor_eval", "train_log"):
        ra, rb = ([json.dumps(clean(json.loads(x)), sort_keys=True) for x in open(f"{d}/{arm}/{f}.jsonl")] for d in (a, b))
        assert ra == rb, (arm, f)
print("  RESUME CHECK PASS: A (stopped at 30) and B (stopped at 40) resumed == uninterrupted; calibration identical")
PY

stamp "--- 2/2 both arms on $DEVICE ---"
R all --device "$DEVICE" --run-name "${TAG}_${DEVICE}"
uv run python - "artifacts/task27_smoke/${TAG}_${DEVICE}/report.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
full = 51200
tot = 0.0
for a, x in r["arms"].items():
    est = full / x["updates_per_second_median"] / 3600
    tot += est
    print(f"  {a}: {x['updates_per_second_median']:.0f} generator updates/s -> full arm about {est:.2f} h (+ evaluation)")
print(f"  lambda {r['calibration']['lambda']:.4g}; start {r['prepared']['start']['checkpoint']} standin={r['prepared']['start']['standin']}")
print(f"  full run (sequential): about {tot:.2f} h of training + about 0.3 h of setup, evaluation and report -> about {8 * (tot + 0.3):.0f} billing-hours at 8/h")
PY
stamp "=== TASK-27 smoke PASSED ($OUT) ==="
