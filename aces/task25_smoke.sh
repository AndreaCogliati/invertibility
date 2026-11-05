#!/bin/bash
# TASK-25 bounded smoke test (every arm, both architecture changes). On ACES, on a PVC node from the repo root:
#   srun --partition=pvc --gres=gpu:pvc:1 --ntasks=1 --cpus-per-task=8 --mem=32G --time=00:45:00 --exclude=ac079 --pty bash
#   module load WebProxy uv && bash aces/task25_smoke.sh
# Locally:  DEVICE=mps LOCAL=1 bash aces/task25_smoke.sh
# Needs runs/task24/task24_v1/{ladder_pairs.pt,init_state.pt} and runs/task23/task23_v1/{pair.pt,B/summary.json}.
# Writes only under runs/task25_smoke and artifacts/task25_smoke; W&B disabled (CLAUDE.md rule 11). Steps:
#   0. unit tests
#   1. bit-exact resume (CPU, deterministic): D and F stopped mid-run and resumed == uninterrupted
#   2. all six arms on the accelerator (4,096 presentations each), report, plots, audio; measured throughput
set -euo pipefail
cd "$(dirname "$0")/.."
DEVICE="${DEVICE:-xpu}"
if [ -z "${LOCAL:-}" ]; then
    [ -n "${SLURM_JOB_ID:-}" ] || { echo "ERROR: run inside a SLURM allocation (or set LOCAL=1 off-cluster)" >&2; exit 1; }
    export UV_PROJECT="$PWD/envs/xpu"
fi
TAG="t25_${SLURM_JOB_ID:-local}_$(date -u +%Y%m%dT%H%M%SZ)"
OUT="artifacts/task25_smoke/$TAG"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/smoke.log") 2>&1
T0=$(date +%s); stamp() { echo "[$(( $(date +%s) - T0 )) s] $*"; }
echo "=== TASK-25 smoke $TAG on $DEVICE | commit $(git rev-parse HEAD) dirty=$(test -n "$(git status --porcelain)" && echo yes || echo no) ==="
R() { uv run python -u scripts/task25_ablation.py "$@" --config configs/task25_smoke.yaml --wandb-mode disabled; }
OK3() { local s=0; "$@" || s=$?; [ $s -eq 3 ] || { echo "expected exit 3, got $s" >&2; exit 1; }; }
ls runs/task24/task24_v1/ladder_pairs.pt runs/task24/task24_v1/init_state.pt runs/task23/task23_v1/pair.pt > /dev/null

stamp "--- 0/2 unit tests ---"
uv run pytest -q tests/test_task25.py

stamp "--- 1/2 bit-exact resume (CPU, deterministic; D and F) ---"
R prepare --device cpu --run-name "${TAG}_cpu_ref" > "$OUT/cpu.log" 2>&1
R train --device cpu --deterministic --run-name "${TAG}_cpu_ref" --arms D,F >> "$OUT/cpu.log" 2>&1
R prepare --device cpu --run-name "${TAG}_cpu_res" >> "$OUT/cpu.log" 2>&1
OK3 R train --device cpu --deterministic --run-name "${TAG}_cpu_res" --arms D,F --stop-after D:100 >> "$OUT/cpu.log" 2>&1
OK3 R train --device cpu --deterministic --run-name "${TAG}_cpu_res" --arms D,F --stop-after F:150 >> "$OUT/cpu.log" 2>&1
R train --device cpu --deterministic --run-name "${TAG}_cpu_res" --arms D,F >> "$OUT/cpu.log" 2>&1
uv run python - "runs/task25_smoke/${TAG}_cpu_ref" "runs/task25_smoke/${TAG}_cpu_res" <<'EOF'
import json, sys
a, b = sys.argv[1:]
for arm in ("D", "F"):
    sa, sb = (json.load(open(f"{d}/{arm}/summary.json")) for d in (a, b))
    assert sa["final_sha256"] == sb["final_sha256"] and sa["presentation_counts_min_max"] == sb["presentation_counts_min_max"], arm
    strip = lambda p: [{k: v for k, v in json.loads(x).items() if k != "wall_seconds"} for x in open(p)]
    for f in ("fit_eval", "monitor_eval"):
        assert strip(f"{a}/{arm}/{f}.jsonl") == strip(f"{b}/{arm}/{f}.jsonl"), (arm, f)
print("  RESUME CHECK PASS: D (stopped at update 100) and F (stopped at 150) resumed == uninterrupted")
EOF

stamp "--- 2/2 all arms on $DEVICE ---"
R all --device "$DEVICE" --run-name "${TAG}_${DEVICE}"
uv run python - "artifacts/task25_smoke/${TAG}_${DEVICE}/report.json" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
full = {"A": 819200, "B": 819200, "C": 51200, "D": 51200, "E": 51200, "F": 51200}
tot = 0.0
for a, x in r["arms"].items():
    est = full[a] / x["updates_per_second_median"] / 3600
    tot += est
    print(f"  {a}: {x['parameters']:,} params, {x['updates_per_second_median']:.0f} updates/s -> full arm about {est:.2f} h")
print(f"  full run (sequential, smoke throughput incl. its denser evaluation): about {tot:.1f} h -> about {8 * (tot + 0.2):.0f} billing-hours at 8/h")
EOF
stamp "=== TASK-25 smoke PASSED ($OUT) ==="
