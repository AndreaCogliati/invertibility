#!/bin/bash
# TASK-24 bounded smoke test. On ACES, interactively on a PVC node from the repo root:
#   srun --partition=pvc --gres=gpu:pvc:1 --ntasks=1 --cpus-per-task=8 --mem=64G --time=01:00:00 --exclude=ac079 --pty bash
#   module load WebProxy uv && bash aces/task24_smoke.sh
# Locally (no SLURM; step 4 skipped, it needs data/cache/task22):  DEVICE=mps LOCAL=1 bash aces/task24_smoke.sh
# Writes only under runs/task24_smoke and artifacts/task24_smoke. Steps (stops at the first failure):
#   0. unit tests; the committed ladder manifest is regenerated identically
#   1. bit-exact resume (CPU, deterministic algorithms): N=4 stopped at update 250 and resumed == uninterrupted
#   2. accelerator ladder smoke (N = 1, 2, 4; TASK-22 smoke cache + TASK-23 anchor): all stages, plots, audio, report
#   3. lock: a second writer on a running arm is refused
#   4. ACES only: the REAL prepare (272 crops from data/cache/task22, anchor bit-exact, cross-check) into the smoke
#      run root, then 3,000 updates of N=256 to measure throughput including whole-set evaluation
set -euo pipefail
cd "$(dirname "$0")/.."
DEVICE="${DEVICE:-xpu}"
if [ -z "${LOCAL:-}" ]; then
    [ -n "${SLURM_JOB_ID:-}" ] || { echo "ERROR: run inside a SLURM allocation (or set LOCAL=1 off-cluster)" >&2; exit 1; }
    export UV_PROJECT="$PWD/envs/xpu"
    export HF_HOME="${HF_HOME:-/scratch/group/p.ele260080.000/hf_cache}"
fi
TAG="t24_${SLURM_JOB_ID:-local}_$(date -u +%Y%m%dT%H%M%SZ)"
OUT="artifacts/task24_smoke/$TAG"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/smoke.log") 2>&1
T0=$(date +%s); stamp() { echo "[$(( $(date +%s) - T0 )) s] $*"; }
echo "=== TASK-24 smoke $TAG on $DEVICE | commit $(git rev-parse HEAD) dirty=$(test -n "$(git status --porcelain)" && echo yes || echo no) ==="
L() { uv run python -u scripts/task24_ladder.py "$@"; }
OK3() { local s=0; "$@" || s=$?; [ $s -eq 3 ] || { echo "expected exit 3, got $s" >&2; exit 1; }; }
ls runs/task23/task23_v1/pair.pt runs/task23/task23_v1/B/checkpoints/milestone_000000.pt > /dev/null
# Small-ladder cache: the TASK-22 smoke cache if present (local), else the real TASK-22 cache (ACES), in which
# case the anchor's cache row must also equal TASK-23's pair bit for bit.
SMOKE_CFG=configs/task24_smoke.yaml
if ! ls data/cache/task22_smoke/task18_v1_L1_*.pt > /dev/null 2>&1; then
    ls data/cache/task22/task18_v1_L1_*.pt > /dev/null || { echo "ERROR: no TASK-22 first-block cache (smoke or full)" >&2; exit 1; }
    SMOKE_CFG="$OUT/task24_smoke_fullcache.yaml"
    sed -e 's#task22_cache_dir: data/cache/task22_smoke#task22_cache_dir: data/cache/task22#' \
        -e 's#anchor_must_be_in_cache: false#anchor_must_be_in_cache: true#' \
        -e 's#restrict_to_cache: true#restrict_to_cache: false#' configs/task24_smoke.yaml > "$SMOKE_CFG"
    grep -q "task22_cache_dir: data/cache/task22$" "$SMOKE_CFG" && grep -q "anchor_must_be_in_cache: true" "$SMOKE_CFG" \
        && grep -q "restrict_to_cache: false" "$SMOKE_CFG" || { echo "ERROR: could not derive the full-cache smoke config" >&2; exit 1; }
fi
echo "small-ladder config: $SMOKE_CFG"

stamp "--- 0/4 unit tests, manifest ---"
uv run pytest -q tests/test_task24.py
L manifest --config configs/task24.yaml --run-name unused | grep "verified (identical)"

stamp "--- 1/4 bit-exact resume (CPU, deterministic) ---"
L all --config "$SMOKE_CFG" --device cpu --deterministic --run-name "${TAG}_cpu_ref" > "$OUT/cpu_ref.log" 2>&1
OK3 L all --config "$SMOKE_CFG" --device cpu --deterministic --run-name "${TAG}_cpu_res" --stop-after 4:250 > "$OUT/cpu_res.log" 2>&1
L all --config "$SMOKE_CFG" --device cpu --deterministic --run-name "${TAG}_cpu_res" >> "$OUT/cpu_res.log" 2>&1
uv run python - "runs/task24_smoke/${TAG}_cpu_ref" "runs/task24_smoke/${TAG}_cpu_res" <<'EOF'
import json, sys
a, b = sys.argv[1:]
for n in ("N0001", "N0002", "N0004"):
    sa, sb = (json.load(open(f"{d}/{n}/summary.json")) for d in (a, b))
    assert sa["final_sha256"] == sb["final_sha256"] and sa["presentation_counts"] == sb["presentation_counts"], n
    for f in ("fit_eval", "monitor_eval"):
        assert open(f"{a}/{n}/{f}.jsonl").read() == open(f"{b}/{n}/{f}.jsonl").read(), (n, f)
print("  RESUME CHECK PASS: N=4 interrupted at 250 + resumed == uninterrupted (weights, sampler counts, eval logs); N=1/N=2 identical")
EOF

stamp "--- 2/4 accelerator ladder smoke ($DEVICE) ---"
L all --config "$SMOKE_CFG" --device "$DEVICE" --run-name "${TAG}_${DEVICE}"
A="artifacts/task24_smoke/${TAG}_${DEVICE}"
ls "$A"/{fit_l1_vs_steps,fit_l1_vs_presentations,monitor_l1_vs_steps,per_crop_distribution_vs_n,anchor_l1_vs_n}.png "$A"/TASK-24_RESULTS.md > /dev/null
ls "$A"/N0004/r0_best_spectrogram.png "$A"/N0004/r3_final_zoom_transient.png "$A"/audio/r0_*_ground_truth_mel_bigvgan.wav > /dev/null
uv run python - "$A/report.json" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
for n, s in r["summaries"].items():
    print(f"  N={n}: final fit L1 {s['final']['l1']:.4f} anchor {s['final']['anchor_l1']:.4f} | {s['updates_per_second_median']:.0f} updates/s | spikes {len(s['spikes'])}")
EOF

stamp "--- 3/4 one writer per arm ---"
mkdir -p "runs/task24_smoke/${TAG}_${DEVICE}_lock/N0001"
echo '{"pid": 0, "note": "simulated running writer"}' > "runs/task24_smoke/${TAG}_${DEVICE}_lock/N0001/LOCK"
cp "runs/task24_smoke/${TAG}_${DEVICE}"/{ladder_pairs.pt,init_state.pt,run_manifest.json} "runs/task24_smoke/${TAG}_${DEVICE}_lock/"
L train --config "$SMOKE_CFG" --device "$DEVICE" --run-name "${TAG}_${DEVICE}_lock" --arms 1 > "$OUT/lock.log" 2>&1 || true
if grep "another writer" "$OUT/lock.log" > /dev/null && [ ! -e "runs/task24_smoke/${TAG}_${DEVICE}_lock/N0001/summary.json" ]; then
    echo "  LOCK CHECK PASS: second writer refused"
else
    echo "ERROR: a second writer was not refused" >&2; exit 1
fi

if [ -z "${LOCAL:-}" ]; then
    stamp "--- 4/4 real prepare (data/cache/task22) + N=256 throughput ---"
    cat > "$OUT/real_probe.yaml" <<'EOF'
base_config: configs/task24.yaml
training: {steps: 3000, lr_drops: [[50000, 0.1], [150000, 0.1]], milestones: [1000, 3000], exposures: [100, 500], log_every: 100,
           fit_eval_every: 1000, monitor_eval_every: 1000, latest_every: 1000, eval_chunk: 64, spike_factor: 2.0, recovery_factor: 1.1}
run_root: runs/task24_smoke
artifact_root: artifacts/task24_smoke
EOF
    L prepare --config "$OUT/real_probe.yaml" --device "$DEVICE" --run-name "${TAG}_real"
    L train --config "$OUT/real_probe.yaml" --device "$DEVICE" --run-name "${TAG}_real" --arms 256
    uv run python - "runs/task24_smoke/${TAG}_real" <<'EOF'
import json, sys
d = sys.argv[1]
m = json.load(open(f"artifacts/task24_smoke/{d.split('/')[-1]}/ladder_meta.json"))
print("  anchor:", m["fit_crops"][0]["source"])
print("  cross-check (rel RMS, mel, wave):", [(round(c["features_rel_rms_diff"], 5), c["mel_max_abs_diff"], c["wave_max_abs_diff"]) for c in m["cross_check"]])
print("  cache:", m["cache"]["path"], m["cache"]["bytes"] // 2**20, "MiB, sha", (m["cache"]["sha256"] or "")[:16], "| MuQ frozen:", m["muq_frozen"])
s = json.load(open(f"{d}/N0256/summary.json"))
ups = 3000 / s["runtime_seconds"]
print(f"  N=256: 3,000 updates in {s['runtime_seconds']:.0f} s incl. 3 whole-set + monitoring evaluations -> {ups:.0f} updates/s")
total_updates = 5 * 200000
print(f"  estimate for the full ladder (5 arms x 200,000 updates): {total_updates / ups / 3600:.2f} h of training "
      f"-> about {total_updates / ups / 3600 + 0.3:.1f} h with prepare/report, {8 * (total_updates / ups / 3600 + 0.3):.0f} billing-hours at 8/h")
EOF
fi
stamp "=== TASK-24 smoke PASSED ($OUT) ==="
