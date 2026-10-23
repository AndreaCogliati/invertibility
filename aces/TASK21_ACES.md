# TASK-21: extended N512 continuation (specification, ACES runbook, reporting template)

**Question.**
- Did TASK-20 stop during a recoverable transient?
- Does much longer training improve the current translator?

This is a duration experiment only: same data, model, loss, optimizer, learning rate, batch,
precision and sampling as TASK-20.

## Specification (`configs/task21.yaml` inherits `configs/task20.yaml`)

**Starting point.** TASK-20's last checkpoint, `latest.pt` at step 4,250, not its best.

- **Restored:**
  - model weights;
  - Adam state, moments and step count (4,250 = global step), not reset;
  - sampler state (RNG, track queue, exposure counts);
  - Python, NumPy and torch-CPU RNG states;
  - global step, best-so-far, validation history and lineage.
- **Not saved by TASK-20, and not needed:**
  - XPU RNG state (no dropout; batches come from the sampler's Python RNG);
  - LR scheduler (constant LR);
  - grad scaler (FP32).
- **Incumbent best.** TASK-20's `best.pt` (step 2,250, val mel L1 0.4753) is copied in as the
  incumbent and replaced only by a lower primary-val mel L1.
- **Startup check.** The loaded model must reproduce TASK-20's logged step-4,250 primary-val mel L1
  within 1e-3, or the run stops.

**Stopping.**
- 48 h of cumulative job runtime across segments, counting start-up.
- No patience, no step ceiling.
- Stops early only on a non-finite loss or gradient norm, or an execution failure.

**Validation and checkpoints.**
- Primary val, correct-conditioning mel L1, same crops and metric. It selects `best.pt`.
- Validation every 250 steps through step 20,000, then every 1,000. No test, test2 or val2
  during training.
- Retention:
  - `best.pt`;
  - `latest.pt` (every 5 min, and on graceful stop, budget, or new best);
  - `final.pt`;
  - 3 rolling `recovery_*.pt` (every 30 min);
  - milestones at 10k, 20k, 50k, 100k, 200k, 500k, 1M, 2M, 5M, 10M and 20M steps.

**Logging.**
- New W&B run, group `task21-n512-extended`, with the parent (`task20_N512_v1`, W&B `524aw8gs`,
  checkpoint sha256s) in its config. TASK-20's W&B run is not touched.
- At every validation: train L1 (window mean and max), grad-norm window max, LR, steps/s,
  cumulative runtime, global step, best step, track cycles, pool passes, sampler exposure, and
  whether this is the first validation after a resume.
- Events (continuation start and check, resumes, graceful stops, budget) are kept in the checkpoint.
- Every W&B log call is mirrored to `runs/task21/<run>/metrics.jsonl`. A W&B failure never stops
  training: logging falls back to offline, or is skipped with a message.

**Code changes.** All config-gated; TASK-18, 19 and 20 configs behave as before.
- `scripts/train_task18_heldout.py`:
  - `init_from_checkpoint` and `--init-from`;
  - `runtime_budget_hours` with `steps: null`;
  - `eval_schedule` and `log_schedule`;
  - `checkpoint_policy`;
  - `train_only`;
  - non-finite gradient stop;
  - diagnostics, robust W&B and the JSONL mirror;
  - continuation events in the report.
- New:
  - `src/muq_inversion/task21.py`;
  - `scripts/task21_check_run.py`, `scripts/task21_analyze.py`;
  - `tests/test_task21.py`;
  - `aces/task21_smoke.sh`, `aces/run_task21_segment.slurm`, `aces/run_task21_eval.slurm`,
    `aces/submit_task21_chain.sh`.

## Verified budget estimate (before submission)

- **Billing rate, verified:** `sacct` for TASK-19 (32 GB) and TASK-20 (64 GB) jobs shows
  `billing=8` for 8 CPUs + 1 PVC; memory did not change it.
- **Training:** 48 h × 8 = **384 billing-hours**.
- **Overhead:**
  - pre-Python per segment: `uv sync` and the manifest check, ~2–3 min;
  - no-op segments: ~3 min each;
  - evaluation job: ~0.5 h, about 4;
  - interactive smoke: ~0.5–1 h, 4–8.
- **Total ≈ 395–400 billing-hours.**
- Whether ACES SUs equal billing-hours is not verified. Check the charge rate with the ACES
  accounting tool before submitting.

**Expected scale.** At TASK-20's measured ~94 steps/s (~110/s once validation thins out), 48 h is
about 16–19 million optimizer steps:
- about 2 million draws per training track;
- about 80,000–90,000 passes over the 12,288-crop pool;
- about 4,000× TASK-20's training.

## ACES runbook

### 0. Check the walltime limit on a login node (required)

```bash
sinfo -p pvc -o "%P %l %D %G"                    # %l = partition time limit
scontrol show partition pvc | grep -iE "MaxTime|DefaultTime|QoS"
sacctmgr -P show qos format=Name,MaxWall,MaxTRESPU
```

Number of segments: `N = ceil(48 h / (walltime − 0.25 h)) + 1`. The +1 is a spare: if training is
already done, a segment exits within minutes. Examples:
- 48:00:00 limit → `N = 3`;
- 24:00:00 → `N = 4`;
- 72:00:00 → `N = 2`.

### 1. Prepare

```bash
cd /home/u.ac370321/project/invertibility && git pull
module load WebProxy uv && export UV_PROJECT=$PWD/envs/xpu && uv sync --frozen
uv run python scripts/build_task20_manifests.py                        # 3 x "unchanged"
ls runs/task20/task20_N512_v1/checkpoints/{latest,best}.pt            # continuation source
```

### 2. Smoke test (interactive PVC node; bounded, separate output directories)

```bash
srun --partition=pvc --gres=gpu:pvc:1 --ntasks=1 --cpus-per-task=8 --mem=32G --time=01:00:00 --pty bash
module load WebProxy uv && bash aces/task21_smoke.sh                   # send back SMOKE_REPORT.md
```

### 3. Submit (only when instructed)

```bash
bash aces/submit_task21_chain.sh <N> <WALLTIME>                        # e.g. 3 48:00:00; prints job IDs
squeue -u $USER                                                        # one segment runs at a time
```

To extend the chain later, add segments after the last job: `bash aces/submit_task21_chain.sh 1 <WALLTIME> <last-jobid>`.

### 4. Evaluate (after the last segment reports "runtime budget already reached" or exit 0)

```bash
sbatch aces/run_task21_eval.slurm
for j in <all TASK-21 job ids>; do sacct -j $j -X --format=JobID,JobName,State,Elapsed,AllocTRES%60; done
```

Copy back `artifacts/task21/`, `runs/task21/*/run_manifest.json`,
`runs/task21/task21_N512ext_v1/metrics.jsonl` and `aces/logs/`. Then:

```bash
uv run python scripts/task21_analyze.py --t21 artifacts/task21/task21_N512ext_v1_eval --sacct-hours <sum of Elapsed>
```

## Reporting template (`artifacts/task21/analysis_v1/TASK-21_RESULTS.md`)

1. **Run:**
   - job IDs per segment;
   - commit (clean?);
   - walltime limit used;
   - cumulative runtime;
   - `sacct` elapsed total;
   - billing-hours (and SUs, if reported).
2. **Continuation check:** step-4,250 val reproduced (value vs TASK-20), and the startup state.
3. **Duration:**
   - best step;
   - final step;
   - optimizer steps added;
   - draws per track, track cycles and pool passes.
4. **Selection:** did the best move from step 2,250? Give its val mel L1 and the improvement over
   TASK-20 best, with a paired track-clustered CI.
5. **Held-out tracks** for TASK-21 best, TASK-21 final, TASK-20 best, A128 and A32, each set
   separately: val, test, val2, test2. The test sets are reused evaluation sets, not untouched
   confirmation sets.
6. **Unseen crops of training tracks and memorized crops**, the same arms.
7. **Curves:** the full run (log step axis) and the early region. Mark the TASK-20 best, any new
   best, spikes and resumes.
8. **Spikes:**
   - each one's coincidence with train-loss or gradient-norm changes;
   - track-cycle position;
   - resumes or other events.

   No cause is assumed.
9. **Interpretation limits:**
   - CIs do not measure seed variability.
   - A null result means only that extended training with this unchanged recipe did not improve
     the selected result. It is not a MuQ information ceiling or exhausted translator capacity.
