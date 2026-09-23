# NEXT

## Current gate

Gate 5 full-GTZAN pilot -- BlueHive package **prepared and locally verified, NOT submitted**
(`tasks/TASK-14.md`). TASK-13 found Gate 4's pooled PASS was a memorization artifact (no
held-out conditioning separation on 4 held-out crops); TASK-14 builds the infrastructure to test
whether full-GTZAN scale (792 training tracks) resolves that, per this task's explicit "do not
run the full job locally / do not submit on BlueHive unless instructed" constraints.

**What exists now, ready to submit:**

- `configs/gate5_full_gtzan.yaml` (300,000 steps, full 792-track training split, random-crop
  sampling) and `configs/gate5_smoke.yaml` (bounded diagnostic).
- `scripts/train_gate5_full_gtzan_wandb.py` -- new training script: `TrainCropSampler`
  (deterministic, resumable, broad random-crop sampling across all train tracks), held-out
  evaluation reusing TASK-13's `gate4_encodec_common.py` unchanged, primary endpoint
  `oracle_relative_waveform_recovery`, outcome classification (A-F), paired-delta statistics.
- `data/manifests/gate5_heldout_crops.csv` (188 val + 226 test crops, fixed/versioned).
- `bluehive/run_task14.slurm` (full job, `--resume auto` always on), `run_task14_diagnostic.slurm`
  (optional BlueHive-side bounded check), `resubmit_task14.sh`.
- **Verified locally** (M4 Max/MPS): fresh smoke run, online-W&B-mode smoke run, and -- most
  importantly -- a genuine partial-train-to-step-10-then-`--resume auto`-to-step-20 run that
  reproduced bit-identical final metrics to an uninterrupted run. Checkpoint/resume mechanism
  (model/optimizer/sampler/RNG/W&B-run-lineage) confirmed genuinely correct, not just non-crashing.
- 24 new tests (`tests/test_gate5_common.py`), full suite 215 passing, lint clean.
- Full preparation report: `artifacts/task14_gate5_full_gtzan_prep/TASK-14_PREPARATION_REPORT.md`.

**Not done, by explicit instruction:** the full 300,000-step training job has not been run,
locally or on BlueHive. No Gate 5 scientific result exists yet.

**Known placeholder**: the full run's 16-hour SLURM wall-clock request is extrapolated from the
local diagnostic's measured per-step rate (~116ms/step including on-the-fly MuQ+EnCodec
encoding), not a real BlueHive measurement. Correct after the first submission.

**Also this session**: fixed a real gap in `wandb_logging.init_run` -- every prior resumed run
(TASK-13's included, though TASK-13 never needed resume in practice) would have silently created
a disconnected new W&B run instead of continuing the same one. Added a backward-compatible
`resume_id` parameter; verified the fix reopens the same run across the resume test above.

## Exact last successful commands

```bash
uv run python scripts/build_gate5_manifests.py                                    # gate5_heldout_crops.csv
uv run pytest tests/test_gate5_common.py -q                                       # 24 passed
uv run python scripts/train_gate5_full_gtzan_wandb.py --config configs/gate5_smoke.yaml --wandb-mode disabled
uv run python scripts/train_gate5_full_gtzan_wandb.py --config configs/gate5_smoke.yaml --wandb-mode online
# partial-then-resume verification (temp configs, steps=10 then steps=20 --resume auto):
uv run python scripts/train_gate5_full_gtzan_wandb.py --config <steps=10 copy> --wandb-mode disabled
uv run python scripts/train_gate5_full_gtzan_wandb.py --config <steps=20 copy> --wandb-mode disabled --resume auto
uv run ruff check .; uv run pytest -q   # 215 passed, lint clean
bash -n bluehive/run_task14.slurm bluehive/run_task14_diagnostic.slurm bluehive/resubmit_task14.sh
```

2026-09-23, macOS, M4 Max, device=mps. All diagnostic runs completed in a few seconds each; no
BlueHive access in this environment (same limitation as TASK-03).

## Run/checkpoint path

- `data/manifests/gtzan_manifest_smoke.csv` (full GTZAN split, sha256
  `0dedd1efa83441ce929a46530c95d8dff6ba3d67d99fbdde2038177a5d4f095e`).
- `data/manifests/gate5_heldout_crops.csv` (sha256
  `719b38a6f9046778ffa4facd451423e283d8b09dfd9cf2788f9983e7d3abbd91`).
- `runs/task14_gate5_smoke/checkpoints/{latest,best,step10,step15,step20}.pt` (git-ignored,
  diagnostic only -- NOT the full run).
- `artifacts/task14_gate5_smoke/report.json` + 12 listening WAVs (git-ignored, diagnostic only).
- `artifacts/task14_gate5_full_gtzan_prep/TASK-14_PREPARATION_REPORT.md` (git-ignored).
- Two W&B runs from the diagnostic (group `task14-gate5-smoke`, project `muq-invertibility`):
  `2xozcddv` (disabled mode) and `nhws11b2` (online mode, real).
- **Full run paths (not yet populated)**: `runs/task14_gate5_full_gtzan/checkpoints/`,
  `artifacts/task14_gate5_full_gtzan/report.json`, W&B group `task14-gate5-full-gtzan`.

## Next command

**Blocked on explicit user instruction to submit the full BlueHive job** -- per this task's own
constraint, TASK-14 prepares and validates only. The next command, when authorized:

```bash
sbatch bluehive/run_task14.slurm
```

Optionally first: `sbatch bluehive/run_task14_diagnostic.slurm` (an extra CUDA-specific check
beyond this task's MPS-only local verification -- not required, since TASK-13's own BlueHive
submissions already validated the CUDA determinism fix this script also relies on).

Do not submit either without the user's explicit go-ahead.

## Blockers

1. **Full BlueHive submission requires explicit authorization** -- not a technical blocker, a
   process one per this task's own instruction.
2. **BlueHive wall-clock estimate unverified** -- 16-hour budget is extrapolated, not measured;
   should be corrected after the first real submission (see `run_task14.slurm`'s own comment).
3. **Decision needed from the user** (carried over from TASK-13): pooled vs. held-out-only Gate 4
   criteria -- still relevant context for interpreting whatever Gate 5 eventually shows.
4. **Listening verdicts pending**: TASK-14's 12 smoke-test WAVs (not scientific evidence, but
   still unreviewed), plus the long-standing backlog: TASK-13's, TASK-12's 80k/320k, TASK-11's,
   TASK-08's, TASK-10's, TASK-01's, TASK-02's.
5. **BlueHive job not yet run** for TASK-03 either -- see [[task03-bluehive-wandb]], still pending.
6. The still-pending conditioning-collapse diagnostic from `experiment/mel-hifigan-sanity` --
   untested, orthogonal.

## Expected verification criteria for Gate 5

Defined in `configs/gate5_full_gtzan.yaml` and `scripts/train_gate5_full_gtzan_wandb.py`'s
outcome-classification logic (Outcomes A-F, `tasks/TASK-14.md`'s own rubric): Gate 5 passes only
if correct conditioning materially outperforms all three controls on BOTH held-out validation and
held-out test data (paired mean delta > 0.05 in `oracle_relative_waveform_recovery` AND >50% of
paired examples positive, for each control, each split), with no variance collapse. Not yet
reached -- the full run has not been executed.

## Repository state at the end of this task (per TASK-14's explicit reporting requirement)

At task start: commit `a74003d` on `main`, clean except `tasks/TASK-14.md` (untracked).

At the end of this task:

```text
On branch main
Your branch is up to date with 'origin/main'.

Changes not staged for commit:
    modified:   bluehive/README.md
    modified:   src/muq_inversion/wandb_logging.py
    modified:   notes/DECISIONS.md
    modified:   notes/MEMORY.md
    modified:   notes/NEXT.md        (this file, being written)

Untracked files:
    bluehive/resubmit_task14.sh
    bluehive/run_task14.slurm
    bluehive/run_task14_diagnostic.slurm
    configs/gate5_full_gtzan.yaml
    configs/gate5_smoke.yaml
    data/manifests/gate5_heldout_crops.csv
    scripts/build_gate5_manifests.py
    scripts/train_gate5_full_gtzan_wandb.py
    src/muq_inversion/gate5_common.py
    tasks/TASK-14.md
    tests/test_gate5_common.py
```

**Not committed**, per this task's explicit "do not commit unless explicitly instructed" rule.
`artifacts/task14_gate5_smoke/`, `artifacts/task14_gate5_full_gtzan_prep/`, and
`runs/task14_gate5_smoke/` are git-ignored and do not appear in `git status`. TASK-13's own files
were not modified.

## Notes for whoever resumes this

- **Read `artifacts/task14_gate5_full_gtzan_prep/TASK-14_PREPARATION_REPORT.md` first** for the
  full required-structure preparation report, then
  [[DECISIONS.md#task14-gate5-full-gtzan-pilot]] for the project-notes-style summary.
- **This task built and verified infrastructure only -- there is no Gate 5 result yet.** Don't
  cite anything from `artifacts/task14_gate5_smoke/` as scientific evidence; it's a 20-step/
  4-track diagnostic, explicitly labeled as such.
- **`oracle_relative_waveform_recovery` is the primary endpoint metric for Gate 5**, distinct from
  TASK-13's `waveform_correlation_to_oracle` (translator-vs-oracle, not anchored to the original)
  -- don't conflate the two when comparing results across tasks.
- To actually run Gate 5: get explicit authorization, then `sbatch bluehive/run_task14.slurm`
  (after confirming `data/raw/gtzan/` has all 999 valid tracks on the BlueHive filesystem -- the
  job checks this and fails fast if not, per `bluehive/README.md`'s TASK-14 addendum).
- If a submitted job is preempted/times out, `bluehive/resubmit_task14.sh` (or just `sbatch
  bluehive/run_task14.slurm` again, since `--resume auto` is always on) continues the same run
  and the same W&B run ID -- verified correct by this task's local partial-then-resume test.
