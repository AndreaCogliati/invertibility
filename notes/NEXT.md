# NEXT

## Current gate

Gate 4 revalidation on the merged architecture (`tasks/TASK-13.md`) -- re-ran CLAUDE.md's Gate 4
checkpoint/resume + four-way conditioning-control protocol on `main`'s actual pipeline after
`experiment/muq-layer-sweep` merged in (commit `a96fd4b`): MuQ `last_hidden_state` ->
repeat_interleave -> `SpectrogramToDacTranslator` -> frozen EnCodec latent -> frozen EnCodec
decoder. One model trained 20,000 steps on `data/manifests/gate4_crops.csv` Experiment B (16
train / 2 val / 2 test track-disjoint crops), then evaluated under `correct`/`shuffled_across`/
`shuffled_time`/`constant_zero` conditioning against that single checkpoint.

**Addendum (2026-09-23): confirmed reproducible on BlueHive.** A first BlueHive submission
(Tesla T4) failed checkpoint/resume equivalence (CUDA kernel nondeterminism, not an RNG bug --
see [[DECISIONS.md#task13-gate4-encodec-revalidation]]'s addendum); fixed in
`scripts/train_gate4_encodec_wandb.py` (commit `895b81b`) and confirmed on a second BlueHive
submission (NVIDIA A40, job `1956502`): resume equivalence now passes exactly, and the
conditioning-control numbers are consistent with the MPS run (correct 0.7843 vs. 0.7857 pooled
mean waveform correlation). `bluehive/run_task13.slurm` is now a validated, working path for
this script.

**Result: PASSED by the pooled decision rule, but this is a memorization artifact, not evidence
of generalized conditioning use.** Pooled across all 20 crops, correct conditioning (0.786 mean
waveform correlation to the EnCodec oracle) beats every control by far more than the 0.05-absolute
margin (shuffled_across -0.008, shuffled_time 0.005, constant_zero 0.001); resume equivalence
exact (0.0000 relative diff both halves); variance ratio 1.080 (no collapse). But split by data
split: train (n=16) correct=0.981 vs. controls ~0 (clean separation, resolving the old
`shuffled_time` ambiguity -- but only on memorized data); held-out val+test (n=4) correct=
0.002-0.007 vs. controls -0.024-to-0.028 -- **no separation at all on held-out crops**. This
reproduces [[muq-to-dac-minimal-gate4]]'s earlier "Gate 3 doesn't generalize, it was
memorization" finding, now confirmed on the EnCodec-target pipeline too. Full reasoning in
`artifacts/task13_gate4_encodec_revalidation/TASK-13_RESULTS.md`, summarized in
[[DECISIONS.md#task13-gate4-encodec-revalidation]]. Per this task's own instruction, **Gate 5 was
not started.**

**Conclusions reached this session:**

1. Merged `experiment/muq-layer-sweep` into `main` (commit `a96fd4b`, zero merge conflicts,
   already pushed to `origin/main`) per the user's explicit instruction, after first committing
   the CLAUDE.md pipeline-shift edit on both branches (`74e2fde` on the experiment branch,
   `2e3fd55` on `main`, identical content).
2. Built the merged-architecture Gate 4 harness (`gate4_encodec_common.py`,
   `train_gate4_encodec_wandb.py`) reusing every existing piece unmodified (`SpectrogramToDacTranslator`,
   `evaluate_prediction`, `aggregate_metrics_across_examples`, `find_best_lag_and_align`-style
   patterns) -- only the MuQ-input + repeat-interleave alignment and the four conditioning-control
   constructions are new, unit-tested (14 tests) before running anything real.
3. **The merged architecture's Gate 4 "pass" is real on training data but does not extend to
   held-out data** -- the single most important, decision-relevant finding this session. Treat any
   future "Gate 4 passed" claim on this pipeline as pooled-metric-only unless explicitly qualified
   by a held-out breakdown.
4. `shuffled_time` (a genuine random frame permutation, not the older fixed circular shift) now
   separates cleanly from correct conditioning on training data -- the prior ambiguity
   ([[gate4-shuffled-time-finding]]) is resolved in the "the decoder does use temporal MuQ
   structure" direction, but only for memorized examples, which meaningfully changes what that
   resolution is worth.

## Exact last successful commands

```bash
git checkout main && git merge --no-commit --no-ff experiment/muq-layer-sweep   # 0 conflicts
git commit -m "Merge experiment/muq-layer-sweep into main"                      # a96fd4b
uv run pytest tests/test_gate4_encodec_common.py -q                             # 14 passed
uv run python scripts/train_gate4_encodec_wandb.py --steps 20 --wandb-mode disabled  # smoke test
uv run python -u scripts/train_gate4_encodec_wandb.py --wandb-mode online       # real run, 20k steps
uv run ruff check scripts/train_gate4_encodec_wandb.py src/muq_inversion/gate4_encodec_common.py tests/test_gate4_encodec_common.py
```

2026-09-23, macOS, M4 Max, device=mps. Launched via `nohup ... &` + `disown`, tracked by a
persistent `Monitor`. Full run (main 20k-step training + resume-check pre/post halves + 4-way
control eval) completed in well under 10 minutes.

## Run/checkpoint path

- `data/manifests/gate4_crops.csv` (tracked, reused unchanged, Experiment B; manifest hash
  `0f9884a3bf451f66451db612da1b7eec880687f71ca59af8062cc9eca5d85ddb`).
- `runs/task13_gate4_encodec_revalidation/checkpoints/{resume_test_checkpoint,final}.pt`
  (git-ignored).
- `artifacts/task13_gate4_encodec_revalidation/report.json` -- full per-condition, per-split,
  per-example metrics; `.../*.wav` -- 12 listening WAVs (test split: original, EnCodec oracle, 4
  conditioning reconstructions x 2 tracks).
- `artifacts/task13_gate4_encodec_revalidation/TASK-13_RESULTS.md` -- the full report in this
  task's own required structure (Executive Summary through Final Git Status).
- One W&B run, group `task13-gate4-encodec-revalidation`: `wandb.ai/
  andrea-cogliati-university-of-rochester/muq-invertibility/runs/gwj2rn8d`.

## Next command

**Blocked on user direction.** Candidate directions that would need the user's go-ahead:

- Decide the Gate 4 decision-rule question this task surfaced but did not resolve: should
  "materially outperforms" be computed on held-out crops only (which would report Gate 4 as NOT
  passed), rather than pooled across train+val+test (which reports PASSED)? This governs whether
  Gate 5 should even be considered next.
- If held-out generalization is the actual bar, consider whether a larger Gate 4 manifest (more
  tracks) would give genuine held-out conditioning sensitivity, or whether (per TASK-01/02's
  data-scaling findings on the DAC pipeline) more data alone is unlikely to help without an
  architecture/objective change -- untested for this pipeline specifically.
- A human listening pass on the 12 TASK-13 test-split WAVs -- not yet done.
- Extending TASK-12's EnCodec compute-response curve further (640k, 1.28M steps) -- still not
  authorized.

Do not start any of these without the user's go-ahead.

## Blockers

1. **Decision needed from the user**: pooled vs. held-out-only Gate 4 criteria (this task's own
   central open question -- see above).
2. **Listening verdicts pending** for TASK-13's 12 WAVs, and the long-standing backlog: TASK-12's
   80k/320k packages, TASK-11's, TASK-08's, TASK-10's, TASK-01's, TASK-02's.
3. **BlueHive job not yet run** -- see [[task03-bluehive-wandb]], still pending.
4. The still-pending conditioning-collapse diagnostic from `experiment/mel-hifigan-sanity` --
   untested, orthogonal to this branch's findings.
5. Listening verdicts for the 80-track scaling experiment, the full-GTZAN scaling experiment, and
   the MuQ layer-sweep were never given -- not blocking, worth noting if any comes up again.

## Expected verification criteria for Gate 5

Not yet reached, and not authorized to start. Per TASK-13's own finding, any future Gate 5
authorization should specify whether pooled or held-out-only conditioning separation is the bar,
given this session's evidence that the two give opposite answers on this pipeline.

## Repository state at the end of this task (per TASK-13's explicit reporting requirement)

At task start: commit `a96fd4b` on `main` (the just-completed merge), clean tree except
`tasks/TASK-13.md` (untracked).

At the end of this task:

```text
On branch main
Your branch is up to date with 'origin/main'.

Changes not staged for commit:
    modified:   notes/DECISIONS.md
    modified:   notes/MEMORY.md
    modified:   notes/NEXT.md        (this file, being written)

Untracked files:
    scripts/train_gate4_encodec_wandb.py
    src/muq_inversion/gate4_encodec_common.py
    tasks/TASK-13.md
    tests/test_gate4_encodec_common.py
```

**Not committed**, per this task's explicit "do not commit unless explicitly instructed" rule.
`artifacts/task13_gate4_encodec_revalidation/` and `runs/task13_gate4_encodec_revalidation/` are
git-ignored and do not appear in `git status`.

## Notes for whoever resumes this

- **Read `artifacts/task13_gate4_encodec_revalidation/TASK-13_RESULTS.md` first** for the full
  required-structure report, then [[DECISIONS.md#task13-gate4-encodec-revalidation]] for the
  project-notes-style summary.
- **The pooled Gate 4 "PASSED" verdict this script prints is not the whole story** -- always check
  the per-split breakdown (`report.json`'s `control_report.<mode>.aggregate_by_split`) before
  citing this result; the held-out numbers tell a materially different story than the pooled ones.
- `scripts/train_gate4_encodec_wandb.py --steps N` overrides the 20,000-step default for smoke
  tests; the real run's config/checkpoints all reflect `steps=20000`.
- `gate4_encodec_common.build_conditioning`'s `shuffled_time` is a true random permutation
  (`torch.randperm`), not the older `gate4_common.shifted_time_control`'s fixed circular shift --
  don't conflate the two when comparing to pre-merge Gate 4 results.
