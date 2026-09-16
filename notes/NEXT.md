# NEXT

## Current gate

Gate-3 training-set-size scaling curve (`tasks/TASK-05.md`). **Result: sharp collapse onset
between N=8 and N=32, then asymptotic decay reaching the GTZAN-scale ceiling by N=512 -- roughly
23-30x fewer examples than either TASK-01 or TASK-02 used. Confirmed by listening, not just by
metric**: user's verdict (2026-09-16) -- "the clips for N=32 are closer to those for N=128 and
N=512 than to those for N=8... the drop between N=8 and N=32 is dramatic." The audible category
boundary matches the metric-located transition exactly; the N=32-to-512 metric decline is
perceptually minor by comparison. Full reasoning in [[DECISIONS.md#task05-gate3-scaling]]. This
is the most direct evidence yet for where and how this investigation's collapse pathology
emerges.

**Conclusions reached this session (full reasoning in
[[DECISIONS.md#task05-gate3-scaling]]):**

1. Built 5 nested, genre-balanced manifests (`gate3_scale_{2,8,32,128,512}.csv`) anchored at
   TASK-04's exact 2 excerpts, all crops already present in TASK-02's dense cache (no new
   precompute). Trained TASK-04's exact pipeline (architecture, loss, optimizer unchanged) with a
   FIXED 20000-step max budget across all 5 levels -- deliberately NOT scaled with N, so
   per-example exposure genuinely decreases as N grows (the variable the experiment needed to
   isolate).
2. **The scaling curve**: N=2/8 near-perfect (norm ratio ~1.0, waveform correlation ~1.0);
   N=32 already substantially collapsed (norm ratio 0.50, waveform correlation 0.47); N=128/512
   continuing to decay toward an asymptote (norm ratio 0.38/0.34, waveform correlation 0.09/0.02).
3. **The N=8-to-N=32 transition is sharp** (every collapse metric moves by more than half its
   total range in one step) -- not a smooth Hypothesis-A gradual decline. But the subsequent
   N=32-to-512 decay is real and continuous, not a Hypothesis-B step-to-final-state either.
4. **By N=512, results are already statistically indistinguishable from TASK-01's (1584 crops)
   and TASK-02's (11873 crops) full GTZAN runs.** The collapse regime this investigation has
   studied since [[muq-to-dac-scale-data]] is fully established by ~512 examples.
5. Interpretation: a capacity/optimization boundary around 8-32 simultaneously-memorizable
   examples, giving way to the regression-to-the-mean mechanism ([[task03-bluehive-wandb]]) once
   individual-example memorization is no longer achievable -- ties together TASK-03's and
   TASK-04's findings into one coherent mechanism story with a measured onset point.

## Exact last successful commands

```bash
uv add matplotlib
uv run python scripts/build_gate3_scale_manifests.py   # 5 manifests, verified nested/genre-balanced
uv run python -u scripts/train_gate3_scale_wandb.py --levels 2 --wandb-mode disabled   # smoke test, PASSED
uv run python -u scripts/train_gate3_scale_wandb.py --wandb-mode online                 # all 5 levels, PASSED, ~53 min total
uv run python scripts/plot_gate3_scale_figures.py       # both required figures
uv run ruff check .; uv run pytest -q                   # 130 passed
```

2026-09-15, macOS, M4 Max, device=mps. Launched via `nohup ... &` + `disown` (unbuffered `-u`),
tracked by real PID and a persistent `Monitor`.

## Run/checkpoint path

- `data/manifests/gate3_scale_{2,8,32,128,512}.csv` (tracked).
- `runs/task05_gate3_scale/n{N}/checkpoints/{best_by_mse,final}.pt` (git-ignored).
- `artifacts/task05_gate3_scale/n{N}/report.json` -- full per-level metrics; `.../n{N}/*.wav` --
  20 listening WAVs (pop.00012/disco.00049 reconstruction + oracle at every level).
- `artifacts/task05_gate3_scale/scaling_summary.json` -- the cross-level summary table.
- `artifacts/task05_gate3_scale/scaling_curve_main.png`,
  `scaling_curve_norm_variance_ratio.png` -- the two required deliverable figures.
- Five live W&B runs, group `gate3-scaling`: `wandb.ai/
  andrea-cogliati-university-of-rochester/muq-invertibility` (`gate3_scale_{2,8,32,128,512}`).

## Next command

**Blocked on user direction.** This task's own scope explicitly excludes optimization/redesign --
diagnosis only. With the collapse onset now localized to N=8-32, the natural next directions
(not yet authorized) are:

- **A dedicated N=16 (and perhaps N=12, N=24) run** to pinpoint the transition more precisely
  within (8, 32] -- cheap, using the exact same pipeline/manifests-generation approach.
- **The loss/target-reformulation direction** flagged after TASK-03/04 is now even more
  specifically motivated: understanding *why* ~8-32 is the ceiling for this optimization loop
  (batch size 1, no scheduler) could point directly at what to change -- e.g. does batching change
  the threshold? Does a learning-rate schedule?
- **Listening verdicts** for the 20 new WAVs would help confirm whether the N=32 "already
  collapsed but not yet asymptotic" regime is audibly distinguishable from the N=512/GTZAN-scale
  ceiling, or perceptually already just as bad.

Do not start any of these without the user's go-ahead.

## Blockers

1. **Decision needed from the user** on which direction to pursue next (pinpointing the exact
   threshold, loss reformulation, or something else).
2. **Listening verdicts** for TASK-01's and TASK-02's WAV packages -- still pending (TASK-04's
   and TASK-05's are now both confirmed by ear).
3. **BlueHive job not yet run** -- see [[task03-bluehive-wandb]], still pending.
4. The still-pending conditioning-collapse diagnostic from `experiment/mel-hifigan-sanity` --
   untested, orthogonal to this branch's findings.
5. The `shuffled_time` finding from Gate 4 (see [[DECISIONS.md#gate4-shuffled-time-finding]]) --
   still dormant.
6. Listening verdicts for the 80-track scaling experiment, the full-GTZAN scaling experiment, and
   the MuQ layer-sweep were never given -- not blocking, worth noting if any comes up again.

## Expected verification criteria for Gate 5

Not yet reached, and not authorized to start.

## Notes for whoever resumes this

- **Read [[DECISIONS.md#task05-gate3-scaling]] first.** It localizes this whole investigation's
  collapse pathology to a specific, small training-set-size range (8-32 examples) for the first
  time -- every earlier experiment either used far more data (TASK-01/02, GTZAN-scale) or far
  less (TASK-04, N=2), never anything in between.
- `src/muq_inversion/gate3_spectrogram_common.py`'s `load_examples_from_manifest` and
  `aggregate_metrics_across_examples` are generic and reusable for any future N-example
  experiment in this line (e.g. a follow-up N=16 pinpointing run) -- no new plumbing needed,
  just a new manifest.
- The fixed-step-budget-across-N design (this task's central methodological choice) is the
  opposite of TASK-02's exposure-preserving convention -- don't default to scaling steps with N
  for any future experiment in this specific line without checking which question is being asked
  (does more data help fit better, vs. does more data cause collapse).
- `scripts/plot_gate3_scale_figures.py` is a small, reusable read-only plotting script (matplotlib,
  new dependency this session) -- rerun it directly against any future
  `scaling_summary.json`-shaped report without modification if the schema doesn't change.
