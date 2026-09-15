# NEXT

## Current gate

Spectrogram Gate-3 memorization control (`tasks/TASK-04.md`). **Result: Strong Success --
Spectrogram->DAC memorizes two fixed training excerpts near-perfectly, matching or exceeding the
successful MuQ Gate-3 control on every latent and audio metric -- confirmed by listening, not
just by metric.** User's verdict (2026-09-15): "the two audio excerpts are already great at 1000
steps. Almost undistinguishable from the final version or the original. The early excerpts show
signs of artifacts, though." Full reasoning in [[DECISIONS.md#task04-spectrogram-gate3]]. Per
this task's own explicit "stop after answering these questions" instruction, do not launch a
larger experiment or new target formulation without direction.

**Conclusions reached this session (full reasoning in
[[DECISIONS.md#task04-spectrogram-gate3]]):**

1. Reused TASK-02's dense cache directly for the two exact excerpts (`pop.00012`, `disco.00049`
   at `t=14.00s`, 2 samples/83us off the MuQ script's own exact center crop) -- no new cache
   built. Reused `SpectrogramToDacTranslator`, the MSE+0.1*cosine loss, and the MuQ Gate-3
   script's exact optimizer/step-budget/presentation-schedule unchanged.
2. **All 11 pre-training integrity checks passed** (cache determinism, spectrogram/DAC-target
   recomputation from source audio, frame alignment, gradient reach to all three submodules,
   frozen DAC, checkpoint save/reload reproducibility). Run twice (disabled-mode dry run, then
   the real online W&B run) with byte-identical results.
3. **Training converged to near-perfect memorization**: cosine 0.15 -> 0.9998 over 3000 steps,
   best checkpoint (step 2536) reaching MSE 0.000547, cosine 0.99999, norm ratio 0.9999,
   variance ratio 0.9998, waveform correlation 0.99999 -- matching or exceeding the MuQ Gate-3
   control's own 0.001936 / 0.99996 / 1.0038 / 1.0073 / 0.99994.
4. **Unplanned discovery while building the required MuQ comparison**: the trained MuQ Gate-3
   checkpoint is extremely sample-alignment-sensitive (a 2-sample/83us crop offset collapses its
   cosine from 0.99998 to 0.577) -- its "successful" memorization is a narrow, sample-exact
   overfit, not a phase-tolerant fit. Corrected the comparison to evaluate MuQ on its own exact
   crop before trusting any number.
5. **This decisively rules out data alignment, spectrogram preprocessing, the 128->1024
   projection, translator implementation, target extraction, loss computation, gradient flow,
   checkpoint restoration, and DAC decoding as explanations for TASK-01/02's GTZAN-scale
   failure** -- every one of those is exercised identically here and works. The failure is
   generalization/training-regime-scale specific, not a basic pipeline defect.
6. **Collapse never emerged at this scale**: lag-1 autocorrelation tracked the oracle's own
   ~0.053 throughout training, vs. ~0.95 at GTZAN scale in TASK-01/02 -- reinforces
   [[task03-bluehive-wandb]]'s regression-to-the-mean finding: collapse looks like a property of
   the *multi-example* training regime (where matching the dataset mean is a cheap MSE shortcut
   unavailable when there's effectively one target per step to fit exactly), not of the
   architecture or loss in isolation.

## Exact last successful commands

```bash
uv run python -u scripts/train_spectrogram_gate3_wandb.py --wandb-mode disabled   # dry run, PASSED, then fixed a crop-offset bug in the MuQ comparison
uv run python -u scripts/train_spectrogram_gate3_wandb.py --wandb-mode disabled   # rerun after fix, PASSED, byte-identical training curve
uv run python -u scripts/train_spectrogram_gate3_wandb.py --wandb-mode online     # the real run, PASSED, W&B run 7kfiziym
uv run ruff check .    # all checks passed
uv run pytest -q       # 117 passed (103 baseline + 14 new gate3-spectrogram tests)
```

2026-09-15, macOS, M4 Max, device=mps. Launched via `nohup ... &` + `disown` (unbuffered `-u`),
tracked by real PID and `Monitor`.

## Run/checkpoint path

- `data/manifests/gate3_spectrogram_two_excerpt.csv` (tracked) -- the two-excerpt manifest.
- `src/muq_inversion/gate3_spectrogram_common.py` -- pure, tested helpers (metrics, gradient
  breakdown, example loading, W&B payload construction).
- `runs/task04_spectrogram_gate3/checkpoints/{step0,best_by_mse,best_by_cosine,best_by_wav_corr,
  final,periodic}.pt` (git-ignored) -- all reloaded and reproducibility-verified after saving.
- `artifacts/task04_spectrogram_gate3/report.json` -- full per-checkpoint metrics, per-example
  breakdown, and the MuQ Gate-3 comparison (computed on MuQ's own exact crop).
- `artifacts/task04_spectrogram_gate3/*.wav` (14 files, git-ignored) -- reference/dac_oracle/
  step0/early/intermediate/final/best_by_mse reconstructions for both tracks.
- Live W&B run: `wandb.ai/andrea-cogliati-university-of-rochester/muq-invertibility/runs/
  7kfiziym` (project `muq-invertibility`, group `spectrogram-gate3`, job_type
  `memorization-control`).

## Next command

**Blocked on user direction**, per this task's own explicit "stop after answering these
questions" instruction. With TASK-01 (representation), TASK-02 (data coverage), and TASK-04
(basic pipeline capability) all now addressed, and TASK-03 (BlueHive reproduction) partially
handed off, the live remaining hypotheses for this investigation's GTZAN-scale saturation are:

- **The training/objective formulation at multi-example scale** -- most directly evidence-backed
  now: TASK-03's regression-to-the-mean finding plus TASK-04's "collapse doesn't happen with one
  example, does happen with many" contrast both point here. A loss/target reformulation that
  doesn't reward the dataset-mean solution as cheaply (per-frame relative error, whitened
  latents, something residual-VQ-aware) is the most pointed next experiment in this whole
  investigation to date.
- **The still-untested architecture/receptive-field-capacity axis** -- lower priority now, since
  TASK-04 shows the existing architecture is not itself incapable of fitting the target.
- **The pending BlueHive run** (see [[task03-bluehive-wandb]]) -- independent track, still needs
  the user to execute `bluehive/run_task03.slurm` and report back.

Do not start any of these without the user's go-ahead.

## Blockers

1. **Decision needed from the user** on whether to pursue the loss/target-reformulation direction
   now most directly pointed to by both TASK-03 and TASK-04's findings.
2. **BlueHive job not yet run** -- still pending, see [[task03-bluehive-wandb]].
3. **Listening verdicts for TASK-01's and TASK-02's 30-WAV packages** -- still pending. TASK-04's
   own listening verdict is now in hand and confirms the metrics (see Current gate above).
4. The still-pending conditioning-collapse diagnostic from `experiment/mel-hifigan-sanity` --
   untested, orthogonal to this branch's findings.
5. The `shuffled_time` finding from Gate 4 (see [[DECISIONS.md#gate4-shuffled-time-finding]]) --
   still dormant.
6. Listening verdicts for the 80-track scaling experiment, the full-GTZAN scaling experiment, and
   the MuQ layer-sweep were never given -- not blocking, worth noting if any comes up again.

## Expected verification criteria for Gate 5

Not yet reached, and not authorized to start.

## Notes for whoever resumes this

- **Read [[DECISIONS.md#task04-spectrogram-gate3]] first**, alongside
  [[DECISIONS.md#task03-bluehive-wandb]] -- together they narrow this investigation's remaining
  hypothesis space to essentially one axis: the training/objective formulation at multi-example
  scale. Representation (TASK-01), data coverage (TASK-02), and basic pipeline capability
  (TASK-04) are now all ruled out as the primary bottleneck.
- The MuQ Gate-3 checkpoint's extreme sub-hop sample-alignment sensitivity (TASK-04's unplanned
  finding) is worth remembering for any future experiment that reuses `runs/
  muq_to_dac_temporal_conv/checkpoints/latest.pt` -- always evaluate it on the crop it was
  actually trained on (exact center-crop formula in
  `scripts/train_muq_to_dac_temporal_conv.py`), never a nearby one.
- `src/muq_inversion/gate3_spectrogram_common.py` (metrics, gradient breakdown, example loading)
  and `wandb_logging.py` (from TASK-03) are both generic enough to reuse for a future loss-
  reformulation experiment without rewriting instrumentation from scratch.
- The regression-to-the-mean / collapse-only-at-scale story (TASK-03 + TASK-04 combined) is now
  this investigation's leading, well-evidenced explanation for the entire GTZAN-scale saturation
  pattern seen since [[muq-to-dac-scale-data]] -- treat it as the working hypothesis for any
  future experiment design in this line, not as one competing theory among several.
