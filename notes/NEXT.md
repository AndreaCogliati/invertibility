# NEXT

## Current gate

Further data-scaling experiment for the MuQ-to-DAC temporal-conv pipeline
(`experiment/muq-to-dac-more-data`, branched off `experiment/muq-to-dac-scale-data`).
**Result: the scaling curve shows diminishing returns / saturation, not continued improvement.**
Do not proceed to transformers, layer sweeps, larger models, or Gate 5 (per this experiment's own
explicit instruction) without further direction.

**Conclusions reached this session (full reasoning in
[[DECISIONS.md#muq-to-dac-more-data]]):**

1. Trained the exact Gate 3/4 architecture (unchanged) on 792 train / 94 val / 113 test
   track-disjoint GTZAN tracks (reusing Gate 1's own committed full-dataset split), 1584/188/226
   crops -- ~5x more crops and ~10x more tracks than the 80-track scaling experiment.
2. Held-out cosine similarity barely improved (0.222/0.223 -> 0.230/0.226, ~4-5% relative) versus
   the previous, similarly-sized scaling step's ~35% relative gain -- a decelerating curve.
3. Waveform correlation to the DAC oracle did not improve at all across any of the three data
   scales tested (16, 320, or 1584 training crops) -- it has stayed at the same noise floor (~0)
   throughout this entire line of investigation.
4. Training was qualitatively new: a turbulent, unstable early phase (steps ~5000-32000, large
   but always-finite loss spikes), then a clean late improvement phase. The validation-MSE-
   selected checkpoint moved to step 50000 of 60000 (83.3%) -- immediate overfitting no longer
   occurs at this scale, but this did not translate into better ultimate generalization.
5. **Final recommendation, from the observed 3-point scaling curve (not subjective expectations):
   further pure data scaling within this exact architecture is unlikely to be productive.** The
   flat waveform-correlation trend across a ~100x crop-count range (16 -> 1584) is the signature
   of saturation (architecture/receptive-field/information ceiling), not a data-limited
   bottleneck. If this line of investigation continues, the next axis to test would be
   architecture (larger receptive field, more capacity) rather than more data -- but that is
   explicitly out of scope for this experiment and requires the user's direction.

## Exact last successful commands

```bash
uv run python scripts/build_more_data_manifest.py       # writes data/manifests/more_data_crops.csv
uv run python scripts/precompute_more_data_cache.py      # writes data/cache/more_data/muq_dac_cache.pt (82.15s)
uv run python scripts/train_muq_to_dac_more_data.py       # 60000 steps, ~15 min total, PASSED (as a measurement)
uv run ruff check .    # all checks passed
uv run pytest -q       # 47 passed (MuQ baseline confirmed untouched)
```

2026-09-09, macOS, M4 Max, device=mps. Launched via `nohup ... &` + `disown`, tracked by real PID
(`kill -0`) and a `Monitor` until-loop -- not the `run_in_background`+`&` combination mistake made
earlier in this investigation (see notes/SCRATCHPAD.md's Gate 3 entry for that lesson).

## Run/checkpoint path

- `data/manifests/more_data_crops.csv` (tracked in git) -- the exact, checksummed 999-track
  (792+94+113), 1998-crop split definition, derived from `data/manifests/gtzan_manifest_smoke.csv`
  (Gate 1's own committed split).
- `data/cache/more_data/muq_dac_cache.pt` (1.5GB, git-ignored) -- precomputed MuQ features and
  DAC latents for all 1998 crops.
- `runs/more_data/checkpoints/best.pt` (60MB, git-ignored) -- the validation-MSE-selected
  checkpoint (step 50000 of 60000).
- `artifacts/more_data/` -- 30 listening WAVs (6 tracks x 5 conditions). A 10-file representative
  subset already sent to the user via SendUserFile.
- `artifacts/more_data/report.json` -- full metrics (MSE, cosine similarity, waveform
  correlation, multi-res STFT distance, for all 3 conditions x 3 splits) plus the full training
  curve (60 points, every 1000 steps), parameter count, receptive field, training/evaluation
  time, peak MPS memory, checkpoint selection info.

## Next command

**Blocked on the user's listening verdict for this experiment** (and, separately, still pending:
a listening verdict for the 80-track scaling experiment, which the user moved past without
confirming -- not blocking, just noted). Once received, the natural next steps (none started,
none greenlit) depend on direction, but per the scaling-curve evidence above, further pure data
scaling is not the recommended next step. Candidates if this line of investigation continues:

- Relax the "no architecture change" constraint that has governed Gate 3 through this experiment:
  test whether a larger receptive field or more capacity (still without transformers/attention,
  if the user wants to stay minimal) closes any of the waveform-correlation gap that data scaling
  alone could not.
- Treat the current 3-point scaling curve as sufficient evidence to conclude this specific
  architecture/approach has a ceiling, and decide whether to pursue a different decoding strategy
  entirely, or revisit the HiFi-GAN-family findings from earlier in this investigation.

Do not start either without the user's go-ahead. Do not proceed to transformers, layer sweeps,
larger models, or Gate 5.

## Blockers

1. **Listening verdict needed from the user** for this experiment -- 10 WAVs already sent via
   SendUserFile (full 30-file set on disk in `artifacts/more_data/`).
2. **Decision needed from the user**: relax the architecture constraint (larger receptive field
   or capacity) as the next axis to test, or treat the current evidence as a stopping point for
   this specific approach.
3. The still-pending conditioning-collapse diagnostic from `experiment/mel-hifigan-sanity` (see
   that branch's DECISIONS.md) remains untested and is orthogonal to this branch's finding.
4. The `shuffled_time` finding from Gate 4 (HiFi-GAN-family, see
   [[DECISIONS.md#gate4-shuffled-time-finding]]) -- still dormant.

## Expected verification criteria for Gate 5

Not yet reached, and explicitly not to be started per this experiment's own instructions.

## Notes for whoever resumes this

- **Read [[DECISIONS.md#muq-to-dac-more-data]] first** -- it completes the 3-point scaling curve
  (Gate 4 -> 80-track -> this experiment) and gives the evidence-based recommendation to stop
  scaling data alone within this architecture.
- This branch sits on top of `experiment/muq-to-dac-scale-data` (not `main`), continuing the
  parent-branch-extension pattern started by Gate 4.
- `data/manifests/gtzan_manifest_smoke.csv` (Gate 1's own committed manifest) is the source of
  truth for this experiment's split -- if it's ever regenerated, `more_data_crops.csv` would need
  rebuilding too (`scripts/build_more_data_manifest.py` re-derives it deterministically).
- **At this data scale, full-training-set evaluation during training becomes expensive enough to
  dominate wall time** -- this script's "train" curve uses a running average over recent steps
  instead (see the module docstring in `scripts/train_muq_to_dac_more_data.py`). Any future
  experiment at this scale or larger should use the same pattern, not Gate 4/the 80-track
  experiment's full-recompute-every-checkpoint approach.
- **The evaluation phase (decode-based metrics across all splits) takes ~8 minutes at this scale**
  -- dominated by re-loading/resampling raw audio per crop plus DAC decode/STFT calls, not by
  `STEPS`. Budget for this when estimating wall time for any larger future run.
- Training at this scale showed a genuinely turbulent early phase (large but finite loss spikes,
  roughly the first half of the 60000-step run) before stabilizing -- worth expecting and not
  panicking over in any future run at this or larger scale, as long as no non-finite values occur.
- Adam betas `(0.9, 0.999)` (PyTorch's default), same as every prior MuQ-to-DAC experiment -- no
  adversarial training in this line of work.
