# NEXT

## Current gate

Gate 4 (Short Pilot) for the MuQ-to-DAC temporal-conv pipeline (`experiment/muq-to-dac-minimal-
gate4`, branched off `experiment/muq-to-dac-temporal-conv`). **Result: the Gate 3 near-perfect
reconstruction does NOT generalize -- it was memorization of the two specific training crops, not
a transferable MuQ-to-DAC mapping.** Do not proceed to Gate 5. Do not increase model complexity
without further direction (per this experiment's own explicit instruction).

**Conclusions reached this session (full reasoning in
[[DECISIONS.md#muq-to-dac-minimal-gate4]]):**

1. Retrained the exact Gate 3 architecture (unchanged) on two generalization tests: Experiment A
   (unseen crops from the same 2 tracks) and Experiment B (unseen tracks, 8 train / 2 val / 2
   test, 4 genres). Both show a textbook overfitting curve: validation MSE increases
   monotonically from the very start of training, and the required validation-MSE checkpoint
   selection picks step 200 of 8000 in both experiments.
2. Held-out waveform correlation to the DAC oracle collapses to ~0 in both experiments (from a
   real 0.25-0.33 on training data) -- and Experiment B's unseen-track performance is no worse
   than Experiment A's unseen-crop-same-track performance. This is an all-or-nothing collapse,
   not a graded "same track helps a little" effect.
3. Shuffled-across/time-shifted controls are clearly beaten by correct conditioning on training
   data but are statistically indistinguishable from it on held-out data -- the model's real
   conditioning-sensitivity exists only for memorized examples.
4. This is a scientifically successful negative result, exactly per this experiment's own success
   criterion: it cleanly measured the generalization gap and answered the central question.

## Exact last successful commands

```bash
uv run python scripts/build_gate4_manifests.py     # writes data/manifests/gate4_crops.csv
uv run python scripts/train_muq_to_dac_gate4.py    # both experiments, 8000 steps each, PASSED (as a measurement)
uv run ruff check .    # all checks passed
uv run pytest -q       # 47 passed (MuQ baseline confirmed untouched)
```

2026-09-09, macOS, M4 Max, device=mps.

## Run/checkpoint path

- `data/manifests/gate4_crops.csv` (tracked in git) -- the exact, checksummed split definition
  for both experiments: track IDs, genres, paths, SHA-256 checksums, split, crop index, crop
  start/duration in seconds.
- `runs/gate4_minimal/{A,B}/checkpoints/best.pt` (git-ignored) -- the validation-MSE-selected
  checkpoint per experiment (step 200 in both cases), not the final-step model.
- `artifacts/gate4_minimal/{A,B}/` -- 30 listening WAVs total (2 experiments x 3 splits x 5
  conditions: reference, dac_oracle, temporal_conv, shuffled_across, time_shifted), descriptive
  filenames (`{split}_{track}_t{crop_start}s_{condition}_gate4_minimal_step200.wav`). A 12-file
  representative subset already sent to the user via SendUserFile.
- `artifacts/gate4_minimal/report.json` -- full metrics (MSE, cosine similarity, waveform
  correlation, multi-res STFT distance, for all 3 conditions x 3 splits x 2 experiments), plus
  parameter count, receptive field, training time, peak MPS memory, checkpoint selection info.

## Next command

**Listening verdict received and CONFIRMED (2026-09-09):** "The listening confirms the poor
result. The model is not learning and generalizing." Metrics and listening agree cleanly --
Gate 3's near-perfect result is confirmed to have been memorization. Next steps (none started,
none greenlit) depend on user direction:

- If the user wants to pursue closing the generalization gap: candidates include more/more
  diverse training data (the current 16-crop training sets are still tiny), regularization
  (weight decay, dropout), or a fundamentally different training regime -- none of these should
  be started without explicit direction, since this experiment's own instructions say not to
  increase model complexity in response to this result.
- If the user wants to treat this as the documented stopping point for the MuQ-to-DAC line of
  investigation, the natural write-up is a synthesis comparing it against the HiFi-GAN-family
  Gate 3 investigation's own findings (neither approach has yet produced both perceptual quality
  AND generalization in the same experiment).

Do not start either without the user's go-ahead. Do not proceed to Gate 5. Do not increase model
complexity. Do not perform further HiFi-GAN, EnCodec, ground-truth-mel, BigVGAN experiments
without explicit instruction.

## Blockers

1. **Decision needed from the user**: pursue closing the generalization gap, or treat this as a
   documented stopping point for this line of investigation.
2. The still-pending conditioning-collapse diagnostic from `experiment/mel-hifigan-sanity` (see
   that branch's DECISIONS.md) remains untested and is orthogonal to this branch's finding.
3. The `shuffled_time` finding from Gate 4 (HiFi-GAN-family, see
   [[DECISIONS.md#gate4-shuffled-time-finding]]) -- still dormant, and now has an interesting
   echo in this branch's own finding that shuffled/shifted controls underperform correct
   conditioning only on training data, not held-out data.

## Expected verification criteria for Gate 5

Not yet reached, and explicitly not to be started per this experiment's own instructions given
the generalization-gap finding above.

## Notes for whoever resumes this

- **Read [[DECISIONS.md#muq-to-dac-minimal-gate4]] first** -- it directly qualifies Gate 3's
  positive result: near-perfect reconstruction was memorization, not evidence of a learned,
  transferable MuQ-to-DAC mapping.
- This branch sits on top of `experiment/muq-to-dac-temporal-conv` (not `main`) -- it's the first
  branch in this investigation to extend a sibling branch directly rather than starting fresh off
  `main`, since it's a direct continuation of that branch's validated model, not an independent
  ablation.
- `scripts/build_gate4_manifests.py` must be run before `scripts/train_muq_to_dac_gate4.py` if
  `data/manifests/gate4_crops.csv` is ever regenerated -- it's committed to git, so this is only
  needed if the split definition changes.
- The tiny-subset sanity check (`run_tiny_subset_sanity_check` in
  `scripts/train_muq_to_dac_gate4.py`) mirrors Gate 3 exactly and is a genuinely useful reusable
  pattern for any future experiment built on this pipeline: verify the *pipeline* still works on
  a trivial case before trusting a harder experiment's slower convergence as a sign of trouble.
- Checkpoint selection uses validation MSE only, evaluated periodically during training (not by
  reloading checkpoints from disk) -- see `compute_mse` in `scripts/train_muq_to_dac_gate4.py`.
- Adam betas `(0.9, 0.999)` (PyTorch's default), same as every prior MuQ-to-DAC experiment -- no
  adversarial training in this line of work, so the GAN-collapse-driven `(0.8, 0.99)` fix from
  the HiFi-GAN branches doesn't apply.
