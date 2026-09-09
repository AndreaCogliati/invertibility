# NEXT

## Current gate

Still investigating Gate 3 (Tiny-Set Overfit) -- explicitly not proceeding to Gate 4/5. This
branch (`experiment/muq-to-dac-temporal-conv`, off `main`) is the most promising result in the
whole investigation so far: giving the MuQ-to-DAC-latent predictor just ~5 frames of local
temporal context closed almost the entire gap the framewise MLP hit.

**Conclusions reached this session (full reasoning in
[[DECISIONS.md#muq-to-dac-temporal-conv]]):**

1. Replacing the framewise MLP's pointwise layers with 3 residual `Conv1D(k=5)` layers (receptive
   field 5 MuQ frames, ~200ms, verified both analytically and empirically) raised cosine
   similarity from 0.64 (the MLP's exact, verified ceiling) to 0.9999-1.0000 -- essentially a full
   latent-space solve, not merely an improvement.
2. The decoded waveform now correlates with the DAC-oracle reconstruction itself (not just the
   reference) at 0.9999/1.0000 -- decoding the predicted latent is essentially indistinguishable
   from decoding the true DAC latent.
3. This directly answers the experiment's critical question: **the framewise MLP's ceiling was a
   limitation of its memoryless architecture, not of MuQ's representation.** MuQ contained the
   missing information all along; it just required minimal local temporal context to access.
4. This is the first result in the entire Gate 3 investigation where a latent-space metric
   improvement is this large and unambiguous -- pending the user's listening confirmation
   (per [[muq-to-dac-mlp]]'s own finding that waveform correlation alone is an unreliable
   perceptual proxy in this project, so listening confirmation matters more here than usual, not
   less).

## Exact last successful commands

```bash
uv run python scripts/train_muq_to_dac_temporal_conv.py   # 3000 steps, cosine sim 0.9999, PASSED
uv run ruff check .    # all checks passed
uv run pytest -q       # 47 passed (MuQ baseline confirmed untouched)
```

2026-09-09, macOS, M4 Max, device=mps.

## Run/checkpoint path

- `runs/muq_to_dac_temporal_conv/checkpoints/{latest,step_003000}.pt` (git-ignored). Intermediate
  numbered checkpoints deleted after the run.
- `artifacts/muq_to_dac_temporal_conv/` holds `{track}_{reference,temporal_conv_reconstruction,
  mlp_reconstruction,dac_oracle,difference}.wav` for both clips -- `mlp_reconstruction` reuses the
  sibling experiment's trained checkpoint (not retrained) for a direct, self-contained comparison.
  All 8 relevant WAVs already sent to the user via SendUserFile.
- `runs/muq_to_dac_mlp/checkpoints/latest.pt` (git-ignored, from the sibling
  `experiment/muq-to-dac-mlp` branch) -- reused directly by this branch's training script; do not
  delete it while this branch is active.

## Next command

**Listening verdict received and CONFIRMED (2026-09-09):** "The listening test shows almost
perfect reconstruction! We nailed it!" This is the first unambiguous positive result in the
entire Gate 3 investigation. Tagged `muq-to-dac-temporal-conv-v1`. Next steps (none started, none
greenlit -- awaiting user direction):

- Test whether this generalizes beyond the 2-clip tiny-overfit setting -- the current result could
  still be an overfitting artifact specific to memorizing 2 fixed clips' local dynamics, not
  evidence of genuine generalizable MuQ->DAC predictability. Per the task's explicit instruction,
  do not proceed to larger models until this was confirmed and analyzed -- it now has been, so a
  broader validation (more clips, held-out clips) is the natural next gate-appropriate step.
- If the direction continues to hold, this project's primary scientific question ("can a decoder
  recover recognizable audio from MuQ's representation?") may be better served by the MuQ -> small
  temporal model -> DAC latent -> frozen DAC decoder path than by the HiFi-GAN-family path this
  investigation started with -- worth discussing with the user before committing further compute
  to either direction, and worth revisiting whether to formally supersede the HiFi-GAN-family
  Gate 3 baseline with this approach.

Do not start either without the user's go-ahead. Do not perform further HiFi-GAN, EnCodec,
ground-truth-mel, BigVGAN experiments or proceed to Gate 4/5 without explicit instruction (this
branch's task scope, like every sibling branch, was narrowly defined).

## Blockers

1. **Decision needed from the user**: generalize this result beyond the 2-clip overfit setting
   (held-out clips, more data), formally adopt this path over the HiFi-GAN-family one, or pursue
   something else.
2. The still-pending conditioning-collapse diagnostic from `experiment/mel-hifigan-sanity` (see
   that branch's DECISIONS.md) remains untested and is orthogonal to this branch's finding.
3. The `shuffled_time` finding from Gate 4 (see [[DECISIONS.md#gate4-shuffled-time-finding]]) --
   still dormant.

## Expected verification criteria for Gate 5

Not yet reached. Per CLAUDE.md: train on the committed GTZAN split, save periodic fixed
validation reconstructions, report objective metrics and qualitative caveats -- blocked behind
Gate 3/4 resolution above.

## Notes for whoever resumes this

- **Read [[DECISIONS.md#muq-to-dac-temporal-conv]] first** -- it's the strongest positive result
  in the entire investigation and directly resolves the open question left by
  [[muq-to-dac-mlp]].
- Six experiment branches now exist, each branched independently off `main`, none merged: `main`
  (capacity investigation), `encodec-hifigan-experiment`, `experiment/mel-hifigan-sanity`,
  `experiment/reference-hifigan`, `experiment/muq-to-dac-mlp`, `experiment/muq-to-dac-temporal-conv`.
  Each has its own isolated notes/DECISIONS.md entries; this file reflects only the branches
  merged into whichever branch is currently checked out.
- This branch's training script (`scripts/train_muq_to_dac_temporal_conv.py`) directly loads
  `runs/muq_to_dac_mlp/checkpoints/latest.pt` from the sibling branch's git-ignored `runs/`
  directory for its comparison audio -- this only works if that directory hasn't been deleted.
  If it's missing, the script still runs but skips the MLP comparison WAV (prints a warning).
- The receptive-field-verification pattern in `train_muq_to_dac_temporal_conv.py`
  (`_measure_receptive_field_empirically`: perturb one frame, check which outputs change) is a
  reusable diagnostic for any future temporal model in this project -- don't just trust a kernel-
  size arithmetic formula for receptive field claims.
- Adam betas `(0.9, 0.999)` (PyTorch's default) were used here without issue, same as
  [[muq-to-dac-mlp]] -- no adversarial training, so the GAN-collapse-driven `(0.8, 0.99)` fix from
  the HiFi-GAN branches doesn't apply.
