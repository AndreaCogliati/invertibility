# NEXT

## Current gate

Still investigating Gate 3 (Tiny-Set Overfit) -- explicitly not proceeding to Gate 4/5. This
branch (`experiment/muq-to-dac-mlp`, off `main`) tested a structurally different approach from
every prior sibling branch: latent-to-latent regression (MuQ -> MLP -> DAC latent -> frozen DAC
decoder) instead of direct waveform synthesis via HiFi-GAN.

**Conclusions reached this session (full reasoning in
[[DECISIONS.md#muq-to-dac-mlp]]):**

1. A tiny (8.4M-param), non-adversarial, non-temporal framewise MLP predicts DAC latents from
   MuQ latents and converges in ~150 steps (vs. 15,000 for any HiFi-GAN variant) to cosine
   similarity 0.64.
2. The convergence plateau is an **exact, verified architectural ceiling**, not an optimization
   failure: DAC's measured frame rate (75 Hz) is exactly 3x MuQ's (25 Hz), so the framewise MLP
   is fed 3 identical repeated MuQ frames per output triplet and cannot, being memoryless,
   produce 3 different outputs. The theoretical floor (best constant predictor per MuQ frame)
   matches the trained model's converged MSE almost exactly (11.04/11.54 vs. ~11.05).
3. Downstream waveform correlation via DAC's frozen decoder (0.303/0.192) beats every
   HiFi-GAN-family variant in this investigation except EnCodec's outlier (0.682/0.602) --
   achieved with dramatically less compute, no GAN, and no waveform loss.
4. This is evidence that MuQ contains substantial, usable information for predicting a
   reconstruction-oriented codec's latent space, and that latent-to-latent regression is a
   fundamentally easier problem than waveform synthesis. The residual gap to DAC's own oracle
   ceiling (0.37/0.57, from decoding the *true* DAC latent) is most likely explained by the
   framewise-architecture constraint rather than insufficient MuQ information, though this has
   not been fully separated from the alternative yet.

## Exact last successful commands

```bash
uv run python scripts/train_muq_to_dac_mlp.py   # 3000 steps, converged ~150 steps, PASSED
uv run ruff check .    # all checks passed
uv run pytest -q       # 47 passed (MuQ baseline confirmed untouched)
```

2026-09-09, macOS, M4 Max, device=mps.

## Run/checkpoint path

- `runs/muq_to_dac_mlp/checkpoints/{latest,step_003000}.pt` (git-ignored, 100.7 MB each --
  notably smaller than any HiFi-GAN-family checkpoint). Intermediate numbered checkpoints
  deleted after the run (384MB -> 192MB).
- `artifacts/muq_to_dac_mlp/` holds `{track}_{reference,reconstruction,dac_oracle,difference}.wav`
  for both clips -- `dac_oracle` is the true-DAC-latent-decoded upper bound, already sent to the
  user alongside the MLP's own reconstruction.

## Next command

**Listening verdict received (2026-09-09):** "Reconstructed audio still recognizable but
noticeably worse than previous experiments. DAC oracle is perfect." This experiment's
waveform-correlation win over three of the four HiFi-GAN-family variants did **not** translate to
better perceived quality -- if anything the opposite. The DAC oracle sounding perfect despite a
mediocre 0.37/0.57 waveform correlation is the sharpest evidence yet in this investigation that
raw waveform-sample correlation carries little-to-no diagnostic value here; see
[[DECISIONS.md#muq-to-dac-mlp]] "Listening verdict" section for the full analysis. Blocked on the
user's direction for what to try next -- options on the table (none started, none greenlit):

- Relax the "no temporal modeling" constraint (a few small 1D-conv layers or a tiny learned
  upsampler instead of nearest-neighbor repeat) to test whether the exact architectural ceiling
  identified in this experiment (cosine similarity 0.64) closes -- worth doing for the latent-space
  question on its own merits, but per the listening verdict, closing it is not guaranteed to
  improve perceived audio quality.
- Re-weight future evaluation toward listening tests and mel/latent-domain metrics, since waveform
  correlation has now been shown to mislead in both directions (rewarding EnCodec's real gain,
  rewarding this experiment's real regression, and scoring a perceptually perfect oracle poorly).
- Run the still-pending conditioning-collapse diagnostic from `experiment/mel-hifigan-sanity`
  (untested, orthogonal to this branch's finding).

Do not start any of these without the user's go-ahead. Do not perform further HiFi-GAN, EnCodec,
ground-truth-mel, BigVGAN experiments or proceed to Gate 4/5 without explicit instruction (this
branch's task scope, like every sibling branch, was narrowly defined).

## Blockers

1. **Decision needed from the user**: relax temporal-modeling constraint on this architecture,
   change the evaluation methodology, run the conditioning-collapse diagnostic, or pursue a
   different angle entirely.
2. The still-pending conditioning-collapse diagnostic from `experiment/mel-hifigan-sanity` (see
   that branch's DECISIONS.md) remains untested and is orthogonal to this branch's finding --
   worth revisiting regardless of which direction the user picks next.
3. The `shuffled_time` finding from Gate 4 (see [[DECISIONS.md#gate4-shuffled-time-finding]]) --
   still dormant.

## Expected verification criteria for Gate 5

Not yet reached. Per CLAUDE.md: train on the committed GTZAN split, save periodic fixed
validation reconstructions, report objective metrics and qualitative caveats -- blocked behind
Gate 3/4 resolution above.

## Notes for whoever resumes this

- **Read [[DECISIONS.md#muq-to-dac-mlp]] first** -- it's the most recent finding and the first
  experiment in this investigation to beat MuQ+HiFi-GAN's waveform correlation by a wide margin
  without adversarial training.
- Five experiment branches now exist, each branched independently off `main`, none merged:
  `main` (capacity investigation), `encodec-hifigan-experiment`, `experiment/mel-hifigan-sanity`,
  `experiment/reference-hifigan`, `experiment/muq-to-dac-mlp`. Each has its own isolated
  notes/DECISIONS.md entries; this file reflects only the branches merged into whichever branch
  is currently checked out.
- `transformers==4.55.0` (already pinned, see [[muq-transformers-compat]]) ships `DacModel`
  natively -- no need for the standalone `descript-audio-codec` PyPI package, which conflicts
  with this project's `tensorboard` pin via `protobuf`. See [[muq-to-dac-mlp]] for the exact
  conflict.
- DAC's printed config (`hop_length: 512`) does NOT match its measured behavior (actual hop 320,
  matching `downsampling_ratios` product) -- another confirmed instance of "measure, don't
  assume" catching a real discrepancy. See `src/muq_inversion/dac_encoder.py`'s docstring.
- `FrozenMuQEncoder.encode()`/`FrozenDacEncoder.encode()` run under `@torch.inference_mode()` --
  caching their outputs for reuse across training steps requires `.detach().clone()`, not just
  `.detach()`, or later autograd use raises `RuntimeError: Inference tensors cannot be saved for
  backward`. See `scripts/train_muq_to_dac_mlp.py`.
- Adam betas `(0.9, 0.999)` (PyTorch's default) were used here without issue -- this experiment
  has no adversarial training, so the GAN-collapse-driven `(0.8, 0.99)` fix from the HiFi-GAN
  branches doesn't apply and wasn't needed.
