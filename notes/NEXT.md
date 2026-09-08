# NEXT

## Current branch and gate

Branch: `encodec-hifigan-experiment` (off `main` at `38d9507`). Testing whether HiFi-GAN itself
is the bottleneck, or MuQ's representation specifically, by swapping only the representation
source and keeping the exact same decoder/training recipe. Still Gate 3 scope -- not proceeding
to Gate 4/5 on either representation, and not starting BigVGAN work, per explicit instruction.

**Headline result, UPDATED with listening verdict (full reasoning in
[[DECISIONS.md#encodec-mediocrity-across-all-variants]]):** EnCodec + HiFi-GAN produced
**waveform correlation 0.682/0.602 -- a 6-9x jump over MuQ's best-ever 0.073/0.104**, the largest
objective-metric effect size of any experiment in this project. But the user's listening
comparison across the tiny baseline, MuQ+MRF-1024, and EnCodec+MRF-1024 found **no audible
difference among any of them** -- "all similarly mediocre." Combined with
[[DECISIONS.md#gate3-capacity-ruled-out]] (capacity also produced a large metric gain with zero
perceptual gain), **two major technical axes are now ruled out**: neither decoder capacity nor
representation source (MuQ vs. EnCodec) explains what's limiting perceived audio quality here.
This is a decision point, not a technical blocker -- see "Next command" below.

## Exact last successful commands

```bash
uv run python -m muq_inversion.cli prepare-data --config configs/encodec_1024.yaml
uv run python -m muq_inversion.cli train --config configs/encodec_1024.yaml   # PASSED, no collapse
uv run pytest -q       # 47 passed
uv run ruff check .    # all checks passed
```

2026-09-08, macOS, M4 Max, device=mps. `configs/smoke.yaml`/`smoke_1024.yaml` (MuQ baseline) were
not touched -- confirmed via `git diff` and by the unaffected 47-test suite throughout.

## Run/checkpoint path

- `runs/gate3_tiny_overfit_encodec/checkpoints/{latest,step_001000..step_015000}.pt`
  (git-ignored) -- EnCodec + MRF-1024, betas `(0.8, 0.99)`, converged cleanly, 15000 steps.
- `artifacts/gate3_tiny_overfit_encodec/<track_id>_{reference,reconstruction,difference}.wav`
  (git-ignored) -- sent to the user for listening comparison against the MuQ+MRF-1024 audio.
- `configs/encodec_1024.yaml` and `data/manifests/gtzan_manifest_encodec_1024.csv` tracked (new).
- New/modified source: `src/muq_inversion/encodec_encoder.py` (new),
  `src/muq_inversion/config.py` (added `EncodecConfig` + `encoder_type` field, default-preserving),
  `src/muq_inversion/train.py` (added the encoder-selection branch + expanded probe diagnostics).
- MuQ-path runs from the prior session (`runs/gate3_tiny_overfit/`,
  `runs/gate3_tiny_overfit_mrf1024/` and their `artifacts/`) are untouched and still the
  comparison baseline for this branch's work.

## Next command

**Blocked on a strategic decision, not a technical step.** The listening verdict came back
negative on the perceptual question (no difference from any variant), which resolves the
technical uncertainty but opens a fork proposed to the user, not yet decided:

1. **Change the shared training paradigm.** Every variant tested so far shares: 2 fixed clips, a
   short training budget (500-15,000 steps), and the same generator/discriminator/loss *design*
   (only width or representation varied). Testing whether more data (moving toward Gate 4/5
   scale) or much longer training resolves the mediocrity would tell us whether the paradigm
   itself -- not capacity, not representation -- was the missing ingredient. This is a real
   scope escalation, not a quick config tweak.
2. **Improve the evaluation method before changing anything else.** A quick, unblinded listening
   pass on 2-second clips may not be resolving real (if subtle) differences the metrics are
   picking up on. A more careful comparison (blinded A/B, longer excerpts, repeated listening)
   could either confirm the "truly no difference" finding more rigorously or overturn it.
3. **Stop here and document the finding.** Treat "recognizable content, mediocre quality, robust
   to both capacity and representation choice" as the interpretability result for this phase --
   consistent with CLAUDE.md's own "fail early" mission framing -- and write up accordingly rather
   than continuing to spend compute on this specific paradigm.

Do not start any of these unilaterally -- they imply different amounts of further work and
different interpretations of everything done so far. Whatever is decided, do not merge this
branch into `main` without an explicit decision about what (if anything) from it should inform
the main MuQ-only investigation.

## Blockers

1. **Strategic decision needed from the user**: escalate the training paradigm (more data/longer
   training), revisit the evaluation method, or stop and document the current finding. Nothing
   else is blocked on anything technical -- every experiment run so far completed cleanly.

## Notes for whoever resumes this

- **Read [[DECISIONS.md#encodec-mediocrity-across-all-variants]] first** -- it's the load-bearing
  conclusion of this branch's work and directly determines what the next experiment should be.
  [[DECISIONS.md#encodec-hifigan]] has the underlying comparison table, the one forced
  architectural adaptation (upsample_rates/crop_frames, required by EnCodec's different measured
  frame rate, not a design choice), and the original four-question analysis (now superseded by
  the generalized synthesis once the listening verdict came back).
- This branch's changes are additive and default-preserving: `encoder_type` defaults to `"muq"`,
  so nothing here should ever affect a config that doesn't explicitly set `encoder_type: encodec`.
  If that invariant ever breaks (e.g. a MuQ config's behavior changes), that's a regression to fix
  immediately, not a design tradeoff to accept.
- EnCodec's continuous latent is obtained via `model.encoder(input_values)` (an internal
  submodule call, not the public `.encode()` API, which returns quantized codes) -- this was
  deliberate, not an oversight; see [[DECISIONS.md#encodec-hifigan]] for why.
- If more EnCodec experiments are needed, remember the crop-alignment invariant: crop_samples
  must equal `training.crop_frames * generator.total_upsample`, and `generator.total_upsample`
  must equal EnCodec's actual measured samples-per-frame ratio (320 for `facebook/encodec_24khz`),
  not MuQ's (960). Don't copy MuQ's `upsample_rates`/`crop_frames` values without re-deriving them
  for whatever encoder is in use.
- All MuQ-branch findings from `main` still apply and are unaffected: capacity is not the
  bottleneck for MuQ+HiFi-GAN ([[DECISIONS.md#gate3-capacity-ruled-out]]), Adam betas must be
  `(0.8, 0.99)` not PyTorch's default ([[DECISIONS.md#gate3-capacity-1024]]), and the generator
  uses a proper MRF design ([[DECISIONS.md#gate3-generator-capacity]]).
- The `transformers==4.55.0` pin remains load-bearing for MuQ ([[DECISIONS.md#muq-transformers-compat]])
  -- EnCodec support happens to already be compatible with this same pinned version, so no
  additional dependency-compatibility work was needed for this branch.
