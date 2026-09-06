# NEXT

## Current gate

Still investigating Gate 3 (Tiny-Set Overfit) -- explicitly not proceeding to Gate 4/5. The
capacity-scaling line of investigation is **closed** (see conclusions below); the audio-quality
question that started it is still open, now reframed around training depth vs. representation
limits rather than decoder size.

**Conclusions reached this session (full reasoning in
[[DECISIONS.md#gate3-capacity-ruled-out]]):**

1. Capacity is not the bottleneck. Tested twice (512 vs 1024 `base_channels`, everything else
   held fixed): once confounded by a training collapse, once cleanly after fixing the optimizer.
   In the clean comparison, 1024 channels won on every objective metric (mel correlation
   0.97/0.98 vs 0.92/0.93, waveform correlation positive vs. near-zero/negative) yet the user
   heard **no perceptible difference** -- "equally mediocre: recognizable but still plagued by
   artifacts." Do not pursue further capacity scaling (e.g. 2048 channels) without first changing
   something else, or expect the same "metrics improve, ears don't" pattern to repeat.
2. Mel/waveform correlation are demonstrably poor proxies for the artifact that matters
   perceptually here (almost certainly phase/fine-temporal structure, which magnitude-domain
   losses and simple correlation metrics are both largely blind to). Do not use metric deltas
   alone to decide whether a change helped, in either direction.
3. Content survives decoding (rhythm, harmony, coarse identity -- recognizable across all
   listening rounds and both capacity levels) but clean phase/fine structure does not. This is
   progress on CLAUDE.md's Scientific Question 3: it rules out "decoder too small" as the
   implementation-side explanation. Two explanations remain open and undistinguished:
   - **(1) Adversarial training is too shallow** -- 15,000 steps on 2 overfit clips is small by
     vocoder-GAN standards. Cheap-ish and directly actionable to test.
   - **(2) MuQ's representation (confirmed 25 Hz / 960 samples-per-frame = 40ms) may impose a
     hard ceiling** on recoverable fine time-domain detail, independent of decoder/training.
     Harder to test; would need either explanation (1) to be ruled out first (train much longer/
     broader and see if the gap closes) or a more targeted diagnostic (error concentration by
     frequency band, transient region, or track/genre).

## Exact last successful commands

```bash
uv run python -m muq_inversion.cli train --config configs/smoke.yaml        # 512, betas fixed, PASSED, no collapse
uv run python -m muq_inversion.cli train --config configs/smoke_1024.yaml   # 1024, betas fixed, PASSED, no collapse
uv run pytest -q       # 47 passed
uv run ruff check .    # all checks passed
```

2026-09-06, macOS, M4 Max, device=mps. No source/config changes since these runs -- this session's
final activity was documentation consolidation only (this file, MEMORY.md, DECISIONS.md,
SCRATCHPAD.md), no new experiments.

## Run/checkpoint path

- `runs/gate3_tiny_overfit/checkpoints/{latest,step_015000}.pt` (git-ignored) -- MRF-512, betas
  `(0.8, 0.99)`, converged cleanly, mel correlation 0.924/0.932. Current "official" working config.
- `runs/gate3_tiny_overfit_mrf1024/checkpoints/{latest,step_001000..step_015000}.pt`
  (git-ignored) -- MRF-1024, betas `(0.8, 0.99)`, converged cleanly, mel correlation 0.965/0.976
  (best objective result of the investigation, not perceptually distinguishable from 512).
- `artifacts/gate3_tiny_overfit/` and `artifacts/gate3_tiny_overfit_mrf1024/` hold the
  reference/reconstruction/difference WAVs for the final (betas-fixed) versions of both configs.
- `configs/smoke_1024.yaml` and `data/manifests/gtzan_manifest_smoke_1024.csv` tracked, retained
  as a documented negative-for-perception (positive-for-metrics) result, not deleted.
- `configs/smoke.yaml` carries this session's two changes: the MRF/1024-vs-512 capacity work and
  the `generator_betas`/`discriminator_betas` optimizer fix (both `(0.8, 0.99)`, replacing
  PyTorch's default `(0.9, 0.999)` which was the root cause of the original 1024-channel collapse).

## Next command

**Blocked on a decision, not a technical step.** Proposed to the user (not yet answered): test
explanation (1) above with a much longer and/or broader adversarial training run before drawing
any conclusion about explanation (2). Concretely, if greenlit, that would mean either:

- Extending `training.steps` substantially beyond 15,000 (e.g. 50,000-100,000) on the same 2
  fixed clips, to see if the adversarial phase eventually resolves the phase/ringing artifact
  given much more time, or
- Moving to a broader training regime (more than 2 clips, i.e. edging toward Gate 4/5 territory)
  since a GAN discriminator trained on only 2 examples may simply not have enough signal to learn
  a general notion of "natural phase" vs. overfitting to two specific clips' spectra.

Do not start either without the user's go-ahead -- this is a real fork (more steps on the tiny set
vs. more data) and the choice affects how to interpret the result afterward.

**Alternative if the user wants to stop here:** treat "recognizable content, capacity-independent
artifacts" as the documented Gate 3 finding, and move toward writing up the interpretability
result as-is, or pivot to the still-open Gate 4 `shuffled_time` finding
([[DECISIONS.md#gate4-shuffled-time-finding]]) instead.

Once Gate 3's audio-quality investigation reaches a stopping point the user is satisfied with,
remaining follow-up work before Gate 5:

1. Port the final Gate 3 architecture/optimizer settings into `pilot.py` (Gate 4 currently has
   none of the MRF/discriminator/adversarial-loss/beta work from this session).
2. Re-investigate [[DECISIONS.md#gate4-shuffled-time-finding]] with the updated decoder.
3. Implement Gate 5 (GTZAN Pilot): batching beyond 1-clip-per-step, a real validation loop, and a
   longer-run-appropriate checkpoint/logging schedule.

## Blockers

1. **Decision needed from the user**: longer-training experiment on the tiny set, broader-data
   experiment, or stop the Gate 3 audio-quality investigation here. Nothing else is blocked on
   anything technical.
2. The `shuffled_time` finding from Gate 4 (see [[DECISIONS.md#gate4-shuffled-time-finding]]) --
   dormant while Gate 3 is being investigated with a decoder Gate 4 hasn't seen yet.

## Expected verification criteria for Gate 5

Not yet reached. Per CLAUDE.md: train on the committed GTZAN split, save periodic fixed
validation reconstructions, report objective metrics and qualitative caveats -- blocked behind
Gate 3/4 resolution above.

## Notes for whoever resumes this

- **Read [[DECISIONS.md#gate3-capacity-ruled-out]] first** -- it's the load-bearing conclusion of
  this entire session's Gate 3 work and directly determines what the next experiment should be.
- The `transformers==4.55.0` pin in `pyproject.toml`/`uv.lock` is load-bearing -- see
  [[DECISIONS.md#muq-transformers-compat]]. Do not let a routine `uv add` bump it past `<5.0`
  without re-running Gate 0.
- Training/validation crop lengths must stay in units of `crop_frames` (multiples of 960 samples)
  -- see [[DECISIONS.md#gate3-crop-alignment]].
- Mel/STFT loss configs are explicit, not copied from speech HiFi-GAN -- see
  [[DECISIONS.md#gate3-loss-configs]]. They are also now known to be poor proxies for the
  perceptual artifact that matters here -- see [[DECISIONS.md#gate3-capacity-ruled-out]] Finding 2.
- MPS training is not bit-exact reproducible even with identical seeds; resume-equivalence
  tolerances must be calibrated per run length -- see [[DECISIONS.md#gate4-mps-resume-tolerance]].
- The GTZAN split is checksum-deduplicated but not artist-aware -- see
  [[DECISIONS.md#gtzan-source-and-defects]].
- Cache entries are single `.pt` files (`torch.save`), not `.npy`/`.npz` -- see
  [[DECISIONS.md#gate2-cache-format]]. Neither Gate 3 nor Gate 4 training reads this cache.
- The generator uses a proper MRF (multi-receptive-field) design -- `upsample_rates=(8,6,5,4)`,
  see [[DECISIONS.md#gate3-generator-capacity]]. Do not increase `base_channels` further as a
  first move -- see the capacity conclusions above.
- **Adam betas for both optimizers are `(0.8, 0.99)`, not PyTorch's default `(0.9, 0.999)`** --
  fixed a real, reproducible GAN training collapse. Do not revert without re-testing for collapse
  (see [[DECISIONS.md#gate3-capacity-1024]] "Update (optimizer-beta fix applied)").
- MuQ's frame rate (25 Hz / 960 samples per frame / 40ms) is a measured, confirmed fact (Gate 2)
  now suspected -- not confirmed -- to be a possible hard ceiling on reconstruction fidelity. See
  [[DECISIONS.md#gate3-capacity-ruled-out]] Finding 3, explanation 2.
