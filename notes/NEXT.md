# NEXT

## Current branch and gate

Branch: `experiment/mel-hifigan-sanity` (created off `main` at `38d9507`, NOT off the sibling
`encodec-hifigan-experiment` branch -- deliberately isolated, no EnCodec code here). Sanity check:
can the current HiFi-GAN pipeline reconstruct audio from the literal representation it was
designed for (ground-truth mel spectrogram)? Still Gate 3 scope -- not touching Gate 4/5, not
starting BigVGAN work, per explicit instruction.

**Headline result, UPDATED with listening verdict (full reasoning in
[[DECISIONS.md#mel-hifigan-sanity]]):** Ground-truth mel + HiFi-GAN converged cleanly with mel
correlation 0.944/0.966 (comparable to MuQ/EnCodec), but waveform correlation only
0.059/-0.010 -- close to MuQ's 0.073/0.104, nowhere near EnCodec's 0.682/0.602. **User's listening
verdict: "mediocre again... almost identical to the other experiments. There must [be] something
wrong with the training pipeline."**

**This raises a more serious hypothesis than the earlier "training paradigm is under-scaled"
framing.** If EnCodec's 6-9x higher waveform correlation reflected a real, audible improvement,
EnCodec should sound at least somewhat different from MuQ/ground-truth-mel -- it reportedly
doesn't. This points toward possible **conditioning collapse** (the generator not meaningfully
using its conditioning input, producing similar output regardless of what's fed in), consistent
with Gate 4's previously unresolved `shuffled_time` finding
([[DECISIONS.md#gate4-shuffled-time-finding]]). **This has not been tested yet** -- see "Next
command" below for the specific, cheap diagnostic that was identified but not run.

## Exact last successful commands

```bash
uv run python -m muq_inversion.cli prepare-data --config configs/mel_1024.yaml
uv run python -m muq_inversion.cli train --config configs/mel_1024.yaml   # PASSED, no collapse
uv run pytest -q       # 47 passed
uv run ruff check .    # all checks passed
```

2026-09-08, macOS, M4 Max, device=mps. `configs/smoke.yaml`/`smoke_1024.yaml` (MuQ baseline) were
not touched -- confirmed via `git diff main` and by the unaffected 47-test suite throughout.

## Run/checkpoint path

- `runs/gate3_tiny_overfit_mel/checkpoints/{latest,step_001000..step_015000}.pt`
  (git-ignored) -- ground-truth-mel + MRF-1024, betas `(0.8, 0.99)`, converged cleanly, 15000
  steps.
- `artifacts/gate3_tiny_overfit_mel/<track_id>_{reference,reconstruction,difference}.wav`
  (git-ignored) -- sent to the user for listening comparison against MuQ and EnCodec audio.
- `configs/mel_1024.yaml` and `data/manifests/gtzan_manifest_mel_1024.csv` tracked (new).
- New/modified source: `src/muq_inversion/mel_encoder.py` (new), `src/muq_inversion/config.py`
  (added `GroundTruthMelConfig` + `encoder_type` field, default-preserving),
  `src/muq_inversion/train.py` (added the encoder-selection branch + expanded probe diagnostics).
- No MuQ or EnCodec runs exist on this branch's `runs/`/`artifacts/` -- the MuQ/EnCodec numbers
  in the comparison table were carried over from the sibling `encodec-hifigan-experiment` branch
  (already established this session), not re-derived here.

## Next command

**The specific next diagnostic, identified but not yet run (interrupted by the user asking to
summarize and commit instead):** test whether the generator is actually using its conditioning at
all, using already-trained checkpoints -- no retraining needed. All three checkpoints (MuQ,
EnCodec, ground-truth-mel + MRF-1024) are still present in the gitignored `runs/` directory,
unaffected by branch switches (confirmed present). Plan:

1. For each checkpoint, load `resolved_config` from the checkpoint file itself (already stored
   there) to reconstruct the exact `MiniGenerator` architecture and load its weights.
2. Compute the *correct* conditioning for a fixed validation crop (reusing whichever encoder that
   checkpoint's config specifies -- `muq`, `transformers.EncodecModel`, or the ground-truth-mel
   transform; all three packages are available in the shared venv regardless of branch).
3. Also compute conditioning under: zero (all-zeros, same shape), shuffled-in-time (permute frame
   order), and cross-track (swap in the other fixed clip's correct conditioning while keeping the
   same target).
4. Run the generator under all four conditioning variants per checkpoint and compare outputs
   (correlation, RMS difference) against the correct-conditioning output.

If outputs barely change across variants -> conditioning collapse confirmed, a genuine pipeline
issue (not a representation or training-depth limitation) -- would need to investigate why (e.g.
input projection bottleneck, adversarial pressure with only 2 training examples overwhelming the
conditioning signal, dead units). If outputs do change substantially with conditioning ->
argues against a collapse bug, back toward the training-paradigm/phase-reconstruction-difficulty
explanation already on record in [[DECISIONS.md#mel-hifigan-sanity]].

Do not merge any of the three experiment branches (`main`, `encodec-hifigan-experiment`,
`experiment/mel-hifigan-sanity`) into each other without an explicit decision first.

## Blockers

1. **The conditioning-collapse diagnostic above has not been run.** This is the single most
   informative next step and should happen before any further training experiments on any
   representation -- continuing to vary representations or training length without first ruling
   out a collapse bug risks repeating the same "big change, no perceptual difference" pattern for
   a reason unrelated to representation or training depth.

## Notes for whoever resumes this

- **Read [[DECISIONS.md#mel-hifigan-sanity]] first** -- full comparison table (MuQ/EnCodec/
  ground-truth-mel side by side), the mel-extraction `center=False` subtlety, and the refined
  hypothesis about magnitude-only vs. reconstruction-trained representations.
- This branch's changes are additive and default-preserving: `encoder_type` defaults to `"muq"`.
  Built independently from the EnCodec branch's own (differently-named but structurally similar)
  `encoder_type` field -- the two branches do not share this code, since they diverged from
  `main` separately. If these branches are ever merged, `config.py`'s `encoder_type` additions
  will need to be reconciled into one field supporting `"muq" | "encodec" | "ground_truth_mel"`.
- If more ground-truth-mel experiments are needed, remember the crop-alignment invariant:
  `crop_samples == training.crop_frames * generator.total_upsample`, and `generator.total_upsample`
  must equal the mel transform's actual samples-per-frame ratio (`hop_length`, 300 for the
  current `n_fft=2048`/`hop_length=300` config) -- not MuQ's 960 or EnCodec's 320.
- `GroundTruthMelEncoder` uses HiFi-GAN's own manual-pad + `center=False` convention, not
  torchaudio's default `center=True` -- changing `n_fft`/`hop_length` requires re-verifying the
  frame count comes out to exactly `crop_samples // hop_length` before trusting any new config.
- All MuQ-branch findings from `main` still apply and are unaffected: capacity is not the
  bottleneck for MuQ+HiFi-GAN ([[DECISIONS.md#gate3-capacity-ruled-out]]), Adam betas must be
  `(0.8, 0.99)` not PyTorch's default ([[DECISIONS.md#gate3-capacity-1024]]), and the generator
  uses a proper MRF design ([[DECISIONS.md#gate3-generator-capacity]]).
- The `transformers==4.55.0` pin remains load-bearing for MuQ ([[DECISIONS.md#muq-transformers-compat]])
  -- irrelevant to this branch's own work (no `transformers` dependency used here at all, since
  the mel transform is pure `torchaudio`), but still governs the MuQ path this branch inherited.
