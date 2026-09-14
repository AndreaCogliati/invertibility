# NEXT

## Current gate

Spectrogram-to-DAC positive control (`tasks/TASK-01.md`, code committed alongside
`experiment/muq-layer-sweep`). **Result: the training pipeline itself cannot productively exploit
a substantially richer input representation either -- audio-relevant metrics (STFT distance,
waveform correlation) did not improve over MuQ, and severe norm/variance collapse plus extreme
over-smoothing persist essentially unchanged.** Full reasoning in
[[DECISIONS.md#spectrogram-positive-control]]. Do not proceed to the optional mel-bin-ablation
follow-up, model scaling, or any new representation experiment without further direction (per
this task's own instruction to prioritize diagnosis over model improvement).

**Conclusions reached this session (full reasoning in
[[DECISIONS.md#spectrogram-positive-control]]):**

1. Built a full parallel pipeline reusing everything possible from `experiment/muq-to-dac-more-
   data` unchanged (manifest, split, DAC cache, `TemporalConvNet`, optimizer, loss, step budget,
   checkpoint-selection rule) -- only the input representation changed (log-mel spectrogram,
   128 mels, hop aligned to DAC's 75Hz grid) plus one required 128->1024 linear input projection
   (spectrogram's channel count doesn't coincidentally match DAC's the way MuQ's did).
2. **Latent-space metrics improved only modestly** (val cosine 0.230 -> 0.298, ~30% relative;
   val MSE 17.29 -> 16.54, ~4%) despite the spectrogram carrying far more raw temporal-spectral
   resolution than MuQ's compressed embedding.
3. **Audio-relevant metrics did not improve at all**: multi-res STFT distance got *worse* (2.65
   -> 3.02 on val), and waveform correlation to the DAC oracle stayed at the same ~0 noise floor
   observed in every MuQ-based experiment to date.
4. **New diagnostics added this session** (`latent_diagnostics.py`, not computed by any prior
   MuQ-to-DAC script) found severe, essentially undiminished norm collapse (predicted/oracle
   vector-norm ratio 0.36), variance collapse (ratio 0.11), and extreme over-smoothing (lag-1
   autocorrelation 0.97 for predictions vs. 0.08 for the oracle) -- exactly the pathologies named
   in CLAUDE.md's own Scientific Motivation section, present in essentially unchanged form under
   a richer input.
5. **Per this task's own pre-registered decision rule, this is Outcome B (similar failure), not
   Outcome A**: the training/objective formulation (a 13-frame-receptive-field residual ConvNet
   doing MSE+cosine regression directly onto DAC's raw continuous latent) is now the leading
   suspect for the saturation ceiling seen throughout this investigation -- ahead of both "wrong
   MuQ layer" ([[muq-layer-sweep]], ruled out) and "insufficient MuQ information" (not exonerated,
   but no longer the most parsimonious explanation, since a richer representation hit the same
   wall).
6. The validation-MSE-selected checkpoint landed at step 1000/60000 -- an extreme case of the
   "early lucky dip" pattern already flagged in [[muq-layer-sweep]]'s notes. End-of-training
   (step 60000) numbers were close (val_mse 17.06 vs. 16.54, val_cos 0.27 vs. 0.30), so this
   specific selection did not appear misleading, but the pattern is worth continuing to watch.

## Exact last successful commands

```bash
uv run python scripts/precompute_spectrogram_cache.py   # 1998 crops, 14.6s -> data/cache/spectrogram_control/mel_cache.pt
uv run python scripts/train_spectrogram_to_dac.py        # 60000 steps, 518s train + eval, PASSED sanity check
uv run ruff check .    # all checks passed
uv run pytest -q       # 73 passed (62 baseline + 11 new spectrogram-control tests)
```

2026-09-14, macOS, M4 Max, device=mps. Launched via `nohup ... &` + `disown`, tracked by real PID
and a `Monitor` until-loop.

## Run/checkpoint path

- `data/cache/spectrogram_control/mel_cache.pt` (git-ignored) -- log-mel features for all 1998
  crops of the reused `more_data_crops.csv` manifest.
- `runs/spectrogram_control/checkpoints/best.pt` (git-ignored) -- validation-MSE-selected
  checkpoint, step 1000.
- `artifacts/spectrogram_control/report.json` -- full metrics (latent MSE/cosine, 3 eval
  conditions, STFT/waveform-correlation, latent norm/variance/autocorrelation diagnostics by
  split, direct comparison block against `artifacts/more_data/report.json`, training curves,
  parameter count, timing, peak memory).
- `artifacts/spectrogram_control/*.wav` (30 files, git-ignored) -- reference / dac_oracle /
  spectrogram_to_dac / shuffled_across / time_shifted, 3 val + 3 test tracks, step 1000. Sent to
  the user for a listening verdict; not yet given as of this note.

## Next command

**Blocked on user direction and the pending listening verdict.** This task's own instructions say
not to prioritize the optional mel-bin-ablation follow-up until the primary result is understood
and (implicitly) reviewed. Given the conclusion narrows the remaining hypothesis space for this
whole investigation to the training/objective formulation itself, natural next directions (not yet
authorized) are:

- **A different loss or target formulation for the DAC latent** -- e.g. something less purely
  pointwise-MSE, or a formulation that respects DAC's residual-VQ structure more directly (the
  requantization/codebook-agreement diagnostics from `scripts/analyze_dac_manifold.py` are
  relevant prior art here).
- **The still-untested architecture/receptive-field-capacity axis** flagged as the next hypothesis
  in [[muq-layer-sweep]]'s own NEXT.md, now with slightly more urgency since this experiment
  suggests architecture/objective rather than representation is the bottleneck.
- **Treating the MuQ-to-DAC-via-`TemporalConvNet` line as having reached a documented ceiling**
  with this specific pipeline design, independent of the input representation.

Do not start any of these without the user's go-ahead.

## Blockers

1. **Listening verdict for this experiment's 30 WAVs** -- pending, needed before treating the
   metric-based "Outcome B" reading as final (metrics and listening have diverged before in this
   project, in both directions -- see [[muq-to-dac-mlp]], [[muq-to-dac-temporal-conv]]).
2. **Decision needed from the user** on which of the three "Next command" directions (loss/target
   reformulation, architecture/receptive-field capacity, or stopping this pipeline design) to
   pursue next.
3. The still-pending conditioning-collapse diagnostic from `experiment/mel-hifigan-sanity` (see
   that branch's DECISIONS.md) remains untested and is orthogonal to this branch's finding.
4. The `shuffled_time` finding from Gate 4 (HiFi-GAN-family, see
   [[DECISIONS.md#gate4-shuffled-time-finding]]) -- still dormant.
5. Listening verdicts for the 80-track scaling experiment, the full-GTZAN scaling experiment, and
   the MuQ layer-sweep were never given by the user -- not blocking, but worth noting if any comes
   up again.

## Expected verification criteria for Gate 5

Not yet reached, and not authorized to start per this and prior experiments' own instructions.

## Notes for whoever resumes this

- **Read [[DECISIONS.md#spectrogram-positive-control]] first** -- it's the first experiment in
  this investigation to test the *pipeline*, not the representation, and its result reframes what
  the remaining hypothesis space looks like.
- `spectrogram_encoder.py`, `latent_diagnostics.py`, and `models/spectrogram_translator.py` are
  new, reusable modules -- `latent_diagnostics.py` in particular (norm ratio, variance ratio,
  lag-1 autocorrelation, all unit-tested) is generic and worth reusing for any future experiment
  that wants the same norm/variance-collapse diagnostics prior scripts computed only ad hoc
  (`scripts/analyze_dac_manifold.py`).
- `TemporalConvNet` itself remains byte-for-byte unchanged across every experiment in this
  investigation to date (Gate 3 through this one) -- if the architecture/receptive-field-capacity
  hypothesis is pursued next, this will be the first experiment to actually change it.
- Validation-MSE-only checkpoint selection continues to be noisy at this pipeline's training
  scale (step 1000/60000 selected here, step 1500/45000 in [[muq-layer-sweep]]'s control run) --
  worth a sanity glance at the full curve, not just the selected checkpoint's number, in any
  future experiment using this style of training loop.
