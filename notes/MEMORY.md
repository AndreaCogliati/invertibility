# MEMORY

Durable, verified facts. Update only with confirmed information.

## Confirmed tensor shapes and frame rates (2026-09-04)

Measured live via `uv run python -m muq_inversion.cli doctor --config configs/smoke.yaml`
on a synthetic 2.0s / 48000-sample 24kHz clip, MuQ `OpenMuQ/MuQ-large-msd-iter`, device `mps`:

- `last_hidden_state` shape: `(batch, num_frames, 1024)` -- feature_channels = 1024.
- Frame rate: 25.0 Hz.
- Samples per frame: 960.0 (exact integer ratio at 24kHz -- convenient for the decoder's
  upsampling stage product).
- Output is finite; repeated forward passes on the same input produce the same shape
  (determinism spot-check, not a full byte-identity check).
- Confirmed against real GTZAN audio (Gate 2, `extract`, 2026-09-04) across 1s/2s/5s/10s/full-length
  crops of 3 tracks (15 probes): `frames_per_second` stayed within `[24.9998, 25.0000]` (spread
  well inside the 0.5 Hz tolerance), `mean_samples_per_frame = 960.0013`. Frame counts matched
  `round(num_samples / mean_samples_per_frame)` within 1 frame for every probe -- i.e. MuQ's
  frame rate is stable across duration and content, not just on synthetic noise.

## Working dependency versions (Apple Silicon, macOS 26.6.2, M4 Max, 2026-09-04)

- Python 3.12.12, torch 2.14.0, torchaudio 2.11.0, numpy 2.5.2.
- `transformers==4.55.0` pinned with `huggingface_hub>=0.34.0,<1.0` (see
  [[muq-transformers-compat]] for why this exact pin is required).
- `muq==0.1.0` (PyPI package `muq`, imports as `from muq import MuQ`).
- MPS backend available and selected by default (`torch.backends.mps.is_available() == True`).

## Stable paths and commands

- Doctor (Gate 0): `uv run python -m muq_inversion.cli doctor --config configs/smoke.yaml`
- Prepare data (Gate 1): `uv run python -m muq_inversion.cli prepare-data --config configs/smoke.yaml`
- Extract / feature contract (Gate 2): `uv run python -m muq_inversion.cli extract --config configs/smoke.yaml`
- Train / tiny-set overfit (Gate 3): `uv run python -m muq_inversion.cli train --config configs/smoke.yaml`
- Short pilot (Gate 4): `uv run python -m muq_inversion.cli pilot --config configs/pilot.yaml`
- Run/checkpoint output: `runs/<run_name>/checkpoints/{latest,step_NNNNNN}.pt` (git-ignored;
  `run_name` is `training.run_name` for Gate 3 or `pilot.run_name` for Gate 4).
- Reference/reconstruction/difference audio: `artifacts/<run_name>/<track_id>_*.wav` (git-ignored).
- Manifest output: `data/manifests/gtzan_manifest_<config-name>.csv` (tracked in git).
- Raw GTZAN audio lives at `data/raw/gtzan/genres/<genre>/<genre>.NNNNN.wav` (git-ignored).
- Cache entries live at `data/cache/<key[:2]>/<key>.pt` (git-ignored), each a
  `torch.save({"features": tensor, "metadata": dict})`, content-addressed by
  (audio checksum, MuQ model id, preprocessing dict, representation name, dtype).

## Generator parameter counts (measured, 2026-09-06)

For the current MRF design (`upsample_rates=(8,6,5,4)`, `resblock_kernel_sizes=(3,7,11)`,
`resblock_dilation_sizes=((1,3,5),(1,3,5),(1,3,5))`, `input_channels=1024`):

- `base_channels=512`: 17,235,585 params (68.9MB fp32), checkpoint file ~250.6MB (includes Adam
  optimizer state + discriminators + discriminator optimizer state).
- `base_channels=1024`: 61,582,593 params (246.3MB fp32, 3.57x), checkpoint file ~782.8MB.
- MPS memory (generator + Adam only, no encoder/discriminators, batch 1): 512 ->
  279.5MB current / 1271.9MB driver-allocated; 1024 -> 1007.0MB / 2496.7MB. Both comfortably
  within the M4 Max's 64GB unified memory.
- Per-step training time (100-step timed sample, 2 clips, discriminators active after warmup):
  512 -> ~0.095s/step; 1024 -> ~0.128s/step (~1.35x, sub-linear vs. the 3.57x parameter increase).

## Cache size (measured, Gate 2, 2026-09-04)

- float32: ~102,399 bytes/audio-second at 1024 channels, 25 fps.
- float16: ~51,200 bytes/audio-second (half, as expected).
- float16 quantization error vs float32: max abs error 0.0074 across all probes, well under the
  documented tolerance of 0.05 (`feature_contract.lowprecision_max_abs_error`).

## Validated design choices

- See [[muq-transformers-compat]], [[gtzan-source-and-defects]], [[decoder-upsample-placeholder]],
  [[gate2-cache-format]], [[gate3-crop-alignment]], [[gate3-loss-configs]],
  [[gate4-mps-resume-tolerance]], [[gate4-shuffled-time-finding]], [[gate3-adversarial-losses]],
  [[gate3-generator-capacity]], [[gate3-capacity-1024]], [[gate3-capacity-ruled-out]].

## Known platform limitations

- `transformers>=5.0` breaks MuQ's internal `Wav2Vec2ConformerEncoder` construction (see
  [[muq-transformers-compat]]) -- do not let a bare `uv add`/`uv sync` drift the resolved
  transformers version past `<5.0` without re-testing Gate 0.

## Benchmark / gate results that passed sanity checks

- Gate 0 (environment doctor): PASSED on 2026-09-04, device=mps, see run output above.
- Gate 1 (data integrity): PASSED on 2026-09-04 -- 1000/1000 GTZAN files indexed, 1 known-corrupted
  file (`jazz.00054.wav`) flagged and excluded, 14 duplicate-checksum pairs detected and kept
  within a single split each, split counts train=793/val=94/test=113, 20-file resample
  verification sample all finite. See [[gtzan-source-and-defects]].
- Gate 2 (feature contract): PASSED on 2026-09-04 -- 15 probes (3 real GTZAN tracks x 5 durations)
  all finite, frame rate consistent within tolerance, alignment within 1 frame, 30 cache
  (tensor, dtype) round-trips exact, float16 quantization error within documented tolerance.
- Gate 3 (tiny-set overfit): PASSED on 2026-09-04 -- 2 fixed clips (`pop.00012`, `disco.00049`),
  500 steps, crop_frames=50 (2.0s / 48000 samples), Adam lr=2e-4. Combined (mel + multi-res STFT)
  loss: mean 7.51 (first 10 steps) -> 3.18 (last 10 steps), ratio 0.42 (PASS threshold <= 0.5).
  Checkpoint round-trip verified (`runs/gate3_tiny_overfit/checkpoints/latest.pt`, step 500, all
  required keys present). Reference/reconstruction/difference WAVs written to
  `artifacts/gate3_tiny_overfit/`, all finite and non-silent (reconstruction peak ~0.53 vs.
  reference peak ~1.0 -- expected for an early, short, tiny-model overfit run, not yet
  high-fidelity). See [[gate3-crop-alignment]], [[gate3-loss-configs]].
- Gate 4 (short pilot): NOT FULLY PASSED on 2026-09-04 (`configs/pilot.yaml`, 6 tracks, 400
  steps). Checkpoint/resume equivalence PASSED (max relative loss diff 0.063, tolerance 0.08 --
  see [[gate4-mps-resume-tolerance]] for why MPS needs a scaled, non-bit-exact tolerance).
  Conditioning controls: `shuffled_across` and `constant_zero` robustly, reproducibly worse than
  correct conditioning (rules out total conditioning collapse); `shuffled_time` was NOT materially
  worse (~11-13% vs. the 20% margin, consistent across 2 independent runs) -- see
  [[gate4-shuffled-time-finding]] for the full evidence table and interpretation. This is an open
  diagnostic finding to re-check at Gate 5's larger scale, not a blocking implementation bug.
- Gate 3 revisited (2026-09-04, same day): user listened to the reconstruction-only artifacts and
  reported "buzzing noise... nothing remotely recognizable." Diagnosed as a phase-incoherence
  symptom of magnitude-only losses (log-mel correlation 0.89-0.91 and frame-energy correlation
  0.98 were both good; raw waveform correlation was ~0). Confirmed more steps alone don't help
  (500->5000 steps only moved loss 2.78->2.64). Added multi-period/multi-scale discriminators +
  adversarial/feature-matching losses (starting after a 500-step reconstruction-only warmup,
  8000 steps total) per CLAUDE.md's own Decoder Design item 6. Re-ran: reconstruction loss ratio
  0.40 (PASS), reconstruction peak amplitude closer to reference's across repeated runs (0.76 ->
  0.92-0.99 for `disco.00049`, ref peak 1.0). Whether this actually fixed the perceptual buzziness
  is a listening judgment pending user feedback -- audio sent directly rather than declared fixed
  from metrics alone. See [[gate3-adversarial-losses]] for full evidence and the pass/fail-metric
  bug this surfaced and fixed (gating on combined loss instead of reconstruction-only loss).
- Gate 3, iteration 2 (2026-09-04, same day): user's listening verdict on iteration 1 was
  "disco.00049 starts to resemble the original... rhythm and drum hits can be heard" but
  "pop.00012 is worse" -- percussive content partially recovering, melodic/harmonic content not.
  Added `training.reconstruction_loss_weight=10.0` (applied to mel+STFT during the adversarial
  phase only) so content fidelity stays dominant over adversarial/feature-matching pressure;
  extended training 8000 -> 15000 steps. Result: final mel loss 0.77 -> 0.59, log-mel correlation
  improved for both tracks (pop 0.82 -> 0.90, disco 0.84 -> 0.94). Second listening pass pending.
  See [[gate3-adversarial-losses]] "Update" section for full reasoning.
- Gate 3, iteration 3 (2026-09-04, same day): user's verdict on iteration 2 -- "both excerpts have
  improved, but they are still very poor: a lot of noise, buzzing, and ringing." Diagnosed as a
  generator-capacity problem (ringing after two rounds of loss/schedule tuning points at
  architecture, not training dynamics): rewrote `MiniGenerator` with a proper multi-receptive-
  field (MRF) design (3 parallel `ResBlock`s, kernel sizes 3/7/11, each with 3 dilated-conv pairs,
  averaged -- not the prior sequential 2-block chain), doubled `base_channels` 256->512, and
  re-factored `upsample_rates` from 6 stages `(5,4,4,3,2,2)` to 4 stages `(8,6,5,4)` (same
  product, 960) so channel width isn't halved away as aggressively. Result: final mel loss
  0.59 -> 0.48 (continuing the trend 0.77 -> 0.59 -> 0.48 across all three iterations), log-mel
  correlation pop.00012 0.90 -> 0.94. Third listening pass pending. See
  [[gate3-generator-capacity]].
- Gate 3, capacity-isolation experiment (2026-09-06): user's verdict on MRF-512 -- disco.00049
  "now recognizable", pop.00012 "still contains substantial buzzing/noise" though rhythm/harmony
  present. Ran a controlled 512-vs-1024-channel experiment (`configs/smoke_1024.yaml`, identical
  in every other respect to `configs/smoke.yaml`). Result: **training collapsed** between step
  3500-3750 (discriminator loss -> 0, generator adversarial loss saturated at its LSGAN maximum
  ~2.0, generator output degenerated to a constant +1.0 signal for the rest of the 15000-step run)
  -- not a clean capacity comparison. Recovered the last pre-collapse checkpoint (step 3000):
  mel correlation 0.945/0.950, matching/exceeding the fully-converged 512-channel model's
  0.937/0.944 in only 20% of the training budget -- suggesting capacity was NOT the bottleneck at
  that point. Diagnosed likely root cause: both optimizers use plain Adam's default betas
  `(0.9, 0.999)`, known in GAN literature to cause exactly this discriminator-overpowers-generator
  collapse (HiFi-GAN's own published config uses `betas=(0.8, 0.99)`). Parameter counts: 512 ->
  17,235,585; 1024 -> 61,582,593 (3.57x). Full comparison table and reasoning in
  [[gate3-capacity-1024]]. Recommendation: fix optimizer betas before any further capacity
  experiments; do not conclude either way on 512-vs-1024 quality from this run. Not proceeding to
  Gate 4/5 -- still investigating Gate 3, per explicit instruction.
- Gate 3, optimizer-beta fix + clean capacity re-test (2026-09-06, same day): applied the
  recommended fix (`training.generator_betas`/`discriminator_betas`, both `(0.8, 0.99)`, HiFi-GAN's
  published config, replacing PyTorch's default `(0.9, 0.999)`). Regression check on MRF-512:
  no collapse, mel correlation 0.924/0.932 (comparable to before). Re-ran MRF-1024 with the fix:
  **no collapse this time** -- clean capacity comparison. Result: mel correlation **0.965/0.976**
  (best of the entire investigation), waveform correlation positive for the first time
  (0.073/0.104). 1024 now strictly dominates 512 on every objective metric under matched
  conditions (3.57x params, ~1.7x wall time, ~52 min for 15000 steps). Adopted `(0.8, 0.99)` betas
  as the new standard for both optimizers. `configs/smoke.yaml` (512) remains the official working
  config pending the user's listening judgment on whether 1024's cost is worth adopting. Full
  comparison table in [[gate3-capacity-1024]] "Final comparison table" section.
- Gate 3, capacity investigation CONCLUDED (2026-09-06, same day): user's final listening verdict
  on the clean 512-vs-1024 comparison -- "I can't hear any noticeable improvement... equally
  mediocre: recognizable but still plagued by artifacts." Despite 1024 winning on every objective
  metric (mel correlation 0.97/0.98 vs 0.92/0.93, waveform correlation positive vs near-zero),
  there was zero perceptible difference. **Conclusion: capacity is not the bottleneck** -- ruled
  out by two independent tests (collapse-confounded, then clean). The gap between improving
  metrics and unchanged perception implicates the training objective (adversarial training likely
  too shallow at 15k steps on 2 clips) or, less testably so far, a hard information ceiling in
  MuQ's 25Hz/40ms-per-frame representation. Neither has been distinguished yet. Recommended next
  step (proposed to user, not yet executed): test much longer/broader adversarial training before
  drawing conclusions about the representation itself. See [[gate3-capacity-ruled-out]] for the
  full three-finding synthesis and reasoning.
- MuQ-to-DAC temporal-conv experiment CONCLUDED (2026-09-09, `experiment/muq-to-dac-temporal-conv`
  branch, off `main`): direct follow-up to the framewise-MLP experiment's exact architectural
  ceiling (cosine similarity 0.64). Replaced the MLP's pointwise `Linear` layers with 3 residual
  `Conv1D(k=5)` layers (receptive field 5 MuQ frames, verified both analytically and empirically
  via a perturbation test) -- everything else (input, target, optimizer, loss, steps, clips, crop)
  held identical, a true single-variable comparison. Result: **cosine similarity jumped to
  0.9999-1.0000** (from 0.6418), MSE dropped to ~0.004-0.014 (from ~11.05) -- essentially a full
  latent-space solve. Decoded waveform correlates with the DAC-oracle reconstruction itself at
  0.9999/1.0000 (not just with the reference), meaning the predicted latent decodes to audio
  essentially indistinguishable from decoding the true DAC latent. 15.7M params, 18s total
  training time. **This answers this investigation's central open question for the MuQ-to-DAC
  axis: the framewise MLP's ceiling was a limitation of its memoryless architecture, not of MuQ's
  representation -- MuQ contained the missing information all along, accessible with just ~200ms
  of local temporal context.** Full write-up in [[muq-to-dac-temporal-conv]]. **Listening verdict
  (2026-09-09): "The listening test shows almost perfect reconstruction! We nailed it!"** --
  confirmed by ear, not just by metric. **This is the first unambiguous positive result in the
  entire Gate 3 audio-quality investigation**, after capacity, representation-source (MuQ/
  EnCodec/ground-truth-mel), and custom-implementation-vs-reference-HiFi-GAN were all ruled out or
  only partially resolved the mediocrity. Tagged `muq-to-dac-temporal-conv-v1` as a durable
  milestone. Notably, the metric that actually predicted this perceptual success was "predicted
  latent decodes to the same audio as the true latent" (correlation to the DAC-oracle
  reconstruction, 0.9999/1.0000) -- not waveform-vs-reference correlation, which has been
  unreliable throughout this project (see [[muq-to-dac-mlp]]).
- Gate 4 minimal generalization experiment CONCLUDED (2026-09-09,
  `experiment/muq-to-dac-minimal-gate4` branch, off `experiment/muq-to-dac-temporal-conv`):
  **the Gate 3 near-perfect result (cosine similarity 0.9999) does NOT generalize -- it was
  memorization, not a transferable MuQ-to-DAC mapping.** Retrained the exact Gate 3 architecture
  (TemporalConvNet, 15.7M params, unchanged) fresh on two independent generalization tests:
  Experiment A (16 train / 4 val / 4 test crops, disjoint 2s windows with 1.0s guard bands, from
  the same 2 Gate 3 tracks) and Experiment B (16 train crops from 8 track-disjoint GTZAN tracks
  across 4 genres / 2 val tracks / 2 test tracks, no track in more than one split). Both show a
  textbook overfitting signature: train cosine similarity reaches 0.40-0.45 by step 8000, but
  validation MSE increases monotonically from the start of training, and the required
  validation-MSE-based checkpoint selection picks **step 200 of 8000** in both experiments.
  Held-out (val/test) waveform correlation to the DAC oracle collapses to ~0 (from a real,
  positive 0.25-0.33 on training data) in BOTH experiments, with Experiment B (unseen tracks) no
  worse than Experiment A (unseen crops, same tracks) -- an all-or-nothing collapse, not a
  graded specificity effect. Shuffled-across and time-shifted controls are clearly beaten by
  correct conditioning on training data (real, well-separated effect) but are statistically
  indistinguishable from correct conditioning on held-out data -- the model's real
  conditioning-sensitivity is entirely confined to memorized examples. DAC oracle quality
  unaffected throughout (0.05-0.67 correlation to reference, consistent with every prior
  experiment); no stop conditions triggered (no split overlap, no frame misalignment, no
  non-finite values, tiny-subset sanity check passed at ratio 0.0002). Full write-up, six-question
  interpretation, and exact manifest in [[muq-to-dac-minimal-gate4]] and
  `data/manifests/gate4_crops.csv`. **Listening verdict (2026-09-09): "The listening confirms the
  poor result. The model is not learning and generalizing."** Metrics and listening agree cleanly
  here (unlike several earlier findings in this project where they diverged in either direction --
  see [[muq-to-dac-mlp]], [[muq-to-dac-temporal-conv]]): Gate 3's near-perfect result is confirmed
  to have been memorization, not a transferable MuQ-to-DAC mapping.
- Data-scaling experiment CONCLUDED (2026-09-09, `experiment/muq-to-dac-scale-data` branch, off
  `experiment/muq-to-dac-minimal-gate4`): tested whether Gate 4's generalization failure was
  caused by insufficient training data, using the exact Gate 3 architecture/hyperparameters
  unchanged on ~20x more training crops (320 vs. 16) and ~10x more unique training tracks (80
  across all 10 GTZAN genres vs. 8 across 4). **Result: cosine similarity improved meaningfully on
  held-out data (0.15-0.18 -> 0.22), and correct conditioning now shows a real, consistent edge
  over shuffled/time-shifted controls on held-out tracks (unlike Gate 4, where they were
  statistically indistinguishable) -- genuine evidence of a weak transferable signal that did not
  exist at Gate 4's scale. But waveform correlation to the DAC oracle did not improve at all,
  staying at the same noise-floor level (~0) as Gate 4.** Immediate overfitting persists
  (validation-MSE-selected checkpoint at step 300/20000, 1.5% through training). Conclusion: the
  MuQ-to-DAC mapping is **data-hungry, not fundamentally non-generalizable** (Gate 4's more
  extreme reading is now too strong) -- but a ~20x data increase, while producing a measurable
  latent-space effect, was nowhere near enough to produce audible-quality improvement. Full
  write-up in [[muq-to-dac-scale-data]]. Listening verdict pending as of this note.
- Further data-scaling experiment CONCLUDED (2026-09-09, `experiment/muq-to-dac-more-data`
  branch, off `experiment/muq-to-dac-scale-data`): tested whether held-out MuQ-to-DAC prediction
  keeps improving with ~5x more crops (1584 vs. 320) and ~10x more tracks (792 vs. 80, reusing
  Gate 1's own committed full-GTZAN split, `data/manifests/gtzan_manifest_smoke.csv`), same
  architecture/hyperparameters unchanged. **Result: cosine similarity barely moved (0.222/0.223
  -> 0.230/0.226, ~4-5% relative vs. the prior step's ~35% relative gain) and waveform correlation
  to the DAC oracle stayed at the same noise floor (~0) as every smaller scale -- a
  decelerating/saturating curve, not continued scaling.** Training itself was qualitatively new:
  a turbulent, unstable early phase (steps ~5000-32000, large but always-finite loss spikes) then
  a clean, late improvement phase, with the validation-MSE-selected checkpoint at **step 50000 of
  60000 (83.3% through training)** -- a dramatic change from Gate 4's 2.5% and the 80-track
  experiment's 1.5%, meaning immediate overfitting no longer occurs at this scale. But this later,
  less-overfit checkpoint did not translate into better ultimate generalization on the
  audio-relevant metric. **Final recommendation (from the observed 3-point scaling curve, not
  expectations): further pure data scaling within this exact architecture is unlikely to be
  productive -- the flat waveform-correlation trend across a ~100x crop-count range points to
  saturation (an architecture/receptive-field/information ceiling), not a data-limited
  bottleneck.** Full write-up and six-question interpretation in [[muq-to-dac-more-data]].
  Listening verdict pending as of this note.
- MuQ hidden-layer sweep CONCLUDED (2026-09-10, `experiment/muq-layer-sweep` branch, off
  `experiment/muq-to-dac-more-data`): tested whether an earlier/intermediate MuQ hidden layer
  predicts DAC latents better than the final layer (this investigation's baseline throughout).
  MuQ measured to have 13 `output_hidden_states` (index 0 = embedding output, indices 1-12 =
  each of 12 Conformer blocks; `hidden_states[12] == last_hidden_state`, confirmed via
  `torch.equal`); 5 candidates chosen by a pure function (`linspace(1,12,5)` rounded) = layers
  [1, 4, 6, 9, 12]; all share shape/channel-count/frame-rate, no adapter needed. Same
  architecture/optimizer/loss/split/manifest as [[muq-to-dac-more-data]], only the MuQ layer
  varied; 45000-step screening budget chosen and documented from that experiment's own curve
  before running anything else. **Result: the final layer (baseline) wins the primary ranking
  metric (validation MSE) outright -- lowest val AND test MSE of all five layers, no candidate
  close.** The earliest layer had a small (~10% relative) edge in cosine similarity and controls
  margin, but this didn't translate to a lower MSE or any waveform-domain gain -- waveform
  correlation to the DAC oracle stayed at ~0 for every layer, no exceptions. **Conclusion: MuQ
  layer choice does not resolve the saturation ceiling found in the data-scaling experiments --
  the final layer was never the bottleneck on this axis.** Per the experiment's own decision
  rule, no layer qualifies for a full follow-up run; architecture/context (receptive field,
  temporal modeling capacity) remains the leading unexplored hypothesis. A genuine, separately
  noteworthy finding: the final-layer control run itself was turbulent through most of its
  budget (val MSE spiking to 142 and 65 at similar step counts to the original more-data run's
  own turbulence), with its best-by-validation-MSE checkpoint landing on an early lucky dip
  (step 1500) rather than the eventual stable region -- a real property of this training regime's
  noise, not a bug, and worth remembering when interpreting any single "best checkpoint step" from
  this pipeline. Full write-up, comparison table, and seven-question interpretation in
  [[muq-layer-sweep]].
- Spectrogram-to-DAC positive control CONCLUDED (2026-09-14, `tasks/TASK-01.md`, code alongside
  `experiment/muq-layer-sweep`): tested whether the MuQ-to-DAC training pipeline itself (not
  MuQ) is the bottleneck, by feeding it a log-mel spectrogram (128 mels, hop aligned to DAC's
  75Hz grid, no repeat-interleave needed) through the exact unmodified `TemporalConvNet` (plus
  one required 128->1024 linear input projection) instead of MuQ's `last_hidden_state`. Same
  manifest/split/DAC-cache/optimizer/loss/step-budget as [[muq-to-dac-more-data]] (the current
  best MuQ-to-DAC model). **Result: latent-space metrics improved only modestly (val cosine 0.230
  -> 0.298, val MSE 17.29 -> 16.54) despite ~100x more raw temporal-spectral resolution in the
  input, while audio-relevant metrics did not improve at all -- multi-res STFT distance got
  worse (2.65 -> 3.02), and waveform correlation to the DAC oracle stayed at the same ~0 noise
  floor as every MuQ-based experiment.** New diagnostics (`latent_diagnostics.py`, predicted-vs-
  oracle on held-out data) found severe, undiminished norm collapse (ratio 0.36), variance
  collapse (ratio 0.11), and extreme over-smoothing (lag-1 autocorrelation 0.97 for predictions
  vs. 0.08 for the oracle) -- the same pathologies named in CLAUDE.md's own Scientific Motivation
  section, present in essentially unchanged form under a richer input representation.
  **Conclusion: the current pipeline (13-frame-receptive-field residual ConvNet, MSE+cosine
  regression directly onto DAC's raw continuous latent) cannot productively exploit a
  substantially richer representation either -- the training/objective formulation, not MuQ's
  information content, is now the leading suspect, and conclusions about MuQ's invertibility
  from this pipeline's waveform-quality failures remain premature.** Validation-MSE-selected
  checkpoint landed at step 1000/60000 (an extreme case of the "early lucky dip" pattern flagged
  in [[muq-layer-sweep]]); end-of-training (step 60000) numbers were close (val_mse 17.06, val_cos
  0.27), so this did not appear to be misleading here. Full write-up, comparison table, and
  interpretation in [[spectrogram-positive-control]]. 11 new tests (`tests/
  test_spectrogram_control.py`), full suite 73 passing. 30 listening WAVs written to
  `artifacts/spectrogram_control/`; listening verdict pending.
- Dense-audio-coverage rerun CONCLUDED (2026-09-14, `tasks/TASK-02.md`): tested whether
  [[spectrogram-positive-control]]'s result was caused by insufficient training data (only
  ~13.3% of GTZAN used, 1998 sparse crops) rather than the pipeline itself, by rebuilding the
  exact same experiment on ~100% of GTZAN's audio (14976 non-overlapping 2.0s crops, 792/94/113
  track-disjoint train/val/test tracks -- same split, 7.50x more crops) with the step budget
  scaled 60000 -> 450000 to preserve per-crop exposure. Representation, architecture, loss,
  optimizer held byte-for-byte identical to TASK-01. **Result: TASK-01 and TASK-02 are
  statistically indistinguishable.** Val MSE 16.54 -> 16.44, val cosine 0.298 -> 0.293, val STFT
  distance 3.02 -> 3.09 (marginally worse), waveform correlation to oracle stayed at the same ~0
  noise floor, and norm/variance collapse ratios moved slightly *further* from 1.0 (0.364 ->
  0.322 norm ratio, 0.110 -> 0.086 variance ratio) rather than toward it -- none of the five
  things a genuine data-coverage effect should move (per the task's own pre-registered Outcome A
  criteria) moved in the expected direction by any margin exceeding this pipeline's known
  run-to-run noise. **Conclusion: Outcome C -- increasing audio coverage does not meaningfully
  address the failure mode.** Combined with [[spectrogram-positive-control]], this rules out both
  "wrong/insufficient input representation" and "insufficient training data" as explanations for
  the norm-collapse/over-smoothing pathology across two independent single-variable experiments;
  the deeper issue is now most plausibly the target formulation / regression objective onto DAC's
  raw continuous latent, not representation or data scale. Best checkpoint again selected at step
  1000 -- this time of 450000, the third independent experiment in this investigation with this
  "early lucky dip" checkpoint-selection pattern ([[muq-layer-sweep]], TASK-01, this one).
  6 new tests (`tests/test_dense_manifest.py`), full suite 79 passing. 30 listening WAVs (same 6
  tracks as TASK-01) written to `artifacts/dense_spectrogram_control/`; listening verdict pending.
  Full write-up in [[dense-spectrogram-positive-control]].
- BlueHive/W&B instrumentation built and validated; early-checkpoint mechanism identified
  (2026-09-14, `tasks/TASK-03.md`): this environment has no SSH/sbatch access to BlueHive, so the
  actual cluster reproduction is handed off (`bluehive/README.md`, `bluehive/run_task03.slurm`) --
  not fabricated here. Built CUDA/MPS-agnostic device selection, gradient/latent/W&B diagnostics
  modules (`grad_diagnostics.py`, extended `latent_diagnostics.py`, `wandb_logging.py`), and
  `scripts/train_spectrogram_to_dac_wandb.py` (TASK-02's exact scientific formulation, plus
  instrumentation only -- documented deviations for cost control). Validated with two real local
  runs on this machine (MPS, NOT BlueHive): a disabled-mode dry run and a real online run against
  the user's actual W&B account (`wandb.ai/andrea-cogliati-university-of-rochester/muq-invertibility`,
  run `gz3uffcf`), both clean, zero errors. The online run's best checkpoint (step 1000,
  val_mse=16.442551) exactly matched TASK-02's own step-1000 result, confirming this pipeline is
  fully deterministic given the same seed/data/code. **Additional-investigation finding (evidence-
  based, `scripts/analyze_early_checkpoint_selection.py`, read-only over TASK-01/TASK-02's
  existing curves): the early-checkpoint phenomenon is regression-to-the-mean, not optimization
  instability or validation noise.** The first validation checkpoint (step 1000) is the GLOBAL
  minimum validation MSE across the entire run in BOTH TASK-01 (rank 1 of 60) and TASK-02 (rank 1
  of 450) -- never beaten again. A trivial "predict the dataset's mean DAC vector, ignore the
  input entirely" baseline achieves MSE within 6-7% of that "best" checkpoint in both runs
  (0.937x/0.933x). Validation MSE and cosine similarity are genuinely anti-correlated across
  checkpoints (r=-0.72 in both runs, excluding turbulence-spike outliers) -- TASK-02's single best
  cosine checkpoint (step 443000, near the end of training) has *higher* MSE than step 1000, so
  pure-MSE selection can never find it. **This is the same underlying mechanism as the norm/
  variance collapse documented in TASK-01/TASK-02: a near-constant prediction is simultaneously
  the MSE-cheap solution, the norm-collapsed solution, and the maximally over-smoothed solution,
  because for this target/loss combination those are the same solution.** 24 new tests, full
  suite 103 passing. Full write-up in [[task03-bluehive-wandb]].
- Spectrogram Gate-3 memorization control CONCLUDED (2026-09-15, `tasks/TASK-04.md`): tested
  whether Spectrogram->DAC can memorize two fixed training excerpts at all, on the exact same
  two clips (`pop.00012`, `disco.00049`) as the successful MuQ Gate-3 memorization control
  (`experiment/muq-to-dac-temporal-conv`, cosine 0.9999-1.0000). Same architecture
  (`SpectrogramToDacTranslator`), loss (MSE+0.1*cosine), and hyperparameters as that control
  (Adam lr=1e-3, 3000 steps, alternating presentation). **Result: Strong Success -- near-perfect
  memorization matching or exceeding the MuQ control** (best checkpoint, step 2536/3000: MSE
  0.000547, cosine 0.99999, norm ratio 0.9999, variance ratio 0.9998, waveform correlation
  0.99999, STFT distance 0.0145 -- vs. MuQ's own 0.001936 / 0.99996 / 1.0038 / 1.0073 / 0.99994 /
  0.0415). Both excerpts memorized evenly (1500/1500 presentations). **This decisively rules out
  data alignment, spectrogram preprocessing, the 128->1024 projection, translator implementation,
  target extraction, loss computation, gradient flow, checkpoint restoration, and DAC decoding as
  explanations for TASK-01/02's GTZAN-scale failure -- the pipeline works; the failure is
  generalization/scale-specific**, consistent with and reinforcing [[task03-bluehive-wandb]]'s
  regression-to-the-mean finding (collapse never emerged here: lag-1 autocorrelation tracked the
  oracle's own ~0.05 throughout, vs. ~0.95 at GTZAN scale -- collapse looks like a property of
  the multi-example training regime, not the architecture/loss in isolation). **Unplanned finding
  while building the required MuQ-comparison table**: the trained MuQ Gate-3 checkpoint is
  extremely sample-alignment-sensitive -- a 2-sample (83us) crop offset collapses its cosine
  similarity from 0.99998 to 0.577, meaning that "successful" memorization is a narrow,
  sample-exact overfit, not a phase-tolerant content-level fit. All 11 pre-training integrity
  checks passed (alignment, caching determinism, gradient reach, checkpoint reproducibility,
  etc.), run twice (disabled + online W&B modes) with byte-identical results, confirming
  determinism. 14 new tests, full suite 117 passing. Live W&B run: `wandb.ai/
  andrea-cogliati-university-of-rochester/muq-invertibility/runs/7kfiziym`. **Listening verdict
  (2026-09-15), confirmed by ear: "the two audio excerpts are already great at 1000 steps. Almost
  undistinguishable from the final version or the original. The early excerpts show signs of
  artifacts, though."** Metrics and perception agree here (unlike several earlier findings in
  this project) -- convergence is perceptually front-loaded (audible quality plateaus well before
  step 3000, consistent with cosine already at 0.98 by step 1000), and the still-artifacted early
  checkpoints match their still-far-from-converged metrics. Full write-up in
  [[task04-spectrogram-gate3]].
- Gate-3 training-set-size scaling curve CONCLUDED (2026-09-15, `tasks/TASK-05.md`): trained the
  exact TASK-04 pipeline at 5 nested training-set sizes (N=2/8/32/128/512, round-robin
  genre-balanced, all crops already in TASK-02's dense cache -- no new precompute) with a FIXED
  20000-step max budget across all levels (not scaled with N, a deliberate choice so per-example
  exposure decreases as N grows -- otherwise every example could trivially memorize regardless of
  N). **Result: a sharp collapse onset between N=8 and N=32, then asymptotic decay to the
  GTZAN-scale ceiling by N=512.** N=2 and N=8 are statistically indistinguishable (both
  near-perfect: norm ratio 1.00/1.00, waveform correlation 1.00/0.997). Between N=8 and N=32,
  every collapse metric moves by more than half its entire range in one step (norm ratio 1.0025
  -> 0.4993, variance ratio 1.0077 -> 0.1773, waveform correlation 0.997 -> 0.469). From N=32
  onward the decline decelerates (norm ratio 0.4993 -> 0.3778 -> 0.3434 at N=32/128/512),
  approaching an asymptote. **By N=512, results (MSE 16.02, cosine 0.327, norm ratio 0.343,
  waveform correlation 0.019) are already statistically indistinguishable from TASK-01's
  1584-crop and TASK-02's 11873-crop full GTZAN runs (MSE 16.47/16.39, cosine 0.295/0.291, norm
  ratio 0.36/0.32)** -- the entire collapse regime this investigation has characterized since
  [[muq-to-dac-scale-data]] is fully established by ~512 examples, 23-30x fewer than either
  GTZAN-scale experiment used. Neither pure Outcome A (gradual), B (sharp-to-final-state), nor C
  (no collapse) -- closest to B but with an asymptotic tail, consistent with a capacity/
  optimization boundary around 8-32 simultaneously-memorizable examples giving way to the
  regression-to-the-mean mechanism ([[task03-bluehive-wandb]]) once individual memorization is no
  longer achievable. Two required figures produced (`artifacts/task05_gate3_scale/
  scaling_curve_main.png`, `scaling_curve_norm_variance_ratio.png`). 13 new tests, full suite 130
  passing. **Listening verdict (2026-09-16), confirms and sharpens the metrics: "the clips for
  N=32 are closer to those for N=128 and N=512 than to those for N=8... the drop between N=8 and
  N=32 is dramatic."** By ear, N=32 already belongs to the "collapsed" perceptual family with
  N=128/512, not the "near-perfect" family with N=2/8 -- the audible category boundary matches
  the metric-located N=8-to-N=32 transition exactly, while the N=32-to-512 metric decline
  (waveform correlation 0.469 -> 0.019, >20x) is comparatively minor perceptually. Full write-up
  in [[task05-gate3-scaling]].
