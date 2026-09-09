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
- Reference HiFi-GAN experiment CONCLUDED (2026-09-08, `experiment/reference-hifigan` branch):
  vendored the original published `jik876/hifi-gan` (MIT license) generator/discriminators/
  losses/mel-preprocessing near-verbatim, trained on ground-truth mel spectrograms for the same
  two GTZAN clips, 15000 steps. Result: best mel-loss convergence of any experiment so far
  (final/initial ratio 0.174 vs. 0.36-0.52 elsewhere), 4-4.4x fewer generator parameters
  (13.9M vs. 51.7-61.6M) than this project's own MRF-1024 generators, stable training throughout.
  **But waveform correlation (0.073/0.122) landed in the same low band as this project's own
  MuQ (0.073/0.104) and ground-truth-mel (0.059/-0.010) results, not EnCodec's outlier
  (0.682/0.602).** A faithfully vendored, independently-authored reference implementation
  reproduced the same fidelity signature as this project's own custom pipeline under the same
  tiny-overfit budget. This substantially weakens the "our implementation has a bug" hypothesis
  and strengthens "the 2-clip/15000-step tiny-overfit training paradigm itself under-trains
  phase/waveform recovery, regardless of whose generator code is used" as the leading explanation.
  Full comparison table and four-question analysis in [[reference-hifigan]]. Listening verdict
  (2026-09-09): "slight improvement... especially on the disco song. However, the pop song still
  has noticeable artifacts" -- the first experiment in this investigation to produce *any*
  reported perceptual improvement, though partial and track-dependent. Matches the relative
  ordering of this experiment's own waveform correlation (disco.00049 0.1221 > pop.00012 0.0728).
  Next experiment will pursue a different angle (not yet specified) rather than further tuning
  this paradigm.
