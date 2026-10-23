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

- Device handling (2026-10-20, TASK-17): all backend calls live in `src/muq_inversion/device.py`;
  `--device auto` = CUDA > XPU > MPS, CPU only when explicit or `allow_cpu_fallback: true`.
  `PYTORCH_ENABLE_MPS_FALLBACK=1` is refused. Local torch 2.14.0 has `torch.xpu` (unavailable here).
- Intel XPU uses the separate uv project `envs/xpu/` (torch 2.14.0+xpu, torchaudio 2.11.0+xpu, torchvision 0.29.0+xpu, triton-xpu 3.8.0 from
  download.pytorch.org/whl/xpu; all other packages pinned to the canonical lock). Linux x86_64 only. PyPI's Linux torchaudio/torchvision link libcudart, so they must come from the XPU index too. Regenerate its constraint list when `uv.lock` changes.
- `scripts/task15_smoke_test.py --device X` passed on MPS and CPU (2026-10-20) and on XPU, ACES
  Intel Data Center GPU Max 1100, torch 2.14.0+xpu, 46.6 s (2026-10-21, run by the user). ACES needs
  >= 32G memory, `HF_HUB_OFFLINE=1` with MuQ pre-downloaded on a login node, and GTZAN copied over.

- TASK-18 (2026-10-21): MuQ's forward pass consumes torch RNG. Seed right before creating any
  trained module, or initialization depends on whether features were cached.
- Translator training on CPU is bit-reproducible only single-threaded (`OMP_NUM_THREADS=1`).
  Multithreaded CPU and MPS give ~1e-5..1e-4 run-to-run weight differences after 40 steps, so
  resume checks must use a measured noise floor.
- TASK-18 manifests: `data/manifests/task18_tracks.csv` (sha256 6a75bda9...) and
  `task18_crops.csv` (73f96d2e...). Throughput on MPS: 896 MuQ crops in 12 s; translator 500
  steps of batch 64 in ~20 s.

## Benchmark / gate results that passed sanity checks

- TASK-20 (2026-10-23, ACES 2246233): N=512, recipe A + early stopping (best 2,250, stop 4,250).
  Held-out mel L1 val/test/val2/test2 = 0.475/0.465/0.481/0.494; vs A128 +4-7% on each set (CI > 0).
  Memorized 0.44: the train/held-out gap is nearly closed. Not audible (blinded A128 = N512).
  Data returns diminish per 4x. See [[task20-n512-result]].

- TASK-19 (2026-10-22, ACES 2244922-5): held-out mel L1, primary/secondary:
  A32 0.550/0.572, A128 0.501/0.514, B32 0.554/0.576, B128 0.508/0.513.
  4x data gives a modest, consistent ~9-10% on both held-out sets (codes NEITHER/DATA_LIMITED
  are threshold labels, not conclusions). A warmup+cosine LR schedule has no effect. The val
  optimum is always at 1-2.5k steps, then it overfits; this coexists with poor absolute training
  fit (UNDERFIT label). The bottleneck is unresolved. Blinded listening: every arm recognizable
  (~2.75/4) and arms indistinguishable. On ACES each arm takes ~21-23 min (~3 billing-hours). See [[task19-data-x-recipe-result]].

- TASK-18 (2026-10-22, ACES job 2240151): MuQ -> BigVGAN mel, 32/8/8 tracks. Held-out mel L1 is
  val 0.576 / test 0.525, against shuffled_time ~0.87 and a round trip of ~0.11. Every control is
  beaten, and held-out audio is recognizable by the user's (unblinded) listening. Pre-registered
  code `UNDERFIT`: memorized mel L1 is 3.2x the round trip. Quality is limited mainly by
  translator fit and optimization (val best at step 1k, widening train/val gap, Adam spike at
  14k); a MuQ limit is not yet measurable. See [[task18-heldout-result]].

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
- N=32 compute-controlled micro-ablation CONCLUDED (2026-09-16, `tasks/TASK-06.md`): held N=32
  fixed (TASK-05's exact manifest/architecture/loss) and varied only step budget -- 20k (TASK-05's
  run reused unchanged, patience-stopped at 8000), 80k, 320k -- to separate "undertraining" from
  "genuine capacity/interference boundary" as explanations for the N=8->N=32 collapse. **Result:
  partial, not full, recovery -- and the key comparison is per-example repetition count, not raw
  steps.** 80k's best checkpoint (step 57000) already gives each of the 32 examples 1781
  repetitions -- 3x more than N=8's best checkpoint needed (594 reps); 320k's best checkpoint
  (step 88000) gives 2750 reps, 4.6x N=8's. Despite that, N=32 plateaus far below N=8: cosine 0.63
  / waveform correlation 0.67 / norm ratio 0.66 at 320k, vs. N=8's cosine 0.998 / waveform
  correlation 0.997 / norm ratio 1.003. Waveform correlation did rise substantially from the
  original 20k number (0.469 -> 0.629 -> 0.666), so compute is not irrelevant -- but returns are
  strongly diminishing (320k found no new global-best checkpoint past step 88000 despite 232000
  more steps of noisy oscillation). **Per-example ranking is stable and reveals preferential
  memorization, not uniform improvement**: `classical.00003` is the single worst example at both
  80k and 320k and does not improve with 4x more compute (0.315 -> 0.276, slightly worse), while
  `pop.00002` is consistently best and does improve (0.875 -> 0.897) -- the mean/median rise is
  driven by the easier subset of the 32 examples, not a uniform lift. Conclusion: the N=8->N=32
  boundary is best explained by genuine interference/capacity competition among examples, not
  compute and not regression-to-the-mean -- exposure explains part of the *original* fixed-budget
  collapse but not the residual gap to N=8, which persists even at 5x N=8's needed per-example
  exposure. Required figure: `artifacts/task06_n32_compute_ablation/
  per_example_waveform_correlation.png`. Full 133-test suite passing (1 new test file, 3 tests,
  `rank_examples_by_metric` added to `gate3_spectrogram_common.py`). Two live W&B runs under group
  `gate3-n32-compute-ablation`. Total compute: 80k run 6257s, 320k run 24249s (~6.7h). **Listening
  verdict (2026-09-17), diverges from the metrics for the first time in this investigation: "the
  best audio was generated at 80K steps. The audio from 320K is generally worse,"** despite 320k's
  higher mean waveform correlation (0.666 vs. 0.629 for 80k). Likely partly an artifact of which
  checkpoint the sent WAVs came from (320k's clips were its final-step checkpoint, a low point in
  that run's oscillation, not its best-by-MSE checkpoint at step 88000) and partly a reminder that
  this task's objective metrics are diagnostics, not proof of perceptual fidelity, especially in
  this collapse-adjacent regime. Full write-up in [[task06-n32-compute-ablation]].
- DAC shift/polarity sensitivity diagnostic CONCLUDED (2026-09-17, `tasks/TASK-07.md`): tested the
  frozen DAC encoder alone (no MuQ, no translator, no training) for sensitivity to small sub-frame
  input time shifts and global polarity flip, on 40 GTZAN val-split crops never used in any
  training set in this project. **Result: DAC's `quantized_representation` is extremely
  shift-fragile (mean cosine similarity 0.716 at a 1-*sample* shift, 0.157 by 16 samples --
  approaching but not fully reaching the random-pair floor of 0.030 -- and never fully reaching it
  even at 256 samples, ~10.7ms, still sub-frame relative to DAC's 320-sample/13.3ms hop), while the
  two candidate translator inputs are comparatively near-invariant at the same shifts (log-mel
  >=0.98, MuQ >=0.94, across the entire tested range up to 256 samples).** The pre-quantizer
  encoder output is somewhat more stable than the post-RVQ representation (0.945 at k=1 vs. 0.716)
  but collapses similarly by k=8-16 -- the fragility is substantially present in DAC's own
  continuous encoder, not purely an RVQ/codebook discretization artifact. **Polarity: `x` and `-x`
  are not merely dissimilar but *anti-correlated* (cosine similarity -0.435 quantized, -0.557
  pre-quantizer) -- further apart than random unrelated crops (0.030/0.036 baseline), not merely
  indifferent to sign.** This directly confirms and generalizes TASK-04's unplanned 2-sample
  (83us) MuQ-checkpoint finding: the fragility is intrinsic to DAC's own target, not an artifact of
  one trained model's narrow overfit, and it is not tracked by an equivalently shifted input --
  ruling out "the target is just following a shifted input" as an alternative explanation. This
  gives a structural, orthogonal-to-everything-else-ruled-out explanation for the collapse
  pathology this investigation has studied since [[muq-to-dac-scale-data]]: a magnitude-only/
  phase-blind translator input (log-mel, MuQ) cannot in principle determine this target uniquely
  at sub-frame precision, so MSE regression must average over the resulting ambiguity set. Per
  this task's explicit non-goal, **no conclusion drawn on whether to proceed with a complex-STFT
  rerun of TASK-05 -- that decision is the user's**. Read-only diagnostic: no training, no
  checkpoint, no cache/manifest modified. Figure: `artifacts/task07_dac_shift_sensitivity/
  shift_sensitivity.png`. 12 new tests, full suite 145 passing. Total run time: 27 seconds. Full
  write-up in [[task07-dac-shift-sensitivity]].
- DAC latent-vs-audio shift diagnostic CONCLUDED (2026-09-21, `tasks/TASK-08.md`, extends
  [[task07-dac-shift-sensitivity]]): measured whether TASK-07's latent-space shift fragility
  actually translates into reconstructed-audio instability, on the exact same 40-crop set, frozen
  DAC encoder-decoder only, no MuQ/translator/training. **Result: Outcome A, strongly supported --
  DAC latent similarity collapses by an order of magnitude (0.72 -> 0.09 cosine) across shifts
  1-1024 samples, while reconstructed-audio similarity stays essentially flat over that same
  range** (waveform correlation 0.955-0.964, log-mel cosine similarity 0.988-0.990, no
  discernible trend). Correlation analysis across 480 crop/shift points: Spearman rho between
  latent cosine similarity and audio similarity is only 0.23-0.35 across all three audio metrics
  (statistically real at this n, but a weak predictor practically) -- Figure 3 shows waveform
  correlation clustering tightly in a narrow 0.85-1.0 band across the *entire* 0-1.0 latent-cosine
  range, no visible slope. **A raw-waveform reference control (shift the pre-DAC audio, then
  align) reads exactly 1.0/1.0/0.0 at every tested shift, by construction** (an integer-sample
  shift is perfectly invertible by cropping, so the source signal carries zero inherent shift
  ambiguity) -- proving the small, shift-independent gap between DAC reconstruction and this
  perfect control is entirely attributable to DAC's own encode-decode process, not to any genuine
  ambiguity in "what the audio should be." **This strengthens and sharpens the target-mismatch
  explanation for TASK-01 through TASK-06**: DAC's latent target isn't merely fragile in a way
  inputs aren't (TASK-07) -- that extra fragility is largely reconstruction-irrelevant, meaning a
  shift-invariant translator input is penalized by MSE for sub-frame misalignments that barely
  affect what any of the three audio-domain metrics used here would call "different audio." Per
  this task's explicit instruction, stopped after answering its five required questions, no
  follow-up experiment proposed. SI-SDR not computed (no existing implementation in this project,
  task instruction is to reuse one rather than add a new dependency). 26 new tests, full suite 159
  passing. Total run time: 93 seconds. Listening package produced
  (`artifacts/task08_dac_latent_vs_audio_shift/listening/`), not yet reviewed by the user. Full
  write-up in [[task08-dac-latent-vs-audio-shift]].
- DAC Optimal Transport geometry diagnostic CONCLUDED (2026-09-22, `tasks/TASK-09.md`, extends
  [[task08-dac-latent-vs-audio-shift]]): tested whether comparing DAC latents as unordered point
  clouds (Optimal Transport / entropic Sinkhorn divergence, `POT` library, sqeuclidean ground
  cost, per-crop-scaled epsilon) predicts reconstruction-audio similarity better than the
  project's pointwise MSE/cosine metrics, on TASK-07/08's exact 40-crop shift-sweep dataset (no
  training, no decode -- audio-similarity numbers reused verbatim from TASK-08's report). **Result:
  OT is a consistently, moderately better predictor of audio similarity than MSE or cosine across
  every audio metric tested, but the improvement is not dramatic** -- |Spearman| correlation with
  waveform correlation: OT=0.521 vs. MSE=0.362 vs. cosine=0.229 (44-127% relative improvement,
  consistent across log-mel-cosine and STFT-distance comparisons too, 0.546/0.625 for OT vs.
  0.382/0.468 for MSE vs. 0.346/0.333 for cosine). **A second, partially independent finding**:
  OT's raw dynamic range across the shift sweep is much smaller than MSE's or cosine's and
  saturates almost immediately (by shift=4, vs. MSE/cosine continuing to drift gradually out to
  shift=1024) -- a genuine but partial Outcome-C-like signature: the *long-range* continued
  pointwise drift looks substantially transportable, but the *short-range* (shifts 1-4) initial
  collapse, the largest single move in every metric, is not explained away by OT at all. Per this
  task's own interpretation rules: **closest to Outcome B** ("the mismatch is deeper than
  framewise comparison"), **with a partial Outcome-C saturation signature confined to the
  long-range portion of the curve** -- neither fits cleanly alone. Strengthens the target-mismatch
  hypothesis with a specific refinement: loss geometry is a contributing cause (OT recovers a
  meaningfully better, though still far from strong, correspondence with audio similarity than the
  project's actual MSE+cosine training objective), but the residual gap (rho still only ~0.5-0.6)
  means geometry mismatch is unlikely to be the entire explanation. New dependency: `pot`
  (python-optimal-transport), added via `uv add pot` -- this task's own instruction explicitly
  permitted a new dependency here (unlike TASK-08's SI-SDR). Per this task's explicit instruction,
  stopped after answering its five required questions, no follow-up experiment proposed or run. 6
  new tests, full suite 165 passing. Total run time: 40.7 seconds (19.9s of which was the 880
  Sinkhorn calls themselves). Full write-up in [[task09-dac-ot-geometry]].
- EnCodec shift-sensitivity replication CONCLUDED (2026-09-22, `tasks/TASK-10.md`, replicates
  TASK-07+TASK-08 on `facebook/encodec_24khz` instead of DAC): tests whether DAC's
  latent-collapses/audio-stays-stable phenomenon is DAC-specific or general to neural codecs,
  same 40-crop dataset and 0-1024-sample shift schedule as TASK-07/08/09. **Note: `tasks/TASK-10.md`
  is truncated mid-sentence in its Final Questions section -- confirmed by direct byte
  inspection, flagged to the user rather than guessed past.** EnCodec: 24kHz, 75Hz frame rate
  (matches DAC exactly), 32 codebooks/1024 codebook size at max bandwidth (also matches DAC),
  128-dim continuous+quantized latent (much smaller than DAC's 1024-dim), decoder reproduces
  input length exactly (unlike DAC's 8-sample-shorter output). **Result: EnCodec's continuous
  latent is dramatically more shift-stable than DAC's (cosine 0.999->0.953 across the full shift
  range vs. DAC's 0.716->0.087) -- but this smoothness does NOT translate into more shift-robust
  audio: EnCodec's reconstructed-audio waveform correlation (~0.89 flat) sits measurably BELOW
  DAC's (~0.96 flat) throughout, despite the latent-stability gap.** A dramatically smoother
  latent space did not buy EnCodec a more shift-robust reconstruction -- latent movement and audio
  change are decoupled in both codecs, just at different points on the latent-stability axis.
  **A further nuance**: EnCodec's latent cosine correlates 1.6-2.1x MORE strongly with its own
  audio similarity than DAC's does (Spearman 0.36/0.60/-0.71 vs. DAC's 0.23/0.35/-0.33 across
  waveform-correlation/log-mel/STFT), despite varying over a much narrower absolute range --
  proportionally more informative, even though there's much less of it. **Mechanistic note**:
  EnCodec's *discrete* codebook indices collapse under shift about as fast as DAC's do (mean-32-
  stage agreement 0.115->0.0055) even though the *continuous* quantized latent does not -- the
  smoothness is specific to the continuous embedding geometry (likely a 128-dim-vs-1024-dim
  effect), not evidence EnCodec's underlying tokenization is intrinsically more shift-robust. Per
  this task's own interpretation rules: **none of Outcome A/B/C fits cleanly alone** -- closest to
  a combination of B (raw latent stability) and C (correlation strength), with the audio-
  robustness complication anticipated by neither. No new dependency (EnCodec ships in this
  project's pinned `transformers==4.55.0`); new module `src/muq_inversion/encodec_encoder.py`
  (`FrozenEncodecEncoder`, mirrors `dac_encoder.py`'s interface, verified once against EnCodec's
  own top-level API before trusting it, no dedicated unit test -- matching this project's existing
  convention for model-loading encoder wrappers). Listening package produced
  (`artifacts/task10_encodec_shift_sensitivity/listening/`), not yet reviewed. Total run time: 11.8
  seconds -- the fastest of the four DAC/EnCodec diagnostics. Full write-up in
  [[task10-encodec-shift-sensitivity]].
- EnCodec continuous-latent controlled scaling probe CONCLUDED (2026-09-22, `tasks/TASK-11.md`, a
  training experiment -- the first since TASK-06 in this line): tested whether TASK-05's sharp
  DAC N=8->N=32 collapse moves when the target codec changes to EnCodec's continuous latent, with
  literally everything else held fixed (same manifests, input, architecture except the mechanically
  -necessary `dac_channels=128` instead of 1024, loss, optimizer, 20000-step budget, seed -- source
  of truth: TASK-05's own report.json). The EnCodec translator has 98.3% *fewer* parameters
  (262,656 vs. 15,863,808) than TASK-05's DAC translator, ruling out a capacity-increase confound.
  **Result: the collapse is dramatically reduced, not just "improved" -- decisive support for the
  target-mismatch hypothesis this project's TASK-07-10 line has been building.** At N=32: DAC's
  norm/variance ratios collapse to 0.499/0.177 (classic collapse signature); EnCodec's stay at
  0.985/0.650 (essentially no norm collapse). Waveform correlation (pred vs. target-decode): DAC
  crashes from 0.997 (N=8) to 0.469 (N=32); EnCodec declines from 0.982 to 0.724 -- real, but
  roughly half DAC's drop. **Oracle-relative (this task's own required framing, since DAC and
  EnCodec have different reconstruction ceilings): EnCodec's translator recovers 79.2% of its own
  oracle ceiling at N=32 (0.724/0.914), vs. DAC's 48.4% (0.469/0.969, using a newly-fixed lag-
  aligned DAC oracle -- see below)** -- a relative, not just raw-number, advantage. Per-example
  N=32: EnCodec's LATENT fit is uniform across all 32 examples (0.98-0.99 band) unlike DAC's wide
  0.43-0.68 spread, but AUDIO-level per-example variation remains (0.37-0.98 waveform correlation)
  despite the uniform latent fit -- pointing to decoder-side/content-dependent variation, not a
  translator-side "easier subset" effect. **Two real bugs found and fixed during this task**: (1)
  DAC's decoder has a previously-undocumented fixed 8-sample group delay relative to its input --
  a naive lag-0 original-vs-oracle correlation gave ~0.37-0.47 (looked broken) until a
  cross-correlation lag search (`find_best_lag_and_align`) recovered the true ~0.96-0.97 ceiling;
  EnCodec has no equivalent delay (best lag 0); (2) EnCodec targets computed fresh are PyTorch
  "inference tensors" and need an explicit `.clone()` before use as autograd targets (TASK-05's
  DAC targets never hit this since they load from a cache as plain tensors). Per this task's own
  interpretation rules: **Outcome B** ("EnCodec substantially outperforms DAC but remains clearly
  below its own oracle... target geometry, optimization, and capacity/interference likely
  coexist") -- not Outcome A (EnCodec N=32 isn't near its own oracle or its own N=8 level) and not
  Outcome C (the two codecs' trajectories clearly differ in magnitude). Scope restricted to N in
  {2,8,32} per the task's own instruction; N=128/512 and any follow-up explicitly not run. 12 new
  tests, full suite 177 passing. Total training time ~5.8 minutes (all three levels). **Not
  committed** -- per this task's explicit "do not commit without explicit user instruction" rule;
  exact repository state (staged/unstaged/untracked) recorded in [[task11-encodec-controlled-scaling]]
  and `notes/NEXT.md`. Full write-up in [[task11-encodec-controlled-scaling]].
- EnCodec N=32 compute ablation CONCLUDED (2026-09-22, `tasks/TASK-12.md`, extends
  [[task11-encodec-controlled-scaling]]): tested whether TASK-11's residual N=32 gap (79.2%
  oracle-relative recovery at a 20k-step budget) is compute-limited, by training the identical
  setup independently to 80k and 320k steps (same seed/init, early stopping disabled, same
  rationale as TASK-06). **Result: Outcome B (Partial Response) -- compute helps substantially
  but with clearly diminishing returns, and a meaningful gap remains at 320k.** Oracle-relative
  recovery: 79.2% (20k) -> 88.0% (80k) -> 92.7% (320k); marginal gain in waveform correlation was
  +8.05 points for 20k->80k (4x compute) but only +4.30 points for 80k->320k (another 4x compute)
  -- roughly half the return for the same additional compute. Norm ratio never collapsed at any
  budget (compute-independent, confirming TASK-11); variance ratio climbed steadily (0.650 ->
  0.668 -> 0.738) but stayed 26 points short of 1.0 even at 320k. **Notable methodological
  finding**: variance ratio (Spearman rho=0.819 vs. audio quality, pooled across budgets) predicts
  audio-reconstruction quality far better than latent cosine similarity (rho=0.605) or latent MSE
  (rho=0.199), despite cosine being the metric closest to its own ceiling -- a concrete,
  actionable refinement of TASK-11's "high latent fit can coexist with audio variation" finding
  for any future diagnostic on this pipeline. **Per-example**: compute preferentially benefits
  examples that started worse (Pearson r=-0.85 between starting performance and subsequent gain)
  while broadly preserving relative rank (Spearman rho 0.94-0.97 across budget pairs) -- an
  equalizing, decelerating effect. One severe but transient instability spike occurred in the 320k
  run (step 268,000, fully recovered within one checkpoint interval, did not affect best-checkpoint
  selection). Both fixes carried over from TASK-11 (target `.clone()`, lag-aware DAC oracle
  policy) preserved without needing rediscovery. No new tests needed (fully reused TASK-05/06/10/11's
  unmodified infrastructure). Full required-template report at `artifacts/
  task12_encodec_n32_compute_ablation/TASK-12_RESULTS.md`. Total training time: 80k run 570.5s
  (~9.5 min), 320k run 2083.1s (~34.7 min) -- far faster than TASK-06's DAC equivalent (~6.7 hours
  for 320k), consistent with EnCodec's much smaller translator. **Not committed** -- per this
  task's explicit instruction; per this task's own instruction, no follow-up experiment proposed
  or run. Full write-up in [[task12-encodec-n32-compute-ablation]].
- Gate 4 revalidation on the merged (post-`experiment/muq-layer-sweep`) architecture CONCLUDED
  (2026-09-23, `tasks/TASK-13.md`, after `main` commit `a96fd4b` merged TASK-04-12's EnCodec
  pipeline): re-ran CLAUDE.md's Gate 4 checkpoint/resume + four-way conditioning-control protocol
  (`correct`/`shuffled_across`/`shuffled_time`/`constant_zero`) on `main`'s actual pipeline (MuQ ->
  repeat_interleave -> `SpectrogramToDacTranslator` -> EnCodec), evaluating all four conditions
  against one 20,000-step checkpoint trained with correct conditioning (`data/manifests/
  gate4_crops.csv` Experiment B, 16 train / 2 val / 2 test track-disjoint crops). **Result: PASSED
  by the pooled decision rule (correct 0.786 mean waveform correlation to oracle vs. -0.008/0.005/
  0.001 for the three controls; resume equivalence exact, 0.0000 relative diff; variance ratio
  1.080, no collapse) -- but this pass is a memorization artifact.** Split by data split: train
  (n=16) correct=0.981 vs. controls ~0 (clean separation, including `shuffled_time` -- resolving
  the old ambiguity on train data); held-out val+test (n=4) correct=0.002-0.007 vs. controls
  -0.024-to-0.028 (**no separation at all** -- correct conditioning is statistically
  indistinguishable from every control, including no conditioning). **This reproduces
  [[muq-to-dac-minimal-gate4]]'s "the Gate 3 result does not generalize, it was memorization"
  finding, now confirmed on the EnCodec-target pipeline too** -- the merged architecture didn't
  resolve the prior Gate 4 `shuffled_time` ambiguity, it relocated it (clean on memorized data,
  null on held-out data). `shuffled_time` is a genuine random frame permutation here (not the
  older pipeline's fixed circular shift), verified correct by 14 new unit tests. **Recommendation:
  do not treat this pooled PASS as license to proceed to Gate 5 without further held-out
  conditioning-sensitivity evidence** -- flagged as an open decision-rule question for the user
  (pooled- vs. held-out-only Gate 4 criteria), not resolved unilaterally. Held-out n is small (2
  val + 2 test) -- a real power caveat, though both splits independently show the same null
  result. Per this task's explicit instruction, Gate 5 not started. **Not committed** -- per this
  task's explicit instruction. Full write-up in [[task13-gate4-encodec-revalidation]].
- Gate 5 full-GTZAN pilot BlueHive package PREPARED, not run (2026-09-23, `tasks/TASK-14.md`):
  built and locally-verified (never executed at scale, per this task's explicit "do not run the
  full job locally / do not submit on BlueHive unless instructed" constraints) the infrastructure
  to test whether TASK-13's held-out null result was caused by too little training data (16
  crops). New: `configs/gate5_full_gtzan.yaml` (300,000-step full-run config, versioned) +
  `configs/gate5_smoke.yaml` (bounded diagnostic); `scripts/train_gate5_full_gtzan_wandb.py` with
  a deterministic resumable random-crop sampler (`gate5_common.TrainCropSampler`) drawing broadly
  across all 792 training tracks (genuinely random start offsets, not a small fixed rotation);
  `data/manifests/gate5_heldout_crops.csv` (188 val + 226 test crops, fixed/versioned, derived
  from `more_data_crops.csv`'s existing split -- no new crop-selection decision). Held-out
  evaluation reuses TASK-13's `gate4_encodec_common.py` manifest loader and all four conditioning
  controls completely unmodified (identical CSV schema). **Primary endpoint is
  `oracle_relative_waveform_recovery`** (original-vs-translator / original-vs-oracle correlation)
  -- deliberately different from TASK-13's translator-vs-oracle metric, per this task's explicit
  "oracle-relative waveform recovery" specification. **Verified, not just built**: ran the bounded
  diagnostic locally three times, including a genuine partial-train-to-step-10-then-resume-to-
  step-20 run that reproduced bit-identical final metrics to an uninterrupted run -- confirms the
  checkpoint/sampler/RNG/W&B-run-lineage round-trip actually works. `wandb_logging.init_run`
  gained a backward-compatible `resume_id` parameter so resumed runs reopen the same W&B run
  (previously every resume created a disconnected new one -- a real gap this task's own explicit
  "same scientific run lineage" requirement caught and fixed). 24 new tests, full suite 215
  passing, lint clean. SLURM package (`bluehive/run_task14.slurm` for the full job,
  `run_task14_diagnostic.slurm` optional BlueHive-side check, `resubmit_task14.sh`) syntax-
  validated but **not submitted** -- this environment still has no BlueHive access. **Known
  placeholder**: the 16-hour wall-clock budget is extrapolated from the local per-step rate
  (~116ms/step), not a real BlueHive measurement -- correct after the first submission. TASK-13's
  own files untouched. **Not committed** -- per this task's explicit instruction; no Gate 5
  scientific conclusion drawn. Full write-up in [[task14-gate5-full-gtzan-pilot]].
- Gate 5 full-GTZAN pilot RESULT (2026-09-23/24, BlueHive job `1956835`, submitted by the user;
  logs + `report.json` + 36 WAVs in `bluehive/logs/` and `bluehive/artifacts/`, git-ignored):
  **Outcome C -- measurable coarse-structure generalization without recognizable audio
  reconstruction; Gate 5 NOT PASSED.** Held-out signal statistically reliable; correct
  conditioning materially affects coarse spectral-energy trajectories; temporal order contributes
  but less than track identity; waveform recovery negligible on average; listening does not
  support recognizable source correspondence. Not B (no practically meaningful or audibly
  convincing effect). Do not say TASK-14 found nothing; do not call the reconstruction
  successful. Log-mel similarity on held-out crops: correct 0.73 vs.
  shuffled_across 0.35-0.39, shuffled_time 0.65-0.66, constant_zero ~0; STFT distance 2.56 vs.
  2.8-3.9. Correct beats shuffled_across on 99-100% and shuffled_time on 98-99% of held-out
  crops, in all 10 genres. Waveform primary endpoint (oracle-relative recovery): correct
  0.0285 val / 0.0315 test vs. controls 0.007-0.017, CIs exclude zero, 74-93% of examples
  positive, but deltas 0.011-0.025 miss the 0.05 materiality bar; every condition is at ~1-3% of
  the EnCodec oracle (0.91); plateaued by ~step 6,000. Best checkpoint step 38,000. Variance ratio
  0.27/0.29 (over-smoothed, not collapsed); latent cosine 0.945-0.975 even with zero conditioning
  (uninformative). **The script's automatic rule printed "Outcome D" because it considers only the
  waveform metric and the 0.05 threshold -- superseded, do not cite it** (an early log-only
  write-up repeated D; corrected). TASK-13's data-scale hypothesis is partly supported (data fixed
  the spectral level, not the waveform level). **Listening (user, 2026-09-24, 6 blues crops, not
  blinded): all `correct` outputs bad; only test_blues.00005_c0 vaguely resembles the source, and
  its controls sound similar.** Contour check on the same WAVs: loudness/brightness contour r with
  the original = 0.78/0.68 for `correct` vs. ~0 for every control -- the envelope transfers, but
  that alone isn't recognizable. Listening selection is now
  worst/median/best per split by the primary metric (was first-N-in-sort-order = all blues);
  `scripts/export_gate5_listening.py` regenerates it for this run once `best.pt` is copied from
  BlueHive -- a robustness check, not a likely route to passing Gate 5. W&B architecture tag
  corrected to `muq-to-encodec-temporal-conv` for future runs (was the stale class name
  `spectrogram-to-dac-translator`). Still open: outcome rule is waveform-only, no `wandb.Audio`. Full write-up in
  [[task14-gate5-full-gtzan-result]].
- TASK-15 BigVGAN-v2 memorization test (2026-10-16, local M4 Max/MPS; separate from EnCodec work):
  **MuQ -> BigVGAN-v2 mel -> frozen BigVGAN-v2 is mechanically viable at small N.** Official
  `nvidia/bigvgan_v2_24khz_100band_256x` @ `c329ede9...` in git-ignored `third_party/` via
  `scripts/fetch_bigvgan.py`; runs on MPS (112.4M params). Mel: 24 kHz, 100 bands, n_fft/win
  1024, hop 256 (93.75 Hz), fmin 0, fmax null, `log(clamp 1e-5)`; track peak-normalized to 0.95.
  Crops 47,872 samples (187 hops) -> MuQ 50 frames; translator interpolates 50 -> 187
  (1,271,908 params). Oracle: resynthesis mel L1 0.109 (unrelated pairs 1.32); waveform corr to
  original ~0.14 is a phase artifact. N=2 passes all pre-registered criteria (mel L1 0.023 vs.
  oracle round-trip ~0.11, pred-vs-oracle corr 0.95/0.98). N=8: 6/8 pass; 2 miss only
  pred-vs-oracle corr >= 0.90 (0.898, 0.888, still rising at the 5k-step budget end); mel
  memorized within vocoder round-trip on all 8. Shuffled-across fails everywhere. Adam + L1 at
  constant LR spikes near the minimum (N=2's final step jumped 0.025 -> 0.41) -- evaluate the best
  checkpoint too. **Listening (user, 2026-10-16): success at N=2 and N=8**, including the two
  near-miss crops -- viable at both N. Not a generalization result. Full write-up in
  [[task15-bigvgan-memorization]].
- TASK-16 (2026-10-19): the two TASK-15 audit corrections -- MuQ->mel interpolation at physical
  frame centers (MuQ token i @ 960·i, BigVGAN frame j @ 256·j+128, derived from the loaded models
  in `time_alignment.py`, impulse-verified) and a decode-time clamp to BigVGAN's log floor
  (log(1e-5), from `spectral_normalize_torch`) -- **don't materially change the TASK-15 N=2
  result**: both crops pass all criteria at best step 4,750; metric changes ≤ ~0.02 in both
  directions; clamp raises 1.5-2.4% of bins. TASK-15 config (switches off) reproduces TASK-15
  bit-for-bit. W&B `fjh80gyn`. Write-up: `artifacts/task16_bigvgan_corrected/TASK-16_RESULTS.md`;
  [[task16-bigvgan-alignment-and-floor]].
