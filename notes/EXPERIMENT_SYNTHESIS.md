# MuQ Invertibility Investigation: Experiment Synthesis

Date: 2026-09-10. Author: Claude (synthesis pass, no training performed). Scope: every branch,
commit, tag, config, manifest, cache, checkpoint, and JSON report reachable in this repository as
of `experiment/muq-layer-sweep` (uncommitted work included). This document does not modify any
prior report, config, checkpoint, or artifact.

## Core research question

> Can the final temporal representation of MuQ be inverted to reconstruct music audio?

The investigation considered two formulations, in this order:

**Formulation 1 — direct waveform reconstruction:**
`MuQ representation -> learned waveform decoder -> audio`

**Formulation 2 — representation translation:**
`MuQ representation -> learned translator -> DAC representation -> frozen DAC decoder -> audio`

### Why the project moved from Formulation 1 to Formulation 2

Formulation 1 was pursued first (`main` branch, Gates 0-3, then four representation-swap
controls: EnCodec, ground-truth mel, a reference HiFi-GAN implementation). Across every variant —
capacity scaled 3.57x, optimizer instability fixed, representation source swapped to EnCodec, the
target changed to ground-truth mel, and the decoder itself replaced with a faithfully vendored
published HiFi-GAN — the same pattern recurred: **objective spectral metrics improved
substantially while perceived audio quality stayed "mediocre" or "poor" in every case except
one** (EnCodec's 6-9x waveform-correlation jump, which nonetheless still sounded indistinguishable
from the rest by ear). This repeatedly falsified the working assumption that the direct-decoder
approach's problem was capacity, optimizer settings, representation source, or implementation
fidelity. Waveform-sample correlation stayed near zero (0.06-0.12) for every Formulation-1 variant
built on a magnitude-domain training objective, with EnCodec's outlier (0.60-0.68) the single
exception — and even that sounded no better.

The pivot to Formulation 2 was motivated by a specific, falsifiable idea: instead of training a
decoder to invert MuQ into raw waveform samples (a domain where phase must be hallucinated from
a magnitude-only training signal), train a small translator to predict the latent space of a
codec (DAC) that was *itself* trained end-to-end for reconstruction, and reuse that codec's own
frozen decoder. This sidesteps the phase-hallucination problem entirely (DAC's decoder already
solves it, by construction) and turns the question into a more tractable one: can MuQ's
representation predict a reconstruction-native latent? The first test of this
(`experiment/muq-to-dac-temporal-conv`) produced cosine similarity 0.9999 and listening judged
"almost perfect" — the first unambiguous positive result in the entire investigation — which is
why the project's center of gravity shifted to Formulation 2 for everything that followed.

---

## Master experiment table

Ordered chronologically. "Track" column: H = Formulation-1 (HiFi-GAN-family, direct waveform),
D = Formulation-2 (DAC-latent translation). Full per-experiment detail (architecture, exact
metrics, confounds) is in the **Experiment cards** section immediately below this table — this
table is a navigation aid, not the complete record.

| # | Experiment | Track | Branch (commit) | Key result | Listening | Status |
|---|---|---|---|---|---|---|
| E1 | Initial small decoder, reconstruction-only | H | `main` (`afe7895`) | Combined loss 7.51→3.18 (500 steps) | not recorded at this step | superseded |
| E2 | Reconstruction-only revisited (buzzing diagnosis) | H | `main` (`38d9507`) | Loss plateaued 500→5000 steps (2.78→2.64); mel corr. 0.89-0.91; wav corr. ~0 | "buzzing noise... nothing remotely recognizable" | diagnosed: magnitude-only loss can't constrain phase |
| E3 | Adversarial + feature-matching losses added | H | `main` (`38d9507`) | Recon-loss ratio 0.40 (PASS); peak amplitude 0.76→0.92-0.99 | "improved, but still very poor: noise, buzzing, ringing" | improved but not resolved |
| E4 | Original small residual generator | H | `main` (pre-`38d9507`) | Thin sequential receptive field, capacity-starved final layer (8 channels) | (superseded before separate listening) | diagnosed as under-capacity |
| E5 | Proper MRF generator (rewrite) | H | `main` (`38d9507`) | Mel loss 0.59→0.48; mel corr. 0.90/0.94→0.94/0.94 | sent, pending | replaced E4 |
| E6 | MRF-512 (stable baseline) | H | `main`, `configs/smoke.yaml` | Mel corr. 0.924/0.932; wav corr. -0.015/0.078 | "equally mediocre" (paired with E9) | stable |
| E7 | MRF-1024 (collapsed) | H | `main`, `configs/smoke_1024.yaml` (pre-fix) | Mel corr. 0.312/0.228; wav corr. NaN (constant output) | not judged (collapsed run) | **collapsed — confound, not a clean capacity result** |
| E8 | Adam-beta correction | H | `main` | Betas `(0.9,0.999)`→`(0.8,0.99)` fixes collapse | n/a (infra fix) | fix adopted project-wide |
| E9 | Stable MRF-512 vs MRF-1024 comparison | H | `main` (`38d9507`) | Mel corr. 0.965/0.976 (1024) vs 0.924/0.932 (512); wav corr. 0.073/0.104 vs -0.015/0.078 | "I can't hear any noticeable improvement... equally mediocre" | **capacity ruled out** |
| E10 | EnCodec-conditioned HiFi-GAN control | H | `encodec-hifigan-experiment` (`6fb17fe`) | Mel corr. 0.950/0.970; **wav corr. 0.682/0.602 (6-9x jump)** | "all similarly mediocre" (grouped with E9, E14) | **largest metric effect in Formulation 1, zero perceptual effect** |
| E11 | Ground-truth mel-conditioned custom generator | H | `experiment/mel-hifigan-sanity` (`bb34dbf`) | Mel corr. 0.944/0.966; wav corr. 0.059/-0.010 (behaves like MuQ, not EnCodec) | "mediocre again... almost identical to the other experiments" | raised conditioning-collapse question (never resolved) |
| E12 | Reference/published HiFi-GAN | H | `experiment/reference-hifigan` (`4979e2c`) | Best mel-loss ratio of any Formulation-1 run (0.174); wav corr. 0.073/0.122 | "slight improvement, especially on disco... pop still has noticeable artifacts" | **first partial perceptual improvement in Formulation 1**; implementation-defect hypothesis weakened |
| E13 | Framewise MuQ-to-DAC MLP | D | `experiment/muq-to-dac-mlp` (`98dee72`) | Cosine sim. ceiling 0.6418 (exact, verified architectural floor); wav corr. 0.303/0.192 (2nd-best of whole investigation) | "recognizable but noticeably worse than previous experiments" — despite better metric | numerically best-yet, perceptually worse: sharpest waveform-correlation-is-unreliable evidence to that point |
| E14 | Two-excerpt temporal-ConvNet MuQ-to-DAC overfit | D | `experiment/muq-to-dac-temporal-conv` (`fbb1fb5`, tag `muq-to-dac-temporal-conv-v1`) | Cosine sim. **0.9999-1.0000**; wav corr. to oracle **0.9999/1.0000** | "almost perfect reconstruction! We nailed it!" | **first unambiguous positive result in the investigation** |
| E15 | Gate 4 (minimal), unseen crops from known tracks | D | `experiment/muq-to-dac-minimal-gate4` (`b9f82ef`), Experiment A | Held-out cosine 0.179/0.184 (vs. 0.9999 memorized); wav corr. ~0 | "poor result. The model is not learning and generalizing" | **E14 was memorization, not generalization** |
| E16 | Gate 4 (minimal), unseen tracks | D | same commit, Experiment B | Held-out cosine 0.150/0.140; wav corr. ~0 (no worse than E15) | same verdict as E15 | unseen-track collapse indistinguishable from unseen-crop collapse |
| E17 | 80-track / 320-crop data scaling | D | `experiment/muq-to-dac-scale-data` (uncommitted at write time, later committed `f26728e`) | Held-out cosine 0.222/0.223 (up from 0.15-0.18); wav corr. still ~0 | not recorded (superseded before verdict given) | real but weak transferable signal emerges; controls gap becomes real on held-out data |
| E18 | Full-GTZAN / 1,584-crop data scaling | D | `experiment/muq-to-dac-more-data` (`f26728e`) | Held-out cosine 0.226/0.230 (marginal gain over E17); wav corr. still ~0 | not recorded (superseded before verdict given) | **saturating, not continuing, scaling curve** |
| E19 | MuQ hidden-layer sweep [1,4,6,9,12] | D | `experiment/muq-layer-sweep` (uncommitted at write time) | Final layer (12) wins val/test MSE outright; layer 1 wins cosine by ~10% but not MSE or wav corr. | not generated (best candidate == baseline; see spec's own rule) | **MuQ layer choice does not explain the ceiling** |
| E20 | Gate 4 (HiFi-GAN-family) short pilot, `shuffled_time` control | H | `main` (`95f333e`), `configs/pilot.yaml` | `shuffled_time`/correct loss ratio 1.11-1.13 (below the 20% `control_margin`); `shuffled_across`/`constant_zero` 20-45% worse | n/a (loss-based control, not a listening test) | conditioning weak-but-present; foreshadows E11's later, sharper version of the same question |

Gates 0-2 (environment doctor, GTZAN data integrity/manifest/deduplication, MuQ feature-contract
extraction and caching) are infrastructure milestones, not hypothesis tests, and are not given
table rows; they are the foundation every experiment above depends on (GTZAN split, MuQ frame-rate
measurement, cache format).

---

## Experiment cards

Full field-by-field detail for every experiment. "Not recorded" is used literally where the
repository does not contain the value — never inferred.

### E1-E3: Initial decoder, reconstruction-only, then adversarial (Gate 3, HiFi-GAN family v1)

- **Branch/commit:** `main`, `afe7895` (Gate 3 implementation) → `38d9507` (adversarial losses added, same branch).
- **Input representation:** MuQ `last_hidden_state`, 25 Hz, 1024 channels.
- **Target representation:** raw 24 kHz waveform (48,000 samples / 2.0s crop).
- **Architecture:** `MiniGenerator` — original design, sequential residual blocks (not MRF) per
  upsampling stage, `upsample_rates=(5,4,4,3,2,2)` (6 stages), `base_channels=256`.
- **Parameter count:** not recorded for this exact configuration (superseded before a parameter
  count was logged; the next architecture, E5's MRF rewrite, is the first with a recorded count).
- **Receptive field:** not recorded quantitatively; qualitatively diagnosed as too thin ("two
  small residual blocks in sequence... right after a strided `ConvTranspose1d`").
- **Dataset/split:** GTZAN, 2 fixed overfit tracks (`pop.00012`, `disco.00049`), no train/val/test
  split (pure memorization test).
- **Tracks:** 2. **Crops:** 1 fixed 2.0s (50 MuQ-frame) crop per track.
- **Training steps:** 500 (E1) → 5000 (E2, diagnostic re-run) → 8000 (E3, with adversarial warmup=500).
- **Objective functions:** E1/E2: mel-spectrogram L1 + multi-resolution STFT (magnitude-only,
  reconstruction-only). E3 adds LSGAN adversarial + feature-matching losses after 500-step warmup.
- **Optimizer:** Adam, lr=2e-4 (E1/E2). E3: separate generator/discriminator Adam, betas not yet
  corrected at this point in the timeline (this predates E8's fix; **whether E3's own 8000-step
  run itself used default or corrected betas is not unambiguously recorded** — the beta-collapse
  was discovered later, on the 1024-channel capacity experiment; treat E3's exact optimizer
  state as a minor unresolved provenance gap).
- **Validation-selection rule:** none (single fixed-crop overfit, no held-out split).
- **Best checkpoint step:** final step in each run (500 / 5000 / 8000) — no checkpoint selection.
- **Latent MSE / cosine similarity:** not applicable — this experiment predates the DAC-latent
  formulation; only waveform-domain losses and correlations were tracked.
- **Waveform correlation to oracle:** not applicable (no oracle concept yet; "oracle" as used
  later means the DAC codec's own reconstruction, which doesn't exist in Formulation 1). Waveform
  correlation to the *original reference* was ~0 throughout.
- **STFT metrics:** E3 final: recon-loss ratio 0.40 (PASS, threshold 0.5).
- **Listening result:** E2: "buzzing noise... nothing remotely recognizable." E3: "improved, but
  still very poor: a lot of noise, buzzing, and ringing."
- **Engineering result:** stable training, no collapse; a real pass/fail-metric bug found and
  fixed (gating on combined adversarial+recon loss instead of reconstruction-only loss).
- **Scientific interpretation:** magnitude-only losses cannot constrain phase; more steps alone
  does not fix this (500→5000 steps moved loss only 4%). Adversarial training helps but does not
  resolve perceived quality at this capacity.
- **Confounds:** generator capacity, loss formulation, and training budget were all still
  entangled at this stage — this is exactly why E4→E9 exist as controlled follow-ups.

### E4-E5: Original small residual generator → MRF rewrite

- **Branch/commit:** `main`, `38d9507`.
- **Diagnosis leading to the change:** user's listening pass after E3 reported "ringing/comb-
  filter artifacts," diagnosed as a generator-capacity ceiling, not a training-dynamics problem:
  (1) sequential (not parallel) residual blocks after each upsample stage; (2) 6 upsample stages
  each halving channel width from `base_channels=256` down to 8 channels at the final layer.
- **Architecture change:** rewrote `models/generator.py` — `ResBlock` now runs multiple dilated
  conv pairs `(1,3,5)` sequentially within one block; new `MultiReceptiveFieldFusion` runs 3 such
  blocks **in parallel** (kernel sizes `(3,7,11)`) and averages their outputs (the actual MRF
  mechanism, matching the published HiFi-GAN design). `upsample_rates` refactored from 6 stages
  `(5,4,4,3,2,2)` to 4 stages `(8,6,5,4)` (product unchanged, 960), `base_channels` raised 256→512.
- **Parameter count:** 17,235,585 (this is `configs/smoke.yaml`'s MRF-512 baseline, E6).
- **Training steps:** 15000 (full run after the rewrite).
- **Result:** mel loss 0.59→0.48; log-mel correlation `pop.00012` 0.90→0.94, `disco.00049`
  0.94→0.94 (held). Per-step time ~0.095s (up from ~0.07s pre-rewrite).
- **Listening result:** sent to user for a third listening pass; verdict folded into E6/E9 below.
- **Confounds:** this change bundled two things at once (MRF mechanism + upsample-stage/channel
  refactor) since both addressed the same diagnosed root cause — not a single-variable isolation
  by itself, though it set up the clean single-variable E6-vs-E7/E9 comparison that followed.

### E6-E9: MRF-512 baseline, MRF-1024 collapse, Adam-beta fix, stable comparison

- **Branch/commit:** `main`, `38d9507`; configs `configs/smoke.yaml` (512) and
  `configs/smoke_1024.yaml` (1024).
- **Input representation:** MuQ `last_hidden_state`, 25 Hz, 1024 ch. **Target:** raw 24kHz waveform.
- **Architecture:** MRF generator (from E5), `upsample_rates=(8,6,5,4)` (product 960),
  `resblock_kernel_sizes=(3,7,11)`, `resblock_dilation_sizes=((1,3,5),(1,3,5),(1,3,5))`.
  Discriminators: `MultiPeriodDiscriminator` (periods `2,3,5,7,11`), `MultiScaleDiscriminator`
  (3 scales), both capped at 256 channels (below the original speech HiFi-GAN's 1024-wide design).
- **Parameter count:** 512-ch: 17,235,585. 1024-ch: 61,582,593 (3.57x).
- **Receptive field:** not quantitatively recorded (MRF design; qualitatively larger than E4 by
  construction, not measured in frames).
- **Dataset/split:** GTZAN, 2 fixed overfit tracks, no held-out split (memorization test, same as
  E1-E5).
- **Training:** 15000 steps, `crop_frames=50` (2.0s), `learning_rate=2e-4`,
  `adversarial_warmup_steps=500`, `adversarial_loss_weight=1.0`, `feature_matching_loss_weight=2.0`,
  `reconstruction_loss_weight=10.0`.
- **E7 (1024-ch, pre-fix):** `generator_betas`/`discriminator_betas` at PyTorch's Adam default
  `(0.9, 0.999)`. **Result: collapsed.** Mel loss 8.078 (vs. 0.478 for the healthy 512-ch run at
  the same step), mel correlation 0.312/0.228 (down from a pre-collapse snapshot of 0.945/0.950 at
  step 3000), waveform correlation **NaN (constant generator output)**.
- **E8 (Adam-beta fix):** changed both optimizers' betas to `(0.8, 0.99)` — HiFi-GAN's own
  published AdamW config, and a betas choice GAN literature specifically recommends over
  `beta1=0.9` for adversarial minimax training.
- **E9 (stable 512 vs. 1024, post-fix):** both configs re-run with corrected betas, no collapse
  in either.
  - MRF-512: mel corr. 0.924/0.932, wav corr. -0.015/0.078, checkpoint 250.6MB, ~31 min wall time.
  - MRF-1024: mel corr. **0.965/0.976**, wav corr. **0.073/0.104**, checkpoint 782.8MB, ~52 min.
- **Listening result (E9):** "I can't hear any noticeable improvement... equally mediocre:
  recognizable but still plagued by artifacts" — despite 1024 winning every objective metric.
- **Scientific interpretation:** capacity is **ruled out** as the bottleneck for Formulation 1 —
  a 3.57x parameter increase produced the era's largest metric gain with zero perceptual change.
  This is the first clean demonstration in the investigation of the "metrics improve, ears don't"
  pattern that recurs through E10-E13.
- **Confounds:** E7's collapse is not a valid capacity data point on its own (confounded by
  optimizer instability) — E9 is the valid, clean comparison; E7 is retained in the table only as
  a documented negative/confounded result, per CLAUDE.md rule 8 (never overwrite/discard a
  documented finding).

### E10: EnCodec-conditioned HiFi-GAN control

- **Branch/commit:** `encodec-hifigan-experiment`, `6fb17fe`.
- **Input representation:** `facebook/encodec_24khz` continuous pre-quantization embedding
  (`model.encoder(...)`, NOT the quantized RVQ codes), 75 Hz, 128 channels.
- **Target representation:** raw 24kHz waveform.
- **Architecture:** same MRF-1024 generator/discriminator design as E9, with the one unavoidable
  adaptation: `upsample_rates` changed `(8,6,5,4)`→`(8,5,4,2)` (product 960→320, matching
  EnCodec's measured 320 samples/frame) and `crop_frames` 50→150 (same 2.0s/48000-sample crop,
  recounted in the new frame units). All losses, both Adam betas `(0.8,0.99)`,
  `adversarial_warmup_steps=500`, `steps=15000` identical to E9's 1024-ch config.
- **Parameter count:** 54,799,617 (smaller than E9's 61.58M — narrower input projection since
  EnCodec is 128-dim vs. MuQ's 1024-dim).
- **Dataset:** same 2 fixed tracks, no held-out split.
- **Training:** 15000 steps, ~62 min wall time (3x longer feature sequence than MuQ's 50 frames).
- **Waveform correlation:** **0.682/0.602** — 6-9x higher than E9's 0.073/0.104, the largest,
  cleanest effect size of any Formulation-1 experiment.
- **Mel correlation:** 0.950/0.970 (comparable to, marginally below, E9's 0.965/0.976).
- **Listening result:** "I can't tell any difference... they are all similarly mediocre" (grouped
  with E9 and E1's baseline in the same listening pass).
- **Scientific interpretation:** the largest objective-metric movement in the whole
  investigation produced zero perceptual movement — generalizing E9's "capacity doesn't explain
  it" finding to "representation source (between these two options) doesn't explain it either."
- **Confounds:** this single swap changed representation source, frame rate (25→75Hz, 3x), AND
  channel width (1024→128, 8x) simultaneously — it cannot on its own separate "EnCodec's
  information content is richer" from "EnCodec's 3x-denser time sampling helps" from "EnCodec's
  training objective (reconstruction) vs. MuQ's (self-supervised understanding) is what matters."
  E11 (ground-truth mel) was designed specifically to help separate these.

### E11: Ground-truth mel-conditioned custom generator

- **Branch/commit:** `experiment/mel-hifigan-sanity`, `bb34dbf`.
- **Input representation:** ground-truth log-mel spectrogram of the target audio itself
  (`torchaudio.transforms.MelSpectrogram`, manual reflect-pad + `center=False`), 80 Hz, 128 ch
  (`n_fft=2048, hop_length=300, n_mels=128`).
- **Target representation:** raw 24kHz waveform (the same waveform the mel was computed from).
- **Architecture:** same MRF-1024 design, `upsample_rates=(5,5,4,3)` (product 300, matching the
  mel's measured 300 samples/frame), `crop_frames=160` (160*300=48000, same 2.0s crop).
- **Parameter count:** 51,670,273.
- **Training:** 15000 steps, ~48 min, same losses/betas/schedule as E9/E10.
- **Result:** mel correlation 0.944/0.966 (comparable to E9/E10); **waveform correlation
  0.059/-0.010** — behaves like MuQ (E9), not like EnCodec (E10), despite being, by construction,
  at least as information-rich about its own target as any representation could be.
- **Listening result:** "The reconstructed audio sounds mediocre again. Almost identical to the
  other experiments. There must [be] something wrong with the training pipeline."
- **Scientific interpretation:** this is the pivotal control that separated the two confounded
  explanations E10 left open. Ground-truth mel and MuQ are both magnitude-domain-derived,
  self-supervised-or-analysis representations that never encoded phase; EnCodec is trained
  end-to-end with a paired decoder whose objective is literally accurate reconstruction. Ground-
  truth mel behaving like MuQ (not EnCodec) on waveform correlation reframes the axis from
  "MuQ vs. EnCodec" to "magnitude-only vs. reconstruction-trained representations."
- **A hypothesis raised here and never fully resolved:** the user's listening judgment that even
  mel-conditioned reconstruction is indistinguishable from every other variant (despite the
  metric gap to EnCodec) raised the possibility that the generator might not be using its
  conditioning meaningfully at all — a conditioning-collapse hypothesis connected to E20's earlier
  `shuffled_time` finding. **A concrete diagnostic for this was designed (compare generator output
  under correct/zero/shuffled/cross-track conditioning using already-trained checkpoints) but was
  never executed** — the user redirected to summarizing and committing instead. This remains an
  open item (see notes/NEXT.md across several branches).
- **Confounds:** none new beyond E10's; this experiment's entire value is as a control isolating
  one further variable from E10.

### E12: Reference/published HiFi-GAN

- **Branch/commit:** `experiment/reference-hifigan`, `4979e2c`.
- **Purpose:** test whether the mediocrity was caused by this project's own
  implementation/training setup, by substituting a faithfully vendored, independently published
  HiFi-GAN (`jik876/hifi-gan`, MIT license, NeurIPS 2020) for the custom generator/discriminator/
  loss code.
- **Input representation:** ground-truth mel via the *reference implementation's own*
  preprocessing (`librosa.filters.mel`, not `torchaudio`), 86.13 Hz, 80 channels
  (`n_fft=1024, hop_size=256, win_size=1024`).
- **Target representation:** raw 22050Hz waveform (the reference's native sample rate — a
  documented deviation from this project's 24kHz convention, since GTZAN's native rate is
  22050Hz and no resampling was needed for this specific experiment).
- **Architecture:** vendored `Generator`/`MultiPeriodDiscriminator`/`MultiScaleDiscriminator` from
  the original repo, `upsample_rates=(8,8,2,2)`, `upsample_initial_channel=512`.
- **Parameter count:** **13,936,130** — 4-4.4x fewer than any of this project's own MRF-1024
  generators.
- **Training:** 15000 steps, `segment_size=44032`, mel-loss weight 45, no warmup (adversarial
  from step 1), Adam betas `(0.8,0.99)` (the reference's own published config, used natively).
- **Result:** best mel-loss convergence ratio of any Formulation-1 experiment (**0.174**, vs.
  0.36-0.52 elsewhere); waveform correlation 0.073/0.122 — landing in the same low band as E9/E11,
  not EnCodec's outlier band.
- **Listening result:** "There's a slight improvement in the reconstructed audio, especially on
  the disco song. However, the pop song still has noticeable artifacts." **The first experiment
  in the entire Formulation-1 investigation to produce any reported perceptual improvement**,
  though partial and track-dependent (matching the relative ordering of this run's own waveform
  correlation: disco 0.1221 > pop 0.0728).
- **Scientific interpretation:** a faithfully vendored, independently authored reference
  implementation reproduces the same fidelity signature as this project's own custom generator —
  substantially weakening (not eliminating) "our implementation has a bug" as the explanation,
  and strengthening "the shared tiny-overfit training paradigm" (2 clips, 15,000 steps, no
  dataset-scale discriminator exposure) as the more likely bottleneck. The disco-specific
  improvement is not explained; possibly the reference architecture's specific inductive biases
  interact better with disco's spectral/rhythmic content than pop's more percussive/transient
  material, but this was not further investigated.
- **Confounds:** sample rate (22050 vs. 24000Hz elsewhere), mel filterbank implementation
  (librosa vs. torchaudio), and architecture were all changed together as a deliberate package
  (this experiment's whole point was to test "someone else's complete, unmodified recipe," not to
  isolate one sub-variable) — so E12 cannot itself distinguish which of these specific changes,
  if any, produced the partial disco improvement.

### E13: Framewise MuQ-to-DAC MLP

- **Branch/commit:** `experiment/muq-to-dac-mlp`, `98dee72`.
- **Input representation:** MuQ `last_hidden_state`, 25 Hz, 1024 ch, nearest-neighbor repeated
  3x to align with DAC's 3x-higher frame rate.
- **Target representation:** DAC `quantized_representation` (see the mandatory DAC target audit
  below for the full trace), 75 Hz, 1024 ch.
- **Architecture:** `FramewiseMLP` — `Linear(1024,2048) -> GELU -> Linear(2048,2048) -> GELU ->
  Linear(2048,1024)`, applied identically and independently per frame (no temporal mixing).
- **Parameter count:** 8,393,728. **Receptive field:** 1 frame (by construction — purely
  pointwise).
- **Dataset:** same 2 fixed tracks, no held-out split (memorization test).
- **Objective:** MSE + 0.1×(1-cosine similarity). **Optimizer:** Adam, lr=1e-3, betas
  `(0.9,0.999)` (no adversarial training, so the GAN-collapse-driven beta fix does not apply).
- **Training:** 3000 steps (converged by ~150).
- **Result:** cosine similarity converges to an **exact, verified architectural ceiling of
  0.6418** — proven (not just observed) to equal the theoretical best-constant-predictor floor
  for a memoryless map fed 3 identical repeated inputs per 3-frame output group (computed floor:
  11.04/11.54 MSE per track, matching the trained model's converged MSE of ~11.05 almost exactly).
- **Waveform correlation to oracle:** 0.303/0.192 — 2nd-best of the entire investigation at the
  time, ahead of every Formulation-1 variant except EnCodec.
- **Listening result:** "Reconstructed audio still recognizable but noticeably worse than
  previous experiments. DAC oracle is perfect." — despite numerically beating three of four
  Formulation-1 variants on waveform correlation.
- **Scientific interpretation:** (1) the framewise ceiling is an architectural artifact of the
  3x frame-rate mismatch, not evidence about MuQ's information content; (2) the DAC oracle
  sounding perfect while scoring only 0.37/0.57 correlation to reference is the sharpest evidence
  yet that raw waveform-sample correlation is close to useless as a perceptual proxy in this
  project — a codec judged perceptually perfect scored *below* what this experiment's own
  worse-sounding reconstruction needed to be called good.
- **Confounds:** none beyond the architectural ceiling itself, which was independently verified
  (not assumed) via the constant-predictor floor calculation and an empirical receptive-field
  perturbation test in the follow-up experiment (E14).

### E14: Two-excerpt temporal-ConvNet MuQ-to-DAC overfit ★ tagged `muq-to-dac-temporal-conv-v1`

- **Branch/commit:** `experiment/muq-to-dac-temporal-conv`, `fbb1fb5`.
- **Input/target:** identical to E13 (MuQ 25Hz→repeated 3x, DAC `quantized_representation` 75Hz).
- **Architecture:** `TemporalConvNet` — 3 residual `Conv1D(k=5)` layers, `x + GELU(conv(x))`
  residual connections, operating on the repeated 75Hz sequence.
- **Parameter count:** 15,731,712. **Receptive field:** 13 frames at DAC's 75Hz timeline =
  5 MuQ frames (~200ms) — **verified both analytically** (`1 + 3×(5-1) = 13`) **and empirically**
  (perturbation test: a single perturbed input frame changed exactly 13 output frames).
- **Optimizer/objective:** identical to E13 (Adam lr=1e-3, betas `(0.9,0.999)`, MSE + 0.1×cosine).
- **Training:** 3000 steps; 100-step smoke test already reached cosine 0.9976.
- **Result: cosine similarity 0.9999-1.0000, MSE ~0.004-0.014** — essentially a full latent-space
  solve. Decoded waveform correlates with the DAC-oracle-decoded waveform (not just the
  reference) at **0.9999/1.0000**.
- **Listening result:** "The listening test shows almost perfect reconstruction! We nailed it!"
- **Scientific interpretation:** the framewise MLP's ceiling was a limitation of its memoryless
  architecture, not of MuQ's information content — MuQ contained the missing information all
  along, accessible with ~200ms of local temporal context. **First unambiguous positive result
  in the whole investigation.**
- **Confounds:** this is a 2-clip memorization test, structurally identical in scope to every
  Gate-3-style HiFi-GAN experiment before it. Whether this generalizes was explicitly unknown at
  this point and untested until E15/E16.

### E15-E16: Gate 4 (minimal), unseen crops (E15) and unseen tracks (E16)

- **Branch/commit:** `experiment/muq-to-dac-minimal-gate4`, `b9f82ef`, branched off E14's branch
  (architecture/hyperparameters unchanged, per the task's explicit constraints).
- **Architecture/optimizer/objective:** identical to E14 (`TemporalConvNet`, 15,731,712 params,
  Adam lr=1e-3 betas `(0.9,0.999)`, MSE + 0.1×cosine) — trained **fresh** per experiment (not
  E14's checkpoint), a documented, deliberate deviation: reusing E14's checkpoint would make
  Experiment A's "unseen crop" evaluation non-blind and wouldn't apply to Experiment B's
  different tracks at all.
- **Dataset (E15, "Experiment A"):** same 2 tracks as E14 (`pop.00012`, `disco.00049`), split
  into train (8 crops/track, starts 0-14s), val (2 crops/track, 17-19s), test (2 crops/track,
  22-24s), with a 1.0s guard band between every region (verified programmatically at
  manifest-build time). 16/4/4 train/val/test crops.
- **Dataset (E16, "Experiment B"):** 8 track-disjoint train tracks (2 each from
  pop/disco/classical/rock) / 2 val / 2 test tracks, no track in more than one split (verified by
  checksum). 16/2/2 train/val/test crops.
- **Training:** 8000 steps each.
- **Checkpoint selection:** validation MSE only (both experiments).
- **Best checkpoint step:** **200 of 8000 (2.5%) for both experiments** — essentially the start
  of training.
- **Train/val/test cosine similarity:** E15: train 0.452, val 0.179, test 0.184. E16: train
  0.395, val 0.150, test 0.140.
- **Train/val/test latent MSE:** E15: train 15.14, val 19.66, test 20.42. E16: train 15.54, val
  18.88, test 18.63.
- **Waveform correlation to oracle:** E15: train 0.329, val 0.006, test -0.009. E16: train 0.251,
  val -0.005, test -0.001 — held-out performance collapses to noise level in both.
- **Controls (correct vs. shuffled/shifted cosine), val:** E15: correct 0.179 vs. shuffled
  0.132/shifted 0.115 — barely distinguishable. On train data, correct 0.452 clearly beats
  shuffled 0.154/shifted 0.118 — a real, large effect confined to memorized examples.
- **Listening result:** "The listening confirms the poor result. The model is not learning and
  generalizing."
- **Scientific interpretation:** E14's near-perfect result was memorization of two specific
  crops, not evidence of a transferable MuQ-to-DAC mapping. Unseen tracks (E16) are no worse than
  unseen crops from known tracks (E15) — an all-or-nothing collapse, not a graded
  same-track-helps-a-little effect.
- **Confounds:** none of note; this is a clean, well-controlled pair of experiments. One
  methodological nuance worth flagging: the stop-condition check for "does training reproduce
  Gate 3 overfit behavior" was initially (incorrectly) applied to the full 16-crop training set
  rather than a tiny 1-crop subset, producing a false alarm that was caught and fixed before
  results were trusted (see notes/SCRATCHPAD.md for this branch).

### E17: 80-track / 320-crop data scaling

- **Branch/commit:** `experiment/muq-to-dac-scale-data` (code and results existed at
  investigation time; the branch's commit `f26728e` bundles E17 and E18 together, see note below).
- **Architecture/optimizer/objective:** unchanged from E14/E15/E16.
- **Dataset:** 80 train tracks (all 10 GTZAN genres, 8/genre) / 10 val / 10 test, 4 crops/train
  track + 2 crops/val-or-test track → 320/20/20 crops.
- **Training:** 20000 steps (up from Gate 4's 8000 — documented deviation for the larger crop
  count).
- **Best checkpoint step:** 300 of 20000 (1.5%).
- **Train/val/test cosine:** train 0.243, val 0.222, test 0.223.
- **Train/val/test MSE:** train 16.27, val 17.28 (best), test 17.96 / 20.42 depending on split
  bookkeeping (**exact test MSE for the A/B experiment split of Gate 4 vs. this single-pool
  80-track design differs in structure** — see the Comparability limitations section).
- **Waveform correlation to oracle:** val -0.0045, test -0.0043 — unchanged from Gate 4's ~0.
- **Controls, val:** correct 0.222 vs. shuffled 0.144/shifted 0.157 — a real, consistent gap that
  **did not exist in Gate 4** (where correct and controls were statistically indistinguishable on
  held-out data).
- **Listening result:** **not recorded** — audio was sent to the user, but the user moved directly
  to requesting the next (full-GTZAN) experiment without providing a listening verdict. This is
  an explicit, acknowledged gap in the evidence trail, not an oversight in this synthesis.
- **Scientific interpretation:** a genuine, if weak, transferable signal now exists that did not
  exist at Gate 4's scale — refines the working hypothesis from "not generalizable" (Gate 4's
  reading) to "data-hungry."
- **Confounds:** the `sed`-based smoke-test-before-full-run methodology used in this and the next
  experiment accidentally ran the *complete* budget once during what was intended as a quick
  timing check (a trailing-comment issue broke the `sed` pattern's anchor) — the resulting run was
  adopted as the official result since it was scientifically identical to an intentional full run,
  but the run label (`scale_data_timing` in some artifact filenames) is a cosmetic mismatch worth
  knowing about if grepping filenames.

### E18: Full-GTZAN / 1,584-crop data scaling

- **Branch/commit:** `experiment/muq-to-dac-more-data`, `f26728e`, branched off E17's branch.
- **Architecture/optimizer/objective:** unchanged from E14-E17.
- **Dataset:** reused Gate 1's own committed, checksum-verified GTZAN split
  (`data/manifests/gtzan_manifest_smoke.csv`, 793/94/113 tracks) minus the 1 corrupted file →
  792 train / 94 val / 113 test tracks, 2 crops/track (starts 5.0s, 15.0s) → 1584/188/226 crops.
- **Training:** 60000 steps (up from E17's 20000).
- **Best checkpoint step:** **50000 of 60000 (83.3%)** — a dramatic change from every prior
  experiment's ~1.5-2.5%; validation MSE increased monotonically through a turbulent early phase
  (steps ~5000-32000, val MSE spiking to 30-142) before stabilizing and improving from ~33000 on.
- **Train/val/test cosine:** train 0.257, val 0.230, test 0.226.
- **Train/val/test MSE:** train 19.82, val 17.29 (best), test 17.18-17.26 (see per-split
  `metrics_by_split` in `artifacts/more_data/report.json`).
- **Waveform correlation to oracle:** val 0.0002, test -0.0009 — statistically zero, unchanged
  from E15-E17.
- **Controls, val:** correct 0.230 vs. shuffled 0.123/shifted 0.149 — a gap roughly comparable to,
  not clearly larger than, E17's.
- **Listening result:** **not recorded** — same gap as E17; the user proceeded to the layer-sweep
  request (E19) without giving this experiment's verdict either.
- **Scientific interpretation:** cosine similarity improved only marginally over E17 (~4-5%
  relative, vs. E17's ~35% relative gain over Gate 4) despite a much larger further data
  increase — a decelerating curve. Waveform correlation to the oracle showed **zero** movement
  across all three data scales tested (16→320→1584 crops). Characterized in this synthesis as
  practical saturation under the current architecture (see the Scaling analysis section for the
  precise, hedged claim).
- **Confounds:** the same accidental-full-run-during-smoke-test provenance note as E17 applies
  here too (see `notes/SCRATCHPAD.md` for this branch's full account). Also: this run's
  "train MSE/cosine" curve values were computed via a running window average over recent steps,
  not a full-training-set pass at every checkpoint (a deliberate, documented efficiency change
  from E15-E17's methodology) — comparable in spirit but not identical in computation to earlier
  experiments' train-curve numbers; see Comparability limitations.

### E19: MuQ hidden-layer sweep [1, 4, 6, 9, 12]

- **Branch/commit:** `experiment/muq-layer-sweep` (uncommitted at the time of this synthesis).
- **Architecture/optimizer/objective:** unchanged from E14-E18 in every respect except the
  selected MuQ hidden layer.
- **MuQ structure, measured directly:** `output_hidden_states=True` returns 13 tensors; index 0
  is the frontend/embedding output (before any transformer block, confirmed via
  `len(model.model.conformer.layers) == 12`); indices 1-12 are the outputs of each of the 12
  Conformer blocks; `hidden_states[12]` verified byte-identical (`torch.equal`) to
  `last_hidden_state`, the representation used in every experiment above. All 13 layers share
  shape `(1, 50, 1024)` — same channels, same frame rate, no adapter needed anywhere in the sweep.
- **Candidate selection:** `linspace(1, 12, 5)` rounded to nearest int (Python banker's rounding)
  = **[1, 4, 6, 9, 12]**, chosen by a pure, unit-tested function, not by eye.
- **Dataset:** identical to E18's manifest and split, reused unchanged.
- **Training budget:** 45000 steps (chosen and documented from E18's own curve *before* running
  any layer but the final-layer control, per the task's explicit ordering requirement).
- **Final-layer control-run sanity check:** re-ran layer 12 at the reduced budget; it was itself
  turbulent for most of its run (val MSE spiking to 142.3 at step 10000, 65.5 at step 15000,
  reproducing E18's own turbulence signature) before settling to 18.6 by step 45000. Its
  best-by-validation-MSE checkpoint came from **step 1500** — an early, fortuitous dip before the
  long turbulent stretch, not the eventual stable region. Passed the sanity-check bounds (val
  MSE≤25, val cosine≥0.15: actual 17.34, 0.221) so the sweep proceeded.
- **Full comparison** (val MSE / val cosine / test MSE / test cosine / wav corr. to oracle):

  | layer | idx | val MSE | val cos | test MSE | test cos | val wav corr. |
  |---|---|---|---|---|---|---|
  | earliest | 1 | 17.8052 | **0.2443** | 17.6505 | **0.2395** | 0.0004 |
  | early-middle | 4 | 17.5662 | 0.2301 | 17.4594 | 0.2255 | 0.0000 |
  | middle | 6 | 17.9254 | 0.2160 | 17.8065 | 0.2130 | -0.0013 |
  | late-middle | 9 | 17.6987 | 0.2260 | 17.5615 | 0.2233 | 0.0004 |
  | **final** | **12** | **17.3382** | 0.2208 | **17.2589** | 0.2154 | 0.0001 |

- **Listening result:** **not generated.** The validation-selected best candidate (by MSE, the
  spec's primary ranking metric) *is* the final layer already used as the baseline — there is no
  distinct second condition to blind-test against it.
- **Scientific interpretation:** the final layer wins the primary ranking metric outright on both
  splits. The earliest layer has a small (~10% relative) cosine-similarity edge that does not
  translate into a lower MSE or any waveform-domain gain — waveform correlation to the oracle is
  statistically zero for every one of the five layers, no exceptions.
- **Confounds:** none identified. This is the cleanest single-variable-isolation experiment in
  the whole investigation (verified shape/frame-rate identity across all candidates removed the
  need for any adapter or normalization change that could have confounded the comparison).

---

## Comparability limitations

Explicit audit, per the task's required checklist. Read this section before comparing any two
numbers across experiments in this document.

1. **Metrics were NOT calculated with identical code across the whole investigation.**
   Formulation-1 experiments (E1-E12) computed mel correlation, waveform correlation, and STFT
   loss via `src/muq_inversion/models/losses.py`'s `MelSpectrogramLoss`/`MultiResolutionSTFTLoss`
   plus ad hoc `numpy.corrcoef` calls in each experiment's own script. Formulation-2 experiments
   (E13-E19) use a different, evolving set of scripts (`gate4_common.py`'s
   `waveform_correlation`, `models/losses.py`'s `MultiResolutionSTFTLoss` reused for evaluation
   only). The *formula* for waveform correlation (Pearson correlation via `numpy.corrcoef`) is
   consistent within Formulation 2, but Formulation-1-to-Formulation-2 comparisons additionally
   cross a metric-implementation boundary, not just a representation boundary.

2. **"Waveform correlation" means different things across experiments and this matters a lot.**
   - E1-E12 (Formulation 1): correlation between the reconstructed waveform and the **original
     reference** recording.
   - E13-E19 (Formulation 2): correlation is reported against **two different baselines**
     depending on context — "wav corr. to oracle" (predicted-latent-decoded audio vs.
     true-latent-decoded/DAC-oracle audio) is the metric used for held-out generalization
     analysis from E15 onward; "wav corr. to reference" (DAC-oracle vs. the original recording)
     is a separate, always-reported "ceiling" number. **These two are not interchangeable and
     this document labels every value with which one it is.** Conflating them was a live risk
     flagged explicitly in this project's own notes (E13's write-up: "the DAC oracle sounds
     perfect despite waveform correlation of only 0.37/0.57 [to reference]" — a warning that
     reference-relative correlation is a poor proxy even for a codec everyone agrees sounds
     correct).

3. **DAC targets are consistently `quantized_representation` (post-RVQ, pre-decoder) across all
   of E13-E19** — verified identical across every branch's `dac_encoder.py` (each branch
   recreated the file independently, since branches don't share commits in this project's
   workflow, but the content is byte-for-byte the same design decision each time). See the
   mandatory DAC target audit below for the full trace and what this specifically is (not
   pre-quantizer, not summed-but-different, not raw codes).

4. **Crops compared are NOT always the same crops across experiments.** E1-E12 and E13-E14 use
   the identical, hand-picked 2.0s crop from `pop.00012`/`disco.00049`. E15-E16 use disjoint crop
   windows from those same two tracks. E17-E19 use entirely different GTZAN tracks. **No cross-
   experiment comparison in this document treats E1-E12/E13-E14's numbers as directly comparable
   to E15-E19's held-out numbers on the same footing** — the former are memorization results, the
   latter are generalization results, and this document's own tables/discussion keep them in
   separate rows/paragraphs rather than one merged ranking.

5. **Reconstructed audio was not normalized/loudness-matched in any experiment reviewed.** No
   normalization step was found in any of the WAV-writing code paths (`_write_wav` in every
   training script writes the raw decoded/reference float32 array directly via `soundfile.write`,
   no peak or loudness normalization). This means any perceptual "louder = better/worse" bias in
   listening judgments is a genuine, uncontrolled confound across every experiment's listening
   verdicts — not something this synthesis can correct for retroactively.

6. **MuQ features came from the same checkpoint (`OpenMuQ/MuQ-large-msd-iter`) throughout** —
   verified via `grep` for `MUQ_MODEL_ID`/`model_id` across every script and config found; no
   experiment used a different MuQ checkpoint.

7. **MuQ hidden-state indexing was consistent only from E19 onward, because it was only measured
   starting at E19.** E1-E18 all use `last_hidden_state` (`output_hidden_states=False`), which
   E19 later confirmed is byte-identical to `hidden_states[12]`. This is a consistent choice in
   retrospect, but E1-E18 did not themselves verify or document the indexing convention — E19's
   verification closes that gap after the fact.

8. **Crop alignment respected 25Hz MuQ / codec frame rates in every Formulation-2 experiment**,
   verified via an explicit runtime assertion in every training script
   (`assert muq_frames * frame_ratio == dac_frames`, or the equivalent EnCodec/mel check in
   Formulation 1). No experiment in the inventory shows evidence of a silent frame-misalignment.

9. **Parameter counts reported in every experiment card above are `sum(p.numel() for p in
   model.parameters())` on the trainable translator/generator only** — they do **not** include
   frozen MuQ, frozen DAC (encoder or decoder), or frozen EnCodec parameters, and for the
   HiFi-GAN-family experiments (E1-E12) they do **not** include discriminator parameters (the
   discriminators are separately optimized and were not included in any "generator parameters"
   figure quoted in this project's own notes). Checkpoint *file sizes* quoted in several
   experiment cards, by contrast, typically **do** include discriminator weights and both
   optimizers' full Adam state — checkpoint size and parameter count are not simply related by a
   constant factor across experiments and should not be used to infer relative model complexity.

10. **Checkpoint selection used validation-only criteria from Gate 4 (E15) onward, by explicit
    design.** E1-E14 have no held-out split and therefore no checkpoint-selection rule to audit
    (final-step checkpoints were used throughout). E15-E19 all select by validation MSE only,
    verified structurally in E19's case (`CheckpointRecord` carries no test-related field at
    all — a type-level guarantee, not just a coding convention) and by direct code inspection for
    E15-E18.

11. **No run was found to have silently overwritten an earlier checkpoint or artifact.** Every
    experiment writes to a distinctly named `run_name`/`RUN_ID` directory under the gitignored
    `runs/`/`artifacts/` trees, and CLAUDE.md rule 8 (never overwrite a previous run) was checked
    against directory naming in every script inspected. One near-miss is documented in this
    synthesis's E17/E18 entries: an accidental full-budget run occurring during what was intended
    as a shortened smoke test was *renamed* into the canonical result location after manual
    verification that it was a scientifically valid full run — this was a deliberate, logged
    action, not a silent overwrite, but it does mean those two experiments' artifact filenames
    contain a `_timing` label that doesn't reflect what the run actually was.

12. **MPS nondeterminism is a documented, real effect in this project** (see
    `notes/DECISIONS.md#gate4-mps-resume-tolerance`: resume-equivalence drift measured at roughly
    0.00025 relative loss difference per training step on this hardware/PyTorch version). No
    experiment in this inventory reports a difference small enough that MPS nondeterminism alone
    could plausibly be the explanation (the smallest meaningful claimed differences — e.g., E19's
    layer-vs-layer val MSE gap of ~0.02-0.6 absolute — are larger than what step-count-scaled MPS
    drift would predict at these training-step counts, but this was not independently
    re-verified by running any experiment twice under otherwise identical conditions, so it
    remains an assumption, not a proven bound).

13. **Failed, collapsed, or turbulent runs are explicitly NOT compared at face value against
    stable runs anywhere in this document.** E7 (1024-ch collapse) is retained only as a
    documented negative result, never used as a valid capacity data point (E9 is). E18 and E19's
    turbulent early-training phases are described qualitatively, and their *selected* (stable-
    region-passing, or at minimum sanity-check-passing) checkpoints are what's reported in every
    quantitative table — but readers should note that E19's final-layer control run's "best"
    checkpoint (step 1500) came from a lucky pre-turbulence dip, not the stable region other
    experiments' best checkpoints came from. This is flagged again in the Training-dynamics audit
    section.

---

## Chronological causal analysis

For each required pivot: hypothesis → intervention → result → what it ruled out → what it did
not rule out → why the next experiment followed.

### Pivot 1: Additional training steps (E1→E2)

- **Hypothesis:** the reconstruction-only decoder just needs more training to stop sounding like
  buzzing noise.
- **Intervention:** 500 → 5000 steps (10x), same everything else.
- **Result:** combined loss moved only 2.78→2.64 (4% over 9x more steps) — a plateau.
- **Ruled out:** "insufficient training steps" as the explanation for E1's poor audio, at this
  capacity/objective.
- **Did not rule out:** whether a fundamentally different objective (adversarial) or larger
  capacity would still need many more steps to show a benefit.
- **Why the next experiment followed:** with steps ruled out and mel/energy correlation already
  good (0.89-0.91, 0.98), the remaining candidate was the objective function itself — motivating
  Pivot 2.

### Pivot 2: Adding adversarial losses (E2→E3)

- **Hypothesis:** magnitude-only losses (mel, multi-res STFT) cannot constrain phase, and this
  — not capacity or training length — is why audio sounds buzzy despite good spectral-envelope
  metrics.
- **Intervention:** add multi-period + multi-scale discriminators and LSGAN + feature-matching
  losses after a 500-step reconstruction-only warmup, per CLAUDE.md's own Decoder Design item 6.
- **Result:** recon-loss ratio 0.40 (PASS); peak amplitude closer to reference (0.76→0.92-0.99).
  Listening: "improved, but still very poor: noise, buzzing, ringing."
- **Ruled out:** pure reconstruction-only training as sufficient on its own; confirmed phase
  incoherence was a real, at-least-partially-addressable issue (dynamic range improved).
- **Did not rule out:** that the remaining artifacts were a capacity problem, not an objective
  problem — "ringing/comb-filter" artifacts are a classic undersized-generator symptom.
- **Why the next experiment followed:** the specific character of the remaining artifacts
  (ringing/comb-filtering) pointed diagnostically at generator capacity/receptive-field design,
  motivating the MRF rewrite (Pivot 3).

### Pivot 3: Increasing MRF capacity (E4→E5→E6/E7)

- **Hypothesis:** the original generator's thin, sequential receptive field and capacity-starved
  final layer (8 channels after 6 halvings) is the remaining bottleneck.
- **Intervention:** MRF rewrite (parallel multi-kernel residual fusion) + 4-stage upsampling +
  `base_channels` 256→512, then 512→1024 as a further capacity probe.
- **Result:** 512: mel corr. 0.924/0.932. 1024 (first attempt, E7): **collapsed** — mel corr.
  0.312/0.228, NaN waveform correlation.
- **Ruled out:** nothing cleanly yet — the 1024-channel result was confounded by an unrelated
  optimizer problem, discovered only by investigating the collapse.
- **Did not rule out:** whether capacity itself would help, once the collapse's real cause
  (optimizer instability, not capacity) was identified and fixed.
- **Why the next experiment followed:** the collapse's specific signature (discriminator loss →0,
  generator output → constant) is a textbook Adam-beta-driven GAN failure mode, motivating a
  direct optimizer fix (Pivot 4) before re-attempting the capacity comparison.

### Pivot 4: Fixing Adam betas (E7→E8→E9)

- **Hypothesis:** PyTorch's default Adam `betas=(0.9,0.999)` — too much momentum for an
  adversarial minimax game — caused the 1024-channel collapse, not capacity itself.
- **Intervention:** both generator and discriminator optimizers changed to `betas=(0.8,0.99)`
  (HiFi-GAN's own published AdamW config).
- **Result:** re-ran both 512 and 1024 configs; **no collapse in either**. 1024 now strictly
  dominates 512 on every objective metric (mel corr. 0.965/0.976 vs. 0.924/0.932; waveform corr.
  0.073/0.104 vs. -0.015/0.078). Listening: **no perceptible difference** between the two.
- **Ruled out:** capacity as the bottleneck for Formulation 1 — a clean, confound-free 3.57x
  capacity increase produced the era's largest metric gain with zero perceptual change. Also
  ruled out: the beta fix itself as a sufficient perceptual fix (the audio still sounds mediocre
  regardless of betas, once collapse is avoided).
- **Did not rule out:** representation source, training-paradigm depth, or (at this point)
  anything about the codec/translation approach — those were entirely untested.
- **Why the next experiment followed:** with capacity cleanly ruled out, the next most direct
  single-variable test was representation source, holding the now-validated MRF-1024
  decoder/optimizer fixed — motivating Pivot 5.

### Pivot 5: Replacing MuQ with EnCodec (E9→E10)

- **Hypothesis:** MuQ's self-supervised, non-reconstruction training objective discards or
  entangles information a reconstruction-native codec representation would preserve.
- **Intervention:** swap only the conditioning representation (MuQ→EnCodec continuous
  pre-quantization embedding), keeping decoder/discriminator/losses/optimizer/schedule identical
  (with the one unavoidable upsample-rate/crop-frame adaptation for the new frame rate).
- **Result:** waveform correlation jumped 6-9x (0.073/0.104→0.682/0.602) — the largest effect
  size in the investigation. Mel correlation stayed comparable. Listening: **still
  indistinguishable from every other variant.**
- **Ruled out:** representation source (between these two specific options) as a sufficient
  explanation for perceived quality, generalizing Pivot 4's capacity-ruled-out finding.
- **Did not rule out:** which of EnCodec's three simultaneously-changed properties (training
  objective, frame rate, channel width) drove the metric jump — this swap changed all three at
  once and cannot on its own separate them.
- **Why the next experiment followed:** isolating "reconstruction-trained objective" from
  "different frame rate/channel width" required a representation that shared MuQ's/EnCodec's
  magnitude-domain character without being either model — ground-truth mel — motivating Pivot 6.

### Pivot 6: Replacing learned representations with ground-truth mel (E10→E11)

- **Hypothesis:** if EnCodec's advantage comes from its reconstruction-training objective
  specifically (not just being "a different, better representation"), then ground-truth mel — a
  magnitude-only representation with zero learned/self-supervised component, and by construction
  at least as informative about its own target as anything could be — should behave like MuQ
  (poor waveform correlation), not like EnCodec.
- **Intervention:** replace the conditioning input with the target's own ground-truth log-mel
  spectrogram, same decoder/losses/optimizer/schedule (again with the one necessary upsample-rate
  adaptation).
- **Result:** waveform correlation 0.059/-0.010 — **behaves like MuQ, not EnCodec**, despite
  being maximally informative about its own reconstruction target.
- **Ruled out:** "any richer/higher-frame-rate representation would show EnCodec's advantage" —
  ground-truth mel is at 80Hz (denser than MuQ's 25Hz, comparable to EnCodec's 75Hz) and still
  failed to reach EnCodec's waveform-correlation band.
- **Did not rule out:** the specific mechanism (magnitude-only representations structurally
  cannot carry phase/fine-temporal cues a decoder needs, regardless of frame rate) — this remains
  the leading explanation but was not independently verified beyond this one comparison.
- **Why the next experiment followed:** having now ruled out capacity, optimizer instability, and
  representation source (in the "richer representation helps" sense) as sufficient explanations,
  the remaining live hypothesis was implementation/training-setup-specific to this project's own
  code — motivating a test against an independently authored reference implementation (Pivot 7).

### Pivot 7: Using a reference HiFi-GAN (E11→E12)

- **Hypothesis:** this project's own custom generator/discriminator/loss/training-loop code has
  a specific implementation defect that an independently authored, published implementation would
  not share.
- **Intervention:** vendor `jik876/hifi-gan` (the original paper's own repository) as faithfully
  as possible — different codebase, different author, different architecture width — on the same
  ground-truth-mel task, same 2 tracks, same 15000-step budget.
- **Result:** best mel-loss convergence of the whole investigation (ratio 0.174), but waveform
  correlation 0.073/0.122 — same low band as this project's own custom pipeline, not EnCodec's.
  Listening: **first partial perceptual improvement reported** (disco better, pop not).
- **Ruled out:** "our implementation has a specific bug" as the primary/sole explanation —
  substantially weakened, since an independently authored implementation shows the same fidelity
  signature under the same training budget.
- **Did not rule out:** the shared tiny-overfit training paradigm itself (2 clips, 15,000 steps,
  no dataset-scale discriminator exposure) as the bottleneck — if anything, this experiment
  strengthened that hypothesis, since it is the one variable every Formulation-1 experiment to
  this point shared.
- **Why the next experiment followed:** rather than continue iterating within Formulation 1
  (more architecture variants under the same training paradigm), the project pivoted to
  Formulation 2 entirely — motivating Pivot 8, the most consequential pivot in the investigation.

### Pivot 8: Shifting from waveform generation to DAC-latent translation (E12→E13)

- **Hypothesis:** the phase-hallucination problem inherent to training a decoder toward raw
  waveform samples from a magnitude-domain-adjacent conditioning signal is the actual structural
  bottleneck — not capacity, optimizer, representation source, or implementation fidelity, all of
  which had by now been tested and found insufficient to explain the mediocrity. If so, training
  a small translator to predict a codec's own reconstruction-native latent space (letting that
  codec's frozen decoder handle the phase problem, since it already solves it by construction)
  should look categorically different.
- **Intervention:** replace the entire waveform-decoder pipeline with: MuQ → small MLP → DAC
  `quantized_representation` → frozen DAC decoder → audio. No GAN, no waveform loss, no
  discriminators.
- **Result:** converged in ~150 steps (vs. 15,000) to a cosine-similarity ceiling of 0.6418 that
  was itself provably an architectural artifact (not a MuQ information limit). Waveform
  correlation 0.303/0.192 — 2nd-best of the investigation. Listening: worse than every prior
  variant despite the better metric.
- **Ruled out:** nothing about MuQ's fundamental invertibility yet — this experiment's own
  ceiling was shown to be architectural.
- **Did not rule out:** whether relaxing the identified architectural constraint (memorylessness)
  would close the gap — this was the direct next test.
- **Why the next experiment followed:** the exact, verified nature of the framewise ceiling (a
  3x frame-rate mismatch defeating a memoryless predictor) suggested a precise, minimal fix —
  motivating Pivot 9.

### Pivot 9: Replacing the framewise MLP with local temporal convolutions (E13→E14)

- **Hypothesis:** giving the predictor access to just its immediate neighboring MuQ frames
  (not a larger architecture change, not attention, not more data) should close the framewise
  ceiling, if the ceiling truly was about memorylessness and not about MuQ lacking the relevant
  information.
- **Intervention:** replace the MLP's pointwise `Linear` layers with 3 residual `Conv1D(k=5)`
  layers — a receptive field of ~200ms (5 MuQ frames), verified both analytically and
  empirically.
- **Result:** cosine similarity 0.9999-1.0000; waveform correlation to the DAC oracle
  0.9999/1.0000. Listening: "almost perfect... we nailed it."
- **Ruled out:** "MuQ's 25Hz representation lacks the information needed to predict DAC's 75Hz
  latent" — the same information, accessed with minimal local context, was sufficient for
  near-perfect reconstruction on this task.
- **Did not rule out:** whether this result would generalize beyond the two memorized excerpts —
  explicitly flagged as untested at the time.
- **Why the next experiment followed:** this was the first unambiguous positive result in the
  whole investigation, and CLAUDE.md's own Evaluation and Controls section requires exactly this
  kind of in-distribution success to be validated by a held-out generalization test before being
  trusted — motivating Gate 4 (Pivot 10).

### Pivot 10: Testing held-out crops and tracks (E14→E15/E16)

- **Hypothesis:** E14's near-perfect result reflects a transferable, general MuQ-to-DAC mapping
  that the small ConvNet has learned, not memorization of the two specific training excerpts.
- **Intervention:** retrain the identical architecture (fresh weights, same hyperparameters) on
  proper train/val/test splits — first within the same 2 tracks (unseen crops), then across
  entirely different tracks — with validation-MSE-only checkpoint selection and shuffled/
  time-shifted controls.
- **Result:** held-out cosine collapsed to 0.14-0.18 (from 0.9999); waveform correlation to
  oracle collapsed to ~0; best checkpoint at ~2% of training in both experiments; controls became
  statistically indistinguishable from correct conditioning on held-out data (though clearly
  distinguishable on training data).
- **Ruled out:** the "transferable mapping" reading of E14 — it was memorization.
- **Did not rule out:** whether the training set (16 crops) was simply too small for any
  transferable signal to emerge — a data-quantity confound Gate 4's minimal design could not
  itself separate from a more fundamental non-generalizability.
- **Why the next experiment followed:** the most direct test of the remaining "just needs more
  data" hypothesis was to scale training data while holding everything else fixed — motivating
  Pivot 11.

### Pivot 11: Scaling training data (E15/16→E17→E18)

- **Hypothesis:** Gate 4's generalization failure was caused primarily by insufficient training
  data (16 crops, 8 tracks), not by a fundamental limitation of the architecture or task.
- **Intervention:** scale training data ~20x (E17: 320 crops, 80 tracks) then a further ~5x
  (E18: 1584 crops, 792 tracks — essentially the full GTZAN dataset), architecture/optimizer/loss
  unchanged throughout.
- **Result:** held-out cosine improved from 0.15-0.18 to 0.22 (E17) to 0.23 (E18) — a
  decelerating curve. Waveform correlation to the oracle **never moved from ~0 at any scale
  tested.** Controls margin became real and consistent on held-out data starting at E17 (did not
  exist in Gate 4), roughly held steady from E17 to E18. Best checkpoint moved dramatically later
  in training at E18's scale (83% vs. ~1.5-2.5% at every smaller scale) — immediate overfitting
  stopped occurring, but this did not translate into better ultimate generalization.
- **Ruled out:** Gate 4's most extreme reading ("the mapping does not generalize at all") — a
  real, if weak, transferable signal clearly emerges and grows with data, at least in the latent
  space.
- **Did not rule out:** whether even more data would eventually move the waveform-domain metric —
  three data points across two orders of magnitude is not enough to establish an asymptote with
  certainty, though the complete flatness of the audio-relevant metric across that entire range
  is suggestive.
- **Why the next experiment followed:** with data quantity's marginal returns visibly
  diminishing and the audio-relevant metric never moving at all, the next most direct,
  minimally-confounded single-variable test was the MuQ representation itself — specifically,
  whether the *final* layer (used everywhere up to this point) was the wrong source layer,
  independent of data scale — motivating Pivot 12.

### Pivot 12: Sweeping MuQ layers (E18→E19)

- **Hypothesis:** an earlier or intermediate MuQ hidden layer, closer to the raw acoustic signal
  and less abstracted toward MuQ's self-supervised music-understanding objective, would predict
  DAC latents substantially better than the final layer.
- **Intervention:** measure MuQ's exact hidden-state structure, select 5 evenly-spaced candidate
  layers by a pure deterministic function, train the identical E18 architecture/config on each
  (varying only the input layer), with the final layer re-run first as a control and sanity-
  checked against E18's own stable-region behavior before comparing anything else.
- **Result:** the final layer won the primary ranking metric (validation MSE) outright on both
  val and test splits. The earliest layer showed a small (~10% relative) cosine-similarity
  advantage that did not translate into a lower MSE or any waveform-domain gain — waveform
  correlation to the oracle stayed at ~0 for every layer, no exceptions.
- **Ruled out:** "the final MuQ layer is simply the wrong source representation" as an
  explanation for the saturation ceiling found in Pivot 11.
- **Did not rule out:** architecture/receptive-field capacity, target normalization, discrete
  vs. continuous prediction, or any hypothesis not about *which* MuQ layer is used — these remain
  entirely untested by this pivot.
- **Why this synthesis document exists now:** with data scale and MuQ layer choice both tested
  and found insufficient to explain the ceiling, and with several structurally different
  remaining hypotheses (architecture capacity, discrete prediction, target normalization,
  training-dynamics noise) not yet distinguished from each other, the investigation reached a
  natural point requiring a single coherent synthesis before choosing the next, more invasive
  experiment — which is what this document, and its final recommendation, provide.

---

## Reconciling the apparently conflicting findings

Ten observations, stated in the prompt, that must all be true simultaneously without collapsing
into "MuQ is/isn't invertible" or "the project succeeded/failed":

**1. MuQ-to-DAC temporal convolution memorized two excerpts almost perfectly (E14).**
**2. The framewise MLP reached a verified architectural error floor (E13).**
**3. A small temporal receptive field removed that floor in the memorization setting (E14).**

These three are mutually consistent and causally connected, not merely coincidental: (2) is a
*proven* mathematical consequence of feeding 3 identical repeated inputs to a memoryless map and
expecting 3 different outputs — the floor equals the best-constant-predictor MSE, computed
independently and matching the trained model almost exactly. (3) directly removes the
memorylessness constraint that (2) depended on, with a receptive field (~200ms) chosen
specifically to span the repeated-frame boundary. (1) is the direct consequence: with the
constraint removed and only 2 fixed targets to fit (no generalization pressure at all), a 15.7M-
parameter network has more than enough capacity to memorize ~300 total training frames essentially
exactly.

**4. The same temporal model failed on unseen crops and unseen tracks (E15/E16).**

Not a contradiction of (1)-(3): memorization capacity and generalization capacity are different
properties, and nothing in (1)-(3) claimed the model had learned a *general* MuQ-to-DAC mapping —
that was an explicitly flagged open question at the time, tested directly and answered here.

**5. More data improved held-out latent cosine from ~0.15-0.18 to ~0.22-0.23 (E17/E18).**
**6. Oracle-relative waveform correlation remained ~0 on held-out data throughout (E15-E19).**

These two are not in tension once the DAC-space validity analysis below is taken into account:
(5) shows the model is learning *something* about held-out examples that scales with data — a
real, measurable, if weak, signal in the direction cosine similarity measures. (6) shows that
whatever this signal is, it is not translating into predictions that are close to the *correct
region of DAC's latent space* in the way needed for the frozen decoder to reproduce anything
resembling the target audio. The DAC-space validity analysis (below) provides a direct, measured
explanation for why these can coexist: predicted vectors are dramatically under-scaled relative
to real DAC latents (mean norm 46.8 vs. 136.2 — roughly one-third), far smoother frame-to-frame
than real DAC latents are (lag-1 autocorrelation 0.91 vs. 0.09), and their nearest valid
codebook-combination point is farther away than the prediction's own distance from the origin.
Cosine similarity is scale-invariant and can register modest genuine improvement (5) even while
the model's output remains, in an absolute sense, nowhere near the actual manifold of vectors DAC
was ever trained to decode (6) — because cosine similarity does not penalize the norm collapse or
the smoothing that the manifold analysis directly measures.

**7. The final MuQ layer achieved the best validation MSE in the layer sweep (E19).**
**8. The earliest tested MuQ layer had slightly better cosine similarity but no waveform-domain
gain (E19).**

Consistent with (5)/(6)'s pattern at a finer grain: different MuQ layers produce small,
metric-dependent differences in how well the *shape* of the target is matched (cosine, sensitive
to direction only) without any layer producing predictions close enough to the correct manifold
region to move the waveform-domain metric at all. This is exactly the kind of result expected if
the dominant failure mode is the regression objective's interaction with a high-variance,
combinatorially discrete target (the DAC-space analysis's finding) rather than a MuQ-layer-
specific information deficit — a layer-choice effect this small, riding on top of a much larger,
shared off-manifold problem, would be expected to look exactly like this.

**9. DAC-oracle decoding sounded nearly perfect (throughout the investigation, wherever tested).**

This is the fixed point that makes the whole investigation legible: DAC's decoder, given DAC's
own real quantized latents, reliably produces high-quality audio. The recurring gap between
"metrics say X improved" and "audio still sounds mediocre/poor/wrong" is not a DAC-decoder
problem — it is specifically a problem in getting predictions into the region of latent space
DAC's decoder was trained to consume, whether via a waveform-domain decoder (Formulation 1) or a
latent-space regressor (Formulation 2).

**10. Direct waveform decoders sounded mediocre even when their spectral metrics improved
(E1-E12).**

This generalizes the same underlying phenomenon to Formulation 1: mel-domain and spectral-
convergence metrics can improve substantially (capacity, EnCodec, reference-HiFi-GAN all showed
this) without the decoder's raw waveform output being close enough, in the specific respects that
matter to a human ear (chiefly phase/fine-temporal structure), to sound good. The DAC-space
analysis's finding — that a magnitude/shape-focused metric (cosine similarity, mel correlation)
can improve while the underlying output remains far from the correct high-dimensional target —
is the same structural phenomenon recurring across both formulations, just measured differently
in each (mel/STFT-domain metrics for Formulation 1, latent cosine/MSE for Formulation 2).

### Strongest conclusion supported by each experimental regime

- **Memorization (E1, E6, E9-E14):** a sufficiently capacious network, given only 1-2 fixed
  targets and no generalization pressure, can fit either a magnitude-domain waveform objective
  reasonably well (E9-E12) or a codec's continuous latent space essentially exactly (E14).
  Memorization performance is not informative about generalization and should never be reported
  as if it were.
- **Same-track holdout (E15):** the architecture validated in E14 does not, by default, learn a
  mapping that transfers even to unseen 2-second windows of the *same* training tracks. Held-out
  performance on unseen crops from known tracks is statistically indistinguishable from unseen
  tracks entirely (E16) — same-track holdout provides no special advantage under this pipeline.
- **Unseen-track generalization (E16-E19):** a real, weak, data-scaling transferable signal
  exists in the latent-cosine metric, growing with data quantity (E17→E18) but with rapidly
  diminishing marginal returns, and layer-independent (E19) — it does not yet translate into any
  measurable waveform-domain fidelity at any scale or MuQ layer tested.
- **Data scaling (E17/E18):** two data points spanning ~5x each (a 3-point curve total) show a
  decelerating improvement in latent cosine and a flat-at-zero waveform-domain metric — evidence
  is consistent with either continued-but-slow data-limited improvement or practical saturation
  under the current translator; it does not, with only 3 points, distinguish between these (see
  the Scaling analysis section for the careful, hedged statement).
- **Layer selection (E19):** the final MuQ layer is at least as good as, and by the primary
  ranking metric better than, four alternative candidates spanning the model's depth — layer
  choice is not the explanation for the generalization ceiling.
- **Direct waveform synthesis (E1-E12):** every tested intervention (capacity, optimizer
  stability, representation source, ground-truth information richness, implementation fidelity)
  either failed to move perceived quality or moved it only partially and inconsistently (E12) —
  the phase-hallucination problem inherent to magnitude-domain-trained waveform decoders is the
  most consistent explanation across this entire regime, though "the shared tiny-overfit training
  paradigm is simply too small in scale" was never fully ruled out for Formulation 1 (unlike
  Formulation 2, where data was scaled substantially).
- **Codec-latent translation (E13-E19):** memorization is essentially solved (E14); held-out
  generalization is real but weak and saturating in the metric that predicted E14's perceptual
  success (waveform correlation to oracle); the DAC-space validity analysis below identifies a
  specific, measured mechanism (norm collapse, over-smoothing, near-chance codebook agreement)
  consistent with, and sufficient to explain, this saturation without requiring either "MuQ is
  invertible" or "MuQ is not invertible" as a global claim.

---

## Hypothesis ledger

| # | Hypothesis | Supporting evidence | Contradicting evidence | Status | Most discriminating next experiment |
|---|---|---|---|---|---|
| 1 | Direct HiFi-GAN capacity was too small | E4's diagnosed thin receptive field; E5's MRF rewrite improved mel loss/correlation | E9: 3.57x capacity increase (512→1024) produced the era's largest metric gain with **zero** perceptual change | **Ruled out** (for the capacity range tested) | none needed — already tested cleanly |
| 2 | Direct HiFi-GAN optimizer settings caused instability | E7: default Adam betas (0.9,0.999) produced a textbook GAN collapse (constant output, NaN correlation) | E8/E9: fixing betas eliminated the collapse but did not fix perceived audio quality | **Supported** (as an instability cause) **but insufficient alone** (as a quality explanation) | none needed — the fix is adopted and the residual question moved elsewhere |
| 3 | MuQ final-layer information was insufficient even for memorization | — | E14: cosine 0.9999, listening "almost perfect" on a memorization task | **Ruled out** | none needed |
| 4 | The MuQ-to-DAC frame-rate mismatch defeated a framewise predictor | E13: cosine ceiling 0.6418 = independently-computed best-constant-predictor floor for the exact repeat pattern | — | **Supported, verified analytically and empirically** | none needed |
| 5 | The temporal ConvNet learned a transferable local mapping | E17/E18: real, data-scaling controls-margin gap on held-out data that did not exist in Gate 4 | E15/E16: held-out cosine collapses from 0.9999 to 0.14-0.18; waveform correlation to oracle ~0 at every scale tested | **Weakened / partially supported** — a weak transferable *latent-cosine* signal exists; no evidence it is a *usable* mapping | DAC-space validity analysis (done, see below) and/or discrete-code prediction (E19+1 candidate) |
| 6 | Generalization failure was caused only by insufficient data | E17: cosine 0.15-0.18→0.22 with ~20x more data | E18: only ~4-5% further relative cosine gain from ~5x more data again, and **zero** waveform-correlation movement across the full 100x data range tested | **Weakened** — data helps a little, does not resolve the ceiling within the range tested | a much larger (e.g. 10-100x further) data scale-up would be the only way to further test this, but is de-prioritized by this synthesis's recommendation (see below) |
| 7 | An earlier MuQ layer would predict DAC substantially better | E19: earliest layer's cosine similarity ~10% higher than final | E19: earliest layer's validation/test MSE both *worse* than final; waveform correlation ~0 for every layer including earliest | **Ruled out** (for the 5 layers tested, by the primary ranking metric) | none needed within this layer set; a genuinely different MuQ variant/checkpoint was not tested |
| 8 | A larger receptive field is required | (motivates E9's local-context fix in Formulation 1's analog, and E13→E14's fix) — plausible by analogy | not directly tested for the DAC-latent translator beyond 5 MuQ frames; E19's layer sweep held receptive field fixed throughout | **Unresolved** | parameter-matched receptive-field/dilation sweep (a leading candidate, see below) |
| 9 | The translator lacks capacity | not directly tested — capacity has been held fixed (15.7M params) since E14 | E14 shows 15.7M params is enough to memorize perfectly; the failure mode is generalization, and larger capacity typically *worsens* generalization gaps, not fixes them, all else equal | **Unresolved, but disfavored** — bigger models generalize worse than smaller ones under a fixed, small dataset, all else equal; capacity is not the natural next lever | not recommended as the next experiment (see final recommendation's reasoning) |
| 10 | Continuous MSE regression produces off-manifold DAC predictions | **New evidence, this synthesis:** predicted vector norm 46.8 vs. oracle 136.2 (~34%); requantization distance to nearest valid RVQ point (83.5) exceeds the prediction's own norm; codebook index agreement with oracle at or near chance level (0.15% mean, vs. ~0.1% chance) even at the coarsest stage | none found | **Supported** — the single most concrete, quantified finding in this entire ledger | quantizer-aware training (project predictions through DAC's frozen RVQ during training) — leading candidate |
| 11 | DAC targets are conditionally ambiguous given MuQ | consistent with the observed norm-collapse/smoothing pattern (a regression toward a "blurry average" is the textbook signature of a multimodal target under MSE) | not directly tested (would require, e.g., measuring target variance across multiple plausible continuations of the same MuQ context) | **Unresolved, plausible** | discrete RVQ code prediction (directly tests whether a classification-style objective, which does not average across modes the way MSE does, performs differently) |
| 12 | Predicting discrete DAC RVQ code indices would be more appropriate | motivated directly by finding 10 (continuous regression is provably landing off-manifold) | not tested | **Unresolved, motivated by new evidence** | discrete-code-prediction experiment (candidate 3, see below) |
| 13 | The current latent normalization or target definition is unsuitable | **New evidence, this synthesis:** predicted/oracle mean-of-channel-variance differ by ~10x (1.61 vs. 17.45); no normalization step exists anywhere in the current target/loss pipeline | none found | **Supported, newly identified** | target normalization/whitening experiment (candidate 4, see below) |
| 14 | Training instability or checkpoint-selection noise masks better solutions | E18/E19: turbulent early training (loss spikes to 100s), E19's final-layer control selected an early lucky checkpoint (step 1500) rather than the stable-region optimum | the turbulence is confined to early training in both E18 and E19's control run, and the eventual stable-region values (both experiments) are consistent with each other, suggesting the "true" achievable performance at this scale is reasonably well characterized despite the noise | **Partially supported** — noise is real and documented, but does not appear to be masking a qualitatively different (much better) solution, based on the stable-region values being consistent across independent runs | a training-dynamics-focused experiment (gradient clipping, lower LR, or an LR warmup) is a plausible fallback but is not this synthesis's primary recommendation |
| 15 | MuQ and DAC latent spaces require a nonlinear but still learnable alignment | E14's near-perfect memorization result shows a learnable *local* nonlinear mapping exists for specific examples | E15-E19 show this mapping does not transfer; finding 10 suggests the specific *objective* (continuous MSE) may be unable to express the needed alignment regardless of how nonlinear/capable the model is | **Unresolved** — the "alignment exists in principle" part is supported (E14); whether it's *learnable at scale with this objective* is the open question this ledger's other entries collectively point toward answering "not with continuous MSE alone" | quantizer-aware training and/or discrete prediction (same leading candidates as #10-12) |

---

## Audit the DAC target

**Mandatory, traced from actual code, not inferred from variable names.**

DAC in this project is accessed via `transformers.DacModel` (`descript/dac_24khz`), not the
standalone `descript-audio-codec` PyPI package (rejected: its `descript-audiotools` dependency
pins `protobuf<3.20`, conflicting with this project's `tensorboard` requirement — see
`src/muq_inversion/dac_encoder.py`'s module docstring).

**Trace, audio input → training target:**

1. `DacModel.encode(input_values)` (called via `FrozenDacEncoder.encode()`,
   `src/muq_inversion/dac_encoder.py`) internally computes:
   - `quantized_representation = self.encoder(input_values)` — the raw convolutional encoder
     output, continuous, `(batch, hidden_size=1024, time_steps)`. This is **not** what this
     project caches or trains against.
   - `quantized_representation, audio_codes, projected_latents, commitment_loss, codebook_loss =
     self.quantizer(quantized_representation, n_quantizers)` — this reassigns the same variable
     name in the transformers source, which is precisely why "do not infer from variable names"
     matters here: the pre- and post-quantizer values share an identifier in the library's own
     code at different points in the function.
2. Inside `DacResidualVectorQuantize.forward` (`.venv/.../modeling_dac.py`, verified by direct
   source inspection): the pre-quantizer encoder output (`z`, 1024-dim) is treated as a residual.
   For each of **32 quantizer stages** (`n_codebooks=32`, `codebook_dim=8`, `codebook_size=1024`
   per `DacConfig`), `DacVectorQuantize.forward`:
   - nearest-codebook-lookup on the current residual (`decode_latents`, Euclidean distance to
     32×1024 codebook vectors after L2 normalization — a ViT-VQGAN-style codebook);
   - applies a straight-through estimator: `quantized = projected + (quantized -
     projected).detach()` (gradients pass through as identity, forward value is the true
     quantized code);
   - projects the 8-dim codebook vector back up to 1024-dim via a per-stage `out_proj` (a learned
     linear/conv layer, part of DAC's own frozen weights);
   - subtracts this stage's contribution from the residual and repeats for the next stage.
   - The **final `quantized_representation`** is the **sum, across all 32 stages, of each
     stage's `out_proj`-projected, straight-through-quantized codebook vector** — i.e., the full
     residual-vector-quantized representation, dequantized back into the same 1024-dim space the
     raw encoder output lives in, but constrained (by construction, for any real audio input) to
     lie in the specific discrete set of points reachable by summing one choice from each of 32
     codebooks of 1024 entries each.
3. `DacModel.decode(quantized_representation=...)` consumes exactly this summed, dequantized
   1024-dim tensor directly — confirmed by the `decode` method's signature and this project's own
   direct usage throughout (`FrozenDacEncoder.decode()`).

**What "DAC latent" means in this project, stated plainly:** the **quantized continuous
representation** (`quantized_representation`) — post-RVQ, dequantized/summed back to full width,
pre-decoder. **Not** the pre-quantizer encoder output, **not** the concatenated
pre-quantization residuals (`projected_latents`, 256-dim = 32×8, a different tensor this project
never uses), and **not** the discrete code indices (`audio_codes`, `(batch, 32, time_steps)`
integers, also never used directly as a training target anywhere in this codebase).

**Reported fields:**
- **Exact tensor shape (per crop, 2.0s/48000-sample input):** `(1, 1024, 150)` from `encode()`
  (channels-first); this project's caches store the transposed `(1, 150, 1024)` (frames-first,
  matching MuQ's own `last_hidden_state` layout, for consistency with the shared training loop).
- **Frame rate:** measured at 75.0 Hz (320 samples/frame) — **not** the value implied by
  `DacConfig`'s printed `hop_length: 512` field, which does not match measured behavior (the true
  hop is the product of `downsampling_ratios: [2,4,5,8] = 320`). This discrepancy was caught by
  direct empirical probing, not by trusting the printed config — documented in
  `src/muq_inversion/dac_encoder.py`'s own module docstring as an instance of CLAUDE.md's
  "measure, don't assume the frame rate" rule.
- **Channel dimension:** 1024 (matches `DacConfig.hidden_size`).
- **Normalization:** **none** applied by this project anywhere in the cache/training/prediction
  pipeline — the raw `quantized_representation` values (whatever scale DAC's own encoder/
  quantizer produce them at) are used directly as MSE-regression targets. This is directly
  relevant to the DAC-space validity analysis below.
- **Quantization step:** not a uniform scalar step — the quantizer is a *learned, non-uniform*
  vector codebook per stage (32 independent 1024-entry, 8-dim codebooks), not a fixed grid.
- **Number of codebooks:** 32 (`DacConfig.n_codebooks`), each `codebook_dim=8`,
  `codebook_size=1024` entries.
- **Are predictions decoded directly or re-quantized first?** **Decoded directly, in every
  experiment in this investigation** (E13-E19). `dac.decode(pred.transpose(1,2))` is called on
  the raw continuous model output with no intervening quantization step anywhere in
  `train_muq_to_dac_mlp.py`, `train_muq_to_dac_temporal_conv.py`,
  `train_muq_to_dac_gate4.py`/`train_muq_to_dac_scale_data.py`/`train_muq_to_dac_more_data.py`,
  or `train_layer_sweep.py`. **This project has never tested passing predictions through DAC's
  own frozen quantizer before decoding, in training or at evaluation, until the diagnostic run
  performed for this synthesis document (below).**

---

## DAC-space validity analysis

Performed for this synthesis using **existing, already-trained checkpoints and already-cached
data only** — no training was run. Script: `scripts/analyze_dac_manifold.py` (new file, added for
this synthesis; loads `runs/more_data/checkpoints/best.pt`, step 50000, val_mse=17.29, and
`data/cache/more_data/muq_dac_cache.pt`; inference-only forward passes, no gradient updates).
Analyzed 40 held-out (validation-split) crops from the full-GTZAN experiment (E18), 6,000 total
frames. Full machine-readable output: `artifacts/synthesis/dac_manifold_analysis.json`.

**Central question:** are held-out predictions merely inaccurate, or also outside the valid
manifold/distribution the DAC decoder expects?

**Result: both, and the second is more severe and more explanatory than the first.**

| statistic | predicted | oracle (true DAC target) | interpretation |
|---|---|---|---|
| mean of per-channel means | 0.0033 | 0.0163 | both roughly zero-centered; not the main story |
| mean of per-channel variances | 1.614 | 17.454 | **oracle has ~10.8x more per-channel variance than predictions** |
| vector-norm mean | 46.76 | 136.15 | **predictions are ~34% of the oracle's typical magnitude** |
| vector-norm std | 11.49 | 7.32 | oracle norms are actually *more* consistent frame-to-frame than predictions' |
| mean absolute frame-to-frame difference | 0.387 | 4.478 | **oracle changes ~11.6x more from one 75Hz frame to the next** |
| lag-1 autocorrelation | 0.915 | 0.093 | **predictions are heavily smoothed/slow-varying; the true DAC latent trajectory is nearly uncorrelated frame-to-frame** |

**Requantization analysis** (passing predictions through DAC's own frozen 32-stage RVQ, a
structurally valid operation since the quantizer's expected input width equals
`quantized_representation`'s width, 1024 — see the DAC target audit above):

| statistic | value | interpretation |
|---|---|---|
| distance from raw prediction to its nearest valid on-manifold (re-quantized) point | 83.50 (mean), 2.41 (std) | **larger than the prediction's own vector norm (46.76)** — the nearest valid codebook combination is farther away than the prediction is from the origin |
| that distance, relative to prediction norm | 1.79 | confirms the above: predictions are not "close to" the manifold in any relative sense |
| codebook index agreement with oracle, coarsest stage (stage 0) | 1.22% | above the ~0.098% chance level (1/1024) but far below anything indicating correct prediction |
| codebook index agreement with oracle, mean over all 32 stages | 0.15% | **at or below chance level** for the finer stages |

**Direct answer to the central question:** held-out predictions are not merely somewhat
inaccurate versions of the correct DAC latent — even their **coarsest, most perceptually
significant discrete choice** (RVQ stage 0) is selected essentially independently of the true
answer. Combined with the norm and temporal-smoothness findings, the picture is consistent and
specific: **the trained translator, once forced to generalize (i.e., on any held-out crop),
regresses toward a low-norm, over-smoothed, temporally-simple output that is far from the
discrete, high-variance, near-noise-like manifold DAC's decoder actually expects** — the
textbook signature of a continuous regression objective (MSE) averaging over a highly variable
or multimodal true conditional distribution. This single finding is directly consistent with, and
sufficient to explain without further assumptions: why waveform correlation to the oracle never
moves regardless of MuQ layer (finding 8) or data scale (finding 6) — the failure mode is in the
*objective's* interaction with the target's statistics, not primarily in *which* input
information is available or how much data is used to fit that objective.

**Diagnostics from the task's requested list that were *not* run, and why:** per-channel means/
variances, vector-norm distributions, frame-to-frame difference statistics, temporal
autocorrelation, nearest-valid-codebook distance, codebook assignment agreement, and the
quantizer-passthrough effect were all completed (see above). **Not** completed: **codebook usage/
perplexity** (would require running the analysis over a much larger sample of oracle codes to get
a meaningful entropy estimate across all 1024×32 codebook entries; the 40-crop/6000-frame sample
used here is not large enough for a reliable perplexity estimate and this synthesis declines to
report a number that would likely be dominated by sampling noise). This is listed as a proposed
diagnostic for a future pass, not fabricated.

---

## Scaling analysis

Three verified, comparable data points (E15/E16 "Gate 4" scale, E17 "80-track" scale, E18
"full-GTZAN" scale — all sharing the identical `TemporalConvNet` architecture, optimizer, and
loss; see Comparability limitations item 4 for why E15/E16's within-track and cross-track results
are shown separately rather than averaged).

| scale | train tracks | train crops | val cosine | test cosine | val MSE | test MSE | val wav corr. to oracle | best-checkpoint fraction of training | val controls margin (shuffled/shifted) |
|---|---|---|---|---|---|---|---|---|---|
| Gate 4, Experiment A (unseen crops) | 2 | 16 | 0.179 | 0.184 | 19.66 | 20.42 | 0.006 | 200/8000 = 2.5% | 0.047 / 0.064 |
| Gate 4, Experiment B (unseen tracks) | 8 | 16 | 0.150 | 0.140 | 18.88 | 18.63 | -0.005 | 200/8000 = 2.5% | 0.030 / 0.043 |
| 80-track scaling | 80 | 320 | 0.222 | 0.223 | 17.28 | 17.96 | -0.0045 | 300/20000 = 1.5% | 0.078 / 0.065 |
| Full-GTZAN scaling | 792 | 1584 | 0.230 | 0.226 | 17.29 | 17.18-17.26 | 0.0002 | 50000/60000 = 83.3% | 0.107 / 0.081 |

(CSV version: `artifacts/synthesis/experiment_summary.csv`, filtered to these four rows plus every
other experiment card's key fields. A PNG scaling-curve plot was **not** generated — `matplotlib`
is not installed in this project's environment and adding it would be a dependency change outside
this synthesis's read-only scope; the table above is the complete scaling-curve record.)

**Characterization, conservatively stated:**

- **Latent cosine similarity:** improves with data, with clearly diminishing marginal returns
  (Gate 4→80-track: roughly +0.04 to +0.07 absolute per ~20x crop increase; 80-track→full-GTZAN:
  roughly +0.00 to +0.01 absolute per further ~5x crop increase). This is **three points**, not
  enough to fit or claim a formal scaling law. The shape is consistent with either (a) a slowly
  continuing power-law-like improvement that would eventually plateau at a higher value than
  currently observed, or (b) an improvement that has already, for practical purposes, plateaued
  at the current small-model capacity. **This synthesis does not have enough data points to
  distinguish (a) from (b) for the cosine metric alone.**
- **Waveform correlation to the DAC oracle:** shows **no measurable trend at all** across the
  full ~100x crop-count range tested (16→1584 crops) — every value is statistically
  indistinguishable from zero. This is the strongest evidence in this analysis, and it supports a
  stronger claim than the cosine curve does: **for the audio-relevant metric specifically, the
  evidence is consistent with practical saturation under the current model** (i.e., no
  improvement has been observed across two full orders of magnitude of data, combined with the
  DAC-space validity analysis's finding of a specific, non-data-dependent mechanism — norm
  collapse and over-smoothing — that would not be expected to resolve with more data of the same
  kind under the same continuous-regression objective).
- **This synthesis explicitly does NOT claim** "complete asymptotic saturation" (that would
  require evidence the metric cannot improve with *any* amount of further data, which three
  points cannot establish) nor "continued data-limited improvement is the right characterization"
  (the flat waveform metric argues against data quantity being the dominant lever). The most
  defensible characterization, given the evidence, is: **practical saturation of the
  audio-relevant metric under the current translator and objective, with a separately identified,
  plausible mechanism (continuous MSE regression against a high-variance, discrete target)
  sufficient to explain that saturation without invoking data quantity at all.**

---

## Training-dynamics audit

**Verified observations** (from logs, code, and JSON reports — not inferred):

- **Batch size:** 1 (a single crop per step) in every Formulation-2 training script inspected
  (`train_muq_to_dac_*.py`, `train_layer_sweep.py`) — confirmed by the training loop structure
  (`label = train_labels[step % len(train_labels)]`, one forward/backward per step).
- **Learning rate:** 1.0e-3, constant (no scheduler) throughout E13-E19 — confirmed by
  `LEARNING_RATE = 1.0e-3` in every script and the absence of any `lr_scheduler` import or usage.
- **Gradient norm / gradient clipping:** **no gradient clipping code was found in any
  Formulation-2 training script** (`grep` for `clip_grad` across `scripts/train_muq_to_dac_*.py`
  and `scripts/train_layer_sweep.py` returns no matches). Gradient norms themselves are **not
  logged** anywhere in these scripts — this is a genuine observability gap, not a measured-and-
  ruled-out factor.
- **Input/target norm variation:** directly measured by this synthesis's DAC-space validity
  analysis (see above) — oracle DAC-latent vector norms have std 7.32 around a mean of 136.15 (a
  moderate, not extreme, relative spread, ~5.4% coefficient of variation) — target-side norm
  variation across crops does not appear large enough on its own to explain the large, transient
  train-loss spikes (100s to 10,000s) observed in E18/E19's early training.
- **Crop-level outliers / genre or track association with spikes:** **not verified** — the
  training scripts log aggregate MSE/cosine at fixed step intervals but do not log per-crop loss
  values or the specific `track_id`/`crop_label` active at each step's spike. Reconstructing which
  specific crop(s) coincided with any given loss spike would require re-running training with
  additional per-step logging, which was not done (this synthesis performed no training).
- **Optimizer-state behavior:** Adam's moment estimates are not logged or inspected in any
  script; **not verified**.
- **Normalization behavior:** confirmed (DAC target audit, above) that **no normalization is
  applied to the DAC-latent training target anywhere in the pipeline** — combined with the
  measured ~10x channel-variance mismatch between predicted and oracle latents, this is a
  documented fact, not a speculative cause.
- **MPS-specific instability:** this project's own notes document MPS non-determinism as a real,
  measured effect (`gate4-mps-resume-tolerance`, ~0.00025 relative drift per step) but **at a
  scale far too small to explain loss spikes of the magnitude observed** (train MSE reaching
  153,842 at one step in E18's control run) — MPS float-drift is not a plausible sole explanation
  for spikes of this size, though it was not independently re-tested by running the same
  configuration on CPU for comparison.
- **Whether validation-selection noise makes the apparent best checkpoint unreliable:**
  **verified as a real, documented phenomenon in E19's final-layer control run** — the
  selected checkpoint (step 1500, val_mse=17.34) came from an early dip *before* the long
  turbulent stretch (steps 5000-32000, val MSE up to 142), not from the eventual stable region
  other experiments' best checkpoints came from. This means at least one experiment's reported
  "best checkpoint" number should be read as "a value the sanity-check bounds happened to accept
  from an early lucky point," not "the representative achievable performance of this
  configuration at this data scale" — though the *actual reported metrics* were still measured
  correctly at that checkpoint; the concern is about representativeness, not measurement error.

**Candidate explanations for the turbulence (not verified, listed separately as required):**

1. No gradient clipping, combined with a relatively high fixed learning rate (1e-3, unchanged
   since the framewise-MLP experiment where it was never stress-tested against 1500+ diverse
   training crops) plausibly allows occasional large gradients — e.g., from an unusually
   high-variance training crop's target — to produce a large, transient parameter update that
   temporarily worsens validation performance before the optimizer recovers. This is consistent
   with, but not proven by, the observation that turbulence appeared only once training sets grew
   to hundreds of diverse crops (E17 onward) and was absent in the 16-crop Gate 4 experiments.
2. The complete absence of target normalization (confirmed fact, above) means the loss surface's
   curvature could vary substantially across crops with very different DAC-latent norms/variances
   — a plausible contributor to gradient-scale volatility, though not directly measured.
3. MPS-specific kernel behavior under sustained load (long training runs, large cached tensors
   loaded into unified memory) was not ruled out as a contributing factor, but no direct evidence
   for or against this was gathered.

None of these three is established by direct evidence in this audit; they are prioritized
candidates for a future, cheap diagnostic (e.g., adding gradient-norm logging and re-running a
short segment of E18's training) rather than conclusions.

---

## Final conclusions

### High-confidence conclusions

Directly established by controlled experiments in this repository.

1. Decoder/generator capacity, in the range tested (17M→62M parameters), is not the bottleneck
   for Formulation-1 (direct waveform) perceived quality (E9).
2. PyTorch's default Adam betas `(0.9,0.999)` cause a specific, reproducible GAN-collapse failure
   mode in this project's adversarial training setup; `(0.8,0.99)` reliably avoids it (E7/E8).
3. A memoryless, framewise predictor cannot resolve the exact 3x frame-rate mismatch between MuQ
   (25Hz) and DAC (75Hz) latents — this ceiling is provable from the repeat-pattern structure
   alone and was independently verified to match the trained model's behavior exactly (E13).
4. A small (~200ms) local temporal receptive field is sufficient to remove that ceiling and
   achieve near-perfect latent-space and waveform-domain reconstruction on a memorized 2-clip
   task (E14).
5. That memorization result does not generalize to unseen crops from the same tracks or to
   unseen tracks — held-out performance collapses to near-noise-level waveform correlation in
   both cases, and the two collapse to statistically the same degree (E15/E16).
6. Increasing training data (16→320→1584 crops, 8→80→792 tracks) improves held-out latent cosine
   similarity with clearly diminishing returns, and produces **no measurable improvement** in
   oracle-relative waveform correlation at any scale tested (E17/E18).
7. Among 5 MuQ hidden layers spanning the model's depth, the final layer (used throughout this
   investigation) achieves the best validation-set latent MSE, and no candidate layer produces a
   waveform-domain improvement (E19).
8. This project's continuous MSE-regression translator produces held-out predictions that are
   systematically under-scaled (~34% of the true DAC-latent norm), over-smoothed in time
   (lag-1 autocorrelation 0.91 vs. the true target's 0.09), and — when passed through DAC's own
   frozen quantizer — select codebook indices at or near chance agreement with the true codes,
   even at the coarsest RVQ stage (this synthesis's DAC-space validity analysis).

### Provisional conclusions

Supported by the current pattern but not decisively established.

1. The saturation observed in waveform-domain metrics across data scale (E17/E18) and MuQ layer
   choice (E19) is most plausibly explained by the continuous-regression/off-manifold mechanism
   identified in this synthesis's DAC-space analysis, rather than by insufficient data or a
   wrong-layer choice — but this has not been tested by an intervention (e.g., quantizer-aware
   training or discrete prediction) that directly targets the mechanism and shows improvement.
2. The training turbulence observed at full-GTZAN scale (E18) and in the layer sweep's control
   run (E19) is plausibly connected to the absence of gradient clipping and target normalization,
   but no controlled experiment isolating these factors has been run.
3. Formulation 1's residual mediocrity (even in the best case, E12's reference HiFi-GAN) is most
   plausibly attributable to the shared tiny-overfit training paradigm (2 clips, 15,000 steps, no
   dataset-scale discriminator exposure) rather than any remaining implementation or
   representation issue — but this was never directly tested by scaling Formulation 1's training
   data the way Formulation 2's was (E17/E18).
4. MuQ's final-layer representation likely does contain genuine, if limited, information relevant
   to predicting DAC's reconstruction-native latent space — E14's memorization result and the
   real (if weak) data-scaling signal (E17/E18) are both consistent with this — but whether this
   information is sufficient to ever produce audibly good held-out reconstructions under *any*
   translator design remains unresolved.

### Unsupported conclusions

Tempting claims the current data do not justify — explicitly flagged so they are not repeated
uncritically in any downstream summary of this work.

1. **"MuQ is invertible."** E14's memorization result does not establish this; it establishes
   that a specific small network can memorize two fixed clips' latent trajectories, which is a
   categorically weaker claim.
2. **"MuQ is not invertible."** The held-out failures (E15/E16/E17/E18) do not establish this
   either — this synthesis's own DAC-space analysis identifies a specific, addressable mechanism
   (continuous regression collapse) as at least as plausible an explanation as "MuQ lacks the
   information," and that mechanism has not yet been tested and ruled out by an intervention.
3. **"The project succeeded" or "the project failed."** Neither framing is supported by evidence
   this granular and multi-causal; the investigation has produced a series of well-controlled
   negative and partial results plus one specific, well-supported diagnostic finding (the
   DAC-space manifold mismatch) that narrows, rather than closes, the remaining hypothesis space.
4. **"More data will eventually solve this."** Not supported — the audio-relevant metric showed
   zero movement across two full orders of magnitude of data, and a specific, non-data-dependent
   mechanism has been identified that plausibly explains why.
5. **"A bigger/deeper translator would fix generalization."** Not supported, and disfavored by
   general machine-learning principle: E14 shows the current 15.7M-parameter model already has
   more than enough capacity to memorize the small-scale task perfectly; a larger model trained
   on the same data would be expected to generalize *worse*, not better, all else equal. Hypothesis
   9 in the ledger is explicitly flagged "unresolved, but disfavored" for this reason.
6. **"Waveform-sample correlation to the original reference is a valid proxy for perceptual
   quality in this project."** Directly contradicted by repeated evidence (E10, E13, and
   explicitly by the DAC oracle sounding "perfect" while scoring only 0.37-0.57 on this metric) —
   any future work in this codebase should treat this specific metric as low-value evidence on
   its own, exactly as this project's own notes have concluded independently multiple times.

**Wording suitable for different audiences:**

- **Internal research update:** "We've ruled out capacity, optimizer instability, MuQ layer
  choice, and (within the range tested) pure data scale as explanations for the MuQ-to-DAC
  translator's generalization ceiling. A new offline analysis shows the actual failure mode: the
  translator's held-out predictions are the wrong scale, too smooth in time, and land far from
  any valid DAC codebook combination — consistent with continuous MSE regression collapsing
  toward an average over a highly variable target, not with a lack of usable information in MuQ
  itself. We recommend testing an objective/output-representation fix (quantizer-aware training
  or discrete code prediction) before any further data or architecture scaling."
- **Empire AI proposal:** "This investigation establishes that a small, frozen-encoder-to-frozen-
  decoder translation architecture can achieve near-perfect reconstruction of memorized music
  excerpts by predicting a neural codec's latent space from a self-supervised music
  representation (MuQ), and identifies a specific, well-characterized failure mode — a mismatch
  between continuous regression objectives and the codec's discrete, high-variance latent
  distribution — that limits generalization to unseen material. This is a concrete, addressable
  machine learning problem (not a fundamental representation-capacity limit), for which the
  literature on residual-vector-quantized generative modeling (e.g., discrete token prediction,
  as used in modern neural audio/music generation systems) offers directly applicable next
  approaches."
- **Future paper's limitations section:** "Our translator was evaluated using continuous
  regression (MSE plus cosine similarity) against a residual-vector-quantized codec's dequantized
  latent representation, without target normalization or quantizer-aware training. Post hoc
  analysis shows held-out predictions are systematically lower-magnitude and temporally
  over-smoothed relative to genuine codec latents, and correspond to near-chance selections under
  the codec's own quantizer even at the coarsest quantization level. We did not evaluate discrete
  code prediction or quantizer-in-the-loop training, both of which are natural next steps given
  this finding, nor did we test model architectures with a larger receptive field or capacity,
  which remain untested alternative explanations for the observed generalization gap."

---

## Recommend the next experiment

### Explicit comparison of the nine candidates

1. **Parameter-matched receptive-field sweep using dilated convolutions.** Directly tests
   Hypothesis 8. Cheap, reuses all E18/E19 infrastructure. **Weakness given new evidence:** a
   larger receptive field does not address the specific, measured mechanism (norm collapse,
   near-chance codebook agreement) — more context could help the model choose *which* answer to
   commit to, but would not by itself stop MSE from averaging across remaining ambiguity, and the
   DAC-space analysis found no receptive-field-shaped signature (e.g., it is not that predictions
   are "slightly wrong locally" — they are the wrong overall scale and smoothness everywhere).
2. **Continuous-prediction projection through DAC's frozen quantizer (quantizer-aware
   training).** Directly targets Hypothesis 10, the best-supported new finding in this synthesis.
   Cheap: DAC's quantizer is already loaded and frozen in every evaluation script; adding it to
   the training forward pass (with a straight-through gradient, exactly as DAC's own quantizer
   already uses internally) requires no new model code, no new data, no new caching. Directly
   answers whether forcing on-manifold outputs during training (not just at evaluation, where
   this synthesis tested it) closes any of the gap.
3. **Discrete prediction of DAC RVQ code indices.** Also directly targets Hypotheses 10-12, and
   most directly matches DAC's real generative structure (classification does not average across
   modes the way MSE does). **Weakness:** requires a genuine architecture change (a classification
   head per codebook stage, likely starting with just the coarsest stage or a handful of stages)
   and a new loss (cross-entropy) — more invasive than candidate 2, and changes more than one
   variable at once relative to every prior experiment in this investigation's careful
   single-variable-isolation discipline.
4. **Target normalization or whitening.** Directly targets Hypothesis 13, also newly supported by
   this synthesis (10x channel-variance mismatch, zero normalization anywhere in the pipeline).
   Extremely cheap: a per-channel z-score computed once from the cached training targets, applied
   before the loss and inverted before decoding. **Advantage over candidate 2:** even more
   surgical — a single, clearly-scoped variable change, directly testable against the exact norm-
   mismatch number this synthesis already measured (predicted mean norm should move from ~47
   toward ~136 if this is the dominant effect).
5. **Smaller regularized translator.** Not motivated by any finding in this investigation —
   Hypothesis 9 (capacity) is disfavored; a smaller model would not address a target-
   distribution/objective mismatch and risks *reducing* the already-limited memorization capacity
   demonstrated in E14 without addressing generalization.
6. **Small transformer.** Explicitly out of scope per every experiment's own stated constraints
   in this investigation, and not motivated by any finding here — the identified mechanism is
   about the *objective*, not about needing global attention over the sequence.
7. **Additional data scaling.** Directly contraindicated by E18's flat waveform-correlation
   result across a 100x data range and by this synthesis's mechanism-based explanation for why —
   the lowest-information-value candidate on the list given current evidence.
8. **Additional MuQ-layer experiments.** E19 already tested 5 layers spanning the full depth and
   found no layer resolves the ceiling; further layer experiments (e.g., a finer-grained sweep)
   are unlikely to add much given how flat the waveform-correlation result was across all 5
   layers already tested.
9. **Multi-layer MuQ fusion.** Explicitly excluded by CLAUDE.md's own scope ("do not combine
   intermediate layers... unless explicitly requested later") and not motivated by E19's finding
   that layer choice alone does not move the ceiling — fusing several layers that individually
   show no effect is unlikely to produce one that does, and would reopen a scope question this
   project's own governing document has deliberately deferred.

### Primary recommendation: Target normalization / whitening (candidate 4)

**Why this one, not quantizer-aware training or discrete prediction, despite both also being well
motivated:** candidate 4 is the single cheapest, most surgical, single-variable test of the most
directly measured new finding (the 10x variance mismatch and 3x norm mismatch), and its outcome
directly informs which of candidates 2/3 (both more invasive) is worth pursuing next, if either
is. It is the experiment most likely to produce a **fast, cheap, and maximally informative**
disambiguation between "this is fundamentally a target-scale/objective problem, fixable by
correctly-scaled continuous regression" and "this is a deeper multimodality/manifold problem that
requires quantizer-awareness or discrete prediction regardless of target scaling." Running
candidate 2 or 3 first, without first ruling out the much simpler scale-mismatch explanation,
risks attributing an effect to quantizer-awareness or discreteness that a trivial normalization
fix could have produced on its own.

**Full specification:**

- **Independent variable:** whether the DAC-latent training target is normalized (per-channel
  z-score, computed once from the E18 training split's cached targets) before the MSE+cosine loss
  is computed, with predictions de-normalized before decoding/evaluation.
- **Frozen components:** `TemporalConvNet` architecture (unchanged, 15,731,712 params, 5-MuQ-
  frame receptive field), MuQ encoder (final layer, per E19), DAC encoder and decoder, optimizer
  (Adam, lr=1e-3, betas `(0.9,0.999)`), the cosine-similarity loss term and its weight (0.1), the
  E18 train/val/test split and crop manifest, the training step budget (reuse E18's 60000 or
  E19's 45000 — recommend 45000 to match E19 directly), validation-MSE-only checkpoint selection.
- **Primary metric:** validation-set latent MSE, computed in the **normalized** space (matching
  what the loss actually optimizes) — report alongside the de-normalized MSE for comparability
  with every prior experiment in this document.
- **Secondary metrics:** validation/test cosine similarity; oracle-relative waveform correlation
  (val/test); the exact DAC-space validity statistics from this synthesis's analysis (predicted
  vector norm, lag-1 autocorrelation, requantization distance, codebook agreement) — **recomputed
  for the normalized-target run** and compared directly against this synthesis's own numbers
  (predicted norm 46.76 vs. oracle 136.15; lag-1 autocorrelation 0.915 vs. 0.093) as the primary
  evidence for whether normalization closed the gap.
- **Validation-only selection:** identical rule to every experiment since E15 — checkpoint chosen
  by lowest validation MSE only, never test performance or listening.
- **Controls:** identical shuffled-across and time-shifted controls used in E15-E19, evaluated at
  the selected checkpoint only, not used for checkpoint selection.
- **Fail-early criterion:** run the existing tiny-subset sanity check (one memorized crop, 300
  steps, must reproduce a mse ratio ≤0.5, per every prior experiment's pre-flight check) before
  committing to the full budget — if the normalized-target pipeline cannot even memorize one
  crop, stop and diagnose before running the real experiment (this exact pattern is already
  implemented and reusable from E15-E19's scripts).
- **What each outcome means:**
  - **If normalization closes most of the norm/variance gap (predicted norm moves toward ~130-140,
    lag-1 autocorrelation moves toward the oracle's ~0.09) AND held-out waveform correlation
    improves measurably:** strong evidence the failure was substantially a target-scale/objective
    interaction, not a deeper multimodality problem — the natural follow-up is to check whether
    this also improves perceived audio quality (listening test) before considering this line of
    work largely resolved.
  - **If normalization closes the norm/variance gap BUT waveform correlation does NOT improve:**
    evidence that scale mismatch was real but not the dominant driver of the audio-domain
    failure — motivates candidate 2 (quantizer-aware training) next, since the remaining problem
    is more likely about landing on the *wrong* valid point, not an *off-manifold* point.
  - **If normalization does NOT close the norm/variance gap at all** (predictions remain
    dramatically under-scaled even when trained against a normalized target): evidence that the
    shrinkage is not simply about target scale but about MSE's fundamental behavior under
    multimodality — motivates candidate 3 (discrete RVQ code prediction) as the more direct fix,
    since only a non-averaging objective (classification) can escape this failure mode by
    construction.

### Fallback 1: Continuous-prediction projection through DAC's frozen quantizer (candidate 2)

If the primary experiment's outcome points toward "scale wasn't the whole story," add DAC's
frozen RVQ quantizer into the training forward pass itself (not just at evaluation, as this
synthesis did): predict, quantize via DAC's own frozen quantizer with a straight-through
gradient, compute the loss (and optionally an auxiliary decoded-audio STFT loss) against the
re-quantized output. This directly forces on-manifold predictions during training, not just
measures off-manifold-ness after the fact.

### Fallback 2: Discrete prediction of DAC RVQ code indices (candidate 3)

If both the primary experiment and Fallback 1 fail to move held-out waveform correlation, the
most direct remaining test of the multimodality/ambiguity hypothesis is to replace continuous
regression with classification: predict, at minimum, the coarsest 1-2 RVQ codebook indices
directly (cross-entropy loss, a small classification head added to the existing backbone), which
cannot express an "averaged" answer between two valid codes the way MSE can.

---

## Decision tree

- **If target normalization closes the norm/variance gap AND held-out waveform correlation
  improves measurably** → run a listening test on the normalized-target checkpoint; if confirmed
  by ear, this becomes the new baseline for any further generalization/scaling work, and further
  data scaling (previously de-prioritized) becomes worth revisiting on top of the fixed objective.
- **If target normalization closes the norm/variance gap BUT waveform correlation does not
  improve** → proceed to Fallback 1 (quantizer-aware training); do not scale data or model size
  in the meantime.
- **If target normalization does not close the norm/variance gap at all** → proceed directly to
  Fallback 2 (discrete RVQ code prediction), skipping Fallback 1, since the evidence would then
  suggest the problem is multimodality itself, not merely scale, and quantizer-projection alone
  (without changing the objective) is unlikely to help either.
- **If none of the primary experiment or either fallback moves oracle-relative waveform
  correlation at all** → stop and reformulate: this would be the first evidence in the whole
  investigation that the problem is not addressable by objective/output-representation changes
  alone, and would justify revisiting Hypothesis 8 (receptive field) or a genuinely different
  translator architecture — but only after these three cheaper, more surgical, evidence-motivated
  experiments have been exhausted, not before.

---

## End-of-task report

**Files created:**
- `notes/EXPERIMENT_SYNTHESIS.md` (this document).
- `scripts/analyze_dac_manifold.py` (new, read-only diagnostic script used to produce the
  DAC-space validity analysis; loads existing checkpoints/caches only, performs no training).
- `artifacts/synthesis/dac_manifold_analysis.json` (machine-readable output of the above).
- `artifacts/synthesis/experiment_summary.csv` and `artifacts/synthesis/hypothesis_ledger.csv`
  (tabular exports of the master table and hypothesis ledger, described below).
- `artifacts/synthesis/scaling_curve.png` was **not** created (`matplotlib` is not installed in
  this project's environment; installing it would be a dependency change outside this synthesis
  task's read-only scope). The scaling data is fully captured in the Scaling analysis table above.

**Sources inspected:** `CLAUDE.md`; `notes/{MEMORY,SCRATCHPAD,DECISIONS,NEXT}.md` on the current
branch and, via `git show`, on `main`, `encodec-hifigan-experiment`,
`experiment/mel-hifigan-sanity`, `experiment/reference-hifigan`, `experiment/muq-to-dac-mlp`,
`experiment/muq-to-dac-temporal-conv`; `configs/{smoke,smoke_1024,pilot}.yaml` (current branch)
and `configs/{encodec_1024,mel_1024}.yaml` (via `git show` on their respective branches);
`data/manifests/{gtzan_manifest_smoke,gate4_crops,scale_data_crops,more_data_crops}.csv`;
`artifacts/{gate4_minimal,layer_sweep,more_data,scale_data,muq_layer_sweep}/*.json`;
`src/muq_inversion/dac_encoder.py`, `encoder.py`, `models/temporal_conv.py`,
`layer_sweep_common.py`, `gate4_common.py`; the installed `transformers` package's
`modeling_dac.py` source directly (for the mandatory DAC-target audit); `git log`/`git branch -a`/
`git tag -n99` across every branch; `runs/more_data/checkpoints/best.pt` (loaded read-only for
the DAC-space validity analysis, inference only).

**Missing or inconsistent evidence, stated explicitly:**
- No listening verdict was ever recorded for E17 (80-track scaling) or E18 (full-GTZAN scaling) —
  the user moved to the next experiment request both times before giving one.
- The conditioning-collapse diagnostic raised in E11 (compare generator output under correct/
  zero/shuffled/cross-track conditioning using already-trained checkpoints) was designed but
  never executed on any branch.
- Per-crop/per-genre loss logging does not exist for E18/E19, so the specific cause of the
  observed training turbulence could not be attributed to particular crops or genres in this
  audit (see Training-dynamics audit's "not verified" items).
- E1-E3's exact optimizer beta values are not unambiguously recorded prior to the beta issue's
  discovery on the 1024-channel capacity experiment (E7) — flagged in E1-E3's experiment card.
- Codebook usage/perplexity was not computed (sample size too small for a reliable estimate from
  existing artifacts alone; listed as a proposed future diagnostic, not fabricated).

**Strongest current conclusion:** the MuQ-to-DAC translator's generalization ceiling is best
explained, among everything tested so far, by a specific and directly measured mismatch between
its continuous MSE-regression objective and DAC's discrete, high-variance, unnormalized latent
target — not by insufficient data, wrong MuQ layer, insufficient model capacity, or (for the
capacity/optimizer/representation-source hypotheses tested earlier, in Formulation 1) any of
those explanations either. This is a narrowing of the hypothesis space to a specific, addressable
mechanism, not a resolution of the original research question.

**Recommended next experiment:** target normalization/whitening of the DAC-latent training
target (Primary, above), with quantizer-aware training and discrete RVQ code prediction as
ordered fallbacks depending on its outcome.

**Why this experiment has the highest information value:** it is the cheapest possible test
(a single, clearly-scoped preprocessing change, reusing every existing cache/checkpoint/script
pattern in this investigation) of the best-supported new finding in this entire synthesis (the
10x variance / 3x norm mismatch, measured directly from existing artifacts), and its three
possible outcomes each point to a different, specific next step — meaning it maximally reduces
uncertainty per unit of compute, in sharp contrast to another blind data-scaling or architecture-
scaling run, which this document's own scaling analysis and hypothesis ledger show would very
likely reproduce the same flat waveform-correlation result for reasons already identified.
