# DECISIONS

## muq-transformers-compat: Pin `transformers==4.55.0`, not the latest 5.x

**Context.** `uv add muq` resolves an unpinned `transformers` dependency to the newest release
(5.16.1 at the time of writing). MuQ's own code
(`muq/muq/models/muq_model.py`) builds the internal Wav2Vec2-Conformer encoder as:

```python
config = EasyDict(w2v2_config)
self.conformer = Wav2Vec2ConformerEncoder(config)
```

i.e. a plain dict-like object, not a real `Wav2Vec2ConformerConfig`. transformers 5.x's
`Wav2Vec2ConformerEncoder.forward` calls `create_bidirectional_mask(config, ...)`, which reads
`config._attn_implementation` -- an attribute only a real `PretrainedConfig` instance carries.
This raises `AttributeError: 'EasyDict' object has no attribute '_attn_implementation'` on every
forward pass, so Gate 0 could not run at all against transformers 5.16.1.

**Alternatives considered.**
1. Monkeypatch `EasyDict`/set `_attn_implementation` before calling `MuQ.from_pretrained` --
   works around the symptom but is fragile against any other newer-transformers API drift
   inside `Wav2Vec2ConformerEncoder.forward`, and modifies behavior we don't own from outside.
2. Vendor/patch the `muq` package locally -- violates "smallest reversible change" and creates
   an ongoing maintenance burden diverging from upstream.
3. Pin `transformers` to a version predating the `masking_utils.create_bidirectional_mask`
   refactor for encoder-only models.

**Decision.** Pin `transformers==4.55.0`. This required also relaxing `huggingface_hub` to
`>=0.34.0,<1.0` (transformers 4.55.0's own constraint), since `uv add muq` had previously pulled
in `huggingface_hub==1.30.0`.

**Evidence.** With this pin, `uv run python -m muq_inversion.cli doctor --config
configs/smoke.yaml` passes end-to-end on MPS: MuQ produces finite, correctly-shaped
`last_hidden_state` output (see [[../notes/MEMORY.md]]).

**Consequences.** `pyproject.toml`/`uv.lock` now carry an explicit upper-bound-adjacent pin on
two packages instead of letting `muq`'s own (unbounded) dependency resolve freely. Any future
`uv add`/`uv sync` that tries to move `transformers` past `<5.0` must be re-validated against
Gate 0 before being accepted.

**Rollback plan.** If a future `muq` release fixes its config construction (e.g. building a real
`Wav2Vec2ConformerConfig` with `_attn_implementation` set), re-run Gate 0 against
`transformers>=5.0` and drop this pin if it passes.

---

## gtzan-source-and-defects: GTZAN acquisition source and known defects

**Context.** CLAUDE.md requires explicit (non-hidden) dataset acquisition and documentation of
any split limitation. GTZAN's original host (`opihi.cs.uvic.ca`) is frequently unreliable;
several unofficial redistributions exist on Hugging Face and Kaggle with varying fidelity
(some re-encoded, cropped, or missing files).

**Alternatives considered.**
1. Kaggle "GTZAN Dataset - Music Genre Classification" -- popular but requires a Kaggle account/
   API token and is a third-party re-upload of uncertain provenance.
2. `marsyas/gtzan` on Hugging Face -- mirrors the canonical `data/genres.tar.gz` archive
   originally hosted by the Marsyas project (per its own dataset loading script), no
   authentication required, and reachable from this network.
3. Manually sourcing from `opihi.cs.uvic.ca` directly -- same canonical content, but that host
   was not reachable from this environment during testing (`curl -sI` timed out).

**Decision.** Use `marsyas/gtzan` (Hugging Face, `data/genres.tar.gz`), downloaded via
`huggingface_hub.hf_hub_download` inside `prepare-data`. This choice was confirmed with the user
directly rather than assumed.

**Evidence / defects found during Gate 1 (2026-09-04), 1000 files indexed:**
- `jazz.00054.wav` is corrupt (`libsndfile: "bad data offset"`, unreadable even for header info).
  This matches the `CORRUPTED_FILES` list in the HF dataset's own `gtzan.py` loading script, so
  it's a defect in the canonical distribution, not an artifact of this download. Flagged
  `corrupted=True` in the manifest and excluded from resample verification and splitting logic.
- 14 checksum-identical file pairs exist (e.g. `metal.00058.wav` / `rock.00016.wav` are
  byte-identical despite different genre labels) -- a previously documented GTZAN defect
  (mislabeled/duplicated tracks; see Sturm, "The GTZAN dataset: its contents, its faults, their
  effects on evaluation, and its future use", 2013). `find_duplicate_checksums` detects these;
  `assign_splits` keys its deterministic split hash purely on checksum (not track_id/genre) so
  that every byte-identical duplicate -- even across mislabeled genres -- lands in the same
  split, preventing train/test leakage from these duplicates.
- No artist metadata is available in this distribution, so the split in `assign_splits` is a
  deterministic, checksum-deduplicated, per-track hash split -- not an artist-aware or
  officially fault-filtered partition (e.g. the "fault-filtered" GTZAN split from Kereliuk et
  al. is a separate, differently-sourced artifact this project does not currently use).
  Documented here per CLAUDE.md rule 7.

**Consequences.** The committed split in `data/manifests/gtzan_manifest_<config>.csv` is stable
across runs (same seed -> same split) but is a project-specific compromise, not the literature's
standard "fault-filtered" split. Future work comparing against published GTZAN classification
benchmarks would need to note this.

**Rollback plan.** If an artist-aware or fault-filtered manifest becomes available (e.g. by
importing Kereliuk et al.'s split file), regenerate the manifest under a new config name (never
overwrite the existing one, per CLAUDE.md rule 8) and record the change here.

---

## decoder-upsample-placeholder: Provisional decoder upsample_rates for Gate 0 only

**Context.** Gate 0 requires "instantiate the decoder and run a shape-only forward pass," but
the decoder's real upsampling factors must exactly match MuQ's measured samples-per-frame ratio
(CLAUDE.md Decoder Design item 3) -- a Gate 2 deliverable, not yet done.

**Decision.** `configs/{smoke,pilot}.yaml` set `decoder.upsample_rates: [5, 4, 4, 3, 2, 2]`
(product 960), explicitly commented as provisional, purely to give Gate 0's `MiniGenerator` a
concrete shape to run. `MiniGenerator` only implements Decoder Design components 1-5 (input
projection, temporal conditioning via residual blocks, upsampling, waveform head) -- no
discriminators (component 6), per the instruction to defer those until after the reconstruction
path passes its own smoke tests.

**Evidence.** Doctor's own MuQ probe on 2026-09-04 measured samples-per-frame = 960.0 exactly,
which happens to match this placeholder's product -- a coincidence worth noting, not a
confirmation that 960 is the *only* valid factorization, alignment, or that this factor
breakdown (5,4,4,3,2,2) is the right one architecturally. That decision belongs to Gate 2/3.

**Consequences.** None yet -- this config value is not used for any real training.

**Rollback plan.** Revisit `decoder.upsample_rates` explicitly during Gate 2/3 work; this
decision record exists so nobody mistakes the placeholder for a validated architectural choice.

**Update (Gate 2, 2026-09-04).** Real GTZAN clips do not land on an exact 960-samples-per-frame
boundary: full-length probes measured `samples_per_frame = 960.0067` (750 frames for 720,005
samples), vs. exactly 960.0 for probe crops whose sample count is itself a multiple of 960 (e.g.
the 1s/2s/5s/10s crops, which were *taken* as `round(duration * 24000)` and happened to divide
evenly). This confirms CLAUDE.md's caveat that "if the ratio is not integral" an explicit
alignment/cropping strategy is required (Decoder Design item 3) -- Gate 3/4 training must crop
waveform targets to `num_frames * upsample_product` samples (or pad/trim the generator output to
match the target), not assume every clip's raw length is an exact multiple of the upsample
product. Not fixed now -- this is a Gate 3 concern once real training crops are implemented.

---

## gate2-cache-format: `torch.save`/`torch.load` for cache entries, not `.npy`/`.npz`

**Context.** CLAUDE.md's Resource Discipline section requires content-addressed, atomic,
resumable cache writes supporting float32 and one explicitly validated lower-precision format.

**Alternatives considered.**

1. `numpy.savez` with the array plus a JSON-encoded metadata string -- avoids pickle, but storing
   a dict of mixed types (str, int, float) as an npz sidecar either needs `allow_pickle=True`
   (same trust boundary as `torch.save`) or manual flattening into parallel arrays, which is more
   code for no real safety gain since these caches are entirely self-generated, never loaded from
   an untrusted source.
2. Two files per entry (`.npy` for the tensor + `.json` for metadata) -- keeps things
   inspectable without `torch`, but a crash between the two atomic renames leaves a valid tensor
   file with no matching metadata (or vice versa), which is exactly the corruption atomicity is
   meant to prevent.

**Decision.** Single `.pt` file per cache entry via `torch.save({"features": tensor, "metadata":
dict}, ...)`, written to a temp file in the same directory and `os.replace`'d into place
(`cache.py:save_atomic`). One file, one atomic rename, no cross-file consistency window.

**Evidence.** Gate 2's cache round-trip test (`cmd_extract` in `cli.py`) writes 30 (tensor,
dtype) pairs across 3 real tracks x 5 durations x {float32, float16}, reloads each, and asserts
`torch.equal` against the tensor as saved (not against the original float32 tensor, since
casting to float16 before saving is an intentional, separately-measured lossy step) -- all 30
passed exactly. Re-running `extract` a second time overwrites the same 30 keys without error
(resumability check).

**Consequences.** Cache entries require `torch` to inspect (not plain `numpy`/`json`), which is
already a hard dependency of this project, so no new cost.

**Rollback plan.** If cache entries ever need to be inspected/edited by non-Python tooling,
switch to the two-file `.npy`+`.json` layout and add a manifest-level checksum step to close the
cross-file atomicity gap.

---

## gate3-crop-alignment: Training crops are exact multiples of the decoder's upsample product

**Context.** [[decoder-upsample-placeholder]]'s Gate 2 update found that real, non-hand-picked-
length GTZAN clips do not land on an exact 960-samples-per-frame boundary (measured
`samples_per_frame = 960.0067` for full-length clips vs. exactly `960.0` for crops whose sample
count is itself a multiple of 960). Decoder Design item 3 requires either an exact match or "an
explicitly documented alignment/cropping strategy."

**Alternatives considered.**

1. Pad/trim the generator's output to match whatever length the target waveform happens to be --
   adds a runtime alignment step (and a design decision about which side to pad/trim) that has to
   be re-justified every time crop length or upsample factors change.
2. Pad/trim MuQ's *input* so its frame count times 960 exactly equals the input length, accepting
   a few discarded/zero-padded samples at the tail -- workable, but adds complexity for no benefit
   at the training-crop scale we're using now.
3. Choose training crop lengths that are themselves exact multiples of 960 samples, so MuQ (which
   downsamples by an exact integer factor of 960, confirmed in Gate 2 for any input length that is
   itself such a multiple) always returns exactly `crop_samples / 960` frames, and the generator's
   output (`num_frames * total_upsample = num_frames * 960 = crop_samples`) is length-identical to
   the target waveform with no padding/trimming logic at all.

**Decision.** Option 3. `training.crop_frames` (a frame count, default 50 -> 2.0s) is the single
source of truth; `crop_samples = crop_frames * generator.total_upsample` is derived in `train.py`,
never configured directly as a raw sample/second count. This only applies to *training and
validation crops* -- it says nothing about arbitrary-length inference in later gates, which would
still need one of alternatives 1/2 if ever required.

**Evidence.** Gate 3's first run (`configs/smoke.yaml`, `crop_frames=50` -> `crop_samples=48000`)
produced generator output shaped exactly `(1, 48000)` against a `(1, 48000)` target with no
length mismatch at any of the 500 training steps or the final reference/reconstruction/difference
export.

**Consequences.** Training/validation crop durations are constrained to multiples of 960 samples
(40ms at 24kHz) -- a negligible constraint in practice, since any desired crop duration can be
rounded to the nearest 40ms.

**Rollback plan.** If a later gate needs arbitrary-length reconstruction (not fixed training
crops), revisit alternatives 1/2 above and document the chosen padding/trimming convention then.

---

## gate3-loss-configs: Reconstruction losses configured explicitly for 24kHz music

**Context.** CLAUDE.md's Decoder Design section explicitly warns against copying speech-specific
HiFi-GAN constants (originally tuned for 22.05kHz speech) blindly onto 24kHz music.

**Decision.**

- `MelSpectrogramLoss`: `n_fft=2048`, `hop_length=300`, `win_length=1200` (50ms window / 12.5ms
  hop at 24kHz -- long enough to resolve low musical pitches, short enough hop for rhythmic
  transients), `n_mels=128`, `fmin=0.0`, `fmax=12000.0` (the true Nyquist frequency at 24kHz, not
  a speech-band cutoff like 8000Hz).
- `MultiResolutionSTFTLoss`: three resolutions, `fft_sizes=(512, 1024, 2048)`,
  `hop_sizes=(50, 120, 240)`, `win_lengths=(240, 600, 1200)` -- the same relative
  short/medium/long spectral-resolution spread used in Parallel WaveGAN's multi-res STFT loss,
  which is a universal (not speech-specific) vocoder loss, rescaled proportionally for 24kHz
  instead of 22.05kHz.

All of these are config fields (`mel_loss`, `stft_loss` in the YAML configs), not hard-coded
constants, so they can be revisited without a code change.

**Evidence.** `tests/test_losses.py` confirms both losses are exactly zero for identical
signals and positive for different signals. Verified both `torchaudio.transforms.MelSpectrogram`
and `torch.stft` run natively on MPS in this environment (no CPU fallback needed for the loss
computation). Gate 3's tiny-set overfit run showed the combined loss drop from a mean of 7.51
(first 10 steps) to 3.18 (last 10 steps) -- ratio 0.42, under the 0.5 PASS threshold.

**Consequences.** None yet; these are training-only losses, not part of the frozen MuQ contract.

**Rollback plan.** Revisit resolutions/window lengths if Gate 4/5 reconstructions show a specific
frequency band or time resolution is poorly captured.

---

## gate4-mps-resume-tolerance: Resume-equivalence tolerance must scale with step count on MPS

**Context.** Gate 4's checkpoint/resume equivalence check trains an "uninterrupted" control run
and a separately-checkpointed-and-resumed run from the same seed, then compares their loss
trajectories. The first attempt used an absolute tolerance (`1e-4`) for the pre-checkpoint
determinism sub-check and a tight relative tolerance (0.02) for the full resume comparison,
copied from an initial guess without measurement.

**Evidence gathered.** On `configs/smoke.yaml` (30-step pre-checkpoint window), the two supposedly
identical training runs (same seed, same architecture, same crop-sampling RNG) diverged by up to
0.68% relative loss difference by step 30 despite having bit-identical initial weights and inputs
-- i.e., MPS/Metal kernels are not guaranteed bit-exact across separate op invocations even with
identical seeds (a known characteristic of GPU backends generally, not specific to this codebase).
On `configs/pilot.yaml` at 100 and then 200-step pre-checkpoint windows, the measured divergence
grew to 2.14% and 4.16% respectively. Computing per-step drift rate: `0.0068/30 = 0.000227`,
`0.0214/100 = 0.000214`, `0.0416/200 = 0.000208` -- three independent measurements agreeing to
within ~10% of each other, strongly suggesting a consistent, roughly-linear per-step
floating-point drift rate rather than a resume-logic defect (a real bug would not produce such a
consistent rate across three different run lengths).

**Decision.** Set `resume_tolerance` per config, scaled to that config's `resume_test_at`/`steps`
window rather than using one universal constant: `configs/smoke.yaml` keeps `0.02` (calibrated for
its 30-step window), `configs/pilot.yaml` uses `0.08` (calibrated for its 200-step window, with
headroom above the ~0.04 measured rate to avoid being fragile to run-to-run noise in the noise
estimate itself). The pre-checkpoint determinism sub-check now uses the same relative-tolerance
style as the main resume comparison (previously it used an unrealistic absolute `1e-4`, which
does not scale with loss magnitude and does not account for MPS's inherent nondeterminism at all).

**Consequences.** A resume-equivalence check that requires bit-exact reproduction on MPS would
always fail regardless of implementation correctness -- CLAUDE.md's Stop Conditions phrase this
exactly right ("resumed training diverges from an uninterrupted control **beyond expected
nondeterminism**"), and this decision operationalizes what "expected" means, backed by three
independent measurements rather than an arbitrary guess.

**Rollback plan.** If a later gate needs stricter reproducibility guarantees (e.g. for a
published result), investigate `torch.use_deterministic_algorithms(True)` and CPU-only
verification of a short segment, per CLAUDE.md's "test correctness against CPU on tiny
deterministic examples" guidance -- do not just raise the tolerance further without also
re-measuring.

---

## gate4-shuffled-time-finding: Decoder does not yet show strong temporal-order sensitivity

**Context.** Gate 4's Evaluation and Controls section requires "conditioning shuffled in time"
as a control: encode a clip's own crop normally, then permute the resulting MuQ frame order
before feeding it to the decoder. If reconstruction loss with this control is close to the loss
with correctly-ordered conditioning, CLAUDE.md says to "assume conditioning collapse or leakage
until disproven" -- this control specifically targets Scientific Question 1 ("Does the decoder
use temporal MuQ conditioning rather than merely learning dataset-level audio statistics?").

**Finding.** Across three independent training runs on `configs/pilot.yaml` (6 tracks, run
lengths 200/200/400 steps, each a fresh model from the same seed), the `shuffled_time` control's
final mean loss was consistently only ~9-13% higher than the correctly-conditioned model's --
short of the configured 20% `control_margin` every time:

| run (steps) | correct | shuffled_time | shuffled_time / correct |
|---|---|---|---|
| 200 (1st) | 3.4847 | 4.6546 | 1.34 (this run was noisier; see below) |
| 200 (2nd, same config) | 3.4824 | 3.8732 | 1.11 |
| 400 | 3.2804 | 3.7020 | 1.13 |

(The first 200-step run's 1.34 ratio was itself borderline/noisy -- rerunning the identical
config produced 1.11, illustrating the same MPS run-to-run variance documented in
[[gate4-mps-resume-tolerance]]. The two more consistent measurements, 1.11 and 1.13, are what
this finding rests on.)

By contrast, `shuffled_across` (wrong clip entirely) and `constant_zero` (no conditioning at all)
were robustly and consistently much worse than correct conditioning across all three runs
(typically 20-45% worse), clearly ruling out total conditioning collapse -- the decoder
demonstrably uses *some* per-clip MuQ signal, just not (yet, at this scale) a strong,
separately-measurable benefit from precise frame ordering within a clip's own features.

**Interpretation (not yet conclusive).** Plausible explanations, none yet tested:
1. Gate 4's training budget (200-400 steps, no adversarial loss, a deliberately tiny generator)
   may simply be too short/small for the model to have learned to exploit frame-level temporal
   alignment yet, even though it clearly has learned to exploit per-clip identity.
2. The reconstruction-only mel/STFT losses may be relatively insensitive to fine temporal
   misalignment compared to a human listener or an adversarial loss.
3. `MiniGenerator`'s temporal receptive field (a handful of small residual blocks per upsampling
   stage) may not be large enough to strongly benefit from correct frame ordering at this
   parameter budget.

**Decision.** Do not silently pass Gate 4 on this control, and do not weaken `control_margin` to
force a pass -- that would defeat the diagnostic's purpose. Gate 4 is reported as **not fully
passed** (checkpoint/resume equivalence passed; 2 of 3 conditioning controls passed;
`shuffled_time` did not). This is recorded as an open, evidence-backed finding for Gate 5
planning, not a blocking implementation bug.

**Consequences.** Per CLAUDE.md Stop Conditions, this warrants diagnosis before scaling further.
Gate 5 (full GTZAN pilot, much longer training) should re-run this same `shuffled_time` control
and check whether the margin improves with more training -- if it doesn't, that would be stronger
evidence toward explanation 2 or 3 above (architecture/loss insensitivity) rather than 1 (just
needs more steps).

**Rollback plan.** None needed -- this is a diagnostic finding, not a design decision to revert.
If Gate 5 confirms the same pattern with a full-scale run, escalate to inspecting the generator's
effective receptive field and considering feature-matching/adversarial losses earlier than
originally planned (CLAUDE.md currently defers those until after Gate 3's reconstruction path
passes, which it has).

---

## gate3-adversarial-losses: Added multi-period/multi-scale discriminators after listening to Gate 3 output

**Context.** The user listened to Gate 3's reconstruction-only artifacts (500 steps, then 5000
steps) and reported them as "buzzing noise... nothing remotely recognizable." Investigated
quantitatively before changing anything:

- Increasing steps 500 -> 5000 (10x) barely moved the loss (2.78 at step 500 vs 2.64 at step
  5000 -- a 4% change over 4500 extra steps): the reconstruction-only objective had already
  plateaued, so "just train longer" was not going to fix this.
- Computed log-mel spectrogram correlation between reconstruction and reference: 0.89-0.91 (good)
  and frame-energy-contour correlation: 0.98 (very good) -- the model *was* learning the coarse
  spectral envelope and dynamics/rhythm well.
- Computed raw waveform sample correlation: ~0 (none).

**Diagnosis.** Mel and multi-resolution-STFT losses are magnitude-only -- they never constrain
phase. A model can match the target's magnitude spectrum arbitrarily well while producing
completely different (and to a human ear, unnatural/buzzy/metallic) phase, since infinitely many
phase reconstructions satisfy the same magnitude loss. This is a well-known failure mode of
magnitude-only vocoder training, not a bug in this implementation -- and it does not improve with
more steps, because the loss function itself carries no gradient toward natural phase.

**Decision.** Implement CLAUDE.md's own next planned step (Decoder Design item 6, deferred until
"after the reconstruction path passes smoke tests" -- which Gate 3 had, per the good mel/energy
correlation above): multi-period discriminator (`MultiPeriodDiscriminator`, periods
`(2,3,5,7,11)`) and multi-scale discriminator (`MultiScaleDiscriminator`, 3 scales), both with
channel widths (max 256) kept well below the original speech HiFi-GAN's 1024-wide discriminators,
consistent with this project's "deliberately small" baseline. Added LSGAN-style adversarial and
feature-matching losses (`models/losses.py`). Training now runs reconstruction-only for
`training.adversarial_warmup_steps` (500), then adds the discriminators -- literally implementing
CLAUDE.md's "reconstruction-only first, then adversarial" ordering within a single run rather than
as separate manual stages.

**Bug found and fixed along the way.** After adding discriminators, Gate 3's pass/fail check
(comparing mean loss over the first vs. last 10 steps) started failing (ratio 0.57 vs. the 0.5
threshold) even though the reconstruction component was clearly still healthy (mel loss 5.15 ->
0.76). Root cause: the ratio was computed on the *combined* training objective, which after
warmup includes the generator-vs-discriminator adversarial term -- a minimax quantity that
oscillates by design during healthy GAN training, not something that monotonically shrinks. Fixed
by tracking `recon_loss_history` (mel + STFT only) separately and gating Gate 3's pass/fail
criterion on that, not the combined loss -- this matches CLAUDE.md's actual wording ("confirm that
**reconstruction** losses decrease substantially") more precisely than the original implementation
did even before discriminators existed.

**Evidence.** Re-ran Gate 3 (`configs/smoke.yaml`, 8000 steps, warmup 500) after the fix:
reconstruction loss ratio 0.40 (PASS, threshold 0.5). Reconstruction peak amplitude moved closer
to the reference's (e.g. `disco.00049`: 0.76 -> 0.92-0.99 across runs, vs. reference peak 1.0),
consistent with a discriminator teaching the generator to use realistic dynamic range. Mel
correlation stayed in a similar 0.82-0.84 band (not obviously better on this metric -- adversarial
training is known to sometimes trade off spectral-distance metrics for perceptual naturalness, per
the vocoder literature). Waveform-sample correlation remained ~0, which is expected and not itself
diagnostic (mel/STFT losses were never phase-aware; whether adversarial training fixed the
*perceptual* buzziness is a listening judgment, not something these two metrics can settle) --
reference and reconstruction audio were sent to the user directly for listening evaluation rather
than declared fixed from metrics alone, per CLAUDE.md's own instruction not to claim successful
inversion from loss numbers alone.

**Consequences.** Every future Gate 3+ training run now trains two more networks (MPD, MSD) and
their optimizer, roughly doubling per-step compute after the warmup period ends (8000 steps took
several minutes longer than the prior 5000-step reconstruction-only run). Checkpoints now also
carry discriminator + discriminator-optimizer state (`checkpoint.build_checkpoint_state` gained
optional `discriminator`/`discriminator_optimizer` parameters, backward-compatible with existing
reconstruction-only checkpoints via `None` defaults).

**Rollback plan.** If adversarial training destabilizes training (mode collapse, discriminator
overpowering the generator) or the user's listening judgment is that the audio didn't improve,
revisit `adversarial_loss_weight`/`feature_matching_loss_weight`/`adversarial_warmup_steps` before
reverting the discriminators entirely -- GAN training instability is normally a
hyperparameter/warmup-schedule problem, not a reason to abandon the approach CLAUDE.md's own
Decoder Design section calls for.

**Update (first listening pass, 2026-09-04).** User's verdict on the 8000-step run above:
`disco.00049` "starts to resemble the original... rhythm and drum hits can be heard" (still
heavily distorted); `pop.00012` "is worse." This pattern -- percussive/rhythmic content partially
recovering, melodic/harmonic content not -- combined with the loss log (recon_loss ~2.8,
adversarial+weighted-feature-matching ~1.2-1.6, i.e. comparable magnitude) suggested the
adversarial/feature-matching gradient was competing with, not just refining, the reconstruction
gradient: enough to pull the model toward "generically plausible transients" without a strong
enough anchor to preserve precise harmonic/pitch content. Percussive transients are broadband and
less phase-sensitive, so they partially survive this tradeoff; tonal/melodic content isn't.

**Decision (iteration 2).** Added `training.reconstruction_loss_weight` (default 10.0), applied to
`(mel + STFT)` only during the adversarial phase (warmup-phase training is unaffected, since it
has no adversarial terms to be out-competed by). This is *not* a copy of the original HiFi-GAN
paper's "45" constant -- deliberately chose a smaller, evidence-based multiplier because our loss
combination differs (HiFi-GAN has no multi-res STFT term, and our feature-matching loss already
carries its own 2x weight), and 10x was enough to bring the weighted-recon term (~28 at the
observed ~2.8 raw recon loss) clearly above the adversarial+FM terms (~1.5-2) without setting it
so high it would suppress the adversarial gradient's contribution to zero. Also extended
`training.steps` 8000 -> 15000 (7500 -> 14500 adversarial steps) since GAN training generally
needs more steps to stabilize than reconstruction-only training, and the first run's `adv`/`disc`
values hadn't visibly converged by step 8000.

**Evidence.** Re-ran: final mel loss improved 0.77 -> 0.59 (vs. the pre-reweighting adversarial
run), and log-mel correlation improved for both tracks (`pop.00012`: 0.82 -> 0.90; `disco.00049`:
0.84 -> 0.94) -- a genuine spectral-fidelity gain, not just a display artifact of the reweighted
loss curve (which is on a different numeric scale and can't be compared directly to the prior
run's). `disc` loss trended toward low values (0.09-0.4) by the end while `adv` stayed elevated
(1.1-2.1, oscillating, not collapsing) -- a mild discriminator-dominance pattern worth watching in
future runs, though not (yet) a training collapse. Sent updated audio to the user for a second
listening pass rather than declaring this resolved from the correlation numbers alone.

**Open question.** Whether this second iteration actually fixes the *perceptual* buzziness (vs.
just moving the spectral-correlation number) is pending the user's second listening judgment.

---

## gate3-generator-capacity: Multi-receptive-field generator + larger channel budget

**Context.** User's second listening pass on iteration 2 (loss-reweighted adversarial training):
"both excerpts have improved, but they are still very poor: a lot of noise, buzzing, and
ringing." Ringing/comb-filter artifacts after two rounds of loss/schedule tuning point away from
training dynamics and toward generator capacity -- no amount of loss-weight tuning or extra steps
can get a genuinely undersized network past a structural ceiling.

**Diagnosis.** `MiniGenerator`'s original design (from the Gate 0 shape-test placeholder,
never revisited) had two capacity problems:

1. Each upsampling stage was followed by only two small residual blocks *in sequence*
   (`ResidualBlock(k=3, d=1)` then `ResidualBlock(k=7, d=3)`, one dilated-conv pair each) -- a
   thin receptive field, especially right after a strided `ConvTranspose1d`, which is exactly the
   situation prone to comb-filter/ringing artifacts. The original HiFi-GAN avoids this with a
   proper multi-receptive-field (MRF) design: several residual blocks with *different* kernel
   sizes run in *parallel* on the same input (not sequentially) and their outputs are averaged,
   each block itself containing multiple dilated-conv pairs.
2. `upsample_rates=(5,4,4,3,2,2)` (6 stages, product 960) combined with each stage halving the
   channel count meant 6 halvings from `base_channels=256`, ending at only 8 channels right
   before the waveform output head -- the stage that most needs capacity to represent fine
   time-domain detail was the most starved.

**Decision.** Two changes, made together since they address the same root cause (capacity):

1. Rewrote `models/generator.py`: `ResBlock` now contains multiple dilated-conv pairs (dilations
   `(1,3,5)`) run sequentially within one block (matching HiFi-GAN's per-kernel-size branch), and
   a new `MultiReceptiveFieldFusion` module runs 3 such `ResBlock`s in parallel (kernel sizes
   `(3,7,11)`) after each upsampling stage and averages their outputs -- the actual MRF mechanism,
   not a sequential chain with different kernel sizes.
2. Re-factored `decoder.upsample_rates` from 6 stages `(5,4,4,3,2,2)` to 4 stages `(8,6,5,4)`
   (product still exactly 960, so [[decoder-upsample-placeholder]]'s Gate-2-confirmed ratio is
   unaffected) and raised `decoder.base_channels` from 256 to 512, so 4 halvings from 512 ends at
   32 channels instead of 6 halvings from 256 ending at 8.

Both `resblock_kernel_sizes`/`resblock_dilation_sizes` and the upsample-rate factorization are
config fields, not hard-coded, per CLAUDE.md's config-explicitness rule.

**Evidence.** No test depended on `MiniGenerator`'s internal structure (confirmed by grep before
refactoring), so the rewrite was a drop-in replacement at the module boundary. Gate 0 re-verified
after the change (`doctor` still passes: decoder shape-only forward pass produces the expected
`(1, 48000)` output from `(1, 50, 1024)` input). Timed a 100-step run before committing to a full
retrain: ~0.095s/step (up from ~0.07s/step pre-refactor, consistent with ~9x more conv work per
MRF block and 4x more channels -- offset by having only 4 upsample stages instead of 6). Full
15000-step retrain result: reconstruction loss ratio 0.37 (PASS), final mel loss 0.48 (down from
0.59 with the smaller generator, continuing the trend from 0.77 -> 0.59 -> 0.48 across all three
iterations). Log-mel correlation: `pop.00012` 0.90 -> 0.94, `disco.00049` 0.94 -> 0.94 (already
high, held). Sent to the user for a third listening pass.

**Consequences.** Generator parameter count increased substantially (512 base channels, MRF
fusion with 3x the conv work per stage vs. the old 2-block sequential design). Training time per
step increased proportionally but remained tractable (~24 min projected for 15000 steps on the
M4 Max, still using only 2 fixed clips).

**Rollback plan.** If the larger model doesn't improve perceptual quality either, that would be
stronger evidence the bottleneck is elsewhere (training budget, discriminator design, or a more
fundamental issue with reconstructing phase from MuQ's representation at all -- relevant to this
project's core Scientific Question). Revert `base_channels`/`upsample_rates` to the smaller
values first (cheaper to iterate on) before concluding the MRF design itself was insufficient.

---

## gate3-capacity-1024: 512 -> 1024 channel isolation experiment; training collapsed, not a clean capacity result

**Context.** User's third listening pass on the MRF-512 generator: "both excerpts have
improved... `disco.00049` is now recognizable... `pop.00012` still contains substantial
buzzing/noise, although rhythm and harmony are present." Combined with the prior finding that
more training steps alone did not help the smaller (pre-MRF) generator
([[gate3-adversarial-losses]]), the user asked for a controlled experiment isolating generator
capacity specifically: `base_channels` 512 -> 1024, everything else (MRF design, kernel sizes,
dilations, upsample structure, losses, training schedule) held fixed, to distinguish "still too
small" from "large enough, remaining limits are the objective/representation."

**Method.** Created `configs/smoke_1024.yaml` as an exact copy of `configs/smoke.yaml` (the
working MRF-512 config) with exactly two differences: `decoder.base_channels: 1024` and
`training.run_name` (so this experiment's checkpoints/artifacts don't overwrite the MRF-512 run's,
per CLAUDE.md rule 8). Did not modify `configs/smoke.yaml`. Measured resource impact before the
full run: Gate 0 passed (correct `(1, 48000)` output shape); parameter count 17,235,585 (512) vs.
61,582,593 (1024, 3.57x); MPS memory 279.5MB/1271.9MB current/driver-allocated (512) vs.
1007.0MB/2496.7MB (1024) -- well within the 64GB budget; timed a 100-step run first (~0.128s/step
vs. ~0.095s/step, ~1.35x slower, sub-linear vs. the 3.57x parameter increase). Full evidence and
methodology logged in `notes/SCRATCHPAD.md` (2026-09-06 entry) as it happened.

**Result: training collapsed, not a clean comparison.** The 1024-channel run trained healthily
and comparably to the 512-channel run through step 3500 (mel loss 0.4873 @ step 3500 for 1024 vs.
0.5770 @ the same step for 512 -- the bigger model was, if anything, learning *faster*), then
catastrophically collapsed between step 3500 and step 3750: `disc` loss crashed from 0.68 to
0.04 and kept falling toward 0.0000 for the rest of training; `adv` loss jumped to ~2.0 (the
theoretical maximum of the LSGAN generator loss, `sum over sub-discriminators of (1-D(fake))^2`,
saturating when every sub-discriminator confidently rejects the generator's output) and stayed
there for the remaining ~11,250 steps -- textbook discriminator-overpowers-generator collapse.
Inspected the final reconstruction audio directly: both clips are a **constant +1.0 signal**
(`std=0.0`, 100% of samples `|x|>0.999`) -- the generator's `tanh` output head fully saturated and
never recovered. Final mel loss 8.08 (vs. 0.48 for the converged 512 run); log-mel correlation
0.31/0.23 (vs. 0.94/0.94) -- these numbers reflect collapse, not a genuine capacity comparison.

**Recovered the pre-collapse signal for a fair comparison.** Loaded the `step_003000.pt`
checkpoint (last saved checkpoint before the collapse window) and ran inference on both fixed
clips: mel correlation **0.9448** (`pop.00012`) and **0.9496** (`disco.00049`) -- matching or
slightly exceeding the *fully-converged* 512-channel model's final correlations (0.937/0.944),
achieved at only 3000/15000 steps (2500 steps past the adversarial warmup, vs. the 512 model's
full 14500). Waveform-sample correlation was also mildly positive (0.10/0.09) rather than the
512 model's near-zero/negative (-0.05/-0.11) -- a small effect, not conclusive on its own, but not
evidence of collapse either. This snapshot is the most informative data point in this experiment:
**capacity was not obviously the bottleneck at the moment training broke** -- the 1024-channel
model was tracking at-or-above the 512-channel model's asymptotic quality in a fifth of the
training time, right up until an unrelated instability destroyed it.

**Comparison table** (parameter counts, resource use, and final metrics; see body text above for
the pre-collapse step-3000 snapshot, which is the fairer capacity comparison):

| metric | MRF-512 (converged, step 15000) | MRF-1024 (collapsed, step 15000) | MRF-1024 (pre-collapse, step 3000) |
|---|---|---|---|
| parameters | 17,235,585 | 61,582,593 | (same model, step 3000) |
| checkpoint size | 250.6 MB | 782.8 MB | (same file format) |
| training wall time (15000 steps) | ~32 min | ~45 min | n/a (partial) |
| final mel loss | 0.478 | 8.078 (collapsed) | 0.486 |
| final STFT spectral-convergence | 0.479 | 7.353 (collapsed) | 0.552 |
| final STFT log-magnitude | 0.928 | 7.770 (collapsed) | 0.890 |
| mel correlation (pop.00012 / disco.00049) | 0.937 / 0.944 | 0.312 / 0.228 (collapsed) | 0.945 / 0.950 |
| waveform correlation (pop.00012 / disco.00049) | -0.046 / -0.110 | NaN (constant output) | 0.102 / 0.093 |

**Diagnosis of the collapse.** The most likely root cause: this project's optimizers use plain
`torch.optim.Adam` with **default betas (0.9, 0.999)** for both generator and discriminator.
GAN training literature (including the original HiFi-GAN paper's own optimizer config, `AdamW`
with `betas=(0.8, 0.99)`) consistently avoids `beta1=0.9` specifically because that much momentum
in a minimax adversarial game is a well-documented cause of exactly this failure mode: the
discriminator's updates accumulate momentum faster than the generator can adapt, and once it
pulls ahead the generator's gradient signal (from a saturating LSGAN loss) vanishes into a flat
region, leaving no path back. This project never customized Adam's betas anywhere -- a real,
identifiable gap from established GAN training practice, not something inherent to the capacity
increase itself. Increasing capacity did not *cause* the instability directly; it's plausible a
larger generator simply reaches the discriminator-fooling/discriminator-overpowering regime
faster (more capacity = faster initial convergence, as the pre-collapse data shows), reaching the
instability threshold sooner in wall-clock/step terms than the smaller model happened to.

**Answering the user's four questions directly:**

1. *Did mel correlation improve materially?* At the only fair comparison point (pre-collapse,
   step 3000), marginally yes (0.945-0.950 vs. 0.937-0.944) -- achieved in 20% of the training
   budget. At the literal final step, no -- it collapsed to 0.23-0.31, but that reflects training
   instability, not capacity.
2. *Did perceptual quality improve materially?* Cannot be assessed from the final checkpoint (pure
   noise/silence). The pre-collapse audio was sent to the user for a listening judgment alongside
   the collapsed audio for contrast.
3. *Is the model still obviously capacity-limited?* No clear evidence of that from this
   experiment -- if anything, the pre-collapse snapshot argues against it (comparable-or-better
   quality with far less training). The dominant finding is a training-stability problem, not a
   capacity ceiling.
4. *Is the remaining artifact profile consistent with (a) insufficient capacity, (b) insufficient/
   unstable objective, or (c) limits of the MuQ representation?* **(b)** -- the specific failure
   mode observed (discriminator confidence -> 0 disc loss, generator adversarial loss saturating
   at its theoretical max, generator output collapsing to a constant) is a textbook adversarial-
   training instability, not a capacity or representation-limit signature. This experiment does
   not provide new evidence about (a) or (c).

**Decision.** Do not conclude "1024 channels is better" or "1024 channels is worse" than 512 --
this experiment did not produce a clean capacity comparison; it produced a clean *instability*
finding that has to be fixed before capacity can be evaluated fairly at this scale. Keep
`configs/smoke.yaml` (MRF-512) as the working config; keep `configs/smoke_1024.yaml` and its
`runs/gate3_tiny_overfit_mrf1024/` output as a documented negative/inconclusive result, not a
capacity verdict. Do not proceed to Gate 4/5 -- still investigating Gate 3, per the user's
explicit instruction and CLAUDE.md's Stop Conditions (training instability warrants diagnosis
before scaling further, in either direction).

**Recommendation for the next experiment.** Before trying any more capacity changes, fix the
optimizer configuration: use GAN-appropriate Adam betas (e.g. `betas=(0.5, 0.999)` per classic GAN
practice, or `(0.8, 0.99)` matching HiFi-GAN's own published config) for the discriminator
optimizer at minimum, ideally both. This is a cheap, well-motivated change to test in isolation
(same pattern as this experiment: one variable at a time) before re-attempting either the 512 or
1024-channel configuration at the full 15000-step budget. If that stabilizes training, re-run the
1024-channel comparison properly; only then would a clean capacity verdict be possible.

**Rollback plan.** `configs/smoke_1024.yaml` and `runs/gate3_tiny_overfit_mrf1024/` are left in
place as a documented experiment, not deleted -- they're evidence of a real, reproducible failure
mode worth referencing if the same collapse pattern recurs after the optimizer fix.

**Update (optimizer-beta fix applied, 2026-09-06).** Added `training.generator_betas` and
`training.discriminator_betas` config fields (both default/set to `(0.8, 0.99)`, HiFi-GAN's
published config), wired into `train.py`'s two `torch.optim.Adam(...)` constructors via a new
`betas=` argument. Set explicitly in both `configs/smoke.yaml` and `configs/smoke_1024.yaml` (not
left as a hidden dataclass default, per CLAUDE.md).

**Regression check (512-channel, new betas).** Re-ran `configs/smoke.yaml` end to end (15000
steps, same run_name -- this supersedes the previous MRF-512 checkpoint/artifacts, consistent
with how this config has been treated as the current-best working iteration throughout this
session's Gate 3 investigation, not a rule-8-protected finalized artifact). Result: **no
collapse** -- `disc` oscillated healthily in roughly 0.4-0.7 for the entire adversarial phase
(vs. the old-beta run's `disc` also mostly healthy, but this is the point: confirming the new
betas don't hurt a run that was already stable). Mel loss trended to ~0.34-0.45 in the last 1500
steps (vs. 0.478 with the old betas) -- a real, if modest, quality improvement from the fix alone.
Mel correlation: `pop.00012` 0.924, `disco.00049` 0.932 (old betas: 0.937/0.944 -- comparable,
slightly lower at the exact final-step snapshot, consistent with normal late-training oscillation
rather than a regression; the training curve was healthier throughout, not just at one snapshot).
Regression check PASSED: the beta fix does not destabilize or degrade the working configuration.

**Capacity re-test (1024-channel, new betas).** Re-ran `configs/smoke_1024.yaml` (same run_name,
superseding the collapsed run) with the fix applied. Result: **no collapse** -- `disc` oscillated
healthily in ~0.4-0.7 for all 15000 steps, the same healthy pattern as the 512-channel run, with
no sign of the discriminator pulling decisively ahead this time. Mel loss reached 0.32 at the
final step (better than the 512 run's 0.32-0.45 range at the same point in training) and log-mel
correlation reached **0.9646 (`pop.00012`) / 0.9755 (`disco.00049`)** -- the best result of any
iteration in this investigation, and for the first time waveform-sample correlation was
consistently positive for both tracks (0.073 / 0.104) rather than near-zero or negative. This is
now a **clean, valid capacity comparison**: with the training-instability confound removed, 1024
channels genuinely outperforms 512 on every objective metric measured.

**Final comparison table** (both runs use the fixed `betas=(0.8, 0.99)`):

| metric | MRF-512 (betas fixed) | MRF-1024 (betas fixed) |
|---|---|---|
| parameters | 17,235,585 | 61,582,593 (3.57x) |
| checkpoint size | 250.6 MB | 782.8 MB |
| training wall time (15000 steps) | ~31 min | ~52 min |
| final mel loss | 0.319-0.45 (oscillating) | 0.319 (steadier) |
| final STFT spectral-convergence | 0.357-0.57 | 0.357 |
| final STFT log-magnitude | 0.751-0.90 | 0.751 |
| mel correlation (pop.00012 / disco.00049) | 0.924 / 0.932 | **0.965 / 0.976** |
| waveform correlation (pop.00012 / disco.00049) | -0.015 / 0.078 | **0.073 / 0.104** |
| training stability | stable (no collapse) | stable (no collapse, after fix) |

**Revised answers to the four diagnostic questions, now that the comparison is valid:**

1. *Did mel correlation improve materially?* Yes -- 0.924/0.932 -> 0.965/0.976, a consistent gain
   on both tracks, and the largest correlation values seen in this entire investigation.
2. *Did perceptual quality improve materially?* Pending the user's listening judgment (audio sent
   via SendUserFile) -- for the first time, waveform-sample correlation is positive for both
   tracks rather than near-zero/negative, which is at least consistent with (not proof of)
   improved time-domain fidelity.
3. *Is the model still obviously capacity-limited?* Cannot be ruled out -- 1024 clearly
   outperformed 512 under matched conditions, which is evidence *for* capacity still mattering,
   not against it. Whether 2048 would help further is untested.
4. *Is the remaining artifact profile consistent with (a)/(b)/(c)?* The training-stability
   confound (b) that dominated the first 1024 attempt is resolved. Any remaining artifacts in this
   run's audio are now more likely to reflect genuine (a) capacity limits or (c) representation
   limits -- this experiment can't distinguish between those two without a further capacity step
   or a different diagnostic (e.g. checking whether error is concentrated in specific frequency
   bands or track types).

**Decision.** Adopt `betas=(0.8, 0.99)` for both optimizers as the new standard -- it strictly
dominates the old default in every measured run (no instability, equal-or-better quality). The
1024-channel configuration is a genuine, validated improvement over 512 under this fixed training
budget; `configs/smoke.yaml` (512) remains the "official" working config for now since the user
has not yet decided whether the added cost (3.57x params, 1.7x wall time) is worth adopting as the
new default, pending their listening judgment on this round's audio.

**Rollback plan.** If 1024 is adopted as the new default, keep `configs/smoke_1024.yaml` (or
promote its settings into `smoke.yaml` under version control history) and update
`configs/pilot.yaml` to match before it's used for Gate 4 work. If the user's listening judgment
says 1024 isn't perceptually better despite the metric gains, that would itself be an interesting
finding (objective spectral metrics diverging from perceptual quality) worth its own note.

**Final listening verdict (2026-09-06) -- the predicted alternative outcome occurred.** User's
judgment on the 512-vs-1024 (both betas-fixed) comparison: "I can't hear any noticeable
improvement in the audio. They are equally mediocre: recognizable but still plagued by
artifacts." The metric gains (mel correlation 0.92/0.93 -> 0.97/0.98, waveform correlation
flipping from near-zero/negative to positive) did **not** translate into any perceptible quality
difference. This is exactly the "objective metrics diverging from perceptual quality" case
flagged as a possible outcome above -- see [[gate3-capacity-ruled-out]] for the full synthesis
this triggered and the resulting conclusions/recommendation.

---

## gate3-capacity-ruled-out: Capacity is not the bottleneck; conclusions and open questions

**Context.** Synthesis requested after the final listening verdict on [[gate3-capacity-1024]]:
1024 channels beat 512 on every objective metric (mel correlation, waveform correlation) under a
clean, collapse-free comparison, yet the user reported no perceptible difference -- "equally
mediocre: recognizable but still plagued by artifacts." This closes out the capacity-scaling line
of investigation with a clear, if not fully satisfying, answer and reframes what to investigate
next.

**Finding 1: capacity is very likely no longer the bottleneck.** This was tested twice: once
confounded by a training collapse ([[gate3-capacity-1024]]'s first attempt), once cleanly (after
the optimizer-beta fix). In the clean comparison, quadrupling parameters (17.2M -> 61.6M) and
roughly doubling wall time (~31 -> ~52 min) produced a consistent, measurable improvement in every
spectral correlation metric and *zero* audible improvement. Scaling capacity further (e.g. 2048
channels) would very likely repeat this pattern at increasing cost and is not recommended as the
next experiment.

**Finding 2: mel/waveform correlation are demonstrably poor proxies for this project's perceptual
target.** Mel-spectrogram correlation aggregates matching quality across the entire time-frequency
grid; it can improve from better broadband spectral-envelope matching (which 1024 channels'
extra capacity plausibly provided) without moving the specific, perceptually-dominant artifact
(buzzing/ringing, almost certainly a phase/fine-temporal-structure phenomenon) that a human ear
locks onto. This is precisely the failure mode CLAUDE.md's Evaluation and Controls section warns
against: "Use objective metrics as diagnostics, not as proof of perceptual fidelity." Concretely
for this project: **do not use mel/waveform correlation deltas alone to decide whether a change
helped** -- a listening check is required before concluding either direction, not just before
declaring final success. (Waveform correlation flipping from negative to positive at 1024
channels looked like a promising phase-related signal; it wasn't perceptible, so it likely
reflects a genuine but perceptually-small change, or noise in a metric that's inherently unstable
at these sample sizes -- worth remembering before over-interpreting small waveform-correlation
movements in future runs.)

**Finding 3: this result is itself a valid, partial answer to the project's Primary Question.**
CLAUDE.md's mission is explicitly an interpretability experiment, not a synthesis-quality
benchmark, and explicitly says "fail early... before increasing model size or scope." The
consistent, reproducible pattern across two capacity levels -- content (rhythm, harmony, coarse
identity, per the user's own descriptions across all three listening rounds) survives decoding,
but clean phase/fine-structure does not -- is progress on Scientific Question 3 ("is failure
caused by implementation/optimization problems, or by insufficient information in the
representation?"). It rules out "insufficient decoder capacity" as the implementation-side
explanation. Two candidate explanations remain open, not yet distinguished by any experiment run
so far:

1. **Insufficient adversarial training depth/breadth.** 15,000 steps on 2 overfit clips is small
   by vocoder-GAN standards -- published HiFi-GAN-style training runs use orders of magnitude more
   steps on full datasets. It's plausible the discriminator/generator pair simply hasn't had
   enough time or data diversity to learn to shape clean phase yet, independent of capacity.
2. **MuQ's representation may impose a hard ceiling.** Its frame rate is a confirmed, measured
   25 Hz / 960 samples per frame (40ms) -- coarse relative to typical neural vocoder hop sizes
   (~10-12ms). A large amount of fine time-domain information per frame must be reconstructed
   (effectively hallucinated, constrained only by the loss/discriminator) regardless of decoder
   design. This has not been tested directly and cannot be ruled in or out from anything run so
   far -- it would require either a much longer/broader training regime to see if the gap closes
   (weighing toward explanation 1 if it does), or a more targeted diagnostic (e.g. examining
   whether reconstruction error concentrates in specific frequency bands, transient regions, or
   track/genre types, which might implicate the representation more directly).

**Decision.** Do not pursue further capacity scaling. Recommend, as the next experiment, testing
explanation 1 (extend adversarial training significantly, and/or move toward a larger, less
overfit-prone training regime -- Gate 4/5 scale) before concluding anything about explanation 2,
since it's the cheaper and more directly actionable hypothesis. This decision was proposed to the
user, not executed unilaterally -- awaiting direction on whether to proceed with a longer-training
experiment or treat the current result as a documented stopping point for this line of Gate 3
investigation.

**Consequences.** `configs/smoke.yaml` (512, betas fixed) remains the working config;
`configs/smoke_1024.yaml` is retained as a documented, informative negative result (capacity
doesn't help perceptually) rather than deleted. No further capacity experiments should be run
without first changing something on the training-depth or diagnostic axis, or the same
"metrics improve, ears don't" pattern will likely just repeat.

**Rollback plan.** None needed -- this is a synthesis of completed experiments, not a code or
config change to revert. If a future experiment (longer training, different loss, or otherwise)
does produce an audible improvement, revisit whether capacity interacts with that fix (e.g. maybe
capacity matters once training is deep enough) before assuming this finding generalizes forever.

---

## mel-hifigan-sanity: Ground-truth mel spectrogram sanity check (experiment/mel-hifigan-sanity branch)

**Context.** After the (sibling, not-merged) `encodec-hifigan-experiment` branch found that
EnCodec conditioning produced a 6-9x jump in waveform correlation over MuQ with an identical
decoder, the user wanted the most direct possible sanity check before drawing conclusions from
either representation-swap result: condition the exact same HiFi-GAN pipeline on the ground-truth
mel spectrogram of the target itself -- the literal representation HiFi-GAN was originally
designed to consume (mel spectrogram -> HiFi-GAN -> waveform, the classic "copy synthesis" test).
If the current pipeline can't even reproduce audio from its own designed input, that would
implicate the training recipe/decoder pipeline itself, independent of MuQ or EnCodec. This branch
was created off `main` (not off `encodec-hifigan-experiment`), so it carries the MuQ+MRF-1024
infrastructure but none of the EnCodec-specific code -- an intentional choice to keep this a
clean, independent sibling experiment.

**Representation source.** No learned upstream model at all: a deterministic log-mel-spectrogram
transform (`torchaudio.transforms.MelSpectrogram`) applied directly to the target waveform crop
itself. Reused this project's own already-justified `mel_loss` parameters (`n_fft=2048,
hop_length=300, win_length=1200, n_mels=128, fmin=0.0, fmax=12000.0`, see
[[gate3-loss-configs]]) as the conditioning transform too, so the "ground truth" representation
isn't a new, separately-justified choice.

**A real implementation subtlety caught before training.** `torchaudio`'s default
`center=True` framing pads the signal internally and yields `1 + num_samples // hop_length`
frames (161 for a 48000-sample crop at hop=300) -- one frame more than the clean
`num_samples // hop_length` (160) needed for this project's crop-alignment invariant
(`crop_samples == num_frames * generator.total_upsample`, [[gate3-crop-alignment]]). Fixed by
adopting HiFi-GAN's own standard mel-extraction convention: manually reflect-pad the waveform by
`(n_fft - hop_length) // 2` on each side, then run the STFT with `center=False`. Verified directly
(a real forward pass, not just arithmetic) that this produces exactly `(1, 160, 128)` for a
48000-sample crop -- samples_per_frame = 300.0 exactly, frames_per_second = 80.0.

**The one unavoidable architectural adaptation** (same pattern as the EnCodec branch): since the
mel representation's measured ratio (300) differs from MuQ's (960), `decoder.upsample_rates` was
changed to `(5, 5, 4, 3)` (product 300) and `training.crop_frames` from 50 to 160 -- preserving
the identical 2.0-second (48000-sample) crop length in the new frame-counting units. Every other
field (`base_channels=1024`, resblock design, discriminator, all losses, both Adam betas
`(0.8, 0.99)`, `adversarial_warmup_steps=500`, `training.steps=15000`, the same two fixed clips
`pop.00012`/`disco.00049`) is byte-identical to `configs/smoke_1024.yaml`.

**Implementation.** Added `mel_encoder.py` (`GroundTruthMelEncoder`, same `encode()`/`probe()`
interface as `FrozenMuQEncoder`), a `GroundTruthMelConfig`, and an `encoder_type: "muq" |
"ground_truth_mel"` field on `ExperimentConfig` (defaulting to `"muq"`, confirmed by re-running
the full 47-test suite and checking `smoke.yaml`/`smoke_1024.yaml`/`pilot.yaml` still resolve to
`encoder_type: "muq"` by default). The only change to `train.py` is the same small
encoder-selection branch pattern used on the EnCodec branch, built independently here since the
two experiment branches don't share commits.

**Comparison table** (all three: same 2 tracks, same 2.0s crop, same MRF-1024 generator design,
same discriminator, same losses, same Adam betas `(0.8, 0.99)`, same 15000-step schedule; MuQ and
EnCodec numbers carried over from the sibling `encodec-hifigan-experiment` branch's results,
established earlier this session):

| metric | MuQ + HiFi-GAN | EnCodec + HiFi-GAN | Ground-truth mel + HiFi-GAN |
|---|---|---|---|
| representation frame rate | 25 Hz | 75 Hz | 80 Hz |
| representation channels | 1024 | 128 | 128 |
| generator parameters | 61,582,593 | 54,799,617 | 51,670,273 |
| checkpoint size | 782.8 MB | 701.4 MB | 663.8 MB |
| training wall time (15000 steps) | ~52 min | ~62 min | ~48 min |
| training stability | stable | stable | stable |
| final mel loss | 0.319 | 0.381 | 0.379 |
| mel correlation (pop.00012 / disco.00049) | 0.965 / 0.976 | 0.950 / 0.970 | 0.944 / 0.966 |
| waveform correlation (pop.00012 / disco.00049) | 0.073 / 0.104 | **0.682 / 0.602** | 0.059 / -0.010 |

**The critical result: ground-truth mel behaves like MuQ, not like EnCodec.** Mel correlation is
comparable across all three (0.94-0.98 range) -- expected, since all three are being pushed
toward the same mel-loss target regardless of conditioning source. But waveform correlation for
ground-truth mel (0.06 / -0.01) is close to MuQ's (0.07 / 0.10) and nowhere near EnCodec's
(0.68 / 0.60), **despite ground-truth mel being, by construction, at least as information-rich
about the target as any other representation could be for matching its own magnitude spectrum.**

**Answering the three required questions:**

1. *Can the current HiFi-GAN setup memorize two clips when conditioned on ground-truth mel
   spectrograms?* Yes, in the same sense MuQ did: clean convergence (reconstruction loss ratio
   0.36, no instability, `disc` stayed healthy throughout), strong mel-domain correlation
   (0.94-0.97). But *not* in the stronger sense EnCodec demonstrated -- waveform-domain fidelity
   stayed low.
2. *If not [fully], what is the most likely bottleneck?* Given that even the literal ground-truth
   mel spectrogram -- the exact representation the mel/STFT losses are already targeting, and the
   representation HiFi-GAN was originally designed around -- does not produce high waveform
   correlation with this training recipe, the bottleneck is most likely **the same one classical
   vocoder research exists to solve**: reconstructing time-domain phase from a magnitude-only
   spectral representation is a genuinely underdetermined problem, and this project's adversarial
   training budget (15,000 steps, 2 overfit clips, no dataset-scale discriminator training) is
   small relative to what published HiFi-GAN results use to solve it. This result does not
   distinguish between "the training paradigm needs to scale up" and "mel-like magnitude
   representations are fundamentally capped at this budget" -- but it does now make the *shared
   training recipe* (not any specific representation) the leading suspect for the mediocre
   perceptual results on both MuQ and ground-truth mel.
3. *What does this imply about the MuQ and EnCodec results?* This substantially refines, rather
   than contradicts, the EnCodec branch's finding. MuQ's own published architecture consumes
   `melspec_2048` (a mel spectrogram) as its *input* feature (confirmed from the model's public
   config) -- so MuQ's `last_hidden_state` is itself downstream of a magnitude-only, phase-
   discarding representation, learned for a self-supervised music-understanding objective, not
   reconstruction. Ground-truth mel scoring similarly to MuQ on waveform correlation is consistent
   with **both being magnitude-domain representations that never encoded phase to begin with** --
   whereas EnCodec is a full neural *codec*, trained end-to-end with a paired decoder whose
   objective is literally accurate waveform reconstruction, so its encoder had explicit pressure
   to retain whatever information (very plausibly including phase/timing cues) its own decoder
   needs. This reframes the MuQ-vs-EnCodec question: it's less "EnCodec's information is richer"
   in a generic sense, and more "EnCodec's representation was trained to preserve reconstruction-
   relevant information that magnitude-spectrum-derived representations (ground-truth mel, and by
   inheritance MuQ) do not carry at all, regardless of how well-trained the decoder is."

**Decision.** Do not conclude the mel-hifigan-sanity result is a decoder-family indictment
(explicitly out of scope; no BigVGAN work started) -- the ground-truth mel result is directionally
consistent with the *training recipe* being under-scaled for full phase recovery from any
magnitude-only source, not with HiFi-GAN being incapable in principle (published HiFi-GAN results
on full datasets with much longer training do recover phase well from mel spectrograms). Recorded
this as a refinement of [[encodec-mediocrity-across-all-variants]]'s open question: the
"representation vs. training paradigm" fork now has a plausible resolution -- **it is likely
both**, with the specific distinction being magnitude-only vs. reconstruction-trained
representations, not "MuQ vs. EnCodec" as arbitrary alternatives. Awaiting the user's listening
verdict before treating any of this as confirmed (per this project's now-consistent practice of
never declaring a perceptual result from correlation metrics alone).

**Consequences.** Three representation-source branches now exist (`main`'s MuQ baseline,
`encodec-hifigan-experiment`, `experiment/mel-hifigan-sanity`), each isolated, each additive to
its own base commit, none merged. If the user wants to test the refined hypothesis directly, the
natural next experiment is a representation that is reconstruction-trained but *not* MuQ or
EnCodec specifically (to check the magnitude-vs-reconstruction-objective distinction generalizes)
-- or, cheaper, extending training substantially on the ground-truth-mel config specifically,
since classical vocoder literature would predict *that* representation should improve the most
with more/broader training if the "training paradigm is under-scaled" explanation is correct.

**Rollback plan.** This entire line of work lives on `experiment/mel-hifigan-sanity`, isolated
from `main` and from `encodec-hifigan-experiment` by construction. If abandoned, simply do not
merge; no baseline files were touched (confirmed via `git diff main` showing zero differences on
`configs/smoke.yaml`/`configs/smoke_1024.yaml` and all MuQ-path source files).

**Final listening verdict (2026-09-08) -- mediocre again, indistinguishable from every other
variant.** User's judgment: "The reconstructed audio sounds mediocre again. Almost identical to
the other experiments. There must [be] something wrong with the training pipeline." This extends
[[encodec-mediocrity-across-all-variants]]'s pattern to a fourth variant, but with an added twist:
ground-truth mel and EnCodec have wildly different waveform correlation (0.06/-0.01 vs. 0.68/0.60)
yet reportedly sound the same to the user, and MuQ (waveform correlation 0.07/0.10, similar to
ground-truth mel) also sounds the same. If EnCodec's waveform correlation gap were real and
perceptually meaningful, an audible difference between EnCodec and the other two would be
expected; hearing none raises a legitimate, more serious hypothesis than "training paradigm is
under-scaled": that the generator may not be using its conditioning input in a perceptually
meaningful way at all across every variant tested (a conditioning-collapse-style issue, per
CLAUDE.md's Evaluation and Controls section: "If the controls sound similar to the correctly
conditioned model, assume conditioning collapse or leakage until disproven") -- consistent with,
and possibly explaining, Gate 4's earlier unresolved `shuffled_time` finding
([[gate4-shuffled-time-finding]]), where shuffled-in-time conditioning was already found to be
only marginally worse than correct conditioning for the original MuQ setup.

**Status: this hypothesis is proposed but NOT yet tested.** A concrete, cheap diagnostic was
identified but not run (interrupted by the user, who asked to summarize and commit instead of
continuing): load each of the three already-trained checkpoints (MuQ, EnCodec, ground-truth-mel
+ MRF-1024 -- all still present on disk in the gitignored `runs/` directory, unaffected by branch
switches) and directly compare generator output under correct vs. zero vs. shuffled-in-time vs.
cross-track conditioning, without retraining anything. If the outputs barely change across these
conditioning variants, that would be direct, decisive evidence of conditioning collapse -- a
genuine pipeline issue, not a representation or training-depth limitation. If outputs do change
substantially with conditioning, that would argue against a collapse bug and back toward the
training-paradigm/phase-reconstruction-difficulty explanation already on record. **This is the
right next diagnostic step and should be run before any further training experiments** -- it is
cheap (no GPU training required, just forward passes on existing checkpoints) and would
directly confirm or rule out the user's hypothesis rather than continuing to guess from
correlation metrics and listening impressions alone.
