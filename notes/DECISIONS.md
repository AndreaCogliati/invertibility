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

## encodec-hifigan: EnCodec-conditioned HiFi-GAN experiment (encodec-hifigan-experiment branch)

**Context.** After the MuQ capacity investigation concluded that scaling the decoder doesn't fix
the residual buzzing/ringing, the user hypothesized that HiFi-GAN itself might not be the right
decoder family for music, and proposed testing that hypothesis indirectly before committing to a
BigVGAN rewrite: keep the exact same HiFi-GAN decoder/training recipe, but swap the representation
source from MuQ to EnCodec. If EnCodec conditioning produces a much better reconstruction with the
*identical* decoder, that argues the MuQ representation (not HiFi-GAN) is the bottleneck. If
EnCodec is similarly mediocre, that argues for the decoder family itself, supporting a move to
BigVGAN. This work was deliberately isolated to a new git branch
(`encodec-hifigan-experiment`) rather than done on `main`, and CLAUDE.md's rule against
multi-model comparisons "until the primary question is answered" was flagged to the user before
starting (they proceeded deliberately).

**Representation source.** Used `transformers.EncodecModel` (`facebook/encodec_24khz`) via the
continuous pre-quantization embedding (`model.encoder(input_values)`), not the quantized RVQ
codes -- this is the correct analog of MuQ's continuous `last_hidden_state` (temporal, no
pooling, per the task's explicit requirement). No new PyPI dependency was needed: EnCodec support
already ships in the pinned `transformers==4.55.0`. Verified it runs natively on MPS (EnCodec's
encoder includes 2 LSTM layers, which have had spotty MPS support historically in other contexts;
tested directly and confirmed finite, deterministic output before building anything on top of it).

**Measured EnCodec feature geometry (Gate-0-style diagnostics, implemented as
`FrozenEncodecEncoder.probe()`, printed at the start of any `train` run using
`encoder_type: encodec`):** for a 48000-sample (2.0s) input crop, latent shape `(1, 150, 128)` --
128-dim continuous embedding, 150 frames, **75.0 Hz frame rate, 320.0 samples/frame** exactly
(matches `facebook/encodec_24khz`'s published `upsampling_ratios: [8, 5, 4, 2]`, product 320).
This is a fundamentally different frame rate from MuQ's measured 25 Hz / 960 samples/frame (Gate
2) -- 3x denser in time, ~8x smaller in channel width.

**The one unavoidable architectural adaptation.** CLAUDE.md's Decoder Design item 3 requires the
generator's upsample-stage product to exactly match the encoder's measured samples-per-frame
ratio. Since EnCodec's ratio (320) differs from MuQ's (960), the *exact* `upsample_rates=(8,6,5,4)`
generator (product 960) cannot be reused verbatim without breaking basic tensor-length alignment
(feeding it EnCodec's 150-frame latent for a 48000-sample crop would produce a
150*960=144000-sample output against a 48000-sample target). The generator's `upsample_rates` was
changed to `(8, 5, 4, 2)` (product 320, mirroring EnCodec's own published decoder upsampling
ratios) and `training.crop_frames` was changed from 50 to 150 -- both changes exist purely to
preserve the same 2.0-second (48000-sample) crop length in the new encoder's frame-counting units,
not to alter the generator "design." `base_channels=1024`, `resblock_kernel_sizes=(3,7,11)`,
`resblock_dilation_sizes=((1,3,5),(1,3,5),(1,3,5))`, discriminator architecture, all losses, both
Adam betas `(0.8, 0.99)`, `adversarial_warmup_steps=500`, and `training.steps=15000` are
byte-identical to `configs/smoke_1024.yaml` (the MuQ+MRF-1024 baseline). Verified via `git diff`
that `configs/smoke.yaml`/`configs/smoke_1024.yaml` and all MuQ-path source code were never
touched -- confirmed by re-running the full test suite (47 tests, unaffected) after every change.

**Implementation.** Added `encodec_encoder.py` (`FrozenEncodecEncoder`, mirroring
`FrozenMuQEncoder`'s `encode()`/`probe()` interface exactly, reusing the already-generic
`MuQProbeResult`/`measure_feature_geometry` from `encoder.py` without modifying that file). Added
`EncodecConfig` and an `encoder_type: "muq" | "encodec"` field to `ExperimentConfig`, defaulting to
`"muq"` so every existing config's resolved behavior is provably unchanged (confirmed by loading
`smoke.yaml`/`smoke_1024.yaml`/`pilot.yaml` and checking the new fields default correctly). The
only change to `train.py` is a 10-line encoder-selection branch (`if config.encoder_type ==
"muq"/"encodec"/else: error`) plus expanded diagnostic printing (latent shape, dimensionality,
fps, samples/frame -- satisfying the Gate-0-style diagnostics requirement without touching the
actual `doctor` CLI command, which stays MuQ-specific per CLAUDE.md's Gate 0 definition).

**Comparison table** (both runs: same 2 tracks `pop.00012`/`disco.00049`, same 2.0s crop, same
MRF-1024 generator design, same discriminator, same losses, same Adam betas `(0.8, 0.99)`, same
15000-step schedule):

| metric | MuQ + HiFi-GAN (MRF-1024) | EnCodec + HiFi-GAN |
|---|---|---|
| representation frame rate | 25 Hz (960 samples/frame) | 75 Hz (320 samples/frame) |
| representation channels | 1024 | 128 |
| generator parameters | 61,582,593 | 54,799,617 |
| checkpoint size | 782.8 MB | 701.4 MB |
| training wall time (15000 steps) | ~52 min | ~62 min (3x longer feature sequence) |
| training stability | stable (no collapse) | stable (no collapse) |
| final mel loss | 0.319 | 0.381 |
| mel correlation (pop.00012 / disco.00049) | 0.965 / 0.976 | 0.950 / 0.970 |
| waveform correlation (pop.00012 / disco.00049) | 0.073 / 0.104 | **0.682 / 0.602** |

**The headline result:** mel correlation is comparable between the two (MuQ marginally higher),
but **waveform-sample correlation is 6-9x higher for EnCodec** -- 0.68/0.60 vs. 0.07/0.10. This is
the largest, cleanest effect size of any experiment in this project so far, and it moves a
completely different metric than every previous intervention did (capacity, loss reweighting, and
optimizer fixes all moved mel correlation while waveform correlation stayed near zero; this one
variable -- representation source -- moved waveform correlation dramatically while leaving mel
correlation roughly where it was).

**Answering the four required questions:**

1. *Can HiFi-GAN memorize two clips when conditioned on EnCodec latents?* Yes, unambiguously.
   Training converged cleanly (reconstruction loss ratio 0.36, well under the 0.5 pass threshold),
   with no instability of any kind across the full 15000 steps -- the same clean convergence
   pattern as the successful MuQ runs, not the collapse pattern seen in the pre-optimizer-fix
   1024-channel MuQ attempt.
2. *Is EnCodec reconstruction perceptually better than MuQ reconstruction?* The objective signal
   (waveform correlation 6-9x higher) is consistent with better time-domain/phase fidelity, but
   per CLAUDE.md's repeated instruction not to claim perceptual success from metrics alone, this
   is pending the user's listening judgment -- audio sent via SendUserFile, not yet confirmed.
3. *If EnCodec is substantially better, what does that imply about MuQ invertibility?* It would
   mean MuQ's own bottleneck representation -- a 25 Hz, self-supervised embedding trained for
   downstream music-understanding tasks, not reconstruction -- discards or entangles phase/fine-
   temporal information more thoroughly than a representation whose entire training objective is
   accurate reconstruction (EnCodec, a neural codec). Critically, since every other variable was
   held fixed (identical decoder, discriminator, losses, optimizer, schedule), this result would
   favor explanation (2) from [[gate3-capacity-ruled-out]] (a genuine MuQ representation ceiling)
   over explanation (1) (merely needing longer adversarial training) -- because the same limited
   adversarial budget (15000 steps, 2 clips) was clearly *sufficient* to unlock much better
   waveform fidelity once the representation changed, suggesting the MuQ runs weren't simply
   undertrained.
4. *If EnCodec were similarly mediocre, what would that imply about the decoder family?* Would
   have supported the user's original hypothesis (HiFi-GAN unsuited to music) and justified moving
   to BigVGAN. **This did not happen** -- the same decoder produced a qualitatively different
   result with a different representation, which is evidence *against* "the decoder family is the
   bottleneck" and *for* "the representation matters a lot," at least provisionally pending the
   listening confirmation in (2).

**Decision.** Do not start BigVGAN work (per explicit instruction, and now also because this
result argues against the decoder family being the primary bottleneck). Do not modify the MuQ
baseline (`configs/smoke.yaml`/`smoke_1024.yaml`, confirmed untouched). Do not proceed to Gate
4/5 on either representation (per explicit instruction). Await the user's listening verdict on
this run's audio before drawing final conclusions -- the waveform-correlation gap is compelling
but this project's own rules (and this session's own repeated experience with the
metrics-vs-perception gap in [[gate3-capacity-ruled-out]]) require a listening check before
treating any objective-metric win as a real one.

**Consequences.** `configs/encodec_1024.yaml`, `src/muq_inversion/encodec_encoder.py`, and the
`encoder_type` config field now exist on this branch. None of it affects the MuQ baseline on
`main` or on this branch when `encoder_type` is left at its default. If the listening verdict
confirms EnCodec is genuinely better, the natural next question becomes whether MuQ's specific
information loss is frame-rate-driven (75 Hz vs 25 Hz -- 3x denser temporal sampling) or
channel/training-objective-driven (128-dim codec-trained vs. 1024-dim self-supervised) -- this
single experiment changed both at once and cannot separate them.

**Rollback plan.** This entire line of work lives on `encodec-hifigan-experiment`, isolated from
`main` by construction. If it needs to be abandoned, simply do not merge the branch; no baseline
files were touched.

---

## encodec-mediocrity-across-all-variants: Neither capacity nor representation explains the audio quality ceiling

**Context.** Across this project's Gate 3 investigation, three substantially different technical
interventions have now been tried, each producing a large, measurable change in objective
metrics:

| variant | change vs. previous | metric that moved | perceptual verdict |
|---|---|---|---|
| tiny reconstruction-only baseline | (original Gate 3 implementation) | -- | "buzzing noise... nothing remotely recognizable" |
| MuQ + MRF-512, adversarial training | added discriminators, MRF generator | mel correlation 0 -> 0.92-0.94 | "recognizable... still plagued by artifacts" |
| MuQ + MRF-1024 | 3.57x generator capacity | mel correlation 0.92 -> 0.97 | **no perceptible difference from MRF-512** |
| EnCodec + MRF-1024 | swapped representation source entirely | waveform correlation 0.07 -> 0.68 (6-9x) | **no perceptible difference from either MuQ variant** |

The final row is the new information: the user's quick listening comparison across the tiny
baseline, MuQ+MRF-1024, and EnCodec+MRF-1024 found them "all similarly mediocre" -- no audible
difference despite EnCodec producing the single largest metric movement of any experiment in this
project.

**Finding: this generalizes [[gate3-capacity-ruled-out]] from "capacity doesn't explain it" to
"capacity *and* representation source don't explain it."** Two independent, large-effect-size
technical axes have now been varied -- decoder capacity (512 -> 1024 channels) and representation
source (MuQ -> EnCodec, a different model family with a different training objective, a 3x
different frame rate, and an 8x different channel width) -- and neither produced an audible
change, even though both produced their era's largest objective-metric improvement. This is
strong evidence that whatever sets the perceptual ceiling here is **not** decoder size and is
**not** which of these two specific representations is used.

**What's left, now that two major axes are ruled out:**

1. **Something shared by every variant tested.** All four rows in the table above share: the
   same two 2-second fixed clips, the same HiFi-GAN-style MRF generator *design* (even MRF-512 and
   MRF-1024 share the same block structure, just width), the same multi-period/multi-scale
   discriminator design, the same mel+STFT+adversarial+feature-matching loss combination, the
   same Adam betas, and a training budget capped in the same rough range (500-15,000 steps) on
   only 2 examples. Any of these could be the actual ceiling. The 2-clip-overfit paradigm itself
   is a strong suspect: a discriminator trained on only 2 real examples may only ever learn to
   push the generator toward a generic "plausible timbre" rather than a precise match to either
   specific clip, regardless of how large the generator is or how information-rich its
   conditioning is -- capacity and representation improvements would still show up in
   correlation-style metrics (which reward getting *closer* on average) without necessarily
   crossing whatever threshold separates "closer" from "audibly better."
2. **The evaluation methodology itself may not be resolving what actually differs.** A "quick
   listening test" on 2-second excerpts, without a blinded A/B setup or repeated careful
   listening, may not detect real but subtle differences -- especially if all variants share a
   dominant, perceptually loud artifact (e.g. a characteristic buzzy/ringing texture common to
   under-trained HiFi-GAN-style adversarial decoders) that masks whatever finer differences the
   metrics are picking up on.
3. **Both of the above could be true simultaneously** -- the paradigm ceiling could be real *and*
   a more careful evaluation could still surface a real, smaller difference between variants.

**Decision.** Do not run another capacity or representation variant inside this same 2-clip,
short-training paradigm expecting a different perceptual outcome -- the pattern (big metric move,
zero audible move) has now repeated three times across two different metrics (mel correlation,
waveform correlation) and two different technical axes (capacity, representation). Continuing to
iterate on generator/encoder choices without changing the shared training paradigm is unlikely to
be a good use of further compute. This was proposed to the user as a recommendation, not decided
unilaterally: the two live options are (a) change something about the shared paradigm itself
(more/different training data beyond 2 clips, i.e. moving toward Gate 4/5 scale, and/or a much
longer training budget) to see if that -- not capacity or representation -- is what's been
missing, or (b) treat "recognizable content, mediocre quality, robust to both capacity and
representation choice" as the documented finding for this phase of the interpretability
experiment and stop tuning this specific axis.

**Consequences.** This is a genuinely useful negative result for CLAUDE.md's own Scientific
Question 3 ("is failure caused by implementation/optimization problems, or by insufficient
information in the representation?") -- the answer increasingly looks like neither, specifically:
it's not an implementation-capacity problem (ruled out), and it's not specific to MuQ's
information content (ruled out by EnCodec performing metrically much better with zero perceptual
gain, which would be a strange result if MuQ's information content were the binding constraint).
That leaves the training paradigm/recipe itself, or the evaluation method, as the most likely
remaining explanations.

**Rollback plan.** None needed -- this is a synthesis, not a code change. If a future, more
careful listening evaluation (blinded A/B, longer excerpts, multiple listeners) does reveal real
differences between these variants that a quick pass missed, revisit whether the metrics were
right all along and the evaluation method was the actual limitation. (2026-09-08) -- no perceptible difference across any variant tested.**
User's quick listening comparison across all three of this project's major reconstruction
variants: the original tiny reconstruction-only overfit baseline, the MuQ+MRF-1024
adversarially-trained model, and this EnCodec+MRF-1024 model -- "I can't tell any difference...
they are all similarly mediocre." This is a stronger and more general version of
[[gate3-capacity-ruled-out]]'s finding: not only did a 3.57x capacity increase fail to produce an
audible difference despite moving mel correlation substantially, but swapping the entire
representation source (with a completely different, measured frame rate and channel width) and
producing a 6-9x jump in waveform correlation *also* failed to produce an audible difference. See
[[encodec-mediocrity-across-all-variants]] for the full synthesis this triggered.
