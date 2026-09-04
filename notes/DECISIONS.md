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
