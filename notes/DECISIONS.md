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
