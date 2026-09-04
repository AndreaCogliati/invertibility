# SCRATCHPAD

Append-only lab notebook. Unverified until summarized into MEMORY.md/DECISIONS.md.

## 2026-09-04 -- Initial scaffold, Gate 0 + Gate 1

- Started from an empty directory containing only `CLAUDE.md`.
- `git init`, `uv init --package --name muq_inversion .`, `uv python pin 3.12`.
- `uv add torch torchaudio librosa soundfile numpy scipy pyyaml tqdm tensorboard muq` then
  `uv add --dev pytest ruff mypy`.
- Confirmed no local GTZAN copy exists (`find ~ -iname '*gtzan*'` empty). Asked the user how to
  acquire it; they chose the Hugging Face route. Searched `huggingface.co/api/datasets?search=gtzan`,
  found `marsyas/gtzan` mirrors the canonical `opihi.cs.uvic.ca/sound/genres.tar.gz` archive
  (confirmed by reading its `gtzan.py` loading script, which also revealed the known corrupted
  file `jazz.00054.wav`).
- Read `OpenMuQ/MuQ-large-msd-iter`'s model card: install via `pip install muq`, use
  `from muq import MuQ; MuQ.from_pretrained(...)`. `config.json` showed `encoder_dim: 1024`,
  `label_rate: 25` (Hz) -- used only as a sanity-check prior, not hardcoded anywhere in code.

- First `doctor` run failed on MuQ forward pass:
  `AttributeError: 'EasyDict' object has no attribute '_attn_implementation'` inside
  transformers 5.16.1's `create_bidirectional_mask`. Traced to MuQ building its Wav2Vec2Conformer
  config as a plain `EasyDict`, not a real `PretrainedConfig`. Tried pinning
  `transformers==4.46.3` -> conflicted with `huggingface_hub>=1.30.0` already resolved.
  Tried `transformers==4.55.0` + `huggingface_hub>=0.34.0,<1.0` together -> resolved cleanly.
  Re-ran doctor: passed. Recorded as [[../notes/DECISIONS.md#muq-transformers-compat]].

- `doctor` (Gate 0) on `configs/smoke.yaml`, device=mps:
  - `muq.feature_shape: (1, 50, 1024)`, `frames_per_second: 25.0`, `samples_per_frame: 960.0`,
    `is_finite: True`, `deterministic_shape_repeat: True`.
  - `MiniGenerator` shape-only pass: input `(1, 50, 1024)` -> output `(1, 48000)`, matching
    `expected_output_samples`. PASSED.

- First `prepare-data` run failed: `sf.info()` itself raises `LibsndfileError: bad data offset`
  on `jazz.00054.wav` -- not just decode-time failure, so `index_gtzan` needed a try/except
  around the `sf.info` call, not just a post-hoc corrupted-file check. Fixed, made it also warn
  (not silently skip) if an *unexpected* file is unreadable.

- Second `prepare-data` run: indexed 1000/1000 files, found 14 duplicate-checksum pairs (e.g.
  `metal.00058.wav` == `rock.00016.wav` byte-for-byte despite different genre labels -- a known
  GTZAN defect, see Sturm 2013). Initial `assign_splits` hashed on
  `f"{genre}/{track_id}"`, which does NOT guarantee cross-genre duplicates land in the same
  split (confirmed by inspection: a duplicate pair spanning two genres would hash differently).
  Fixed by hashing purely on `checksum`. Verified post-fix with a standalone check: zero
  checksums span multiple splits.
  Deleted the manifest from the pre-fix run before re-running, since CLAUDE.md rule 8 (never
  overwrite a previous run's manifest) applies to *committed* prior runs, not this session's own
  just-created, not-yet-committed test output.

- `ruff check src/` flagged blind `except Exception` (BLE001) in `data.py` and
  `logging_utils.py`, plus an unnecessary `.encode("utf-8")`. Narrowed the except clauses to
  specific exception types (`sf.LibsndfileError`/`RuntimeError`/`OSError`;
  `subprocess.CalledProcessError`/`TimeoutExpired`/`FileNotFoundError`;
  `importlib.metadata.PackageNotFoundError`). Noticed `_package_version` was dead code -- wired
  it into `capture_environment_manifest` so package versions (torch, torchaudio, muq,
  transformers, librosa, soundfile, huggingface_hub) actually get recorded, per CLAUDE.md rule 2.
  `ruff check src/` now clean.

- 13 unit tests added (`tests/test_audio.py`, `tests/test_data.py`) covering resample length
  tolerance, waveform validation rejections, deterministic/duplicate-safe splitting, and
  manifest round-trip. All pass (`uv run pytest -q`).

- Did not commit to git yet -- scope was explicitly Gate 0 + Gate 1 only; leaving the commit
  decision to the user per instructions to only commit when asked.
- User asked to commit; committed as `95da3c5` on `main`.

## 2026-09-04 (cont.) -- Gate 2 (Feature Contract)

- Added `cache.py`: content-addressed key = sha256(audio_checksum | muq_model_id |
  sorted(preprocessing.items()) | representation_name | dtype); entries sharded into
  `data/cache/<key[:2]>/<key>.pt`; atomic write via `tempfile.mkstemp` in the same directory +
  `os.replace`. Considered `.npy`+`.json` two-file layout but rejected it (crash between the two
  renames leaves an inconsistent pair) -- see [[DECISIONS.md#gate2-cache-format]].
- Refactored `encoder.py`: split `FrozenMuQEncoder.probe()` into `encode()` (forward pass) +
  a new module-level `measure_feature_geometry(features, num_samples, sample_rate)` (pure stats),
  so `extract` can get both the geometry stats *and* the raw feature tensor from a single forward
  pass instead of encoding twice.
- Made `data.py`'s `_deterministic_unit_interval` public (`deterministic_unit_interval`) so
  `cli.py`'s `extract` command can reuse the same deterministic-sample-selection logic used for
  the Gate 1 resample-verification sample, picking the same kind of stable subset of tracks to
  probe.
- Added `cache`, `feature_contract` sections to `config.py` and both YAML configs:
  `cache.dtype` (float32 default), `feature_contract.probe_durations_seconds` (1/2/5/10s + full),
  `probe_track_count` (3), `frame_rate_tolerance_hz` (0.5), `alignment_tolerance_frames` (1),
  `lowprecision_dtype` (float16), `lowprecision_max_abs_error` (0.05, chosen as a round,
  conservative bound -- not derived from any measurement in advance).
- `extract` command: for `probe_track_count` deterministically-chosen real GTZAN tracks, crop to
  each configured duration (skipping durations longer than the clip; always including the
  clip's true full length), run MuQ, check finiteness, then aggregate: frame-rate consistency
  across all probes (min/max fps spread vs. tolerance), a single shared `feature_channels` value,
  and per-probe alignment (`num_frames` vs `round(num_samples / mean_samples_per_frame)`).
  Then, for every probe, cache the feature tensor at both `cache.dtype` and
  `feature_contract.lowprecision_dtype`, reload, and assert bit-exact equality against the tensor
  *as saved* (not the original float32, since the float16 cast is an intentional lossy step
  measured separately via max-abs-error vs. the float32 tensor).
- First `extract` run on `configs/smoke.yaml`: all 15 probes (3 tracks x 5 durations) finite;
  `frames_per_second` in `[24.9998, 25.0000]` (well under the 0.5 Hz tolerance);
  `mean_samples_per_frame = 960.0013`; alignment passed for all 15; 30 cache round-trips exact;
  float16 max abs error 0.0074 (under the 0.05 tolerance). Noted: full-length clips give
  `samples_per_frame = 960.0067`, not exactly 960 -- confirms the ratio isn't perfectly integral
  for real (non-hand-picked-length) audio, recorded as a Gate 3 concern in
  [[DECISIONS.md#decoder-upsample-placeholder]] (see the "Update (Gate 2, ...)" addendum there).
- Second `extract` run (same config, cache already populated): completed identically with no
  errors -- confirms idempotency/resumability of the content-addressed cache writes.
- Added `tests/test_cache.py` (5 tests: key determinism, key sensitivity to every input field,
  save/load round trip, missing-key returns None, no leftover tmp file after a successful save).
  `ruff check .` clean, `uv run pytest -q` -> 18 passed.
- Registered `extract` in `cli.py`'s argparse subcommands and `main()` dispatch.
- User asked to commit Gate 2 work; committed as `78692c0` on `main`.

## 2026-09-04 (cont.) -- Gate 3 (Tiny-Set Overfit)

- Tested `torch.stft` and `torchaudio.transforms.MelSpectrogram` directly on `mps` before writing
  any loss code -- both ran without error on a synthetic batch, so no CPU fallback needed for the
  loss computation (would have needed an explicit, non-silent fallback path otherwise per
  CLAUDE.md rule 10).
- `models/losses.py`: `MelSpectrogramLoss` (log-mel L1) and `MultiResolutionSTFTLoss`
  (spectral-convergence + log-magnitude L1, averaged over 3 resolutions). Configs chosen
  explicitly for 24kHz music, not copied from speech HiFi-GAN -- see
  [[DECISIONS.md#gate3-loss-configs]].
- `checkpoint.py`: `build_checkpoint_state` assembles the full CLAUDE.md-required field list
  (weights, optimizer state, step, RNG states incl. `torch.mps.get_rng_state()` when MPS is
  available, resolved config, manifest hash, `uv.lock` hash, environment manifest, best metric).
  `save_checkpoint_atomic` writes via temp-file + `os.replace` (same pattern as `cache.py`), then
  immediately reloads and checks required keys are present -- deletes and raises if not. Full
  resume-equivalence testing is explicitly deferred to Gate 4 (CLAUDE.md's own gate ordering
  already separates "train on tiny set" from "validate checkpoint/resume equivalence").
- `train.py`: `select_overfit_tracks` reuses the same `deterministic_unit_interval` hash used for
  Gate 1/2's deterministic sampling, restricted to `split == "train"` and `not corrupted`, so the
  same 1-4 clips are chosen every run unless `training.overfit_track_ids` is set explicitly.
  Measured `feature_channels` from a real MuQ forward pass before constructing `MiniGenerator`
  (never hard-coded 1024) -- caught this mid-write: first draft had a
  `hasattr(config.muq, "feature_channels")` fallback to a hard-coded 1024, which would have
  silently assumed the channel count instead of measuring it; replaced with an actual probe call.
  Training crops are `crop_frames * generator.total_upsample` samples (see
  [[DECISIONS.md#gate3-crop-alignment]]) so the generator's output length always exactly equals
  the cropped target waveform -- no padding/trimming logic needed at this stage.
- Encoder features are computed via `encoder.encode(crop).clone()` -- `encode()` runs under its
  own `@torch.inference_mode()`; cloning *outside* that context (after the `with` block/decorator
  scope exits) converts the inference tensor into a normal, autograd-eligible leaf tensor for the
  generator's input, per PyTorch's documented workaround, without ever requiring gradients
  through the frozen MuQ encoder.
- First `train` run on `configs/smoke.yaml`: 2 clips (`pop.00012`, `disco.00049`), crop_frames=50
  (48000 samples/2.0s), 500 steps, Adam lr=2e-4, alternating clips per step. Loss went
  10.78 -> ~2.8-3.2 (mean first-10 7.51 -> mean last-10 3.18, ratio 0.42 vs. 0.5 PASS threshold).
  Checkpoints written to `runs/gate3_tiny_overfit/checkpoints/` (`step_000250.pt`, `step_000500.pt`,
  `latest.pt`, ~31MB each); reloaded `latest.pt` standalone afterward and confirmed all required
  keys present, `step == 500`, hashes recorded. Reference/reconstruction/difference WAVs written
  to `artifacts/gate3_tiny_overfit/`; spot-checked with soundfile -- all finite, reconstruction
  peak ~0.53 vs. reference peak ~1.0 (expected for a short, tiny-model, non-adversarial overfit
  run -- amplitude is not yet the point, per-clip loss decrease is).
- Added `tests/test_losses.py` (zero-loss-for-identical-signal + positive-for-different-signal for
  both losses, plus a mismatched-lengths rejection test), `tests/test_checkpoint.py` (round trip,
  no leftover tmp file, rejects a checkpoint missing required keys), `tests/test_train.py`
  (deterministic/explicit track selection, corrupted/non-train exclusion, crop length/rejection).
  `ruff check .` clean; `uv run pytest -q` -> 33 passed.
- Registered `train` in `cli.py`'s argparse subcommands and `main()` dispatch. Also fixed a
  pre-existing loose `object` type annotation on `cmd_extract`'s `probes` list (flagged by the
  IDE's mypy diagnostics while editing nearby code) to the real `MuQProbeResult` type.
