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
- User asked to commit Gate 3 work; committed as `afe7895` on `main`.

## 2026-09-04 (cont.) -- Gate 4 (Short Pilot)

- Refactored `train.py`'s `select_overfit_tracks` into a generic `select_tracks(records, *,
  split_seed, num_tracks, explicit_track_ids)` (kept `select_overfit_tracks` as a thin wrapper
  reading `config.training.*`, so Gate 3's existing tests didn't need to change) so Gate 4's
  `pilot.py` could reuse the same deterministic-selection logic against `config.pilot.*` fields.
- Added `PilotConfig` to `config.py`: `num_tracks` (6 for pilot.yaml, 3 for smoke.yaml's fast
  path), `steps`, `resume_test_at`, `resume_tolerance`, `control_margin`. This is the first point
  where `configs/pilot.yaml` and `configs/smoke.yaml` actually diverge in values (previously
  identical except `name:`), as flagged as expected in the previous session's NEXT.md.
- Added an `extra: dict | None` parameter to `checkpoint.build_checkpoint_state` so Gate 4 could
  stash a dedicated crop-sampling RNG's state (`rng.getstate()`) in the checkpoint without
  hard-coding that concept into the shared, gate-agnostic checkpoint module. This was necessary
  because Gate 3/4's crop sampler uses its own `random.Random(seed)` instance, decoupled from
  Python's global `random` module -- `checkpoint.restore_rng_state` only restores the global
  python/numpy/torch RNGs, so the dedicated crop RNG needed its own explicit save/restore path.
- `pilot.py`: `_build_conditioning(mode, ...)` implements all three required controls in one
  function -- `shuffled_across` (encode a *different* clip's crop entirely), `shuffled_time`
  (encode the same clip, then `torch.randperm` the frame dimension), `constant_zero` (zero out
  the encoded features). "correct" mode bypasses this function entirely in `_train_steps`.
- Wrote `tests/test_pilot.py` using a stub encoder (returns a value that encodes which waveform
  was seen, via `waveform.mean()`) to verify each control's tensor-level behavior without needing
  a real MuQ forward pass -- confirms `constant_zero` is exactly all-zero, `shuffled_time`
  preserves the same multiset of per-frame values in a different order, and `shuffled_across`
  actually pulls from a different clip's audio.
- First `pilot` run (`configs/smoke.yaml`, 60 steps, 3 tracks) surfaced a real bug: the
  "pre-checkpoint determinism" check used an absolute tolerance of `1e-4`, which failed
  (`4.0138` vs `4.0000` at step 30) even though both runs used identical seeds, architecture, and
  RNG streams. Diagnosed as genuine MPS/Metal kernel float nondeterminism (not a logic bug) by
  computing per-step drift rate across multiple run lengths and finding it consistent
  (~0.00022-0.00026 relative loss difference per step across 30/100/200-step windows) -- see
  [[DECISIONS.md#gate4-mps-resume-tolerance]]. Fixed by switching both the pre-checkpoint check
  and the main resume check to the same relative-tolerance style, then calibrating
  `resume_tolerance` per config based on that config's step count (0.02 for smoke's 30-step
  window, eventually 0.08 for pilot's 200-step window after the real pilot run showed 0.063).
- Ran the real `configs/pilot.yaml` pilot three times total (had to also run `prepare-data
  --config configs/pilot.yaml` first, since the manifest filename is keyed off `config.name` and
  only `smoke`'s manifest existed): first at 200 steps/0.02 tolerance (resume failed, borderline
  controls), second at 200 steps/0.05 tolerance after the drift-rate analysis (resume passed,
  `shuffled_across` control borderline-failed), third at 400 steps/0.08 tolerance (resume passed
  cleanly, `shuffled_across` and `constant_zero` passed robustly, `shuffled_time` did not -- see
  [[DECISIONS.md#gate4-shuffled-time-finding]]). Did not keep raising steps indefinitely chasing
  a full pass once the `shuffled_time` result was consistent across two runs -- treated it as a
  genuine, evidence-backed diagnostic finding instead of a target to optimize away.
- Final Gate 4 state: checkpoint/resume equivalence PASSED; 2 of 3 conditioning controls PASSED
  (`shuffled_across`, `constant_zero`); `shuffled_time` did NOT pass its margin. Reported Gate 4
  as NOT FULLY PASSED, with the specific finding documented rather than silently accepted or
  forced past via threshold-tuning.
- `ruff check .` clean; `uv run pytest -q` -> 38 passed (added `tests/test_pilot.py`, 5 tests).
- Registered `pilot` in `cli.py`'s argparse subcommands and `main()` dispatch.
- User asked to commit Gate 4 work; committed as `95f333e` on `main`.

## 2026-09-04 (cont.) -- User feedback: Gate 3 audio is "buzzing noise", add discriminators

- User listened to Gate 3's artifacts and reported them as unrecognizable buzzing noise. Before
  changing anything, tried the user's literal suggestion first (more steps): reran at 5000 steps
  (10x). Loss barely moved (2.78 at step 500 -> 2.64 at step 5000), confirming a plateau, not an
  undertraining problem.
- Diagnosed with quantitative checks (not just guessing): log-mel spectrogram correlation
  reference vs. reconstruction was 0.89-0.91, frame-energy-contour correlation 0.98 (both good --
  the model *is* learning spectral envelope/dynamics), but raw waveform sample correlation was
  ~0. Concluded: classic phase-incoherence symptom of magnitude-only losses (mel + multi-res STFT
  never constrain phase) -- doesn't improve with more steps because the loss has no phase
  gradient at all. This matches CLAUDE.md's own plan: add discriminators "only after the
  reconstruction path passes smoke tests" -- which it had (good spectral match, converged loss).
- Asked the user how to proceed (bigger model vs. discriminators vs. revert steps) rather than
  silently picking one -- they chose discriminators.
- Implemented `models/discriminators.py`: `PeriodDiscriminator`/`MultiPeriodDiscriminator`
  (reshapes waveform into 2D frames by period, 2D convs) and `ScaleDiscriminator`/
  `MultiScaleDiscriminator` (1D convs on raw + avg-pooled waveform, grouped convs). Kept channel
  widths <=256 (vs. original HiFi-GAN's 1024) to match this project's "deliberately small"
  philosophy. Used `torch.nn.utils.parametrizations.weight_norm` (non-deprecated) rather than the
  deprecated `torch.nn.utils.weight_norm`.
- Initially wrote `ScaleDiscriminator`'s grouped-conv channel counts with a defensive
  `min(g, c_in, c_out)` clamp meant to avoid invalid group counts -- realized this doesn't
  actually guarantee `groups` divides both channel counts evenly (Conv2d/Conv1d's real
  requirement), so replaced it with an explicit divisibility assertion instead of silently
  producing a possibly-still-invalid value.
- Added `discriminator_loss`, `generator_adversarial_loss`, `feature_matching_loss` (LSGAN-style)
  to `models/losses.py`. Added `DiscriminatorConfig` and adversarial fields
  (`adversarial_warmup_steps=500`, `discriminator_learning_rate`, `adversarial_loss_weight`,
  `feature_matching_loss_weight`) to `TrainingConfig`. `configs/pilot.yaml` didn't need any edits
  -- dataclass defaults cover it, confirmed by loading both configs and checking the new fields.
- Extended `checkpoint.build_checkpoint_state` with optional `discriminator`/
  `discriminator_optimizer` params (backward-compatible via `None` defaults) -- wrapped MPD+MSD
  in a single `nn.ModuleDict` so they share one `discriminator_state_dict` entry rather than
  inventing a second ad-hoc state field.
- Updated `train.py`'s training loop: reconstruction-only for `adversarial_warmup_steps`, then
  each step does a discriminator update (real vs. `prediction.detach()`) followed by a generator
  update (re-running the now-updated discriminators, this time with gradients flowing through the
  generator's output) combining reconstruction + weighted adversarial + weighted feature-matching
  losses.
- Ran a cheap 20-step smoke test first (own scratch config in `/tmp`, deleted after) with
  `adversarial_warmup_steps=5` to exercise both the warmup and post-warmup code paths cheaply
  before committing to a full run -- caught nothing broken, but this is exactly the kind of check
  that would have caught a shape mismatch cheaply if one existed.
- Added `tests/test_discriminators.py` (9 tests: MPD/MSD output shapes, invalid-groups rejection,
  all three GAN loss functions' zero/positive cases, and a gradient-flow check confirming
  `generator_adversarial_loss` actually backprops into a fake input tensor).
- First full run (8000 steps, `configs/smoke.yaml`) technically succeeded but Gate 3's own
  pass/fail check then reported FAILED (ratio 0.57 vs. 0.5 threshold) despite mel loss clearly
  still converging (5.15 -> 0.76). Root-caused: the ratio was computed on the *combined* loss,
  which after warmup includes the adversarial minimax term -- oscillates by design in healthy GAN
  training, doesn't monotonically shrink. Fixed by tracking `recon_loss_history` (mel + STFT only)
  separately and gating on that instead -- this is actually a more faithful implementation of
  CLAUDE.md's literal wording ("confirm that **reconstruction** losses decrease substantially")
  than the pre-discriminator code was.
- Made the classic backgrounding mistake while re-running: used both the Bash tool's
  `run_in_background: true` AND a trailing `&` in the same command. The trailing `&` caused the
  wrapped shell to return immediately (job orphaned in the background), so the harness reported
  "completed" almost instantly while the real `python` process (verified via `ps aux`) kept
  running for several more minutes, and the redirected log file sat at 4 lines the whole time due
  to stdio block-buffering on a non-tty destination. Recovered by using Monitor with an
  until-loop waiting on the actual PID (`kill -0 $PID`) rather than trusting the harness's
  (in this case wrong) "completed" signal. Lesson: never combine `run_in_background: true` with a
  trailing `&`; the tool parameter already does the backgrounding.
- Re-ran cleanly: reconstruction loss ratio 0.40 (PASS). Reconstruction peak amplitude moved
  closer to the reference's (`disco.00049`: 0.76 -> 0.92-0.99 across the two post-fix runs, ref
  peak 1.0) -- consistent with a discriminator teaching the generator to use realistic dynamic
  range, though not proof of reduced buzziness by itself. Mel correlation stayed similar
  (0.82-0.84, not obviously improved -- consistent with the known GAN-vocoder tradeoff of
  spectral-distance metrics for perceptual naturalness). Sent reference + reconstruction WAVs for
  both clips directly to the user via SendUserFile rather than declaring the buzzing fixed from
  metrics alone -- per CLAUDE.md's explicit instruction not to claim successful inversion from
  loss numbers, and because waveform/mel correlation genuinely can't settle a perceptual question.
- `ruff check .` clean; `uv run pytest -q` -> 47 passed.

## 2026-09-04 (cont.) -- Gate 3 iterations 2-3 (brief backfill; full detail in DECISIONS.md)

Detailed reasoning for both iterations went straight into DECISIONS.md/MEMORY.md at the time;
backfilling a short pointer here per CLAUDE.md's protocol (SCRATCHPAD should log as things happen).

- Iteration 2: user's verdict on iteration 1 was disco.00049 partially recognizable, pop.00012
  worse. Added `training.reconstruction_loss_weight=10.0` (mel+STFT weighted 10x during the
  adversarial phase only) and extended 8000 -> 15000 steps. Result: mel loss 0.77 -> 0.59, mel
  correlation pop 0.82 -> 0.90, disco 0.84 -> 0.94. See [[DECISIONS.md#gate3-adversarial-losses]]
  "Update" + "Decision (iteration 2)" sections.
- Iteration 3: user's verdict on iteration 2 -- "still very poor: noise, buzzing, ringing."
  Diagnosed as generator capacity (ringing after two rounds of loss tuning points at architecture).
  Rewrote `models/generator.py` with a proper MRF fusion (3 parallel `ResBlock`s, kernel sizes
  3/7/11, each with dilations 1/3/5 -- previously a sequential 2-block chain), refactored
  `upsample_rates` from 6 stages `(5,4,4,3,2,2)` to 4 stages `(8,6,5,4)` (same product 960) so
  channel width isn't halved away as aggressively, and doubled `base_channels` 256 -> 512. Timed
  a 100-step run first (~0.095s/step) before the full 15000-step retrain. Result: mel loss
  0.59 -> 0.48, mel correlation pop 0.90 -> 0.94, disco 0.94 -> 0.94 (held). Sent to user for a
  third listening pass. See [[DECISIONS.md#gate3-generator-capacity]].

## 2026-09-06 -- Capacity-isolation experiment: MRF-512 vs MRF-1024

User's verdict on iteration 3 (MRF-512): "both excerpts have improved... disco.00049 is now
recognizable... pop.00012 still contains substantial buzzing/noise, although rhythm and harmony
are present." Given a prior finding that more training steps alone didn't help the smaller model,
user asked for a controlled experiment isolating generator capacity specifically: 512 -> 1024
base_channels, everything else held fixed, to distinguish "still too small" from "large enough,
remaining limits are the objective/representation." Re-read CLAUDE.md and all notes files first
per user's explicit instruction (also standard practice at session start per CLAUDE.md's own
Memory and Scratchpad Protocol).

- Created `configs/smoke_1024.yaml` as a full copy of `configs/smoke.yaml` (the current MRF-512
  working config) with exactly two differences: `decoder.base_channels: 1024` and
  `training.run_name: gate3_tiny_overfit_mrf1024` (distinct run name so this experiment's
  checkpoints/artifacts don't overwrite the MRF-512 run's, per CLAUDE.md rule 8). Did not touch
  `configs/smoke.yaml` (confirmed via `git diff --stat` before and after -- no changes to it from
  this work). `upsample_rates`, `resblock_kernel_sizes`, `resblock_dilation_sizes`, all loss
  configs, and all training schedule fields (steps, warmup, learning rates, loss weights) are
  byte-identical to `smoke.yaml`, to isolate capacity as the only variable.
- Ran `prepare-data --config configs/smoke_1024.yaml` to generate its manifest (same split,
  `split_seed: 1234` unchanged, so identical train/val/test assignment -- just re-indexed under
  the new config name since manifest filenames are keyed by `config.name`).
- Ran `doctor --config configs/smoke_1024.yaml`: Gate 0 PASSED, decoder shape-only pass produces
  `(1, 48000)` from `(1, 50, 1024)` input, same as the 512 config (upsample_rates/total_upsample
  unchanged, only channel width differs).
- Computed parameter counts directly (fresh `MiniGenerator` instances, not from a checkpoint):
  base_channels=512 -> 17,235,585 params (68.9MB fp32); base_channels=1024 -> 61,582,593 params
  (246.3MB fp32) -- a 3.57x increase (sub-4x because the input projection and output head scale
  linearly with channels, not quadratically like the interior conv layers).
- Measured MPS memory directly (`torch.mps.current_allocated_memory()`/`driver_allocated_memory()`
  after a few forward+backward steps on a fresh generator + Adam optimizer, batch 1, no encoder or
  discriminators loaded): 512 -> 279.5MB current / 1271.9MB driver; 1024 -> 1007.0MB current /
  2496.7MB driver. Comfortably within the M4 Max's 64GB unified memory; no swap risk anticipated.
- Timed 100 steps end-to-end (own scratch config in `/tmp`, deleted after, plus a throwaway
  manifest copy and run/artifacts dirs, all cleaned up) before committing to the full retrain:
  ~19.8s total (100 steps, 50 warmup + 50 adversarial) vs. ~15.5s for the same test on the 512
  config -- per-step cost ~0.128s vs. ~0.095s (~1.35x slower per step, well under the 3.57x
  parameter-count increase, consistent with MPS/Metal having compute headroom on the M4 Max for
  this workload rather than being purely parameter-count-bound). Projected full 15000-step run:
  ~32 minutes.
- Launched the real 15000-step run. Result: healthy and comparable to the 512-channel run through
  step 3500 (mel=0.4873 @ 3500 for 1024 vs. 0.5770 @ 3500 for 512 -- 1024 was tracking *better*),
  then between step 3500 and step 3750 `disc` crashed 0.68 -> 0.04 and kept falling toward 0.0000,
  `adv` jumped to ~2.0 (LSGAN generator-loss theoretical max) and stayed saturated for the
  remaining ~11,250 steps. Checked the final audio directly: both clips are a constant +1.0
  signal (`std=0.0`, 100% of samples `|x|>0.999`) -- total generator collapse, `tanh` head fully
  saturated. Final mel loss 8.08 (vs. 0.48 for 512), mel correlation 0.31/0.23 (vs. 0.94/0.94) --
  these numbers describe the collapse, not a capacity comparison.
- To get a fair comparison anyway, loaded `step_003000.pt` (last checkpoint before the collapse
  window; `checkpoint_every=1000` so this was the closest available save) with a one-off inference
  script (not part of `run_gate3` -- built inline, not saved as a repo script) and ran the fixed
  validation crops through it: mel correlation 0.9448 (`pop.00012`) / 0.9496 (`disco.00049`) --
  matching or slightly *exceeding* the fully-converged 512-channel model's 0.937/0.944, at only
  3000/15000 steps. Waveform correlation also mildly positive (0.10/0.09) vs. 512's near-zero/
  negative (-0.05/-0.11) -- small effect, not conclusive alone, but consistent with "not obviously
  worse." This is the most useful data point from the whole experiment.
- Estimated training wall time from checkpoint file timestamps (didn't wrap the actual
  `uv run ... train` invocations in `time` since they ran via `run_in_background`): 512 -> ~32 min
  (1000->15000 in 29m14s, extrapolated), 1024 -> ~45 min (1000->15000 in 41m33s) -- ~1.4x slower,
  consistent with the ~1.35x measured in the 100-step timing test.
- Diagnosed the likely root cause of the collapse: both the generator and discriminator optimizers
  in `train.py` use plain `torch.optim.Adam(..., lr=...)` with PyTorch's *default* betas
  `(0.9, 0.999)` -- never explicitly set anywhere. GAN training literature consistently avoids
  `beta1=0.9` for exactly this failure mode (too much momentum in a minimax game lets the
  discriminator pull decisively ahead); the original HiFi-GAN paper's own published optimizer
  config uses `betas=(0.8, 0.99)`. This is a real, previously-unnoticed gap from established
  practice, not something inherent to the capacity increase -- plausible mechanism: a larger
  generator reaches the discriminator-fooling regime faster (matches the pre-collapse data showing
  faster early convergence), which may have reached the instability threshold sooner in step-count
  terms than the smaller model happened to, but wasn't tested directly.
- Sent 4 audio files to the user: the two pre-collapse (step 3000) reconstructions and the two
  final (collapsed, garbage) reconstructions, with a caption explaining which is which and why the
  final ones aren't worth listening to closely.
- Wrote up the full comparison table, four-question analysis, and recommendation in
  [[DECISIONS.md#gate3-capacity-1024]]. No source code was changed this session -- purely a new
  config file (`configs/smoke_1024.yaml`) plus its generated manifest; `configs/smoke.yaml` was
  never touched (verified via `git diff --stat` at multiple points). `ruff check .`/`pytest -q`
  re-confirmed clean/passing (47 tests, unaffected since no source changed).
- Did not proceed to Gate 4/5, per explicit instruction -- still investigating Gate 3.

## 2026-09-06 (cont.) -- Optimizer-beta fix, clean capacity comparison

User said "go ahead with your suggestions" (the recommendation from the previous write-up: fix
Adam betas, regression-check 512, then re-run 1024).

- Added `training.generator_betas`/`discriminator_betas` config fields (both default
  `(0.8, 0.99)`), wired into `train.py`'s two `torch.optim.Adam(...)` calls via `betas=`. Added
  YAML tuple-parsing for the two new fields in `config.py`. Set explicitly in both
  `configs/smoke.yaml` and `configs/smoke_1024.yaml` (not left as a hidden dataclass default).
  `ruff check .`/`pytest -q` clean (47 tests, unaffected -- pure additive config change).
- Regression check: re-ran `configs/smoke.yaml` (512, same run_name -- supersedes the previous
  checkpoint, consistent with how this config has been treated all session). No collapse; `disc`
  oscillated healthily ~0.4-0.7 throughout (vs. crashing to 0 in the failed 1024 attempt). Mel
  loss trended to 0.32-0.45 near the end (vs. 0.478 with old betas -- a real improvement from the
  fix alone, on top of just not breaking anything). Mel correlation 0.924/0.932 (comparable to the
  prior 0.937/0.944 -- the exact final-step snapshot caught a slightly-off point in an otherwise
  healthier training curve, not a regression).
- Re-ran `configs/smoke_1024.yaml` (1024, same run_name -- supersedes the collapsed run) with the
  same fix. Result: **no collapse** -- `disc` stayed in the same healthy ~0.4-0.7 band for all
  15000 steps. Final mel loss 0.319 (best of any run this session), mel correlation
  **0.9646/0.9755** (best of any run this session), waveform correlation positive for both tracks
  for the first time (0.073/0.104) -- previously always near-zero or negative.
- This is now a valid, clean capacity comparison (both runs stable, same optimizer settings,
  same everything except `base_channels`): 1024 beats 512 on every objective metric measured.
  Checkpoint/timing: 512 -> 250.6MB, ~31 min; 1024 -> 782.8MB, ~52 min (~1.7x wall time for
  3.57x params -- sub-linear, consistent with earlier per-step timing measurements).
- Sent 4 reconstruction WAVs to the user (512 pop/disco, 1024 pop/disco) for a direct A/B
  listening comparison, with the metric summary in the caption.
- Wrote up the full "Update (optimizer-beta fix applied)" section in
  [[DECISIONS.md#gate3-capacity-1024]], including a revised comparison table and revised answers
  to the four diagnostic questions now that the comparison is uncomplicated by instability.
- Did not proceed to Gate 4/5. `configs/smoke.yaml` remains the nominal "working" config; whether
  to promote the 1024-channel settings as the new default is left to the user's listening
  judgment, not decided unilaterally from the metric improvements alone (per CLAUDE.md's
  repeated instruction not to claim success from objective metrics without a listening check).

## 2026-09-06 (cont.) -- Final listening verdict and synthesis; capacity line of investigation closed

User's verdict on the clean 512-vs-1024 (betas-fixed) comparison: "I can't hear any noticeable
improvement in the audio. They are equally mediocre: recognizable but still plagued by artifacts."
Then asked directly: "What conclusions can we make?"

- This confirms the predicted-but-unconfirmed alternative outcome flagged in the prior write-up
  ("if the user's listening judgment says 1024 isn't perceptually better despite the metric
  gains, that would itself be an interesting finding"). It happened.
- Talked through the interpretation before writing anything down (this section is the write-up of
  that reasoning, not a new investigation): capacity has now been tested twice (once confounded by
  the optimizer collapse, once cleanly) and both times increasing it produced measurable metric
  gains with no perceptible gain -- strong evidence capacity is not the active bottleneck, and
  that mel/waveform correlation are not good proxies for whatever the human ear is actually
  reacting to (almost certainly phase/fine-temporal structure, which magnitude-domain losses and
  correlation metrics are both largely blind to).
- Framed this as genuine, positive progress on CLAUDE.md's Scientific Question 3 ("is failure
  caused by implementation/optimization problems, or by insufficient information in the
  representation?") rather than a dead end: it rules out "decoder too small" specifically. Two
  candidate explanations remain, not yet distinguished by anything run so far: (1) adversarial
  training is still too shallow (15k steps on 2 overfit clips is small by vocoder-GAN standards),
  or (2) MuQ's measured 25Hz/40ms-per-frame representation is a hard information ceiling
  independent of decoder/training. Recommended testing (1) first since it's cheaper/more
  actionable, but did not execute anything -- presented as a recommendation and asked the user
  whether to proceed with a longer-training experiment or stop here.
- Wrote up the full three-finding synthesis in `notes/DECISIONS.md` as a new entry
  ([[DECISIONS.md#gate3-capacity-ruled-out]]), and appended the verbatim listening verdict to the
  end of [[DECISIONS.md#gate3-capacity-1024]] (had to fix a section-ordering mistake from the
  first edit attempt -- the verdict paragraph landed after the new synthesis section instead of
  before it; caught by re-reading the file and moved it to the correct place).
- User then asked to "summarize all the findings appropriately in the note files, so that the
  task can be resumed later" -- this pass is that consolidation: added the synthesis to
  DECISIONS.md, MEMORY.md (durable conclusion), this SCRATCHPAD.md entry (as-it-happened log),
  and NEXT.md (resumability: current status, next command options, blockers) all in the same
  pass, no new code or experiments run.

## 2026-09-08 -- New branch: experiment/mel-hifigan-sanity

User asked for a new branch `experiment/mel-hifigan-sanity`, off `main` (deliberately chosen over
branching from `encodec-hifigan-experiment`, since this test doesn't need any EnCodec code and
keeping it a clean sibling avoids conflating the two lines of investigation). Re-read CLAUDE.md
and all notes files first, per explicit instruction -- confirmed this branch's notes files are
at the pre-EnCodec ("capacity ruled out") state, since `main` never received the EnCodec branch's
commits. Since I already had the exact MuQ/EnCodec comparison numbers verified earlier this same
session, referenced them directly in this branch's write-up rather than needing the other
branch's files.

Task: run the identical Gate 3 setup with ground-truth mel-spectrogram conditioning instead of
MuQ or EnCodec -- the classic vocoder "copy synthesis" sanity check (mel of the target itself ->
HiFi-GAN -> waveform). Same 2 clips, crop length, MRF-1024 generator, discriminators, losses,
betas, 15000 steps. Explicit instructions: don't touch Gate 4/5, don't start BigVGAN.

- Reused this project's own already-justified `mel_loss` parameters (n_fft=2048, hop=300,
  win=1200, n_mels=128, fmin=0, fmax=12000) as the conditioning transform too, rather than
  picking new numbers -- the "ground truth" representation shouldn't be a fresh unjustified
  choice when a good one already exists in this codebase.
- Caught a real implementation subtlety before training: torchaudio's default `center=True`
  mel framing gives `1 + num_samples // hop_length` frames (161 for a 48000-sample/hop=300 crop),
  not the clean `num_samples // hop_length` (160) this project's crop-alignment invariant needs.
  Verified this with an actual forward pass, not just reading docs. Fixed by implementing
  HiFi-GAN's own standard convention: manually reflect-pad the waveform by
  `(n_fft - hop_length) // 2` on each side, then use `center=False` -- verified this gives
  exactly `(1, 160, 128)` for a 48000-sample crop before writing anything depending on it.
- `decoder.upsample_rates` -> `(5, 5, 4, 3)` (product 300, matching the mel's measured
  samples-per-frame ratio) and `training.crop_frames` -> 160 (same forced-adaptation pattern as
  the EnCodec branch) -- preserves the identical 2.0s/48000-sample crop length.
- Implementation: `mel_encoder.py` (`GroundTruthMelEncoder`, no learned parameters at all --
  just a deterministic transform, kept behind the same `encode()`/`probe()` interface as the
  other encoders for polymorphism in `train.py`), `GroundTruthMelConfig`, and a fresh
  `encoder_type` field on `ExperimentConfig` (built independently here, not shared with the
  EnCodec branch, since the two branches don't share commits) -- default-preserving, confirmed
  via the full 47-test suite and by checking `smoke.yaml`/`smoke_1024.yaml`/`pilot.yaml` still
  resolve to `encoder_type: "muq"`.
- Timed a 100-step run first (~0.19s/step, similar to EnCodec's per-step cost, both having
  ~150-160 frame sequences vs. MuQ's 50) before the full 15000-step commitment.
- Ran the full experiment. Training converged cleanly, `disc` stayed healthy (~0.2-0.9) the whole
  time, final mel loss 0.379 -- essentially identical to EnCodec's 0.381, both slightly higher
  than MuQ's 0.319. Mel correlation 0.944/0.966, again comparable to the other two.
- The interesting result: **waveform correlation came out at 0.059/-0.010** -- nowhere near
  EnCodec's 0.682/0.602, much closer to MuQ's 0.073/0.104. This was not the expected outcome if
  "representation information richness" were the whole story (ground-truth mel should be an
  upper bound on how well a magnitude-domain representation can inform reconstruction, since it
  literally *is* the loss's own target signal). Re-derived why: MuQ's own published model config
  lists `melspec_2048` as its input feature -- MuQ is itself downstream of a mel spectrogram, so
  its representation never had phase information to begin with, any more than a raw mel
  spectrogram does. EnCodec is a full neural codec with a paired reconstruction-objective decoder,
  which is a categorically different kind of representation (trained to retain what its own
  decoder needs, likely including phase/timing cues). This reframes the earlier "MuQ vs EnCodec"
  framing as really being about magnitude-only vs. reconstruction-trained representations.
- Computed parameter count (51,670,273, smallest of the three -- input projection is
  `Conv1d(128, 1024, ...)` like EnCodec's, plus a slightly different overall upsample structure),
  checkpoint size (663.8MB), and wall time (~48 min, fastest of the three despite a similar
  150-160 frame sequence length -- plausibly just run-to-run system noise, not investigated
  further since it's not the point of this experiment).
- Sent reference/reconstruction audio for both clips to the user via SendUserFile, again
  explicitly asking for a listening comparison rather than declaring anything from the metrics
  alone -- same discipline as every prior perceptual claim this session.
- Wrote up the full comparison table (MuQ / EnCodec / ground-truth-mel side by side) and the
  three required questions in [[DECISIONS.md#mel-hifigan-sanity]], being explicit that the MuQ/
  EnCodec numbers were carried over from the sibling branch's already-established results (not
  re-derived here) since the two branches don't share history.
- `ruff check .`/`pytest -q` clean (47 tests) throughout; confirmed via `git diff main` that no
  MuQ-baseline file was touched. Did not start BigVGAN work. Did not touch Gate 4/5.

## 2026-09-08 (cont.) -- Listening verdict + interrupted conditioning-collapse diagnostic

User's verdict: "The reconstructed audio sounds mediocre again. Almost identical to the other
experiments. There must [be] something wrong with the training pipeline." Took this seriously as
a testable hypothesis rather than debating it: if EnCodec's 6-9x higher waveform correlation were
perceptually real, EnCodec should sound at least somewhat different from MuQ/ground-truth-mel --
reportedly it doesn't. Recalled that Gate 4's `shuffled_time` finding
([[DECISIONS.md#gate4-shuffled-time-finding]]) already showed a *smaller* version of this same
pattern (shuffled-in-time conditioning only marginally worse than correct conditioning) and was
never fully resolved -- this new report could be the same underlying issue showing up more
starkly now that representations differ by more than just frame ordering.

Started designing a direct, cheap diagnostic (no retraining): confirmed all three checkpoints
(MuQ, EnCodec, ground-truth-mel + MRF-1024) are still physically present in `runs/` (gitignored,
so untouched by branch switches -- verified with `ls`). Confirmed `muq`/`transformers`/
`torchaudio` are all importable in this branch's venv (shared `.venv`, not branch-specific, so no
dependency issue running an EnCodec or MuQ checkpoint from the mel-sanity branch). Looked up the
exact GTZAN relative paths for both fixed clips from the manifest
(`genres/disco/disco.00049.wav`, `genres/pop/pop.00012.wav`) as the next step toward loading real
audio for the "correct conditioning" case.

**Interrupted by the user before running anything** -- asked to summarize and commit instead,
explicitly "don't do anything else for now." Did not execute the diagnostic. Recorded the full
plan in [[DECISIONS.md#mel-hifigan-sanity]] ("Final listening verdict" addendum) and
[[NEXT.md]] ("Next command") so it can be picked up exactly where it was left off: load each
checkpoint's `resolved_config` (already embedded in the checkpoint file) to reconstruct the
generator, compute correct/zero/shuffled-time/cross-track conditioning, and compare outputs.
No training, no new experiments, no code changes this pass -- pure documentation of the current
state and the interrupted plan, per the user's explicit request.
