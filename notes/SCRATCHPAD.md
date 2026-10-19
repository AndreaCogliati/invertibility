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

## 2026-09-09 -- `experiment/muq-to-dac-temporal-conv` branch: does local temporal context close the framewise MLP's gap?

Direct follow-up to `experiment/muq-to-dac-mlp`'s finding that its framewise MLP hit an exact
architectural ceiling (cosine similarity 0.64) because DAC's frame rate is exactly 3x MuQ's and a
memoryless per-frame map can't distinguish the 3 repeated-input output slots. Branched off `main`
(not off the MLP branch, per this project's standing pattern) -- had to recreate
`dac_encoder.py` and `models/framewise_mlp.py` identically since branches don't share commits;
the MLP's *trained checkpoint* (`runs/muq_to_dac_mlp/checkpoints/latest.pt`, git-ignored) did
survive the branch switch and was reloaded directly for a matching comparison WAV rather than
retrained.

**Single-variable change from the MLP experiment:** same input/target/optimizer/loss/step budget/
clips/crop, only the framewise `Linear` layers replaced by `Conv1D(k=5)` layers with trivial
residual connections (`x + GELU(conv(x))`), operating on the same repeated-to-75Hz sequence used
for temporal alignment. New modules: `src/muq_inversion/models/temporal_conv.py`
(`TemporalConvNet`), `scripts/train_muq_to_dac_temporal_conv.py`.

**Receptive field.** Analytical: `1 + 3*(5-1) = 13` frames at DAC's 75Hz timeline, translating to
`ceil(13/3) = 5` MuQ frames (roughly the current MuQ frame plus ~2 neighbors each side).
Verified empirically, not just trusted: perturbed one frame in a random input sequence and
measured which output frames changed by more than floating-point noise -- got exactly 13 frames
affected, matching the formula exactly. This is a good example of CLAUDE.md's "measure, don't
assume" principle applied to a structural/architectural claim, not just a data-geometry one.

**100-step smoke test result was already dramatic:** cosine similarity 0.9976 by step 100 (vs.
the framewise MLP's *converged, 3000-step* ceiling of 0.6418) and MSE down to 0.09 (vs. the MLP's
floor of ~11.05). This was surprising enough that the full 3000-step run was launched immediately
to confirm it wasn't a fluke or an early favorable coincidence.

**Full 3000-step run:** converged to cosine similarity 0.9999-1.0000 and MSE ~0.004-0.014 (final
10-step mean 0.004072, final/initial ratio 0.0002) -- essentially a full latent-space solve, not
just an improvement. Training time 18.12s total (6.04 ms/step) -- slower than the pure-MLP's
per-step cost (unsurprising, Conv1D vs Linear) but still trivial compared to any HiFi-GAN variant.
15,731,712 parameters (roughly 1.9x the framewise MLP's 8.4M, from the 3 extra `Conv1D` kernel
dimensions vs. `Linear`'s per-position weights).

**Waveform-domain result, decoded through the frozen DAC decoder:**
```
                 temporal-conv   MLP (reused ckpt)   DAC-oracle
pop.00012:       0.3683          0.3028              0.3666
disco.00049:     0.5753          0.1918              0.5735
```
And directly comparing the temporal-conv reconstruction against the DAC-oracle reconstruction
(not the original reference): **0.9999/1.0000** -- i.e. decoding the *predicted* latent produces
audio essentially indistinguishable, sample-for-sample-correlated, from decoding the *true* DAC
latent. Given the user's own prior verdict that the DAC oracle "is perfect," this strongly
predicts the temporal-conv reconstruction should also sound at or near that same ceiling. Sent
all four variants (reference/temporal-conv/MLP/DAC-oracle) for both clips to the user for a
listening verdict; not yet received as of this note. See `notes/DECISIONS.md#muq-to-dac-temporal-conv`
for the full write-up and answer to the experiment's critical question.

**Listening verdict (2026-09-09), confirmed after this scratchpad entry was written:** "The
listening test shows almost perfect reconstruction! We nailed it!" Committed as `fbb1fb5` and
tagged `muq-to-dac-temporal-conv-v1`. See `notes/DECISIONS.md#muq-to-dac-temporal-conv`'s
"Listening verdict" section (added retroactively there, not duplicated here per this file's
append-only convention).

## 2026-09-09 -- `experiment/muq-to-dac-minimal-gate4` branch: does the Gate 3 result generalize?

Branched off `experiment/muq-to-dac-temporal-conv` directly (not off `main`, unlike every prior
sibling experiment) since this is a direct extension of that branch's validated architecture,
not an independent ablation -- inherits `dac_encoder.py`, `models/framewise_mlp.py`,
`models/temporal_conv.py` via git history instead of recreating them.

**Manifest first.** Wrote `scripts/build_gate4_manifests.py` before any training code, producing
`data/manifests/gate4_crops.csv` (44 rows) with built-in stop-condition checks run at build time:
region-disjointness with a 1.0s guard band for Experiment A (verified programmatically, not just
by eyeballing the docstring's numbers), and a checksum/track-split-uniqueness check for
Experiment B (confirmed no duplicate-checksum GTZAN defect among the 14 candidate tracks, and no
track appears in more than one Experiment B split).

- **Experiment A** (`pop.00012`, `disco.00049`, both 30.00s): train=8 crops/track at
  starts 0,2,...,14 (covers [0,16)), 1.0s guard, val=2 crops/track at 17,19 (covers [17,21)),
  1.0s guard, test=2 crops/track at 22,24 (covers [22,26)). 16/4/4 train/val/test crops total.
- **Experiment B**: 8 train tracks (2 each from pop/disco/classical/rock, excluding the two
  Experiment A tracks), 2 val tracks (pop.00003, disco.00003), 2 test tracks (classical.00003,
  rock.00003) -- verified via direct checksum comparison to have no duplicate-checksum collisions
  with each other or with Experiment A's tracks. Train tracks get 2 crops each (starts 5.0s,
  15.0s); val/test get 1 fixed crop each (10.0s). 16/2/2 train/val/test crops total.

**First bug caught by my own stop-condition check -- and it was my own check that was
miscalibrated, not the pipeline.** Applied the "training loss must reproduce Gate 3 overfit
behavior" stop condition to the *full* 16-crop training set at step 300 (threshold: mse ratio
<=0.5) -- it failed (ratio 0.77 for A, 0.69 for B). Investigated before concluding anything was
broken: the spec says "on a tiny subset," and fitting 16 diverse crops jointly is a genuinely
harder problem than Gate 3's 1-2-crop overfit, so slower convergence there is expected data, not
evidence of a broken pipeline. Fixed by splitting this into two separate checks: (1) a new
`run_tiny_subset_sanity_check` that mirrors Gate 3 exactly (fresh model, ONE crop, 300 steps,
same 0.5 threshold) as a pre-flight gate before trusting the real experiments at all -- this
passed cleanly (ratio 0.0002, matching Gate 3's own convergence pace almost exactly); (2) the
full-training-set ratio at step 300 is now printed as informational only, not a stop condition.

**Training (8000 steps each, up from Gate 3's 3000 -- documented deviation, justified by needing
more exposures per crop across 16 diverse training examples instead of 1-2 memorized ones; same
per-step cost, ~7ms, so still trivial wall-clock: 54.7s for A, 56.4s for B).** Both experiments
show the same clean, unambiguous pattern: **train MSE decreases steadily throughout training
(cosine similarity reaching 0.40-0.45 by step 8000) while validation MSE increases monotonically
from the very start of training.** The checkpoint-selection rule (lowest validation MSE, not
final step, not test performance, per this experiment's explicit requirement) picked **step 200
out of 8000** for both experiments -- essentially right at the start, before any real fitting had
happened. This is about as clean a memorization/overfitting signature as this kind of experiment
can produce.

**Full metrics** (train/val/test, correct-conditioning cosine similarity and waveform correlation
to the DAC oracle) at the selected checkpoint:
```
Experiment A:  train cos=0.452 wav_corr=0.329  |  val cos=0.179 wav_corr=0.006  |  test cos=0.184 wav_corr=-0.009
Experiment B:  train cos=0.395 wav_corr=0.251  |  val cos=0.150 wav_corr=-0.005 |  test cos=0.140 wav_corr=-0.001
```
Held-out performance (val/test) is nearly identical between Experiment A (unseen crops, same
tracks) and Experiment B (entirely unseen tracks) -- both collapse to roughly the same
near-zero-correlation floor, with no clear advantage for "at least it's the same track."

**Controls.** On train data, correct conditioning clearly beats both shuffled-across and
time-shifted controls (e.g. Experiment A train: cos 0.452 correct vs. 0.154/0.118 shuffled/
shifted) -- a real, well-separated effect confirming the model has learned *something*
conditioning-dependent on its training crops. On held-out data, correct vs. controls are barely
distinguishable and all near-zero (Experiment A val: cos 0.179 correct vs. 0.132/0.115 shuffled/
shifted; waveform correlation to oracle 0.006/0.001/0.003 -- all indistinguishable from noise).
This is the sharpest evidence for memorization: the model's conditioning-sensitivity, which is
large and real on training data, essentially vanishes once evaluated out-of-distribution.

**Stop conditions checked, none triggered:** no split/track overlap (verified twice, at
manifest-build time and at run time), frame alignment held exactly (ratio=3 for every one of the
44 crops across both experiments, asserted per-crop), the tiny-subset sanity check passed,
evaluation controls did NOT perform similarly to correct conditioning on training data (they did
on held-out data, which is itself the finding, not a broken-pipeline signal), the DAC oracle's
own quality was unaffected (waveform correlation to reference 0.39-0.67 across both experiments,
consistent with every prior MuQ-to-DAC experiment's oracle numbers), no non-finite losses or
latents anywhere (`nonfinite_seen: False` in both experiment reports), and validation metrics
were never used to touch test data (test crops were only touched in the final evaluation pass,
after checkpoint selection was already fixed by validation MSE alone).

**Listening artifacts:** 30 WAVs written (2 experiments x 3 splits x 5 conditions), descriptive
filenames (`{split}_{track}_t{crop_start}s_{condition}_{run_id}_step{n}.wav`) in
`artifacts/gate4_minimal/{A,B}/`. Sent a representative 12-file subset to the user (train
reconstruction + full 5-condition held-out comparison, for each experiment) for a listening
verdict; not yet received as of this note.

Full parameter count (15,731,712, unchanged from Gate 3 by construction), receptive field (13
DAC-frames / 5 MuQ-frames, unchanged), and peak MPS memory (~2.45-2.47 GB) recorded in
`artifacts/gate4_minimal/report.json` and reproduced in `notes/DECISIONS.md#muq-to-dac-minimal-gate4`.

**Gate 4 listening verdict (2026-09-09), confirmed after this scratchpad entry was written:**
"The listening confirms the poor result. The model is not learning and generalizing." Committed
as `b9f82ef`. See `notes/DECISIONS.md#muq-to-dac-minimal-gate4`'s "Listening verdict" section.

## 2026-09-09 -- `experiment/muq-to-dac-scale-data` branch: is Gate 4's failure data-hungriness or something fundamental?

Branched off `experiment/muq-to-dac-minimal-gate4` directly (continuing the same
branch-off-the-parent-experiment pattern started by Gate 4), inheriting `gate4_common.py`,
`temporal_conv.py`, `dac_encoder.py` unchanged via git history.

**Track selection:** 100 GTZAN tracks, `<genre>.00010`-`<genre>.00019` for all 10 genres (avoids
the known-corrupted `jazz.00054.wav` entirely by construction). Verified via direct SHA-256
checksum comparison across all 100 candidates: zero duplicate-checksum collisions. Split: 8 train
+ 1 val + 1 test per genre = 80/10/10 tracks, no track in more than one split. Crops: 4 per train
track (starts 4,10,16,22s), 2 per val/test track (starts 8,18s) -> 320/20/20 crops. Written by
`scripts/build_scale_data_manifest.py` -> `data/manifests/scale_data_crops.csv` (360 rows).

**Precompute + cache, as required ("avoid recomputing encoders during training").**
`scripts/precompute_scale_data_cache.py` encodes all 360 crops with both frozen encoders once
(15.10s total -- MuQ+DAC encoding is fast, ~34ms/crop combined) and writes one consolidated file,
`data/cache/scale_data/muq_dac_cache.pt` (281MB, git-ignored). The training script loads this and
never touches `FrozenMuQEncoder` again (only `FrozenDacEncoder.decode()` at evaluation time, for
turning latents back into audio).

**A genuinely useful accident during smoke-testing.** Tried to smoke-test with a reduced step
count via `sed -i '' 's/^STEPS = 20000$/STEPS = 1000/' ...` -- the substitution silently failed
because the real line has a trailing comment (`STEPS = 20000  # documented deviation...`), so the
`$`-anchored pattern never matched, while the `LOG_EVERY`/`VAL_EVERY`/`RUN_ID` substitutions (no
trailing comments) succeeded. The result: an accidental but fully valid execution of the complete
20000-step experiment, just with a finer validation cadence (every 100 steps instead of the
intended 500) and the cosmetic run label `scale_data_timing`. Recognized this from the run's own
duration (~6 minutes, far longer than a real smoke test) and the printed step counts going all
the way to 20000/20000 -- verified by grepping the actual executed file's constants rather than
trusting my own sed invocation, and confirmed the `best_checkpoint_step: 300` finding was
consistent with a 100-step validation cadence (LCM(100,250)=500 explains why printed log lines
only appeared every 500 steps despite VAL_EVERY=100). Adopted this run as the official result --
scientifically identical to a "properly" smoke-tested-then-launched run, and if anything more
rigorously monitored -- by renaming `runs/scale_data_timing/` -> `runs/scale_data/` and
`artifacts/scale_data_timing/` -> `artifacts/scale_data/` rather than spending another ~6 minutes
re-running an identical experiment under a different label. (Lesson for next time: sed patterns
anchored with `$` are fragile against trailing comments -- match on the value plus optional
comment, or edit via the Edit tool instead of `sed` for anything with inline documentation.)

**Result.** Training time 252.4s (up from Gate 4's ~55s, mostly from the finer VAL_EVERY=100
evaluation cadence evaluating all 320 train crops every 100 steps -- 200 such evaluations, each
340 forward passes, dominates the wall clock; the core per-step training cost is unchanged).
Training was noisier than Gate 4's (occasional large train-MSE spikes, e.g. 224.7 at step 2500,
64.4 at step 12000, both self-correcting within a few hundred steps) -- expected given the far
greater genre/content diversity per cyclically-rotated training step, not an instability
requiring intervention (no non-finite values, no divergence). Best-by-validation-MSE checkpoint:
step 300 of 20000 (1.5% through training) -- still very early, though `val_cos` alone (not the
selection criterion) peaks somewhat later, around steps 5000-6500 (~0.21-0.23), before drifting
back down -- MSE-based and cosine-based selection don't perfectly agree here, a minor nuance
worth flagging since the task requires MSE-only selection.

**Metrics (train/val/test, correct-conditioning cosine similarity and waveform correlation to
DAC oracle), compared against Gate 4:**
```
                Gate 4 (16 train crops)          Scale-data (320 train crops, 20x)
val:   cos=0.15-0.18  wav_corr=0.006/-0.005   ->  cos=0.222  wav_corr=-0.0045
test:  cos=0.14-0.18  wav_corr=-0.009/-0.001  ->  cos=0.223  wav_corr=-0.0043
```
Cosine similarity improved meaningfully on held-out data (~25-45% relative). Waveform correlation
to the oracle did not move at all -- stayed at the same noise-floor level in both experiments.

**Controls, the most interesting difference from Gate 4:** on train data, correct beats shuffled/
shifted clearly but by a *smaller* relative margin than Gate 4's train gap (cos 0.243 vs.
0.142/0.156 here, vs. Gate 4's 0.40-0.45 vs. 0.11-0.20) -- expected, since fitting 320 diverse
crops is harder than fitting 16. On held-out data, **correct conditioning now shows a real,
consistent edge over both controls** (val: 0.222 vs. 0.144/0.157; test: 0.223 vs. 0.111/0.142) --
qualitatively different from Gate 4, where correct and controls were statistically
indistinguishable on held-out data. This is evidence of a genuine (if weak) transferable signal
that did not exist at Gate 4's data scale.

Listening artifacts: 30 WAVs (6 tracks -- 3 val + 3 test -- x 5 conditions) written to
`artifacts/scale_data/`, descriptive filenames. Sent a 10-file representative subset (one val,
one test track, all 5 conditions each) to the user; verdict pending as of this note.

**Note:** the user moved directly to requesting the further data-scaling experiment below without
providing a listening verdict for this 80-track experiment. Not blocking -- proceeded per the
explicit new request. If a verdict for this experiment arrives later, record it as an addendum
to this section, not to the more-data section below.

## 2026-09-09 -- `experiment/muq-to-dac-more-data` branch: does the scaling trend continue at full-GTZAN scale?

Branched off `experiment/muq-to-dac-scale-data` directly (continuing the parent-branch-extension
pattern started by Gate 4), carrying forward that branch's uncommitted work automatically (git
preserves working-tree changes across a same-commit branch creation) -- nothing from the prior
experiment was lost or needed re-doing.

**Track selection: reused Gate 1's own committed split instead of building another one.**
`data/manifests/gtzan_manifest_smoke.csv` (committed 2026-09-04) already has a full 1000-track,
checksummed, duplicate-safe, corruption-flagged train/val/test split (793/94/113) from Gate 1 --
using it directly for this experiment (dropping just the 1 corrupted row) is both more thorough
(uses essentially the whole dataset, not another hand-picked subset) and safer (leakage-checked
machinery that already exists, documented in `notes/DECISIONS.md#gtzan-source-and-defects`) than
re-deriving a new split. Re-verified track-disjointness and checksum-based no-cross-split-leakage
at manifest-build time anyway (`scripts/build_more_data_manifest.py`), rather than trusting the
source file's own claims uncritically. Result: 792 train / 94 val / 113 test tracks, 2 crops each
(starts 5.0s, 15.0s) -> 1584/188/226 crops (1998 total).

**Precompute:** `scripts/precompute_more_data_cache.py`, 1998 crops in 82.15s (~41ms/crop,
matching the 80-track experiment's per-crop rate) -> `data/cache/more_data/muq_dac_cache.pt`
(1.5GB, git-ignored).

**A deliberate departure from the prior two experiments' evaluation pattern, to control wall
time.** Both Gate 4 and the 80-track experiment recomputed full-training-set MSE/cosine at every
validation checkpoint for the logged curve -- fine at 16-320 crops, but would dominate wall time
at 1584. Replaced with a running average over the training steps *since the last checkpoint*
(free, already computed during the normal forward/backward) for the "train" curve, while still
evaluating the *full* validation set (188 crops, cheap) for checkpoint selection, per this
experiment's explicit "validation MSE only" requirement. A one-time full-train-set pass is run
once at the end, on the selected checkpoint, for the final reported train metrics.

**Smoke-testing lesson applied.** Used the Edit tool (not `sed`) to reduce `STEPS`/`VAL_EVERY`/
`RUN_ID` for a smoke test this time, specifically because of the prior experiment's `sed`-anchor
failure against a trailing comment. Verified the edits actually took effect by grepping the
constants in the copied file before running -- caught nothing wrong this time, and the smoke test
(1500 steps) completed correctly. Its biggest surprise: the **evaluation phase took ~498s**, far
longer than training itself (11.72s for 1500 steps) -- almost entirely from reloading and
resampling raw audio per crop during evaluation (2000 crops, each a real disk read + `librosa`
`soxr_hq` resample) plus ~8000 DAC decode calls and ~6000 multi-resolution STFT computations. This
cost is fixed regardless of `STEPS` (it only depends on dataset size), so it was budgeted into
the full run's time estimate (~16-17 min total) rather than optimized away.

**Full run:** launched properly in the background this time (`nohup ... &` + `disown`, tracked by
real PID via `kill -0` and a `Monitor` until-loop -- not the `run_in_background`+`&` combination
mistake made twice earlier in this investigation). 60000 steps, training 403.55s (6.73ms/step,
consistent with every prior experiment's per-step cost), evaluation 477.0s. Total wall time ~15
minutes, matching the smoke-test-based estimate closely.

**The training curve was genuinely different from every prior experiment -- a turbulent early
phase, then a late, clean improvement phase.** Steps ~5000-32000 showed large, repeated loss
spikes (train MSE up to 153,842 at step 7000; val MSE up to 413.9 at step 11000) -- never
non-finite (no stop condition triggered), but clearly unstable, unlike any smaller-scale
experiment's smooth training. From step ~33000 onward, training stabilized and improved cleanly:
val MSE dropped from ~18-21 down to ~17.3, val cosine climbed from ~0.16-0.21 up to ~0.22-0.23.
**Best-by-validation-MSE checkpoint: step 50000 of 60000 (83.3% through training)** -- a dramatic
qualitative change from Gate 4's step 200/8000 (2.5%) and the 80-track experiment's step 300/20000
(1.5%). At this data scale, the model does not immediately overfit; validation performance
actually improves for most of training, after an initial unstable period.

**Metrics comparison across all three data scales (val/test cosine similarity and waveform
correlation to oracle, correct conditioning):**
```
                    Gate 4 (16 crops, 8 tracks)   80-track (320 crops, 80 tracks)   More-data (1584 crops, 792 tracks)
val cos:            0.150-0.179                    0.222                              0.230
test cos:           0.140-0.184                    0.223                              0.226
val wav_corr:       0.006/-0.005                   -0.0045                            0.0002
test wav_corr:      -0.009/-0.001                  -0.0043                            -0.0009
best ckpt step:     200/8000 (2.5%)                300/20000 (1.5%)                   50000/60000 (83.3%)
```
Cosine similarity: a large jump from Gate 4 to the 80-track experiment (~0.16 -> ~0.22), then a
much smaller jump from the 80-track experiment to this one (~0.22 -> ~0.23) despite a
similarly-sized (in orders of magnitude) further data increase -- classic diminishing-returns
shape. Waveform correlation to the oracle: unchanged at ~0 across all three scales, no signal of
improvement at any point.

**Controls:** held-out cosine gap (correct minus controls) was roughly similar to, or modestly
larger than, the 80-track experiment's gap (val: correct 0.230 vs. shuffled 0.123/shifted 0.149,
gap ~0.081-0.107; test: correct 0.226 vs. shuffled 0.119/shifted 0.134, gap ~0.092-0.107) -- not a
dramatic further increase, roughly holding steady rather than continuing to grow.

Listening artifacts: 30 WAVs (6 tracks x 5 conditions) written to `artifacts/more_data/`. Sent a
10-file representative subset (one val, one test track, all 5 conditions each) to the user;
verdict pending as of this note. See `notes/DECISIONS.md#muq-to-dac-more-data` for the full
write-up, six-question interpretation, and final scaling-curve-based recommendation.

## 2026-09-10 -- `experiment/muq-layer-sweep` branch: is the final MuQ layer the wrong source representation?

Branched off `experiment/muq-to-dac-more-data` (continuing the parent-branch-extension pattern),
inheriting `gate4_common.py`, `temporal_conv.py`, the more-data manifest, and its DAC cache
unchanged via git history / gitignored `data/`+`runs/` persistence.

**Inspected MuQ's hidden-state structure directly, per CLAUDE.md's "measure, don't assume" rule
applied to layer indexing** (`scripts/inspect_muq_hidden_states.py`): `output_hidden_states=True`
returns 13 tensors; measured `len(model.model.conformer.layers) == 12`, confirming index 0 is the
frontend/embedding output (before any transformer block) and indices 1-12 are the outputs after
each of the 12 Conformer blocks, with `hidden_states[12]` verified byte-identical
(`torch.equal`) to `last_hidden_state` -- the representation used throughout every prior
experiment. All 13 hidden states share shape `(1, 50, 1024)` (measured on a 2s crop) -- same
channel count, same frame rate -- so no shape adapter is needed anywhere in this sweep.

**Candidate selection.** `layer_sweep_common.select_evenly_spaced_layers(12, 5)` --
`linspace(1, 12, 5)` rounded to nearest int, Python's banker's rounding turning the exact
midpoint 6.5 into 6 rather than 7 -- gives **[1, 4, 6, 9, 12]** (earliest / early-middle / middle
/ late-middle / final). This is a pure, unit-tested function (`tests/test_layer_sweep.py`), not
a value chosen by eye.

**Caching.** One MuQ forward pass per crop (`output_hidden_states=True`), extracting all 5
candidate layers at once, for the SAME 1998-crop `more_data_crops.csv` manifest (reused
unchanged, not rebuilt) -- 45.91s total, `data/cache/layer_sweep/muq_layers_cache.pt` (2.05GB,
git-ignored, new file). DAC latents were NOT recomputed: read directly from the existing
`data/cache/more_data/muq_dac_cache.pt`, with an explicit runtime assertion that its mtime never
changes during the precompute run (verifying it truly stays read-only, not just documented as
such).

**Budget choice, made before looking at any layer but the final-layer control.** Reviewed
`artifacts/more_data/report.json`'s curve: turbulent/unstable through ~step 32000 (val MSE
spikes to 30-142), then a comparatively stable region to step 60000 (val MSE 17.3-21.4, best at
50000). Chose **45000 steps** -- past the turbulent phase, samples several near-best points,
without fully repeating the 60000-step run. Documented in the training script's own docstring
*before* running anything, per the task's explicit ordering requirement.

**A genuine surprise: the "final-layer control run" was itself turbulent for most of its
budget, only settling in the last third.** Its val MSE spiked to 142.3 (step 10000) and 65.5
(step 15000) -- reproducing the exact turbulent signature the original more-data run showed at
similar step counts -- before improving to 18.6 by step 45000. But the *best* checkpoint by
validation MSE turned out to be from **step 1500** (val_mse=17.34), an early, lucky dip before
the turbulence, not from the stable late region. This is a new, slightly uncomfortable finding
about this training regime: validation-MSE-only checkpoint selection can occasionally pick an
early, possibly less-representative checkpoint if it happens to land on a lucky low point before
a long turbulent stretch -- worth flagging even though it didn't change this experiment's
conclusion (the sanity-check bounds, val_mse<=25 and val_cos>=0.15, were still comfortably met:
17.34 and 0.2208).

**Full comparison table** (ordered by MuQ depth; primary ranking metric val MSE, lower better):
```
        layer  idx  best_step  val_mse  val_cos  test_mse  test_cos  margin_shuf  margin_shift  val_wav_corr
     earliest    1      18000  17.8052   0.2443   17.6505    0.2395       0.1242        0.0818        0.0004
 early_middle    4       4500  17.5662   0.2301   17.4594    0.2255       0.1121        0.0731        0.0000
       middle    6       2500  17.9254   0.2160   17.8065    0.2130       0.1012        0.0689       -0.0013
  late_middle    9      33000  17.6987   0.2260   17.5615    0.2233       0.1005        0.0719        0.0004
        final   12       1500  17.3382   0.2208   17.2589    0.2154       0.1022        0.0733        0.0001
```
**Final layer wins the primary metric (val AND test MSE, both lowest of the five) -- no
candidate layer beats the baseline.** Interestingly, "earliest" (layer 1) has the *highest*
cosine similarity on both val and test (0.244/0.240, vs. final's 0.221/0.215) and the largest
controls margin -- a real but small secondary-metric edge that doesn't come with a lower MSE or
any waveform-domain gain (`val_wav_corr` is ~0 for literally every layer, no exceptions).
Val/test rankings agree closely for both metrics (same top-2 under each), so this isn't a
test-performance artifact -- it's a genuine, if minor and metric-dependent, MuQ-depth effect.

**Decision rule applied literally:** "a layer qualifies for a full follow-up run only if it
improves validation latent prediction reproducibly and substantially relative to the re-run
final-layer baseline." No layer does -- final already has the lowest val MSE. Per the spec's own
fallback ("if all layers perform similarly... document architecture/context as the next
hypothesis"), no follow-up run is recommended.

**Listening screen: not generated.** The spec calls for a blinded A/B between "the current
final-layer baseline" and "the validation-selected best candidate layer" -- but the
validation-selected best candidate *is* the final layer. There is no second, distinct condition
to blind-test against the baseline, so producing WAV files here would just be comparing the same
checkpoint against itself. Noted explicitly rather than silently skipped, and rather than
substituting a different layer (e.g. "earliest," the cosine-similarity standout) that the primary
ranking metric did not select -- doing so would violate "do not declare success from a tiny
numerical difference."

Full six-plus-one-question interpretation and final recommendation in
`notes/DECISIONS.md#muq-layer-sweep`.

## 2026-09-14: Spectrogram-to-DAC positive control (tasks/TASK-01.md)

Hypothesis: does the MuQ-to-DAC training pipeline itself (architecture/loss/optimizer) work at
all, when given a representation known to carry far more information than MuQ?

Commands run:
```
uv run python scripts/precompute_spectrogram_cache.py   # 1998 crops, 14.58s
uv run python scripts/train_spectrogram_to_dac.py        # tiny-subset sanity PASSED, 60000 steps, 517.98s train
uv run ruff check .    # clean
uv run pytest -q       # 73 passed
```

Training curve was turbulent (same character as `more_data`'s own instability) -- one huge but
finite val_mse spike (4.7e8) at step 15000, recovered by step 20000, no non-finite-loss stop
triggered. Best-by-val-MSE checkpoint landed at step 1000/60000 (first validation check) --
double-checked this wasn't a misleading fluke by comparing to the step-60000 end-of-training
numbers (17.06/0.27 vs. the selected 16.54/0.30) -- close enough that it doesn't look like a
lucky-dip artifact this time, unlike the muq-layer-sweep control run's step-1500 selection.

Direct comparison against `artifacts/more_data/report.json` (val split):
- MSE: 17.29 (MuQ) -> 16.54 (spectrogram), ~4% better
- cosine: 0.230 -> 0.298, ~30% relative better
- STFT distance (pred vs oracle): 2.65 -> 3.02, WORSE
- waveform correlation (pred vs oracle): 0.0002 -> -0.0008, both ~noise floor, unchanged

New latent diagnostics (val split, predicted vs oracle):
- norm_ratio = 0.364 (severe collapse)
- variance_ratio = 0.110 (severe collapse)
- lag1_autocorr: predicted 0.970 vs oracle 0.092 (extreme over-smoothing -- oracle is nearly
  uncorrelated frame-to-frame, predictions are nearly a random walk)

Interpretation: modest latent-metric gains, zero audio-domain gain, same-character
norm/variance collapse and over-smoothing as every MuQ-based run. Outcome B (similar failure) per
the task's own pre-registered decision rule -- full write-up in
`notes/DECISIONS.md#spectrogram-positive-control`.

30 listening WAVs written to `artifacts/spectrogram_control/`. Sent to the user; listening
verdict not yet in hand as of this note -- treat the "Outcome B" reading as provisional until
confirmed by ear, per this project's own repeated experience of metrics and listening
disagreeing (muq-to-dac-mlp, muq-to-dac-temporal-conv).

## 2026-09-14: Dense-audio-coverage rerun (tasks/TASK-02.md)

Hypothesis: was TASK-01's negative result caused by using only ~13.3% of GTZAN, rather than the
pipeline itself?

Commands:
```
uv run python scripts/build_dense_gtzan_manifest.py    # 14976 crops, 999 tracks, matches estimate exactly
uv run python scripts/precompute_dense_cache.py          # 354.5s, 10.36GB cache (est. 10.35GB)
uv run python -u scripts/train_spectrogram_to_dac_dense.py   # sanity PASSED, 450000 steps, 4677s train, 3805s eval
uv run ruff check .    # clean
uv run pytest -q       # 79 passed
```

Training curve, same turbulent character as every prior run in this line: val_mse readings at
75k/150k/225k/300k/375k/450k steps were 17.17/31.32/17.57/1051.51/26.03/20.03 -- one huge but
finite spike at step 300000, recovered by 375000, no stop condition triggered.

Best-by-val-MSE checkpoint: step 1000/450000 (again the earliest possible selection point --
third experiment in a row with this pattern, see NEXT.md).

Three-way comparison (val):
                    MuQ->DAC   TASK-01(sparse)  TASK-02(dense)
MSE                 17.29      16.54            16.44
cosine               0.230      0.298            0.293
STFT distance         2.65       3.02             3.09
wav corr (~0)        0.0002    -0.0008          -0.0014
norm ratio            n/a       0.364            0.322
variance ratio        n/a       0.110            0.086
lag1 autocorr pred     n/a       0.970            0.952

TASK-01 vs TASK-02: no metric moved meaningfully in the "more data helps" direction; norm/
variance ratios actually moved slightly further from 1.0. Outcome C per the task's own
pre-registered rule: increasing coverage doesn't address the pathology.

30 listening WAVs written to artifacts/dense_spectrogram_control/, same 6 tracks as TASK-01
(t=4.0s vs t=5.0s). Sent to the user alongside TASK-01's for a direct comparison. Full write-up:
notes/DECISIONS.md#dense-spectrogram-positive-control.

## 2026-09-14: TASK-03 (BlueHive + W&B instrumentation)

No SSH/sbatch access to BlueHive from this environment -- surfaced to user via AskUserQuestion
before starting, user confirmed: build everything runnable, validate locally, hand off cluster
execution. Found real W&B credentials in ~/.netrc (andrea-cogliati-university-of-rochester).

Commands:
```
uv add wandb
uv run ruff check .; uv run pytest -q   # 103 passed
uv run python -u scripts/train_spectrogram_to_dac_wandb.py --steps 50 --wandb-mode disabled     # PASSED
uv run python -u scripts/train_spectrogram_to_dac_wandb.py --steps 2500 --wandb-mode online      # PASSED, real run
uv run python scripts/analyze_early_checkpoint_selection.py
```

Online smoke test: real W&B run at wandb.ai/andrea-cogliati-university-of-rochester/
muq-invertibility/runs/gz3uffcf. 5 files, 9 media, 20 artifacts synced, zero errors. Step-1000
checkpoint val_mse=16.442551 -- exact match to TASK-02's own number, confirming determinism.

Early-checkpoint analysis (the interesting part): checked TASK-01/TASK-02's full training_curves
against (a) rank of the step-1000 checkpoint among all checkpoints by val_mse, (b) a trivial
"predict the mean DAC vector" baseline computed directly from the cache, (c) correlation between
val_mse and val_cos across checkpoints.

Results:
- TASK-01: step 1000 is rank 1/60 (never beaten). Baseline MSE 17.65 vs selected 16.54 (0.937x).
  corr(mse,cos) excl. spikes = -0.72.
- TASK-02: step 1000 is rank 1/450 (never beaten across 449000 more steps!). Baseline MSE 17.63
  vs selected 16.44 (0.933x). corr(mse,cos) excl. spikes = -0.72. Best cosine (0.300) occurs at
  step 443000 with MSE 16.60 -- WORSE than step 1000's MSE, so MSE-only selection structurally
  can't find it even though it's a real late-training improvement.

Conclusion: regression-to-the-mean, not instability/noise. Same mechanism as the norm/variance
collapse already documented -- a near-constant prediction is the MSE-cheap, norm-collapsed, and
maximally-smooth solution simultaneously, for this loss/target combination.

Full write-up: notes/DECISIONS.md#task03-bluehive-wandb. bluehive/README.md +
bluehive/run_task03.slurm are the handoff -- genuinely untested against a real cluster, flagged
as such.

## 2026-09-15: Spectrogram Gate-3 memorization control (tasks/TASK-04.md)

Hypothesis: can Spectrogram->DAC memorize the same 2 fixed excerpts (pop.00012, disco.00049) as
the successful MuQ Gate-3 control?

Found the exact MuQ Gate-3 crop (scripts/train_muq_to_dac_temporal_conv.py: centered crop,
start=(len-48000)//2 -> 14.000083s for both tracks) and confirmed it's already in TASK-02's dense
cache at t=14.00s (crop_index=7), 2 samples/83us off. Reused directly, no new cache.

Commands:
```
uv run python -u scripts/train_spectrogram_gate3_wandb.py --wandb-mode disabled   # first pass
```
First pass revealed a bug: comparing the MuQ checkpoint against examples[track].dac (TASK-02's
dense-grid target) gave mean_cos=0.577, mean_mse=15.9 -- wildly worse than the documented
0.9999-1.0000. Debugged directly: reproduced independently, found the MuQ model is EXTREMELY
sensitive to the 2-sample/83us crop offset -- evaluating on its exact own crop gives cos=0.99998,
evaluating 2 samples off gives cos=0.577. Real finding, not a bug in my harness. Fixed the
comparison section to recompute MuQ's input/target from its own exact crop formula.

Reran clean:
```
uv run python -u scripts/train_spectrogram_gate3_wandb.py --wandb-mode disabled   # PASSED, byte-identical curve to first pass (as expected, bug was only in the post-hoc comparison)
uv run python -u scripts/train_spectrogram_gate3_wandb.py --wandb-mode online     # the real run, 3000 steps is small enough this IS the real experiment, not just a smoke test
uv run ruff check .; uv run pytest -q   # 117 passed
```

Refactored testable logic (evaluate_prediction, three_way_grad_norms, count_nonfinite,
ExampleData/load_examples, build_memorization_log_payload) into
src/muq_inversion/gate3_spectrogram_common.py, following the gate4_common.py/
layer_sweep_common.py precedent, rather than importing from scripts/ in tests (no prior
precedent for that in this repo).

Final result (best checkpoint step 2536/3000): mse=0.000547, cos=0.99999, norm_ratio=0.9999,
var_ratio=0.9998, wav_corr=0.99999, stft=0.0145 -- matching/exceeding MuQ's own (re-evaluated
correctly): mse=0.001936, cos=0.99996, norm_ratio=1.0038, var_ratio=1.0073, wav_corr=0.99994,
stft=0.0415. Strong Success. Full write-up: notes/DECISIONS.md#task04-spectrogram-gate3.

Live W&B run: wandb.ai/andrea-cogliati-university-of-rochester/muq-invertibility/runs/7kfiziym

## 2026-09-15: TASK-04 listening verdict

User's verdict on the 4 sent WAVs (pop.00012 reference/dac_oracle/best_by_mse-step2536/step0):
"the two audio excerpts are already great at 1000 steps. Almost undistinguishable from the final
version or the original. The early excerpts show signs of artifacts, though."

Confirms the metrics by ear -- Strong Success holds perceptually, not just numerically. Also
confirms perceptual convergence is front-loaded (matches cosine already at 0.98 by step 1000),
and that early/unconverged checkpoints are audibly worse (matches their far-from-converged
metrics). Recorded in notes/DECISIONS.md#task04-spectrogram-gate3 and notes/MEMORY.md.

## 2026-09-15: TASK-05 (Gate-3 scaling curve, N=2/8/32/128/512)

Goal: find where collapse begins between TASK-04's N=2 (perfect) and TASK-01/02's N~15000
(collapsed).

Key design decision: FIXED 20000-step max budget across all 5 levels (not scaled with N like
TASK-02 did) -- otherwise every example could trivially memorize regardless of N, making the
experiment unable to detect anything. Stop early on patience (best MSE no improvement >1e-5 in
5000 steps).

Manifest construction: round-robin across 10 genres alphabetically, deterministic track order,
anchored at TASK-04's exact 2 tracks for N=2. All crops at crop_index=7 (t=14.0s) -- already in
TASK-02's dense cache for every one of 999 tracks, so zero new precompute needed. Verified
nesting (2⊂8⊂32⊂128⊂512) and genre balance in tests.

Commands:
```
uv add matplotlib
uv run python scripts/build_gate3_scale_manifests.py
uv run python -u scripts/train_gate3_scale_wandb.py --levels 2 --wandb-mode disabled   # smoke, PASSED, 9000 steps, 666s
uv run python -u scripts/train_gate3_scale_wandb.py --wandb-mode online   # real run, all 5 levels, ~53 min total
uv run python scripts/plot_gate3_scale_figures.py
uv run ruff check .; uv run pytest -q   # 130 passed
```

Results (best-by-MSE checkpoint per level):
N=2:   mse=0.000005 cos=1.0000 norm=1.0000 var=1.0000 wav_corr=1.0000
N=8:   mse=0.077636 cos=0.9980 norm=1.0025 var=1.0077 wav_corr=0.9970
N=32:  mse=12.88     cos=0.5161 norm=0.4993 var=0.1773 wav_corr=0.4689
N=128: mse=15.61     cos=0.3610 norm=0.3778 var=0.0630 wav_corr=0.0901
N=512: mse=16.02     cos=0.3266 norm=0.3434 var=0.0480 wav_corr=0.0193

Sharp break between N=8 and N=32 (>50% of the total range moves in one step), then decelerating
decay N=32→128→512 approaching an asymptote. N=512's numbers already match TASK-01 (1584 crops)
and TASK-02 (11873 crops) almost exactly -- the collapse regime is fully established by 512
examples, ~23-30x less data than either GTZAN-scale run used.

N=512's training itself was turbulent (huge finite spikes, mse up to 126000, norm_ratio up to
39x) -- same instability pattern as TASK-01/02's GTZAN training, now reproduced at much smaller
scale. Best checkpoint (step 3000/8000) unaffected since selection reloads from saved checkpoint,
not the noisy final state.

Full write-up: notes/DECISIONS.md#task05-gate3-scaling. Figures:
artifacts/task05_gate3_scale/scaling_curve_main.png, scaling_curve_norm_variance_ratio.png.
20 listening WAVs across the 5 levels, not yet reviewed.

## 2026-09-16: TASK-05 listening verdict

User's verdict on the 4 sent pop.00012 reconstructions (N=2/8/32/512): "the clips for N=32 are
closer to those for N=128 and N=512 than to those for N=8. In other words, there's a small
degradation after N=32, but the drop between N=8 and N=32 is dramatic."

Confirms and sharpens the metric-based reading: by ear, N=32 already sounds like it belongs to
the collapsed family (with N=128/512), not the near-perfect family (with N=2/8). The audible
category boundary lines up exactly with the metric-located N=8-to-N=32 transition. The
N=32-to-512 metric decline (waveform correlation 0.469 -> 0.019, >20x) is real numerically but
comparatively minor perceptually. Recorded in notes/DECISIONS.md#task05-gate3-scaling and
notes/MEMORY.md.

## 2026-09-16: TASK-06 N=32 compute-controlled micro-ablation

Hypothesis: TASK-05's fixed-20000-step-budget design confounds "N grows" with "per-example
exposure shrinks." Does N=32 recover if given N=8-equivalent (or more) per-example exposure,
holding N fixed at 32?

Checked TASK-05's checkpoint save code first -- confirmed it stores only `{"step",
"model_state_dict"}`, no optimizer/RNG state, so a true resume from the 8000-step checkpoint is
impossible. Used same-seed deterministic replay instead (documented in the new script's
docstring) and reused TASK-05's actual N=32 report unchanged for the "20k" row rather than
retraining (it stopped at step 8000 via patience, not the full 20000 -- noted explicitly).

Benchmarked raw per-step cost before committing to the multi-hour run: ~38.5ms/step
training-only, wandb.log() overhead ~0.09ms/call (negligible). Ran a --budget 500 smoke test
first (temporarily added to argparse choices, reverted after) -- passed cleanly, 55.34s, correct
report.json shape (checkpoint_history len=5, right metric keys). Cleaned up smoke artifacts before
the real run.

Disabled early stopping (patience) entirely for the 80k/320k runs -- TASK-05's own 5000-step
patience window, reused naively, would likely have killed these runs early during one of the
large turbulent dips this pipeline consistently shows at scale (confirmed this concern was
justified: both runs show >10x MSE swings within a few thousand steps throughout, e.g. 320k step
128000 mse=11.4 -> step 132000 mse=305.8).

Commands:
```
uv run python -u scripts/train_gate3_n32_compute_ablation.py --budget 80000 --wandb-mode online
uv run python -u scripts/train_gate3_n32_compute_ablation.py --budget 320000 --wandb-mode online
uv run python scripts/plot_gate3_n32_compute_ablation.py
uv run ruff check .; uv run pytest -q   # 133 passed
```
Launched as one chained background job (`... --budget 80000 ... && ... --budget 320000 ...`),
monitored via persistent Monitor + periodic ScheduleWakeup rather than continuous polling. 80k:
6257s (104 min). 320k: 24249s (6.7h). Total wall time both runs: ~8.4 hours.

Results (best-by-MSE checkpoint per budget, mean across all 32 examples):
```
             best step  reps/example  mse      cosine  norm    var     wav_corr  stft_dist
20k (TASK-05,  3000       94          12.881   0.516   0.499   0.177   0.469*    1.830*
 patience@8000)
80k            57000      1781        10.763   0.609   0.644   0.377   0.629     1.701
320k           88000      2750        10.252   0.633   0.656   0.403   0.666     1.588
N=8 (ref)       4750       594        0.078    0.998   1.003   1.008   0.997*    0.238*
```
(*20k/N=8 wav_corr/stft computed on fixed 2-track subset only, not all N -- TASK-05 didn't decode
audio for every example every checkpoint.)

Key finding: 80k's best checkpoint ALREADY exceeds N=8's needed per-example exposure by 3x (1781
vs. 594 reps), 320k by 4.6x (2750 vs. 594) -- and N=32 still isn't close to N=8's quality. So
"just needs more steps" is not the right story, even though more steps clearly helped
(0.469->0.629->0.666 wav_corr).

Per-example ranking (via new `rank_examples_by_metric`, sorted worst-to-best):
- Worst at 80k: classical.00003 (0.315). Worst at 320k: classical.00003 again (0.276, WORSE
  despite 4x compute).
- Best at 80k: pop.00002 (0.875). Best at 320k: pop.00002 again (0.897, improved).
- Mean/median rose substantially 80k->320k; worst-case flat-to-down. Compute helps the easy
  examples, not the hard one. This is the strongest evidence in the whole write-up for
  "interference/capacity" over "just needs more compute."

320k's own trajectory: best-by-MSE checkpoint found at step 88000 (right after the 80k
endpoint), then NO new global best for the remaining 232000 steps, despite several close local
peaks (steps 188000/252000/300000 all ~0.60 wav_corr, never beating 88000's 0.666). Noisy
plateau, not continued scaling.

Answered TASK-06's four required questions directly in notes/DECISIONS.md#task06-n32-compute-ablation
(reproduced there in full). Stopped there per the task's own explicit instruction not to propose
a next experiment.

Figure: artifacts/task06_n32_compute_ablation/per_example_waveform_correlation.png (3 panels,
20k/80k/320k, sorted worst-to-best, worst in red / best in green).

Full write-up in notes/DECISIONS.md#task06-n32-compute-ablation and notes/MEMORY.md.

## 2026-09-17: TASK-06 listening verdict

User's verdict on the sent WAVs (80k best/worst, 320k best/worst, figure): "the best audio was
generated at 80K steps. The audio from 320K is generally worse."

This is the first metric/perception divergence in this investigation -- 320k's mean waveform
correlation (0.666) is higher than 80k's (0.629), but 320k sounds worse. Two candidate
explanations, not disambiguated: (1) the 320k WAVs sent were from its final-step checkpoint (step
320000, wav_corr_mean=0.228 in the middle of a trough), not its actual best-by-MSE checkpoint
(step 88000, wav_corr_mean=0.666) -- so the A/B comparison as heard may not have compared best-vs-
best at all; (2) waveform correlation may simply track perceptual quality less tightly in this
collapse-adjacent regime than in the near-perfect N=2/8 regime where TASK-04/05's listening
verdicts agreed with the metrics. Recorded in notes/DECISIONS.md#task06-n32-compute-ablation and
notes/MEMORY.md. Does not change the core conclusion (partial, plateauing recovery with
preferential per-example memorization) but flags 80k, not 320k, as this experiment's
best-sounding result, and flags checkpoint-selection-for-listening as a methodological gap worth
fixing in any future compute-ablation-style task (always export best-by-MSE audio at every
milestone, not just fixed-step milestones).

## 2026-09-17: TASK-07 DAC shift/polarity sensitivity diagnostic

Motivation: TASK-04's unplanned finding (trained MuQ checkpoint's cosine similarity collapsed
0.99998 -> 0.577 under a 2-sample/83us crop offset) never established whether the fragility is in
DAC's own target or specific to that one trained model. This task isolates DAC alone, no
training, to check before spending compute on a complex-STFT TASK-05 rerun.

First checked whether DAC's pre-quantizer output is reachable without a `transformers`
monkeypatch (task's stop condition #1). Read modeling_dac.py directly: `DacModel.encoder` and
`DacModel.quantizer` are already public nn.Module submodules; `DacModel.encode()` is just
`quantized_representation = self.encoder(input_values); self.quantizer(quantized_representation)`.
Verified this composition reproduces `encode()`'s output exactly (`torch.allclose` on
quantized_representation, `torch.equal` on audio_codes) before trusting it -- no monkeypatch
needed, first stop condition never triggered. Added `FrozenDacEncoder.encode_stages()` wrapping
this. n_codebooks measured from the loaded checkpoint's config = 32 (matches task's "32 stages"
language, not assumed).

Design decision on how to implement "shift the input by k samples": extract crops from a
padded buffer with 256 samples (=MAX_SHIFT) of genuine surrounding audio on each side (crop
starts chosen with much larger margin, e.g. 5.0s/15.0s into a 30s track), then a shift-by-k crop
is a literal re-slice of that buffer -- never torch.roll/wraparound, never zero-padded. Both
canonical and every shifted crop are full CROP_SAMPLES-length real-audio inputs -> same frame
count always -> 0 frames ever need dropping for invalidity (documented explicitly rather than
silently assumed, per the task's "say how many frames were dropped" requirement).

Commands:
```
uv run python -u scripts/analyze_dac_shift_sensitivity.py   # 27s total, well under budget
uv run ruff check .; uv run pytest -q   # 145 passed (12 new tests)
```

Results (mean cosine similarity to canonical, both directions averaged, no asymmetry >0.05
threshold ever triggered):
```
shift(samples)  DAC quantized  DAC pre-quant  DAC codebook-stage0  log-mel  MuQ
1               0.716          0.945          0.783                0.9994   0.9999
2               0.572          0.827          0.626                0.9991   0.9998
4               0.290          0.540          0.410                0.9987   0.9996
8               0.207          0.334          0.212                0.9984   0.9993
16              0.157          0.229          0.067                0.9981   0.9991
32              0.155          0.230          0.024                0.9975   0.9981
64              0.132          0.201          0.021                0.9959   0.9950
128             0.116          0.174          0.019                0.9915   0.9833
256             0.118          0.180          0.017                0.9802   0.9434
random baseline 0.030          0.036          0.0025               --       --
```
Polarity (x vs -x): quantized cosine=-0.435, pre-quant cosine=-0.557 (NEGATIVE -- anti-correlated,
not just dissimilar), codebook agreement 0.002/0.0003 (at/below random-pair floor).

Striking result: DAC's own target collapses toward the random-pair floor by 16-32 samples (a
small fraction of its own 320-sample/13.3ms frame hop), while log-mel and MuQ -- the two
candidate translator inputs -- stay >0.98/0.94 across the ENTIRE tested range up to 256 samples.
This directly rules out "the target is just tracking a shifted input" and confirms the ambiguity-
set concern from the task's Context section: a magnitude-only/phase-blind input cannot in
principle determine this target uniquely at sub-frame precision.

Per the task's explicit non-goal, did NOT draw a conclusion about whether to pursue a complex-STFT
rerun of TASK-05 -- reported the numbers only, left the decision to the user.

Full write-up in notes/DECISIONS.md#task07-dac-shift-sensitivity and notes/MEMORY.md. Figure:
artifacts/task07_dac_shift_sensitivity/shift_sensitivity.png.

## 2026-09-21: TASK-08 DAC latent-vs-audio shift diagnostic

Extends TASK-07: does the latent-space shift fragility TASK-07 found actually show up in the
reconstructed AUDIO, or does DAC's decoder smooth it back out? Same crop set, same constraint (no
training, no translator), decode step added.

First checked DAC's decode output length vs input: input 48000 samples -> decode output 47992
(8 samples shorter, measured not assumed). Both canonical and shifted reconstructions come out
this same length, so alignment logic doesn't need to handle a length mismatch between the two
sides of a comparison, only the shift-induced offset.

Key design decision: how to "correct for the shift before evaluating similarity" (task's explicit
requirement). Implemented `align_shifted_pair(a, b, shift)`: b's content started `shift` samples
later than a's in source-time, so crop a's head (or tail, for negative shift) and b's tail (or
head) by `shift` samples so the remaining overlap covers literally the same source-time span.
Wrote unit tests confirming this with a synthetic torch.arange signal shifted by a known amount --
aligned_a == aligned_b exactly when b really is a's content shifted by that amount.

Reused TASK-07's crop selection by calling select_stratified_tracks again (deterministic, same
manifest/split/tracks_per_genre) rather than hardcoding a track list -- then added a runtime
assertion that the resulting 40 crop_ids exactly match TASK-07's own report.json "used" list, to
make "reuse the exact crop list" a verified fact, not an assumption from "same code should give
same result."

Moved TASK-07 script's crop-loading helpers (load_padded_crops, slice_shifted, crop_id) into the
shared dac_shift_sensitivity_common.py module, parameterized by max_shift (1024 here vs TASK-07's
256) instead of a module constant -- this let TASK-08 reuse identical crop-extraction logic
without touching TASK-07's own script file (task explicitly says "do not modify any existing
experiment").

Ran into a moment of doubt during the raw-reference computation: the "reference comparison"
(shift raw audio, align, compare) came back EXACTLY 1.0/1.0/0.0 at every single shift including
1024 samples -- initially looked like a bug (why would nothing at all change?). Worked through it:
this is mathematically correct, not a bug -- align_shifted_pair, after cropping the non-overlapping
edge for a PURE integer-sample shift of the SAME underlying digital signal, returns literally
identical sample sequences on both sides (no interpolation, no resampling happened, just window
repositioning). This is actually a valuable, clean control: it proves the raw signal has zero
inherent ambiguity near a shifted crop boundary, so 100% of any DAC-reconstruction gap from 1.0 is
attributable to DAC's own encode-decode process. Documented this explicitly in DECISIONS.md so a
future reader doesn't "fix" it.

Commands:
```
uv run python -u scripts/analyze_dac_latent_vs_audio_shift.py   # 93s total
uv run ruff check .; uv run pytest -q   # 159 passed (26 new tests)
```

Headline result (mean across 40 crops, both directions):
```
shift    latent_cos   DAC_wav_corr   DAC_logmel_cos   DAC_stft_dist   raw_ref (all 3)
1        0.716        0.958          0.990            0.653           1.000/1.000/0.000
16       0.157        0.964          0.989            0.702           1.000/1.000/0.000
256      0.118        0.963          0.988            0.709           1.000/1.000/0.000
1024     0.087        0.963          0.988            0.712           1.000/1.000/0.000
```
Latent collapses ~8x (0.716->0.087) while DAC reconstruction waveform correlation barely moves
(0.958->0.963, actually slightly UP) and log-mel cosine barely moves (0.990->0.988). STFT distance
jumps immediately at shift=1 (0->0.65) then stays flat -- doesn't track the latent's gradual decay
either, just saturates instantly (spectral-convergence term is evidently very alignment-sensitive
even for genuinely tiny, perceptually negligible differences).

Correlation analysis (Pearson/Spearman, latent cosine vs each audio metric, 480 points):
waveform_correlation: r=0.203, rho=0.229. logmel_cosine: r=0.525, rho=0.346. stft_similarity:
r=0.742, rho=0.333. All weak-to-moderate Spearman, meaning latent similarity is a poor rank
predictor of audio similarity despite some of these Pearson numbers looking moderate (likely
driven by a few outlier crops, not a real monotonic relationship -- the Pearson/Spearman gap for
STFT similarity is the tell).

This is a clean, decisive Outcome A result per the task's own interpretation rules: "DAC latent
geometry is poorly aligned with perceptual similarity. The latent is a poor regression target even
though DAC reconstructs correctly." Strengthens the target-mismatch story from TASK-07 rather than
just repeating it -- now we know the extra latent fragility isn't just "different from the input's
own stability," it's also mostly NOT predictive of what the decoder actually produces.

Exported a small listening package (2 representative crops x 5 listening shifts) but have not
gotten a listening verdict from the user yet -- flagged as a blocker in NEXT.md, though the
quantitative evidence here is unusually multi-metric-consistent already (unlike some earlier tasks
in this project where perception and metrics diverged).

Full write-up in notes/DECISIONS.md#task08-dac-latent-vs-audio-shift and notes/MEMORY.md. Figures:
artifacts/task08_dac_latent_vs_audio_shift/figure{1,2,3}_*.png.

## 2026-09-22: TASK-09 DAC Optimal Transport geometry diagnostic

Tests a specific hypothesis from TASK-08's finding (latent collapses, audio doesn't): is the
mismatch caused by comparing latents pointwise (frame i vs frame i) instead of as unordered point
clouds? OT/Sinkhorn distance treats each 150x1024 latent as a point cloud and asks how much mass
must move, not requiring frame alignment.

Neither `geomloss` nor `ot` (POT) was installed. Task explicitly permits adding one (unlike
TASK-08's SI-SDR instruction) -- chose POT (`uv add pot`) over geomloss: lighter footprint, more
direct primitive (`empirical_sinkhorn_divergence` takes raw point clouds directly).

Prototyped on synthetic data first before committing to an implementation:
- `ot.sinkhorn2` (plain entropic OT cost): does NOT return 0 for identical point clouds (tested:
  ~14 for two copies of the same random cloud with a reasonable epsilon) -- an O(epsilon) entropy
  bias. Would make "shift=0" a nonzero anchor, awkward for comparing against MSE=0/cosine=1.
- `ot.bregman.empirical_sinkhorn_divergence` (debiased): confirmed returns exactly 0.0 for
  identical clouds. Used this instead.

Epsilon selection: tried a FIXED absolute epsilon derived from one synthetic trial first, but
real DAC latents have much larger scale (std~4.25, cost matrix entries ~2000-2500) than N(0,1)
synthetic test data (cost matrix entries ~2-3) -- a fixed epsilon would be wildly off-scale
depending on the crop. Switched to per-crop-relative epsilon: 0.1 x median(nonzero pairwise
sqeuclidean distances within that crop's own canonical point cloud). Tested frac=0.05 (didn't
converge on a real DAC latent pair, got a UserWarning), frac=0.1 (converged cleanly), frac=0.2
(also fine but more smoothing/bias) -- picked 0.1, documented as a single empirical check per the
task's "do not tune aggressively" instruction, not a sweep.

Quick real-latent trial (blues.00014, shift 1 vs 8 vs 64 vs 256 vs 1024) showed something
striking early: OT distance jumps sharply from shift=1 to shift=8, then stays almost completely
FLAT from shift=8 to shift=1024 (24350 -> 24388, <0.2% change) -- while MSE/cosine both continue
drifting gradually over that same long range. This pattern held up in the full 40-crop run too
(see Figure 1). Early sign this might land closer to Outcome C (OT roughly invariant beyond onset)
than a plain "moderately better" Outcome B -- ended up being a hybrid: true for the LONG-RANGE
part of the curve, not true for the initial SHORT-RANGE collapse (shifts 1-4), where OT falls just
as fast as MSE/cosine.

Efficient design: TASK-08's report.json already has audio-similarity numbers per (crop, shift
magnitude), averaged across +/- directions -- exactly the granularity TASK-09 needs. Skipped
DAC decode entirely; just loaded TASK-08's report and reused those numbers verbatim. Verified
crop set matches TASK-08's report at runtime (same pattern as TASK-08 verifying against TASK-07).
This made the whole run much faster than TASK-08 (41s vs 93s) since no decode/audio-metric
computation was needed at all, only DAC encode (for the latents) + Sinkhorn calls.

Commands:
```
uv add pot
uv run python -u scripts/analyze_dac_ot_geometry.py   # 40.7s total, 19.9s of which was OT calls
uv run ruff check .; uv run pytest -q   # 165 passed (6 new tests)
```

Headline result (|Spearman| correlation with audio similarity, 480 crop/shift points):
```
                waveform_corr   logmel_cosine   stft_distance
MSE             0.362           0.382           0.468
Cosine          0.229           0.346           0.333
OT (Sinkhorn)   0.521           0.546           0.625
```
OT wins on every single comparison, consistently, by 44-127% relative -- real and robust across
three independent audio metrics, not cherry-picked. But not dramatic (task's Outcome A example
was MSE=0.25/Cosine=0.30/OT=0.80 -- we got MSE=0.36/Cosine=0.23/OT=0.52, a much smaller gap).
Figure 4 (rank-agreement scatter) shows OT visually tracking the "most different" region better
than MSE/cosine, but still noisy everywhere -- no metric is a strong predictor in absolute terms.

Settled on "closest to Outcome B, with a partial Outcome-C saturation signature confined to the
long-range portion of the curve" as the honest characterization -- doesn't force-fit into a single
bucket, consistent with this project's established practice (TASK-05 did the same).

Answered all five required questions in notes/DECISIONS.md#task09-dac-ot-geometry. Stopped there
per the task's explicit instruction not to propose/run a follow-up.

Full write-up in notes/DECISIONS.md#task09-dac-ot-geometry and notes/MEMORY.md. Figures:
artifacts/task09_dac_ot_geometry/figure{1,2,3,4}_*.png.

## 2026-09-22: TASK-10 EnCodec shift-sensitivity replication

Tests whether DAC's collapse-but-audio-stable phenomenon is DAC-specific or general to codecs, by
repeating TASK-07+TASK-08 on EnCodec.

First surprise: the task file itself is truncated. Read it via the Read tool, last line printed
was "> Does EnCodec latent similarity" with no continuation. Checked via `tail -c 300` on the raw
file directly (not a Read-tool artifact) -- confirmed the file genuinely ends there, 500 lines,
mid-sentence. Every section before "Final Questions" is complete though (Objective through
Deliverables), so proceeded with full execution and flagged the truncation explicitly in the
write-up rather than fabricating the missing question text or refusing to proceed.

Checked EnCodec availability: `transformers.EncodecModel` already available (same pinned
transformers==4.55.0 DAC uses, no new dependency). Found `facebook/encodec_24khz` was already the
checkpoint used on a different git branch (`encodec-hifigan-experiment`, referenced in
notes/EXPERIMENT_SYNTHESIS.md) -- used the same model ID per the task's "same checkpoint already
used elsewhere in the project" instruction.

Probed EnCodec's structure before writing any wrapper code (mirroring the same discipline used for
DAC's encode_stages in TASK-07): confirmed `model.encoder(x)` -> continuous embeddings (128-dim),
`model.quantizer.encode(embeddings, bandwidth)` -> codes (num_quantizers, batch, frames),
`model.quantizer.decode(codes)` -> quantized continuous representation, `model.decoder(quantized)`
-> reconstructed audio -- and verified this manual path reproduces `model.encode()`/`.decode()`'s
top-level output exactly (torch.allclose/torch.equal) before trusting it, same verification
discipline as DAC's own encode_stages.

Key geometry facts checked directly, not assumed: frame_rate=75Hz (matches DAC exactly -- lucky,
no resampling needed for a fair per-shift comparison), num_quantizers=32 at max bandwidth 24.0kbps
(matches DAC's 32 exactly), codebook_size=1024 (matches DAC), but latent dim=128 (much smaller
than DAC's 1024) and decoder output length EXACTLY matches input (48000->48000, unlike DAC's
47992) -- documented this last one explicitly since it's a real preprocessing difference that
could matter for anyone reusing align_shifted_pair.

Bug during first run: hardcoded `task07_report["dac_shift_sensitivity"][str(k)]` for DAC's latent
cosine comparison, but TASK-07 only ran shifts up to 256, while this task needs 0-1024 (TASK-08/09
extended the schedule). KeyError at shift=512. Fixed by switching to TASK-09's report instead,
which recomputed DAC's latent cosine with identical methodology across the FULL 0-1024 range
already -- better source for this comparison than TASK-07's now-superseded partial range anyway.

No dedicated test file for encodec_encoder.py -- checked first that dac_encoder.py and encoder.py
(MuQ) have none either (`ls tests | grep -i encoder` -> empty), confirming this is the project's
established convention for model-loading wrapper classes (verified once via a documented manual
check, not unit-tested), not an oversight to fix.

Commands:
```
uv run python -u scripts/analyze_encodec_shift_sensitivity.py   # 11.8s total, fastest of the 4
uv run ruff check .; uv run pytest -q   # 165 passed (no new tests -- see encoder-wrapper convention above)
```

Headline results:
```
                    shift=1   shift=16   shift=256   shift=1024
DAC latent cosine   0.716     0.157      0.118       0.087
EnCodec lat cosine  0.999     0.974      0.961       0.953

DAC wav corr        0.958     0.964      0.963       0.963
EnCodec wav corr     0.940     0.892      0.889       0.889
```
EnCodec's latent barely moves at all (stays >0.95 the whole way) -- a much cleaner Outcome-B-style
result than DAC's near-total latent collapse. But EnCodec's own audio reconstruction is actually
somewhat WORSE (lower, not just different) than DAC's throughout, not better -- despite the vastly
smoother latent. This directly complicates Outcome B's stated interpretation ("EnCodec becomes a
much more attractive target representation") -- attractive for latent-matching smoothness, yes,
but that didn't buy more shift-robust audio here.

Latent-to-audio correlation (Spearman): EnCodec's numbers (0.36/0.60/-0.71) are all noticeably
STRONGER than DAC's (0.23/0.35/-0.33) despite EnCodec's latent varying over a much narrower range
-- the little bit of movement EnCodec's latent does show is proportionally more informative than
DAC's much larger movement. An Outcome-C flavor layered on top of the Outcome-B-flavored raw
number.

Dug one level deeper (not explicitly required, but seemed important for interpreting the above):
checked EnCodec's DISCRETE codebook agreement (stage0 + mean-32-stage) alongside the continuous
latent cosine. Codebook agreement collapses about as fast as DAC's own codebook collapse (TASK-07
numbers) -- meaning the smooth CONTINUOUS latent isn't because EnCodec's tokenization is
inherently more shift-robust; the discrete assignments reshuffle just as readily. The smoothness
must be a property of the continuous embedding geometry itself (128-dim summed-codebook-vectors
vs DAC's 1024-dim), most likely a dimensionality/averaging effect. Documented this as Result 4,
a mechanistic note not explicitly asked for but directly relevant to interpreting Results 1 and 3
together honestly.

Settled on "none of Outcome A/B/C fits cleanly, closest to a combination of B and C, with a
complication neither anticipates" -- again not forcing a single bucket, consistent with this
project's established practice across TASK-05/09.

Answered the two legible final questions; flagged the file truncation clearly rather than guessing
past it. Per the task's explicit instruction, proposed no follow-up experiment.

Full write-up in notes/DECISIONS.md#task10-encodec-shift-sensitivity and notes/MEMORY.md. Figures:
artifacts/task10_encodec_shift_sensitivity/figure{1,2,3}_*.png.

## 2026-09-22: TASK-11 EnCodec continuous-latent controlled scaling probe

First TRAINING task since TASK-06 in this line -- the direct causal test of everything TASK-07-10
found: does swapping only the target codec (DAC -> EnCodec) move TASK-05's sharp N=8->N=32
collapse? Recorded git status at start per the task's explicit instruction first: clean except
tasks/TASK-11.md itself untracked -- TASK-09/10 were already committed (b54418f) from the prior
turn, contrary to the task text's own assumption that they were still uncommitted. Noted this
discrepancy rather than silently trusting the task's stale premise.

Checked `SpectrogramToDacTranslator`'s `dac_channels` param -- already parameterized (not
hardcoded to 1024), so `dac_channels=128` for EnCodec is a one-line change, no architecture
surgery needed. `TemporalConvNet`'s receptive field (13 frames) is independent of channel count,
confirmed from its own formula. Computed exact param counts: DAC version 15,863,808 (matches
TASK-05's own stored value exactly), EnCodec version 262,656 -- 98.3% FEWER, not more. Important:
this rules out "EnCodec just got a bigger network" as a confound before even running anything.

Key design decision: reuse `evaluate_prediction` from gate3_spectrogram_common.py UNCHANGED for
EnCodec too. Checked its source first -- it only ever calls `.decode(...)` on its `dac` parameter
(duck-typed), and FrozenEncodecEncoder.decode (built in TASK-10) already mirrors
FrozenDacEncoder.decode's exact interface. Wrote a dedicated test
(test_evaluate_prediction_accepts_encodec_encoder_via_duck_typing) to verify this BEFORE writing
the training script, not after -- this was the single biggest reuse risk in the whole task and
worth confirming early.

Built load_examples_from_manifest_encodec: reuses TASK-05's own load_examples_from_manifest for
the mel input (unmodified), computes EnCodec target fresh from the manifest's own
path/crop_start_seconds/crop_duration_seconds columns (dense cache only ever had DAC targets).

First smoke test (--levels 2, MAX_STEPS=100 override) crashed immediately:
"RuntimeError: Inference tensors cannot be saved for backward." FrozenEncodecEncoder.encode runs
under @torch.inference_mode(), so its raw output can't be used as an autograd target later even
though no grad is needed ON the target itself -- PyTorch tracks "inference tensor" as a property
of the tensor's origin, not the current context. Fixed with .clone() right after transpose(1,2),
per PyTorch's own error message suggestion. TASK-05 never hit this since its DAC targets load
from a torch.save'd cache file (plain tensors by the time training sees them).

Second smoke test (after the clone fix) ran clean -- but a startup sanity check revealed something
worth double-checking: EnCodec's own oracle wav_corr was 0.897 while a freshly-added DAC oracle
metric (original vs DAC's own encode-decode, never computed anywhere in this project before) came
back at only 0.47 -- looked broken for a supposedly high-fidelity codec. Investigated by hand:
searched lag -30..+30 for the best Pearson correlation between original and DAC's own oracle
reconstruction. Found best_lag=8, corr=0.973 (vs lag-0's 0.366) -- DAC's decoder has a genuine
fixed 8-sample GROUP DELAY relative to its input, not just the already-known 8-sample-shorter
output length. Checked EnCodec the same way: best_lag=0, no delay. This was a real bug in my own
new oracle-comparison code (naive first-N-samples truncation), not a DAC defect -- fixed by adding
find_best_lag_and_align (cross-correlation search + alignment) and using it for both codecs'
oracle metrics. Unit-tested with a synthetic 8-sample-shifted signal to confirm the search
recovers the exact known shift.

Ran the REAL training (all three levels, background job via nohup+disown, monitored via
persistent Monitor with ~30 print-line granularity, no polling):
```
uv run python -u scripts/train_gate3_encodec_scale_wandb.py --levels 2,8,32 --wandb-mode online
```
N=2: 15250 steps (patience), 96.8s -- near-perfect immediately (cosine 1.0 by step ~4000).
N=8: 20000 steps (max_steps, no early stop), 126.3s -- near-perfect (cosine 0.999).
N=32: 20000 steps (patience triggered exactly at the boundary), 127.0s -- the decisive run.
Total: ~5.8 minutes for all three levels combined -- much faster than TASK-05's DAC equivalents
(660+716+592s ~ 33 min for the same three N), consistent with EnCodec's much smaller 128-dim
target and correspondingly tiny 262K-param translator.

N=32 trajectory: climbed steadily from wav_corr~0.06 (step 1000) to a peak ~0.77 (step 16000),
then a sharp dip to ~0.18 (step 17000, same turbulence pattern as every large/long run in this
project since TASK-01), ending at 0.378 at step 20000 (final). Best-by-MSE checkpoint (step
15000) captured wav_corr=0.7235, well before the late-run dip -- exactly why the script tracks
best-by-MSE AND best-by-wav_corr separately and never trusts the final checkpoint automatically,
per this task's own explicit TASK-06-derived instruction.

Headline numbers (best-by-MSE checkpoint):
```
              N=2      N=8      N=32
DAC cosine    1.0000   0.9980   0.5161
EnCodec cos   1.0000   0.9991   0.9879

DAC wav_corr  1.0000   0.9970   0.4689
EnCodec wc    1.0000   0.9819   0.7235   (pred vs target-decode, both codecs' own metric)

DAC oracle    0.9612   0.9665   0.9690   (orig vs codec's own encode-decode, lag-aligned)
EnCodec orcl  0.8975   0.9043   0.9140
```
Oracle-relative at N=32: EnCodec recovers 0.7235/0.9140 = 79.2% of its own ceiling; DAC recovers
0.4689/0.9690 = 48.4% of its own ceiling. This normalized comparison is the load-bearing number
the task explicitly asked for ("do not compare raw EnCodec wav_corr directly with DAC without
also presenting the codec oracle") -- confirms the raw-number gap isn't just "EnCodec is an
easier codec to reconstruct through in general."

Figure 2 (DAC vs EnCodec scaling, 3 panels) is the single clearest visual in this whole task:
DAC's three curves (cosine, wav_corr, variance ratio) all crash sharply between N=8 and N=32;
EnCodec's decline over the same range is comparatively mild on every one. Figure 4 (per-example
N=32) shows EnCodec's latent cosine in a tight 0.98-0.99 band across ALL 32 examples (vs DAC's
0.43-0.68 spread) -- uniform lift, not an easier-subset effect -- but EnCodec's own per-example
AUDIO correlation still spans 0.37-0.98 despite that uniform latent fit, an interesting residual
finding not explicitly asked for but worth flagging (decoder-side sensitivity, not translator-side
unevenness).

Outcome classification: closest to Outcome B (substantial-but-partial improvement, target
geometry + optimization/capacity/interference likely coexist) -- not Outcome A (EnCodec N=32 is
well below its own oracle and its own N=8 level) and not Outcome C (the two codecs' trajectories
are clearly different in magnitude, no shared sharp transition).

Per the task's explicit instructions: did NOT extend to N=128/512, did NOT propose a follow-up
experiment, did NOT commit anything. Reported exact final git status (staged/unstaged/untracked)
in notes/NEXT.md as required.

Full write-up in notes/DECISIONS.md#task11-encodec-controlled-scaling and notes/MEMORY.md.
Figures: artifacts/task11_gate3_encodec_scale/figure{1,2,3,4}_*.png.

## 2026-09-22: TASK-12 EnCodec N=32 compute ablation

Extends TASK-11: is the residual N=32 gap (79.2% oracle-relative recovery at 20k steps)
compute-limited? Directly mirrors TASK-06's DAC compute ablation design, substituting EnCodec.

Reproducibility checks first, per the task's explicit instructions: recorded git status (clean
except tasks/TASK-12.md itself -- confirmed TASK-09/10 were ALREADY committed from a prior turn,
contradicting the task text's own stale assumption, same situation as TASK-11 had with TASK-09/10),
recorded commit hash (79805b4...), confirmed TASK-11's report.json/checkpoints/WAVs all present,
checksummed the N=32 manifest before touching anything.

Wrote train_gate3_encodec_n32_compute_ablation.py as a close mirror of TASK-06's
train_gate3_n32_compute_ablation.py, substituting: FrozenEncodecEncoder for FrozenDacEncoder,
gate3_encodec_common.load_examples_from_manifest_encodec for load_examples_from_manifest,
ex.target for ex.dac, dac_channels=128 for the translator. Reused evaluate_prediction/
aggregate_metrics_across_examples/rank_examples_by_metric UNCHANGED (same duck-typing trick
verified in TASK-11 -- didn't need to re-verify, TASK-11's own test already covers it). Added
best_by_wav_corr checkpoint tracking (TASK-11 pattern) plus explicit final-vs-best degradation
check per TASK-12's own requirement.

Smoke test (--budget 500 temporarily added to choices, --wandb-mode disabled): 10.18s, clean,
correct schema. Reverted the temporary choices/val_every overrides before the real run.

Launched the real run as one chained background command (80k then 320k, same nohup+disown+Monitor
pattern as every prior long run in this project):
```
nohup bash -c "uv run python -u scripts/train_gate3_encodec_n32_compute_ablation.py --budget 80000 --wandb-mode online && uv run python -u scripts/train_gate3_encodec_n32_compute_ablation.py --budget 320000 --wandb-mode online" > /tmp/task12_real.log 2>&1 &
```
80k: 570.5s (~9.5 min). 320k: 2083.1s (~34.7 min). Total ~44 min for both -- dramatically faster
than TASK-06's DAC equivalent (320k alone took ~6.7 hours for DAC), consistent with EnCodec's
much smaller 262,656-param translator (vs DAC's 15.9M).

Confirmed determinism throughout monitoring: the 80k and 320k runs' logged per-checkpoint values
matched TASK-11's own 20k run EXACTLY at every shared step (e.g. step 20000: both showed
wav_corr_mean=0.3554-ish and the identical best/median/worst example IDs) -- same seed, same
deterministic alternating-presentation order, confirms nothing was accidentally randomized.

Headline results (best-by-MSE checkpoint):
```
budget   best_step   wav_corr   oracle_rel   var_ratio
20k      15000       0.7235     79.2%        0.650
80k      75000       0.8040     88.0%        0.668
320k     300000      0.8470     92.7%        0.738
```
Classic diminishing returns: +8.05pp for the first 4x compute jump, only +4.30pp for the second
4x jump. Norm ratio stayed flat near 1.0 throughout (0.985-0.993) -- confirms TASK-11's "no norm
collapse" finding is compute-independent, not just a 20k-budget artifact.

One severe instability event during the 320k run: step 268000 showed wav_corr crashing to 0.35
(cosine to 0.94, mse spiking to 4.16) from a stable ~0.84 the checkpoint before -- fully recovered
by the very next checkpoint (step 272000, only 4000 steps later). Checked the checkpoint_history
directly to confirm this was a single-point transient, not a multi-checkpoint degradation, and
that it didn't affect best-checkpoint selection (actual best came later, at step 300000).

Computed the required per-example analyses with scipy (spearmanr for rank stability, pearsonr for
gain-vs-starting-point):
- Rank stability: spearman 0.964 (20k vs 80k), 0.973 (80k vs 320k), 0.944 (20k vs 320k) -- high
  but not perfect.
- Gain vs starting point: pearson r=-0.846 (20k->80k) and r=-0.839 (80k->320k) -- STRONG negative
  correlation, meaning examples that started worse gained more. An equalizing, decelerating
  effect (mean gain shrinks: 0.0805 -> 0.0430; std also shrinks: 0.054 -> 0.029).

Unplanned but important finding while doing the latent-vs-audio analysis: pooled all 96 points
(3 budgets x 32 examples) and computed Spearman rho between each latent metric and waveform
correlation. Cosine similarity: rho=0.605. Latent MSE: rho=0.199 (weak). VARIANCE RATIO: rho=0.819
-- much stronger than either. This makes sense in retrospect: cosine similarity is already
squeezed into a tiny 0.988-0.992 range across all three budgets (ceiling effect), so its residual
variation carries little signal, while variance ratio has real headroom (0.650-0.738) and tracks
the actual audio outcome much more tightly. Flagged this as a methodologically useful finding for
future diagnostics on this pipeline, not just a TASK-12-specific footnote.

Built 5 figures (compute-response curve, marginal gains, rank-stability scatter x3, gain-vs-
starting-point scatter x2, latent-vs-audio scatter x2) via a new plot_gate3_encodec_n32_compute_ablation.py
script, read-only over the three reports.

Wrote the full TASK-12_RESULTS.md report in the EXACT structure the task specified (Executive
Summary through Final Git Status, 15 sections) -- this is a different deliverable format than
this project's usual DECISIONS.md entries, so wrote it as a dedicated file in the artifacts
directory rather than trying to force it into the DECISIONS.md format, then wrote a normal
DECISIONS.md summary entry linking to it.

Outcome classification: B (Partial Response) -- clear gains at 80k, smaller-but-real gains at
320k, meaningful residual gap (7.3pp oracle-relative, variance ratio still 0.262 short of 1.0).
Explicitly ruled out A (not strong/sustained enough), C (audio improved right alongside latent,
not latent-only), D (changes are large, not minimal), and primarily-E (oscillation was present
but didn't prevent net progress).

Per the task's explicit instructions: no follow-up experiment proposed, not committed. Recorded
exact final git status in notes/NEXT.md as required.

Full write-up in notes/DECISIONS.md#task12-encodec-n32-compute-ablation, notes/MEMORY.md, and the
dedicated artifacts/task12_encodec_n32_compute_ablation/TASK-12_RESULTS.md report.

---

## 2026-10-16 -- TASK-15: MuQ -> BigVGAN-v2 mel -> BigVGAN-v2 memorization (N=2, N=8)

Interpreted `tasks/TASK-15.md` (which opens "Create the next task for...") as the task to execute.

Fetch: `scripts/fetch_bigvgan.py` -> `snapshot_download(nvidia/bigvgan_v2_24khz_100band_256x,
revision=c329ede9e9bbc100ddf5c91e2330a61921262370)`, code + config.json + bigvgan_generator.pt only
(skipped 2x 1.4 GB discriminator/optimizer states and the *_3msteps checkpoint). Generator sha256
6f9c5715...76f5. Read config.json, meldataset.py, bigvgan.py before importing. Deps already present
(matplotlib, scipy, librosa 1.0.0, huggingface_hub 0.36.2) -- no pyproject/uv.lock change.

Official mel: sr 24000, num_mels 100, n_fft 1024, hop 256, win 1024, fmin 0, fmax None,
librosa Slaney mel, Hann, reflect pad (n_fft-hop)//2=384, center=False,
sqrt(re^2+im^2+1e-9), log(clamp 1e-5). Training path peak-normalizes (normalize(audio)*0.95);
README inference snippet doesn't. Used the training path.

Geometry probe (MPS, BigVGAN 112,414,512 params after remove_weight_norm):
  L=48000 -> mel 187 frames -> wav 47872;  L=47872 -> 187 -> 47872;  L=48128 -> 188 -> 48128.
  MuQ: 50 frames for all three. -> crop 47872 samples (187 hops), interpolate 50 -> 187.

Oracle (8 crops): resynth mel L1 0.109, STFT 0.922; waveform corr to original 0.138 with lag
search pinned at +-32 on half the crops (phase not in mel -- not a defect). Unrelated-pair
baseline: mel L1 1.322, STFT 2.794.

Test failure: official_mel on 1 hop (256 samples) -> torch reflect pad error (pad 384 > input).
Test bug, not pipeline; test now uses >= 2 hops.

N=2 run 1: met all criteria steps 3250-4750 (mel L1 ~0.024, pred-vs-oracle corr ~0.97), then
loss 0.025 -> 0.41 at step 5000 (final) -> both crops FAIL on final weights. 63 steps > 0.1 loss
after step 1000. Adam + L1 + constant LR. Added best-by-eval-mel-L1 checkpoint (train = eval
crops) alongside final; deleted run 1's local outputs and reran -- bit-identical trajectory.
Best step 4750: both PASS.

N=8: best step 4500, 6/8 PASS; hiphop.00001 (0.898) and jazz.00000 (0.888) miss only
pred-vs-oracle corr >= 0.90. Mean corr trajectory 0.27 (1k) -> 0.76 (3k) -> 0.92 (4.5k) -- still
rising. Not extended (post-hoc budget change). Final step 0.042 mel L1, corr 0.919.

Checked n8/jazz.00000_mel_comparison.png by eye: pred indistinguishable from target, flat diff
map; shuffled panel clearly the paired crop's mel.

Runs: W&B seni7345 (N=2), xl3nmbjt (N=8). Train time 38 s / 69 s on MPS.

2026-10-16 (later): user listening verdict -- success at N=2 and N=8, including the two N=8
crops below the 0.90 pred-vs-oracle waveform-corr bar (hiphop.00001 0.898, jazz.00000 0.888).
That bar was stricter than perceptual equivalence. TASK-15 closed: viable at N=2 and N=8.

---

## 2026-10-19 -- TASK-16: audit corrections + N=2 rerun

MuQ geometry read from the loaded model: mel_stft.spectrogram hop 240, center True, pad 0;
Conv2dSubsampling blocks: conv1 k3 s(2,2) p1, conv2 k3 s1 p1, shortcut conv3 = conv1 geometry
(time axis = last dim). Composed: token i -> mel frame 4i -> sample 960i. (MelSpectrogram has no
`.center`; it lives on `.spectrogram`.)

Test failure: explicit interpolation vs F.interpolate differed by 1.1e-5 (float rounding of
50/187 inside PyTorch) -> atol 1e-4.

Regression: TASK-15 config with switches off, after the refactor -> loss history identical over
5,000 steps, metrics identical. Kept as artifacts/task16_bigvgan_corrected/task15_baseline_rerun.

Corrected run: best 4750 (same as TASK-15), final step 5000 spiked again (0.195). Clamp: 288/440 of
18,700 bins. Last 3 mel frames clamp to token 49.

Audit trace updated to the corrected pipeline; pre-fix files archived with _task15 /
_pre_task16 suffixes.
