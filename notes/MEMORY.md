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

## Cache size (measured, Gate 2, 2026-09-04)

- float32: ~102,399 bytes/audio-second at 1024 channels, 25 fps.
- float16: ~51,200 bytes/audio-second (half, as expected).
- float16 quantization error vs float32: max abs error 0.0074 across all probes, well under the
  documented tolerance of 0.05 (`feature_contract.lowprecision_max_abs_error`).

## Validated design choices

- See [[muq-transformers-compat]], [[gtzan-source-and-defects]], [[decoder-upsample-placeholder]],
  [[gate2-cache-format]], [[gate3-crop-alignment]], [[gate3-loss-configs]],
  [[gate4-mps-resume-tolerance]], [[gate4-shuffled-time-finding]].

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
