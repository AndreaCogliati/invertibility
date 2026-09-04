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

## Working dependency versions (Apple Silicon, macOS 26.6.2, M4 Max, 2026-09-04)

- Python 3.12.12, torch 2.14.0, torchaudio 2.11.0, numpy 2.5.2.
- `transformers==4.55.0` pinned with `huggingface_hub>=0.34.0,<1.0` (see
  [[muq-transformers-compat]] for why this exact pin is required).
- `muq==0.1.0` (PyPI package `muq`, imports as `from muq import MuQ`).
- MPS backend available and selected by default (`torch.backends.mps.is_available() == True`).

## Stable paths and commands

- Doctor (Gate 0): `uv run python -m muq_inversion.cli doctor --config configs/smoke.yaml`
- Prepare data (Gate 1): `uv run python -m muq_inversion.cli prepare-data --config configs/smoke.yaml`
- Manifest output: `data/manifests/gtzan_manifest_<config-name>.csv` (tracked in git).
- Raw GTZAN audio lives at `data/raw/gtzan/genres/<genre>/<genre>.NNNNN.wav` (git-ignored).

## Validated design choices

- See [[muq-transformers-compat]], [[gtzan-source-and-defects]], [[decoder-upsample-placeholder]].

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
