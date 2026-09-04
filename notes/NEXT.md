# NEXT

## Current gate

Gate 1 (Data Integrity) complete. Gate 0 (Environment Doctor) complete. Gate 2 (Feature
Contract) not started.

## Exact last successful commands

```bash
uv run python -m muq_inversion.cli doctor --config configs/smoke.yaml
uv run python -m muq_inversion.cli prepare-data --config configs/smoke.yaml
uv run pytest -q
uv run ruff check src/
```

All four passed on 2026-09-04 (macOS 26.6.2, M4 Max, device=mps).

## Run/checkpoint path

No `runs/` directories yet -- Gate 0/1 don't train anything. Data artifacts:
- `data/raw/gtzan/` (git-ignored, extracted GTZAN audio, marker file `.extracted`)
- `data/manifests/gtzan_manifest_smoke.csv` (tracked, 1000 rows, 793/94/113 train/val/test)

## Next command

Implement Gate 2 (Feature Contract):
```bash
uv run python -m muq_inversion.cli extract --config configs/pilot.yaml
```
This does not exist yet -- `cli.py` currently only has `doctor` and `prepare-data` subcommands,
scoped intentionally per this session's task. Building `extract` requires:
- A `cache.py` module implementing content-addressed caching (checksum + MuQ checkpoint +
  preprocessing config + representation name + dtype), atomic writes, per CLAUDE.md's Resource
  Discipline section.
- Extracting MuQ `last_hidden_state` for several real GTZAN clips of varying duration, checking
  frame-rate consistency (should match the doctor-measured 25.0 Hz / 960 samples-per-frame from
  [[MEMORY.md]]) and alignment.
- A round-trip test: cache a tensor, reload it, assert equality within dtype tolerance.

## Blockers

None currently. `configs/pilot.yaml` is presently identical to `configs/smoke.yaml` except for
`name:` -- it should probably diverge (e.g. subset size, crop length) once Gate 3/4 introduce
training-specific config fields; not needed yet.

## Expected verification criteria for Gate 2

Per CLAUDE.md: confirm frame-rate consistency and alignment across several durations; round-trip
cached tensors and verify equality within dtype tolerance. Do not proceed to Gate 3 (tiny-set
overfit) until this passes.

## Notes for whoever resumes this

- The `transformers==4.55.0` pin in `pyproject.toml`/`uv.lock` is load-bearing -- see
  [[DECISIONS.md#muq-transformers-compat]]. Do not let a routine `uv add` bump it past `<5.0`
  without re-running Gate 0.
- `decoder.upsample_rates` in the configs is a Gate-0-only placeholder, not a real design
  decision -- see [[DECISIONS.md#decoder-upsample-placeholder]].
- The GTZAN split is checksum-deduplicated but not artist-aware -- see
  [[DECISIONS.md#gtzan-source-and-defects]].
