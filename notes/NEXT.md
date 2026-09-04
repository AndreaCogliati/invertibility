# NEXT

## Current gate

Gate 0 (Environment Doctor), Gate 1 (Data Integrity), and Gate 2 (Feature Contract) complete.
Gate 3 (Tiny-Set Overfit) not started.

## Exact last successful commands

```bash
uv run python -m muq_inversion.cli doctor --config configs/smoke.yaml
uv run python -m muq_inversion.cli prepare-data --config configs/smoke.yaml
uv run python -m muq_inversion.cli extract --config configs/smoke.yaml
uv run pytest -q       # 18 passed
uv run ruff check .    # all checks passed
```

All passed on 2026-09-04 (macOS 26.6.2, M4 Max, device=mps). `extract` was run twice to confirm
idempotent/resumable cache writes.

## Run/checkpoint path

No `runs/` directories yet -- Gates 0-2 don't train anything. Data artifacts:

- `data/raw/gtzan/` (git-ignored, extracted GTZAN audio, marker file `.extracted`)
- `data/manifests/gtzan_manifest_smoke.csv` (tracked, 1000 rows, 793/94/113 train/val/test)
- `data/cache/<key[:2]>/<key>.pt` (git-ignored, 30 entries from the Gate 2 probe: 3 tracks x 5
  durations x {float32, float16})

## Next command

Implement Gate 3 (Tiny-Set Overfit): train on 1-4 fixed clips and confirm reconstruction loss
decreases substantially. This does not exist yet -- `cli.py` currently only has `doctor`,
`prepare-data`, and `extract` subcommands. Building it requires, per CLAUDE.md:

- `train.py`: a training loop, reconstruction-only (mel + multi-resolution STFT losses per
  Decoder Design -- no adversarial/discriminator losses yet, those come after Gate 3 passes).
- `checkpoint.py`: atomic checkpointing with the full state list from CLAUDE.md's Checkpointing
  section (generator/optimizer/scheduler state, RNG states, epoch/step counters, resolved config,
  manifest hash, `uv.lock` hash, git commit + dirty status).
- Resolving [[DECISIONS.md#decoder-upsample-placeholder]]'s open question: real audio does not
  land on an exact samples-per-frame multiple (measured 960.0067 for full-length clips vs. an
  exact 960 for hand-picked crop lengths) -- Gate 3's data loader needs an explicit crop/pad
  alignment strategy between the waveform target and `num_frames * upsample_product`, not an
  assumption that they already match.
- Decide on a fixed segment/crop length for the "short aligned random crops" CLAUDE.md's Data
  Pipeline section calls for -- not chosen yet, no config field exists for it.
- Mel and multi-resolution STFT loss implementations belong in `models/losses.py` (currently
  doesn't exist).

## Blockers

None currently. `configs/pilot.yaml` is presently identical to `configs/smoke.yaml` except for
`name:` -- it should probably diverge (e.g. subset size, crop length) once Gate 3/4 introduce
training-specific config fields; not needed yet.

## Expected verification criteria for Gate 3

Per CLAUDE.md: train on 1-4 fixed clips, confirm reconstruction losses decrease substantially,
save reference/reconstruction/difference audio. If this gate fails, do not launch a full run
(i.e. do not proceed to Gate 4).

## Notes for whoever resumes this

- The `transformers==4.55.0` pin in `pyproject.toml`/`uv.lock` is load-bearing -- see
  [[DECISIONS.md#muq-transformers-compat]]. Do not let a routine `uv add` bump it past `<5.0`
  without re-running Gate 0.
- `decoder.upsample_rates` in the configs is still a placeholder, now with an important caveat
  from Gate 2 (non-integral samples-per-frame on real audio) -- see
  [[DECISIONS.md#decoder-upsample-placeholder]].
- The GTZAN split is checksum-deduplicated but not artist-aware -- see
  [[DECISIONS.md#gtzan-source-and-defects]].
- Cache entries are single `.pt` files (`torch.save`), not `.npy`/`.npz` -- see
  [[DECISIONS.md#gate2-cache-format]]. `feature_contract.lowprecision_max_abs_error` (0.05) is a
  round conservative bound, not derived from data in advance; measured float16 error so far is
  0.0074, well under it.
