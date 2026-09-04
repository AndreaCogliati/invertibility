# NEXT

## Current gate

Gates 0-3 complete (Environment Doctor, Data Integrity, Feature Contract, Tiny-Set Overfit).
Gate 4 (Short Pilot) not started.

## Exact last successful commands

```bash
uv run python -m muq_inversion.cli doctor --config configs/smoke.yaml
uv run python -m muq_inversion.cli prepare-data --config configs/smoke.yaml
uv run python -m muq_inversion.cli extract --config configs/smoke.yaml
uv run python -m muq_inversion.cli train --config configs/smoke.yaml
uv run pytest -q       # 33 passed
uv run ruff check .    # all checks passed
```

All passed on 2026-09-04 (macOS 26.6.2, M4 Max, device=mps).

## Run/checkpoint path

- `runs/gate3_tiny_overfit/checkpoints/{latest,step_000250,step_000500}.pt` (git-ignored) --
  step 500, 2 fixed clips, loss ratio 0.42 (PASS threshold 0.5).
- `artifacts/gate3_tiny_overfit/{pop.00012,disco.00049}_{reference,reconstruction,difference}.wav`
  (git-ignored).
- `data/cache/<key[:2]>/<key>.pt` (git-ignored, Gate 2's 30 probe entries; Gate 3 does NOT read
  from this cache -- it re-encodes each training crop live through frozen MuQ, since crop offsets
  are randomized per step and the Gate 2 cache only holds a few fixed-duration prefix crops).

## Next command

Implement Gate 4 (Short Pilot): train on a small deterministic subset (more than 4 clips),
validate checkpoint/resume equivalence, and run shuffled/constant-conditioning controls. Per
CLAUDE.md this requires:

- A `--resume auto` path in `cli.py`'s `train` command (or a new flag) that finds the newest
  valid checkpoint, quarantines a corrupt partial one if found, and restores optimizer/RNG state
  via `checkpoint.restore_rng_state` (already written, not yet wired into `train.py`'s loop) --
  then a resumed run's next N steps must be compared against an uninterrupted control run for
  numerical equivalence (CLAUDE.md Stop Conditions: "resumed training diverges from an
  uninterrupted control beyond expected nondeterminism").
- Real batching/dataset iteration over a subset of the train split (Gate 3 only ever alternates
  between 1-4 fixed clips) -- needs an actual `DataLoader`/sampler with a resumable cursor
  (CLAUDE.md Checkpointing: "data-sampler state or equivalent deterministic cursor").
  `sample_training_crop`/`fixed_validation_crop` in `train.py` generalize to this without changes,
  but track selection/iteration order does not yet.
- Shuffled-conditioning and constant/zero-conditioning controls (CLAUDE.md Evaluation and
  Controls) -- not implemented at all yet. These are Gate 4 requirements, deliberately not built
  during Gate 3 since Gate 3 only needed to prove the reconstruction path works at all.
- A validation split subset and validation-loss tracking (Gate 3 has no val/test evaluation --
  it only trains and reconstructs the same fixed clips it trained on, which is correct for an
  overfit test but not for a pilot).

## Blockers

None currently. `configs/pilot.yaml` is still identical to `configs/smoke.yaml` except `name:` --
Gate 4 should give it a real distinct identity (larger `training.num_overfit_tracks`-equivalent
subset size, `steps`, etc.) since it's meant for a larger, non-toy run.

## Expected verification criteria for Gate 4

Per CLAUDE.md: checkpoint/resume equivalence validated, real conditioning materially
outperforms shuffled/constant-conditioning controls (if controls sound similar to correctly-
conditioned output, assume conditioning collapse/leakage until disproven). Do not proceed to
Gate 5 until this passes.

## Notes for whoever resumes this

- The `transformers==4.55.0` pin in `pyproject.toml`/`uv.lock` is load-bearing -- see
  [[DECISIONS.md#muq-transformers-compat]]. Do not let a routine `uv add` bump it past `<5.0`
  without re-running Gate 0.
- Training/validation crop lengths must stay in units of `training.crop_frames` (multiples of the
  decoder's `total_upsample`, currently 960 samples/frame) -- see
  [[DECISIONS.md#gate3-crop-alignment]]. Do not introduce a raw-seconds crop-length config field
  without also handling the non-integral alignment case it would reintroduce.
- Mel/STFT loss configs are explicit, not copied from speech HiFi-GAN -- see
  [[DECISIONS.md#gate3-loss-configs]]; revisit if Gate 4/5 reconstructions show a specific
  frequency band or time resolution captured poorly.
- The GTZAN split is checksum-deduplicated but not artist-aware -- see
  [[DECISIONS.md#gtzan-source-and-defects]].
- Cache entries are single `.pt` files (`torch.save`), not `.npy`/`.npz` -- see
  [[DECISIONS.md#gate2-cache-format]]. Gate 3 training does not use this cache (see "Run/
  checkpoint path" above) -- only `extract` (Gate 2) populates/reads it so far.
- `checkpoint.restore_rng_state` exists but is not called anywhere yet -- it's for Gate 4's
  resume path.
