# NEXT

## Current gate

Gates 0-3 complete and passed. Gate 4 (Short Pilot) implemented and run, but **NOT FULLY
PASSED**: checkpoint/resume equivalence passed; 2 of 3 conditioning controls
(`shuffled_across`, `constant_zero`) passed; `shuffled_time` did not (see
[[DECISIONS.md#gate4-shuffled-time-finding]]). Per CLAUDE.md Stop Conditions, do not proceed to
Gate 5 (full GTZAN pilot) until this is investigated further -- this is the single blocker.

## Exact last successful commands

```bash
uv run python -m muq_inversion.cli doctor --config configs/smoke.yaml
uv run python -m muq_inversion.cli prepare-data --config configs/smoke.yaml
uv run python -m muq_inversion.cli prepare-data --config configs/pilot.yaml   # separate manifest, same split
uv run python -m muq_inversion.cli extract --config configs/smoke.yaml
uv run python -m muq_inversion.cli train --config configs/smoke.yaml
uv run python -m muq_inversion.cli pilot --config configs/pilot.yaml         # Gate 4 -- see caveat above
uv run pytest -q       # 38 passed
uv run ruff check .    # all checks passed
```

All ran successfully (exit code 1 for `pilot` is expected/correct given the finding above -- the
command itself works, the *result* is the thing not yet resolved). 2026-09-04, macOS 26.6.2,
M4 Max, device=mps.

## Run/checkpoint path

- `runs/gate4_short_pilot/checkpoints/{latest,resume_test_checkpoint}.pt` (git-ignored) -- final
  run: 6 tracks, 400 steps, resume equivalence PASSED (max relative diff 0.063, tolerance 0.08).
- `artifacts/gate4_short_pilot/<track_id>_{reference,reconstruction}.wav` (git-ignored, 6 tracks).
- `runs/gate3_tiny_overfit/`, `artifacts/gate3_tiny_overfit/` still present from Gate 3.
- `data/manifests/gtzan_manifest_{smoke,pilot}.csv` both tracked (identical split, since both use
  `split_seed: 1234` -- separate files only because the manifest filename is keyed by config name).

## Next command

**Before Gate 5:** investigate the `shuffled_time` finding from
[[DECISIONS.md#gate4-shuffled-time-finding]]. Concretely, in order of cost:

1. Re-run `uv run python -m muq_inversion.cli pilot --config configs/pilot.yaml` with
   `pilot.steps` increased further (e.g. 800-1000) to see whether the `shuffled_time` margin
   improves with more training -- if it doesn't, that points away from "just needs more steps."
2. Listen to `artifacts/gate4_short_pilot/*_reconstruction.wav` vs `*_reference.wav` directly --
   objective loss numbers are diagnostics, not proof, per CLAUDE.md's Evaluation and Controls
   section ("Do not claim successful inversion solely from a low training loss").
3. If the pattern persists, consider whether `MiniGenerator`'s receptive field (currently just a
   couple of small residual blocks per upsampling stage, sized for a Gate-0 shape test, not
   tuned since) is large enough to exploit frame-level temporal alignment at all.

Once resolved (or explicitly accepted as a documented, re-checked-at-Gate-5 open question),
implement Gate 5 (GTZAN Pilot): train on the full committed split, save periodic fixed validation
reconstructions, report objective metrics. Requires:

- Batching beyond the current 1-clip-per-step rotation -- Gate 5 trains on the full train split,
  not a hand-picked handful of tracks.
- A real validation loop against the `val` split (Gates 3/4 only ever reconstruct clips they
  trained on).
- Periodic checkpointing/sample-saving on a schedule appropriate for a longer run (current
  `checkpoint_every`/`log_every` values are tuned for Gate 3/4's short runs).

## Blockers

The `shuffled_time` finding above. Not a code defect (implementation verified via
`tests/test_pilot.py`'s stub-encoder tests and three independent full pilot runs) -- an open
scientific question this project's Gate 4 was specifically designed to surface.

## Expected verification criteria for Gate 5

Per CLAUDE.md: train on the committed GTZAN split, save periodic fixed validation
reconstructions, report objective metrics and qualitative caveats.

## Notes for whoever resumes this

- The `transformers==4.55.0` pin in `pyproject.toml`/`uv.lock` is load-bearing -- see
  [[DECISIONS.md#muq-transformers-compat]]. Do not let a routine `uv add` bump it past `<5.0`
  without re-running Gate 0.
- Training/validation crop lengths must stay in units of `crop_frames` (multiples of 960 samples)
  -- see [[DECISIONS.md#gate3-crop-alignment]].
- Mel/STFT loss configs are explicit, not copied from speech HiFi-GAN -- see
  [[DECISIONS.md#gate3-loss-configs]].
- MPS training is not bit-exact reproducible even with identical seeds; resume-equivalence
  tolerances must be calibrated per run length, not copy-pasted -- see
  [[DECISIONS.md#gate4-mps-resume-tolerance]]. If `pilot.steps` changes, re-measure and adjust
  `pilot.resume_tolerance` rather than assuming the current 0.08 still holds.
- The GTZAN split is checksum-deduplicated but not artist-aware -- see
  [[DECISIONS.md#gtzan-source-and-defects]].
- Cache entries are single `.pt` files (`torch.save`), not `.npy`/`.npz` -- see
  [[DECISIONS.md#gate2-cache-format]]. Neither Gate 3 nor Gate 4 training reads this cache --
  only `extract` (Gate 2) populates/reads it so far; both training loops re-encode crops live.
