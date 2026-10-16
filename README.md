# muq-inversion

Proof-of-concept: can a small HiFi-GAN-style decoder invert MuQ's final time-resolved
representation (`last_hidden_state`) back into recognizable music audio? See `CLAUDE.md` for the
full experimental scope, gates, and reproducibility rules.

## Setup

```bash
uv sync
```

## Gates implemented so far

```bash
uv run python -m muq_inversion.cli doctor --config configs/smoke.yaml       # Gate 0
uv run python -m muq_inversion.cli prepare-data --config configs/smoke.yaml # Gate 1
uv run pytest -q
uv run ruff check .
```

## BigVGAN-v2 memorization test (TASK-15)

Uses NVIDIA's official `nvidia/bigvgan_v2_24khz_100band_256x` code and weights, fetched once at a
pinned revision into the git-ignored `third_party/` directory (no new Python dependencies):

```bash
uv run python scripts/fetch_bigvgan.py                             # ~450 MB, one-time
uv run python scripts/task15_bigvgan_oracle.py                     # positive control
uv run python scripts/train_task15_bigvgan_memorization.py --n 2   # then --n 8
```

`tests/test_bigvgan_common.py` skips its BigVGAN-dependent tests until the fetch has run. Results:
`artifacts/task15_bigvgan_memorization/TASK-15_RESULTS.md` (local) and
`notes/DECISIONS.md#task15-bigvgan-memorization`.

See `notes/NEXT.md` for current status and the next command to run.
