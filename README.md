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

See `notes/NEXT.md` for current status and the next command to run.
