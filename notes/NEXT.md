# NEXT

## Current gate

BlueHive/W&B instrumentation for `tasks/TASK-03.md` -- code built and validated locally, actual
cluster run handed off to the user (this environment has no BlueHive access). **The
early-checkpoint-selection mechanism is now identified with direct evidence: regression-to-the-
mean, the same underlying phenomenon as the norm/variance collapse documented in TASK-01/TASK-02.**
Full reasoning in [[DECISIONS.md#task03-bluehive-wandb]]. Do not start a new
architecture/loss/objective experiment without further direction -- see "Next command" below for
what this evidence now points toward.

**Conclusions reached this session (full reasoning in
[[DECISIONS.md#task03-bluehive-wandb]]):**

1. **BlueHive access gap surfaced up front, not worked around silently**: no SSH/sbatch access
   from this environment. User confirmed scope: build everything runnable, validate locally, hand
   off the cluster execution. A real W&B API key was found in `~/.netrc` and used for genuine
   (not simulated) W&B logging.
2. Built CUDA/MPS-agnostic device selection (`logging_utils.select_device`, additive, MPS
   behavior unchanged), gradient diagnostics (`grad_diagnostics.py`), extended latent diagnostics
   (median norm, frame-to-frame variance), and a W&B wrapper (`wandb_logging.py`) -- 24 new tests.
3. `scripts/train_spectrogram_to_dac_wandb.py` reproduces TASK-02's scientific formulation
   exactly (manifest, architecture, loss, optimizer, step budget, checkpoint rule, eval protocol
   all unchanged); four documented, cost-driven instrumentation deviations (reduced-frequency
   STFT/waveform-correlation, audio logging, gradient logging; a `--steps` override reserved for
   infrastructure validation only).
4. **Validated with two real local runs (MPS, not BlueHive)**: a disabled-mode dry run and a real
   online run against the user's actual W&B account
   (`wandb.ai/andrea-cogliati-university-of-rochester/muq-invertibility`, run `gz3uffcf`), both
   clean. The online run's step-1000 checkpoint exactly matched TASK-02's own number
   (val_mse=16.442551), confirming determinism.
5. **Additional-investigation finding, evidence-based**: the first validation checkpoint (step
   1000) is the GLOBAL minimum validation MSE across the ENTIRE run in both TASK-01 (rank 1/60)
   and TASK-02 (rank 1/450) -- never beaten again. A trivial "predict the dataset mean DAC
   vector" baseline gets within 6-7% of that "best" MSE in both runs. Validation MSE and cosine
   similarity are anti-correlated across checkpoints (r=-0.72, excluding turbulence spikes) --
   pure-MSE checkpoint selection is structurally biased toward the least-informative, most-
   collapsed solution. **Conclusion: regression-to-the-mean, not optimization instability or
   validation noise -- and the same mechanism as the norm/variance-collapse findings, not a
   separate phenomenon.**
6. **BlueHive execution itself is NOT done** -- `bluehive/README.md` + `bluehive/run_task03.slurm`
   are ready, with `CHANGE_ME` placeholders for account/partition details this environment cannot
   know. The user must run this on the actual cluster and report back.

## Exact last successful commands

```bash
uv add wandb    # dependency added, uv.lock updated
uv run ruff check .    # all checks passed
uv run pytest -q       # 103 passed (79 baseline + 9 device-selection + 4 grad-diagnostics + 11 wandb-logging)
uv run python -u scripts/train_spectrogram_to_dac_wandb.py --steps 50 --wandb-mode disabled     # PASSED, no errors
uv run python -u scripts/train_spectrogram_to_dac_wandb.py --steps 2500 --wandb-mode online      # PASSED, real W&B run gz3uffcf, no errors
uv run python scripts/analyze_early_checkpoint_selection.py    # regression-to-the-mean evidence, both runs
```

2026-09-14, macOS, M4 Max, device=mps. Launched via `nohup ... &` + `disown` (unbuffered `-u`),
tracked by real PID and `Monitor`.

## Run/checkpoint path

- `src/muq_inversion/grad_diagnostics.py`, `wandb_logging.py`, extended `latent_diagnostics.py`,
  extended `logging_utils.py` -- new/modified diagnostics modules.
- `scripts/train_spectrogram_to_dac_wandb.py` -- the instrumented reproduction script (run this
  on BlueHive with default `--steps` for the real reproduction).
- `scripts/analyze_early_checkpoint_selection.py` -- the additional-investigation analysis;
  `artifacts/task03_wandb_repro/early_checkpoint_analysis.json` -- its output.
- `bluehive/README.md`, `bluehive/run_task03.slurm` -- the cluster handoff.
- `artifacts/task03_wandb_repro/report.json`, `config.json` -- from the two local validation runs
  (both labeled `is_smoke_test: true`; NOT the real reproduction numbers).
- Live W&B project: `wandb.ai/andrea-cogliati-university-of-rochester/muq-invertibility`
  (group `spectrogram-control`), run `gz3uffcf` from the 2500-step online validation.

## Next command

**Blocked on the user running the BlueHive job.** Once that's done:

1. Fill in `bluehive/run_task03.slurm`'s `CHANGE_ME` placeholders with real account/partition
   details and submit (`sbatch bluehive/run_task03.slurm`).
2. Report back wall-clock time, GPU type, and any errors so the reproduction-comparison table
   (M4 Max/TASK-02 vs. BlueHive) can be completed and the "does TASK-02 reproduce on different
   hardware" question actually answered.

**Independent of BlueHive**, the regression-to-the-mean finding (item 5 above) is itself now
actionable and doesn't need cluster access to pursue further:

- **A loss/target reformulation that doesn't reward the dataset-mean solution as cheaply** is now
  the most directly evidence-backed next direction across this entire investigation -- e.g.
  down-weighting or replacing plain MSE (per-frame relative error, a loss on normalized/
  whitened latents, or something that respects DAC's residual-VQ structure more directly, per
  `scripts/analyze_dac_manifold.py`'s codebook-agreement diagnostics).
- The architecture/receptive-field-capacity axis (flagged in [[muq-layer-sweep]]'s NEXT.md,
  never addressed by TASK-01/02/03) remains untested but is now a less urgent lead than the loss
  reformulation, since the mechanism identified this session is a loss-geometry property, not
  obviously a capacity limitation.

Do not start either without the user's go-ahead.

## Blockers

1. **BlueHive job not yet run** -- this environment cannot do it; the user must execute
   `bluehive/run_task03.slurm` (after filling in placeholders) and report results back.
2. **Listening verdicts for TASK-01's and TASK-02's 30-WAV packages** -- still pending (see
   [[DECISIONS.md#dense-spectrogram-positive-control]]).
3. **Decision needed from the user** on whether to pursue the loss/target-reformulation direction
   the regression-to-the-mean finding now points toward.
4. The still-pending conditioning-collapse diagnostic from `experiment/mel-hifigan-sanity` --
   untested, orthogonal to this branch's findings.
5. The `shuffled_time` finding from Gate 4 (see [[DECISIONS.md#gate4-shuffled-time-finding]]) --
   still dormant.
6. Listening verdicts for the 80-track scaling experiment, the full-GTZAN scaling experiment, and
   the MuQ layer-sweep were never given -- not blocking, worth noting if any comes up again.

## Expected verification criteria for Gate 5

Not yet reached, and not authorized to start.

## Notes for whoever resumes this

- **Read [[DECISIONS.md#task03-bluehive-wandb]] first.** It closes off "is this an optimization-
  instability or noise artifact" for the early-checkpoint phenomenon with direct evidence, and
  reframes the norm-collapse/over-smoothing/early-checkpoint findings from TASK-01/02 as three
  faces of one mechanism (regression-to-the-mean under plain MSE on a high-variance, weakly-
  conditioned target), not three separate open questions.
- `scripts/analyze_early_checkpoint_selection.py` is read-only and cheap (~seconds) -- rerun it
  against any future experiment's `report.json` (as long as it has a `training_curves` field) to
  get the same rank/baseline/correlation diagnostics without needing a fresh training run.
- The W&B integration (`wandb_logging.py`) is generic enough to reuse for any future training
  script in this project, not just this one -- `init_run`, `log_checkpoint_tables`, and
  `log_audio_examples` don't assume anything TASK-03-specific.
- Real credentials exist in `~/.netrc` for `api.wandb.ai` (account
  `andrea-cogliati-university-of-rochester`) -- any future script that calls `wandb.init()` will
  connect to the real account by default; use `mode="disabled"` for any future automated/CI-style
  test that shouldn't create real W&B runs.
- `bluehive/` is new infrastructure, never exercised end-to-end (no cluster access here) --
  treat `bluehive/run_task03.slurm` as a well-reasoned first draft, not a proven-working script,
  until the user actually runs it once and reports back.
