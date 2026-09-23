# NEXT

## Current gate

EnCodec N=32 compute ablation (`tasks/TASK-12.md`) -- extends TASK-11 by testing whether its
residual N=32 gap (79.2% oracle-relative recovery at a 20k-step budget) is compute-limited.
Trained the identical setup independently to 80k and 320k steps (same seed/init, early stopping
disabled, mirroring TASK-06's DAC compute ablation exactly, substituting EnCodec).

**Result: Outcome B (Partial Response) -- compute helps substantially but with clearly
diminishing returns; a meaningful gap remains at 320k.** Oracle-relative recovery: 79.2% (20k) ->
88.0% (80k) -> 92.7% (320k). Marginal gain in waveform correlation: +8.05 points for 20k->80k (4x
compute) but only +4.30 points for 80k->320k (another 4x compute) -- roughly half the return for
the same additional compute. Norm ratio never collapsed at any budget; variance ratio climbed
steadily (0.650 -> 0.668 -> 0.738) but stayed 26 points short of 1.0 even at 320k. **Notable
methodological finding**: variance ratio (Spearman rho=0.819 vs. audio quality) predicts audio
quality far better than latent cosine similarity (rho=0.605) or latent MSE (rho=0.199), despite
cosine being the metric closest to its own ceiling. Per-example: compute preferentially benefits
examples that started worse (Pearson r=-0.85) while broadly preserving relative rank (Spearman
rho 0.94-0.97). One severe but transient instability spike occurred in the 320k run (step
268,000), fully recovered within one checkpoint interval, did not affect best-checkpoint
selection. Full reasoning, every required section (per this task's own template), and all figures
in `artifacts/task12_encodec_n32_compute_ablation/TASK-12_RESULTS.md`, summarized in
[[DECISIONS.md#task12-encodec-n32-compute-ablation]]. Per this task's own instruction, **no
follow-up experiment proposed or run.**

**Conclusions reached this session (full reasoning in the dedicated results report and
[[DECISIONS.md#task12-encodec-n32-compute-ablation]]):**

1. Verified TASK-11's outputs preserved and the exact N=32 configuration unchanged (manifest
   checksum, parameter count, architecture) before running anything -- per this task's explicit
   reproducibility-check requirement.
2. Reused TASK-11's 20k run unchanged as the baseline data point (per the task's "preferred
   training strategy": independent runs from the same seed/init, not one extended trajectory).
3. **The residual TASK-11 gap is partially, not primarily, compute-limited** -- 16x more compute
   recovered 13.5 additional percentage points of oracle-relative performance but did not close
   the gap, with each successive 4x compute increase buying roughly half the previous gain.
4. **Variance ratio, not latent cosine similarity, is the best available latent-space predictor of
   audio quality** in this regime -- worth using instead of cosine similarity in any future
   diagnostic on this pipeline.
5. **Compute preferentially benefits harder (lower-starting) examples** in absolute terms, an
   equalizing but decelerating effect, while largely preserving relative example ranking.
6. Both fixes carried over from TASK-11 (target `.clone()`, lag-aware DAC oracle policy) worked
   unchanged, no rediscovery needed.

## Exact last successful commands

```bash
uv run python -u scripts/train_gate3_encodec_n32_compute_ablation.py --budget 80000 --wandb-mode online
uv run python -u scripts/train_gate3_encodec_n32_compute_ablation.py --budget 320000 --wandb-mode online
uv run python scripts/plot_gate3_encodec_n32_compute_ablation.py   # five figures
uv run ruff check .; uv run pytest -q   # 177 passed (no new tests needed)
```

2026-09-22, macOS, M4 Max, device=mps. Launched via `nohup bash -c "... --budget 80000 && ...
--budget 320000" &` + `disown`, tracked by a persistent `Monitor`. Training time: 80k run 570.5s
(~9.5 min), 320k run 2083.1s (~34.7 min) -- far faster than TASK-06's DAC equivalent (~6.7 hours
for 320k), consistent with EnCodec's much smaller 262,656-parameter translator.

## Run/checkpoint path

- `data/manifests/gate3_scale_32.csv` (tracked, reused unchanged from TASK-05/06/11, checksum-
  verified before use: `4a65f3f79c3031594cd79570957e4e6248bfb306`).
- `runs/task12_encodec_n32_compute_ablation/budget_{80000,320000}/checkpoints/
  {best_by_mse,best_by_wav_corr,final}.pt` (git-ignored).
- `artifacts/task12_encodec_n32_compute_ablation/budget_{80000,320000}/report.json` -- full
  per-budget metrics (aggregate + per-example, best and final checkpoints); `.../budget_{N}/*.wav`
  -- listening WAVs (19 files for 80k, 29 for 320k; best/median/worst at the best checkpoint).
- `artifacts/task12_encodec_n32_compute_ablation/figure{1..5}_*.png` -- the five required
  compute-response/marginal-gain/rank-stability/latent-vs-audio figures.
- `artifacts/task12_encodec_n32_compute_ablation/TASK-12_RESULTS.md` -- the full report in this
  task's own required structure (Executive Summary through Final Git Status).
- `artifacts/task11_gate3_encodec_scale/n32/report.json` -- the reused 20k baseline (TASK-11's
  own completed run, unchanged).
- Two live W&B runs, group `gate3-encodec-n32-compute-ablation`: `wandb.ai/
  andrea-cogliati-university-of-rochester/muq-invertibility` (`encodec_n32_80k_*`,
  `encodec_n32_320k_*`).

## Next command

**Blocked on user direction.** Per this task's own instruction ("do not propose or run a new
experiment automatically"), no next experiment is proposed here. Candidate directions that would
need the user's go-ahead if raised later:

- Extending the compute-response curve further (640k, 1.28M steps) to see whether the diminishing-
  returns curve is approaching an asymptote near ~95-97% or a harder floor closer to 92.7% --
  explicitly out of this task's own scope, and the marginal-gain deceleration alone can't resolve
  which.
- Investigating whether the variance-ratio-vs-cosine-similarity predictive gap found here
  generalizes to DAC's own compute ablation (TASK-06) or is EnCodec-specific -- not checked.
- Repeating the 320k run with a different seed to check whether the step-268,000 instability spike
  is a general property of this optimization setup at long horizons or a one-off.
- A human listening pass on the 80k/320k packages -- not yet done, purely objective-metric
  conclusions so far.

Do not start any of these without the user's go-ahead.

## Blockers

1. **Decision needed from the user** on how to weigh TASK-07-12 together for any future direction
   (none of the six tasks requests one).
2. **Listening verdict for TASK-12's packages** (80k and 320k) -- not yet reviewed, nor TASK-11's
   or TASK-08's or TASK-10's.
3. **Listening verdicts** for TASK-01's and TASK-02's WAV packages -- still pending.
4. **BlueHive job not yet run** -- see [[task03-bluehive-wandb]], still pending.
5. The still-pending conditioning-collapse diagnostic from `experiment/mel-hifigan-sanity` --
   untested, orthogonal to this branch's findings.
6. The `shuffled_time` finding from Gate 4 (see [[DECISIONS.md#gate4-shuffled-time-finding]]) --
   still dormant.
7. Listening verdicts for the 80-track scaling experiment, the full-GTZAN scaling experiment, and
   the MuQ layer-sweep were never given -- not blocking, worth noting if any comes up again.

## Expected verification criteria for Gate 5

Not yet reached, and not authorized to start.

## Repository state at the end of this task (per TASK-12's explicit reporting requirement)

At task start: commit `79805b40007bca79ae415590960089cf520c712b`, clean tree except
`tasks/TASK-12.md` itself (untracked). TASK-11's outputs confirmed preserved.

At the end of this task:

```text
Changes not staged for commit:
    modified:   notes/DECISIONS.md
    modified:   notes/MEMORY.md
    modified:   notes/NEXT.md        (this file, being written)
    modified:   notes/SCRATCHPAD.md  (pending)

Untracked files:
    scripts/plot_gate3_encodec_n32_compute_ablation.py
    scripts/train_gate3_encodec_n32_compute_ablation.py
    tasks/TASK-12.md
```

**Not committed**, per this task's explicit "do not commit unless explicitly instructed" rule. No
existing tracked file was modified except the four `notes/*.md` files (additive appends/rewrites
only, consistent with every prior task in this project). `artifacts/
task12_encodec_n32_compute_ablation/` and `runs/task12_encodec_n32_compute_ablation/` are
git-ignored and do not appear in `git status`.

## Notes for whoever resumes this

- **Read `artifacts/task12_encodec_n32_compute_ablation/TASK-12_RESULTS.md` first** for the full
  required-structure report, then [[DECISIONS.md#task12-encodec-n32-compute-ablation]] for the
  project-notes-style summary.
- **Variance ratio > cosine similarity as an audio-quality proxy** is now an established finding
  for this pipeline -- prefer it over cosine similarity when judging "is this checkpoint actually
  better" in any future diagnostic here, especially once cosine similarity is already near ceiling.
- `scripts/train_gate3_encodec_n32_compute_ablation.py` restricts `--budget` to {80000, 320000} by
  explicit `choices=` -- the 20k point is TASK-11's own run, never retrained by this script.
- `scripts/plot_gate3_encodec_n32_compute_ablation.py` depends on TASK-11's report.json existing
  (for the 20k baseline) -- rerun TASK-11 first if that file is ever regenerated differently.
