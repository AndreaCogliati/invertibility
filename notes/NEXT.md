# NEXT

## Current gate

N=32 compute-controlled micro-ablation (`tasks/TASK-06.md`). **Result: partial, not full,
recovery -- by the metrics.** Held N=32 fixed and varied only step budget (20k/80k/320k) to test
whether TASK-05's sharp N=8->N=32 collapse was an undertraining artifact. Waveform correlation did
rise substantially with more compute (0.469 -> 0.629 -> 0.666), so exposure is a real contributing
factor -- but even at 320k steps (2750 repetitions per example, 4.6x more than N=8's best
checkpoint needed) N=32 plateaus far below N=8's near-perfect level (cosine 0.63 vs. 0.998,
waveform correlation 0.67 vs. 0.997). Per-example ranking is stable across budgets: one example
(`classical.00003`) never improves regardless of compute while another (`pop.00002`) does --
preferential memorization of a subset, not uniform improvement. **Listening verdict (2026-09-17)
diverges from the metrics: "the best audio was generated at 80K steps. The audio from 320K is
generally worse,"** despite 320k's higher mean metric. Full reasoning and the divergence caveat in
[[DECISIONS.md#task06-n32-compute-ablation]]. Per this task's own explicit instruction, **stopped
after answering its four required questions -- no next experiment proposed by this task.**

**Conclusions reached this session (full reasoning in
[[DECISIONS.md#task06-n32-compute-ablation]]):**

1. Reused TASK-05's N=32 manifest and pipeline unchanged; ran 80k and 320k step budgets with
   patience-based early stopping deliberately disabled (reusing TASK-05's 5000-step patience
   window would likely have triggered a premature stop during this run's characteristic
   multi-thousand-step turbulent dips).
2. **Per-example repetition count, not raw step count, is the right unit for comparison against
   N=8.** 80k's best checkpoint already exceeds N=8's needed per-example exposure by 3x; 320k's by
   4.6x. Neither approaches N=8's quality.
3. **Compute has strongly diminishing, then flat, returns.** The 320k run's best-by-MSE checkpoint
   was found at step 88000 -- barely past the 80k run's own endpoint -- and no new global optimum
   emerged in the remaining 232000 steps despite many local peaks approaching it.
4. **The worst example does not improve with compute; the model preferentially memorizes a
   subset.** `classical.00003` is worst at both 80k and 320k (0.315 -> 0.276, slightly worse
   despite 4x more compute) while `pop.00002` improves (0.875 -> 0.897) alongside the overall
   mean/median rise.
5. Interpretation: the N=8->N=32 boundary is best explained by genuine interference/capacity
   competition among examples for this architecture's limited representational capacity, not by
   compute alone and not by regression-to-the-mean ([[task03-bluehive-wandb]]'s mechanism doesn't
   fit -- scores here are far above degenerate dataset-mean levels and per-example structure is
   consistent across budgets).

## Exact last successful commands

```bash
uv run python -u scripts/train_gate3_n32_compute_ablation.py --budget 80000 --wandb-mode online
uv run python -u scripts/train_gate3_n32_compute_ablation.py --budget 320000 --wandb-mode online
uv run python scripts/plot_gate3_n32_compute_ablation.py   # required figure
uv run ruff check .; uv run pytest -q                       # 133 passed
```

2026-09-16, macOS, M4 Max, device=mps. Launched via `nohup ... &` chained with `&&` + `disown`
(unbuffered `-u`), tracked by real PID and a persistent `Monitor`. 80k run: 6257s (~104 min). 320k
run: 24249s (~6.7 hours).

## Run/checkpoint path

- `data/manifests/gate3_scale_32.csv` (tracked, reused unchanged from TASK-05).
- `runs/task06_n32_compute_ablation/budget_{80000,320000}/checkpoints/{best_by_mse,final}.pt`
  (git-ignored).
- `artifacts/task06_n32_compute_ablation/budget_{80000,320000}/report.json` -- full per-budget
  metrics (32-example aggregate + per-example); `.../budget_{N}/*.wav` -- listening WAVs at steps
  {20000,40000,80000,160000,320000} (best/median/worst/fixed-subset).
- `artifacts/task06_n32_compute_ablation/per_example_waveform_correlation.png` -- the required
  figure (one panel per budget, all 32 examples).
- `artifacts/task05_gate3_scale/n32/report.json` -- the reused 20k reference (TASK-05's original
  patience-stopped run).
- Two live W&B runs, group `gate3-n32-compute-ablation`: `wandb.ai/
  andrea-cogliati-university-of-rochester/muq-invertibility` (`n32_80k_*`, `n32_320k_*`).

## Next command

**Blocked on user direction.** Per TASK-06's own explicit instruction ("stop after answering
these questions; do not automatically propose the next experiment"), no next experiment is
proposed here. Candidate directions that would need the user's go-ahead if raised later:

- A dedicated N=16/N=24 pinpointing run now has a cleaner methodological baseline (compute is
  confirmed to only partially explain the collapse, so a pinpointing run wouldn't need to also
  disentangle exposure).
- Investigating *why* the capacity boundary sits around 8-32 examples specifically -- e.g. does
  batching multiple examples per gradient step change the threshold? Does model capacity (more
  parameters) push it higher?
- Listening verdict for the TASK-06 WAVs (not yet requested from the user).

Do not start any of these without the user's go-ahead.

## Blockers

1. **Decision needed from the user** on which direction to pursue next, if any (TASK-06 itself
   does not request one).
2. **Listening verdicts** for TASK-01's and TASK-02's WAV packages -- still pending (TASK-04's,
   TASK-05's, and TASK-06's are all now confirmed by ear).
3. **BlueHive job not yet run** -- see [[task03-bluehive-wandb]], still pending.
4. The still-pending conditioning-collapse diagnostic from `experiment/mel-hifigan-sanity` --
   untested, orthogonal to this branch's findings.
5. The `shuffled_time` finding from Gate 4 (see [[DECISIONS.md#gate4-shuffled-time-finding]]) --
   still dormant.
6. Listening verdicts for the 80-track scaling experiment, the full-GTZAN scaling experiment, and
   the MuQ layer-sweep were never given -- not blocking, worth noting if any comes up again.

## Expected verification criteria for Gate 5

Not yet reached, and not authorized to start.

## Notes for whoever resumes this

- **Read [[DECISIONS.md#task06-n32-compute-ablation]] first.** It resolves the "is the N=8->N=32
  collapse just undertraining?" question TASK-05 left open: no, not primarily -- exposure matters
  but plateaus far short of N=8's quality even given 5x N=8's needed per-example repetitions.
- Per-example repetition count (`best_checkpoint_step / N`) is the correct unit for comparing
  compute-equivalence across different N -- raw step counts are not comparable across N.
- `src/muq_inversion/gate3_spectrogram_common.py`'s `rank_examples_by_metric` is generic and
  reusable for any future per-example best/median/worst identification -- no new plumbing needed.
- `scripts/plot_gate3_n32_compute_ablation.py` is read-only and reusable if a future N=32-fixed
  run with a different budget schedule is added -- update its report-path constants only.
- Early stopping (patience) is disabled by design in
  `scripts/train_gate3_n32_compute_ablation.py` for its 80k/320k budgets -- do not re-enable it
  without re-deriving why (see [[DECISIONS.md#task06-n32-compute-ablation]]'s resume-policy
  section) if reusing this script for a future compute-budget experiment.
