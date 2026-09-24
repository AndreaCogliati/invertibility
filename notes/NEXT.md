# NEXT

## Current gate

**Gate 5 full-GTZAN pilot: run complete, NOT PASSED -- Outcome B (partial generalization).**

The user ran the full job on BlueHive (`sbatch bluehive/run_task14.slurm`, job `1956835`, Tesla
T4, 31,078.4s) and copied its logs, `report.json`, and 36 listening WAVs into `bluehive/logs/` and
`bluehive/artifacts/task14_gate5_full_gtzan/` (git-ignored). Full write-up:
`artifacts/task14_gate5_full_gtzan/TASK-14_RESULTS.md`; summary in
[[DECISIONS.md#task14-gate5-full-gtzan-result]].

**Result in one line:** at full scale the translator generalizes *spectral* content to unseen
tracks, using temporal order, but not waveform detail.

- Waveform (primary endpoint, oracle-relative recovery): correct 0.0285 val / 0.0315 test vs.
  controls 0.007-0.017. Statistically real (CIs exclude zero, 74-93% of examples positive) but
  far below the 0.05 materiality bar. Every condition is at ~1-3% of the oracle.
- Log-mel similarity: correct 0.73 vs. shuffled_across ~0.37, shuffled_time ~0.65, zero ~0.
  Correct wins on 98-100% of held-out crops against each control, in all 10 genres.
- Variance ratio 0.27/0.29: over-smoothed, not collapsed.

**Correction made this session:** a first, log-only write-up classified this as Outcome D ("no
held-out conditioning dependence") and said it refuted TASK-13's data-scale hypothesis. Wrong on
both counts once `report.json`'s per-example data was read. The report and notes are corrected.
The script's own printed "Outcome D" is superseded (its rule only checks the waveform metric).

## Exact last successful commands

```bash
# BlueHive (by the user):
sbatch bluehive/run_task14.slurm        # job 1956835, completed

# Local analysis of the copied report (read-only, no code changes):
uv run python -c "...paired_delta_stats over report.json per_example..."
uv run pytest -q; uv run ruff check .   # 215 passed, clean (unchanged since bb61acd)
```

## Run/checkpoint path

- `bluehive/logs/muq-gate5-full-gtzan-1956835.{out,err}` (git-ignored).
- `bluehive/artifacts/task14_gate5_full_gtzan/report.json` + 36 WAVs (git-ignored).
- Checkpoints (`best.pt` = step 38,000): BlueHive scratch only,
  `/gpfs/fs2/scratch/acogliat/invertibility/runs/task14_gate5_full_gtzan/checkpoints/`.
- W&B: `wandb.ai/andrea-cogliati-university-of-rochester/muq-invertibility/runs/01bb8xot`.
- Results report: `artifacts/task14_gate5_full_gtzan/TASK-14_RESULTS.md` (git-ignored).

## Next command

**Blocked on user direction.** Nothing new is authorized. Cheapest informative next steps:

1. **Listening pass** on the 36 WAVs, and ideally on a genre-diverse held-out subset (the saved
   subset is all blues because the script picks the first labels in sort order). Suggested
   additions: `test_classical.00031_c1` (best), `test_hiphop.00085_c1` (median),
   `test_jazz.00032_c0` (worst). Producing those needs `best.pt` from BlueHive scratch. What to
   listen for: does correct conditioning share the original's timbre/brightness/energy contour
   where `shuffled_across` does not, and does it miss onsets and pitch contour?
2. **Decide Gate 5's primary endpoint**: keep it waveform-based (then the path forward is the
   waveform gap: objective, over-smoothing, phase), or adopt a spectral endpoint (would need a
   pre-registered threshold, since none was defined for log-mel).
3. Small script fixes if another Gate 5 run is planned: outcome rule should use more than the
   waveform metric; fix the mislabeled periodic val print; add `wandb.Audio`; pick a genre-diverse
   listening subset; optionally track log-mel per checkpoint.

## Blockers

1. **User decision** on the endpoint question and next direction (above).
2. **Listening verdicts pending**: TASK-14 full run, plus the long-standing backlog (TASK-13,
   TASK-12 80k/320k, TASK-11, TASK-10, TASK-08, TASK-02, TASK-01).
3. **Carried over from TASK-13**: pooled vs. held-out-only Gate 4 criteria (TASK-14 used held-out
   only, per its own spec).
4. TASK-03's BlueHive job still not run; `experiment/mel-hifigan-sanity` diagnostic still pending.

## Expected verification criteria for Gate 5

Evaluated: NOT PASSED. The primary waveform endpoint missed materiality (max paired delta 0.025 vs.
0.05) and listening was not done. Replication across val/test: yes. Breadth across tracks and
genres: yes. Collapse: no. Defects: none found.

## Repository state at the end of this session

`notes/DECISIONS.md`, `notes/MEMORY.md`, `notes/NEXT.md` updated (being committed now). No code
changed. Last code commit: `bb61acd`.

## Notes for whoever resumes this

- Read `artifacts/task14_gate5_full_gtzan/TASK-14_RESULTS.md` first.
- **Don't cite the script's printed "Outcome D"** -- superseded. The per-example data supports B.
- **Separate waveform from spectral claims.** "No generalization" and "full generalization" are
  both wrong; the accurate statement is "spectral yes, waveform no."
- Latent cosine is useless as a discriminator here (0.945-0.975 even with zero conditioning). Use
  log-mel, STFT, variance ratio, and the waveform metrics.
