# NEXT

## Current gate

**Gate 5 full-GTZAN pilot: run complete, NOT PASSED. Outcome C, possibly C->B pending listening.**

**Headline: MuQ generalizes spectral structure to unseen tracks surprisingly well, but the
current pipeline cannot yet turn that into strong waveform-level reconstruction. TASK-14 did not
find "nothing."** There is a small but statistically reliable held-out conditioning dependence at
the waveform level, and a large, near-universal one at the spectral level.

The user ran the full job on BlueHive (`sbatch bluehive/run_task14.slurm`, job `1956835`, Tesla
T4, 31,078.4s) and copied its logs, `report.json`, and 36 WAVs into `bluehive/logs/` and
`bluehive/artifacts/task14_gate5_full_gtzan/` (git-ignored). Full write-up:
`artifacts/task14_gate5_full_gtzan/TASK-14_RESULTS.md`; summary in
[[DECISIONS.md#task14-gate5-full-gtzan-result]].

- **Spectral (held-out):** log-mel similarity correct 0.73 vs. shuffled_across ~0.37,
  shuffled_time ~0.65, zero ~0; STFT distance agrees. Correct wins on 98-100% of held-out crops
  against each control, in all 10 genres.
- **Waveform (Gate 5's primary endpoint, oracle-relative recovery):** correct 0.0285 val / 0.0315
  test vs. controls 0.007-0.017. CIs exclude zero, 74-93% of examples positive, but deltas miss
  the 0.05 materiality bar. Every condition is at ~1-3% of the oracle.
- **Over-smoothed, not collapsed:** variance ratio 0.27/0.29.

**The script's printed "Outcome D" is superseded.** Its automatic rule considers only the
waveform metric and the 0.05 threshold. An early log-only write-up repeated that D; corrected.

## Exact last successful commands

```bash
# BlueHive (by the user):
sbatch bluehive/run_task14.slurm        # job 1956835, completed

# Local, after the run:
uv run python scripts/train_gate5_full_gtzan_wandb.py --config configs/gate5_smoke.yaml --wandb-mode disabled   # exit 0, new listening selection
uv run python scripts/export_gate5_listening.py --config configs/gate5_smoke.yaml \
    --checkpoint runs/task14_gate5_smoke/checkpoints/best.pt \
    --report artifacts/task14_gate5_smoke/report.json --out-dir /tmp/g5_listen        # 36 WAVs
uv run pytest -q; uv run ruff check .   # 219 passed, clean
```

## Run/checkpoint path

- `bluehive/logs/muq-gate5-full-gtzan-1956835.{out,err}` (git-ignored).
- `bluehive/artifacts/task14_gate5_full_gtzan/report.json` + 36 WAVs (git-ignored; the WAVs are
  the old all-blues selection).
- Checkpoints (`best.pt` = step 38,000): BlueHive scratch only,
  `/gpfs/fs2/scratch/acogliat/invertibility/runs/task14_gate5_full_gtzan/checkpoints/`.
- W&B: `wandb.ai/andrea-cogliati-university-of-rochester/muq-invertibility/runs/01bb8xot`.
- Results report: `artifacts/task14_gate5_full_gtzan/TASK-14_RESULTS.md` (git-ignored).

## Next command

**Blocked on user direction.** Nothing new is authorized. Cheapest informative next steps:

1. **Generate and listen to a representative set.** Copy `best.pt` from BlueHive scratch to
   `runs/task14_gate5_full_gtzan/checkpoints/best.pt`, then:
   ```bash
   uv run python scripts/export_gate5_listening.py \
       --config configs/gate5_full_gtzan.yaml \
       --checkpoint runs/task14_gate5_full_gtzan/checkpoints/best.pt \
       --report bluehive/artifacts/task14_gate5_full_gtzan/report.json
   ```
   That writes worst/median/best per split: val hiphop.00016 / country.00031 / reggae.00012,
   test hiphop.00004 / pop.00098 / classical.00031. Listen for whether correct conditioning keeps
   the original's timbre, brightness, and energy contour where `shuffled_across` does not, and
   whether it misses onsets and pitch contour. The result decides C vs. B.
2. **Decide Gate 5's primary endpoint.** Keep it waveform-only (then work on the waveform gap:
   objective, over-smoothing, phase), or add a spectral endpoint (needs a pre-registered
   threshold, since none was defined for log-mel).
3. **Remaining script fixes** if another Gate 5 run is planned: make the outcome rule use more
   than the waveform metric; add `wandb.Audio`; optionally track log-mel per checkpoint.

## Blockers

1. **User decision** on the endpoint question and next direction.
2. **`best.pt` not local** -- needed for the representative listening set.
3. **Listening verdicts pending**: TASK-14, plus the backlog (TASK-13, TASK-12 80k/320k, TASK-11,
   TASK-10, TASK-08, TASK-02, TASK-01).
4. Carried over from TASK-13: pooled vs. held-out-only Gate 4 criteria.
5. TASK-03's BlueHive job still not run; `experiment/mel-hifigan-sanity` diagnostic pending.

## Expected verification criteria for Gate 5

Evaluated: NOT PASSED. Primary waveform endpoint missed materiality (max paired delta 0.025 vs.
0.05); listening not done. Replication across val/test: yes. Breadth across tracks and genres:
yes. Collapse: no. Defects: none found.

## Notes for whoever resumes this

- Read `artifacts/task14_gate5_full_gtzan/TASK-14_RESULTS.md` first.
- **Do not conclude TASK-14 found nothing.** It found reliable, temporally-ordered, per-track
  spectral generalization from MuQ to unseen tracks. What failed is waveform-level recovery.
- **Do not cite the script's printed "Outcome D".** Its rule is waveform-only.
- **Separate waveform from spectral claims.** "No generalization" and "full generalization" are
  both wrong.
- Latent cosine is useless as a discriminator here (0.945-0.975 even with zero conditioning). Use
  log-mel, STFT, variance ratio, and the waveform metrics.
