# TASK-24 on ACES: fixed-crop decoder fitting ladder

Brief: `tasks/TASK-24.md`. Config: `configs/task24.yaml` (smoke: `configs/task24_smoke.yaml`).
Entry point: `scripts/task24_ladder.py {manifest,prepare,train,report,all}`. Helpers: `src/muq_inversion/task24.py`.
Tests: `tests/test_task24.py`. Manifest (tracked): `data/manifests/task24_ladder.{csv,json}`.

**Question:** where does the first-block MuQ → mel decoder lose its ability to fit **training** crops as the number of fixed pairs grows?

## Design

- **Nested subsets:** N = 1, 4, 16, 64, 256 are prefixes of one ordering, so each subset contains every smaller one.
  - Rank 0 is the TASK-23 anchor `rock.00009:train_pool:pool:18`.
  - Every crop comes from a different training track; genres are interleaved round-robin from a seeded track order (seed 0), with one seeded crop per track.
  - Genres per N: N=4 is rock, jazz, disco, pop; N=16 covers all 10 genres (1–2 each); N=64 has 6–7 per genre; N=256 has 25–26 per genre.
  - Nothing depends on decoder performance.
- **Monitoring set:** the next 16 tracks in that order, which are training tracks with no crop in the 256 pool, one crop each.
  - It only monitors transfer. It never selects a checkpoint, changes the schedule or stops training.
  - It is not a test set, and no validation or test crop is used.
- **Data:** gathered from TASK-22's first-block cache (`data/cache/task22`, job 2261697) with no recomputation.
  - The anchor's cache row must equal TASK-23's `pair.pt` bit for bit (features, target mel, waveform).
  - For the first 4 crops, a fresh frozen-MuQ extraction (hook-checked) and a fresh mel and waveform must agree with the cache:

    | Check | Threshold | Measured |
    | --- | --- | --- |
    | Feature relative RMS | ≤ 0.05 | 7e-4 to 8.5e-3 between backends; a wrong layer gives about 1.07 |
    | Mel | ≤ 1e-2 | - |
    | Waveform | ≤ 1e-6 | soxr resampling differs by up to 8.9e-8 between ACES and the Mac |

  - Every crop's tensors are hashed (`ladder_pairs.pt`, verified on every load). Crop length, isolated-crop extraction, peak normalization, the mask (all frames valid) and the 12 boundary frames are TASK-22/23's.
- **Starting weights:** TASK-23 B's step-0 weights for every arm (`init_state.pt`; sha `09150937…`, which equals a fresh seed-0 TASK-22 build). No arm is warm-started from another.
- **Recipe:** unchanged from TASK-23 B: `MuQToMelTranslator`, direct masked mel L1, Adam 1e-3 (0.9/0.999, weight decay 0), no clipping, FP32, batch 1. There is no "unregularized" arm, since TASK-23 found nothing to disable.
- **Sampling:** seeded shuffled passes without replacement within a pass. Presentation counts per crop are exact (after k·N updates every crop has been seen k times) and the sampler state is checkpointed.
- **Budget:** 200,000 updates per arm with no early stopping.
  - LR is 1e-3 up to update 50,000, then 1e-4 to update 150,000, then 1e-5. This schedule is a deliberate change from TASK-23.
  - Checkpoints at 20k, 50k and 150k (each just before a reduction), 100k and 200k, plus best (lowest whole-fitting-set L1) and latest (every 5k updates and on TERM). That's about 100 MB per arm.
- **Matched exposure:** metrics plus full fitting-set and monitoring predictions after exactly 100·N and 500·N updates (N=256: 25,600 and 128,000). These points fall at different update counts and LR phases for different N; matched steps give different exposure per crop.
- **Evaluation:** the whole fitting subset in evaluation mode, with masked absolute error aggregated over every valid mel element.
  - It runs every 1,000 updates and at every comparison step; monitoring runs every 5,000 updates and at every comparison step.
  - It reports per-crop L1 (mean, median, p90, max), the anchor, interior and boundary L1, time and frequency first differences (diagnostic), gradient norm, LR, throughput and presentations.
  - Evaluation passes restore RNG and refuse to change model state.
- **Spikes:** recorded as events where the whole-set L1 exceeds 2× the best so far, closed when it drops below 1.1× the pre-spike best. Causes are not attributed.
- **Report:**
  - Tables for matched steps, matched exposure and best checkpoint.
  - Plots: L1 vs updates and vs presentations, monitoring separately, per-crop distributions vs N, anchor L1 vs N, LR and spike annotations.
  - For each N, spectrograms of the anchor plus 3 crops chosen in advance (ranks 1–3), with transient and harmonic zooms.
  - BigVGAN WAVs of the best and final predictions plus the original and ground-truth-mel references, as float WAV at gain 1.0.

## Local verification (2026-11-05, M4 Max)

- `uv run pytest tests/test_task24.py`: 11 passed. They cover:
  - nested manifests and split disjointness (the committed manifest regenerates identically);
  - cached-pair hashes with tamper detection and the exact-anchor check;
  - identical starting weights across arms;
  - sampler passes, counts and resume;
  - element-wise and per-crop aggregation;
  - evaluation without state or RNG changes (BatchNorm and Dropout model);
  - frozen MuQ;
  - LR schedule, checkpoint and exposure timing;
  - bit-exact save/resume;
  - the writer lock.
- `DEVICE=mps LOCAL=1 bash aces/task24_smoke.sh`: PASS in 189 s.
  - Manifest verified.
  - Deterministic CPU: N=4 interrupted at update 250 and resumed equals the uninterrupted run (weights, sampler counts, evaluation logs).
  - MPS ladder N = 1, 2, 4: all stages ran, about 158 updates/s.
  - A second writer was refused.

  It uses the local TASK-22 smoke cache (6 blues tracks, built on CPU) plus the anchor, so its numbers are not results.
- **The full ladder could not run locally.** TASK-22's cache of the other 271 crops exists only on ACES, and the task forbids recomputing them.

## Runbook

1. Commit and push, then `git pull` on ACES.
2. **Smoke** (about 15 min) on a PVC node:
   ```bash
   srun --partition=pvc --gres=gpu:pvc:1 --ntasks=1 --cpus-per-task=8 --mem=64G --time=01:00:00 --exclude=ac079 --pty bash
   module load WebProxy uv && bash aces/task24_smoke.sh
   ```
   Pass criteria:
   - "RESUME CHECK PASS" and "LOCK CHECK PASS";
   - step 4 prints the anchor as "TASK-22 cache row … == TASK-23 pair.pt (bit-exact)", the cross-check values under their thresholds, MuQ frozen, and **the measured N=256 throughput with the resulting estimate** for the full run.
3. **Full run**, only when you say so:
   ```bash
   mkdir -p aces/logs && sbatch --exclude=ac079 aces/run_task24_v1.slurm
   ```
   If it exits with code 3, resubmit the same command. Finished arms are skipped and the interrupted arm resumes.
4. **Copy back:**
   - `artifacts/task24/task24_v1/` (report, plots, WAVs);
   - `runs/task24/task24_v1/` (about 0.6 GB with checkpoints; add `--exclude 'checkpoints/'` for logs, summaries and predictions only);
   - `aces/logs/muq-task24-v1-*`.

## Resources

| | Estimate | Basis |
| --- | --- | --- |
| Prepare | about 5 min | 7 GB memory-mapped cache plus its SHA-256, MuQ/BigVGAN load, 4-crop cross-check |
| 5 arms × 200,000 updates | about 50–60 min | TASK-23 on PVC: 340 batch-1 updates/s with reporting; whole-set evaluation of up to 256 crops adds a few ms per 1,000 updates |
| Report | about 3 min | 48 BigVGAN decodes plus plots |
| **Total** | **about 1–1.5 h**, walltime 6 h | **about 8–12 billing-hours** at 8 per job-hour (`billing=8` in sacct AllocTRES) |

Smoke step 4 replaces this estimate with one measured from the real N=256 arm.
