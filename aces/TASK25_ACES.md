# TASK-25: architecture vs optimization on the N=256 fitting pool

Brief: `tasks/TASK-25.md`. Config: `configs/task25.yaml` (smoke: `configs/task25_smoke.yaml`).
Runner: `scripts/task25_ablation.py {prepare,train,report,all}`. Helpers: `src/muq_inversion/task25.py`. Tests: `tests/test_task25.py`.

**Where it ran:** the full run executes **locally on the M4 Max (MPS)**. TASK-24's cached pairs and initial state were already
downloaded, so nothing needs recomputing; the task says to run on an available accelerator. The ACES scripts below are only
needed to rerun on PVC.

## Arms

All arms share:
- **Data:** TASK-24's 256 fitting pairs and 16 monitoring crops (`runs/task24/task24_v1/ladder_pairs.pt`). Per-crop hashes are
  verified, the crop list matches `data/manifests/task24_ladder.csv`, monitoring tracks don't overlap the fitting pool, and
  rank 0 equals TASK-23's anchor pair bit for bit.
- **Recipe:** first-block features, raw log-mel targets, all-valid masks, direct masked mel L1. Adam (0.9, 0.999) with weight
  decay 0, **no gradient clipping** (unchanged), FP32, frozen MuQ (cached features), no dropout, normalization, augmentation or
  noise.

| arm | decoder | batch | initial LR | init | params | fwd MACs/crop | MuQ frames per output frame (min/median/max) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | current (hidden 256) | 1 | 1e-3 | shared (TASK-23 B step 0) | 1,271,908 | 201.8M | 2 / 5 / 6 |
| B | current | 1 | 1e-4 | shared | 1,271,908 | 201.8M | 2 / 5 / 6 |
| C | current | 16 | 1e-3 | shared | 1,271,908 | 201.8M | 2 / 5 / 6 |
| D | current (**candidate recipe**) | 16 | 1e-4 | shared | 1,271,908 | 201.8M | 2 / 5 / 6 |
| E | width ×2 (hidden 512; same depth, kernels, 13-frame RF, GELU, linear interpolation) | 16 | 1e-4 | seed 0, fresh build | 4,509,796 | 771.3M | 2 / 5 / 6 |
| F | learned resampler (below), current trunk | 16 | 1e-4 | seed 0 + shared trunk copied | 1,305,764 | 208.0M | 3 / 7 / 8 |

Different architectures can't share complete weights:
- **E** has a fresh seed-0 build.
- **F** copies the whole shared trunk and adds a seed-0 resampler, whose zero-initialized output layer makes F start
  **bit-identical** to A–D's output (verified in `prepared.json` and in a test).

## F: learned temporal resampler (`task25.PhaseResampler`)

The fixed linear interpolation at frame-center positions is replaced; everything else in the decoder is identical.
- **Positions:** mel frame j sits at fractional MuQ position p_j = lo_j + w_j (frame-center alignment, from 0.133 to 49.0).
  There are 3.75 mel frames per MuQ frame, so the phase w_j cycles through 15 values.
- **Combination:** output(c, j) = Σ_k (base_k(w_j) + δ_{k,c}(w_j)) · h(c, lo_j + k) over taps k ∈ {−1, 0, 1, 2}.
  - base = [0, 1−w, w, 0], which is exactly the old linear interpolation.
  - δ = MLP(w): 1 → 32 → GELU → 4·256, with a **zero-initialized** output layer.
  - Tap indices are clamped to [0, 49] (edge replication, as before).
- **Locality:** one rule shared by all crops and positions; it depends only on the phase. It's local, with no absolute-position
  table and no crop-specific mapping.
- **Support:** 2 → 4 MuQ frames per mel frame. Influence per output frame goes from a median of 5 MuQ frames (range 2–6) to 7
  (3–8).
- **Cost:** +33,856 parameters (+2.7%), +3.1% MACs.
- **Gradients:** finite. At step 0 the MLP's first layer gets zero gradient, because the output layer is zero; it moves from the
  second update on (tested).

## Budget and schedule (identical for every arm)

- **Budget:** 819,200 crop presentations (3,200 per crop) = 819,200 updates at batch 1, 51,200 at batch 16.
- **Sampling:** one seeded shuffled-pass stream (`task25-sampler-0`) shared by all arms; batches are consecutive groups of 16.
  Because 256 is divisible by 16, a batch never spans two passes or repeats a crop.
- **LR schedule:** by cumulative presentations. The initial LR runs until 204,800, then ×0.1, then ×0.1 again at 614,400.
  Checkpoints just before each reduction.
- **Matched-update comparison:** at 10,000 and 50,000 updates (secondary: unequal exposure).
- **Evaluation:** whole 256-crop pool every 16,384 presentations; monitoring every 65,536; both also at the LR boundaries, the
  matched updates and the end.
- **Selection:** best checkpoint by training-pool error only; final checkpoints are the primary comparison.
- No early stopping and no schedule changes.
- **Logging:** W&B, one run per arm in group `task25-arch-vs-opt`, with a JSONL mirror.

## Verification before the full run (2026-11-05, local)

- **Unit tests:** `uv run pytest tests/test_task25.py`, 12 passed. They cover:
  - arm definitions and budget arithmetic;
  - TASK-24 pair hashes, manifest identity, anchor and monitoring disjointness;
  - identical A–D initial weights and the F trunk copy;
  - the resampler: initialization equals linear interpolation bit for bit (and so does F's whole output), tap alignment and
    clamping, gradients;
  - the exposure-based LR schedule;
  - batches as groupings of the same sample stream, with exact counts;
  - masked aggregation and paired differences;
  - checkpoints at LR boundaries;
  - bit-exact resume for the current decoder and F;
  - evaluation without state or RNG changes.
- **Smoke run on MPS:** `uv run python scripts/task25_ablation.py all --config configs/task25_smoke.yaml --device mps
  --run-name <tag> --wandb-mode disabled`. All six arms, report, plots and audio completed in 105 s.
- **Throughput on MPS:**

  | | updates/s | presentations/s |
  | --- | --- | --- |
  | Batch 1 | about 150 | about 150 |
  | Batch 16 | about 95 | about 1,500 |
  | E | 39 | 620 |
  | F | 81 | 1,300 |

  During the full run, with three concurrent processes, A and B each reach about 102 updates/s.

## Resource estimate

| | batch-1 arms (A, B) | batch-16 arms (C, D) | E | F | total |
| --- | --- | --- | --- | --- | --- |
| MPS, sequential | about 90 min each | about 8 min each | about 22 min | about 11 min | about 3.9 h |
| MPS, 3 parallel processes (as run) | about 2.2 h (concurrent) | | | | **about 2.2 h, free** |
| ACES PVC, sequential (TASK-24: 370 batch-1 updates/s) | about 37 min each | | | | about 1.5–2 h, **about 12–16 billing-hours** |

## ACES (only to rerun there)

1. Make sure `runs/task24/task24_v1/{ladder_pairs.pt,init_state.pt}` and `runs/task23/task23_v1/{pair.pt,B/summary.json}` are
   present on ACES. They are, from the TASK-23/24 jobs.
2. Smoke: `srun ... --pty bash`, then `module load WebProxy uv && bash aces/task25_smoke.sh`. It checks bit-exact CPU resume for
   D and F, runs all arms on XPU, and prints the measured estimate.
3. Full: `mkdir -p aces/logs && sbatch --exclude=ac079 aces/run_task25_v1.slurm`. Exit 3 means resubmit.
