# TASK-23 on ACES: one-crop decoder-overfit diagnostic

Brief: `tasks/TASK-23.md`. Config: `configs/task23.yaml` (smoke: `configs/task23_smoke.yaml`).
Entry point: `scripts/task23_overfit.py`. Helpers: `src/muq_inversion/task23.py`. Tests: `tests/test_task23.py`.

**Question:** can the TASK-22 decoder reproduce one fixed target mel from its fixed first-block MuQ
features? This tests fitting ability, not invertibility. There are no new losses and no decoder changes.

## What runs

| Step | What | Length |
| --- | --- | --- |
| Pair | `rock.00009:train_pool:pool:18`: training track, 19.63 s, 47,872 samples. Same crop as TASK-18..22's memorized crop 18; TASK-22 best mel L1 on it was 0.238 | - |
| A | Free (1, 100, 187) tensor. Same masked L1, Adam LR 0.01, weight decay 0. Starts at the target mean (−5.05) plus 0.01 × N(0,1) noise, seed 0 | Up to 20,000 updates, stops at the criterion; plateau rule ×0.1 after 2,000 updates without 1% improvement (max 2) |
| B | Fresh `MuQToMelTranslator` (1,271,908 parameters), TASK-22 seed 0 and build order. Adam 1e-3, betas (0.9, 0.999), weight decay 0, no clipping, FP32, batch 1 | 20,000 updates; if eval L1 is above 1% of its initial value, continue to 50,000 at LR/10 |
| C | B's step-0 weights (hash-checked). Disables every explicit regularizer: **there is none** (see below) | As B |
| C_seed1 | Seed-1 initialization, regularizers disabled | Only if C misses the criterion |

At every report point (every 100 updates) and at steps 0/100/1k/5k/20k/50k, the run records:
- direct mel L1 in training and in evaluation mode, measured without changing parameters, buffers or RNG;
- interior and boundary L1, time and frequency first-difference errors (diagnostic only), gradient norm, LR, runtime, and non-finite counts.

Outputs per run:
- checkpoints at those steps plus best and final, and predictions at the same steps;
- target, prediction and error spectrograms (shared target/prediction colour scale, separate error scale), with transient and harmonic zooms;
- a progression figure and learning curves;
- BigVGAN audio of the best and final predictions as float WAV at gain 1.0;
- `report.json` and `TASK-23_RESULTS.md`.

## Verified in the code, not assumed

- **Loss:** TASK-22's loss is `(pred - mel).abs().mean()` over every bin and frame. Crops are exactly 187 mel hops of real audio, so there is no padding and the valid mask is all ones. The masked L1 equals that loss exactly at initialization (recorded difference 0.0).
- **Normalization:** none on the features (raw input to the Linear) and none on the target (raw log-mel). Before vocoding, the inverse is the identity followed by the same clamp to BigVGAN's log floor as TASK-22. The target has 0 bins below the floor.
- **Regularizers:** `MuQToMelTranslator` has no dropout, no normalization layers and no buffers. Adam weight decay is 0. Gradient clipping is measure-only (`clip_grad_norm_(..., inf)`). There is no input noise, no augmentation and no random crop. So **C differs from B only in nothing**, and C is effectively a replicate of B. Train and eval mode are identical by construction, and the runs measure that (max difference 0.0 in the smoke).
- **First-block identity:** `hidden_states[1]` is hook-checked on every extraction. MuQ stays in eval mode with no gradients, and its parameter checksums are unchanged after extraction.
- **Extraction context:** like TASK-22, features come from the isolated 47,872-sample crop of the peak-normalized track (0.95). The full run uses **the exact tensor TASK-22 trained on**, from `data/cache/task22` (job 2261697). A fresh extraction is compared against it and must not differ by more than 0.1.
  - Same backend, CPU batch 16 vs batch 1: 1.3e-5.
  - MPS against a CPU-built cache: 0.051.
  - A wrong layer or crop differs by O(1–10).

  The cached mel and waveform must equal the recomputed ones bit for bit; they did on CPU and MPS.
- **Alignment:**
  - Mel frame j is centered at 256j + 128 samples (reflect-padded, `center=False`).
  - MuQ token centers come from the conv frontend geometry, with a 960-sample hop.
  - Source positions run from 0.13 to 49.0; the last 3 mel frames (184–186) are edge-replicated.
- **Boundary frames:** 12. That's 6 frames at each end, half the translator's 13-frame receptive field, where its convolutions zero-pad; the 3 edge-replicated frames fall inside them.
- **Unresolved alignment uncertainty** (written into the report):
  - a token's "center" is a convention after global attention;
  - edge tokens see only the crop's audio;
  - the frame-center mapping is the TASK-15/16 audit and was not re-measured here.
- **BigVGAN:** on the ground-truth mel it gives 47,872 samples, all finite, with a waveform→mel round-trip L1 of 0.110. That's a round-trip number, not a direct mel error.

## Nondeterminism (found while preparing TASK-23)

- **Cause:** the backward pass of the frame-center interpolation (`interpolate_at_positions`) accumulates with `index_put`. That isn't deterministic on MPS, or on CPU with several threads. TASK-22 had the same behaviour.
- **Effect at batch 1:** with Adam and an L1 loss, the differences grow. Two identical 100-update runs end with weight differences of about 3e-3 on MPS and 6e-3 on CPU.
- **Bit-exact resume:** with `--deterministic` (`torch.use_deterministic_algorithms(True)`) CPU training is bit-exact, so that's how resume is verified (smoke step 1). MPS has no deterministic implementation and raises; XPU support is tested in smoke step 2.
- **Full run:** it uses the TASK-22 recipe, without `--deterministic`. The report measures the B-vs-C spread as the run-to-run noise floor. If B and C come out differently against the criterion, the report says the split isn't evidence about regularization.

## Local verification (2026-11-05, M4 Max)

- `uv run pytest tests/test_task23.py`: 12 passed. They cover pair hashes and tampering, masked loss with padded frames, detail metrics on valid pairs only, gradient flow to the free tensor and every decoder parameter, frozen MuQ, identical B/C initialization (1,271,908 parameters, TASK-22 order), diagnostic passes that leave BatchNorm running statistics and RNG unchanged, bit-exact checkpoint save/resume, and the extension and criterion rules.
- `DEVICE=mps LOCAL=1 bash aces/task23_smoke.sh`: PASS in 109 s.
  - CPU deterministic: B stopped at update 150 and resumed equals uninterrupted C, bit-exact at every report point and in the final weights.
  - MPS: every arm and branch ran (A reached the criterion at 650 updates; B/C/C_seed1 extended at 300). All plots and audio were written.
  - B-C spread on MPS: max |Δ eval L1| 0.082, first difference at update 50.
- TASK-22 smoke cache path (`blues.00000` pool crop 6 from `data/cache/task22_smoke`): passed on CPU (diff 1.3e-5) and MPS (0.051).
- Throughput: 130–160 updates/s for the decoder arms (MPS and CPU), about 800 updates/s for A on MPS.

Smoke numbers are not results.

## Runbook

1. **Commit and push**, then on ACES: `git pull`.
2. **Smoke** (about 10 min) on a PVC node:
   ```bash
   srun --partition=pvc --gres=gpu:pvc:1 --ntasks=1 --cpus-per-task=8 --mem=32G --time=00:45:00 --exclude=ac079 --pty bash
   module load WebProxy uv && bash aces/task23_smoke.sh
   ```
   Pass criteria:
   - "RESUME CHECK PASS";
   - every arm finishes on XPU;
   - the line saying whether deterministic algorithms are supported on XPU (information only);
   - step 3 shows a fresh-vs-TASK-22-cache difference at most 0.1 (expected about 1e-4 or less on XPU), with the mel and waveform equal.
3. **Full run**, only when you say so:
   ```bash
   mkdir -p aces/logs && sbatch --exclude=ac079 aces/run_task23_v1.slurm
   ```
4. **Copy back:**
   - `artifacts/task23/task23_v1/` (report, plots, WAVs, pair metadata);
   - `runs/task23/task23_v1/` (pair.pt, metrics.jsonl, summaries, checkpoints, predictions: about 110 MB per decoder arm, about 350 MB in all);
   - `aces/logs/muq-task23-v1-*.{out,err}`.

## Resources

| | Estimate | Basis |
| --- | --- | --- |
| Pair preparation | about 5 min | MuQ + BigVGAN load, memory-mapped TASK-22 cache (about 7 GB), selection scoring on 32 tracks |
| A | under 1 min | about 650 updates in the smoke |
| B, C, C_seed1 | up to about 7 min each at 50,000 updates | about 130 updates/s (smoke; TASK-22 ran 131 steps/s at batch 64 on PVC) |
| Outputs | about 1 min | plots + 4–8 BigVGAN decodes |
| **Total** | **about 35 min**, walltime 2 h | about 5 billing-hours at 8 per job-hour |

## Report template

`TASK-23_RESULTS.md` is generated. It contains:
- the pair, verification and alignment caveats;
- for each arm: initial/best/final eval L1, the criterion step, the extension, the train/eval difference, interior/boundary, time/frequency difference, and runtime;
- audio paths and peaks, plus the vocoded round-trip L1 (labelled separately from direct mel L1);
- the regularizer audit and what was disabled (nothing), B-C nondeterminism, and the interpretation rule applied mechanically;
- the four required limits. Listening stays **pending** until a human rates the audio.
