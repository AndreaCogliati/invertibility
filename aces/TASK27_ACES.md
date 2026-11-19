# TASK-27: adversarial mel-spectrogram reconstruction

Brief: `tasks/TASK-27.md`. Configs: `configs/task27.yaml` (full run), `configs/task27_smoke.yaml` (ACES smoke),
`configs/task27_smoke_local.yaml` (local smoke with a labeled stand-in start).
Runner: `scripts/task27_adversarial.py {prepare,calibrate,train,report,all,score-listening}`.
Helpers: `src/muq_inversion/task27.py`. Tests: `tests/test_task27.py` (19).

**Where it runs:** ACES. The common start, TASK-25 C's final checkpoint (`runs/task25/task25_v1/C/checkpoints/final.pt`),
exists only on ACES, because the TASK-25 download excluded checkpoints. SSH from the Mac to ACES was refused
(`Permission denied (publickey)`) on 2026-11-19, so it could not be copied, and nothing was submitted.

## What the run does

1. **prepare.** Checks the inputs and fixes everything that must be decided before training.
   - Data:
     - verifies TASK-24's 256 + 16 cached pairs (per-crop hashes), the manifest, monitoring disjointness and the
       TASK-23 anchor.
   - Start checkpoint:
     - its state-dict SHA-256 must be `e52fc84e…` (TASK-25 C's recorded `final_sha256`);
     - its spec must be C's (batch 16, LR 1e-3, 819,200 presentations), at update 51,200;
     - re-evaluated, its fit L1 must equal the recorded 0.169569 (±1e-4).
     - Any mismatch stops the run. No other model is substituted.
   - Fixes before training:
     - the discriminator input normalization;
     - the 8 training + 8 monitoring listening examples.
2. **calibrate.** Measures the adversarial weight with a temporary generator and discriminator.
   - The generator holds the start weights and is not updated.
   - The discriminator (seed 0) gets 100 D-only warm-up updates on training pairs, from its own sample stream.
   - On 8 fixed training batches, it measures the generator gradient norms of L1 and of the adversarial term.
   - It sets lambda = 0.1 × median |g_L1| / median |g_adv|.
   - It stops if a result is non-finite or degenerate.
3. **train.** For each arm, everything is reset:
   - `seed_everything(0)`;
   - start weights reloaded;
   - new optimizers;
   - the sample stream `task27-sampler-0` from its beginning.

   Then:
   - **A:** masked mel L1 only.
   - **B:** first 100 D-only warm-up updates, then L1 + lambda(u)·hinge adversarial loss. Lambda ramps linearly from
     0 over the first 5,120 of 51,200 generator updates, then stays fixed.
   - **Both arms:**
     - batch 16; 819,200 presentations;
     - generator Adam (0.9, 0.999) at LR 1e-4; discriminator Adam (0.5, 0.9) at LR 1e-4;
     - both LRs ×0.1 at 409,600 and 614,400 presentations;
     - FP32; no clipping; no early stopping.
4. **report.**
   - Final checkpoints are the primary result; best-training-L1 checkpoints are secondary.
   - Fit pool and monitoring set are reported separately. Metrics:
     - mel L1; per-crop distribution; anchor;
     - first-difference errors in time and frequency;
     - boundary vs interior error;
     - correct vs shuffled conditioning;
     - discriminator logits and saturation; gradient norms; compute.
   - Spectrograms on shared scales, with transient and harmonic zooms.
   - The blinded listening package, with the key in a separate folder.

## Discriminator (measured by `prepare`)

| | |
| --- | --- |
| input | clamp(mel, ln 1e-5) → (x − μ)/σ, with μ, σ from the 256 clamped training targets; identical for real and generated |
| conditioning | cached block-1 MuQ features → the generator's frame-center linear interpolation to the 187 mel frames → SN Conv1d(1024→16, k=1), broadcast over the 100 bins |
| layers | SN Conv2d 17→32 k3 s1, 32→64 k4 s2, 64→128 k4 s2, 128→128 k3 s1, 128→1 k3 s1; all unpadded; LeakyReLU 0.2; no normalization; no sigmoid |
| parameters | 334,097 |
| receptive field | 28 bins × 28 frames = 298.7 ms (crop: 100 × 187); logit stride 4 |
| logits | 19 × 40 per crop. Every receptive field lies inside the crop. Logits touching invalid frames would be masked; all frames are valid here. |
| forward MACs | about 0.5 G per crop, vs 0.2 G for the generator |

## Procedure

1. Commit and push; then `git pull` on ACES.
2. Smoke test, interactive (about 10–15 min):

   ```bash
   srun --partition=pvc --gres=gpu:pvc:1 --ntasks=1 --cpus-per-task=8 --mem=32G --time=00:45:00 --exclude=ac079 --pty bash
   module load WebProxy uv && bash aces/task27_smoke.sh
   ```

   - Pass criteria:
     - 19 unit tests pass;
     - `RESUME CHECK PASS`: deterministic CPU, using `configs/task27_resume.yaml` (64 updates); A stopped at 30 and B at 40,
       resumed equal to uninterrupted; calibration identical;
     - both arms and the report run on XPU;
     - prepare prints the real start (`e52fc84e…`, `STAND-IN` absent).
   - It prints measured generator updates/s and a full-run estimate.
3. Full run, only when instructed:

   ```bash
   mkdir -p aces/logs && sbatch --exclude=ac079 aces/run_task27_v1.slurm
   ```

   Exit 3 means resubmit.
4. Copy back:
   - `artifacts/task27/task27_v1/`;
   - `runs/task27/task27_v1/` (checkpoints are optional);
   - `aces/logs/muq-task27-v1-*`.

## Resource estimate

**Measured locally** with `DEVICE=mps LOCAL=1 bash aces/task27_smoke.sh`, which PASSED in 496 s.
- **Accelerator smoke on MPS:**
  - A ran at 85 generator updates/s, giving about 0.17 h for the full arm.
  - B ran at 19 updates/s, giving about 0.74 h.
  - Setup, evaluation and report add about 0.3 h.
  - Total: about 1.2 h, so about 10 billing-hours. This is an upper estimate, because XPU was faster than MPS for TASK-25:
    TASK-25 C ran at 262 updates/s on XPU vs about 95 on MPS.
- **Expected on ACES:** about 0.5–1.2 h, i.e. about 4–10 billing-hours at 8/h. The walltime is set to 4 h.
- **Interactive ACES smoke:** about 10–15 min. Its deterministic CPU resume step took 7.3 min locally.

The ACES smoke prints the measured XPU figure; use it to update this estimate.

## Caveats found in the local smoke

- **Lambda depends on a warm-up that does not learn much.** After 100 D-only updates at LR 1e-4, the discriminator is
  at chance (sign accuracy about 0.50; hinge loss 2.000 → 1.995).
  - Lambda is therefore calibrated against an almost uninformative discriminator.
  - Its value depends on the warm-up trajectory: 0.0151 on MPS, 0.0089 on deterministic CPU, from identical seeds and
    data. The 8-batch spread within one backend is small.
  - The run uses the lambda calibrated on XPU. On XPU the actual run's warm-up is not bit-identical to the calibration
    warm-up.
  - As the discriminator learns, the weighted adversarial share of the gradient grows: 0.10 at calibration, 0.54 after
    256 smoke updates.
  - This follows the brief as written. Report it with the results.
- **The local smoke used a stand-in start** (`runs/task25/task25_v1_local_incomplete/C/checkpoints/final.pt`, MPS,
  SHA-256 `47844330…`), labeled in every output. It is never used for results.
