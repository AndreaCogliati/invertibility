Implement TASK-27: adversarial mel-spectrogram reconstruction.

First inspect any existing TASK-26 specification. Preserve compatible decisions, document conflicts, and do not overwrite completed artifacts. Inspect TASK-22 through TASK-25 code and results before implementing.

GOAL
Test whether adding an adversarial loss to the existing first-block MuQ → mel translator improves perceptual reconstruction without sacrificing correspondence to the source.

This is a loss ablation, not an architecture or data-diversity experiment.

1. DATA AND COMMON START
- Reuse TASK-25's exact 256 fixed training pairs and 16-crop unseen-track monitoring set.
- Preserve features, target normalization, masks, crop context, and manifests.
- Use the original-width decoder and TASK-25 C's final checkpoint as the common starting point.
- Verify the checkpoint identity and compatibility. If unavailable, report the missing prerequisite rather than silently substituting another model.
- Both arms start from identical generator weights.
- Reset generator optimizer state identically in both arms, documenting that this is a new fine-tuning phase.
- Keep MuQ and BigVGAN frozen. BigVGAN is used for evaluation only.

2. TWO MATCHED ARMS
A: L1-only continuation.
B: L1 + adversarial continuation.

Keep generator architecture, initialization, data order, batch size, learning rate, exposure budget, and evaluation schedule identical.

Do not add feature-matching, latent-consistency, spectral-gradient, or waveform losses in this experiment.

3. DISCRIMINATOR
Implement a modest conditional 2D PatchGAN discriminator:
- Inputs: candidate target mel and its corresponding frozen MuQ conditioning features.
- Align conditioning to the target time grid using a documented deterministic operation.
- Project/broadcast conditioning as needed without altering the generator.
- Output local real/fake logits, with no sigmoid.
- Use spectral normalization and no batch normalization.
- Document layers, parameter count, and receptive field in mel bins and milliseconds.
- Ensure its temporal receptive field fits the crop.

Use identical conditioning for each real/fake pair.
Do not let the discriminator distinguish examples through inconsistent padding, normalization, or preprocessing.

Exclude invalid/padded regions from discriminator inputs or mask logits whose receptive fields intersect invalid regions. Document boundary handling.

4. LOSSES AND GRADIENT FLOW
Use hinge losses:

L_D = mean(relu(1 - D(z, m)))
    + mean(relu(1 + D(z, stopgrad(m_hat))))

L_G_adv = -mean(D(z, m_hat))

L_G = L_mel_L1 + lambda_adv * L_G_adv

- Preserve the existing masked direct mel L1 exactly.
- Detach generated mel during discriminator updates.
- Freeze discriminator parameters during generator updates while preserving gradients to generated mel.
- Use one discriminator update per generator update.
- Do not backpropagate into cached MuQ features.

5. ADVERSARIAL WEIGHT
Avoid choosing lambda solely from incomparable loss magnitudes.

Before full training:
- On eight fixed TRAINING batches, measure generator gradient norms for L1 and adversarial terms separately.
- Use a temporary discriminator after 100 discriminator-only warm-up updates on training pairs.
- Set lambda so the median weighted adversarial gradient norm is approximately 10% of the L1 gradient norm.
- Record the measurements and resulting lambda.
- Reset generator, discriminator, optimizers, sampler, and RNG to their prescribed initial states before the actual run.
- If calibration is non-finite or degenerate, stop and diagnose rather than choosing an arbitrary weight.

During the actual run:
- Repeat the 100 discriminator-only warm-up updates.
- Ramp lambda linearly from zero to its calibrated value over the first 10% of generator updates.
- Keep it fixed afterward.
- No monitoring-driven tuning.

6. TRAINING BUDGET
For both arms:
- Batch size 16.
- 51,200 generator updates: 819,200 crop presentations.
- Initial generator LR 1e-4.
- Reduce LR by 10x at 50% and 75% of the exposure budget.
- Preserve generator Adam settings from TASK-25 otherwise.
- No early stopping.

For B:
- Discriminator Adam: LR 1e-4, betas (0.5, 0.9).
- Apply the same LR reduction milestones.
- No additional discriminator regularizers in this first experiment.

Use full precision initially.
Do not silently introduce gradient clipping.
Log spikes, non-finite values, discriminator saturation, and gradient norms.
Stop for genuine numerical failure and retain diagnostic artifacts.

7. EVALUATION
Primary comparison: final checkpoints at matched generator exposure.
Also retain best-training-L1 checkpoints, clearly labeled as secondary.

Evaluate the full fitting pool and monitoring set separately:
- Direct masked mel L1.
- Per-crop error distribution and TASK-23 anchor.
- Temporal/frequency first-difference errors.
- Boundary versus interior errors.
- Conditioning sensitivity using correct versus shuffled MuQ inputs.
- Generator/discriminator losses, logits, gradient norms, runtime, and compute.

Do not treat discriminator scores or sharper spectrograms as proof of fidelity.
Do not use monitoring results for tuning, checkpoint selection, or stopping.
Do not use existing test sets.

8. BLINDED LISTENING
Preselect eight training and eight monitoring examples before training.

Generate:
- Original audio.
- Ground-truth mel → BigVGAN.
- Common starting checkpoint → BigVGAN.
- A final → BigVGAN.
- B final → BigVGAN.

Create randomized A/B comparisons with a separate key.
Ask for separate ratings of:
- Overall quality/artifacts.
- Transient sharpness.
- Pitch/timbre fidelity to the reference.
- Overall preference.

Use common playback gain, not independent loudness normalization.
Do not invent ratings.

Save comparable spectrograms with shared color scales and enlarged transient/harmonic regions.

9. TESTS AND EXECUTION
Test:
- Exact data/checkpoint identity.
- Identical initial generator weights.
- Correct gradient isolation for D and G.
- Conditioning alignment and valid-region handling.
- Loss formulas and lambda ramp.
- Exposure counts and matched sample order.
- Save/resume of both optimizers, discriminator, RNG, sampler, and schedule.
- No evaluation-state mutation.

Run unit tests and smoke tests in separate output directories.

Provide one runner, ACES Slurm scripts, exact commands, and measured runtime/allocation estimates.
Run fully only if a suitable accelerator is already allocated and available. Otherwise prepare submission commands without submitting paid jobs automatically.

10. REPORT
Distinguish:
- Better realism from better source fidelity.
- Training improvements from unseen-track improvements.
- Mel-error changes from listening outcomes.
- Equal generator exposure from unequal total compute.

An adversarial improvement does not establish recovery of previously inaccessible information.
A null result applies only to this discriminator, weighting, data pool, and budget.
Single-seed and reused-monitoring results are exploratory.

Deliver implementation, configs, tests, plots, audio, listening package, and a concise report stating exactly what ran.