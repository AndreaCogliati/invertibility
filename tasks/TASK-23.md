Implement TASK-23: a runnable one-crop decoder-overfit diagnostic for the MuQ reconstruction project. 

GOAL
Determine whether the current TASK-22 decoder can reproduce one fixed target mel-spectrogram from its fixed first-encoder-block MuQ features.

This tests fitting ability, not invertibility. Do not introduce new losses or redesign the decoder in this experiment.

1. INSPECT AND REUSE
Inspect TASK-22 code, configs, feature extraction, normalization, loss, masking, checkpoints, and ACES execution scripts.

Reuse existing components. Verify all assumptions against the code rather than relying on task descriptions.

2. FIX ONE INPUT–TARGET PAIR
Select one existing TRAINING crop with both harmonic structure and a clear transient, using the normal crop duration. Do not select from validation or test.

Cache:
- Frozen first-encoder-block MuQ features.
- Target mel in the exact representation used by the training loss.
- Valid-frame mask.
- Original waveform.
- Crop timestamps, track identifier, extraction settings, normalization metadata, and tensor hashes.

Preserve TASK-22 extraction context: do not switch between whole-track and isolated-crop feature extraction.

Disable random cropping, augmentation, input noise, and stochastic feature extraction. Every update must use the identical cached pair.

Verify:
- Shapes, finite values, nonempty valid mask, and target dynamic range.
- Exact first-block identity.
- Feature and target frame timestamps, padding, centering, and crop boundaries.
- Normalization and inverse normalization.
- Target compatibility with BigVGAN.

Document any unresolved alignment uncertainty. Matching tensor lengths alone is not proof of alignment.

3. THREE CONTROLS

A. FREE TARGET-TENSOR CONTROL
Optimize a freely trainable tensor with the target mel's shape using the same masked direct mel L1 implementation.

Initialize near the target's global mean with small seeded noise, not at the target.

Use Adam with zero weight decay. Select and document a suitable learning rate. If convergence stalls, allow a documented learning-rate reduction.

This bypasses the decoder and tests loss, masking, gradient flow, and optimizer plumbing. It does not validate feature alignment.

B. EXACT DECODER RECIPE
Initialize a fresh TASK-22 decoder using its original seed.
Preserve its architecture, optimizer, learning rate, regularization, clipping, and reconstruction loss.
Use batch size one and the fixed cached pair.

C. UNREGULARIZED DECODER
Start from the SAME initial model weights as B.
Keep the architecture, loss, optimizer, and initial learning rate unchanged.
Disable dropout, weight decay, and other explicit training regularizers.
List exactly what was disabled.
Do not remove architectural layers or silently change normalization behavior.

For B and C:
- Keep MuQ frozen and in evaluation mode.
- Train decoder parameters only.
- Use full precision initially, with autocast and gradient scaling disabled.
- Do not load trained decoder weights.
- Do not add adversarial, latent-consistency, gradient-detail, or other losses.
- Do not silently replace normalization layers.

4. TRAINING AND EXTENSION RULES
B and C:
- Run 20,000 updates with no validation-based stopping.
- Log every 100 updates.
- Save predictions and checkpoints at initialization, 100, 1,000, 5,000, and 20,000 updates.
- If evaluation-mode L1 remains above 1% of that run's initial evaluation-mode L1, continue to 50,000 updates.
- For that extension, reduce the learning rate by 10x at step 20,000.
- Save additional outputs at 50,000.
- Treat the 1% threshold as a provisional fitting criterion, not a perceptual threshold.
- If C still fails, repeat C with one additional initialization seed.

A:
- Run up to 20,000 updates, stopping earlier if the provisional fitting criterion is reached.
- Record the achieved numerical floor and convergence history.
- If A fails, investigate and report the control failure before interpreting B/C.

Stop for non-finite values or genuine execution failures, preserving diagnostic logs.

5. TRAIN/EVAL MODE CHECKS
At each reporting point, measure direct mel L1 in:
- Training mode.
- Evaluation mode.

Prevent diagnostic forward passes from changing persistent model state, including batch-normalization running statistics. Preserve RNG state around diagnostic passes so monitoring does not change the training trajectory.

Inspect and document batch-dependent normalization behavior at batch size one. Do not silently alter it.

6. METRICS
Primary:
- Direct predicted-mel versus target-mel masked L1 BEFORE vocoding.
- Absolute error and reduction relative to initialization.

Also report:
- Interior-frame and boundary-frame L1 separately.
- A documented boundary definition based on available context/receptive field where feasible.
- Temporal first-difference error.
- Frequency first-difference error.
- Gradient norm, learning rate, runtime, and non-finite counts.

Difference metrics are diagnostic only, not training losses.
Apply valid masks correctly to adjacent-frame/bin pairs.

Do not confuse direct mel error with waveform-to-mel round-trip error.

7. OUTPUTS
For every run, save:
- Machine-readable metrics and config.
- Best and final checkpoints.
- Target, prediction, and absolute-error spectrograms.
- Target/prediction plots with identical axes and shared color scales.
- Error plots with a separately labeled error scale.
- Enlarged transient and harmonic-region views.
- Learning curves for direct L1 and detail metrics.

Generate audio using the existing frozen BigVGAN:
- Original waveform.
- Ground-truth mel → BigVGAN.
- Best predicted mel → BigVGAN.
- Final predicted mel → BigVGAN if different.

Invert normalization correctly before vocoding. Do not independently loudness-normalize comparison clips in a way that conceals amplitude errors. Document any common playback gain.

Do not invent listening ratings or claim perceptual success without human feedback.

8. TESTS AND EXECUTION
Implement tests for:
- Cached-pair determinism and hashes.
- Masked-loss correctness, including padded frames.
- Gradient flow to the free tensor and decoder.
- Frozen MuQ parameters.
- Identical initial weights for B/C.
- No state mutation during diagnostic mode comparisons.
- Checkpoint save/resume.

Run unit tests and a short smoke test in a separate output directory.

Provide a single entry point to run the complete diagnostic and a compatible ACES Slurm script. Reuse the working environment and respect verified resource limits.

Run the full diagnostic if a suitable accelerator is already available in the current execution environment. Otherwise prepare exact submission commands, but do not submit paid cluster jobs automatically.

Do not overwrite previous task artifacts.

9. REPORT
Write a concise results report separating observations from hypotheses.

Interpretation:
- A fails: loss/optimization plumbing remains unresolved.
- B fails but C succeeds: regularization or training-mode behavior limits fitting.
- B/C succeed: the decoder can memorize this pair; this does not establish transferable reconstruction or invertibility.
- A succeeds but B/C plateau: decoder parameterization, normalization, upsampling, or optimization needs investigation; do not conclude information loss.
- Predicted mel is nearly exact but audio remains poor: compare with ground-truth-mel audio to isolate vocoder limitations.

Explicitly note:
- One-pair memorization cannot establish information preservation.
- Success cannot establish correct temporal alignment.
- Failure cannot establish irrecoverable information loss.
- This experiment does not test whether more training data would improve generalization.

Deliver the implementation, tests, runnable commands, configs, artifacts from any completed runs, and a clear statement of what was executed versus only prepared.
