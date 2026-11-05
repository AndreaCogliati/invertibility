Implement TASK-24: fixed-crop decoder fitting ladder.

GOAL
Determine where the current first-block MuQ → mel decoder loses its ability to fit training examples as the number of fixed pairs increases.

TASK-23 established successful one-pair memorization. It did NOT rule out capacity, architecture, optimization, alignment, or representation ambiguity as limitations in multi-example reconstruction.

1. INSPECT AND REUSE
Inspect TASK-22 and TASK-23 code, configs, cached pairs, metrics, and execution infrastructure.

Reuse:
- TASK-22 first-encoder-block features and target-mel convention.
- TASK-23 direct masked mel L1, diagnostic plots, audio generation, and checkpoint handling.
- The existing decoder architecture and optimizer.

Verify actual implementation details. Do not assume task descriptions are exact.

2. NESTED DATASETS
Create fixed nested training subsets with:
N = 1, 4, 16, 64, 256 crops.

Requirements:
- N=1 must be the exact TASK-23 cached pair.
- Every smaller subset must be contained in every larger subset.
- Select additional crops deterministically from TASK-22 TRAINING data only.
- Prefer distinct tracks for additional crops, with reasonable genre coverage.
- Do not choose examples based on decoder performance.
- Preserve crop duration, extraction context, normalization, padding, and masks.
- Reuse existing cached features and targets rather than recomputing them unnecessarily.
- Save manifests, track IDs, timestamps, tensor hashes, and selection seed.
- Verify cache identity and subset nesting.

Use no random crops, augmentation, input noise, or changing targets.

3. MONITORING SET
Select a small fixed set of 16 crops from training tracks not represented in the 256-crop fitting pool.

Use this set only to monitor transfer to unseen tracks:
- No checkpoint selection.
- No scheduler decisions.
- No stopping decisions.

Do not use existing validation/test sets for this diagnostic.
Call this a monitoring set, not an untouched final test set.

4. CONTROLLED TRAINING
For each N:
- Initialize a fresh decoder from the SAME initial state used for TASK-23 B, if recoverable.
- Otherwise generate one documented initial state and use it for ALL ladder arms.
- Never warm-start a larger subset from a smaller subset.
- Keep architecture, direct mel L1, optimizer, initial learning rate, and clipping policy unchanged from TASK-23 B.
- Use batch size one for every arm.
- Use full precision, matching TASK-23.
- Keep MuQ frozen and in evaluation mode.
- Do not add losses, regularization, fresh crops, or architectural changes.

Sample with seeded shuffled passes through each subset, without replacement within each pass.
Record actual presentation counts per crop and preserve sampler state across resumes.

Do not create a redundant “unregularized” arm: TASK-23 found nothing to disable.

5. FIXED OPTIMIZATION BUDGET
Run every arm for 200,000 optimizer updates, without early stopping.

Use the same predefined learning-rate schedule for every arm:
- Initial TASK-23 learning rate through step 50,000.
- Multiply by 0.1 at step 50,000.
- Multiply by 0.1 again at step 150,000.

This schedule is a deliberate diagnostic change from TASK-23.
Do not interpret comparisons with TASK-23 as a pure dataset-size comparison.

Do not adjust the schedule based on monitoring-set results.

Save checkpoints at:
- 20,000
- 50,000, immediately before the first LR reduction
- 100,000
- 150,000, immediately before the second LR reduction
- 200,000
- Best training-set evaluation L1
- Resumable last

Keep checkpoint storage bounded.

6. MATCHED EXPOSURE AND MATCHED STEPS
Equal optimizer steps do not provide equal exposure per crop.

Provide BOTH:
A. Matched-step comparisons across all N.
B. Matched-exposure comparisons at 100 and 500 complete presentations per crop.

At batch size one, capture the exposure checkpoints after exactly:
- 100 × N updates
- 500 × N updates

Save their metrics and predictions. Full model checkpoints at these points are optional if storage is constrained.

Explain that matched exposure also entails different numbers of optimizer updates and potentially different learning-rate phases. Neither comparison alone isolates every factor.

7. EVALUATION
Evaluate the ENTIRE fitting subset, not just the most recent training example.

Use evaluation mode and aggregate masked absolute error by valid mel element.
Also report the distribution of per-crop L1:
- Mean
- Median
- 90th percentile
- Maximum
- TASK-23 anchor-crop error

Report:
- Direct pre-vocoder mel L1.
- Interior and boundary errors.
- Temporal and frequency first-difference errors, for diagnosis only.
- Monitoring-set L1 separately.
- Gradient norm, LR, runtime, throughput, and presentation counts.

Suggested cadence:
- Training logs every 100 updates.
- Full fitting-set evaluation every 1,000 updates.
- Monitoring-set evaluation every 5,000 updates.
- Additional evaluations at all required comparison checkpoints.

Ensure diagnostic passes do not modify model state or perturb training RNG.

Track spikes and recovery, but do not attribute their cause without evidence.
Stop only for genuine failures such as non-finite state.

8. PLOTS AND AUDIO
Produce:
- Training-set L1 versus optimizer steps.
- Training-set L1 versus presentations per crop.
- Monitoring-set curves, clearly separated.
- Per-crop error distributions versus N.
- Anchor-crop error versus N.
- LR-change and spike annotations.

For every N, save target/prediction/error spectrograms for:
- The TASK-23 anchor.
- Up to three additional examples selected in advance.

Use consistent target/prediction color scales and separately labeled error scales.
Include enlarged transient and harmonic regions.

Generate audio from best and final predictions using frozen BigVGAN.
Include original audio and ground-truth-mel vocoding references.
Use consistent playback gain, not independent loudness normalization.

Do not invent listening ratings.

9. TESTS
Test:
- Nested manifests and split disjointness.
- Cached-pair hashes and exact anchor reuse.
- Identical initial model weights across arms.
- Sampler coverage and exposure counts.
- Masked metrics and per-crop aggregation.
- Frozen MuQ parameters.
- LR schedule and checkpoint timing.
- Save/resume, including sampler and RNG state.
- No model-state mutation during evaluation.

Run unit tests and a smoke test in a separate output directory.

10. EXECUTION
Provide one entry point for the full ladder and compatible ACES Slurm scripts.

Estimate runtime and allocation charge from measured smoke-test throughput and verified billing settings.
Respect actual partition walltime limits and support resumable segments.
Never allow concurrent writers to the same arm.

Run the full ladder if a suitable accelerator is already available in the current execution environment.
Otherwise prepare exact submission commands, but do not submit paid cluster jobs automatically.

Do not overwrite prior task artifacts.

11. REPORT AND INTERPRETATION
Separate observations from hypotheses.

Key questions:
- Does absolute training error rise with N?
- Does the original anchor become harder to fit as other examples are added?
- Are errors broad-based or concentrated in a few crops?
- Do predefined LR reductions materially improve fitting?
- Is the remaining limitation training fit, transfer, or both?

Interpret cautiously:
- Low training error with poor monitoring performance suggests a transfer problem, not necessarily insufficient data alone.
- Increasing training error with N motivates capacity/architecture and optimization ablations; it does not identify which is responsible.
- A plateau does not establish information loss.
- Successful fitting does not establish invertibility or correct alignment.
- This is one nested sample sequence and one initialization, not a seed-robust scaling law.

Report absolute errors prominently. Do not rely on “percent of initial error,” which depends on initialization.

Deliver:
- Minimal implementation/config changes.
- Tests and manifests.
- Runnable commands and Slurm scripts.
- Resource estimate.
- Machine-readable metrics, plots, audio, and report from completed runs.
- A clear statement of what was executed versus only prepared.
