Implement TASK-21: extended N512 continuation, with a 48-hour training budget and no validation-based early stopping.

Goal
Determine whether TASK-20 stopped during a recoverable transient, and whether substantially longer training improves the current translator. This is a duration experiment, not an architecture or optimizer experiment.

First inspect TASK-20 code, configuration, checkpoints, manifests, and logs. Reuse existing infrastructure.

1. Exact continuation
- Resume from TASK-20's LAST checkpoint at step 4250, not its best checkpoint.
- Restore model, optimizer, scaler, scheduler if present, global step, RNG states, and sampler state where supported.
- Explicitly document any state that was not saved and cannot be restored.
- Preserve the existing best checkpoint at step 2250 as the incumbent.
- Keep all 512 training tracks, splits, crops, architecture, representation, preprocessing, loss, optimizer, learning rate, batch size, precision, and sampling policy unchanged.
- Do not silently reset optimizer state, introduce decay, or change gradient clipping.

2. Training budget
- Disable patience-based early stopping completely.
- Target 48 cumulative hours of additional training-job runtime across resumable segments.
- Inspect actual ACES partition/QOS walltime limits before preparing submission scripts. Do not assume a single 48-hour job is allowed.
- If necessary, use sequential resumable jobs, never concurrent writers to the same run.
- Stop cleanly before each job's walltime limit and save a resumable checkpoint.
- Remove the old 20,000-step ceiling. Use the runtime budget as the intended stopping condition.
- Stop early only for non-finite loss/gradients, corrupted state, or another genuine execution failure, not a plateau or transient spike.
- Report the estimated allocation charge using verified billing settings before submission.

3. Validation and checkpoint selection
- Use ONLY primary validation correct-conditioning mel L1 for checkpoint selection.
- Validate every 250 steps through global step 20,000, then every 1,000 steps.
- Use the same fixed validation crops and metric implementation throughout.
- Keep the best checkpoint across TASK-20 and this continuation.
- Save resumable last checkpoints periodically and before graceful shutdown.
- Keep bounded storage: best, last, a small rolling recovery set, and sparse milestone checkpoints. Do not save every validation checkpoint indefinitely.
- Do not evaluate test/test2 during training or use val2, listening, or test results to change the run.

4. Monitoring and diagnostics
- Continue logging to W&B with clear TASK-20 lineage and no overwritten history.
- Log training loss, primary validation loss, learning rate, gradient norm, throughput, elapsed runtime, global step, and best step.
- Track effective dataset passes and sampling coverage using the actual sampling implementation. Distinguish optimizer steps from exposure per track.
- Record whether validation spikes coincide with training-loss/gradient changes, data-order boundaries, checkpoint resumes, or other logged events. Do not assume a cause.
- Keep expensive diagnostic evaluation sparse so it does not dominate training.
- Handle temporary W&B connectivity failures without losing local logs or training progress.

5. Verification
- Test checkpoint save/resume and graceful shutdown in a separate smoke-test output directory.
- Verify that optimizer state and global step survive resumption.
- Verify unchanged manifests and configuration, apart from duration, stopping, logging, and checkpoint-management settings.
- Do not overwrite TASK-20 artifacts.

6. Final evaluation
- Evaluate the overall validation-selected best checkpoint and the final checkpoint using existing TASK-20 metrics and controls.
- Compare with TASK-20 best, A128, and A32 on identical evaluation crops.
- Report validation, test, val2, and test2 separately.
- Retain paired track-clustered confidence intervals, while noting that they do not measure seed variability.
- Describe test sets as reused evaluation sets, not untouched final confirmation sets.
- Plot the full continuation and the early-training region separately, marking the original best, any new best, spikes, and resumes.
- Report best step, final step, runtime, allocation charge, effective dataset passes, and improvement over TASK-20.
- If no improvement occurs, conclude only that extended training with this unchanged recipe did not improve the selected result. Do not claim a MuQ information ceiling or exhausted translator capacity.

Deliver
- Minimal code/config changes and tests.
- ACES submission/resumption scripts.
- Exact submission commands and verified budget estimate.
- A concise TASK-21 specification and reporting template.

Prepare everything, but do not submit automatically.
