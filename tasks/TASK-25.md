Implement TASK-25: architecture-versus-optimization ablation on the fixed TASK-24 N=256 fitting pool.

GOAL
Determine whether multi-example fitting improves through:
1. Lower learning rate.
2. Larger batch size.
3. Greater decoder width.
4. Learned temporal upsampling.

Keep MuQ features, targets, reconstruction loss, and data fixed. Do not add adversarial or latent-consistency losses.

1. INSPECT AND REUSE
Inspect TASK-22/23/24 implementations, configs, manifests, checkpoints, and metrics.

Reuse exactly:
- TASK-24's 256 cached training pairs.
- Its 16-crop unseen-track monitoring set.
- First-encoder-block MuQ features.
- Target normalization, masks, extraction context, and direct mel L1.
- Existing evaluation, plotting, audio, and ACES infrastructure.

Verify tensor hashes and the TASK-23 anchor.
Do not recompute features unnecessarily.
Do not add crops or change splits.

2. EXPERIMENTAL ARMS

Optimization factorial, using the CURRENT decoder:
A: batch 1, initial LR 1e-3.
B: batch 1, initial LR 1e-4.
C: batch 16, initial LR 1e-3.
D: batch 16, initial LR 1e-4.

Architecture comparisons, using D's optimization settings:
E: wider decoder, batch 16, initial LR 1e-4.
F: learned temporal upsampling, batch 16, initial LR 1e-4.

Comparisons:
- A versus B and C versus D isolate initial LR.
- A versus C and B versus D compare batch size at matched exposure.
- D versus E tests width.
- D versus F tests temporal upsampling.

Call D the candidate optimization recipe, not a proven improvement.
Do not select a recipe using monitoring performance.

3. ARCHITECTURE DEFINITIONS

E: Width only
- Double the existing hidden-channel width.
- Preserve depth, kernel sizes, dilation, receptive field, activation functions, and linear-interpolation method.
- Preserve input/output dimensions.
- Report actual parameter count and compute.

F: Upsampling only
- Preserve the current decoder's hidden width and convolutional trunk.
- Replace fixed linear interpolation with a learned temporal resampler.
- Use a shared local resampling rule, not a crop-specific dense time mapping or learned absolute-position table.
- Account explicitly for the noninteger input/output frame-rate ratio and frame-center alignment.
- Prefer a small local phase-conditioned resampler: predict interpolation weights from relative target-frame position, then combine neighboring source frames.
- Initialize to reproduce the existing linear interpolation where feasible.
- Document support, boundary handling, parameter count, and any receptive-field change.
- Do not add a wider/deeper trunk or other unrelated architectural changes.

Before full execution, document the exact F implementation and verify shapes, alignment, finite gradients, and initialization behavior.

If the proposed resampler is incompatible with the actual implementation, implement the closest minimal local learned replacement and explicitly document the deviation. Do not silently redesign the model.

4. INITIALIZATION AND FIXED SETTINGS
- A/B/C/D must start from identical decoder weights.
- Reuse TASK-24's initial state if available.
- E/F use the same initialization seed; document that different architectures cannot have identical complete weights.
- Copy compatible trunk weights from the common initial state into F where possible.
- Train every arm from scratch, never from a trained checkpoint.
- Freeze MuQ; use cached features.
- Use full precision.
- Preserve Adam settings other than the stated LR.
- Preserve the existing clipping policy. Do not silently add clipping.
- Do not add dropout, normalization, weight decay, augmentation, or input noise.

5. EXPOSURE-MATCHED PRIMARY BUDGET
Run every arm for exactly 819,200 crop presentations:
- 3,200 presentations per crop.
- Batch 1: 819,200 optimizer updates.
- Batch 16: 51,200 optimizer updates.

Use identical seeded shuffled dataset passes.
Group the same sample stream into batches.
Preserve sampler/RNG state across resumes.

Schedule LR by cumulative crop presentations:
- Initial LR through 204,800 presentations.
- Multiply by 0.1 at 204,800.
- Multiply by 0.1 again at 614,400.
- End at 819,200.

No early stopping.
No monitoring-driven schedule changes.

This is a new exposure-based schedule, not an exact TASK-24 reproduction. Treat TASK-24 as historical context, not a clean causal control.

IMPORTANT:
Matched exposure does not match optimizer-update count or compute.
Report all three. Do not claim batch-size comparisons isolate gradient noise alone.

6. SECONDARY MATCHED-UPDATE COMPARISON
From the same trajectories, compare all arms at:
- 10,000 optimizer updates.
- 50,000 optimizer updates.

Report exposure, LR phase, runtime, and training error at those points.
These comparisons have unequal exposure and may have different LR phases.
Do not present them as fully controlled substitutes for the primary comparison.

Do not extend arms opportunistically after inspecting outcomes.

7. EVALUATION AND CHECKPOINTS
Evaluate the entire 256-crop fitting pool in evaluation mode.

Primary metric:
- Direct pre-vocoder masked mel L1, aggregated by valid mel element.

Also report:
- Per-crop mean, median, p90, and maximum.
- TASK-23 anchor error.
- Interior and boundary errors.
- Temporal/frequency first-difference errors, diagnostic only.
- Monitoring-set L1 separately.
- Gradient norms, spike frequency, recovery duration, LR, throughput, runtime, and allocation charge.

Suggested cadence:
- Training logs every 1,000 presentations.
- Full fitting-set evaluation every 16,384 presentations.
- Monitoring evaluation every 65,536 presentations.
- Additional evaluations at LR boundaries and matched-update checkpoints.

Save:
- Initial state.
- Best training-set evaluation checkpoint.
- Final checkpoint.
- Resumable last.
- LR-boundary checkpoints.

Use training-set error only for best-checkpoint selection.
Report final checkpoints as the primary fixed-budget comparison.
Do not select on monitoring data.

Evaluation must not mutate model state or perturb training RNG.
Do not use validation/test sets.

8. ARTIFACTS
Produce:
- Fitting and monitoring curves versus presentations, optimizer updates, and wall time.
- Per-crop paired differences for the controlled arm comparisons.
- Error distributions and anchor-crop curves.
- Parameter counts, receptive fields, compute estimates, and measured throughput.
- Explicit LR-phase and spike annotations.

For the anchor and three examples selected before training:
- Target/prediction/error spectrograms.
- Enlarged transient and harmonic views.
- Original audio.
- Ground-truth mel → frozen BigVGAN.
- Best and final predicted mel → frozen BigVGAN.

Use consistent spectrogram scales and common playback gain.
Do not invent listening ratings.

9. TESTS AND EXECUTION
Test:
- Manifest/hash identity and monitoring disjointness.
- Identical A/B/C/D initial weights.
- Exact exposure counts and sample-stream grouping.
- Exposure-based LR schedule.
- Masked metric aggregation.
- Learned-resampler alignment, boundaries, gradients, and initialization.
- Checkpoint/resume, including RNG and sampler state.
- No evaluation-state mutation.

Run unit tests and short smoke tests for every architecture.

Estimate runtime and allocation charge from measured throughput before full execution.
Provide a single runner and compatible ACES Slurm scripts.
Respect verified walltime limits and prevent concurrent writers.

Run full experiments only if a suitable accelerator is already available in the current execution environment. Otherwise prepare exact submission commands, but do not submit paid cluster jobs automatically.

Do not overwrite previous artifacts.

10. INTERPRETATION
Separate observations from hypotheses.

- Lower LR improving fit supports an optimization explanation under this budget.
- Batch-size effects must be interpreted with unequal update counts at matched exposure.
- E improving over D supports a width/capacity contribution, but width also changes optimization and compute.
- F improving over D supports the learned resampling parameterization; it does not prove interpolation was the sole bottleneck.
- Lower training error without better monitoring performance is a fitting improvement, not a demonstrated generalization improvement.
- A null result does not establish representation information loss or exhausted architectural capacity.
- Single-seed results are exploratory, not seed-robust conclusions.

Do not describe a low-LR plateau as proof of convergence.
Do not attribute spikes to a specific mechanism without an intervention that tests it.

DELIVER
- Minimal implementation/config changes.
- Tests and exact arm definitions.
- Resource estimate and runnable commands.
- Machine-readable metrics, plots, audio, and analysis from completed runs.
- Clear distinction between executed and prepared work.
