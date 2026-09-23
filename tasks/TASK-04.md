# TASK-04: Spectrogram Gate-3 Memorization Control with Weights & Biases

## Required Reading

Before modifying or running anything, read:

- `CLAUDE.md`
- `EXPERIMENT_SYNTHESIS.md`
- `notes/DECISIONS.md`
- `notes/MEMORY.md`
- `notes/NEXT.md`
- The implementation and reports for TASK-01 and TASK-02
- The original Gate-3 memorization experiment
- The successful two-excerpt MuQ→DAC memorization experiment, including its exact manifest, training configuration, checkpoint-selection logic, and evaluation procedure

Inspect the existing code before deciding whether any new files are necessary.

---

# Objective

Run a minimal **Spectrogram Gate-3 memorization experiment**.

Replace the MuQ input in the successful tiny memorization experiment with the mel-spectrogram input implemented for TASK-01 and TASK-02:

```text
Mel spectrogram
        ↓
SpectrogramTranslator
        ↓
Post-RVQ summed DAC latent
        ↓
Frozen DAC decoder
        ↓
Audio
```

Use only the same two fixed excerpts used by the corresponding successful MuQ memorization control, if those exact excerpts and their manifest remain available.

The central question is:

> Can the current Spectrogram→DAC pipeline memorize two fixed training excerpts nearly perfectly?

This is a training-set memorization test, not a generalization experiment.

---

# Scientific Motivation

The project has established the following:

1. A small temporal ConvNet can memorize two fixed MuQ→DAC examples nearly perfectly.
2. Spectrogram→DAC produces modestly better latent-space validation metrics than MuQ→DAC on GTZAN.
3. Spectrogram→DAC nevertheless exhibits severe norm collapse, variance collapse, temporal over-smoothing, and approximately zero waveform correlation on held-out examples.
4. Increasing GTZAN coverage from approximately 13.3% to approximately 99.94% did not improve those failure modes.
5. The best validation-MSE checkpoint in TASK-02 occurred extremely early in training.

Before changing the target, loss, architecture, or quantization strategy, determine whether the current spectrogram pipeline can fit the training examples themselves.

A mel spectrogram is a highly informative input. Complete or near-complete memorization of two fixed examples should therefore be an easy positive control for a sufficiently expressive and correctly aligned pipeline.

---

# Hypotheses

## Hypothesis H1: The pipeline can memorize

If Spectrogram→DAC memorizes the two excerpts nearly perfectly, then:

- The spectrogram extraction and alignment are usable.
- The translator has enough capacity for the tiny problem.
- The post-RVQ DAC target can be fitted at least sample-specifically.
- The frozen DAC decoder path works.
- The TASK-01 and TASK-02 failures are primarily failures of generalization, optimization at scale, checkpoint selection, or some interaction among them.

This result would not prove that the post-RVQ target is a good generalizable regression target.

## Hypothesis H2: The pipeline cannot memorize

If Spectrogram→DAC cannot memorize the two excerpts, then the failure cannot reasonably be attributed to validation generalization or insufficient dataset diversity.

Investigate:

- Spectrogram/DAC temporal alignment
- Input/target crop correspondence
- Spectrogram preprocessing
- The learned 128→1024 projection
- Translator implementation
- Target extraction
- Loss computation
- Gradient flow
- Checkpoint restoration
- Evaluation and audio export
- Any implementation difference from the successful MuQ memorization control

Do not launch larger experiments until the failure is understood.

---

# Experimental Design

## Dataset

Use exactly two fixed excerpts.

Prefer the exact two excerpts used in the successful MuQ→DAC memorization experiment.

Do not randomly resample excerpts at every epoch or step.

Each training step must present one of the same two fixed input-target pairs.

Record:

- Source track identifier
- Crop start position
- Crop duration
- Sample rate
- Number of audio samples
- Number of spectrogram frames
- Number of DAC target frames
- Exact manifest path
- Hash of the manifest
- Hash or stable identifier for each cached example

There is no meaningful validation or test split for the primary memorization question.

If the existing training loop requires validation, use a clearly labeled diagnostic evaluation of the same two examples. Do not describe this as held-out validation.

Optionally evaluate on other crops after memorization, but label them strictly as out-of-sample diagnostics. Generalization is not a success criterion for this task.

---

## Input Representation

Reuse exactly the mel-spectrogram implementation and configuration from TASK-01 and TASK-02.

Do not change:

- Number of mel bins
- FFT size
- Window length
- Hop length
- Window function
- Frequency range
- Log or amplitude transformation
- Spectrogram normalization
- Padding policy
- Temporal alignment policy

Reuse the existing spectrogram cache when the exact excerpts are already present and verified.

If a new tiny cache is required, generate only what is needed for this experiment.

Before training, verify numerically that:

- Each spectrogram corresponds to the intended audio crop.
- Each DAC target corresponds to the same audio crop.
- Input and target frames are aligned according to the TASK-01/TASK-02 policy.
- No unintended offset, truncation, padding mismatch, or crop mismatch is present.

Report the resulting input and target tensor shapes.

---

## Target Representation

Use exactly the same target used in:

- The successful MuQ memorization experiment
- TASK-01
- TASK-02

That target should remain the current post-RVQ summed DAC latent unless inspection of the existing code shows otherwise.

Do not change the target definition.

Do not introduce:

- Pre-quantizer targets
- Discrete code prediction
- Re-quantization
- Target standardization
- Per-channel normalization
- PCA
- Latent clipping
- Alternative DAC layers

---

## Architecture

Reuse the existing `SpectrogramTranslator` from TASK-01 and TASK-02 unchanged.

This includes the existing learned mel-channel projection required to map the spectrogram input dimension to the translator input dimension.

Do not add:

- Attention
- Transformers
- U-Nets
- Skip connections
- Larger hidden dimensions
- Larger receptive fields
- Per-example embeddings
- Track identifiers
- Positional lookup tables designed to memorize crop identity

The translator must infer the DAC target from the spectrogram sequence, not from an explicit example identifier.

Report:

- Total parameter count
- Trainable parameter count
- Receptive field
- Input shape
- Output shape

---

## Loss

Reuse exactly the current Spectrogram→DAC loss from TASK-01 and TASK-02:

```text
total loss = MSE component + cosine component
```

Preserve the same weights and implementation.

Do not introduce:

- Target normalization
- Quantization loss
- Adversarial loss
- Spectral loss
- Waveform loss
- Perceptual loss
- Codebook loss
- Temporal derivative loss
- Variance regularization
- Norm matching
- Weight changes between the existing loss components

Log the two loss components separately.

---

## Optimizer and Training Configuration

Reproduce the successful MuQ tiny memorization experiment as closely as possible while retaining the established spectrogram model and loss.

Use the successful Gate-3 configuration for:

- Optimizer
- Learning rate
- Batch construction
- Number of steps
- Evaluation interval
- Checkpoint interval
- Random seed, where supported
- Gradient-reset behavior
- Model initialization policy

If TASK-01/TASK-02 and the original Gate-3 run differ, explicitly document the difference and choose the configuration that best preserves the purpose of replicating Gate-3.

Do not silently tune hyperparameters.

Because the dataset contains only two fixed samples, ensure that both samples are presented evenly. Log presentation counts for each example.

Do not scale the training budget according to GTZAN crop counts. This should remain a small memorization experiment.

---

# Pre-Training Integrity Checks

Before optimization, run assertions or tests confirming:

1. Both examples can be loaded repeatedly without changing.
2. Spectrogram caching is deterministic.
3. DAC target caching is deterministic.
4. Input and target refer to the same crop.
5. Input and target sequence lengths satisfy the expected alignment policy.
6. The model output shape exactly matches the target shape.
7. Loss values are finite.
8. Gradients reach:
   - The mel projection
   - The temporal translator
   - The output layer
9. Frozen DAC components remain frozen.
10. Decoding the oracle target yields the expected DAC oracle audio.
11. Saving and reloading a checkpoint reproduces identical predictions within numerical tolerance.

Fail early if any integrity check fails.

Do not proceed to a long training loop with an unverified alignment or checkpoint path.

---

# Weights & Biases Integration

Integrate W&B as a first-class logger, but retain local logs and artifacts so the experiment remains usable if W&B is unavailable.

## Project Organization

Use:

```text
project: muq-invertibility
group: spectrogram-gate3
job_type: memorization-control
```

Use a run name similar to:

```text
spec_gate3_two_excerpt_<timestamp>
```

Add tags such as:

```text
spectrogram
gate3
memorization
positive-control
two-excerpt
post-rvq-target
```

If the repository already defines W&B naming conventions, follow them instead and document the mapping.

---

## W&B Configuration

Store in the W&B run configuration:

- Task identifier
- Git commit hash
- Dirty working-tree status
- Hostname
- Operating system
- Hardware backend
- Device type
- Python version
- PyTorch version
- DAC package/model version
- W&B version
- Random seed
- Source manifest path
- Manifest hash
- Number of examples
- Crop duration
- Sample rate
- Spectrogram parameters
- Input tensor shape
- Target tensor shape
- Model parameter count
- Optimizer
- Learning rate
- Loss weights
- Number of training steps
- Evaluation interval
- Checkpoint interval

Do not log credentials, API keys, access tokens, usernames, or private filesystem details that are unnecessary for reproduction.

---

## W&B Operating Modes

Support at least:

```text
WANDB_MODE=online
WANDB_MODE=offline
WANDB_MODE=disabled
```

The run must not fail merely because W&B authentication or network access is unavailable.

In offline or disabled mode:

- Preserve all metrics locally.
- Preserve all generated audio locally.
- Preserve the final report locally.
- Clearly record the logging mode.

Do not hard-code a W&B API key.

---

# Required W&B Metrics

Use consistent metric namespaces.

## Optimization Metrics

Log at an appropriate training interval:

```text
train/total_loss
train/mse_loss
train/cosine_loss
train/learning_rate
train/example_0_presentations
train/example_1_presentations
```

Do not log every scalar at every step if that creates excessive overhead. Use a documented interval while retaining enough early-step resolution to inspect rapid convergence.

For the beginning of training, use finer-grained logging so that the initial trajectory is visible.

---

## Gradient Diagnostics

Log:

```text
grad/global_l2_norm
grad/global_max_abs
grad/mel_projection_l2_norm
grad/translator_l2_norm
grad/output_layer_l2_norm
```

Also log:

```text
update/model_parameter_l2_norm
update/update_to_parameter_ratio
```

if these can be computed cleanly without materially changing training.

Avoid W&B full-parameter watching on every step if it imposes substantial overhead. Prefer explicit scalar gradient statistics. If `wandb.watch()` is used, use a conservative logging frequency and document it.

Record:

- Nonfinite gradient count
- Nonfinite parameter count
- Nonfinite prediction count

---

## Per-Example Memorization Metrics

Evaluate the two fixed examples separately.

For each example, log:

```text
memorization/example_0/mse
memorization/example_0/cosine
memorization/example_0/norm_ratio
memorization/example_0/variance_ratio
memorization/example_0/lag1_autocorrelation_prediction
memorization/example_0/lag1_autocorrelation_oracle
memorization/example_0/frame_difference_mse
memorization/example_0/waveform_correlation_to_oracle
memorization/example_0/stft_distance_to_oracle
```

and the equivalent metrics for `example_1`.

Also log averages:

```text
memorization/mean_mse
memorization/mean_cosine
memorization/mean_norm_ratio
memorization/mean_variance_ratio
memorization/mean_waveform_correlation_to_oracle
memorization/mean_stft_distance_to_oracle
```

Use the same metric implementations as TASK-01 and TASK-02.

Do not silently change metric definitions.

---

## Prediction and Target Statistics

For each example and for their aggregate, log:

- Prediction mean
- Prediction standard deviation
- Target mean
- Target standard deviation
- Mean prediction frame norm
- Mean target frame norm
- Prediction variance
- Target variance
- Norm ratio
- Variance ratio
- Prediction frame-to-frame difference norm
- Target frame-to-frame difference norm
- Prediction lag-1 autocorrelation
- Target lag-1 autocorrelation

These metrics are central because prior runs showed low-norm, low-variance, highly autocorrelated predictions.

---

## Training-Set Audio

For both fixed examples, log:

1. Original reference audio
2. DAC oracle reconstruction
3. Current Spectrogram→DAC reconstruction
4. Untrained-model reconstruction or step-0 reconstruction
5. Shuffled-input control, if this is already supported by the existing evaluation pipeline

Use the same audio normalization policy as TASK-01 and TASK-02.

Log audio at:

- Step 0
- Early training
- Intermediate training
- Best training-MSE checkpoint
- Final checkpoint

Select a small, fixed set of informative steps rather than uploading audio at every evaluation.

Attach clear captions containing:

- Example identifier
- Checkpoint step
- Audio type

Preserve WAV files locally in addition to logging them to W&B.

---

## Visualizations

Log or construct the data needed for the following comparisons:

### Learning curves

- Training MSE versus step
- Training cosine versus step
- Total loss versus step
- Per-example MSE versus step
- Per-example cosine versus step

### Collapse diagnostics

- Norm ratio versus step
- Variance ratio versus step
- Prediction autocorrelation versus step
- Frame-difference norm versus step

### Optimization diagnostics

- Global gradient norm versus step
- Update-to-parameter ratio versus step
- Learning rate versus step

### Metric relationships

- MSE versus norm ratio
- MSE versus variance ratio
- Cosine similarity versus waveform correlation
- MSE versus waveform correlation

The purpose is to determine whether low MSE corresponds to meaningful DAC reconstruction or merely to a low-energy average prediction.

---

# Checkpoint Policy

Save at least:

- Initial checkpoint at step 0
- Best checkpoint by mean training MSE
- Best checkpoint by mean training cosine
- Best checkpoint by waveform correlation to the DAC oracle, if the metric is sufficiently stable
- Final checkpoint

Because this is a memorization experiment, do not select the primary checkpoint using held-out validation MSE.

For each saved checkpoint, record:

- Global step
- Total loss
- MSE
- Cosine
- Norm ratio
- Variance ratio
- Waveform correlation
- STFT distance
- Checkpoint file hash

Log checkpoints as W&B artifacts when W&B is enabled.

Always preserve checkpoints locally.

After saving, reload each primary checkpoint and verify that its reported metrics are reproducible.

---

# Success Criteria

The primary success criterion is near-perfect fitting of the two fixed training examples.

Compare directly with the successful MuQ memorization experiment.

Do not declare success based only on a modest reduction in MSE.

## Strong Success

A strong success should show most or all of the following:

- Training MSE approaches the successful MuQ memorization result.
- Training cosine approaches 1.
- Prediction norm approaches oracle norm.
- Prediction variance approaches oracle variance.
- Prediction temporal dynamics approach oracle dynamics.
- Waveform correlation to the DAC oracle becomes high.
- Decoded audio closely resembles the DAC oracle by listening.
- Both excerpts are memorized, rather than one excerpt dominating the average.

Interpretation:

```text
The spectrogram pipeline can fit the post-RVQ DAC latent target
on a tiny fixed dataset.

The failures in TASK-01 and TASK-02 are therefore not caused by
a basic inability to optimize or decode the training examples.

The next investigation should focus on why this mapping does not
generalize or remain stable at larger dataset scale.
```

## Partial Success

Possible pattern:

- Very low training MSE
- High cosine
- Persistent norm or variance mismatch
- Poor waveform correspondence
- Audible reconstruction substantially worse than the DAC oracle

Interpretation:

```text
The optimization objective can fit its own latent-space metrics
without recovering an audio-equivalent DAC trajectory.

The loss and evaluation criteria may be misaligned with decoded
audio quality.
```

Do not report this as complete pipeline validation.

## Failure

Failure includes:

- Training metrics plateau far from the successful MuQ memorization result.
- Severe norm or variance collapse remains.
- Predictions remain extremely smooth.
- Waveform correlation remains approximately zero.
- Decoded training audio remains unrecognizable or unrelated.
- Only one of the two examples is learned.
- Performance depends on an early lucky checkpoint and subsequently degrades.

Interpretation:

```text
The Spectrogram→DAC implementation cannot reproduce the existing
tiny memorization control.

Do not launch larger training runs.

Investigate alignment, preprocessing, target construction,
optimization, checkpointing, and implementation differences.
```

---

# Required Comparisons

Produce a direct comparison among:

1. Successful MuQ Gate-3 memorization
2. Spectrogram Gate-3 memorization
3. TASK-01 sparse-GTZAN Spectrogram→DAC
4. TASK-02 dense-GTZAN Spectrogram→DAC

Use a table containing at least:

| Metric | MuQ Gate-3 | Spectrogram Gate-3 | TASK-01 | TASK-02 |
|---|---:|---:|---:|---:|
| Number of training crops | | | | |
| Total unique audio duration | | | | |
| Best checkpoint step | | | | |
| Training MSE | | | | |
| Training cosine | | | | |
| Norm ratio | | | | |
| Variance ratio | | | | |
| Prediction lag-1 autocorrelation | | | | |
| Oracle lag-1 autocorrelation | | | | |
| STFT distance to oracle | | | | |
| Waveform correlation to oracle | | | | |

Do not compare a training metric from one experiment against a validation metric from another without labeling the distinction clearly.

---

# Implementation Constraints

Prefer reusing:

- `spectrogram_encoder.py`
- `latent_diagnostics.py`
- `models/spectrogram_translator.py`
- Existing cache utilities
- Existing audio comparison functions
- Existing TASK-01/TASK-02 metric implementations
- Existing W&B utilities, if TASK-03 already introduced any reusable instrumentation

Avoid duplicating metric implementations.

If W&B integration is not yet present in the working tree, isolate it in a small reusable logging module rather than scattering direct logging calls throughout the training code.

All W&B-specific code must be optional.

Existing tests must continue to pass.

Add tests for:

- W&B disabled mode
- Metric logging payload construction
- Deterministic two-example loading
- Checkpoint save/reload consistency
- Per-example metric separation
- Input-target alignment assertions

Run:

- Full test suite
- Ruff
- Any repository-standard type or formatting checks specified in `CLAUDE.md`

---

# Scope Limits

Do not:

- Train on full GTZAN
- Train on FMA
- Run a layer sweep
- Change the spectrogram representation
- Change the DAC target
- Normalize the target
- Add quantizer-aware training
- Predict discrete codes
- Increase model capacity
- Tune multiple learning rates
- Introduce a scheduler
- Add gradient clipping unless an existing Gate-3 configuration already used it
- Commit changes unless explicitly instructed

If the experiment fails, diagnose the existing setup. Do not compensate by immediately adding more model capacity or a different loss.

---

# Deliverables

Produce:

1. Exact identification of the two fixed excerpts
2. Dataset and crop manifest
3. Input/target alignment report
4. Training configuration
5. W&B run name and run identifier
6. Local report independent of W&B
7. Learning curves
8. Gradient diagnostics
9. Per-example memorization metrics
10. Norm, variance, and temporal diagnostics
11. Checkpoints and checkpoint comparison
12. Listening package for both excerpts
13. Direct comparison with the successful MuQ memorization experiment
14. Updated project notes
15. Test and lint results

Update:

- `notes/DECISIONS.md`
- `notes/MEMORY.md`
- `notes/NEXT.md`
- `notes/SCRATCHPAD.md`

Do not overwrite the reports or artifacts from TASK-01 or TASK-02.

---

# Final Questions

The final report must answer, directly and without overclaiming:

> Can the existing Spectrogram→DAC pipeline nearly perfectly memorize the same two fixed excerpts used by the successful MuQ memorization control?

> Does low training loss correspond to restoration of DAC latent norm, variance, and temporal dynamics?

> Does better latent fitting produce audio that closely matches the DAC oracle?

> If the experiment fails, at what point does it fail: data alignment, optimization, latent prediction, checkpoint selection, or DAC decoding?

> What do the W&B trajectories reveal about when norm collapse, variance collapse, and temporal over-smoothing emerge?

Stop after answering these questions. Do not proceed automatically to a larger experiment or a new target formulation.
