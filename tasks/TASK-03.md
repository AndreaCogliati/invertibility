# TASK-03: BlueHive Reproduction + Weights & Biases Instrumentation

Read:

- `CLAUDE.md`
- `EXPERIMENT_SYNTHESIS.md`
- `notes/DECISIONS.md`
- `notes/MEMORY.md`
- `notes/NEXT.md`
- Results from TASK-01 and TASK-02

Before making any changes, fully understand the conclusions of the previous experiments.

---

# Objective

Run an exact reproduction of the current best Spectrogram→DAC experiment (TASK-02) on BlueHive and integrate comprehensive Weights & Biases (W&B) monitoring.

This task has two independent goals:

1. Determine whether the behavior observed on the M4 Max reproduces on a different hardware and software environment.
2. Obtain substantially deeper training diagnostics using W&B so we can investigate the recurring norm-collapse, variance-collapse, and early-checkpoint-selection phenomena.

This is an infrastructure and observability task, not a model-development task.

---

# Scientific Motivation

After TASK-01 and TASK-02:

- Replacing MuQ with mel spectrograms did not solve the problem.
- Increasing GTZAN coverage from ~13% to ~100% did not solve the problem.
- Norm collapse persists.
- Variance collapse persists.
- Waveform correlation remains approximately zero.
- Validation-selected checkpoints consistently occur extremely early in training.

The next priority is understanding **how training evolves** rather than introducing new architectures or objectives.

BlueHive provides:

- Larger compute resources
- Better experiment reproducibility
- Longer training runs
- Centralized experiment tracking

W&B provides:

- Rich diagnostics
- Historical comparisons
- Gradient tracking
- Artifact management
- Failure-mode visualization

---

# Critical Constraint

Do not redesign the experiment.

Do not change:

- Dataset
- Split
- Spectrogram parameters
- Architecture
- Loss function
- Latent targets
- Training objective

The training task must remain directly comparable to TASK-02.

---

# Baseline Experiment

Reproduce TASK-02 exactly:

```text
Dense GTZAN coverage
↓
Mel spectrogram input
↓
SpectrogramTranslator
↓
DAC latent target
↓
MSE + cosine loss
```

Use:

- Same dense manifest
- Same crop generation
- Same train/val/test split
- Same optimizer
- Same scheduler (if any)
- Same learning rate
- Same batch size
- Same checkpointing policy
- Same evaluation protocol

Any deviation must be documented and justified.

---

# BlueHive Integration

Create a clean BlueHive execution path.

Tasks:

- Create BlueHive job scripts
- Validate environment setup
- Verify DAC dependencies
- Verify cache paths
- Verify checkpoint locations
- Verify artifact storage
- Use Astral `uv` as the package manager, using native project commands only
- Update the code so it can run on MPS or CUDA
- Create instructions on manually moving and running the experiments on BlueHive

Document:

- Cluster configuration
- GPU type
- Number of GPUs
- CPU resources
- Memory allocation

Record all runtime environment information.

---

# Weights & Biases Integration

Integrate W&B as a first-class experiment logger.

The goal is to make future debugging substantially easier.

---

# Required W&B Configuration

Create a dedicated project:

```text
muq-invertibility
```

Suggested grouping:

```text
spectrogram-control
```

Suggested run naming scheme:

```text
spec_dense_gtzan_<timestamp>
```

Log:

- Git commit hash
- Hostname
- Machine type
- BlueHive job ID
- Hyperparameters
- Dataset statistics

at run start.

---

# Hyperparameter Logging

Record:

- Learning rate
- Batch size
- Optimizer
- Scheduler
- Weight decay
- Number of steps
- Model parameter count
- Spectrogram configuration
- Dataset sizes

All hyperparameters should appear in the W&B config panel.

---

# Training Metrics

Log every training interval:

## Loss Components

- Total loss
- MSE loss
- Cosine loss

## Gradient Statistics

- Total gradient norm
- Gradient norm by module
- Maximum gradient
- Minimum gradient

Specifically monitor whether gradients collapse or explode.

---

# Prediction Statistics

For every validation cycle log:

## Target Statistics

- Mean latent norm
- Median latent norm
- Latent variance

## Prediction Statistics

- Mean prediction norm
- Median prediction norm
- Prediction variance

## Ratios

- Norm ratio
- Variance ratio

This is now one of the central diagnostics of the project.

---

# Temporal Dynamics Diagnostics

Log:

- Lag-1 autocorrelation
- Frame-to-frame variance
- Temporal smoothness measures

Create dedicated W&B charts for these metrics.

We have repeated evidence that these may be the most informative indicators of failure.

---

# Validation Metrics

Track:

- Validation MSE
- Validation cosine similarity
- Validation STFT distance
- Waveform correlation to oracle

for every validation pass.

---

# Checkpoint Diagnostics

A central question in the project is:

> Why does the best checkpoint often occur extremely early?

For every saved checkpoint record:

- Global step
- Validation MSE
- Validation cosine
- Norm ratio
- Variance ratio

Generate W&B tables showing:

- Top-10 checkpoints by validation MSE
- Top-10 checkpoints by cosine similarity
- Top-10 checkpoints by norm ratio

We want to determine whether these objectives disagree.

---

# Visualization Requirements

Create W&B visualizations for:

## Training Trajectories

- Loss curves
- Learning curves
- Gradient curves

## Distribution Tracking

- Norm distributions
- Variance distributions

## Temporal Diagnostics

- Autocorrelation trends
- Frame-variance trends

## Correlation Analysis

Scatter plots of:

```text
Validation MSE
vs.
Norm Ratio
```

```text
Validation MSE
vs.
Variance Ratio
```

```text
Cosine Similarity
vs.
Waveform Correlation
```

These may reveal why model selection is choosing pathological checkpoints.

---

# Audio Logging

For a fixed held-out evaluation set:

Log to W&B:

- Ground truth audio
- DAC oracle reconstruction
- Spectrogram→DAC reconstruction

for the same examples across training.

Use a fixed sample set.

This allows listening to training evolution.

---

# Artifact Management

Save as W&B artifacts:

- Checkpoints
- Evaluation reports
- Dataset manifest
- Training configuration
- Diagnostic summaries

The experiment should be fully reproducible from W&B.

---

# Reproduction Analysis

At the end of training compare:

| Metric | M4 Max (TASK-02) | BlueHive |
|----------|----------|----------|
| MSE | | |
| Cosine | | |
| STFT | | |
| Waveform Corr | | |
| Norm Ratio | | |
| Variance Ratio | | |
| Best Checkpoint Step | | |

Determine whether the conclusions reproduce.

---

# Additional Investigation

The repeated appearance of a best checkpoint near step 1000 is now a major scientific observation.

Produce a dedicated analysis answering:

> Is the early-checkpoint phenomenon caused by:
>
> - optimization instability,
> - validation-noise fluctuations,
> - regression-to-the-mean behavior,
> - oversmoothing,
> - or another mechanism?

Use W&B visualizations to support the conclusion.

Do not speculate without evidence.

---

# Deliverables

Produce:

1. BlueHive training infrastructure
2. W&B integration
3. Full reproduction run
4. Diagnostic dashboard
5. Checkpoint analysis
6. Audio comparisons
7. Reproduction report

Most importantly, answer:

> Does the TASK-02 result reproduce unchanged on BlueHive?

and

> What do the new W&B diagnostics reveal about the persistent norm-collapse and early-checkpoint-selection behavior?

Do not modify the scientific formulation.

The objective of TASK-03 is observability, reproducibility, and diagnosis.
