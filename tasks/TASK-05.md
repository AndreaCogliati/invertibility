# TASK-05: Spectrogram Gate-3 Scaling Curve

## Required Reading

Before making any changes, read:

- `CLAUDE.md`
- `EXPERIMENT_SYNTHESIS.md`
- `notes/DECISIONS.md`
- `notes/MEMORY.md`
- `notes/NEXT.md`
- Results and implementation from TASK-01, TASK-02, and TASK-04
- The W&B dashboard from TASK-04

Carefully understand the conclusions already established:

- TASK-01: Spectrogram→DAC control
- TASK-02: Dense GTZAN coverage
- TASK-04: Spectrogram Gate-3 memorization

This task is explicitly designed to bridge the gap between:

```text
N = 2 examples
(perfect memorization)

and

N ≈ 15,000 examples
(norm collapse)
```

---

# Objective

Determine **where the collapse begins**.

The current evidence shows:

```text
2 examples
→ perfect reconstruction

~15,000 examples
→ norm collapse
→ variance collapse
→ temporal over-smoothing
→ no meaningful waveform recovery
```

The transition point is completely unknown.

This experiment constructs a controlled scaling curve:

```text
2 examples
8 examples
32 examples
128 examples
512 examples
```

using the exact Spectrogram→DAC pipeline that succeeded in TASK-04.

The goal is not generalization.

The goal is understanding:

> At what dataset size does the reconstruction regime change?

---

# Core Scientific Question

Which of the following is true?

### Hypothesis A

The collapse emerges gradually.

Example:

```text
N = 2
norm ratio = 1.0

N = 8
norm ratio = 0.98

N = 32
norm ratio = 0.9

N = 128
norm ratio = 0.6

N = 512
norm ratio = 0.35
```

This would strongly support a regression-to-the-mean explanation.

---

### Hypothesis B

There is a sharp transition.

Example:

```text
N = 2
norm ratio = 1.0

N = 8
norm ratio = 1.0

N = 32
norm ratio = 0.99

N = 128
norm ratio = 0.98

N = 512
norm ratio = 0.3
```

This would suggest a capacity, optimization, or target-complexity threshold.

---

### Hypothesis C

Collapse never emerges on the training set.

Example:

```text
all training examples memorized well
```

even for hundreds of excerpts.

This would imply that the real problem is entirely one of generalization.

---

# Critical Experimental Principle

This is a training-set scaling experiment.

Do **not** focus on validation performance.

Do **not** attempt to solve the task.

Do **not** redesign the architecture.

Do **not** introduce new losses.

Do **not** change the target representation.

This experiment is intended to isolate one variable:

```text
Number of training examples
```

Nothing else.

---

# Dataset Construction

Use examples from the existing dense GTZAN cache.

Do not regenerate features unless necessary.

Create nested datasets:

```text
Level 1:   2 examples
Level 2:   8 examples
Level 3:  32 examples
Level 4: 128 examples
Level 5: 512 examples
```

Each level should contain all examples from the previous level.

Example:

```text
2 ⊂ 8 ⊂ 32 ⊂ 128 ⊂ 512
```

This makes comparisons cleaner.

Prefer balanced genre coverage whenever practical.

Document exactly how examples were selected.

Save manifests:

```text
gate3_scale_2.csv
gate3_scale_8.csv
gate3_scale_32.csv
gate3_scale_128.csv
gate3_scale_512.csv
```

---

# Representation

Reuse exactly the same configuration from TASK-04:

```text
Mel spectrogram
↓
Learned projection
↓
TemporalConvNet
↓
DAC latent
```

No representation changes.

No preprocessing changes.

No alignment changes.

---

# Architecture

Reuse the exact SpectrogramTranslator from TASK-04.

Do not change:

- depth
- width
- receptive field
- parameter count

The entire purpose is to keep the model fixed.

---

# Loss

Reuse exactly:

```text
MSE + cosine
```

with identical weighting.

Do not introduce any regularization.

Do not normalize targets.

Do not change objectives.

---

# Training Procedure

Use the same optimizer settings as TASK-04.

Training should continue until one of the following occurs:

### Condition A

Training metrics saturate.

### Condition B

Metrics show no meaningful improvement for a long period.

### Condition C

A predefined maximum step budget is reached.

Document the criterion used.

The objective is not to maximize performance.

The objective is to characterize scaling behavior.

---

# Weights & Biases

Reuse and extend TASK-04 W&B instrumentation.

Project:

```text
muq-invertibility
```

Group:

```text
gate3-scaling
```

Suggested names:

```text
gate3_scale_2
gate3_scale_8
gate3_scale_32
gate3_scale_128
gate3_scale_512
```

All runs should be directly comparable.

---

# Required Metrics

For every scale:

## Training Metrics

Log:

```text
train/mse
train/cosine
train/total_loss
```

---

## Collapse Metrics

The primary diagnostics are:

```text
norm_ratio
variance_ratio
lag1_autocorrelation
```

Track them continuously.

These are now among the most important project metrics.

---

## Temporal Dynamics

Track:

```text
frame_difference_norm
temporal_variance
prediction_autocorrelation
oracle_autocorrelation
```

---

## Audio Metrics

Track:

```text
waveform_correlation_to_oracle
stft_distance
```

on a fixed subset of training examples.

---

# Memorization Analysis

For each scale compute:

```text
Best training MSE
Best cosine
Best waveform correlation
Best norm ratio
Best variance ratio
```

Also compute:

```text
Mean
Median
Worst-case
```

across examples.

The worst-case example may become particularly informative.

---

# Audio Logging

Use a fixed set of representative examples.

Log:

```text
Reference
DAC Oracle
Current Reconstruction
```

at multiple checkpoints.

Use exactly the same audio logging workflow as TASK-04.

---

# Main Deliverable

Produce a scaling figure with:

```text
x-axis = number of training examples
```

and the following curves:

```text
training MSE
cosine similarity
norm ratio
variance ratio
waveform correlation
```

This figure is the primary outcome of TASK-05.

---

# Additional Required Figure

Plot:

```text
Number of Training Examples
vs
Norm Ratio
```

and

```text
Number of Training Examples
vs
Variance Ratio
```

These are likely the most important plots in the entire project so far.

---

# Interpretation Rules

## Outcome A

Collapse grows gradually as dataset size increases.

Interpretation:

```text
Strong evidence for regression-to-the-mean
and averaging effects.
```

---

## Outcome B

A sharp threshold exists.

Interpretation:

```text
Suggests capacity, optimization,
or target-complexity limits.
```

---

## Outcome C

Training-set memorization remains strong even at N=512.

Interpretation:

```text
The true failure mode is likely
generalization rather than fitting.
```

This would significantly change project direction.

---

# Direct Comparison Table

Produce:

| Metric | N=2 | N=8 | N=32 | N=128 | N=512 |
|----------|----------|----------|----------|----------|----------|
| Training MSE | | | | | |
| Cosine | | | | | |
| Norm Ratio | | | | | |
| Variance Ratio | | | | | |
| Lag-1 AC | | | | | |
| Waveform Corr | | | | | |
| STFT Distance | | | | | |

---

# Explicitly Avoid

Do not:

- Train on full GTZAN
- Train on FMA
- Run BlueHive
- Modify DAC targets
- Introduce pre-quantizer prediction
- Predict discrete codebooks
- Normalize targets
- Change loss weights
- Increase model size
- Run layer sweeps

The purpose of this task is diagnosis, not optimization.

---

# Final Questions

The final report must answer:

> At what dataset size does norm collapse first become measurable?

> At what dataset size does variance collapse first?
