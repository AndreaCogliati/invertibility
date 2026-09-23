# TASK-06: N=32 Compute-Controlled Micro-Ablation

## Required Reading

Before making any changes, read:

- `CLAUDE.md`
- `EXPERIMENT_SYNTHESIS.md`
- `notes/DECISIONS.md`
- `notes/MEMORY.md`
- `notes/NEXT.md`
- TASK-04 (Spectrogram Gate-3)
- TASK-05 (Gate-3 Scaling Curve)

Carefully review the W&B dashboards and reports for:

```text
N = 8
N = 32
```

from TASK-05.

---

# Objective

Determine whether the dramatic collapse observed at:

```text
N = 32
```

is caused primarily by:

1. insufficient optimization exposure per example, or
2. a genuine capacity / interference / objective boundary.

This experiment is intentionally minimal.

Do not repeat the scaling curve.

Do not train larger datasets.

Do not modify the model.

Only investigate the existing:

```text
N = 32
```

configuration.

---

# Scientific Motivation

TASK-05 established:

| N | Waveform Corr | Norm Ratio | Variance Ratio |
|---|---:|---:|---:|
| 2 | 1.000 | 1.000 | 1.000 |
| 8 | 0.997 | 1.003 | 1.008 |
| 32 | 0.469 | 0.499 | 0.177 |
| 128 | 0.090 | 0.378 | 0.063 |
| 512 | 0.019 | 0.343 | 0.048 |

The transition between:

```text
N = 8
```

and

```text
N = 32
```

is abrupt and perceptually obvious.

However, TASK-05 deliberately used a fixed training budget:

```text
20,000 steps
```

for every scale.

This means that two variables changed simultaneously:

## Variable 1

Dataset size increased.

```text
N = 8 → N = 32
```

## Variable 2

Optimization exposure per example decreased.

Approximate per-example presentations:

| N | Presentations / Example |
|---:|---:|
| 2 | 10,000 |
| 8 | 2,500 |
| 32 | 625 |

Therefore TASK-05 does **not** isolate whether the collapse arises from:

```text
More examples
```

or

```text
Fewer updates per example
```

This task isolates that question.

---

# Experimental Principle

Everything remains fixed except:

```text
Training duration
```

No other variable may change.

---

# Dataset

Reuse exactly:

```text
data/manifests/gate3_scale_32.csv
```

from TASK-05.

Do not modify it.

Do not resample.

Do not create a new manifest.

---

# Representation

Reuse exactly the TASK-04 and TASK-05 configuration:

```text
Mel Spectrogram
↓
learned projection
↓
TemporalConvNet
↓
post-RVQ DAC latent
```

No preprocessing changes.

No caching changes.

No alignment changes.

---

# Architecture

Reuse the identical model checkpoint initialization and architecture from TASK-05.

Report:

- parameter count
- receptive field
- input dimensionality
- output dimensionality

for confirmation only.

---

# Loss

Reuse exactly:

```text
MSE + cosine
```

with unchanged weighting.

No normalization.

No additional losses.

No regularization changes.

---

# Training Schedule

Run three budgets:

```text
20,000 steps
80,000 steps
320,000 steps
```

The existing TASK-05 run already provides:

```text
20k
```

If possible, reuse it rather than retraining.

Primary targets:

```text
80k
320k
```

The purpose is to approximately equalize exposure per example with the successful smaller-scale experiments.

Approximate exposure:

| Steps | Presentations / Example |
|---:|---:|
| 20k | 625 |
| 80k | 2,500 |
| 320k | 10,000 |

These correspond roughly to:

```text
TASK-05 N=32
TASK-05 N=8
TASK-05 N=2
```

respectively.

---

# Resume Policy

If optimizer state was saved correctly:

```text
Resume from 20k
```

instead of retraining.

Otherwise document why a fresh run was necessary.

Do not silently restart from scratch if a valid continuation path exists.

---

# Weights & Biases

Reuse the existing setup.

Project:

```text
muq-invertibility
```

Group:

```text
gate3-n32-compute-ablation
```

Run names:

```text
n32_20k
n32_80k
n32_320k
```

or equivalent.

---

# Critical New Diagnostics

TASK-05 mostly examined averages.

This experiment must examine distributions.

---

## Per-Example Metrics

For all 32 training examples compute:

```text
MSE
cosine
norm ratio
variance ratio
waveform correlation
STFT distance
```

at each checkpoint.

Store per-example metrics.

Do not only compute means.

---

## Distribution Statistics

Log:

```text
mean
median
min
max
std
```

for every metric across the 32 examples.

---

## Worst-Case Analysis

At each checkpoint identify:

```text
best example
median example
worst example
```

and export their reconstructions.

This may reveal whether:

```text
all examples partially collapse
```

or

```text
a subset dominates training
```

---

# New Figure Requirement

Create a figure showing:

```text
Per-example waveform correlation
```

for all 32 examples.

One panel each for:

```text
20k
80k
320k
```

This is a key diagnostic.

---

# Audio Logging

For the same fixed examples, export:

```text
Reference
DAC Oracle
Reconstruction
```

at:

```text
20k
40k
80k
160k
320k
```

Use a small representative subset.

---

# Main Hypotheses

## Hypothesis A: Undertraining

As training continues:

```text
norm ratio rises
variance ratio rises
waveform correlation rises
```

and approaches the N=8 regime.

Interpretation:

```text
TASK-05's collapse was largely caused by
insufficient optimization exposure per example.
```

---

## Hypothesis B: Partial Recovery

Metrics improve but plateau.

Example:

```text
waveform corr
0.47 → 0.70
```

without approaching N=8.

Interpretation:

```text
Optimization matters,
but interference among examples is real.
```

---

## Hypothesis C: No Recovery

Metrics remain close to:

```text
N=32 @ 20k
```

even at:

```text
320k
```

Interpretation:

```text
The sharp N=8→32 transition reflects
a genuine capacity/objective/interference boundary.
```

---

# Required Comparison Table

Produce:

| Metric | 20k | 80k | 320k |
|---|---:|---:|---:|
| Mean MSE | | | |
| Mean Cosine | | | |
| Mean Norm Ratio | | | |
| Mean Variance Ratio | | | |
| Mean Waveform Corr | | | |
| Mean STFT Distance | | | |
| Worst-Case Waveform Corr | | | |
| Worst-Case Norm Ratio | | | |
| Worst-Case Variance Ratio | | | |

---

# Explicitly Avoid

Do not:

- Change the model
- Change the target
- Change the loss
- Change the dataset
- Change the spectrogram representation
- Train N=64
- Train N=128
- Train full GTZAN
- Train FMA
- Introduce normalization
- Introduce codebook prediction
- Introduce pre-quantizer prediction

This is a single-variable compute ablation.

---

# Final Questions

The report must answer:

> Is the collapse at N=32 caused primarily by insufficient optimization exposure per example?

> If N=32 receives approximately the same number of presentations per example as N=8 or N=2, does it recover?

> Do all examples improve together, or does the model preferentially memorize a subset?

> Is the observed boundary between N=8 and N=32 better explained by compute, interference, capacity, or regression-to-the-mean?

Stop after answering these questions.

Do not automatically propose the next experiment.