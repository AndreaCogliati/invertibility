# TASK-09: Optimal Transport Diagnostic for DAC Latent Geometry

## Required Reading

Before doing anything, read:

- `CLAUDE.md`
- `notes/DECISIONS.md`
- `notes/MEMORY.md`
- `notes/NEXT.md`
- TASK-07
- TASK-08

Pay particular attention to the combined interpretation of TASK-07 and TASK-08:

> DAC latent similarity collapses under tiny waveform shifts, while decoded audio similarity remains nearly unchanged.

This experiment is designed to test a specific suggestion:

> Is the problem that we are using the wrong geometry (pointwise MSE) to compare latent trajectories?

This is **not** a new training experiment.

No learning is involved.

---

# Objective

Evaluate whether an Optimal Transport (OT) distance between DAC latent trajectories is better aligned with reconstructed-audio similarity than the project's current latent metrics.

Current comparison:

```text
latent MSE
latent cosine similarity
```

Proposed comparison:

```text
Optimal Transport distance
```

The goal is to determine whether the geometry of latent space is the problem.

---

# Scientific Question

TASK-08 established:

```text
latent similarity
   ↓↓↓

audio similarity
   ≈ constant
```

This suggests:

> Large Euclidean changes in DAC latent space may correspond to very small perceptual changes.

OT offers a different notion of similarity:

Instead of requiring:

```text
frame i
  ↔
frame i
```

OT asks:

```text
How much mass must move
to transform one latent distribution into another?
```

This experiment tests whether OT aligns better with the decoder's behavior.

---

# Critical Constraints

No training.

No checkpoints.

No optimization.

No translator models.

No MuQ prediction.

This is a pure analysis task.

---

# Dataset

Reuse exactly the same shift experiment dataset from TASK-08.

Do not create a new sample set.

Use:

- the same crops
- the same shifts
- the same latent extractions
- the same reconstructed audio

This guarantees direct comparability.

---

# Inputs

For each crop:

```text
z0 = DAC latent (shift 0)
zk = DAC latent (shift k)
```

for:

```text
k ∈ {1,2,4,8,16,32,64,128,256,512,1024}
```

Reuse cached results wherever possible.

---

# Metrics to Compute

## Baseline 1: MSE

Reuse the current latent-space distance:

```text
MSE(z0, zk)
```

---

## Baseline 2: Cosine Similarity

Reuse:

```text
1 - cosine(z0, zk)
```

or cosine similarity itself.

Match the existing implementation.

---

## New Metric: Optimal Transport

Treat each latent trajectory as a cloud of points:

```text
150 frames × 1024 dimensions
```

and compute an OT distance between:

```text
z0
and
zk
```

Use a standard implementation:

Preferred:

```python
geomloss
```

or

```python
POT
```

Use entropic Sinkhorn OT for tractability.

Document:

- exact implementation
- epsilon
- normalization
- runtime

Do not tune hyperparameters aggressively.

This is a diagnostic.

---

# Audio Metrics

Reuse TASK-08's audio metrics:

## Waveform Correlation

```text
corr(audio0, audiok)
```

---

## Log-Mel Similarity

Reuse TASK-08 implementation.

---

## STFT Distance

Reuse TASK-08 implementation.

---

# Core Analysis

For every crop/shift pair:

Compute:

```text
MSE
Cosine
OT
```

and compare to:

```text
Waveform Correlation
Log-Mel Similarity
STFT Distance
```

---

# Correlation Analysis

Compute:

## Spearman Correlation

Between:

```text
latent MSE
and
audio similarity
```

---

Between:

```text
cosine similarity
and
audio similarity
```

---

Between:

```text
OT distance
and
audio similarity
```

---

Also compute:

## Pearson Correlation

for the same comparisons.

Report both.

---

# Required Figures

## Figure 1

### Latent Distance vs Shift

Plot:

```text
MSE
Cosine
OT
```

against shift.

---

## Figure 2

### Audio Similarity vs Shift

Reuse TASK-08 metrics.

---

## Figure 3 (Most Important)

### Distance Metric vs Audio Similarity

Three scatter plots:

```text
MSE vs waveform correlation
Cosine vs waveform correlation
OT vs waveform correlation
```

Include best-fit line and Spearman value.

---

## Figure 4

### Ranking Comparison

For all crop/shift pairs:

Rank pairs by:

```text
MSE
Cosine
OT
```

Compare each ranking against:

```text
Audio similarity ranking
```

This directly tests which metric agrees best with the decoder.

---

# Interpretation Rules

## Outcome A

OT greatly outperforms MSE and cosine.

Example:

```text
Spearman:

MSE     = 0.25
Cosine  = 0.30
OT      = 0.80
```

Interpretation:

```text
The latent manifold itself is not necessarily the problem.

Pointwise regression is imposing
the wrong geometry.

OT becomes a strong candidate
for future objectives.
```

---

## Outcome B

OT is only modestly better.

Example:

```text
0.25
0.30
0.40
```

Interpretation:

```text
The mismatch is deeper than
framewise comparison.

The latent space itself may be
poorly aligned with reconstruction.
```

---

## Outcome C

OT becomes nearly invariant to shifts.

Example:

```text
Tiny OT change
Huge MSE change
Tiny audio change
```

Interpretation:

```text
A large fraction of the latent variation
identified in TASK-07 is effectively
transportable mass rather than
meaningful reconstruction information.
```

This would provide a clean explanation for why decoded audio remains stable.

---

# Deliverables

Produce:

1. Machine-readable report
2. OT implementation details
3. Correlation tables
4. All four figures
5. Runtime statistics
6. Written interpretation

Update:

- `notes/DECISIONS.md`
- `notes/MEMORY.md`
- `notes/NEXT.md`
- `notes/SCRATCHPAD.md`

Do not create or modify any training code.

Do not perform any learning.

---

# Final Questions

The report must answer:

> Does Optimal Transport correlate with reconstructed-audio similarity better than MSE?

> Does Optimal Transport correlate with reconstructed-audio similarity better than cosine similarity?

> Is the latent-collapse phenomenon identified in TASK-07 largely explainable as transportable rather than perceptually meaningful variation?

> Does OT provide evidence that the current MSE-plus-cosine objective is using the wrong geometry?

> Do the results strengthen or weaken the target-mismatch hypothesis emerging from TASK-07 and TASK-08?

Stop after answering these questions.

Do not propose or run a follow-up experiment automatically.
