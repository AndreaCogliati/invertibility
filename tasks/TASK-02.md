# TASK-02: Full-Audio GTZAN Positive Control (Spectrogram → DAC Latent)

Read:

- `CLAUDE.md`
- `EXPERIMENT_SYNTHESIS.md`
- `notes/DECISIONS.md`
- `notes/MEMORY.md`
- `notes/NEXT.md`

Pay particular attention to the results of **TASK-01 (Spectrogram → DAC Positive Control)** before proceeding.

---

## Objective

Run the **identical Spectrogram→DAC positive-control experiment**, changing only one variable:

> Replace the sparse 2-crops-per-track training set with dense coverage of the full GTZAN audio.

The purpose is to determine whether the apparent failure of TASK-01 was caused by:

1. The learning objective / target formulation, or
2. Insufficient effective training data.

The previous experiment used:

```text
999 tracks
1,998 crops
2 seconds per crop
3,996 total seconds
≈ 66.6 minutes of audio
```

while GTZAN contains approximately:

```text
999 tracks
30 seconds per track
≈ 29,970 total seconds
≈ 8.3 hours of audio
```

The previous experiment therefore used only ~13.3% of the available audio.

This task tests whether substantially increasing audio coverage changes the outcome.

---

## Scientific Question

The central question is:

> Does Spectrogram→DAC remain fundamentally limited when trained on most or all available GTZAN audio?

or

> Does increasing audio coverage eliminate the norm-collapse and over-smoothing pathologies observed in TASK-01?

This experiment should provide a direct answer.

---

## Critical Constraint

Do **not** redesign the experiment.

Do **not** introduce:

- New architectures
- New losses
- Target normalization
- Quantization-aware objectives
- Different optimizers
- Different learning rates
- Different schedulers
- Different training strategies
- Different latent targets

The only intended experimental change is:

```text
More audio
```

All other variables should remain as close as possible to TASK-01.

---

# Dataset Construction

## Goal

Reuse the exact same GTZAN train/val/test split.

Do not change:

- Track assignments
- Validation tracks
- Test tracks
- Corrupted-track handling

Maintain track-disjoint evaluation.

---

## Dense Audio Coverage

Construct a new crop manifest that uses substantially more audio from each track.

Several approaches are acceptable:

### Option A (preferred)

Dense overlapping crops.

Example:

```text
2-second crops
50% overlap
across entire track
```

yielding approximately:

```text
30-second track
→ ~29 crops
```

per track.

---

### Option B

Non-overlapping coverage.

Example:

```text
30-second track
→ fifteen 2-second crops
```

per track.

---

### Option C

Longer segments.

Example:

```text
5-second crops
or
10-second crops
```

covering most or all of each track.

---

Before implementation:

1. Estimate resulting dataset size.
2. Estimate storage requirements.
3. Estimate spectrogram cache size.

Choose the simplest option that maximizes audio coverage while remaining practical on the local machine.

Document the rationale.

---

# Data Accounting

Report:

```text
train tracks
val tracks
test tracks

train crops
val crops
test crops

total audio duration

effective increase over TASK-01
```

Compute the exact numbers.

Do not estimate after training; report actual values.

---

# Representation

Reuse the same spectrogram configuration from TASK-01.

Do not change:

- FFT parameters
- Hop size
- Window size
- Number of mel bins
- Temporal alignment

The representation must remain identical.

---

# Architecture

Reuse:

```text
SpectrogramTranslator
```

from TASK-01 unchanged.

Do not increase model size.

Do not alter receptive field.

Do not introduce attention.

Do not add new modules.

The only acceptable architectural difference is whatever was already necessary in TASK-01.

---

# Loss Function

Reuse exactly:

```text
MSE + cosine loss
```

Do not alter loss weighting.

Do not introduce additional objectives.

---

# Training Procedure

Match TASK-01 as closely as possible:

- Same optimizer
- Same batch size
- Same learning rate
- Same checkpoint selection strategy
- Same evaluation frequency
- Same logging

The goal is a single-variable experiment.

---

# Diagnostics

Run the full TASK-01 diagnostic suite.

## Latent Metrics

Report:

- MSE
- Cosine similarity
- Norm ratio
- Variance ratio
- Lag-1 autocorrelation
- Temporal variance

for:

- Train
- Validation
- Test

---

## Audio Metrics

Report:

- STFT distance
- Waveform correlation to oracle

using the same evaluation procedure as TASK-01.

---

## Audio Exports

Generate the same listening package:

```text
Reference
DAC Oracle
Spectrogram→DAC
Shuffled Control
```

using the same held-out examples as TASK-01 whenever possible.

This allows direct listening comparison.

---

# Direct Comparison

Prepare a side-by-side comparison:

| Metric | TASK-01 | TASK-02 |
|----------|----------|----------|
| Audio seconds used | | |
| MSE | | |
| Cosine | | |
| STFT | | |
| Waveform correlation | | |
| Norm ratio | | |
| Variance ratio | | |
| Lag-1 autocorrelation | | |

Focus specifically on whether:

- Norm collapse decreases
- Variance collapse decreases
- Excessive smoothing decreases
- Audio quality improves

---

# Interpretation Rules (Pre-Registered)

## Outcome A

Substantially increased audio coverage leads to:

- Better latent metrics
- Better audio metrics
- Reduced norm collapse
- Reduced variance collapse
- Reduced smoothing

Interpretation:

```text
Data volume and/or audio coverage
were significant limiting factors.

The negative result from TASK-01
cannot be attributed primarily
to the training pipeline.
```

---

## Outcome B

Latent metrics improve modestly, but:

- Audio metrics remain unchanged
- Norm collapse persists
- Variance collapse persists
- Over-smoothing persists

Interpretation:

```text
The pathology is largely independent
of GTZAN audio coverage.

Attention should shift toward:

- target formulation
- DAC latent geometry
- regression objectives
- quantization-aware approaches

rather than simple data scaling.
```

---

## Outcome C

Performance is essentially unchanged.

Interpretation:

```text
Increasing audio coverage across GTZAN
does not meaningfully address the failure mode.

The current formulation likely has a deeper issue.
```

---

# Deliverables

Produce:

1. Exact dataset statistics
2. Storage statistics
3. Training curves
4. Validation curves
5. Test metrics
6. Full latent diagnostic report
7. Listening package
8. Side-by-side TASK-01 vs TASK-02 comparison
9. Scientific interpretation

Most importantly, answer:

> Was the apparent failure of the spectrogram positive control caused by using only ~13% of GTZAN's audio?

and

> After exposure to most or all of GTZAN, does the existing Spectrogram→DAC pipeline finally learn a useful inverse mapping?

Do not optimize the model.

Do not redesign the architecture.

Do not introduce new objectives.

Answer the data-coverage question as rigorously as possible.
