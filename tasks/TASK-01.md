# Positive Control Experiment: Spectrogram → DAC Latent (GTZAN)

Read `CLAUDE.md`, `EXPERIMENT_SYNTHESIS.md`, and the existing MuQ→DAC training codebase before making changes.

## Objective

Design and run a **positive-control inversion experiment** on GTZAN.

The goal is **not** to improve MuQ inversion.

The goal is to determine whether the existing training pipeline can successfully learn a mapping from a highly informative representation to the same DAC latent target currently used in the MuQ project.

Specifically:

```text
Mel Spectrogram
      ↓
TemporalConvNet Translator
      ↓
DAC Latent
      ↓
Frozen DAC Decoder
      ↓
Audio
```

Use exactly the same DAC latent target currently used by the MuQ→DAC experiments.

---

## Scientific Motivation

Current MuQ→DAC experiments show:

- Near-perfect memorization on tiny datasets
- Poor generalization
- Strong norm collapse
- Strong temporal smoothing
- Poor waveform recovery

These observations could arise from:

1. Insufficient information in MuQ
2. Insufficient training data
3. A mismatch between regression and DAC latent geometry
4. Shortcomings in the current training pipeline

This experiment serves as a **positive control**.

A mel spectrogram is known to retain substantially more reconstruction-relevant information than MuQ.

If the current pipeline cannot learn Spectrogram→DAC, conclusions about MuQ invertibility become premature.

---

## Experimental Philosophy

This experiment should isolate a single question:

> Can the existing architecture, loss function, dataset, and training pipeline learn an inverse mapping when provided with a representation that is known to contain abundant reconstruction information?

Avoid introducing new ideas, losses, architectures, normalization schemes, or quantization-aware methods.

The purpose is diagnostic, not optimization.

---

## Reuse Existing Infrastructure

Reuse as much of the current codebase as possible:

- GTZAN manifests
- Train/validation/test split
- Cropping strategy
- Dataset loaders
- Caching infrastructure
- Checkpointing
- Logging
- DAC latent extraction
- DAC decoding
- Evaluation scripts
- Existing diagnostic framework

Minimize code changes.

---

## Representation

Replace MuQ embeddings with mel-spectrogram features.

### Requirements

Inspect the current DAC latent frame rate and temporal resolution.

Choose spectrogram parameters that produce a closely aligned temporal grid.

Suggested starting point:

```python
n_mels = 128
```

but determine the final parameters based on DAC frame timing.

Document:

- FFT size
- Window length
- Hop size
- Number of mel bins
- Spectrogram normalization
- Input dimensionality
- Sequence length
- Temporal alignment procedure

---

## Architecture

Start with exactly the same architecture currently used for:

```text
MuQ → DAC Latent
```

Do not introduce:

- Transformers
- Attention mechanisms
- U-Nets
- Diffusion models
- Larger architectures
- Multi-scale encoders
- Skip connections that do not already exist

The goal is a fair positive-control comparison.

---

## Dataset

Use the same GTZAN split employed in the full-scale MuQ experiments.

Reuse the same:

- Training tracks
- Validation tracks
- Test tracks
- Crop selection strategy

Record:

- Number of tracks
- Number of crops
- Batch size
- Number of training steps
- Wall-clock training time

---

## Loss Function

Use exactly the current baseline loss:

```text
MSE + Cosine Similarity Loss
```

Do not introduce:

- Target normalization
- Quantization losses
- Adversarial losses
- Perceptual losses
- Spectral losses
- Contrastive objectives

The objective is to validate the existing training pipeline.

---

## Required Diagnostics

Compute exactly the same diagnostics used in the MuQ experiments.

### Latent-Space Metrics

Report:

- MSE
- Cosine similarity
- Latent norm ratio
- Latent variance ratio
- Temporal variance
- Lag-1 autocorrelation
- Validation and test performance

### Audio-Space Metrics

Generate and save:

1. Ground-truth audio
2. DAC oracle reconstruction
3. Spectrogram→DAC reconstruction

for the same held-out examples used in prior analyses.

Export listening examples.

---

## Additional Diagnostics

### Training Curves

Track:

- Total loss
- MSE component
- Cosine component
- Validation loss
- Validation cosine similarity

### Prediction Statistics

Track over training:

- Mean target norm
- Mean predicted norm
- Mean target variance
- Mean predicted variance

One major question is whether norm collapse persists when spectrograms are used instead of MuQ embeddings.

---

## Direct Comparison with MuQ

Produce a side-by-side comparison against the current best MuQ→DAC model.

Compare:

### Latent Metrics

- MSE
- Cosine similarity
- Norm ratio
- Variance ratio
- Temporal statistics

### Audio Metrics

- Reconstruction quality
- Similarity to DAC oracle reconstruction
- Subjective listening quality

### Failure Modes

Specifically assess whether the following remain present:

- Norm collapse
- Variance collapse
- Over-smoothing
- Excessive temporal autocorrelation
- Near-constant predictions

---

## Success Criteria

Define the interpretation **before training**.

### Outcome A: Strong Success

Spectrogram→DAC significantly outperforms MuQ→DAC.

Possible indicators:

- Much higher cosine similarity
- Realistic latent norms
- Realistic latent variance
- Reduced smoothing
- Recognizable reconstructed audio

Interpretation:

```text
The current pipeline is capable of learning an inverse mapping.

The primary bottleneck lies in:
- MuQ information content,
- available training data,
- target formulation,
or some combination thereof.
```

---

### Outcome B: Similar Failure

Spectrogram→DAC exhibits the same pathologies observed in MuQ→DAC.

Examples:

- Norm collapse
- Variance collapse
- Excessive smoothing
- Poor audio reconstruction

Interpretation:

```text
The problem extends beyond MuQ itself.

Conclusions about MuQ invertibility should be postponed until
the underlying training formulation is better understood.
```

---

## Optional Follow-Up

Only if the primary experiment succeeds convincingly:

Run an information-ablation study using:

```text
128 mel bins
64 mel bins
32 mel bins
16 mel bins
```

to estimate how reconstruction quality degrades as information is progressively removed.

Do not prioritize this until the primary positive-control experiment is complete.

---

## Deliverables

Produce:

1. A detailed experimental report
2. Training curves
3. Validation and test metrics
4. Norm and variance diagnostics
5. Representative audio examples
6. Side-by-side comparison with the best MuQ→DAC run
7. Scientific interpretation of the results

Most importantly, answer:

> Does the current translation pipeline successfully learn a highly informative Spectrogram→DAC mapping on GTZAN?

and

> How large is the performance gap between Spectrogram→DAC and MuQ→DAC?

Focus on diagnosis and scientific understanding rather than model improvement.
