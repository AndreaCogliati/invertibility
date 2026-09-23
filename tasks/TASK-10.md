# TASK-10: EnCodec Shift-Sensitivity and Decoder-Stability Replication

## Required Reading

Before making any changes, read:

- `CLAUDE.md`
- `notes/DECISIONS.md`
- `notes/MEMORY.md`
- `notes/NEXT.md`
- TASK-07
- TASK-08
- TASK-09

Pay particular attention to the conclusions of TASK-07 and TASK-08:

> DAC latent similarity collapses under tiny waveform shifts.

and

> Reconstructed audio similarity remains nearly constant even when latent similarity collapses.

This task asks whether that phenomenon is:

```text
DAC-specific
```

or

```text
a property of neural audio codecs generally.
```

---

# Objective

Repeat the equivalent of:

```text
TASK-07
+
TASK-08
```

using EnCodec instead of DAC.

The purpose is not to compare reconstruction quality.

The purpose is to compare latent-space geometry.

---

# Scientific Question

Current working hypothesis:

> DAC latent space is poorly aligned with reconstruction similarity.

A competing hypothesis:

> Codec latents generally behave this way because they are optimized for compression rather than smooth geometry.

This experiment tests that distinction.

---

# Critical Constraint

No training.

No translators.

No MuQ prediction.

No optimization.

No checkpoint selection.

This is purely:

```text
waveform
↓
EnCodec
↓
latent
↓
EnCodec decoder
↓
audio
```

analysis.

---

# Exact Dataset Reuse

Reuse exactly the same dataset used in:

- TASK-07
- TASK-08
- TASK-09

Do not change:

- clips
- crops
- durations
- shift schedule

This must remain directly comparable.

---

# Shift Schedule

Reuse exactly:

```text
0
1
2
4
8
16
32
64
128
256
512
1024
```

sample shifts.

Maintain the same shift implementation.

Document any unavoidable EnCodec-specific preprocessing differences.

---

# EnCodec Representation Extraction

Use the same EnCodec checkpoint already used elsewhere in the project.

Document:

- model name
- sample rate
- frame rate
- codebook count
- latent dimensionality

before running analyses.

---

# Representations to Analyze

For every crop and shift:

## Representation A

Continuous EnCodec latent

Use the analogous continuous representation to the one used in prior experiments whenever exposed.

If multiple continuous latent choices exist, document them and choose the closest analog to the DAC experiment.

---

## Representation B

Quantized EnCodec latent

Use the representation closest to:

```text
DAC quantized_representation
```

from TASK-07.

Document the exact implementation.

---

## Representation C (Optional)

Per-codebook analysis.

Only if trivial.

Do not significantly expand the scope.

---

# Part 1: TASK-07 Replication

Repeat the latent-only analysis.

For every shift:

Compute:

## Cosine Similarity

```text
cos(z0, zk)
```

---

## MSE

```text
MSE(z0, zk)
```

---

## Norm Ratio

```text
||zk|| / ||z0||
```

---

## Any Existing Latent Diagnostic

Reuse established project metrics wherever practical.

---

# Part 2: TASK-08 Replication

Now decode.

For every shift:

```text
audio
↓
EnCodec encode
↓
EnCodec decode
```

Compute:

---

## Waveform Correlation

After shift compensation exactly as performed for TASK-08.

---

## Log-Mel Similarity

Reuse TASK-08 implementation.

---

## STFT Distance

Reuse TASK-08 implementation.

---

# Most Important Analysis

Construct:

## Latent Similarity vs Shift

for EnCodec.

---

## Audio Similarity vs Shift

for EnCodec.

---

## Latent Similarity vs Audio Similarity

Scatter plots:

```text
latent cosine
vs
waveform correlation
```

```text
latent cosine
vs
log-mel cosine
```

```text
latent cosine
vs
STFT distance
```

---

# Direct Comparison With DAC

Create side-by-side tables.

---

## Latent Stability

| Shift | DAC Cosine | EnCodec Cosine |
|---------|---------:|---------:|
| 1 | | |
| 2 | | |
| 4 | | |
| 8 | | |
| 16 | | |
| 32 | | |
| 64 | | |
| 128 | | |
| 256 | | |
| 512 | | |
| 1024 | | |

---

## Audio Stability

| Shift | DAC Wave Corr | EnCodec Wave Corr |
|---------|---------:|---------:|
| 1 | | |
| 2 | | |
| 4 | | |
| 8 | | |
| 16 | | |
| 32 | | |
| 64 | | |
| 128 | | |
| 256 | | |
| 512 | | |
| 1024 | | |

---

## Latent→Audio Alignment

Compare:

```text
Spearman(latent, waveform)
Spearman(latent, mel)
Spearman(latent, STFT)
```

for:

```text
DAC
EnCodec
```

---

# Interpretation Rules

## Outcome A

EnCodec behaves similarly to DAC.

Example:

```text
latent similarity collapses
audio remains stable
```

Interpretation:

```text
The phenomenon is not DAC-specific.

Neural codec latent geometry appears
generally poorly aligned with
reconstruction similarity.

This strengthens the target-mismatch
hypothesis.
```

---

## Outcome B

EnCodec is much smoother.

Example:

```text
DAC cosine:
1.0 → 0.1

EnCodec cosine:
1.0 → 0.8
```

Interpretation:

```text
DAC is unusually fragile.

EnCodec becomes a much more attractive
target representation.
```

---

## Outcome C

EnCodec latent instability tracks audio.

Interpretation:

```text
Latent instability is more
reconstruction-relevant in EnCodec.

DAC may be the outlier.
```

---

# Listening Package

Export:

For several representative clips:

```text
shift 0
shift 2
shift 8
shift 32
shift 128
```

including:

- original audio
- EnCodec reconstruction
- shifted EnCodec reconstruction

Use exactly the same procedure and examples as TASK-08 whenever possible.

---

# Deliverables

Produce:

1. EnCodec latent-similarity report
2. EnCodec audio-similarity report
3. DAC vs EnCodec comparison tables
4. Shift plots
5. Latent-vs-audio plots
6. Listening package
7. Written interpretation

Update:

- `notes/DECISIONS.md`
- `notes/MEMORY.md`
- `notes/NEXT.md`
- `notes/SCRATCHPAD.md`

Do not perform any training.

Do not modify any translator.

Do not propose or run a new experiment automatically.

---

# Final Questions

The report must answer:

> Is EnCodec latent space more stable than DAC latent space under tiny waveform shifts?

> Does EnCodec latent similarity 