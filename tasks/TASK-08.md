# TASK-08: DAC Shift Sensitivity in Latent Space vs Reconstructed Audio

## Required Reading

Read:

- `CLAUDE.md`
- `notes/MEMORY.md`
- `notes/DECISIONS.md`
- TASK-04
- TASK-07

Pay particular attention to TASK-07's finding that DAC latents are highly sensitive to tiny waveform shifts while MuQ and log-mel are comparatively stable.

This task extends TASK-07.

Do not repeat the latent-only analysis.

The goal is to determine whether DAC latent instability translates into reconstructed-audio instability.

---

# Objective

Measure the relationship between:

```text
waveform shift
    ↓
DAC latent
    ↓
DAC decoder
    ↓
reconstructed audio
```

The central question is:

> Does a large change in DAC latent space imply a large change in reconstructed audio?

TASK-07 only answered the first half.

This task measures the second half.

---

# Scientific Motivation

TASK-07 found:

```text
1 sample shift
    ↓
large latent change

16 sample shift
    ↓
near-collapse of latent similarity
```

while:

```text
MuQ
log-mel
```

remained largely invariant.

However, latent-space instability only matters if it manifests in the audio itself.

Two possibilities remain:

## Hypothesis A

### Latents unstable, audio stable

```text
tiny shift
    ↓
latent changes dramatically
    ↓
decoder reconstructs nearly the same sound
```

Interpretation:

```text
DAC latent geometry is poorly aligned
with perceptual similarity.

The latent is a poor regression target
even though DAC reconstructs correctly.
```

---

## Hypothesis B

### Latents unstable, audio unstable

```text
tiny shift
    ↓
latent changes dramatically
    ↓
decoder produces different audio
```

Interpretation:

```text
DAC genuinely encodes alignment-sensitive
information.

The instability is physically meaningful.
```

---

# Critical Constraint

No training.

No translators.

No MuQ prediction.

No optimization.

This is purely a frozen DAC encoder-decoder analysis.

---

# Dataset

Reuse the exact crop list from TASK-07.

Do not change:

- track selection
- crop duration
- crop positions

This allows direct comparison.

---

# Shift Schedule

Use exactly the same shifts as TASK-07:

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

samples.

Apply the same shift policy as TASK-07.

---

# Required Representations

For each crop and each shift:

## Latent

Compute:

```text
DAC quantized_representation
```

exactly as in TASK-07.

---

## Reconstructed Audio

Compute:

```text
audio
    ↓
DAC encoder
    ↓
quantized_representation
    ↓
DAC decoder
    ↓
reconstructed audio
```

for:

```text
shift = 0
shift = k
```

independently.

Do not align latents manually.

Do not align audio manually beyond the normal shift compensation described below.

---

# Audio Similarity Metrics

The two reconstructed waveforms differ by an intentional shift.

Correct for that shift before evaluating similarity.

Specifically:

For a shift of:

```text
k samples
```

remove the corresponding edge samples so that the comparison reflects content similarity rather than trivial timing offset.

Document exactly how this alignment is performed.

---

## Required Metrics

Report:

### Waveform Correlation

```text
corr(audio0, audiok)
```

---

### STFT Distance

Use the same implementation already used throughout the project.

---

### Log-Mel Distance

Compute:

```text
mel(audio0)
vs
mel(audiok)
```

using the project's existing mel configuration.

---

### SI-SDR (if already available)

Reuse an existing implementation.

Do not add a new dependency solely for this metric.

---

# Reference Comparison

Also compute the same metrics directly on:

```text
original waveform
shifted waveform
```

This provides the perceptual baseline.

We need to know whether DAC reconstruction amplifies or attenuates the instability.

---

# Required Figures

## Figure 1

### Latent Similarity vs Shift

Reuse TASK-07's latent cosine curve.

---

## Figure 2

### Audio Similarity vs Shift

X-axis:

```text
shift (samples)
```

Y-axis:

```text
audio similarity
```

Plot:

- waveform correlation
- log-mel similarity
- STFT similarity

---

## Figure 3

### Latent vs Audio Similarity

X-axis:

```text
latent cosine similarity
```

Y-axis:

```text
audio similarity
```

One point per crop/shift pair.

This is arguably the most important figure.

It directly answers:

> Does latent similarity predict audio similarity?

---

# Listening Evaluation

Required.

Export representative examples for:

```text
shift 0
shift 2
shift 8
shift 32
shift 128
```

For each:

1. Original audio
2. Shifted audio
3. DAC reconstruction (shift 0)
4. DAC reconstruction (shift k)
5. Difference signal, if practical

Use the same tracks from TASK-07.

---

# Additional Analysis

Compute:

```text
latent similarity
```

versus

```text
audio similarity
```

for all examples.

Report:

- Pearson correlation
- Spearman correlation

between latent-space and audio-space metrics.

This directly evaluates whether DAC latent distance is a meaningful proxy for reconstruction distance.

---

# Interpretation Rules

## Outcome A

Latent similarity collapses rapidly.

Audio similarity remains high.

Interpretation:

```text
DAC latent space is much more sensitive than
the reconstructed audio.

Latent-distance objectives likely exaggerate
meaningful differences.

This strongly supports the target-mismatch
hypothesis.
```

---

## Outcome B

Latent similarity and audio similarity collapse together.

Interpretation:

```text
The latent instability is physically meaningful.

DAC genuinely preserves alignment-sensitive
details relevant to reconstruction.
```

---

## Outcome C

Intermediate behavior.

Interpretation:

```text
Some DAC latent dimensions correspond to
perceptually meaningful variation, others do not.

Latent distance and audio distance are only
partially aligned.
```

---

# Deliverables

Produce:

1. Machine-readable report
2. Shift-vs-audio-similarity figure
3. Latent-vs-audio figure
4. Listening package
5. Correlation analysis
6. Written interpretation

Update:

- `notes/DECISIONS.md`
- `notes/MEMORY.md`
- `notes/NEXT.md`
- `notes/SCRATCHPAD.md`

Do not perform any training.

Do not modify any existing experiment.

---

# Final Questions

The report must answer:

> How much audio change results from the latent changes observed in TASK-07?

> Does DAC latent similarity predict DAC reconstruction similarity?

> Are DAC latents substantially more sensitive than the reconstructed audio itself?

> Is DAC latent distance a meaningful proxy for perceptual or reconstruction distance?

> Do the results strengthen or weaken the target-mismatch explanation for TASK-01 through TASK-06?

Stop after answering these questions.

Do not propose a follow-up experiment automatically.