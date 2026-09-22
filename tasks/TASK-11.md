# TASK-11: EnCodec Continuous-Latent Controlled Scaling Probe

## Required Reading

Before changing code or running anything, read:

- `CLAUDE.md`
- `notes/DECISIONS.md`
- `notes/MEMORY.md`
- `notes/NEXT.md`
- `notes/SCRATCHPAD.md`
- TASK-04
- TASK-05
- TASK-06
- TASK-07
- TASK-08
- TASK-09
- TASK-10

Pay particular attention to:

1. TASK-04's successful two-example memorization control.
2. TASK-05's DAC scaling transition between `N=8` and `N=32`.
3. TASK-06's partial compute recovery and training instability.
4. TASK-10's finding that EnCodec's continuous latent is dramatically more shift-stable than DAC's:
   - EnCodec continuous-latent cosine remains approximately `0.999 → 0.953` over shifts of `1 → 1024` samples.
   - DAC latent cosine falls approximately `0.716 → 0.087`.
5. TASK-10's finding that EnCodec discrete code indices remain shift-fragile.

This task therefore tests the **continuous EnCodec latent**, not EnCodec token indices.

---

# Objective

Determine whether replacing the continuous DAC target with the substantially smoother continuous EnCodec target moves the memorization-collapse boundary observed in TASK-05.

The central question is:

> Does the same translator architecture and training recipe scale beyond eight examples when the target is EnCodec's continuous latent rather than DAC's continuous post-RVQ latent?

This is a controlled target-replacement experiment.

---

# Primary Hypothesis

TASK-05 found:

```text
DAC continuous target

N = 2   → near-perfect memorization
N = 8   → near-perfect memorization
N = 32  → sharp collapse
```

TASK-10 found that EnCodec's continuous latent is much smoother under waveform shifts and more strongly aligned with EnCodec reconstruction similarity.

Therefore:

> If target geometry materially contributes to the TASK-05 transition, the EnCodec continuous target should retain substantially better memorization at `N=32`.

---

# Critical Experimental Constraint

Change only:

```text
target codec:
DAC continuous latent
→
EnCodec continuous latent
```

Keep fixed wherever technically possible:

- crop manifest
- crop duration
- selected examples
- dataset ordering
- input representation
- translator architecture
- optimizer
- learning rate
- batch size
- number of training steps
- loss formulation
- checkpoint interval
- evaluation interval
- random seeds
- best-checkpoint selection rule

Do not introduce:

- a transformer
- discrete-token prediction
- cross-entropy
- Optimal Transport as a training loss
- a decoder-aware loss
- a new optimizer
- warmup
- learning-rate decay
- gradient clipping
- larger batches
- target normalization
- architectural normalization layers
- EMA weights

Those may be reasonable later, but they would confound the target comparison.

---

# Experimental Scope

Run only:

```text
N ∈ {2, 8, 32}
```

Do not run:

```text
N = 128
N = 512
```

within this task.

The purpose is to test whether EnCodec moves the original `N=8 → N=32` boundary.

Stop after interpreting the three required runs.

---

# Input Representation

Use the exact input representation used in TASK-05.

If TASK-05 used:

```text
log-mel spectrogram
```

reuse:

- sample rate
- FFT size
- hop size
- window size
- number of mel bins
- magnitude/log scaling
- crop construction
- temporal layout
- normalization policy

Do not substitute MuQ or another input representation.

This task isolates the **target representation**.

---

# EnCodec Model

Use the exact EnCodec checkpoint and extraction pathway validated in TASK-10.

Before training, record:

- checkpoint identifier
- input sample rate
- supported channel configuration
- frame rate
- temporal hop
- continuous-latent dimensionality
- continuous-latent tensor orientation
- decoder input expectation
- EnCodec bandwidth or active-codebook configuration
- whether the continuous representation is pre-quantizer, post-quantizer, or another explicitly exposed internal tensor

Use the same continuous representation whose shift stability was measured in TASK-10.

Do not silently substitute discrete code indices or a different internal EnCodec tensor.

---

# Target Definition

For each training crop, extract:

```text
z_encodec = EnCodec continuous latent
```

The translator must predict this continuous latent directly.

The predicted latent must be passed through the frozen EnCodec decoding path in the same manner validated by the existing project infrastructure.

Document the exact data flow:

```text
waveform
↓
frozen EnCodec encoder
↓
continuous target latent
```

and:

```text
input representation
↓
learned translator
↓
predicted continuous EnCodec latent
↓
frozen EnCodec decoder
↓
reconstructed waveform
```

---

# Translator Architecture

Reuse the exact TASK-05 translator architecture wherever dimensions permit.

Expected conceptual structure:

```text
input projection
↓
temporal ConvNet
↓
temporal-rate conversion
↓
output projection
```

Only make mechanically necessary changes for:

- EnCodec target dimensionality
- EnCodec frame rate
- target sequence length
- temporal upsampling/downsampling ratio

Do not increase:

- depth
- channel width
- receptive field
- parameter count

unless a dimension-compatible implementation is otherwise impossible.

If parameter count changes because the output dimension changes, report:

```text
TASK-05 DAC translator parameter count
TASK-11 EnCodec translator parameter count
absolute difference
relative difference
```

Clearly distinguish unavoidable dimensional changes from intentional capacity changes.

---

# Temporal Alignment Validation

Before training, verify:

1. Input crop duration matches the EnCodec target duration.
2. Predicted and target EnCodec sequences have identical shapes.
3. The EnCodec decoder accepts the predicted tensor without hidden reshaping or truncation.
4. No off-by-one target frames are silently removed.
5. Re-encoding and decoding the oracle continuous latent produces the expected EnCodec reconstruction.
6. The translator output path and oracle path use the same decoder interface.

Add assertions for all shape and duration invariants.

Fail early if alignment is ambiguous.

---

# Dataset and Example Selection

Reuse the exact TASK-05 nested example sets.

The required relationship is:

```text
N=2 examples
⊂
N=8 examples
⊂
N=32 examples
```

Do not create a new random example selection.

Record the complete crop list and verify that it exactly matches TASK-05.

If exact TASK-05 manifests cannot be reused, stop and report the discrepancy rather than silently generating a replacement.

---

# Training Budget

Use the same fixed training budget as TASK-05:

```text
20,000 optimizer steps per N
```

unless the repository's TASK-05 record explicitly specifies a different exact budget.

Use the repository record as the source of truth.

Do not compensate for dataset size by changing step count.

This task must reproduce TASK-05's fixed-compute design.

---

# Optimizer and Training Recipe

Reuse TASK-05 exactly:

```text
optimizer: Adam
learning rate: 1e-3
betas: (0.9, 0.999)
batch size: 1
learning-rate schedule: none
warmup: none
gradient clipping: none
```

If the code or TASK-05 report differs from the summary above, use the recorded TASK-05 configuration and document the discrepancy.

Do not improve the optimizer in this task.

The purpose is a direct DAC-versus-EnCodec target comparison under the same optimization conditions.

---

# Loss

Reuse the TASK-05 continuous regression objective exactly:

```text
MSE
+
cosine loss
```

Use the same weighting and reduction strategy.

Document the exact formula:

```text
L_total =
    lambda_mse * L_mse
    +
    lambda_cos * L_cos
```

Do not tune the weights for EnCodec.

Do not normalize EnCodec targets unless TASK-05 normalized DAC targets in the same way.

---

# Instrumentation

TASK-06 showed that final-step evaluation can be misleading because the training trajectory is unstable.

Log at minimum:

```text
step
learning rate
train MSE
train cosine similarity
total loss
global gradient norm
prediction norm
target norm
norm ratio
prediction variance
target variance
variance ratio
```

If the current infrastructure already provides:

```text
waveform correlation
log-mel similarity
STFT distance
```

at evaluation checkpoints, retain those metrics.

Do not add expensive per-step decoding if it materially changes the controlled training loop. Periodic evaluation is sufficient.

---

# Checkpointing

Save:

```text
latest checkpoint
best checkpoint by the same latent criterion used in TASK-05
best checkpoint by waveform correlation, if already supported
```

Do not use the final checkpoint automatically for listening or final reporting.

For each run, report:

- best checkpoint step
- final checkpoint step
- best metric values
- final metric values
- whether the final checkpoint materially degraded from the best checkpoint

This is required because TASK-06 exposed large peak-to-trough variability.

---

# EnCodec Oracle Baseline

For every crop, compute the frozen codec oracle:

```text
original waveform
↓
EnCodec encoder
↓
oracle continuous latent
↓
EnCodec decoder
↓
EnCodec oracle reconstruction
```

Report:

```text
original vs EnCodec oracle
```

using:

- waveform correlation
- log-mel cosine similarity
- STFT distance
- any established project reconstruction metrics

The EnCodec oracle defines the maximum performance available through the selected frozen EnCodec pathway.

---

# Required Evaluation

For each:

```text
N = 2
N = 8
N = 32
```

evaluate both:

## Latent Reconstruction

- MSE
- cosine similarity
- norm ratio
- variance ratio

## Audio Reconstruction

- waveform correlation
- log-mel similarity
- STFT distance
- the project's existing listening metrics

## Oracle-Relative Performance

For metrics where a ratio is mathematically meaningful, report:

```text
translator performance relative to EnCodec oracle
```

Where a ratio would be misleading, present:

```text
original
EnCodec oracle
translator reconstruction
```

side by side instead.

Do not compare raw EnCodec waveform correlation directly with DAC without also presenting the codec oracle.

---

# Listening Package

For every required `N`, export representative WAV files from:

```text
best checkpoint
```

not only the final checkpoint.

Include:

1. Original waveform
2. EnCodec oracle reconstruction
3. Translator reconstruction
4. Final-checkpoint reconstruction if it differs materially from the best checkpoint

Keep the same example-selection policy used in TASK-05:

- best
- median
- worst

where those labels are determined by the same per-example metric used previously.

Document the ranking metric.

---

# Required Comparison Table

Produce one table containing at least:

| Target | N | Best Step | Latent Cosine | Norm Ratio | Variance Ratio | Wave Corr | Log-Mel Similarity | STFT Distance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DAC | 2 | | | | | | | |
| EnCodec | 2 | | | | | | | |
| DAC | 8 | | | | | | | |
| EnCodec | 8 | | | | | | | |
| DAC | 32 | | | | | | | |
| EnCodec | 32 | | | | | | | |

Reuse TASK-05 DAC values from its machine-readable report.

Do not recompute DAC results unless necessary.

Clearly label metrics that are not directly comparable because the codec oracle differs.

---

# Required Figures

## Figure 1: EnCodec Scaling Curve

X-axis:

```text
N = 2, 8, 32
```

Plot:

- latent cosine
- norm ratio
- variance ratio
- waveform correlation

Use separate panels or appropriate normalization so scales remain readable.

---

## Figure 2: DAC vs EnCodec Scaling

Compare the two targets across:

```text
N = 2, 8, 32
```

Include at minimum:

- latent cosine
- waveform correlation
- variance ratio

Use paired panels, not a single cluttered axis.

---

## Figure 3: Oracle-Relative Audio Quality

For each codec and N, show:

```text
codec oracle
translator reconstruction
```

for:

- waveform correlation
- log-mel similarity
- STFT distance

The figure must make clear that the frozen DAC and EnCodec pathways have different reconstruction ceilings.

---

## Figure 4: Per-Example N=32 Results

Plot N=32 per-example EnCodec:

- latent cosine
- waveform correlation

and compare with the TASK-05 DAC N=32 per-example values if available.

We need to know whether EnCodec:

```text
lifts all examples
```

or:

```text
only benefits an easier subset
```

---

# Decision Rules

## Outcome A: EnCodec moves the boundary

Example qualitative pattern:

```text
N=2   near oracle
N=8   near oracle
N=32  near oracle or dramatically better than DAC
```

Interpretation:

```text
The TASK-05 transition was strongly
target-dependent.

The TemporalConvNet's apparent capacity
boundary was at least partly induced by
DAC target geometry.
```

Do not extend the scale curve automatically.

Stop and report.

---

## Outcome B: Partial improvement

Example qualitative pattern:

```text
EnCodec N=32
substantially outperforms DAC N=32
but remains clearly below EnCodec oracle
```

Interpretation:

```text
Smoother target geometry helps but does
not fully remove the scaling problem.

Target geometry, optimization, and
capacity/interference likely coexist.
```

---

## Outcome C: Similar collapse

Example qualitative pattern:

```text
EnCodec and DAC both transition sharply
between N=8 and N=32
```

Interpretation:

```text
Shift smoothness is not sufficient to
explain the TASK-05 boundary.

Attention should return to optimization,
translator capacity, conditional ambiguity,
or information missing from the input.
```

---

## Outcome D: Positive control failure

Example:

```text
EnCodec cannot closely memorize N=2
```

Interpretation:

```text
The experiment is not yet informative
about scaling.

Investigate representation extraction,
alignment, latent scale, or decoder interface.
```

Do not proceed to `N=8` or `N=32` after a genuine `N=2` pipeline failure.

---

# Execution Gates

## Gate 0: Static validation

Before training:

- verify manifests
- verify tensor shapes
- verify decoder interface
- verify oracle reconstruction
- verify deterministic extraction
- run all existing tests and new tests

---

## Gate 1: N=2

Train and evaluate `N=2`.

Proceed only if:

- loss decreases substantially
- latent cosine approaches the EnCodec oracle target
- decoded audio is recognizably close to the EnCodec oracle
- no shape or alignment problem is present

If the result is below expectation but technically valid, diagnose whether this is optimization or pipeline failure before deciding whether the gate passed.

Do not silently continue.

---

## Gate 2: N=8

Train and evaluate `N=8`.

Compare directly with:

- EnCodec `N=2`
- DAC `N=8`

Proceed to `N=32` if the pipeline remains valid, even if EnCodec shows some degradation. The `N=32` point is required for the main hypothesis.

---

## Gate 3: N=32

Train and evaluate `N=32`.

This is the decisive run.

After evaluation, stop.

Do not run larger N values.

---

# Tests

Add focused tests for:

- EnCodec continuous-latent extraction shape
- deterministic extraction
- crop-duration consistency
- target/prediction shape equality
- decoder acceptance of oracle latent
- decoder acceptance of translator-shaped latent
- manifest equality with TASK-05
- nested N sets
- metric calculation
- checkpoint selection
- report schema

Follow the project's existing testing conventions.

Do not add a large test harness unrelated to this experiment.

---

# Machine-Readable Report

Write a JSON report including:

```text
configuration
git state
checkpoint identifier
manifest identity
representation geometry
translator parameter count
training metrics
best checkpoint
final checkpoint
per-example metrics
oracle metrics
oracle-relative metrics
DAC comparison values
runtime
device
random seeds
```

The report must be sufficient to reconstruct every comparison table and figure.

---

# Notes Updates

Update:

- `notes/DECISIONS.md`
- `notes/MEMORY.md`
- `notes/NEXT.md`
- `notes/SCRATCHPAD.md`

Use the established per-task format.

Correct the truncated tail of `tasks/TASK-10.md` only if the intended original text can be recovered exactly from repository history or prior task material.

Do not invent missing TASK-10 wording.

If it cannot be recovered exactly, record the truncation in the TASK-11 notes and leave TASK-10 unchanged.

---

# Repository State and Commit Discipline

TASK-09 and TASK-10 changes are currently uncommitted.

Before making TASK-11 changes:

1. Record the exact current `git status`.
2. Do not discard, reset, or overwrite TASK-09 or TASK-10 work.
3. Keep TASK-11 changes distinguishable from prior staged and unstaged changes.
4. Do not commit anything unless explicitly asked by the user.
5. Report the final staged, unstaged, and untracked state clearly.

---

# Final Questions

The final report must answer:

> Can the existing TemporalConvNet memorize the continuous EnCodec target at N=2?

> Does EnCodec preserve near-perfect memorization at N=8?

> At N=32, does EnCodec substantially outperform DAC relative to each codec's own oracle?

> Does the EnCodec target move or remove the sharp N=8→N=32 transition observed with DAC?

> Are norm and variance collapse reduced when predicting EnCodec continuous latents?

> Does EnCodec improve all N=32 examples or preferentially improve an easier subset?

> Does the result support target geometry, optimizer instability, capacity/interference, or a combination of these explanations?

> Is continuous EnCodec a materially better regression target than continuous DAC under the same translator, loss, compute budget, and optimizer?

Stop after answering these questions.

Do not:

- extend to N=128 or N=512
- introduce a transformer
- change the optimizer
- add an OT training loss
- switch to discrete tokens
- propose or run another experiment automatically
- commit changes without explicit user instruction