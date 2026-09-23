# TASK-12: EnCodec N=32 Compute Ablation

## Purpose

TASK-11 showed that replacing the DAC continuous latent target with the continuous EnCodec latent target substantially reduced the N=8→N=32 collapse:

- DAC N=32: waveform correlation 0.469, norm ratio 0.499, variance ratio 0.177, oracle-relative recovery 48.4%.
- EnCodec N=32: waveform correlation 0.724, norm ratio 0.985, variance ratio 0.650, oracle-relative recovery 79.2%.
- The EnCodec translator used 262,656 parameters versus 15,863,808 for DAC, ruling out a larger-model confound.

The remaining question is whether the residual N=32 gap is primarily due to the fixed 20k-step training budget or whether it persists after substantially more optimization.

## Scientific Question

Does increasing training compute improve the N=32 EnCodec condition beyond the TASK-11 baseline, or has the current ConvNet already reached most of its achievable performance?

---

## Experimental Design

Use the completed TASK-11 EnCodec N=32 run as the baseline.

Run:

| Condition | Steps |
|----------|--------|
| Baseline | 20,000 |
| Compute Ablation 1 | 80,000 |
| Compute Ablation 2 | 320,000 |

Do not run N=2, N=8, N=128, or N=512.

The only intended independent variable is:

```text
optimizer steps
```

---

## Required Controls

Keep identical to TASK-11:

- Exact N=32 manifest
- Exact train/eval examples
- Input representation
- Continuous EnCodec target definition
- EnCodec checkpoint
- Target alignment
- ConvNet architecture
- MSE + cosine loss
- Loss weights
- Batch size 1
- Adam
- Learning rate 1e-3
- Betas (0.9, 0.999)
- No warmup
- No decay
- No gradient clipping
- No EMA
- No normalization changes
- Same seed
- Same checkpoint-selection procedure
- Same evaluation code

Do not modify the optimizer recipe.

This is a compute-ablation experiment, not an optimization-improvement experiment.

---

## Reproducibility Checks

Before running:

1. Record `git status`
2. Record current commit hash
3. Confirm TASK-11 outputs are preserved
4. Confirm identical N=32 configuration
5. Record staged, unstaged, and untracked files

Do not commit unless explicitly instructed.

---

## Important TASK-11 Fixes

Preserve both fixes introduced during TASK-11:

### EnCodec target extraction

Freshly extracted targets must be cloned where required:

```python
target = target.clone()
```

to avoid the inference-tensor/autograd failure.

### DAC reference metrics

If DAC appears in comparison tables, preserve the documented 8-sample DAC decoder delay correction.

Do not recalculate DAC oracle metrics using naive unaligned waveform comparison.

---

## Preferred Training Strategy

For the cleanest comparison:

- Initialize each condition from the same seed and initial weights.
- Train independently to 20k, 80k, and 320k.

If instead a single run is extended to 320k and intermediate checkpoints are evaluated, clearly document that the results come from one training trajectory.

---

## Checkpointing

Save and evaluate:

- Best checkpoint
- Final checkpoint
- 20k checkpoint
- 80k checkpoint
- 320k checkpoint

Record:

- Best step
- Best metric
- Final metric
- Best-to-final delta

Do not judge performance using only the final checkpoint.

TASK-06 showed that final checkpoints can lie in optimization troughs.

---

## Primary Endpoint

Primary endpoint:

> Best-checkpoint oracle-relative waveform recovery at N=32 as a function of training compute.

Baseline from TASK-11:

```text
79.2% oracle-relative recovery at 20k steps
```

Compare this directly against 80k and 320k.

---

## Required Metrics

### Audio Metrics

Report for best and final checkpoints:

- Waveform correlation
- Oracle-relative waveform recovery
- Log-mel similarity
- STFT metric (if already implemented)

### Latent Metrics

Report:

- Latent MSE
- Latent cosine
- Norm ratio
- Variance ratio

### Per-Example Metrics

For all 32 examples report:

- Example ID
- Oracle metric
- 20k result
- 80k result
- 320k result
- Oracle-relative performance
- Latent MSE
- Latent cosine
- Norm ratio
- Variance ratio

---

## Derived Analyses

### Compute Response Curve

Plot or tabulate:

- Waveform correlation
- Oracle-relative recovery
- Latent MSE
- Latent cosine
- Norm ratio
- Variance ratio

for:

```text
20k → 80k → 320k
```

### Marginal Gains

Compute:

```text
20k → 80k
80k → 320k
20k → 320k
```

for all primary metrics.

### Per-Example Rank Stability

Determine whether extra compute:

- Helps all examples similarly
- Mainly helps easy examples
- Mainly helps hard examples
- Changes example rankings

### Latent-vs-Audio Improvement

Determine whether improvements in:

- Latent MSE
- Latent cosine
- Variance ratio

track improvements in audio reconstruction.

TASK-11 showed that very high latent fit can coexist with substantial audio variation.

---

## Listening Package

For representative examples (or all 32 if practical), generate:

```text
original
EnCodec oracle
20k best checkpoint
80k best checkpoint
320k best checkpoint
```

Keep loudness handling consistent.

Best-checkpoint audio is the primary listening result.

---

## Outcome Criteria

### Outcome A: Strong Compute Response

Characteristics:

- Large gains at 80k and 320k
- Performance approaches oracle
- Variance ratio moves substantially toward 1.0
- Norm ratio stays near 1.0

Interpretation:

> The remaining TASK-11 gap was largely compute-limited.

### Outcome B: Partial Response

Characteristics:

- Clear gains at 80k
- Smaller gains at 320k
- Meaningful residual gap remains

Interpretation:

> Compute helps, but architecture, objective, or information limits remain.

### Outcome C: Latent Improves, Audio Does Not

Characteristics:

- Better latent metrics
- Small audio gains

Interpretation:

> Additional optimization improves coordinate matching without proportional reconstruction benefit.

### Outcome D: Minimal Response

Characteristics:

- Little change from 20k

Interpretation:

> Compute is not the dominant remaining bottleneck.

### Outcome E: Instability-Limited

Characteristics:

- Better best checkpoints
- Poor final checkpoints
- Large oscillations

Interpretation:

> Optimization instability remains a major confound.

---

## Deliverables

Produce:

1. JSON results
2. Aggregate tables
3. Per-example results
4. Best vs final checkpoint comparisons
5. Compute-response plots
6. Marginal-gain analysis
7. Rank-stability analysis
8. Latent-vs-audio analysis
9. Listening package
10. Updated notes
11. Final git status

Do not commit unless explicitly instructed.

---

## Final Report Structure

```markdown
# TASK-12 Results

## Executive Summary
## Repository State
## Experimental Verification
## Aggregate Results
## Oracle-Relative Results
## Norm and Variance Behavior
## Best vs Final Checkpoints
## Optimization Stability
## Per-Example Analysis
## Latent-to-Audio Analysis
## Listening Observations
## Outcome Classification
## Supported Conclusions
## Remaining Ambiguities
## Files Changed
## Final Git Status
```