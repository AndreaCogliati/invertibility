# TASK-13: Gate 4 Revalidation with the Merged Architecture

## Context and prior results

The branch `experiment/muq-layer-sweep` has been merged into `main`. Revalidate Gate 4 using the architecture now present on `main`; do not recreate or assume the pre-merge architecture.

The previous Gate 4 run used the earlier architecture:

- Checkpoint/resume equivalence passed.
- `shuffled_across` conditioning was worse than correct conditioning.
- `constant_zero` conditioning was worse than correct conditioning.
- `shuffled_time` did not clearly underperform correct conditioning, so Gate 4 was not fully passed and Gate 5 remained blocked.

Subsequent EnCodec experiments changed the main reconstruction path and established the current baseline:

- Pipeline: `audio -> frozen MuQ last_hidden_state -> learned translator -> frozen EnCodec continuous latent -> frozen EnCodec decoder -> reconstructed audio`.
- DAC is retained only as historical evidence and is not the target for this task.
- At N=32, the EnCodec translator reached 79.2% of its oracle ceiling at 20k steps, 88.0% at 80k, and 92.7% at 320k.
- The 320k variance ratio was 0.738, with no DAC-style norm collapse.
- Variance ratio is the primary latent-collapse diagnostic. Cosine similarity and MSE are secondary diagnostics.

## Objective

Repeat the complete Gate 4 short-pilot protocol on the merged architecture and determine whether it uses temporally ordered MuQ conditioning rather than dataset-level statistics or temporally unordered features.

The primary question is:

> Does correct MuQ conditioning materially outperform all three conditioning controls, especially `shuffled_time`, with the architecture now on `main`?

Do not proceed to Gate 5 in this task.

## Preflight

Before changing or running anything:

1. Read `CLAUDE.md`, `notes/MEMORY.md`, `notes/DECISIONS.md`, and `notes/NEXT.md`.
2. Run `git status` and record the current commit hash, branch, staged files, unstaged files, and untracked files.
3. Inspect the merged architecture and the existing Gate 4 implementation, configuration, tests, and artifacts.
4. Preserve all prior runs and artifacts. Do not overwrite them.
5. Do not reset, discard, or commit unrelated work.
6. Do not commit unless explicitly requested.

## Experimental controls

Evaluate the same trained checkpoint with:

1. `correct`: correctly aligned MuQ conditioning.
2. `shuffled_across`: conditioning taken from another example.
3. `shuffled_time`: conditioning frames permuted in time within the example.
4. `constant_zero`: constant or zero conditioning, matching the established Gate 4 control.

Use the same deterministic pilot examples and fixed validation crops across conditions. Ensure that only the conditioning intervention changes. Verify control construction with tests rather than assuming it is correct.

Keep the current project pipeline fixed:

- frozen MuQ `last_hidden_state` input;
- translator architecture currently on `main` after the merge;
- frozen continuous EnCodec target and decoder;
- current loss, optimizer, preprocessing, alignment, and checkpoint policy unless a code defect prevents execution;
- no DAC training target;
- no transformer, OT loss, optimizer redesign, or unrelated architecture change.

If the merged architecture mechanically requires a configuration change, document it precisely and keep it identical across all four conditions.

## Weights & Biases monitoring

Training must be monitored with Weights & Biases. Use the repository's existing W&B integration if present; extend it minimally if required.

Create one W&B run for the Gate 4 training run and log enough information to diagnose convergence, collapse, instability, and conditioning dependence.

### Required W&B configuration and metadata

Log or store in the run configuration:

- task: `TASK-13`;
- gate: `Gate 4 revalidation`;
- Git commit and dirty-tree status;
- architecture identifier and parameter count;
- MuQ and EnCodec model identifiers;
- resolved YAML configuration;
- dataset manifest path and hash;
- random seed;
- device and dtype;
- crop geometry and tensor shapes;
- optimizer, learning rate, loss weights, batch size, and total steps;
- checkpoint-selection metric;
- W&B run ID in the local run manifest and final report.

Use explicit, stable run naming and tags that identify `TASK-13`, `gate4`, `encodec`, and the merged architecture. Do not hard-code credentials or API keys. Support the repository's normal authenticated, offline, or disabled modes without silently losing local metrics.

### Required W&B time-series metrics

Log at the existing training/evaluation cadence:

- total training loss;
- MSE loss component;
- cosine loss component;
- validation loss;
- learning rate;
- gradient norm if it is already available without altering optimization;
- predicted/target norm ratio;
- predicted/target variance ratio;
- latent cosine similarity;
- latent MSE;
- waveform correlation;
- oracle-relative waveform recovery;
- log-mel similarity and STFT metric if already implemented;
- checkpoint step and best-so-far metric;
- wall-clock or throughput metrics already supported by the project.

Do not change optimization behavior merely to obtain a metric. Variance ratio is the primary collapse diagnostic.

### Required W&B control summaries

For each of the four conditioning conditions, log a clearly namespaced final summary containing:

- waveform correlation;
- oracle-relative waveform recovery;
- log-mel/STFT metrics already supported;
- variance ratio;
- norm ratio;
- latent cosine;
- latent MSE;
- delta from correct conditioning;
- per-example metrics or an artifact containing them.

Use names such as:

```text
controls/correct/...
controls/shuffled_across/...
controls/shuffled_time/...
controls/constant_zero/...
```

Upload or link the machine-readable results, resolved configuration, plots, and listening manifest as W&B artifacts when supported by the existing integration. Do not upload large audio collections blindly; follow existing artifact policy and record local paths when audio remains local.

## Checkpoint and resume verification

Repeat Gate 4 checkpoint/resume equivalence with the merged architecture.

Record:

- uninterrupted result;
- resumed result;
- comparison metric and tolerance;
- maximum observed difference;
- whether equivalence passed;
- exact checkpoint lineage.

Do not reuse the old MPS tolerance automatically. Re-measure it for this run and justify any tolerance change.

## Evaluation

For each fixed pilot example, save or preserve:

- original waveform;
- EnCodec oracle reconstruction;
- correct-conditioning reconstruction;
- `shuffled_across` reconstruction;
- `shuffled_time` reconstruction;
- `constant_zero` reconstruction.

Report aggregate and per-example metrics for all four conditions. Listening comparisons are required and must use consistent loudness and alignment handling.

The primary comparison is correct conditioning versus each control. Report absolute values and paired deltas. Do not infer success from training loss alone.

## Gate 4 decision rule

Gate 4 passes only if:

- checkpoint/resume equivalence passes;
- correct conditioning materially outperforms `shuffled_across`;
- correct conditioning materially outperforms `constant_zero`;
- correct conditioning materially outperforms `shuffled_time`;
- objective metrics and listening evidence are directionally consistent;
- variance ratio does not indicate latent collapse;
- no pipeline, alignment, leakage, or control-construction defect explains the result.

If `shuffled_time` again matches correct conditioning, classify Gate 4 as not fully passed and analyze whether the result is consistent with temporal invariance, insufficient receptive-field use, leakage, or a defective control. Do not proceed to Gate 5.

## Required deliverables

Produce:

1. a machine-readable run configuration;
2. W&B run URL or run ID, plus local/offline run location if applicable;
3. training and validation curves;
4. aggregate control-comparison table;
5. per-example control results;
6. variance, norm, cosine, MSE, and audio-level diagnostics;
7. checkpoint/resume equivalence report;
8. listening package and manifest;
9. concise result summary;
10. explicit Gate 4 pass/fail decision;
11. updates to `notes/MEMORY.md`, `notes/DECISIONS.md`, and `notes/NEXT.md` as appropriate;
12. exact final `git status`.

## Final report structure

```markdown
# TASK-13 Results

## Executive Summary
## Prior Result Being Re-tested
## Repository and Merge Verification
## Architecture and Experimental Configuration
## Weights & Biases Run
## Training Dynamics
## Checkpoint/Resume Equivalence
## Conditioning-Control Results
## Variance and Collapse Diagnostics
## Per-Example Analysis
## Listening Observations
## Gate 4 Decision
## Supported Conclusions
## Remaining Ambiguities
## Files Changed
## Final Git Status
```

## Constraints

- Do not proceed to Gate 5.
- Do not change the scientific target from continuous EnCodec latents.
- Do not introduce a new loss, optimizer recipe, transformer, or OT objective.
- Do not overwrite prior runs or artifacts.
- Do not commit unless explicitly requested.
- Do not claim that MuQ inversion succeeds solely from low loss or high latent cosine.
