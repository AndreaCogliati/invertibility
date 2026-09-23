# TASK-14: Full GTZAN Pilot on BlueHive

## Purpose

TASK-13 showed that the merged architecture can use temporally ordered MuQ conditioning on training crops, but correct conditioning did not outperform the controls on the four held-out validation and test crops. The likely explanation is that the short pilot used too little training data to demonstrate held-out generalization.

Gate 5 therefore trains on the full committed GTZAN training split and tests whether conditioning dependence emerges with substantially more data.

The current pipeline is:

```text
audio
  -> frozen MuQ last_hidden_state
  -> learned translator
  -> frozen EnCodec continuous latent
  -> frozen EnCodec decoder
  -> reconstructed audio
```

MuQ and EnCodec remain frozen. Only the translator is trained.

## Primary scientific question

> After training on the full committed GTZAN training split, does temporally correct MuQ conditioning improve reconstruction of unseen tracks relative to shuffled and absent conditioning?

The conclusion must be based on held-out validation and test results. Do not pool training examples with held-out examples for the Gate 5 decision.

## Scope

### Included

- Full committed GTZAN training split.
- Frozen MuQ final `last_hidden_state`.
- Current translator architecture on `main`.
- Frozen continuous EnCodec target and decoder.
- Current MSE plus cosine objective.
- Held-out validation during training.
- Final test evaluation.
- Conditioning controls on validation and test data.
- Variance-ratio diagnostics.
- Weights & Biases monitoring.
- Periodic checkpoints and fixed listening outputs.
- Resume and reproducibility verification.
- BlueHive execution through Slurm.

### Excluded

Do not introduce:

- DAC as a training target;
- a transformer;
- an OT loss;
- a waveform, spectral, or decoder-aware training loss;
- optimizer redesign;
- alternate MuQ layers or learned layer mixing;
- genre supervision;
- adversarial training;
- a hyperparameter sweep.

This task tests data scale, not a new model or objective.

## Execution policy

The full Gate 5 training run must execute on BlueHive through a Slurm batch job.

Do not run the full training job locally.

Local or BlueHive login-node execution is allowed only for small, bounded diagnostic work such as:

- static inspection;
- unit tests;
- linting;
- configuration validation;
- manifest validation;
- a tiny CPU smoke test;
- a very short subset run that verifies imports, tensor geometry, caching, W&B initialization, checkpoint writing, and command-line arguments.

Do not use a diagnostic subset run as scientific evidence. Label its outputs clearly as smoke-test artifacts and keep them separate from the full Gate 5 run.

Do not run sustained computation on a BlueHive login node. Submit the full run through `sbatch`.

## Repository preflight

Before changing or running anything:

1. Read:
   - `CLAUDE.md`
   - `notes/MEMORY.md`
   - `notes/DECISIONS.md`
   - `notes/NEXT.md`
   - `TASK-13_RESULTS.md`
   - TASK-13's machine-readable report and resolved configuration.
2. Run `git status`.
3. Record:
   - branch;
   - commit hash;
   - dirty-tree status;
   - staged files;
   - unstaged files;
   - untracked files.
4. Confirm that `experiment/muq-layer-sweep` is present in the current `main` history.
5. Preserve TASK-13 code, runs, reports, and artifacts.
6. Run the full test and lint suites before submission.
7. Do not reset, discard, overwrite, or commit unrelated work.
8. Do not commit unless explicitly requested.

## BlueHive environment discovery

Inspect the existing repository documentation, scripts, prior BlueHive job files, and the current BlueHive environment before writing the batch file.

Do not invent BlueHive account names, partitions, module versions, GPU types, paths, or resource limits. Reuse established project conventions when they exist. If a required value cannot be determined from the repository or environment, use an explicit placeholder in the generated batch file and identify it in the final preparation report.

Do not install dependencies globally. Preserve the project's `uv` workflow and pinned dependency state.

## Submission and resumption helpers

If consistent with repository conventions, create small helper scripts or documented commands for:

- initial submission with `sbatch`;
- checking job state with normal Slurm tools;
- viewing stdout and stderr;
- resubmitting from the latest valid checkpoint;
- identifying the W&B run associated with the Slurm job.

The resumption path must append to the same scientific run lineage and must not create ambiguous duplicate outputs.

Do not submit the full job as part of TASK-14 unless explicitly instructed. TASK-14 prepares, validates, and documents the BlueHive run package and may execute only the bounded smoke test described below.

## Bounded diagnostic run

Create and run a small diagnostic configuration before finalizing the Slurm file.

The diagnostic may use a tiny subset and very few optimizer steps. It must verify at least:

- imports and dependency resolution;
- GTZAN manifest loading;
- one or more training crops;
- MuQ feature extraction or cache loading;
- EnCodec target extraction;
- translator forward and backward passes;
- finite loss;
- checkpoint writing and loading;
- W&B initialization in the intended BlueHive mode;
- output path creation;
- the full training command's argument parsing.

The diagnostic must not:

- use the full GTZAN training split;
- run the full step budget;
- be reported as Gate 5 evidence;
- overwrite TASK-13 or future TASK-14 full-run outputs.

If local hardware cannot exercise a BlueHive-specific path, validate that path through shell syntax checks, configuration checks, and a short submitted diagnostic Slurm job if appropriate. Never run the full training locally.

## Dataset and split

Use the existing committed GTZAN manifest and its current train, validation, and test assignments.

At runtime:

1. Verify the manifest hash.
2. Report the exact number of valid tracks in each split.
3. Exclude only entries already marked invalid or corrupt by the established data pipeline.
4. Verify that checksum-identical files do not cross split boundaries.
5. Do not regenerate or alter the split.
6. Do not combine validation and test tracks.
7. Do not use genre labels for training.
8. Document that the split is checksum-deduplicated but not artist-aware.

## Crop construction and sampling

### Training crops

Use aligned crops whose waveform length obeys the established MuQ and EnCodec frame geometry.

For every sampled crop, retain enough state to reproduce or audit:

- track ID;
- split;
- start and end samples;
- crop length;
- MuQ tensor shape;
- EnCodec tensor shape;
- seed or sampler state.

Training must sample from the full training split rather than rotate over a small fixed set of clips.

Use a deterministic, resumable sampler that:

- visits tracks broadly;
- samples positions across tracks;
- saves sufficient state for continuation;
- never samples validation or test tracks;
- reports per-track exposure so concentration on a small subset can be detected.

### Validation crops

Create a fixed, versioned validation-crop manifest before training.

It must:

- contain only validation tracks;
- use fixed crop offsets;
- remain unchanged throughout training;
- be reused for all checkpoints and conditioning controls;
- include enough tracks and crop positions that the result is not determined by a handful of examples.

### Test crops

Create a separate fixed test-crop manifest.

Use validation metrics for checkpoint selection. Reserve test evaluation for the final selected checkpoint and any diagnostic checkpoints explicitly predefined before training.

## Models

### MuQ

Use only:

```python
with torch.inference_mode():
    output = muq(waveform_24khz, output_hidden_states=False)
    features = output.last_hidden_state
```

Requirements:

- frozen parameters;
- evaluation mode;
- detached output;
- current project checkpoint;
- full temporal sequence;
- no pooling;
- no intermediate-layer mixing.

### Translator

Use the architecture currently on `main` after the layer-sweep merge.

Record:

- architecture identifier;
- exact parameter count;
- input and output geometry;
- temporal conversion mechanism;
- initialization seed.

Do not alter model capacity during this task.

### EnCodec

Use the existing frozen `facebook/encodec_24khz` integration and the same continuous latent definition used in TASK-11 through TASK-13.

Preserve the target `.clone()` correction where required.

Use the frozen EnCodec oracle reconstruction as the reconstruction ceiling.

## Training configuration

Use the TASK-13 recipe as the baseline:

- same loss and weights;
- same optimizer and learning rate;
- same batch-size behavior;
- same initialization policy;
- no new warmup or decay;
- no new gradient clipping;
- no EMA;
- no normalization changes.

Scale-related implementation changes are allowed only when mechanically necessary for full GTZAN or BlueHive execution, such as:

- data-loader workers;
- cache placement;
- logging cadence;
- checkpoint cadence;
- gradient accumulation;
- distributed or device launch mechanics already supported by the repository.

Document every such change. Do not change the scientific model, target, or objective.

### Training budget

Use a versioned configuration with a predefined maximum optimizer-step budget and predefined evaluation checkpoints.

Include at least:

- an early diagnostic checkpoint;
- intermediate checkpoints;
- final checkpoint;
- best validation checkpoint.

Define the schedule before the full job starts. Do not extend training merely because the desired result has not appeared.

## Weights & Biases

Monitor the full BlueHive job with the W&B integration validated in TASK-13.

### Run identity

Create a distinct run with:

- task: `TASK-14`;
- gate: `Gate 5 Full GTZAN Pilot`;
- tags including `task14`, `gate5`, `gtzan-full`, `muq`, `encodec`, `bluehive`, and the architecture identifier.

Store the W&B run ID in:

- local run manifest;
- checkpoints;
- Slurm log metadata;
- `notes/NEXT.md`;
- final report.

If compute nodes cannot reach W&B reliably, use the repository's supported offline mode and preserve the offline run directory for later synchronization. Do not silently drop metrics.

### Required metadata

Log:

- Slurm job ID;
- hostname and device;
- Git commit and dirty-tree status;
- dataset and crop-manifest hashes;
- split counts;
- model identifiers;
- translator architecture and parameter count;
- resolved configuration;
- seeds;
- package and platform versions;
- optimizer and step budget;
- crop geometry;
- batch size and accumulation;
- checkpoint and validation cadence;
- scratch and run identifiers without exposing secrets.

### Required metrics

Log over time:

- total loss;
- MSE component;
- cosine component;
- learning rate;
- gradient norm if already available without altering optimization;
- throughput and wall-clock metrics;
- samples and unique tracks seen;
- norm ratio;
- variance ratio;
- latent cosine;
- latent MSE;
- validation waveform correlation;
- oracle-relative waveform recovery;
- log-mel similarity;
- established STFT metric;
- best-so-far metric and checkpoint step.

Log control-specific validation and final test summaries under clear namespaces:

```text
controls/correct/...
controls/shuffled_across/...
controls/shuffled_time/...
controls/constant_zero/...
```

## Conditioning controls

Evaluate the same model checkpoint and same held-out crop under:

1. `correct`
2. `shuffled_across`
3. `shuffled_time`
4. `constant_zero`

No model weights may change between conditions.

### Correct

Use the crop's correctly aligned MuQ representation.

### Shuffled across

Use conditioning from a different held-out track.

Requirements:

- deterministic pairing;
- no self-pairing;
- matched geometry;
- saved pairing manifest.

### Shuffled time

Permute MuQ frames within the same crop.

Requirements:

- deterministic non-identity permutation;
- same values and dimensions;
- saved permutation seed or indices.

### Constant zero

Use the control implementation validated in TASK-13.

Training controls may be reported separately as memorization diagnostics, but must never be pooled with held-out results.

## Primary endpoint

The primary endpoint is the paired held-out difference in oracle-relative waveform recovery between correct conditioning and each control.

Report separately for:

- validation;
- test;
- validation and test combined only as a secondary held-out summary.

A successful Gate 5 result requires correct conditioning to outperform all three controls on held-out data.

## Required metrics

For every held-out example and condition, report:

### Audio

- original-to-oracle waveform correlation;
- original-to-translator waveform correlation;
- translator-to-oracle waveform correlation if supported;
- oracle-relative waveform recovery;
- log-mel similarity;
- established STFT distance;
- alignment or lag applied.

### Latent

- MSE;
- cosine similarity;
- norm ratio;
- variance ratio;
- existing framewise metrics.

Variance ratio is the primary collapse diagnostic. Cosine and MSE are secondary.

### Paired deltas

For every example and control, compute:

```text
correct metric - control metric
```

Report:

- mean and median paired delta;
- distribution;
- proportion of positive deltas;
- confidence interval or deterministic bootstrap interval if correctly implemented;
- paired test if explicitly defined.

Statistical significance alone is insufficient. The effect must also be practically meaningful and audible.

## Checkpoint selection

Select the best checkpoint using a validation-only metric defined before training.

Use validation correct-conditioning oracle-relative waveform recovery unless the existing project already defines a different valid criterion.

Do not select by test metrics or informal listening.

Report:

- best and final checkpoint steps;
- best and final validation values;
- best-to-final delta;
- whether final lies in a trough;
- checkpoint lineage across any resumed Slurm jobs.

## Listening evaluation

For a fixed representative held-out subset, save:

1. original;
2. EnCodec oracle;
3. correct-conditioning reconstruction;
4. `shuffled_across` reconstruction;
5. `shuffled_time` reconstruction;
6. `constant_zero` reconstruction.

Use identical alignment, duration, and loudness handling.

Listening notes must address source correspondence, including rhythm, onset timing, pitch contour, instrumentation, spectral envelope, dynamics, and whether correct conditioning is more source-specific than controls.

Do not claim success solely because audio is pleasant.

## Sanity checks

Before interpretation, verify:

- disjoint train, validation, and test tracks;
- duplicate checksums do not cross splits;
- oracle metrics are invariant across controls;
- controls alter only conditioning;
- no source or target waveform leaks into translator inputs;
- intended crop and conditioning alignment;
- EnCodec target/output alignment;
- valid tensor lengths and frame rates;
- no NaN or Inf;
- W&B and local logs agree;
- checkpoint/resume preserves the intended sampler and RNG state;
- broad training-track exposure;
- validation tracks never enter optimization;
- Slurm job metadata matches the recorded run manifest.

## Decision outcomes

### Outcome A: Held-out conditioning generalization

Correct conditioning materially and consistently outperforms all controls on validation and test data, with metrics and listening aligned.

Interpretation: full-scale data resolves the tiny-data limitation observed in TASK-13.

### Outcome B: Partial generalization

Correct conditioning beats controls on some metrics or subsets, but the effect is small, inconsistent, or not audibly clear.

Interpretation: more data produces a transferable signal, but the current system is not yet robust.

### Outcome C: Latent generalization without audio generalization

Correct conditioning improves latent metrics or variance preservation, but audio metrics and listening remain comparable to controls.

Interpretation: MuQ predicts aspects of EnCodec space that are not sufficiently decoder-relevant.

### Outcome D: No held-out conditioning dependence

Correct conditioning remains indistinguishable from controls on validation and test data.

Interpretation: TASK-13 is not explained solely by its tiny training set. Investigate architecture, objective geometry, target ambiguity, or information availability before further scaling.

### Outcome E: Collapse or instability

Variance collapses, best-to-final degradation is severe, or optimization instability prevents a reliable comparison.

Interpretation: Gate 5 is inconclusive as a generalization test.

### Outcome F: Pipeline failure

Leakage, target misalignment, invalid controls, unstable oracle metrics, or tensor errors are found.

Interpretation: correct the pipeline and rerun affected conditions before drawing scientific conclusions.

## Gate 5 pass criteria

Gate 5 passes scientifically only if:

- correct conditioning outperforms every control on held-out validation data;
- the result replicates on the held-out test set;
- gains appear across multiple tracks rather than a few outliers;
- listening supports greater source correspondence;
- variance ratio does not indicate collapse;
- no leakage, alignment, control, or Slurm-execution defect explains the result.

Training performance cannot convert held-out failure into a pass.

## Required deliverables

1. Versioned full-run configuration.
2. Versioned diagnostic configuration.
3. Slurm batch file.
4. Submission and resume instructions or helper scripts.
5. Starting repository state.
6. BlueHive environment and resource summary.
7. Dataset and crop-manifest hashes.
8. Exact valid split counts.
9. Diagnostic-run report.
10. Full test and lint results.
11. W&B run naming and logging verification.
12. Expected output, checkpoint, log, and artifact paths.
13. Syntax-validated Slurm batch file.
14. Machine-readable run manifest template.
15. Updates to project notes.
16. Exact final `git status`.

After the full Slurm job is explicitly submitted and completes, its result package must additionally contain:

17. Slurm job ID and terminal state.
18. W&B run ID or URL and local/offline path if applicable.
19. Training and validation curves.
20. Best-versus-final comparison.
21. Aggregate validation results.
22. Aggregate test results.
23. Per-example control metrics.
24. Conditioning-delta analysis.
25. Variance and norm diagnostics.
26. Listening package and manifest.
27. JSON report and per-example CSV.
28. Outcome A-F classification.
29. Explicit Gate 5 pass/fail decision.
30. Final job and repository status.

## Final preparation report structure

```markdown
# TASK-14 BlueHive Preparation Report

## Executive Summary
## Repository State
## BlueHive Environment
## Dataset and Split Verification
## Architecture and Configuration
## Diagnostic Run
## Weights & Biases Verification
## Slurm Batch File
## Resource Requests
## Submission Command
## Resume Procedure
## Expected Outputs
## Tests and Lint
## Files Changed
## Known Placeholders or Blockers
## Final Git Status
```

## Final scientific report structure

```markdown
# TASK-14 Results: Full GTZAN Pilot

## Executive Summary
## Scientific Question
## Repository and Slurm State
## Dataset and Split Verification
## Crop Manifests
## Architecture and Configuration
## Weights & Biases Run
## Training Dynamics
## Validation Learning Curve
## Best vs Final Checkpoint
## Held-Out Conditioning Controls
## Validation Results
## Test Results
## Variance and Collapse Diagnostics
## Per-Example Analysis
## Listening Observations
## Outcome Classification
## Gate 5 Decision
## Supported Conclusions
## Limitations
## Files Changed
## Test and Lint Results
## Final Git Status
```

## Required conclusion discipline

Do not claim that:

- good training reconstruction demonstrates generalization;
- a diagnostic subset is Gate 5 evidence;
- high latent cosine demonstrates successful inversion;
- full GTZAN establishes asymptotic performance;
- held-out failure proves MuQ lacks reconstructive information;
- success on GTZAN guarantees cross-dataset generalization;
- genre labels contributed to reconstruction;
- pooled metrics substitute for held-out results;
- a Slurm job was submitted or completed unless its job ID and state were actually recorded.

Do not run the full training locally. Do not submit the full BlueHive job unless explicitly instructed. Do not commit unless explicitly requested.
