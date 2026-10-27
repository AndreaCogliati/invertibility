Implement TASK-22: first-layer MuQ → mel reconstruction.

Goal
Test whether the output of MuQ’s first encoder block supports better reconstruction than the representation used in TASK-21, with the current translator and reconstruction objective.

Inspect the existing implementation and TASK-20/21 artifacts first. Reuse infrastructure and verify assumptions against the code.

1. Representation
- Extract the output of the FIRST encoder block, not the input embedding or frontend output.
- Inspect the installed MuQ implementation to establish the correct hidden-state index. Do not assume hidden_states[0] means the first encoder block.
- Document the exact module, index, tensor shape, frame rate, and extraction settings.
- Freeze MuQ and use evaluation mode.
- Verify feature/target temporal alignment and valid-frame masking.
- Use a separate, explicitly layer-keyed feature cache. Never overwrite or accidentally reuse the current-layer cache.
- Preserve the existing feature-normalization policy. If statistics are learned from data, recompute them using training data only.

2. Controlled comparison
- Train the translator FROM SCRATCH using the same initialization seed and recipe as TASK-20/21.
- Preserve the exact N512 tracks, finite training-crop pool, sampling policy, held-out splits, evaluation crops, mel targets, and frozen BigVGAN.
- Keep translator architecture, loss, optimizer, learning rate, batch size, precision, and clipping unchanged.
- Do not add fresh crops, latent-consistency loss, adversarial loss, or a new learning-rate schedule.
- Verify feature dimensions are compatible with the existing translator. If not, document the minimum required input-layer change and resulting parameter-count difference. Do not silently redesign the model.
- Record a config diff showing every change from the baseline.

3. Training
- Train for 200,000 optimizer steps with NO patience-based early stopping.
- Validate every 250 steps through step 20,000, then every 1,000 steps, matching TASK-21 where applicable.
- Select the best checkpoint using ONLY primary validation correct-conditioning mel L1.
- Save resumable last checkpoints, the best checkpoint, and checkpoints at steps 60,000 and 200,000.
- Preserve optimizer, scaler, RNG, sampler state where supported, and global step across resumes.
- Stop early only for genuine execution failures, not validation deterioration.
- Use existing ACES PVC/XPU and Slurm infrastructure, respecting verified walltime limits.

4. Tests and monitoring
- Test layer selection, cache separation, split disjointness, alignment, finite features, and checkpoint resumption.
- Run a short smoke test in a separate output directory.
- Log to W&B: layer identity, config, seed, commit, manifest hashes, training loss, primary validation loss, learning rate, gradient norm, throughput, runtime, effective dataset passes, and best step.
- Plot memorized-core, unseen-training-crop, and primary-validation curves.
- Keep test/test2 out of training-time monitoring and all selection decisions.

5. Evaluation
- Compare with TASK-21’s current-layer baseline at MATCHED optimizer steps, including 60,000 and 200,000 where checkpoints exist.
- Also compare validation-selected best checkpoints within the SAME 200,000-step window and validation-check schedule.
- Inspect TASK-21 history to identify its best checkpoint within that window; do not assume it.
- If a required baseline checkpoint is unavailable, report the limitation rather than substitute a different step without disclosure.
- Report primary validation, primary test, secondary validation, and secondary test separately.
- Use identical held-out crops and paired track-clustered confidence intervals.
- Report per-track differences and distinguish unseen tracks from unseen crops of training tracks.
- Create a blinded listening comparison using the same held-out excerpts for first-layer, current-layer, and BigVGAN oracle outputs, with a separate key.
- Treat listening results as pending until a human supplies ratings.
- Note that this is a single-seed comparison using reused evaluation sets.

Interpretation
A first-layer advantage demonstrates greater recoverability with this translator and training recipe. It does not prove deeper layers discarded information. A null result does not establish equal information content.

Deliver
- Minimal code/config changes and tests.
- Layer-extraction verification and config diff.
- Slurm script, exact submission command, and resource estimate.
- TASK-22 specification and results-report template.

Prepare everything, but do not submit automatically.
