Implement TASK-20: one N=512 data-scaling run with the current MuQ → mel → BigVGAN pipeline on ACES.

First inspect the TASK-19 implementation, configs, split manifests, and results. Reuse existing infrastructure rather than creating a parallel pipeline.

Experiment
- Train one model from scratch using TASK-19 recipe A.
- Keep the translator architecture, MuQ representation, mel preprocessing, frozen BigVGAN, loss, optimizer, learning rate, batch size, precision, and seed unchanged from A128.
- Change only training-set size and introduce validation-based early stopping.
- Target 512 training tracks, including all original 128. Add tracks deterministically, maintaining genre balance where feasible.
- Preserve all existing val/test and val2/test2 tracks and evaluation crops exactly. Exclude every held-out track from training.
- If fewer than 512 eligible tracks exist, use the largest eligible set and report the actual N. Never shrink or contaminate held-out sets.

Early stopping
- Select checkpoints using ONLY primary validation correct-conditioning mel L1, with the existing metric implementation.
- Proposed settings: validate every 250 optimizer steps; patience 8 validation checks; min_delta 0; maximum 20,000 steps.
- Save the best checkpoint and a separate resumable last checkpoint.
- Preserve early-stopping state across resumes.
- Do not use test, val2, test2, listening scores, or combined val+test metrics for checkpoint selection or stopping.
- Keep the learning-rate recipe unchanged; do not introduce a new schedule.

Checks and execution
- Verify split disjointness, inclusion of the original 128 tracks, crop/target alignment, and unchanged model/config settings.
- Run a short smoke test before the full run.
- Reuse the working ACES PVC/XPU environment and Slurm configuration.
- Log configuration, seed, commit, split-manifest hashes, actual N, learning rate, training/validation curves, best step, and stopping reason to W&B.
- Prepare the runnable Slurm script and exact submission command. Do not submit automatically.

Evaluation and reporting
- Evaluate the selected checkpoint using the existing TASK-19 metrics and controls.
- Report primary validation and test separately, and secondary validation and test separately. Retain combined summaries only for comparison with TASK-19.
- Compare against A128 and A32 on identical held-out crops, using paired track-clustered confidence intervals.
- Clearly distinguish held-out-track performance from unseen crops of training tracks.
- Create a small blinded listening comparison of A128, TASK-20, and the BigVGAN oracle on the same held-out examples, with a separate key.
- State that this is a single-seed scaling experiment, not a measurement of MuQ's information ceiling.

Deliver the config, minimal code changes, tests, Slurm script, and a concise TASK-20 specification. After execution, generate results and analysis reports using the existing conventions.
