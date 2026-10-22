Create TASK-19: N=128 data × recipe experiment on TAMU ACES PVC, using the TASK-18 MuQ → BigVGAN-mel pipeline.

Scientific question:
Is held-out mel reconstruction currently limited by training data, by optimization, or by both?

Arms (2 × 2):
| Arm  | Training tracks | Recipe |
|------|-----------------|--------|
| A32  | 32  | TASK-18 recipe; reuse the existing TASK-18 checkpoints, do not retrain |
| A128 | 128 | TASK-18 recipe |
| B32  | 32  | improved recipe |
| B128 | 128 | improved recipe |

Preflight:
- Read CLAUDE.md, notes/{MEMORY,DECISIONS,NEXT}.md, tasks/TASK-18.md and the TASK-18 results/config.
- Record git status and commit. The TASK-18 notes and the WebProxy edit to aces/run_task18_v1.slurm should be committed before any submission; if they are not, stop and tell me.
- Record sacct/SU usage for TASK-18 job 2240151 and use it to estimate the SU cost of each new arm.

Splits:
- 128 training tracks = the TASK-18 32 + 96 new, genre-balanced, deterministic.
- Primary held-out: the identical TASK-18 8 validation + 8 test tracks and crop manifests.
- Secondary held-out: 24 additional validation + 24 additional test tracks, reported separately.
- Before training, check all training and held-out tracks against the published GTZAN fault list (repeated recordings and versions, not just checksum duplicates). Do not let repeats cross splits. If any TASK-18 track is affected, report it and stop rather than silently changing the TASK-18 manifests.
- Save all manifests and hashes.

Fixed across arms:
MuQ last_hidden_state, physical-time alignment, the 1.27M-parameter translator, BigVGAN 100-band mel target, decode-time floor clamp, frozen BigVGAN-v2, crop geometry, seed, step budget (same as TASK-18), evaluation code, controls and metrics. No architecture or loss changes.

Improved recipe (B):
- Define it in the task file before any run: learning-rate warmup plus decay, and batch size > 1 only if the existing code supports it without architectural change.
- Document every setting. Do not tune it on held-out results.

Evaluation (all four arms, including re-evaluating the A32 checkpoint on the secondary held-out set):
- Primary endpoint: mel L1 for unseen crops of training tracks, validation tracks and test tracks, compared with BigVGAN round-trip error and TASK-18 values.
- Controls: shuffled-across, shuffled-time, constant-zero, training-set average mel.
- Secondary: variance ratio, loudness and brightness contours, chroma, onset envelope.
- Checkpoint selection: validation mel L1 only. Report best and final checkpoints.
- Log full training and validation curves to W&B, one run per arm, grouped as TASK-19.

Listening:
- Same fixed held-out crops for all four arms.
- Blinded package: randomized file names with a separate, unopened key file. Include original, BigVGAN oracle, each arm's correct prediction, and controls.

Interpretation table (pre-registered):
- A128 much better than A32 → data-limited.
- B32 much better than A32, A128 ≈ A32 → optimization-limited.
- Both improve → both matter.
- Neither improves → capacity or objective limit.
- Training error falls but held-out does not → generalization gap.
Define "much better" numerically in the task file before training.

ACES execution:
- One Slurm batch file per arm, one PVC GPU each, using the verified XPU environment and WebProxy setup.
- Run only bounded smoke tests interactively.
- Do not submit the full jobs unless I explicitly say so.

Deliverables: task file, manifests, fault-list report, Slurm files, smoke-test report, SU estimate, then (after the runs) aggregate and per-example results, curves, blinded listening package, outcome classification against the pre-registered table, updated notes and exact git status.

Do not commit unless explicitly requested.
