Create the next task for a 32/8/8 held-out generalization experiment using the MuQ → BigVGAN-mel → BigVGAN pipeline on TAMU ACES PVC.

Scientific question:
Can a translator trained on 32 GTZAN tracks predict BigVGAN-compatible mel spectrograms for unseen crops and unseen tracks from frozen MuQ last_hidden_state?

Use the corrected, ACES-validated TASK-15 pipeline:

audio
→ frozen MuQ last_hidden_state
→ physical-time MuQ-to-mel alignment
→ learned translator
→ raw 100-band BigVGAN log-mel
→ decode-time BigVGAN mel-floor clamp
→ frozen nvidia/bigvgan_v2_24khz_100band_256x
→ audio

Experimental split:
- 32 training tracks
- 8 validation tracks
- 8 test tracks
- Use a deterministic, genre-balanced selection where possible.
- Prevent duplicate/checksum-identical audio from crossing splits.
- Save the exact track manifest and hashes.
- Do not use genre labels as model inputs.

Crops:
- Use the established 1.995-second crop geometry.
- Sample multiple training crops per track across different offsets.
- Create fixed validation and test crop manifests before training.
- Also create fixed evaluation crops from the 32 training tracks that were not used as training crops.

This must distinguish:
1. memorized training crops;
2. unseen crops from known training tracks;
3. unseen crops from validation tracks;
4. unseen crops from test tracks.

Models:
- Freeze MuQ and BigVGAN-v2.
- Train only the existing TASK-15 translator.
- Do not change architecture, MuQ layer, mel representation, loss, optimizer, or BigVGAN.
- Use XPU/PVC through the repository’s portable device layer.
- Use FP32 initially.

Training:
- Use a predefined step budget and evaluation schedule.
- Select the best checkpoint using validation mel metrics only.
- Save best and final checkpoints.
- Monitor with Weights & Biases.
- Log training-track exposure to ensure broad sampling.
- Support exact checkpoint/resume lineage where practical.

Primary endpoint:
Held-out target-vs-predicted BigVGAN mel error, reported separately for:
- unseen crops from known training tracks;
- validation tracks;
- test tracks.

Compare predicted-mel error with:
- BigVGAN’s own mel round-trip error;
- shuffled-across MuQ;
- shuffled-time MuQ;
- constant-zero MuQ;
- a simple training-set average-mel baseline.

Secondary metrics:
- mel cosine or existing spectral similarity;
- predicted/target mel variance ratio;
- RMS/loudness-contour correlation;
- spectral-centroid/brightness correlation;
- chroma similarity;
- onset-envelope correlation at several smoothing timescales;
- decoded audio metrics that do not require sample-level phase agreement.

Listening:
For fixed representative examples, save:
- original;
- BigVGAN oracle;
- correct prediction;
- shuffled-across;
- shuffled-time;
- constant-zero.

Use validation data for checkpoint selection. Do not use test metrics or listening to choose the checkpoint.

ACES execution:
- Create a versioned Slurm batch file for one PVC GPU.
- Reuse the verified XPU environment and TASK-15 ACES setup.
- Run only bounded smoke tests interactively.
- Submit full training through Slurm, not on a login node.
- Record job ID, SU/resource request, environment, commit, configuration, W&B run ID, and output paths.
- Do not submit the full job unless explicitly instructed.

Decision outcomes:
- If unseen crops from known tracks succeed but unseen tracks fail, classify this as track-specific generalization only.
- If validation and test tracks beat every control and produce recognizably source-related audio, the mapping shows held-out track generalization.
- If correct conditioning only preserves coarse contours, report that without calling it recognizable reconstruction.
- If all regimes fail, determine whether the translator underfits even known tracks before attributing failure to MuQ.
- Do not infer MuQ invertibility from memorized training crops.

Produce:
- task protocol/configuration;
- manifests;
- Slurm batch file;
- smoke-test report;
- W&B setup;
- aggregate and per-example results;
- listening package;
- explicit outcome classification;
- exact final git status.

Do not modify the established TASK-15 scientific pipeline beyond dataset sampling, validation, checkpointing, and required full-scale execution infrastructure. Do not commit unless explicitly requested.
