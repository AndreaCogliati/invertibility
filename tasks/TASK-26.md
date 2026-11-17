# TASK-26: synthesis report of the MuQ -> mel -> BigVGAN experiments through TASK-25

(Given in chat on 2026-11-06; recorded verbatim.)

Create a self-contained LaTeX report summarizing the BigVGAN-based MuQ reconstruction experiments through TASK-25.

PURPOSE
Help me step back and understand:
1. The complete pipeline from original audio back to reconstructed audio.
2. Exactly what each intermediate representation contains.
3. What the experiments have established, what remains uncertain, and why.

This is a synthesis and documentation task. Do not launch new training experiments or modify experimental results.

1. INSPECT THE REPOSITORY
Read the relevant code, configs, checkpoint metadata, task specifications, results, analysis, listening notes, and decision records.

Identify where the MuQ → mel → BigVGAN experiment sequence began. Cover the relevant earlier BigVGAN tasks as well as TASK-19 through TASK-25. Do not assume TASK-19 was the beginning.

Keep earlier DAC/EnCodec work to a short background paragraph if needed.

Use actual implementation and saved artifacts as the source of truth. Resolve discrepancies between notes and code explicitly. Never invent missing dimensions, frame rates, checkpoint identities, or results.

2. DELIVERABLES
Create a dedicated report directory containing:
- Main LaTeX source.
- Bibliography or equivalent source-reference section.
- Figure sources and included assets.
- A compiled PDF, if a LaTeX compiler is available.
- A README with exact build instructions and any unresolved gaps.

Use an existing LaTeX installation if available. Do not install large dependencies without approval.

If compilation is unavailable, still deliver complete source and clearly state that the PDF was not compiled.

Aim for approximately 10–15 readable pages, allowing a technical appendix for detailed tables. Prioritize clarity over exhaustive log reproduction.

3. OPEN WITH A ONE-PAGE OVERVIEW
Explain in plain language:
- What reconstruction problem we are studying.
- What is frozen and what is trained.
- Why MuQ representation depth matters.
- Why accurate mel reconstruction and accurate audio reconstruction are different questions.
- The main findings through TASK-25.
- The principal unresolved questions.

Clearly separate observations, interpretations, and hypotheses.

4. DOCUMENT THE ENTIRE AUDIO-TO-AUDIO PIPELINE
Trace every implemented transformation, including:

Original audio
→ channel handling/resampling/cropping
→ MuQ acoustic frontend
→ frontend projection/subsampling
→ selected MuQ encoder hidden state
→ feature normalization/cache
→ trainable translator
→ temporal resampling/output projection
→ predicted target mel
→ inverse normalization or other vocoder preparation
→ frozen BigVGAN
→ reconstructed waveform.

Show the separate ground-truth target branch:
Original audio → target-mel extraction → ground-truth mel → BigVGAN.

Show where the training loss compares predicted and target mel.

Verify the actual ordering of operations from code. Do not force the implementation into the illustrative ordering above.

For EVERY stage, provide a representation ledger with:
- Stage/module name.
- Mathematical symbol.
- Tensor layout and shape, with batch/time/feature axes labeled.
- Exact shape for one actual approximately 2-second crop.
- Feature dimension or number of channels.
- Sample rate for waveforms.
- Frame rate and hop duration for framed representations.
- Window length, FFT size, hop size, and mel-bin count where relevant.
- Continuous-valued representation versus discrete indices.
- Numerical dtype, separately from continuous/discrete status.
- Normalization, log scaling, clipping, and units where known.
- Frozen, trainable, or deterministic.
- Temporal context, padding, and boundary handling.

Distinguish nominal frame rate from actual finite-crop frame count. Explain rounding, padding, and centering effects.

Explicitly distinguish:
- Audio sample rate from feature frame rate.
- Vector dimension from number of frames.
- Continuous hidden states from discrete quantizer tokens.
- Floating-point storage from mathematical continuity.
- MuQ's pretraining quantization targets from the representations actually passed to our decoder.

Verify whether any discrete bottleneck is traversed in OUR inference pipeline. Do not imply that MuQ hidden states are discrete merely because MuQ pretraining uses quantized targets.

5. EXPLAIN THE MODELS AND ALIGNMENT
MuQ:
- Exact model/checkpoint.
- Frontend and temporal downsampling.
- Exact meaning of “first block,” including hidden-state indexing.
- Original last-layer representation versus first-block representation.
- Whether extraction uses isolated crops or longer audio context.

Translator:
- Architecture, parameter count, width, depth, kernels, activations.
- Receptive field in frames AND seconds, identifying the frame grid.
- Temporal interpolation/resampling and input/output frame-center conventions.
- Output parameterization and any constraints.
- What changed in TASK-25's wider and learned-resampler arms.

BigVGAN:
- Exact checkpoint/configuration.
- Expected mel convention and waveform sample rate.
- Frozen status and whether gradients traverse it during training.
- Whether training loss is applied before or after vocoding.

Explain alignment assumptions and what has actually been checked. Matching tensor lengths is not proof of timestamp alignment.

6. USE TECHNICAL FIGURES
Create clear, source-faithful figures, preferably vector diagrams using TikZ or another reproducible method.

Required:
A. Full pipeline diagram, including target branch, loss location, frozen/trainable components, rates, dimensions, and continuous/discrete labels.
B. Temporal-grid diagram showing waveform samples, MuQ frames, and target-mel frames over the same interval.
C. Compact experiment timeline or decision tree showing the question each task addressed.

Where existing artifacts support them, include:
- Ground-truth, last-layer prediction, and first-block prediction spectrograms on the SAME crop.
- TASK-23 one-crop reconstruction versus the full-data reconstruction on its anchor.
- TASK-24 fitting-ladder curves.
- TASK-25 training-fit versus monitoring-error comparison.

Use existing measured data and saved predictions. Do not fabricate illustrative experimental results.

Use shared color scales for comparable spectrograms. Label any schematic as schematic. Avoid decorative images.

7. SUMMARIZE THE EXPERIMENT SEQUENCE
For each relevant task, state:
- Question/hypothesis.
- Data and split.
- Input representation.
- Architecture and objective.
- Optimization budget, batch size, LR schedule, and selection rule.
- Key numerical results.
- Listening observations and whether blinded.
- Supported conclusion.
- Important caveats.

Include a compact cross-task table, but do not present incompatible metrics as directly comparable.

Explain:
- Direct pre-vocoder mel L1 versus round-trip metrics.
- Training crops versus unseen crops of training tracks versus unseen tracks.
- Validation-selected best versus final checkpoints.
- Reused evaluation sets versus untouched confirmation data.
- Matched steps versus matched presentations versus matched compute.
- Crop/track confidence intervals versus variability across training seeds.

8. PRESERVE THE CORRECTED INTERPRETATIONS
TASK-21:
Longer training recovered from the early transient and helped modestly, then increasingly favored training fit.

TASK-22:
First-block features substantially improved reconstruction relative to the last layer. This demonstrates greater recoverability with the tested decoder, not proof that deeper layers discarded information.

TASK-23:
One-crop memorization succeeded and informal listening detected no decoder-specific difference relative to ground-truth-mel vocoding.
This does NOT rule out capacity or optimization limitations across many examples.
B/C were same-initialization repeats with no actual regularization difference.
The shared click does not require decoder error; its exact cause remains unresolved.

TASK-24:
Fitting worsened as the fixed pool grew, while monitoring improved.
The anchor became harder to fit as other examples were added.
Architecture and optimization were not separated by that experiment.

TASK-25:
LR and batch size interacted.
Width improved fitting under the tested recipe.
The tested learned resampler produced only a small fitting improvement.
Better fitting did not improve unseen-track monitoring.
Listening found little difference between arms on the sampled training clips.

Verify all statements and exact numbers against saved artifacts. Correct this outline if the primary evidence differs.

Do not claim:
- A MuQ information ceiling.
- Proven invertibility or non-invertibility.
- Optimization convergence from a plateau.
- That L1 necessarily causes the observed blur.
- That more data, regularization, or adversarial loss is already established as the solution.
- That a ground-truth-mel vocoder reference is a mathematical upper bound on all possible reconstruction methods.

9. END WITH A SYNTHESIS
Organize the remaining possibilities into:
- Representation accessibility/information.
- Decoder architecture/capacity.
- Optimization.
- Data diversity/generalization.
- Objective/perceptual mismatch.
- Preprocessing/alignment.
- Vocoder limitations.

For each, state:
- Evidence supporting it.
- Evidence against it, if any.
- What remains untested.

Conclude with a short list of discriminating next experiments, not a new training plan.

10. SOURCE TRACEABILITY AND QUALITY CHECKS
Reference local task reports, configs, code paths, and artifact paths using readable identifiers.
Include checkpoint/commit identifiers where relevant.
Use original papers or model documentation only where available and verified; do not invent bibliographic entries.

Before delivery:
- Check every numerical claim against its source.
- Verify rates and shapes with a lightweight inspection script if necessary.
- Check that diagrams agree with the implementation.
- Compile and inspect for clipped figures, overflowing tables, broken references, and unreadable labels.
- Record unresolved facts explicitly rather than filling them with typical values.

Report exactly what was generated, what compiled successfully, and what remains uncertain.
