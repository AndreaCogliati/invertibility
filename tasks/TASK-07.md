# TASK-07: DAC latent shift/polarity sensitivity — pre-training diagnostic

## Context

TASK-01 through TASK-06 established that the MuQ/spectrogram-to-DAC translator's held-out
collapse (norm collapse, over-smoothing, near-chance codebook agreement) is not explained by
representation choice (MuQ layer, mel vs. MuQ), data scale (16 to 11873 crops), or MuQ layer
depth. TASK-04 surfaced, as an unplanned finding, that a trained checkpoint's cosine similarity
collapsed from 0.99998 to 0.577 under a 2-sample (83us) input-crop shift — suggesting the DAC
target may be extremely sensitive to sub-frame alignment, independent of any translator.

This task tests that sensitivity directly, on the DAC encoder alone, with no training and no
translator involved. It is a diagnostic, not an experiment on the translator pipeline: it answers
whether a magnitude-only or phase-blind input (log-mel, MuQ) could in principle determine the DAC
target uniquely, before spending compute on a complex-STFT rerun of TASK-05.

**Read `notes/MEMORY.md` and `notes/DECISIONS.md#task04-spectrogram-gate3` before starting** —
the 2-sample finding and its caveats (MuQ's own exact-crop sensitivity) are directly relevant and
should not be re-derived from scratch.

## Question

For real GTZAN audio, how sensitive is the DAC `quantized_representation` (and, separately, the
pre-quantizer encoder output) to (a) small sub-frame time shifts of the input waveform and (b)
global sign flip of the input waveform (`x` vs. `-x`)? If DAC latents are highly sensitive to
either, a translator conditioned on a shift-invariant or polarity-blind input (log-mel magnitude,
MuQ) cannot in principle recover the correct latent uniquely — the target is not a function of
the input the translator receives, and MSE regression will average over the resulting ambiguity
set. If DAC latents are near-invariant to both, this concern does not apply and the held-out
collapse must be explained some other way.

## Method

Use the existing `FrozenDacEncoder` (`src/muq_inversion/dac_encoder.py`) directly — no MuQ, no
translator, no training loop. New script: `scripts/analyze_dac_shift_sensitivity.py`, read-only
over already-downloaded GTZAN audio (reuse TASK-02's dense cache's source list or Gate 1's
manifest for track selection — do not re-download anything).

1. **Sample set.** 20 tracks, 2 crops each (2.0s / 48000 samples, same crop length used
   throughout this project), stratified across all 10 GTZAN genres (2 tracks/genre) — enough for
   a stable mean/std without a large compute cost. Use the val-split tracks from
   `data/manifests/gtzan_manifest_smoke.csv` so this diagnostic's sample never overlaps any
   training set used elsewhere in this project.

2. **Shift sensitivity.** For each crop, compute the DAC `quantized_representation` at the
   canonical (unshifted) alignment. Then recompute it at every shift `k` in
   `{1, 2, 4, 8, 16, 32, 64, 128, 256}` samples (both directions; report the average of + and -
   unless they differ meaningfully, in which case report both). For each shift, report:
   - cosine similarity between shifted and unshifted `quantized_representation`, averaged over
     all overlapping frames (crop the non-overlapping edge frames created by the shift before
     computing, and say how many frames were dropped);
   - the same, for the **pre-quantizer encoder output** (`self.encoder(input_values)`, before RVQ
     — see the DAC target audit in `notes/DECISIONS.md` for how to obtain this without modifying
     `transformers`' `modeling_dac.py`);
   - codebook index agreement (stage 0 only, and mean over all 32 stages) between shifted and
     unshifted, at each shift.
   Produce one figure: cosine similarity (both pre- and post-quantizer) vs. shift, log-x axis,
   one line per quantity, with a horizontal reference line at the shift-0 self-similarity (should
   be 1.0) and at random-pair baseline similarity (computed once, using unrelated crops).

3. **Polarity sensitivity.** For the same 40 crops, compute `quantized_representation` and the
   pre-quantizer encoder output for `x` and for `-x`. Report cosine similarity between the two,
   per crop and averaged, for both representations. Also report codebook index agreement (stage 0
   and mean-over-32) between the `x` and `-x` codes.

4. **Random-pair baseline (required for interpreting both above).** Cosine similarity and
   codebook agreement between `quantized_representation` of unrelated crops (different tracks),
   computed once over ~40 random pairs from the same sample set. This is the "chance" floor every
   other number in this task should be read against — do not report a similarity number without
   this baseline alongside it.

5. **Log-mel and MuQ crop-shift sensitivity, for comparison.** Repeat step 2's shift sweep (same
   shift values, same crops) for (a) the project's 128-band log-mel spectrogram and (b) MuQ
   `last_hidden_state` — the two candidate translator inputs. This directly answers whether the
   *input* representations are also shift-fragile at comparable shifts (in which case a shifted
   DAC target might just be tracking a shifted input, which would undercut the ambiguity
   argument) or whether they are comparatively shift-stable while DAC is not (which would support
   it). Use the project's existing feature-extraction code unchanged; do not add new
   preprocessing.

## Required outputs

- `artifacts/task07_dac_shift_sensitivity/shift_sensitivity_report.json` — every number above,
  machine-readable, with the exact crop list (track IDs + crop indices) used.
- `artifacts/task07_dac_shift_sensitivity/shift_sensitivity.png` — the required figure from step
  2, extended to overlay the log-mel and MuQ curves from step 5 on the same shift axis (twin
  y-axis if the cosine-similarity ranges differ too much to share one axis cleanly).
- A short written summary (append to `notes/DECISIONS.md`, following this project's existing
  entry format) stating, in one paragraph each:
  - whether DAC's `quantized_representation` is shift-fragile, and at what shift magnitude
    similarity drops to within the random-pair baseline's range;
  - whether the pre-quantizer encoder output shows the same fragility or is more stable (this
    matters for whether RVQ itself, versus the underlying continuous encoding, is the source of
    any fragility found);
  - the polarity-sensitivity result and whether `x`/`-x` latents are closer to each other than to
    random pairs;
  - how the input representations' (log-mel, MuQ) own shift-sensitivity compares to DAC's at the
    same shift values — this is the load-bearing comparison for the open question.

## Explicit non-goals

- No training. No translator. No modification to any existing checkpoint, manifest, or cache.
- Do not draw a final conclusion about whether to proceed with the complex-STFT TASK-05 rerun —
  report the numbers and the three-way comparison (DAC vs. log-mel vs. MuQ shift-sensitivity);
  the decision on whether/how to proceed is the user's, to be made from this task's output.

## Stop conditions

- If `FrozenDacEncoder` does not expose (or cannot easily be made to expose) the pre-quantizer
  encoder output without a fragile monkeypatch of `transformers` internals, stop and report this
  rather than approximating it — report shift/polarity sensitivity for
  `quantized_representation` only, with this limitation stated explicitly, not silently.
- If any of the 40 sampled crops fails a finite-value check, drop it, log which one and why, and
  continue with the remainder — do not fail the whole run over one bad crop.

## Estimated cost

Forward passes only, no gradients, no gradient accumulation — this should run in at most a few
minutes on the M4 Max. Confirm before running that this stays within that budget; if it does not,
stop and report why before scaling anything up.
