# NEXT

## Current gate

DAC shift/polarity sensitivity diagnostic (`tasks/TASK-07.md`), a pre-training diagnostic on the
frozen DAC encoder alone -- no MuQ, no translator, no training. **Result: DAC's own target is
extremely shift-fragile while the candidate translator inputs (log-mel, MuQ) are comparatively
shift-stable at the same shift magnitudes.** `quantized_representation`'s mean cosine similarity
to its unshifted self drops to 0.716 at a single-*sample* shift and to 0.157 by 16 samples
(approaching, but never fully reaching, the random-pair floor of 0.030, even at the largest tested
shift of 256 samples -- still sub-frame relative to DAC's 320-sample hop). log-mel and MuQ stay
>=0.98/0.94 across the entire same range. Polarity flip (`x` vs. `-x`) produces *anti-correlated*
latents (cosine -0.44 quantized), not merely dissimilar ones -- further apart than random unrelated
crops. This confirms and generalizes TASK-04's unplanned 2-sample (83us) finding: the fragility is
intrinsic to DAC's target, not an artifact of one trained checkpoint's overfit, and gives a
structural explanation for the collapse pathology (a magnitude-only/phase-blind translator input
cannot in principle determine this target uniquely) that is orthogonal to everything TASK-01
through TASK-06 have already ruled out. Full reasoning and the complete shift-sweep table in
[[DECISIONS.md#task07-dac-shift-sensitivity]]. Per this task's own explicit non-goal, **no
conclusion is drawn here on whether to proceed with a complex-STFT rerun of TASK-05 -- that
decision is the user's, from these numbers.**

**Conclusions reached this session (full reasoning in
[[DECISIONS.md#task07-dac-shift-sensitivity]]):**

1. Ran the shift sweep (k in {1,2,4,8,16,32,64,128,256} samples, both directions) and polarity
   test on 40 GTZAN val-split crops (2/genre x 20 tracks x 2 crops), stratified and deterministic,
   never overlapping any training set used elsewhere in this project. 27 seconds total runtime.
2. **DAC's `quantized_representation` is severely shift-fragile at sub-frame magnitudes** --
   collapses toward (not fully into) the random-pair floor by 16-32 samples, out of a 320-sample
   frame hop.
3. **The pre-quantizer (pre-RVQ) encoder output is somewhat more stable but not qualitatively
   different** -- 0.945 at k=1 vs. quantized's 0.716, but it too collapses by k=8-16. RVQ
   quantization amplifies an already-present continuous-encoder fragility; it does not create it.
4. **Polarity flip produces anti-correlated latents**, not just dissimilar ones -- `x`/`-x` cosine
   similarity is negative and codebook agreement is at or below the random-pair floor.
5. **log-mel and MuQ are both comparatively near-invariant to the same shifts** -- this rules out
   "the DAC target is just tracking an equivalently shifted input" as an alternative explanation,
   and is the load-bearing evidence for the ambiguity-set argument this task set out to test.
6. `DacModel.encoder`/`DacModel.quantizer` (already-public submodules) expose the pre-quantizer
   stage cleanly -- verified to reproduce `DacModel.encode()` exactly, no monkeypatch needed, so
   this task's first stop condition never triggered.

## Exact last successful commands

```bash
uv run python -u scripts/analyze_dac_shift_sensitivity.py
uv run ruff check .; uv run pytest -q   # 145 passed
```

2026-09-17, macOS, M4 Max, device=mps. Single foreground run, 27 seconds total (forward passes
only, no gradients, no background job needed).

## Run/checkpoint path

No checkpoints -- this task trains nothing. Artifacts:

- `artifacts/task07_dac_shift_sensitivity/shift_sensitivity_report.json` -- every number
  (shift sweep for DAC quantized/pre-quantizer/codebook-agreement, log-mel, MuQ; polarity
  sensitivity; random-pair baseline; the exact 40-crop list with track IDs and crop start times).
- `artifacts/task07_dac_shift_sensitivity/shift_sensitivity.png` -- the required figure (DAC
  quantized + pre-quantizer vs. log-mel vs. MuQ, cosine similarity vs. shift magnitude, log-x,
  with the random-pair baseline as a reference line).
- No W&B run (this task doesn't use W&B -- no training loop to log).

## Next command

**Blocked on user direction.** Per this task's explicit non-goal ("do not draw a final conclusion
about whether to proceed with the complex-STFT TASK-05 rerun -- the decision on whether/how to
proceed is the user's"), no next experiment is proposed here. The evidence this task produced
(DAC target is shift-fragile at sub-frame magnitudes; candidate translator inputs are not) is the
input to that decision, not the decision itself. If the user does want to proceed:

- A complex-STFT (phase-aware) representation for the translator input, motivated directly by this
  task's finding that magnitude-only inputs cannot in principle resolve DAC's sub-frame-sensitive
  target -- this was the TASK-07 Context section's own motivating rationale for building this
  diagnostic first.
- Alternatively, investigating whether more precise crop-boundary alignment during data
  preparation (rather than a representation change) narrows the practical impact of this
  fragility, if the sensitivity is dominated by alignment noise already present in this project's
  own crop-extraction pipeline.

Do not start either without the user's go-ahead.

## Blockers

1. **Decision needed from the user** on whether/how to proceed given this task's shift-sensitivity
   findings (TASK-07 itself does not request one).
2. **Listening verdicts** for TASK-01's and TASK-02's WAV packages -- still pending (TASK-04's,
   TASK-05's, and TASK-06's are all now confirmed by ear; TASK-07 has no audio output to review).
3. **BlueHive job not yet run** -- see [[task03-bluehive-wandb]], still pending.
4. The still-pending conditioning-collapse diagnostic from `experiment/mel-hifigan-sanity` --
   untested, orthogonal to this branch's findings.
5. The `shuffled_time` finding from Gate 4 (see [[DECISIONS.md#gate4-shuffled-time-finding]]) --
   still dormant.
6. Listening verdicts for the 80-track scaling experiment, the full-GTZAN scaling experiment, and
   the MuQ layer-sweep were never given -- not blocking, worth noting if any comes up again.

## Expected verification criteria for Gate 5

Not yet reached, and not authorized to start.

## Notes for whoever resumes this

- **Read [[DECISIONS.md#task07-dac-shift-sensitivity]] first.** It gives the clearest structural
  explanation yet for why this project's collapse pathology exists: the DAC target this whole
  investigation regresses against is intrinsically fragile to sub-frame input alignment in a way
  none of the candidate translator inputs are, at the same shift magnitudes.
- `src/muq_inversion/dac_encoder.py`'s new `FrozenDacEncoder.encode_stages` is the clean way to
  get DAC's pre-quantizer output + per-stage codebook indices in one call, for any future
  diagnostic needing them -- no monkeypatch, verified equivalent to `encode()`.
- `src/muq_inversion/dac_shift_sensitivity_common.py`'s `select_stratified_tracks`,
  `cosine_similarity_per_frame`, `codebook_agreement`, and `sample_cross_track_pairs` are generic
  and reusable for any future shift/polarity/alignment-sensitivity diagnostic on any of this
  project's three representations (DAC, log-mel, MuQ) -- no new plumbing needed.
- This task's crop selection (val-split, `crop_start_seconds` in {5.0, 15.0}) deliberately never
  overlaps any training manifest in this project -- keep that property if extending this
  diagnostic (e.g. to more shift magnitudes or a MuQ polarity test) rather than reusing a training
  crop.
- `scripts/analyze_dac_shift_sensitivity.py` is read-only and fast (27s) -- safe to rerun directly
  if the DAC checkpoint, GTZAN manifest, or shift magnitudes ever need revisiting.
