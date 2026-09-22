# NEXT

## Current gate

EnCodec shift-sensitivity replication (`tasks/TASK-10.md`), testing whether DAC's
latent-collapses/audio-stays-stable phenomenon (TASK-07/08) is DAC-specific or general to neural
codecs. **Note: `tasks/TASK-10.md` is truncated** (ends mid-sentence in its Final Questions
section, confirmed by direct byte inspection) -- flagged to the user, not guessed past; every
other section of the task was complete and was followed in full.

**Result: neither Outcome A, B, nor C fits cleanly -- EnCodec's latent is dramatically smoother
than DAC's, but that smoothness does not buy more shift-robust audio.** On the exact same 40-crop
dataset and 0-1024-sample shift schedule as TASK-07/08/09: EnCodec's continuous (quantized)
latent cosine similarity stays at 0.999->0.953 across the full shift range, vs. DAC's
0.716->0.087 -- an order of magnitude or more smoother, closely matching this task's Outcome B
example. But EnCodec's reconstructed-audio waveform correlation (~0.89, flat) sits measurably
*below* DAC's (~0.96, flat) throughout -- the latent smoothness did not translate into better
shift-robustness downstream. A further complication: EnCodec's latent cosine is nonetheless a
1.6-2.1x *better* predictor of its own audio similarity than DAC's latent cosine is of DAC's
(Spearman 0.36/0.60/-0.71 vs. 0.23/0.35/-0.33), despite the much narrower range it varies over.
Mechanistically, EnCodec's *discrete* codebook indices collapse under shift about as fast as
DAC's -- the continuous-latent smoothness is specific to the embedding geometry (likely a
128-dim-vs-1024-dim effect), not evidence the underlying tokenization itself is more robust. Full
reasoning, all four comparison tables, and both legible-question answers in
[[DECISIONS.md#task10-encodec-shift-sensitivity]]. Per this task's own instruction, **no follow-up
experiment proposed or run.**

**Conclusions reached this session (full reasoning in
[[DECISIONS.md#task10-encodec-shift-sensitivity]]):**

1. `tasks/TASK-10.md` is truncated mid-sentence; every section up through "Final Questions" is
   complete and was followed, but the question list itself cuts off after two questions. Flagged
   explicitly rather than fabricating the missing text.
2. Built `src/muq_inversion/encodec_encoder.py` (`FrozenEncodecEncoder`), mirroring
   `dac_encoder.py`'s interface -- EnCodec (24kHz, 75Hz frame rate, 32 codebooks/1024 codebook
   size at max bandwidth) matches DAC's geometry closely except for a much smaller 128-dim latent
   and exact-length-preserving decoder (vs. DAC's 8-sample-shorter output).
3. **EnCodec's latent is dramatically more shift-stable than DAC's** -- confirmed decisively
   (Outcome-B-shaped result on the raw stability axis).
4. **This latent stability does not translate into audio-reconstruction stability** -- EnCodec's
   audio similarity is flat but sits below DAC's throughout; the two properties are evidently
   decoupled in both codecs, not a DAC-specific artifact.
5. **EnCodec's (small) latent movement is proportionally more audio-relevant than DAC's (large)
   movement** -- an Outcome-C-flavored refinement layered on the Outcome-B-flavored headline.
6. The smoothness is a continuous-embedding-geometry effect, not a discrete-tokenization one --
   EnCodec's codebook indices are about as shift-fragile as DAC's.

## Exact last successful commands

```bash
uv run python -u scripts/analyze_encodec_shift_sensitivity.py
uv run ruff check .; uv run pytest -q   # 165 passed
```

2026-09-22, macOS, M4 Max, device=mps. Single foreground run, 11.8 seconds total -- the fastest of
the four DAC/EnCodec diagnostics (no MuQ, smaller 128-dim latent).

## Run/checkpoint path

No checkpoints -- this task trains nothing. Artifacts:

- `artifacts/task10_encodec_shift_sensitivity/encodec_shift_sensitivity_report.json` -- every
  number (per-shift EnCodec latent + audio summaries, the DAC comparison tables, correlation
  analysis, EnCodec model/geometry details, runtime).
- `artifacts/task10_encodec_shift_sensitivity/figure1_latent_similarity_vs_shift.png` -- EnCodec
  vs. DAC latent cosine vs. shift.
- `artifacts/task10_encodec_shift_sensitivity/figure2_audio_similarity_vs_shift.png` -- EnCodec
  vs. DAC waveform correlation vs. shift.
- `artifacts/task10_encodec_shift_sensitivity/figure3_latent_vs_audio_similarity.png` -- EnCodec
  latent cosine vs. all three audio metrics, one point per crop/shift pair.
- `artifacts/task10_encodec_shift_sensitivity/listening/` -- 2 representative crops (same
  selection method as TASK-08) x shifts {0,2,8,32,128}: original, shifted, EnCodec reconstruction
  at shift 0 and shift k. Not yet reviewed by the user.
- No W&B run (no training loop to log).

## Next command

**Blocked on user direction.** Per this task's own instruction ("do not propose or run a new
experiment automatically"), no next experiment is proposed here. Two things worth flagging to the
user directly, though:

- **The task file itself is incomplete** -- if there were more final questions intended, the user
  may want to supply the missing text so they can be answered from the results already computed
  (no rerun needed; the report has everything the visible sections asked for).
- The EnCodec-vs-DAC audio-robustness gap (EnCodec's flat similarity sitting below DAC's, despite
  EnCodec's far smoother latent) is a genuinely new, somewhat counterintuitive finding across
  TASK-07/08/09/10 combined, worth the user's attention regardless of what direction (if any) is
  pursued next -- it complicates any simple "switch to EnCodec" recommendation the latent-only
  result alone might have suggested.

Do not start any further experiment without the user's go-ahead.

## Blockers

1. **The task file's missing final questions** -- worth asking the user for the complete text if
   they had something specific in mind beyond what's already answered here.
2. **Decision needed from the user** on how to weigh TASK-07/08/09/10 together for any future
   direction (none of the four tasks requests one).
3. **Listening verdict for TASK-08's and TASK-10's packages** -- neither yet reviewed by the user.
4. **Listening verdicts** for TASK-01's and TASK-02's WAV packages -- still pending.
5. **BlueHive job not yet run** -- see [[task03-bluehive-wandb]], still pending.
6. The still-pending conditioning-collapse diagnostic from `experiment/mel-hifigan-sanity` --
   untested, orthogonal to this branch's findings.
7. The `shuffled_time` finding from Gate 4 (see [[DECISIONS.md#gate4-shuffled-time-finding]]) --
   still dormant.
8. Listening verdicts for the 80-track scaling experiment, the full-GTZAN scaling experiment, and
   the MuQ layer-sweep were never given -- not blocking, worth noting if any comes up again.

## Expected verification criteria for Gate 5

Not yet reached, and not authorized to start.

## Notes for whoever resumes this

- **Read [[DECISIONS.md#task10-encodec-shift-sensitivity]] first**, alongside TASK-07/08/09's
  entries -- together these four form the most complete picture yet of the target-mismatch
  investigation: not just "DAC's target is fragile and reconstruction-irrelevant" (TASK-07/08),
  not just "pointwise geometry is part of the problem" (TASK-09), but now also "a much smoother
  codec latent doesn't automatically fix the audio-robustness side of the equation" (TASK-10).
- `src/muq_inversion/encodec_encoder.py` is now available for any future EnCodec-based diagnostic
  or experiment in this project -- no dedicated unit test file, matching the existing convention
  for `encoder.py`/`dac_encoder.py` (verified once via a documented manual check instead).
- `scripts/analyze_encodec_shift_sensitivity.py` depends on TASK-08's and TASK-09's report.json
  files existing (for DAC's reused comparison numbers) -- rerun those first if either file is ever
  regenerated with different crops/shifts.
- Don't assume "smoother latent = better regression target" from TASK-10's Result 1 alone without
  also weighing Result 2 (audio-robustness gap) -- the full picture is more nuanced than either
  number in isolation suggests.
