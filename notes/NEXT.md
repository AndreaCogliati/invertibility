# NEXT

## Current gate

DAC latent-vs-audio shift diagnostic (`tasks/TASK-08.md`), extending TASK-07's latent-only
finding through the DAC decoder. **Result: Outcome A, strongly supported -- DAC latent
instability does not translate into reconstructed-audio instability.** On the exact same 40-crop
set TASK-07 used, latent cosine similarity collapses from 1.0 to 0.09 across shifts of 1-1024
samples, while reconstructed-audio waveform correlation stays flat at 0.955-0.964 and log-mel
cosine similarity at 0.988-0.990 over that same range -- no discernible trend. Across 480
crop/shift points, Spearman correlation between latent similarity and audio similarity is only
0.23-0.35 for every audio metric tested: statistically real but practically a weak predictor. A
raw-waveform reference control (shift the audio before any DAC involvement, then align) reads
exactly 1.0/1.0/0.0 at every shift by construction, proving the small DAC-vs-perfect-control gap
is entirely DAC's own encode-decode behavior, not genuine ambiguity in the source signal. Full
reasoning, the complete shift-vs-both-domains table, and all five required question answers in
[[DECISIONS.md#task08-dac-latent-vs-audio-shift]]. Per this task's own instruction, **stopped
after answering its five required questions -- no next experiment proposed by this task.**

**Conclusions reached this session (full reasoning in
[[DECISIONS.md#task08-dac-latent-vs-audio-shift]]):**

1. Reused TASK-07's exact 40-crop set (verified identical at runtime against TASK-07's own report,
   not just assumed), extended the shift schedule to include 0, 512, and 1024 samples.
2. **Reconstructed-audio similarity is essentially flat across the entire shift range where latent
   similarity collapses by an order of magnitude** -- the two do not move together.
3. **The raw-reference control (exactly 1.0/1.0/0.0 at every shift) is the load-bearing evidence**:
   it proves the underlying audio carries zero inherent shift ambiguity, isolating the small,
   shift-independent DAC reconstruction gap as purely a property of DAC's own encode-decode
   process, not of any genuine target ambiguity near a shifted crop boundary.
4. **Latent cosine similarity is a weak predictor of reconstruction-audio similarity** (Spearman
   rho 0.23-0.35 across three independent audio metrics, n=480) -- Figure 3 shows no visible slope.
5. **This strengthens and sharpens the target-mismatch explanation** for TASK-01 through TASK-06:
   the extra shift-fragility TASK-07 found in DAC's latent, beyond what candidate translator
   inputs show, is now shown to be largely reconstruction-irrelevant -- a shift-invariant input is
   being penalized by MSE for latent differences that mostly don't correspond to differences a
   listener (or any of these three audio metrics) would perceive.
6. SI-SDR was not computed -- no existing implementation found anywhere in this project, and the
   task's own instruction was to reuse one rather than add a dependency for this metric alone.

## Exact last successful commands

```bash
uv run python -u scripts/analyze_dac_latent_vs_audio_shift.py
uv run ruff check .; uv run pytest -q   # 159 passed
```

2026-09-21, macOS, M4 Max, device=mps. Single foreground run, 93 seconds total (forward passes
only, no gradients).

## Run/checkpoint path

No checkpoints -- this task trains nothing. Artifacts:

- `artifacts/task08_dac_latent_vs_audio_shift/latent_vs_audio_shift_report.json` -- every number
  (per-shift summary for latent cosine + DAC-reconstruction audio metrics + raw-reference audio
  metrics, the full correlation analysis, the 480-point crop/shift dataset underlying Figure 3).
- `artifacts/task08_dac_latent_vs_audio_shift/figure1_latent_similarity_vs_shift.png` -- latent
  curve, extended to 512/1024 samples beyond TASK-07's original range.
- `artifacts/task08_dac_latent_vs_audio_shift/figure2_audio_similarity_vs_shift.png` -- DAC
  reconstruction vs. raw-reference audio similarity, three metrics, vs. shift.
- `artifacts/task08_dac_latent_vs_audio_shift/figure3_latent_vs_audio_similarity.png` -- the
  required latent-vs-audio scatter, one point per crop/shift pair.
- `artifacts/task08_dac_latent_vs_audio_shift/listening/` -- listening package for 2
  representative crops x shifts {0,2,8,32,128}: original, shifted, DAC reconstruction (shift 0
  and shift k), and the aligned difference signal. Not yet reviewed by the user.
- No W&B run (no training loop to log).

## Next command

**Blocked on user direction.** Per this task's own instruction ("stop after answering these
questions... do not propose a follow-up experiment automatically"), no next experiment is
proposed here. If the user wants to act on this and TASK-07's combined findings:

- A human listening pass on `artifacts/task08_dac_latent_vs_audio_shift/listening/` would confirm
  (or challenge) the objective-metric conclusion above -- worth doing given this project's several
  earlier metric/perception divergences (see [[task06-n32-compute-ablation]]), even though this
  task's evidence is unusually consistent across three independent audio metrics already.
- TASK-07's own still-open question -- whether to pursue a complex-STFT (phase-aware) translator
  input -- is now informed by a further nuance: the shift-fragility that motivated that idea is
  shown here to be substantially reconstruction-irrelevant, which may weaken (not strengthen) the
  case for chasing sub-frame alignment precision specifically, even though the broader
  target-mismatch explanation (representation choice vs. this MSE-latent-regression setup) is
  reinforced.

Do not start either without the user's go-ahead.

## Blockers

1. **Decision needed from the user** on how to weigh TASK-07 and TASK-08 together for any future
   direction (neither task requests one).
2. **Listening verdict for TASK-08's package** -- not yet reviewed by the user (unlike TASK-04
   through TASK-06, which are all confirmed by ear).
3. **Listening verdicts** for TASK-01's and TASK-02's WAV packages -- still pending.
4. **BlueHive job not yet run** -- see [[task03-bluehive-wandb]], still pending.
5. The still-pending conditioning-collapse diagnostic from `experiment/mel-hifigan-sanity` --
   untested, orthogonal to this branch's findings.
6. The `shuffled_time` finding from Gate 4 (see [[DECISIONS.md#gate4-shuffled-time-finding]]) --
   still dormant.
7. Listening verdicts for the 80-track scaling experiment, the full-GTZAN scaling experiment, and
   the MuQ layer-sweep were never given -- not blocking, worth noting if any comes up again.

## Expected verification criteria for Gate 5

Not yet reached, and not authorized to start.

## Notes for whoever resumes this

- **Read [[DECISIONS.md#task08-dac-latent-vs-audio-shift]] first**, alongside
  [[DECISIONS.md#task07-dac-shift-sensitivity]] -- together they are the most direct evidence yet
  that this investigation's collapse pathology has a structural, representation-level cause (a
  fragile, reconstruction-irrelevant regression target), not purely an optimization/capacity one.
- The raw-reference control reading exactly 1.0/1.0/0.0 at every shift is *correct*, not a bug --
  see the dedicated note in [[DECISIONS.md#task08-dac-latent-vs-audio-shift]] before "fixing" it
  in any future reuse of `align_shifted_pair`.
- `src/muq_inversion/dac_shift_sensitivity_common.py` now holds the full toolkit for any future
  shift/alignment diagnostic on any of this project's representations: track/crop selection,
  padded-buffer shifting, per-frame cosine/L2 distance, codebook agreement, audio-domain
  alignment (`align_shifted_pair`), and Pearson/Spearman correlation (`correlate_metrics`) -- no
  new plumbing needed for a similar future diagnostic.
- `scripts/analyze_dac_latent_vs_audio_shift.py` is read-only and fast (93s) -- safe to rerun
  directly if the DAC checkpoint, GTZAN manifest, or shift schedule ever need revisiting.
