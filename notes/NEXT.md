# NEXT

## Current gate

Still investigating Gate 3 (Tiny-Set Overfit) -- explicitly not proceeding to Gate 4/5. Four
independent axes have now been varied on this branch and its siblings, none of which explain the
perceptual mediocrity that started this investigation:

**Conclusions reached across this investigation (full reasoning in
[[DECISIONS.md#gate3-capacity-ruled-out]], [[DECISIONS.md#encodec-mediocrity-across-all-variants]],
`experiment/mel-hifigan-sanity`'s DECISIONS.md, and [[DECISIONS.md#reference-hifigan]]):**

1. **Decoder capacity is not the bottleneck** (512 vs. 1024 `base_channels`, `main` branch). 1024
   won on every objective metric but the user heard no perceptible difference.
2. **Representation source (MuQ vs. EnCodec) is not the bottleneck** in the way it first appeared
   (`encodec-hifigan-experiment` branch). EnCodec produced a 6-9x waveform-correlation jump with
   zero perceptual difference from MuQ.
3. **Ground-truth mel spectrogram conditioning behaves like MuQ, not EnCodec**
   (`experiment/mel-hifigan-sanity` branch) -- waveform correlation stayed low (0.06/-0.01) despite
   being, by construction, at least as information-rich as any representation could be for
   matching its own mel-loss target. This reframed the axis as "magnitude-only vs.
   reconstruction-trained representations," not "MuQ vs. EnCodec" as arbitrary alternatives -- and
   raised a more serious hypothesis (conditioning collapse, still untested, see Blockers below).
4. **This project's own custom generator/discriminator/training-loop implementation is not the
   bottleneck either** (`experiment/reference-hifigan` branch, just concluded). A faithfully
   vendored, independently-published `jik876/hifi-gan` implementation, with 4-4.4x fewer generator
   parameters, trained on ground-truth mel for the same 2 clips/15000 steps, converged best-in-class
   on mel loss (ratio 0.174) but landed in the *same low waveform-correlation band* (0.073/0.122)
   as this project's own MuQ and ground-truth-mel results -- not EnCodec's outlier band. This
   substantially weakens "our implementation has a bug" as an explanation.

**What's left, now that capacity, representation source (MuQ/EnCodec), and implementation are all
ruled out or substantially weakened:** the 2-clip/15000-step tiny-overfit training paradigm itself
(shared by every variant tested, including the reference implementation), and/or the still-untested
conditioning-collapse hypothesis from `experiment/mel-hifigan-sanity`.

## Exact last successful commands

```bash
# experiment/reference-hifigan branch, 2026-09-08:
uv run python scripts/train_reference_hifigan.py   # 15000 steps, PASSED, mel-loss ratio 0.174
uv run ruff check .    # all checks passed
uv run pytest -q       # 47 passed (MuQ baseline confirmed untouched)
```

Post-hoc mel/waveform correlation computed from the written WAVs using the vendored
`mel_spectrogram` function (script not saved as a permanent artifact, results recorded in
`notes/SCRATCHPAD.md`): pop.00012 mel_corr=0.9496 wav_corr=0.0728; disco.00049 mel_corr=0.9480
wav_corr=0.1221.

Reference/reconstruction/difference WAVs sent to the user for a listening verdict. **Verdict
received (2026-09-09):** "There's a slight improvement in the reconstructed audio, especially on
the disco song. However, the pop song still has noticeable artifacts." First experiment in this
investigation to produce any reported perceptual improvement, though partial and track-dependent
-- see [[DECISIONS.md#reference-hifigan]] "Listening verdict" section. User has indicated the next
experiment will pursue a different angle; not yet specified as of this note.

## Run/checkpoint path

- `runs/reference_hifigan/checkpoints/{latest,step_015000}.pt` (git-ignored, 969 MB each) --
  reference HiFi-GAN generator (13.9M params) + MPD/MSD discriminators, converged cleanly.
  Intermediate numbered checkpoints (step_001000..step_014000) deleted after the run to reclaim
  disk (15GB -> 1.9GB); this run is a self-contained diagnostic, not something needing mid-run
  resume.
- `artifacts/reference_hifigan/` holds the reference/reconstruction/difference WAVs for both
  clips (already sent to the user).
- `runs/gate3_tiny_overfit*/`, `runs/encodec_1024/` (or equivalent), and the
  `experiment/mel-hifigan-sanity` branch's `runs/` entries are untouched, still on disk
  (git-ignored, survive branch switches) -- see each sibling branch's own NEXT.md/DECISIONS.md for
  their exact paths.

## Next command

**Blocked on the user's listening verdict for this experiment**, then on a decision between two
live options (both previously proposed, neither yet greenlit):

1. **Run the still-pending conditioning-collapse diagnostic** from `experiment/mel-hifigan-sanity`
   (see that branch's DECISIONS.md, "Status: this hypothesis is proposed but NOT yet tested"):
   load already-trained checkpoints (MuQ, EnCodec, ground-truth-mel, and now reference-HiFi-GAN --
   all still on disk) and compare generator output under correct/zero/shuffled-in-time/
   cross-track conditioning, without retraining anything. Cheap (forward passes only) and would
   directly confirm or rule out whether the generator meaningfully uses its conditioning at all.
2. **Test the training-paradigm-scale hypothesis directly**: much longer training on the same 2
   clips (e.g. 50,000-100,000 steps) and/or a broader training regime (more than 2 clips, edging
   toward Gate 4/5 territory) to see if the shared "converges in mel domain, stays low in
   waveform-correlation domain" pattern resolves with more adversarial exposure.

Do not start either without the user's go-ahead. Do not perform further MuQ/EnCodec/BigVGAN
experiments or proceed to Gate 4/5 without explicit instruction (per this branch's task scope).

## Blockers

1. **Listening verdict needed from the user** for the reference-HiFi-GAN result specifically --
   audio already sent via SendUserFile.
2. **Decision needed from the user**: conditioning-collapse diagnostic vs. longer/broader training
   experiment vs. stopping the Gate 3 audio-quality investigation here and writing it up as the
   documented finding.
3. The `shuffled_time` finding from Gate 4 (see [[DECISIONS.md#gate4-shuffled-time-finding]]) --
   dormant while Gate 3 is being investigated, but now directly relevant to the conditioning-
   collapse hypothesis above.

## Expected verification criteria for Gate 5

Not yet reached. Per CLAUDE.md: train on the committed GTZAN split, save periodic fixed
validation reconstructions, report objective metrics and qualitative caveats -- blocked behind
Gate 3/4 resolution above.

## Notes for whoever resumes this

- **Read [[DECISIONS.md#reference-hifigan]] first** -- it's the most recent and most decisive
  finding: a faithfully vendored, independently-published HiFi-GAN reproduces this project's own
  fidelity signature, weakening the "implementation bug" hypothesis substantially.
- Four experiment branches now exist, each branched independently off `main`, none merged:
  `main` (capacity investigation), `encodec-hifigan-experiment`, `experiment/mel-hifigan-sanity`,
  `experiment/reference-hifigan`. Each has its own isolated notes/DECISIONS.md entries for its
  specific findings; this file summarizes the cross-branch synthesis as of the most recent branch.
- The `transformers==4.55.0` pin in `pyproject.toml`/`uv.lock` is load-bearing -- see
  [[DECISIONS.md#muq-transformers-compat]]. Do not let a routine `uv add` bump it past `<5.0`
  without re-running Gate 0.
- **Adam betas for both optimizers are `(0.8, 0.99)`, not PyTorch's default `(0.9, 0.999)`** --
  fixed a real, reproducible GAN training collapse, confirmed load-bearing across every branch
  including the reference-HiFi-GAN implementation (which uses the same published betas natively).
- MuQ's frame rate (25 Hz / 960 samples per frame / 40ms) is a measured, confirmed fact (Gate 2).
  No longer suspected to be the primary ceiling -- ground-truth mel (80 Hz) and the reference
  implementation's own mel (86.13 Hz) show the same waveform-correlation ceiling as MuQ, which
  argues against frame-rate/information-density being the dominant factor.
- `librosa==1.0.0`'s `librosa.filters.mel` makes every argument keyword-only, unlike the
  positional-argument API the original `jik876/hifi-gan` repo was written against -- see the
  vendored `src/muq_inversion/reference_hifigan/mel.py` docstring for the exact compatibility fix
  (calling with explicit keywords; filterbank output is otherwise identical).
- Never combine the Bash tool's `run_in_background: true` with a trailing `&`/`disown` in the same
  command -- the tool reports "exited with code 0" for the outer wrapper while the real process
  keeps running detached, which looks like premature completion. Track the real PID directly
  (`kill -0 <pid>`) or use a persistent `Monitor` instead. (Documented previously; repeated once
  more on this branch before being caught.)
