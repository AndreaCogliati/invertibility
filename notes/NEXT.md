# NEXT

## Current gate

EnCodec continuous-latent controlled scaling probe (`tasks/TASK-11.md`) -- the first *training*
experiment since TASK-06 in the TASK-07-11 target-mismatch investigation line. Tested whether
TASK-05's sharp DAC N=8->N=32 collapse moves when only the target codec changes to EnCodec's
continuous latent, everything else held fixed (same manifests, input, architecture except the
mechanically-necessary 128-vs-1024 output width, loss, optimizer, 20000-step budget, seed).

**Result: the collapse is dramatically reduced -- decisive support for the target-mismatch
hypothesis.** At N=32: DAC's norm/variance ratios collapse to 0.499/0.177; EnCodec's stay at
0.985/0.650 (no norm collapse). Waveform correlation (pred vs. target-decode): DAC crashes 0.997
(N=8) -> 0.469 (N=32); EnCodec declines 0.982 -> 0.724 -- real but about half DAC's drop.
**Oracle-relative: EnCodec's translator recovers 79.2% of its own reconstruction ceiling at N=32,
vs. DAC's 48.4%** (using a newly-fixed lag-aligned DAC oracle metric -- see below). Per-example
N=32: EnCodec's latent fit is uniform across all 32 examples; audio-level per-example variation
remains despite that uniform latent fit. Per this task's own rules: **Outcome B** (substantial
but partial improvement -- target geometry, optimization, and capacity/interference likely
coexist). Full reasoning, the complete comparison table, and all eight required question answers
in [[DECISIONS.md#task11-encodec-controlled-scaling]]. Per this task's own instruction, **stopped
after answering its eight required questions -- no N=128/512 extension, no follow-up experiment
proposed or run.**

**Two real bugs found and fixed during this task:**
1. **DAC's decoder has a previously-undocumented fixed 8-sample group delay** relative to its
   input -- a naive lag-0 original-vs-oracle correlation gave ~0.37-0.47 (looked broken) until a
   cross-correlation lag search (`find_best_lag_and_align`, `gate3_encodec_common.py`) recovered
   the true ~0.96-0.97 ceiling. EnCodec has no equivalent delay (best lag 0). This had never
   surfaced before because every prior same-codec audio comparison in this project (TASK-08's
   `align_shifted_pair`) compares two DECODED outputs to each other, where a shared delay cancels
   -- never original audio to a single decoder output directly.
2. **EnCodec targets computed fresh are PyTorch "inference tensors"** and crash autograd
   ("Inference tensors cannot be saved for backward") unless explicitly `.clone()`-d. TASK-05's
   DAC targets never hit this since they load from TASK-02's cache as plain tensors.

## Exact last successful commands

```bash
uv run python -u scripts/train_gate3_encodec_scale_wandb.py --levels 2,8,32 --wandb-mode online
uv run python scripts/plot_gate3_encodec_scale_figures.py   # four required figures
uv run ruff check .; uv run pytest -q   # 177 passed
```

2026-09-22, macOS, M4 Max, device=mps. Launched via `nohup ... &` + `disown`, tracked by a
persistent `Monitor`. Training time: N=2 96.8s, N=8 126.3s, N=32 127.0s (~5.8 min total -- far
faster than TASK-05's DAC runs at the same N, consistent with EnCodec's much smaller 128-dim
target and correspondingly smaller translator).

## Run/checkpoint path

- `data/manifests/gate3_scale_{2,8,32}.csv` (tracked, reused unchanged from TASK-05, verified
  nested and identical at runtime).
- `runs/task11_gate3_encodec_scale/n{2,8,32}/checkpoints/{best_by_mse,best_by_wav_corr,final}.pt`
  (git-ignored).
- `artifacts/task11_gate3_encodec_scale/n{2,8,32}/report.json` -- full per-level metrics (latent +
  audio, EnCodec oracle, DAC oracle, per-example, DAC comparison values pulled from TASK-05);
  `.../n{N}/*.wav` -- listening WAVs (best/median/worst at the best checkpoint, plus final-
  checkpoint audio for N=8/32 where the final checkpoint materially degraded from the best).
- `artifacts/task11_gate3_encodec_scale/figure{1,2,3,4}_*.png` -- the four required figures.
- `artifacts/task11_gate3_encodec_scale/scaling_summary.json` -- cross-level summary.
- Three live W&B runs, group `gate3-encodec-scale`: `wandb.ai/
  andrea-cogliati-university-of-rochester/muq-invertibility` (`gate3_encodec_scale_{2,8,32}`).

## Next command

**Blocked on user direction.** Per this task's own instruction ("do not propose or run a new
experiment automatically"), no next experiment is proposed here. Candidate directions that would
need the user's go-ahead if raised later:

- Extending the EnCodec scaling curve to N=128/512 to see whether EnCodec has its own, later
  collapse point -- explicitly out of this task's scope, but a natural next question given how
  cleanly N=2/8/32 differentiated DAC from EnCodec.
- Investigating the audio-level per-example variation at N=32 that survives despite uniform
  latent fit (Figure 4) -- is it decoder sensitivity, content complexity, or something else?
- Now that DAC's own 8-sample decoder delay is known and fixed, revisiting any earlier
  original-vs-DAC-oracle comparison in this project that might have used the naive (unaligned)
  correlation -- worth an audit, though the fix here was scoped narrowly to this task's own new
  oracle metric, not applied retroactively to any prior task's numbers.

Do not start any of these without the user's go-ahead.

## Blockers

1. **Decision needed from the user** on how to weigh TASK-07-11 together for any future direction
   (none of the five tasks requests one).
2. **Listening verdict for TASK-11's package** -- not yet reviewed by the user (nor TASK-08's or
   TASK-10's).
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

## Repository state at the end of this task (per TASK-11's explicit reporting requirement)

At the start of this task, `git status` showed a clean tree except `tasks/TASK-11.md` itself
(untracked) -- TASK-09 and TASK-10, which TASK-11's own text assumed were still uncommitted, had
in fact already been committed as `b54418f` in the prior session turn (explicit user instruction:
"Summarize the results and commit"). Nothing from TASK-09/10 was at risk during this task.

At the end of this task:

```text
Changes not staged for commit:
    modified:   notes/DECISIONS.md
    modified:   notes/MEMORY.md
    modified:   notes/NEXT.md        (this file, being written)
    modified:   notes/SCRATCHPAD.md  (pending)

Untracked files:
    scripts/plot_gate3_encodec_scale_figures.py
    scripts/train_gate3_encodec_scale_wandb.py
    src/muq_inversion/gate3_encodec_common.py
    tasks/TASK-11.md
    tests/test_gate3_encodec_common.py
```

**Not committed**, per this task's explicit "do not commit changes without explicit user
instruction" rule. No existing tracked file was modified except the four `notes/*.md` files
(additive appends/rewrites only, consistent with every prior task in this project).

## Notes for whoever resumes this

- **Read [[DECISIONS.md#task11-encodec-controlled-scaling]] first**, alongside TASK-07-10's
  entries -- this is the decisive, causal test of the whole line: a controlled target swap (with
  *fewer*, not more, translator parameters) removes most of TASK-05's sharp collapse, directly
  implicating DAC's target geometry as a primary (not sole) cause.
- `src/muq_inversion/gate3_encodec_common.py`'s `find_best_lag_and_align` is now the correct way
  to compare any codec's original-vs-oracle audio in this project -- do not reuse a naive lag-0
  comparison for a new codec without first checking whether it has its own fixed decoder delay
  (verified: DAC has one, 8 samples; EnCodec does not).
- The `.clone()` requirement for freshly-computed (not cached) codec targets used as autograd
  targets is a real PyTorch gotcha (`torch.inference_mode()`-decorated encode methods) -- keep it
  if reusing `load_examples_from_manifest_encodec`'s pattern for any future non-DAC target.
- `scripts/train_gate3_encodec_scale_wandb.py` restricts `--levels` to {2,8,32} by explicit
  assertion -- this is a deliberate scope guard from this task's own instruction, not an
  oversight; remove it deliberately (and update the docstring) if a future task authorizes
  N=128/512.
