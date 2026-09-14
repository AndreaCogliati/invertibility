# NEXT

## Current gate

MuQ hidden-layer sweep for the MuQ-to-DAC temporal-conv pipeline (`experiment/muq-layer-sweep`,
branched off `experiment/muq-to-dac-more-data`). **Result: MuQ layer choice does not resolve the
saturation ceiling -- the final layer (this investigation's baseline throughout) already wins.**
Do not proceed to layer fusion, model scaling, transformers, additional datasets, or Gate 5 (per
this experiment's own explicit instruction) without further direction.

**Conclusions reached this session (full reasoning in [[DECISIONS.md#muq-layer-sweep]]):**

1. Measured MuQ's `output_hidden_states` structure directly: 13 tensors (index 0 = embedding
   output, indices 1-12 = each of 12 Conformer blocks), `hidden_states[12] ==
   last_hidden_state` confirmed. Chose 5 candidates [1, 4, 6, 9, 12] via a pure, unit-tested
   function -- no shape adapter needed (all layers share shape/channels/frame-rate).
2. Trained the exact `experiment/muq-to-dac-more-data` architecture/config, varying only the
   input MuQ layer, at a documented 45000-step screening budget chosen from that experiment's
   own curve before running anything else.
3. **Final layer wins the primary ranking metric (validation MSE) outright** -- lowest val AND
   test MSE of all five layers. "Earliest" (layer 1) had a small cosine-similarity edge (~10%
   relative) but no MSE or waveform-domain advantage; waveform correlation to the DAC oracle
   stayed at ~0 for every layer.
4. Per the experiment's own decision rule, no layer qualifies for a full follow-up run.
   Architecture/context (receptive field, temporal modeling capacity) remains the leading
   unexplored hypothesis for the saturation ceiling.
5. A separately noteworthy finding: the final-layer control run was itself turbulent through
   most of its 45000-step budget, with its best-by-validation-MSE checkpoint landing on an early
   lucky dip (step 1500) rather than the eventual stable region -- a property of this training
   regime's noise worth remembering when interpreting any "best checkpoint step" from this
   pipeline, not a bug in this experiment specifically.

## Exact last successful commands

```bash
uv run python scripts/inspect_muq_hidden_states.py       # writes artifacts/muq_layer_sweep/hidden_states_inspection.json
uv run python scripts/precompute_layer_sweep_cache.py     # writes data/cache/layer_sweep/muq_layers_cache.pt (45.91s)
uv run python scripts/train_layer_sweep.py                 # 5 layers x 45000 steps, ~35 min total, PASSED (as a measurement)
uv run ruff check .    # all checks passed
uv run pytest -q       # 62 passed (47 baseline + 15 new layer-sweep tests)
```

2026-09-10, macOS, M4 Max, device=mps. Launched via `nohup ... &` + `disown`, tracked by real PID
and a `Monitor` until-loop.

## Run/checkpoint path

- `artifacts/muq_layer_sweep/hidden_states_inspection.json` -- the measured MuQ hidden-state
  structure and exact candidate-layer selection, tracked as a durable record.
- `data/cache/layer_sweep/muq_layers_cache.pt` (2.05GB, git-ignored) -- all 5 candidate MuQ
  layers for all 1998 crops. `data/cache/more_data/muq_dac_cache.pt` (DAC targets) reused
  read-only, unmodified (verified via mtime check at runtime).
- `runs/layer_sweep/<layer_label>/checkpoints/best.pt` (5 x 60MB, git-ignored) -- one
  validation-MSE-selected checkpoint per layer.
- `artifacts/layer_sweep/report.json` -- full metrics for all 5 layers (MSE, cosine similarity,
  controls margins, waveform correlation, STFT distance, training curves, parameter count,
  timing, peak memory).
- No listening WAVs were generated this round -- see [[DECISIONS.md#muq-layer-sweep]]'s
  "Listening screen: not generated" section for why (the validation-selected best candidate
  coincides with the baseline, so there is no second condition to blind-test).

## Next command

**Blocked on user direction.** This experiment's own instructions are explicit: do not proceed
to layer fusion, model scaling, transformers, additional datasets, or Gate 5 based on this result
alone. The negative result here, combined with [[muq-to-dac-more-data]]'s data-scaling
saturation, narrows the remaining hypothesis space for this investigation's MuQ-to-DAC line to:

- **Architecture/receptive-field capacity**: the TemporalConvNet's 5-MuQ-frame receptive field
  and 3-layer depth have been held fixed since Gate 3 -- untested as a variable in this entire
  investigation. A receptive-field or capacity sweep (mirroring this experiment's own minimal-
  confounding methodology) would be the natural next test, but is explicitly out of scope here.
- **A fundamentally different translator design** (still not transformers/attention, if the user
  wants to stay within this investigation's established constraints) -- also out of scope
  without direction.
- **Treating the MuQ-to-DAC line as having reached a documented ceiling** with this specific
  small-ConvNet-translator approach, and revisiting whether to invest further here versus the
  HiFi-GAN-family findings from earlier in the broader investigation.

Do not start any of these without the user's go-ahead.

## Blockers

1. **Decision needed from the user**: pursue architecture/receptive-field capacity as the next
   axis (the one remaining untested variable after data and MuQ-layer-choice), or treat the
   current evidence as a stopping point for this specific translator design.
2. The still-pending conditioning-collapse diagnostic from `experiment/mel-hifigan-sanity` (see
   that branch's DECISIONS.md) remains untested and is orthogonal to this branch's finding.
3. The `shuffled_time` finding from Gate 4 (HiFi-GAN-family, see
   [[DECISIONS.md#gate4-shuffled-time-finding]]) -- still dormant.
4. Listening verdicts for the 80-track scaling experiment and the full-GTZAN scaling experiment
   were never given by the user (superseded by this and the prior request) -- not blocking, but
   worth noting if either comes up again.

## Expected verification criteria for Gate 5

Not yet reached, and explicitly not to be started per this experiment's own instructions.

## Notes for whoever resumes this

- **Read [[DECISIONS.md#muq-layer-sweep]] first** -- it closes off "wrong MuQ layer" as an
  explanation for the saturation ceiling, narrowing what's left to test.
- This branch sits on top of `experiment/muq-to-dac-more-data` (not `main`), continuing the
  parent-branch-extension pattern started by Gate 4 and followed by every experiment since.
- `src/muq_inversion/encoder.py` gained `encode_all_hidden_states()` -- additive only, the
  existing `encode()` method (used by every other branch/experiment) is untouched.
- `src/muq_inversion/layer_sweep_common.py` holds the pure, unit-tested layer-selection,
  cache-key, and checkpoint-selection logic -- reuse it rather than re-deriving similar logic if
  a future architecture sweep needs the same kind of deterministic-selection/checkpoint-selection
  machinery.
- MuQ's hidden-state indexing convention (0 = embedding output, 1-12 = each Conformer block,
  12 = `last_hidden_state`) is now a measured, documented fact
  (`artifacts/muq_layer_sweep/hidden_states_inspection.json`) -- reuse it directly rather than
  re-inspecting if any future experiment needs a different MuQ layer.
- Validation-MSE-only checkpoint selection can land on an early, lucky low point in this training
  regime's noisy loss landscape (see the final-layer control run's step-1500 selection) rather
  than a representative late-training optimum -- worth a sanity glance at the full curve, not
  just the selected checkpoint's number, in any future experiment using this pipeline.
- Adam betas `(0.9, 0.999)` (PyTorch's default), same as every prior MuQ-to-DAC experiment -- no
  adversarial training in this line of work.
