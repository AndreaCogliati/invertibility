# NEXT

## Latest: TASK-23 one-crop overfit diagnostic, prepared for ACES (2026-11-05)

Gate: diagnostic (decoder fitting ability on one pair). Spec, runbook, resources: `aces/TASK23_ACES.md`.
Protocol: [[DECISIONS.md#task23-one-crop-overfit-protocol]]. Last successful command:
`DEVICE=mps LOCAL=1 bash aces/task23_smoke.sh` (PASS). Not submitted, not committed.

**Next (user):**
1. Commit and push; `git pull` on ACES.
2. On a PVC node: `bash aces/task23_smoke.sh`. Pass = "RESUME CHECK PASS", all arms on XPU, cache diff <= 0.1.
3. Only when instructed: `mkdir -p aces/logs && sbatch --exclude=ac079 aces/run_task23_v1.slurm`.
4. Copy back `artifacts/task23/task23_v1/`, `runs/task23/task23_v1/` (about 350 MB) and `aces/logs/muq-task23-v1-*`.
5. Read `artifacts/task23/task23_v1/TASK-23_RESULTS.md`; listen to the WAVs (ratings pending).

## Previous: TASK-22 result (2026-10-28)

ACES 2261697 (L1) and 2261638 (T21eval) finished cleanly; all artifacts were downloaded and checked.
Block-1 output beats `last_hidden_state` by ~43-45% mel L1 on every held-out set, at every matched step,
on every track. Analysis: `artifacts/task22/analysis_v1/`. See [[DECISIONS.md#task22-first-block-result]].

**Next (user):**
1. Rate the blinded package `artifacts/task22/listening_blind_v1/` (do not open `_KEY`). Then run
   `uv run python scripts/task22_listening_package.py --score artifacts/task22/listening_blind_v1`.
2. Commit the notes and the Slurm GPU check (uncommitted).

## Previous: TASK-22 first-block representation, prepared for ACES (2026-10-27)

The translator is trained from scratch on MuQ conformer block 1 output (`hidden_states[1]`,
hook-verified) for 200k steps, then compared with TASK-21 (`last_hidden_state`) at matched steps and
window best. Spec, config diff, resources (~14 billing-hours), runbook and report template:
`aces/TASK22_ACES.md`. Protocol: [[DECISIONS.md#task22-first-block-protocol]]. Not submitted.

**Next (user):**
1. Commit and push.
2. On a PVC node: `bash aces/task22_smoke.sh`.
3. Only when instructed: `mkdir -p aces/logs && sbatch aces/run_task22_L1_v1.slurm && sbatch aces/run_task22_T21eval_v1.slurm`.
4. Copy back `artifacts/task22/`, `runs/task22/task22_L1_v1/` and `aces/logs/`.
5. Run `scripts/task22_analyze.py`, then `scripts/task22_listening_package.py --l1 artifacts/task22/task22_L1_v1`.

## Previous: TASK-21 result (2026-10-26)

ACES 2248110: 47 h, 22.2 M steps, ~378.5 billing-hours, clean.
- TASK-20's stop was a recoverable transient: new best at step 60,000 (val 0.455 vs 0.475).
- On non-selection held-out sets the gain is only ~2-3% (test +2.2%, val2 +1.8%, test2 +2.9%;
  two CIs at or including 0).
- After ~100k steps held-out error slowly rises while memorized keeps falling; the final
  checkpoint is no better than TASK-20 best.
- See [[DECISIONS.md#task21-n512-extended-result]] and `artifacts/task21/analysis_v1/`.

**Next:** user's call. Not committed. Optional: copy back `runs/task21/task21_N512ext_v1/{run_manifest.json,metrics.jsonl}`.

## Previous: TASK-21 extended N512 continuation, prepared for ACES (2026-10-23)

One 48 h ACES job: 47.0 h of training continuing TASK-20's last checkpoint (no early stopping),
then in-job evaluation. Spec, budget (~379 billing-hours at the verified billing=8) and runbook:
`aces/TASK21_ACES.md`.
Protocol: [[DECISIONS.md#task21-n512-extended-protocol]]. Not submitted.

**Next (user):**
1. Commit and push.
2. The pvc walltime limit is 2-00:00:00 (verified 2026-10-23), so the run is a single job (user's choice).
3. Run `bash aces/task21_smoke.sh` on a PVC node.
4. Only when instructed: `mkdir -p aces/logs && sbatch aces/run_task21_v1.slurm`.
5. Afterwards: `scripts/task21_analyze.py --t21 artifacts/task21/task21_N512ext_v1`.

## Latest: TASK-20 N=512 run finished (2026-10-23)

Job 2246233: COMPLETED in 22:33 at billing=8 (~3.0 billing-hours); commit dacda13, clean. Best step
2,250; early-stopped at 4,250. Results are local in `artifacts/task20/task20_N512_v1/`.

**Result.** Held-out mel L1 improves over A128 on every set separately: val +7.3%, test +5.1%,
val2 +5.7%, test2 +4.4%, all with CIs > 0. Returns per 4x are diminishing. The train/held-out gap
is nearly closed (memorized 0.44, held-out 0.47-0.49), on a floor ~4.3x the oracle round trip.
Blinded listening: A128 = N512 (2.25 each), oracle 3.88. Caveat: an Adam spike at ~3.5k steps
inside the patience window may have stopped training early. See
[[DECISIONS.md#task20-n512-result]] and `artifacts/task20/analysis_v1/`.

**Next:** user's call. Not committed.

## Previous: TASK-20 N=512 scaling run, prepared for ACES (2026-10-22)

Built and smoke-tested locally; not submitted. Spec and instructions: `aces/TASK20_ACES.md`;
protocol: [[DECISIONS.md#task20-n512-protocol]]. N = 512 (all 128 TASK-19 training tracks
included); held-out sets and crops unchanged.

**Next (user):**
1. Commit and push.
2. On ACES: `bash aces/task20_smoke.sh` on a PVC node.
3. Only when instructed: `sbatch aces/run_task20_N512_v1.slurm`.
4. Afterwards: `scripts/task20_analyze.py`, then `scripts/task20_listening_package.py`.

## Latest: TASK-19 result (2026-10-22)

**Pre-registered codes (labels only): primary `NEITHER_CAPACITY_OR_OBJECTIVE_LIMIT`, secondary `DATA_LIMITED`.**
The data effect is +8.9% on primary held-out (every track improves), just under the 10% bar,
and +10.2% on secondary. Recipe B has no effect. Every arm's val optimum is at 1-2.5k steps,
then it overfits. The gap to the oracle stays ~4.5x the round trip. See
[[DECISIONS.md#task19-data-x-recipe-result]] and `artifacts/task19/analysis_v1/`.

**Blinded listening done (2026-10-22).** All arms are recognizable (mean 2.69-2.75/4) and
indistinguishable from each other; they are far above the controls and below the oracle (4.00).
The mel gain from 4x data is not audible. Examples 05/06 were corrected by the user after
re-listening; they are no longer blind, and nothing depends on them.

**Interpretation (revised at the user's direction).**
- Recognizable reconstruction now has stronger (blinded) listening support.
- Data scaling is promising but modest, on both held-out sets.
- Schedule B is not the solution.
- Training overfits after ~1-2k steps, alongside poor absolute training fit.
- The remaining bottleneck is unresolved. The pre-registered code NEITHER_CAPACITY_OR_OBJECTIVE_LIMIT
  is a label, not a conclusion.

**Next (user is preparing the task):** one larger-data run with the current translator and
validation-based early stopping, same fixed held-out sets, test reported separately from
validation. No capacity or loss changes yet.

## Previous: TASK-19 data x recipe, prepared for ACES (2026-10-22)

TASK-19 (2x2: 32 vs 128 tracks x recipe A vs B) is built, pre-registered in `configs/task19.yaml`,
and smoke-tested locally. Nothing has been submitted. Protocol:
[[DECISIONS.md#task19-data-x-recipe-protocol]]. Instructions and the SU estimate:
`aces/TASK19_ACES.md`. Fault list: no TASK-18 track is affected (7 have repetition partners,
none used).

```bash
uv run python scripts/build_task19_manifests.py --verify-audio      # 3x unchanged, 192 WAVs verified
uv run pytest -q; uv run ruff check .                                # 327 passed, clean
```

**Next (user):**
1. Commit and push (TASK-19 files are uncommitted).
2. On ACES: login-node prep, then `bash aces/task19_smoke.sh` on a PVC node.
3. Only when instructed: `sbatch aces/run_task19_{A32,A128,B32,B128}_v1.slurm`.
4. Afterwards: `scripts/task19_analyze.py`, `scripts/task19_listening_package.py`, then blinded
   listening.

## Latest: TASK-18 result (2026-10-22)

**Held-out track generalization at a recognizable level; pre-registered code `UNDERFIT`.** Job
2240151 (W&B 274iituu). Val and test beat every control. The user heard all held-out `correct`
examples as recognizable, though worse than memorized. This is the first positive held-out
result.
- Current quality is limited mainly by translator fit and optimization (memorized mel L1 is
  3.2x the round trip). Any MuQ-imposed limit is not yet measurable.
- It is not pure underfitting: val is best at step 1,000 while training error keeps falling,
  with the train/val gap widening from 0.20 to 0.36, then an Adam spike at 14k. Diagnose the
  optimization before attributing anything to underfitting.
- The TASK-14 comparison is uncontrolled: target, decoder, translator size, alignment, data and
  metrics all differ.
- Caveats: unblinded listening; split not artist-aware. Details: [[DECISIONS.md#task18-heldout-result]] and
`artifacts/task18_heldout/task18_heldout_v1/TASK-18_RESULTS.md`.

**Next (user's call; nothing authorized):**
- diagnose optimization on the same manifests (W&B train/val curves, LR schedule or warmup, clipping), then capacity;
- blinded listening;
- an artist-aware held-out set;
- more training tracks.

Resources: job 2240151 COMPLETED in 10:27, billing=8 (cpu=8, 1 PVC, 32G), so ~1.4 billing-hours.

## Previous: TASK-18 held-out generalization, prepared for ACES (2026-10-21)

TASK-18 (32/8/8 held-out generalization of the corrected MuQ -> BigVGAN-mel pipeline) is built
and smoke-tested locally. It has **not** been run at full scale; the full job waits for the
user's go-ahead. Protocol: [[DECISIONS.md#task18-heldout-protocol]]. Instructions:
`aces/TASK18_ACES.md`. Local smoke report: `artifacts/task18_smoke/LOCAL_SMOKE_REPORT.md`.

```bash
uv run python scripts/build_task18_manifests.py --verify-audio      # manifests unchanged, 48 WAVs verified
uv run python scripts/train_task18_heldout.py --config configs/task18_smoke.yaml --device mps --run-name smoke_mps_final --wandb-mode disabled
uv run pytest -q; uv run ruff check .                                # 309 passed, clean
```

**Next (user):**
1. On ACES, do the login-node prep from `aces/TASK18_ACES.md`, step 0.
2. ~~ACES smoke~~ **PASSED 2026-10-21 (job 2240126)**. Clean the git tree on ACES before submitting; the smoke ran on a dirty tree.
3. **Submitted by the user 2026-10-21: job 2240151** (`aces/run_task18_v1.slurm`, run `task18_heldout_v1`). Record the job ID, `sacct`, and SUs.
4. Copy back `artifacts/task18_heldout/task18_heldout_v1/`, `run_manifest.json`, and the logs.
5. Listen to `listening/val` and `listening/test`. I then write up the outcome.

Not committed.

## Latest: TASK-17 multi-backend device handling (2026-10-20)

TASK-15/16 scripts now take `--device auto|cuda|xpu|mps|cpu`. All backend calls are in
`src/muq_inversion/device.py`. Intel XPU uses the separate uv project `envs/xpu/`. There's a
bounded per-backend smoke test. The scientific pipeline is unchanged. See
[[DECISIONS.md#task17-multi-backend-device-handling]] and the README section "Running TASK-15/16 on
MPS, CUDA, XPU, or CPU".

```bash
uv run pytest -q; uv run ruff check .                        # 285 passed, clean
uv run python scripts/task15_smoke_test.py --device mps      # PASSED
uv run python scripts/task15_smoke_test.py --device cpu      # PASSED
```

**2026-10-21: XPU smoke test PASSED on ACES** (Intel Data Center GPU Max 1100, 46.6 s). Next:
`mkdir -p aces/logs && sbatch aces/run_task15.slurm` (XPU, W&B offline, HF offline; then
`uv run wandb sync wandb/offline-run-*` on a login node). Earlier procedure, for reference: on ACES PVC, follow the README's "ACES PVC smoke-test procedure"
(`uv sync --project envs/xpu --frozen`, then
`uv run --project envs/xpu python scripts/task15_smoke_test.py --device xpu`). Verify the
partition and GRES names against ACES docs first. The `bluehive/run_task15.slurm` job now runs
the CUDA smoke test first and passes `--device cuda`. Not committed.

## Latest: TASK-16 audit corrections, N=2 rerun (2026-10-19)

**The two TASK-15 audit corrections don't materially change the TASK-15 N=2 result.** Both crops
still pass all four criteria at the best checkpoint (step 4,750); changes are small and mixed in
direction; the shuffled-across control is unchanged. The decode-time clamp raises 1.5-2.4% of mel
bins. Write-up: `artifacts/task16_bigvgan_corrected/TASK-16_RESULTS.md`; summary in
[[DECISIONS.md#task16-bigvgan-alignment-and-floor]]. The audit Markdown now traces the corrected
pipeline (raw vs clamped mel shown separately).

```bash
uv run python scripts/train_task15_bigvgan_memorization.py --config configs/task16_bigvgan_corrected.yaml --n 2   # W&B fjh80gyn
uv run python scripts/task16_compare_n2.py
uv run python artifacts/task15_bigvgan_memorization/audit/trace_pipeline.py   # corrected-pipeline trace
uv run pytest -q; uv run ruff check .   # 243 passed, clean
```

**Next (user's call):** listen to `artifacts/task16_bigvgan_corrected/n2/<track>_predicted.wav`
against `..._bigvgan_oracle.wav` (and TASK-15's `predicted`). TASK-16 forbids N=8 and
generalization runs. `bluehive/run_task15.slurm` still runs TASK-15's original (uncorrected) config.
Committed on request; artifacts are git-ignored.

## Latest: TASK-15 BigVGAN-v2 memorization test (2026-10-16)

**MuQ -> BigVGAN-v2 mel -> frozen BigVGAN-v2 is mechanically viable in the small-N memorization
regime.** Separate from the EnCodec line; not a generalization result. Write-up:
`artifacts/task15_bigvgan_memorization/TASK-15_RESULTS.md`; summary in
[[DECISIONS.md#task15-bigvgan-memorization]].

- Oracle (original -> official mel -> BigVGAN): resynthesis mel L1 0.109 vs. 1.32 for unrelated
  tracks. Good.
- N=2: all pre-registered criteria pass at the best checkpoint (step 4,750). The final step hit
  an Adam/L1 spike, so the evaluation now reports best and final.
- N=8: mel memorized within the vocoder's round-trip error on all 8 crops; 6/8 pass every
  criterion. hiphop.00001 and jazz.00000 miss only pred-vs-oracle waveform corr >= 0.90 (0.898,
  0.888), still rising when the fixed 5,000-step budget ended. Not extended.
- **Listening (user, 2026-10-16): success at both N=2 and N=8**, including the two near-miss
  crops. Conclusion: mechanically viable at N=2 and N=8.
- Shuffled-across control fails on every crop.

**Exact commands:**

```bash
uv run python scripts/fetch_bigvgan.py                                   # official code + weights, pinned
uv run python scripts/task15_bigvgan_oracle.py                           # positive control
uv run python scripts/train_task15_bigvgan_memorization.py --n 2         # W&B seni7345
uv run python scripts/train_task15_bigvgan_memorization.py --n 8         # W&B xl3nmbjt
```

**Next (user's call; nothing authorized):** TASK-15 is complete -- listening done (success at
N=2 and N=8), so the optional longer N=8 rerun isn't needed. TASK-15 explicitly excluded held-out
tests, N > 8, MuQ-layer changes, and loss/architecture sweeps; any of those would be a new task.

Committed on request (code, config, tests, notes); artifacts and `third_party/` are git-ignored.

## Previous gate

**Gate 5 full-GTZAN pilot: run complete, NOT PASSED. Outcome C -- measurable coarse-structure
generalization without recognizable audio reconstruction.**

1. The held-out signal is statistically reliable.
2. Correct conditioning materially affects coarse spectral-energy trajectories.
3. Temporal ordering contributes, but less than track identity.
4. Waveform recovery is negligible on average.
5. Listening does not support recognizable source correspondence.
6. Gate 5 does not pass.

**Bottom line: do not say TASK-14 found nothing; do not call the reconstruction successful.**
Correct MuQ conditioning transfers a measurable coarse envelope to unseen tracks, but not enough
to yield recognizable music in the listened examples. Not B: that would need a practically
meaningful or audibly convincing effect.

**Listening (user, 2026-09-24; 6 blues crops, not blinded):** all `correct` reconstructions are
bad. Only `test_blues.00005_c0_correct` vaguely resembles the original/oracle, and its
`shuffled_across`/`shuffled_time` versions sound similar. A contour check on the same WAVs:
`correct` matches the original's loudness/brightness contours at r = 0.78 / 0.68 vs. ~0 for every
control, and the listener's pick has the best match (0.94 / 0.97). The envelope transfers; nothing
recognizable does.

The user ran the full job on BlueHive (`sbatch bluehive/run_task14.slurm`, job `1956835`, Tesla
T4, 31,078.4s) and copied its logs, `report.json`, and 36 WAVs into `bluehive/logs/` and
`bluehive/artifacts/task14_gate5_full_gtzan/` (git-ignored). Full write-up:
`artifacts/task14_gate5_full_gtzan/TASK-14_RESULTS.md`; summary in
[[DECISIONS.md#task14-gate5-full-gtzan-result]].

- **Spectral (held-out):** log-mel similarity correct 0.73 vs. shuffled_across ~0.37,
  shuffled_time ~0.65, zero ~0; STFT distance agrees. Correct wins on 98-100% of held-out crops
  against each control, in all 10 genres.
- **Waveform (Gate 5's primary endpoint, oracle-relative recovery):** correct 0.0285 val / 0.0315
  test vs. controls 0.007-0.017. CIs exclude zero, 74-93% of examples positive, but deltas miss
  the 0.05 materiality bar. Every condition is at ~1-3% of the oracle.
- **Over-smoothed, not collapsed:** variance ratio 0.27/0.29.

**The script's printed "Outcome D" is superseded.** Its automatic rule considers only the
waveform metric and the 0.05 threshold. An early log-only write-up repeated that D; corrected.

## Exact last successful commands

```bash
# BlueHive (by the user):
sbatch bluehive/run_task14.slurm        # job 1956835, completed

# Local, after the run:
uv run python scripts/train_gate5_full_gtzan_wandb.py --config configs/gate5_smoke.yaml --wandb-mode disabled   # exit 0, new listening selection
uv run python scripts/export_gate5_listening.py --config configs/gate5_smoke.yaml \
    --checkpoint runs/task14_gate5_smoke/checkpoints/best.pt \
    --report artifacts/task14_gate5_smoke/report.json --out-dir /tmp/g5_listen        # 36 WAVs
uv run pytest -q; uv run ruff check .   # 219 passed, clean
```

## Run/checkpoint path

- `bluehive/logs/muq-gate5-full-gtzan-1956835.{out,err}` (git-ignored).
- `bluehive/artifacts/task14_gate5_full_gtzan/report.json` + 36 WAVs (git-ignored; the WAVs are
  the old all-blues selection).
- Checkpoints (`best.pt` = step 38,000): BlueHive scratch only,
  `/gpfs/fs2/scratch/acogliat/invertibility/runs/task14_gate5_full_gtzan/checkpoints/`.
- W&B: `wandb.ai/andrea-cogliati-university-of-rochester/muq-invertibility/runs/01bb8xot`.
- Results report: `artifacts/task14_gate5_full_gtzan/TASK-14_RESULTS.md` (git-ignored).

## Next command

**Blocked on user direction.** Nothing new is authorized. Cheapest informative next steps:

1. **Generate and listen to a representative set.** Copy `best.pt` from BlueHive scratch to
   `runs/task14_gate5_full_gtzan/checkpoints/best.pt`, then:
   ```bash
   uv run python scripts/export_gate5_listening.py \
       --config configs/gate5_full_gtzan.yaml \
       --checkpoint runs/task14_gate5_full_gtzan/checkpoints/best.pt \
       --report bluehive/artifacts/task14_gate5_full_gtzan/report.json
   ```
   That writes worst/median/best per split: val hiphop.00016 / country.00031 / reggae.00012,
   test hiphop.00004 / pop.00098 / classical.00031. This is a robustness check on the Outcome C
   conclusion across genres, not a likely route to passing Gate 5. Start with
   `test_classical.00031_c1`, the crop most likely to be recognizable.
2. **Decide Gate 5's primary endpoint.** Keep it waveform-only (then work on the waveform gap:
   objective, over-smoothing, phase), or add a spectral endpoint (needs a pre-registered
   threshold, since none was defined for log-mel).
3. **Remaining script fixes** if another Gate 5 run is planned: make the outcome rule use more
   than the waveform metric; add `wandb.Audio`; optionally track log-mel per checkpoint. (The
   stale `spectrogram-to-dac-translator` W&B tag is already fixed in both configs.)

## Blockers

1. **User decision** on the endpoint question and next direction.
2. **`best.pt` not local** -- needed for the representative listening set.
3. **Listening**: TASK-14 done on the blues-only subset (see above); genre-diverse set pending.
   Backlog still pending (TASK-13, TASK-12 80k/320k, TASK-11,
   TASK-10, TASK-08, TASK-02, TASK-01).
4. Carried over from TASK-13: pooled vs. held-out-only Gate 4 criteria.
5. TASK-03's BlueHive job still not run; `experiment/mel-hifigan-sanity` diagnostic pending.

## Expected verification criteria for Gate 5

Evaluated: NOT PASSED. Primary waveform endpoint missed materiality (max paired delta 0.025 vs.
0.05); listening (limited) does not support source correspondence. Replication across val/test: yes. Breadth across tracks and genres:
yes. Collapse: no. Defects: none found.

## Notes for whoever resumes this

- Read `artifacts/task14_gate5_full_gtzan/TASK-14_RESULTS.md` first.
- **Do not conclude TASK-14 found nothing, and do not oversell it.** It found a reliable,
  temporally-ordered, per-track loudness/brightness envelope transferring from MuQ to unseen
  tracks. It is measurable but barely audible; recognizable content and waveform-level recovery
  did not transfer.
- **Do not cite the script's printed "Outcome D".** Its rule is waveform-only.
- **Separate waveform from spectral claims.** "No generalization" and "full generalization" are
  both wrong.
- Latent cosine is useless as a discriminator here (0.945-0.975 even with zero conditioning). Use
  log-mel, STFT, variance ratio, and the waveform metrics.
