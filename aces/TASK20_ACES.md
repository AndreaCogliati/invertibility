# TASK-20: N=512 data-scaling run (specification and ACES instructions)

**Question.** Does going from 128 to 512 training tracks keep improving held-out mel
reconstruction under the unchanged TASK-19 recipe A, with validation-based early stopping?
Single seed: a scaling data point, **not** a measurement of MuQ's information ceiling.

## Specification (`configs/task20.yaml`, inheriting `configs/task19.yaml`)

| | TASK-19 A128 | TASK-20 N512 |
| --- | --- | --- |
| Training tracks | 128 | **512** = the 128 + 384 new |
| Translator, MuQ layer, mel target, BigVGAN, clamp, alignment | — | unchanged |
| Loss, optimizer, LR (constant 1e-3), batch 64, FP32, seed 0 | — | unchanged |
| Validation cadence | every 500 steps | every **250** steps |
| Stopping | fixed 20,000 steps | **early stopping**: patience 8 checks (2,000 steps), min_delta 0, max 20,000 |
| Checkpoint selection | lowest primary val mel L1 | same metric (primary val, correct conditioning) |
| Eval sets and crops | TASK-19 | identical (asserted crop by crop) |

**Data.**
- **Training tracks.** The 384 new tracks come from the committed train split under TASK-19's
  eligibility rules: no Sturm (2013) repetition member, no distortion, no checksum duplicate,
  at least 29 s. 532 were eligible, so the target of 512 is met.
- **Genre balance.** Genres get 51 tracks each, with metal and reggae at 52. Pop is capped at
  51 by availability.
- **Held-out sets.** The 64 held-out tracks (primary 8 + 8, secondary 24 + 24) and all their
  crops are unchanged. No held-out track can be a training candidate. No repetition group spans
  splits.
- **Crops.** The 384 new tracks get only training-pool crops (24 per track, outside their
  reserved windows); they are not evaluated.
- **Manifests.** `data/manifests/task20_{tracks,crops}.csv`, `task20_fault_report.json`.

**Early stopping.** It uses only primary val (the 8 TASK-18 val tracks), correct conditioning,
with the existing mel L1. Test, val2, test2 and listening are never used for selection or
stopping. The state (counter, stop flag, reason, step) lives in every checkpoint, so a resumed
job continues the same count. `best.pt` is the selected model; `latest.pt` is resumable; `final.pt`
is the state at the stop.

**Code changes (all inactive for TASK-18/19 configs).**
- `scripts/train_task18_heldout.py`:
  - optional `early_stopping`;
  - features computed only for crops in use;
  - chained `base_config`;
  - smoke grouping by cohort;
  - N, stop step and stop reason logged to the report and to W&B.
- New: `src/muq_inversion/task20.py`, `scripts/build_task20_manifests.py`,
  `scripts/task20_analyze.py`, `scripts/task20_listening_package.py`, `tests/test_task20.py`.

**Reporting** (`scripts/task20_analyze.py`).
- Each set separately: val, test, val2, test2, unseen crops of training tracks (core 32, new 96),
  and memorized crops.
- Combined val+test and val2+test2 only for comparison with TASK-19.
- N512 vs A128 and vs A32 on identical crops: paired, track-clustered 95% CI.
- N512 controls, and curves with the stopping point.

**Listening.** `scripts/task20_listening_package.py` builds a blinded package: the 16 fixed
TASK-18/19 crops; per example the reference original plus 3 random clips (oracle, A128, N512);
the key in a separate folder.

## ACES

**Cost.** TASK-19 A128 took 22:50 at billing=8 (20,000 steps, 3.6k feature crops). N512 has
~13.4k feature crops (~10 min of MuQ), at most 20,000 steps with validation twice as often,
and the same evaluation (~15 min). Estimate: ~25–40 min, ~4–6 billing-hours (64 GB requested;
check whether billing changes).

```bash
# login node (login.aces.hprc.tamu.edu), repo at /home/u.ac370321/project/invertibility
git pull && module load WebProxy uv && export UV_PROJECT=$PWD/envs/xpu && uv sync --frozen
uv run python scripts/build_task20_manifests.py --verify-audio      # 3 x "unchanged", 576 WAVs verified

# bounded smoke (interactive PVC node)
srun --partition=pvc --gres=gpu:pvc:1 --ntasks=1 --cpus-per-task=8 --mem=32G --time=01:00:00 --pty bash
module load WebProxy uv && bash aces/task20_smoke.sh                # send back ACES_SMOKE_REPORT.md

# full run -- only when instructed
mkdir -p aces/logs && sbatch aces/run_task20_N512_v1.slurm
```

W&B: group `task20-n512-scaling` (online via WebProxy). The run logs:
- config, seed, commit and manifest hashes;
- N, learning rate and lr curve;
- training and validation curves;
- the early-stopping counter;
- best step, final step and stopping reason (in the run summary).

**After the run.** Copy back `artifacts/task20/`, `runs/task20/*/run_manifest.json` and
`aces/logs/`, then run on the Mac:

```bash
uv run python scripts/task20_analyze.py --n512 artifacts/task20/task20_N512_v1
uv run python scripts/task20_listening_package.py --n512 artifacts/task20/task20_N512_v1
```
