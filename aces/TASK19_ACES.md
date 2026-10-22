# TASK-19 on ACES: data × recipe (A32, A128, B32, B128)

Question: is held-out mel reconstruction limited by training data, by optimization, or by both?
Protocol, arms and the interpretation table are pre-registered in `configs/task19.yaml`; rationale
in `notes/DECISIONS.md#task19-data-x-recipe-protocol`.

| Arm | Training tracks | Recipe | Job |
| --- | --- | --- | --- |
| A32 | 32 (TASK-18) | A = TASK-18 (constant LR 1e-3) | **evaluation only** of the TASK-18 checkpoints |
| A128 | 128 | A | train 20,000 steps |
| B32 | 32 (TASK-18) | B = A + 1k-step warmup + cosine decay to 1e-5 | train 20,000 steps |
| B128 | 128 | B | train 20,000 steps |

Everything else is identical to TASK-18: pipeline, translator, loss, batch 64, step budget, seed,
evaluation code, controls and metrics. All four arms are evaluated on the same crops:

- the TASK-18 regimes on the 32 TASK-18 training tracks;
- the TASK-18 primary held-out set (8 val + 8 test tracks, identical crops);
- 96 new training tracks;
- a secondary held-out set (24 + 24 new tracks).

The checkpoint is selected on primary val mel L1 only.

**Fault list.** All 192 tracks were checked against Sturm's (2013) published GTZAN fault list
(`data/manifests/gtzan_sturm2013_faults.yaml`, transcribed from arXiv:1306.1461 Table 2: exact,
recording and version repetitions, distortions, artists). Report: `data/manifests/task19_fault_report.json`.

- No repetition crosses splits.
- No TASK-18 track is distorted.
- 7 TASK-18 tracks belong to repetition groups. Their partners are not used anywhere in TASK-18
  or TASK-19, so the TASK-18 manifests are unchanged:
  - train: `country.00008`, `hiphop.00072`, `pop.00090`, `reggae.00056`, `reggae.00092`;
  - val: `pop.00093`;
  - test: `disco.00071`.
- New tracks avoid every repetition group, every distorted track and every checksum duplicate.
- Artist overlap (Sturm's partial artist list):
  - 6 of the 16 TASK-18 held-out tracks share an artist with the 32 TASK-18 training tracks;
  - 33 of the 64 TASK-19 held-out tracks share one with the 128 training tracks; 19 have no listed artist.
  - The analysis reports held-out results split by artist overlap.

## Cost (SU estimate)

TASK-18 job 2240151 used 10 min 27 s at `billing=8` (8 CPUs, 1 PVC, 32 GB), so ~1.4 billing-hours.
TASK-19 arms are larger mainly in evaluation (768 evaluation crops vs 192) and features
(3,584 unique crops vs 896). Training takes the same number of steps.

Breakdown of TASK-18's 627 s on ACES (estimated):
- ~60 s environment and model loading;
- ~40 s MuQ features (0.046 s/crop, measured in the TASK-18 smoke run);
- 167 s training (measured);
- ~360 s evaluation, about 1.9 s per evaluation crop (5 conditions, decoding plus CPU audio
  metrics).

Scaled to TASK-19 (estimate only; ±50%):

| Job | Features | Training | Evaluation | Wall time | Billing-hours (×8) |
| --- | --- | --- | --- | --- | --- |
| A32 (evaluation only) | ~165 s | — | ~1,460 s | ~28 min | ~3.8 |
| A128, B32, B128 (each) | ~165 s | ~250 s (more curve sets) | ~1,460 s | ~32 min | ~4.3 |
| **All four** | | | | ~2 h of PVC time | **~17 (range ~9–25)** |
| Interactive smoke (`task19_smoke.sh`) | | | | ~20–30 min held | ~3–4 |

Billing is for the allocation's wall time, so the interactive smoke counts while the shell is open.
The 3 h limit per job leaves a wide margin. Record the actual `sacct` figures afterwards.

## 0. Login node (once)

```bash
cd /home/u.ac370321/project/invertibility     # or wherever the repo lives on ACES
git pull                                       # TASK-19 code, configs, manifests (commit/push from the Mac first)
module load WebProxy uv
export UV_PROJECT=$PWD/envs/xpu
uv sync --frozen
uv run python scripts/build_task19_manifests.py --verify-audio   # 3 x "unchanged", "verified sha256 of 192 WAVs"
ls runs/task18_heldout/task18_heldout_v1/checkpoints/{best,final}.pt   # A32 evaluates these
```

All 192 tracks must be in `data/raw/gtzan` (the full GTZAN folder is enough).

## 1. Bounded smoke test (interactive PVC node)

```bash
srun --partition=pvc --gres=gpu:pvc:1 --ntasks=1 --cpus-per-task=8 --mem=32G --time=01:30:00 --pty bash
cd /home/u.ac370321/project/invertibility && module load WebProxy uv
bash aces/task19_smoke.sh
```

The script runs these steps and stops at the first failure:
1. the XPU backend smoke test;
2. the manifests rebuild unchanged and all 192 checksums are verified;
3. all four arms on a tiny subset, 40 steps each; A32 evaluates the real TASK-18 checkpoints;
4. the cross-arm analysis and the blinded package;
5. resume equivalence for recipe B.

Send back `artifacts/task19_smoke/aces_<job>_<time>/ACES_SMOKE_REPORT.md`. Its numbers are not
results. They show that every path runs, and the A32 sanity line compares the re-evaluation
against TASK-18.

## 2. Full runs (only when instructed)

```bash
git status --short          # clean, so the recorded commit is exact
mkdir -p aces/logs
for a in A32 A128 B32 B128; do sbatch aces/run_task19_${a}_v1.slurm; done
```

- **Jobs.** The four jobs are independent, each on 1 PVC with a 3 h limit. They share the
  content-addressed feature cache. Writes to it are atomic, so concurrent jobs are safe.
- **W&B.** Runs are online through WebProxy: group `task19-data-x-recipe`, one run per arm. A32
  replays the TASK-18 training curves into its own run. If the proxy fails, resubmit with
  `WANDB_MODE_ARG=offline sbatch ...` and later run `uv run wandb sync $(ls -d wandb/offline-run-* | sort)`.
- **Resuming.** Resubmitting a job resumes it from `latest.pt`. A finished run is never
  overwritten.

## 3. After the runs

```bash
for j in <jobids>; do sacct -j $j --format=JobID,JobName,State,Elapsed,AllocTRES%60; done
uv run python scripts/task19_analyze.py \
    --arm A32=artifacts/task19/task19_A32_v1 --arm A128=artifacts/task19/task19_A128_v1 \
    --arm B32=artifacts/task19/task19_B32_v1 --arm B128=artifacts/task19/task19_B128_v1
uv run python scripts/task19_listening_package.py \
    --arm A32=artifacts/task19/task19_A32_v1 --arm A128=artifacts/task19/task19_A128_v1 \
    --arm B32=artifacts/task19/task19_B32_v1 --arm B128=artifacts/task19/task19_B128_v1
```

Copy back `artifacts/task19/` (all arms, `analysis_v1/`, `listening_blind_v1/` and
`listening_blind_v1_KEY/`), `runs/task19/*/run_manifest.json`, and `aces/logs/`.

## 4. Blinded listening

`listening_blind_v1/example_NN/`: 16 examples (the TASK-18 listening crops), in random order. Each has
`reference_original.wav` and `clip_A`–`clip_H`. The clips, in random order per example:

- the BigVGAN oracle;
- each arm's correct prediction (4);
- A32's shuffled-across, shuffled-time and constant-zero controls.

Rate each clip in `ratings.csv` (0–4: recognizably the same music as the reference). Do **not** open
`listening_blind_v1_KEY/` until every rating is done. Then send me `ratings.csv`, and I'll
unblind it against the key.
