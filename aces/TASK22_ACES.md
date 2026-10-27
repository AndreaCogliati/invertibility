# TASK-22: first-conformer-block MuQ features → mel (specification, ACES runbook, reporting template)

**Question.** Does the output of MuQ's first encoder block support better reconstruction than
`last_hidden_state` (TASK-20/21) with the current translator and reconstruction objective?

This is a representation experiment only. The translator is trained from scratch. Everything else is
TASK-20/21's: tracks, crops, sampler, splits, targets, decoder, model, loss, optimizer, LR, batch,
precision and validation schedule.

## 1. Representation: layer-extraction verification

The installed MuQ implementation was read, then checked on real crops with forward hooks:
- `muq` 0.1.0 wrapping transformers 4.55.0 `Wav2Vec2ConformerEncoder`;
- script `scripts/task22_verify_layer.py`;
- output `artifacts/task22/layer_verification/layer_verification.json` (MPS, 8 primary-val crops);
- verdict **PASS**.

The encoder loop appends `hidden_states` *before* each block. After the loop it applies an encoder
`LayerNorm` and appends the result. So:

| `hidden_states[k]` | is exactly (hook-verified, `torch.equal`) |
| --- | --- |
| 0 | the **input** of block 1: the `Conv2dSubsampling` frontend output (dropout is identity in eval). Not a block output |
| 1 | the **output of block 1**, `model.model.conformer.layers[0]`, after that block's own `final_layer_norm` = the input of block 2 |
| k = 2..11 | output of block k |
| 12 | encoder `layer_norm`(output of block 12) == `last_hidden_state` (the baseline) |

**Selected:** `muq_layer: {block: 1}`, giving `hidden_states[1]`, named `conformer_block_1_output`.

**Extraction settings:**
- `MuQ(x, output_hidden_states=True).hidden_states[1]` under `torch.inference_mode()`;
- MuQ is frozen: `eval()`, `requires_grad=False`;
- no `attention_mask`;
- float32;
- the same peak-normalized 47,872-sample (1.995 s) crops at 24 kHz.

`FrozenMuQEncoder.encode_block_output` re-checks the index on **every extraction batch** with hooks:
- `hidden_states[0]` is the input of block 1;
- `hidden_states[1]` is the output of block 1, and differs from index 0;
- values are finite.

It stops otherwise. The check result is stored in the cache, the run manifest and the W&B config
(`muq_feature.index_check`).

**Geometry:** (B, 50, 1024) per crop, 25.07 frames/s. This is identical to `last_hidden_state`, so
the existing translator fits unchanged (1,271,908 parameters, no input-layer change).

**Alignment and valid frames:**
- Conformer blocks keep the sequence length and order (no striding or pooling).
- Token i therefore has the frontend's time center at every depth, so `muq_token_centers` and the
  frame-center interpolation positions are the same as for TASK-21.
- Mel source positions lie in [0, 49].
- Every crop is full-length real audio with no padding or mask, so all 50 tokens are valid.

**Determinism and finiteness:** two passes are bit-exact (MPS), and all values are finite.

**Normalization policy, preserved:** none. Raw features go into the translator's input `Linear`, as
for every earlier task. There are no statistics to recompute.

Block-1 features differ in scale from `last_hidden_state`:
- std 1.72 vs 0.85;
- mean frame L2 norm 55 vs 27;
- mean per-frame cosine to `last_hidden_state` 0.11.

This scale difference is part of the representation as given. With unchanged Adam and LR it is part
of what "with this translator and recipe" means.

**Cache separation:**
- The cache lives in `data/cache/task22/`. Its file stem is `task18_v1_L1_<digest>`, and the digest
  includes `muq_layer` and `hidden_state_index`.
- A cache is used only if its stored key parts match exactly.
- The training script refuses a non-default layer that points at `data/cache/task18`.
- `last_hidden_state` cache names are unchanged, so TASK-21's cache is still hit by the baseline
  evaluation.

## 2. Controlled comparison: config diff (`uv run python scripts/task22_config_diff.py`)

Resolved TASK-21 arm `N512ext` vs TASK-22 arm `L1`. Every changed key:

| key | TASK-21 | TASK-22 L1 |
| --- | --- | --- |
| `muq_layer` | null (`last_hidden_state`) | `{block: 1}` (`hidden_states[1]`) |
| `feature_cache_dir` | `data/cache/task18` | `data/cache/task22` |
| `init_from_checkpoint` | TASK-20 `latest.pt` (step 4,250) | null (from scratch) |
| `steps` | null | 200000 |
| `runtime_budget_hours` | 47.0 | null |
| `checkpoint_policy.milestones` | 10k, 20k, 50k, 100k, 200k, 500k … 20M | 10k, 20k, 50k, **60k**, 100k, 200k |
| `extra_checkpoints` | — | 10k, 20k, 50k, 60k, 100k (+ best, final = 200k) |
| `task`, `arm`, `run_name_prefix`, `run_root`, `artifact_root`, `results_name`, `wandb_group`, `wandb_tags` | task21 … | task22 … |

**Unchanged** (asserted in `tests/test_task22.py`):
- manifests (`task20_tracks.csv`, `task20_crops.csv`): 512 training tracks, the 12,288-crop pool,
  and every evaluation crop;
- sampler and seed 0, so the batch sequence and translator init are identical;
- eval sets, quick-eval sets `[memorized, train_unseen, val]`;
- validation schedule (every 250 steps through 20,000, then every 1,000) and logging schedule;
- L1 on log-mel, Adam 1e-3 (0.9/0.999), constant LR, batch 64, FP32, no gradient clipping
  (norm measured only);
- `hidden_channels` 256, frame-center alignment, decode clamp, BigVGAN revision, MuQ model ID.

Not added: fresh crops, latent-consistency or adversarial losses, and LR schedules.

**Why TASK-21 is a valid from-scratch baseline:**
- Its trajectory is TASK-20 steps 0–4,250 (seed 0, same recipe; early stopping only ended the run)
  plus TASK-21's continuation.
- The continuation restored model, Adam, sampler and RNG, and reproduced the parent's step-4,250
  validation exactly (0.505893).
- Validation ran every 250 steps through 20,000 in both.

**TASK-21's window best, verified from its own history (not assumed):**
- 260 checks at steps ≤ 200,000; the lowest primary-val mel L1 is at **step 60,000** (0.45540).
- That is exactly TASK-21's `best.pt` (step 60,000).
- So the 60k matched step and TASK-21's window best are the same checkpoint.
- Milestones at 10k, 20k, 50k, 100k and 200k exist (`runs/task21/task21_N512ext_v1/checkpoints/`).

## 3. Training (`configs/task22.yaml`, arm L1)

- 200,000 optimizer steps from scratch, no patience. Stops early only on a non-finite loss or
  gradient, or an execution failure.
- `best.pt` is selected on primary-val correct-conditioning mel L1 only.
- Test, test2 and val2 are never evaluated during training.
- Checkpoints:
  - `latest.pt` every 5 min, on every new best, and on graceful stop;
  - 3 rolling `recovery_*.pt` files;
  - `best.pt`, `final.pt` (200k);
  - milestones at 10k, 20k, 50k, 60k, 100k and 200k.
- Each checkpoint carries model, Adam, sampler, Python/NumPy/torch-CPU RNG and global step. There is
  no scaler (FP32) and no scheduler (constant LR).
- SIGTERM (Slurm `--signal=B:TERM@600`) → `latest.pt`, exit 3; resubmitting resumes.
- **W&B** (group `task22-first-layer`) logs:
  - config, including `muq_feature` (layer identity and index check), seed, commit, and manifest,
    lock and BigVGAN hashes;
  - train L1, primary val, LR and gradient norm;
  - steps/s, cumulative runtime, pool passes, track cycles and best step;
  - all mirrored to `runs/task22/task22_L1_v1/metrics.jsonl`.

## 4. Evaluation

**TASK-22 job:**
- `best.pt` gets the full evaluation: all 8 sets × 5 conditions, audio metrics, listening WAVs.
- `final.pt` (200k) and the 10k, 20k, 50k, 60k and 100k milestones get mel-level evaluation.

**Baseline job (`T21eval`, evaluation only):**
- TASK-21's 10k, 20k, 50k, 100k and 200k milestones get mel-level evaluation, on the existing
  `last_hidden_state` cache.
- TASK-21's `best.pt` (60k) and `final.pt` are re-evaluated. This doubles as a reproducibility check
  against TASK-21's own evaluation.
- A missing milestone is reported as unavailable, never substituted.

**`scripts/task22_analyze.py`:**
- Comparisons at matched steps 10k, 20k, 50k, 60k, 100k and 200k, plus window best vs window best.
- Each set reported separately:
  - unseen tracks: val, test, val2, test2;
  - unseen crops of training tracks: core 32 and the 96;
  - memorized crops: core 32 and the 96.
- Paired by crop on identical crops, with track-clustered 95% CIs, and per-track differences on the
  held-out sets.
- Controls at each window-best checkpoint.
- Curves: memorized-core, unseen-training-crop and primary-val, for both runs, plus the val
  difference at every shared validation step.

**Listening (`scripts/task22_listening_package.py`):**
- The 16 fixed primary val/test listening crops (unseen tracks).
- Per example: the labeled original plus 3 blinded clips (BigVGAN oracle, L1 window best, L12 window
  best). The key is in a separate `_KEY` folder.
- The package stops if either run's audio is not its window-best checkpoint.
- Results are **pending** until ratings are supplied (`--score`).

## 5. Resources (from measured ACES runs, verified billing rate 8 per job-hour)

| job | what | expected | walltime | billing-hours |
| --- | --- | --- | --- | --- |
| `run_task22_L1_v1.slurm` | ~3 min start-up; ~2 min block-1 extraction for 12,800 crops (TASK-20: 92 s); ~26 min for 200k steps (TASK-21: 0.42 h for its first 200k at ~132 steps/s); ~20 min best/final evaluation; ~5–10 min for 5 milestone evaluations | **~1 h** | 4 h | ~8–9 (max 32) |
| `run_task22_T21eval_v1.slurm` | start-up; cache hit; W&B replay of 22,297 history entries; ~20 min best/final evaluation; ~5–10 min for 5 milestones | **~35 min** | 2 h | ~5 (max 16) |
| smoke (interactive) | `aces/task22_smoke.sh` | ~20–30 min | 1 h | ~4–8 |

Total: about **14 billing-hours**, plus the smoke. Disk: ~12 checkpoints of ~15 MB each, plus a
~2.6 GB feature cache in `data/cache/task22`. Whether SUs equal billing-hours is not verified.

## 6. ACES runbook

```bash
# 1. login node (login.aces.hprc.tamu.edu), repo at /home/u.ac370321/project/invertibility
git pull && module load WebProxy uv && export UV_PROJECT=$PWD/envs/xpu && uv sync --frozen
uv run python scripts/build_task20_manifests.py                          # 3 x "unchanged"
ls runs/task21/task21_N512ext_v1/checkpoints/{best,final,milestone_000200000}.pt

# 2. bounded smoke test (interactive PVC node; separate output directories)
srun --partition=pvc --gres=gpu:pvc:1 --ntasks=1 --cpus-per-task=8 --mem=32G --time=01:00:00 --pty bash
module load WebProxy uv && bash aces/task22_smoke.sh                     # send back SMOKE_REPORT.md

# 3. the runs -- only when instructed; independent, can run at the same time
mkdir -p aces/logs && sbatch aces/run_task22_L1_v1.slurm && sbatch aces/run_task22_T21eval_v1.slurm

# 4. afterwards
sacct -j <jobid1>,<jobid2> -X --format=JobID,JobName,State,Elapsed,AllocTRES%60
```

Copy back:
- `artifacts/task22/` (`task22_L1_v1/`, `task22_T21eval_v1/`, `layer_verification_<job>/`);
- `runs/task22/task22_L1_v1/{run_manifest.json,metrics.jsonl}` and `runs/task22/task22_L1_v1/checkpoints/`;
- `aces/logs/`.

Then run on the Mac:

```bash
uv run python scripts/task22_analyze.py --t22 artifacts/task22/task22_L1_v1 --t21eval artifacts/task22/task22_T21eval_v1 \
    --sacct-hours <L1 elapsed h> <T21eval elapsed h>
uv run python scripts/task22_listening_package.py --l1 artifacts/task22/task22_L1_v1
```

## 7. Reporting template (`artifacts/task22/analysis_v1/TASK-22_RESULTS.md`)

1. **Runs:**
   - job IDs;
   - commit (clean?);
   - elapsed time;
   - billing-hours;
   - W&B IDs.
2. **Layer verification:**
   - index map;
   - module;
   - shape and frame rate;
   - per-batch index check (in the report's `muq_feature`);
   - cache path, separate from `data/cache/task18`.
3. **Comparability:**
   - identical manifest, lock and BigVGAN hashes, pool, eval crops, parameters and seed (checked by
     the analysis);
   - the baseline re-evaluation reproduces TASK-21's own numbers (max |Δ| mel L1).
4. **Training:**
   - L1 best step and final step;
   - draws per track and pool passes;
   - steps/s;
   - any instabilities: spikes, with train-loss and grad-norm coincidence.
5. **Matched steps** (10k, 20k, 50k, 60k, 100k, 200k), each set separately:
   - L12, L1, relative difference, paired Δ [95% CI], fraction of tracks.
   - Unavailable checkpoints are named.
6. **Window best vs window best** (≤ 200k):
   - L1 step vs L12 step 60,000;
   - the same table;
   - per-track held-out differences.
7. **Controls** at each window best (shuffled across, shuffled in time, zero, average mel).
8. **Unseen tracks vs unseen crops of training tracks vs memorized crops:** where any L1 difference
   appears.
9. **Curves:** `curves.png` and `val_difference.png`.
10. **Listening:** PENDING until ratings are supplied. Then mean rating per system and per-example
    L1 vs L12 counts.
11. **Interpretation limits:**
    - single seed;
    - reused evaluation sets; selection on primary val only;
    - CIs are track-clustered and do not measure seed variability;
    - an L1 advantage shows greater recoverability with this translator and recipe, not that deeper
      layers discarded information;
    - a null result does not establish equal information content;
    - the input scale differs (×2) under an unchanged recipe.
