# Running TASK-03 on BlueHive

This is a manual handoff, not an automated deployment: the environment that produced this
repository has no SSH access to BlueHive and no `sbatch`/`squeue`/`srun` available locally, so
none of this has been run on the actual cluster. Everything below was validated the way it *can*
be validated from here -- unit tests, and two real W&B-logged dry runs on the M4 Max (`--steps 50`
in `--wandb-mode disabled`, then `--steps 2500` in `--wandb-mode online`, both clean, no errors --
see `notes/DECISIONS.md#task03-bluehive-wandb`). You'll need to do the cluster-side steps
yourself and report back what you find so `run_task03.slurm`'s placeholder resource requests can
be corrected with real numbers.

## 1. Get the code onto BlueHive

```bash
# From your machine, or directly on a BlueHive login node:
git clone <this repo's remote> muq-invertibility   # or scp/rsync if there is no shared remote
cd muq-invertibility
git checkout experiment/muq-layer-sweep   # or whichever branch has tasks/TASK-03.md's code
```

## 2. Install `uv` and sync the environment

BlueHive login nodes typically have outbound network access for package installs (compute nodes
may not -- check your site's network policy). Astral `uv` only, per CLAUDE.md rule 4/5:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh    # if uv isn't already available/module-loaded
uv python pin 3.12
uv sync --frozen
```

`uv sync` resolves `torch>=2.14.0` (this project's exact pin, unchanged) to the CUDA-enabled
Linux wheel automatically -- no separate CUDA-specific index needed for a standard PyPI install.
Confirm CUDA is visible before submitting anything:

```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If this prints `False`, do not proceed to a GPU job -- diagnose the module/CUDA-driver setup
first (`nvidia-smi`, `module avail cuda`, `module load cuda/<version>` per your site's docs).

## 3. Get GTZAN and the precomputed caches onto BlueHive

Per CLAUDE.md's "no hidden downloads during training" rule, dataset preparation is a separate,
explicit step from training -- and per that same rule, `run_task03.slurm` refuses to start
without these already present. Two options:

**Option A (recommended): copy the caches built on the M4 Max.** They're deterministic given the
same manifest/audio/code (content-addressed, per this project's cache discipline), so copying is
equivalent to regenerating:

```bash
# From the M4 Max, to a BlueHive login node:
rsync -avz --progress \
  data/manifests/dense_crops.csv \
  data/cache/dense/mel_dac_cache.pt \
  data/raw/gtzan/ \
  bluehive:/path/to/muq-invertibility/data/
```

`data/cache/dense/mel_dac_cache.pt` is 10.36GB -- check your BlueHive home/project quota before
transferring (`quota -s` or your site's equivalent).

**Option B: regenerate on BlueHive.** Requires `data/raw/gtzan/` present (GTZAN itself, ~1.2GB;
this project does not auto-download it -- see the top-level `README.md`/`CLAUDE.md` for the
expected layout) and `data/manifests/gtzan_manifest_smoke.csv` (tracked in git, already present
after cloning):

```bash
uv run python scripts/build_dense_gtzan_manifest.py
uv run python scripts/precompute_dense_cache.py   # took 354.5s on the M4 Max; CUDA should be comparable or faster
```

## 4. W&B credentials

```bash
uv run wandb login   # interactive; caches to ~/.netrc on the login node (shared with compute nodes via NFS home, typically)
```

Confirm: `cat ~/.netrc` should show a `machine api.wandb.ai` block. `run_task03.slurm` checks for
this (or `$WANDB_API_KEY`) before starting and fails fast with a clear message if neither is
present, rather than running for hours and failing silently at the first `wandb.init()` call.

## 5. Fill in the job script's placeholders

`run_task03.slurm` has `CHANGE_ME` placeholders for `--partition`, `--account`, and
`--mail-user` -- these are specific to your BlueHive allocation and unknown from this
environment. Find them with:

```bash
sacctmgr show assoc user=$USER format=account,partition   # your account(s) and available partitions
sinfo -o "%P %G"                                            # partitions and their GPU resources
```

## 6. Submit

```bash
sbatch bluehive/run_task03.slurm
squeue -u $USER                       # watch it queue/run
tail -f muq-spec-dense-wandb-<jobid>.out
```

The W&B run URL appears near the top of the job's stdout (`wandb: 🚀 View run ... at:
https://wandb.ai/...`) within the first minute or two -- you can watch training live from there
without needing to tail cluster logs.

## 7. Record the cluster configuration (CLAUDE.md rule 2 / this task's explicit requirement)

Once you've run it, fill in and keep for the record (`notes/DECISIONS.md` or
`artifacts/task03_wandb_repro/bluehive_environment.md` is a reasonable place):

- GPU type and count (`nvidia-smi --query-gpu=name,count --format=csv`, also printed by the job
  script itself into its `.out` file)
- CPU count and memory allocation actually granted (`seff <jobid>` after completion, or `sacct -j
  <jobid> --format=JobID,MaxRSS,AllocCPUS,Elapsed`)
- Wall-clock time actually taken (compare against the M4 Max's 4677s train + 3805s eval)
- CUDA/driver version (`nvidia-smi` header)
- BlueHive job ID (already logged automatically to W&B's run config via `$SLURM_JOB_ID` --
  `logging_utils.capture_environment_manifest` picks it up, and `--job-id` on the training script
  is also available as an explicit override)

## What this script does NOT do

- It does not modify the scientific formulation in any way (dataset, split, spectrogram
  parameters, architecture, loss, optimizer, step budget all match TASK-02 exactly by default --
  see `scripts/train_spectrogram_to_dac_wandb.py`'s module docstring for the short, explicitly
  documented list of instrumentation-only deviations).
- It does not auto-scale to multiple GPUs. This model doesn't need it (see the resource
  justification comment in `run_task03.slurm`); if BlueHive assigns you a multi-GPU node,
  request only 1 GPU (`--gres=gpu:1`) rather than changing the training loop.
- `--steps` on the training script exists ONLY for infrastructure validation, as done on the M4
  Max before this handoff. Never pass `--steps` when running the actual reproduction -- the
  default (450000) is what makes the run comparable to TASK-02.
