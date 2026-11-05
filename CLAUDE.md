# CLAUDE.md

## Project Mission

Build a small, reproducible proof-of-concept that tests whether the final, time-resolved MuQ representation can be decoded into recognizable music audio.

This is an interpretability experiment, not a codec benchmark and not an attempt to achieve state-of-the-art synthesis quality. Fail early: validate the data path, tensor geometry, conditioning, loss behavior, checkpointing, and basic reconstruction before increasing model size or scope.

## Fixed Experimental Scope

- Development machine: Apple MacBook Pro with M4 Max and 64 GB unified memory.
- Dataset: GTZAN, 1,000 mono WAV tracks, about 30 seconds each, originally sampled at 22.05 kHz.
- Encoder: frozen `OpenMuQ/MuQ-large-msd-iter`.
- Encoder input: mono audio resampled to 24 kHz.
- Reconstruction pipeline (current): audio -> frozen MuQ `last_hidden_state` -> learned translator -> frozen EnCodec continuous latent -> frozen EnCodec decoder -> reconstructed audio. The translator is the only trained component; MuQ and EnCodec are both frozen throughout.
- Decoder: frozen, pretrained EnCodec (`facebook/encodec_24khz`). This project does not train, fine-tune, or design a decoder -- EnCodec's own decoder is used as-is to turn a predicted continuous latent into audio.
- Translator input: MuQ `last_hidden_state`, meaning the final time-resolved encoder representation. In this project, call this the "bottleneck" representation.
- Initial experiment: one representation only. Do not combine intermediate layers, use skip connections from MuQ, or implement learned layer mixing unless explicitly requested later.
- Training target: EnCodec's continuous (post-quantization) latent, not the raw waveform directly. The translator is trained to predict this latent; the frozen EnCodec decoder then turns it into the reconstructed waveform.
- DAC (`descript/dac_24khz`) is no longer the primary regression target: its continuous latent proved highly sensitive to sub-frame input shifts and poorly aligned with decoded-audio similarity. EnCodec's continuous latent substantially improved scaling behavior and avoided DAC-style norm collapse under the same translator, loss, and optimizer. DAC-based findings remain valid prior experimental evidence, not invalidated results -- see `notes/DECISIONS.md` for the full history.
- Latest recorded benchmark (N=32 training examples, 320,000 training steps): the existing small ConvNet-based translator reached 92.7% of the EnCodec oracle reconstruction ceiling (waveform correlation), with diminishing but still-positive returns from additional training compute, and a variance ratio of 0.738 (short of the 1.0 no-collapse reference).
- Package manager: Astral `uv`, using native project commands only.

## Scientific Questions

Primary question:

> Can a decoder recover recognizable audio from MuQ's final time-resolved representation alone?

Secondary diagnostics:

1. Does the decoder use temporal MuQ conditioning rather than merely learning dataset-level audio statistics?
2. Which coarse attributes survive: rhythm, pitch, instrumentation, dynamics, and spectral envelope?
3. Is failure caused by implementation or optimization problems, or by insufficient information in the representation?

Do not broaden the project to layer-wise inversion, multi-model comparisons, BigVGAN, flow matching, or full-scale perceptual evaluation until the primary question is answered.

## Non-Negotiable Reproducibility Rules

1. Keep MuQ frozen and in evaluation mode at all times.
2. Record the exact MuQ model identifier, package versions, Python version, macOS version, PyTorch version, device, dtype, random seed, and Git commit in every run manifest.
3. Preserve `pyproject.toml` and `uv.lock`. Never hand-edit dependency versions without updating the lockfile.
4. Never use `pip`, `uv pip`, `requirements.txt`, Conda, Poetry, or an independently managed virtual environment.
5. Run project commands through `uv run`.
6. Store all tunable parameters in versioned YAML configuration files. Do not hide important defaults in code.
7. Split GTZAN deterministically and save the exact manifests. Prefer known fault-filtered or artist-aware partitions when available; otherwise document the limitation clearly.
8. Never overwrite checkpoints, run manifests, metrics, or generated samples from a previous run.
9. All long-running stages must be resumable and idempotent.
10. Do not silently fall back from MPS to CPU. Log the selected device and fail clearly unless a configuration explicitly permits fallback.
11. Log every experiment to Weights & Biases, whether or not the task brief mentions it. This covers training runs, evaluation-only jobs, diagnostics, and smoke tests.
    - **Helper:** use `muq_inversion.wandb_logging.init_run`, which logs to the project's W&B project and records the environment manifest in the run config.
    - **Grouping:** one W&B group per task (e.g. `task24-fitting-ladder`); one W&B run per arm or independent run; tags for the task and the representation.
    - **What to log:** the resolved config and the training, evaluation and monitoring metrics at their native cadence. Store the W&B run id in the checkpoint, so a resumed run continues the same W&B run.
    - **Failures:** if online initialization fails, continue offline rather than stopping training. Log failures must never stop training.
    - **Local copy:** always keep a local JSONL mirror of everything logged. The local files stay the source of truth for reports.
    - **Smoke tests:** run with `--wandb-mode disabled` (or offline) so smoke runs do not clutter the project.

## Native `uv` Workflow

Use commands in this style:

```bash
uv init --package
uv python pin 3.12
uv add torch torchaudio librosa soundfile numpy scipy pyyaml tqdm tensorboard
uv add --dev pytest ruff mypy
uv sync
uv lock
uv run python -m muq_inversion.cli doctor
uv run python -m muq_inversion.cli prepare-data --config configs/pilot.yaml
uv run python -m muq_inversion.cli extract --config configs/pilot.yaml
uv run python -m muq_inversion.cli train --config configs/pilot.yaml
uv run python -m muq_inversion.cli evaluate --config configs/pilot.yaml
uv run pytest
uv run ruff check .
```

Dependency compatibility on Apple Silicon must be verified before implementation. Prefer released wheels that support macOS arm64. If MuQ or a dependency is incompatible, document the exact failure in `notes/DECISIONS.md` and make the smallest reversible change.

## Proposed Repository Layout

```text
.
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── uv.lock
├── .python-version
├── .gitignore
├── configs/
│   ├── smoke.yaml
│   └── pilot.yaml
├── data/
│   ├── raw/                 # ignored; GTZAN source audio
│   ├── processed/           # ignored; 24 kHz mono audio or indices
│   ├── manifests/           # tracked split manifests and checksums
│   └── cache/               # ignored; optional MuQ features
├── src/muq_inversion/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── data.py
│   ├── audio.py
│   ├── encoder.py
│   ├── cache.py
│   ├── models/
│   │   ├── generator.py
│   │   ├── discriminators.py
│   │   └── losses.py
│   ├── train.py
│   ├── checkpoint.py
│   ├── evaluate.py
│   └── logging_utils.py
├── tests/
├── runs/                    # ignored; immutable run directories
├── artifacts/               # ignored; generated WAVs and figures
├── notes/
│   ├── MEMORY.md
│   ├── SCRATCHPAD.md
│   ├── DECISIONS.md
│   └── NEXT.md
├── tasks/
│   └── TASK-NN.md           # tracked; task prompts given to the assistant
└── scripts/
    └── inspect_checkpoint.py
```

## Data Pipeline

- Treat GTZAN genre labels as metadata only. This is waveform reconstruction, not genre classification.
- Convert audio to mono float waveform and resample deterministically to 24 kHz before MuQ inference.
- Retain source file identity, original sample rate, original length, resampled length, checksum, and split in manifests.
- Use short aligned random crops for training. Begin with a conservative crop length that fits comfortably in unified memory.
- Validation crops must be deterministic and fixed across runs.
- Verify waveform amplitude conventions and reject NaN, Inf, empty, clipped, or corrupt inputs.
- Keep the train, validation, and test split stable after it is committed.

## MuQ Feature Contract

Use only:

```python
with torch.inference_mode():
    output = muq(waveform_24khz, output_hidden_states=False)
    features = output.last_hidden_state
```

At startup, probe the installed model with synthetic and real clips and record:

- waveform shape and duration;
- feature shape;
- channel dimension;
- frames per second;
- effective waveform-samples-per-feature-frame ratio;
- dtype and device;
- whether results are finite and deterministic.

Do not assume the MuQ frame rate or channel dimension. Derive them from the installed checkpoint and assert them in the model adapter.

The decoder must receive the full temporal sequence. Never pool with a mean, standard deviation, or global average.

## Translator Design

The only trained component in the current pipeline is the translator: a learned mapping from MuQ `last_hidden_state` to EnCodec's continuous latent space. The decoder itself is frozen, pretrained EnCodec -- do not design, train, fine-tune, or add discriminators to a decoder.

Historical note: the original plan for this section called for training a small HiFi-GAN-style decoder with discriminators directly from MuQ features, outputting the waveform directly. That approach was superseded by the frozen-EnCodec pipeline above (see `notes/DECISIONS.md` for the evidence). The original design notes are preserved below as historical context, not current instructions:

1. A learned input projection from MuQ channels to decoder channels.
2. Temporal convolutional conditioning at the MuQ frame rate.
3. Upsampling stages whose product exactly matches the measured sample-to-frame ratio, or an explicitly documented alignment/cropping strategy if the ratio is not integral.
4. Multi-receptive-field residual blocks.
5. A waveform output head.
6. Multi-period and multi-scale discriminators only after the reconstruction path passes smoke tests.

(Historical) Start with reconstruction-only training using mel and multi-resolution STFT losses. Add adversarial and feature-matching losses only after the model can overfit a tiny subset and produce time-aligned audio.

(Historical) Do not copy speech-specific HiFi-GAN constants blindly. Sampling rate, FFT sizes, hop sizes, upsampling factors, segment lengths, and discriminator settings must be explicit in configuration and justified for 24 kHz music.

## Required Validation Gates

Complete these gates in order. Do not skip ahead.

### Gate 0: Environment doctor

- Import all dependencies.
- Report Apple Silicon architecture, available unified memory if discoverable, PyTorch version, MPS availability, and selected device.
- Instantiate MuQ and run a short inference.
- Instantiate the decoder and run a shape-only forward pass.

### Gate 1: Data integrity

- Index GTZAN.
- Detect unreadable/duplicate files where practical.
- Create deterministic manifests.
- Resample several files and verify length, finiteness, and amplitude.

### Gate 2: Feature contract

- Extract MuQ final-layer features for several durations.
- Confirm frame-rate consistency and alignment.
- Round-trip cached tensors and verify equality within dtype tolerance.

### Gate 3: Tiny-set overfit

- Train on one to four fixed clips.
- Confirm that reconstruction losses decrease substantially.
- Save reference, reconstruction, and difference audio.
- If this gate fails, do not launch a full run.

### Gate 4: Short pilot

- Train on a small deterministic subset.
- Validate checkpoint/resume equivalence.
- Run shuffled-conditioning and constant-conditioning controls.
- Confirm that real conditioning materially outperforms controls.

### Gate 5: GTZAN pilot

- Train on the committed GTZAN split.
- Save periodic fixed validation reconstructions.
- Report objective metrics and qualitative caveats.

## Checkpointing and Resumption

Every training checkpoint must contain enough state for a numerically faithful continuation:

- generator and discriminator weights;
- all optimizer and scheduler states;
- gradient-scaler state if used;
- epoch, global step, batch-in-epoch, and samples seen;
- Python, NumPy, and Torch RNG states;
- data-sampler state or equivalent deterministic cursor;
- complete resolved configuration;
- dataset manifest hash;
- `uv.lock` hash;
- Git commit and dirty-tree status;
- best validation metric and checkpoint lineage.

Write checkpoints atomically: save to a temporary path, flush, then rename. Keep `latest`, `best`, and periodic numbered checkpoints. Validate a checkpoint immediately after writing it by loading metadata and key tensors.

On startup, `--resume auto` should find the newest valid checkpoint. A corrupt partial checkpoint must be quarantined and the prior valid checkpoint used. Resuming must append to the existing run history rather than creating ambiguous duplicate runs.

Handle SIGINT and SIGTERM by saving an interruption checkpoint when it is safe to do so.

## Memory and Scratchpad Protocol

Long tasks must externalize state aggressively. Never rely on conversational context alone.

### `notes/MEMORY.md`

Maintain durable facts that should survive across sessions:

- confirmed tensor shapes and frame rates;
- working dependency versions;
- stable paths and commands;
- validated design choices;
- known platform limitations;
- benchmark results that have passed sanity checks.

Update only with verified information. Include dates and links to run IDs or commits.

### `notes/SCRATCHPAD.md`

Use as an append-only laboratory notebook for active work:

- hypotheses;
- commands executed;
- partial outputs;
- errors and stack traces;
- temporary calculations;
- experiments attempted;
- immediate interpretation.

Summarize useful conclusions into `MEMORY.md` or `DECISIONS.md`. Do not treat unverified scratchpad notes as facts.

### `notes/DECISIONS.md`

For each consequential decision, record:

- context;
- alternatives considered;
- decision;
- evidence;
- consequences;
- rollback plan.

### `notes/NEXT.md`

Before ending any session or long command, record:

- current gate;
- exact last successful command;
- run/checkpoint path;
- next command;
- blockers;
- expected verification criteria.

At the start of every session, read `CLAUDE.md`, `notes/MEMORY.md`, `notes/DECISIONS.md`, and `notes/NEXT.md` before making changes.

### `tasks/`

Tracked, append-only record of the task prompts given to the assistant, one file per task (e.g. `tasks/TASK-01.md`). Each file is the verbatim brief, not a summary. Do not edit a task file after the fact; if scope changes, record that in `notes/DECISIONS.md` and start a new task file for the next request. Read the relevant task file for full context and intent before resuming work on it.

## Resource Discipline on the M4 Max

- Prefer MPS when supported, but test correctness against CPU on tiny deterministic examples.
- Keep MuQ frozen and use inference mode to avoid autograd storage.
- Start with batch size 1, short crops, and no adversarial discriminators; increase only after measuring memory use.
- Use gradient accumulation rather than forcing a large physical batch.
- Avoid retaining MuQ activations after the final hidden state is detached.
- Limit DataLoader workers initially because macOS multiprocessing and unified memory can make excessive workers counterproductive.
- Monitor system memory pressure and swap. Treat sustained swap growth as a configuration failure.
- Cache features only after measuring actual bytes per audio second. Support `float32` and an explicitly validated lower-precision cache format.
- Make cache entries content-addressed by audio checksum, MuQ checkpoint, preprocessing configuration, representation name, and dtype.
- Cache writes must be atomic and resumable.

## Evaluation and Controls

Minimum outputs for each validation item:

- original WAV;
- reconstructed WAV;
- log-mel comparison;
- multi-resolution STFT loss;
- waveform and feature shapes;
- run ID and checkpoint step.

Required controls:

1. Correct MuQ conditioning.
2. Conditioning shuffled across examples.
3. Conditioning shuffled in time.
4. Constant or zero conditioning.

If the controls sound similar to the correctly conditioned model, assume conditioning collapse or leakage until disproven.

Use objective metrics as diagnostics, not as proof of perceptual fidelity. Do not claim successful inversion solely from a low training loss. The initial success criterion is conservative: reconstructions are recognizably tied to the source and outperform the controls.

Treat the variance ratio (predicted-to-oracle per-channel latent variance) as the primary latent-collapse diagnostic. Across this project's DAC and EnCodec experiments, cosine similarity and MSE have repeatedly saturated near their ceiling values well before decoded-audio quality did, making them unreliable as the primary signal for whether collapse is actually resolved. Retain cosine similarity and MSE as secondary/supporting metrics, not as the primary indicator.

## Coding Standards

- Use typed Python and small testable modules.
- Prefer dataclasses or validated configuration objects.
- Add shape annotations in docstrings for every tensor-facing function.
- Assert shapes at module boundaries and include file/run identifiers in errors.
- No hidden downloads during training. Dataset and model acquisition should be explicit preparation steps.
- No notebooks as the canonical implementation. Notebooks may inspect results but must call package code.
- Unit-test resampling lengths, crop alignment, feature caching, upsampling geometry, checkpoint round trips, and resume behavior.
- Keep platform-specific code isolated.
- Run formatting, linting, and tests before committing.

## Stop Conditions

Stop and diagnose rather than scaling when any of the following occurs:

- MuQ outputs contain NaN or Inf;
- feature frame rate changes unexpectedly;
- generator output length does not align with the target;
- tiny-set overfit fails;
- resumed training diverges from an uninterrupted control beyond expected nondeterminism;
- the generator ignores conditioning;
- memory pressure causes sustained swapping;
- generated samples improve aesthetically while source correspondence degrades.

## Definition of Done for the Initial Experiment

The proof of concept is complete only when:

1. The environment can be recreated with `uv sync --frozen` from `pyproject.toml` and `uv.lock`.
2. GTZAN manifests and preprocessing are deterministic.
3. MuQ final-layer feature geometry is measured and documented.
4. A small learned translator trains to predict frozen EnCodec's continuous latent from that representation, decodable by EnCodec's own frozen decoder.
5. Training can stop and resume from an atomic checkpoint with state restored.
6. Fixed validation examples produce source-related reconstructions that outperform shuffled and constant-conditioning controls.
7. Results, failures, resource use, and next-step recommendations are recorded in the notes and run manifest.

## Fundamental instructions

### Done means done

Not half done. Not done except for the part you decided to skip. And not a report about how it will be done.

Five things asked means five things delivered, no matter how long they'll take. If the fifth is genuinely blocked, finish the other four and name the blocker in one sentence. The specific blocker. Not "this needs more investigation."

### Act. Don't ask

Reversible and cheap? Do it, then tell me. Research, data pulls, analysis,
drafts, refactors inside the scope I gave you, testing an API. A question costs me more than a re-run costs you.

Ask first only for: anything reaching an audience, anything we cannot undo,
anything expensive.

Something is broken? Fix it. Reporting an issue you could have fixed turns your work into my to-do list.

### A question is a question

When I ask a question, answer it. Do not implement it.

"Should we use X?" is not "migrate everything to X." "What would it take to add Y?" is not "add Y."

When in doubt, assume it's a question. Answer first. Act when I say go.
