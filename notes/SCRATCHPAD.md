# SCRATCHPAD

Append-only lab notebook. Unverified until summarized into MEMORY.md/DECISIONS.md.

## 2026-09-04 -- Initial scaffold, Gate 0 + Gate 1

- Started from an empty directory containing only `CLAUDE.md`.
- `git init`, `uv init --package --name muq_inversion .`, `uv python pin 3.12`.
- `uv add torch torchaudio librosa soundfile numpy scipy pyyaml tqdm tensorboard muq` then
  `uv add --dev pytest ruff mypy`.
- Confirmed no local GTZAN copy exists (`find ~ -iname '*gtzan*'` empty). Asked the user how to
  acquire it; they chose the Hugging Face route. Searched `huggingface.co/api/datasets?search=gtzan`,
  found `marsyas/gtzan` mirrors the canonical `opihi.cs.uvic.ca/sound/genres.tar.gz` archive
  (confirmed by reading its `gtzan.py` loading script, which also revealed the known corrupted
  file `jazz.00054.wav`).
- Read `OpenMuQ/MuQ-large-msd-iter`'s model card: install via `pip install muq`, use
  `from muq import MuQ; MuQ.from_pretrained(...)`. `config.json` showed `encoder_dim: 1024`,
  `label_rate: 25` (Hz) -- used only as a sanity-check prior, not hardcoded anywhere in code.

- First `doctor` run failed on MuQ forward pass:
  `AttributeError: 'EasyDict' object has no attribute '_attn_implementation'` inside
  transformers 5.16.1's `create_bidirectional_mask`. Traced to MuQ building its Wav2Vec2Conformer
  config as a plain `EasyDict`, not a real `PretrainedConfig`. Tried pinning
  `transformers==4.46.3` -> conflicted with `huggingface_hub>=1.30.0` already resolved.
  Tried `transformers==4.55.0` + `huggingface_hub>=0.34.0,<1.0` together -> resolved cleanly.
  Re-ran doctor: passed. Recorded as [[../notes/DECISIONS.md#muq-transformers-compat]].

- `doctor` (Gate 0) on `configs/smoke.yaml`, device=mps:
  - `muq.feature_shape: (1, 50, 1024)`, `frames_per_second: 25.0`, `samples_per_frame: 960.0`,
    `is_finite: True`, `deterministic_shape_repeat: True`.
  - `MiniGenerator` shape-only pass: input `(1, 50, 1024)` -> output `(1, 48000)`, matching
    `expected_output_samples`. PASSED.

- First `prepare-data` run failed: `sf.info()` itself raises `LibsndfileError: bad data offset`
  on `jazz.00054.wav` -- not just decode-time failure, so `index_gtzan` needed a try/except
  around the `sf.info` call, not just a post-hoc corrupted-file check. Fixed, made it also warn
  (not silently skip) if an *unexpected* file is unreadable.

- Second `prepare-data` run: indexed 1000/1000 files, found 14 duplicate-checksum pairs (e.g.
  `metal.00058.wav` == `rock.00016.wav` byte-for-byte despite different genre labels -- a known
  GTZAN defect, see Sturm 2013). Initial `assign_splits` hashed on
  `f"{genre}/{track_id}"`, which does NOT guarantee cross-genre duplicates land in the same
  split (confirmed by inspection: a duplicate pair spanning two genres would hash differently).
  Fixed by hashing purely on `checksum`. Verified post-fix with a standalone check: zero
  checksums span multiple splits.
  Deleted the manifest from the pre-fix run before re-running, since CLAUDE.md rule 8 (never
  overwrite a previous run's manifest) applies to *committed* prior runs, not this session's own
  just-created, not-yet-committed test output.

- `ruff check src/` flagged blind `except Exception` (BLE001) in `data.py` and
  `logging_utils.py`, plus an unnecessary `.encode("utf-8")`. Narrowed the except clauses to
  specific exception types (`sf.LibsndfileError`/`RuntimeError`/`OSError`;
  `subprocess.CalledProcessError`/`TimeoutExpired`/`FileNotFoundError`;
  `importlib.metadata.PackageNotFoundError`). Noticed `_package_version` was dead code -- wired
  it into `capture_environment_manifest` so package versions (torch, torchaudio, muq,
  transformers, librosa, soundfile, huggingface_hub) actually get recorded, per CLAUDE.md rule 2.
  `ruff check src/` now clean.

- 13 unit tests added (`tests/test_audio.py`, `tests/test_data.py`) covering resample length
  tolerance, waveform validation rejections, deterministic/duplicate-safe splitting, and
  manifest round-trip. All pass (`uv run pytest -q`).

- Did not commit to git yet -- scope was explicitly Gate 0 + Gate 1 only; leaving the commit
  decision to the user per instructions to only commit when asked.
