"""TASK-15: explicit, revision-pinned download of the official NVIDIA BigVGAN-v2 code and
checkpoint from Hugging Face (`nvidia/bigvgan_v2_24khz_100band_256x`).

The HF model repo ships NVIDIA's own BigVGAN implementation (`bigvgan.py`, `meldataset.py`,
`activations.py`, `alias_free_activation/`, `env.py`, `utils.py`) next to `config.json` and the
generator weights; its README's supported loading path is
`bigvgan.BigVGAN.from_pretrained(<repo or local dir>, use_cuda_kernel=False)` plus
`meldataset.get_mel_spectrogram(wav, model.h)`. This script fetches exactly those files at a
pinned revision into `third_party/bigvgan_v2_24khz_100band_256x/` (git-ignored) so training never
downloads anything (CLAUDE.md: "No hidden downloads during training"). The 1.4 GB discriminator/
optimizer states and the alternative `*_3msteps.pt` checkpoints are not needed for inference and
are skipped.

    uv run python scripts/fetch_bigvgan.py
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "nvidia/bigvgan_v2_24khz_100band_256x"
REVISION = "c329ede9e9bbc100ddf5c91e2330a61921262370"  # repo head on 2026-10-16 (last modified 2024-09-05)
REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_DIR = REPO_ROOT / "third_party" / "bigvgan_v2_24khz_100band_256x"
ALLOW_PATTERNS = [
    "*.py",
    "alias_free_activation/**",
    "config.json",
    "bigvgan_generator.pt",
    "LICENSE",
    "README.md",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=REPO_ID, revision=REVISION, local_dir=LOCAL_DIR, allow_patterns=ALLOW_PATTERNS
    )
    for name in ("config.json", "bigvgan.py", "meldataset.py", "bigvgan_generator.pt"):
        path = LOCAL_DIR / name
        print(f"{sha256(path)}  {path.relative_to(REPO_ROOT)}  ({path.stat().st_size:,} bytes)")
    (LOCAL_DIR / "REVISION").write_text(f"{REPO_ID}@{REVISION}\n")


if __name__ == "__main__":
    main()
