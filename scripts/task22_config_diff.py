"""TASK-22: every resolved-config change from the baseline, as a Markdown table.

    uv run python scripts/task22_config_diff.py [--baseline task21|task20]

Resolved = the config after `base_config` inheritance and the arm override, exactly as the training
script builds it (TASK-21 arm N512ext / TASK-20 arm N512 vs TASK-22 arm L1).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from muq_inversion.task22 import config_diff

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from train_task18_heldout import load_config

BASELINES = {"task21": ("configs/task21.yaml", "N512ext"), "task20": ("configs/task20.yaml", "N512")}


def resolved(path: str, arm: str) -> dict:
    cfg = load_config(REPO_ROOT / path)
    return {**cfg, **cfg["arms"][arm], "arm": arm}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", choices=sorted(BASELINES), default="task21")
    args = parser.parse_args()
    base = resolved(*BASELINES[args.baseline])
    new = resolved("configs/task22.yaml", "L1")
    print(f"| key | {args.baseline} ({BASELINES[args.baseline][1]}) | task22 (L1) |\n| --- | --- | --- |")
    for d in config_diff(base, new, ignore=("arms",)):
        print(f"| `{d['key']}` | `{json.dumps(d['baseline'])}` | `{json.dumps(d['task22'])}` |")
    print("\n(`arms` differs by construction: it lists each task's arms; the selected arm's keys are merged above.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
