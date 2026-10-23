"""TASK-21: check a continuation run's saved state against its parent checkpoint.

    uv run python scripts/task21_check_run.py PARENT_LATEST.pt RUN_DIR [--expect-final]

Checks (exit 1 on failure):
- Adam's step count equals the global step in latest.pt (optimizer state carried, never reset):
  parent Adam step == parent step, and continuation Adam step == continuation step;
- the continuation recorded its parent (path, sha256, step) and the continuation check passed;
- the incumbent best is the parent's best unless a lower primary-val mel L1 was found later;
- lineage has one entry per job segment, and resumes are logged as events;
- recovery checkpoints are bounded by the policy (`keep_recovery`);
- with --expect-final: final.pt exists and the runtime budget was reached.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from muq_inversion.device import load_portable


def adam_step(ck: dict) -> int:
    steps = {int(float(s["step"])) for s in ck["optimizer_state_dict"]["state"].values()}
    if len(steps) != 1:
        raise SystemExit(f"inconsistent Adam step counts {steps}")
    return steps.pop()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("parent_latest", type=Path)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expect-final", action="store_true")
    args = parser.parse_args()
    parent = load_portable(args.parent_latest)
    ck = load_portable(args.run_dir / "checkpoints" / "latest.pt")
    st = ck["train_state"]
    checks = {
        "parent Adam step == parent global step": adam_step(parent) == parent["train_state"]["step"],
        "continuation Adam step == global step": adam_step(ck) == st["step"],
        "continuation started at the parent's last step": st["parent"]["latest"]["step"] == parent["train_state"]["step"],
        "continuation check passed": any(e["event"] == "continuation_check"
                                         and abs(e["val_mel_l1"] - e["parent_val_mel_l1"]) <= 1e-3 for e in st["events"]),
        "best is parent's best or a later improvement": (
            st["best"] == parent["train_state"]["best"]
            or (st["best"]["step"] > parent["train_state"]["step"] and st["best"]["val_mel_l1"] < parent["train_state"]["best"]["val_mel_l1"])),
        "early stopping disabled": st.get("early_stopping") is None,
        "resume events logged for every extra segment": (
            sum(e["event"] == "resume" for e in st["events"]) == len([x for x in st["lineage"] if "run" not in x]) - 1),
        "recovery checkpoints bounded": len(list((args.run_dir / "checkpoints").glob("recovery_*.pt")))
                                        <= ck["resolved_config"]["checkpoint_policy"]["keep_recovery"],
    }
    if args.expect_final:
        checks["final.pt exists"] = (args.run_dir / "checkpoints" / "final.pt").exists()
        checks["runtime budget reached"] = bool(st.get("budget_reached"))
    print(f"run {args.run_dir.name}: step {st['step']}, best step {st['best']['step']} "
          f"({st['best']['val_mel_l1']:.4f}), runtime {st.get('runtime_seconds', 0) / 3600:.4f} h, "
          f"segments {len([x for x in st['lineage'] if 'run' not in x])}, events {[e['event'] for e in st['events']]}")
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    ok = all(checks.values())
    print("STATE CHECK:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
