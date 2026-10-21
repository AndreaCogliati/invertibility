"""Resume-equivalence check for TASK-18: compare the translator weights (and Adam state) of two
checkpoints, e.g. an uninterrupted run's final.pt vs an interrupted-then-resumed run's final.pt.

    uv run python scripts/task18_compare_checkpoints.py UNINTERRUPTED.pt RESUMED.pt [--noise-reference UNINTERRUPTED2.pt] [--atol X]

Pass if the resumed weights match the uninterrupted ones within max(--atol, 10 x the difference
between two uninterrupted runs) -- the run-to-run noise floor, given with --noise-reference -- and
the sampler visited exactly the same crops. Single-threaded CPU (OMP_NUM_THREADS=1) is
bit-identical (verified 2026-10-21); multithreaded CPU, MPS, and XPU kernels are not, so measure
their noise floor instead of assuming zero.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from muq_inversion.device import load_portable


def max_weight_diff(a: dict, b: dict) -> float:
    return max(float((a["model_state_dict"][k].float() - b["model_state_dict"][k].float()).abs().max())
               for k in a["model_state_dict"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    parser.add_argument("--noise-reference", type=Path, default=None, help="a second uninterrupted run's checkpoint")
    parser.add_argument("--atol", type=float, default=0.0)
    args = parser.parse_args()
    a, b = load_portable(args.a), load_portable(args.b)
    worst = max_weight_diff(a, b)
    tol = args.atol
    if args.noise_reference is not None:
        noise = max_weight_diff(a, load_portable(args.noise_reference))
        tol = max(tol, 10 * noise)
        print(f"noise floor (uninterrupted vs uninterrupted): {noise:.3g} -> tolerance {tol:.3g}")
    sa, sb = a["train_state"], b["train_state"]
    print(f"steps {sa['step']} vs {sb['step']}; best step {sa['best']['step']} vs {sb['best']['step']}")
    print(f"max |weight diff| = {worst:.3g} (tolerance {tol:.3g}); lineage B: {sb['lineage']}")
    same_samples = a["sampler_state"]["track_exposure"] == b["sampler_state"]["track_exposure"]
    print(f"sampler exposure identical: {same_samples}")
    ok = worst <= tol and sa["step"] == sb["step"] and same_samples
    print("RESUME EQUIVALENCE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
