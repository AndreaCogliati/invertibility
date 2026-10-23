"""TASK-21: extended N512 continuation (runtime budget, no early stopping).

Pure helpers: step-dependent validation/logging cadence, the cumulative runtime budget, and the
bounded checkpoint-retention policy. The training loop itself is the shared TASK-18 script.
"""

from __future__ import annotations

import time
from pathlib import Path


def due(step: int, schedule: list[dict]) -> bool:
    """`schedule`: ordered [{"until": S, "every": k}, ..., {"every": k}] (last entry has no `until`).
    True when `step` is a multiple of the interval in force at that step."""
    for seg in schedule:
        if "until" not in seg or step <= seg["until"]:
            return step % seg["every"] == 0
    return False


class RuntimeBudget:
    """Cumulative job runtime across resumable segments. `used_before` is restored from the checkpoint;
    this segment's share is wall time since `segment_start` (process start, so start-up counts)."""

    def __init__(self, budget_hours: float | None, used_before: float, segment_start: float | None = None):
        self.budget = None if budget_hours is None else budget_hours * 3600.0
        self.used_before = used_before
        self.segment_start = time.time() if segment_start is None else segment_start

    def used(self, now: float | None = None) -> float:
        return self.used_before + ((time.time() if now is None else now) - self.segment_start)

    def exhausted(self, now: float | None = None) -> bool:
        return self.budget is not None and self.used(now) >= self.budget


def recovery_to_delete(existing: list[Path], keep: int) -> list[Path]:
    """Rolling recovery set: keep the `keep` most recent `recovery_<step>.pt` files (by step)."""
    def step_of(p: Path) -> int:
        return int(p.stem.split("_")[-1])
    ordered = sorted(existing, key=step_of)
    return ordered[:-keep] if keep > 0 else ordered
