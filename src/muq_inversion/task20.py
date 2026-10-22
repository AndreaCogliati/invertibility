"""TASK-20: N=512 data-scaling run (recipe A + validation-based early stopping).

Pure pieces only: capped genre-balanced extension of the TASK-19 training set and the early-stopping
rule. Everything else reuses TASK-18/19 code.
"""

from __future__ import annotations

import collections
import random
from collections.abc import Sequence

from muq_inversion.task18 import GENRES


def balanced_genre_totals(target: int, have: dict[str, int], caps: dict[str, int], rng: random.Random) -> dict[str, int]:
    """Per-genre totals summing to min(target, sum(caps)), as equal as the caps allow: start every
    genre at max(have, floor(target / G)) capped at its cap, then hand out the remainder one track
    at a time to genres with spare capacity, lowest total first, ties broken by a seeded shuffle.
    Never below `have` (existing tracks are kept)."""
    total_cap = sum(caps.values())
    target = min(target, total_cap)
    base = target // len(GENRES)
    totals = {g: min(caps[g], max(have.get(g, 0), base)) for g in GENRES}
    order = sorted(GENRES)
    rng.shuffle(order)
    while sum(totals.values()) < target:
        open_genres = [g for g in order if totals[g] < caps[g]]
        g = min(open_genres, key=lambda x: (totals[x], order.index(x)))
        totals[g] += 1
    if sum(totals.values()) != target:
        raise ValueError(f"cannot reach {target} tracks: totals {totals}")
    return totals


def extend_training_tracks(source: Sequence[dict], existing: Sequence[dict], target: int, faults: dict,
                           seed: int) -> tuple[list[dict], dict[str, int]]:
    """New training tracks (committed train split only, so held-out tracks are never candidates)
    bringing the training set to `target` (or the largest eligible size). Eligibility as in TASK-19:
    not corrupted, unique checksum, >= 29 s, not in any Sturm (2013) repetition group, not distorted,
    not already selected (in any split). Returns (new rows, per-genre totals)."""
    checksum_counts = collections.Counter(r["checksum"] for r in source)
    taken = {r["track_id"] for r in existing}
    have = collections.Counter(r["genre"] for r in existing if r["split"] == "train")
    eligible = {g: sorted((r for r in source if r["split"] == "train" and r["genre"] == g and r["corrupted"] != "True"
                           and checksum_counts[r["checksum"]] == 1 and float(r["duration_seconds"]) >= 29.0
                           and r["track_id"] not in faults["group_of"] and r["track_id"] not in faults["distorted"]
                           and r["track_id"] not in taken), key=lambda r: r["track_id"]) for g in GENRES}
    rng = random.Random(f"task20-{seed}-train")
    totals = balanced_genre_totals(target, have, {g: have[g] + len(eligible[g]) for g in GENRES}, rng)
    new = []
    for g in GENRES:
        new += [{**r, "split": "train"} for r in rng.sample(eligible[g], totals[g] - have[g])]
    return new, totals


def early_stopping_update(es: dict, val_metric: float, best_before: float, cfg: dict) -> dict:
    """One validation check. `es` = {"checks_since_improvement", "stopped", "reason", "stopped_at_step"}.
    Improvement = val_metric < best_before - min_delta (lower is better). Stops once
    `checks_since_improvement` reaches `patience`. Returns the updated state (new dict)."""
    es = dict(es)
    if val_metric < best_before - cfg["min_delta"]:
        es["checks_since_improvement"] = 0
    else:
        es["checks_since_improvement"] += 1
    if es["checks_since_improvement"] >= cfg["patience"]:
        es["stopped"], es["reason"] = True, f"patience: {cfg['patience']} validation checks without improvement"
    return es
