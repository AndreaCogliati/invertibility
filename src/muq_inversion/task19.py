"""TASK-19: N=128 data x recipe experiment on the TASK-18 pipeline.

Pure pieces: Sturm (2013) fault groups and artist labels, deterministic extension of the TASK-18
track selection, the improved-recipe learning-rate schedule, and the pre-registered comparison of
the four arms (A32, A128, B32, B128).
"""

from __future__ import annotations

import collections
import math
import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import yaml

from muq_inversion.task18 import GENRES, cluster_bootstrap_delta

ARMS = ("A32", "A128", "B32", "B128")


# --- GTZAN faults (Sturm 2013, Table 2) ----------------------------------------------------

def _tid(genre: str, n: int | str) -> str:
    return f"{genre}.{int(n):05d}"


def _ranges(spec: str) -> set[int]:
    out: set[int] = set()
    for part in str(spec).split(","):
        if "-" in part:
            a, b = part.split("-")
            out |= set(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def load_faults(path: Path) -> dict:
    """Returns {"groups": [(kind, [track_id, ...])], "group_of": {track_id: frozenset of all tracks
    repeated with it (transitively)}, "distorted": {track_id: description}, "artist": {track_id: name}}."""
    f = yaml.safe_load(Path(path).read_text())
    groups = []
    for kind in ("exact", "recording", "version"):
        for g, gs in f[kind].items():
            groups += [(kind, [_tid(g, n) for n in grp]) for grp in gs]
    groups += [("cross_genre", [_tid(*x.split(".")) for x in grp]) for grp in f["cross_genre"]]
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            x = parent[x]
        return x

    for _, members in groups:
        for m in members[1:]:
            parent[find(m)] = find(members[0])
    comps: dict[str, set] = collections.defaultdict(set)
    for x in list(parent):
        comps[find(x)].add(x)
    return {
        "source": f["source"], "groups": groups,
        "group_of": {x: frozenset(comps[find(x)]) for x in parent},
        "distorted": {_tid(g, n): d for g, dd in f["distortions"].items() for n, d in dd.items()},
        "artist": {_tid(g, n): a for g, d in f["artists"].items() for a, s in d.items() for n in _ranges(s)},
    }


def fault_report(tracks: Sequence[dict], faults: dict) -> dict:
    """tracks: rows with track_id and split. Lists every selected track that is in a repetition
    group or distorted, and every repetition group whose selected members span >1 split (the
    stop condition). Also counts held-out tracks whose (known) artist appears in training."""
    split = {t["track_id"]: t["split"] for t in tracks}
    members, crossing = [], []
    seen_groups = set()
    for t in tracks:
        grp = faults["group_of"].get(t["track_id"])
        if grp:
            partners = sorted(grp - {t["track_id"]})
            members.append({"track_id": t["track_id"], "split": t["split"], "repeats": partners,
                            "repeats_selected": {p: split[p] for p in partners if p in split}})
            if grp not in seen_groups:
                seen_groups.add(grp)
                splits = {split[x] for x in grp if x in split}
                if len(splits) > 1:
                    crossing.append({"group": sorted(grp), "splits": {x: split[x] for x in grp if x in split}})
    train_artists = {faults["artist"].get(t["track_id"]) for t in tracks if t["split"] == "train"} - {None}
    heldout = [t for t in tracks if t["split"] != "train"]
    artist_seen = {t["track_id"]: faults["artist"].get(t["track_id"]) for t in heldout
                   if faults["artist"].get(t["track_id"]) in train_artists}
    return {
        "source": faults["source"], "n_tracks": len(tracks), "repetition_members": members,
        "repetition_groups_crossing_splits": crossing,
        "distorted": {t["track_id"]: faults["distorted"][t["track_id"]] for t in tracks if t["track_id"] in faults["distorted"]},
        "heldout_artist_seen_in_training": artist_seen,
        "heldout_artist_unknown": sorted(t["track_id"] for t in heldout if t["track_id"] not in faults["artist"]),
    }


# --- track selection ----------------------------------------------------------------------

def extend_selection(source: Sequence[dict], existing: Sequence[dict], targets: dict[str, int], faults: dict,
                     seed: int) -> list[dict]:
    """New tracks per split so that the split reaches `targets[split]` tracks in total (existing +
    new), genre-balanced: each genre ends with floor(target/10) or one more, extra genres chosen by
    a seeded RNG. Candidates: same committed split, not corrupted, unique checksum, >= 29 s, not in
    any Sturm repetition group, not distorted, not already selected. Returns only the new rows."""
    checksum_counts = collections.Counter(r["checksum"] for r in source)
    taken = {r["track_id"] for r in existing}
    new = []
    for split, target in targets.items():
        have = collections.Counter(r["genre"] for r in existing if r["split"] == split)
        rng = random.Random(f"task19-{seed}-{split}")
        base, extra = divmod(target, len(GENRES))
        extra_genres = set(rng.sample(sorted(GENRES), extra))
        for g in GENRES:
            want = base + (g in extra_genres) - have[g]
            if want < 0:
                raise ValueError(f"{split}/{g}: already {have[g]} tracks, target {base + (g in extra_genres)}")
            pool = sorted((r for r in source if r["split"] == split and r["genre"] == g and r["corrupted"] != "True"
                           and checksum_counts[r["checksum"]] == 1 and float(r["duration_seconds"]) >= 29.0
                           and r["track_id"] not in faults["group_of"] and r["track_id"] not in faults["distorted"]
                           and r["track_id"] not in taken), key=lambda r: r["track_id"])
            if len(pool) < want:
                raise ValueError(f"{split}/{g}: need {want} new eligible tracks, have {len(pool)}")
            new += [{**r, "split": split} for r in rng.sample(pool, want)]
    return new


# --- improved recipe (B) ------------------------------------------------------------------

def lr_multiplier(step: int, schedule: dict, total_steps: int) -> float:
    """Multiplier on the base LR for optimizer step `step` (1-based). `constant` -> 1.
    `warmup_cosine`: linear from 1/warmup to 1 over `warmup_steps`, then cosine from 1 down to
    `min_lr_ratio` at `total_steps`."""
    kind = schedule.get("type", "constant")
    if kind == "constant":
        return 1.0
    if kind != "warmup_cosine":
        raise ValueError(f"unknown lr schedule {kind!r}")
    w = schedule["warmup_steps"]
    if step <= w:
        return step / w
    progress = min(1.0, (step - w) / max(1, total_steps - w))
    lo = schedule["min_lr_ratio"]
    return lo + (1 - lo) * 0.5 * (1 + math.cos(math.pi * progress))


# --- arm comparison and pre-registered interpretation --------------------------------------

def paired_arm_delta(per_example: Sequence[dict], arm_a: str, arm_b: str, eval_sets: Sequence[str], metric: str = "mel_l1",
                     seed: int = 0, num_bootstrap: int = 2000) -> dict:
    """Improvement of arm_b over arm_a (positive = arm_b better, lower `metric`) on the same crops,
    best checkpoint, correct conditioning. per_example rows need arm, checkpoint, eval_set, crop_id,
    track_id, condition, `metric`."""
    def table(arm):
        return {r["crop_id"]: r for r in per_example if r["arm"] == arm and r["checkpoint"] == "best"
                and r["condition"] == "correct" and r["eval_set"] in eval_sets}
    a, b = table(arm_a), table(arm_b)
    crops = sorted(a.keys() & b.keys())
    if not crops:
        raise ValueError(f"no shared crops for {arm_a} vs {arm_b} on {eval_sets}")
    va = np.array([float(a[c][metric]) for c in crops])
    vb = np.array([float(b[c][metric]) for c in crops])
    d = cluster_bootstrap_delta(list(vb), list(va), [a[c]["track_id"] for c in crops], lower_is_better=True,
                                seed=seed, num_bootstrap=num_bootstrap)
    return {**d, "mean_a": float(va.mean()), "mean_b": float(vb.mean()),
            "relative_improvement": float((va.mean() - vb.mean()) / va.mean())}


def judge(delta: dict, rule: dict) -> str:
    """'much_better' | 'better' | 'similar' | 'worse' for one paired comparison (pre-registered)."""
    rel, lo, hi, frac = delta["relative_improvement"], delta["ci_low"], delta["ci_high"], delta["proportion_tracks_positive"]
    if rel >= rule["much_better_rel"] and lo > 0 and frac >= rule["min_track_fraction"]:
        return "much_better"
    if lo > 0:
        return "better"
    if hi < 0:
        return "worse"
    return "similar"


def classify_arms(primary: dict[str, dict], training: dict[str, dict], rule: dict) -> dict:
    """primary[arm]: paired delta vs A32 on primary held-out crops (val+test); training[arm]: the same
    on memorized core crops. Maps to the pre-registered table in configs/task19.yaml."""
    data, recipe, both = (judge(primary[a], rule) for a in ("A128", "B32", "B128"))
    if data == "much_better" and recipe == "much_better":
        code = "BOTH_MATTER"
    elif data == "much_better":
        code = "DATA_LIMITED"
    elif recipe == "much_better" and data in ("similar", "worse", "better"):
        code = "OPTIMIZATION_LIMITED"
    elif both == "much_better":
        code = "BOTH_NEEDED_JOINTLY"
    else:
        code = "NEITHER_CAPACITY_OR_OBJECTIVE_LIMIT"
    gaps = [a for a in ("A128", "B32", "B128")
            if judge(training[a], rule) == "much_better" and judge(primary[a], rule) in ("similar", "worse")]
    return {"code": code, "judgements": {"A128_vs_A32": data, "B32_vs_A32": recipe, "B128_vs_A32": both},
            "generalization_gap_arms": gaps}
