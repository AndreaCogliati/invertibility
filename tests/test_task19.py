"""TASK-19: fault list, track/crop extension, LR schedule, arm comparison and outcome table."""

from __future__ import annotations

import collections
import csv
import math
import re
from pathlib import Path

import pytest
import yaml

from muq_inversion import task19 as t19

REPO_ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((REPO_ROOT / "configs/task19.yaml").read_text())


def _csv(path):
    with open(REPO_ROOT / path, newline="") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def faults():
    return t19.load_faults(REPO_ROOT / CFG["fault_list"])


@pytest.fixture(scope="module")
def tracks():
    return _csv(CFG["track_manifest"])


@pytest.fixture(scope="module")
def crops():
    return _csv(CFG["crop_manifest"])


def test_fault_list_transcription(faults):
    # Spot checks against Sturm (2013) Table 2.
    g = faults["group_of"]
    assert g["metal.00058"] == {"metal.00058", "rock.00016"}
    assert g["disco.00014"] == {"disco.00014", "metal.00085"}
    assert {"pop.00015", "pop.00021", "pop.00022", "pop.00037"} <= g["pop.00015"]
    assert g["disco.00050"] == {"disco.00050", "disco.00051", "disco.00070"}
    assert faults["distorted"]["reggae.00086"] == "last 25 s useless"
    assert faults["artist"]["blues.00029"] == "Kelly Joe Phelps"


def test_every_checksum_duplicate_is_in_a_fault_group(faults):
    src = _csv(CFG["source_manifest"])
    dups = collections.defaultdict(list)
    for r in src:
        dups[r["checksum"]].append(r["track_id"])
    for ids in (v for v in dups.values() if len(v) > 1):
        assert all(i in faults["group_of"] for i in ids), ids


def test_selection_counts_balance_and_task18_superset(tracks):
    c = collections.Counter((t["split"], t["tier"]) for t in tracks)
    assert c == {("train", "train"): 128, ("val", "primary"): 8, ("val", "secondary"): 24,
                 ("test", "primary"): 8, ("test", "secondary"): 24}
    for split, lo, hi in (("train", 12, 13), ("val", 3, 4), ("test", 3, 4)):
        g = collections.Counter(t["genre"] for t in tracks if t["split"] == split)
        assert set(g) == set(t19.GENRES) and min(g.values()) >= lo and max(g.values()) <= hi
    t18 = {(t["track_id"], t["split"]) for t in _csv(CFG["task18_track_manifest"])}
    assert t18 == {(t["track_id"], t["split"]) for t in tracks if t["core"] == "1"}


def test_no_repetition_crosses_splits_and_new_tracks_are_clean(tracks, faults):
    rep = t19.fault_report(tracks, faults)
    assert rep["repetition_groups_crossing_splits"] == []
    for t in tracks:
        if t["core"] == "0":
            assert t["track_id"] not in faults["group_of"] and t["track_id"] not in faults["distorted"]


def test_task18_crops_reproduced_and_primary_heldout_identical(crops):
    t18 = {r["crop_id"]: r for r in _csv(CFG["task18_crop_manifest"])}
    mine = {r["crop_id"]: r for r in crops}
    for cid, r in t18.items():
        for k in ("eval_set", "split", "track_id", "path", "checksum_sha256", "crop_start_sample", "crop_num_samples"):
            assert mine[cid][k] == r[k], (cid, k)
    by = collections.Counter(r["eval_set"] or "pool" for r in crops)
    assert by == {"pool": 3072, "memorized": 64, "train_unseen": 64, "val": 32, "test": 32,
                  "memorized_new": 192, "train_unseen_new": 192, "val2": 96, "test2": 96}
    listen = [r for r in crops if r["listening"] == "1"]
    assert len(listen) == 16 and {r["eval_set"] for r in listen} == {"val", "test"}


def test_lr_schedule():
    sched = CFG["arms"]["B128"]["lr_schedule"]
    assert t19.lr_multiplier(5, {"type": "constant"}, 20000) == 1.0
    assert t19.lr_multiplier(1, sched, 20000) == pytest.approx(1e-3)
    assert t19.lr_multiplier(1000, sched, 20000) == pytest.approx(1.0)
    assert t19.lr_multiplier(10500, sched, 20000) == pytest.approx(0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * 0.5)))
    assert t19.lr_multiplier(20000, sched, 20000) == pytest.approx(0.01)
    assert CFG["arms"]["A32"]["lr_schedule"] == CFG["arms"]["A128"]["lr_schedule"] == {"type": "constant"}


def _delta(rel, lo, hi, frac):
    return {"relative_improvement": rel, "ci_low": lo, "ci_high": hi, "proportion_tracks_positive": frac}


MB, SIM = _delta(0.2, 0.05, 0.2, 0.9), _delta(0.01, -0.02, 0.03, 0.5)


@pytest.mark.parametrize(("a128", "b32", "b128", "code"), [
    (MB, SIM, MB, "DATA_LIMITED"),
    (SIM, MB, MB, "OPTIMIZATION_LIMITED"),
    (MB, MB, MB, "BOTH_MATTER"),
    (SIM, SIM, MB, "BOTH_NEEDED_JOINTLY"),
    (SIM, SIM, SIM, "NEITHER_CAPACITY_OR_OBJECTIVE_LIMIT"),
])
def test_interpretation_table(a128, b32, b128, code):
    out = t19.classify_arms({"A128": a128, "B32": b32, "B128": b128}, {"A128": SIM, "B32": SIM, "B128": SIM}, CFG["arm_comparison"])
    assert out["code"] == code and out["generalization_gap_arms"] == []


def test_much_better_needs_size_ci_and_tracks():
    rule = CFG["arm_comparison"]
    assert t19.judge(_delta(0.09, 0.01, 0.1, 1.0), rule) == "better"        # < 10%
    assert t19.judge(_delta(0.20, -0.01, 0.3, 1.0), rule) == "similar"      # CI includes 0
    assert t19.judge(_delta(0.20, 0.01, 0.3, 0.70), rule) == "better"       # < 75% of tracks
    assert t19.judge(_delta(-0.2, -0.3, -0.1, 0.0), rule) == "worse"


def test_generalization_gap_flag():
    out = t19.classify_arms({"A128": SIM, "B32": SIM, "B128": SIM}, {"A128": MB, "B32": SIM, "B128": SIM}, CFG["arm_comparison"])
    assert out["generalization_gap_arms"] == ["A128"]


def test_paired_arm_delta_on_shared_crops():
    rows = []
    for arm, v in (("A32", 1.0), ("B32", 0.8)):
        for i in range(16):
            rows.append({"arm": arm, "checkpoint": "best", "condition": "correct", "eval_set": "val",
                         "crop_id": f"c{i}", "track_id": f"t{i // 2}", "mel_l1": v})
    d = t19.paired_arm_delta(rows, "A32", "B32", ("val",), num_bootstrap=200)
    assert d["relative_improvement"] == pytest.approx(0.2) and d["ci_low"] > 0 and t19.judge(d, CFG["arm_comparison"]) == "much_better"


@pytest.mark.parametrize("rel", ["src/muq_inversion/task19.py", "scripts/build_task19_manifests.py",
                                 "scripts/task19_analyze.py", "scripts/task19_listening_package.py"])
def test_task19_paths_have_no_direct_backend_calls(rel):
    forbidden = re.compile(r"torch\.cuda\.|torch\.xpu\.|torch\.mps\.|\.cuda\(|autocast|GradScaler|\.half\(|bfloat16")
    assert not [ln for ln in (REPO_ROOT / rel).read_text().splitlines() if forbidden.search(ln.split("#")[0])]
