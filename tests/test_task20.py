"""TASK-20: 512-track manifest, disjointness, unchanged recipe, early stopping."""

from __future__ import annotations

import collections
import csv
import random
import sys
from pathlib import Path

import pytest

from muq_inversion import task20 as t20
from muq_inversion.task19 import fault_report, load_faults

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from train_task18_heldout import load_config

CFG = load_config(REPO_ROOT / "configs/task20.yaml")
A128 = {**load_config(REPO_ROOT / "configs/task19.yaml"), **load_config(REPO_ROOT / "configs/task19.yaml")["arms"]["A128"]}


def _csv(path):
    with open(REPO_ROOT / path, newline="") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def tracks():
    return _csv(CFG["track_manifest"])


@pytest.fixture(scope="module")
def crops():
    return _csv(CFG["crop_manifest"])


def test_512_training_tracks_include_all_128_and_are_balanced(tracks):
    train = [t for t in tracks if t["split"] == "train"]
    assert len(train) == 512 == CFG["train_target"]
    t19_train = {t["track_id"] for t in _csv("data/manifests/task19_tracks.csv") if t["split"] == "train"}
    assert len(t19_train) == 128 and t19_train <= {t["track_id"] for t in train}
    g = collections.Counter(t["genre"] for t in train)
    assert min(g.values()) >= 51 and max(g.values()) <= 52 and sum(g.values()) == 512


def test_heldout_tracks_unchanged_and_disjoint_from_training(tracks):
    held = {(t["track_id"], t["split"], t["tier"]) for t in tracks if t["split"] != "train"}
    t19_held = {(t["track_id"], t["split"], t["tier"]) for t in _csv("data/manifests/task19_tracks.csv") if t["split"] != "train"}
    assert held == t19_held and len(held) == 64
    assert not {h[0] for h in held} & {t["track_id"] for t in tracks if t["split"] == "train"}
    rep = fault_report(tracks, load_faults(REPO_ROOT / CFG["fault_list"]))
    assert rep["repetition_groups_crossing_splits"] == [] and rep["distorted"] == {}


def test_all_task19_crops_identical_and_pool_never_on_heldout(crops):
    t19 = {r["crop_id"]: r for r in _csv("data/manifests/task19_crops.csv")}
    mine = {r["crop_id"]: r for r in crops}
    for cid, r in t19.items():
        assert all(mine[cid][k] == v for k, v in r.items()), cid
    assert {r["split"] for r in crops if r["role"] == "train_pool"} == {"train"}
    assert sum(r["role"] == "train_pool" for r in crops) == 512 * 24
    # crop/target alignment: every crop is exactly 187 BigVGAN hops
    assert {r["crop_num_samples"] for r in crops} == {str(187 * 256)}


def test_recipe_and_model_unchanged_from_a128():
    for k in ("muq_model_id", "bigvgan_revision", "hidden_channels", "temporal_alignment", "decode_clamp_to_mel_floor",
              "learning_rate", "adam_betas", "batch_size", "seed", "loss", "num_mel_hops", "eval_sets"):
        assert CFG[k] == A128[k], k
    arm = CFG["arms"]["N512"]
    assert arm["lr_schedule"] == A128["lr_schedule"] == {"type": "constant"} and arm["train_tracks"] == "all"
    assert CFG["early_stopping"] == {"patience": 8, "min_delta": 0.0} and CFG["eval_every"] == 250 and CFG["steps"] == 20000


def test_balanced_genre_totals_respects_caps():
    caps = {g: 75 for g in t20.GENRES} | {"pop": 51, "jazz": 53}
    have = dict.fromkeys(t20.GENRES, 12)
    tot = t20.balanced_genre_totals(512, have, caps, random.Random(0))
    assert sum(tot.values()) == 512 and tot["pop"] == 51 and max(tot.values()) - min(tot.values()) <= 1
    small = t20.balanced_genre_totals(10_000, have, caps, random.Random(0))
    assert small == caps  # target larger than available -> largest eligible set


def test_early_stopping_rule():
    cfg = {"patience": 3, "min_delta": 0.0}
    es = {"checks_since_improvement": 0, "stopped": False, "reason": None, "stopped_at_step": None}
    best = 1.0
    for v in (0.9, 0.95, 0.9, 0.91):  # improve, then 3 non-improvements (equal is not an improvement)
        es = t20.early_stopping_update(es, v, best, cfg)
        best = min(best, v)
    assert es["stopped"] and es["checks_since_improvement"] == 3
    es2 = t20.early_stopping_update({"checks_since_improvement": 2, "stopped": False, "reason": None, "stopped_at_step": None}, 0.5, 0.6, cfg)
    assert es2["checks_since_improvement"] == 0 and not es2["stopped"]


def test_early_stopping_uses_primary_val_only():
    src = (REPO_ROOT / "scripts/train_task18_heldout.py").read_text()
    call = [ln for ln in src.splitlines() if "early_stopping_update(" in ln and "import" not in ln]
    assert len(call) == 1 and 'ev["val/correct/mel_l1"]' in call[0]
    assert "val2" not in call[0] and "test" not in call[0]
