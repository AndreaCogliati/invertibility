"""TASK-18: split/crop construction, sampler, controls, metrics, statistics, outcome rule."""

from __future__ import annotations

import collections
import csv
import random
import re
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from muq_inversion import task18 as t18
from muq_inversion.task18_report import render_results

REPO_ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((REPO_ROOT / "configs/task18_heldout.yaml").read_text())


def _csv(path):
    with open(REPO_ROOT / path, newline="") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def tracks():
    return _csv(CFG["track_manifest"])


@pytest.fixture(scope="module")
def crops():
    return _csv(CFG["crop_manifest"])


# --- tracks ---------------------------------------------------------------------------------

def test_track_selection_is_balanced_deduplicated_and_deterministic():
    source = _csv(CFG["source_manifest"])
    a = t18.select_tracks(source, CFG["track_counts"], CFG["seed"])
    b = t18.select_tracks(source, CFG["track_counts"], CFG["seed"])
    assert [r["track_id"] for r in a] == [r["track_id"] for r in b]
    by_split = collections.defaultdict(list)
    for r in a:
        by_split[r["split"]].append(r)
    assert {s: len(v) for s, v in by_split.items()} == {"train": 32, "val": 8, "test": 8}
    train_genres = collections.Counter(r["genre"] for r in by_split["train"])
    assert set(train_genres) == set(t18.GENRES) and set(train_genres.values()) <= {3, 4}
    for s in ("val", "test"):
        assert len({r["genre"] for r in by_split[s]}) == 8  # one track per genre
    assert {r["genre"] for r in by_split["val"] + by_split["test"]} == set(t18.GENRES)
    dup = {c for c, n in collections.Counter(r["checksum"] for r in source).items() if n > 1}
    assert not any(r["checksum"] in dup for r in a), "duplicate-checksum audio selected"
    committed = {r["split"]: r for r in source}
    assert all(r["split"] == next(x for x in source if x["track_id"] == r["track_id"])["split"] for r in a)
    assert committed  # selections respect the committed split


def test_committed_manifests_match_a_fresh_build(tracks, crops):
    source = _csv(CFG["source_manifest"])
    selected = t18.select_tracks(source, CFG["track_counts"], CFG["seed"])
    assert [r["track_id"] for r in selected] == [r["track_id"] for r in tracks]
    rebuilt = t18.build_crop_rows(tracks, CFG, CFG["seed"])
    assert [{k: str(v) for k, v in r.items()} for r in rebuilt] == crops


def test_no_track_crosses_splits_and_unseen_windows_never_overlap_training(crops):
    counts = t18.check_crop_rows(crops)
    assert counts == {"train_unseen": 64, "train_pool": 768, "memorized": 64, "val": 32, "test": 32}
    split_of = {}
    for r in crops:
        assert split_of.setdefault(r["track_id"], r["split"]) == r["split"]
    assert {r["split"] for r in crops if r["role"] == "train_pool"} == {"train"}
    n = int(crops[0]["crop_num_samples"])
    assert n == 187 * 256 == 47_872


def test_listening_set_fixed_and_covers_every_regime(crops):
    chosen = [r for r in crops if r["listening"] == "1"]
    by_set = collections.Counter(r["eval_set"] for r in chosen)
    assert by_set == {"val": 8, "test": 8, "train_unseen": 4, "memorized": 4}


def test_allowed_intervals_and_stratified_starts():
    L = 100
    iv = t18.allowed_intervals(1000, [300, 700], L)
    assert iv == [(0, 200), (400, 600), (800, 1000)]
    starts = t18.stratified_starts(iv, 30, random.Random(0))
    assert all(not (s < e + L and e < s + L) for s in starts for e in (300, 700))
    assert min(starts) < 200 and max(starts) > 800  # spread over the whole track


# --- sampler --------------------------------------------------------------------------------

def test_sampler_is_track_balanced_and_exactly_resumable():
    pool = {f"t{i}": list(range(i * 5, i * 5 + 5)) for i in range(8)}
    s = t18.TrackBalancedSampler(pool, 4, random.Random(1))
    ref = [s.next_batch() for _ in range(40)]
    ex = s.exposure_summary()
    assert ex["tracks_visited"] == 8 and ex["track_exposure_min"] == ex["track_exposure_max"] == 20
    s2 = t18.TrackBalancedSampler(pool, 4, random.Random(1))
    first = [s2.next_batch() for _ in range(13)]
    state = s2.state_dict()
    s3 = t18.TrackBalancedSampler(pool, 4, random.Random(999))
    s3.load_state_dict(state)
    assert first + [s3.next_batch() for _ in range(27)] == ref


# --- controls -------------------------------------------------------------------------------

def test_shuffled_across_never_pairs_a_crop_with_its_own_track():
    ids = ["a", "a", "b", "b", "c", "c", "d", "d"]
    j = t18.shuffled_across_index(ids)
    assert all(ids[i] != ids[k] for i, k in enumerate(j))
    uneven = ["a", "a", "a", "b"]
    assert all(uneven[i] != uneven[k] for i, k in enumerate(t18.shuffled_across_index(uneven)))
    with pytest.raises(ValueError):
        t18.shuffled_across_index(["a", "a"])


def test_shuffled_time_and_conditioned_features():
    feats = torch.randn(3, 50, 8)
    perms = t18.shuffled_time_perms(3, 50, seed=0)
    assert torch.equal(perms, t18.shuffled_time_perms(3, 50, seed=0))
    assert all(torch.equal(p.sort().values, torch.arange(50)) for p in perms)
    x = t18.conditioned_features("shuffled_time", feats, torch.tensor([1, 2, 0]), perms)
    assert torch.equal(x[0], feats[0, perms[0]])
    assert torch.equal(t18.conditioned_features("shuffled_across", feats, torch.tensor([1, 2, 0]), perms)[0], feats[1])
    assert not t18.conditioned_features("constant_zero", feats, torch.tensor([1, 2, 0]), perms).any()


# --- metrics --------------------------------------------------------------------------------

def test_mel_metrics_identity_and_scale():
    m = torch.randn(100, 187)
    same = t18.mel_metrics(m, m, floor=-11.5)
    assert same["mel_l1"] == 0 and same["mel_cosine"] == pytest.approx(1) and same["mel_variance_ratio"] == pytest.approx(1)
    flat = t18.mel_metrics(torch.zeros(100, 187), m, floor=-11.5)
    assert flat["mel_variance_ratio"] == 0 and np.isnan(flat["mel_pearson"])


def test_contour_metrics_identity_and_independence():
    rng = np.random.default_rng(0)
    sr = 24_000
    t = np.arange(47_872) / sr
    x = (np.sin(2 * np.pi * 220 * t) * (1 + np.sin(2 * np.pi * 2 * t)) + 0.05 * rng.standard_normal(t.size)).astype(np.float32)
    same = t18.contour_metrics(x, x, sr, [0, 50, 200, 500])
    assert all(v == pytest.approx(1, abs=1e-6) for v in same.values())
    other = t18.contour_metrics(x, rng.standard_normal(t.size).astype(np.float32), sr, [0, 50])
    assert other["rms_corr"] < 0.5 and other["chroma_cosine"] < same["chroma_cosine"]


def test_cluster_bootstrap_delta_sign_and_ci():
    clusters = [f"t{i // 4}" for i in range(32)]
    correct = [1.0] * 32
    worse = [1.5 + 0.01 * i for i in range(32)]
    d = t18.cluster_bootstrap_delta(correct, worse, clusters, lower_is_better=True, num_bootstrap=500)
    assert d["mean_delta"] > 0 and d["ci_low"] > 0 and d["n_tracks"] == 8 and d["proportion_tracks_positive"] == 1
    d2 = t18.cluster_bootstrap_delta(correct, correct, clusters, lower_is_better=True, num_bootstrap=200)
    assert d2["mean_delta"] == 0 and d2["ci_low"] <= 0 <= d2["ci_high"]
    hb = t18.cluster_bootstrap_delta([0.9] * 32, [0.1] * 32, clusters, lower_is_better=False, num_bootstrap=200)
    assert hb["mean_delta"] == pytest.approx(0.8)


# --- outcome rule ---------------------------------------------------------------------------

def _agg(correct: float, controls: float, beats: bool, rt: float = 0.1) -> dict:
    ci = 0.05 if beats else -0.05
    return {
        "conditions": {c: {"mel_l1": {"mean": correct if c == "correct" else controls}} for c in t18.CONDITIONS},
        "oracle_round_trip_mel_l1": {"mean": rt},
        "paired_mel_l1": {c: {"ci_low": ci, "proportion_tracks_positive": 1.0 if beats else 0.4}
                          for c in t18.CONDITIONS if c != "correct"},
    }


@pytest.mark.parametrize(("aggs", "code"), [
    ({"memorized": (0.5, 1.4, False), "train_unseen": (1, 1.4, True), "val": (1, 1.4, True), "test": (1, 1.4, True)}, "UNDERFIT"),
    ({"memorized": (0.3, 1.4, True), "train_unseen": (1, 1.4, True), "val": (1, 1.4, True), "test": (1, 1.4, True)}, "UNDERFIT"),
    ({"memorized": (0.12, 1.4, True), "train_unseen": (1, 1.4, False), "val": (1, 1.4, False), "test": (1, 1.4, False)}, "ALL_FAIL"),
    ({"memorized": (0.12, 1.4, True), "train_unseen": (0.5, 1.4, True), "val": (1, 1.4, True), "test": (1, 1.4, False)}, "TRACK_SPECIFIC_ONLY"),
    ({"memorized": (0.12, 1.4, True), "train_unseen": (0.5, 1.4, True), "val": (0.9, 1.4, True), "test": (0.9, 1.4, True)}, "HELDOUT_COARSE"),
    ({"memorized": (0.11, 1.4, True), "train_unseen": (0.12, 1.4, True), "val": (0.12, 1.4, True), "test": (0.12, 1.4, True)}, "HELDOUT_NEAR_ORACLE"),
])
def test_outcome_rule(aggs, code):
    out = t18.classify_outcome({s: _agg(*v) for s, v in aggs.items()}, CFG["outcome_rule"])
    assert out["code"] == code and out["listening_verdict"] == "pending"


def test_beating_a_control_also_needs_most_tracks():
    agg = _agg(0.12, 1.4, True)
    agg["paired_mel_l1"]["shuffled_time"]["proportion_tracks_positive"] = 0.5  # CI > 0 but only half the tracks
    st = t18.regime_status(agg, CFG["outcome_rule"])
    assert not st["beats"]["shuffled_time"] and not st["beats_all"] and st["beats"]["shuffled_across"]


def test_report_renders():
    aggs = {s: _agg(0.5, 1.4, True) for s in t18.EVAL_SETS}
    for a in aggs.values():
        a |= {"n_crops": 32, "n_tracks": 8}
        for c in a["conditions"].values():
            c["mel_cosine"] = {"mean": 0.9}
        for d in a["paired_mel_l1"].values():
            d |= {"mean_delta": 0.9, "ci_high": 1.0}
    report = {
        "run_name": "x", "wandb_run_id": "abc", "final_step": 10, "best": aggs,
        "outcome_best": t18.classify_outcome(aggs, CFG["outcome_rule"]), "outcome_final": {"code": "UNDERFIT"},
        "best_checkpoint": {"step": 5, "selection": "val"}, "listening": [],
        "resolved_config": {"device_metadata": {"device_name": "cpu", "backend": "cpu", "torch_version": "2"},
                            "track_manifest_sha256": "a" * 64, "crop_manifest_sha256": "b" * 64},
        "exposure": {"tracks_visited": 1, "tracks_total": 1, "track_exposure_min": 1, "track_exposure_max": 1,
                     "track_exposure_max_over_min": 1.0, "pool_crops_visited": 1, "pool_crops_total": 1},
        "lineage": [], "paths": {},
    }
    md = render_results(report)
    assert f"`{report['outcome_best']['code']}`" in md and "| val | 32 / 8 |" in md and "Training exposure" in md


# --- portability ----------------------------------------------------------------------------

@pytest.mark.parametrize("rel", ["scripts/train_task18_heldout.py", "scripts/build_task18_manifests.py",
                                 "scripts/task18_compare_checkpoints.py", "src/muq_inversion/task18.py",
                                 "src/muq_inversion/task18_report.py"])
def test_task18_paths_have_no_direct_backend_calls(rel):
    forbidden = re.compile(r"torch\.cuda\.|torch\.xpu\.|torch\.mps\.|\.cuda\(|autocast|GradScaler|\.half\(|bfloat16")
    hits = [line for line in (REPO_ROOT / rel).read_text().splitlines() if forbidden.search(line.split("#")[0])]
    assert not hits, hits
