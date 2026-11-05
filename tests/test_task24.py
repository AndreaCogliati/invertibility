"""TASK-24: nested manifests and split disjointness, cached-pair hashes and exact anchor reuse,
identical initial weights across arms, sampler coverage and exposure counts, masked per-crop
aggregation, frozen MuQ, LR schedule and checkpoint timing, save/resume with sampler and RNG
state, no model-state mutation during evaluation, and the per-arm writer lock."""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from muq_inversion.device import seed_everything
from muq_inversion.models.muq_mel_translator import MuQToMelTranslator
from muq_inversion.task23 import boundary_frames, state_dict_sha256, tensor_sha256
from muq_inversion.task24 import (
    ArmLock,
    LadderSpec,
    ShuffledPassSampler,
    anchor_row_matches,
    evaluate_set,
    lr_at,
    select_ladder,
    train_ladder_arm,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
CPU = torch.device("cpu")


def rows(path: str) -> list[dict]:
    with open(REPO_ROOT / path, newline="") as f:
        return list(csv.DictReader(f))


# --- manifests ---------------------------------------------------------------------------

def test_committed_ladder_manifest_is_nested_disjoint_training_only_and_reproducible():
    lad = rows("data/manifests/task24_ladder.csv")
    crops = {r["crop_id"]: r for r in rows("data/manifests/task20_crops.csv")}
    tracks = {t["track_id"]: t for t in rows("data/manifests/task20_tracks.csv")}
    meta = json.loads((REPO_ROOT / "data/manifests/task24_ladder.json").read_text())
    fit = [r for r in lad if r["role"] in ("anchor", "fit")]
    mon = [r for r in lad if r["role"] == "monitor"]
    assert [int(r["rank"]) for r in fit] == list(range(256)) and len(mon) == 16
    assert fit[0]["role"] == "anchor" and fit[0]["crop_id"] == "rock.00009:train_pool:pool:18"
    sizes = meta["sizes"]
    for r in fit:  # nesting: rank r belongs to every N > r, and min_n is the smallest such N
        assert int(r["min_n"]) == min(n for n in sizes if int(r["rank"]) < n)
    for r in fit + mon:  # training-pool crops of training tracks only; manifest fields match the source
        src = crops[r["crop_id"]]
        assert src["role"] == "train_pool" and src["split"] == "train" and tracks[r["track_id"]]["split"] == "train"
        assert src["crop_start_sample"] == r["crop_start_sample"] and src["path"] == r["path"]
    fit_tracks = [r["track_id"] for r in fit]
    assert len(set(fit_tracks)) == 256  # distinct tracks
    assert not set(fit_tracks) & {r["track_id"] for r in mon} and len({r["track_id"] for r in mon}) == 16
    heldout = {t for t, v in tracks.items() if v["split"] != "train"}
    assert not (set(fit_tracks) | {r["track_id"] for r in mon}) & heldout
    pool = [r for r in crops.values() if r["role"] == "train_pool" and r["split"] == "train"]
    f2, m2 = select_ladder(pool, fit[0]["crop_id"], 256, 16, meta["selection_seed"])
    assert [r["crop_id"] for r in f2] == [r["crop_id"] for r in fit] and [r["crop_id"] for r in m2] == [r["crop_id"] for r in mon]
    assert all(len(meta["genre_counts"]["256"]) == 10 for _ in [0]) and min(meta["genre_counts"]["64"].values()) >= 5


def _toy_pool(n_tracks=6, per=3, genres=("a", "b")):
    return [{"crop_id": f"t{t}:train_pool:pool:{c}", "track_id": f"t{t}", "genre": genres[t % len(genres)], "path": f"p{t}",
             "crop_start_sample": str(1000 * c), "role": "train_pool", "split": "train"} for t in range(n_tracks) for c in range(per)]


def test_select_ladder_prefix_nesting_fallback_and_cache_restriction():
    pool = _toy_pool()
    f8, _ = select_ladder(pool, "t0:train_pool:pool:1", 8, 0, seed=0)  # 6 tracks: needs the fallback
    f4, mon = select_ladder(pool, "t0:train_pool:pool:1", 4, 2, seed=0)
    assert f8[:4] == f4 and f8[0]["crop_id"] == "t0:train_pool:pool:1"
    assert len({r["track_id"] for r in f8[:6]}) == 6 and len({r["crop_id"] for r in f8}) == 8  # distinct tracks first
    assert {r["track_id"] for r in mon}.isdisjoint({r["track_id"] for r in f4}) and len(mon) == 2
    with pytest.raises(ValueError, match="unused tracks"):
        select_ladder(pool, "t0:train_pool:pool:1", 6, 1, seed=0)
    avail = {(r["path"], int(r["crop_start_sample"])) for r in pool if r["track_id"] in ("t1", "t2", "t3")}
    fr, mr = select_ladder(pool, "t0:train_pool:pool:1", 3, 1, seed=0, available=avail)
    assert {r["track_id"] for r in fr[1:]} <= {"t1", "t2", "t3"} and mr[0]["track_id"] in {"t1", "t2", "t3"}
    with pytest.raises(ValueError):
        select_ladder(pool, "missing", 3, 1, seed=0)


# --- cached pairs ------------------------------------------------------------------------

def test_ladder_pair_hashes_detect_tampering_and_anchor_must_match(tmp_path):
    from task24_ladder import load_ladder

    g = torch.Generator().manual_seed(0)
    fit = {"features": torch.randn(2, 6, 8, generator=g), "mel": torch.randn(2, 5, 12, generator=g), "wave": torch.randn(2, 48, generator=g)}
    info = [{"crop_id": f"c{j}", **{f"{k}_sha256": tensor_sha256(fit[k][j]) for k in ("features", "mel", "wave")}} for j in range(2)]
    d = {"tensors": {"fit": fit, "monitor": {k: v[:1].clone() for k, v in fit.items()}}, "meta": {"fit_crops": info, "monitor_crops": info[:1]}}
    torch.save(d, tmp_path / "l.pt")
    load_ladder(tmp_path / "l.pt")
    d["tensors"]["fit"]["mel"][1, 0, 0] += 1e-4
    torch.save(d, tmp_path / "l.pt")
    with pytest.raises(RuntimeError, match="c1 mel hash"):
        load_ladder(tmp_path / "l.pt")
    pair = {"features": fit["features"][:1].clone(), "target_mel": fit["mel"][:1].clone(), "waveform": fit["wave"][0].clone()}
    cache = {"muq": fit["features"], "mel": fit["mel"], "wave": fit["wave"]}
    assert anchor_row_matches(cache, 0, pair) and not anchor_row_matches(cache, 1, pair)


# --- sampler / schedule ------------------------------------------------------------------

def test_sampler_passes_without_replacement_exact_counts_and_resume():
    s = ShuffledPassSampler(7, "seed")
    draws = [s.next() for _ in range(7 * 5)]
    for p in range(5):
        assert sorted(draws[7 * p:7 * (p + 1)]) == list(range(7))  # each pass is a permutation
    assert s.counts == [5] * 7 and s.passes == 5
    mid = ShuffledPassSampler(7, "seed")
    [mid.next() for _ in range(10)]
    st = json.loads(json.dumps(mid.state_dict(), default=list))
    st["rng"] = mid.state_dict()["rng"]
    cont = ShuffledPassSampler(7, "seed")
    cont.load_state_dict(st)
    assert [cont.next() for _ in range(25)] == [mid.next() for _ in range(25)]


def test_lr_schedule_boundaries():
    drops = ((50000, 0.1), (150000, 0.1))
    assert lr_at(1, 1e-3, drops) == 1e-3 and lr_at(50000, 1e-3, drops) == 1e-3
    assert lr_at(50001, 1e-3, drops) == pytest.approx(1e-4) and lr_at(150000, 1e-3, drops) == pytest.approx(1e-4)
    assert lr_at(150001, 1e-3, drops) == pytest.approx(1e-5) and lr_at(200000, 1e-3, drops) == pytest.approx(1e-5)


# --- evaluation --------------------------------------------------------------------------

def test_evaluate_set_aggregates_by_valid_element_and_leaves_state_and_rng():
    class Toy(nn.Module):  # BatchNorm + Dropout: would mutate state / RNG if evaluation were careless
        def __init__(self):
            super().__init__()
            self.bn, self.drop = nn.BatchNorm1d(5), nn.Dropout(0.5)

        def forward(self, x, t_mel, _pos):
            return self.drop(self.bn(x[:, :5, :t_mel]))

    m = Toy()
    m.train()
    feats = torch.randn(3, 5, 12)
    tgt = torch.zeros(3, 5, 12)
    mask = torch.ones(12, dtype=torch.bool)
    mask[-2:] = False
    before = {k: v.clone() for k, v in m.state_dict().items()}
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    agg, per_crop, preds = evaluate_set(m, feats, tgt, mask, torch.zeros(12, dtype=torch.bool), torch.zeros(12), CPU, chunk=2)
    draws = (random.random(), np.random.rand(), torch.rand(1))
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    assert draws == (random.random(), np.random.rand(), torch.rand(1))
    assert all(torch.equal(v, before[k]) for k, v in m.state_dict().items()) and m.training
    m.eval()
    with torch.no_grad():
        expect = m(feats, 12, None)[..., :10].abs()
    assert torch.allclose(preds, m(feats, 12, None).detach())
    assert agg["l1"] == pytest.approx(float(expect.mean()), rel=1e-6)
    assert per_crop == pytest.approx(expect.mean((1, 2)).tolist(), rel=1e-6)
    assert agg["per_crop_max"] == pytest.approx(max(per_crop)) and agg["per_crop_median"] == pytest.approx(float(np.median(per_crop)))


def test_frozen_muq_encoder(monkeypatch):
    import muq_inversion.encoder as enc

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin, self.drop = nn.Linear(4, 4), nn.Dropout(0.5)

    monkeypatch.setattr(enc.MuQ, "from_pretrained", staticmethod(lambda _id: Tiny()))
    muq = enc.FrozenMuQEncoder("x", CPU)
    assert not muq.model.training and all(not p.requires_grad for p in muq.model.parameters())


# --- training ----------------------------------------------------------------------------

@pytest.fixture
def deterministic():
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(prev)


def _data(n):
    g = torch.Generator().manual_seed(0)
    pos = torch.linspace(0, 5, 12)
    return {"fit_feats": torch.randn(n, 6, 8, generator=g), "fit_mel": torch.randn(n, 5, 12, generator=g) - 4,
            "mon_feats": torch.randn(2, 6, 8, generator=g), "mon_mel": torch.randn(2, 5, 12, generator=g) - 4,
            "mask": torch.ones(12, dtype=torch.bool), "boundary": boundary_frames(12, 2, pos, 6), "positions": pos}


def _init_state():
    seed_everything(0, CPU)
    return MuQToMelTranslator(8, 16, 5).state_dict()


def _run(tmp_path, name, n, stop=None, steps=60):
    model = MuQToMelTranslator(8, 16, 5)
    model.load_state_dict(_init_state())
    spec = LadderSpec(n=n, steps=steps, base_lr=1e-3, betas=(0.9, 0.999), lr_drops=((20, 0.1), (40, 0.1)), milestones=(20, 40, steps),
                      exposures=(5, 100), log_every=5, fit_eval_every=10, monitor_eval_every=20, latest_every=10,
                      sampler_seed=f"s-{n}", showcase=tuple(range(min(n, 2))))
    return train_ladder_arm(spec, model, _data(n), tmp_path / name, CPU, stop_after_step=stop, log=lambda _s: None)


def test_identical_initial_weights_across_arms(tmp_path):
    s1, s3 = _run(tmp_path, "n1", 1), _run(tmp_path, "n3", 3)
    assert s1["init_sha256"] == s3["init_sha256"]
    seed_everything(0, CPU)
    assert s1["init_sha256"] == state_dict_sha256(MuQToMelTranslator(8, 16, 5))


def test_lr_schedule_checkpoint_and_exposure_timing(tmp_path):
    s = _run(tmp_path, "n3", 3)
    ck = tmp_path / "n3/checkpoints"
    for step, lr in ((20, 1e-3), (40, 1e-4), (60, 1e-5)):  # milestones 20/40 = just before each reduction
        c = torch.load(ck / f"milestone_{step:06d}.pt", weights_only=False)
        assert c["state"]["step"] == step and c["state"]["lr"] == pytest.approx(lr)
    e = torch.load(tmp_path / "n3/preds/exposure_0005x.pt", weights_only=False)
    assert e["step"] == 15 and e["presentation_counts"] == [5, 5, 5] and e["fit"].shape == (3, 5, 12)
    assert s["unreached_exposures"] == [100] and s["presentation_counts"] == [20, 20, 20]
    fit_steps = [json.loads(x)["step"] for x in (tmp_path / "n3/fit_eval.jsonl").read_text().splitlines()]
    assert {0, 10, 15, 20, 30, 40, 50, 60} <= set(fit_steps)
    mon_steps = [json.loads(x)["step"] for x in (tmp_path / "n3/monitor_eval.jsonl").read_text().splitlines()]
    assert {0, 15, 20, 40, 60} <= set(mon_steps)
    assert (ck / "best.pt").exists() and (tmp_path / "n3/preds/final.pt").exists()


def test_resume_restores_model_optimizer_sampler_and_logs_exactly(tmp_path, deterministic):
    ref = _run(tmp_path, "ref", 3)
    assert _run(tmp_path, "res", 3, stop=27) is None
    res = _run(tmp_path, "res", 3)
    assert res["final_sha256"] == ref["final_sha256"] and res["presentation_counts"] == ref["presentation_counts"]
    for f in ("fit_eval", "monitor_eval", "train_log"):
        a, b = ((tmp_path / d / f"{f}.jsonl").read_text().splitlines() for d in ("ref", "res"))
        drop = ("updates_per_second",)
        assert [{k: v for k, v in json.loads(x).items() if k not in drop} for x in a] == \
               [{k: v for k, v in json.loads(x).items() if k not in drop} for x in b], f


def test_arm_lock_excludes_a_second_writer(tmp_path):
    lock = ArmLock(tmp_path / "N0004/LOCK")
    with pytest.raises(SystemExit, match="another writer"):
        ArmLock(tmp_path / "N0004/LOCK")
    ArmLock(tmp_path / "N0004/LOCK", steal=True).release()
    lock.release()
    assert not (tmp_path / "N0004/LOCK").exists()
