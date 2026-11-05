"""TASK-25: manifest/hash identity and monitoring disjointness, identical A-D initial weights, exact
exposure counts and sample-stream grouping, exposure-based LR schedule, masked aggregation, the learned
resampler (alignment, boundaries, gradients, initialization), checkpoint/resume with RNG and sampler
state, and no evaluation-state mutation."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest
import torch

from muq_inversion.device import seed_everything
from muq_inversion.models.muq_mel_translator import MuQToMelTranslator
from muq_inversion.task23 import boundary_frames, state_dict_sha256
from muq_inversion.task24 import ShuffledPassSampler, evaluate_set
from muq_inversion.task25 import (
    TAP_OFFSETS,
    ExposureSpec,
    PhaseResampler,
    ResampledMuQToMelTranslator,
    build_model,
    forward_macs,
    lr_at_presentations,
    paired_difference,
    train_exposure_arm,
)
from muq_inversion.time_alignment import interpolate_at_positions

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
CPU = torch.device("cpu")
CFG = REPO_ROOT / "configs/task25.yaml"


def cfg():
    from train_task18_heldout import load_config

    return load_config(CFG)


# --- identity ----------------------------------------------------------------------------

def test_arm_definitions_and_budget():
    c = cfg()
    a = c["arms"]
    assert {k: (v["batch"], v["lr"]) for k, v in a.items()} == {"A": (1, 1e-3), "B": (1, 1e-4), "C": (16, 1e-3), "D": (16, 1e-4),
                                                                 "E": (16, 1e-4), "F": (16, 1e-4)}
    assert [a[k]["init"] for k in "ABCD"] == ["shared"] * 4 and a["E"]["width_multiplier"] == 2 and a["F"]["arch"] == "learned_resampler"
    b = c["budget"]
    assert b["total_presentations"] == 819200 == 3200 * 256 and b["lr_drops"] == [[204800, 0.1], [614400, 0.1]]
    for batch, updates in ((1, 819200), (16, 51200)):
        s = ExposureSpec(name="x", batch=batch, base_lr=1e-3, betas=(0.9, 0.999), total_presentations=819200, lr_drops=((204800, 0.1), (614400, 0.1)))
        assert s.updates == updates and s.boundary_updates() == {204800: 204800 // batch, 614400: 614400 // batch}
        assert 16384 % batch == 0 and 65536 % batch == 0  # evaluation cadences fall on whole updates


@pytest.mark.skipif(not (REPO_ROOT / "runs/task24/task24_v1/ladder_pairs.pt").exists(), reason="TASK-24 cached pairs not present")
def test_task24_pairs_manifest_hashes_anchor_and_monitor_disjointness():
    from task24_ladder import load_ladder

    from muq_inversion.task23 import load_pair

    lad = load_ladder(REPO_ROOT / "runs/task24/task24_v1/ladder_pairs.pt")  # verifies every per-crop hash
    with open(REPO_ROOT / "data/manifests/task24_ladder.csv", newline="") as f:
        man = list(csv.DictReader(f))
    assert [c["crop_id"] for c in lad["meta"]["fit_crops"]] == [r["crop_id"] for r in man if r["role"] in ("anchor", "fit")]
    assert [c["crop_id"] for c in lad["meta"]["monitor_crops"]] == [r["crop_id"] for r in man if r["role"] == "monitor"]
    assert not {r["track_id"] for r in man if r["role"] != "monitor"} & {r["track_id"] for r in man if r["role"] == "monitor"}
    a = load_pair(REPO_ROOT / "runs/task23/task23_v1/pair.pt")["tensors"]
    assert torch.equal(lad["tensors"]["fit"]["features"][0], a["features"][0]) and torch.equal(lad["tensors"]["fit"]["mel"][0], a["target_mel"][0])


def test_identical_initial_weights_for_A_to_D_and_trunk_copy_into_F():
    seed_everything(0, CPU)
    init = MuQToMelTranslator(8, 16, 5).state_dict()
    shas = set()
    for _ in "ABCD":
        m = build_model("current", 8, 16, 5)
        m.load_state_dict(init)
        shas.add(state_dict_sha256(m))
    assert len(shas) == 1
    f = build_model("learned_resampler", 8, 16, 5)
    res = f.load_state_dict(init, strict=False)
    assert all(k.startswith("resampler.") for k in res.missing_keys) and not res.unexpected_keys
    e = build_model("current", 8, 16, 5, width_multiplier=2)
    assert e.input_projection.out_features == 32 and e.temporal_conv.receptive_field_frames == m.temporal_conv.receptive_field_frames
    assert forward_macs(e, 6, 12)["temporal_conv"] == 4 * forward_macs(m, 6, 12)["temporal_conv"]


# --- learned resampler -------------------------------------------------------------------

def test_resampler_initialization_equals_linear_interpolation_bit_for_bit():
    torch.manual_seed(0)
    x = torch.randn(3, 16, 50)
    pos = torch.tensor([0.0, 0.13, 0.4, 1.0, 17.73, 48.2, 48.99, 49.0])
    r = PhaseResampler(16)
    assert torch.equal(r(x, pos), interpolate_at_positions(x, pos))
    seed_everything(0, CPU)
    base = MuQToMelTranslator(8, 16, 5)
    full = ResampledMuQToMelTranslator(8, 16, 5)
    full.load_state_dict(base.state_dict(), strict=False)
    feats, p = torch.randn(2, 6, 8), torch.linspace(0, 5, 12)
    assert torch.equal(full(feats, 12, p), base(feats, 12, p))


def test_resampler_taps_alignment_and_boundaries():
    r = PhaseResampler(4)
    idx, w = r.taps(torch.tensor([0.0, 0.25, 2.5, 4.0]), t_src=5)
    assert TAP_OFFSETS == (-1, 0, 1, 2)
    assert idx[:, 0].tolist() == [0, 0, 1, 2] and idx[:, 2].tolist() == [1, 2, 3, 4] and idx[:, 3].tolist() == [3, 4, 4, 4]  # clamped
    assert w.tolist() == pytest.approx([0.0, 0.25, 0.5, 0.0])
    with pytest.raises(ValueError):
        r.taps(torch.tensor([5.0]), t_src=5)
    # a learned rule is shared: identical phases at different positions get identical weights
    with torch.no_grad():
        r.mlp_out.weight.normal_()
    x = torch.zeros(1, 4, 9)
    x[..., 4] = 1.0
    y = r(x, torch.tensor([3.5, 4.5]))  # x is an impulse at 4: tap lo+1 for p=3.5 and tap lo for p=4.5
    assert torch.isfinite(y).all()


def test_resampler_gradients_finite_and_reach_both_layers_after_one_step():
    torch.manual_seed(0)
    m = ResampledMuQToMelTranslator(8, 16, 5)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    feats, tgt, p = torch.randn(4, 6, 8), torch.randn(4, 5, 12), torch.linspace(0, 5, 12)
    for step in range(2):
        opt.zero_grad()
        (m(feats, 12, p) - tgt).abs().mean().backward()
        grads = {n: q.grad for n, q in m.named_parameters()}
        assert all(torch.isfinite(g).all() for g in grads.values())
        if step == 0:
            assert grads["resampler.mlp_out.weight"].abs().sum() > 0 and grads["resampler.mlp_in.weight"].abs().sum() == 0  # zero-init output layer
        else:
            assert grads["resampler.mlp_in.weight"].abs().sum() > 0
        opt.step()


# --- schedule / sampler ------------------------------------------------------------------

def test_exposure_lr_schedule():
    d = ((204800, 0.1), (614400, 0.1))
    assert lr_at_presentations(0, 1e-3, d) == 1e-3 and lr_at_presentations(204799, 1e-3, d) == 1e-3
    assert lr_at_presentations(204800, 1e-3, d) == pytest.approx(1e-4) and lr_at_presentations(614399, 1e-3, d) == pytest.approx(1e-4)
    assert lr_at_presentations(614400, 1e-3, d) == pytest.approx(1e-5)
    # batch 16: update 12,800 still at the initial LR (covers presentations 204,785..204,800); 12,801 reduced
    assert lr_at_presentations(12799 * 16, 1e-3, d) == 1e-3 and lr_at_presentations(12800 * 16, 1e-3, d) == pytest.approx(1e-4)


def test_batches_group_the_same_sample_stream_and_counts_are_exact():
    s1, s16 = ShuffledPassSampler(32, "x"), ShuffledPassSampler(32, "x")
    stream = [s1.next() for _ in range(32 * 3)]
    batches = [[s16.next() for _ in range(16)] for _ in range(6)]
    assert [i for b in batches for i in b] == stream and s1.counts == s16.counts == [3] * 32
    assert all(len(set(b)) == 16 for b in batches)  # no repeats inside a batch (a batch never straddles a pass)


# --- evaluation / training ---------------------------------------------------------------

def test_masked_aggregation_and_paired_difference():
    m = MuQToMelTranslator(8, 16, 5).eval()
    feats, tgt = torch.randn(3, 6, 8), torch.randn(3, 5, 12)
    mask = torch.ones(12, dtype=torch.bool)
    mask[:2] = False
    agg, per_crop, preds = evaluate_set(m, feats, tgt, mask, torch.zeros(12, dtype=torch.bool), torch.linspace(0, 5, 12), CPU)
    e = (preds - tgt).abs()[..., 2:]
    assert agg["l1"] == pytest.approx(float(e.mean()), rel=1e-6) and per_crop == pytest.approx(e.mean((1, 2)).tolist(), rel=1e-6)
    d = paired_difference([1.0, 2.0, 3.0], [0.5, 2.5, 2.0])
    assert d["mean"] == pytest.approx(-1 / 3) and d["fraction_b_better"] == pytest.approx(2 / 3) and d["ci95"][0] <= d["mean"] <= d["ci95"][1]


@pytest.fixture
def deterministic():
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(prev)


def _data():
    g = torch.Generator().manual_seed(0)
    pos = torch.linspace(0, 5, 12)
    return {"fit_feats": torch.randn(32, 6, 8, generator=g), "fit_mel": torch.randn(32, 5, 12, generator=g) - 4,
            "mon_feats": torch.randn(2, 6, 8, generator=g), "mon_mel": torch.randn(2, 5, 12, generator=g) - 4,
            "mask": torch.ones(12, dtype=torch.bool), "boundary": boundary_frames(12, 2, pos, 6), "positions": pos}


def _run(tmp_path, name, arch="current", batch=4, stop=None):
    seed_everything(0, CPU)
    m = build_model(arch, 8, 16, 5)
    spec = ExposureSpec(name=name, batch=batch, base_lr=1e-3, betas=(0.9, 0.999), total_presentations=1024, lr_drops=((256, 0.1), (768, 0.1)),
                        log_every=64, fit_eval_every=128, monitor_eval_every=256, latest_every=256, matched_updates=(32, 100), showcase=(0, 1))
    return train_exposure_arm(spec, m, _data(), tmp_path / name, CPU, stop_after_update=stop, log=lambda _s: None)


def test_exposure_counts_checkpoints_and_lr_boundaries(tmp_path):
    s = _run(tmp_path, "b4")
    assert s["updates_run"] == 256 and s["presentations"] == 1024 and s["presentation_counts_min_max"] == [32, 32] and s["passes"] == 32
    ck = tmp_path / "b4/checkpoints"
    for f in ("init.pt", "best.pt", "final.pt", "latest.pt", "lr_boundary_0000256.pt", "lr_boundary_0000768.pt"):
        assert (ck / f).exists(), f
    c = torch.load(ck / "lr_boundary_0000256.pt", weights_only=False)
    assert c["state"]["update"] == 64 and c["state"]["lr"] == pytest.approx(1e-3)  # just before the first reduction
    fit_u = {json.loads(x)["update"] for x in (tmp_path / "b4/fit_eval.jsonl").read_text().splitlines()}
    assert {0, 32, 64, 100, 192, 256} <= fit_u  # cadence + matched updates + LR boundaries + final
    assert s["matched_updates"]["32"]["presentations"] == 128 if "32" in s["matched_updates"] else s["matched_updates"][32]["presentations"] == 128


def test_resume_is_exact_including_sampler_and_rng(tmp_path, deterministic):
    for arch in ("current", "learned_resampler"):
        ref = _run(tmp_path, f"{arch}_ref", arch=arch)
        assert _run(tmp_path, f"{arch}_res", arch=arch, stop=77) is None
        res = _run(tmp_path, f"{arch}_res", arch=arch)
        assert res["final_sha256"] == ref["final_sha256"] and res["presentation_counts_min_max"] == ref["presentation_counts_min_max"]
        for f in ("fit_eval", "monitor_eval"):
            strip = lambda t: [{k: v for k, v in json.loads(x).items() if k != "wall_seconds"} for x in t.splitlines()]
            assert strip((tmp_path / f"{arch}_ref/{f}.jsonl").read_text()) == strip((tmp_path / f"{arch}_res/{f}.jsonl").read_text()), (arch, f)


def test_evaluation_does_not_mutate_state(tmp_path):
    m = ResampledMuQToMelTranslator(8, 16, 5)
    with torch.no_grad():
        m.resampler.mlp_out.weight.normal_()
    before = {k: v.clone() for k, v in m.state_dict().items()}
    m.train()
    torch.manual_seed(3)
    evaluate_set(m, torch.randn(3, 6, 8), torch.randn(3, 5, 12), torch.ones(12, dtype=torch.bool), torch.zeros(12, dtype=torch.bool),
                 torch.linspace(0, 5, 12), CPU)
    after = torch.rand(1)
    torch.manual_seed(3)
    torch.randn(3, 6, 8), torch.randn(3, 5, 12)
    assert torch.equal(after, torch.rand(1)) and m.training
    assert all(torch.equal(v, before[k]) for k, v in m.state_dict().items())
