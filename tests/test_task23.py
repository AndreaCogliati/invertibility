"""TASK-23: cached-pair hashes, masked loss and detail metrics with padded frames, gradient flow,
frozen MuQ, identical B/C initialization, state-preserving train/eval diagnostics, and
checkpoint save/resume of the one-pair trainer."""

from __future__ import annotations

import json
import random

import numpy as np
import pytest
import torch
from torch import nn

from muq_inversion.device import seed_everything
from muq_inversion.models.muq_mel_translator import MuQToMelTranslator
from muq_inversion.task23 import (
    ArmSpec,
    FreeMel,
    boundary_frames,
    disable_regularizers,
    load_pair,
    masked_diff_errors,
    masked_l1,
    mode_predictions,
    regularizer_audit,
    save_pair,
    state_dict_sha256,
    tensor_sha256,
    train_arm,
)

CPU = torch.device("cpu")


def toy_pair(t_muq: int = 6, c: int = 8, m: int = 5, t: int = 12) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    return {"features": torch.randn(1, t_muq, c, generator=g), "target_mel": torch.randn(1, m, t, generator=g) - 4,
            "mask": torch.ones(t, dtype=torch.bool), "waveform": torch.randn(t * 4, generator=g),
            "positions": torch.linspace(0, t_muq - 1, t)}


def test_pair_cache_round_trip_hashes_and_tamper(tmp_path):
    pair = toy_pair()
    hashes = save_pair(tmp_path / "pair.pt", pair, {"crop": "x"})
    again = save_pair(tmp_path / "pair2.pt", {k: v.clone() for k, v in pair.items()}, {"crop": "x"})
    assert hashes == again  # deterministic: identical tensors, identical hashes
    d = load_pair(tmp_path / "pair.pt")
    assert all(torch.equal(d["tensors"][k], pair[k]) for k in pair) and d["meta"] == {"crop": "x"}
    raw = torch.load(tmp_path / "pair.pt", weights_only=False)
    raw["tensors"]["features"][0, 0, 0] += 1e-3  # silent corruption, hash left as recorded
    torch.save(raw, tmp_path / "pair.pt")
    with pytest.raises(RuntimeError, match="features hash"):
        load_pair(tmp_path / "pair.pt")
    assert tensor_sha256(torch.zeros(2, 3)) != tensor_sha256(torch.zeros(3, 2))


def test_pair_validation_rejects_bad_tensors(tmp_path):
    for k, bad in (("mask", torch.zeros(12, dtype=torch.bool)), ("target_mel", torch.full((1, 5, 12), float("nan"))),
                   ("positions", torch.linspace(0, 9, 12))):
        with pytest.raises(ValueError):
            save_pair(tmp_path / f"{k}.pt", toy_pair() | {k: bad}, {})


def test_masked_l1_ignores_padded_frames_and_matches_task22_loss():
    g = torch.Generator().manual_seed(1)
    pred = torch.randn(1, 4, 10, generator=g, requires_grad=True)
    target = torch.randn(1, 4, 10, generator=g)
    mask = torch.ones(10, dtype=torch.bool)
    mask[7:] = False
    poisoned = pred.detach().clone()
    poisoned[..., 7:] = 1e6  # padded frames may hold anything
    expect = (pred.detach()[..., :7] - target[..., :7]).abs().mean()
    assert torch.allclose(masked_l1(poisoned, target, mask), expect)
    loss = masked_l1(pred, target, mask)
    loss.backward()
    assert torch.all(pred.grad[..., 7:] == 0) and torch.all(pred.grad[..., :7] != 0)
    full = torch.ones(10, dtype=torch.bool)  # TASK-22: no padding -> exactly the unmasked mean
    assert torch.allclose(masked_l1(pred.detach(), target, full), (pred.detach() - target).abs().mean(), rtol=0, atol=1e-7)
    with pytest.raises(ValueError):
        masked_l1(pred.detach(), target, torch.zeros(10, dtype=torch.bool))


def test_detail_metrics_use_only_valid_adjacent_pairs():
    target = torch.zeros(1, 3, 6)
    pred = target.clone()
    pred[..., 4:] = 5.0  # a jump between frame 3 and 4: frame 4 onward is padding
    mask = torch.tensor([1, 1, 1, 1, 0, 0], dtype=torch.bool)
    d = masked_diff_errors(pred, target, mask)
    assert d["time_diff_l1"] == 0.0 and d["freq_diff_l1"] == 0.0
    d2 = masked_diff_errors(pred, target, torch.ones(6, dtype=torch.bool))
    assert d2["time_diff_l1"] > 0


def test_boundary_frames_edges_and_replicated_positions():
    pos = torch.tensor([0.0, 0.0, 0.3, 1.0, 2.0, 3.0, 4.0, 4.0])
    b = boundary_frames(8, 1, pos, num_tokens=5)
    assert b.tolist() == [True, True, False, False, False, False, True, True]


def test_gradient_flow_to_free_tensor_and_decoder():
    pair = toy_pair()
    free = FreeMel(tuple(pair["target_mel"].shape), float(pair["target_mel"].mean()), 0.01, seed=0)
    assert not torch.allclose(free.mel, pair["target_mel"])  # not initialized at the target
    masked_l1(free(), pair["target_mel"], pair["mask"]).backward()
    assert torch.isfinite(free.mel.grad).all() and free.mel.grad.abs().sum() > 0
    dec = MuQToMelTranslator(8, 16, 5)
    masked_l1(dec(pair["features"], 12, pair["positions"]), pair["target_mel"], pair["mask"]).backward()
    for n, p in dec.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, n


def test_frozen_muq_encoder(monkeypatch):
    import muq_inversion.encoder as enc

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin, self.drop = nn.Linear(4, 4), nn.Dropout(0.5)

    monkeypatch.setattr(enc.MuQ, "from_pretrained", staticmethod(lambda _id: Tiny()))
    muq = enc.FrozenMuQEncoder("x", CPU)
    assert not muq.model.training and all(not p.requires_grad for p in muq.model.parameters())


def test_b_and_c_start_from_identical_weights_task22_order():
    def build(seed):
        seed_everything(seed, CPU)
        return MuQToMelTranslator(1024, 256, 100)

    b, c, other = build(0), build(0), build(1)
    assert state_dict_sha256(b) == state_dict_sha256(c) != state_dict_sha256(other)
    assert sum(p.numel() for p in b.parameters()) == 1_271_908  # TASK-22's translator


def test_task22_translator_has_no_regularizers_and_disable_is_explicit():
    dec = MuQToMelTranslator(8, 16, 5)
    spec = ArmSpec(name="C", learning_rate=1e-3, steps=1)
    audit = regularizer_audit(dec, spec)
    assert audit["dropout_modules"] == {} and audit["normalization_modules"] == {} and audit["buffers"] == []
    assert disable_regularizers(dec, spec) == []  # nothing to disable in the TASK-22 recipe
    net = nn.Sequential(nn.Linear(3, 3), nn.Dropout(0.3), nn.LayerNorm(3))
    spec2 = ArmSpec(name="C", learning_rate=1e-3, steps=1, weight_decay=0.1)
    changed = disable_regularizers(net, spec2)
    assert len(changed) == 2 and net[1].p == 0.0 and spec2.weight_decay == 0.0
    assert isinstance(net[2], nn.LayerNorm)  # normalization untouched


def test_mode_predictions_do_not_mutate_state_or_rng():
    torch.manual_seed(3)
    net = nn.Sequential(nn.Conv1d(2, 4, 3, padding=1), nn.BatchNorm1d(4), nn.Dropout(0.5))
    net.train()
    x = torch.randn(1, 2, 7)
    before = {k: v.clone() for k, v in net.state_dict().items()}
    random.seed(5)
    np.random.seed(5)
    torch.manual_seed(5)
    out = mode_predictions(net, lambda m: m(x), CPU)
    after_draws = (random.random(), np.random.rand(), torch.rand(1))
    random.seed(5)
    np.random.seed(5)
    torch.manual_seed(5)
    assert after_draws == (random.random(), np.random.rand(), torch.rand(1))  # RNG restored
    assert all(torch.equal(v, before[k]) for k, v in net.state_dict().items())  # BN running stats untouched
    assert net.training and not torch.allclose(out["train"], out["eval"])


def _toy_run(tmp_path, name, stop_after=None, steps=40, extension=None, criterion=0.01, stop_at_criterion=False):
    pair = toy_pair()
    seed_everything(0, CPU)
    dec = MuQToMelTranslator(8, 16, 5)
    spec = ArmSpec(name=name, learning_rate=1e-3, steps=steps, report_every=10, save_steps=(0, 10, 40),
                   latest_every_steps=10, extension_steps=extension, criterion_fraction=criterion,
                   stop_at_criterion=stop_at_criterion)
    pos = pair["positions"]
    return train_arm(spec, dec, lambda m: m(pair["features"], 12, pos), pair["target_mel"], pair["mask"],
                     boundary_frames(12, 2, pos, 6), tmp_path / name, CPU, stop_after_step=stop_after, log=lambda _s: None)


@pytest.fixture
def deterministic():
    """The interpolation backward accumulates with index_put, which is not deterministic multithreaded."""
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(prev)


def test_checkpoint_save_resume_is_exact(tmp_path, deterministic):
    ref = _toy_run(tmp_path, "ref")
    assert _toy_run(tmp_path, "resumed", stop_after=15) is None  # interrupted between latest saves
    res = _toy_run(tmp_path, "resumed")
    assert res["final_sha256"] == ref["final_sha256"] and res["init_sha256"] == ref["init_sha256"]
    drop = ("runtime_seconds",)
    hist = [[{k: v for k, v in json.loads(e).items() if k not in drop} for e in (tmp_path / d / "metrics.jsonl").read_text().splitlines()]
            for d in ("ref", "resumed")]
    assert hist[0] == hist[1]
    init = torch.load(tmp_path / "ref/checkpoints/milestone_000000.pt", weights_only=False)
    dec = MuQToMelTranslator(8, 16, 5)
    dec.load_state_dict(init["model_state_dict"])
    assert state_dict_sha256(dec) == ref["init_sha256"]
    for f in ("best.pt", "final.pt", "milestone_000010.pt", "milestone_000040.pt"):
        assert (tmp_path / "ref/checkpoints" / f).exists()
    assert ref["max_train_eval_abs_diff"] == 0.0  # no dropout/BN: train and eval mode agree exactly


def test_extension_and_criterion_rules(tmp_path):
    ext = _toy_run(tmp_path, "ext", steps=20, extension=40, criterion=1e-9)
    assert ext["extension"]["extended"] and ext["steps_run"] == 40 and ext["final"]["lr"] == pytest.approx(1e-4)
    stop = _toy_run(tmp_path, "crit", steps=40, criterion=0.999, stop_at_criterion=True)
    assert stop["criterion_met"] and stop["steps_run"] == stop["criterion"]["reached_at_step"] < 40
