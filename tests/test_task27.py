"""TASK-27: adversarial fine-tuning helpers (src/muq_inversion/task27.py). Synthetic tiny data on CPU."""

from __future__ import annotations

import json

import pytest
import torch

from muq_inversion.models.muq_mel_translator import MuQToMelTranslator
from muq_inversion.task23 import masked_l1, state_dict_sha256
from muq_inversion.task27 import (
    D_LAYERS,
    AdvSpec,
    FrozenD,
    align_conditioning,
    build_discriminator,
    calibrate_lambda,
    d_hinge_loss,
    d_input,
    d_scores,
    d_step,
    g_adv_loss,
    g_hinge_loss,
    lambda_at,
    logit_valid_mask,
    output_length,
    receptive_field,
    train_adversarial_arm,
    verify_start_checkpoint,
)
from muq_inversion.time_alignment import interpolate_at_positions

CPU = torch.device("cpu")
N, N_IN, T_MUQ, M, T_MEL = 8, 6, 12, 32, 40
D_CFG = {"cond_channels": 4, "channels": [8, 8, 8, 8], "leaky_slope": 0.2}


def prep(m: torch.Tensor) -> torch.Tensor:
    return d_input(m, -11.5129, -5.0, 3.0)


def tiny_data(seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {"fit_feats": torch.randn(N, T_MUQ, N_IN, generator=g), "fit_mel": torch.randn(N, M, T_MEL, generator=g) - 5,
            "mon_feats": torch.randn(4, T_MUQ, N_IN, generator=g), "mon_mel": torch.randn(4, M, T_MEL, generator=g) - 5,
            "mask": torch.ones(T_MEL, dtype=torch.bool), "boundary": torch.zeros(T_MEL, dtype=torch.bool),
            "positions": torch.linspace(0, T_MUQ - 1, T_MEL)}


def gen(seed: int = 0) -> MuQToMelTranslator:
    torch.manual_seed(seed)
    return MuQToMelTranslator(N_IN, 8, M)


def disc(seed: int = 0):
    return build_discriminator(D_CFG, N_IN, seed)


def spec(name: str, adv: bool, total: int = 64, **kw) -> AdvSpec:
    base = {"name": name, "adversarial": adv, "batch": 4, "base_lr": 1e-3, "betas": (0.9, 0.999), "total_presentations": total,
            "lr_drops": ((total // 2, 0.1), (3 * total // 4, 0.1)), "lam": 0.5 if adv else 0.0, "d_warmup_updates": 3,
            "log_every": 8, "fit_eval_every": 16, "monitor_eval_every": 32, "latest_every": 32, "chunk": 4, "showcase": (0, 1)}
    return AdvSpec(**(base | kw))


VALID = logit_valid_mask(torch.ones(T_MEL, dtype=torch.bool), M)


# --- identity ---------------------------------------------------------------------------------------

def test_start_checkpoint_identity_is_exact():
    g = gen()
    ck = {"model_state_dict": g.state_dict(), "spec": {"name": "C", "batch": 16}, "state": {"update": 51200}}
    exp = {"expected_sha256": state_dict_sha256(g), "expected_spec": {"name": "C", "batch": 16}, "expected_update": 51200}
    assert verify_start_checkpoint(g, ck, exp) == exp["expected_sha256"]
    with pytest.raises(ValueError, match="SHA-256"):
        verify_start_checkpoint(gen(1), ck, exp)
    with pytest.raises(ValueError, match="spec batch"):
        verify_start_checkpoint(g, ck | {"spec": {"name": "C", "batch": 1}}, exp)
    with pytest.raises(ValueError, match="update"):
        verify_start_checkpoint(g, ck | {"state": {"update": 100}}, exp)


def test_both_arms_start_from_identical_generator_weights(tmp_path):
    data, start = tiny_data(), gen().state_dict()
    out = {}
    for name, adv in (("A", False), ("B", True)):
        g = MuQToMelTranslator(N_IN, 8, M)
        g.load_state_dict(start)
        train_adversarial_arm(spec(name, adv), g, disc() if adv else None, data, tmp_path / name, CPU, prep, VALID, log=lambda *_: None)
        out[name] = torch.load(tmp_path / name / "checkpoints" / "init.pt", weights_only=False)
    assert out["A"]["sha256"] == out["B"]["sha256"]
    sa, sb = (torch.load(tmp_path / a / "preds" / "start.pt", weights_only=False) for a in "AB")
    assert torch.equal(sa["fit"], sb["fit"])


# --- gradient isolation -----------------------------------------------------------------------------

def test_discriminator_step_does_not_touch_generator():
    data, g, d = tiny_data(), gen(), disc()
    opt_d = torch.optim.Adam(d.parameters(), lr=1e-3, betas=(0.5, 0.9))
    before_g, before_d = state_dict_sha256(g), state_dict_sha256(d)
    pred = g(data["fit_feats"][:4], T_MEL, data["positions"])
    d_step(d, opt_d, data["fit_mel"][:4], pred, align_conditioning(data["fit_feats"][:4], data["positions"]), VALID, prep, 1e-3)
    assert state_dict_sha256(g) == before_g
    assert all(p.grad is None for p in g.parameters())
    assert state_dict_sha256(d) != before_d


def test_generator_adversarial_loss_freezes_discriminator_but_reaches_generator():
    data, g, d = tiny_data(), gen(), disc()
    before = {k: v.clone() for k, v in d.state_dict().items()}
    pred = g(data["fit_feats"][:4], T_MEL, data["positions"])
    adv = g_adv_loss(d, pred, align_conditioning(data["fit_feats"][:4], data["positions"]), VALID, prep)
    adv.backward()
    assert all(p.grad is None for p in d.parameters())
    assert all(p.requires_grad for p in d.parameters()) and d.training
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in g.parameters())
    assert all(torch.equal(v, before[k]) for k, v in d.state_dict().items())  # eval mode: no spectral-norm power iteration


def test_frozen_d_restores_flags_and_mode():
    d = disc()
    d.train()
    with FrozenD(d):
        assert not d.training and not any(p.requires_grad for p in d.parameters())
    assert d.training and all(p.requires_grad for p in d.parameters())


def test_features_get_no_gradient():
    data = tiny_data()
    f = data["fit_feats"][:2].clone().requires_grad_(True)
    c = align_conditioning(f, data["positions"])
    assert not c.requires_grad


# --- conditioning alignment and valid regions --------------------------------------------------------

def test_conditioning_uses_the_generators_frame_center_interpolation():
    data = tiny_data()
    f = data["fit_feats"][:3]
    assert torch.equal(align_conditioning(f, data["positions"]), interpolate_at_positions(f.transpose(1, 2), data["positions"]))


def test_receptive_field_and_grid_fit_the_crop():
    rf, jump = receptive_field(D_LAYERS)
    assert (rf, jump) == (28, 4)
    assert output_length(100) == 19 and output_length(187) == 40
    d = build_discriminator({"cond_channels": 16, "channels": [32, 64, 128, 128], "leaky_slope": 0.2}, 1024, 0)
    desc = d.describe(100, 187, 1000 * 256 / 24000)
    assert desc["fits_crop"] and desc["parameters"] == 334_097 and abs(desc["receptive_field_ms"] - 298.67) < 0.01


def test_logit_receptive_fields_match_the_valid_mask_empirically():
    d = disc()
    d.eval()
    mel, cond = torch.randn(1, M, T_MEL), torch.randn(1, N_IN, T_MEL)
    base = d(mel, cond)
    frame = 30
    mel2 = mel.clone()
    mel2[..., frame] += 5.0
    changed = (d(mel2, cond) - base).abs().amax(dim=(0, 1)) > 0  # per logit time index
    mask = torch.ones(T_MEL, dtype=torch.bool)
    mask[frame] = False
    invalid = ~logit_valid_mask(mask, M)[0, 0]
    assert torch.equal(changed, invalid)


def test_masked_logits_are_excluded_from_losses():
    lv = torch.ones(1, 2, 3, dtype=torch.bool)
    lv[..., 2] = False
    real, fake = torch.zeros(1, 2, 3), torch.zeros(1, 2, 3)
    real[..., 2], fake[..., 2] = -100.0, 100.0  # would dominate if not masked
    lr_, lf = d_hinge_loss(real, fake, lv)
    assert float(lr_) == pytest.approx(1.0) and float(lf) == pytest.approx(1.0)


def test_real_and_fake_get_identical_preprocessing():
    m = torch.tensor([[-20.0, -11.5129, -3.0]])
    assert torch.equal(prep(m), (m.clamp(min=-11.5129) + 5.0) / 3.0)


# --- losses and schedule ----------------------------------------------------------------------------

def test_hinge_formulas():
    v = torch.ones(1, 1, 4, dtype=torch.bool)
    real = torch.tensor([[[2.0, 0.5, -1.0, 1.0]]])
    fake = torch.tensor([[[-2.0, -0.5, 1.0, 0.0]]])
    lr_, lf = d_hinge_loss(real, fake, v)
    assert float(lr_) == pytest.approx((0 + 0.5 + 2 + 0) / 4)
    assert float(lf) == pytest.approx((0 + 0.5 + 2 + 1) / 4)
    assert float(g_hinge_loss(fake, v)) == pytest.approx(-(-2 - 0.5 + 1 + 0) / 4)


def test_lambda_ramp():
    assert lambda_at(1, 2.0, 10) == 0.0
    assert lambda_at(6, 2.0, 10) == pytest.approx(1.0)
    assert lambda_at(11, 2.0, 10) == 2.0 and lambda_at(500, 2.0, 10) == 2.0
    s = spec("B", True, total=51200 * 16, batch=16)
    assert s.updates == 51200 and s.ramp_updates == 5120


def test_calibration_measures_and_flags_degenerate():
    data, g, d = tiny_data(), gen(), disc()
    opt_d = torch.optim.Adam(d.parameters(), lr=1e-3, betas=(0.5, 0.9))
    cfg = {"warmup_updates": 5, "warmup_sampler_seed": "w", "n_batches": 3, "batch_sampler_seed": "c", "target_ratio": 0.1,
           "min_grad_norm": 1e-8, "max_saturation": 0.99}
    res = calibrate_lambda(g, d, opt_d, data, cfg, 1e-3, 4, prep, VALID, CPU)
    assert res["lambda"] > 0 and res["weighted_adv_over_l1_median"] == pytest.approx(0.1, rel=0.6)
    assert len(res["batches"]) == 3 and res["warmup"]["updates"] == 5
    bad = calibrate_lambda(gen(), disc(), torch.optim.Adam(disc().parameters()), data, cfg | {"min_grad_norm": 1e9}, 1e-3, 4, prep, VALID, CPU)
    assert bad["lambda"] is None and bad["problems"]


# --- exposure, order, resume, evaluation ---------------------------------------------------------------

TIMING = ("wall_seconds", "updates_per_second", "presentations_per_second")


def _clean(row):
    return {k: _clean(v) if isinstance(v, dict) else v for k, v in row.items() if k not in TIMING}


def _strip(path):
    """Rows without timing fields (also nested), serialized so that NaN entries compare equal."""
    return [json.dumps(_clean(json.loads(x)), sort_keys=True) for x in path.read_text().splitlines()]


def test_exposure_and_sample_order_match_between_arms(tmp_path):
    data, start = tiny_data(), gen().state_dict()
    s = {}
    for name, adv in (("A", False), ("B", True)):
        g = MuQToMelTranslator(N_IN, 8, M)
        g.load_state_dict(start)
        s[name] = train_adversarial_arm(spec(name, adv), g, disc() if adv else None, data, tmp_path / name, CPU, prep, VALID, log=lambda *_: None)
    assert s["A"]["order_sha256"] == s["B"]["order_sha256"]
    assert s["A"]["presentations"] == s["B"]["presentations"] == 64
    assert s["A"]["presentation_counts_min_max"] == s["B"]["presentation_counts_min_max"] == [8, 8]
    assert s["A"]["final_sha256"] != s["B"]["final_sha256"]


def test_resume_is_bit_exact_for_the_adversarial_arm(tmp_path):
    torch.use_deterministic_algorithms(True)
    try:
        data, start = tiny_data(), gen().state_dict()
        runs = {}
        for tag, stop in (("ref", None), ("res", 5)):
            g = MuQToMelTranslator(N_IN, 8, M)
            g.load_state_dict(start)
            out = train_adversarial_arm(spec("B", True), g, disc(), data, tmp_path / tag, CPU, prep, VALID, stop_after_update=stop,
                                        log=lambda *_: None)
            if stop is not None:
                assert out is None
                g = MuQToMelTranslator(N_IN, 8, M)  # fresh objects, everything from latest.pt
                g.load_state_dict(start)
                out = train_adversarial_arm(spec("B", True), g, disc(7), data, tmp_path / tag, CPU, prep, VALID, log=lambda *_: None)
            runs[tag] = out
        a, b = runs["ref"], runs["res"]
        assert a["final_sha256"] == b["final_sha256"] and a["discriminator_final_sha256"] == b["discriminator_final_sha256"]
        assert a["order_sha256"] == b["order_sha256"] and a["presentation_counts_min_max"] == b["presentation_counts_min_max"]
        for f in ("fit_eval", "monitor_eval", "train_log"):
            assert _strip(tmp_path / "ref" / f"{f}.jsonl") == _strip(tmp_path / "res" / f"{f}.jsonl"), f
    finally:
        torch.use_deterministic_algorithms(False)


def test_lr_schedule_applies_to_both_optimizers(tmp_path):
    data = tiny_data()
    s = train_adversarial_arm(spec("B", True), gen(), disc(), data, tmp_path / "B", CPU, prep, VALID, log=lambda *_: None)
    rows = [json.loads(r) for r in _strip(tmp_path / "B" / "train_log.jsonl")]
    lrs = {r["presentations"]: (r["lr"], r["d_lr"]) for r in rows}
    assert lrs[8] == (1e-3, 1e-4) and lrs[40] == pytest.approx((1e-4, 1e-5)) and lrs[64] == pytest.approx((1e-5, 1e-6))
    assert s["d_warmup"]["updates"] == 3


def test_evaluation_does_not_mutate_discriminator():
    data, d = tiny_data(), disc()
    before = state_dict_sha256(d)
    d_scores(d, data["fit_feats"], data["fit_mel"], data["fit_mel"] + 0.1, data["positions"], VALID, prep, CPU, 3)
    assert state_dict_sha256(d) == before and d.training


def test_l1_term_is_the_existing_masked_l1(tmp_path):
    data, g = tiny_data(), gen()
    pred = g(data["fit_feats"][:4], T_MEL, data["positions"])
    assert torch.equal(masked_l1(pred, data["fit_mel"][:4], data["mask"]), (pred - data["fit_mel"][:4]).abs().mean())
