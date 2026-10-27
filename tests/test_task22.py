"""TASK-22: layer selection on a real (tiny) transformers conformer, cache separation, config diff,
split disjointness, alignment, and window/matched-step selection helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch
from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer import (
    Wav2Vec2ConformerConfig,
    Wav2Vec2ConformerEncoder,
)

from muq_inversion.task22 import (
    checkpoint_steps,
    config_diff,
    feature_cache_stem,
    hidden_state_with_check,
    label_at_step,
    resolve_muq_layer,
    window_best,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from train_task18_heldout import feature_cache_path, load_config

T21 = load_config(REPO_ROOT / "configs/task21.yaml")
T22 = load_config(REPO_ROOT / "configs/task22.yaml")


def arm(cfg: dict, name: str) -> dict:
    return {**cfg, **cfg["arms"][name]}


@pytest.fixture(scope="module")
def tiny_conformer():
    torch.manual_seed(0)
    cfg = Wav2Vec2ConformerConfig(hidden_size=16, num_hidden_layers=3, num_attention_heads=2, intermediate_size=32,
                                  conv_depthwise_kernel_size=3, position_embeddings_type="relative")
    enc = Wav2Vec2ConformerEncoder(cfg).eval()
    return enc


def test_resolve_layer():
    assert resolve_muq_layer(None, 12) is None
    r = resolve_muq_layer({"block": 1}, 12)
    assert r["hidden_state_index"] == 1 and r["module"] == "model.model.conformer.layers.0" and r["tag"] == "L1"
    for bad in (0, 12, "1", 1.0):
        with pytest.raises(ValueError):
            resolve_muq_layer({"block": bad}, 12)


def test_hidden_states_index_semantics_on_installed_transformers(tiny_conformer):
    """Index 0 is the input of block 1, index k the output of block k, and the last entry carries the
    encoder LayerNorm -- the convention the TASK-22 extraction relies on (and checks on every batch)."""
    enc = tiny_conformer
    x = torch.randn(2, 7, 16)
    with torch.inference_mode():
        hs = enc(x, output_hidden_states=True).hidden_states
        rel = enc.embed_positions(x)
        manual = [x]
        for blk in enc.layers:
            manual.append(blk(manual[-1], relative_position_embeddings=rel)[0])
        normed = enc.layer_norm(manual[-1])
    assert len(hs) == len(enc.layers) + 1
    assert torch.equal(hs[0], x) and torch.equal(hs[1], manual[1]) and torch.equal(hs[2], manual[2])
    assert not torch.equal(hs[-1], manual[-1]) and torch.equal(hs[-1], normed)
    f, check = hidden_state_with_check(lambda t: enc(t, output_hidden_states=True).hidden_states, enc.layers, x, 1)
    assert torch.equal(f, manual[1]) and check["index_is_block_output"] and check["index0_is_block1_input"]
    assert f.shape[:2] == x.shape[:2]  # same sequence length/order: frontend token centers still apply


def test_wrong_index_convention_is_detected(tiny_conformer):
    enc = tiny_conformer
    x = torch.randn(1, 5, 16)

    def shifted(t):  # a library that dropped the frontend entry: index 1 would be block 2's output
        hs = enc(t, output_hidden_states=True).hidden_states
        return hs[1:] + (hs[-1],)

    with pytest.raises(RuntimeError, match="STOP"):
        hidden_state_with_check(shifted, enc.layers, x, 1)


def test_cache_separation_and_baseline_name_unchanged():
    keys = [["data/raw/a.wav", 0], ["data/raw/a.wav", 960]]
    base_parts = {"version": 1, "muq": "m", "bigvgan": "b", "num_hops": 187, "backend": "xpu", "torch": "2"}
    layer = resolve_muq_layer({"block": 1}, 12)
    l1_parts = base_parts | {"muq_layer": layer["name"], "hidden_state_index": 1}
    p_base = feature_cache_path(arm(T22, "T21eval"), base_parts, keys)
    p_l1 = feature_cache_path(arm(T22, "L1"), l1_parts, keys, layer)
    digest = hashlib.sha256(json.dumps({**base_parts, "keys": keys}, sort_keys=True).encode()).hexdigest()[:20]
    assert p_base == REPO_ROOT / "data/cache/task18" / f"task18_v1_{digest}.pt"  # TASK-18..21 naming, cache still hit
    assert p_l1.parent == REPO_ROOT / "data/cache/task22" and p_l1.name.startswith("task18_v1_L1_")
    assert p_l1.name.split("_")[-1] != p_base.name.split("_")[-1]  # different digest, too
    assert feature_cache_stem(None) == "task18_v1"


def test_only_the_representation_and_duration_changed_from_task21():
    l1, t21 = arm(T22, "L1"), arm(T21, "N512ext")
    changed = {d["key"] for d in config_diff(t21, l1)}
    allowed = {"task", "init_from_checkpoint", "steps", "runtime_budget_hours", "checkpoint_policy", "muq_layer",
               "feature_cache_dir", "extra_checkpoints", "arms", "arm", "run_name_prefix", "run_root", "artifact_root",
               "results_name", "wandb_group", "wandb_tags"}
    assert changed <= allowed, changed - allowed
    for k in ("track_manifest", "crop_manifest", "eval_sets", "learning_rate", "adam_betas", "batch_size", "seed",
              "hidden_channels", "temporal_alignment", "decode_clamp_to_mel_floor", "loss", "train_tracks", "lr_schedule",
              "eval_schedule", "log_schedule", "quick_eval_sets", "num_mel_hops", "muq_model_id", "bigvgan_revision"):
        assert l1[k] == t21[k], k
    assert l1["muq_layer"] == {"block": 1} and l1["feature_cache_dir"] != t21["feature_cache_dir"]
    assert l1["steps"] == 200000 and l1["early_stopping"] is None and l1["init_from_checkpoint"] is None
    assert {60000, 200000} <= set(l1["checkpoint_policy"]["milestones"])
    assert not {"test", "test2", "val2"} & set(l1["quick_eval_sets"])
    base = arm(T22, "T21eval")
    assert base["muq_layer"] is None and base["feature_cache_dir"] == t21["feature_cache_dir"]
    assert base["evaluate_checkpoints"] == "runs/task21/task21_N512ext_v1/checkpoints"


def test_splits_disjoint():
    with open(REPO_ROOT / T22["crop_manifest"], newline="") as f:
        crops = list(csv.DictReader(f))
    train = {r["track_id"] for r in crops if r["split"] == "train"}
    held = {r["track_id"] for r in crops if r["eval_set"] in ("val", "test", "val2", "test2")}
    assert len(train) == 512 and held and not train & held
    pool = {r["track_id"] for r in crops if r["role"] == "train_pool"}
    assert pool == train


def test_window_best_and_matched_labels():
    hist = [{"step": 250, "val/correct/mel_l1": 0.5}, {"step": 500, "val/correct/mel_l1": 0.4},
            {"step": 1000, "val/correct/mel_l1": 0.4}, {"step": 3000, "val/correct/mel_l1": 0.3}]
    assert window_best(hist, 2000)["step"] == 500 and window_best(hist, 10**9)["step"] == 3000
    with pytest.raises(ValueError):
        window_best(hist, 100)
    rep = {"best_checkpoint": {"step": 60000}, "final_step": 200000,
           "extra_checkpoints": {"step_60000": {"step": 60000, "available": True},
                                 "step_150000": {"step": 150000, "available": False}}}
    steps = checkpoint_steps(rep)
    assert "step_150000" not in steps
    assert label_at_step(steps, 60000) == "step_60000" and label_at_step(steps, 200000) == "final"
    assert label_at_step(steps, 150000) is None


@pytest.mark.skipif(not (REPO_ROOT / "artifacts/task21/task21_N512ext_v1/report.json").exists(), reason="TASK-21 artifacts absent")
def test_task21_window_best_is_its_best_checkpoint():
    rep = json.loads((REPO_ROOT / "artifacts/task21/task21_N512ext_v1/report.json").read_text())
    assert window_best(rep["history"], 200000)["step"] == rep["best_checkpoint"]["step"] == 60000
