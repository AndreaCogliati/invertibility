import torch

from muq_inversion.checkpoint import load_checkpoint, save_checkpoint_atomic


def test_save_checkpoint_round_trip(tmp_path):
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.Adam(model.parameters())
    state = {
        "generator_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": 42,
        "resolved_config": {"name": "test"},
        "manifest_hash": "abc",
        "uv_lock_hash": "def",
    }
    path = tmp_path / "checkpoint.pt"
    save_checkpoint_atomic(path, state)

    loaded = load_checkpoint(path)
    assert loaded["step"] == 42
    assert loaded["manifest_hash"] == "abc"
    assert set(loaded["generator_state_dict"].keys()) == set(model.state_dict().keys())


def test_save_checkpoint_leaves_no_tmp_file(tmp_path):
    state = {
        "generator_state_dict": {},
        "optimizer_state_dict": {},
        "step": 1,
        "resolved_config": {},
        "manifest_hash": "x",
        "uv_lock_hash": "y",
    }
    path = tmp_path / "checkpoint.pt"
    save_checkpoint_atomic(path, state)
    remaining = list(tmp_path.iterdir())
    assert remaining == [path]


def test_save_checkpoint_rejects_missing_required_keys(tmp_path):
    import pytest

    path = tmp_path / "checkpoint.pt"
    with pytest.raises(RuntimeError):
        save_checkpoint_atomic(path, {"step": 1})
    assert not path.exists()
