import torch

from muq_inversion.cache import (
    CacheKeyInputs,
    cache_path,
    compute_cache_key,
    load,
    save_atomic,
)


def test_compute_cache_key_deterministic():
    inputs = CacheKeyInputs(
        audio_checksum="abc123",
        muq_model_id="OpenMuQ/MuQ-large-msd-iter",
        preprocessing={"target_sample_rate": 24000},
        representation_name="last_hidden_state",
        dtype="float32",
    )
    assert compute_cache_key(inputs) == compute_cache_key(inputs)


def test_compute_cache_key_changes_with_any_input():
    base = CacheKeyInputs(
        audio_checksum="abc123",
        muq_model_id="OpenMuQ/MuQ-large-msd-iter",
        preprocessing={"target_sample_rate": 24000},
        representation_name="last_hidden_state",
        dtype="float32",
    )
    variants = [
        CacheKeyInputs(**{**base.__dict__, "audio_checksum": "different"}),
        CacheKeyInputs(**{**base.__dict__, "muq_model_id": "different"}),
        CacheKeyInputs(**{**base.__dict__, "preprocessing": {"target_sample_rate": 22050}}),
        CacheKeyInputs(**{**base.__dict__, "representation_name": "different"}),
        CacheKeyInputs(**{**base.__dict__, "dtype": "float16"}),
    ]
    base_key = compute_cache_key(base)
    for variant in variants:
        assert compute_cache_key(variant) != base_key


def test_save_and_load_round_trip(tmp_path):
    features = torch.randn(1, 50, 1024, dtype=torch.float32)
    metadata = {"track_id": "blues.00000", "duration_label": "full"}
    key = "deadbeef"
    path = cache_path(tmp_path, key)

    save_atomic(path, features, metadata)
    loaded_features, loaded_metadata = load(path)

    assert torch.equal(loaded_features, features)
    assert loaded_metadata == metadata


def test_load_missing_returns_none(tmp_path):
    path = cache_path(tmp_path, "doesnotexist")
    assert load(path) is None


def test_save_atomic_leaves_no_tmp_file_on_success(tmp_path):
    features = torch.zeros(1, 10, 4)
    path = cache_path(tmp_path, "somekey")
    save_atomic(path, features, {})
    remaining = list(path.parent.iterdir())
    assert remaining == [path]
