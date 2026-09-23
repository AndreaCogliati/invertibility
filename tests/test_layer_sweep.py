import csv
from pathlib import Path

from muq_inversion.layer_sweep_common import (
    CheckpointRecord,
    layer_cache_label,
    select_best_checkpoint,
    select_evenly_spaced_layers,
)

# --- exact hidden-layer selection ---


def test_select_evenly_spaced_layers_matches_measured_muq_structure():
    """MuQ has 12 transformer layers (measured in scripts/inspect_muq_hidden_states.py); this
    experiment's 5 candidates must be exactly [1, 4, 6, 9, 12], with the final one equal to the
    total transformer-layer count (last_hidden_state)."""
    assert select_evenly_spaced_layers(12, 5) == [1, 4, 6, 9, 12]


def test_select_evenly_spaced_layers_excludes_embedding_output():
    """Index 0 (the frontend/embedding output, before any transformer block) must never be
    selected -- the earliest candidate is index 1, the first transformer block's output."""
    for num_layers in range(2, 25):
        indices = select_evenly_spaced_layers(num_layers, num_candidates=5)
        assert 0 not in indices
        assert min(indices) >= 1


def test_select_evenly_spaced_layers_final_is_last_transformer_layer():
    for num_layers in (6, 12, 24, 48):
        indices = select_evenly_spaced_layers(num_layers, num_candidates=5)
        assert indices[-1] == num_layers


# --- deterministic candidate selection ---


def test_select_evenly_spaced_layers_is_deterministic():
    results = [select_evenly_spaced_layers(12, 5) for _ in range(10)]
    assert all(r == results[0] for r in results)


def test_select_evenly_spaced_layers_rejects_too_few_candidates():
    import pytest

    with pytest.raises(ValueError):
        select_evenly_spaced_layers(12, num_candidates=1)


# --- cache separation by layer / no accidental use of the final layer for every configuration ---


def test_layer_cache_label_differs_per_layer_index():
    labels = {
        layer_idx: layer_cache_label("val_pop.00003_t5.0s", layer_idx, "deadbeef" * 8, "OpenMuQ/MuQ-large-msd-iter", "0.1.0", "torch.float32")
        for layer_idx in (1, 4, 6, 9, 12)
    }
    assert len(set(labels.values())) == 5, "every candidate layer must produce a distinct cache key"


def test_layer_cache_label_includes_required_provenance_fields():
    label = layer_cache_label("val_pop.00003_t5.0s", 6, "deadbeef" * 8, "OpenMuQ/MuQ-large-msd-iter", "0.1.0", "torch.float32")
    assert "val_pop.00003_t5.0s" in label  # crop label (carries crop boundaries/split/track)
    assert "L6" in label  # exact hidden-state index
    assert "deadbeef" in label  # source-audio checksum
    assert "OpenMuQ/MuQ-large-msd-iter" in label  # MuQ checkpoint identifier
    assert "0.1.0" in label  # MuQ package version
    assert "torch.float32" in label  # dtype


def test_layer_cache_label_differs_per_crop():
    a = layer_cache_label("val_pop.00003_t5.0s", 6, "deadbeef" * 8, "OpenMuQ/MuQ-large-msd-iter", "0.1.0", "torch.float32")
    b = layer_cache_label("val_pop.00003_t18.0s", 6, "deadbeef" * 8, "OpenMuQ/MuQ-large-msd-iter", "0.1.0", "torch.float32")
    assert a != b


# --- shape consistency (measured, not assumed -- see scripts/inspect_muq_hidden_states.py) ---


def test_inspection_report_confirms_uniform_shape_across_candidates():
    """Reads the committed inspection report (if present) and asserts all 5 candidate layers
    share the same channel count and frame rate -- the precondition this experiment relies on to
    avoid needing a shape adapter. Skipped if the report hasn't been generated yet."""
    import json
    from pathlib import Path

    report_path = Path(__file__).resolve().parents[1] / "artifacts/muq_layer_sweep/hidden_states_inspection.json"
    if not report_path.exists():
        import pytest

        pytest.skip("hidden_states_inspection.json not generated in this environment")
    report = json.loads(report_path.read_text())
    candidates = report["candidate_layers"]
    channels = {c["channels"] for c in candidates}
    frame_rates = {c["frames_per_second"] for c in candidates}
    assert len(channels) == 1, f"expected one shared channel count, got {channels}"
    assert len(frame_rates) == 1, f"expected one shared frame rate, got {frame_rates}"
    assert report["shape_adapter_needed"] is False


# --- identical split/crop manifests across runs ---


def test_layer_sweep_reuses_more_data_manifest_unchanged():
    """This experiment must reuse experiment/muq-to-dac-more-data's exact manifest and DAC
    cache, not a rebuilt or modified one. `scripts/` has no `__init__.py` (not an importable
    package in this project), so this checks the script's source text directly rather than
    importing it."""
    script_path = Path(__file__).resolve().parents[1] / "scripts/train_layer_sweep.py"
    source = script_path.read_text()
    assert 'MANIFEST_PATH = REPO_ROOT / "data/manifests/more_data_crops.csv"' in source
    assert 'DAC_CACHE_PATH = REPO_ROOT / "data/cache/more_data/muq_dac_cache.pt"' in source


def test_manifest_track_disjoint_across_splits():
    manifest_path = (Path(__file__).resolve().parents[1] / "data/manifests/more_data_crops.csv").resolve()
    if not manifest_path.exists():
        import pytest

        pytest.skip("more_data_crops.csv not present in this environment")
    track_to_splits: dict[str, set[str]] = {}
    with open(manifest_path, newline="") as f:
        for row in csv.DictReader(f):
            track_to_splits.setdefault(row["track_id"], set()).add(row["split"])
    offending = {t: s for t, s in track_to_splits.items() if len(s) > 1}
    assert not offending, f"tracks spanning multiple splits: {offending}"


# --- validation-only checkpoint selection ---


def test_select_best_checkpoint_picks_lowest_val_mse():
    history = [
        CheckpointRecord(step=100, val_mse=20.0),
        CheckpointRecord(step=200, val_mse=15.0),
        CheckpointRecord(step=300, val_mse=17.0),
    ]
    best = select_best_checkpoint(history)
    assert best.step == 200
    assert best.val_mse == 15.0


def test_select_best_checkpoint_never_sees_test_metrics():
    """CheckpointRecord structurally carries no test-related field, so checkpoint selection
    cannot accidentally use test performance -- a type-level guarantee, not just a convention."""
    assert set(CheckpointRecord.__dataclass_fields__.keys()) == {"step", "val_mse"}


def test_select_best_checkpoint_breaks_ties_by_earliest_step():
    history = [
        CheckpointRecord(step=300, val_mse=15.0),
        CheckpointRecord(step=100, val_mse=15.0),
        CheckpointRecord(step=200, val_mse=15.0),
    ]
    assert select_best_checkpoint(history).step == 100


def test_select_best_checkpoint_rejects_empty_history():
    import pytest

    with pytest.raises(ValueError):
        select_best_checkpoint([])
