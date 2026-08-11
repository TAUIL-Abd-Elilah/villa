import copy
from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from vesuvius.neural_tracing.inference.run_copy_cycle_experiment import (
    phase_source_wraps,
    validate_config_structure,
    validate_test_authorization,
)


def _manifest():
    return {
        "scroll_id": "Scroll",
        "volume_url": "https://example.test/volume.zarr/",
        "volume_scale": 1,
        "voxel_size_um": 4.0,
        "splits": {
            "development_edges": [[1, 2]],
            "validation_edges": [[3, 4], [4, 5]],
            "sealed_test_edges": [[7, 8], [8, 9]],
        },
    }


def _config(phase, source_wraps):
    return {
        "schema_version": 1,
        "scroll_id": "Scroll",
        "phase": phase,
        "source_wraps": source_wraps,
        "volume_path": "https://example.test/volume.zarr/",
        "volume_scale": 1,
        "tifxyz_voxel_size_um": 4.0,
        "copy_args": {"tta": True, "batch_size": 1},
    }


def test_phase_source_wraps_uses_edge_endpoints():
    assert phase_source_wraps(_manifest(), "validation") == {3, 4, 5}


def test_validation_requires_complete_source_block():
    with pytest.raises(ValueError, match="complete source set"):
        validate_config_structure(_config("validation", [3, 4]), _manifest())

    phase, sources = validate_config_structure(
        _config("validation", [5, 3, 4]), _manifest()
    )
    assert phase == "validation"
    assert sources == [3, 4, 5]


def test_runner_owned_copy_arguments_are_rejected():
    config = _config("development", [1])
    config["copy_args"]["iterations"] = 2

    with pytest.raises(ValueError, match="runner-owned"):
        validate_config_structure(config, _manifest())


def test_tta_cannot_be_disabled_with_non_boolean_false_value():
    config = _config("development", [1])
    config["copy_args"]["tta"] = 0

    with pytest.raises(ValueError, match="TTA enabled"):
        validate_config_structure(config, _manifest())


def test_phase_example_configs_are_valid_and_freeze_identical_copy_args():
    docs = Path(__file__).resolve().parents[2] / "docs"
    config_names = (
        "copy_cycle_development_config.example.json",
        "copy_cycle_validation_config.example.json",
        "copy_cycle_pherc0343p_test_config.example.json",
    )
    configs = [
        json.loads((docs / name).read_text(encoding="utf-8"))
        for name in config_names
    ]
    for config in configs:
        manifest = json.loads(
            (docs / config["manifest"]).read_text(encoding="utf-8")
        )
        validate_config_structure(config, manifest)

    assert configs[0]["copy_args"] == configs[1]["copy_args"] == configs[2]["copy_args"]


def test_sealed_test_authorization_is_commit_bound():
    validation_receipt_sha = "d" * 64
    checkpoint_sha = "c" * 64
    manifest_sha = "b" * 64
    authorization = {
        "schema_version": 1,
        "status": "authorized",
        "validation_status": "validation_positive",
        "implementation_commit": "abc123",
        "selected": {"alpha": 0.5, "tau": 24.0},
        "validation_score_sha256": "a" * 64,
        "validation_receipt_sha256": validation_receipt_sha,
        "validation_public_url": "https://github.com/example/project/tree/results",
        "overlap_audit_utc": "2026-08-11T01:00:00Z",
    }
    validation_score = {
        "schema_version": 1,
        "mode": "grid",
        "phase": "validation",
        "scroll_id": "PHerc0500P2",
        "implementation_commit": "abc123",
        "checkpoint_sha256": checkpoint_sha,
        "manifest_sha256": manifest_sha,
        "receipt_sha256": validation_receipt_sha,
        "direction_identity": [
            {"source": 5, "target": 6},
            {"source": 6, "target": 5},
            {"source": 6, "target": 7},
            {"source": 7, "target": 6},
        ],
        "result": {
            "selection": {
                "status": "validation_positive",
                "selected": {"alpha": 0.5, "tau": 24.0},
            }
        },
    }
    validation_receipt = {
        "completed": True,
        "phase": "validation",
        "scroll_id": "PHerc0500P2",
        "implementation_commit": "abc123",
        "checkpoint_sha256": checkpoint_sha,
        "manifest_sha256": manifest_sha,
        "requested_copy_args": {"tta": True, "batch_size": 1},
        "sources": [{"wrap": 5}, {"wrap": 6}, {"wrap": 7}],
    }
    validation_args = {
        "validation_score": validation_score,
        "validation_receipt": validation_receipt,
        "actual_validation_score_sha256": "a" * 64,
        "actual_validation_receipt_sha256": validation_receipt_sha,
        "expected_checkpoint_sha256": checkpoint_sha,
        "expected_validation_manifest_sha256": manifest_sha,
        "expected_copy_args": {"tta": True, "batch_size": 1},
        "authorization_now_utc": datetime(2026, 8, 11, 2, tzinfo=timezone.utc),
    }

    assert validate_test_authorization(
        authorization, "abc123", **validation_args
    ) == (0.5, 24.0)
    modified = copy.deepcopy(authorization)
    modified["implementation_commit"] = "different"
    with pytest.raises(ValueError, match="checked-out commit"):
        validate_test_authorization(modified, "abc123", **validation_args)

    changed_settings = copy.deepcopy(validation_args)
    changed_settings["expected_copy_args"] = {"tta": True, "batch_size": 2}
    with pytest.raises(ValueError, match="exactly match"):
        validate_test_authorization(
            authorization, "abc123", **changed_settings
        )
