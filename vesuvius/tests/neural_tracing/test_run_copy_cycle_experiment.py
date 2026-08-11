import copy
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pytest

from vesuvius.neural_tracing.evaluation.copy_cycle_calibration import (
    CalibrationRows,
    fit_calibration_bundle,
)
from vesuvius.neural_tracing.evaluation.copy_cycle_io import sha256_file
from vesuvius.neural_tracing.inference.run_copy_cycle_experiment import (
    load_and_validate_test_authorization,
    phase_source_wraps,
    validate_config_structure,
    validate_learned_test_authorization,
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


def test_learned_test_authorization_binds_model_score_receipt_and_gate(tmp_path):
    commit = "a" * 40
    checkpoint_sha = "c" * 64
    manifest_sha = "b" * 64
    receipt_sha = "d" * 64
    score_sha = "e" * 64
    model_sha = "f" * 64
    copy_args = {"tta": True, "batch_size": 1}
    directions = ("1->2", "2->1", "2->3", "3->2", "3->4", "4->3")
    rng = np.random.default_rng(20260811)
    rows = CalibrationRows(
        cycle_local=rng.normal(size=(60, 3)),
        displacement_local=rng.normal(size=(60, 3)),
        target_correction_local=rng.normal(size=(60, 3)),
    )
    training = {
        "stage": "final",
        "directions": list(directions),
        "sources": [1, 2, 3, 4],
        "receipts": [{"path": "/tmp/development.json", "sha256": "9" * 64}],
        "scroll_id": "PHerc0500P2",
        "inference_implementation_commit": commit,
        "checkpoint_sha256": checkpoint_sha,
        "manifest_sha256": manifest_sha,
        "requested_copy_args": copy_args,
        "volume_path": "https://example.test/volume.zarr/",
        "volume_backend": "zarr",
        "volume_scale_requested": 0,
        "volume_scale_resolved": 0,
        "tifxyz_voxel_size_um": 4.317,
        "crop_size": [128, 384, 384],
        "runtime": {"torch": "test"},
        "rows_by_direction": {direction: 10 for direction in directions},
        "total_rows": 60,
    }
    model = fit_calibration_bundle(
        rows,
        implementation_commit=commit,
        training=training,
    )
    validation_receipt = {
        "completed": True,
        "phase": "validation",
        "scroll_id": "PHerc0500P2",
        "implementation_commit": commit,
        "checkpoint_sha256": checkpoint_sha,
        "manifest_sha256": manifest_sha,
        "requested_copy_args": copy_args,
        "volume_path": "https://example.test/volume.zarr/",
        "volume_backend": "zarr",
        "volume_scale_requested": 0,
        "volume_scale_resolved": 0,
        "copy_args": {"tifxyz_voxel_size_um": 4.317},
        "crop_size": [128, 384, 384],
        "runtime": {"torch": "test"},
        "sources": [{"wrap": 5}, {"wrap": 6}, {"wrap": 7}],
    }

    def arm(penalty):
        return {
            "penalized_target_distance_mean": penalty,
            "valid_prediction_cells": 100,
        }

    direction_arms = {
        "baseline": arm(10.0),
        "combined": arm(8.0),
        "displacement_only": arm(9.0),
        "cycle_only": arm(8.7),
        "fitted_scalar": arm(9.2),
        "physical_scalar": arm(9.5),
    }
    aggregate = {
        name: {
            "penalized_target_distance_mean": values[
                "penalized_target_distance_mean"
            ],
            "target_distance_p95_valid": 10.0,
            "sheet_switch_rate_all_eligible": 0.004,
        }
        for name, values in direction_arms.items()
    }
    validation_score = {
        "schema_version": 1,
        "method": "copy_cycle_local_linear_v1_score",
        "stage": "validation",
        "implementation_commit": commit,
        "model_sha256": model_sha,
        "manifest_sha256": manifest_sha,
        "receipts": [{"path": "/tmp/validation.json", "sha256": receipt_sha}],
        "scroll_id": "PHerc0500P2",
        "directions": [
            {"source": source, "target": target, "arms": direction_arms}
            for source, target in ((5, 6), (6, 5), (6, 7), (7, 6))
        ],
        "aggregate": aggregate,
        "gate": {
            "passed": True,
            "conditions": {
                "aggregate_penalty_improves_at_least_10pct": True,
                "required_directions_improve": True,
                "beats_each_displacement_control": True,
                "incremental_gain_over_best_control_at_least_1pct_baseline": True,
                "coverage_unchanged_each_direction": True,
                "p95_noninferiority": True,
                "sheet_switch_gate": True,
            },
            "improved_directions": 4,
            "required_improved_directions": 3,
        },
    }
    authorization = {
        "schema_version": 1,
        "method": "copy_cycle_local_linear_v1",
        "status": "authorized",
        "validation_status": "validation_positive",
        "implementation_commit": commit,
        "calibration_model_sha256": model_sha,
        "validation_score_sha256": score_sha,
        "validation_receipt_sha256": receipt_sha,
        "validation_public_url": "https://github.com/example/project/tree/results",
        "overlap_audit_utc": "2026-08-11T01:00:00Z",
    }
    validation_args = {
        "calibration_model": model,
        "validation_score": validation_score,
        "validation_receipt": validation_receipt,
        "actual_calibration_model_sha256": model_sha,
        "actual_validation_score_sha256": score_sha,
        "actual_validation_receipt_sha256": receipt_sha,
        "expected_checkpoint_sha256": checkpoint_sha,
        "expected_validation_manifest_sha256": manifest_sha,
        "expected_copy_args": copy_args,
        "authorization_now_utc": datetime(2026, 8, 11, 2, tzinfo=timezone.utc),
    }

    assert validate_learned_test_authorization(
        authorization, commit, **validation_args
    ) == {"method": "copy_cycle_local_linear_v1", "model_sha256": model_sha}
    modified_score_args = copy.deepcopy(validation_args)
    modified_score_args["validation_score"]["gate"]["conditions"][
        "p95_noninferiority"
    ] = False
    with pytest.raises(ValueError, match="every frozen gate"):
        validate_learned_test_authorization(
            authorization, commit, **modified_score_args
        )
    modified_authorization = copy.deepcopy(authorization)
    modified_authorization["calibration_model_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="model file SHA-256 mismatch"):
        validate_learned_test_authorization(
            modified_authorization, commit, **validation_args
        )

    model_path = tmp_path / "model.json"
    receipt_path = tmp_path / "receipt.json"
    score_path = tmp_path / "score.json"
    authorization_path = tmp_path / "authorization.json"
    model_path.write_text(json.dumps(model.to_json(), sort_keys=True), encoding="utf-8")
    receipt_path.write_text(
        json.dumps(validation_receipt, sort_keys=True), encoding="utf-8"
    )
    file_score = copy.deepcopy(validation_score)
    file_score["model_sha256"] = sha256_file(model_path)
    file_score["receipts"][0]["sha256"] = sha256_file(receipt_path)
    score_path.write_text(json.dumps(file_score, sort_keys=True), encoding="utf-8")
    file_authorization = copy.deepcopy(authorization)
    file_authorization.update(
        {
            "calibration_model_path": model_path.name,
            "calibration_model_sha256": sha256_file(model_path),
            "validation_score_path": score_path.name,
            "validation_score_sha256": sha256_file(score_path),
            "validation_receipt_path": receipt_path.name,
            "validation_receipt_sha256": sha256_file(receipt_path),
            "overlap_audit_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    authorization_path.write_text(
        json.dumps(file_authorization, sort_keys=True), encoding="utf-8"
    )

    assert load_and_validate_test_authorization(
        authorization_path,
        commit,
        expected_checkpoint_sha256=checkpoint_sha,
        expected_validation_manifest_sha256=manifest_sha,
        expected_copy_args=copy_args,
    ) == {
        "method": "copy_cycle_local_linear_v1",
        "model_sha256": sha256_file(model_path),
    }
