import copy

import numpy as np
import pytest

from vesuvius.neural_tracing.evaluation.copy_cycle_calibration import (
    CalibrationBundle,
    CalibrationRows,
    LocalLinearModel,
    apply_local_linear_model,
    apply_scalar_displacement_correction,
    build_local_frames,
    extract_calibration_rows,
    fit_calibration_bundle,
    fit_local_linear_model,
    fit_scalar_displacement_correction,
    inference_receipt_signature,
    vectors_to_local,
)
from vesuvius.neural_tracing.evaluation.copy_cycle_metrics import (
    DistanceIndex,
    score_prediction,
)
from vesuvius.neural_tracing.evaluation.score_copy_cycle_calibration import (
    evaluate_calibration_gate,
    validate_scoring_provenance,
)


def _plane(shape=(5, 6)):
    rows, columns = np.indices(shape, dtype=np.float32)
    grid = np.stack((np.zeros(shape), rows, columns), axis=2)
    return grid, np.ones(shape, dtype=bool)


def _rows(seed=20260811, count=200):
    rng = np.random.default_rng(seed)
    cycle = rng.normal(size=(count, 3))
    displacement = rng.normal(size=(count, 3))
    features = np.column_stack((cycle, displacement))
    coefficients = np.array(
        [
            [0.2, -0.1, 0.0],
            [0.0, 0.3, 0.1],
            [-0.2, 0.0, 0.4],
            [0.5, 0.1, 0.0],
            [0.0, -0.3, 0.2],
            [0.1, 0.0, -0.2],
        ]
    )
    target = features @ coefficients
    return CalibrationRows(cycle, displacement, target), coefficients


def _score(prediction_value):
    source = np.zeros((4, 4, 3), dtype=np.float32)
    valid = np.ones((4, 4), dtype=bool)
    prediction = source.copy()
    prediction[..., 2] = float(prediction_value)
    target = np.array([[0.0, 0.0, 10.0]], dtype=np.float32)
    wrong = np.array([[0.0, 100.0, 0.0]], dtype=np.float32)
    return score_prediction(source, valid, target, wrong, prediction, valid)


def _training_provenance(stage="holdout", requested_copy_args=None, total_rows=200):
    directions = (
        ["1->2", "2->1", "2->3"]
        if stage == "holdout"
        else ["1->2", "2->1", "2->3", "3->2", "3->4", "4->3"]
    )
    sources = [1, 2] if stage == "holdout" else [1, 2, 3, 4]
    quotient, remainder = divmod(total_rows, len(directions))
    rows = {
        direction: quotient + (index < remainder)
        for index, direction in enumerate(directions)
    }
    return {
        "stage": stage,
        "directions": directions,
        "sources": sources,
        "receipts": [{"path": "/tmp/receipt.json", "sha256": "c" * 64}],
        "scroll_id": "PHerc0500P2",
        "inference_implementation_commit": "d" * 40,
        "checkpoint_sha256": "b" * 64,
        "manifest_sha256": "e" * 64,
        "requested_copy_args": requested_copy_args or {"tta": True},
        "volume_path": "https://example.test/volume.zarr/",
        "volume_backend": "zarr",
        "volume_scale_requested": 0,
        "volume_scale_resolved": 0,
        "tifxyz_voxel_size_um": 4.317,
        "crop_size": [128, 384, 384],
        "runtime": {"torch": "test"},
        "rows_by_direction": rows,
        "total_rows": sum(rows.values()),
    }


def _receipt(**updates):
    receipt = {
        "scroll_id": "PHerc0500P2",
        "implementation_commit": "d" * 40,
        "checkpoint_sha256": "b" * 64,
        "manifest_sha256": "e" * 64,
        "requested_copy_args": {"tta": True},
        "volume_path": "https://example.test/volume.zarr/",
        "volume_backend": "zarr",
        "volume_scale_requested": 0,
        "volume_scale_resolved": 0,
        "copy_args": {"tifxyz_voxel_size_um": 4.317},
        "crop_size": [128, 384, 384],
        "runtime": {"torch": "test"},
    }
    receipt.update(updates)
    return receipt


def test_local_frames_are_orthonormal_and_follow_chart_orientation():
    grid, valid = _plane()

    frames, frame_valid = build_local_frames(grid, valid)

    assert frame_valid.all()
    assert np.allclose(frames[..., 0, :], [0.0, 0.0, 1.0])
    assert np.allclose(frames[..., 1, :], [0.0, 1.0, 0.0])
    assert np.allclose(frames[..., 2, :], [-1.0, 0.0, 0.0])
    gram = np.einsum("...ik,...jk->...ij", frames, frames)
    assert np.allclose(gram, np.eye(3))


def test_local_features_are_invariant_to_proper_world_rotation():
    grid, valid = _plane()
    frames, _ = build_local_frames(grid, valid)
    vector = np.broadcast_to(np.array([2.0, 3.0, 4.0]), grid.shape)
    rotation = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    rotated_grid = np.einsum("ij,...j->...i", rotation, grid)
    rotated_vector = np.einsum("ij,...j->...i", rotation, vector)

    rotated_frames, _ = build_local_frames(rotated_grid, valid)

    assert np.allclose(
        vectors_to_local(vector, frames),
        vectors_to_local(rotated_vector, rotated_frames),
    )


def test_no_intercept_ridge_recovers_synthetic_linear_map():
    rows, coefficients = _rows()

    model = fit_local_linear_model(
        rows, "cycle_displacement", ridge=0.0, correction_cap=8.0
    )

    recovered_raw_coefficients = model.coefficients / model.rms[:, None]
    assert np.allclose(recovered_raw_coefficients, coefficients, atol=1e-10)


def test_extract_rows_uses_common_mask_and_target_correction():
    source, valid = _plane((3, 3))
    forward = source.copy()
    forward[..., 0] += 2.0
    returned = source.copy()
    target_cloud = forward.reshape(-1, 3).copy()
    target_cloud[:, 0] += 0.5
    eligible = valid.copy()
    eligible[0, 0] = False
    return_valid = valid.copy()
    return_valid[0, 1] = False

    rows = extract_calibration_rows(
        source,
        valid,
        forward,
        valid,
        returned,
        return_valid,
        eligible,
        DistanceIndex.from_clouds((target_cloud,)),
    )

    assert rows.count == 7
    assert np.allclose(rows.cycle_local, 0.0)
    assert np.allclose(np.linalg.norm(rows.target_correction_local, axis=1), 0.5)


def test_local_application_preserves_validity_and_caps_correction():
    source, valid = _plane()
    forward = source.copy()
    forward[..., 2] += 2.0
    model = LocalLinearModel(
        mode="displacement",
        rms=np.ones(3),
        coefficients=100.0 * np.eye(3),
        ridge=10.0,
        correction_cap=8.0,
    )
    forward_valid = valid.copy()
    forward_valid[0, 0] = False

    corrected, corrected_valid, applied = apply_local_linear_model(
        model, source, valid, forward, forward_valid
    )

    assert np.array_equal(corrected_valid, forward_valid)
    assert not applied[0, 0]
    correction_norm = np.linalg.norm(corrected[applied] - forward[applied], axis=1)
    assert np.allclose(correction_norm, 8.0)


def test_scalar_control_fit_and_application_use_correction_beta():
    displacement = np.array([[1.0, 2.0, 3.0], [-2.0, 1.0, 0.5]])
    rows = CalibrationRows(
        cycle_local=np.ones_like(displacement),
        displacement_local=displacement,
        target_correction_local=0.25 * displacement,
    )
    source, valid = _plane((3, 3))
    forward = source.copy()
    forward[..., 2] += 4.0

    beta = fit_scalar_displacement_correction(rows)
    corrected, corrected_valid, applied = apply_scalar_displacement_correction(
        source,
        valid,
        forward,
        valid,
        beta=beta,
        correction_cap=None,
    )

    assert beta == pytest.approx(0.25)
    assert corrected_valid.all() and applied.all()
    assert np.allclose(corrected - source, 1.25 * (forward - source))


def test_bundle_round_trip_is_strict_and_records_frozen_models():
    rows, _ = _rows()
    bundle = fit_calibration_bundle(
        rows,
        implementation_commit="a" * 40,
        training=_training_provenance(),
    )

    loaded = CalibrationBundle.from_json(bundle.to_json())

    assert loaded.implementation_commit == "a" * 40
    assert loaded.combined.mode == "cycle_displacement"
    assert loaded.displacement_only.mode == "displacement"
    assert loaded.cycle_only.mode == "cycle"
    malformed = copy.deepcopy(bundle.to_json())
    malformed["combined"]["feature_order"][0] = "wrong"
    with pytest.raises(ValueError, match="feature order"):
        CalibrationBundle.from_json(malformed)
    malformed = copy.deepcopy(bundle.to_json())
    malformed["training"]["total_rows"] += 1
    with pytest.raises(ValueError, match="total training rows"):
        CalibrationBundle.from_json(malformed)


def test_holdout_gate_requires_incremental_gain_over_controls():
    baseline = [_score(8.0) for _ in range(3)]
    combined = [_score(10.0) for _ in range(3)]
    control = [_score(9.0) for _ in range(3)]
    results = {
        "baseline": baseline,
        "combined": combined,
        "displacement_only": control,
        "cycle_only": control,
        "fitted_scalar": control,
        "physical_scalar": control,
    }

    passing = evaluate_calibration_gate("development_holdout", results)

    assert passing["passed"] is True
    tied = dict(results)
    tied["displacement_only"] = combined
    failing = evaluate_calibration_gate("development_holdout", tied)
    assert failing["passed"] is False
    assert failing["conditions"]["beats_each_displacement_control"] is False


def test_scoring_provenance_rejects_nonfrozen_copy_arguments():
    rows, _ = _rows()
    bundle = fit_calibration_bundle(
        rows,
        implementation_commit="a" * 40,
        training=_training_provenance(),
    )
    receipt = _receipt()

    validate_scoring_provenance(
        bundle,
        [receipt],
        stage="development_holdout",
        implementation_commit="a" * 40,
    )
    receipt["requested_copy_args"] = {"tta": False}
    with pytest.raises(ValueError, match="requested_copy_args"):
        validate_scoring_provenance(
            bundle,
            [receipt],
            stage="development_holdout",
            implementation_commit="a" * 40,
        )


def test_scoring_provenance_rejects_same_scroll_manifest_or_runtime_change():
    rows, _ = _rows()
    bundle = fit_calibration_bundle(
        rows,
        implementation_commit="a" * 40,
        training=_training_provenance(),
    )

    for key, value in (("manifest_sha256", "f" * 64), ("runtime", {"torch": "other"})):
        receipt = _receipt(**{key: value})
        with pytest.raises(ValueError, match=key):
            validate_scoring_provenance(
                bundle,
                [receipt],
                stage="development_holdout",
                implementation_commit="a" * 40,
            )


def test_sealed_scoring_allows_frozen_cross_scroll_fields_only():
    rows, _ = _rows()
    bundle = fit_calibration_bundle(
        rows,
        implementation_commit="a" * 40,
        training=_training_provenance(stage="final"),
    )
    receipt = _receipt(
        scroll_id="PHerc0343P",
        manifest_sha256="f" * 64,
        volume_path="https://example.test/other.zarr/",
        volume_scale_requested=1,
        volume_scale_resolved=1,
        copy_args={"tifxyz_voxel_size_um": 2.215},
        selected_parameters={
            "method": "copy_cycle_local_linear_v1",
            "model_sha256": "7" * 64,
        },
    )

    validate_scoring_provenance(
        bundle,
        [receipt],
        stage="sealed_test",
        implementation_commit="a" * 40,
        model_sha256="7" * 64,
    )
    receipt["selected_parameters"]["model_sha256"] = "8" * 64
    with pytest.raises(ValueError, match="not bound"):
        validate_scoring_provenance(
            bundle,
            [receipt],
            stage="sealed_test",
            implementation_commit="a" * 40,
            model_sha256="7" * 64,
        )
    receipt["selected_parameters"]["model_sha256"] = "7" * 64
    receipt["implementation_commit"] = "9" * 40
    with pytest.raises(ValueError, match="inference_implementation_commit"):
        validate_scoring_provenance(
            bundle,
            [receipt],
            stage="sealed_test",
            implementation_commit="a" * 40,
            model_sha256="7" * 64,
        )


def test_inference_signature_rejects_missing_or_invalid_fields():
    signature = inference_receipt_signature(_receipt())
    assert signature["crop_size"] == [128, 384, 384]

    with pytest.raises(ValueError, match="crop size"):
        inference_receipt_signature(_receipt(crop_size=[128, 384]))
    with pytest.raises(ValueError, match="checkpoint SHA-256"):
        inference_receipt_signature(_receipt(checkpoint_sha256="short"))


def test_calibration_rows_reject_wrong_rank():
    values = np.ones((2, 2, 3))
    with pytest.raises(ValueError, match=r"shape \[N, 3\]"):
        CalibrationRows(values, values, values)
