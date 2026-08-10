import math

import numpy as np
import pytest

from vesuvius.neural_tracing.evaluation.copy_cycle_metrics import (
    aggregate_scores,
    apply_cycle_guard,
    binary_ranking_metrics,
    central_fraction_mask,
    choose_return_branch,
    densify_surface_grid,
    score_prediction,
)


def _constant_grid(value, shape=(4, 4)):
    grid = np.zeros((*shape, 3), dtype=np.float32)
    grid[..., 2] = float(value)
    return grid, np.ones(shape, dtype=bool)


def test_central_fraction_mask_is_ground_truth_only():
    valid = np.ones((10, 10), dtype=bool)
    valid[0, 0] = False

    mask = central_fraction_mask(valid, 0.7)

    assert mask.sum() == 64
    assert mask[1:9, 1:9].all()
    assert not mask[0].any()
    assert not mask[:, 0].any()


def test_densify_surface_grid_adds_edges_and_bilinear_center():
    grid = np.array(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 10.0]],
            [[0.0, 10.0, 0.0], [0.0, 10.0, 10.0]],
        ],
        dtype=np.float32,
    )

    points = densify_surface_grid(grid, np.ones((2, 2), dtype=bool), spacing=5.0)

    unique = np.unique(points, axis=0)
    assert unique.shape == (9, 3)
    assert np.any(np.all(np.isclose(unique, [0.0, 5.0, 5.0]), axis=1))


def test_densify_surface_grid_ignores_nonfinite_invalid_cells():
    grid = np.zeros((2, 2, 3), dtype=np.float32)
    grid[0, 0] = np.nan
    valid = np.ones((2, 2), dtype=bool)
    valid[0, 0] = False

    points = densify_surface_grid(grid, valid, spacing=5.0)

    assert points.shape == (3, 3)
    assert np.isfinite(points).all()


def test_cycle_guard_uses_frozen_correction_sign_and_threshold():
    source, valid = _constant_grid(0.0)
    forward, _ = _constant_grid(10.0)
    returned, _ = _constant_grid(2.0)
    returned_over_threshold, _ = _constant_grid(5.0)

    accepted = apply_cycle_guard(
        source,
        valid,
        forward,
        valid,
        returned,
        valid,
        alpha=0.5,
        tau=4.0,
    )
    rejected = apply_cycle_guard(
        source,
        valid,
        forward,
        valid,
        returned_over_threshold,
        valid,
        alpha=0.5,
        tau=4.0,
    )

    assert accepted.valid.all()
    assert np.allclose(accepted.grid[..., 2], 9.0)
    assert np.allclose(accepted.residual, 2.0)
    assert not rejected.valid.any()


def test_choose_return_branch_uses_median_cycle_error_without_target():
    source, valid = _constant_grid(0.0)
    near, _ = _constant_grid(1.0)
    far, _ = _constant_grid(5.0)

    selected, medians = choose_return_branch(
        source, valid, {"front": (far, valid), "back": (near, valid)}
    )

    assert selected == "back"
    assert medians == {"front": 5.0, "back": 1.0}


def test_score_prediction_penalizes_missing_cells_and_wrong_sheet():
    source, source_valid = _constant_grid(0.0)
    prediction, prediction_valid = _constant_grid(10.0)
    target = np.array([[0.0, 0.0, 10.0]], dtype=np.float32)
    wrong = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

    full = score_prediction(
        source, source_valid, target, wrong, prediction, prediction_valid
    )
    half_valid = prediction_valid.copy()
    half_valid[:, :2] = False
    half = score_prediction(source, source_valid, target, wrong, prediction, half_valid)
    stayed = score_prediction(source, source_valid, target, wrong, source, source_valid)

    assert full.metrics["coverage"] == 1.0
    assert full.metrics["penalized_target_distance_mean"] == 0.0
    assert half.metrics["eligible_cells"] == full.metrics["eligible_cells"]
    assert half.metrics["coverage"] == 0.5
    assert half.metrics["penalized_target_distance_mean"] == 40.0
    assert stayed.metrics["sheet_switch_rate_all_eligible"] == 1.0
    aggregate = aggregate_scores([full, half])
    assert aggregate["penalized_target_distance_mean"] == 20.0


def test_binary_ranking_metrics_distinguishes_signal_from_inverse():
    labels = np.array([False, False, True, True])

    signal = binary_ranking_metrics(np.array([0.0, 1.0, 2.0, 3.0]), labels)
    inverse = binary_ranking_metrics(np.array([3.0, 2.0, 1.0, 0.0]), labels)

    assert signal["auroc"] == 1.0
    assert signal["average_precision"] == 1.0
    assert inverse["auroc"] == 0.0
    assert math.isclose(inverse["average_precision"], 5.0 / 12.0)


def test_guard_rejects_unregistered_parameters():
    grid, valid = _constant_grid(0.0)

    with pytest.raises(ValueError, match="alpha"):
        apply_cycle_guard(grid, valid, grid, valid, grid, valid, alpha=0.1, tau=4.0)
