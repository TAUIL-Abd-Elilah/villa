"""Frozen geometry and cycle-guard metrics for adjacent-wrap copy tracing."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
import math
from typing import Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import rankdata


INNER_FRACTION = 0.70
GROUND_TRUTH_SAMPLE_SPACING = 5.0
GROUND_TRUTH_MAX_EDGE = 60.0
MAX_SOURCE_TO_TARGET = 80.0
MISSING_PENALTY = 80.0
SHEET_SWITCH_MARGIN = 5.0
BAD_CELL_DISTANCE = 40.0
SHIFT_ROWS = 17
SHIFT_COLS = 11
ALPHA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
TAU_GRID = (4.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, math.inf)


@dataclass(frozen=True)
class GuardResult:
    grid: np.ndarray
    valid: np.ndarray
    residual: np.ndarray
    residual_valid: np.ndarray


@dataclass(frozen=True)
class ScoreResult:
    metrics: dict[str, float | int]
    eligible: np.ndarray
    prediction_valid: np.ndarray
    target_distance: np.ndarray
    wrong_distance: np.ndarray
    sheet_switch: np.ndarray


@dataclass(frozen=True)
class DistanceIndex:
    """Nearest-point index over one or more surface point clouds."""

    trees: tuple[cKDTree, ...]

    @classmethod
    def from_clouds(cls, clouds: Sequence[np.ndarray]) -> "DistanceIndex":
        trees: list[cKDTree] = []
        for cloud in clouds:
            cloud_arr = np.asarray(cloud, dtype=np.float32)
            if cloud_arr.ndim != 2 or cloud_arr.shape[1] != 3 or cloud_arr.shape[0] == 0:
                raise ValueError(
                    f"distance-index clouds must have non-empty shape [N, 3], got {cloud_arr.shape}"
                )
            trees.append(cKDTree(cloud_arr))
        if not trees:
            raise ValueError("distance index requires at least one point cloud")
        return cls(trees=tuple(trees))

    def query(self, points: np.ndarray) -> np.ndarray:
        points_arr = np.asarray(points, dtype=np.float32)
        if points_arr.ndim != 2 or points_arr.shape[1] != 3:
            raise ValueError(f"query points must have shape [N, 3], got {points_arr.shape}")
        minimum = np.full((points_arr.shape[0],), np.inf, dtype=np.float64)
        for tree in self.trees:
            minimum = np.minimum(minimum, tree.query(points_arr, workers=-1)[0])
        return minimum


def _distance_index(value: np.ndarray | DistanceIndex) -> DistanceIndex:
    if isinstance(value, DistanceIndex):
        return value
    return DistanceIndex.from_clouds((np.asarray(value, dtype=np.float32),))


def _checked_grid(grid: np.ndarray, valid: np.ndarray, label: str) -> tuple[np.ndarray, np.ndarray]:
    grid_arr = np.asarray(grid, dtype=np.float32)
    valid_arr = np.asarray(valid, dtype=bool)
    if grid_arr.ndim != 3 or grid_arr.shape[-1] != 3:
        raise ValueError(f"{label} grid must have shape [H, W, 3], got {grid_arr.shape}")
    if valid_arr.shape != grid_arr.shape[:2]:
        raise ValueError(
            f"{label} valid mask must have shape {grid_arr.shape[:2]}, got {valid_arr.shape}"
        )
    valid_arr = valid_arr & np.isfinite(grid_arr).all(axis=2)
    return grid_arr, valid_arr


def central_fraction_mask(valid: np.ndarray, fraction: float = INNER_FRACTION) -> np.ndarray:
    valid_arr = np.asarray(valid, dtype=bool)
    if valid_arr.ndim != 2:
        raise ValueError(f"valid mask must be 2D, got {valid_arr.shape}")
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError(f"fraction must satisfy 0 < fraction <= 1, got {fraction}")
    output = np.zeros_like(valid_arr)
    rows, cols = np.where(valid_arr)
    if rows.size == 0:
        return output
    row_extent = int(rows.max() - rows.min() + 1)
    col_extent = int(cols.max() - cols.min() + 1)
    trim_rows = int(math.floor(row_extent * (1.0 - float(fraction)) / 2.0))
    trim_cols = int(math.floor(col_extent * (1.0 - float(fraction)) / 2.0))
    row_start = int(rows.min()) + trim_rows
    row_stop = int(rows.max()) - trim_rows + 1
    col_start = int(cols.min()) + trim_cols
    col_stop = int(cols.max()) - trim_cols + 1
    output[row_start:row_stop, col_start:col_stop] = True
    return output & valid_arr


def _densify_edges(
    start: np.ndarray,
    end: np.ndarray,
    valid: np.ndarray,
    spacing: float,
    max_edge: float,
) -> list[np.ndarray]:
    lengths = np.linalg.norm(end - start, axis=-1)
    usable = np.asarray(valid, dtype=bool) & np.isfinite(lengths) & (lengths <= max_edge)
    subdivisions = np.ones(lengths.shape, dtype=np.int16)
    subdivisions[usable] = np.maximum(
        1, np.ceil(lengths[usable] / spacing).astype(np.int16)
    )
    pieces: list[np.ndarray] = []
    for count in sorted(int(value) for value in np.unique(subdivisions[usable]) if value > 1):
        selected = usable & (subdivisions == count)
        start_selected = start[selected]
        delta_selected = end[selected] - start_selected
        for index in range(1, count):
            pieces.append(start_selected + delta_selected * (float(index) / float(count)))
    return pieces


def densify_surface_grid(
    grid: np.ndarray,
    valid: np.ndarray,
    *,
    spacing: float = GROUND_TRUTH_SAMPLE_SPACING,
    max_edge: float = GROUND_TRUTH_MAX_EDGE,
) -> np.ndarray:
    """Densify valid grid quads deterministically with bilinear samples."""

    if float(spacing) <= 0.0 or float(max_edge) <= 0.0:
        raise ValueError("spacing and max_edge must be positive")
    grid_arr, valid_arr = _checked_grid(grid, valid, "surface")
    if not bool(valid_arr.any()):
        return np.zeros((0, 3), dtype=np.float32)

    pieces: list[np.ndarray] = [grid_arr[valid_arr]]

    horizontal_valid = valid_arr[:, :-1] & valid_arr[:, 1:]
    pieces.extend(
        _densify_edges(
            grid_arr[:, :-1],
            grid_arr[:, 1:],
            horizontal_valid,
            float(spacing),
            float(max_edge),
        )
    )
    vertical_valid = valid_arr[:-1, :] & valid_arr[1:, :]
    pieces.extend(
        _densify_edges(
            grid_arr[:-1, :],
            grid_arr[1:, :],
            vertical_valid,
            float(spacing),
            float(max_edge),
        )
    )

    p00 = grid_arr[:-1, :-1]
    p01 = grid_arr[:-1, 1:]
    p10 = grid_arr[1:, :-1]
    p11 = grid_arr[1:, 1:]
    quad_valid = (
        valid_arr[:-1, :-1]
        & valid_arr[:-1, 1:]
        & valid_arr[1:, :-1]
        & valid_arr[1:, 1:]
    )
    boundary_lengths = np.stack(
        [
            np.linalg.norm(p01 - p00, axis=-1),
            np.linalg.norm(p10 - p00, axis=-1),
            np.linalg.norm(p11 - p01, axis=-1),
            np.linalg.norm(p11 - p10, axis=-1),
        ],
        axis=0,
    )
    max_boundary_length = np.max(boundary_lengths, axis=0)
    quad_valid &= np.isfinite(max_boundary_length) & (
        max_boundary_length <= float(max_edge)
    )
    subdivisions = np.ones(max_boundary_length.shape, dtype=np.int16)
    subdivisions[quad_valid] = np.maximum(
        1,
        np.ceil(max_boundary_length[quad_valid] / float(spacing)).astype(np.int16),
    )

    for count in sorted(int(value) for value in np.unique(subdivisions[quad_valid]) if value > 1):
        selected = quad_valid & (subdivisions == count)
        q00 = p00[selected]
        q01 = p01[selected]
        q10 = p10[selected]
        q11 = p11[selected]
        for row_index in range(1, count):
            row_fraction = float(row_index) / float(count)
            for col_index in range(1, count):
                col_fraction = float(col_index) / float(count)
                samples = (
                    (1.0 - row_fraction) * (1.0 - col_fraction) * q00
                    + (1.0 - row_fraction) * col_fraction * q01
                    + row_fraction * (1.0 - col_fraction) * q10
                    + row_fraction * col_fraction * q11
                )
                pieces.append(samples)

    return np.concatenate(pieces, axis=0).astype(np.float32, copy=False)


def central_dense_surface(
    grid: np.ndarray,
    valid: np.ndarray,
    *,
    fraction: float = INNER_FRACTION,
    spacing: float = GROUND_TRUTH_SAMPLE_SPACING,
    max_edge: float = GROUND_TRUTH_MAX_EDGE,
) -> np.ndarray:
    _, valid_arr = _checked_grid(grid, valid, "surface")
    return densify_surface_grid(
        grid,
        central_fraction_mask(valid_arr, fraction),
        spacing=spacing,
        max_edge=max_edge,
    )


def eligibility_mask(
    source_grid: np.ndarray,
    source_valid: np.ndarray,
    target_points: np.ndarray | DistanceIndex,
    *,
    inner_fraction: float = INNER_FRACTION,
    max_source_to_target: float = MAX_SOURCE_TO_TARGET,
) -> tuple[np.ndarray, np.ndarray]:
    source_arr, source_mask = _checked_grid(source_grid, source_valid, "source")
    target_index = _distance_index(target_points)
    central = central_fraction_mask(source_mask, inner_fraction)
    distances = np.full(source_mask.shape, np.nan, dtype=np.float32)
    if bool(central.any()):
        distances[central] = target_index.query(source_arr[central])
    eligible = central & (distances <= float(max_source_to_target))
    return eligible, distances


def score_prediction(
    source_grid: np.ndarray,
    source_valid: np.ndarray,
    target_points: np.ndarray | DistanceIndex,
    wrong_points: np.ndarray | DistanceIndex,
    prediction_grid: np.ndarray,
    prediction_valid: np.ndarray,
    *,
    missing_penalty: float = MISSING_PENALTY,
    switch_margin: float = SHEET_SWITCH_MARGIN,
    max_source_to_target: float = MAX_SOURCE_TO_TARGET,
) -> ScoreResult:
    source_arr, source_mask = _checked_grid(source_grid, source_valid, "source")
    prediction_arr, prediction_mask = _checked_grid(
        prediction_grid, prediction_valid, "prediction"
    )
    if prediction_arr.shape != source_arr.shape:
        raise ValueError(
            f"prediction shape {prediction_arr.shape} must equal source shape {source_arr.shape}"
        )
    target_index = _distance_index(target_points)
    wrong_index = _distance_index(wrong_points)

    eligible, _ = eligibility_mask(
        source_arr,
        source_mask,
        target_index,
        max_source_to_target=max_source_to_target,
    )
    n_eligible = int(eligible.sum())
    if n_eligible == 0:
        raise ValueError("directed task has zero eligible source cells")
    scored_valid = eligible & prediction_mask
    n_valid = int(scored_valid.sum())

    target_distance = np.full(eligible.shape, np.nan, dtype=np.float32)
    wrong_distance = np.full(eligible.shape, np.nan, dtype=np.float32)
    sheet_switch = np.zeros(eligible.shape, dtype=bool)
    if n_valid > 0:
        predicted_points = prediction_arr[scored_valid]
        target_values = target_index.query(predicted_points)
        wrong_values = wrong_index.query(predicted_points)
        target_distance[scored_valid] = target_values
        wrong_distance[scored_valid] = wrong_values
        sheet_switch[scored_valid] = wrong_values + float(switch_margin) < target_values
        valid_distances = target_values.astype(np.float64, copy=False)
    else:
        valid_distances = np.zeros((0,), dtype=np.float64)

    missing_count = n_eligible - n_valid
    capped_sum = float(np.minimum(valid_distances, float(missing_penalty)).sum())
    capped_sum += float(missing_count) * float(missing_penalty)
    switch_count = int(sheet_switch.sum())

    def percentile(value: float) -> float:
        if n_valid == 0:
            return math.inf
        return float(np.percentile(valid_distances, value))

    metrics: dict[str, float | int] = {
        "eligible_cells": n_eligible,
        "valid_prediction_cells": n_valid,
        "missing_prediction_cells": missing_count,
        "coverage": float(n_valid / n_eligible),
        "target_distance_mean_valid": (
            float(valid_distances.mean()) if n_valid > 0 else math.inf
        ),
        "target_distance_median_valid": percentile(50.0),
        "target_distance_p90_valid": percentile(90.0),
        "target_distance_p95_valid": percentile(95.0),
        "within_10_rate_valid": (
            float(np.mean(valid_distances <= 10.0)) if n_valid > 0 else 0.0
        ),
        "within_20_rate_valid": (
            float(np.mean(valid_distances <= 20.0)) if n_valid > 0 else 0.0
        ),
        "within_40_rate_valid": (
            float(np.mean(valid_distances <= 40.0)) if n_valid > 0 else 0.0
        ),
        "within_10_rate_all_eligible": float(np.sum(valid_distances <= 10.0) / n_eligible),
        "within_20_rate_all_eligible": float(np.sum(valid_distances <= 20.0) / n_eligible),
        "within_40_rate_all_eligible": float(np.sum(valid_distances <= 40.0) / n_eligible),
        "penalized_target_distance_sum": capped_sum,
        "penalized_target_distance_mean": float(capped_sum / n_eligible),
        "sheet_switch_cells": switch_count,
        "sheet_switch_rate_valid": float(switch_count / n_valid) if n_valid > 0 else 0.0,
        "sheet_switch_rate_all_eligible": float(switch_count / n_eligible),
    }
    return ScoreResult(
        metrics=metrics,
        eligible=eligible,
        prediction_valid=scored_valid,
        target_distance=target_distance,
        wrong_distance=wrong_distance,
        sheet_switch=sheet_switch,
    )


def choose_return_branch(
    source_grid: np.ndarray,
    source_valid: np.ndarray,
    branches: Mapping[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[str, dict[str, float]]:
    source_arr, source_mask = _checked_grid(source_grid, source_valid, "source")
    central = central_fraction_mask(source_mask)
    if not branches:
        raise ValueError("at least one return branch is required")
    medians: dict[str, float] = {}
    for name, (branch_grid, branch_valid) in branches.items():
        branch_arr, branch_mask = _checked_grid(branch_grid, branch_valid, f"return branch {name}")
        if branch_arr.shape != source_arr.shape:
            raise ValueError(f"return branch {name} shape does not match source")
        comparable = central & branch_mask
        medians[str(name)] = (
            float(np.median(np.linalg.norm(branch_arr[comparable] - source_arr[comparable], axis=1)))
            if bool(comparable.any())
            else math.inf
        )
    selected = min(sorted(medians), key=lambda name: medians[name])
    if not math.isfinite(medians[selected]):
        raise ValueError("no return branch has finite central-70% support")
    return selected, medians


def assign_baseline_branches(
    branches: Sequence[str],
    targets: Sequence[int],
    scores: Mapping[tuple[str, int], ScoreResult],
) -> dict[int, str]:
    """Assign arbitrary chart-side labels using baseline cost only."""

    branch_names = tuple(sorted(str(name) for name in branches))
    target_ids = tuple(sorted(int(target) for target in targets))
    if not branch_names or not target_ids:
        raise ValueError("branches and targets must both be non-empty")
    if len(target_ids) > len(branch_names):
        raise ValueError("there must be at least as many branches as targets")
    missing = [
        (branch, target)
        for branch in branch_names
        for target in target_ids
        if (branch, target) not in scores
    ]
    if missing:
        raise ValueError(f"missing baseline branch-target scores: {missing}")

    assignments: list[tuple[float, tuple[str, ...]]] = []
    for selected_branches in permutations(branch_names, len(target_ids)):
        total_cost = sum(
            float(scores[(branch, target)].metrics["penalized_target_distance_mean"])
            for target, branch in zip(target_ids, selected_branches)
        )
        assignments.append((total_cost, selected_branches))
    _, winning_branches = min(assignments, key=lambda item: (item[0], item[1]))
    return {target: branch for target, branch in zip(target_ids, winning_branches)}


def apply_cycle_guard(
    source_grid: np.ndarray,
    source_valid: np.ndarray,
    forward_grid: np.ndarray,
    forward_valid: np.ndarray,
    return_grid: np.ndarray,
    return_valid: np.ndarray,
    *,
    alpha: float,
    tau: float,
) -> GuardResult:
    return _apply_cycle_guard_with_sign(
        source_grid,
        source_valid,
        forward_grid,
        forward_valid,
        return_grid,
        return_valid,
        alpha=alpha,
        tau=tau,
        correction_sign=-1.0,
    )


def apply_wrong_sign_null(
    source_grid: np.ndarray,
    source_valid: np.ndarray,
    forward_grid: np.ndarray,
    forward_valid: np.ndarray,
    return_grid: np.ndarray,
    return_valid: np.ndarray,
    *,
    alpha: float,
    tau: float,
) -> GuardResult:
    return _apply_cycle_guard_with_sign(
        source_grid,
        source_valid,
        forward_grid,
        forward_valid,
        return_grid,
        return_valid,
        alpha=alpha,
        tau=tau,
        correction_sign=1.0,
    )


def _apply_cycle_guard_with_sign(
    source_grid: np.ndarray,
    source_valid: np.ndarray,
    forward_grid: np.ndarray,
    forward_valid: np.ndarray,
    return_grid: np.ndarray,
    return_valid: np.ndarray,
    *,
    alpha: float,
    tau: float,
    correction_sign: float,
) -> GuardResult:
    source_arr, source_mask = _checked_grid(source_grid, source_valid, "source")
    forward_arr, forward_mask = _checked_grid(forward_grid, forward_valid, "forward")
    return_arr, return_mask = _checked_grid(return_grid, return_valid, "return")
    if forward_arr.shape != source_arr.shape or return_arr.shape != source_arr.shape:
        raise ValueError("source, forward, and return grids must have identical shapes")
    if float(alpha) not in ALPHA_GRID:
        raise ValueError(f"alpha must be one of {ALPHA_GRID}, got {alpha}")
    if float(tau) not in TAU_GRID:
        raise ValueError(f"tau must be one of {TAU_GRID}, got {tau}")

    residual_valid = source_mask & forward_mask & return_mask
    residual_vector = np.zeros_like(source_arr, dtype=np.float32)
    residual_vector[residual_valid] = return_arr[residual_valid] - source_arr[residual_valid]
    residual = np.full(source_mask.shape, np.nan, dtype=np.float32)
    residual[residual_valid] = np.linalg.norm(residual_vector[residual_valid], axis=1)
    accepted = residual_valid & (residual <= float(tau))

    corrected = np.full_like(source_arr, -1.0, dtype=np.float32)
    corrected[accepted] = (
        forward_arr[accepted]
        + float(correction_sign) * float(alpha) * residual_vector[accepted]
    ).astype(np.float32, copy=False)
    return GuardResult(
        grid=corrected,
        valid=accepted,
        residual=residual,
        residual_valid=residual_valid,
    )


def aggregate_scores(results: Sequence[ScoreResult]) -> dict[str, float | int]:
    if not results:
        raise ValueError("at least one score result is required")
    eligible = sum(int(result.metrics["eligible_cells"]) for result in results)
    valid = sum(int(result.metrics["valid_prediction_cells"]) for result in results)
    switches = sum(int(result.metrics["sheet_switch_cells"]) for result in results)
    penalty_sum = sum(float(result.metrics["penalized_target_distance_sum"]) for result in results)
    valid_distances = np.concatenate(
        [result.target_distance[result.prediction_valid] for result in results]
    )
    return {
        "directed_tasks": len(results),
        "eligible_cells": eligible,
        "valid_prediction_cells": valid,
        "coverage": float(valid / eligible),
        "penalized_target_distance_mean": float(penalty_sum / eligible),
        "sheet_switch_cells": switches,
        "sheet_switch_rate_all_eligible": float(switches / eligible),
        "target_distance_p95_valid": (
            float(np.percentile(valid_distances, 95.0)) if valid_distances.size else math.inf
        ),
    }


def select_validation_candidate(
    baseline: Sequence[ScoreResult],
    candidates: Mapping[tuple[float, float], Sequence[ScoreResult]],
) -> dict[str, object]:
    """Apply the frozen validation coverage gate and lexicographic selection."""

    if not baseline:
        raise ValueError("baseline validation scores cannot be empty")
    no_op_key = (0.0, math.inf)
    if no_op_key not in candidates:
        raise ValueError("candidate grid must include the no-op key (0.0, infinity)")
    direction_count = len(baseline)
    for key, results in candidates.items():
        if float(key[0]) not in ALPHA_GRID or float(key[1]) not in TAU_GRID:
            raise ValueError(f"candidate key is outside the frozen grid: {key}")
        if len(results) != direction_count:
            raise ValueError(
                f"candidate {key} has {len(results)} directions, expected {direction_count}"
            )

    eligible: list[tuple[tuple[float, float], dict[str, float | int]]] = []
    coverage_failures: dict[str, list[int]] = {}
    for key, results in candidates.items():
        failed_directions = [
            index
            for index, (baseline_result, candidate_result) in enumerate(zip(baseline, results))
            if float(candidate_result.metrics["coverage"])
            < 0.9 * float(baseline_result.metrics["coverage"])
        ]
        if failed_directions:
            coverage_failures[f"alpha={key[0]:g},tau={key[1]:g}"] = failed_directions
            continue
        aggregate = aggregate_scores(results)
        baseline_aggregate = aggregate_scores(baseline)
        if float(aggregate["coverage"]) < 0.9 * float(baseline_aggregate["coverage"]):
            coverage_failures[f"alpha={key[0]:g},tau={key[1]:g}"] = [-1]
            continue
        eligible.append((key, aggregate))

    if not eligible:
        return {
            "status": "validation_negative",
            "reason": "no frozen-grid candidate passed every coverage gate",
            "selected": None,
            "coverage_failures": coverage_failures,
        }

    def rank(item: tuple[tuple[float, float], dict[str, float | int]]) -> tuple[float, ...]:
        (alpha, tau), aggregate = item
        return (
            float(aggregate["penalized_target_distance_mean"]),
            float(aggregate["sheet_switch_rate_all_eligible"]),
            float(alpha),
            -float(tau),
        )

    selected_key, selected_aggregate = min(eligible, key=rank)
    no_op_results = candidates[no_op_key]
    improved_directions = sum(
        float(selected.metrics["penalized_target_distance_mean"])
        < float(no_op.metrics["penalized_target_distance_mean"])
        for selected, no_op in zip(candidates[selected_key], no_op_results)
    )
    required_improvements = int(math.ceil(0.75 * direction_count))
    if selected_key == no_op_key or improved_directions < required_improvements:
        return {
            "status": "validation_negative",
            "reason": (
                "the lexicographic winner was no-op or did not improve the required "
                "three-quarters of directed tasks"
            ),
            "selected": None,
            "best_grid_key": {"alpha": selected_key[0], "tau": selected_key[1]},
            "improved_directions_vs_no_op": improved_directions,
            "required_improved_directions": required_improvements,
            "coverage_failures": coverage_failures,
        }
    return {
        "status": "validation_positive",
        "reason": "frozen coverage and direction-improvement gates passed",
        "selected": {"alpha": selected_key[0], "tau": selected_key[1]},
        "selected_aggregate": selected_aggregate,
        "improved_directions_vs_no_op": improved_directions,
        "required_improved_directions": required_improvements,
        "coverage_failures": coverage_failures,
    }


def binary_ranking_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float | int]:
    score_arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    label_arr = np.asarray(labels, dtype=bool).reshape(-1)
    finite = np.isfinite(score_arr)
    score_arr = score_arr[finite]
    label_arr = label_arr[finite]
    positives = int(label_arr.sum())
    negatives = int(label_arr.size - positives)
    if positives == 0 or negatives == 0:
        return {
            "cells": int(label_arr.size),
            "positives": positives,
            "negatives": negatives,
            "auroc": math.nan,
            "average_precision": math.nan,
        }

    ranks = rankdata(score_arr, method="average")
    positive_rank_sum = float(ranks[label_arr].sum())
    auroc = (
        positive_rank_sum - (positives * (positives + 1) / 2.0)
    ) / float(positives * negatives)

    order = np.argsort(-score_arr, kind="mergesort")
    sorted_scores = score_arr[order]
    sorted_labels = label_arr[order]
    group_end = np.r_[np.flatnonzero(np.diff(sorted_scores) != 0), sorted_scores.size - 1]
    cumulative_true = np.cumsum(sorted_labels, dtype=np.int64)
    true_at_threshold = cumulative_true[group_end]
    predicted_at_threshold = group_end + 1
    recall = true_at_threshold / float(positives)
    precision = true_at_threshold / predicted_at_threshold
    recall_increment = np.diff(np.r_[0.0, recall])
    average_precision = float(np.sum(recall_increment * precision))
    return {
        "cells": int(label_arr.size),
        "positives": positives,
        "negatives": negatives,
        "auroc": float(auroc),
        "average_precision": average_precision,
    }


def compare_shifted_residual(
    residual: np.ndarray,
    residual_valid: np.ndarray,
    score: ScoreResult,
    *,
    shift_rows: int = SHIFT_ROWS,
    shift_cols: int = SHIFT_COLS,
) -> dict[str, dict[str, float | int]]:
    residual_arr = np.asarray(residual, dtype=np.float32)
    residual_mask = np.asarray(residual_valid, dtype=bool) & np.isfinite(residual_arr)
    if residual_arr.shape != score.eligible.shape or residual_mask.shape != score.eligible.shape:
        raise ValueError("residual shape must match score maps")
    bad = score.prediction_valid & (
        (score.target_distance > BAD_CELL_DISTANCE) | score.sheet_switch
    )
    shifted = np.roll(residual_arr, shift=(int(shift_rows), int(shift_cols)), axis=(0, 1))
    shifted_mask = np.roll(
        residual_mask, shift=(int(shift_rows), int(shift_cols)), axis=(0, 1)
    )
    common = score.eligible & residual_mask & shifted_mask
    return {
        "real": binary_ranking_metrics(residual_arr[common], bad[common]),
        "shifted": binary_ranking_metrics(shifted[common], bad[common]),
    }
