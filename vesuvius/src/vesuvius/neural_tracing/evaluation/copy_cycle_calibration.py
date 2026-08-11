"""Deterministic local-frame calibration for neural copy displacement.

The calibration is deliberately small: a no-intercept ridge map from local
forward/cycle vectors to a local correction vector.  Training uses only
development ground truth; applying a fitted model does not use a target.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from vesuvius.neural_tracing.evaluation.copy_cycle_metrics import DistanceIndex


CALIBRATION_SCHEMA_VERSION = 1
FROZEN_RIDGE = 10.0
FROZEN_CORRECTION_CAP = 8.0
COPY_TRAINING_VOXEL_SIZE_UM = 4.8

FeatureMode = Literal["cycle_displacement", "displacement", "cycle"]
FEATURE_ORDERS: dict[FeatureMode, tuple[str, ...]] = {
    "cycle_displacement": (
        "cycle_column",
        "cycle_row",
        "cycle_normal",
        "displacement_column",
        "displacement_row",
        "displacement_normal",
    ),
    "displacement": (
        "displacement_column",
        "displacement_row",
        "displacement_normal",
    ),
    "cycle": ("cycle_column", "cycle_row", "cycle_normal"),
}
INFERENCE_SIGNATURE_KEYS = (
    "scroll_id",
    "inference_implementation_commit",
    "checkpoint_sha256",
    "manifest_sha256",
    "requested_copy_args",
    "volume_path",
    "volume_backend",
    "volume_scale_requested",
    "volume_scale_resolved",
    "tifxyz_voxel_size_um",
    "crop_size",
    "runtime",
)
CROSS_SCROLL_INVARIANT_KEYS = (
    "inference_implementation_commit",
    "checkpoint_sha256",
    "requested_copy_args",
    "runtime",
)
_TRAINING_DIRECTIONS = {
    "holdout": ("1->2", "2->1", "2->3"),
    "final": ("1->2", "2->1", "2->3", "3->2", "3->4", "4->3"),
}


def _checked_grid(
    grid: np.ndarray, valid: np.ndarray, label: str
) -> tuple[np.ndarray, np.ndarray]:
    grid_arr = np.asarray(grid, dtype=np.float64)
    valid_arr = np.asarray(valid, dtype=bool)
    if grid_arr.ndim != 3 or grid_arr.shape[2] != 3:
        raise ValueError(f"{label} grid must have shape [R, C, 3], got {grid_arr.shape}")
    if valid_arr.shape != grid_arr.shape[:2]:
        raise ValueError(
            f"{label} validity shape {valid_arr.shape} does not match {grid_arr.shape[:2]}"
        )
    return grid_arr, valid_arr & np.isfinite(grid_arr).all(axis=2)


def _required_hash(value: Any, length: int, label: str) -> str:
    normalized = str(value)
    if re.fullmatch(rf"[0-9a-f]{{{int(length)}}}", normalized) is None:
        raise ValueError(f"{label} must be a full lowercase hexadecimal hash")
    return normalized


def _validate_inference_signature(payload: Mapping[str, Any]) -> dict[str, Any]:
    if set(payload) != set(INFERENCE_SIGNATURE_KEYS):
        raise ValueError("inference provenance has unexpected or missing keys")
    scroll_id = str(payload["scroll_id"])
    volume_path = str(payload["volume_path"])
    volume_backend = str(payload["volume_backend"])
    if not scroll_id or not volume_path or not volume_backend:
        raise ValueError("inference provenance strings cannot be empty")
    requested_copy_args = payload["requested_copy_args"]
    runtime = payload["runtime"]
    if not isinstance(requested_copy_args, Mapping):
        raise ValueError("requested copy arguments must be an object")
    if not isinstance(runtime, Mapping) or not runtime:
        raise ValueError("runtime provenance must be a non-empty object")
    scale_requested = payload["volume_scale_requested"]
    scale_resolved = payload["volume_scale_resolved"]
    if (
        isinstance(scale_requested, bool)
        or not isinstance(scale_requested, int)
        or scale_requested < 0
        or isinstance(scale_resolved, bool)
        or not isinstance(scale_resolved, int)
        or scale_resolved < 0
    ):
        raise ValueError("volume scales must be non-negative integers")
    voxel_size = float(payload["tifxyz_voxel_size_um"])
    if not math.isfinite(voxel_size) or voxel_size <= 0.0:
        raise ValueError("TIFXYZ voxel size must be finite and positive")
    crop_size = payload["crop_size"]
    if (
        not isinstance(crop_size, (list, tuple))
        or len(crop_size) != 3
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in crop_size
        )
    ):
        raise ValueError("crop size must contain three positive integers")
    return {
        "scroll_id": scroll_id,
        "inference_implementation_commit": _required_hash(
            payload["inference_implementation_commit"], 40, "inference implementation commit"
        ),
        "checkpoint_sha256": _required_hash(
            payload["checkpoint_sha256"], 64, "checkpoint SHA-256"
        ),
        "manifest_sha256": _required_hash(
            payload["manifest_sha256"], 64, "manifest SHA-256"
        ),
        "requested_copy_args": dict(requested_copy_args),
        "volume_path": volume_path,
        "volume_backend": volume_backend,
        "volume_scale_requested": int(scale_requested),
        "volume_scale_resolved": int(scale_resolved),
        "tifxyz_voxel_size_um": voxel_size,
        "crop_size": [int(value) for value in crop_size],
        "runtime": dict(runtime),
    }


def inference_receipt_signature(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the inference fields that must stay fixed across a run block."""

    copy_args = receipt.get("copy_args")
    if not isinstance(copy_args, Mapping):
        raise ValueError("run receipt copy_args must be an object")
    return _validate_inference_signature(
        {
            "scroll_id": receipt.get("scroll_id"),
            "inference_implementation_commit": receipt.get("implementation_commit"),
            "checkpoint_sha256": receipt.get("checkpoint_sha256"),
            "manifest_sha256": receipt.get("manifest_sha256"),
            "requested_copy_args": receipt.get("requested_copy_args"),
            "volume_path": receipt.get("volume_path"),
            "volume_backend": receipt.get("volume_backend"),
            "volume_scale_requested": receipt.get("volume_scale_requested"),
            "volume_scale_resolved": receipt.get("volume_scale_resolved"),
            "tifxyz_voxel_size_um": copy_args.get("tifxyz_voxel_size_um"),
            "crop_size": receipt.get("crop_size"),
            "runtime": receipt.get("runtime"),
        }
    )


def training_inference_signature(training: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and extract the inference signature serialized with a model."""

    return _validate_inference_signature(
        {key: training.get(key) for key in INFERENCE_SIGNATURE_KEYS}
    )


def _validate_training_provenance(training: Mapping[str, Any]) -> None:
    expected_keys = set(INFERENCE_SIGNATURE_KEYS) | {
        "stage",
        "directions",
        "sources",
        "receipts",
        "rows_by_direction",
        "total_rows",
    }
    if set(training) != expected_keys:
        raise ValueError("training provenance has unexpected or missing keys")
    training_inference_signature(training)
    stage = str(training["stage"])
    if stage not in _TRAINING_DIRECTIONS:
        raise ValueError(f"unknown calibration training stage: {stage!r}")
    directions = list(training["directions"])
    if directions != list(_TRAINING_DIRECTIONS[stage]):
        raise ValueError("training directions do not match the frozen stage")
    expected_sources = [1, 2] if stage == "holdout" else [1, 2, 3, 4]
    if list(training["sources"]) != expected_sources:
        raise ValueError("training sources do not match the frozen stage")
    rows = training["rows_by_direction"]
    if not isinstance(rows, Mapping) or set(rows) != set(directions):
        raise ValueError("training row counts must cover every frozen direction")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in rows.values()
    ):
        raise ValueError("training row counts must be positive integers")
    total_rows = training["total_rows"]
    if (
        isinstance(total_rows, bool)
        or not isinstance(total_rows, int)
        or total_rows != sum(rows.values())
    ):
        raise ValueError("total training rows must equal the per-direction sum")
    receipts = training["receipts"]
    if not isinstance(receipts, list) or not receipts:
        raise ValueError("training provenance must contain receipt records")
    seen_paths: set[str] = set()
    for receipt in receipts:
        if not isinstance(receipt, Mapping) or set(receipt) != {"path", "sha256"}:
            raise ValueError("training receipt records must contain path and SHA-256")
        path = str(receipt["path"])
        if not path or path in seen_paths:
            raise ValueError("training receipt paths must be non-empty and unique")
        seen_paths.add(path)
        _required_hash(receipt["sha256"], 64, "training receipt SHA-256")


def _axis_difference(
    grid: np.ndarray, valid: np.ndarray, axis: int
) -> tuple[np.ndarray, np.ndarray]:
    plus = np.roll(grid, -1, axis=axis)
    minus = np.roll(grid, 1, axis=axis)
    plus_valid = np.roll(valid, -1, axis=axis)
    minus_valid = np.roll(valid, 1, axis=axis)
    high_edge = [slice(None), slice(None)]
    high_edge[axis] = -1
    plus_valid[tuple(high_edge)] = False
    low_edge = [slice(None), slice(None)]
    low_edge[axis] = 0
    minus_valid[tuple(low_edge)] = False

    both = valid & plus_valid & minus_valid
    forward = valid & plus_valid & ~minus_valid
    backward = valid & minus_valid & ~plus_valid
    difference = np.zeros_like(grid, dtype=np.float64)
    difference[both] = 0.5 * (plus[both] - minus[both])
    difference[forward] = plus[forward] - grid[forward]
    difference[backward] = grid[backward] - minus[backward]
    return difference, both | forward | backward


def build_local_frames(
    source_grid: np.ndarray, source_valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return orthonormal [column, row, normal] frame rows per source cell."""

    source, valid = _checked_grid(source_grid, source_valid, "source")
    row_delta, row_ok = _axis_difference(source, valid, axis=0)
    column_delta, column_ok = _axis_difference(source, valid, axis=1)

    column_norm = np.linalg.norm(column_delta, axis=2)
    column_good = column_norm > 1e-6
    column = np.zeros_like(column_delta)
    column[column_good] = (
        column_delta[column_good] / column_norm[column_good, None]
    )

    row = row_delta - np.sum(row_delta * column, axis=2)[..., None] * column
    row_norm = np.linalg.norm(row, axis=2)
    row_good = row_norm > 1e-6
    row[row_good] = row[row_good] / row_norm[row_good, None]

    normal = np.cross(column, row)
    normal_norm = np.linalg.norm(normal, axis=2)
    normal_good = normal_norm > 1e-6
    normal[normal_good] = normal[normal_good] / normal_norm[normal_good, None]

    frame_valid = (
        valid & row_ok & column_ok & row_good & column_good & normal_good
    )
    frames = np.stack((column, row, normal), axis=2)
    frames[~frame_valid] = 0.0
    return frames, frame_valid


def vectors_to_local(vectors: np.ndarray, frames: np.ndarray) -> np.ndarray:
    return np.einsum("...ij,...j->...i", frames, vectors)


def vectors_to_world(vectors: np.ndarray, frames: np.ndarray) -> np.ndarray:
    return np.einsum("...i,...ij->...j", vectors, frames)


def _nearest_points(index: DistanceIndex, points: np.ndarray) -> np.ndarray:
    points_arr = np.asarray(points, dtype=np.float64)
    if points_arr.ndim != 2 or points_arr.shape[1] != 3:
        raise ValueError(f"nearest-point queries require shape [N, 3], got {points_arr.shape}")
    best_distance = np.full(points_arr.shape[0], np.inf, dtype=np.float64)
    best_points = np.zeros_like(points_arr)
    for tree in index.trees:
        distance, indices = tree.query(points_arr, workers=-1)
        better = distance < best_distance
        if np.any(better):
            best_distance[better] = distance[better]
            best_points[better] = np.asarray(tree.data[indices[better]], dtype=np.float64)
    if not np.isfinite(best_distance).all():
        raise ValueError("target index returned a non-finite nearest-point distance")
    return best_points


@dataclass(frozen=True)
class CalibrationRows:
    cycle_local: np.ndarray
    displacement_local: np.ndarray
    target_correction_local: np.ndarray

    def __post_init__(self) -> None:
        shapes = [
            tuple(np.asarray(self.cycle_local).shape),
            tuple(np.asarray(self.displacement_local).shape),
            tuple(np.asarray(self.target_correction_local).shape),
        ]
        if len(set(shapes)) != 1 or len(shapes[0]) != 2 or shapes[0][1] != 3:
            raise ValueError("calibration rows must share shape [N, 3]")
        if np.asarray(self.cycle_local).shape[0] == 0:
            raise ValueError("calibration rows cannot be empty")
        if not all(
            np.isfinite(np.asarray(values, dtype=np.float64)).all()
            for values in (
                self.cycle_local,
                self.displacement_local,
                self.target_correction_local,
            )
        ):
            raise ValueError("calibration rows must be finite")

    @property
    def count(self) -> int:
        return int(np.asarray(self.cycle_local).shape[0])


def extract_calibration_rows(
    source_grid: np.ndarray,
    source_valid: np.ndarray,
    forward_grid: np.ndarray,
    forward_valid: np.ndarray,
    return_grid: np.ndarray,
    return_valid: np.ndarray,
    eligible: np.ndarray,
    target_index: DistanceIndex,
) -> CalibrationRows:
    """Extract common, target-supervised rows for all frozen control models."""

    source, source_mask = _checked_grid(source_grid, source_valid, "source")
    forward, forward_mask = _checked_grid(forward_grid, forward_valid, "forward")
    returned, return_mask = _checked_grid(return_grid, return_valid, "return")
    if forward.shape != source.shape or returned.shape != source.shape:
        raise ValueError("source, forward, and return grids must share a shape")
    eligible_mask = np.asarray(eligible, dtype=bool)
    if eligible_mask.shape != source.shape[:2]:
        raise ValueError("eligible mask must match the source lattice")

    frames, frame_valid = build_local_frames(source, source_mask)
    training_mask = (
        eligible_mask & source_mask & forward_mask & return_mask & frame_valid
    )
    if not np.any(training_mask):
        raise ValueError("no common eligible calibration rows")

    cycle = returned - source
    displacement = forward - source
    nearest = _nearest_points(target_index, forward[training_mask])
    target_correction = nearest - forward[training_mask]
    selected_frames = frames[training_mask]
    return CalibrationRows(
        cycle_local=vectors_to_local(cycle[training_mask], selected_frames),
        displacement_local=vectors_to_local(
            displacement[training_mask], selected_frames
        ),
        target_correction_local=vectors_to_local(
            target_correction, selected_frames
        ),
    )


def concatenate_calibration_rows(rows: Sequence[CalibrationRows]) -> CalibrationRows:
    if not rows:
        raise ValueError("at least one calibration row block is required")
    return CalibrationRows(
        cycle_local=np.concatenate([item.cycle_local for item in rows], axis=0),
        displacement_local=np.concatenate(
            [item.displacement_local for item in rows], axis=0
        ),
        target_correction_local=np.concatenate(
            [item.target_correction_local for item in rows], axis=0
        ),
    )


def _features(rows: CalibrationRows, mode: FeatureMode) -> np.ndarray:
    if mode == "cycle_displacement":
        return np.column_stack((rows.cycle_local, rows.displacement_local))
    if mode == "displacement":
        return np.asarray(rows.displacement_local, dtype=np.float64)
    if mode == "cycle":
        return np.asarray(rows.cycle_local, dtype=np.float64)
    raise ValueError(f"unknown calibration feature mode: {mode!r}")


@dataclass(frozen=True)
class LocalLinearModel:
    mode: FeatureMode
    rms: np.ndarray
    coefficients: np.ndarray
    ridge: float = FROZEN_RIDGE
    correction_cap: float = FROZEN_CORRECTION_CAP

    def __post_init__(self) -> None:
        if self.mode not in FEATURE_ORDERS:
            raise ValueError(f"unknown calibration feature mode: {self.mode!r}")
        rms = np.asarray(self.rms, dtype=np.float64)
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        expected = len(FEATURE_ORDERS[self.mode])
        if rms.shape != (expected,) or coefficients.shape != (expected, 3):
            raise ValueError(
                f"{self.mode} model expects RMS [{expected}] and coefficients "
                f"[{expected}, 3], got {rms.shape} and {coefficients.shape}"
            )
        if not np.isfinite(rms).all() or np.any(rms <= 0.0):
            raise ValueError("model RMS values must be finite and positive")
        if not np.isfinite(coefficients).all():
            raise ValueError("model coefficients must be finite")
        if not math.isfinite(float(self.ridge)) or float(self.ridge) < 0.0:
            raise ValueError("ridge must be finite and non-negative")
        if not math.isfinite(float(self.correction_cap)) or float(self.correction_cap) <= 0.0:
            raise ValueError("correction cap must be finite and positive")

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": CALIBRATION_SCHEMA_VERSION,
            "type": "local_no_intercept_ridge",
            "mode": self.mode,
            "feature_order": list(FEATURE_ORDERS[self.mode]),
            "rms": np.asarray(self.rms, dtype=np.float64).tolist(),
            "coefficients": np.asarray(
                self.coefficients, dtype=np.float64
            ).tolist(),
            "ridge": float(self.ridge),
            "correction_cap": float(self.correction_cap),
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "LocalLinearModel":
        expected_keys = {
            "schema_version",
            "type",
            "mode",
            "feature_order",
            "rms",
            "coefficients",
            "ridge",
            "correction_cap",
        }
        if set(payload) != expected_keys:
            raise ValueError("local calibration model has unexpected or missing keys")
        if int(payload["schema_version"]) != CALIBRATION_SCHEMA_VERSION:
            raise ValueError("unsupported local calibration model schema")
        if payload["type"] != "local_no_intercept_ridge":
            raise ValueError("unsupported local calibration model type")
        mode = str(payload["mode"])
        if mode not in FEATURE_ORDERS:
            raise ValueError(f"unknown calibration feature mode: {mode!r}")
        if tuple(payload["feature_order"]) != FEATURE_ORDERS[mode]:
            raise ValueError("calibration feature order does not match its mode")
        return cls(
            mode=mode,
            rms=np.asarray(payload["rms"], dtype=np.float64),
            coefficients=np.asarray(payload["coefficients"], dtype=np.float64),
            ridge=float(payload["ridge"]),
            correction_cap=float(payload["correction_cap"]),
        )


def fit_local_linear_model(
    rows: CalibrationRows,
    mode: FeatureMode,
    *,
    ridge: float = FROZEN_RIDGE,
    correction_cap: float = FROZEN_CORRECTION_CAP,
) -> LocalLinearModel:
    features = np.asarray(_features(rows, mode), dtype=np.float64)
    target = np.asarray(rows.target_correction_local, dtype=np.float64)
    rms = np.sqrt(np.mean(np.square(features), axis=0))
    if np.any(~np.isfinite(rms)) or np.any(rms <= 1e-12):
        raise ValueError("calibration features must have non-zero finite RMS")
    normalized = features / rms
    regularizer = float(ridge) * np.eye(normalized.shape[1], dtype=np.float64)
    coefficients = np.linalg.solve(
        normalized.T @ normalized + regularizer,
        normalized.T @ target,
    )
    return LocalLinearModel(
        mode=mode,
        rms=rms,
        coefficients=coefficients,
        ridge=float(ridge),
        correction_cap=float(correction_cap),
    )


def _cap_vectors(vectors: np.ndarray, cap: float) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1)
    scales = np.minimum(1.0, float(cap) / np.maximum(norms, 1e-12))
    return vectors * scales[:, None]


def apply_local_linear_model(
    model: LocalLinearModel,
    source_grid: np.ndarray,
    source_valid: np.ndarray,
    forward_grid: np.ndarray,
    forward_valid: np.ndarray,
    return_grid: np.ndarray | None = None,
    return_valid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply a fitted model without target data and preserve baseline validity."""

    source, source_mask = _checked_grid(source_grid, source_valid, "source")
    forward, forward_mask = _checked_grid(forward_grid, forward_valid, "forward")
    if forward.shape != source.shape:
        raise ValueError("source and forward grids must share a shape")
    frames, frame_valid = build_local_frames(source, source_mask)
    application_mask = source_mask & forward_mask & frame_valid
    displacement_local = vectors_to_local(forward - source, frames)

    cycle_local: np.ndarray | None = None
    if model.mode in {"cycle_displacement", "cycle"}:
        if return_grid is None or return_valid is None:
            raise ValueError(f"{model.mode} model requires a return grid")
        returned, return_mask = _checked_grid(return_grid, return_valid, "return")
        if returned.shape != source.shape:
            raise ValueError("source and return grids must share a shape")
        application_mask &= return_mask
        cycle_local = vectors_to_local(returned - source, frames)

    if model.mode == "cycle_displacement":
        assert cycle_local is not None
        full_features = np.concatenate((cycle_local, displacement_local), axis=2)
    elif model.mode == "displacement":
        full_features = displacement_local
    else:
        assert cycle_local is not None
        full_features = cycle_local

    selected = full_features[application_mask]
    local_correction = (selected / np.asarray(model.rms)) @ np.asarray(
        model.coefficients
    )
    world_correction = vectors_to_world(
        local_correction, frames[application_mask]
    )
    world_correction = _cap_vectors(world_correction, model.correction_cap)
    corrected = forward.copy()
    corrected[application_mask] += world_correction
    return corrected.astype(np.float32), forward_mask.copy(), application_mask


def fit_scalar_displacement_correction(rows: CalibrationRows) -> float:
    displacement = np.asarray(rows.displacement_local, dtype=np.float64)
    target = np.asarray(rows.target_correction_local, dtype=np.float64)
    denominator = float(np.sum(displacement * displacement))
    if not math.isfinite(denominator) or denominator <= 1e-12:
        raise ValueError("cannot fit scalar correction to zero displacement")
    beta = float(np.sum(displacement * target) / denominator)
    if not math.isfinite(beta):
        raise ValueError("fitted scalar correction is non-finite")
    return beta


def apply_scalar_displacement_correction(
    source_grid: np.ndarray,
    source_valid: np.ndarray,
    forward_grid: np.ndarray,
    forward_valid: np.ndarray,
    *,
    beta: float,
    correction_cap: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source, source_mask = _checked_grid(source_grid, source_valid, "source")
    forward, forward_mask = _checked_grid(forward_grid, forward_valid, "forward")
    if forward.shape != source.shape:
        raise ValueError("source and forward grids must share a shape")
    if not math.isfinite(float(beta)):
        raise ValueError("scalar correction beta must be finite")
    application_mask = source_mask & forward_mask
    correction = float(beta) * (forward[application_mask] - source[application_mask])
    if correction_cap is not None:
        if not math.isfinite(float(correction_cap)) or float(correction_cap) <= 0.0:
            raise ValueError("scalar correction cap must be finite and positive")
        correction = _cap_vectors(correction, float(correction_cap))
    corrected = forward.copy()
    corrected[application_mask] += correction
    return corrected.astype(np.float32), forward_mask.copy(), application_mask


@dataclass(frozen=True)
class CalibrationBundle:
    combined: LocalLinearModel
    displacement_only: LocalLinearModel
    cycle_only: LocalLinearModel
    fitted_scalar_beta: float
    implementation_commit: str
    training: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.combined.mode != "cycle_displacement":
            raise ValueError("combined model must use cycle_displacement features")
        if self.displacement_only.mode != "displacement":
            raise ValueError("displacement-only model has the wrong feature mode")
        if self.cycle_only.mode != "cycle":
            raise ValueError("cycle-only model has the wrong feature mode")
        if not math.isfinite(float(self.fitted_scalar_beta)):
            raise ValueError("fitted scalar beta must be finite")
        _required_hash(self.implementation_commit, 40, "implementation commit")
        if not isinstance(self.training, Mapping):
            raise ValueError("training provenance must be an object")
        _validate_training_provenance(self.training)

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": CALIBRATION_SCHEMA_VERSION,
            "method": "copy_cycle_local_linear_v1",
            "implementation_commit": self.implementation_commit,
            "copy_training_voxel_size_um": COPY_TRAINING_VOXEL_SIZE_UM,
            "combined": self.combined.to_json(),
            "displacement_only": self.displacement_only.to_json(),
            "cycle_only": self.cycle_only.to_json(),
            "fitted_scalar_beta": float(self.fitted_scalar_beta),
            "training": dict(self.training),
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "CalibrationBundle":
        expected_keys = {
            "schema_version",
            "method",
            "implementation_commit",
            "copy_training_voxel_size_um",
            "combined",
            "displacement_only",
            "cycle_only",
            "fitted_scalar_beta",
            "training",
        }
        if set(payload) != expected_keys:
            raise ValueError("calibration bundle has unexpected or missing keys")
        if int(payload["schema_version"]) != CALIBRATION_SCHEMA_VERSION:
            raise ValueError("unsupported calibration bundle schema")
        if payload["method"] != "copy_cycle_local_linear_v1":
            raise ValueError("unsupported calibration bundle method")
        if not math.isclose(
            float(payload["copy_training_voxel_size_um"]),
            COPY_TRAINING_VOXEL_SIZE_UM,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError("copy training voxel size does not match the frozen protocol")
        return cls(
            combined=LocalLinearModel.from_json(payload["combined"]),
            displacement_only=LocalLinearModel.from_json(
                payload["displacement_only"]
            ),
            cycle_only=LocalLinearModel.from_json(payload["cycle_only"]),
            fitted_scalar_beta=float(payload["fitted_scalar_beta"]),
            implementation_commit=str(payload["implementation_commit"]),
            training=payload["training"],
        )


def fit_calibration_bundle(
    rows: CalibrationRows,
    *,
    implementation_commit: str,
    training: Mapping[str, Any],
) -> CalibrationBundle:
    if training.get("total_rows") != rows.count:
        raise ValueError("training provenance row total does not match fitted rows")
    return CalibrationBundle(
        combined=fit_local_linear_model(rows, "cycle_displacement"),
        displacement_only=fit_local_linear_model(rows, "displacement"),
        cycle_only=fit_local_linear_model(rows, "cycle"),
        fitted_scalar_beta=fit_scalar_displacement_correction(rows),
        implementation_commit=implementation_commit,
        training=training,
    )
