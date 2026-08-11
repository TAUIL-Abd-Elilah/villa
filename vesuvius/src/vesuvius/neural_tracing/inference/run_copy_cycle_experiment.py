"""Run forward and round-trip copy inference for a frozen benchmark phase."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Sequence

from vesuvius.neural_tracing.evaluation.copy_cycle_calibration import (
    FROZEN_CORRECTION_CAP,
    FROZEN_RIDGE,
    CalibrationBundle,
    inference_receipt_signature,
    training_inference_signature,
)
from vesuvius.neural_tracing.evaluation.copy_cycle_io import (
    clean_git_commit,
    sha256_file,
    sha256_tifxyz,
    write_json_atomic,
)
from vesuvius.neural_tracing.evaluation.copy_cycle_metrics import (
    ALPHA_GRID,
    TAU_GRID,
    choose_return_branch,
)
from vesuvius.neural_tracing.evaluation.prepare_copy_cycle_benchmark import (
    load_manifest,
    prepare_benchmark,
)


_DISALLOWED_COPY_ARGS = {
    "checkpoint_path",
    "iter_direction",
    "iterations",
    "keep_previous_wrap",
    "out_dir",
    "output_prefix",
    "save_original_copy",
    "tifxyz_path",
    "tifxyz_voxel_size_um",
    "volume_path",
    "volume_scale",
}

_LEARNED_METHOD = "copy_cycle_local_linear_v1"
_LEARNED_SCORE_METHOD = "copy_cycle_local_linear_v1_score"
_LEARNED_VALIDATION_CONDITIONS = {
    "aggregate_penalty_improves_at_least_10pct",
    "required_directions_improve",
    "beats_each_displacement_control",
    "incremental_gain_over_best_control_at_least_1pct_baseline",
    "coverage_unchanged_each_direction",
    "p95_noninferiority",
    "sheet_switch_gate",
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("rt", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _resolve_path(config_dir: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    return path.resolve()


def _phase_edges(manifest: dict[str, Any], phase: str) -> list[tuple[int, int]]:
    key = {
        "development": "development_edges",
        "validation": "validation_edges",
        "sealed_test": "sealed_test_edges",
    }.get(str(phase))
    if key is None:
        raise ValueError(f"unknown phase: {phase!r}")
    source = manifest.get("splits", manifest)
    raw = source.get(key)
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"manifest does not define non-empty {key}")
    return [(int(edge[0]), int(edge[1])) for edge in raw]


def phase_source_wraps(manifest: dict[str, Any], phase: str) -> set[int]:
    return {wrap for edge in _phase_edges(manifest, phase) for wrap in edge}


def _parse_frozen_tau(value: Any) -> float:
    if isinstance(value, str) and value.strip().lower() in {"inf", "infinity"}:
        return float("inf")
    return float(value)


def _require_sha256(value: Any, label: str) -> str:
    normalized = str(value)
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{label} is not a lowercase SHA-256")
    return normalized


def _validate_authorization_envelope(
    authorization: dict[str, Any],
    current_commit: str,
    *,
    actual_validation_score_sha256: str,
    actual_validation_receipt_sha256: str,
    authorization_now_utc: datetime | None,
) -> None:
    if int(authorization.get("schema_version", -1)) != 1:
        raise ValueError("test authorization schema_version must be 1")
    if str(authorization.get("status")) != "authorized":
        raise ValueError("test authorization status must be 'authorized'")
    if str(authorization.get("validation_status")) != "validation_positive":
        raise ValueError("sealed test requires a validation_positive authorization")
    if str(authorization.get("implementation_commit")) != str(current_commit):
        raise ValueError(
            "test authorization implementation_commit must equal the checked-out commit"
        )
    authorization_score_sha = _require_sha256(
        authorization.get("validation_score_sha256"),
        "test authorization validation_score_sha256",
    )
    authorization_receipt_sha = _require_sha256(
        authorization.get("validation_receipt_sha256"),
        "test authorization validation_receipt_sha256",
    )
    if authorization_score_sha != actual_validation_score_sha256:
        raise ValueError("test authorization validation score file SHA-256 mismatch")
    if authorization_receipt_sha != actual_validation_receipt_sha256:
        raise ValueError("test authorization validation receipt file SHA-256 mismatch")
    overlap_audit_utc = str(authorization.get("overlap_audit_utc", "")).strip()
    if not overlap_audit_utc:
        raise ValueError("test authorization must record overlap_audit_utc")
    try:
        parsed_audit_time = datetime.fromisoformat(
            overlap_audit_utc.replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise ValueError(
            "test authorization overlap_audit_utc must be ISO-8601"
        ) from exc
    if parsed_audit_time.tzinfo is None:
        raise ValueError("test authorization overlap_audit_utc must include a timezone")
    reference_time = authorization_now_utc or datetime.now(timezone.utc)
    if reference_time.tzinfo is None:
        raise ValueError("authorization_now_utc must include a timezone")
    audit_age = reference_time.astimezone(timezone.utc) - parsed_audit_time.astimezone(
        timezone.utc
    )
    if audit_age.total_seconds() < -300.0 or audit_age.total_seconds() > 72.0 * 3600.0:
        raise ValueError("test authorization overlap audit must be within the last 72 hours")
    public_url = str(authorization.get("validation_public_url", "")).strip()
    if not public_url.startswith("https://github.com/"):
        raise ValueError("test authorization validation_public_url must be a GitHub URL")


def _validate_validation_receipt(
    validation_receipt: dict[str, Any],
    current_commit: str,
    *,
    expected_checkpoint_sha256: str,
    expected_validation_manifest_sha256: str,
    expected_copy_args: dict[str, Any],
) -> None:
    expected_checkpoint_sha256 = _require_sha256(
        expected_checkpoint_sha256, "expected checkpoint SHA-256"
    )
    expected_validation_manifest_sha256 = _require_sha256(
        expected_validation_manifest_sha256, "expected validation manifest SHA-256"
    )
    if (
        validation_receipt.get("completed") is not True
        or validation_receipt.get("phase") != "validation"
        or validation_receipt.get("scroll_id") != "PHerc0500P2"
    ):
        raise ValueError("validation receipt is not a complete PHerc0500P2 validation run")
    if str(validation_receipt.get("implementation_commit")) != str(current_commit):
        raise ValueError("validation receipt implementation commit does not match test code")
    if str(validation_receipt.get("checkpoint_sha256")) != expected_checkpoint_sha256:
        raise ValueError("validation receipt checkpoint does not match the frozen checkpoint")
    if str(validation_receipt.get("manifest_sha256")) != expected_validation_manifest_sha256:
        raise ValueError("validation receipt does not use the frozen validation manifest")
    if validation_receipt.get("requested_copy_args") != expected_copy_args:
        raise ValueError(
            "sealed-test copy_args must exactly match the validation inference settings"
        )
    receipt_sources = validation_receipt.get("sources")
    if (
        not isinstance(receipt_sources, list)
        or len(receipt_sources) != 3
        or not all(isinstance(item, dict) for item in receipt_sources)
        or {int(item.get("wrap", -1)) for item in receipt_sources} != {5, 6, 7}
    ):
        raise ValueError("validation receipt does not contain source wraps 5, 6, and 7")


def validate_test_authorization(
    authorization: dict[str, Any],
    current_commit: str,
    *,
    validation_score: dict[str, Any],
    validation_receipt: dict[str, Any],
    actual_validation_score_sha256: str,
    actual_validation_receipt_sha256: str,
    expected_checkpoint_sha256: str,
    expected_validation_manifest_sha256: str,
    expected_copy_args: dict[str, Any],
    authorization_now_utc: datetime | None = None,
) -> tuple[float, float]:
    _validate_authorization_envelope(
        authorization,
        current_commit,
        actual_validation_score_sha256=actual_validation_score_sha256,
        actual_validation_receipt_sha256=actual_validation_receipt_sha256,
        authorization_now_utc=authorization_now_utc,
    )
    selected = authorization.get("selected")
    if not isinstance(selected, dict):
        raise ValueError("test authorization must contain selected alpha/tau")
    alpha = float(selected.get("alpha"))
    tau = _parse_frozen_tau(selected.get("tau"))
    if alpha not in ALPHA_GRID or tau not in TAU_GRID:
        raise ValueError("test authorization selected parameters are outside the frozen grid")
    expected_checkpoint_sha256 = _require_sha256(
        expected_checkpoint_sha256, "expected checkpoint SHA-256"
    )
    expected_validation_manifest_sha256 = _require_sha256(
        expected_validation_manifest_sha256, "expected validation manifest SHA-256"
    )
    if int(validation_score.get("schema_version", -1)) != 1:
        raise ValueError("validation score schema_version must be 1")
    if (
        validation_score.get("mode") != "grid"
        or validation_score.get("phase") != "validation"
        or validation_score.get("scroll_id") != "PHerc0500P2"
    ):
        raise ValueError("authorization requires the frozen PHerc0500P2 validation grid")
    if str(validation_score.get("implementation_commit")) != str(current_commit):
        raise ValueError("validation score implementation commit does not match test code")
    if str(validation_score.get("checkpoint_sha256")) != expected_checkpoint_sha256:
        raise ValueError("validation score checkpoint does not match the frozen checkpoint")
    if str(validation_score.get("manifest_sha256")) != expected_validation_manifest_sha256:
        raise ValueError("validation score does not use the frozen validation manifest")
    if str(validation_score.get("receipt_sha256")) != actual_validation_receipt_sha256:
        raise ValueError("validation score is not bound to the supplied validation receipt")
    direction_identity = validation_score.get("direction_identity")
    expected_directions = {(5, 6), (6, 5), (6, 7), (7, 6)}
    if (
        not isinstance(direction_identity, list)
        or len(direction_identity) != 4
        or not all(isinstance(item, dict) for item in direction_identity)
        or {
            (int(item.get("source", -1)), int(item.get("target", -1)))
            for item in direction_identity
        }
        != expected_directions
    ):
        raise ValueError("validation score does not contain the four frozen directions")
    result = validation_score.get("result")
    selection = result.get("selection") if isinstance(result, dict) else None
    if not isinstance(selection, dict) or selection.get("status") != "validation_positive":
        raise ValueError("validation score is not validation_positive")
    score_selected = selection.get("selected")
    if not isinstance(score_selected, dict) or (
        float(score_selected.get("alpha")),
        _parse_frozen_tau(score_selected.get("tau")),
    ) != (alpha, tau):
        raise ValueError("authorization parameters do not match validation selection")

    _validate_validation_receipt(
        validation_receipt,
        current_commit,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_validation_manifest_sha256=expected_validation_manifest_sha256,
        expected_copy_args=expected_copy_args,
    )
    return alpha, tau


def _finite_metric(payload: dict[str, Any], key: str, label: str) -> float:
    try:
        value = float(payload[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain numeric {key}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{label} {key} must be finite")
    return value


def _validate_learned_validation_gate(validation_score: dict[str, Any]) -> None:
    expected_directions = {(5, 6), (6, 5), (6, 7), (7, 6)}
    directions = validation_score.get("directions")
    if (
        not isinstance(directions, list)
        or len(directions) != 4
        or not all(isinstance(item, dict) for item in directions)
        or {
            (int(item.get("source", -1)), int(item.get("target", -1)))
            for item in directions
        }
        != expected_directions
    ):
        raise ValueError("learned validation score does not contain the four frozen directions")

    arm_names = {
        "baseline",
        "combined",
        "displacement_only",
        "cycle_only",
        "fitted_scalar",
        "physical_scalar",
    }
    improved_directions = 0
    for direction in directions:
        arms = direction.get("arms")
        if not isinstance(arms, dict) or set(arms) != arm_names:
            raise ValueError("learned validation direction is missing a frozen score arm")
        if not all(isinstance(arms[name], dict) for name in arm_names):
            raise ValueError("learned validation arm metrics must be objects")
        baseline = arms["baseline"]
        combined = arms["combined"]
        baseline_penalty = _finite_metric(
            baseline, "penalized_target_distance_mean", "baseline direction"
        )
        candidate_penalty = _finite_metric(
            combined, "penalized_target_distance_mean", "combined direction"
        )
        improved_directions += candidate_penalty < baseline_penalty
        try:
            baseline_valid = int(baseline["valid_prediction_cells"])
            candidate_valid = int(combined["valid_prediction_cells"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "learned validation directions must record valid prediction cells"
            ) from exc
        if candidate_valid != baseline_valid:
            raise ValueError("learned validation changed prediction coverage")
    if improved_directions < 3:
        raise ValueError("learned validation improves fewer than three frozen directions")

    aggregate = validation_score.get("aggregate")
    if not isinstance(aggregate, dict) or set(aggregate) != arm_names:
        raise ValueError("learned validation aggregate is missing a frozen score arm")
    if not all(isinstance(aggregate[name], dict) for name in arm_names):
        raise ValueError("learned validation aggregate arms must be objects")
    baseline = aggregate["baseline"]
    combined = aggregate["combined"]
    baseline_penalty = _finite_metric(
        baseline, "penalized_target_distance_mean", "baseline aggregate"
    )
    candidate_penalty = _finite_metric(
        combined, "penalized_target_distance_mean", "combined aggregate"
    )
    if baseline_penalty <= 0.0 or candidate_penalty > 0.9 * baseline_penalty:
        raise ValueError("learned validation does not improve aggregate penalty by 10%")
    control_penalties = [
        _finite_metric(
            aggregate[name], "penalized_target_distance_mean", f"{name} aggregate"
        )
        for name in ("displacement_only", "fitted_scalar", "physical_scalar")
    ]
    if not all(candidate_penalty < penalty for penalty in control_penalties):
        raise ValueError("learned validation does not beat every displacement control")
    if min(control_penalties) - candidate_penalty < 0.01 * baseline_penalty:
        raise ValueError(
            "learned validation lacks the frozen incremental gain over controls"
        )
    baseline_p95 = _finite_metric(
        baseline, "target_distance_p95_valid", "baseline aggregate"
    )
    candidate_p95 = _finite_metric(
        combined, "target_distance_p95_valid", "combined aggregate"
    )
    if candidate_p95 > 1.05 * baseline_p95:
        raise ValueError("learned validation fails p95 non-inferiority")
    baseline_switch = _finite_metric(
        baseline, "sheet_switch_rate_all_eligible", "baseline aggregate"
    )
    candidate_switch = _finite_metric(
        combined, "sheet_switch_rate_all_eligible", "combined aggregate"
    )
    switch_passed = (
        candidate_switch <= 0.75 * baseline_switch
        if baseline_switch >= 0.005
        else candidate_switch <= baseline_switch + 0.001
    )
    if not switch_passed:
        raise ValueError("learned validation fails the sheet-switch gate")

    gate = validation_score.get("gate")
    conditions = gate.get("conditions") if isinstance(gate, dict) else None
    try:
        required_improved_directions = int(gate["required_improved_directions"])
        recorded_improved_directions = int(gate["improved_directions"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("learned validation gate direction counts are invalid") from exc
    if (
        not isinstance(conditions, dict)
        or set(conditions) != _LEARNED_VALIDATION_CONDITIONS
        or any(value is not True for value in conditions.values())
        or gate.get("passed") is not True
        or required_improved_directions != 3
        or recorded_improved_directions != improved_directions
    ):
        raise ValueError("learned validation score does not pass every frozen gate")


def validate_learned_test_authorization(
    authorization: dict[str, Any],
    current_commit: str,
    *,
    calibration_model: CalibrationBundle,
    validation_score: dict[str, Any],
    validation_receipt: dict[str, Any],
    actual_calibration_model_sha256: str,
    actual_validation_score_sha256: str,
    actual_validation_receipt_sha256: str,
    expected_checkpoint_sha256: str,
    expected_validation_manifest_sha256: str,
    expected_copy_args: dict[str, Any],
    authorization_now_utc: datetime | None = None,
) -> dict[str, str]:
    _validate_authorization_envelope(
        authorization,
        current_commit,
        actual_validation_score_sha256=actual_validation_score_sha256,
        actual_validation_receipt_sha256=actual_validation_receipt_sha256,
        authorization_now_utc=authorization_now_utc,
    )
    if authorization.get("method") != _LEARNED_METHOD:
        raise ValueError(f"learned test authorization method must be {_LEARNED_METHOD}")
    actual_model_sha = _require_sha256(
        actual_calibration_model_sha256, "actual calibration model SHA-256"
    )
    authorization_model_sha = _require_sha256(
        authorization.get("calibration_model_sha256"),
        "test authorization calibration_model_sha256",
    )
    if authorization_model_sha != actual_model_sha:
        raise ValueError("test authorization calibration model file SHA-256 mismatch")

    expected_checkpoint_sha256 = _require_sha256(
        expected_checkpoint_sha256, "expected checkpoint SHA-256"
    )
    expected_validation_manifest_sha256 = _require_sha256(
        expected_validation_manifest_sha256, "expected validation manifest SHA-256"
    )
    if calibration_model.implementation_commit != current_commit:
        raise ValueError("calibration model was produced by a different implementation commit")
    if calibration_model.training.get("stage") != "final":
        raise ValueError("sealed test requires the frozen final calibration model")
    for model in (
        calibration_model.combined,
        calibration_model.displacement_only,
        calibration_model.cycle_only,
    ):
        if not math.isclose(model.ridge, FROZEN_RIDGE, rel_tol=0.0, abs_tol=0.0):
            raise ValueError("calibration model ridge differs from the frozen protocol")
        if not math.isclose(
            model.correction_cap,
            FROZEN_CORRECTION_CAP,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError("calibration model cap differs from the frozen protocol")
    training_signature = training_inference_signature(calibration_model.training)
    if training_signature["scroll_id"] != "PHerc0500P2":
        raise ValueError("calibration model was not trained on PHerc0500P2")
    if training_signature["inference_implementation_commit"] != current_commit:
        raise ValueError("calibration training inference used a different implementation commit")
    if training_signature["checkpoint_sha256"] != expected_checkpoint_sha256:
        raise ValueError("calibration training used a different checkpoint")
    if training_signature["manifest_sha256"] != expected_validation_manifest_sha256:
        raise ValueError("calibration training used a different PHerc0500P2 manifest")
    if training_signature["requested_copy_args"] != expected_copy_args:
        raise ValueError("calibration training copy_args differ from sealed-test copy_args")

    if (
        int(validation_score.get("schema_version", -1)) != 1
        or validation_score.get("method") != _LEARNED_SCORE_METHOD
        or validation_score.get("stage") != "validation"
        or validation_score.get("scroll_id") != "PHerc0500P2"
    ):
        raise ValueError("authorization requires the learned PHerc0500P2 validation score")
    if str(validation_score.get("implementation_commit")) != current_commit:
        raise ValueError("learned validation score implementation commit does not match test code")
    if str(validation_score.get("model_sha256")) != actual_model_sha:
        raise ValueError("learned validation score is not bound to the calibration model")
    if str(validation_score.get("manifest_sha256")) != expected_validation_manifest_sha256:
        raise ValueError("learned validation score does not use the frozen manifest")
    score_receipts = validation_score.get("receipts")
    if (
        not isinstance(score_receipts, list)
        or len(score_receipts) != 1
        or not isinstance(score_receipts[0], dict)
        or str(score_receipts[0].get("sha256"))
        != actual_validation_receipt_sha256
    ):
        raise ValueError("learned validation score is not bound to the supplied receipt")
    _validate_learned_validation_gate(validation_score)
    _validate_validation_receipt(
        validation_receipt,
        current_commit,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_validation_manifest_sha256=expected_validation_manifest_sha256,
        expected_copy_args=expected_copy_args,
    )
    if inference_receipt_signature(validation_receipt) != training_signature:
        raise ValueError("learned validation inference provenance differs from training")
    return {"method": _LEARNED_METHOD, "model_sha256": actual_model_sha}


def load_and_validate_test_authorization(
    authorization_path: Path,
    current_commit: str,
    *,
    expected_checkpoint_sha256: str,
    expected_validation_manifest_sha256: str,
    expected_copy_args: dict[str, Any],
) -> dict[str, Any]:
    authorization = _load_json(authorization_path)
    for key in ("validation_score_path", "validation_receipt_path"):
        if not isinstance(authorization.get(key), str) or not authorization[key].strip():
            raise ValueError(f"test authorization must define {key}")
    score_path = _resolve_path(
        authorization_path.parent, authorization.get("validation_score_path")
    )
    receipt_path = _resolve_path(
        authorization_path.parent, authorization.get("validation_receipt_path")
    )
    actual_score_sha256 = sha256_file(score_path)
    actual_receipt_sha256 = sha256_file(receipt_path)
    validation_score = _load_json(score_path)
    validation_receipt = _load_json(receipt_path)
    method = authorization.get("method")
    if method == _LEARNED_METHOD:
        model_value = authorization.get("calibration_model_path")
        if not isinstance(model_value, str) or not model_value.strip():
            raise ValueError("learned test authorization must define calibration_model_path")
        model_path = _resolve_path(authorization_path.parent, model_value)
        return validate_learned_test_authorization(
            authorization,
            current_commit,
            calibration_model=CalibrationBundle.from_json(_load_json(model_path)),
            validation_score=validation_score,
            validation_receipt=validation_receipt,
            actual_calibration_model_sha256=sha256_file(model_path),
            actual_validation_score_sha256=actual_score_sha256,
            actual_validation_receipt_sha256=actual_receipt_sha256,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
            expected_validation_manifest_sha256=expected_validation_manifest_sha256,
            expected_copy_args=expected_copy_args,
        )
    if method not in (None, "copy_cycle_scalar_grid_v1"):
        raise ValueError(f"unsupported test authorization method: {method!r}")
    alpha, tau = validate_test_authorization(
        authorization,
        current_commit,
        validation_score=validation_score,
        validation_receipt=validation_receipt,
        actual_validation_score_sha256=actual_score_sha256,
        actual_validation_receipt_sha256=actual_receipt_sha256,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_validation_manifest_sha256=expected_validation_manifest_sha256,
        expected_copy_args=expected_copy_args,
    )
    return {
        "alpha": alpha,
        "tau": "infinity" if tau == float("inf") else tau,
    }


def validate_config_structure(
    config: dict[str, Any], manifest: dict[str, Any]
) -> tuple[str, list[int]]:
    if int(config.get("schema_version", -1)) != 1:
        raise ValueError("experiment config schema_version must be 1")
    if str(config.get("scroll_id")) != str(manifest.get("scroll_id")):
        raise ValueError("experiment config scroll_id must match its manifest")
    phase = str(config.get("phase"))
    allowed = phase_source_wraps(manifest, phase)
    raw_sources = config.get("source_wraps")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("source_wraps must be a non-empty list")
    sources = [int(value) for value in raw_sources]
    if len(set(sources)) != len(sources):
        raise ValueError("source_wraps must be unique")
    if not set(sources).issubset(allowed):
        raise ValueError(
            f"source_wraps must be inside phase set {sorted(allowed)}, got {sorted(sources)}"
        )
    if phase in {"validation", "sealed_test"} and set(sources) != allowed:
        raise ValueError(
            f"{phase} requires the complete source set {sorted(allowed)}, got {sorted(sources)}"
        )
    if int(config.get("volume_scale", -1)) != int(manifest.get("volume_scale", 0)):
        raise ValueError("config volume_scale must match the benchmark manifest")
    if str(config.get("volume_path", "")).rstrip("/") != str(
        manifest.get("volume_url", "")
    ).rstrip("/"):
        raise ValueError("config volume_path must equal the frozen manifest volume_url")
    copy_args = config.get("copy_args", {})
    if not isinstance(copy_args, dict):
        raise ValueError("copy_args must be an object")
    normalized_keys = {str(key).replace("-", "_") for key in copy_args}
    disallowed = normalized_keys & _DISALLOWED_COPY_ARGS
    if disallowed:
        raise ValueError(f"copy_args contains runner-owned keys: {sorted(disallowed)}")
    if "tta" in copy_args and copy_args.get("tta") is not True:
        raise ValueError("the frozen baseline requires TTA enabled")
    expected_tifxyz_voxel_size = manifest.get(
        "native_voxel_size_um", manifest.get("voxel_size_um")
    )
    try:
        configured_tifxyz_voxel_size = float(config.get("tifxyz_voxel_size_um"))
        expected_tifxyz_voxel_size = float(expected_tifxyz_voxel_size)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "config and manifest must define a numeric tifxyz voxel size"
        ) from exc
    if configured_tifxyz_voxel_size != expected_tifxyz_voxel_size:
        raise ValueError(
            "config tifxyz_voxel_size_um must match the native stored-coordinate "
            f"voxel size {expected_tifxyz_voxel_size:g}"
        )
    return phase, sorted(sources)


def _safe_prefix(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.")
    if not normalized:
        raise ValueError(f"unable to form safe output prefix from {value!r}")
    return normalized


def _relative_output(output_root: Path, value: str | Path) -> str:
    resolved = Path(value).resolve()
    try:
        relative = resolved.relative_to(output_root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"inference output escaped run directory: {resolved}") from exc
    return relative.as_posix()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _refuse_nonempty_output(output_root: Path) -> None:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"output directory must be absent or empty to prevent overwrite: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)


def _runtime_versions() -> dict[str, Any]:
    import torch

    def package_version(distribution: str) -> str | None:
        try:
            return metadata.version(distribution)
        except metadata.PackageNotFoundError:
            return None

    cuda_name = None
    if torch.cuda.is_available():
        cuda_name = torch.cuda.get_device_name(0)
    return {
        "executable": sys.executable,
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "zarr": package_version("zarr"),
        "numcodecs": package_version("numcodecs"),
        "fsspec": package_version("fsspec"),
        "s3fs": package_version("s3fs"),
        "volume_cartographer": package_version("volume-cartographer"),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_0": cuda_name,
        "platform": sys.platform,
    }


def run_experiment(
    config_path: Path, test_authorization_path: Path | None = None
) -> dict[str, Any]:
    config_path = config_path.resolve()
    config_dir = config_path.parent
    config = _load_json(config_path)
    manifest_path = _resolve_path(config_dir, config.get("manifest"))
    manifest = load_manifest(manifest_path)
    phase, sources = validate_config_structure(config, manifest)
    current_commit = clean_git_commit(Path(__file__).resolve().parents[4])
    expected_checkpoint_sha256 = _require_sha256(
        config.get("checkpoint_sha256"), "config checkpoint_sha256"
    )
    provenance_path = (
        Path(__file__).resolve().parents[4]
        / "docs"
        / "copy_cycle_checkpoint_provenance.json"
    )
    frozen_checkpoint_sha256 = _require_sha256(
        _load_json(provenance_path).get("sha256"),
        "frozen checkpoint provenance SHA-256",
    )
    if expected_checkpoint_sha256 != frozen_checkpoint_sha256:
        raise ValueError("config checkpoint_sha256 must match the frozen official checkpoint")

    selected_parameters = None
    authorization_sha256 = None
    if phase == "sealed_test":
        if test_authorization_path is None:
            raise ValueError("sealed_test requires --test-authorization")
        test_authorization_path = test_authorization_path.resolve()
        validation_manifest_path = (
            Path(__file__).resolve().parents[4]
            / "docs"
            / "copy_cycle_pherc0500p2_manifest.json"
        )
        selected_parameters = load_and_validate_test_authorization(
            test_authorization_path,
            current_commit,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
            expected_validation_manifest_sha256=sha256_file(
                validation_manifest_path
            ),
            expected_copy_args=dict(config.get("copy_args", {})),
        )
        authorization_sha256 = sha256_file(test_authorization_path)
    elif test_authorization_path is not None:
        raise ValueError("--test-authorization is only valid for sealed_test")

    data_root = _resolve_path(config_dir, config.get("data_root"))
    benchmark_receipt = prepare_benchmark(
        manifest,
        data_root,
        download_missing=False,
    )
    checkpoint_path = _resolve_path(config_dir, config.get("checkpoint_path"))
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if checkpoint_sha256 != expected_checkpoint_sha256:
        raise RuntimeError(
            f"checkpoint SHA-256 mismatch: expected {expected_checkpoint_sha256}, "
            f"found {checkpoint_sha256}"
        )
    output_root = _resolve_path(config_dir, config.get("output_dir"))

    from vesuvius.neural_tracing.inference.infer_rowcol_triplet_wraps import (
        _COPY_ARG_TO_CLI,
        _canonicalize_tta_settings,
        _copy_args_to_argv,
        _load_input_grid,
        _open_vc_volume_level,
        _resolve_crop_size,
        _run_single_iteration,
        parse_args as parse_copy_args,
        normalize_copy_args,
        resolve_tifxyz_params,
    )
    from vesuvius.neural_tracing.inference.displacement_helpers import load_model

    entries = {int(entry["wrap"]): entry for entry in manifest["wraps"]}
    first_entry = entries[sources[0]]
    first_input = data_root / str(
        first_entry.get("local_dir", f"wrap{sources[0]:02d}")
    )
    copy_args = dict(config.get("copy_args", {}))
    copy_args.update(
        {
            "tifxyz_path": str(first_input),
            "volume_path": str(config["volume_path"]),
            "checkpoint_path": str(checkpoint_path),
            "volume_scale": int(config["volume_scale"]),
            "tifxyz_voxel_size_um": float(config["tifxyz_voxel_size_um"]),
            "out_dir": str(output_root),
        }
    )
    copy_args = normalize_copy_args(copy_args)
    known_copy_args = set(_COPY_ARG_TO_CLI) | {
        "bbox_prune",
        "compile_model",
        "keep_previous_wrap",
        "save_original_copy",
        "tta",
        "verbose",
    }
    unknown_copy_args = set(copy_args) - known_copy_args
    if unknown_copy_args:
        raise ValueError(f"unknown copy_args keys: {sorted(unknown_copy_args)}")
    _refuse_nonempty_output(output_root)
    args = _canonicalize_tta_settings(
        parse_copy_args(_copy_args_to_argv(copy_args))
    )
    model_state = load_model(args)
    model_config = model_state["model_config"]
    crop_size = _resolve_crop_size(args, model_config)
    volume_arr, resolved_volume_level = _open_vc_volume_level(
        args.volume_path,
        volume_scale=args.volume_scale,
        cache_dir=args.volume_cache_dir,
        chunk_cache_gb=args.volume_chunk_cache_gb,
        retry_seconds=args.volume_cache_retry_seconds,
    )
    if int(resolved_volume_level) != int(args.volume_scale):
        raise RuntimeError(
            f"frozen volume scale {args.volume_scale} was unavailable; "
            f"backend resolved level {resolved_volume_level} instead"
        )
    retarget_factor = float(2 ** int(args.volume_scale))
    save_scale_factor = int(2 ** int(args.volume_scale))

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "completed": False,
        "started_utc": _utc_now(),
        "finished_utc": None,
        "phase": phase,
        "scroll_id": manifest["scroll_id"],
        "implementation_commit": current_commit,
        "config_sha256": sha256_file(config_path),
        "manifest_sha256": sha256_file(manifest_path),
        "checkpoint_filename": checkpoint_path.name,
        "checkpoint_sha256": checkpoint_sha256,
        "benchmark_file_count": int(benchmark_receipt["file_count"]),
        "benchmark_total_bytes": int(benchmark_receipt["total_bytes"]),
        "volume_path": str(args.volume_path),
        "volume_backend": str(volume_arr.backend_name),
        "volume_scale_requested": int(args.volume_scale),
        "volume_scale_resolved": int(resolved_volume_level),
        "crop_size": [int(value) for value in crop_size],
        "requested_copy_args": dict(config.get("copy_args", {})),
        "copy_args": copy_args,
        "effective_copy_args": dict(vars(args)),
        "runtime": _runtime_versions(),
        "test_authorization_sha256": authorization_sha256,
        "selected_parameters": selected_parameters,
        "sources": [],
    }
    receipt_path = output_root / "run_receipt.json"
    write_json_atomic(receipt_path, receipt)

    for wrap in sources:
        entry = entries[wrap]
        local_dir = str(entry.get("local_dir", f"wrap{wrap:02d}"))
        input_path = data_root / local_dir
        args.tifxyz_path = str(input_path)
        surface, input_grid, input_valid = _load_input_grid(
            str(input_path), retarget_factor=retarget_factor
        )
        tifxyz_step_size, tifxyz_voxel_size_um, stored_scale_rc = resolve_tifxyz_params(
            args,
            model_config,
            args.volume_scale,
            input_scale=surface.get_scale_tuple(),
        )
        source_prefix = _safe_prefix(f"source_{wrap:02d}_{local_dir}")
        source_output_dir = output_root / source_prefix
        source_output_dir.mkdir(parents=False, exist_ok=False)
        forward_result = _run_single_iteration(
            args=args,
            model_state=model_state,
            crop_size=crop_size,
            volume_arr=volume_arr,
            input_tifxyz_path=str(input_path),
            out_dir=str(source_output_dir),
            out_prefix=f"{source_prefix}_forward",
            retarget_factor=retarget_factor,
            tifxyz_step_size=tifxyz_step_size,
            tifxyz_voxel_size_um=tifxyz_voxel_size_um,
            stored_scale_rc=stored_scale_rc,
            save_scale_factor=save_scale_factor,
            iteration_index=1,
            iterations_requested=1,
            iterative_mode=False,
            iter_direction=None,
            keep_previous_wrap=True,
            preloaded_input=(surface, input_grid, input_valid),
        )
        forward_paths = {
            branch: str(forward_result["outputs"][branch])
            for branch in ("front", "back")
        }
        roundtrip_receipt: dict[str, Any] = {}
        for branch in ("front", "back"):
            roundtrip_result = _run_single_iteration(
                args=args,
                model_state=model_state,
                crop_size=crop_size,
                volume_arr=volume_arr,
                input_tifxyz_path=forward_paths[branch],
                out_dir=str(source_output_dir),
                out_prefix=f"{source_prefix}_roundtrip_from_{branch}",
                retarget_factor=retarget_factor,
                tifxyz_step_size=tifxyz_step_size,
                tifxyz_voxel_size_um=tifxyz_voxel_size_um,
                stored_scale_rc=stored_scale_rc,
                save_scale_factor=save_scale_factor,
                iteration_index=1,
                iterations_requested=1,
                iterative_mode=False,
                iter_direction=None,
                keep_previous_wrap=True,
                preloaded_input=None,
            )
            return_paths = {
                name: str(roundtrip_result["outputs"][name])
                for name in ("front", "back")
            }
            return_grids = {
                name: _load_input_grid(path, retarget_factor=retarget_factor)[1:]
                for name, path in return_paths.items()
            }
            selected_return, medians = choose_return_branch(
                input_grid,
                input_valid,
                return_grids,
            )
            roundtrip_receipt[branch] = {
                "outputs": {
                    name: _relative_output(output_root, path)
                    for name, path in return_paths.items()
                },
                "output_sha256": {
                    name: sha256_tifxyz(path) for name, path in return_paths.items()
                },
                "selected_return": selected_return,
                "cycle_median_by_return_branch": medians,
                "n_predicted_cells": {
                    "front": int(roundtrip_result["n_pred_front_cells"]),
                    "back": int(roundtrip_result["n_pred_back_cells"]),
                },
            }

        source_receipt = {
            "wrap": wrap,
            "local_dir": local_dir,
            "input_sha256": sha256_tifxyz(input_path),
            "forward": {
                branch: _relative_output(output_root, path)
                for branch, path in forward_paths.items()
            },
            "forward_sha256": {
                branch: sha256_tifxyz(path) for branch, path in forward_paths.items()
            },
            "roundtrip": roundtrip_receipt,
            "n_predicted_cells": {
                "front": int(forward_result["n_pred_front_cells"]),
                "back": int(forward_result["n_pred_back_cells"]),
            },
        }
        receipt["sources"].append(source_receipt)
        write_json_atomic(receipt_path, receipt)

    receipt["completed"] = True
    receipt["finished_utc"] = _utc_now()
    write_json_atomic(receipt_path, receipt)
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a frozen forward/round-trip copy-cycle benchmark phase."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--test-authorization", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    receipt = run_experiment(args.config, args.test_authorization)
    print(
        json.dumps(
            {
                "completed": receipt["completed"],
                "phase": receipt["phase"],
                "scroll_id": receipt["scroll_id"],
                "sources": len(receipt["sources"]),
            }
        )
    )


if __name__ == "__main__":
    main()
