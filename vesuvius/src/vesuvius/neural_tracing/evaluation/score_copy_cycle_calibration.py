"""Score the frozen learned copy calibration and its mandatory controls."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

from vesuvius.neural_tracing.evaluation.copy_cycle_calibration import (
    COPY_TRAINING_VOXEL_SIZE_UM,
    CROSS_SCROLL_INVARIANT_KEYS,
    FROZEN_CORRECTION_CAP,
    FROZEN_RIDGE,
    CalibrationBundle,
    apply_local_linear_model,
    apply_scalar_displacement_correction,
    inference_receipt_signature,
    training_inference_signature,
)
from vesuvius.neural_tracing.evaluation.copy_cycle_io import (
    clean_git_commit,
    sha256_file,
    write_json_atomic,
)
from vesuvius.neural_tracing.evaluation.copy_cycle_metrics import (
    ScoreResult,
    aggregate_scores,
    score_prediction,
)
from vesuvius.neural_tracing.evaluation.score_copy_cycle_experiment import (
    DirectionContext,
    build_direction_contexts,
)


ARM_NAMES = (
    "baseline",
    "combined",
    "displacement_only",
    "cycle_only",
    "fitted_scalar",
    "physical_scalar",
)
EXPECTED_DIRECTIONS = {
    "development_holdout": {(3, 2), (3, 4), (4, 3)},
    "validation": {(5, 6), (6, 5), (6, 7), (7, 6)},
}
EXPECTED_PHASE = {
    "development_holdout": "development",
    "validation": "validation",
    "sealed_test": "sealed_test",
}
EXPECTED_SCROLL_ID = {
    "development_holdout": "PHerc0500P2",
    "validation": "PHerc0500P2",
    "sealed_test": "PHerc0343P",
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("rt", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _validate_frozen_bundle(bundle: CalibrationBundle) -> None:
    for model in (bundle.combined, bundle.displacement_only, bundle.cycle_only):
        if not math.isclose(model.ridge, FROZEN_RIDGE, rel_tol=0.0, abs_tol=0.0):
            raise ValueError("model ridge differs from the frozen protocol")
        if not math.isclose(
            model.correction_cap,
            FROZEN_CORRECTION_CAP,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError("model correction cap differs from the frozen protocol")


def _effective_voxel_size_um(manifest: dict[str, Any]) -> float:
    if "effective_model_voxel_size_um" in manifest:
        value = float(manifest["effective_model_voxel_size_um"])
    elif "voxel_size_um" in manifest:
        value = float(manifest["voxel_size_um"]) * float(
            2 ** int(manifest.get("volume_scale", 0))
        )
    elif "native_voxel_size_um" in manifest:
        value = float(manifest["native_voxel_size_um"]) * float(
            2 ** int(manifest.get("volume_scale", 0))
        )
    else:
        raise ValueError("manifest does not declare an effective voxel size")
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("manifest effective voxel size must be finite and positive")
    return value


def load_scoring_contexts(
    receipt_paths: Sequence[Path],
    manifest_path: Path,
    data_root: Path,
    *,
    stage: str,
) -> tuple[list[DirectionContext], list[dict[str, str]], list[dict[str, Any]], dict[str, Any]]:
    if stage not in EXPECTED_PHASE:
        raise ValueError(f"unknown calibration scoring stage: {stage!r}")
    if not receipt_paths:
        raise ValueError("at least one scoring receipt is required")
    manifest = _load_json(manifest_path)
    contexts: list[DirectionContext] = []
    records: list[dict[str, str]] = []
    receipts: list[dict[str, Any]] = []
    common_signature: dict[str, Any] | None = None
    for path in receipt_paths:
        receipt = _load_json(path)
        if receipt.get("completed") is not True:
            raise ValueError(f"scoring receipt is incomplete: {path}")
        if receipt.get("phase") != EXPECTED_PHASE[stage]:
            raise ValueError(
                f"{stage} requires a {EXPECTED_PHASE[stage]} receipt: {path}"
            )
        signature = inference_receipt_signature(receipt)
        if common_signature is None:
            common_signature = signature
        elif signature != common_signature:
            raise ValueError("scoring receipts do not share inference provenance")
        loaded, _, _ = build_direction_contexts(path, manifest_path, data_root)
        contexts.extend(loaded)
        receipts.append(receipt)
        records.append({"path": str(path.resolve()), "sha256": sha256_file(path)})

    directions = [(item.source, item.target) for item in contexts]
    if len(set(directions)) != len(directions):
        raise ValueError("scoring receipts contain duplicate directed tasks")
    if stage in EXPECTED_DIRECTIONS and set(directions) != EXPECTED_DIRECTIONS[stage]:
        raise ValueError(
            f"{stage} requires directions {sorted(EXPECTED_DIRECTIONS[stage])}, "
            f"got {sorted(directions)}"
        )
    if stage == "sealed_test" and len(directions) != 12:
        raise ValueError(f"sealed test requires 12 directions, got {len(directions)}")
    assert common_signature is not None
    if common_signature["scroll_id"] != EXPECTED_SCROLL_ID[stage]:
        raise ValueError(
            f"{stage} requires scroll {EXPECTED_SCROLL_ID[stage]}, "
            f"got {common_signature['scroll_id']}"
        )
    if stage == "sealed_test":
        for receipt in receipts:
            authorization_sha = str(receipt.get("test_authorization_sha256", ""))
            if len(authorization_sha) != 64 or any(
                character not in "0123456789abcdef" for character in authorization_sha
            ):
                raise ValueError("sealed-test receipt lacks a valid authorization SHA-256")
    contexts.sort(key=lambda item: (item.source, item.target))
    return contexts, records, receipts, manifest


def _score_grid(context: DirectionContext, grid, valid) -> ScoreResult:
    return score_prediction(
        context.source_grid,
        context.source_valid,
        context.target_index,
        context.wrong_index,
        grid,
        valid,
    )


def score_context(
    context: DirectionContext,
    bundle: CalibrationBundle,
    *,
    effective_voxel_size_um: float,
) -> tuple[dict[str, ScoreResult], dict[str, int]]:
    combined_grid, combined_valid, combined_applied = apply_local_linear_model(
        bundle.combined,
        context.source_grid,
        context.source_valid,
        context.forward_grid,
        context.forward_valid,
        context.return_grid,
        context.return_valid,
    )
    displacement_grid, displacement_valid, displacement_applied = (
        apply_local_linear_model(
            bundle.displacement_only,
            context.source_grid,
            context.source_valid,
            context.forward_grid,
            context.forward_valid,
        )
    )
    cycle_grid, cycle_valid, cycle_applied = apply_local_linear_model(
        bundle.cycle_only,
        context.source_grid,
        context.source_valid,
        context.forward_grid,
        context.forward_valid,
        context.return_grid,
        context.return_valid,
    )
    fitted_grid, fitted_valid, fitted_applied = apply_scalar_displacement_correction(
        context.source_grid,
        context.source_valid,
        context.forward_grid,
        context.forward_valid,
        beta=bundle.fitted_scalar_beta,
        correction_cap=FROZEN_CORRECTION_CAP,
    )
    physical_scale = COPY_TRAINING_VOXEL_SIZE_UM / float(effective_voxel_size_um)
    physical_grid, physical_valid, physical_applied = (
        apply_scalar_displacement_correction(
            context.source_grid,
            context.source_valid,
            context.forward_grid,
            context.forward_valid,
            beta=physical_scale - 1.0,
            correction_cap=None,
        )
    )
    scores = {
        "baseline": context.baseline,
        "combined": _score_grid(context, combined_grid, combined_valid),
        "displacement_only": _score_grid(
            context, displacement_grid, displacement_valid
        ),
        "cycle_only": _score_grid(context, cycle_grid, cycle_valid),
        "fitted_scalar": _score_grid(context, fitted_grid, fitted_valid),
        "physical_scalar": _score_grid(context, physical_grid, physical_valid),
    }
    application_counts = {
        "combined": int(combined_applied.sum()),
        "displacement_only": int(displacement_applied.sum()),
        "cycle_only": int(cycle_applied.sum()),
        "fitted_scalar": int(fitted_applied.sum()),
        "physical_scalar": int(physical_applied.sum()),
    }
    return scores, application_counts


def _switch_condition(baseline: float, candidate: float) -> bool:
    if baseline >= 0.005:
        return candidate <= 0.75 * baseline
    return candidate <= baseline + 0.001


def evaluate_calibration_gate(
    stage: str, results: dict[str, list[ScoreResult]]
) -> dict[str, Any]:
    if set(results) != set(ARM_NAMES):
        raise ValueError("calibration gate requires every frozen arm")
    count = len(results["baseline"])
    if count == 0 or any(len(values) != count for values in results.values()):
        raise ValueError("calibration arms must contain the same non-zero task count")

    aggregate = {name: aggregate_scores(values) for name, values in results.items()}
    baseline_penalty = float(aggregate["baseline"]["penalized_target_distance_mean"])
    candidate_penalty = float(aggregate["combined"]["penalized_target_distance_mean"])
    if baseline_penalty <= 0.0:
        raise ValueError("baseline penalty must be positive for a relative gate")
    direction_improvements = [
        float(candidate.metrics["penalized_target_distance_mean"])
        < float(baseline.metrics["penalized_target_distance_mean"])
        for baseline, candidate in zip(results["baseline"], results["combined"])
    ]
    direction_noninferiority = [
        float(candidate.metrics["penalized_target_distance_mean"])
        <= 1.02 * float(baseline.metrics["penalized_target_distance_mean"])
        for baseline, candidate in zip(results["baseline"], results["combined"])
    ]
    coverage_equal = all(
        int(candidate.metrics["valid_prediction_cells"])
        == int(baseline.metrics["valid_prediction_cells"])
        for baseline, candidate in zip(results["baseline"], results["combined"])
    )
    control_names = ("displacement_only", "fitted_scalar", "physical_scalar")
    control_penalties = {
        name: float(aggregate[name]["penalized_target_distance_mean"])
        for name in control_names
    }
    best_control_penalty = min(control_penalties.values())
    required_improvements = {
        "development_holdout": 2,
        "validation": 3,
        "sealed_test": 9,
    }[stage]
    conditions = {
        "aggregate_penalty_improves_at_least_10pct": (
            candidate_penalty <= 0.9 * baseline_penalty
        ),
        "required_directions_improve": (
            sum(direction_improvements) >= required_improvements
        ),
        "beats_each_displacement_control": all(
            candidate_penalty < value for value in control_penalties.values()
        ),
        "incremental_gain_over_best_control_at_least_1pct_baseline": (
            best_control_penalty - candidate_penalty >= 0.01 * baseline_penalty
        ),
        "coverage_unchanged_each_direction": coverage_equal,
        "p95_noninferiority": (
            float(aggregate["combined"]["target_distance_p95_valid"])
            <= 1.05 * float(aggregate["baseline"]["target_distance_p95_valid"])
        ),
        "sheet_switch_gate": _switch_condition(
            float(aggregate["baseline"]["sheet_switch_rate_all_eligible"]),
            float(aggregate["combined"]["sheet_switch_rate_all_eligible"]),
        ),
    }
    if stage == "development_holdout":
        conditions["no_direction_worsens_over_2pct"] = all(
            direction_noninferiority
        )
    return {
        "passed": all(conditions.values()),
        "conditions": conditions,
        "improved_directions": sum(direction_improvements),
        "required_improved_directions": required_improvements,
        "direction_improvements": direction_improvements,
        "direction_noninferiority": direction_noninferiority,
        "baseline_penalty": baseline_penalty,
        "candidate_penalty": candidate_penalty,
        "relative_improvement": (baseline_penalty - candidate_penalty)
        / baseline_penalty,
        "control_penalties": control_penalties,
        "best_control_penalty": best_control_penalty,
        "incremental_gain_over_best_control_fraction_of_baseline": (
            best_control_penalty - candidate_penalty
        )
        / baseline_penalty,
    }


def validate_scoring_provenance(
    bundle: CalibrationBundle,
    receipts: Sequence[dict[str, Any]],
    *,
    stage: str,
    implementation_commit: str,
    model_sha256: str | None = None,
) -> None:
    _validate_frozen_bundle(bundle)
    if bundle.implementation_commit != implementation_commit:
        raise ValueError("loaded model was produced by a different implementation commit")
    expected_training_stage = (
        "holdout" if stage == "development_holdout" else "final"
    )
    if bundle.training.get("stage") != expected_training_stage:
        raise ValueError(
            f"{stage} scoring requires a {expected_training_stage} training model"
        )
    if stage == "sealed_test":
        normalized_model_sha = str(model_sha256 or "")
        if len(normalized_model_sha) != 64 or any(
            character not in "0123456789abcdef"
            for character in normalized_model_sha
        ):
            raise ValueError("sealed-test scoring requires the calibration model SHA-256")
        expected_selection = {
            "method": "copy_cycle_local_linear_v1",
            "model_sha256": normalized_model_sha,
        }
        for receipt in receipts:
            if receipt.get("selected_parameters") != expected_selection:
                raise ValueError(
                    "sealed-test receipt is not bound to the supplied calibration model"
                )
    training_signature = training_inference_signature(bundle.training)
    for receipt in receipts:
        scoring_signature = inference_receipt_signature(receipt)
        compared_keys = (
            CROSS_SCROLL_INVARIANT_KEYS
            if stage == "sealed_test"
            else tuple(training_signature)
        )
        for key in compared_keys:
            if scoring_signature[key] != training_signature[key]:
                raise ValueError(
                    f"scoring inference provenance differs from training: {key}"
                )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score the frozen local copy-cycle calibration and controls."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, action="append", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=tuple(EXPECTED_PHASE), required=True
    )
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite calibration score: {args.output}")
    current_commit = clean_git_commit(Path(__file__).resolve().parents[4])
    if args.implementation_commit != current_commit:
        raise ValueError(
            "--implementation-commit must equal the clean checked-out commit"
        )
    model_sha256 = sha256_file(args.model)
    bundle = CalibrationBundle.from_json(_load_json(args.model))
    contexts, receipt_records, receipts, manifest = load_scoring_contexts(
        args.receipt,
        args.manifest,
        args.data_root,
        stage=args.stage,
    )
    validate_scoring_provenance(
        bundle,
        receipts,
        stage=args.stage,
        implementation_commit=args.implementation_commit,
        model_sha256=model_sha256,
    )
    effective_voxel_size_um = _effective_voxel_size_um(manifest)
    by_arm: dict[str, list[ScoreResult]] = {name: [] for name in ARM_NAMES}
    directions = []
    for context in contexts:
        scores, application_counts = score_context(
            context,
            bundle,
            effective_voxel_size_um=effective_voxel_size_um,
        )
        for name, score in scores.items():
            by_arm[name].append(score)
        directions.append(
            {
                "source": context.source,
                "target": context.target,
                "branch": context.branch,
                "return_branch": context.return_branch,
                "application_cells": application_counts,
                "arms": {name: score.metrics for name, score in scores.items()},
            }
        )
    aggregate = {name: aggregate_scores(values) for name, values in by_arm.items()}
    gate = evaluate_calibration_gate(args.stage, by_arm)
    payload = {
        "schema_version": 1,
        "method": "copy_cycle_local_linear_v1_score",
        "stage": args.stage,
        "implementation_commit": args.implementation_commit,
        "model_path": str(args.model.resolve()),
        "model_sha256": model_sha256,
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "receipts": receipt_records,
        "scroll_id": manifest.get("scroll_id"),
        "effective_voxel_size_um": effective_voxel_size_um,
        "physical_scalar": COPY_TRAINING_VOXEL_SIZE_UM
        / effective_voxel_size_um,
        "directions": directions,
        "aggregate": aggregate,
        "gate": gate,
        "test_authorization_sha256": (
            receipts[0].get("test_authorization_sha256")
            if args.stage == "sealed_test"
            else None
        ),
    }
    write_json_atomic(args.output, payload)


if __name__ == "__main__":
    main()
