"""Score preregistered copy-cycle runs without changing their frozen protocol."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

import numpy as np

from vesuvius.neural_tracing.evaluation.copy_cycle_io import (
    load_tifxyz_grid,
    sha256_file,
    sha256_tifxyz,
    write_json_atomic,
)
from vesuvius.neural_tracing.evaluation.copy_cycle_metrics import (
    ALPHA_GRID,
    BAD_CELL_DISTANCE,
    SHIFT_COLS,
    SHIFT_ROWS,
    TAU_GRID,
    DistanceIndex,
    ScoreResult,
    aggregate_scores,
    apply_cycle_guard,
    apply_wrong_sign_null,
    assign_baseline_branches,
    binary_ranking_metrics,
    central_dense_surface,
    choose_return_branch,
    compare_shifted_residual,
    score_prediction,
    select_validation_candidate,
)
from vesuvius.neural_tracing.evaluation.prepare_copy_cycle_benchmark import (
    load_manifest,
    prepare_benchmark,
)


@dataclass(frozen=True)
class DirectionContext:
    source: int
    target: int
    branch: str
    return_branch: str
    return_medians: dict[str, float]
    source_grid: np.ndarray
    source_valid: np.ndarray
    forward_grid: np.ndarray
    forward_valid: np.ndarray
    return_grid: np.ndarray
    return_valid: np.ndarray
    target_index: DistanceIndex
    wrong_index: DistanceIndex
    baseline: ScoreResult
    source_stay: ScoreResult


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("rt", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _phase_edges(manifest: dict[str, Any], phase: str) -> list[tuple[int, int]]:
    phase_key = {
        "development": "development_edges",
        "validation": "validation_edges",
        "sealed_test": "sealed_test_edges",
    }.get(str(phase))
    if phase_key is None:
        raise ValueError(f"unknown experiment phase: {phase!r}")
    source = manifest.get("splits", manifest)
    raw_edges = source.get(phase_key)
    if not isinstance(raw_edges, list) or not raw_edges:
        raise ValueError(f"manifest does not define non-empty {phase_key}")
    edges: list[tuple[int, int]] = []
    for edge in raw_edges:
        if not isinstance(edge, list) or len(edge) != 2:
            raise ValueError(f"invalid {phase_key} edge: {edge!r}")
        first, second = int(edge[0]), int(edge[1])
        if first == second:
            raise ValueError(f"self-edge is not allowed: {edge!r}")
        edges.append((first, second))
    return edges


def _resolve_receipt_path(output_root: Path, value: Any) -> Path:
    relative = PurePosixPath(str(value))
    if relative.is_absolute() or not relative.parts or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ValueError(f"receipt output path must be a safe relative POSIX path: {value!r}")
    resolved = output_root.joinpath(*relative.parts).resolve()
    try:
        resolved.relative_to(output_root.resolve())
    except ValueError as exc:
        raise ValueError(f"receipt output escapes its run directory: {value!r}") from exc
    return resolved


def _parameter_json(alpha: float, tau: float) -> dict[str, float | str]:
    return {
        "alpha": float(alpha),
        "tau": "infinity" if math.isinf(float(tau)) else float(tau),
    }


def _normalize_selection(selection: dict[str, object]) -> dict[str, object]:
    output = dict(selection)
    for key in ("selected", "best_grid_key"):
        value = output.get(key)
        if isinstance(value, dict) and "alpha" in value and "tau" in value:
            output[key] = _parameter_json(float(value["alpha"]), float(value["tau"]))
    return output


def _manifest_entries(manifest: dict[str, Any]) -> dict[int, dict[str, Any]]:
    entries = manifest.get("wraps")
    if not isinstance(entries, list) or not entries:
        raise ValueError("manifest wraps must be a non-empty list")
    output: dict[int, dict[str, Any]] = {}
    for entry in entries:
        wrap = int(entry["wrap"])
        if wrap in output:
            raise ValueError(f"duplicate manifest wrap: {wrap}")
        output[wrap] = entry
    return output


def _load_ground_truth(
    manifest: dict[str, Any], data_root: Path
) -> tuple[
    dict[int, tuple[np.ndarray, np.ndarray]],
    dict[int, np.ndarray],
    dict[int, DistanceIndex],
]:
    divisor = float(2 ** int(manifest.get("volume_scale", 0)))
    grids: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    clouds: dict[int, np.ndarray] = {}
    indexes: dict[int, DistanceIndex] = {}
    excluded = {int(value) for value in manifest.get("excluded_wraps", [])}
    for wrap, entry in sorted(_manifest_entries(manifest).items()):
        if wrap in excluded or entry.get("included_in_test") is False:
            continue
        local_dir = str(entry.get("local_dir", f"wrap{wrap:02d}"))
        grid, valid = load_tifxyz_grid(
            data_root / local_dir,
            coordinate_divisor=divisor,
        )
        cloud = central_dense_surface(grid, valid)
        grids[wrap] = (grid, valid)
        clouds[wrap] = cloud
        indexes[wrap] = DistanceIndex.from_clouds((cloud,))
    return grids, clouds, indexes


def _wrong_index(
    target: int,
    indexes: dict[int, DistanceIndex],
) -> DistanceIndex:
    trees = tuple(
        index.trees[0]
        for wrap, index in sorted(indexes.items())
        if int(wrap) != int(target)
    )
    if not trees:
        raise ValueError("wrong-sheet index requires at least one non-target surface")
    return DistanceIndex(trees=trees)


def _source_targets(edges: Sequence[tuple[int, int]]) -> dict[int, list[int]]:
    output: dict[int, set[int]] = {}
    for first, second in edges:
        output.setdefault(first, set()).add(second)
        output.setdefault(second, set()).add(first)
    return {source: sorted(targets) for source, targets in output.items()}


def _validate_receipt(
    receipt: dict[str, Any],
    manifest: dict[str, Any],
    edges: Sequence[tuple[int, int]],
) -> dict[int, dict[str, Any]]:
    if int(receipt.get("schema_version", -1)) != 1:
        raise ValueError("run receipt schema_version must be 1")
    if receipt.get("completed") is not True:
        raise ValueError("run receipt is partial; completed must be true before scoring")
    if str(receipt.get("scroll_id")) != str(manifest.get("scroll_id")):
        raise ValueError("run receipt scroll_id does not match the benchmark manifest")
    phase = str(receipt.get("phase"))
    source_records = receipt.get("sources")
    if not isinstance(source_records, list) or not source_records:
        raise ValueError("run receipt sources must be a non-empty list")
    records: dict[int, dict[str, Any]] = {}
    for record in source_records:
        wrap = int(record["wrap"])
        if wrap in records:
            raise ValueError(f"duplicate source wrap in receipt: {wrap}")
        records[wrap] = record

    allowed_sources = set(_source_targets(edges))
    unknown = set(records) - allowed_sources
    if unknown:
        raise ValueError(f"receipt contains sources outside phase {phase}: {sorted(unknown)}")
    if phase in {"validation", "sealed_test"} and set(records) != allowed_sources:
        raise ValueError(
            f"{phase} receipt must contain every source {sorted(allowed_sources)}, "
            f"got {sorted(records)}"
        )
    return records


def build_direction_contexts(
    receipt_path: Path,
    manifest_path: Path,
    data_root: Path,
) -> tuple[list[DirectionContext], dict[str, Any], dict[str, Any]]:
    receipt = _load_json(receipt_path)
    manifest = load_manifest(manifest_path)
    prepare_benchmark(manifest, data_root, download_missing=False)
    if str(receipt.get("manifest_sha256")) != sha256_file(manifest_path):
        raise ValueError("run receipt manifest SHA-256 does not match the supplied manifest")
    phase = str(receipt.get("phase"))
    edges = _phase_edges(manifest, phase)
    records = _validate_receipt(receipt, manifest, edges)
    grids, _, indexes = _load_ground_truth(manifest, data_root)
    targets_by_source = _source_targets(edges)
    missing_scored_wraps = set(targets_by_source) - set(indexes)
    if missing_scored_wraps:
        raise ValueError(
            "phase edges reference excluded or unavailable scored wraps: "
            f"{sorted(missing_scored_wraps)}"
        )
    output_root = receipt_path.resolve().parent
    output_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def load_output(relative_path: str) -> tuple[np.ndarray, np.ndarray]:
        if relative_path not in output_cache:
            divisor = float(2 ** int(manifest.get("volume_scale", 0)))
            output_cache[relative_path] = load_tifxyz_grid(
                _resolve_receipt_path(output_root, relative_path),
                coordinate_divisor=divisor,
            )
        return output_cache[relative_path]

    contexts: list[DirectionContext] = []
    wrong_indexes = {target: _wrong_index(target, indexes) for target in indexes}
    for source, record in sorted(records.items()):
        source_grid, source_valid = grids[source]
        forward_paths = record.get("forward")
        if not isinstance(forward_paths, dict) or set(forward_paths) != {"front", "back"}:
            raise ValueError(f"source {source} forward outputs must contain front and back")
        forward_hashes = record.get("forward_sha256")
        if not isinstance(forward_hashes, dict) or set(forward_hashes) != {"front", "back"}:
            raise ValueError(f"source {source} forward hashes must contain front and back")
        forward = {}
        for branch, relative_path in forward_paths.items():
            resolved_path = _resolve_receipt_path(output_root, relative_path)
            actual_sha256 = sha256_tifxyz(resolved_path)
            if actual_sha256 != str(forward_hashes[branch]):
                raise ValueError(f"source {source} forward {branch} SHA-256 mismatch")
            forward[branch] = load_output(str(relative_path))
        input_entry = _manifest_entries(manifest)[source]
        input_local_dir = str(input_entry.get("local_dir", f"wrap{source:02d}"))
        if sha256_tifxyz(data_root / input_local_dir) != str(record.get("input_sha256")):
            raise ValueError(f"source {source} input SHA-256 mismatch")
        targets = targets_by_source[source]
        branch_target_scores: dict[tuple[str, int], ScoreResult] = {}
        for branch, (forward_grid, forward_valid) in forward.items():
            for target in targets:
                branch_target_scores[(branch, target)] = score_prediction(
                    source_grid,
                    source_valid,
                    indexes[target],
                    wrong_indexes[target],
                    forward_grid,
                    forward_valid,
                )
        assignment = assign_baseline_branches(
            tuple(forward), tuple(targets), branch_target_scores
        )

        roundtrip = record.get("roundtrip")
        if not isinstance(roundtrip, dict) or set(roundtrip) != {"front", "back"}:
            raise ValueError(
                f"source {source} roundtrip outputs must contain front and back"
            )
        return_choices: dict[
            str, tuple[str, dict[str, float], tuple[np.ndarray, np.ndarray]]
        ] = {}
        for branch in ("front", "back"):
            roundtrip_entry = roundtrip.get(branch)
            if not isinstance(roundtrip_entry, dict):
                raise ValueError(f"source {source} is missing roundtrip for {branch}")
            return_paths = roundtrip_entry.get("outputs")
            if not isinstance(return_paths, dict) or set(return_paths) != {"front", "back"}:
                raise ValueError(
                    f"source {source} roundtrip {branch} must contain front and back"
                )
            return_hashes = roundtrip_entry.get("output_sha256")
            if not isinstance(return_hashes, dict) or set(return_hashes) != {"front", "back"}:
                raise ValueError(
                    f"source {source} roundtrip {branch} hashes must contain front and back"
                )
            return_outputs = {}
            for name, path in return_paths.items():
                resolved_path = _resolve_receipt_path(output_root, path)
                if sha256_tifxyz(resolved_path) != str(return_hashes[name]):
                    raise ValueError(
                        f"source {source} roundtrip {branch}/{name} SHA-256 mismatch"
                    )
                return_outputs[name] = load_output(str(path))
            return_branch, return_medians = choose_return_branch(
                source_grid, source_valid, return_outputs
            )
            recorded_return = roundtrip_entry.get("selected_return")
            if recorded_return is not None and str(recorded_return) != return_branch:
                raise ValueError(
                    f"source {source} roundtrip {branch} selected-return mismatch: "
                    f"receipt={recorded_return}, recomputed={return_branch}"
                )
            return_choices[branch] = (
                return_branch,
                return_medians,
                return_outputs[return_branch],
            )

        for target in targets:
            branch = assignment[target]
            return_branch, return_medians, selected_return = return_choices[branch]
            forward_grid, forward_valid = forward[branch]
            return_grid, return_valid = selected_return
            contexts.append(
                DirectionContext(
                    source=source,
                    target=target,
                    branch=branch,
                    return_branch=return_branch,
                    return_medians=return_medians,
                    source_grid=source_grid,
                    source_valid=source_valid,
                    forward_grid=forward_grid,
                    forward_valid=forward_valid,
                    return_grid=return_grid,
                    return_valid=return_valid,
                    target_index=indexes[target],
                    wrong_index=wrong_indexes[target],
                    baseline=branch_target_scores[(branch, target)],
                    source_stay=score_prediction(
                        source_grid,
                        source_valid,
                        indexes[target],
                        wrong_indexes[target],
                        source_grid,
                        source_valid,
                    ),
                )
            )
    contexts.sort(key=lambda context: (context.source, context.target))
    return contexts, receipt, manifest


def _candidate_score(
    context: DirectionContext,
    alpha: float,
    tau: float,
    *,
    wrong_sign: bool = False,
) -> tuple[ScoreResult, Any]:
    apply = apply_wrong_sign_null if wrong_sign else apply_cycle_guard
    guard = apply(
        context.source_grid,
        context.source_valid,
        context.forward_grid,
        context.forward_valid,
        context.return_grid,
        context.return_valid,
        alpha=alpha,
        tau=tau,
    )
    score = score_prediction(
        context.source_grid,
        context.source_valid,
        context.target_index,
        context.wrong_index,
        guard.grid,
        guard.valid,
    )
    return score, guard


def _aggregate_detector(
    contexts: Sequence[DirectionContext], guards: Sequence[Any]
) -> dict[str, dict[str, float | int]]:
    real_values: list[np.ndarray] = []
    shifted_values: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for context, guard in zip(contexts, guards):
        residual = np.asarray(guard.residual, dtype=np.float32)
        residual_valid = np.asarray(guard.residual_valid, dtype=bool) & np.isfinite(residual)
        shifted = np.roll(residual, shift=(SHIFT_ROWS, SHIFT_COLS), axis=(0, 1))
        shifted_valid = np.roll(
            residual_valid, shift=(SHIFT_ROWS, SHIFT_COLS), axis=(0, 1)
        )
        common = context.baseline.eligible & residual_valid & shifted_valid
        bad = context.baseline.prediction_valid & (
            (context.baseline.target_distance > BAD_CELL_DISTANCE)
            | context.baseline.sheet_switch
        )
        real_values.append(residual[common])
        shifted_values.append(shifted[common])
        labels.append(bad[common])
    return {
        "real": binary_ranking_metrics(np.concatenate(real_values), np.concatenate(labels)),
        "shifted": binary_ranking_metrics(
            np.concatenate(shifted_values), np.concatenate(labels)
        ),
    }


def evaluate_primary_gate(
    baseline: Sequence[ScoreResult],
    candidate: Sequence[ScoreResult],
    source_stay: Sequence[ScoreResult],
    wrong_sign: Sequence[ScoreResult],
    detector: dict[str, dict[str, float | int]],
) -> dict[str, Any]:
    if not baseline or not (
        len(baseline) == len(candidate) == len(source_stay) == len(wrong_sign)
    ):
        raise ValueError("all primary-gate arms must contain the same non-zero directions")
    baseline_aggregate = aggregate_scores(baseline)
    candidate_aggregate = aggregate_scores(candidate)
    source_stay_aggregate = aggregate_scores(source_stay)
    wrong_sign_aggregate = aggregate_scores(wrong_sign)
    baseline_penalty = float(baseline_aggregate["penalized_target_distance_mean"])
    candidate_penalty = float(candidate_aggregate["penalized_target_distance_mean"])
    relative_improvement = (
        (baseline_penalty - candidate_penalty) / baseline_penalty
        if baseline_penalty > 0.0
        else -math.inf
    )
    improved_directions = sum(
        float(candidate_result.metrics["penalized_target_distance_mean"])
        < float(baseline_result.metrics["penalized_target_distance_mean"])
        for baseline_result, candidate_result in zip(baseline, candidate)
    )
    required_improved_directions = int(math.ceil(0.75 * len(baseline)))
    baseline_switch = float(baseline_aggregate["sheet_switch_rate_all_eligible"])
    candidate_switch = float(candidate_aggregate["sheet_switch_rate_all_eligible"])
    switch_gate = (
        candidate_switch <= 0.75 * baseline_switch
        if baseline_switch >= 0.005
        else candidate_switch <= baseline_switch + 0.001
    )
    real_detector = detector["real"]
    shifted_detector = detector["shifted"]
    detector_gate = all(
        math.isfinite(float(real_detector[key]))
        and math.isfinite(float(shifted_detector[key]))
        and float(real_detector[key]) > float(shifted_detector[key])
        for key in ("auroc", "average_precision")
    )
    conditions = {
        "coverage_each_direction": all(
            float(candidate_result.metrics["coverage"])
            >= 0.9 * float(baseline_result.metrics["coverage"])
            for baseline_result, candidate_result in zip(baseline, candidate)
        ),
        "coverage_aggregate": float(candidate_aggregate["coverage"])
        >= 0.9 * float(baseline_aggregate["coverage"]),
        "penalized_distance_relative_improvement_at_least_10pct": relative_improvement
        >= 0.10,
        "improved_directions_at_least_three_quarters": improved_directions
        >= required_improved_directions,
        "sheet_switch_gate": switch_gate,
        "p95_noninferiority": float(candidate_aggregate["target_distance_p95_valid"])
        <= 1.05 * float(baseline_aggregate["target_distance_p95_valid"]),
        "cycle_detector_beats_shifted_null": detector_gate,
        "candidate_beats_source_stay_null": candidate_penalty
        < float(source_stay_aggregate["penalized_target_distance_mean"]),
        "candidate_beats_wrong_sign_null": candidate_penalty
        < float(wrong_sign_aggregate["penalized_target_distance_mean"]),
    }
    return {
        "passed": all(conditions.values()),
        "conditions": conditions,
        "relative_penalized_distance_improvement": relative_improvement,
        "improved_directions": improved_directions,
        "required_improved_directions": required_improved_directions,
        "baseline": baseline_aggregate,
        "candidate": candidate_aggregate,
        "source_stay": source_stay_aggregate,
        "wrong_sign": wrong_sign_aggregate,
        "detector": detector,
    }


def score_grid(contexts: Sequence[DirectionContext]) -> dict[str, Any]:
    baseline = [context.baseline for context in contexts]
    candidate_map: dict[tuple[float, float], list[ScoreResult]] = {}
    grid_records: list[dict[str, Any]] = []
    for alpha in ALPHA_GRID:
        for tau in TAU_GRID:
            scores = [_candidate_score(context, alpha, tau)[0] for context in contexts]
            candidate_map[(alpha, tau)] = scores
            grid_records.append(
                {
                    **_parameter_json(alpha, tau),
                    "aggregate": aggregate_scores(scores),
                    "directions": [score.metrics for score in scores],
                }
            )
    selection = _normalize_selection(select_validation_candidate(baseline, candidate_map))
    return {
        "baseline_aggregate": aggregate_scores(baseline),
        "grid": grid_records,
        "selection": selection,
    }


def score_fixed(
    contexts: Sequence[DirectionContext], alpha: float, tau: float
) -> dict[str, Any]:
    baseline = [context.baseline for context in contexts]
    source_stay = [context.source_stay for context in contexts]
    candidate_pairs = [_candidate_score(context, alpha, tau) for context in contexts]
    wrong_sign_pairs = [
        _candidate_score(context, alpha, tau, wrong_sign=True) for context in contexts
    ]
    candidate = [item[0] for item in candidate_pairs]
    guards = [item[1] for item in candidate_pairs]
    wrong_sign = [item[0] for item in wrong_sign_pairs]
    detector = _aggregate_detector(contexts, guards)
    directions = []
    for context, candidate_score, wrong_sign_score, guard in zip(
        contexts, candidate, wrong_sign, guards
    ):
        directions.append(
            {
                "source": context.source,
                "target": context.target,
                "branch": context.branch,
                "return_branch": context.return_branch,
                "return_medians": context.return_medians,
                "baseline": context.baseline.metrics,
                "candidate": candidate_score.metrics,
                "source_stay": context.source_stay.metrics,
                "wrong_sign": wrong_sign_score.metrics,
                "detector": compare_shifted_residual(
                    guard.residual, guard.residual_valid, context.baseline
                ),
            }
        )
    return {
        "parameters": _parameter_json(alpha, tau),
        "directions": directions,
        "primary_gate": evaluate_primary_gate(
            baseline, candidate, source_stay, wrong_sign, detector
        ),
    }


def parse_tau(value: str) -> float:
    normalized = str(value).strip().lower()
    if normalized in {"inf", "infinity"}:
        return math.inf
    return float(normalized)


def validate_scoring_request(
    receipt: dict[str, Any], mode: str, alpha: float | None, tau: float | None
) -> None:
    phase = str(receipt.get("phase"))
    if phase == "validation" and mode != "grid":
        raise ValueError("validation receipts must be scored in frozen grid mode")
    if phase != "sealed_test":
        return
    if mode != "fixed":
        raise ValueError("sealed-test receipts may only be scored in fixed mode")
    selected = receipt.get("selected_parameters")
    if not isinstance(selected, dict):
        raise ValueError("sealed-test receipt is missing validation-selected parameters")
    selected_alpha = float(selected.get("alpha"))
    selected_tau = parse_tau(str(selected.get("tau")))
    if selected_alpha not in ALPHA_GRID or selected_tau not in TAU_GRID:
        raise ValueError("sealed-test receipt parameters are outside the frozen grid")
    if alpha is None or tau is None or (
        float(alpha), float(tau)
    ) != (selected_alpha, selected_tau):
        raise ValueError(
            "sealed-test scoring parameters must exactly match the run receipt"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score a frozen copy-cycle run receipt.")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("grid", "fixed"), required=True)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--tau", type=parse_tau, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.mode == "fixed":
        if args.alpha is None or args.tau is None:
            parser.error("--mode fixed requires --alpha and --tau")
        if float(args.alpha) not in ALPHA_GRID:
            parser.error(f"--alpha must be one of {ALPHA_GRID}")
        if float(args.tau) not in TAU_GRID:
            parser.error(f"--tau must be one of {TAU_GRID}")
    elif args.alpha is not None or args.tau is not None:
        parser.error("--alpha/--tau are only valid with --mode fixed")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite score output: {args.output}")
    preflight_receipt = _load_json(args.receipt)
    validate_scoring_request(preflight_receipt, args.mode, args.alpha, args.tau)
    contexts, receipt, manifest = build_direction_contexts(
        args.receipt, args.manifest, args.data_root
    )
    direction_identity = [
        {
            "source": context.source,
            "target": context.target,
            "branch": context.branch,
            "return_branch": context.return_branch,
        }
        for context in contexts
    ]
    if args.mode == "grid":
        result = score_grid(contexts)
    else:
        result = score_fixed(contexts, float(args.alpha), float(args.tau))
    payload = {
        "schema_version": 1,
        "mode": args.mode,
        "phase": receipt["phase"],
        "scroll_id": manifest["scroll_id"],
        "implementation_commit": receipt.get("implementation_commit"),
        "checkpoint_sha256": receipt.get("checkpoint_sha256"),
        "config_sha256": receipt.get("config_sha256"),
        "receipt_sha256": sha256_file(args.receipt),
        "manifest_sha256": sha256_file(args.manifest),
        "direction_identity": direction_identity,
        "result": result,
    }
    write_json_atomic(args.output, payload)
    print(json.dumps({"output": str(args.output.resolve()), "directions": len(contexts)}))


if __name__ == "__main__":
    main()
