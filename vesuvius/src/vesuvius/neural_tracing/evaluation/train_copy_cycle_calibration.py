"""Fit the frozen cycle-conditioned calibration on development receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from vesuvius.neural_tracing.evaluation.copy_cycle_calibration import (
    concatenate_calibration_rows,
    extract_calibration_rows,
    fit_calibration_bundle,
    inference_receipt_signature,
)
from vesuvius.neural_tracing.evaluation.copy_cycle_io import (
    clean_git_commit,
    sha256_file,
    write_json_atomic,
)
from vesuvius.neural_tracing.evaluation.score_copy_cycle_experiment import (
    DirectionContext,
    build_direction_contexts,
)


EXPECTED_DIRECTIONS = {
    "holdout": {(1, 2), (2, 1), (2, 3)},
    "final": {(1, 2), (2, 1), (2, 3), (3, 2), (3, 4), (4, 3)},
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("rt", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def load_training_contexts(
    receipt_paths: Sequence[Path],
    manifest_path: Path,
    data_root: Path,
    *,
    training_stage: str,
) -> tuple[list[DirectionContext], dict[str, Any]]:
    if training_stage not in EXPECTED_DIRECTIONS:
        raise ValueError(f"unknown training stage: {training_stage!r}")
    if not receipt_paths:
        raise ValueError("at least one training receipt is required")

    contexts: list[DirectionContext] = []
    receipt_records: list[dict[str, str]] = []
    common_signature: dict[str, Any] | None = None
    for receipt_path in receipt_paths:
        receipt = _load_json(receipt_path)
        if receipt.get("completed") is not True:
            raise ValueError(f"training receipt is incomplete: {receipt_path}")
        if receipt.get("phase") != "development":
            raise ValueError(f"training receipt is not development data: {receipt_path}")
        signature = inference_receipt_signature(receipt)
        if common_signature is None:
            common_signature = signature
        elif signature != common_signature:
            raise ValueError("training receipts do not share inference provenance")
        loaded, _, _ = build_direction_contexts(
            receipt_path, manifest_path, data_root
        )
        contexts.extend(loaded)
        receipt_records.append(
            {
                "path": str(receipt_path.resolve()),
                "sha256": sha256_file(receipt_path),
            }
        )

    directions = [(item.source, item.target) for item in contexts]
    if len(set(directions)) != len(directions):
        raise ValueError("training receipts contain duplicate directed tasks")
    expected = EXPECTED_DIRECTIONS[training_stage]
    if set(directions) != expected:
        raise ValueError(
            f"{training_stage} training requires directions {sorted(expected)}, "
            f"got {sorted(directions)}"
        )
    contexts.sort(key=lambda item: (item.source, item.target))
    assert common_signature is not None
    provenance = {
        "stage": training_stage,
        "directions": [f"{source}->{target}" for source, target in sorted(expected)],
        "sources": sorted({source for source, _ in expected}),
        "receipts": sorted(receipt_records, key=lambda item: item["path"]),
        **common_signature,
    }
    return contexts, provenance


def train_bundle(
    contexts: Sequence[DirectionContext],
    *,
    implementation_commit: str,
    provenance: dict[str, Any],
):
    row_blocks = []
    rows_by_direction: dict[str, int] = {}
    for context in contexts:
        rows = extract_calibration_rows(
            context.source_grid,
            context.source_valid,
            context.forward_grid,
            context.forward_valid,
            context.return_grid,
            context.return_valid,
            context.baseline.eligible,
            context.target_index,
        )
        direction = f"{context.source}->{context.target}"
        rows_by_direction[direction] = rows.count
        row_blocks.append(rows)
    combined_rows = concatenate_calibration_rows(row_blocks)
    training = dict(provenance)
    training["rows_by_direction"] = rows_by_direction
    training["total_rows"] = combined_rows.count
    return fit_calibration_bundle(
        combined_rows,
        implementation_commit=implementation_commit,
        training=training,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit the frozen local copy-cycle calibration."
    )
    parser.add_argument("--receipt", type=Path, action="append", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--training-stage", choices=tuple(EXPECTED_DIRECTIONS), required=True
    )
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite calibration model: {args.output}")
    current_commit = clean_git_commit(Path(__file__).resolve().parents[4])
    if args.implementation_commit != current_commit:
        raise ValueError(
            "--implementation-commit must equal the clean checked-out commit"
        )
    contexts, provenance = load_training_contexts(
        args.receipt,
        args.manifest,
        args.data_root,
        training_stage=args.training_stage,
    )
    bundle = train_bundle(
        contexts,
        implementation_commit=args.implementation_commit,
        provenance=provenance,
    )
    write_json_atomic(args.output, bundle.to_json())


if __name__ == "__main__":
    main()
