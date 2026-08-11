import hashlib
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import tifffile

from vesuvius.neural_tracing.evaluation.copy_cycle_io import sha256_file, sha256_tifxyz
from vesuvius.neural_tracing.evaluation.copy_cycle_metrics import score_prediction
from vesuvius.neural_tracing.evaluation.score_copy_cycle_experiment import (
    build_direction_contexts,
    evaluate_primary_gate,
    parse_tau,
    validate_scoring_request,
)


def _score(prediction_value):
    source = np.zeros((4, 4, 3), dtype=np.float32)
    valid = np.ones((4, 4), dtype=bool)
    prediction = source.copy()
    prediction[..., 2] = float(prediction_value)
    target = np.array([[0.0, 0.0, 10.0]], dtype=np.float32)
    wrong = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    return score_prediction(source, valid, target, wrong, prediction, valid)


def test_parse_tau_accepts_frozen_infinity_spelling():
    assert np.isinf(parse_tau("infinity"))
    assert np.isinf(parse_tau("inf"))
    assert parse_tau("24") == 24.0


def test_primary_gate_reports_each_frozen_condition():
    baseline = [_score(8.0) for _ in range(4)]
    candidate = [_score(10.0) for _ in range(4)]
    source_stay = [_score(0.0) for _ in range(4)]
    wrong_sign = [_score(6.0) for _ in range(4)]
    detector = {
        "real": {"auroc": 0.9, "average_precision": 0.8},
        "shifted": {"auroc": 0.5, "average_precision": 0.4},
    }

    gate = evaluate_primary_gate(
        baseline, candidate, source_stay, wrong_sign, detector
    )

    assert gate["passed"] is True
    assert gate["improved_directions"] == 4
    assert gate["required_improved_directions"] == 3
    assert all(gate["conditions"].values())


def test_sealed_scoring_is_fixed_to_receipt_parameters():
    receipt = {
        "phase": "sealed_test",
        "selected_parameters": {"alpha": 0.5, "tau": 24.0},
    }

    validate_scoring_request(receipt, "fixed", 0.5, 24.0)
    with pytest.raises(ValueError, match="only be scored in fixed mode"):
        validate_scoring_request(receipt, "grid", None, None)
    with pytest.raises(ValueError, match="exactly match"):
        validate_scoring_request(receipt, "fixed", 0.75, 24.0)


def _write_tifxyz(path: Path, x_value: float) -> None:
    path.mkdir(parents=True)
    shape = (10, 10)
    x_values = np.full(shape, x_value, dtype=np.float32)
    y_values = np.zeros(shape, dtype=np.float32)
    z_values = np.full(shape, 100.0, dtype=np.float32)
    tifffile.imwrite(path / "x.tif", x_values)
    tifffile.imwrite(path / "y.tif", y_values)
    tifffile.imwrite(path / "z.tif", z_values)
    (path / "meta.json").write_text(
        json.dumps(
            {
                "scale": [1.0, 1.0],
                "bbox": [[x_value, 0.0, 100.0], [x_value, 0.0, 100.0]],
                "format": "tifxyz",
                "type": "seg",
                "uuid": path.name,
            }
        ),
        encoding="utf-8",
    )


def _file_specs(path: Path):
    output = {}
    for filename in ("meta.json", "x.tif", "y.tif", "z.tif"):
        payload = (path / filename).read_bytes()
        output[filename] = {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    return output


def test_build_direction_contexts_verifies_receipt_and_assigns_baseline(
    tmp_path, monkeypatch
):
    cv2_stub = types.ModuleType("cv2")
    cv2_stub.IMREAD_UNCHANGED = -1
    cv2_stub.imread = lambda path, _mode: tifffile.imread(path)
    monkeypatch.setitem(sys.modules, "cv2", cv2_stub)
    numba_stub = types.ModuleType("numba")
    numba_stub.njit = lambda *args, **kwargs: (
        (lambda function: function) if not args or not callable(args[0]) else args[0]
    )
    numba_stub.prange = range
    monkeypatch.setitem(sys.modules, "numba", numba_stub)
    data_root = tmp_path / "data"
    _write_tifxyz(data_root / "wrap01", 0.0)
    _write_tifxyz(data_root / "wrap02", 10.0)
    _write_tifxyz(data_root / "wrap03", 100.0)
    manifest = {
        "schema_version": 1,
        "scroll_id": "Synthetic",
        "bucket_base_url": "https://example.test",
        "expected_wraps": [1, 2, 3],
        "excluded_wraps": [3],
        "volume_url": "https://example.test/volume.zarr/",
        "volume_scale": 0,
        "splits": {"validation_edges": [[1, 2]]},
        "wraps": [
            {
                "wrap": wrap,
                "included_in_test": wrap != 3,
                "segment_id": f"segment-{wrap}",
                "tifxyz_dir": "mesh/surface.tifxyz",
                "files": _file_specs(data_root / f"wrap{wrap:02d}"),
            }
            for wrap in (1, 2, 3)
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    run_root = tmp_path / "run"
    run_root.mkdir()

    source_specs = {
        1: {
            "source": 0.0,
            "forward": {"front": 10.0, "back": -10.0},
            "roundtrip": {
                "front": {"front": 20.0, "back": 0.0},
                "back": {"front": 0.0, "back": -20.0},
            },
        },
        2: {
            "source": 10.0,
            "forward": {"front": 20.0, "back": 0.0},
            "roundtrip": {
                "front": {"front": 30.0, "back": 10.0},
                "back": {"front": 10.0, "back": -10.0},
            },
        },
    }
    source_receipts = []
    for wrap, spec in source_specs.items():
        source_dir = run_root / f"source_{wrap}"
        forward_paths = {}
        forward_hashes = {}
        roundtrip_receipt = {}
        for branch, value in spec["forward"].items():
            path = source_dir / f"forward_{branch}"
            _write_tifxyz(path, value)
            forward_paths[branch] = path.relative_to(run_root).as_posix()
            forward_hashes[branch] = sha256_tifxyz(path)
        for from_branch, outputs in spec["roundtrip"].items():
            paths = {}
            hashes = {}
            for return_branch, value in outputs.items():
                path = source_dir / f"roundtrip_{from_branch}_{return_branch}"
                _write_tifxyz(path, value)
                paths[return_branch] = path.relative_to(run_root).as_posix()
                hashes[return_branch] = sha256_tifxyz(path)
            selected = min(outputs, key=lambda name: abs(outputs[name] - spec["source"]))
            roundtrip_receipt[from_branch] = {
                "outputs": paths,
                "output_sha256": hashes,
                "selected_return": selected,
            }
        source_receipts.append(
            {
                "wrap": wrap,
                "input_sha256": sha256_tifxyz(data_root / f"wrap{wrap:02d}"),
                "forward": forward_paths,
                "forward_sha256": forward_hashes,
                "roundtrip": roundtrip_receipt,
            }
        )
    receipt = {
        "schema_version": 1,
        "completed": True,
        "phase": "validation",
        "scroll_id": "Synthetic",
        "manifest_sha256": sha256_file(manifest_path),
        "sources": source_receipts,
    }
    receipt_path = run_root / "run_receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    contexts, _, _ = build_direction_contexts(
        receipt_path, manifest_path, data_root
    )

    assert [(item.source, item.target, item.branch) for item in contexts] == [
        (1, 2, "front"),
        (2, 1, "back"),
    ]
    assert all(item.baseline.metrics["penalized_target_distance_mean"] == 0.0 for item in contexts)
    assert all(len(item.wrong_index.trees) == 1 for item in contexts)

    unassigned_return = run_root / "source_1" / "roundtrip_back_front" / "x.tif"
    unassigned_return.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="roundtrip back/front SHA-256 mismatch"):
        build_direction_contexts(receipt_path, manifest_path, data_root)
