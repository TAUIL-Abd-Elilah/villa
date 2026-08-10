import copy
import hashlib
from pathlib import Path

import pytest

from vesuvius.neural_tracing.evaluation.prepare_copy_cycle_benchmark import (
    default_manifest_path,
    iter_manifest_files,
    load_manifest,
    prepare_benchmark,
    validate_manifest,
    verify_file,
)


def test_preregistered_manifest_is_complete_and_valid():
    manifest = load_manifest(default_manifest_path())

    entries = list(iter_manifest_files(manifest, Path("benchmark")))

    assert len(entries) == 52
    assert {entry[0] for entry in entries} == set(range(1, 14))
    assert all(entry[1].startswith("https://") for entry in entries)
    assert entries[0][2] == Path("benchmark/wrap01/meta.json")


def test_validate_manifest_rejects_path_traversal():
    manifest = load_manifest(default_manifest_path())
    modified = copy.deepcopy(manifest)
    modified["wraps"][0]["tifxyz_dir"] = "../private"

    with pytest.raises(ValueError, match="safe relative"):
        validate_manifest(modified)


def test_verify_and_prepare_existing_files(tmp_path):
    payloads = {
        "meta.json": b"{}\n",
        "x.tif": b"x",
        "y.tif": b"yy",
        "z.tif": b"zzz",
    }
    files = {
        name: {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        for name, payload in payloads.items()
    }
    wraps = []
    for wrap in range(1, 14):
        wraps.append(
            {
                "wrap": wrap,
                "segment_id": f"segment-{wrap:02d}",
                "tifxyz_dir": "mesh/surface.tifxyz",
                "files": copy.deepcopy(files),
            }
        )
        wrap_dir = tmp_path / f"wrap{wrap:02d}"
        wrap_dir.mkdir()
        for name, payload in payloads.items():
            (wrap_dir / name).write_bytes(payload)

    manifest = {
        "schema_version": 1,
        "scroll_id": "PHerc0500P2",
        "bucket_base_url": "https://example.test",
        "wraps": wraps,
    }

    receipt = prepare_benchmark(manifest, tmp_path, download_missing=False)

    assert receipt["file_count"] == 52
    assert receipt["total_bytes"] == 13 * sum(map(len, payloads.values()))


def test_verify_file_rejects_same_size_wrong_bytes(tmp_path):
    target = tmp_path / "x.tif"
    target.write_bytes(b"bad")

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        verify_file(target, 3, hashlib.sha256(b"not").hexdigest())
