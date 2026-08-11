import json

import numpy as np

from vesuvius.neural_tracing.evaluation.copy_cycle_io import (
    json_safe,
    sha256_tifxyz,
    write_json_atomic,
)


def test_json_safe_and_atomic_writer_emit_strict_json(tmp_path):
    target = tmp_path / "receipt.json"

    write_json_atomic(
        target,
        {"finite": np.float32(1.5), "infinite": float("inf"), "array": np.array([1, 2])},
    )

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload == {"array": [1, 2], "finite": 1.5, "infinite": None}
    assert json_safe(float("nan")) is None


def test_sha256_tifxyz_is_stable_and_filename_sensitive(tmp_path):
    surface = tmp_path / "surface"
    surface.mkdir()
    for filename in ("meta.json", "x.tif", "y.tif", "z.tif"):
        (surface / filename).write_bytes(filename.encode("ascii"))

    first = sha256_tifxyz(surface)
    second = sha256_tifxyz(surface)
    (surface / "x.tif").write_bytes(b"changed")

    assert first == second
    assert sha256_tifxyz(surface) != first
