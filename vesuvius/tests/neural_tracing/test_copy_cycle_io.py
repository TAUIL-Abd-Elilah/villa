import json
import subprocess

import numpy as np
import pytest

from vesuvius.neural_tracing.evaluation.copy_cycle_io import (
    clean_git_commit,
    json_safe,
    sha256_tifxyz,
    write_json_atomic,
)


def test_clean_git_commit_rejects_uncommitted_state(tmp_path):
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "tracked.txt"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Copy Cycle Test",
            "-c",
            "user.email=copy-cycle@example.test",
            "commit",
            "-m",
            "initial",
        ],
        check=True,
        capture_output=True,
    )

    commit = clean_git_commit(tmp_path)

    assert len(commit) == 40
    tracked.write_text("dirty\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="clean worktree"):
        clean_git_commit(tmp_path)


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
