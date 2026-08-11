"""Shared, auditable I/O helpers for copy-cycle experiments."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np


def load_tifxyz_grid(
    path: str | Path, *, coordinate_divisor: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Load a stored-resolution TIFXYZ as a ZYX grid and validity mask."""

    divisor = float(coordinate_divisor)
    if not math.isfinite(divisor) or divisor <= 0.0:
        raise ValueError(f"coordinate_divisor must be finite and positive, got {divisor}")
    from vesuvius.tifxyz import read_tifxyz

    surface = read_tifxyz(str(path))
    surface.use_stored_resolution()
    x_values, y_values, z_values, valid_values = surface[:]
    grid = np.stack([z_values, y_values, x_values], axis=-1).astype(np.float32, copy=False)
    if divisor != 1.0:
        grid = grid / divisor
    valid = np.asarray(valid_values, dtype=bool) & np.isfinite(grid).all(axis=2)
    grid = grid.copy()
    grid[~valid] = -1.0
    return grid, valid


def sha256_file(path: str | Path, buffer_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        while chunk := file_obj.read(int(buffer_size)):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tifxyz(path: str | Path) -> str:
    """Hash filenames and bytes for the four canonical TIFXYZ files."""

    root = Path(path)
    digest = hashlib.sha256()
    for filename in ("meta.json", "x.tif", "y.tif", "z.tif"):
        file_path = root / filename
        if not file_path.is_file():
            raise FileNotFoundError(file_path)
        digest.update(filename.encode("utf-8"))
        digest.update(b"\0")
        with file_path.open("rb") as file_obj:
            while chunk := file_obj.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json_atomic(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    safe_payload = json_safe(payload)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as temp_obj:
            temp_path = Path(temp_obj.name)
            json.dump(safe_payload, temp_obj, indent=2, sort_keys=True, allow_nan=False)
            temp_obj.write("\n")
        os.replace(temp_path, target)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
