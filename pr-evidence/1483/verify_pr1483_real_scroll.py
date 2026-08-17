#!/usr/bin/env python3
"""Replay Villa's flat robust preprocessing on a public PHerc0139 CT cube."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import tifffile
import zarr

import vesuvius.ink_detection.inference.infer as infer_module
from vesuvius.ink_detection.inference.infer import (
    Block,
    FlatBlockDataset,
    FlatPatchReader,
    normalize_flat_patch,
)


EXPECTED_SOURCE_SHA256 = (
    "d4bd2dfaf5ee1518560e15ac767b76ea836bf43e0a96a8be92445e5082e73dc3"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        type=Path,
        help="Path to the extracted z04352_y03072_x02560.tif source cube",
    )
    return parser.parse_args()


def git_commit() -> str:
    repo = Path(infer_module.__file__).resolve().parents[5]
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def create_zarr(path: Path, real_zyx: np.ndarray) -> None:
    options = {
        "mode": "w",
        "shape": real_zyx.shape,
        "chunks": real_zyx.shape,
        "dtype": "u1",
    }
    try:
        source = zarr.open(path, zarr_format=2, **options)
    except TypeError:
        source = zarr.open(path, zarr_version=2, **options)
    source[:] = real_zyx


def main() -> int:
    source_path = parse_args().source.resolve()
    source_bytes = source_path.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if source_sha256 != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(f"public CT cube identity mismatch: {source_sha256}")

    cube_zyx = tifffile.imread(source_path)
    real_zyx = np.ascontiguousarray(cube_zyx[62:65], dtype=np.uint8)
    with tempfile.TemporaryDirectory(prefix="villa-pr1483-real-scroll-") as tmp:
        input_path = Path(tmp) / "public-pherc0139.zarr"
        create_zarr(input_path, real_zyx)
        reader = FlatPatchReader(
            input_path=input_path,
            resolution="0",
            depth_axis_first=True,
            height=128,
            width=128,
            layer_indices=np.arange(3),
            output_depth=5,
            preprocessing="tifxyz_robust",
        )
        actual, metadata = FlatBlockDataset(
            reader=reader,
            blocks=[Block(0, 0, 128, 128)],
            patch_size=128,
            preprocessing="tifxyz_robust",
        )[0]

    actual_zyx = actual.numpy()[0]
    expected_zyx = np.zeros((5, 128, 128), dtype=np.float32)
    expected_zyx[1:4] = normalize_flat_patch(
        real_zyx.copy(), "tifxyz_robust"
    )
    valid = np.zeros_like(expected_zyx, dtype=bool)
    valid[1:4] = True
    real_error = np.abs(actual_zyx[valid] - expected_zyx[valid])
    padding_nonzero = int(np.count_nonzero(actual_zyx[~valid]))
    padding_total = int((~valid).sum())
    passed = bool(
        np.array_equal(actual_zyx, expected_zyx) and padding_nonzero == 0
    )

    print("Villa flat robust-MAD real-scroll parity check")
    print(f"implementation_commit={git_commit()}")
    print(f"implementation_file={Path(infer_module.__file__).resolve()}")
    print("source_scroll=PHerc0139")
    print(
        "source_uri=s3://vesuvius-challenge-open-data/PHerc0139/volumes/"
        "20250728140407-9.362um-1.2m-113keV-masked.zarr"
    )
    print(f"source_cube_sha256={source_sha256}")
    print("source_global_l0_zyx=[4352:4480,3072:3200,2560:2688]")
    print("replay=local_z[62:65] centered into depth 5")
    print(
        "source_nonzero_fraction="
        f"{np.count_nonzero(real_zyx) / real_zyx.size:.6f}"
    )
    print(f"real_voxel_mae_vs_training={float(real_error.mean()):.6f}")
    print(f"real_voxel_max_abs_vs_training={float(real_error.max()):.6f}")
    print(f"nonzero_padded_voxels={padding_nonzero}/{padding_total}")
    print(f"metadata={metadata.tolist()}")
    print(f"RESULT={'PASS' if passed else 'FAIL'}")
    if not passed:
        print("ERROR: flat inference input differs from training semantics")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
