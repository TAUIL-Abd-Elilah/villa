"""Download and verify a preregistered copy-cycle benchmark manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any, Iterable
from urllib.request import urlopen


_EXPECTED_FILES = ("meta.json", "x.tif", "y.tif", "z.tif")
_BUFFER_SIZE = 1024 * 1024


def default_manifest_path() -> Path:
    return Path(__file__).resolve().parents[4] / "docs" / "copy_cycle_pherc0500p2_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        while chunk := file_obj.read(_BUFFER_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _require_safe_relative_path(value: Any, label: str) -> str:
    path = PurePosixPath(str(value))
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{label} must be a safe relative POSIX path, got {value!r}")
    return path.as_posix()


def validate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    if int(manifest.get("schema_version", -1)) != 1:
        raise ValueError("copy-cycle manifest schema_version must be 1")

    bucket_base = str(manifest.get("bucket_base_url", "")).rstrip("/")
    if not bucket_base.startswith("https://"):
        raise ValueError("bucket_base_url must be an https URL")
    scroll_id = _require_safe_relative_path(manifest.get("scroll_id"), "scroll_id")
    if "/" in scroll_id:
        raise ValueError("scroll_id must have one path component")

    expected_wraps_raw = manifest.get("expected_wraps")
    if not isinstance(expected_wraps_raw, list) or not expected_wraps_raw:
        raise ValueError("expected_wraps must be a non-empty list")
    expected_wraps = {int(value) for value in expected_wraps_raw}
    if len(expected_wraps) != len(expected_wraps_raw) or min(expected_wraps) < 1:
        raise ValueError("expected_wraps must contain unique positive integers")

    wraps = manifest.get("wraps")
    if not isinstance(wraps, list) or not wraps:
        raise ValueError("manifest wraps must be a non-empty list")

    excluded_raw = manifest.get("excluded_wraps", [])
    if not isinstance(excluded_raw, list):
        raise ValueError("excluded_wraps must be a list when present")
    excluded_wraps = {int(value) for value in excluded_raw}
    if len(excluded_wraps) != len(excluded_raw) or not excluded_wraps.issubset(
        expected_wraps
    ):
        raise ValueError("excluded_wraps must contain unique expected wrap identifiers")

    seen_wraps: set[int] = set()
    seen_remote_paths: set[str] = set()
    for entry in wraps:
        if not isinstance(entry, dict):
            raise ValueError("each wrap entry must be an object")
        wrap = int(entry.get("wrap", -1))
        if wrap < 1 or wrap in seen_wraps:
            raise ValueError(f"wrap identifiers must be unique positive integers, got {wrap}")
        seen_wraps.add(wrap)

        segment_id = _require_safe_relative_path(entry.get("segment_id"), f"wrap {wrap} segment_id")
        if "/" in segment_id:
            raise ValueError(f"wrap {wrap} segment_id must have one path component")
        tifxyz_dir = _require_safe_relative_path(entry.get("tifxyz_dir"), f"wrap {wrap} tifxyz_dir")
        local_dir = _require_safe_relative_path(
            entry.get("local_dir", f"wrap{wrap:02d}"), f"wrap {wrap} local_dir"
        )
        if "/" in local_dir:
            raise ValueError(f"wrap {wrap} local_dir must have one path component")

        files = entry.get("files")
        if not isinstance(files, dict) or tuple(sorted(files)) != tuple(sorted(_EXPECTED_FILES)):
            raise ValueError(f"wrap {wrap} must contain exactly {_EXPECTED_FILES!r}")
        for filename in _EXPECTED_FILES:
            spec = files[filename]
            if not isinstance(spec, dict):
                raise ValueError(f"wrap {wrap} {filename} spec must be an object")
            byte_count = int(spec.get("bytes", -1))
            sha256 = str(spec.get("sha256", "")).lower()
            if byte_count < 1:
                raise ValueError(f"wrap {wrap} {filename} byte count must be positive")
            if len(sha256) != 64 or any(char not in "0123456789abcdef" for char in sha256):
                raise ValueError(f"wrap {wrap} {filename} has an invalid SHA-256")
            remote_path = f"{scroll_id}/segments/{segment_id}/{tifxyz_dir}/{filename}"
            if remote_path in seen_remote_paths:
                raise ValueError(f"duplicate remote path in manifest: {remote_path}")
            seen_remote_paths.add(remote_path)

    if seen_wraps != expected_wraps:
        raise ValueError(
            f"benchmark wraps must equal expected_wraps={sorted(expected_wraps)}, "
            f"got {sorted(seen_wraps)}"
        )
    marked_excluded = {
        int(entry["wrap"])
        for entry in wraps
        if entry.get("included_in_test") is False
    }
    if marked_excluded != excluded_wraps:
        raise ValueError(
            "excluded_wraps must exactly match wraps marked included_in_test=false"
        )
    edge_source = manifest.get("splits", manifest)
    for edge_key in (
        "development_edges",
        "validation_edges",
        "sealed_test_edges",
        "buffer_edges",
    ):
        raw_edges = edge_source.get(edge_key)
        if raw_edges is None:
            continue
        if not isinstance(raw_edges, list) or not raw_edges:
            raise ValueError(f"{edge_key} must be a non-empty list when present")
        seen_edges: set[tuple[int, int]] = set()
        for raw_edge in raw_edges:
            if not isinstance(raw_edge, list) or len(raw_edge) != 2:
                raise ValueError(f"invalid {edge_key} edge: {raw_edge!r}")
            first, second = int(raw_edge[0]), int(raw_edge[1])
            if first == second or first not in expected_wraps or second not in expected_wraps:
                raise ValueError(f"invalid {edge_key} edge: {raw_edge!r}")
            canonical = (min(first, second), max(first, second))
            if canonical in seen_edges:
                raise ValueError(f"duplicate {edge_key} edge: {raw_edge!r}")
            seen_edges.add(canonical)
            if edge_key != "buffer_edges" and ({first, second} & excluded_wraps):
                raise ValueError(f"{edge_key} references an excluded wrap: {raw_edge!r}")
    return manifest


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("rt", encoding="utf-8") as file_obj:
        manifest = json.load(file_obj)
    if not isinstance(manifest, dict):
        raise ValueError("copy-cycle manifest root must be an object")
    return validate_manifest(manifest)


def iter_manifest_files(
    manifest: dict[str, Any], destination: Path
) -> Iterable[tuple[int, str, Path, int, str]]:
    base_url = str(manifest["bucket_base_url"]).rstrip("/")
    scroll_id = str(manifest["scroll_id"])
    for entry in sorted(manifest["wraps"], key=lambda item: int(item["wrap"])):
        wrap = int(entry["wrap"])
        remote_root = (
            f"{base_url}/{scroll_id}/segments/{entry['segment_id']}/{entry['tifxyz_dir']}"
        )
        local_root = destination / str(entry.get("local_dir", f"wrap{wrap:02d}"))
        for filename in _EXPECTED_FILES:
            spec = entry["files"][filename]
            yield (
                wrap,
                f"{remote_root}/{filename}",
                local_root / filename,
                int(spec["bytes"]),
                str(spec["sha256"]).lower(),
            )


def verify_file(path: Path, expected_bytes: int, expected_sha256: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_bytes = path.stat().st_size
    if actual_bytes != int(expected_bytes):
        raise RuntimeError(
            f"size mismatch for {path}: expected {expected_bytes}, found {actual_bytes}"
        )
    actual_sha256 = sha256_file(path)
    if actual_sha256 != str(expected_sha256).lower():
        raise RuntimeError(
            f"SHA-256 mismatch for {path}: expected {expected_sha256}, found {actual_sha256}"
        )
    return {"path": str(path.resolve()), "bytes": actual_bytes, "sha256": actual_sha256}


def _download_verified(
    url: str,
    target: Path,
    expected_bytes: int,
    expected_sha256: str,
    timeout_seconds: float,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{target.name}.", suffix=".part", dir=target.parent, delete=False
        ) as temp_obj:
            temp_path = Path(temp_obj.name)
            with urlopen(url, timeout=timeout_seconds) as response:
                while chunk := response.read(_BUFFER_SIZE):
                    temp_obj.write(chunk)
        verify_file(temp_path, expected_bytes, expected_sha256)
        os.replace(temp_path, target)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def prepare_benchmark(
    manifest: dict[str, Any],
    destination: Path,
    *,
    download_missing: bool,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    validate_manifest(manifest)
    destination = destination.resolve()
    receipts: list[dict[str, Any]] = []
    for wrap, url, path, expected_bytes, expected_sha256 in iter_manifest_files(
        manifest, destination
    ):
        if not path.exists():
            if not download_missing:
                raise FileNotFoundError(
                    f"missing benchmark file {path}; rerun with --download-missing"
                )
            _download_verified(
                url,
                path,
                expected_bytes,
                expected_sha256,
                timeout_seconds,
            )
        receipt = verify_file(path, expected_bytes, expected_sha256)
        receipt.update({"wrap": wrap, "url": url})
        receipts.append(receipt)

    return {
        "schema_version": 1,
        "scroll_id": manifest["scroll_id"],
        "destination": str(destination),
        "file_count": len(receipts),
        "total_bytes": sum(int(item["bytes"]) for item in receipts),
        "files": receipts,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download missing copy-cycle benchmark files and verify every byte."
    )
    parser.add_argument("--manifest", type=Path, default=default_manifest_path())
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--download-missing", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--receipt", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = load_manifest(args.manifest)
    receipt = prepare_benchmark(
        manifest,
        args.destination,
        download_missing=bool(args.download_missing),
        timeout_seconds=float(args.timeout_seconds),
    )
    payload = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.receipt is not None:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
