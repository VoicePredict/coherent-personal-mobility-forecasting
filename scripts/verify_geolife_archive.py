#!/usr/bin/env python3
"""Verify the manually downloaded official GeoLife 1.3 archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path, PurePosixPath


EXPECTED_FILENAME = "Geolife Trajectories 1.3.zip"
EXPECTED_SIZE_BYTES = 313_164_406
EXPECTED_SHA256 = "1107c5ac064d0a23c8d021a8736a77e53abc75b227062e6260342c6a8d86bdb6"
EXPECTED_ROOT = "Geolife Trajectories 1.3"
EXPECTED_PLT_FILES = 18_670
EXPECTED_LABEL_FILES = 69
OFFICIAL_DOWNLOAD_PAGE = "https://www.microsoft.com/en-us/download/details.aspx?id=52367"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_members(archive: zipfile.ZipFile) -> dict[str, int]:
    plt_files = 0
    label_files = 0
    user_guide = 0
    root_prefix = f"{EXPECTED_ROOT}/"
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        if path.is_absolute() or ".." in path.parts:
            raise RuntimeError(f"Unsafe path in archive: {info.filename}")
        if info.filename and not (
            info.filename == f"{EXPECTED_ROOT}/" or info.filename.startswith(root_prefix)
        ):
            raise RuntimeError(f"Unexpected archive root: {info.filename}")
        if info.filename.lower().endswith(".plt"):
            plt_files += 1
        if info.filename.endswith("/labels.txt"):
            label_files += 1
        if info.filename == f"{EXPECTED_ROOT}/User Guide-1.3.pdf":
            user_guide += 1
    if plt_files != EXPECTED_PLT_FILES:
        raise RuntimeError(
            f"Expected {EXPECTED_PLT_FILES} PLT files, found {plt_files}."
        )
    if label_files != EXPECTED_LABEL_FILES:
        raise RuntimeError(
            f"Expected {EXPECTED_LABEL_FILES} label files, found {label_files}."
        )
    if user_guide != 1:
        raise RuntimeError("GeoLife 1.3 User Guide was not found exactly once.")
    return {
        "plt_files": plt_files,
        "label_files": label_files,
        "user_guide_files": user_guide,
    }


def extracted_counts(root: Path) -> tuple[int, int, int]:
    data = root / "Data"
    users = sum(1 for path in data.iterdir() if path.is_dir()) if data.is_dir() else 0
    plt_files = sum(1 for _ in data.glob("*/Trajectory/*.plt")) if data.is_dir() else 0
    label_files = sum(1 for _ in data.glob("*/labels.txt")) if data.is_dir() else 0
    return users, plt_files, label_files


def extract_archive(archive: zipfile.ZipFile, destination: Path, force: bool) -> Path:
    destination = destination.expanduser().resolve()
    root = destination / EXPECTED_ROOT
    if root.exists():
        counts = extracted_counts(root)
        if counts == (182, EXPECTED_PLT_FILES, EXPECTED_LABEL_FILES) and not force:
            return root
        if not force:
            raise RuntimeError(
                f"Extraction target already exists but is incomplete: {root}. "
                "Use --force-extract to replace it."
            )
        shutil.rmtree(root)
    destination.mkdir(parents=True, exist_ok=True)
    archive.extractall(destination)
    counts = extracted_counts(root)
    if counts != (182, EXPECTED_PLT_FILES, EXPECTED_LABEL_FILES):
        raise RuntimeError(f"Extracted dataset has unexpected counts: {counts}")
    return root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--extract-to")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--json-out")
    args = parser.parse_args()

    archive_path = Path(args.archive).expanduser().resolve()
    if not archive_path.is_file():
        raise SystemExit(
            f"GeoLife archive not found: {archive_path}\n"
            f"Download {EXPECTED_FILENAME} manually from {OFFICIAL_DOWNLOAD_PAGE}."
        )
    size = archive_path.stat().st_size
    if size != EXPECTED_SIZE_BYTES:
        raise SystemExit(
            f"Archive size mismatch: expected {EXPECTED_SIZE_BYTES}, found {size}."
        )
    digest = sha256(archive_path)
    if digest != EXPECTED_SHA256:
        raise SystemExit(
            f"Archive SHA-256 mismatch: expected {EXPECTED_SHA256}, found {digest}."
        )

    with zipfile.ZipFile(archive_path) as archive:
        member_counts = validate_members(archive)
        extracted_root = None
        if args.extract_to:
            extracted_root = extract_archive(
                archive, Path(args.extract_to), args.force_extract
            )

    report = {
        "status": "passed",
        "required_release": "1.3",
        "archive_filename": archive_path.name,
        "archive_size_bytes": size,
        "archive_sha256": digest,
        "official_download_page": OFFICIAL_DOWNLOAD_PAGE,
        **member_counts,
        "extracted_root": str(extracted_root) if extracted_root else None,
    }
    payload = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.json_out:
        output = Path(args.json_out).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
