#!/usr/bin/env python3
"""Acquire, verify, extract, and prepare the official GeoLife 1.3 release.

GeoLife is intentionally not stored in this repository. Automatic download is
performed only after the caller confirms that the original dataset terms have
been reviewed. All source and coordinate-bearing derived files are written to
ignored local directories.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

import verify_geolife_archive as verifier


ROOT = Path(__file__).resolve().parents[1]
DIRECT_DOWNLOAD_URL = (
    "https://download.microsoft.com/download/f/4/8/"
    "f4894aa5-fdbc-481e-9285-d5f8c4c4f039/"
    "Geolife%20Trajectories%201.3.zip"
)
DEFAULT_ARCHIVE = ROOT / "data" / "private" / verifier.EXPECTED_FILENAME
DEFAULT_EXTRACT_DIR = ROOT / "data" / "private" / "geolife-1.3"
DEFAULT_PREPARED_DIR = ROOT / "outputs" / "private" / "source_preparation"


def download(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    partial.unlink(missing_ok=True)
    request = urllib.request.Request(
        DIRECT_DOWNLOAD_URL,
        headers={"User-Agent": "coherent-personal-mobility-forecasting/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as source, partial.open("wb") as target:
            shutil.copyfileobj(source, target, length=4 * 1024 * 1024)
        if partial.stat().st_size != verifier.EXPECTED_SIZE_BYTES:
            raise RuntimeError(
                "Downloaded archive has the wrong size: "
                f"expected {verifier.EXPECTED_SIZE_BYTES}, found {partial.stat().st_size}."
            )
        digest = verifier.sha256(partial)
        if digest != verifier.EXPECTED_SHA256:
            raise RuntimeError(
                "Downloaded archive failed SHA-256 verification: "
                f"expected {verifier.EXPECTED_SHA256}, found {digest}."
            )
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def verified_extract(archive_path: Path, extract_dir: Path, force: bool) -> Path:
    if archive_path.stat().st_size != verifier.EXPECTED_SIZE_BYTES:
        raise RuntimeError(f"Unexpected archive size: {archive_path}")
    digest = verifier.sha256(archive_path)
    if digest != verifier.EXPECTED_SHA256:
        raise RuntimeError(
            f"GeoLife archive checksum mismatch: expected {verifier.EXPECTED_SHA256}, found {digest}."
        )
    with zipfile.ZipFile(archive_path) as archive:
        verifier.validate_members(archive)
        return verifier.extract_archive(archive, extract_dir, force)


def run_checked(arguments: list[str]) -> None:
    subprocess.run(arguments, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--accept-license",
        action="store_true",
        help="Confirm that the official GeoLife terms were reviewed before automatic download.",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        help="Use an already downloaded GeoLife 1.3 ZIP instead of downloading it.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        help="Use an already extracted GeoLife 1.3 directory and skip archive acquisition.",
    )
    parser.add_argument("--extract-dir", type=Path, default=DEFAULT_EXTRACT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PREPARED_DIR)
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    if args.archive is not None and args.raw_root is not None:
        raise SystemExit("Use either --archive or --raw-root, not both.")

    if args.raw_root is not None:
        supplied = args.raw_root.expanduser().resolve()
        candidates = [supplied, supplied / verifier.EXPECTED_ROOT]
        extracted_root = next(
            (path for path in candidates if verifier.extracted_counts(path) == (182, 18670, 69)),
            None,
        )
        if extracted_root is None:
            raise SystemExit(
                f"The extracted GeoLife directory has unexpected release counts: {supplied}"
            )
    else:
        archive_path = (args.archive or DEFAULT_ARCHIVE).expanduser().resolve()
        if not archive_path.is_file():
            if args.archive is not None:
                raise SystemExit(f"GeoLife archive not found: {archive_path}")
            if not args.accept_license:
                raise SystemExit(
                    "GeoLife is not redistributed by this repository. Review the official terms at\n"
                    f"  {verifier.OFFICIAL_DOWNLOAD_PAGE}\n"
                    "then rerun with --accept-license, pass --archive /path/to/Geolife Trajectories 1.3.zip, "
                    "or pass --raw-root /path/to/the/extracted/release."
                )
            print(f"Downloading the official GeoLife 1.3 archive to {archive_path}", flush=True)
            download(archive_path)

        extracted_root = verified_extract(
            archive_path,
            args.extract_dir.expanduser().resolve(),
            args.force_extract,
        )
    print(f"Verified GeoLife 1.3 source: {extracted_root}", flush=True)
    if args.verify_only:
        return 0

    output_dir = args.output_dir.expanduser().resolve()
    validation = output_dir / "validation_report.json"
    if validation.is_file() and not args.force_prepare:
        print(
            f"Prepared GeoLife input already exists at {output_dir}; "
            "use --force-prepare to rebuild it.",
            flush=True,
        )
        return 0
    if output_dir.exists() and args.force_prepare:
        shutil.rmtree(output_dir)

    run_checked(
        [
            sys.executable,
            str(ROOT / "scripts" / "prepare_geolife.py"),
            "--config",
            str(ROOT / "config" / "publication.json"),
            "--raw-root",
            str(extracted_root),
            "--output-dir",
            str(output_dir),
        ]
    )
    run_checked(
        [
            sys.executable,
            str(ROOT / "scripts" / "validate_geolife_preparation.py"),
            "--output-dir",
            str(output_dir),
            "--require-complete-release",
        ]
    )
    print(f"Prepared private forecasting input: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
