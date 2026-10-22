#!/usr/bin/env python3
"""Validate source-faithful GeoLife preparation without reading coordinates."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fail(message: str) -> None:
    raise RuntimeError(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs" / "private" / "source_preparation"),
    )
    parser.add_argument(
        "--require-complete-release",
        action="store_true",
        help="Reject diagnostic runs restricted to selected users.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    private_dir = output_dir / "private"
    summary_path = output_dir / "preparation_summary.json"
    source_path = private_dir / "geolife_source.csv.gz"
    audit_path = private_dir / "source_trajectory_audit.csv"
    for path in (summary_path, source_path, audit_path):
        if not path.is_file():
            fail(f"Missing required preparation output: {path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "completed":
        fail("Preparation summary does not report completed status.")
    if args.require_complete_release and summary.get("scan_scope") != "complete_release":
        fail("The preparation did not scan the complete GeoLife release.")
    if summary.get("trajectory_unit") != "one original PLT file":
        fail("Trajectory identity is not source-faithful.")
    invariants = {
        "manual_segmentation": False,
        "trajectory_subsampling": False,
        "point_downsampling": False,
        "row_level_mode_filtering": False,
        "complete_trajectory_mode_filtering": True,
        "large_gap_filtering": False,
        "high_implied_speed_filtering": False,
        "preserve_all_valid_points": True,
    }
    for key, expected in invariants.items():
        if summary.get(key) != expected:
            fail(f"Preparation invariant failed for {key}: {summary.get(key)!r}")

    if sha256(source_path) != summary["private_standardized_csv_sha256"]:
        fail("Prepared coordinate CSV checksum does not match the summary.")
    if sha256(audit_path) != summary["private_audit_sha256"]:
        fail("Trajectory audit checksum does not match the summary.")

    selected_users = {int(value) for value in summary["selected_users"]}
    expected_trips = int(summary["selected_trajectories"])
    expected_points = int(summary["selected_point_rows"])
    seen_users: set[int] = set()
    seen_trips: set[tuple[int, int]] = set()
    point_rows = 0
    with gzip.open(source_path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "ID",
            "trip_id",
            "latitude",
            "longitude",
            "datetime",
            "Transportation Mode",
        }
        missing = sorted(required.difference(reader.fieldnames or []))
        if missing:
            fail(f"Prepared CSV is missing columns: {missing}")
        for row in reader:
            user_id = int(row["ID"])
            trip_id = int(row["trip_id"])
            seen_users.add(user_id)
            seen_trips.add((user_id, trip_id))
            point_rows += 1

    if seen_users != selected_users:
        fail("User IDs in the prepared CSV differ from the preparation summary.")
    if len(seen_trips) != expected_trips:
        fail(
            f"Prepared trip count mismatch: expected {expected_trips}, "
            f"found {len(seen_trips)}."
        )
    if point_rows != expected_points:
        fail(
            f"Prepared point-row mismatch: expected {expected_points}, "
            f"found {point_rows}."
        )

    report = {
        "status": "passed",
        "scan_scope": summary["scan_scope"],
        "selected_user_count": len(seen_users),
        "selected_trajectories": len(seen_trips),
        "selected_point_rows": point_rows,
        "mode_filtering_unit": "complete source trajectory",
        "trajectory_subsampling": False,
    }
    report_path = output_dir / "validation_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
