#!/usr/bin/env python3
"""Convert official GeoLife PLT files to the unchanged forecast input schema.

Each source PLT file remains one trajectory. No manual or gap-based
segmentation, trajectory subsampling, or point downsampling is introduced.
The pre-existing transportation-mode restrictions are applied to complete
source trajectories rather than by deleting labelled rows from a trajectory.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import hashlib
import io
import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "publication.json"
EARTH_RADIUS_M = 6_371_008.8


@dataclass(frozen=True)
class Point:
    latitude: float
    longitude: float
    altitude_ft: float
    timestamp: datetime


@dataclass(frozen=True)
class LabelInterval:
    start: datetime
    end: datetime
    mode: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_mode(value: str) -> str:
    return value.strip().lower().replace(" ", "_") or "unknown"


def resolve_raw_root(value: str | Path, expected_root_name: str) -> Path:
    root = Path(value).expanduser().resolve()
    candidates = [root, root / expected_root_name]
    for candidate in candidates:
        if (candidate / "Data").is_dir():
            return candidate
    raise FileNotFoundError(
        f"GeoLife Data directory was not found under {root}. "
        f"Expected either {root / 'Data'} or {root / expected_root_name / 'Data'}."
    )


def parse_label_time(value: str) -> datetime:
    return datetime.strptime(value.strip(), "%Y/%m/%d %H:%M:%S")


def load_labels(user_dir: Path) -> tuple[list[LabelInterval], list[datetime]]:
    path = user_dir / "labels.txt"
    if not path.is_file():
        return [], []
    intervals: list[LabelInterval] = []
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            try:
                start = parse_label_time(row["Start Time"])
                end = parse_label_time(row["End Time"])
                mode = normalize_mode(row["Transportation Mode"])
            except (KeyError, TypeError, ValueError):
                continue
            if end >= start:
                intervals.append(LabelInterval(start=start, end=end, mode=mode))
    intervals.sort(key=lambda item: (item.start, item.end, item.mode))
    return intervals, [item.start for item in intervals]


def mode_at(
    timestamp: datetime,
    intervals: list[LabelInterval],
    starts: list[datetime],
) -> str:
    position = bisect.bisect_right(starts, timestamp) - 1
    if position < 0:
        return "unknown"
    interval = intervals[position]
    return interval.mode if timestamp <= interval.end else "unknown"


def parse_point(fields: list[str]) -> Point | None:
    if len(fields) < 7:
        return None
    try:
        latitude = float(fields[0])
        longitude = float(fields[1])
        timestamp = datetime.fromisoformat(
            f"{fields[5].strip()} {fields[6].strip()}"
        )
    except (TypeError, ValueError):
        return None
    if not (-90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0):
        return None
    try:
        altitude = float(fields[3])
    except (TypeError, ValueError):
        altitude = float("nan")
    return Point(
        latitude=latitude,
        longitude=longitude,
        altitude_ft=altitude,
        timestamp=timestamp,
    )


def iter_plt_points(path: Path) -> Iterator[Point]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        for _ in range(6):
            next(reader, None)
        for fields in reader:
            point = parse_point(fields)
            if point is not None:
                yield point


def haversine_m(a: Point, b: Point) -> float:
    lat1 = math.radians(a.latitude)
    lat2 = math.radians(b.latitude)
    dlat = lat2 - lat1
    dlon = math.radians(b.longitude - a.longitude)
    value = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(value)))


def dominant_known_mode(counts: Counter[str]) -> str:
    known = [(mode, count) for mode, count in counts.items() if mode != "unknown"]
    if not known:
        return "unknown"
    known.sort(key=lambda item: (-item[1], item[0]))
    return known[0][0]


def inspect_trajectory(
    path: Path,
    raw_root: Path,
    user_id: int,
    intervals: list[LabelInterval],
    starts: list[datetime],
    selection: dict,
) -> dict[str, object]:
    raw_rows = 0
    valid_points = 0
    invalid_rows = 0
    first_time: datetime | None = None
    last_time: datetime | None = None
    minimum_time: datetime | None = None
    maximum_time: datetime | None = None
    previous: Point | None = None
    path_length_m = 0.0
    max_gap_s = 0.0
    max_speed_mps = 0.0
    nonpositive_steps = 0
    mode_counts: Counter[str] = Counter()

    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        for _ in range(6):
            next(reader, None)
        for fields in reader:
            raw_rows += 1
            point = parse_point(fields)
            if point is None:
                invalid_rows += 1
                continue
            valid_points += 1
            if first_time is None:
                first_time = point.timestamp
            last_time = point.timestamp
            minimum_time = (
                point.timestamp
                if minimum_time is None
                else min(minimum_time, point.timestamp)
            )
            maximum_time = (
                point.timestamp
                if maximum_time is None
                else max(maximum_time, point.timestamp)
            )
            mode_counts[mode_at(point.timestamp, intervals, starts)] += 1
            if previous is not None:
                elapsed_s = (point.timestamp - previous.timestamp).total_seconds()
                distance_m = haversine_m(previous, point)
                path_length_m += distance_m
                if elapsed_s <= 0:
                    nonpositive_steps += 1
                else:
                    max_gap_s = max(max_gap_s, elapsed_s)
                    max_speed_mps = max(max_speed_mps, distance_m / elapsed_s)
            previous = point

    duration_s = (
        (maximum_time - minimum_time).total_seconds()
        if minimum_time is not None and maximum_time is not None
        else 0.0
    )
    known_modes = sorted(mode for mode in mode_counts if mode != "unknown")
    disallowed_modes = sorted(
        set(known_modes).difference(selection["allowed_modes"])
    )
    status = "eligible"
    reason = "retained"
    if valid_points == 0:
        status, reason = "excluded", "no_valid_points"
    elif valid_points < int(selection["min_valid_points_per_trajectory"]):
        status, reason = "excluded", "too_few_valid_points"
    elif duration_s < float(selection["min_duration_seconds"]):
        status, reason = "excluded", "duration_below_minimum"
    elif disallowed_modes:
        status, reason = "excluded", "whole_trajectory_disallowed_mode_overlap"

    return {
        "user_id": user_id,
        "trip_id": path.stem,
        "source_file": str(path.relative_to(raw_root)),
        "raw_rows": raw_rows,
        "valid_points": valid_points,
        "invalid_rows": invalid_rows,
        "start_datetime": minimum_time.isoformat() if minimum_time else "",
        "end_datetime": maximum_time.isoformat() if maximum_time else "",
        "duration_s": duration_s,
        "dominant_mode": dominant_known_mode(mode_counts),
        "known_modes": "|".join(known_modes),
        "disallowed_modes": "|".join(disallowed_modes),
        "labelled_point_share": (
            1.0 - mode_counts.get("unknown", 0) / valid_points
            if valid_points
            else 0.0
        ),
        "duplicate_or_nonpositive_time_steps": nonpositive_steps,
        "max_recording_gap_s": max_gap_s,
        "path_length_m": path_length_m,
        "max_implied_speed_mps": max_speed_mps,
        "status": status,
        "reason": reason,
        "_path": path,
    }


def write_audit(path: Path, rows: list[dict[str, object]]) -> None:
    public_keys = [key for key in rows[0] if not key.startswith("_")]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=public_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in public_keys})


def deterministic_gzip_text(path: Path) -> tuple[io.TextIOWrapper, object, object]:
    raw = path.open("wb")
    compressed = gzip.GzipFile(
        filename="",
        mode="wb",
        compresslevel=6,
        fileobj=raw,
        mtime=0,
    )
    text = io.TextIOWrapper(compressed, encoding="utf-8", newline="")
    return text, compressed, raw


def write_standardized_points(
    path: Path,
    selected: list[dict[str, object]],
    raw_root: Path,
) -> int:
    header = [
        "ID",
        "trip_id",
        "latitude",
        "longitude",
        "altitude_feet",
        "datetime",
        "date",
        "time",
        "Transportation Mode",
    ]
    text, compressed, raw = deterministic_gzip_text(path)
    row_count = 0
    try:
        writer = csv.writer(text)
        writer.writerow(header)
        for metadata in sorted(
            selected,
            key=lambda row: (
                int(row["user_id"]),
                str(row["start_datetime"]),
                str(row["trip_id"]),
            ),
        ):
            source_path = Path(metadata["_path"])
            user_id = int(metadata["user_id"])
            if not str(metadata["trip_id"]).isdigit():
                raise RuntimeError(f"Non-numeric GeoLife PLT stem: {source_path}")
            trip_id = int(str(metadata["trip_id"]))
            user_dir = raw_root / "Data" / f"{user_id:03d}"
            intervals, starts = load_labels(user_dir)
            for point in iter_plt_points(source_path):
                mode = mode_at(point.timestamp, intervals, starts)
                writer.writerow(
                    [
                        user_id,
                        trip_id,
                        f"{point.latitude:.12g}",
                        f"{point.longitude:.12g}",
                        (
                            ""
                            if not math.isfinite(point.altitude_ft)
                            else f"{point.altitude_ft:.12g}"
                        ),
                        point.timestamp.isoformat(sep=" "),
                        point.timestamp.date().isoformat(),
                        point.timestamp.time().isoformat(),
                        mode,
                    ]
                )
                row_count += 1
        text.flush()
    finally:
        text.close()
        if not compressed.closed:
            compressed.close()
        if not raw.closed:
            raw.close()
    return row_count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--raw-root", required=True)
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs" / "private" / "source_preparation"),
    )
    parser.add_argument(
        "--user-id",
        action="append",
        type=int,
        default=[],
        help="Diagnostic-only source restriction; never used by the full run.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    selection = dict(config["data_selection"])
    selection["allowed_modes"] = {
        normalize_mode(value) for value in selection["allowed_modes"]
    }
    raw_root = resolve_raw_root(
        args.raw_root, str(selection["archive_root_directory"])
    )
    data_root = raw_root / "Data"
    all_user_dirs = sorted(
        (
            path
            for path in data_root.iterdir()
            if path.is_dir() and path.name.isdigit()
        ),
        key=lambda path: int(path.name),
    )
    total_plt_files = sum(
        1 for user_dir in all_user_dirs for _ in (user_dir / "Trajectory").glob("*.plt")
    )
    total_label_files = sum(1 for user_dir in all_user_dirs if (user_dir / "labels.txt").is_file())
    expected_counts = (
        int(selection["expected_user_directories"]),
        int(selection["expected_plt_files"]),
        int(selection["expected_label_files"]),
    )
    observed_counts = (len(all_user_dirs), total_plt_files, total_label_files)
    if observed_counts != expected_counts:
        raise RuntimeError(
            f"GeoLife source count mismatch: expected {expected_counts}, "
            f"found {observed_counts}."
        )

    requested = set(args.user_id)
    if requested:
        available = {int(path.name) for path in all_user_dirs}
        missing = sorted(requested - available)
        if missing:
            raise RuntimeError(f"Requested GeoLife users do not exist: {missing}")
        scanned_user_dirs = [
            path for path in all_user_dirs if int(path.name) in requested
        ]
        scan_scope = "requested_users"
    else:
        scanned_user_dirs = all_user_dirs
        scan_scope = "complete_release"

    audits: list[dict[str, object]] = []
    for user_dir in scanned_user_dirs:
        user_id = int(user_dir.name)
        intervals, starts = load_labels(user_dir)
        for plt_path in sorted((user_dir / "Trajectory").glob("*.plt")):
            audits.append(
                inspect_trajectory(
                    plt_path,
                    raw_root,
                    user_id,
                    intervals,
                    starts,
                    selection,
                )
            )
    if not audits:
        raise RuntimeError("No GeoLife PLT files were inspected.")

    eligible = [row for row in audits if row["status"] == "eligible"]
    eligible_counts = Counter(int(row["user_id"]) for row in eligible)
    minimum = int(selection["min_eligible_trajectories_per_user"])
    selected_users = sorted(
        user_id
        for user_id, count in eligible_counts.items()
        if count >= minimum
    )
    if requested:
        insufficient = sorted(requested - set(selected_users))
        if insufficient:
            raise RuntimeError(
                f"Requested users fail the unchanged minimum-history rule: "
                f"{insufficient}"
            )
    selected_user_set = set(selected_users)
    selected = [
        row for row in eligible if int(row["user_id"]) in selected_user_set
    ]
    if not selected:
        raise RuntimeError("No trajectories remain after the unchanged filters.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    private_dir = output_dir / "private"
    private_dir.mkdir(parents=True, exist_ok=True)
    audit_path = private_dir / "source_trajectory_audit.csv"
    source_csv = private_dir / "geolife_source.csv.gz"
    write_audit(audit_path, audits)
    point_rows = write_standardized_points(source_csv, selected, raw_root)

    reason_counts = Counter(str(row["reason"]) for row in audits)
    mode_counts = Counter(str(row["dominant_mode"]) for row in selected)
    selected_counts = Counter(int(row["user_id"]) for row in selected)
    summary = {
        "status": "completed",
        "scan_scope": scan_scope,
        "raw_root_recorded_as": raw_root.name,
        "source_release": str(selection["required_release"]),
        "source_users_total": len(all_user_dirs),
        "source_plt_files_total": total_plt_files,
        "source_label_files_total": total_label_files,
        "inspected_users": len(scanned_user_dirs),
        "inspected_trajectories": len(audits),
        "eligible_trajectories_before_user_selection": len(eligible),
        "minimum_eligible_trajectories_per_user": minimum,
        "selected_user_count": len(selected_users),
        "selected_users": selected_users,
        "selected_trajectories": len(selected),
        "selected_point_rows": point_rows,
        "selected_trip_counts_by_user": {
            str(key): int(value) for key, value in sorted(selected_counts.items())
        },
        "dominant_mode_counts_selected": {
            str(key): int(value) for key, value in sorted(mode_counts.items())
        },
        "exclusion_reasons_in_scan": {
            str(key): int(value) for key, value in sorted(reason_counts.items())
        },
        "quality_audit_selected": {
            "trajectories_with_nonpositive_time_steps": sum(
                int(row["duplicate_or_nonpositive_time_steps"]) > 0
                for row in selected
            ),
            "trajectories_with_gap_over_1h": sum(
                float(row["max_recording_gap_s"]) > 3600.0 for row in selected
            ),
            "trajectories_with_implied_speed_over_100mps": sum(
                float(row["max_implied_speed_mps"]) > 100.0 for row in selected
            ),
        },
        "trajectory_unit": "one original PLT file",
        "manual_segmentation": False,
        "trajectory_subsampling": False,
        "point_downsampling": False,
        "row_level_mode_filtering": False,
        "complete_trajectory_mode_filtering": True,
        "large_gap_filtering": False,
        "high_implied_speed_filtering": False,
        "preserve_all_valid_points": True,
        "private_standardized_csv_sha256": sha256(source_csv),
        "private_audit_sha256": sha256(audit_path),
    }
    summary_path = output_dir / "preparation_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
