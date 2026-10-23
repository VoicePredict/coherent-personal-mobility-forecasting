#!/usr/bin/env python3
"""Validate private case rows and public aggregates for the diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["dataset", "driver_id", "split_label", "trip_id"]
VARIANTS = ["full", "no_grid", "no_soft_coherence", "equal_weights", "top15_retrieval"]
FINITE_COLUMNS = [
    "emitted", "product_success", "dest_within_1000m",
    "la_within_1000m",
]


def validate_case_matrix(rows: pd.DataFrame, *, expected_cases: int = 8526) -> int:
    case_count = int(rows[KEYS].drop_duplicates().shape[0])
    if case_count != expected_cases:
        raise AssertionError(f"case count is {case_count:,}, expected {expected_cases:,}")
    expected_rows = expected_cases * len(VARIANTS)
    if len(rows) != expected_rows:
        raise AssertionError(f"row count is {len(rows):,}, expected {expected_rows:,}")
    duplicates = int(rows.duplicated(KEYS + ["variant"]).sum())
    if duplicates:
        raise AssertionError(f"duplicate case × variant rows: {duplicates}")
    if set(rows["variant"].dropna().unique()) != set(VARIANTS):
        raise AssertionError("variant set is incomplete or unexpected")
    counts = rows.groupby(KEYS)["variant"].nunique()
    if not counts.eq(len(VARIANTS)).all():
        raise AssertionError("one or more cases do not contain every variant")
    return case_count


def validate_finite_values(rows: pd.DataFrame) -> None:
    for column in FINITE_COLUMNS:
        values = pd.to_numeric(rows[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise AssertionError(f"non-finite values in {column}")
    emitted = pd.to_numeric(rows["emitted"], errors="coerce").eq(1)
    for column in [
        "destination_error_m", "lookahead_error_m",
        "route_endpoint_to_destination_m", "route_step_accuracy",
    ]:
        values = pd.to_numeric(rows.loc[emitted, column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise AssertionError(f"non-finite emitted geometry in {column}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--temporal-audit", type=Path, required=True)
    parser.add_argument("--public-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = pd.read_csv(args.predictions)
    temporal = pd.read_csv(args.temporal_audit)
    failures: list[str] = []

    try:
        validate_case_matrix(rows)
    except AssertionError as error:
        failures.append(str(error))
    try:
        validate_finite_values(rows)
    except AssertionError as error:
        failures.append(str(error))
    if temporal[KEYS].drop_duplicates().shape[0] != 8526 or len(temporal) != 8526:
        failures.append("temporal audit does not contain one row for every case")

    required_public = [
        "variant_metrics.csv", "paired_equal_user_bootstrap.csv",
        "variant_metrics_by_dataset_split.csv",
        "emission_overlap.csv", "common_emission_metrics.csv",
        "selected_candidate_families.csv", "horizon_endpoint_use.csv",
        "evaluation_region_radii.csv", "temporal_availability_summary.csv",
        "summary.json",
    ]
    missing = [name for name in required_public if not (args.public_dir / name).is_file()]
    if missing:
        failures.append(f"missing public outputs: {missing}")
    for name in required_public:
        path = args.public_dir / name
        if path.suffix == ".csv" and path.is_file():
            frame = pd.read_csv(path)
            numeric_frame = frame.select_dtypes(include=[np.number])
            if not np.isfinite(numeric_frame.to_numpy(dtype=float)).all():
                failures.append(f"non-finite numeric aggregate in {name}")

    report = {
        "status": "passed" if not failures else "failed",
        "n_cases": int(rows[KEYS].drop_duplicates().shape[0]),
        "n_rows": int(len(rows)),
        "variants": sorted(rows["variant"].dropna().unique().tolist()),
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise SystemExit("; ".join(failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
