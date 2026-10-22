#!/usr/bin/env python3
"""Validate final admissible-system outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["dataset", "driver_id", "split_label", "trip_id"]
ARMS = {"combined_1000m": 1000.0, "combined_500m": 500.0, "combined_350m": 350.0}
METHODS = {
    "destination_only",
    "route_analogue",
    "transition_grid_route",
    "most_frequent_destination",
    "coherent_product",
    "coherent_abstain",
}
EXPECTED_CASES = 8526


def require(condition: bool, message: object) -> None:
    if not condition:
        raise AssertionError(message)


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    require(column in frame.columns, f"Missing column: {column}")
    return pd.to_numeric(frame[column], errors="coerce")


def validate_case_matrix(rows: pd.DataFrame, expected_cases: int = EXPECTED_CASES) -> int:
    required = set(KEYS) | {"arm", "method", "emitted"}
    missing = sorted(required - set(rows.columns))
    require(not missing, f"Missing columns: {missing}")
    require(not rows[KEYS + ["arm", "method"]].isna().any().any(), "Null case keys")

    n_cases = rows[KEYS].drop_duplicates().shape[0]
    require(n_cases == expected_cases, f"Expected {expected_cases} cases, found {n_cases}")
    require(set(rows["arm"].astype(str).unique()) == set(ARMS), "Unexpected arm set")
    require(set(rows["method"].astype(str).unique()) == METHODS, "Unexpected method set")

    combination_keys = KEYS + ["arm", "method"]
    duplicate = rows.duplicated(combination_keys, keep=False)
    require(not duplicate.any(), f"Duplicate case-arm-method rows: {int(duplicate.sum())}")

    expected_per_case = len(ARMS) * len(METHODS)
    require(
        len(rows) == n_cases * expected_per_case,
        f"Expected {n_cases * expected_per_case} rows, found {len(rows)}",
    )
    per_case = rows.groupby(KEYS, dropna=False).size()
    require(
        per_case.eq(expected_per_case).all(),
        f"Incomplete case matrices: {int((~per_case.eq(expected_per_case)).sum())}",
    )

    emitted = numeric(rows, "emitted")
    require(np.isfinite(emitted).all(), "Non-finite emitted indicator")
    require(emitted.isin([0, 1]).all(), "Emitted indicator must be binary")
    return int(n_cases)


def validate_geometry(rows: pd.DataFrame) -> dict[str, int]:
    required = {
        "selected_route_start_gap_m",
        "selected_geometry_source",
        "selected_grid_reached_destination",
        "route_endpoint_to_destination_m",
    }
    missing = sorted(required - set(rows.columns))
    require(not missing, f"Missing geometry columns: {missing}")

    violations: dict[str, int] = {}
    emitted_indicator = numeric(rows, "emitted")
    for arm, threshold in ARMS.items():
        emitted = rows[rows["arm"].eq(arm) & emitted_indicator.eq(1)].copy()
        start_gap = numeric(emitted, "selected_route_start_gap_m")
        violations[f"{arm}_start_gap_nonfinite"] = int((~np.isfinite(start_gap)).sum())
        violations[f"{arm}_start_gap"] = int(
            (np.isfinite(start_gap) & (start_gap > threshold + 1e-6)).sum()
        )

        discrepancy = numeric(emitted, "route_endpoint_to_destination_m")
        violations[f"{arm}_endpoint_nonfinite"] = int((~np.isfinite(discrepancy)).sum())
        # The gate is evaluated in the split-specific metric plane. The output
        # discrepancy is recomputed geodesically, so a 1 m numerical allowance
        # is used only for validation of the exported diagnostic.
        violations[f"{arm}_endpoint"] = int(
            (np.isfinite(discrepancy) & (discrepancy > 501.0)).sum()
        )

        grid = emitted[emitted["selected_geometry_source"].eq("grid_beam")]
        reached = numeric(grid, "selected_grid_reached_destination")
        violations[f"{arm}_grid_reach_nonfinite"] = int((~np.isfinite(reached)).sum())
        violations[f"{arm}_grid_reach"] = int(
            (np.isfinite(reached) & reached.ne(1)).sum()
        )

    require(not any(violations.values()), violations)
    return violations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--public-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = pd.read_csv(args.predictions)
    n_cases = validate_case_matrix(rows)
    violations = validate_geometry(rows)
    required = [
        "metrics_by_arm_method.csv",
        "paired_equal_user_bootstrap.csv",
        "confidence_operating_points.csv",
        "risk_coverage_curves.csv",
        "risk_coverage_aurc.csv",
        "matched_coverage_score_comparison.csv",
        "metrics_by_dataset_split_policy.csv",
        "historical_support_strata.csv",
        "leave_development_users_out_metrics.csv",
        "leave_development_users_out_bootstrap.csv",
        "temporal_sensitivity_metrics.csv",
        "risk_coverage_ead.pdf",
        "summary.json",
    ]
    for name in required:
        require((args.public_dir / name).is_file(), f"Missing public result: {name}")

    forbidden = {"lat", "lon", "latitude", "longitude", "x", "y"}
    public_csvs = sorted(args.public_dir.glob("*.csv"))
    for path in public_csvs:
        columns = {str(column).lower() for column in pd.read_csv(path, nrows=0).columns}
        require(columns.isdisjoint(forbidden), (path, columns & forbidden))

    result = {
        "status": "passed",
        "n_cases": int(n_cases),
        "n_rows": int(len(rows)),
        "violations": violations,
        "public_csv_count": len(public_csvs),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
