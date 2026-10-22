#!/usr/bin/env python3
"""Validate the top-15 guarded analogue experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["dataset", "driver_id", "split_label", "trip_id"]
ARMS = {"combined_1000m": 1000.0, "combined_500m": 500.0, "combined_350m": 350.0}
METHODS = {"route_analogue_a1_guarded", "route_analogue_top15_guarded"}
EXPECTED_CASES = 8526


def require(condition: bool, message: object) -> None:
    if not condition:
        raise AssertionError(message)


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    require(column in frame.columns, f"Missing column: {column}")
    return pd.to_numeric(frame[column], errors="coerce")


def equal_numeric(left: pd.Series, right: pd.Series, tolerance: float = 1e-8) -> bool:
    a = pd.to_numeric(left, errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(right, errors="coerce").to_numpy(dtype=float)
    return bool(np.all(np.isclose(a, b, rtol=0.0, atol=tolerance, equal_nan=True)))


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
        "selected_endpoint_discrepancy_m",
        "selected_analogue_rank",
    }
    missing = sorted(required - set(rows.columns))
    require(not missing, f"Missing geometry columns: {missing}")

    violations: dict[str, int] = {}
    emitted_indicator = numeric(rows, "emitted")
    for arm, threshold in ARMS.items():
        for method in METHODS:
            accepted = rows[
                rows["arm"].eq(arm)
                & rows["method"].eq(method)
                & emitted_indicator.eq(1)
            ]
            prefix = f"{arm}_{method}"
            start_gap = numeric(accepted, "selected_route_start_gap_m")
            endpoint = numeric(accepted, "selected_endpoint_discrepancy_m")
            rank = numeric(accepted, "selected_analogue_rank")
            violations[f"{prefix}_start_gap_nonfinite"] = int((~np.isfinite(start_gap)).sum())
            violations[f"{prefix}_start_gap"] = int(
                (np.isfinite(start_gap) & (start_gap > threshold + 1e-6)).sum()
            )
            violations[f"{prefix}_endpoint_nonfinite"] = int((~np.isfinite(endpoint)).sum())
            violations[f"{prefix}_endpoint"] = int(
                (np.isfinite(endpoint) & (endpoint > 500.0 + 1e-6)).sum()
            )
            violations[f"{prefix}_rank_nonfinite"] = int((~np.isfinite(rank)).sum())
            upper = 1 if method == "route_analogue_a1_guarded" else 15
            violations[f"{prefix}_rank"] = int(
                (np.isfinite(rank) & ~rank.between(1, upper)).sum()
            )

    require(not any(violations.values()), violations)
    return violations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--final-predictions", type=Path, required=True)
    parser.add_argument("--public-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = pd.read_csv(args.predictions)
    final = pd.read_csv(args.final_predictions)
    n_cases = validate_case_matrix(rows)
    violations = validate_geometry(rows)
    for arm in ARMS:
        top15 = rows[
            rows["arm"].eq(arm)
            & rows["method"].eq("route_analogue_top15_guarded")
        ].copy()

        a1 = rows[
            rows["arm"].eq(arm)
            & rows["method"].eq("route_analogue_a1_guarded")
        ][KEYS + [
            "emitted",
            "product_success",
            "dest_within_1000m",
            "la_within_1000m",
            "route_step_accuracy",
            "route_trip_id",
            "selected_route_start_gap_m",
        ]].copy()
        old = final[
            final["arm"].eq(arm) & final["method"].eq("route_analogue")
        ][KEYS + [
            "emitted",
            "product_success",
            "dest_within_1000m",
            "la_within_1000m",
            "route_step_accuracy",
            "route_trip_id",
            "selected_route_start_gap_m",
        ]].copy()
        require(
            not a1.duplicated(KEYS, keep=False).any(),
            f"Duplicate a1 reference rows for {arm}",
        )
        require(
            not old.duplicated(KEYS, keep=False).any(),
            f"Duplicate final reference rows for {arm}",
        )
        joined = a1.merge(old, on=KEYS, suffixes=("_new", "_old"), validate="one_to_one")
        require(len(joined) == n_cases, f"Unpaired a1 reference rows for {arm}")
        for column in [
            "emitted",
            "product_success",
            "dest_within_1000m",
            "la_within_1000m",
            "route_step_accuracy",
            "route_trip_id",
            "selected_route_start_gap_m",
        ]:
            require(
                equal_numeric(joined[f"{column}_new"], joined[f"{column}_old"]),
                (arm, column),
            )

        a1_emit = numeric(joined, "emitted_new").eq(1)
        t15_indexed = top15.set_index(KEYS).loc[
            pd.MultiIndex.from_frame(joined.loc[a1_emit, KEYS])
        ]
        require(numeric(t15_indexed, "emitted").eq(1).all(), f"Top-15 lost a1 emissions: {arm}")
        require(
            numeric(t15_indexed, "selected_analogue_rank").eq(1).all(),
            f"Top-15 did not preserve admissible a1: {arm}",
        )

    required = [
        "metrics_by_arm_policy.csv",
        "metrics_by_dataset_split_arm_policy.csv",
        "paired_equal_user_bootstrap.csv",
        "leave_development_users_out_bootstrap.csv",
        "selected_analogue_rank_distribution.csv",
        "top15_backoff_summary.csv",
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
        "a1_reference_reproduced": True,
        "violations": violations,
        "public_csv_count": len(public_csvs),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
