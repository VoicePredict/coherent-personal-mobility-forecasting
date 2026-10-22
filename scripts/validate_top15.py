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


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def equal_numeric(left: pd.Series, right: pd.Series, tolerance: float = 1e-8) -> bool:
    a = pd.to_numeric(left, errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(right, errors="coerce").to_numpy(dtype=float)
    return bool(np.all(np.isclose(a, b, rtol=0.0, atol=tolerance, equal_nan=True)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--final-predictions", type=Path, required=True)
    parser.add_argument("--public-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = pd.read_csv(args.predictions)
    final = pd.read_csv(args.final_predictions)
    n_cases = rows[KEYS].drop_duplicates().shape[0]
    assert n_cases == 8526
    assert len(rows) == n_cases * len(ARMS) * len(METHODS)
    assert set(rows["arm"].unique()) == set(ARMS)
    assert set(rows["method"].unique()) == METHODS

    violations: dict[str, int] = {}
    for arm, threshold in ARMS.items():
        top15 = rows[
            rows["arm"].eq(arm)
            & rows["method"].eq("route_analogue_top15_guarded")
        ].copy()
        emitted = top15[numeric(top15, "emitted").eq(1)]
        violations[f"{arm}_start_gap"] = int(
            (numeric(emitted, "selected_route_start_gap_m") > threshold + 1e-6).sum()
        )
        violations[f"{arm}_endpoint"] = int(
            (numeric(emitted, "selected_endpoint_discrepancy_m") > 500.0 + 1e-6).sum()
        )
        violations[f"{arm}_rank"] = int(
            (~numeric(emitted, "selected_analogue_rank").between(1, 15)).sum()
        )

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
        joined = a1.merge(old, on=KEYS, suffixes=("_new", "_old"), validate="one_to_one")
        assert len(joined) == n_cases
        for column in [
            "emitted",
            "product_success",
            "dest_within_1000m",
            "la_within_1000m",
            "route_step_accuracy",
            "route_trip_id",
            "selected_route_start_gap_m",
        ]:
            assert equal_numeric(joined[f"{column}_new"], joined[f"{column}_old"]), (
                arm,
                column,
            )

        a1_emit = numeric(joined, "emitted_new").eq(1)
        t15_indexed = top15.set_index(KEYS).loc[
            pd.MultiIndex.from_frame(joined.loc[a1_emit, KEYS])
        ]
        assert numeric(t15_indexed, "emitted").eq(1).all()
        assert numeric(t15_indexed, "selected_analogue_rank").eq(1).all()

    assert not any(violations.values()), violations

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
        assert (args.public_dir / name).is_file(), name

    forbidden = {"lat", "lon", "latitude", "longitude", "x", "y"}
    public_csvs = sorted(args.public_dir.glob("*.csv"))
    for path in public_csvs:
        columns = {str(column).lower() for column in pd.read_csv(path, nrows=0).columns}
        assert columns.isdisjoint(forbidden), (path, columns & forbidden)

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
