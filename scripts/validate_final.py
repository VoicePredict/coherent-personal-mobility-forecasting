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
METHODS = 6


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--public-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = pd.read_csv(args.predictions)
    n_cases = rows[KEYS].drop_duplicates().shape[0]
    assert n_cases == 8526
    assert len(rows) == n_cases * len(ARMS) * METHODS
    assert set(rows["arm"].unique()) == set(ARMS)

    violations: dict[str, int] = {}
    for arm, threshold in ARMS.items():
        emitted = rows[rows["arm"].eq(arm) & pd.to_numeric(rows["emitted"]).eq(1)].copy()
        start_gap = pd.to_numeric(emitted["selected_route_start_gap_m"], errors="coerce")
        violations[f"{arm}_start_gap"] = int((start_gap > threshold + 1e-6).sum())
        grid = emitted[emitted["selected_geometry_source"].eq("grid_beam")]
        reached = pd.to_numeric(grid["selected_grid_reached_destination"], errors="coerce")
        violations[f"{arm}_grid_reach"] = int((reached != 1).sum())
        discrepancy = pd.to_numeric(emitted["route_endpoint_to_destination_m"], errors="coerce")
        # The gate is evaluated in the split-specific metric plane. The output
        # discrepancy is recomputed geodesically, so a 1 m numerical allowance
        # is used only for validation of the exported diagnostic.
        violations[f"{arm}_endpoint"] = int((discrepancy > 501.0).sum())

    assert not any(violations.values()), violations
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
        "violations": violations,
        "public_csv_count": len(public_csvs),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
