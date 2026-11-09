#!/usr/bin/env python3
"""Validate the journal-review control outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["dataset", "driver_id", "split_label", "trip_id", "variant"]
VARIANTS = {"full", "fixed_family_order", "top15_retrieval"}


def require_finite(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite values in {label}: {columns}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--public-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions, low_memory=False)
    if set(predictions["variant"].unique()) != VARIANTS:
        raise ValueError("Missing or unexpected selection variants")
    if predictions.duplicated(KEYS).any():
        raise ValueError("Duplicate case x variant rows")
    case_counts = predictions.groupby(KEYS[:-1])["variant"].nunique()
    if len(case_counts) != 8526 or not case_counts.eq(len(VARIANTS)).all():
        raise ValueError("Incomplete variant coverage over the 8,526 cases")

    emitted = predictions.loc[predictions["emitted"].eq(1)]
    require_finite(
        emitted,
        [
            "destination_error_m",
            "lookahead_error_m",
            "route_step_accuracy",
            "route_endpoint_to_destination_m",
        ],
        "emitted predictions",
    )
    same_pool = emitted.loc[
        emitted["variant"].isin(["full", "fixed_family_order"])
    ]
    require_finite(same_pool, ["selected_route_start_gap_m"], "same-pool forecasts")
    if (emitted["route_endpoint_to_destination_m"] > 500.0 + 1e-6).any():
        raise ValueError("Endpoint guard violation")
    if (same_pool["selected_route_start_gap_m"] > 1000.0 + 1e-6).any():
        raise ValueError("Start-gap guard violation")

    policy = pd.read_csv(args.public_dir / "policy_metrics.csv")
    bootstrap = pd.read_csv(args.public_dir / "paired_equal_user_bootstrap.csv")
    temporal = pd.read_csv(args.public_dir / "short_horizon_observed_target.csv")
    require_finite(policy, [c for c in policy if c.endswith("_pct")], "policy metrics")
    require_finite(
        bootstrap,
        [
            "reference_equal_user_pct",
            "comparator_equal_user_pct",
            "delta_reference_minus_comparator_pp",
            "ci95_low_pp",
            "ci95_high_pp",
        ],
        "bootstrap metrics",
    )
    require_finite(
        temporal,
        [c for c in temporal if c.endswith("_pct")],
        "short-horizon metrics",
    )

    result = {
        "passed": True,
        "cases": int(len(case_counts)),
        "prediction_rows": int(len(predictions)),
        "variants": sorted(VARIANTS),
        "public_files": [
            "paired_equal_user_bootstrap.csv",
            "policy_metrics.csv",
            "short_horizon_observed_target.csv",
            "summary.json",
        ],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
