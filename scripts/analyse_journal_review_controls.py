#!/usr/bin/env python3
"""Aggregate same-pool, chronological, and observed-target controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["dataset", "driver_id", "split_label", "trip_id"]
OUTCOMES = {
    "product_success": "ead",
    "dest_within_1000m": "destination_1km",
    "la_within_1000m": "short_horizon_1km",
    "route_step_accuracy": "route_score",
}


def emitted_metric(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    return values.where(frame["emitted"].eq(1), 0.0)


def metric_records(frame: pd.DataFrame, scope: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for variant, group in frame.groupby("variant", sort=False):
        emitted = group.loc[group["emitted"].eq(1)]
        record: dict[str, object] = {
            "scope": scope,
            "variant": variant,
            "n_cases": int(len(group)),
            "n_emitted": int(len(emitted)),
            "coverage_pct": 100.0 * float(group["emitted"].mean()),
        }
        for column, label in OUTCOMES.items():
            record[f"accepted_{label}_pct"] = (
                100.0 * float(pd.to_numeric(emitted[column], errors="coerce").mean())
                if len(emitted)
                else float("nan")
            )
            record[f"all_case_{label}_pct"] = 100.0 * float(
                emitted_metric(group, column).mean()
            )
        records.append(record)
    return records


def paired_user_bootstrap(
    frame: pd.DataFrame,
    *,
    reference: str,
    comparator: str,
    scope: str,
    draws: int,
    seed: int,
) -> list[dict[str, object]]:
    columns = KEYS + ["emitted"] + list(OUTCOMES)
    left = frame.loc[frame["variant"].eq(reference), columns].copy()
    right = frame.loc[frame["variant"].eq(comparator), columns].copy()
    paired = left.merge(
        right, on=KEYS, suffixes=("_reference", "_comparator"), validate="one_to_one"
    )
    if len(paired) != len(left) or len(paired) != len(right):
        raise RuntimeError(f"Incomplete pairing for {scope}: {reference} vs {comparator}")

    quantities = {"emitted": "coverage", **OUTCOMES}
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    for column, label in quantities.items():
        if column == "emitted":
            ref = pd.to_numeric(paired[f"{column}_reference"], errors="raise")
            cmp = pd.to_numeric(paired[f"{column}_comparator"], errors="raise")
        else:
            ref = pd.to_numeric(
                paired[f"{column}_reference"], errors="coerce"
            ).fillna(0.0)
            cmp = pd.to_numeric(
                paired[f"{column}_comparator"], errors="coerce"
            ).fillna(0.0)
            ref = ref.where(paired["emitted_reference"].eq(1), 0.0)
            cmp = cmp.where(paired["emitted_comparator"].eq(1), 0.0)
        per_user = pd.DataFrame(
            {
                "driver_id": paired["driver_id"].astype(int),
                "reference": ref,
                "comparator": cmp,
            }
        ).groupby("driver_id", sort=True).mean()
        differences = (
            per_user["reference"] - per_user["comparator"]
        ).to_numpy(dtype=float)
        samples = differences[
            rng.integers(0, len(differences), size=(draws, len(differences)))
        ].mean(axis=1)
        low, high = np.quantile(samples, [0.025, 0.975])
        records.append(
            {
                "scope": scope,
                "reference": reference,
                "comparator": comparator,
                "metric": label,
                "n_users": int(len(per_user)),
                "n_cases": int(len(paired)),
                "reference_equal_user_pct": 100.0 * float(per_user["reference"].mean()),
                "comparator_equal_user_pct": 100.0 * float(per_user["comparator"].mean()),
                "delta_reference_minus_comparator_pp": 100.0 * float(differences.mean()),
                "ci95_low_pp": 100.0 * float(low),
                "ci95_high_pp": 100.0 * float(high),
                "bootstrap_draws": draws,
                "seed": seed,
            }
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--geolife-temporal-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261020)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(args.predictions, low_memory=False)
    geolife = predictions.loc[predictions["dataset"].eq("geolife_ground")].copy()
    gta = predictions.loc[predictions["dataset"].eq("gta_ground")].copy()

    metrics: list[dict[str, object]] = []
    metrics.extend(metric_records(geolife, "geolife_all_splits"))
    metrics.extend(metric_records(gta, "gta_all_splits"))
    metrics.extend(
        metric_records(
            geolife.loc[geolife["split_label"].eq("chronological")],
            "geolife_chronological",
        )
    )
    pd.DataFrame(metrics).to_csv(args.output_dir / "policy_metrics.csv", index=False)

    bootstrap: list[dict[str, object]] = []
    bootstrap.extend(
        paired_user_bootstrap(
            geolife,
            reference="full",
            comparator="fixed_family_order",
            scope="geolife_all_splits",
            draws=args.bootstrap_draws,
            seed=args.seed,
        )
    )
    bootstrap.extend(
        paired_user_bootstrap(
            geolife.loc[geolife["split_label"].eq("chronological")],
            reference="full",
            comparator="top15_retrieval",
            scope="geolife_chronological",
            draws=args.bootstrap_draws,
            seed=args.seed,
        )
    )
    pd.DataFrame(bootstrap).to_csv(
        args.output_dir / "paired_equal_user_bootstrap.csv", index=False
    )

    audit = pd.read_csv(args.geolife_temporal_audit, low_memory=False)
    audit_keys = ["driver_id", "trip_id"]
    audit = audit[
        audit_keys + ["target_after_trip_end", "target_bracket_gap_over_1h"]
    ].copy()
    for column in audit_keys:
        audit[column] = pd.to_numeric(audit[column], errors="raise").astype(int)
        geolife[column] = pd.to_numeric(geolife[column], errors="raise").astype(int)
    geolife = geolife.merge(audit, on=audit_keys, how="left", validate="many_to_one")
    flags = ["target_after_trip_end", "target_bracket_gap_over_1h"]
    if geolife[flags].isna().any().any():
        raise RuntimeError("GeoLife temporal audit does not cover every evaluated case")

    temporal_records: list[dict[str, object]] = []
    scopes = {
        "geolife_target_strictly_before_endpoint": ~geolife["target_after_trip_end"].astype(bool),
        "geolife_target_before_endpoint_and_bracket_le_1h": (
            ~geolife["target_after_trip_end"].astype(bool)
            & ~geolife["target_bracket_gap_over_1h"].astype(bool)
        ),
    }
    for scope, mask in scopes.items():
        subset = geolife.loc[
            mask & geolife["variant"].isin(["full", "top15_retrieval"])
        ]
        temporal_records.extend(metric_records(subset, scope))
    gta_inside = gta.loc[
        gta["horizon_beyond_recording_end"].eq(0)
        & gta["variant"].isin(["full", "top15_retrieval"])
    ]
    temporal_records.extend(
        metric_records(gta_inside, "gta_target_not_beyond_endpoint")
    )
    pd.DataFrame(temporal_records).to_csv(
        args.output_dir / "short_horizon_observed_target.csv", index=False
    )

    summary = {
        "candidate_pool": "identical eligible candidates for full and fixed_family_order",
        "fixed_family_order": [
            "route_analogue by decreasing route evidence",
            "grid_transition by decreasing grid destination score",
            "destination region by decreasing destination score",
        ],
        "geolife_users": int(geolife["driver_id"].nunique()),
        "geolife_cases": int(geolife[KEYS].drop_duplicates().shape[0]),
        "gta_cases": int(gta[KEYS].drop_duplicates().shape[0]),
        "bootstrap_unit": "GeoLife user",
        "bootstrap_draws": args.bootstrap_draws,
        "bootstrap_seed": args.seed,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
