#!/usr/bin/env python3
"""Create aggregate reports for candidate-geometry admissibility arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


KEYS = ["dataset", "driver_id", "split_label", "trip_id"]
SUCCESS = {
    "ead": "product_success",
    "destination_1km": "dest_within_1000m",
    "short_horizon_1km": "la_within_1000m",
    "route_score": "route_step_accuracy",
}
THRESHOLDS = [0.0, 0.48, 0.60, 0.70, 0.75, 0.80]


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def no_connector(frame: pd.DataFrame) -> pd.Series:
    source = frame["selected_geometry_source"].astype(str)
    reached = numeric(frame, "selected_grid_reached_destination").eq(1)
    return ~source.eq("grid_beam") | reached


def metric_record(frame: pd.DataFrame, arm: str, method: str) -> dict[str, Any]:
    emitted = numeric(frame, "emitted").fillna(0).eq(1)
    accepted = frame[emitted]
    record: dict[str, Any] = {
        "arm": arm,
        "method": method,
        "n_cases": int(len(frame)),
        "n_emitted": int(emitted.sum()),
        "coverage_pct": 100.0 * float(emitted.mean()),
    }
    for label, column in SUCCESS.items():
        accepted_value = numeric(accepted, column).mean() if len(accepted) else np.nan
        all_value = (numeric(frame, column).fillna(0.0) * emitted.astype(float)).mean()
        record[f"accepted_{label}_pct"] = 100.0 * float(accepted_value)
        record[f"all_case_{label}_pct"] = 100.0 * float(all_value)
    record["accepted_conflict_gt500_pct"] = (
        100.0 * float(numeric(accepted, "route_destination_conflict_gt500").mean())
        if len(accepted)
        else np.nan
    )
    record["accepted_start_gap_gt1000_pct"] = (
        100.0 * float((numeric(accepted, "selected_route_start_gap_m") > 1000.0).mean())
        if len(accepted)
        else np.nan
    )
    record["accepted_synthetic_connector_pct"] = (
        100.0 * float((numeric(accepted, "selected_synthetic_connector_m") > 0.0).mean())
        if len(accepted)
        else np.nan
    )
    record["accepted_geometry_valid_1000m_pct"] = (
        100.0
        * float(
            (
                no_connector(accepted)
                & numeric(accepted, "selected_route_start_gap_m").le(1000.0)
            ).mean()
        )
        if len(accepted)
        else np.nan
    )
    return record


def summarize_metrics(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (arm, method), frame in rows.groupby(["arm", "method"], sort=False):
        records.append(metric_record(frame, str(arm), str(method)))
    return pd.DataFrame(records)


def summarize_reselection(rows: pd.DataFrame) -> pd.DataFrame:
    product = rows[rows["method"].eq("coherent_product")].copy()
    records: list[dict[str, Any]] = []
    for arm, frame in product.groupby("arm", sort=False):
        emitted = numeric(frame, "emitted").fillna(0).eq(1)
        original_valid = numeric(frame, "original_coherent_candidate_valid").eq(1)
        changed = numeric(frame, "selection_changed").eq(1)
        rescued = ~original_valid & emitted
        records.append(
            {
                "arm": arm,
                "n_cases": int(len(frame)),
                "original_selected_invalid_n": int((~original_valid).sum()),
                "original_selected_invalid_pct": 100.0 * float((~original_valid).mean()),
                "reselected_to_valid_alternative_n": int(rescued.sum()),
                "reselected_to_valid_alternative_pct_all": 100.0 * float(rescued.mean()),
                "rescue_pct_among_original_invalid": (
                    100.0 * float(rescued.sum() / (~original_valid).sum())
                    if (~original_valid).sum()
                    else np.nan
                ),
                "abstained_after_filter_n": int((~emitted).sum()),
                "coverage_after_filter_pct": 100.0 * float(emitted.mean()),
                "selection_changed_n": int(changed.sum()),
                "selection_changed_pct": 100.0 * float(changed.mean()),
            }
        )
    return pd.DataFrame(records)


def confidence_geometry(rows: pd.DataFrame) -> pd.DataFrame:
    product = rows[rows["method"].eq("coherent_product")].copy()
    records: list[dict[str, Any]] = []
    for arm, frame in product.groupby("arm", sort=False):
        model_emitted = numeric(frame, "emitted").fillna(0).eq(1)
        score = numeric(frame, "candidate_confidence").fillna(-np.inf)
        for threshold in THRESHOLDS:
            accepted_mask = model_emitted & score.ge(threshold)
            accepted = frame[accepted_mask]
            record: dict[str, Any] = {
                "arm": arm,
                "confidence_threshold": threshold,
                "n_cases": int(len(frame)),
                "n_emitted": int(accepted_mask.sum()),
                "coverage_pct": 100.0 * float(accepted_mask.mean()),
            }
            for label, column in SUCCESS.items():
                record[f"accepted_{label}_pct"] = (
                    100.0 * float(numeric(accepted, column).mean())
                    if len(accepted)
                    else np.nan
                )
            record["synthetic_connector_pct"] = (
                100.0 * float((numeric(accepted, "selected_synthetic_connector_m") > 0.0).mean())
                if len(accepted)
                else np.nan
            )
            record["start_gap_gt1000_pct"] = (
                100.0 * float((numeric(accepted, "selected_route_start_gap_m") > 1000.0).mean())
                if len(accepted)
                else np.nan
            )
            record["geometry_availability_1000m_pct"] = (
                100.0
                * float(
                    (
                        no_connector(accepted)
                        & numeric(accepted, "selected_route_start_gap_m").le(1000.0)
                    ).mean()
                )
                if len(accepted)
                else np.nan
            )
            records.append(record)
    return pd.DataFrame(records)


def family_counts(rows: pd.DataFrame) -> pd.DataFrame:
    frame = rows[rows["method"].eq("coherent_product")].copy()
    frame["reported_family"] = frame["candidate_family"].where(
        numeric(frame, "emitted").eq(1),
        "abstain",
    )
    counts = (
        frame.groupby(["arm", "reported_family"], dropna=False)
        .size()
        .rename("n")
        .reset_index()
    )
    totals = counts.groupby("arm")["n"].transform("sum")
    counts["pct"] = 100.0 * counts["n"] / totals
    return counts


def prepare_user_outcomes(rows: pd.DataFrame, arm: str, method: str, cohort: str) -> pd.DataFrame:
    frame = rows[rows["arm"].eq(arm) & rows["method"].eq(method)].copy()
    if cohort == "geolife":
        frame = frame[frame["dataset"].eq("geolife_ground")].copy()
    frame["user_key"] = frame["dataset"].astype(str) + ":" + frame["driver_id"].astype(str)
    emitted = numeric(frame, "emitted").fillna(0.0)
    outcome = pd.DataFrame({"user_key": frame["user_key"], "coverage": emitted})
    for label, column in SUCCESS.items():
        outcome[label] = emitted * numeric(frame, column).fillna(0.0)
    return outcome.groupby("user_key", as_index=False).mean(numeric_only=True)


def bootstrap_difference(
    rows: pd.DataFrame,
    *,
    cohort: str,
    arm_a: str,
    method_a: str,
    arm_b: str,
    method_b: str,
    draws: int,
    seed: int,
) -> list[dict[str, Any]]:
    a = prepare_user_outcomes(rows, arm_a, method_a, cohort)
    b = prepare_user_outcomes(rows, arm_b, method_b, cohort)
    merged = a.merge(b, on="user_key", suffixes=("_a", "_b"), validate="one_to_one")
    if len(merged) != len(a) or len(merged) != len(b):
        raise ValueError(f"User mismatch for {arm_a}/{method_a} versus {arm_b}/{method_b}")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(merged), size=(draws, len(merged)))
    records: list[dict[str, Any]] = []
    for metric in ["coverage", *SUCCESS.keys()]:
        differences = (
            numeric(merged, f"{metric}_a").to_numpy(dtype=float)
            - numeric(merged, f"{metric}_b").to_numpy(dtype=float)
        )
        estimates = differences[indices].mean(axis=1) * 100.0
        records.append(
            {
                "cohort": cohort,
                "arm_a": arm_a,
                "method_a": method_a,
                "arm_b": arm_b,
                "method_b": method_b,
                "metric": metric,
                "n_users": int(len(merged)),
                "difference_pp": 100.0 * float(differences.mean()),
                "ci95_low_pp": float(np.quantile(estimates, 0.025)),
                "ci95_high_pp": float(np.quantile(estimates, 0.975)),
                "bootstrap_draws": draws,
                "seed": seed,
            }
        )
    return records


def bootstrap_reports(rows: pd.DataFrame, draws: int, seed: int) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    guarded_arms = ["combined_1000m", "combined_500m", "combined_350m"]
    baselines = [
        "destination_only",
        "route_analogue",
        "transition_grid_route",
        "most_frequent_destination",
    ]
    for cohort in ["all", "geolife"]:
        for arm in guarded_arms:
            records.extend(
                bootstrap_difference(
                    rows,
                    cohort=cohort,
                    arm_a=arm,
                    method_a="coherent_product",
                    arm_b="unfiltered_reference",
                    method_b="coherent_product",
                    draws=draws,
                    seed=seed,
                )
            )
            for baseline in baselines:
                records.extend(
                    bootstrap_difference(
                        rows,
                        cohort=cohort,
                        arm_a=arm,
                        method_a="coherent_product",
                        arm_b=arm,
                        method_b=baseline,
                        draws=draws,
                        seed=seed,
                    )
                )
    return pd.DataFrame(records)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261020)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(args.predictions)

    metrics = summarize_metrics(rows)
    reselection = summarize_reselection(rows)
    risk_geometry = confidence_geometry(rows)
    families = family_counts(rows)
    bootstrap = bootstrap_reports(rows, args.bootstrap_draws, args.seed)
    metrics.to_csv(args.output_dir / "metrics_by_arm_method.csv", index=False)
    reselection.to_csv(args.output_dir / "coherent_reselection_summary.csv", index=False)
    risk_geometry.to_csv(args.output_dir / "confidence_geometry_operating_points.csv", index=False)
    families.to_csv(args.output_dir / "coherent_candidate_families.csv", index=False)
    bootstrap.to_csv(args.output_dir / "paired_equal_user_bootstrap.csv", index=False)

    report = {
        "n_cases": int(rows[KEYS].drop_duplicates().shape[0]),
        "n_rows": int(len(rows)),
        "arms": sorted(rows["arm"].unique().tolist()),
        "methods": sorted(rows["method"].unique().tolist()),
        "bootstrap_draws": args.bootstrap_draws,
        "bootstrap_seed": args.seed,
        "interpretation": {
            "accepted_metrics": "Computed only among emitted joint forecasts.",
            "all_case_metrics": "Abstentions count as unsuccessful forecasts; all cases remain in the denominator.",
            "bootstrap": "Individuals, not cases, are resampled with replacement after per-individual aggregation.",
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(metrics.round(3).to_string(index=False), flush=True)
    print(reselection.round(3).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
