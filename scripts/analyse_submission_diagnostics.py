#!/usr/bin/env python3
"""Aggregate the fixed submission diagnostics without exposing coordinates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


KEYS = ["dataset", "driver_id", "split_label", "trip_id"]
VARIANTS = ["full", "no_grid", "no_soft_coherence", "equal_weights", "top15_retrieval"]
OUTCOMES = {
    "ead": "product_success",
    "destination_1km": "dest_within_1000m",
    "short_horizon_1km": "la_within_1000m",
    "route_score": "route_step_accuracy",
}


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def user_outcomes(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["user_key"] = work["dataset"].astype(str) + ":" + work["driver_id"].astype(str)
    emitted = numeric(work, "emitted").fillna(0.0)
    out = pd.DataFrame({"user_key": work["user_key"], "coverage": emitted})
    for label, column in OUTCOMES.items():
        out[label] = emitted * numeric(work, column).fillna(0.0)
    return out.groupby("user_key", as_index=False).mean(numeric_only=True)


def summary(frame: pd.DataFrame) -> dict[str, float | int]:
    emitted = numeric(frame, "emitted").fillna(0).eq(1)
    accepted = frame[emitted]
    record: dict[str, float | int] = {
        "n_cases": int(len(frame)),
        "n_emitted": int(emitted.sum()),
        "coverage_pct": 100.0 * float(emitted.mean()),
    }
    for label, column in OUTCOMES.items():
        values = numeric(frame, column).fillna(0.0)
        record[f"all_case_{label}_pct"] = 100.0 * float((values * emitted.astype(float)).mean())
        record[f"accepted_{label}_pct"] = (
            100.0 * float(numeric(accepted, column).mean()) if len(accepted) else float("nan")
        )
    return record


def bootstrap_difference(
    a: pd.DataFrame,
    b: pd.DataFrame,
    *,
    draws: int,
    seed: int,
) -> list[dict[str, Any]]:
    ua = user_outcomes(a)
    ub = user_outcomes(b)
    merged = ua.merge(ub, on="user_key", suffixes=("_a", "_b"), validate="one_to_one")
    if len(merged) != len(ua) or len(merged) != len(ub):
        raise ValueError("Compared variants do not contain identical user sets")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(merged), size=(draws, len(merged)))
    rows: list[dict[str, Any]] = []
    for metric in ["coverage", *OUTCOMES]:
        differences = (
            numeric(merged, f"{metric}_a").to_numpy(dtype=float)
            - numeric(merged, f"{metric}_b").to_numpy(dtype=float)
        )
        sampled = differences[indices].mean(axis=1) * 100.0
        rows.append({
            "metric": metric,
            "n_users": len(merged),
            "difference_pp": 100.0 * float(differences.mean()),
            "ci95_low_pp": float(np.quantile(sampled, 0.025)),
            "ci95_high_pp": float(np.quantile(sampled, 0.975)),
            "bootstrap_draws": draws,
            "seed": seed,
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--temporal-audit", type=Path, required=True)
    parser.add_argument("--region-radii", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261020)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = pd.read_csv(args.predictions)
    temporal = pd.read_csv(args.temporal_audit)
    radii = pd.read_csv(args.region_radii)
    selective = rows[rows["variant"].eq("full")].copy()
    selective["variant"] = "full_q048"
    selective["method"] = "full_q048"
    selective["emitted"] = (
        numeric(selective, "emitted").eq(1)
        & numeric(selective, "candidate_confidence").ge(0.48)
    ).astype(int)
    reporting_rows = pd.concat([rows, selective], ignore_index=True, sort=False)
    reporting_variants = [*VARIANTS, "full_q048"]
    records: list[dict[str, Any]] = []
    for cohort in ["all", "geolife", "gta"]:
        cohort_rows = reporting_rows
        if cohort == "geolife":
            cohort_rows = reporting_rows[reporting_rows["dataset"].eq("geolife_ground")]
        elif cohort == "gta":
            cohort_rows = reporting_rows[~reporting_rows["dataset"].eq("geolife_ground")]
        for variant in reporting_variants:
            frame = cohort_rows[cohort_rows["variant"].eq(variant)]
            record = {"cohort": cohort, "variant": variant}
            record.update(summary(frame))
            equal_user = user_outcomes(frame)
            for metric in ["coverage", *OUTCOMES]:
                record[f"equal_user_all_case_{metric}_pct"] = 100.0 * float(equal_user[metric].mean())
            records.append(record)
    pd.DataFrame(records).to_csv(args.output_dir / "variant_metrics.csv", index=False)

    split_records: list[dict[str, Any]] = []
    for (dataset, split_label, variant), frame in reporting_rows.groupby(
        ["dataset", "split_label", "variant"], sort=False
    ):
        record = {"dataset": dataset, "split_label": split_label, "variant": variant}
        record.update(summary(frame))
        split_records.append(record)
    pd.DataFrame(split_records).to_csv(
        args.output_dir / "variant_metrics_by_dataset_split.csv", index=False
    )

    bootstrap_rows: list[dict[str, Any]] = []
    for cohort in ["all", "geolife"]:
        cohort_rows = rows if cohort == "all" else rows[rows["dataset"].eq("geolife_ground")]
        full = cohort_rows[cohort_rows["variant"].eq("full")]
        for comparator in ["no_grid", "no_soft_coherence", "equal_weights", "top15_retrieval"]:
            comparison = bootstrap_difference(
                full,
                cohort_rows[cohort_rows["variant"].eq(comparator)],
                draws=args.bootstrap_draws,
                seed=args.seed,
            )
            for record in comparison:
                record.update({"cohort": cohort, "variant_a": "full", "variant_b": comparator})
            bootstrap_rows.extend(comparison)
    pd.DataFrame(bootstrap_rows).to_csv(
        args.output_dir / "paired_equal_user_bootstrap.csv", index=False
    )

    full = rows[rows["variant"].eq("full")].set_index(KEYS)
    retrieval = rows[rows["variant"].eq("top15_retrieval")].set_index(KEYS)
    joined = full.add_suffix("_full").join(retrieval.add_suffix("_retrieval"), how="inner")
    full_emit = numeric(joined, "emitted_full").eq(1)
    retrieval_emit = numeric(joined, "emitted_retrieval").eq(1)
    group = np.select(
        [full_emit & retrieval_emit, full_emit & ~retrieval_emit, ~full_emit & retrieval_emit],
        ["both_emit", "full_only", "retrieval_only"],
        default="both_abstain",
    )
    joined["emission_group"] = group
    overlap = (
        joined.reset_index()
        .groupby(["dataset", "emission_group"], dropna=False)
        .size().rename("n").reset_index()
    )
    overlap["pct_within_dataset"] = 100.0 * overlap["n"] / overlap.groupby("dataset")["n"].transform("sum")
    overlap.to_csv(args.output_dir / "emission_overlap.csv", index=False)

    common = joined[full_emit & retrieval_emit].reset_index()
    common_records: list[dict[str, Any]] = []
    for cohort in ["all", "geolife", "gta"]:
        frame = common
        if cohort == "geolife":
            frame = frame[frame["dataset"].eq("geolife_ground")]
        elif cohort == "gta":
            frame = frame[~frame["dataset"].eq("geolife_ground")]
        for method_suffix in ["full", "retrieval"]:
            frame_copy = frame.copy()
            frame_copy["emitted"] = 1
            for column in OUTCOMES.values():
                frame_copy[column] = frame_copy[f"{column}_{method_suffix}"]
            record = {"cohort": cohort, "common_emission_method": method_suffix}
            record.update(summary(frame_copy))
            user_values = user_outcomes(frame_copy)
            for metric in OUTCOMES:
                record[f"equal_user_{metric}_pct"] = 100.0 * float(user_values[metric].mean())
            common_records.append(record)
    pd.DataFrame(common_records).to_csv(
        args.output_dir / "common_emission_metrics.csv", index=False
    )

    full_rows = rows[rows["variant"].eq("full")].copy()
    full_rows["selected_family"] = full_rows["candidate_family"].where(
        numeric(full_rows, "emitted").eq(1), "abstain"
    )
    family = (
        full_rows.groupby(["dataset", "selected_family"], dropna=False)
        .size().rename("n").reset_index()
    )
    family["pct_within_dataset"] = 100.0 * family["n"] / family.groupby("dataset")["n"].transform("sum")
    family.to_csv(args.output_dir / "selected_candidate_families.csv", index=False)

    horizon_records: list[dict[str, Any]] = []
    for cohort, frame in [
        ("all", full_rows),
        ("geolife", full_rows[full_rows["dataset"].eq("geolife_ground")]),
        ("gta", full_rows[~full_rows["dataset"].eq("geolife_ground")]),
    ]:
        flag = numeric(frame, "horizon_beyond_recording_end").eq(1)
        horizon_records.append({
            "cohort": cohort,
            "n_cases": len(frame),
            "horizon_beyond_recording_end_n": int(flag.sum()),
            "horizon_beyond_recording_end_pct": 100.0 * float(flag.mean()),
        })
    pd.DataFrame(horizon_records).to_csv(args.output_dir / "horizon_endpoint_use.csv", index=False)

    radius_records: list[dict[str, Any]] = []
    for cohort, frame in [
        ("all", radii),
        ("geolife", radii[radii["dataset"].eq("geolife_ground")]),
        ("gta", radii[~radii["dataset"].eq("geolife_ground")]),
    ]:
        values = numeric(frame, "radius_p90_m").dropna().to_numpy(dtype=float)
        radius_records.append({
            "cohort": cohort,
            "n_regions": len(values),
            "median_radius_m": float(np.median(values)),
            "p90_radius_m": float(np.quantile(values, 0.9)),
            "max_radius_m": float(np.max(values)),
        })
    pd.DataFrame(radius_records).to_csv(args.output_dir / "evaluation_region_radii.csv", index=False)

    temporal_records: list[dict[str, Any]] = []
    for dataset, frame in temporal[temporal["is_chronological"].eq(1)].groupby("dataset"):
        violated = numeric(frame, "temporal_availability_violation").eq(1)
        temporal_records.append({
            "dataset": dataset,
            "chronological_cases_checked": len(frame),
            "violating_cases": int(violated.sum()),
            "violating_cases_pct": 100.0 * float(violated.mean()),
            "unavailable_training_records": int(numeric(frame, "unavailable_train_trip_count").sum()),
        })
    pd.DataFrame(temporal_records).to_csv(
        args.output_dir / "temporal_availability_summary.csv", index=False
    )

    summary_json = {
        "status": "completed",
        "n_cases": int(rows[KEYS].drop_duplicates().shape[0]),
        "variants": VARIANTS,
        "bootstrap_unit": "individual",
        "bootstrap_draws": args.bootstrap_draws,
        "bootstrap_seed": args.seed,
        "selection_uses_test_outcomes": False,
        "chronological_history_rule": "training trip end time must not exceed the observed-prefix prediction time",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary_json, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
