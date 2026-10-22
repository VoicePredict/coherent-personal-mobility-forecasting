#!/usr/bin/env python3
"""Post-hoc development-overlap, timestamp, and endpoint-coherence audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEVELOPMENT_USERS = set(range(1, 14))
REFERENCE = "coherent_product"
COMPARATORS = [
    "destination_only",
    "route_analogue",
    "transition_grid_route",
    "most_frequent_destination",
]
METRICS = {
    "product_success": "EAD",
    "dest_within_1000m": "destination_within_1km",
    "la_within_1000m": "short_horizon_within_1km",
    "route_step_accuracy": "route_step_accuracy",
    "route_destination_conflict_gt500": "endpoint_conflict_gt500m",
}
KEYS = ["dataset", "driver_id", "split_label", "trip_id"]


def percentile(values: pd.Series, q: float) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    return float(np.quantile(clean, q)) if len(clean) else float("nan")


def aggregate_metrics(frame: pd.DataFrame, scope: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for method, part in frame.groupby("method", sort=True):
        record: dict[str, object] = {
            "scope": scope,
            "method": method,
            "n_cases": int(len(part)),
            "n_users": int(part["driver_id"].nunique()),
        }
        for column, label in METRICS.items():
            record[f"{label}_pct"] = 100.0 * float(
                pd.to_numeric(part[column], errors="coerce").mean()
            )
        records.append(record)
    return records


def paired_macro_bootstrap(
    frame: pd.DataFrame,
    *,
    reference: str,
    comparator: str,
    reps: int,
    seed: int,
) -> list[dict[str, object]]:
    columns = KEYS + list(METRICS)
    left = frame.loc[frame["method"].eq(reference), columns]
    right = frame.loc[frame["method"].eq(comparator), columns]
    paired = left.merge(right, on=KEYS, suffixes=("_reference", "_comparator"), validate="one_to_one")
    if len(paired) != len(left) or len(paired) != len(right):
        raise RuntimeError(f"Unpaired cases for {reference} and {comparator}")
    paired["user_key"] = paired["dataset"].astype(str) + ":" + paired["driver_id"].astype(str)
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    for column, label in METRICS.items():
        values = paired[["user_key", f"{column}_reference", f"{column}_comparator"]].copy()
        values["reference"] = pd.to_numeric(values[f"{column}_reference"], errors="coerce")
        values["comparator"] = pd.to_numeric(values[f"{column}_comparator"], errors="coerce")
        values = values.dropna(subset=["reference", "comparator"])
        grouped = values.groupby("user_key", sort=True).agg(
            reference=("reference", "mean"), comparator=("comparator", "mean")
        )
        delta = grouped["reference"].to_numpy() - grouped["comparator"].to_numpy()
        draws = delta[rng.integers(0, len(delta), size=(reps, len(delta)))].mean(axis=1)
        low, high = np.quantile(draws, [0.025, 0.975])
        records.append(
            {
                "reference": reference,
                "comparator": comparator,
                "metric": label,
                "n_users": int(len(grouped)),
                "n_paired_cases": int(len(values)),
                "reference_macro_pct": 100.0 * float(grouped["reference"].mean()),
                "comparator_macro_pct": 100.0 * float(grouped["comparator"].mean()),
                "delta_reference_minus_comparator_pp": 100.0 * float(delta.mean()),
                "ci95_low_pp": 100.0 * float(low),
                "ci95_high_pp": 100.0 * float(high),
                "bootstrap_reps": reps,
                "seed": seed,
            }
        )
    return records


def relevant_trip_temporal_records(
    prepared_path: Path,
    relevant_keys: set[tuple[int, int]],
    observed_fraction: float,
    horizon_s: float,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    carry = pd.DataFrame()
    usecols = ["ID", "trip_id", "datetime"]

    def process(group: pd.DataFrame) -> None:
        driver_id = int(group["ID"].iloc[0])
        trip_id = int(group["trip_id"].iloc[0])
        if (driver_id, trip_id) not in relevant_keys:
            return
        times = pd.to_datetime(group["datetime"], errors="raise").astype("int64").to_numpy() / 1e9
        n = len(times)
        observed_idx = max(1, min(n - 2, int(round((n - 1) * observed_fraction))))
        target = float(times[observed_idx] + horizon_s)
        duplicate_steps = int(np.sum(np.diff(times) <= 0.0))
        target_after_end = bool(target >= float(times[-1]))
        bracket_gap_s = 0.0
        if not target_after_end:
            idx = int(np.searchsorted(times, target, side="right"))
            idx = max(1, min(idx, n - 1))
            bracket_gap_s = float(times[idx] - times[idx - 1])
        records.append(
            {
                "driver_id": driver_id,
                "trip_id": trip_id,
                "trip_points": n,
                "observed_index_zero_based": observed_idx,
                "sorted_nonpositive_steps": duplicate_steps,
                "target_after_trip_end": target_after_end,
                "target_bracket_gap_s": bracket_gap_s,
                "target_bracket_gap_over_1h": bool(bracket_gap_s > 3600.0),
            }
        )

    for chunk in pd.read_csv(prepared_path, usecols=usecols, chunksize=500_000, low_memory=False):
        if not carry.empty:
            chunk = pd.concat([carry, chunk], ignore_index=True)
        last_id = int(chunk["ID"].iloc[-1])
        last_trip = int(chunk["trip_id"].iloc[-1])
        last_mask = chunk["ID"].eq(last_id) & chunk["trip_id"].eq(last_trip)
        complete = chunk.loc[~last_mask]
        carry = chunk.loc[last_mask].copy()
        for _, group in complete.groupby(["ID", "trip_id"], sort=False):
            process(group)
    if not carry.empty:
        process(carry)
    out = pd.DataFrame(records)
    if len(out) != len(relevant_keys):
        missing = relevant_keys.difference(set(zip(out["driver_id"], out["trip_id"])))
        raise RuntimeError(f"Temporal records missing for {len(missing)} test trips")
    return out


def cohort_history(audit: pd.DataFrame, selected_users: set[int]) -> dict[str, object]:
    selected = audit.loc[
        audit["driver_id"].isin(selected_users) & audit["status"].eq("eligible")
    ].copy()
    selected["start_datetime"] = pd.to_datetime(selected["start_datetime"], errors="raise")
    selected["end_datetime"] = pd.to_datetime(selected["end_datetime"], errors="raise")
    grouped = selected.groupby("driver_id").agg(
        trips=("trip_id", "nunique"),
        first=("start_datetime", "min"),
        last=("end_datetime", "max"),
    )
    grouped["history_days"] = (grouped["last"] - grouped["first"]).dt.total_seconds() / 86400.0

    def stats(column: str) -> dict[str, float]:
        values = grouped[column].to_numpy(dtype=float)
        return {
            "median": float(np.median(values)),
            "q1": float(np.quantile(values, 0.25)),
            "q3": float(np.quantile(values, 0.75)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    return {
        "n_users": int(len(grouped)),
        "n_trajectories": int(grouped["trips"].sum()),
        "trips_per_user": stats("trips"),
        "history_span_days": stats("history_days"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--prepared-geolife", required=True)
    parser.add_argument("--trajectory-audit", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260807)
    args = parser.parse_args()

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(args.predictions, low_memory=False)
    geolife = predictions.loc[predictions["dataset"].astype(str).str.contains("geo", case=False)].copy()
    geolife["driver_id"] = pd.to_numeric(geolife["driver_id"], errors="raise").astype(int)

    endpoint_records = []
    for method in [REFERENCE, "destination_only"]:
        values = predictions.loc[predictions["method"].eq(method), "route_endpoint_to_destination_m"]
        endpoint_records.append(
            {
                "method": method,
                "n": int(pd.to_numeric(values, errors="coerce").notna().sum()),
                "median_m": percentile(values, 0.50),
                "p90_m": percentile(values, 0.90),
                "p95_m": percentile(values, 0.95),
            }
        )
    pd.DataFrame(endpoint_records).to_csv(output / "endpoint_discrepancy.csv", index=False)

    leaveout = geolife.loc[~geolife["driver_id"].isin(DEVELOPMENT_USERS)].copy()
    full_scope = f"all_{geolife['driver_id'].nunique()}_geolife_users"
    leaveout_metrics = pd.DataFrame(
        aggregate_metrics(geolife, full_scope)
        + aggregate_metrics(leaveout, "exclude_all_13_development_users")
    )
    leaveout_metrics.to_csv(output / "leave_development_users_metrics.csv", index=False)
    bootstrap = []
    for comparator in COMPARATORS:
        bootstrap.extend(
            paired_macro_bootstrap(
                leaveout,
                reference=REFERENCE,
                comparator=comparator,
                reps=args.bootstrap_reps,
                seed=args.bootstrap_seed,
            )
        )
    pd.DataFrame(bootstrap).to_csv(output / "leave_development_users_bootstrap.csv", index=False)

    reference_cases = geolife.loc[geolife["method"].eq(REFERENCE), KEYS].drop_duplicates()
    relevant_keys = set(zip(reference_cases["driver_id"].astype(int), reference_cases["trip_id"].astype(int)))
    temporal_cache = output / "temporal_case_audit.csv"
    audit = pd.read_csv(args.trajectory_audit, low_memory=False)
    audit = audit.rename(columns={"user_id": "driver_id"})
    audit["driver_id"] = pd.to_numeric(audit["driver_id"], errors="raise").astype(int)
    audit["trip_id"] = pd.to_numeric(audit["trip_id"], errors="raise").astype(int)
    audit_cols = [
        "driver_id",
        "trip_id",
        "duplicate_or_nonpositive_time_steps",
        "max_recording_gap_s",
        "max_implied_speed_mps",
    ]
    if temporal_cache.exists():
        temporal = pd.read_csv(temporal_cache, low_memory=False)
    else:
        temporal = relevant_trip_temporal_records(
            Path(args.prepared_geolife), relevant_keys, observed_fraction=0.66, horizon_s=300.0
        )
        temporal = temporal.merge(
            audit[audit_cols], on=["driver_id", "trip_id"], how="left", validate="one_to_one"
        )
        if temporal[audit_cols[2:]].isna().any().any():
            raise RuntimeError("Missing source-audit fields for evaluated trajectories")
        temporal["source_nonpositive"] = temporal["duplicate_or_nonpositive_time_steps"].gt(0)
        temporal["source_gap_over_1h"] = temporal["max_recording_gap_s"].gt(3600.0)
        temporal["source_speed_over_100mps"] = temporal["max_implied_speed_mps"].gt(100.0)
        temporal.to_csv(temporal_cache, index=False)

    case_temporal = reference_cases.merge(temporal, on=["driver_id", "trip_id"], how="left", validate="many_to_one")
    scopes = {
        "all_geolife_cases": np.ones(len(case_temporal), dtype=bool),
        "exclude_target_interpolation_gap_over_1h": ~case_temporal["target_bracket_gap_over_1h"],
        "exclude_any_source_nonpositive_or_gap_over_1h": ~(
            case_temporal["source_nonpositive"] | case_temporal["source_gap_over_1h"]
        ),
        "exclude_any_audited_temporal_or_speed_anomaly": ~(
            case_temporal["source_nonpositive"]
            | case_temporal["source_gap_over_1h"]
            | case_temporal["source_speed_over_100mps"]
        ),
    }
    sensitivity_records = []
    for scope, mask in scopes.items():
        keys = case_temporal.loc[mask, KEYS]
        scoped = geolife.merge(keys, on=KEYS, how="inner", validate="many_to_one")
        sensitivity_records.extend(aggregate_metrics(scoped, scope))
    pd.DataFrame(sensitivity_records).to_csv(output / "temporal_sensitivity_metrics.csv", index=False)

    history = cohort_history(audit, set(reference_cases["driver_id"].astype(int)))
    (output / "cohort_history_summary.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "development_users": sorted(DEVELOPMENT_USERS),
        "development_users_in_final_geolife_cohort": sorted(
            DEVELOPMENT_USERS.intersection(set(reference_cases["driver_id"].astype(int)))
        ),
        "geolife_users_full": int(reference_cases["driver_id"].nunique()),
        "geolife_users_leaveout": int(leaveout.loc[leaveout["method"].eq(REFERENCE), "driver_id"].nunique()),
        "geolife_cases_full": int(len(reference_cases)),
        "geolife_cases_leaveout": int(leaveout.loc[leaveout["method"].eq(REFERENCE)].shape[0]),
        "unique_test_trajectories": int(len(temporal)),
        "split_specific_cases_target_gap_over_1h": int(case_temporal["target_bracket_gap_over_1h"].sum()),
        "split_specific_cases_source_nonpositive": int(case_temporal["source_nonpositive"].sum()),
        "split_specific_cases_source_gap_over_1h": int(case_temporal["source_gap_over_1h"].sum()),
        "split_specific_cases_source_speed_over_100mps": int(case_temporal["source_speed_over_100mps"].sum()),
        "split_specific_cases_target_after_trip_end": int(case_temporal["target_after_trip_end"].sum()),
        "bootstrap": {
            "unit": "GeoLife user",
            "paired": True,
            "reps": args.bootstrap_reps,
            "seed": args.bootstrap_seed,
            "interval": "two-sided percentile 95%",
        },
        "temporal_sensitivity_limitation": (
            "Retrospective test-case filtering; models were not retrained after removing anomalous training trajectories."
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
