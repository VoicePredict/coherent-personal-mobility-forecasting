#!/usr/bin/env python3
"""Run the predeclared submission diagnostics on the frozen 8,526 cases.

The frozen data loader, candidate generator, geometry guard, and evaluation
function are imported without modification.  This experiment adds four fixed
selection variants, an admissible top-15 retrieval comparator, and audits
whether chronological training trips had ended by each prediction time.
Case-level outputs contain GeoLife-derived quantities and remain private.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from pyproj import Transformer


ROOT = Path(__file__).resolve().parents[1]
GUARD_SCRIPT = ROOT / "scripts/geometry_guard.py"
FINAL_SCRIPT = ROOT / "scripts/run_final_admissible.py"
TOP15_SCRIPT = ROOT / "scripts/run_top15_baseline.py"
KEYS = ["dataset", "driver_id", "split_label", "trip_id"]
VARIANTS = ["full", "no_grid", "no_soft_coherence", "equal_weights", "top15_retrieval"]
PRIMARY_ARM = {
    "require_grid_reach": True,
    "max_start_gap_m": 1000.0,
    "max_endpoint_gap_m": 500.0,
}


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Python module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def trip_end_time(trip: Any) -> pd.Timestamp:
    elapsed = float(trip.t_s[-1]) if len(trip.t_s) else 0.0
    return pd.Timestamp(trip.start_time) + pd.to_timedelta(elapsed, unit="s")


def prediction_time(trip: Any, prefix_idx: int) -> pd.Timestamp:
    elapsed = float(trip.t_s[prefix_idx])
    return pd.Timestamp(trip.start_time) + pd.to_timedelta(elapsed, unit="s")


def coherent_score(model: Any, candidate: Any, variant: str, threshold_m: float) -> float:
    coherence = math.exp(-model.candidate_conflict_m(candidate) / max(1.0, threshold_m))
    if variant == "no_soft_coherence":
        return float(0.50 * candidate.dest_score + 0.35 * candidate.route_score)
    if variant == "equal_weights":
        return float((candidate.dest_score + candidate.route_score + coherence) / 3.0)
    return float(0.50 * candidate.dest_score + 0.35 * candidate.route_score + 0.15 * coherence)


def select_variant(model: Any, candidates: list[Any], variant: str, threshold_m: float) -> Any | None:
    pool = candidates
    if variant == "no_grid":
        pool = [candidate for candidate in candidates if candidate.family != "grid_transition"]
    if not pool:
        return None
    return max(pool, key=lambda candidate: coherent_score(model, candidate, variant, threshold_m))


def build_structures(
    model: Any,
    dataset_spec: Any,
    driver_trips: list[Any],
    train_trips: list[Any],
    split_label: str,
    driver_id: int,
    config: dict[str, Any],
) -> tuple[Any, dict[str, Any], list[Any], dict[int, int], Any]:
    split_spec, projection_meta = model.projection_for_split(dataset_spec, train_trips, config)
    model.reproject_trips(driver_trips, split_spec.epsg)
    clusters, trip_to_cluster = model.train_destination_clusters(
        train_trips,
        config,
        transformer_to_latlon=Transformer.from_crs(split_spec.epsg, "EPSG:4326", always_xy=True),
    )
    grid_model = model.build_grid_model(
        spec=split_spec,
        driver_id=driver_id,
        driver_name=driver_trips[0].driver_name,
        split_label=split_label,
        train_trips=train_trips,
        config=config,
    )
    return split_spec, projection_meta, clusters, trip_to_cluster, grid_model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-script", type=Path, required=True)
    parser.add_argument("--reference-predictions", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path, required=True)
    parser.add_argument("--temporal-audit-output", type=Path, required=True)
    parser.add_argument("--region-radii-output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=15)
    args = parser.parse_args()
    if args.top_k != 15:
        raise ValueError("The predeclared retrieval budget is 15")
    for path in [args.predictions_output, args.temporal_audit_output, args.region_radii_output]:
        path.parent.mkdir(parents=True, exist_ok=True)

    model = load_module(args.model_script, "submission_diagnostics_model")
    guard = load_module(GUARD_SCRIPT, "submission_diagnostics_guard")
    final = load_module(FINAL_SCRIPT, "submission_diagnostics_final")
    top15 = load_module(TOP15_SCRIPT, "submission_diagnostics_top15")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    reference = pd.read_csv(args.reference_predictions, usecols=KEYS).drop_duplicates()
    if len(reference) != 8526:
        raise ValueError(f"Expected 8,526 reference cases, found {len(reference):,}")

    exp = config["experiment"]
    observed_fraction = float(exp.get("observed_fraction", 0.66))
    horizon_s = float(exp.get("lookahead_minutes", 5.0)) * 60.0
    min_trips = int(exp.get("min_trips_per_driver", 20))
    train_fraction = float(exp.get("train_fraction", 0.75))
    coherence_threshold = float(exp.get("coherence_threshold_m", 500.0))
    predictions: list[dict[str, Any]] = []
    temporal_rows: list[dict[str, Any]] = []
    radius_rows: list[dict[str, Any]] = []

    for dataset_spec in model.dataset_specs(config):
        print(f"Loading {dataset_spec.name}", flush=True)
        trips = model.load_trips(dataset_spec, config)
        by_driver: dict[int, list[Any]] = {}
        for trip in trips:
            by_driver.setdefault(int(trip.driver_id), []).append(trip)
        driver_ids = [
            driver_id for driver_id, driver_trips in sorted(by_driver.items())
            if len(driver_trips) >= min_trips
        ]
        for driver_number, driver_id in enumerate(driver_ids, start=1):
            driver_trips = by_driver[driver_id]
            for split in exp.get("splits", []):
                split_label = str(split["label"])
                split_mode = str(split["mode"])
                train_trips, test_trips = model.split_driver_trips(
                    driver_trips,
                    split_label=split_label,
                    split_mode=split_mode,
                    train_fraction=train_fraction,
                    seed=int(split.get("seed", 13)),
                )
                base = build_structures(
                    model, dataset_spec, driver_trips, train_trips,
                    split_label, driver_id, config,
                )
                base_spec, base_meta, base_clusters, base_trip_to_cluster, base_grid = base
                for cluster in base_clusters:
                    radius_rows.append({
                        "dataset": dataset_spec.name,
                        "driver_id": driver_id,
                        "split_label": split_label,
                        "cluster_id": int(cluster.cluster_id),
                        "support": int(cluster.support),
                        "radius_p90_m": float(cluster.radius_p90_m),
                    })

                print(
                    f"{dataset_spec.name} user={driver_id} split={split_label}: "
                    f"train={len(train_trips)} test={len(test_trips)}",
                    flush=True,
                )
                for test_index, test_trip in enumerate(test_trips, start=1):
                    prefix_idx = model.observed_index(test_trip, observed_fraction)
                    forecast_time = prediction_time(test_trip, prefix_idx)
                    unavailable = [trip for trip in train_trips if trip_end_time(trip) > forecast_time]
                    violation = split_mode == "chronological" and bool(unavailable)
                    available_train = (
                        [trip for trip in train_trips if trip_end_time(trip) <= forecast_time]
                        if violation else train_trips
                    )
                    if not available_train:
                        raise ValueError(
                            f"No temporally available history for {dataset_spec.name}/{driver_id}/{test_trip.trip_id}"
                        )
                    temporal_rows.append({
                        "dataset": dataset_spec.name,
                        "driver_id": driver_id,
                        "split_label": split_label,
                        "trip_id": int(test_trip.trip_id),
                        "is_chronological": int(split_mode == "chronological"),
                        "train_trip_count": len(train_trips),
                        "available_train_trip_count": len(available_train),
                        "unavailable_train_trip_count": len(unavailable) if split_mode == "chronological" else 0,
                        "temporal_availability_violation": int(violation),
                    })

                    if violation:
                        working = build_structures(
                            model, dataset_spec, driver_trips, available_train,
                            split_label, driver_id, config,
                        )
                    else:
                        working = base
                    split_spec, projection_meta, clusters, trip_to_cluster, grid_model = working
                    analogues = model.retrieve_analogues(test_trip, available_train, config)
                    candidates = model.make_candidates(
                        test_trip,
                        available_train,
                        clusters,
                        trip_to_cluster,
                        analogues,
                        config,
                        grid_model=grid_model,
                    )
                    geometries = [
                        guard.candidate_geometry(model, candidate, test_trip, prefix_idx, grid_model, config)
                        for candidate in candidates
                    ]
                    geometry_by_id = {
                        id(candidate): geometry for candidate, geometry in zip(candidates, geometries)
                    }
                    eligible = [
                        candidate for candidate, geometry in zip(candidates, geometries)
                        if final.final_is_eligible(candidate, geometry, PRIMARY_ARM)
                    ]

                    cluster_score = model.cluster_scores(
                        analogues, clusters, trip_to_cluster, test_trip, config
                    )
                    analogue_candidates = [
                        top15.make_analogue_candidate(
                            model,
                            analogue,
                            rank=rank,
                            cluster_scores=cluster_score,
                            trip_to_cluster=trip_to_cluster,
                        )
                        for rank, analogue in enumerate(analogues[: args.top_k], start=1)
                    ]
                    analogue_geometries = [
                        top15.candidate_geometry(model, candidate, test_trip, prefix_idx)
                        for candidate in analogue_candidates
                    ]
                    retrieval, retrieval_rank, _ = top15.first_admissible(
                        analogue_candidates,
                        analogue_geometries,
                        max_start_gap_m=float(PRIMARY_ARM["max_start_gap_m"]),
                        max_endpoint_gap_m=float(PRIMARY_ARM["max_endpoint_gap_m"]),
                    )

                    selections = {
                        variant: select_variant(model, eligible, variant, coherence_threshold)
                        for variant in VARIANTS if variant != "top15_retrieval"
                    }
                    selections["top15_retrieval"] = retrieval
                    for variant, candidate in selections.items():
                        emitted = candidate is not None
                        row = model.evaluate_candidate(
                            dataset=test_trip.dataset,
                            dataset_label=split_spec.label,
                            driver_id=test_trip.driver_id,
                            driver_name=test_trip.driver_name,
                            split_label=split_label,
                            method=variant,
                            test_trip=test_trip,
                            prefix_idx=prefix_idx,
                            candidate=candidate,
                            emitted=emitted,
                            train_count=len(available_train),
                            test_count=len(test_trips),
                            analogues=analogues,
                            clusters=clusters,
                            config=config,
                        )
                        selected_geometry = geometry_by_id.get(id(candidate), {})
                        row.update(projection_meta)
                        row.update({
                            "variant": variant,
                            "original_train_trip_count": len(train_trips),
                            "available_train_trip_count": len(available_train),
                            "temporal_availability_violation": int(violation),
                            "eligible_candidate_count": len(eligible),
                            "selection_score": (
                                coherent_score(model, candidate, variant, coherence_threshold)
                                if candidate is not None and variant != "top15_retrieval" else np.nan
                            ),
                            "selected_route_start_gap_m": selected_geometry.get("route_start_gap_m", np.nan),
                            "selected_grid_reached_destination": selected_geometry.get("grid_reached_destination", np.nan),
                            "selected_analogue_rank": retrieval_rank if variant == "top15_retrieval" else np.nan,
                            "horizon_beyond_recording_end": int(
                                float(test_trip.t_s[prefix_idx]) + horizon_s > float(test_trip.t_s[-1])
                            ),
                        })
                        predictions.append(row)

                    if violation:
                        model.reproject_trips(driver_trips, base_spec.epsg)
                    if test_index % 25 == 0:
                        print(f"  processed {test_index}/{len(test_trips)}", flush=True)
            print(
                f"Completed {dataset_spec.name} user {driver_number}/{len(driver_ids)}",
                flush=True,
            )

    output = pd.DataFrame(predictions)
    observed = output[KEYS].drop_duplicates().sort_values(KEYS).reset_index(drop=True)
    expected = reference.sort_values(KEYS).reset_index(drop=True)
    if not observed.equals(expected):
        raise ValueError("Generated case keys do not match the frozen reference")
    expected_rows = len(reference) * len(VARIANTS)
    if len(output) != expected_rows:
        raise ValueError(f"Expected {expected_rows:,} rows, found {len(output):,}")
    output.to_csv(args.predictions_output, index=False)
    pd.DataFrame(temporal_rows).to_csv(args.temporal_audit_output, index=False)
    pd.DataFrame(radius_rows).to_csv(args.region_radii_output, index=False)
    print(f"Wrote {len(output):,} prediction rows", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
