#!/usr/bin/env python3
"""Audit train-only provenance and hidden-suffix invariance.

One deterministic test case is audited for every individual--split model. The
case is forecast twice: first with its recorded hidden suffix and then with all
post-prefix coordinates and times replaced. Candidate construction, primary
admissibility, and coherent selection must remain unchanged.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
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


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Python module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def replace_hidden_suffix(trip: Any, prefix_idx: int) -> Any:
    """Return a copy with the observed prefix intact and the suffix replaced."""
    arrays = {
        name: np.asarray(getattr(trip, name)).copy()
        for name in ("lat", "lon", "x", "y", "t_s")
    }
    suffix = np.arange(prefix_idx + 1, int(trip.n), dtype=int)
    if len(suffix) == 0:
        raise ValueError("The audited trip has no hidden suffix")
    step = np.arange(1, len(suffix) + 1, dtype=float)
    arrays["x"][suffix] = float(arrays["x"][prefix_idx]) + 1_000_000.0 + 137.0 * step
    arrays["y"][suffix] = float(arrays["y"][prefix_idx]) - 1_000_000.0 - 193.0 * step
    arrays["lat"][suffix] = np.clip(float(arrays["lat"][prefix_idx]) + 0.25, -89.0, 89.0)
    arrays["lon"][suffix] = (
        (float(arrays["lon"][prefix_idx]) + 0.25 + 180.0) % 360.0
    ) - 180.0
    arrays["t_s"][suffix] = float(arrays["t_s"][prefix_idx]) + 86_400.0 * step
    return replace(trip, **arrays)


def scalar(value: Any) -> Any:
    if value is None:
        return None
    number = float(value)
    if math.isnan(number):
        return "nan"
    if math.isinf(number):
        return "inf" if number > 0 else "-inf"
    return number.hex()


def vector(value: Any) -> list[Any] | None:
    if value is None:
        return None
    return [scalar(item) for item in np.asarray(value, dtype=float).tolist()]


def candidate_signature(candidate: Any | None) -> Any:
    if candidate is None:
        return None
    route_key = None
    if candidate.route_trip is not None:
        route_key = (
            str(candidate.route_trip.dataset),
            int(candidate.route_trip.driver_id),
            int(candidate.route_trip.trip_id),
        )
    return (
        str(candidate.name),
        str(candidate.family),
        scalar(candidate.dest_x),
        scalar(candidate.dest_y),
        scalar(candidate.dest_lat),
        scalar(candidate.dest_lon),
        route_key,
        None if candidate.route_index is None else int(candidate.route_index),
        scalar(candidate.dest_score),
        scalar(candidate.route_score),
        scalar(candidate.confidence),
        None if candidate.cluster_id is None else int(candidate.cluster_id),
        str(candidate.source),
        vector(candidate.route_x),
        vector(candidate.route_y),
        vector(candidate.route_lat),
        vector(candidate.route_lon),
        scalar(candidate.lookahead_x),
        scalar(candidate.lookahead_y),
        scalar(candidate.lookahead_lat),
        scalar(candidate.lookahead_lon),
    )


def train_projection_center(train_trips: list[Any]) -> tuple[float, float]:
    lat = np.asarray([float(np.median(trip.lat)) for trip in train_trips], dtype=float)
    lon = np.asarray([float(np.median(trip.lon)) for trip in train_trips], dtype=float)
    return float(np.median(lat)), float(np.median(lon))


def primary_candidates(
    model: Any,
    guard: Any,
    candidates: list[Any],
    test_trip: Any,
    prefix_idx: int,
    grid_model: Any,
    config: dict[str, Any],
) -> list[Any]:
    retained: list[Any] = []
    for candidate in candidates:
        geometry = guard.candidate_geometry(
            model,
            candidate,
            test_trip,
            prefix_idx,
            grid_model,
            config,
        )
        endpoint = model.candidate_route_endpoint(candidate)
        discrepancy = math.hypot(
            float(endpoint[0]) - float(candidate.dest_x),
            float(endpoint[1]) - float(candidate.dest_y),
        )
        if not math.isfinite(discrepancy) or discrepancy > 500.0:
            continue
        start_gap = float(geometry["route_start_gap_m"])
        if not math.isfinite(start_gap) or start_gap > 1000.0:
            continue
        if candidate.family == "grid_transition" and int(
            geometry["grid_reached_destination"]
        ) != 1:
            continue
        retained.append(candidate)
    return retained


def infer(
    model: Any,
    guard: Any,
    test_trip: Any,
    train_trips: list[Any],
    clusters: list[Any],
    trip_to_cluster: dict[int, int],
    grid_model: Any,
    config: dict[str, Any],
) -> tuple[list[Any], Any | None, bool, set[int]]:
    observed_fraction = float(config["experiment"].get("observed_fraction", 0.66))
    prefix_idx = model.observed_index(test_trip, observed_fraction)
    analogues = model.retrieve_analogues(test_trip, train_trips, config)
    candidates = model.make_candidates(
        test_trip,
        train_trips,
        clusters,
        trip_to_cluster,
        analogues,
        config,
        grid_model=grid_model,
    )
    admissible = primary_candidates(
        model,
        guard,
        candidates,
        test_trip,
        prefix_idx,
        grid_model,
        config,
    )
    selected = model.select_coherent_product(admissible, config)
    threshold = float(config["experiment"].get("abstain_confidence_threshold", 0.48))
    emitted = selected is not None and float(selected.confidence) >= threshold
    route_ids = {
        int(candidate.route_trip.trip_id)
        for candidate in candidates
        if candidate.route_trip is not None
    }
    return candidates, selected, bool(emitted), route_ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-script", type=Path, required=True)
    parser.add_argument(
        "--guard-script",
        type=Path,
        default=ROOT / "scripts" / "geometry_guard.py",
    )
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    parser.add_argument("--expected-models", type=int, default=210)
    args = parser.parse_args()

    model = load_module(args.model_script, "leakage_audit_model")
    guard = load_module(args.guard_script, "leakage_audit_guard")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    exp = config["experiment"]
    min_trips = int(exp.get("min_trips_per_driver", 20))
    train_fraction = float(exp.get("train_fraction", 0.75))
    observed_fraction = float(exp.get("observed_fraction", 0.66))
    rows: list[dict[str, Any]] = []

    for dataset_spec in model.dataset_specs(config):
        trips = model.load_trips(dataset_spec, config)
        by_driver: dict[int, list[Any]] = {}
        for trip in trips:
            by_driver.setdefault(int(trip.driver_id), []).append(trip)
        driver_ids = [
            driver_id
            for driver_id, driver_trips in sorted(by_driver.items())
            if len(driver_trips) >= min_trips
        ]
        for driver_id in driver_ids:
            driver_trips = by_driver[driver_id]
            for split in exp.get("splits", []):
                split_label = str(split["label"])
                train_trips, test_trips = model.split_driver_trips(
                    driver_trips,
                    split_label=split_label,
                    split_mode=str(split["mode"]),
                    train_fraction=train_fraction,
                    seed=int(split.get("seed", 13)),
                )
                split_spec, projection_meta = model.projection_for_split(
                    dataset_spec,
                    train_trips,
                    config,
                )
                model.reproject_trips(driver_trips, split_spec.epsg)
                clusters, trip_to_cluster = model.train_destination_clusters(
                    train_trips,
                    config,
                    transformer_to_latlon=Transformer.from_crs(
                        split_spec.epsg,
                        "EPSG:4326",
                        always_xy=True,
                    ),
                )
                grid_model = model.build_grid_model(
                    spec=split_spec,
                    driver_id=driver_id,
                    driver_name=driver_trips[0].driver_name,
                    split_label=split_label,
                    train_trips=train_trips,
                    config=config,
                )

                train_ids = {int(trip.trip_id) for trip in train_trips}
                test_ids = {int(trip.trip_id) for trip in test_trips}
                test_trip = test_trips[0]
                prefix_idx = model.observed_index(test_trip, observed_fraction)
                altered_trip = replace_hidden_suffix(test_trip, prefix_idx)

                original = infer(
                    model,
                    guard,
                    test_trip,
                    train_trips,
                    clusters,
                    trip_to_cluster,
                    grid_model,
                    config,
                )
                altered = infer(
                    model,
                    guard,
                    altered_trip,
                    train_trips,
                    clusters,
                    trip_to_cluster,
                    grid_model,
                    config,
                )
                original_candidates, original_selected, original_emitted, original_routes = original
                altered_candidates, altered_selected, altered_emitted, altered_routes = altered

                chronological_ok = True
                if str(split["mode"]) == "chronological":
                    chronological_ok = max(t.start_time for t in train_trips) <= min(
                        t.start_time for t in test_trips
                    )
                projection_ok = True
                if projection_meta["projection_strategy"] == "train_centered_aeqd":
                    center_lat, center_lon = train_projection_center(train_trips)
                    projection_ok = bool(
                        math.isclose(
                            center_lat,
                            float(projection_meta["projection_center_lat"]),
                            rel_tol=0.0,
                            abs_tol=1e-10,
                        )
                        and math.isclose(
                            center_lon,
                            float(projection_meta["projection_center_lon"]),
                            rel_tol=0.0,
                            abs_tol=1e-10,
                        )
                    )

                checks = {
                    "train_test_disjoint": train_ids.isdisjoint(test_ids),
                    "chronological_order": chronological_ok,
                    "projection_center_train_only": projection_ok,
                    "destination_regions_train_only": all(
                        set(cluster.train_trip_ids).issubset(train_ids) for cluster in clusters
                    ),
                    "grid_structures_train_only": set(
                        int(value) for value in grid_model.train_sequences_by_trip
                    ).issubset(train_ids),
                    "candidate_routes_train_only": (
                        original_routes.issubset(train_ids)
                        and altered_routes.issubset(train_ids)
                    ),
                    "hidden_suffix_candidate_set_invariant": (
                        [candidate_signature(value) for value in original_candidates]
                        == [candidate_signature(value) for value in altered_candidates]
                    ),
                    "hidden_suffix_selection_invariant": (
                        candidate_signature(original_selected)
                        == candidate_signature(altered_selected)
                        and original_emitted == altered_emitted
                    ),
                }
                rows.append(
                    {
                        "dataset": dataset_spec.name,
                        "driver_id": driver_id,
                        "split_label": split_label,
                        "audited_trip_id": int(test_trip.trip_id),
                        "candidate_count": len(original_candidates),
                        "selected_candidate": (
                            None if original_selected is None else str(original_selected.name)
                        ),
                        **{name: int(value) for name, value in checks.items()},
                    }
                )

    audit = pd.DataFrame(rows)
    check_columns = [
        column
        for column in audit.columns
        if column
        not in {
            "dataset",
            "driver_id",
            "split_label",
            "audited_trip_id",
            "candidate_count",
            "selected_candidate",
        }
    ]
    failures = {column: int((audit[column] != 1).sum()) for column in check_columns}
    dataset_models = {
        str(name): int(count)
        for name, count in audit.groupby("dataset", sort=True).size().items()
    }
    status = (
        "passed"
        if len(audit) == int(args.expected_models) and not any(failures.values())
        else "failed"
    )
    report = {
        "status": status,
        "expected_models": int(args.expected_models),
        "audited_models": int(len(audit)),
        "hidden_suffix_replacements": int(len(audit)),
        "models_by_dataset": dataset_models,
        "failure_counts": failures,
    }
    args.private_output.parent.mkdir(parents=True, exist_ok=True)
    args.public_output.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.private_output, index=False)
    args.public_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if status != "passed":
        raise AssertionError(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
