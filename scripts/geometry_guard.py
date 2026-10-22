#!/usr/bin/env python3
"""Re-select forecast candidates after geometry-admissibility filtering.

The frozen forecasting implementation is imported without modification.  Its
candidate generator and scores are reused exactly; only the eligibility gate
and the subsequent selection are added.  Case-level outputs contain GeoLife
coordinates and are therefore private analysis artifacts.
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


KEYS = ["dataset", "driver_id", "split_label", "trip_id"]
METHODS = [
    "destination_only",
    "route_analogue",
    "transition_grid_route",
    "most_frequent_destination",
    "coherent_product",
    "coherent_abstain",
]
ARMS: dict[str, dict[str, float | bool]] = {
    "unfiltered_reference": {"require_grid_reach": False, "max_start_gap_m": math.inf},
    "grid_guard_only": {"require_grid_reach": True, "max_start_gap_m": math.inf},
    "start_guard_1000m_only": {"require_grid_reach": False, "max_start_gap_m": 1000.0},
    "combined_1000m": {"require_grid_reach": True, "max_start_gap_m": 1000.0},
    "combined_500m": {"require_grid_reach": True, "max_start_gap_m": 500.0},
    "combined_350m": {"require_grid_reach": True, "max_start_gap_m": 350.0},
}


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Python module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def beam_reached_destination(
    model: Any,
    grid_model: Any,
    *,
    start_cell: str,
    destination_cell: str,
    max_steps: int,
    beam_width: int,
) -> tuple[bool, float]:
    """Repeat the frozen beam before its synthetic endpoint append."""
    beam: list[tuple[list[str], float]] = [([start_cell], 1.0)]
    completed: list[tuple[list[str], float]] = []
    for _ in range(max(1, int(max_steps))):
        expanded: list[tuple[list[str], float]] = []
        for path, score in beam:
            current = path[-1]
            if current == destination_cell:
                completed.append((path, score))
                expanded.append((path, score))
                continue
            transitions = model.transition_probabilities(grid_model, current)
            for dst, probability, _ in transitions[: max(1, int(beam_width))]:
                loop_penalty = 0.65 if dst in path else 1.0
                expanded.append((path + [dst], score * probability * loop_penalty))
        if not expanded:
            break
        expanded.sort(key=lambda item: item[1], reverse=True)
        beam = expanded[: max(1, int(beam_width))]
        completed.extend(item for item in beam if item[0][-1] == destination_cell)

    pool = completed if completed else beam
    if not pool:
        end_cell = start_cell
    else:
        def final_score(item: tuple[list[str], float]) -> float:
            path, score = item
            end_x, end_y = model.grid_cell_center_xy(grid_model, path[-1])
            dst_x, dst_y = model.grid_cell_center_xy(grid_model, destination_cell)
            distance_penalty = math.exp(
                -model.euclid(end_x, end_y, dst_x, dst_y)
                / max(1.0, grid_model.cell_size_m)
            )
            destination_bonus = 1.0 if path[-1] == destination_cell else 0.5
            return score * (0.5 + 0.5 * distance_penalty) * destination_bonus

        end_cell = max(pool, key=final_score)[0][-1]
    reached = end_cell == destination_cell
    end_x, end_y = model.grid_cell_center_xy(grid_model, end_cell)
    dst_x, dst_y = model.grid_cell_center_xy(grid_model, destination_cell)
    connector_m = 0.0 if reached else model.euclid(end_x, end_y, dst_x, dst_y)
    return bool(reached), float(connector_m)


def candidate_geometry(
    model: Any,
    candidate: Any,
    test_trip: Any,
    prefix_idx: int,
    grid_model: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    current_x = float(test_trip.x[prefix_idx])
    current_y = float(test_trip.y[prefix_idx])
    if candidate.family == "grid_transition":
        start_cell = model.grid_cell_id(grid_model, current_x, current_y)
        destination_cell = model.grid_cell_id(
            grid_model,
            float(candidate.dest_x),
            float(candidate.dest_y),
        )
        reached, connector_m = beam_reached_destination(
            model,
            grid_model,
            start_cell=start_cell,
            destination_cell=destination_cell,
            max_steps=int(config["experiment"].get("grid_route_max_steps", 30)),
            beam_width=int(config["experiment"].get("grid_route_beam_width", 8)),
        )
        route_start_x = float(candidate.route_x[0]) if candidate.route_x is not None and len(candidate.route_x) else current_x
        route_start_y = float(candidate.route_y[0]) if candidate.route_y is not None and len(candidate.route_y) else current_y
        return {
            "route_start_gap_m": float(model.euclid(current_x, current_y, route_start_x, route_start_y)),
            "grid_reached_destination": int(reached),
            "synthetic_connector_m": connector_m,
            "geometry_source": "grid_beam",
        }
    if candidate.route_trip is not None and candidate.route_index is not None:
        route_index = int(candidate.route_index)
        gap = model.euclid(
            current_x,
            current_y,
            float(candidate.route_trip.x[route_index]),
            float(candidate.route_trip.y[route_index]),
        )
        return {
            "route_start_gap_m": float(gap),
            "grid_reached_destination": np.nan,
            "synthetic_connector_m": 0.0,
            "geometry_source": "historical_suffix",
        }
    return {
        "route_start_gap_m": float("nan"),
        "grid_reached_destination": np.nan,
        "synthetic_connector_m": float("nan"),
        "geometry_source": "missing",
    }


def is_eligible(candidate: Any, geometry: dict[str, Any], arm: dict[str, float | bool]) -> bool:
    if candidate.family == "grid_transition":
        return not bool(arm["require_grid_reach"]) or int(geometry["grid_reached_destination"]) == 1
    gap = float(geometry["route_start_gap_m"])
    return math.isfinite(gap) and gap <= float(arm["max_start_gap_m"])


def selected_for_method(model: Any, candidates: list[Any], method: str, config: dict[str, Any]) -> Any | None:
    if method in {"coherent_product", "coherent_abstain"}:
        return model.select_coherent_product(candidates, config)
    if method == "destination_only":
        # Preserve the destination-score order but allow the policy to use the
        # next destination-region candidate when its first route attachment is
        # inadmissible.
        for candidate in candidates:
            if candidate.name == "destination_only" or candidate.family == "destination_cluster_topk":
                return candidate
        return None
    if method == "transition_grid_route":
        # Grid candidates are generated in descending destination-score order.
        # After the reachability gate, retain the first remaining member.
        for candidate in candidates:
            if candidate.family == "grid_transition":
                return candidate
        return None
    return model.select_named(candidates, method)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-script", type=Path, required=True)
    parser.add_argument("--reference-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    model = load_module(args.model_script, "frozen_geometry_guard_model")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    reference = pd.read_csv(args.reference_predictions)
    reference_keys = reference[KEYS].drop_duplicates()
    expected_cases = len(reference_keys)
    if expected_cases != 8526:
        raise ValueError(f"Expected 8,526 reference cases, found {expected_cases:,}")

    exp = config["experiment"]
    observed_fraction = float(exp.get("observed_fraction", 0.66))
    min_trips = int(exp.get("min_trips_per_driver", 20))
    train_fraction = float(exp.get("train_fraction", 0.75))
    abstain_threshold = float(exp.get("abstain_confidence_threshold", 0.48))
    splits = exp.get("splits", [])
    rows: list[dict[str, Any]] = []

    for dataset_spec in model.dataset_specs(config):
        print(f"Loading {dataset_spec.name}", flush=True)
        trips = model.load_trips(dataset_spec, config)
        by_driver: dict[int, list[Any]] = {}
        for trip in trips:
            by_driver.setdefault(int(trip.driver_id), []).append(trip)
        driver_ids = [
            driver_id
            for driver_id, driver_trips in sorted(by_driver.items())
            if len(driver_trips) >= min_trips
        ]
        print(f"{dataset_spec.name}: {len(driver_ids)} eligible individuals", flush=True)

        for driver_number, driver_id in enumerate(driver_ids, start=1):
            driver_trips = by_driver[driver_id]
            for split in splits:
                split_label = str(split["label"])
                train_trips, test_trips = model.split_driver_trips(
                    driver_trips,
                    split_label=split_label,
                    split_mode=str(split["mode"]),
                    train_fraction=train_fraction,
                    seed=int(split.get("seed", 13)),
                )
                split_spec, projection_meta = model.projection_for_split(dataset_spec, train_trips, config)
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
                print(
                    f"{dataset_spec.name} user={driver_id} split={split_label}: "
                    f"train={len(train_trips)} test={len(test_trips)}",
                    flush=True,
                )
                for test_index, test_trip in enumerate(test_trips, start=1):
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
                    geometry = [
                        candidate_geometry(model, candidate, test_trip, prefix_idx, grid_model, config)
                        for candidate in candidates
                    ]
                    geometry_by_identity = {id(candidate): record for candidate, record in zip(candidates, geometry)}
                    reference_product = model.select_coherent_product(candidates, config)

                    for arm_name, arm in ARMS.items():
                        eligible = [
                            candidate
                            for candidate, record in zip(candidates, geometry)
                            if is_eligible(candidate, record, arm)
                        ]
                        original_valid = (
                            reference_product is not None
                            and reference_product in eligible
                        )
                        for method in METHODS:
                            candidate = selected_for_method(model, eligible, method, config)
                            emitted = candidate is not None
                            if method == "coherent_abstain":
                                emitted = emitted and float(candidate.confidence) >= abstain_threshold
                            row = model.evaluate_candidate(
                                dataset=test_trip.dataset,
                                dataset_label=split_spec.label,
                                driver_id=test_trip.driver_id,
                                driver_name=test_trip.driver_name,
                                split_label=split_label,
                                method=method,
                                test_trip=test_trip,
                                prefix_idx=prefix_idx,
                                candidate=candidate,
                                emitted=bool(emitted),
                                train_count=len(train_trips),
                                test_count=len(test_trips),
                                analogues=analogues,
                                clusters=clusters,
                                config=config,
                            )
                            selected_geometry = geometry_by_identity.get(id(candidate), {})
                            row.update(projection_meta)
                            row.update(
                                {
                                    "arm": arm_name,
                                    "require_grid_reach": int(bool(arm["require_grid_reach"])),
                                    "max_start_gap_m": float(arm["max_start_gap_m"]),
                                    "candidate_count": len(candidates),
                                    "eligible_candidate_count": len(eligible),
                                    "original_coherent_candidate_name": None if reference_product is None else reference_product.name,
                                    "original_coherent_candidate_family": None if reference_product is None else reference_product.family,
                                    "original_coherent_candidate_valid": int(original_valid),
                                    "selection_changed": int(
                                        method in {"coherent_product", "coherent_abstain"}
                                        and candidate is not reference_product
                                    ),
                                    "selected_route_start_gap_m": selected_geometry.get("route_start_gap_m", np.nan),
                                    "selected_grid_reached_destination": selected_geometry.get("grid_reached_destination", np.nan),
                                    "selected_synthetic_connector_m": selected_geometry.get("synthetic_connector_m", np.nan),
                                    "selected_geometry_source": selected_geometry.get("geometry_source", "missing"),
                                }
                            )
                            rows.append(row)
                    if test_index % 25 == 0:
                        print(f"  processed {test_index}/{len(test_trips)}", flush=True)
            print(
                f"Completed {dataset_spec.name} user {driver_number}/{len(driver_ids)}",
                flush=True,
            )

    output = pd.DataFrame(rows)
    observed_cases = output[KEYS].drop_duplicates().shape[0]
    if observed_cases != expected_cases:
        raise ValueError(f"Output has {observed_cases:,} cases; expected {expected_cases:,}")
    expected_rows = expected_cases * len(ARMS) * len(METHODS)
    if len(output) != expected_rows:
        raise ValueError(f"Output has {len(output):,} rows; expected {expected_rows:,}")
    output.to_csv(args.output, index=False)
    print(f"Wrote {args.output} ({len(output):,} rows)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
