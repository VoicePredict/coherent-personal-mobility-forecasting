#!/usr/bin/env python3
"""Evaluate guarded a1 and top-15 personal route-analogue policies.

The frozen model is imported without modification.  Candidate choice uses only
the original train-derived similarity order and the predeclared geometry gate.
Ground-truth test outcomes are accessed only after selection, by the unchanged
evaluation function.
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
METHODS = ["route_analogue_a1_guarded", "route_analogue_top15_guarded"]
ARMS = {
    "combined_1000m": {"max_start_gap_m": 1000.0, "max_endpoint_gap_m": 500.0},
    "combined_500m": {"max_start_gap_m": 500.0, "max_endpoint_gap_m": 500.0},
    "combined_350m": {"max_start_gap_m": 350.0, "max_endpoint_gap_m": 500.0},
}


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Python module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_analogue_candidate(
    model: Any,
    analogue: Any,
    *,
    rank: int,
    cluster_scores: dict[int, float],
    trip_to_cluster: dict[int, int],
) -> Any:
    cluster_id = trip_to_cluster.get(int(analogue.trip.trip_id))
    analogue.cluster_id = cluster_id
    end_lat, end_lon = analogue.trip.endpoint_latlon
    end_x, end_y = analogue.trip.endpoint_xy
    return model.Candidate(
        name=f"route_analogue_top{rank}",
        family="route_analogue",
        dest_x=float(end_x),
        dest_y=float(end_y),
        dest_lat=float(end_lat),
        dest_lon=float(end_lon),
        route_trip=analogue.trip,
        route_index=int(analogue.index),
        dest_score=float(cluster_scores.get(cluster_id, analogue.score)),
        route_score=float(analogue.score),
        confidence=float(analogue.score),
        cluster_id=cluster_id,
        source="similarity_ranked_personal_route_analogue_top15",
    )


def candidate_geometry(model: Any, candidate: Any, test_trip: Any, prefix_idx: int) -> tuple[float, float]:
    if candidate.route_trip is None or candidate.route_index is None:
        return float("nan"), float("nan")
    route_index = int(candidate.route_index)
    start_gap = model.euclid(
        float(test_trip.x[prefix_idx]),
        float(test_trip.y[prefix_idx]),
        float(candidate.route_trip.x[route_index]),
        float(candidate.route_trip.y[route_index]),
    )
    route_end_x, route_end_y = candidate.route_trip.endpoint_xy
    endpoint_gap = model.euclid(
        float(route_end_x),
        float(route_end_y),
        float(candidate.dest_x),
        float(candidate.dest_y),
    )
    return float(start_gap), float(endpoint_gap)


def first_admissible(
    candidates: list[Any],
    geometries: list[tuple[float, float]],
    *,
    max_start_gap_m: float,
    max_endpoint_gap_m: float,
) -> tuple[Any | None, int | None, int]:
    admissible_count = 0
    selected: Any | None = None
    selected_rank: int | None = None
    for rank, (candidate, (start_gap, endpoint_gap)) in enumerate(
        zip(candidates, geometries), start=1
    ):
        admissible = (
            math.isfinite(start_gap)
            and start_gap <= max_start_gap_m
            and math.isfinite(endpoint_gap)
            and endpoint_gap <= max_endpoint_gap_m
        )
        if admissible:
            admissible_count += 1
            if selected is None:
                selected = candidate
                selected_rank = rank
    return selected, selected_rank, admissible_count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-script", type=Path, required=True)
    parser.add_argument("--reference-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=15)
    args = parser.parse_args()
    if args.top_k != 15:
        raise ValueError("The frozen protocol requires top-k=15")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    model = load_module(args.model_script, "frozen_top15_analogue_model")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    reference = pd.read_csv(args.reference_predictions, usecols=KEYS).drop_duplicates()
    if len(reference) != 8526:
        raise ValueError(f"Expected 8,526 reference cases, found {len(reference):,}")

    exp = config["experiment"]
    if int(exp.get("destination_analogue_top_k", 15)) != 15:
        raise ValueError("Frozen configuration does not use destination_analogue_top_k=15")
    observed_fraction = float(exp.get("observed_fraction", 0.66))
    min_trips = int(exp.get("min_trips_per_driver", 20))
    train_fraction = float(exp.get("train_fraction", 0.75))
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
                split_spec, projection_meta = model.projection_for_split(
                    dataset_spec, train_trips, config
                )
                model.reproject_trips(driver_trips, split_spec.epsg)
                clusters, trip_to_cluster = model.train_destination_clusters(
                    train_trips,
                    config,
                    transformer_to_latlon=Transformer.from_crs(
                        split_spec.epsg, "EPSG:4326", always_xy=True
                    ),
                )
                print(
                    f"{dataset_spec.name} user={driver_id} split={split_label}: "
                    f"train={len(train_trips)} test={len(test_trips)}",
                    flush=True,
                )

                for test_index, test_trip in enumerate(test_trips, start=1):
                    prefix_idx = model.observed_index(test_trip, observed_fraction)
                    analogues = model.retrieve_analogues(test_trip, train_trips, config)
                    scores = model.cluster_scores(
                        analogues, clusters, trip_to_cluster, test_trip, config
                    )
                    top_analogues = analogues[: args.top_k]
                    candidates = [
                        make_analogue_candidate(
                            model,
                            analogue,
                            rank=rank,
                            cluster_scores=scores,
                            trip_to_cluster=trip_to_cluster,
                        )
                        for rank, analogue in enumerate(top_analogues, start=1)
                    ]
                    geometries = [
                        candidate_geometry(model, candidate, test_trip, prefix_idx)
                        for candidate in candidates
                    ]

                    for arm_name, arm in ARMS.items():
                        top15, top15_rank, admissible_count = first_admissible(
                            candidates,
                            geometries,
                            max_start_gap_m=float(arm["max_start_gap_m"]),
                            max_endpoint_gap_m=float(arm["max_endpoint_gap_m"]),
                        )
                        a1_candidates = candidates[:1]
                        a1_geometries = geometries[:1]
                        a1, a1_rank, _ = first_admissible(
                            a1_candidates,
                            a1_geometries,
                            max_start_gap_m=float(arm["max_start_gap_m"]),
                            max_endpoint_gap_m=float(arm["max_endpoint_gap_m"]),
                        )
                        selections = {
                            "route_analogue_a1_guarded": (a1, a1_rank),
                            "route_analogue_top15_guarded": (top15, top15_rank),
                        }
                        for method, (candidate, selected_rank) in selections.items():
                            emitted = candidate is not None
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
                                emitted=emitted,
                                train_count=len(train_trips),
                                test_count=len(test_trips),
                                analogues=analogues,
                                clusters=clusters,
                                config=config,
                            )
                            if candidate is None or selected_rank is None:
                                selected_start_gap = float("nan")
                                selected_endpoint_gap = float("nan")
                                selected_similarity = float("nan")
                            else:
                                selected_start_gap, selected_endpoint_gap = geometries[
                                    selected_rank - 1
                                ]
                                selected_similarity = float(
                                    top_analogues[selected_rank - 1].score
                                )
                            row.update(projection_meta)
                            row.update(
                                {
                                    "arm": arm_name,
                                    "max_start_gap_m": float(arm["max_start_gap_m"]),
                                    "max_endpoint_gap_m": float(arm["max_endpoint_gap_m"]),
                                    "analogue_budget": args.top_k,
                                    "retrieved_analogue_count": len(analogues),
                                    "considered_analogue_count": len(top_analogues),
                                    "admissible_analogue_count_top15": admissible_count,
                                    "selected_analogue_rank": selected_rank,
                                    "selected_similarity_score": selected_similarity,
                                    "selected_route_start_gap_m": selected_start_gap,
                                    "selected_endpoint_discrepancy_m": selected_endpoint_gap,
                                    "selection_backed_off": int(
                                        selected_rank is not None and selected_rank > 1
                                    ),
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
    observed_keys = output[KEYS].drop_duplicates().sort_values(KEYS).reset_index(drop=True)
    reference_keys = reference.sort_values(KEYS).reset_index(drop=True)
    if not observed_keys.equals(reference_keys):
        raise ValueError("Generated case keys do not match the frozen 8,526-case reference")
    expected_rows = len(reference_keys) * len(ARMS) * len(METHODS)
    if len(output) != expected_rows:
        raise ValueError(f"Output has {len(output):,} rows; expected {expected_rows:,}")
    output.to_csv(args.output, index=False)
    print(f"Wrote {args.output} ({len(output):,} rows)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
