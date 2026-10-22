#!/usr/bin/env python3
"""Run the frozen forecaster with the final candidate-admissibility rule."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = ROOT / "scripts" / "geometry_guard.py"


def load_base() -> Any:
    spec = importlib.util.spec_from_file_location("base_guarded_reselection", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def route_endpoint(candidate: Any) -> tuple[float, float] | None:
    if candidate.route_x is not None and candidate.route_y is not None and len(candidate.route_x):
        return float(candidate.route_x[-1]), float(candidate.route_y[-1])
    if candidate.route_trip is not None:
        x, y = candidate.route_trip.endpoint_xy
        return float(x), float(y)
    return None


def final_is_eligible(candidate: Any, geometry: dict[str, Any], arm: dict[str, Any]) -> bool:
    endpoint = route_endpoint(candidate)
    if endpoint is None:
        return False
    discrepancy = math.hypot(endpoint[0] - float(candidate.dest_x), endpoint[1] - float(candidate.dest_y))
    if not math.isfinite(discrepancy) or discrepancy > float(arm["max_endpoint_gap_m"]):
        return False

    start_gap = float(geometry["route_start_gap_m"])
    if not math.isfinite(start_gap) or start_gap > float(arm["max_start_gap_m"]):
        return False

    if candidate.family == "grid_transition" and bool(arm["require_grid_reach"]):
        return int(geometry["grid_reached_destination"]) == 1
    return True


def main() -> int:
    base = load_base()
    base.ARMS = {
        "combined_1000m": {
            "require_grid_reach": True,
            "max_start_gap_m": 1000.0,
            "max_endpoint_gap_m": 500.0,
        },
        "combined_500m": {
            "require_grid_reach": True,
            "max_start_gap_m": 500.0,
            "max_endpoint_gap_m": 500.0,
        },
        "combined_350m": {
            "require_grid_reach": True,
            "max_start_gap_m": 350.0,
            "max_endpoint_gap_m": 500.0,
        },
    }
    base.is_eligible = final_is_eligible
    return int(base.main())


if __name__ == "__main__":
    raise SystemExit(main())
