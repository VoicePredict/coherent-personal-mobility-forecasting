#!/usr/bin/env python3
"""Coherent personal mobility forecasting experiment.

This script is intentionally self-contained: it does not import historical
experiment notebooks or versioned experiment code. It reads trajectory CSVs,
builds per-driver train/test splits, learns destination clusters from train
endpoints, retrieves personal route analogues, produces several forecasting
policies, and evaluates destination, route, LookAhead, and coherence metrics.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from sklearn.cluster import DBSCAN


ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "config"
OUT_DIR = ROOT / "outputs" / "private" / "base_forecast"


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else ROOT / path


def project_relative_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    label: str
    path: Path
    epsg: str
    kind: str


@dataclass
class Trip:
    dataset: str
    driver_id: int
    driver_name: str
    trip_id: int
    lat: np.ndarray
    lon: np.ndarray
    x: np.ndarray
    y: np.ndarray
    t_s: np.ndarray
    start_time: pd.Timestamp
    mode: str = ""

    @property
    def n(self) -> int:
        return int(len(self.x))

    @property
    def duration_s(self) -> float:
        return float(self.t_s[-1] - self.t_s[0]) if self.n else 0.0

    @property
    def endpoint_xy(self) -> tuple[float, float]:
        return float(self.x[-1]), float(self.y[-1])

    @property
    def endpoint_latlon(self) -> tuple[float, float]:
        return float(self.lat[-1]), float(self.lon[-1])

    @property
    def start_xy(self) -> tuple[float, float]:
        return float(self.x[0]), float(self.y[0])

    @property
    def start_hour(self) -> float:
        ts = self.start_time
        return float(ts.hour + ts.minute / 60.0 + ts.second / 3600.0)


@dataclass
class DestinationCluster:
    cluster_id: int
    support: int
    center_x: float
    center_y: float
    center_lat: float
    center_lon: float
    radius_p90_m: float
    train_trip_ids: set[int]


@dataclass
class Analogue:
    trip: Trip
    index: int
    score: float
    current_distance_m: float
    start_distance_m: float
    direction_score: float
    speed_score: float
    time_score: float
    cluster_id: int | None = None


@dataclass
class Candidate:
    name: str
    family: str
    dest_x: float
    dest_y: float
    dest_lat: float
    dest_lon: float
    route_trip: Trip | None
    route_index: int | None
    dest_score: float
    route_score: float
    confidence: float
    cluster_id: int | None
    source: str
    route_x: np.ndarray | None = None
    route_y: np.ndarray | None = None
    route_lat: np.ndarray | None = None
    route_lon: np.ndarray | None = None
    lookahead_x: float | None = None
    lookahead_y: float | None = None
    lookahead_lat: float | None = None
    lookahead_lon: float | None = None


@dataclass
class GridModel:
    dataset: str
    driver_id: int
    driver_name: str
    split_label: str
    epsg: str
    cell_size_m: float
    origin_x: float
    origin_y: float
    transformer_to_latlon: Transformer
    transition_counts: dict[str, dict[str, int]]
    destination_counts: dict[str, int]
    cell_visit_counts: dict[str, int]
    train_sequences_by_trip: dict[int, list[str]]


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def train_centered_aeqd_crs(train_trips: list[Trip]) -> tuple[str, float, float]:
    """Build a metric CRS from train data only, without a fixed UTM zone."""
    if not train_trips:
        raise ValueError("Cannot build a train-centred projection without train trips.")
    trip_center_lat = np.array([float(np.median(trip.lat)) for trip in train_trips], dtype=float)
    trip_center_lon = np.array([float(np.median(trip.lon)) for trip in train_trips], dtype=float)
    center_lat = float(np.median(trip_center_lat))
    center_lon = float(np.median(trip_center_lon))
    crs_definition = (
        f"+proj=aeqd +lat_0={center_lat:.10f} +lon_0={center_lon:.10f} "
        "+datum=WGS84 +units=m +no_defs"
    )
    CRS.from_user_input(crs_definition)
    return crs_definition, center_lat, center_lon


def reproject_trips(trips: list[Trip], target_crs: str | CRS) -> None:
    """Recompute metric coordinates from the immutable WGS84 arrays."""
    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    for trip in trips:
        x, y = transformer.transform(trip.lon, trip.lat)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(
                f"Projection produced non-finite coordinates for "
                f"{trip.dataset} driver={trip.driver_id} trip={trip.trip_id}."
            )
        trip.x = np.asarray(x, dtype=float)
        trip.y = np.asarray(y, dtype=float)


def projection_for_split(
    spec: DatasetSpec,
    train_trips: list[Trip],
    config: dict[str, Any],
) -> tuple[DatasetSpec, dict[str, Any]]:
    strategy = str(config["experiment"].get("projection_strategy", "fixed")).strip().lower()
    if strategy == "fixed" or spec.kind != "geolife":
        crs = CRS.from_user_input(spec.epsg)
        return spec, {
            "projection_strategy": "fixed",
            "projection_crs": crs.to_string(),
            "projection_center_lat": float("nan"),
            "projection_center_lon": float("nan"),
        }
    if strategy != "train_centered_aeqd":
        raise ValueError(
            f"Unknown projection_strategy={strategy!r}; expected fixed or train_centered_aeqd."
        )
    crs_definition, center_lat, center_lon = train_centered_aeqd_crs(train_trips)
    split_spec = replace(spec, epsg=crs_definition)
    return split_spec, {
        "projection_strategy": strategy,
        "projection_crs": crs_definition,
        "projection_center_lat": center_lat,
        "projection_center_lon": center_lon,
    }


def cyclic_hour_delta(a: float, b: float) -> float:
    raw = abs(float(a) - float(b)) % 24.0
    return min(raw, 24.0 - raw)


def euclid(ax: float, ay: float, bx: float, by: float) -> float:
    return float(math.hypot(float(ax) - float(bx), float(ay) - float(by)))


def path_length(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    return float(np.hypot(np.diff(x), np.diff(y)).sum())


def point_at_elapsed(trip: Trip, start_idx: int, elapsed_s: float) -> tuple[float, float, float, float]:
    target_t = float(trip.t_s[start_idx] + max(0.0, elapsed_s))
    if target_t >= float(trip.t_s[-1]):
        return float(trip.x[-1]), float(trip.y[-1]), float(trip.lat[-1]), float(trip.lon[-1])
    idx = int(np.searchsorted(trip.t_s, target_t, side="right"))
    idx = max(1, min(idx, trip.n - 1))
    t0 = float(trip.t_s[idx - 1])
    t1 = float(trip.t_s[idx])
    alpha = 0.0 if t1 <= t0 else (target_t - t0) / (t1 - t0)
    x = float(trip.x[idx - 1] + alpha * (trip.x[idx] - trip.x[idx - 1]))
    y = float(trip.y[idx - 1] + alpha * (trip.y[idx] - trip.y[idx - 1]))
    lat = float(trip.lat[idx - 1] + alpha * (trip.lat[idx] - trip.lat[idx - 1]))
    lon = float(trip.lon[idx - 1] + alpha * (trip.lon[idx] - trip.lon[idx - 1]))
    return x, y, lat, lon


def resample_suffix_xy(trip: Trip, start_idx: int, steps: int) -> np.ndarray:
    start_idx = max(0, min(int(start_idx), trip.n - 1))
    xs = trip.x[start_idx:]
    ys = trip.y[start_idx:]
    if len(xs) == 0:
        return np.empty((0, 2), dtype=float)
    if len(xs) == 1 or steps <= 1:
        return np.array([[float(xs[-1]), float(ys[-1])]], dtype=float)
    d = np.hypot(np.diff(xs), np.diff(ys))
    s = np.concatenate([[0.0], np.cumsum(d)])
    if s[-1] <= 1e-9:
        return np.repeat(np.array([[float(xs[-1]), float(ys[-1])]]), steps, axis=0)
    targets = np.linspace(0.0, float(s[-1]), int(steps))
    out_x = np.interp(targets, s, xs)
    out_y = np.interp(targets, s, ys)
    return np.column_stack([out_x, out_y])


def resample_polyline_xy(points: np.ndarray, steps: int) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0, 2), dtype=float)
    if len(points) == 1 or steps <= 1:
        return np.repeat(points[-1:, :], max(1, steps), axis=0)
    d = np.hypot(np.diff(points[:, 0]), np.diff(points[:, 1]))
    s = np.concatenate([[0.0], np.cumsum(d)])
    if s[-1] <= 1e-9:
        return np.repeat(points[-1:, :], int(steps), axis=0)
    targets = np.linspace(0.0, float(s[-1]), int(steps))
    return np.column_stack([
        np.interp(targets, s, points[:, 0]),
        np.interp(targets, s, points[:, 1]),
    ])


def interpolate_polyline_by_distance(points: np.ndarray, distance_m: float) -> tuple[float, float]:
    if len(points) == 0:
        return float("nan"), float("nan")
    if len(points) == 1:
        return float(points[0, 0]), float(points[0, 1])
    d = np.hypot(np.diff(points[:, 0]), np.diff(points[:, 1]))
    s = np.concatenate([[0.0], np.cumsum(d)])
    if s[-1] <= 1e-9:
        return float(points[-1, 0]), float(points[-1, 1])
    target = max(0.0, min(float(distance_m), float(s[-1])))
    return float(np.interp(target, s, points[:, 0])), float(np.interp(target, s, points[:, 1]))


def grid_cell_id(model: GridModel, x: float, y: float) -> str:
    ix = int(math.floor((float(x) - model.origin_x) / model.cell_size_m))
    iy = int(math.floor((float(y) - model.origin_y) / model.cell_size_m))
    return f"{ix}:{iy}"


def parse_cell_id(cell_id: str) -> tuple[int, int]:
    left, right = str(cell_id).split(":", 1)
    return int(left), int(right)


def grid_cell_center_xy(model: GridModel, cell_id: str) -> tuple[float, float]:
    ix, iy = parse_cell_id(cell_id)
    return (
        float(model.origin_x + (ix + 0.5) * model.cell_size_m),
        float(model.origin_y + (iy + 0.5) * model.cell_size_m),
    )


def grid_cell_center_latlon(model: GridModel, cell_id: str) -> tuple[float, float]:
    x, y = grid_cell_center_xy(model, cell_id)
    lon, lat = model.transformer_to_latlon.transform(x, y)
    return float(lat), float(lon)


def trip_cell_sequence(trip: Trip, model: GridModel) -> list[str]:
    raw = [grid_cell_id(model, x, y) for x, y in zip(trip.x, trip.y)]
    compressed: list[str] = []
    prev = None
    for cell in raw:
        if cell != prev:
            compressed.append(cell)
            prev = cell
    return compressed


def build_grid_model(
    *,
    spec: DatasetSpec,
    driver_id: int,
    driver_name: str,
    split_label: str,
    train_trips: list[Trip],
    config: dict[str, Any],
) -> GridModel:
    exp = config["experiment"]
    cell_size = float(exp.get("grid_cell_size_m", 350.0))
    all_x = np.concatenate([trip.x for trip in train_trips]) if train_trips else np.array([0.0])
    all_y = np.concatenate([trip.y for trip in train_trips]) if train_trips else np.array([0.0])
    origin_x = math.floor(float(np.nanmin(all_x)) / cell_size) * cell_size
    origin_y = math.floor(float(np.nanmin(all_y)) / cell_size) * cell_size
    model = GridModel(
        dataset=spec.name,
        driver_id=driver_id,
        driver_name=driver_name,
        split_label=split_label,
        epsg=spec.epsg,
        cell_size_m=cell_size,
        origin_x=float(origin_x),
        origin_y=float(origin_y),
        transformer_to_latlon=Transformer.from_crs(spec.epsg, "EPSG:4326", always_xy=True),
        transition_counts={},
        destination_counts={},
        cell_visit_counts={},
        train_sequences_by_trip={},
    )
    for trip in train_trips:
        seq = trip_cell_sequence(trip, model)
        if not seq:
            continue
        model.train_sequences_by_trip[int(trip.trip_id)] = seq
        for cell in seq:
            model.cell_visit_counts[cell] = model.cell_visit_counts.get(cell, 0) + 1
        end_cell = seq[-1]
        model.destination_counts[end_cell] = model.destination_counts.get(end_cell, 0) + 1
        for src, dst in zip(seq, seq[1:]):
            model.transition_counts.setdefault(src, {})
            model.transition_counts[src][dst] = model.transition_counts[src].get(dst, 0) + 1
    return model


def transition_probabilities(model: GridModel, cell_id: str) -> list[tuple[str, float, int]]:
    counts = model.transition_counts.get(cell_id, {})
    total = float(sum(counts.values()))
    if total <= 0:
        return []
    out = [(dst, count / total, count) for dst, count in counts.items()]
    out.sort(key=lambda item: (-item[1], item[0]))
    return out


def transition_reachability(model: GridModel, start_cell: str, max_steps: int) -> dict[str, float]:
    mass: dict[str, float] = {start_cell: 1.0}
    reached: dict[str, float] = {start_cell: 1.0}
    for _ in range(max(1, int(max_steps))):
        next_mass: dict[str, float] = {}
        for cell, value in mass.items():
            transitions = transition_probabilities(model, cell)
            if not transitions:
                continue
            for dst, prob, _ in transitions:
                next_mass[dst] = next_mass.get(dst, 0.0) + value * prob
        if not next_mass:
            break
        for cell, value in next_mass.items():
            reached[cell] = reached.get(cell, 0.0) + value
        mass = next_mass
    total = max(reached.values()) if reached else 0.0
    if total > 0:
        return {cell: value / total for cell, value in reached.items()}
    return reached


def beam_search_transition_route(
    model: GridModel,
    *,
    start_cell: str,
    destination_cell: str,
    max_steps: int,
    beam_width: int,
) -> tuple[list[str], float]:
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
            transitions = transition_probabilities(model, current)
            if not transitions:
                continue
            for dst, prob, _ in transitions[: max(1, int(beam_width))]:
                loop_penalty = 0.65 if dst in path else 1.0
                expanded.append((path + [dst], score * prob * loop_penalty))
        if not expanded:
            break
        expanded.sort(key=lambda item: item[1], reverse=True)
        beam = expanded[: max(1, int(beam_width))]
        completed.extend(item for item in beam if item[0][-1] == destination_cell)
    pool = completed if completed else beam
    if not pool:
        return [start_cell, destination_cell], 0.0

    def final_score(item: tuple[list[str], float]) -> float:
        path, score = item
        end_x, end_y = grid_cell_center_xy(model, path[-1])
        dst_x, dst_y = grid_cell_center_xy(model, destination_cell)
        distance_penalty = math.exp(-euclid(end_x, end_y, dst_x, dst_y) / max(1.0, model.cell_size_m))
        destination_bonus = 1.0 if path[-1] == destination_cell else 0.5
        return score * (0.5 + 0.5 * distance_penalty) * destination_bonus

    best_path, best_score = max(pool, key=final_score)
    if best_path[-1] != destination_cell:
        best_path = best_path + [destination_cell]
    return best_path, float(best_score)


def route_arrays_from_cells(model: GridModel, cells: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xy = np.array([grid_cell_center_xy(model, cell) for cell in cells], dtype=float)
    if len(xy) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    lon, lat = model.transformer_to_latlon.transform(xy[:, 0], xy[:, 1])
    return xy[:, 0], xy[:, 1], np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)


def route_step_accuracy(
    pred_trip: Trip | None,
    pred_start_idx: int | None,
    true_trip: Trip,
    true_start_idx: int,
    *,
    steps: int,
    threshold_m: float,
) -> float:
    if pred_trip is None or pred_start_idx is None:
        return float("nan")
    pred = resample_suffix_xy(pred_trip, pred_start_idx, steps)
    true = resample_suffix_xy(true_trip, true_start_idx, steps)
    n = min(len(pred), len(true))
    if n == 0:
        return float("nan")
    dist = np.hypot(pred[:n, 0] - true[:n, 0], pred[:n, 1] - true[:n, 1])
    return float(np.mean(dist <= threshold_m))


def candidate_route_step_accuracy(
    candidate: Candidate,
    true_trip: Trip,
    true_start_idx: int,
    *,
    steps: int,
    threshold_m: float,
) -> float:
    if candidate.route_trip is not None and candidate.route_index is not None:
        return route_step_accuracy(
            candidate.route_trip,
            candidate.route_index,
            true_trip,
            true_start_idx,
            steps=steps,
            threshold_m=threshold_m,
        )
    if candidate.route_x is None or candidate.route_y is None:
        return float("nan")
    pred_points = np.column_stack([candidate.route_x, candidate.route_y])
    pred = resample_polyline_xy(pred_points, steps)
    true = resample_suffix_xy(true_trip, true_start_idx, steps)
    n = min(len(pred), len(true))
    if n == 0:
        return float("nan")
    dist = np.hypot(pred[:n, 0] - true[:n, 0], pred[:n, 1] - true[:n, 1])
    return float(np.mean(dist <= threshold_m))


def read_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dataset_specs(config: dict[str, Any]) -> list[DatasetSpec]:
    specs = []
    for item in config["datasets"]:
        specs.append(
            DatasetSpec(
                name=str(item["name"]),
                label=str(item.get("label", item["name"])),
                path=resolve_project_path(str(item["path"])),
                epsg=str(item["epsg"]),
                kind=str(item.get("kind", item["name"])),
            )
        )
    return specs


def load_trips(spec: DatasetSpec, config: dict[str, Any]) -> list[Trip]:
    header = pd.read_csv(spec.path, nrows=0).columns.tolist()
    needed = ["ID", "trip_id", "latitude", "longitude"]
    optional = ["datetime", "date", "time", "Transportation Mode", "char"]
    usecols = [c for c in needed + optional if c in header]
    df = pd.read_csv(spec.path, usecols=usecols, low_memory=False)
    df = df.rename(columns={"ID": "driver_id"})
    df["driver_id"] = pd.to_numeric(df["driver_id"], errors="coerce")
    df["trip_id"] = pd.to_numeric(df["trip_id"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["driver_id", "trip_id", "latitude", "longitude"]).copy()
    df["driver_id"] = df["driver_id"].astype(int)
    df["trip_id"] = df["trip_id"].astype(int)

    if "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"], errors="coerce")
    elif "date" in df.columns and "time" in df.columns:
        dt = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), errors="coerce")
    else:
        dt = pd.Series(pd.NaT, index=df.index)
    df["timestamp"] = dt
    df = df.dropna(subset=["timestamp"]).copy()
    df = df[
        df["latitude"].between(-90.0, 90.0)
        & df["longitude"].between(-180.0, 180.0)
    ].copy()

    transformer = Transformer.from_crs("EPSG:4326", spec.epsg, always_xy=True)
    x, y = transformer.transform(df["longitude"].to_numpy(), df["latitude"].to_numpy())
    df["x"] = x
    df["y"] = y
    df = df.sort_values(["driver_id", "trip_id", "timestamp"]).reset_index(drop=True)

    min_points = int(config["experiment"].get("min_points_per_trip", 10))
    min_duration = float(config["experiment"].get("min_duration_min", 5.0)) * 60.0
    trips: list[Trip] = []
    for (driver_id, trip_id), group in df.groupby(["driver_id", "trip_id"], sort=False):
        if len(group) < min_points:
            continue
        t_ns = group["timestamp"].astype("int64").to_numpy()
        t_s = (t_ns - t_ns[0]) / 1_000_000_000.0
        if float(t_s[-1] - t_s[0]) < min_duration:
            continue
        if "char" in group.columns and group["char"].notna().any():
            driver_name = str(group["char"].dropna().astype(str).mode().iloc[0])
        else:
            driver_name = f"user{int(driver_id)}"
        mode = ""
        if "Transportation Mode" in group.columns and group["Transportation Mode"].notna().any():
            mode = str(group["Transportation Mode"].dropna().astype(str).mode().iloc[0])
        trips.append(
            Trip(
                dataset=spec.name,
                driver_id=int(driver_id),
                driver_name=driver_name,
                trip_id=int(trip_id),
                lat=group["latitude"].to_numpy(dtype=float),
                lon=group["longitude"].to_numpy(dtype=float),
                x=group["x"].to_numpy(dtype=float),
                y=group["y"].to_numpy(dtype=float),
                t_s=t_s.astype(float),
                start_time=pd.Timestamp(group["timestamp"].iloc[0]),
                mode=mode,
            )
        )
    return trips


def split_driver_trips(
    trips: list[Trip],
    split_label: str,
    split_mode: str,
    train_fraction: float,
    seed: int,
) -> tuple[list[Trip], list[Trip]]:
    ordered = sorted(trips, key=lambda t: (t.start_time, t.trip_id))
    n = len(ordered)
    n_train = int(math.floor(n * train_fraction))
    n_train = max(1, min(n - 1, n_train))
    if split_mode == "chronological":
        return ordered[:n_train], ordered[n_train:]
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    train_idx = set(int(i) for i in indices[:n_train])
    train = [ordered[i] for i in range(n) if i in train_idx]
    test = [ordered[i] for i in range(n) if i not in train_idx]
    return train, test


def observed_index(trip: Trip, observed_fraction: float) -> int:
    idx = int(round((trip.n - 1) * float(observed_fraction)))
    return max(1, min(trip.n - 2, idx))


def trip_speed(trip: Trip, start_idx: int, end_idx: int) -> float:
    start_idx = max(0, min(start_idx, trip.n - 1))
    end_idx = max(start_idx + 1, min(end_idx, trip.n - 1))
    dist = path_length(trip.x[start_idx : end_idx + 1], trip.y[start_idx : end_idx + 1])
    dt = float(trip.t_s[end_idx] - trip.t_s[start_idx])
    return dist / dt if dt > 0 else 0.0


def direction_score(test: Trip, test_idx: int, train: Trip, train_idx: int) -> float:
    a = np.array([test.x[test_idx] - test.x[0], test.y[test_idx] - test.y[0]], dtype=float)
    b = np.array([train.x[train_idx] - train.x[0], train.y[train_idx] - train.y[0]], dtype=float)
    an = float(np.linalg.norm(a))
    bn = float(np.linalg.norm(b))
    if an <= 1e-9 or bn <= 1e-9:
        return 0.5
    cos = float(np.clip(np.dot(a, b) / (an * bn), -1.0, 1.0))
    return 0.5 + 0.5 * cos


def retrieve_analogues(
    test_trip: Trip,
    train_trips: list[Trip],
    config: dict[str, Any],
) -> list[Analogue]:
    exp = config["experiment"]
    obs = float(exp.get("observed_fraction", 0.66))
    test_idx = observed_index(test_trip, obs)
    test_speed = trip_speed(test_trip, 0, test_idx)
    current_sigma = float(exp.get("analogue_current_sigma_m", 1200.0))
    start_sigma = float(exp.get("analogue_start_sigma_m", 2500.0))
    time_sigma = float(exp.get("analogue_time_sigma_hours", 4.0))
    speed_sigma = float(exp.get("analogue_speed_sigma_log", 0.8))

    analogues: list[Analogue] = []
    for train in train_trips:
        idx = observed_index(train, obs)
        current_dist = euclid(test_trip.x[test_idx], test_trip.y[test_idx], train.x[idx], train.y[idx])
        start_dist = euclid(test_trip.x[0], test_trip.y[0], train.x[0], train.y[0])
        dir_score = direction_score(test_trip, test_idx, train, idx)
        train_speed = trip_speed(train, 0, idx)
        if test_speed > 1e-6 and train_speed > 1e-6:
            speed_score = math.exp(-abs(math.log(train_speed / test_speed)) / speed_sigma)
        else:
            speed_score = 0.5
        hour_delta = cyclic_hour_delta(test_trip.start_hour, train.start_hour)
        time_score = math.exp(-(hour_delta * hour_delta) / (2.0 * time_sigma * time_sigma))
        current_score = math.exp(-current_dist / current_sigma)
        start_score = math.exp(-start_dist / start_sigma)
        score = (
            0.42 * current_score
            + 0.18 * start_score
            + 0.20 * dir_score
            + 0.10 * speed_score
            + 0.10 * time_score
        )
        analogues.append(
            Analogue(
                trip=train,
                index=idx,
                score=float(score),
                current_distance_m=float(current_dist),
                start_distance_m=float(start_dist),
                direction_score=float(dir_score),
                speed_score=float(speed_score),
                time_score=float(time_score),
            )
        )
    analogues.sort(key=lambda a: a.score, reverse=True)
    return analogues


def train_destination_clusters(
    train_trips: list[Trip],
    config: dict[str, Any],
    transformer_to_latlon: Transformer | None = None,
) -> tuple[list[DestinationCluster], dict[int, int]]:
    exp = config["experiment"]
    eps = float(exp.get("destination_cluster_eps_m", 1000.0))
    min_radius = float(exp.get("destination_cluster_min_radius_m", 200.0))
    endpoints = np.array([t.endpoint_xy for t in train_trips], dtype=float)
    if len(endpoints) == 0:
        return [], {}
    labels = DBSCAN(eps=eps, min_samples=1).fit_predict(endpoints)
    trip_to_cluster: dict[int, int] = {}
    clusters: list[DestinationCluster] = []
    for cluster_id in sorted(set(int(x) for x in labels)):
        idx = np.where(labels == cluster_id)[0]
        pts = endpoints[idx]
        center_x = float(np.mean(pts[:, 0]))
        center_y = float(np.mean(pts[:, 1]))
        d = np.hypot(pts[:, 0] - center_x, pts[:, 1] - center_y)
        radius = max(min_radius, float(np.quantile(d, 0.9)) if len(d) else min_radius)
        if transformer_to_latlon is not None:
            center_lon, center_lat = transformer_to_latlon.transform(center_x, center_y)
        else:
            medoid_local = int(idx[int(np.argmin(d))]) if len(idx) else int(idx[0])
            center_lat, center_lon = train_trips[medoid_local].endpoint_latlon
        trip_ids = {int(train_trips[i].trip_id) for i in idx}
        for i in idx:
            trip_to_cluster[int(train_trips[i].trip_id)] = cluster_id
        clusters.append(
            DestinationCluster(
                cluster_id=cluster_id,
                support=int(len(idx)),
                center_x=center_x,
                center_y=center_y,
                center_lat=float(center_lat),
                center_lon=float(center_lon),
                radius_p90_m=radius,
                train_trip_ids=trip_ids,
            )
        )
    clusters.sort(key=lambda c: (-c.support, c.cluster_id))
    return clusters, trip_to_cluster


def nearest_cluster(
    x: float,
    y: float,
    clusters: list[DestinationCluster],
) -> tuple[DestinationCluster | None, float]:
    if not clusters:
        return None, float("nan")
    best = min(clusters, key=lambda c: euclid(x, y, c.center_x, c.center_y))
    return best, euclid(x, y, best.center_x, best.center_y)


def cluster_scores(
    analogues: list[Analogue],
    clusters: list[DestinationCluster],
    trip_to_cluster: dict[int, int],
    test_trip: Trip,
    config: dict[str, Any],
) -> dict[int, float]:
    top_k = int(config["experiment"].get("destination_analogue_top_k", 15))
    max_support = max((c.support for c in clusters), default=1)
    by_id = {c.cluster_id: c for c in clusters}
    mass: dict[int, float] = {c.cluster_id: 0.0 for c in clusters}
    for analogue in analogues[:top_k]:
        cid = trip_to_cluster.get(int(analogue.trip.trip_id))
        if cid is not None:
            mass[cid] += analogue.score
            analogue.cluster_id = cid
    max_mass = max(mass.values()) if mass else 0.0
    out: dict[int, float] = {}
    for cid, c in by_id.items():
        mass_score = mass[cid] / max_mass if max_mass > 0 else 0.0
        support_score = math.log1p(c.support) / math.log1p(max_support) if max_support > 0 else 0.0
        # Time support: average temporal proximity of train endpoints in this cluster.
        member_times = [a.time_score for a in analogues if trip_to_cluster.get(int(a.trip.trip_id)) == cid]
        time_score = float(np.mean(member_times)) if member_times else 0.0
        out[cid] = float(0.68 * mass_score + 0.22 * support_score + 0.10 * time_score)
    return out


def best_analogue_for_cluster(analogues: list[Analogue], cluster_id: int | None) -> Analogue | None:
    if cluster_id is None:
        return analogues[0] if analogues else None
    for analogue in analogues:
        if analogue.cluster_id == cluster_id:
            return analogue
    return analogues[0] if analogues else None


def grid_destination_scores(
    *,
    test_trip: Trip,
    prefix_idx: int,
    analogues: list[Analogue],
    grid_model: GridModel,
    config: dict[str, Any],
) -> dict[str, float]:
    if not grid_model.destination_counts:
        return {}
    exp = config["experiment"]
    start_cell = grid_cell_id(grid_model, test_trip.x[prefix_idx], test_trip.y[prefix_idx])
    max_steps = int(exp.get("grid_route_max_steps", 30))
    reach = transition_reachability(grid_model, start_cell, max_steps=max_steps)
    max_dest_count = max(grid_model.destination_counts.values()) if grid_model.destination_counts else 1

    analogue_top_k = int(exp.get("destination_analogue_top_k", 15))
    analogue_mass: dict[str, float] = {}
    for analogue in analogues[:analogue_top_k]:
        seq = grid_model.train_sequences_by_trip.get(int(analogue.trip.trip_id), [])
        if seq:
            analogue_mass[seq[-1]] = analogue_mass.get(seq[-1], 0.0) + analogue.score
    max_analogue_mass = max(analogue_mass.values()) if analogue_mass else 0.0

    out: dict[str, float] = {}
    for cell, count in grid_model.destination_counts.items():
        prior = count / max_dest_count if max_dest_count else 0.0
        reachable = reach.get(cell, 0.0)
        analogue_score = analogue_mass.get(cell, 0.0) / max_analogue_mass if max_analogue_mass > 0 else 0.0
        out[cell] = float(0.45 * reachable + 0.30 * prior + 0.25 * analogue_score)
    return out


def add_grid_transition_candidates(
    *,
    candidates: list[Candidate],
    test_trip: Trip,
    prefix_idx: int,
    analogues: list[Analogue],
    grid_model: GridModel | None,
    config: dict[str, Any],
) -> None:
    if grid_model is None:
        return
    scores = grid_destination_scores(
        test_trip=test_trip,
        prefix_idx=prefix_idx,
        analogues=analogues,
        grid_model=grid_model,
        config=config,
    )
    if not scores:
        return

    exp = config["experiment"]
    lookahead_s = float(exp.get("lookahead_minutes", 5.0)) * 60.0
    prefix_speed = trip_speed(test_trip, 0, prefix_idx)
    lookahead_distance = max(0.0, prefix_speed * lookahead_s)
    start_cell = grid_cell_id(grid_model, test_trip.x[prefix_idx], test_trip.y[prefix_idx])
    top_k = int(exp.get("grid_destination_top_k_candidates", 3))
    max_steps = int(exp.get("grid_route_max_steps", 30))
    beam_width = int(exp.get("grid_route_beam_width", 8))

    for rank, dest_cell in enumerate(sorted(scores, key=scores.get, reverse=True)[:top_k], start=1):
        dest_x, dest_y = grid_cell_center_xy(grid_model, dest_cell)
        dest_lat, dest_lon = grid_cell_center_latlon(grid_model, dest_cell)
        route_cells, route_score = beam_search_transition_route(
            grid_model,
            start_cell=start_cell,
            destination_cell=dest_cell,
            max_steps=max_steps,
            beam_width=beam_width,
        )
        route_x, route_y, route_lat, route_lon = route_arrays_from_cells(grid_model, route_cells)
        if len(route_x):
            la_x, la_y = interpolate_polyline_by_distance(
                np.column_stack([route_x, route_y]),
                lookahead_distance,
            )
            la_lon, la_lat = grid_model.transformer_to_latlon.transform(la_x, la_y)
        else:
            la_x, la_y, la_lat, la_lon = dest_x, dest_y, dest_lat, dest_lon
        dest_score = float(scores[dest_cell])
        route_quality = float(min(1.0, route_score * 10.0))
        candidates.append(
            Candidate(
                name="transition_grid_route" if rank == 1 else f"transition_grid_route_top{rank}",
                family="grid_transition",
                dest_x=dest_x,
                dest_y=dest_y,
                dest_lat=dest_lat,
                dest_lon=dest_lon,
                route_trip=None,
                route_index=None,
                dest_score=dest_score,
                route_score=route_quality,
                confidence=float(0.58 * dest_score + 0.42 * route_quality),
                cluster_id=None,
                source="train_grid_transition_matrix_and_destination_prior",
                route_x=route_x,
                route_y=route_y,
                route_lat=route_lat,
                route_lon=route_lon,
                lookahead_x=float(la_x),
                lookahead_y=float(la_y),
                lookahead_lat=float(la_lat),
                lookahead_lon=float(la_lon),
            )
        )


def make_candidates(
    test_trip: Trip,
    train_trips: list[Trip],
    clusters: list[DestinationCluster],
    trip_to_cluster: dict[int, int],
    analogues: list[Analogue],
    config: dict[str, Any],
    grid_model: GridModel | None = None,
) -> list[Candidate]:
    if not analogues:
        return []
    prefix_idx = observed_index(test_trip, float(config["experiment"].get("observed_fraction", 0.66)))
    scores = cluster_scores(analogues, clusters, trip_to_cluster, test_trip, config)
    by_id = {c.cluster_id: c for c in clusters}
    best_cluster_id = max(scores, key=scores.get) if scores else None
    frequent_cluster = clusters[0] if clusters else None
    best_analogue = analogues[0]
    best_analogue_cluster = trip_to_cluster.get(int(best_analogue.trip.trip_id))
    best_analogue.cluster_id = best_analogue_cluster

    candidates: list[Candidate] = []

    def add_cluster_candidate(name: str, family: str, cluster_id: int | None, source: str) -> None:
        if cluster_id is None or cluster_id not in by_id:
            return
        c = by_id[cluster_id]
        route = best_analogue_for_cluster(analogues, cluster_id)
        route_score = route.score if route is not None else 0.0
        dest_score = scores.get(cluster_id, 0.0)
        candidates.append(
            Candidate(
                name=name,
                family=family,
                dest_x=c.center_x,
                dest_y=c.center_y,
                dest_lat=c.center_lat,
                dest_lon=c.center_lon,
                route_trip=route.trip if route is not None else None,
                route_index=route.index if route is not None else None,
                dest_score=float(dest_score),
                route_score=float(route_score),
                confidence=float(0.60 * dest_score + 0.40 * route_score),
                cluster_id=cluster_id,
                source=source,
            )
        )

    add_cluster_candidate(
        "destination_only",
        "destination_cluster",
        best_cluster_id,
        "best_destination_cluster_with_route_analogue",
    )
    add_cluster_candidate(
        "most_frequent_destination",
        "destination_prior",
        frequent_cluster.cluster_id if frequent_cluster else None,
        "most_frequent_train_destination_cluster",
    )

    end_lat, end_lon = best_analogue.trip.endpoint_latlon
    end_x, end_y = best_analogue.trip.endpoint_xy
    candidates.append(
        Candidate(
            name="route_analogue",
            family="route_analogue",
            dest_x=end_x,
            dest_y=end_y,
            dest_lat=end_lat,
            dest_lon=end_lon,
            route_trip=best_analogue.trip,
            route_index=best_analogue.index,
            dest_score=float(scores.get(best_analogue_cluster, best_analogue.score)),
            route_score=float(best_analogue.score),
            confidence=float(best_analogue.score),
            cluster_id=best_analogue_cluster,
            source="nearest_personal_route_analogue",
        )
    )

    top_cluster_ids = sorted(scores, key=scores.get, reverse=True)[
        : int(config["experiment"].get("destination_top_k_candidates", 3))
    ]
    for rank, cid in enumerate(top_cluster_ids, start=1):
        add_cluster_candidate(
            f"destination_cluster_top{rank}",
            "destination_cluster_topk",
            cid,
            "top_destination_cluster_candidate",
        )
    add_grid_transition_candidates(
        candidates=candidates,
        test_trip=test_trip,
        prefix_idx=prefix_idx,
        analogues=analogues,
        grid_model=grid_model,
        config=config,
    )
    return candidates


def candidate_route_endpoint(candidate: Candidate) -> tuple[float, float]:
    if candidate.route_x is not None and candidate.route_y is not None and len(candidate.route_x):
        return float(candidate.route_x[-1]), float(candidate.route_y[-1])
    if candidate.route_trip is None:
        return candidate.dest_x, candidate.dest_y
    return candidate.route_trip.endpoint_xy


def candidate_conflict_m(candidate: Candidate) -> float:
    ex, ey = candidate_route_endpoint(candidate)
    return euclid(ex, ey, candidate.dest_x, candidate.dest_y)


def select_coherent_product(candidates: list[Candidate], config: dict[str, Any]) -> Candidate | None:
    if not candidates:
        return None
    threshold = float(config["experiment"].get("coherence_threshold_m", 500.0))

    def score(c: Candidate) -> float:
        conflict = candidate_conflict_m(c)
        coherence = math.exp(-conflict / max(1.0, threshold))
        return float(0.50 * c.dest_score + 0.35 * c.route_score + 0.15 * coherence)

    coherent = [c for c in candidates if candidate_conflict_m(c) <= threshold]
    pool = coherent if coherent else candidates
    return max(pool, key=score)


def select_named(candidates: list[Candidate], name: str) -> Candidate | None:
    for c in candidates:
        if c.name == name:
            return c
    return None


def true_lookahead_point(test_trip: Trip, prefix_idx: int, lookahead_min: float) -> tuple[float, float, float, float]:
    return point_at_elapsed(test_trip, prefix_idx, lookahead_min * 60.0)


def predicted_lookahead_point(candidate: Candidate, lookahead_min: float) -> tuple[float, float, float, float]:
    if (
        candidate.lookahead_x is not None
        and candidate.lookahead_y is not None
        and candidate.lookahead_lat is not None
        and candidate.lookahead_lon is not None
    ):
        return (
            float(candidate.lookahead_x),
            float(candidate.lookahead_y),
            float(candidate.lookahead_lat),
            float(candidate.lookahead_lon),
        )
    if candidate.route_trip is None or candidate.route_index is None:
        return candidate.dest_x, candidate.dest_y, candidate.dest_lat, candidate.dest_lon
    return point_at_elapsed(candidate.route_trip, candidate.route_index, lookahead_min * 60.0)


def evaluate_candidate(
    *,
    dataset: str,
    dataset_label: str,
    driver_id: int,
    driver_name: str,
    split_label: str,
    method: str,
    test_trip: Trip,
    prefix_idx: int,
    candidate: Candidate | None,
    emitted: bool,
    train_count: int,
    test_count: int,
    analogues: list[Analogue],
    clusters: list[DestinationCluster],
    config: dict[str, Any],
) -> dict[str, Any]:
    lookahead_min = float(config["experiment"].get("lookahead_minutes", 5.0))
    route_steps = int(config["experiment"].get("route_eval_steps", 10))
    route_threshold = float(config["experiment"].get("route_step_threshold_m", 350.0))
    analogue_threshold = float(config["experiment"].get("train_analogue_score_threshold", 0.45))
    best_analogue_score = analogues[0].score if analogues else 0.0
    has_train_analogue = int(best_analogue_score >= analogue_threshold)

    true_x, true_y = test_trip.endpoint_xy
    true_lat, true_lon = test_trip.endpoint_latlon
    true_cluster, true_cluster_distance = nearest_cluster(true_x, true_y, clusters)
    true_cluster_covered = 0
    true_cluster_support = 0
    true_cluster_radius = float("nan")
    if true_cluster is not None:
        true_cluster_support = true_cluster.support
        true_cluster_radius = true_cluster.radius_p90_m
        true_cluster_covered = int(true_cluster_distance <= true_cluster.radius_p90_m)

    if candidate is None:
        candidate = Candidate(
            name="missing",
            family="missing",
            dest_x=float("nan"),
            dest_y=float("nan"),
            dest_lat=float("nan"),
            dest_lon=float("nan"),
            route_trip=None,
            route_index=None,
            dest_score=0.0,
            route_score=0.0,
            confidence=0.0,
            cluster_id=None,
            source="missing",
        )
        emitted = False

    dest_error_m = euclid(candidate.dest_x, candidate.dest_y, true_x, true_y)
    pred_cluster_distance = float("nan")
    cluster_hit = 0
    if true_cluster is not None:
        pred_cluster_distance = euclid(candidate.dest_x, candidate.dest_y, true_cluster.center_x, true_cluster.center_y)
        cluster_hit = int(true_cluster_covered == 1 and pred_cluster_distance <= true_cluster.radius_p90_m)

    true_la_x, true_la_y, true_la_lat, true_la_lon = true_lookahead_point(test_trip, prefix_idx, lookahead_min)
    pred_la_x, pred_la_y, pred_la_lat, pred_la_lon = predicted_lookahead_point(candidate, lookahead_min)
    la_error_m = euclid(pred_la_x, pred_la_y, true_la_x, true_la_y)
    route_acc = candidate_route_step_accuracy(
        candidate,
        test_trip,
        prefix_idx,
        steps=route_steps,
        threshold_m=route_threshold,
    )
    conflict_m = candidate_conflict_m(candidate)
    product_success = int(cluster_hit == 1 if true_cluster_covered else dest_error_m <= 1000.0)

    return {
        "dataset": dataset,
        "dataset_label": dataset_label,
        "driver_id": driver_id,
        "driver_name": driver_name,
        "split_label": split_label,
        "method": method,
        "trip_id": test_trip.trip_id,
        "train_trip_count": train_count,
        "test_trip_count": test_count,
        "prefix_points": int(prefix_idx + 1),
        "trip_points": test_trip.n,
        "emitted": int(emitted),
        "candidate_name": candidate.name,
        "candidate_family": candidate.family,
        "candidate_source": candidate.source,
        "candidate_confidence": candidate.confidence,
        "candidate_destination_score": candidate.dest_score,
        "candidate_route_score": candidate.route_score,
        "best_analogue_score": best_analogue_score,
        "has_train_analogue": has_train_analogue,
        "true_destination_lat": true_lat,
        "true_destination_lon": true_lon,
        "pred_destination_lat": candidate.dest_lat,
        "pred_destination_lon": candidate.dest_lon,
        "destination_error_m": dest_error_m,
        "dest_within_500m": int(dest_error_m <= 500.0),
        "dest_within_1000m": int(dest_error_m <= 1000.0),
        "true_cluster_id": None if true_cluster is None else true_cluster.cluster_id,
        "true_cluster_support": true_cluster_support,
        "true_cluster_radius_p90_m": true_cluster_radius,
        "true_cluster_covered_p90": true_cluster_covered,
        "pred_to_true_cluster_p90_m": pred_cluster_distance,
        "cluster_hit_p90_all": cluster_hit,
        "product_success": product_success,
        "true_lookahead_lat": true_la_lat,
        "true_lookahead_lon": true_la_lon,
        "pred_lookahead_lat": pred_la_lat,
        "pred_lookahead_lon": pred_la_lon,
        "lookahead_error_m": la_error_m,
        "la_within_500m": int(la_error_m <= 500.0),
        "la_within_1000m": int(la_error_m <= 1000.0),
        "route_step_accuracy": route_acc,
        "route_endpoint_to_destination_m": conflict_m,
        "route_destination_conflict_gt500": int(conflict_m > 500.0),
        "route_trip_id": None if candidate.route_trip is None else candidate.route_trip.trip_id,
    }


def predict_one_trip(
    *,
    dataset_label: str,
    split_label: str,
    train_trips: list[Trip],
    test_trips: list[Trip],
    test_trip: Trip,
    clusters: list[DestinationCluster],
    trip_to_cluster: dict[int, int],
    grid_model: GridModel | None,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    prefix_idx = observed_index(test_trip, float(config["experiment"].get("observed_fraction", 0.66)))
    analogues = retrieve_analogues(test_trip, train_trips, config)
    candidates = make_candidates(
        test_trip,
        train_trips,
        clusters,
        trip_to_cluster,
        analogues,
        config,
        grid_model=grid_model,
    )
    product = select_coherent_product(candidates, config)
    abstain_threshold = float(config["experiment"].get("abstain_confidence_threshold", 0.48))
    product_emitted = product is not None
    abstain_emitted = product is not None and product.confidence >= abstain_threshold

    selections: dict[str, tuple[Candidate | None, bool]] = {
        "destination_only": (select_named(candidates, "destination_only"), True),
        "route_analogue": (select_named(candidates, "route_analogue"), True),
        "transition_grid_route": (select_named(candidates, "transition_grid_route"), True),
        "most_frequent_destination": (select_named(candidates, "most_frequent_destination"), True),
        "coherent_product": (product, product_emitted),
        "coherent_abstain": (product, abstain_emitted),
    }
    rows = []
    for method, (candidate, emitted) in selections.items():
        rows.append(
            evaluate_candidate(
                dataset=test_trip.dataset,
                dataset_label=dataset_label,
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
        )
    return rows


def summarize(group: pd.DataFrame) -> dict[str, Any]:
    emitted = group[pd.to_numeric(group["emitted"], errors="coerce").fillna(0).eq(1)].copy()
    n_total = int(len(group))
    n_emit = int(len(emitted))

    def mean_col(col: str, frame: pd.DataFrame = emitted) -> float:
        if frame.empty or col not in frame.columns:
            return float("nan")
        return float(pd.to_numeric(frame[col], errors="coerce").mean())

    return {
        "n": n_total,
        "emitted": n_emit,
        "coverage_pct": 100.0 * n_emit / n_total if n_total else float("nan"),
        "product_success_pct": 100.0 * mean_col("product_success"),
        "cluster_hit_p90_all_pct": 100.0 * mean_col("cluster_hit_p90_all"),
        "dest_within_500m_pct": 100.0 * mean_col("dest_within_500m"),
        "dest_within_1000m_pct": 100.0 * mean_col("dest_within_1000m"),
        "la_within_500m_pct": 100.0 * mean_col("la_within_500m"),
        "la_within_1000m_pct": 100.0 * mean_col("la_within_1000m"),
        "route_step_accuracy_pct": 100.0 * mean_col("route_step_accuracy"),
        "conflict_gt500_pct": 100.0 * mean_col("route_destination_conflict_gt500"),
        "has_train_analogue_pct": 100.0 * mean_col("has_train_analogue"),
        "true_cluster_covered_pct": 100.0 * mean_col("true_cluster_covered_p90"),
        "median_destination_error_m": float(pd.to_numeric(emitted["destination_error_m"], errors="coerce").median()) if n_emit else float("nan"),
        "median_lookahead_error_m": float(pd.to_numeric(emitted["lookahead_error_m"], errors="coerce").median()) if n_emit else float("nan"),
    }


def write_summaries(rows: pd.DataFrame, out_dir: Path) -> None:
    summary_specs = [
        (["method"], "metrics_by_method.csv"),
        (["dataset", "method"], "metrics_by_dataset_method.csv"),
        (["dataset", "driver_id", "driver_name", "split_label", "method"], "metrics_by_driver_split_method.csv"),
        (["split_label", "method"], "metrics_by_split_method.csv"),
    ]
    for cols, filename in summary_specs:
        records = []
        for keys, group in rows.groupby(cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            record = dict(zip(cols, keys))
            record.update(summarize(group))
            records.append(record)
        pd.DataFrame(records).sort_values(cols).to_csv(out_dir / filename, index=False)

    support = rows.copy()
    support["analogue_bucket"] = np.where(
        pd.to_numeric(support["has_train_analogue"], errors="coerce").fillna(0).eq(1),
        "has_train_analogue",
        "no_train_analogue",
    )
    support["destination_bucket"] = np.where(
        pd.to_numeric(support["true_cluster_covered_p90"], errors="coerce").fillna(0).eq(1),
        "known_destination_cluster",
        "unseen_destination_cluster",
    )
    records = []
    for keys, group in support.groupby(["method", "analogue_bucket", "destination_bucket"], dropna=False):
        record = dict(zip(["method", "analogue_bucket", "destination_bucket"], keys))
        record.update(summarize(group))
        records.append(record)
    pd.DataFrame(records).sort_values(["method", "analogue_bucket", "destination_bucket"]).to_csv(
        out_dir / "metrics_by_historical_support.csv", index=False
    )


def grid_artifact_records(
    model: GridModel,
    *,
    train_count: int,
    test_count: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    n_transition_edges = sum(len(dst_counts) for dst_counts in model.transition_counts.values())
    all_cells = set(model.cell_visit_counts) | set(model.destination_counts)
    for src, dst_counts in model.transition_counts.items():
        all_cells.add(src)
        all_cells.update(dst_counts)
    summary = {
        "dataset": model.dataset,
        "driver_id": model.driver_id,
        "driver_name": model.driver_name,
        "split_label": model.split_label,
        "cell_size_m": model.cell_size_m,
        "origin_x": model.origin_x,
        "origin_y": model.origin_y,
        "train_trip_count": train_count,
        "test_trip_count": test_count,
        "train_sequence_count": len(model.train_sequences_by_trip),
        "grid_cell_count": len(all_cells),
        "transition_edge_count": n_transition_edges,
        "destination_cell_count": len(model.destination_counts),
    }

    transition_rows: list[dict[str, Any]] = []
    for src, dst_counts in model.transition_counts.items():
        total = float(sum(dst_counts.values()))
        src_ix, src_iy = parse_cell_id(src)
        for dst, count in sorted(dst_counts.items()):
            dst_ix, dst_iy = parse_cell_id(dst)
            transition_rows.append(
                {
                    "dataset": model.dataset,
                    "driver_id": model.driver_id,
                    "driver_name": model.driver_name,
                    "split_label": model.split_label,
                    "from_cell": src,
                    "from_ix": src_ix,
                    "from_iy": src_iy,
                    "to_cell": dst,
                    "to_ix": dst_ix,
                    "to_iy": dst_iy,
                    "transition_count": int(count),
                    "transition_probability": float(count / total) if total > 0 else float("nan"),
                }
            )

    destination_rows: list[dict[str, Any]] = []
    total_dest = float(sum(model.destination_counts.values()))
    for cell, count in sorted(model.destination_counts.items()):
        ix, iy = parse_cell_id(cell)
        x, y = grid_cell_center_xy(model, cell)
        lat, lon = grid_cell_center_latlon(model, cell)
        destination_rows.append(
            {
                "dataset": model.dataset,
                "driver_id": model.driver_id,
                "driver_name": model.driver_name,
                "split_label": model.split_label,
                "cell": cell,
                "ix": ix,
                "iy": iy,
                "center_x": x,
                "center_y": y,
                "center_lat": lat,
                "center_lon": lon,
                "destination_count": int(count),
                "destination_prior": float(count / total_dest) if total_dest > 0 else float("nan"),
            }
        )

    cell_rows: list[dict[str, Any]] = []
    for cell in sorted(all_cells):
        ix, iy = parse_cell_id(cell)
        x, y = grid_cell_center_xy(model, cell)
        lat, lon = grid_cell_center_latlon(model, cell)
        cell_rows.append(
            {
                "dataset": model.dataset,
                "driver_id": model.driver_id,
                "driver_name": model.driver_name,
                "split_label": model.split_label,
                "cell": cell,
                "ix": ix,
                "iy": iy,
                "center_x": x,
                "center_y": y,
                "center_lat": lat,
                "center_lon": lon,
                "visit_count": int(model.cell_visit_counts.get(cell, 0)),
                "destination_count": int(model.destination_counts.get(cell, 0)),
            }
        )
    return summary, transition_rows, destination_rows, cell_rows


def write_grid_artifacts(
    out_dir: Path,
    summaries: list[dict[str, Any]],
    transition_rows: list[dict[str, Any]],
    destination_rows: list[dict[str, Any]],
    cell_rows: list[dict[str, Any]],
) -> None:
    grid_dir = out_dir / "grid_transition"
    grid_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(grid_dir / "grid_model_summary.csv", index=False)
    pd.DataFrame(transition_rows).to_csv(grid_dir / "transition_edges.csv", index=False)
    pd.DataFrame(destination_rows).to_csv(grid_dir / "destination_priors.csv", index=False)
    pd.DataFrame(cell_rows).to_csv(grid_dir / "grid_cells.csv", index=False)


def run_experiment(config: dict[str, Any], args: argparse.Namespace) -> pd.DataFrame:
    all_rows: list[dict[str, Any]] = []
    grid_summaries: list[dict[str, Any]] = []
    grid_transition_rows: list[dict[str, Any]] = []
    grid_destination_rows: list[dict[str, Any]] = []
    grid_cell_rows: list[dict[str, Any]] = []
    min_trips = int(config["experiment"].get("min_trips_per_driver", 20))
    train_fraction = float(config["experiment"].get("train_fraction", 0.75))
    splits = config["experiment"].get("splits", [])
    max_drivers = args.max_drivers_per_dataset
    max_test_trips = args.max_test_trips_per_split

    for spec in dataset_specs(config):
        print(f"Loading {spec.name}: {project_relative_path(spec.path)}", flush=True)
        trips = load_trips(spec, config)
        by_driver: dict[int, list[Trip]] = {}
        for trip in trips:
            by_driver.setdefault(trip.driver_id, []).append(trip)
        driver_ids = [
            driver_id
            for driver_id, driver_trips in sorted(by_driver.items())
            if len(driver_trips) >= min_trips
        ]
        if max_drivers is not None:
            driver_ids = driver_ids[: int(max_drivers)]
        print(f"{spec.name}: {len(driver_ids)} eligible drivers", flush=True)

        for driver_id in driver_ids:
            driver_trips = by_driver[driver_id]
            for split in splits:
                split_label = str(split["label"])
                split_mode = str(split["mode"])
                seed = int(split.get("seed", 13))
                train_trips, test_trips = split_driver_trips(
                    driver_trips,
                    split_label=split_label,
                    split_mode=split_mode,
                    train_fraction=train_fraction,
                    seed=seed,
                )
                split_spec, projection_meta = projection_for_split(spec, train_trips, config)
                reproject_trips(driver_trips, split_spec.epsg)
                if max_test_trips is not None:
                    test_trips = test_trips[: int(max_test_trips)]
                clusters, trip_to_cluster = train_destination_clusters(
                    train_trips,
                    config,
                    transformer_to_latlon=Transformer.from_crs(
                        split_spec.epsg,
                        "EPSG:4326",
                        always_xy=True,
                    ),
                )
                grid_model = build_grid_model(
                    spec=split_spec,
                    driver_id=driver_id,
                    driver_name=driver_trips[0].driver_name if driver_trips else f"user{driver_id}",
                    split_label=split_label,
                    train_trips=train_trips,
                    config=config,
                )
                summary, transition_rows, destination_rows, cell_rows = grid_artifact_records(
                    grid_model,
                    train_count=len(train_trips),
                    test_count=len(test_trips),
                )
                grid_summaries.append(summary)
                grid_transition_rows.extend(transition_rows)
                grid_destination_rows.extend(destination_rows)
                grid_cell_rows.extend(cell_rows)
                print(
                    f"{spec.name} driver={driver_id} split={split_label}: "
                    f"train={len(train_trips)} test={len(test_trips)} "
                    f"clusters={len(clusters)} grid_cells={summary['grid_cell_count']} "
                    f"transition_edges={summary['transition_edge_count']}",
                    flush=True,
                )
                for i, test_trip in enumerate(test_trips, start=1):
                    rows = predict_one_trip(
                        dataset_label=split_spec.label,
                        split_label=split_label,
                        train_trips=train_trips,
                        test_trips=test_trips,
                        test_trip=test_trip,
                        clusters=clusters,
                        trip_to_cluster=trip_to_cluster,
                        grid_model=grid_model,
                        config=config,
                    )
                    for row in rows:
                        row.update(projection_meta)
                    all_rows.extend(rows)
                    if i % 25 == 0:
                        print(f"  predicted {i}/{len(test_trips)}", flush=True)
    out_dir_value = getattr(args, "grid_artifact_dir", None)
    if out_dir_value:
        write_grid_artifacts(
            Path(out_dir_value),
            grid_summaries,
            grid_transition_rows,
            grid_destination_rows,
            grid_cell_rows,
        )
    return pd.DataFrame(all_rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(EXP_DIR / "publication.json"))
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--max-drivers-per-dataset", type=int, default=None)
    parser.add_argument("--max-test-trips-per-split", type=int, default=None)
    parser.add_argument(
        "--projection-strategy",
        choices=["fixed", "train_centered_aeqd"],
        default=None,
        help="Override the metric projection strategy used by the forecasting model.",
    )
    args = parser.parse_args()

    config_path = resolve_project_path(args.config)
    config = read_config(config_path)
    if args.projection_strategy is not None:
        config["experiment"]["projection_strategy"] = args.projection_strategy
    out_dir = resolve_project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    args.grid_artifact_dir = str(out_dir)

    rows = run_experiment(config, args)
    rows.to_csv(out_dir / "trip_predictions.csv", index=False)
    write_summaries(rows, out_dir)

    report = {
        "experiment": "coherent_personal_mobility_forecasting",
        "config": project_relative_path(config_path),
        "n_prediction_rows": int(len(rows)),
        "n_cases": int(rows[["dataset", "driver_id", "split_label", "trip_id"]].drop_duplicates().shape[0]) if not rows.empty else 0,
        "methods": sorted(rows["method"].dropna().unique().tolist()) if not rows.empty else [],
        "outputs": [
            "grid_transition/",
            "trip_predictions.csv",
            "metrics_by_method.csv",
            "metrics_by_dataset_method.csv",
            "metrics_by_driver_split_method.csv",
            "metrics_by_split_method.csv",
            "metrics_by_historical_support.csv",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not rows.empty:
        print(pd.read_csv(out_dir / "metrics_by_method.csv").round(3).to_string(index=False), flush=True)
    print(f"Wrote {project_relative_path(out_dir)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
