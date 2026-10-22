#!/usr/bin/env python3
"""Prepare the repository's raw mobility CSV files for forecasting."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else ROOT / path


def project_relative_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def parse_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(part).strip().lower() for part in value if str(part).strip()]
    return [part.strip().lower() for part in str(value).split(",") if part.strip()]


def normalize_mode(series: pd.Series) -> pd.Series:
    return (
        series.fillna("unknown")
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"": "unknown", "nan": "unknown", "none": "unknown"})
    )


def build_exclude_mask(modes: pd.Series, patterns: list[str]) -> pd.Series:
    escaped = [re.escape(pattern) for pattern in patterns if pattern]
    if not escaped:
        return pd.Series(False, index=modes.index)
    return modes.str.contains("|".join(escaped), regex=True, na=False)


def decimal_comma_to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.strip().str.replace(",", ".", regex=False),
        errors="coerce",
    )


def hms_to_seconds(series: pd.Series) -> pd.Series:
    td = pd.to_timedelta(series.astype(str).str.strip(), errors="coerce")
    return td.dt.total_seconds()


def build_monotonic_gta_datetimes(raw: pd.DataFrame, time_col: str = "time") -> pd.Series:
    """Use GTA in-game time while preserving original point order per trip."""

    out = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")
    base_dates = pd.to_datetime(raw["date"], errors="coerce")
    seconds = hms_to_seconds(raw[time_col])

    for _, idx_values in raw.groupby(["ID", "trip_id"], sort=False).groups.items():
        idx = list(idx_values)
        base_date = base_dates.loc[idx].dropna()
        if base_date.empty:
            continue
        base = base_date.iloc[0].normalize()

        group_seconds = seconds.loc[idx].to_numpy(dtype=float)
        adjusted: list[float] = []
        day_offset = 0.0
        prev = np.nan
        for sec in group_seconds:
            if not np.isfinite(sec):
                adjusted.append(np.nan)
                continue
            if np.isfinite(prev) and sec < prev:
                day_offset += 24.0 * 60.0 * 60.0
            adjusted.append(float(sec + day_offset))
            prev = sec

        offsets_us = np.arange(len(idx), dtype=np.int64)
        values = []
        for adj_sec, offset_us in zip(adjusted, offsets_us):
            if not np.isfinite(adj_sec):
                values.append(pd.NaT)
            else:
                values.append(
                    base
                    + pd.to_timedelta(adj_sec, unit="s")
                    + pd.to_timedelta(int(offset_us), unit="us")
                )
        out.loc[idx] = values
    return out


def build_standard_datetime(df: pd.DataFrame) -> pd.Series:
    if "datetime" in df.columns:
        return pd.to_datetime(df["datetime"], errors="coerce")
    if {"date", "time"}.issubset(df.columns):
        return pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), errors="coerce")
    raise ValueError("Input must contain either `datetime` or both `date` and `time` columns.")


def normalize_gta_raw(raw_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(raw_path, low_memory=False)
    required = {
        "ID",
        "trip_id",
        "latitude",
        "longitude",
        "date",
        "time",
        "time_rw",
        "veh_class",
        "char",
        "length_m",
        "dur_s",
        "n_points",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"{raw_path} is missing required GTA columns: {missing}")

    raw = raw.copy()
    raw["_source_row"] = np.arange(len(raw), dtype=np.int64)
    normalized = pd.DataFrame()
    normalized["ID"] = pd.to_numeric(raw["ID"], errors="coerce").astype("Int64")
    normalized["trip_id"] = pd.to_numeric(raw["trip_id"], errors="coerce").astype("Int64")
    normalized["latitude"] = decimal_comma_to_float(raw["latitude"])
    normalized["longitude"] = decimal_comma_to_float(raw["longitude"])
    normalized["datetime"] = build_monotonic_gta_datetimes(raw, time_col="time")
    normalized["date"] = normalized["datetime"].dt.strftime("%Y-%m-%d")
    normalized["time"] = normalized["datetime"].dt.strftime("%H:%M:%S.%f")
    normalized["Transportation Mode"] = raw["veh_class"].astype(str).str.strip()
    normalized["char"] = raw["char"].astype(str).str.strip()
    normalized["veh_class"] = raw["veh_class"].astype(str).str.strip()
    normalized["game_time"] = raw["time"].astype(str).str.strip()
    normalized["real_recorded_time"] = raw["time_rw"].astype(str).str.strip()
    normalized["source_date"] = raw["date"].astype(str).str.strip()
    normalized["length_m"] = decimal_comma_to_float(raw["length_m"])
    normalized["dur_s"] = pd.to_numeric(raw["dur_s"], errors="coerce")
    normalized["n_points"] = pd.to_numeric(raw["n_points"], errors="coerce")
    normalized["source_row"] = raw["_source_row"]
    return normalized


def read_source_frame(raw_path: Path, source_format: str) -> pd.DataFrame:
    if source_format == "gta_decimal_comma_csv":
        return normalize_gta_raw(raw_path)
    if source_format == "standard_csv":
        return pd.read_csv(raw_path, low_memory=False)
    raise ValueError(f"Unsupported source_format: {source_format}")


def prepare_one_dataset(spec: dict[str, Any]) -> dict[str, Any]:
    name = str(spec["name"])
    raw_path = resolve_project_path(spec["raw_path"])
    prepared_path = resolve_project_path(spec["prepared_path"])
    source_format = str(spec.get("source_format", "standard_csv"))
    mode_col = str(spec.get("mode_column", "Transportation Mode"))
    include_modes = parse_list(spec.get("include_modes"))
    exclude_patterns = parse_list(spec.get("exclude_mode_patterns"))
    min_trips = int(spec.get("min_trips_per_user", 20))

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw dataset for {name!r} was not found: {raw_path}. "
            "Place the source CSV at this path or update config/publication.json."
        )

    df = read_source_frame(raw_path, source_format)
    required = {"ID", "trip_id", "latitude", "longitude"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{raw_path} is missing required columns: {missing}")

    original_rows = int(len(df))
    if mode_col not in df.columns:
        df[mode_col] = "unknown"
    df[mode_col] = normalize_mode(df[mode_col])

    include_filtered_rows = 0
    include_filtered_mode_counts: dict[str, int] = {}
    if include_modes:
        include_mask = df[mode_col].isin(include_modes)
        include_filtered_rows = int((~include_mask).sum())
        include_filtered_mode_counts = {
            str(k): int(v)
            for k, v in df.loc[~include_mask, mode_col].value_counts(dropna=False).items()
        }
        df = df.loc[include_mask].copy()

    excluded_mask = build_exclude_mask(df[mode_col], exclude_patterns)
    excluded_mode_counts = {
        str(k): int(v)
        for k, v in df.loc[excluded_mask, mode_col].value_counts(dropna=False).items()
    }
    df = df.loc[~excluded_mask].copy()

    if "datetime" not in df.columns or pd.to_datetime(df["datetime"], errors="coerce").isna().all():
        df["datetime"] = build_standard_datetime(df)
    else:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["ID"] = pd.to_numeric(df["ID"], errors="coerce").astype("Int64")
    df["trip_id"] = pd.to_numeric(df["trip_id"], errors="coerce").astype("Int64")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["ID", "trip_id", "latitude", "longitude", "datetime"]).copy()

    trip_counts_before_min = df.groupby("ID")["trip_id"].nunique().sort_values(ascending=False)
    keep_users = trip_counts_before_min[trip_counts_before_min >= min_trips].index.tolist()
    df = df[df["ID"].isin(keep_users)].copy()
    df["ID"] = df["ID"].astype(int)
    df["trip_id"] = df["trip_id"].astype(int)
    sort_cols = ["ID", "trip_id", "datetime"]
    if "source_row" in df.columns:
        sort_cols.append("source_row")
    df = df.sort_values(sort_cols).reset_index(drop=True)

    prepared_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(prepared_path, index=False)

    final_trip_counts = df.groupby("ID")["trip_id"].nunique().sort_values(ascending=False)
    summary = {
        "name": name,
        "input": project_relative_path(raw_path),
        "output": project_relative_path(prepared_path),
        "source_format": source_format,
        "original_rows": original_rows,
        "exclude_mode_patterns": exclude_patterns,
        "include_modes": include_modes if include_modes else None,
        "include_filtered_rows": include_filtered_rows,
        "include_filtered_mode_counts": include_filtered_mode_counts,
        "excluded_rows": int(excluded_mask.sum()),
        "excluded_mode_counts": excluded_mode_counts,
        "min_trips_per_user": min_trips,
        "rows": int(len(df)),
        "users": int(df["ID"].nunique()),
        "trips": int(df["trip_id"].nunique()),
        "mode_counts": {str(k): int(v) for k, v in df[mode_col].value_counts(dropna=False).items()},
        "trip_counts_by_user": {str(k): int(v) for k, v in final_trip_counts.items()},
    }
    if "char" in df.columns:
        summary["trip_counts_by_driver_name"] = {
            f"{driver} ({user_id})": int(count)
            for (user_id, driver), count in (
                df.groupby(["ID", "char"])["trip_id"].nunique().sort_values(ascending=False).items()
            )
        }

    summary_path = prepared_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def prepare_all(config: dict[str, Any]) -> list[dict[str, Any]]:
    preprocessing = config.get("preprocessing", {})
    dataset_specs = preprocessing.get("datasets", [])
    if not dataset_specs:
        raise ValueError("config.json must contain preprocessing.datasets for a raw-data run.")
    return [prepare_one_dataset(dict(spec)) for spec in dataset_specs]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config" / "publication.json"))
    args = parser.parse_args()

    config_path = resolve_project_path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    prepare_all(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
