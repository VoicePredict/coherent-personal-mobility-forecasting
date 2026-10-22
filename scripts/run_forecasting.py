#!/usr/bin/env python3
"""Run the full reproducible experiment from raw data to final metrics."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

from forecasting import EXP_DIR, OUT_DIR, ROOT, run_experiment, write_summaries
from data_preparation import prepare_all, resolve_project_path


def read_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_effective_config(config: dict[str, Any], out_dir: Path) -> Path:
    effective_path = out_dir / "effective_config.json"
    effective_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return effective_path


def project_relative_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def config_path_value(path: Path) -> str:
    return project_relative_path(path)


def parse_raw_path_overrides(values: list[str]) -> dict[str, Path]:
    overrides: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                "--raw-path must use DATASET_NAME=CSV_PATH, "
                "for example geolife_ground=data/geolife.csv"
            )
        dataset_name, path_value = value.split("=", 1)
        dataset_name = dataset_name.strip()
        path_value = path_value.strip()
        if not dataset_name or not path_value:
            raise ValueError("--raw-path requires non-empty DATASET_NAME and CSV_PATH.")
        overrides[dataset_name] = resolve_project_path(path_value)
    return overrides


def apply_raw_path_overrides(
    config: dict[str, Any],
    raw_data_dir: str | None,
    raw_path_values: list[str],
) -> dict[str, Any]:
    configured = copy.deepcopy(config)
    preprocessing_specs = configured.get("preprocessing", {}).get("datasets", [])

    if raw_data_dir:
        raw_dir = resolve_project_path(raw_data_dir)
        for prep_spec in preprocessing_specs:
            original_name = Path(str(prep_spec["raw_path"])).name
            prep_spec["raw_path"] = config_path_value(raw_dir / original_name)

    raw_path_overrides = parse_raw_path_overrides(raw_path_values)
    for dataset_name, raw_path in raw_path_overrides.items():
        matched = False
        for prep_spec in preprocessing_specs:
            if str(prep_spec["name"]) == dataset_name:
                prep_spec["raw_path"] = config_path_value(raw_path)
                matched = True
                break
        if not matched:
            known = ", ".join(str(spec.get("name")) for spec in preprocessing_specs)
            raise ValueError(f"Unknown dataset in --raw-path: {dataset_name!r}. Known datasets: {known}")

    return configured


def configure_prepared_paths(config: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    configured = copy.deepcopy(config)
    prepared_dir = out_dir / "prepared_data"
    prep_by_name: dict[str, str] = {}
    for prep_spec in configured.get("preprocessing", {}).get("datasets", []):
        name = str(prep_spec["name"])
        prepared_path = prepared_dir / f"{name}.csv"
        prep_spec["prepared_path"] = config_path_value(prepared_path)
        prep_by_name[name] = config_path_value(prepared_path)
    for dataset_spec in configured.get("datasets", []):
        name = str(dataset_spec["name"])
        if name in prep_by_name:
            dataset_spec["path"] = prep_by_name[name]
    return configured


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(EXP_DIR / "publication.json"))
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument(
        "--raw-data-dir",
        default=None,
        help=(
            "Optional directory containing the raw CSV files named as in config.json. "
            "Paths are resolved relative to the project root unless absolute."
        ),
    )
    parser.add_argument(
        "--raw-path",
        action="append",
        default=[],
        metavar="DATASET_NAME=CSV_PATH",
        help=(
            "Override one raw input path, e.g. "
            "geolife_ground=outputs/private/source_preparation/private/geolife_source.csv.gz. "
            "May be passed more than once."
        ),
    )
    parser.add_argument(
        "--skip-data-preparation",
        action="store_true",
        help="Use already prepared files listed in config.json instead of rebuilding them from raw data.",
    )
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
    out_dir = resolve_project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = read_config(config_path)
    config = apply_raw_path_overrides(config, args.raw_data_dir, args.raw_path)
    if args.projection_strategy is not None:
        config["experiment"]["projection_strategy"] = args.projection_strategy
    if not args.skip_data_preparation:
        config = configure_prepared_paths(config, out_dir)
    if not args.skip_data_preparation:
        print("Prepare raw input data", flush=True)
        preparation_summary = prepare_all(config)
        (out_dir / "data_preparation_summary.json").write_text(
            json.dumps(preparation_summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    else:
        print("Skip data preparation; using prepared files from config.json", flush=True)

    effective_config_path = write_effective_config(config, out_dir)
    print(f"Effective config: {project_relative_path(effective_config_path)}", flush=True)

    forecast_args = SimpleNamespace(
        max_drivers_per_dataset=args.max_drivers_per_dataset,
        max_test_trips_per_split=args.max_test_trips_per_split,
        grid_artifact_dir=str(out_dir),
    )
    rows = run_experiment(config, forecast_args)
    rows.to_csv(out_dir / "trip_predictions.csv", index=False)
    write_summaries(rows, out_dir)

    report = {
        "experiment": "coherent_personal_mobility_forecasting",
        "run_type": "raw_data_to_metrics",
        "config": project_relative_path(config_path),
        "effective_config": project_relative_path(effective_config_path),
        "data_preparation": "skipped" if args.skip_data_preparation else "completed",
        "n_prediction_rows": int(len(rows)),
        "n_cases": int(rows[["dataset", "driver_id", "split_label", "trip_id"]].drop_duplicates().shape[0]) if not rows.empty else 0,
        "methods": sorted(rows["method"].dropna().unique().tolist()) if not rows.empty else [],
        "outputs": [
            "prepared_data/",
            "grid_transition/",
            "data_preparation_summary.json",
            "effective_config.json",
            "trip_predictions.csv",
            "metrics_by_method.csv",
            "metrics_by_dataset_method.csv",
            "metrics_by_driver_split_method.csv",
            "metrics_by_split_method.csv",
            "metrics_by_historical_support.csv",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not rows.empty:
        print(pd.read_csv(out_dir / "metrics_by_method.csv").round(3).to_string(index=False), flush=True)
    print(f"Wrote {project_relative_path(out_dir)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
