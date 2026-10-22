#!/usr/bin/env python3
"""Aggregate and bootstrap the top-15 guarded analogue comparison."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_ANALYSIS = ROOT / "scripts" / "analyse_guarded.py"
KEYS = ["dataset", "driver_id", "split_label", "trip_id"]
ARMS = ["combined_1000m", "combined_500m", "combined_350m"]
NEW_METHODS = ["route_analogue_a1_guarded", "route_analogue_top15_guarded"]
FINAL_METHODS = ["coherent_product", "coherent_abstain"]
OUTCOMES = {
    "ead": "product_success",
    "destination_1km": "dest_within_1000m",
    "short_horizon_1km": "la_within_1000m",
    "route_score": "route_step_accuracy",
}
DEVELOPMENT_USERS = set(range(1, 14))


def load_base() -> Any:
    spec = importlib.util.spec_from_file_location("top15_base_analysis", BASE_ANALYSIS)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {BASE_ANALYSIS}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def metric_summary(frame: pd.DataFrame) -> dict[str, float | int]:
    emitted = numeric(frame, "emitted").fillna(0).eq(1)
    accepted = frame[emitted]
    out: dict[str, float | int] = {
        "n_cases": int(len(frame)),
        "n_emitted": int(emitted.sum()),
        "coverage_pct": 100.0 * float(emitted.mean()),
    }
    for label, column in OUTCOMES.items():
        out[f"accepted_{label}_pct"] = (
            100.0 * float(numeric(accepted, column).mean())
            if len(accepted)
            else float("nan")
        )
        out[f"all_case_{label}_pct"] = 100.0 * float(
            (numeric(frame, column).fillna(0.0) * emitted.astype(float)).mean()
        )
    return out


def metric_tables(rows: pd.DataFrame, output_dir: Path) -> None:
    records: list[dict[str, Any]] = []
    for (arm, method), frame in rows.groupby(["arm", "method"], sort=False):
        record = {"arm": arm, "method": method}
        record.update(metric_summary(frame))
        records.append(record)
    pd.DataFrame(records).to_csv(output_dir / "metrics_by_arm_policy.csv", index=False)

    records = []
    for (dataset, split, arm, method), frame in rows.groupby(
        ["dataset", "split_label", "arm", "method"], sort=False
    ):
        record = {
            "dataset": dataset,
            "split_label": split,
            "arm": arm,
            "method": method,
        }
        record.update(metric_summary(frame))
        records.append(record)
    pd.DataFrame(records).to_csv(
        output_dir / "metrics_by_dataset_split_arm_policy.csv", index=False
    )


def bootstrap_tables(base: Any, rows: pd.DataFrame, output_dir: Path, draws: int, seed: int) -> None:
    records: list[dict[str, Any]] = []
    for cohort in ["all", "geolife"]:
        for arm in ARMS:
            for selector in FINAL_METHODS:
                records.extend(
                    base.bootstrap_difference(
                        rows,
                        cohort=cohort,
                        arm_a=arm,
                        method_a=selector,
                        arm_b=arm,
                        method_b="route_analogue_top15_guarded",
                        draws=draws,
                        seed=seed,
                    )
                )
            records.extend(
                base.bootstrap_difference(
                    rows,
                    cohort=cohort,
                    arm_a=arm,
                    method_a="route_analogue_top15_guarded",
                    arm_b=arm,
                    method_b="route_analogue_a1_guarded",
                    draws=draws,
                    seed=seed,
                )
            )
    pd.DataFrame(records).to_csv(
        output_dir / "paired_equal_user_bootstrap.csv", index=False
    )

    heldout = rows[
        ~(
            rows["dataset"].eq("geolife_ground")
            & rows["driver_id"].isin(DEVELOPMENT_USERS)
        )
    ].copy()
    records = []
    for selector in FINAL_METHODS:
        records.extend(
            base.bootstrap_difference(
                heldout,
                cohort="geolife",
                arm_a="combined_1000m",
                method_a=selector,
                arm_b="combined_1000m",
                method_b="route_analogue_top15_guarded",
                draws=draws,
                seed=seed,
            )
        )
    records.extend(
        base.bootstrap_difference(
            heldout,
            cohort="geolife",
            arm_a="combined_1000m",
            method_a="route_analogue_top15_guarded",
            arm_b="combined_1000m",
            method_b="route_analogue_a1_guarded",
            draws=draws,
            seed=seed,
        )
    )
    pd.DataFrame(records).to_csv(
        output_dir / "leave_development_users_out_bootstrap.csv", index=False
    )


def rank_outputs(new_rows: pd.DataFrame, output_dir: Path) -> None:
    top15 = new_rows[new_rows["method"].eq("route_analogue_top15_guarded")].copy()
    top15["reported_rank"] = numeric(top15, "selected_analogue_rank").fillna(0).astype(int)
    rank_counts = (
        top15.groupby(["arm", "reported_rank"], dropna=False)
        .size()
        .rename("n")
        .reset_index()
    )
    totals = rank_counts.groupby("arm")["n"].transform("sum")
    rank_counts["pct_all_cases"] = 100.0 * rank_counts["n"] / totals
    rank_counts["rank_label"] = np.where(
        rank_counts["reported_rank"].eq(0),
        "abstain",
        "a" + rank_counts["reported_rank"].astype(str),
    )
    rank_counts.to_csv(output_dir / "selected_analogue_rank_distribution.csv", index=False)

    records: list[dict[str, Any]] = []
    for arm in ARMS:
        a1 = new_rows[
            new_rows["arm"].eq(arm)
            & new_rows["method"].eq("route_analogue_a1_guarded")
        ].set_index(KEYS)
        t15 = new_rows[
            new_rows["arm"].eq(arm)
            & new_rows["method"].eq("route_analogue_top15_guarded")
        ].set_index(KEYS)
        joined = a1.add_suffix("_a1").join(t15.add_suffix("_top15"), how="inner")
        a1_emit = numeric(joined, "emitted_a1").eq(1)
        top15_emit = numeric(joined, "emitted_top15").eq(1)
        record: dict[str, Any] = {
            "arm": arm,
            "n_cases": int(len(joined)),
            "a1_emitted_n": int(a1_emit.sum()),
            "top15_emitted_n": int(top15_emit.sum()),
            "rescued_after_inadmissible_a1_n": int((~a1_emit & top15_emit).sum()),
            "top15_abstained_n": int((~top15_emit).sum()),
            "selected_lower_rank_n": int(
                (numeric(joined, "selected_analogue_rank_top15") > 1).sum()
            ),
        }
        for label, column in OUTCOMES.items():
            a1_all = a1_emit.astype(float) * numeric(joined, f"{column}_a1").fillna(0.0)
            t15_all = top15_emit.astype(float) * numeric(joined, f"{column}_top15").fillna(0.0)
            record[f"{label}_improved_n"] = int((t15_all > a1_all).sum())
            record[f"{label}_worsened_n"] = int((t15_all < a1_all).sum())
            record[f"{label}_unchanged_n"] = int((t15_all == a1_all).sum())
        records.append(record)
    pd.DataFrame(records).to_csv(output_dir / "top15_backoff_summary.csv", index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-predictions", type=Path, required=True)
    parser.add_argument("--final-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261020)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base = load_base()
    new_rows = pd.read_csv(args.new_predictions)
    final_rows = pd.read_csv(args.final_predictions)
    final_rows = final_rows[
        final_rows["arm"].isin(ARMS) & final_rows["method"].isin(FINAL_METHODS)
    ].copy()
    rows = pd.concat([final_rows, new_rows], ignore_index=True, sort=False)

    for method in [*FINAL_METHODS, *NEW_METHODS]:
        method_cases = rows[rows["method"].eq(method)][KEYS].drop_duplicates()
        if len(method_cases) != 8526:
            raise ValueError(f"{method} has {len(method_cases):,} cases, expected 8,526")

    metric_tables(rows, args.output_dir)
    bootstrap_tables(
        base, rows, args.output_dir, args.bootstrap_draws, args.seed
    )
    rank_outputs(new_rows, args.output_dir)

    summary = {
        "status": "completed",
        "n_cases": 8526,
        "analogue_budget": 15,
        "arms": ARMS,
        "policies": [*FINAL_METHODS, *NEW_METHODS],
        "primary_contrast": "coherent_product minus route_analogue_top15_guarded under combined_1000m",
        "bootstrap_draws": args.bootstrap_draws,
        "bootstrap_seed": args.seed,
        "selection_uses_test_outcomes": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
