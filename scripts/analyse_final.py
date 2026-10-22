#!/usr/bin/env python3
"""Aggregate final admissible-system results and confidence diagnostics."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_ANALYSIS = ROOT / "scripts" / "analyse_guarded.py"
ARMS = ["combined_1000m", "combined_500m", "combined_350m"]
BASELINES = [
    "destination_only",
    "route_analogue",
    "transition_grid_route",
    "most_frequent_destination",
]
OUTCOMES = {
    "ead": "product_success",
    "destination_1km": "dest_within_1000m",
    "short_horizon_1km": "la_within_1000m",
    "route_score": "route_step_accuracy",
}
Q_THRESHOLDS = [0.0, 0.48, 0.60, 0.70, 0.75, 0.80]
DEVELOPMENT_USERS = set(range(1, 14))


def load_base() -> Any:
    spec = importlib.util.spec_from_file_location("base_final_analysis", BASE_ANALYSIS)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {BASE_ANALYSIS}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def metric_summary(frame: pd.DataFrame, total_cases: int) -> dict[str, float | int]:
    emitted = numeric(frame, "emitted").fillna(0).eq(1)
    accepted = frame[emitted]
    result: dict[str, float | int] = {
        "n_cases": int(total_cases),
        "n_emitted": int(emitted.sum()),
        "coverage_pct": 100.0 * float(emitted.sum() / total_cases),
    }
    for label, column in OUTCOMES.items():
        result[f"accepted_{label}_pct"] = (
            100.0 * float(numeric(accepted, column).mean()) if len(accepted) else float("nan")
        )
        result[f"all_case_{label}_pct"] = (
            100.0 * float((numeric(frame, column).fillna(0.0) * emitted.astype(float)).sum() / total_cases)
        )
    return result


def paired_bootstrap(base: Any, rows: pd.DataFrame, draws: int, seed: int) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for cohort in ["all", "geolife"]:
        for arm in ARMS:
            for selector in ["coherent_product", "coherent_abstain"]:
                for baseline in BASELINES:
                    records.extend(
                        base.bootstrap_difference(
                            rows,
                            cohort=cohort,
                            arm_a=arm,
                            method_a=selector,
                            arm_b=arm,
                            method_b=baseline,
                            draws=draws,
                            seed=seed,
                        )
                    )
    return pd.DataFrame(records)


def risk_curve(frame: pd.DataFrame, score_column: str, total_cases: int) -> pd.DataFrame:
    ordered = frame.sort_values(
        [score_column, "dataset", "driver_id", "split_label", "trip_id"],
        ascending=[False, True, True, True, True],
    ).copy()
    k = np.arange(1, len(ordered) + 1, dtype=float)
    result = pd.DataFrame(
        {
            "rank": k.astype(int),
            "coverage_pct": 100.0 * k / float(total_cases),
            "threshold": numeric(ordered, score_column).to_numpy(dtype=float),
        }
    )
    for label, column in OUTCOMES.items():
        values = numeric(ordered, column).fillna(0.0).to_numpy(dtype=float)
        result[f"accepted_{label}_pct"] = 100.0 * np.cumsum(values) / k
        result[f"risk_{label}"] = 1.0 - np.cumsum(values) / k
    return result


def normalized_aurc(curve: pd.DataFrame, risk_column: str) -> float:
    coverage = curve["coverage_pct"].to_numpy(dtype=float) / 100.0
    risk = curve[risk_column].to_numpy(dtype=float)
    x = np.concatenate([[0.0], coverage])
    y = np.concatenate([[risk[0]], risk])
    width = float(x[-1])
    return float(np.trapezoid(y, x) / width) if width > 0 else float("nan")


def rank_top(frame: pd.DataFrame, score_column: str, n: int) -> pd.DataFrame:
    return frame.sort_values(
        [score_column, "dataset", "driver_id", "split_label", "trip_id"],
        ascending=[False, True, True, True, True],
    ).head(n)


def confidence_outputs(rows: pd.DataFrame, total_cases: int, output_dir: Path) -> None:
    final = rows[
        rows["arm"].eq("combined_1000m") & rows["method"].eq("coherent_product")
    ].copy()
    available = final[numeric(final, "emitted").eq(1)].copy()
    if len(final) != total_cases:
        raise ValueError("Unexpected final coherent case count")

    score_specs = {
        "multi_output_q": "candidate_confidence",
        "analogue_s1": "best_analogue_score",
    }
    curves: list[pd.DataFrame] = []
    aurc_records: list[dict[str, Any]] = []
    for score, column in score_specs.items():
        curve = risk_curve(available, column, total_cases)
        curve.insert(0, "score", score)
        curves.append(curve)
        record: dict[str, Any] = {"score": score, "n_admissible": int(len(available))}
        for label in OUTCOMES:
            record[f"normalized_{label}_aurc"] = normalized_aurc(curve, f"risk_{label}")
        aurc_records.append(record)
    curve_table = pd.concat(curves, ignore_index=True)
    curve_table.to_csv(output_dir / "risk_coverage_curves.csv", index=False)
    pd.DataFrame(aurc_records).to_csv(output_dir / "risk_coverage_aurc.csv", index=False)

    matched: list[dict[str, Any]] = []
    q_values = numeric(available, "candidate_confidence")
    for threshold in Q_THRESHOLDS:
        q_selected = available[q_values.ge(threshold)].copy()
        n = len(q_selected)
        for score, column in score_specs.items():
            selected = q_selected if score == "multi_output_q" else rank_top(available, column, n)
            record = {
                "q_threshold_defining_count": threshold,
                "score": score,
                "n_total": total_cases,
                "n_admissible": int(len(available)),
                "n_accepted": int(len(selected)),
                "coverage_pct": 100.0 * len(selected) / total_cases,
                "score_cutoff_at_rank": (
                    float(numeric(selected, column).min()) if len(selected) else float("nan")
                ),
            }
            for label, outcome in OUTCOMES.items():
                record[f"accepted_{label}_pct"] = (
                    100.0 * float(numeric(selected, outcome).mean()) if len(selected) else float("nan")
                )
            matched.append(record)
    pd.DataFrame(matched).to_csv(output_dir / "matched_coverage_score_comparison.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    styles = {"multi_output_q": ("#1f5a99", "Multi-output evidence score q"), "analogue_s1": ("#c45a2c", "Best-analogue score S1")}
    for score, (colour, label) in styles.items():
        curve = curve_table[curve_table["score"].eq(score)]
        keep = np.unique(np.linspace(0, len(curve) - 1, min(450, len(curve))).astype(int))
        ax.plot(
            curve.iloc[keep]["coverage_pct"],
            100.0 * curve.iloc[keep]["risk_ead"],
            color=colour,
            linewidth=1.8,
            label=label,
        )
    ax.set_xlabel("Coverage over all evaluation cases (%)")
    ax.set_ylabel("Accepted EAD risk (%)")
    ax.set_xlim(left=0.0, right=86.0)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "risk_coverage_ead.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "risk_coverage_ead.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def stratified_outputs(base: Any, rows: pd.DataFrame, output_dir: Path) -> None:
    final = rows[rows["arm"].eq("combined_1000m")].copy()

    source_records: list[dict[str, Any]] = []
    for (dataset, split, method), frame in final.groupby(
        ["dataset", "split_label", "method"], sort=False
    ):
        record = {"dataset": dataset, "split_label": split, "method": method}
        record.update(metric_summary(frame, len(frame)))
        source_records.append(record)
    pd.DataFrame(source_records).to_csv(output_dir / "metrics_by_dataset_split_policy.csv", index=False)

    support_records: list[dict[str, Any]] = []
    for policy in ["coherent_product", "coherent_abstain"]:
        coherent = final[final["method"].eq(policy)].copy()
        coherent["analogue_available"] = numeric(coherent, "has_train_analogue").fillna(0).astype(int)
        coherent["destination_covered"] = numeric(coherent, "true_cluster_covered_p90").fillna(0).astype(int)
        for (analogue, covered), frame in coherent.groupby(
            ["analogue_available", "destination_covered"], sort=False
        ):
            record = {"policy": policy, "A": int(analogue), "Z": int(covered)}
            record.update(metric_summary(frame, len(frame)))
            support_records.append(record)
    pd.DataFrame(support_records).sort_values(
        ["policy", "A", "Z"], ascending=[True, False, False]
    ).to_csv(
        output_dir / "historical_support_strata.csv", index=False
    )

    heldout = final[
        ~(
            final["dataset"].eq("geolife_ground")
            & final["driver_id"].isin(DEVELOPMENT_USERS)
        )
    ].copy()
    heldout_metrics = base.summarize_metrics(heldout)
    heldout_metrics.to_csv(output_dir / "leave_development_users_out_metrics.csv", index=False)
    records: list[dict[str, Any]] = []
    for selector in ["coherent_product", "coherent_abstain"]:
        for baseline in BASELINES:
            records.extend(
                base.bootstrap_difference(
                    heldout,
                    cohort="geolife",
                    arm_a="combined_1000m",
                    method_a=selector,
                    arm_b="combined_1000m",
                    method_b=baseline,
                    draws=10000,
                    seed=20261020,
                )
            )
    pd.DataFrame(records).to_csv(output_dir / "leave_development_users_out_bootstrap.csv", index=False)


def temporal_outputs(rows: pd.DataFrame, temporal_audit_path: Path, output_dir: Path) -> None:
    temporal = pd.read_csv(temporal_audit_path)
    flags = [
        "target_bracket_gap_over_1h",
        "source_nonpositive",
        "source_gap_over_1h",
        "source_speed_over_100mps",
    ]
    temporal = temporal[["driver_id", "trip_id", *flags]].drop_duplicates(
        ["driver_id", "trip_id"]
    )
    records: list[dict[str, Any]] = []
    for policy in ["coherent_product", "coherent_abstain"]:
        coherent = rows[
            rows["arm"].eq("combined_1000m")
            & rows["method"].eq(policy)
            & rows["dataset"].eq("geolife_ground")
        ].copy()
        merged = coherent.merge(
            temporal,
            on=["driver_id", "trip_id"],
            how="left",
            validate="many_to_one",
        )
        if merged[flags].isna().any().any():
            raise ValueError("Missing temporal-audit flags for final GeoLife cases")
        scopes = {
            "all_geolife_cases": np.ones(len(merged), dtype=bool),
            "exclude_target_interpolation_gap_over_1h": ~merged["target_bracket_gap_over_1h"].astype(bool),
            "exclude_any_source_nonpositive_or_gap_over_1h": ~(
                merged["source_nonpositive"].astype(bool)
                | merged["source_gap_over_1h"].astype(bool)
            ),
            "exclude_any_audited_temporal_or_speed_anomaly": ~(
                merged["source_nonpositive"].astype(bool)
                | merged["source_gap_over_1h"].astype(bool)
                | merged["source_speed_over_100mps"].astype(bool)
            ),
        }
        for scope, mask in scopes.items():
            frame = merged.loc[mask].copy()
            record = {"policy": policy, "scope": scope}
            record.update(metric_summary(frame, len(frame)))
            records.append(record)
    pd.DataFrame(records).to_csv(output_dir / "temporal_sensitivity_metrics.csv", index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--temporal-audit", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base = load_base()
    rows = pd.read_csv(args.predictions)
    total_cases = int(rows[["dataset", "driver_id", "split_label", "trip_id"]].drop_duplicates().shape[0])
    if total_cases != 8526:
        raise ValueError(f"Expected 8,526 cases, found {total_cases:,}")

    base.summarize_metrics(rows).to_csv(args.output_dir / "metrics_by_arm_method.csv", index=False)
    base.summarize_reselection(rows).to_csv(args.output_dir / "coherent_reselection_summary.csv", index=False)
    base.confidence_geometry(rows).to_csv(args.output_dir / "confidence_operating_points.csv", index=False)
    base.family_counts(rows).to_csv(args.output_dir / "coherent_candidate_families.csv", index=False)
    paired_bootstrap(base, rows, args.bootstrap_draws, args.seed).to_csv(
        args.output_dir / "paired_equal_user_bootstrap.csv", index=False
    )
    confidence_outputs(rows, total_cases, args.output_dir)
    stratified_outputs(base, rows, args.output_dir)
    temporal_outputs(rows, args.temporal_audit, args.output_dir)

    summary = {
        "status": "completed",
        "n_cases": total_cases,
        "n_rows": int(len(rows)),
        "arms": ARMS,
        "policies": sorted(rows["method"].unique().tolist()),
        "admissibility": {
            "max_endpoint_discrepancy_m": 500,
            "primary_max_start_gap_m": 1000,
            "grid_destination_cell_must_be_reached": True,
        },
        "bootstrap_draws": args.bootstrap_draws,
        "bootstrap_seed": args.seed,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
