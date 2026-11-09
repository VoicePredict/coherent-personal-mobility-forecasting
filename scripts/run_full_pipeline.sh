#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
PRIVATE="$ROOT/outputs/private"
PUBLIC="$ROOT/outputs/public"
CONFIG="$ROOT/config/publication.json"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python environment not found: $PYTHON" >&2
  echo "Create it with: python3 -m venv .venv && .venv/bin/pip install -r requirements.lock.txt" >&2
  exit 2
fi

mkdir -p "$PRIVATE" "$PUBLIC"
cd "$ROOT"

setup_args=()
if [[ "${ACCEPT_GEOLIFE_LICENSE:-0}" == "1" ]]; then
  setup_args+=(--accept-license)
fi
if [[ -n "${GEOLIFE_ARCHIVE:-}" ]]; then
  setup_args+=(--archive "$GEOLIFE_ARCHIVE")
fi
if [[ -n "${GEOLIFE_RAW_ROOT:-}" ]]; then
  setup_args+=(--raw-root "$GEOLIFE_RAW_ROOT")
fi

echo "[1/14] Verify, extract, and prepare GeoLife 1.3"
"$PYTHON" scripts/setup_geolife.py "${setup_args[@]}"

echo "[2/14] Generate the frozen candidate families (DBSCAN epsilon 500 m)"
"$PYTHON" scripts/run_forecasting.py \
  --config "$CONFIG" \
  --output-dir "$PRIVATE/base_forecast"

echo "[3/14] Audit train-only provenance and hidden-suffix invariance"
mkdir -p "$PRIVATE/leakage" "$PUBLIC/final_admissible"
"$PYTHON" scripts/leakage_audit.py \
  --config "$PRIVATE/base_forecast/effective_config.json" \
  --model-script "$ROOT/scripts/forecasting.py" \
  --private-output "$PRIVATE/leakage/model_audit.csv" \
  --public-output "$PUBLIC/final_admissible/leakage_audit.json" \
  --expected-models 210

echo "[4/14] Build the temporal robustness audit used by the final analysis"
"$PYTHON" scripts/robustness_audit.py \
  --predictions "$PRIVATE/base_forecast/trip_predictions.csv" \
  --prepared-geolife "$PRIVATE/base_forecast/prepared_data/geolife_ground.csv" \
  --trajectory-audit "$PRIVATE/source_preparation/private/source_trajectory_audit.csv" \
  --output-dir "$PRIVATE/robustness"

echo "[5/14] Apply final geometric admissibility and re-selection"
mkdir -p "$PRIVATE/final"
"$PYTHON" scripts/run_final_admissible.py \
  --config "$PRIVATE/base_forecast/effective_config.json" \
  --model-script "$ROOT/scripts/forecasting.py" \
  --reference-predictions "$PRIVATE/base_forecast/trip_predictions.csv" \
  --output "$PRIVATE/final/final_trip_predictions.csv"

echo "[6/14] Aggregate final results, confidence curves, and bootstrap intervals"
mkdir -p "$PUBLIC/final_admissible"
"$PYTHON" scripts/analyse_final.py \
  --predictions "$PRIVATE/final/final_trip_predictions.csv" \
  --output-dir "$PUBLIC/final_admissible" \
  --bootstrap-draws 10000 \
  --seed 20261020 \
  --temporal-audit "$PRIVATE/robustness/temporal_case_audit.csv"
"$PYTHON" scripts/validate_final.py \
  --predictions "$PRIVATE/final/final_trip_predictions.csv" \
  --public-dir "$PUBLIC/final_admissible" \
  --output "$PUBLIC/final_admissible/validation.json"

echo "[7/14] Evaluate the admissibility-matched top-15 retrieval baseline"
mkdir -p "$PRIVATE/top15"
"$PYTHON" scripts/run_top15_baseline.py \
  --config "$PRIVATE/base_forecast/effective_config.json" \
  --model-script "$ROOT/scripts/forecasting.py" \
  --reference-predictions "$PRIVATE/base_forecast/trip_predictions.csv" \
  --output "$PRIVATE/top15/top15_analogue_predictions.csv" \
  --top-k 15

echo "[8/14] Aggregate the strengthened-baseline comparisons"
mkdir -p "$PUBLIC/top15_baseline"
"$PYTHON" scripts/analyse_top15.py \
  --new-predictions "$PRIVATE/top15/top15_analogue_predictions.csv" \
  --final-predictions "$PRIVATE/final/final_trip_predictions.csv" \
  --output-dir "$PUBLIC/top15_baseline" \
  --bootstrap-draws 10000 \
  --seed 20261020
"$PYTHON" scripts/validate_top15.py \
  --predictions "$PRIVATE/top15/top15_analogue_predictions.csv" \
  --final-predictions "$PRIVATE/final/final_trip_predictions.csv" \
  --public-dir "$PUBLIC/top15_baseline" \
  --output "$PUBLIC/top15_baseline/validation.json"

echo "[9/14] Run ablation, emission-overlap, and temporal-availability diagnostics"
mkdir -p "$PRIVATE/submission_diagnostics" "$PUBLIC/submission_diagnostics"
"$PYTHON" scripts/run_submission_diagnostics.py \
  --config "$PRIVATE/base_forecast/effective_config.json" \
  --model-script "$ROOT/scripts/forecasting.py" \
  --reference-predictions "$PRIVATE/base_forecast/trip_predictions.csv" \
  --predictions-output "$PRIVATE/submission_diagnostics/diagnostic_predictions.csv" \
  --temporal-audit-output "$PRIVATE/submission_diagnostics/temporal_availability_cases.csv" \
  --region-radii-output "$PRIVATE/submission_diagnostics/evaluation_region_radii.csv"

echo "[10/14] Aggregate and validate the submission diagnostics"
"$PYTHON" scripts/analyse_submission_diagnostics.py \
  --predictions "$PRIVATE/submission_diagnostics/diagnostic_predictions.csv" \
  --temporal-audit "$PRIVATE/submission_diagnostics/temporal_availability_cases.csv" \
  --region-radii "$PRIVATE/submission_diagnostics/evaluation_region_radii.csv" \
  --output-dir "$PUBLIC/submission_diagnostics" \
  --bootstrap-draws 10000 \
  --seed 20261020
"$PYTHON" scripts/validate_submission_diagnostics.py \
  --predictions "$PRIVATE/submission_diagnostics/diagnostic_predictions.csv" \
  --temporal-audit "$PRIVATE/submission_diagnostics/temporal_availability_cases.csv" \
  --public-dir "$PUBLIC/submission_diagnostics" \
  --output "$PUBLIC/submission_diagnostics/validation.json"

echo "[11/14] Run same-pool and observed-target review controls"
mkdir -p "$PRIVATE/journal_review_controls" "$PUBLIC/journal_review_controls"
"$PYTHON" scripts/run_journal_review_controls.py \
  --config "$PRIVATE/base_forecast/effective_config.json" \
  --model-script "$ROOT/scripts/forecasting.py" \
  --reference-predictions "$PRIVATE/base_forecast/trip_predictions.csv" \
  --predictions-output "$PRIVATE/journal_review_controls/control_predictions.csv" \
  --temporal-audit-output "$PRIVATE/journal_review_controls/temporal_availability_cases.csv" \
  --region-radii-output "$PRIVATE/journal_review_controls/evaluation_region_radii.csv"

echo "[12/14] Aggregate and validate the review controls"
"$PYTHON" scripts/analyse_journal_review_controls.py \
  --predictions "$PRIVATE/journal_review_controls/control_predictions.csv" \
  --geolife-temporal-audit "$PRIVATE/robustness/temporal_case_audit.csv" \
  --output-dir "$PUBLIC/journal_review_controls" \
  --bootstrap-draws 10000 \
  --seed 20261020
"$PYTHON" scripts/validate_journal_review_controls.py \
  --predictions "$PRIVATE/journal_review_controls/control_predictions.csv" \
  --public-dir "$PUBLIC/journal_review_controls" \
  --output "$PUBLIC/journal_review_controls/validation.json"

echo "[13/14] Compare regenerated aggregate tables with the article reference"
"$PYTHON" scripts/validate_public_results.py \
  --generated "$PUBLIC/final_admissible" \
  --reference "$ROOT/results/reference/final_admissible" \
  --output "$PUBLIC/final_admissible/reference_comparison.json"
"$PYTHON" scripts/validate_public_results.py \
  --generated "$PUBLIC/top15_baseline" \
  --reference "$ROOT/results/reference/top15_baseline" \
  --output "$PUBLIC/top15_baseline/reference_comparison.json"
"$PYTHON" scripts/validate_public_results.py \
  --generated "$PUBLIC/submission_diagnostics" \
  --reference "$ROOT/results/reference/submission_diagnostics" \
  --output "$PUBLIC/submission_diagnostics/reference_comparison.json"
"$PYTHON" scripts/validate_public_results.py \
  --generated "$PUBLIC/journal_review_controls" \
  --reference "$ROOT/results/reference/journal_review_controls" \
  --output "$PUBLIC/journal_review_controls/reference_comparison.json"

echo "[14/14] Complete"
echo "Coordinate-bearing outputs: $PRIVATE (ignored by Git)"
echo "Coordinate-free aggregate outputs: $PUBLIC"
