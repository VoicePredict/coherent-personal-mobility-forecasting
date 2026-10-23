# Scripts

## Data acquisition and preparation

- `setup_geolife.py`: optional official download, SHA-256 verification, safe
  extraction, and complete source preparation.
- `verify_geolife_archive.py`: stand-alone archive and member-count check.
- `prepare_geolife.py`: one-PLT-one-trip conversion, complete-trip mode
  filtering, and source-quality audit.
- `validate_geolife_preparation.py`: preparation invariants and checksums.
- `data_preparation.py`: normalization of the prepared GeoLife and GTA-V inputs
  for the forecasting code.

## Forecasting and selection

- `forecasting.py`: train-only destination regions, route analogue
  retrieval, grid transitions, candidate scoring, and evaluation.
- `run_forecasting.py`: the two-split 8,526-case base run.
- `geometry_guard.py`: route-start, endpoint, and grid-reachability checks.
- `run_final_admissible.py`: candidate filtering and coherent
  re-selection at 1,000, 500, and 350 m start-gap tolerances.
- `run_top15_baseline.py`: first admissible route analogue among the
  similarity-ranked top 15.
- `run_submission_diagnostics.py`: fixed ablation variants, top-15 emission
  overlap, candidate-family composition, and case-level temporal-availability
  checks for chronological evaluation.

## Analysis and validation

- `analyse_guarded.py`, `analyse_final.py`, and `analyse_top15.py`: aggregate
  metrics, support strata, risk--coverage analysis, and paired user-clustered
  bootstrap.
- `analyse_submission_diagnostics.py`: equal-user ablation, paired bootstrap,
  common-emission, target-endpoint, and evaluation-region aggregates.
- `robustness_audit.py`: development-overlap and timestamp sensitivity data.
- `validate_final.py`, `validate_top15.py`, and
  `validate_submission_diagnostics.py`: complete case matrices, cohort and
  geometric invariants, finite values, and expected public outputs.
- `validate_public_results.py`: comparison with tracked coordinate-free
  reference tables.
- `run_full_pipeline.sh`: end-to-end reproduction.
