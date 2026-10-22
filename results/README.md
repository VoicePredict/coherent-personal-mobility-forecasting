# Coordinate-Free Reference Results

The tracked files under `reference/` are the aggregate outputs used in the
article and by `scripts/validate_public_results.py`.

- `reference/final_admissible/` contains policy metrics, confidence operating
  points, risk--coverage tables, support strata, start-gap sensitivity,
  development-user exclusion, temporal sensitivity, and paired bootstrap
  intervals.
- `reference/top15_baseline/` contains the strengthened route-retrieval
  baseline, selected-rank counts, comparisons with the single-analogue
  diagnostic, and paired bootstrap intervals.

No tracked file contains latitude, longitude, projected coordinates, route
geometry, or per-case GeoLife predictions. Fresh coordinate-bearing results
are created only under the ignored `outputs/private/` directory.
