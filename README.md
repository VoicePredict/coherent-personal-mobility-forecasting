# Confidence-Aware Personalized Mobility Forecasting with Destination--Route Coherence

This repository is the reproducibility artifact for the article
*Confidence-Aware Personalized Mobility Forecasting with Destination--Route
Coherence*. The final method builds train-only personal destination regions,
route analogues, and grid-transition candidates. Candidates are retained only
when the route begins within 1 km of the observed prefix, ends within 500 m of
its proposed destination, and, for a grid route, reaches the destination cell
without an artificial connector. The highest-ranked admissible candidate is
emitted directly or filtered by the multi-output evidence score `q`.

The artifact matches the final 8,526-case evaluation:

- 8,466 split-specific cases from 103 eligible GeoLife users;
- 60 cases from two GTA-V histories;
- DBSCAN destination neighbourhood `eps = 500 m`;
- random-seed-13 and chronological 75/25 per-individual splits;
- a top-15 admissibility-matched route-retrieval baseline;
- paired 10,000-draw user-clustered bootstrap intervals;
- start-gap sensitivities at 1,000, 500, and 350 m.

## GeoLife Is Not Redistributed

The repository contains neither GeoLife 1.3 nor coordinate-bearing data
derived from it. Obtain the dataset from the
[official Microsoft download page](https://www.microsoft.com/en-us/download/details.aspx?id=52367)
under its original terms. The setup script downloads from Microsoft's official
file URL only after explicit acknowledgement, verifies the pinned SHA-256,
checks the archive structure, extracts it under `data/private/`, and prepares
the forecasting input under `outputs/private/`. Both locations are ignored by
Git.

Pinned release:

```text
File:    Geolife Trajectories 1.3.zip
Size:    313164406 bytes
SHA-256: 1107c5ac064d0a23c8d021a8736a77e53abc75b227062e6260342c6a8d86bdb6
```

See [DATA_NOTICE.md](DATA_NOTICE.md) and [data/README.md](data/README.md) before
running the preparation step.

## Environment

Python 3.10.12 was used for the reported run.

```bash
python3.10 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.lock.txt
```

## Download and Prepare GeoLife

Automatic download, verification, extraction, and preparation:

```bash
.venv/bin/python scripts/setup_geolife.py --accept-license
```

To use an archive downloaded manually from Microsoft:

```bash
.venv/bin/python scripts/setup_geolife.py \
  --archive "/path/to/Geolife Trajectories 1.3.zip"
```

An already extracted release can be used without copying it:

```bash
.venv/bin/python scripts/setup_geolife.py \
  --raw-root "/path/to/Geolife Trajectories 1.3"
```

The preparation is source-faithful: each original PLT file remains one trip;
no trips are split or merged and no points are downsampled. A complete trip is
excluded if any supplied label identifies an explicit air or water mode.
Car, taxi, bus, walking, cycling, subway, train, and `unknown` are retained.

## Reproduce the Article Results

After GeoLife has been prepared, run:

```bash
bash scripts/run_full_pipeline.sh
```

For a first run that should also download GeoLife automatically:

```bash
ACCEPT_GEOLIFE_LICENSE=1 bash scripts/run_full_pipeline.sh
```

For an existing archive or extraction, set `GEOLIFE_ARCHIVE` or
`GEOLIFE_RAW_ROOT` when invoking the same pipeline.

The pipeline performs source preparation, frozen candidate generation, final
geometric admissibility and re-selection, confidence analysis, the strengthened
top-15 baseline, paired user-clustered bootstrap, robustness analysis, and
comparison with the tracked aggregate reference tables.

Outputs are separated by redistribution status:

```text
outputs/private/   GeoLife-derived tables and case-level predictions; ignored
outputs/public/    coordinate-free aggregate results; ignored regenerations
results/reference/ tracked coordinate-free article results
```

Do not publish or commit `data/private/` or `outputs/private/`.

## Principal Reference Results

Under the primary 1-km route-start tolerance:

| Policy | Coverage | All-case EAD | Accepted EAD | Destination <=1 km | Short horizon <=1 km | Route score |
|---|---:|---:|---:|---:|---:|---:|
| Coherent, admissibility only | 84.38% | 60.30% | 71.46% | 73.76% | 79.15% | 62.64% |
| Coherent, `q >= 0.48` | 81.48% | 59.61% | 73.15% | 75.47% | 80.77% | 64.23% |
| Route retrieval, first admissible of top 15 | 81.23% | 56.77% | 69.88% | 72.26% | 79.45% | 60.75% |

Over the 103 GeoLife users, the admissibility-only coherent policy exceeded
top-15 retrieval by 3.17 percentage points in equal-user all-case EAD (95% CI
[1.30, 4.94]). Case-level and coordinate-bearing outputs are deliberately not
part of the repository.

## Repository Layout

```text
config/publication.json      Frozen article configuration
data/raw/                    Project-owned compressed GTA-V input only
data/private/                Local GeoLife archive and extraction (ignored)
scripts/setup_geolife.py     Official download, checksum, extraction, preparation
scripts/prepare_geolife.py   Source-faithful PLT conversion and filtering
scripts/forecasting.py       Train-only candidate construction and evaluation
scripts/geometry_guard.py    Candidate admissibility and re-selection
scripts/run_final_admissible.py
scripts/run_top15_baseline.py
scripts/analyse_*.py         Aggregate tables, confidence, and bootstrap
scripts/run_full_pipeline.sh End-to-end reproduction
results/reference/           Coordinate-free article aggregates
```

The GTA-V file is project data and is retained so that the descriptive
two-history slice can be reproduced. GeoLife remains subject to Microsoft's
separate dataset terms and is never supplied by this repository.
