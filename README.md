# Confidence-Aware Personalized Mobility Forecasting with Destination--Route Coherence

This repository is the reproducibility artifact for the article
*Confidence-Aware Personalized Mobility Forecasting with Destination--Route
Coherence*. The method builds train-only personal destination regions,
route analogues, and grid-transition candidates. Candidates are retained only
when the route begins within 1 km of the observed prefix, ends within 500 m of
its proposed destination, and, for a grid route, reaches the destination cell
without an artificial connector. The highest-ranked admissible candidate is
emitted directly or filtered by the multi-output evidence score `q`.

The artifact reproduces the 8,526-case evaluation reported in the article:

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

## Tests

Run the unit and validation-regression tests from the repository root:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The tests cover the frozen article configuration, candidate-admissibility
rules, GeoLife archive identity, top-15 fallback order, complete
case--arm--method matrices, and rejection of non-finite geometry diagnostics.

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

The pipeline performs source preparation, candidate generation, geometric
admissibility filtering and re-selection, confidence analysis, the
admissibility-matched top-15 baseline, paired user-clustered bootstrap,
robustness analysis, and comparison with the tracked aggregate reference
tables.

Outputs are separated by redistribution status:

```text
outputs/private/   GeoLife-derived tables and case-level predictions; ignored
outputs/public/    coordinate-free aggregate results; ignored regenerations
results/reference/ tracked coordinate-free article results
```

Do not publish or commit `data/private/` or `outputs/private/`.

## Computational Requirements

The reported runs used Python 3.10.12 on Linux with an Intel Core i5-8259U
processor (4 physical cores, 8 hardware threads) and 16 GiB of RAM. No GPU is
required. The computationally intensive stages were effectively
single-threaded.

On that system, the stages in `run_full_pipeline.sh` required approximately
50 min in total when starting from an already extracted GeoLife release; data
download, archive extraction, and environment installation are not included.
The measured peak resident memory for the epsilon-500 forecasting and final
selection stages was approximately 8.5 GiB. Runtime depends on processor and
storage performance; at least 12 GiB of available RAM is recommended.

## Principal Reference Results

Under the primary 1-km route-start tolerance:

| Policy | Coverage | All-case EAD | Accepted EAD | Destination <=1 km | Short horizon <=1 km | Route score |
|---|---:|---:|---:|---:|---:|---:|
| Coherent, admissibility only | 84.38% | 60.30% | 71.46% | 73.76% | 79.15% | 62.64% |
| Coherent, `q >= 0.48` | 81.48% | 59.61% | 73.15% | 75.47% | 80.77% | 64.23% |
| Route retrieval, first admissible of top 15 | 81.23% | 56.77% | 69.88% | 72.26% | 79.45% | 60.75% |

Over the 103 GeoLife users, the admissibility-only coherent policy exceeded
top-15 retrieval by 3.17 percentage points in equal-user all-case EAD (95% CI
[1.30, 4.94]). Case-level and coordinate-bearing outputs are not distributed.

## Repository Layout

```text
config/publication.json      Article configuration
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
separate dataset terms and is not supplied by this repository.

## Versioned Artifact

The results reported in the article correspond to
[version v1.0](https://github.com/VoicePredict/coherent-personal-mobility-forecasting/tree/v1.0).
Use this tag for publication reproduction; the default branch may receive
subsequent documentation or validation maintenance.

## License

Original code, documentation, and the project-generated GTA-V input are
distributed under the [MIT License](LICENSE). GeoLife and third-party
dependencies remain subject to their respective upstream terms; see
[DATA_NOTICE.md](DATA_NOTICE.md).
