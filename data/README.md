# Data Preparation

Only the project-generated GTA-V input is tracked. GeoLife is acquired from
Microsoft by the user and remains local.

## GeoLife 1.3

Run either:

```bash
.venv/bin/python scripts/setup_geolife.py --accept-license
```

or:

```bash
.venv/bin/python scripts/setup_geolife.py \
  --archive "/path/to/Geolife Trajectories 1.3.zip"
```

For an already extracted release, pass `--raw-root` instead of `--archive`.

The script performs the following checks before any model input is created:

1. exact archive size and SHA-256;
2. safe ZIP member paths and the expected release root;
3. 182 user directories, 18,670 PLT files, and 69 label files;
4. one original PLT file per trip, without segmentation or downsampling;
5. at least 10 valid points and at least 5 min duration per retained trip;
6. complete-trip removal if an explicit air or water label occurs;
7. at least 20 eligible trips per retained user.

Car, taxi, bus, walking, cycling, subway, train, and unlabeled (`unknown`)
trajectories are retained. The mode attached to each point is obtained from
the supplied interval labels. A trip's descriptive modal label is its most
frequent known point label, or `unknown` if no known label is present.

The source-preparation procedure yields 103 eligible users and 16,764 retained
GeoLife trips. The two split protocols produce 8,466 GeoLife evaluation cases.

Generated paths, relative to the repository root:

```text
data/private/Geolife Trajectories 1.3.zip
data/private/geolife-1.3/Geolife Trajectories 1.3/
outputs/private/source_preparation/private/geolife_source.csv.gz
outputs/private/source_preparation/private/source_trajectory_audit.csv
```

All are ignored by Git and must remain uncommitted.

## GTA-V

The tracked compressed input is `data/raw/trips_lines_GTA.csv.gz`. The
forecasting launcher normalizes decimal-comma coordinates and in-game time,
then applies the same minimum trip-count, point-count, and duration rules.
