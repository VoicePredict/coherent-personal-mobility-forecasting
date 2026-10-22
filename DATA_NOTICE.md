# Data Notice

## GeoLife

GeoLife GPS Trajectories 1.3 is a Microsoft Research dataset with terms that
are separate from this repository. Neither the archive, its PLT files, a
compiled coordinate CSV, prepared coordinate arrays, nor per-trajectory
coordinate predictions are distributed here.

The official source is:

- download page: <https://www.microsoft.com/en-us/download/details.aspx?id=52367>
- file: `Geolife Trajectories 1.3.zip`
- size: `313164406` bytes
- SHA-256: `1107c5ac064d0a23c8d021a8736a77e53abc75b227062e6260342c6a8d86bdb6`

The user must review and comply with the original terms. The helper script
downloads only from Microsoft's official file host after `--accept-license`
has been supplied, or accepts a manually downloaded archive. All source and
coordinate-bearing derived files are written to Git-ignored directories.

## GTA-V

`data/raw/trips_lines_GTA.csv.gz` contains project-generated virtual-city
trajectories used for the descriptive GTA-V slice. It is not derived from
GeoLife. Its SHA-256 is
`7eaf8e923bb91d18b5dd6d7680cb46b850f316006fca56757565f4572ef04c3e`.

## Tracked Results

`results/reference/` contains coordinate-free aggregate tables only. It does
not contain user trajectories, route geometries, prepared coordinate tables,
or case-level predictions.
