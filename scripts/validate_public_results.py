#!/usr/bin/env python3
"""Compare regenerated coordinate-free tables with the frozen article tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def compare_csv(observed_path: Path, expected_path: Path) -> None:
    observed = pd.read_csv(observed_path)
    expected = pd.read_csv(expected_path)
    if list(observed.columns) != list(expected.columns):
        raise AssertionError(f"Column mismatch: {observed_path.name}")
    if len(observed) != len(expected):
        raise AssertionError(f"Row-count mismatch: {observed_path.name}")
    for column in observed.columns:
        left = pd.to_numeric(observed[column], errors="coerce")
        right = pd.to_numeric(expected[column], errors="coerce")
        numeric = left.notna() | right.notna()
        if numeric.any():
            if not np.allclose(
                left[numeric].to_numpy(dtype=float),
                right[numeric].to_numpy(dtype=float),
                rtol=0.0,
                atol=1e-10,
                equal_nan=True,
            ):
                raise AssertionError(f"Numeric mismatch: {observed_path.name}:{column}")
        text_rows = ~numeric
        if text_rows.any():
            a = observed.loc[text_rows, column].fillna("").astype(str)
            b = expected.loc[text_rows, column].fillna("").astype(str)
            if not a.reset_index(drop=True).equals(b.reset_index(drop=True)):
                raise AssertionError(f"Text mismatch: {observed_path.name}:{column}")


def compare_directory(observed: Path, expected: Path) -> list[str]:
    checked: list[str] = []
    for expected_path in sorted(expected.glob("*")):
        if expected_path.suffix not in {".csv", ".json"}:
            continue
        observed_path = observed / expected_path.name
        if not observed_path.is_file():
            raise AssertionError(f"Missing regenerated result: {observed_path}")
        if expected_path.suffix == ".csv":
            compare_csv(observed_path, expected_path)
        else:
            actual_json = json.loads(observed_path.read_text(encoding="utf-8"))
            expected_json = json.loads(expected_path.read_text(encoding="utf-8"))
            if actual_json != expected_json:
                raise AssertionError(f"JSON mismatch: {observed_path.name}")
        checked.append(expected_path.name)
    return checked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checked = compare_directory(args.generated, args.reference)
    report = {"status": "passed", "checked_files": checked}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
