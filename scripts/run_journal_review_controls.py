#!/usr/bin/env python3
"""Run the same-pool fixed-family-order publication control."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTIC_SCRIPT = ROOT / "scripts" / "run_submission_diagnostics.py"


def load_diagnostics() -> Any:
    spec = importlib.util.spec_from_file_location(
        "journal_review_base_diagnostics", DIAGNOSTIC_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {DIAGNOSTIC_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def select_family_priority(candidates: list[Any]) -> Any | None:
    """Select by the frozen analogue, grid, destination-family order."""
    indexed = list(enumerate(candidates))
    family_groups = (
        {"route_analogue"},
        {"grid_transition"},
        {"destination_cluster", "destination_cluster_topk", "destination_prior"},
    )
    for families in family_groups:
        members = [
            (index, candidate)
            for index, candidate in indexed
            if str(candidate.family) in families
        ]
        if not members:
            continue
        if families == {"route_analogue"}:
            members.sort(key=lambda item: (-float(item[1].route_score), item[0]))
        else:
            members.sort(key=lambda item: (-float(item[1].dest_score), item[0]))
        return members[0][1]
    return None


def main() -> int:
    diagnostics = load_diagnostics()
    original_select = diagnostics.select_variant

    def select_variant(
        model: Any,
        candidates: list[Any],
        variant: str,
        threshold_m: float,
    ) -> Any | None:
        if variant == "fixed_family_order":
            return select_family_priority(candidates)
        return original_select(model, candidates, variant, threshold_m)

    diagnostics.VARIANTS = ["full", "fixed_family_order", "top15_retrieval"]
    diagnostics.select_variant = select_variant
    return int(diagnostics.main())


if __name__ == "__main__":
    raise SystemExit(main())
