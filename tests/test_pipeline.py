from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_final_admissible as final  # noqa: E402
import run_top15_baseline as top15  # noqa: E402
import verify_geolife_archive as geolife  # noqa: E402


def candidate(*, family: str = "route_analogue", end_x: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(
        family=family,
        dest_x=0.0,
        dest_y=0.0,
        route_x=np.asarray([0.0, end_x]),
        route_y=np.asarray([0.0, 0.0]),
        route_trip=None,
    )


class PublicationPipelineTests(unittest.TestCase):
    def test_frozen_configuration_matches_article(self) -> None:
        config = json.loads((ROOT / "config" / "publication.json").read_text())
        experiment = config["experiment"]
        self.assertEqual(experiment["destination_cluster_eps_m"], 500.0)
        self.assertEqual(experiment["projection_strategy"], "train_centered_aeqd")
        self.assertEqual(experiment["destination_analogue_top_k"], 15)
        self.assertEqual(config["data_selection"]["required_release"], "1.3")
        self.assertIn("bike", config["data_selection"]["allowed_modes"])

    def test_final_gate_checks_start_and_endpoint(self) -> None:
        arm = {
            "require_grid_reach": True,
            "max_start_gap_m": 1000.0,
            "max_endpoint_gap_m": 500.0,
        }
        self.assertTrue(
            final.final_is_eligible(
                candidate(),
                {"route_start_gap_m": 999.0, "grid_reached_destination": np.nan},
                arm,
            )
        )
        self.assertFalse(
            final.final_is_eligible(
                candidate(),
                {"route_start_gap_m": 1001.0, "grid_reached_destination": np.nan},
                arm,
            )
        )
        self.assertFalse(
            final.final_is_eligible(
                candidate(end_x=501.0),
                {"route_start_gap_m": 0.0, "grid_reached_destination": np.nan},
                arm,
            )
        )

    def test_grid_candidate_must_reach_destination(self) -> None:
        arm = {
            "require_grid_reach": True,
            "max_start_gap_m": 1000.0,
            "max_endpoint_gap_m": 500.0,
        }
        geometry = {"route_start_gap_m": 0.0, "grid_reached_destination": 0}
        self.assertFalse(final.final_is_eligible(candidate(family="grid_transition"), geometry, arm))
        geometry["grid_reached_destination"] = 1
        self.assertTrue(final.final_is_eligible(candidate(family="grid_transition"), geometry, arm))

    def test_top15_selects_first_admissible_in_original_order(self) -> None:
        candidates = [SimpleNamespace(rank=value) for value in (1, 2, 3)]
        geometries = [(1500.0, 0.0), (800.0, 0.0), (100.0, 0.0)]
        selected, rank, count = top15.first_admissible(
            candidates,
            geometries,
            max_start_gap_m=1000.0,
            max_endpoint_gap_m=500.0,
        )
        self.assertIs(selected, candidates[1])
        self.assertEqual(rank, 2)
        self.assertEqual(count, 2)

    def test_pinned_archive_identity(self) -> None:
        self.assertEqual(geolife.EXPECTED_PLT_FILES, 18670)
        self.assertEqual(geolife.EXPECTED_LABEL_FILES, 69)
        self.assertEqual(
            geolife.EXPECTED_SHA256,
            "1107c5ac064d0a23c8d021a8736a77e53abc75b227062e6260342c6a8d86bdb6",
        )


if __name__ == "__main__":
    unittest.main()
