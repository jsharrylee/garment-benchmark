from __future__ import annotations

import json
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "benchmark" / "configs" / "tshirt_causal_proof_v1.json"

EXPECTED_AXES = (
    "neck_width",
    "neck_depth",
    "shoulder_slope",
    "armhole_depth",
    "bodice_length",
    "sleeve_cap",
    "sleeve_length",
    "sleeve_width",
)
INTERVENTION_KINDS = {
    "DIRECT_INTERVENTION",
    "PROXY_INTERVENTION",
    "NEW_GENERATOR_REQUIRED",
}


class TShirtCausalProofConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_exactly_the_eight_requested_axes_are_declared_in_order(self) -> None:
        self.assertEqual(self.config["schema_version"], "tshirt-causal-proof-config/v1")
        self.assertEqual(tuple(self.config["axis_order"]), EXPECTED_AXES)
        axes = self.config["axes"]
        self.assertEqual(tuple(axis["axis_id"] for axis in axes), EXPECTED_AXES)
        self.assertEqual(len({axis["axis_id"] for axis in axes}), len(EXPECTED_AXES))
        self.assertEqual(self.config["scope"]["axis_count"], len(EXPECTED_AXES))

    def test_each_axis_has_an_explicit_intervention_and_physical_target_contract(self) -> None:
        for axis in self.config["axes"]:
            with self.subTest(axis=axis["axis_id"]):
                kind = axis["intervention_kind"]
                self.assertIn(kind, INTERVENTION_KINDS)
                self.assertTrue(axis["formula_targets"])
                self.assertTrue(axis["semantic_definition"])
                if kind in {"DIRECT_INTERVENTION", "PROXY_INTERVENTION"}:
                    self.assertIsInstance(axis["source_field"], str)
                    self.assertTrue(axis["source_field"])
                else:
                    self.assertIsNone(axis["source_field"])
                    self.assertTrue(axis["proposed_source_field"])

                bounds = axis["provisional_conservative_range"]
                self.assertTrue(math.isfinite(float(bounds["lower"])))
                self.assertTrue(math.isfinite(float(bounds["upper"])))
                self.assertLess(float(bounds["lower"]), float(bounds["upper"]))
                self.assertEqual(bounds["expert_validation"], "PENDING")

    def test_no_pattern_only_pair_is_mislabeled_as_visual_causal_truth(self) -> None:
        self.assertEqual(self.config["scope"]["current_true_four_view_pair_count"], 0)
        for axis in self.config["axes"]:
            with self.subTest(axis=axis["axis_id"]):
                availability = axis["current_availability"]
                self.assertEqual(availability["true_four_view_pair_count"], 0)
                self.assertLessEqual(
                    availability["simulation_geometry_clean_pair_count"],
                    availability["pattern_training_pair_count"],
                )
                if axis["intervention_kind"] == "NEW_GENERATOR_REQUIRED":
                    self.assertEqual(availability["pattern_training_pair_count"], 0)

    def test_pair_and_split_contracts_prevent_single_axis_and_baseline_leakage(self) -> None:
        pair = self.config["pairing_contract"]
        self.assertEqual(pair["changed_author_facing_field_count"], 1)
        self.assertIn("body_yaml_and_body_mesh", pair["fixed_within_pair"])
        self.assertIn("baseline_not_self_intersecting", pair["required_pattern_gates"])
        self.assertIn("intervention_not_self_intersecting", pair["required_pattern_gates"])
        self.assertEqual(self.config["split_contract"]["group_key"], "baseline_state_id")
        self.assertIn("pair_level_random_split", self.config["split_contract"]["forbidden"])

    def test_visual_receipt_requires_semantic_render_passes(self) -> None:
        receipt = self.config["render_receipt_contract"]
        self.assertEqual(set(receipt["required_views"]), {"front", "back", "left", "right"})
        for required in ("alpha", "depth", "surface_normal", "panel_id", "pattern_uv"):
            self.assertIn(required, receipt["required_member_outputs"])
        self.assertTrue(receipt["visual_training_eligible_only_after_receipt_validation"])


if __name__ == "__main__":
    unittest.main()
