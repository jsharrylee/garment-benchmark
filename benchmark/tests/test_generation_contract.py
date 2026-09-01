from __future__ import annotations

import json
import unittest
from pathlib import Path


class GenerationContractTests(unittest.TestCase):
    def test_retrieval_anchored_generation_is_default(self):
        config_path = Path(__file__).parents[1] / "configs" / "benchmark.yaml"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        contract = config["reconstruction_contract"]

        self.assertEqual(contract["mode"], "retrieval_anchored_v2")
        self.assertFalse(contract["generate_variable_topology"])
        self.assertTrue(contract["template_retrieval_allowed"])
        self.assertTrue(contract["nearest_pattern_selection_allowed"])
        self.assertTrue(contract["selection_requires_structural_validation"])
        self.assertTrue(contract["selection_requires_simulation_rerank_for_final_acceptance"])
        self.assertIn("anchor_retrieval", contract["garmentcode_roles"])

    def test_legacy_generation_contract_is_preserved_for_reproduction(self):
        config_path = Path(__file__).parents[1] / "configs" / "benchmark.yaml"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        legacy = config["legacy_reconstruction_contract"]

        self.assertEqual(legacy["mode"], "generative_v1")
        self.assertTrue(legacy["generate_variable_topology"])
        self.assertFalse(legacy["template_retrieval_allowed"])


if __name__ == "__main__":
    unittest.main()
