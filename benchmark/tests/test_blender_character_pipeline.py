import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark.adapters.blender_character import prepare_layer_bundles
from benchmark.evaluation.generative_routing import assess_generated_patterns


class BlenderCharacterPipelineTests(unittest.TestCase):
    def test_falls_back_between_generators_not_templates(self):
        result = assess_generated_patterns(
            {"structural_export": "FAILED_VALIDATION"},
            {"valid": True, "panel_count": 12, "edge_count": 64, "stitch_pair_count": 24, "panel_closure_gap_max": 2.0},
            {"structural_export": "PASS"},
        )
        self.assertEqual(result["primary_generated_draft"], "garment_particles")
        self.assertEqual(result["technical_status"], "DRAFT_PATTERN_AVAILABLE")
        self.assertFalse(result["generation_contract"]["template_retrieval"])
        self.assertFalse(result["generation_contract"]["nearest_pattern_selection"])

    def test_no_generated_output_means_no_draft(self):
        result = assess_generated_patterns({"structural_export": "FAILED_VALIDATION"}, {"valid": False}, {"structural_export": "FAILED_VALIDATION"})
        self.assertIsNone(result["primary_generated_draft"])
        self.assertEqual(result["technical_status"], "NO_VALID_GENERATED_DRAFT")

    def test_particle_arrays_can_be_retained_as_repair_required_draft(self):
        result = assess_generated_patterns(
            {"structural_export": "FAILED_VALIDATION"},
            {"valid": True, "panel_closure_gap_max": 1.0},
            {"structural_export": "FAILED_VALIDATION"},
        )
        self.assertEqual(result["primary_generated_draft"], "garment_particles")
        self.assertEqual(result["technical_status"], "DRAFT_REQUIRES_REPAIR")

    def test_large_particle_closure_gap_requires_review(self):
        result = assess_generated_patterns(
            {"structural_export": "PASS"},
            {"valid": True, "panel_closure_gap_max": 7.2},
            {"structural_export": "FAILED_VALIDATION"},
        )
        self.assertEqual(result["primary_generated_draft"], "reweaver")
        self.assertTrue(result["garment_particles"]["closure_review_required"])

    def test_layer_bundle_separates_masks_and_preserves_four_view_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rgb = root / "reweaver" / "render_output" / "rgb"
            masks = root / "masks"
            rgb.mkdir(parents=True)
            masks.mkdir(parents=True)
            source = np.full((518, 518, 3), 255, dtype=np.uint8)
            source[100:420, 180:340] = (40, 80, 180)
            mask = np.zeros((518, 518), dtype=np.uint8)
            mask[100:420, 180:340] = 255
            for camera in ("CAM000", "CAM001", "CAM002", "CAM003"):
                Image.fromarray(source).save(rgb / f"{camera}.png")
                Image.fromarray(mask).save(masks / f"{camera}.png")
            manifest = prepare_layer_bundles(root, split_ratio=0.5)
            upper = np.asarray(Image.open(root / "layers" / "upper" / "masks" / "CAM000.png")) > 0
            lower = np.asarray(Image.open(root / "layers" / "lower" / "masks" / "CAM000.png")) > 0
            self.assertFalse(np.logical_and(upper, lower).any())
            np.testing.assert_array_equal(np.logical_or(upper, lower), mask > 0)
            self.assertFalse(manifest["source_pattern_inferred_from_uv"])


if __name__ == "__main__":
    unittest.main()
