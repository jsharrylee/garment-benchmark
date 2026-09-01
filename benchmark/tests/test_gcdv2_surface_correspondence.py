from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np

from benchmark.drafting_semantics.gcdv2_surface_correspondence import (
    ELEMENT_NAMES,
    TSHIRT_PARAMETER_NAMES,
    build_tshirt_surface_example,
    build_tshirt_visual_correspondence_model,
    collapse_uv_expanded_vertices,
    extract_tshirt_physical_parameters,
    project_gcdv2_orthographic,
    sample_specification_edge,
)


ROOT = Path(__file__).resolve().parents[2]


class SurfaceCorrespondenceTests(unittest.TestCase):
    def test_front_and_back_elements_remain_distinct(self) -> None:
        self.assertIn("front_shoulder", ELEMENT_NAMES)
        self.assertIn("back_shoulder", ELEMENT_NAMES)
        self.assertIn("front_armhole", ELEMENT_NAMES)
        self.assertIn("back_armhole", ELEMENT_NAMES)
        self.assertNotIn("shoulder", ELEMENT_NAMES)
        self.assertNotIn("armhole", ELEMENT_NAMES)

    def test_body_length_uses_complete_panel_extent(self) -> None:
        record = {
            "panels": [
                {
                    "id": "front",
                    "role": "front_bodice",
                    "vertices_cm": [[0.0, 0.0], [5.0, 60.0], [10.0, 20.0]],
                    "edges": [
                        {
                            "role": "center_front",
                            "length_cm": 20.0,
                            "start_cm": [0.0, 0.0],
                            "end_cm": [0.0, 20.0],
                        }
                    ],
                },
                {
                    "id": "back",
                    "role": "back_bodice",
                    "vertices_cm": [[0.0, -2.0], [5.0, 62.0], [10.0, 20.0]],
                    "edges": [
                        {
                            "role": "center_back",
                            "length_cm": 16.0,
                            "start_cm": [0.0, -2.0],
                            "end_cm": [0.0, 14.0],
                        }
                    ],
                },
            ]
        }
        values, valid, definitions = extract_tshirt_physical_parameters(
            record, {"pattern": {"panels": {}}}
        )
        self.assertTrue(bool(valid[4]))
        self.assertAlmostEqual(float(values[4]), 62.0)
        self.assertIn("complete", definitions["body_length_cm"])

    def test_resolved_shoulder_role_is_used_for_physical_parameters(self) -> None:
        record = {
            "panels": [
                {
                    "id": "front",
                    "role": "front_bodice",
                    "vertices_cm": [[0.0, 0.0], [2.0, 1.0]],
                    "edges": [
                        {
                            "id": "front:e0",
                            "role": "side_seam",
                            "length_cm": 2.24,
                            "start_cm": [0.0, 0.0],
                            "end_cm": [2.0, 1.0],
                        }
                    ],
                }
            ]
        }
        values, valid, _ = extract_tshirt_physical_parameters(
            record,
            {"pattern": {"panels": {}}},
            resolved_roles={"front:e0": "shoulder"},
        )
        self.assertTrue(bool(valid[2]))
        self.assertAlmostEqual(float(values[2]), 26.565, places=3)

    def test_uv_collapse_preserves_first_occurrence_order(self) -> None:
        vertices = np.asarray(
            ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 2.0, 0.0), (1.0, 0.0, 0.0)),
            dtype=np.float32,
        )
        unique, first = collapse_uv_expanded_vertices(vertices)
        np.testing.assert_array_equal(first, np.asarray((0, 2)))
        np.testing.assert_allclose(unique, vertices[[0, 2]])

    def test_intrinsic_line_and_cubic_sampling(self) -> None:
        panel = {
            "vertices": [[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]],
            "edges": [
                {"endpoints": [0, 1]},
                {
                    "endpoints": [1, 2],
                    "curvature": {
                        "type": "cubic",
                        "params": [[0.25, 0.2], [0.75, -0.2]],
                    },
                },
            ],
        }
        line = sample_specification_edge(panel, 0, samples=9)
        cubic = sample_specification_edge(panel, 1, samples=9)
        np.testing.assert_allclose(line[0], panel["vertices"][0])
        np.testing.assert_allclose(line[-1], panel["vertices"][1])
        np.testing.assert_allclose(cubic[0], panel["vertices"][1])
        np.testing.assert_allclose(cubic[-1], panel["vertices"][2])
        self.assertGreater(float(np.ptp(cubic[:, 1])), 0.0)

    def test_projection_uses_semantic_view_order(self) -> None:
        vertices = np.asarray(((-1.0, 0.0, 2.0), (1.0, 0.0, -2.0), (0.0, 4.0, 0.0)))
        xy, depth = project_gcdv2_orthographic(vertices)
        self.assertEqual(xy.shape, (4, 3, 2))
        self.assertEqual(depth.shape, (4, 3))
        self.assertGreater(depth[0, 0], depth[0, 1])  # front is +dataset-Z
        self.assertGreater(depth[1, 1], depth[1, 0])  # back is -dataset-Z
        self.assertGreater(depth[2, 0], depth[2, 1])  # left is -dataset-X
        self.assertGreater(depth[3, 1], depth[3, 0])  # right is +dataset-X

    def test_model_output_contract(self) -> None:
        import torch

        model = build_tshirt_visual_correspondence_model(width=32, heads=4, layers=1)
        output = model(torch.randn(2, 4, 85, 256))
        self.assertEqual(
            tuple(output["element_location_logits"].shape),
            (2, len(ELEMENT_NAMES), 4, 85),
        )
        self.assertEqual(
            tuple(output["parameter_mean"].shape),
            (2, len(TSHIRT_PARAMETER_NAMES)),
        )
        self.assertEqual(
            tuple(output["parameter_element_attention"].shape),
            (2, 4, len(TSHIRT_PARAMETER_NAMES), len(ELEMENT_NAMES)),
        )

    @unittest.skipUnless(
        (ROOT / "data/processed/garmentcode_v2/batch_0_full/rand_05H6WP8PJX").is_dir(),
        "local GCDv2 technical-evaluation data is not installed",
    )
    def test_real_sample_has_all_elements_and_parameters(self) -> None:
        index = next(
            json.loads(line)
            for line in (ROOT / "artifacts/gcdv2_exact_pairs_v1/index.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            if "rand_05H6WP8PJX" in line
        )
        record = next(
            json.loads(line)
            for line in (
                ROOT / "artifacts/drafting_semantics/gcdv2_multigarment/records.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            if "rand_05H6WP8PJX" in line
        )
        example = build_tshirt_surface_example(
            index_row=index,
            semantic_record=record,
            raw_root=ROOT / "data/processed/garmentcode_v2/batch_0_full",
        )
        self.assertTrue(bool(np.all(example.element_valid)))
        self.assertTrue(bool(np.all(example.parameter_valid)))
        self.assertGreater(int(example.element_vertex_counts.min()), 0)
        self.assertFalse(bool(example.audit["causal_claim"]))

    @unittest.skipUnless(
        (ROOT / "data/processed/garmentcode_v2/batch_0_full/rand_1JGSONFALQ").is_dir(),
        "local GCDv2 technical-evaluation data is not installed",
    )
    def test_previous_front_shoulder_quarantine_is_resolved(self) -> None:
        sample_id = "rand_1JGSONFALQ"
        index = next(
            json.loads(line)
            for line in (ROOT / "artifacts/gcdv2_exact_pairs_v1/index.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            if sample_id in line
        )
        record = next(
            json.loads(line)
            for line in (
                ROOT / "artifacts/drafting_semantics/gcdv2_multigarment/records.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            if sample_id in line
        )
        example = build_tshirt_surface_example(
            index_row=index,
            semantic_record=record,
            raw_root=ROOT / "data/processed/garmentcode_v2/batch_0_full",
        )
        self.assertTrue(bool(np.all(example.element_valid)))
        self.assertTrue(bool(np.all(example.parameter_valid)))
        resolver = example.audit["semantic_role_resolver"]
        self.assertGreater(int(resolver["override_count"]), 0)
        self.assertIn(
            "shoulder", {value["resolved"] for value in resolver["overrides"].values()}
        )


if __name__ == "__main__":
    unittest.main()
