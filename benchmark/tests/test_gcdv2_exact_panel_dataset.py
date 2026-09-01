from __future__ import annotations

import unittest

from benchmark.gcdv2_exact.panel_dataset import canonical_boundary, infer_panel_role, panel_target


def _panel():
    vertices = [[0.0, 0.0], [10.0, 0.0], [10.0, 20.0], [0.0, 20.0]]
    edges = []
    for index, endpoints in enumerate(([2, 1], [0, 3], [1, 0], [3, 2])):
        start, end = (vertices[value] for value in endpoints)
        edges.append(
            {
                "edge_id": f"front.edge_{index}",
                "edge_index": index,
                "endpoints": list(endpoints),
                "start_cm": list(start),
                "end_cm": list(end),
                "length_cm": ((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5,
                "chord_direction_deg": 0.0,
                "start_tangent_deg": 0.0,
                "end_tangent_deg": 0.0,
                "curve": {"type": "line", "controls_cm": [], "source_type": "line", "source_params": None},
            }
        )
    return {
        "panel_id": "right_ftorso",
        "source_order_index": 0,
        "source_label": "body",
        "vertices_cm": vertices,
        "edges": edges,
        "local_curve_bbox_cm": [0.0, 0.0, 10.0, 20.0],
    }


class ExactSinglePanelDatasetTests(unittest.TestCase):
    def test_shuffled_reversed_edges_become_one_ccw_cycle(self):
        panel = _panel()
        order, edges = canonical_boundary(panel)
        self.assertEqual(len(order), 4)
        self.assertEqual(len(edges), 4)
        self.assertEqual({value for value in order}, {0, 1, 2, 3})
        area = 0.5 * sum(
            panel["vertices_cm"][a][0] * panel["vertices_cm"][b][1]
            - panel["vertices_cm"][b][0] * panel["vertices_cm"][a][1]
            for a, b in zip(order, (*order[1:], order[0]))
        )
        self.assertGreater(area, 0.0)

    def test_fixed_metric_frame_preserves_dimensions_and_incidence(self):
        sample = {
            "sample_id": "sample",
            "category": "top",
            "source_dataset": "GarmentCodeData v2",
            "source_license": "CC BY 4.0",
            "source_specification_sha256": "0" * 64,
            "views": [],
        }
        target = panel_target(sample, _panel(), canvas_size=1024, pixels_per_cm=3.0)
        self.assertEqual(target["geometry"]["width_cm"], 10.0)
        self.assertEqual(target["geometry"]["height_cm"], 20.0)
        self.assertEqual(
            target["input_contract"]["metric_validation_image"]["pixels_per_cm"], 3.0
        )
        self.assertTrue(
            target["input_contract"]["normalized_panel_image"][
                "absolute_length_requires_scale_token"
            ]
        )
        self.assertEqual(target["geometry"]["boundary_sequence"], [0, 1, 2, 3])
        for index, edge in enumerate(target["geometry"]["edges"]):
            self.assertEqual(edge["start_vertex_index"], index)
            self.assertEqual(edge["end_vertex_index"], (index + 1) % 4)

    def test_role_is_weak_and_keeps_source_identifier(self):
        role = infer_panel_role("pant_b_r", "leg")
        self.assertEqual(role["part"], "pants_leg")
        self.assertEqual(role["surface"], "back")
        self.assertEqual(role["side"], "right")
        self.assertFalse(role["expert_verified"])
        self.assertEqual(role["source_panel_id"], "pant_b_r")


if __name__ == "__main__":
    unittest.main()
