from __future__ import annotations

import unittest

from benchmark.gcdv2_exact.neurosymbolic_dataset import formal_graph, junction_records


def _target():
    vertices = [
        {"source_vertex_index": index, "centered_xy_cm": value, "image_xy_px": [512 + value[0] * 10, 512 - value[1] * 10]}
        for index, value in enumerate(([-10, -10], [10, -10], [10, 10], [-10, 10]))
    ]
    edges = []
    for index, angle in enumerate((0, 90, 180, -90)):
        edges.append({
            "edge_index": index,
            "source_edge_index": index,
            "source_edge_id": f"source_{index}",
            "start_vertex_index": index,
            "end_vertex_index": (index + 1) % 4,
            "curve_type": "line",
            "curve_parameters": {},
            "centered_controls_cm": [],
            "length_cm": 20.0,
            "chord_direction_deg_y_up": angle,
            "start_tangent_deg_y_up": angle,
            "end_tangent_deg_y_up": angle,
        })
    return {
        "panel_uid": "sample:panel",
        "garment_category": "top",
        "source": {"panel_id": "right_ftorso"},
        "role_labels": {"part": "bodice"},
        "geometry": {"vertices": vertices, "edges": edges},
    }


class NeurosymbolicDatasetTests(unittest.TestCase):
    def test_corner_observability_uses_tangent_discontinuity(self):
        junctions = junction_records(_target())
        self.assertEqual(len(junctions), 4)
        self.assertTrue(all(value["observability"] == "VISIBLE_CORNER" for value in junctions))

    def test_formal_graph_is_closed_and_cyclic_start_is_not_semantic(self):
        graph = formal_graph(_target())
        self.assertEqual(len(graph["points"]), 4)
        self.assertEqual(len(graph["curves"]), 4)
        self.assertEqual(sum(value["predicate"] == "NEXT" for value in graph["relations"]), 4)
        self.assertTrue(graph["serialization_equivalence"]["cyclic_rotations_are_same_shape"])
        self.assertTrue(graph["serialization_equivalence"]["canonical_start_is_not_a_semantic_landmark"])


if __name__ == "__main__":
    unittest.main()
