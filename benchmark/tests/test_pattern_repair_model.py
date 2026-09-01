import unittest

import numpy as np

from benchmark.pattern_pipeline.schema import Edge, Panel, PatternDocument
from benchmark.pattern_pipeline.validation import validate_pattern
from benchmark.pattern_repair.application import panel_unique_nodes, rebuild_panel, repair_document
from benchmark.pattern_repair.data import corrupt_loop, generate_clean_loop, loop_features, strict_self_intersections, synthetic_batch


def panel_from_loop(panel_id: str, points: np.ndarray) -> Panel:
    edges = []
    for index, start in enumerate(points):
        end = points[(index + 1) % len(points)]
        edges.append(Edge(f"{panel_id}.e{index}", (tuple(start), tuple(end))))
    return Panel(panel_id, tuple(edges))


class PatternRepairModelTests(unittest.TestCase):
    def test_official_clean_pool_probability_can_be_forced(self):
        rng = np.random.default_rng(11)
        clean = generate_clean_loop(rng, 24, 24)
        _, targets, mask = synthetic_batch(
            rng,
            2,
            maximum_nodes=32,
            clean_pool=(clean,),
            clean_pool_probability=1.0,
        )
        self.assertTrue(np.all(mask[:, :24]))
        np.testing.assert_allclose(targets[0, :24], clean)

    def test_synthetic_pair_has_variable_nodes_and_ten_features(self):
        rng = np.random.default_rng(7)
        clean = generate_clean_loop(rng, 24, 40)
        pair = corrupt_loop(clean, rng)
        self.assertGreaterEqual(len(clean), 24)
        self.assertLessEqual(len(clean), 40)
        self.assertEqual(loop_features(pair.corrupted).shape, (len(clean), 10))
        self.assertEqual(pair.clean.shape, pair.corrupted.shape)
        self.assertEqual(strict_self_intersections(clean), 0)

    def test_panel_round_trip_preserves_edge_topology(self):
        points = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]], dtype=np.float32)
        panel = panel_from_loop("panel", points)
        nodes, spans = panel_unique_nodes(panel)
        rebuilt = rebuild_panel(panel, nodes, spans)
        self.assertEqual([edge.id for edge in rebuilt.edges], [edge.id for edge in panel.edges])
        self.assertTrue(validate_pattern(PatternDocument("roundtrip", "test", (rebuilt,), ())).accepted)

    def test_learned_application_can_repair_bowtie_without_changing_topology(self):
        import torch

        bowtie = np.array([[0.0, 0.0], [2.0, 2.0], [0.0, 2.0], [2.0, 0.0]], dtype=np.float32)
        clean = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]], dtype=np.float32)
        untouched = panel_from_loop("clean_panel", clean + np.array([4.0, 0.0], dtype=np.float32))
        document = PatternDocument("bowtie", "generator", (panel_from_loop("panel", bowtie), untouched), ())
        target = (clean - clean.mean(axis=0)) / 2.0

        class FixedRepair(torch.nn.Module):
            repair_config = {"maximum_nodes": 16}

            def forward(self, features, valid_mask):
                output = features[..., :2].clone()
                output[0, :4] = torch.from_numpy(target).to(output)
                return output

        repaired, receipt = repair_document(FixedRepair(), document, "cpu", (1.0,))
        self.assertFalse(validate_pattern(document).accepted)
        self.assertTrue(validate_pattern(repaired).accepted)
        self.assertTrue(receipt["improved"])
        self.assertEqual(receipt["panel_count_before"], receipt["panel_count_after"])
        self.assertEqual(receipt["edge_count_before"], receipt["edge_count_after"])
        self.assertFalse(receipt["template_retrieval"])
        self.assertEqual(repaired.panels[1], untouched)


if __name__ == "__main__":
    unittest.main()
