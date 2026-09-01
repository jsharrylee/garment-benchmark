from __future__ import annotations

import unittest

import numpy as np

from benchmark.gcdv2_exact.visual_dsl_retrieval import (
    bidirectional_infonce,
    build_visual_dsl_retrieval_model,
    paired_retrieval_metrics,
    topology_signature_from_program,
    train_bank_retrieval_metrics,
)


class VisualDSLTopologyTests(unittest.TestCase):
    def test_signature_ignores_panel_order_and_cycle_origin(self) -> None:
        commands = np.full((3, 6), 4, dtype=np.int64)
        valid = np.zeros((3, 6), dtype=bool)
        panels = np.asarray([True, True, False])
        commands[0, :4] = [0, 1, 0, 2]
        commands[1, :3] = [0, 0, 3]
        valid[0, :4] = True
        valid[1, :3] = True
        first = topology_signature_from_program(0, commands, valid, panels)

        reordered = commands.copy()
        reordered[[0, 1]] = reordered[[1, 0]]
        reordered_valid = valid.copy()
        reordered_valid[[0, 1]] = reordered_valid[[1, 0]]
        # Rotate the second panel after reordering.
        reordered[1, :4] = [1, 0, 2, 0]
        second = topology_signature_from_program(0, reordered, reordered_valid, panels)
        self.assertEqual(first, second)

    def test_topology_changes_with_command(self) -> None:
        commands = np.asarray([[0, 1, 0, 2]], dtype=np.int64)
        valid = np.ones_like(commands, dtype=bool)
        panels = np.asarray([True])
        first = topology_signature_from_program(0, commands, valid, panels)
        commands[0, 1] = 3
        second = topology_signature_from_program(0, commands, valid, panels)
        self.assertNotEqual(first, second)


class VisualDSLModelTests(unittest.TestCase):
    def test_forward_and_loss_shapes(self) -> None:
        import torch

        config = {
            "spatial_dim": 16,
            "dsl_dim": 12,
            "hidden_dim": 32,
            "embedding_dim": 20,
            "heads": 4,
            "visual_layers": 1,
            "pattern_layers": 1,
            "max_spatial_tokens": 7,
            "max_panels": 5,
            "pool_queries_per_view": 2,
            "dropout": 0.0,
        }
        model = build_visual_dsl_retrieval_model(config)
        output = model(
            torch.randn(3, 4, 7, 16),
            torch.randn(3, 5, 12),
            torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [1, 0, 0, 0, 0]], dtype=torch.bool),
        )
        self.assertEqual(tuple(output["visual_embedding"].shape), (3, 20))
        self.assertEqual(tuple(output["pattern_embedding"].shape), (3, 20))
        self.assertTrue(torch.isfinite(bidirectional_infonce(model, output["visual_embedding"], output["pattern_embedding"])))

    def test_metrics_distinguish_present_and_absent_gallery(self) -> None:
        values = np.eye(3, dtype=np.float32)
        paired = paired_retrieval_metrics(values, values)
        self.assertEqual(paired["recall_at_1"], 1.0)
        metrics, order = train_bank_retrieval_metrics(
            values[:2],
            np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32),
            np.asarray([0, 1]),
            np.asarray([0, 1, 2]),
            np.asarray(["a", "b"]),
            np.asarray(["a", "x", "z"]),
        )
        self.assertEqual(order.shape, (2, 3))
        self.assertTrue(metrics["exact_target_present"] is False)
        self.assertEqual(metrics["category_match_at_1"], 1.0)
        self.assertEqual(metrics["exact_topology_compatibility_at_1"], 0.5)


if __name__ == "__main__":
    unittest.main()
