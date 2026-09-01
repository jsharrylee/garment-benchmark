from __future__ import annotations

import unittest

import numpy as np

from benchmark.drafting_semantics.multigarment_learning import (
    EDGE_FEATURE_DIM,
    GARMENT_ROLES,
    MULTIGARMENT_EDGE_ROLES,
    MULTIGARMENT_PANEL_ROLES,
    MultiGarmentExample,
    MultiPanelExample,
    build_multigarment_model,
    padded_garment_batch,
    randomize_boundary_serialization,
)


def _panel(role: str, edge_roles: tuple[str, ...], lengths: tuple[float, ...]) -> MultiPanelExample:
    features = np.zeros((len(edge_roles), EDGE_FEATURE_DIM), dtype=np.float32)
    features[:, 17] = np.asarray(lengths) / 100.0
    return MultiPanelExample(
        panel_id=role,
        panel_target=MULTIGARMENT_PANEL_ROLES.index(role),
        features=features,
        edge_targets=np.asarray([MULTIGARMENT_EDGE_ROLES.index(value) for value in edge_roles]),
        edge_lengths_cm=np.asarray(lengths, dtype=np.float32),
        edge_ids=tuple(f"{role}.{index}" for index in range(len(edge_roles))),
        panel_scale_cm=100.0,
    )


class MultiGarmentLearningTests(unittest.TestCase):
    def test_cross_panel_sleeve_ratio_is_preserved(self) -> None:
        example = MultiGarmentExample(
            "tee",
            "train",
            "unit",
            GARMENT_ROLES.index("top"),
            (
                _panel("front_bodice", ("armhole", "armhole"), (20.0, 22.0)),
                _panel("back_bodice", ("armhole",), (38.0,)),
                _panel("sleeve", ("sleeve_head", "sleeve_head"), (41.0, 43.0)),
            ),
        )
        batch = padded_garment_batch((example,), maximum_panels=4, maximum_edges=4)
        self.assertTrue(batch["seam_ratio_mask"][0])
        self.assertAlmostEqual(float(batch["seam_ratio_targets"][0]), 84.0 / 80.0, places=6)

    def test_random_reindex_keeps_targets_and_lengths_paired(self) -> None:
        panel = _panel("front_bodice", ("neckline", "shoulder", "armhole"), (1.0, 2.0, 3.0))
        example = MultiGarmentExample("x", "train", "unit", 0, (panel,))
        changed = randomize_boundary_serialization(example, np.random.default_rng(7))
        paired = sorted(zip(changed.panels[0].edge_targets.tolist(), changed.panels[0].edge_lengths_cm.tolist()))
        expected = sorted(zip(panel.edge_targets.tolist(), panel.edge_lengths_cm.tolist()))
        self.assertEqual(paired, expected)

    def test_model_returns_headwise_attention(self) -> None:
        import torch

        config = {
            "width": 32,
            "heads": 4,
            "local_layers": 2,
            "global_layers": 1,
            "feedforward_multiplier": 2,
            "dropout": 0.0,
        }
        model = build_multigarment_model(config)
        features = torch.zeros((2, 3, 5, EDGE_FEATURE_DIM))
        edge_valid = torch.ones((2, 3, 5), dtype=torch.bool)
        panel_valid = torch.ones((2, 3), dtype=torch.bool)
        output = model(features, edge_valid, panel_valid, capture_attention=True)
        self.assertEqual(output["edge_logits"].shape, (2, 3, 5, len(MULTIGARMENT_EDGE_ROLES)))
        self.assertEqual(output["local_attention"][0].shape, (6, 4, 6, 6))
        self.assertEqual(output["global_attention"][0].shape, (2, 4, 4, 4))


if __name__ == "__main__":
    unittest.main()
