from __future__ import annotations

import unittest

import torch

from benchmark.gcdv2_exact.garment_panel_set_learning import (
    CATEGORIES,
    CURVE_TYPES,
    IMAGE_SIZE,
    MAXIMUM_EDGES,
    build_model,
    collate_garments,
    garment_disjoint_split,
    model_loss,
)
from benchmark.gcdv2_exact.garment_panel_set_learning import GarmentExample


class GarmentPanelSetLearningTests(unittest.TestCase):
    def test_split_is_stratified_and_garment_disjoint(self):
        garments = []
        for category in CATEGORIES:
            for index in range(20):
                garments.append(GarmentExample(f"{category}_{index}", category, ()))
        assignments, audit = garment_disjoint_split(garments)
        self.assertTrue(audit["garment_disjoint"])
        self.assertEqual(set(assignments), {garment.sample_id for garment in garments})
        for category in CATEGORIES:
            observed = {assignments[f"{category}_{index}"] for index in range(20)}
            self.assertEqual(observed, {"train", "validation", "test"})

    def test_model_consumes_panel_set_and_backpropagates_graph_heads(self):
        count = 5
        item = {
            "sample_id": "sample",
            "category": 2,
            "images": torch.rand(3, 1, IMAGE_SIZE, IMAGE_SIZE).numpy(),
            "scales": torch.zeros(3, 1).numpy(),
            "panel_uids": ["a", "b", "c"],
            "targets": [],
        }
        for panel in range(3):
            target = {
                "source_id": panel,
                "part": panel,
                "surface": panel % 3,
                "side": panel % 3,
                "count": count,
                "vertices": torch.rand(MAXIMUM_EDGES, 2).numpy(),
                "edge_types": torch.cat((torch.arange(count) % len(CURVE_TYPES), torch.full((MAXIMUM_EDGES-count,), -1))).numpy(),
                "lengths": torch.rand(MAXIMUM_EDGES).numpy(),
                "directions": torch.nn.functional.normalize(torch.rand(MAXIMUM_EDGES, 2), dim=-1).numpy(),
                "tangents": torch.nn.functional.normalize(torch.rand(MAXIMUM_EDGES, 2, 2), dim=-1).reshape(MAXIMUM_EDGES, 4).numpy(),
                "controls": torch.rand(MAXIMUM_EDGES, 4).numpy(),
                "control_masks": torch.zeros(MAXIMUM_EDGES, 4).numpy(),
                "arc_radius": torch.rand(MAXIMUM_EDGES).numpy(),
                "arc_flags": torch.zeros(MAXIMUM_EDGES, 2).numpy(),
                "arc_mask": torch.zeros(MAXIMUM_EDGES).numpy(),
                "cm_per_pixel": 0.1,
            }
            item["targets"].append(target)
        batch = collate_garments([item, item])
        model = build_model(5, {"width": 32, "heads": 4, "set_layers": 1, "graph_layers": 1, "dropout": 0.0})
        output = model(batch["images"], batch["scales"], batch["panel_mask"])
        self.assertEqual(tuple(output["vertices"].shape), (2, 3, MAXIMUM_EDGES, 2))
        self.assertEqual(tuple(output["category_logits"].shape), (2, len(CATEGORIES)))
        loss = model_loss(output, batch)["loss"]
        self.assertTrue(torch.isfinite(loss))
        loss.backward()


if __name__ == "__main__":
    unittest.main()
