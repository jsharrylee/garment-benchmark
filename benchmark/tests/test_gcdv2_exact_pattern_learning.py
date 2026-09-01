from __future__ import annotations

import copy
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from benchmark.gcdv2_exact.pattern_learning import (
    MAXIMUM_EDGES,
    MAXIMUM_PANELS,
    PatternExample,
    build_pattern_parser_model,
    family_disjoint_split,
    padded_pattern_batch,
    pattern_parser_loss,
    spatial_token_metadata,
    target_from_label,
)


def _edge(edge_id, index, endpoints, start, end, kind, length):
    return {
        "edge_id": edge_id,
        "edge_index": index,
        "endpoints": endpoints,
        "packed_start_uv": start,
        "packed_end_uv": end,
        "length_cm": length,
        "curve": {"type": kind},
    }


def _label():
    return {
        "schema_version": "gcdv2-exact-pair-1.0",
        "sample_id": "sample",
        "category": "top",
        "pattern_image": "pattern.png",
        "packing": {
            "late": {"canvas_size_px": [1024, 1024], "scale_px_per_cm": 8.0},
            "early": {"canvas_size_px": [1024, 1024], "scale_px_per_cm": 8.0},
        },
        "panels": [
            {
                "panel_id": "late",
                "source_order_index": 0,
                "vertices_cm": [[0, 0], [10, 0], [0, 10]],
                "packed_bbox_px": [600, 600, 800, 800],
                "edges": [
                    _edge("late.edge_1", 1, [1, 2], [0.78, 0.60], [0.60, 0.78], "quadratic_bezier", 20),
                    _edge("late.edge_0", 0, [0, 1], [0.60, 0.60], [0.78, 0.60], "line", 10),
                ],
            },
            {
                "panel_id": "early",
                "source_order_index": 1,
                "vertices_cm": [[0, 0], [10, 0], [10, 10]],
                "packed_bbox_px": [100, 100, 400, 400],
                "edges": [
                    _edge("early.edge_1", 1, [1, 2], [0.35, 0.10], [0.35, 0.35], "circular_arc", 202.5),
                    # Deliberately reversed in the source.  Packed canonical
                    # orientation must still be upper/left to lower/right.
                    _edge("early.edge_0", 0, [0, 1], [0.35, 0.10], [0.10, 0.10], "line", 25),
                ],
            },
        ],
    }


class ExactPatternLearningTests(unittest.TestCase):
    def test_targets_are_order_invariant_and_keep_long_arc(self):
        label = _label()
        first = target_from_label(label, label_path=Path("labels.json"))
        permuted = copy.deepcopy(label)
        permuted["panels"].reverse()
        for panel in permuted["panels"]:
            panel["edges"].reverse()
        second = target_from_label(permuted, label_path=Path("labels.json"))
        np.testing.assert_allclose(first.panel_boxes, second.panel_boxes)
        np.testing.assert_allclose(first.edge_geometry, second.edge_geometry)
        np.testing.assert_array_equal(first.edge_types, second.edge_types)
        self.assertGreater(float(first.edge_geometry[:, 4].max()), 1.0)
        # The reversed horizontal edge is canonicalized left-to-right.
        index = next(
            index
            for index, ref in enumerate(first.edge_refs)
            if ref["source_edge_id"] == "early.edge_0"
        )
        self.assertLess(float(first.edge_geometry[index, 0]), float(first.edge_geometry[index, 2]))

    def test_model_set_outputs_and_loss_are_finite(self):
        import torch

        example = target_from_label(_label(), label_path=Path("labels.json"))
        feature = np.random.default_rng(7).normal(
            size=(len(spatial_token_metadata()), 256)
        ).astype(np.float32)
        example = replace(example, spatial_features=feature)
        raw = padded_pattern_batch([example, example])
        batch = {
            key: torch.from_numpy(value)
            for key, value in raw.items()
            if isinstance(value, np.ndarray)
        }
        model = build_pattern_parser_model(
            {"width": 32, "heads": 4, "encoder_layers": 1, "decoder_layers": 1, "feedforward_multiplier": 2, "dropout": 0.0}
        )
        # Prove the length head is not artificially bounded by one.
        with torch.no_grad():
            model.edge_geometry_head[-1].bias[4] = 2.0
        output = model(batch["spatial_features"])
        self.assertEqual(tuple(output["edge_geometry"].shape), (2, MAXIMUM_EDGES, 7))
        self.assertEqual(tuple(output["panel_boxes"].shape), (2, MAXIMUM_PANELS, 4))
        self.assertGreater(float(output["edge_geometry"][..., 4].max()), 1.0)
        result = pattern_parser_loss(output, batch)
        self.assertTrue(torch.isfinite(result["loss"]))
        result["loss"].backward()

    def test_split_keeps_topology_families_disjoint(self):
        base = target_from_label(_label(), label_path=Path("labels.json"))
        examples = []
        for category in ("top", "skirt", "pants"):
            for family in range(4):
                for member in range(2):
                    examples.append(
                        replace(
                            base,
                            sample_id=f"{category}_{family}_{member}",
                            category=category,
                            family_id=f"{category}_family_{family}",
                        )
                    )
        assignments, audit = family_disjoint_split(examples)
        self.assertTrue(audit["family_disjoint"])
        self.assertEqual(set(assignments.values()), {"train", "validation", "test"})
        by_family = {}
        for example in examples:
            by_family.setdefault(example.family_id, set()).add(assignments[example.sample_id])
        self.assertTrue(all(len(values) == 1 for values in by_family.values()))


if __name__ == "__main__":
    unittest.main()
