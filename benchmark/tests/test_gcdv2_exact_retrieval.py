from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from benchmark.gcdv2_exact.retrieval_learning import (
    CURVE_TYPES,
    EDGE_FEATURE_INDEX,
    EDGE_FEATURE_NAMES,
    ExactRetrievalCorpus,
    ExactRetrievalExample,
    build_crossmodal_retrieval_model,
    deterministic_stratified_split,
    make_retrieval_batch,
    normalized_geometry_distance,
    paired_retrieval_metrics,
    tokenize_exact_pattern,
)


def _label(curve_type="quadratic_bezier"):
    curve = {
        "type": curve_type,
        "controls_cm": [[6.0, 8.0]] if curve_type == "quadratic_bezier" else [],
    }
    if curve_type == "circular_arc":
        curve["arc"] = {
            "radius_cm": 8.0,
            "large_arc": False,
            "sweep_y_up": True,
        }
    return {
        "category": "top",
        "panels": [
            {
                "panel_id": "front",
                "vertices_cm": [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]],
                "local_curve_bbox_cm": [0.0, 0.0, 10.0, 10.0],
                "initial_3d_placement": {
                    "translation_cm": [0.0, 80.0, 20.0],
                    "normal": [0.0, 0.0, 1.0],
                },
                "edges": [
                    {
                        "edge_id": "front.edge_0",
                        "edge_index": 0,
                        "endpoints": [0, 1],
                        "start_cm": [0.0, 0.0],
                        "end_cm": [10.0, 0.0],
                        "length_cm": 10.0,
                        "chord_direction_deg": 0.0,
                        "start_tangent_deg": 0.0,
                        "end_tangent_deg": 0.0,
                        "curve": {"type": "line", "controls_cm": []},
                    },
                    {
                        "edge_id": "front.edge_1",
                        "edge_index": 1,
                        "endpoints": [1, 2],
                        "start_cm": [10.0, 0.0],
                        "end_cm": [10.0, 10.0],
                        "length_cm": 12.0,
                        "chord_direction_deg": 90.0,
                        "start_tangent_deg": 80.0,
                        "end_tangent_deg": 100.0,
                        "curve": curve,
                    },
                ],
            }
        ],
    }


class ExactRetrievalGeometryTests(unittest.TestCase):
    def test_tokenizer_preserves_exact_curve_contract(self):
        result = tokenize_exact_pattern(_label())
        self.assertEqual(result.tokens.shape, (2, len(EDGE_FEATURE_NAMES)))
        second = result.tokens[1]
        self.assertEqual(second[EDGE_FEATURE_INDEX["curve_is_quadratic_bezier"]], 1.0)
        self.assertEqual(second[EDGE_FEATURE_INDEX["control_1_present"]], 1.0)
        self.assertAlmostEqual(second[EDGE_FEATURE_INDEX["control_1_u_in_panel"]], 0.6)
        self.assertAlmostEqual(second[EDGE_FEATURE_INDEX["control_1_v_in_panel"]], 0.8)
        self.assertAlmostEqual(second[EDGE_FEATURE_INDEX["curve_length_over_sample"]], 1.2)

    def test_topology_and_distance_change_with_curve_kind(self):
        first = tokenize_exact_pattern(_label("quadratic_bezier"))
        same = tokenize_exact_pattern(_label("quadratic_bezier"))
        other = tokenize_exact_pattern(_label("circular_arc"))
        self.assertEqual(first.topology_signature, same.topology_signature)
        self.assertNotEqual(first.topology_signature, other.topology_signature)
        self.assertEqual(
            normalized_geometry_distance(
                first.tokens, same.tokens, topology_compatible=True
            ),
            0.0,
        )
        self.assertGreater(
            normalized_geometry_distance(
                first.tokens, other.tokens, topology_compatible=False
            ),
            0.0,
        )

    def test_split_is_deterministic_and_stratified(self):
        rows = [
            {"sample_id": f"{category}_{index}", "category": category}
            for category in ("top", "skirt", "pants")
            for index in range(20)
        ]
        first = deterministic_stratified_split(rows, seed=11)
        second = deterministic_stratified_split(list(reversed(rows)), seed=11)
        self.assertEqual(first, second)
        for category in ("top", "skirt", "pants"):
            values = {first[f"{category}_{index}"] for index in range(20)}
            self.assertEqual(values, {"train", "validation", "test"})

    def test_paired_metrics_use_exact_ids(self):
        embeddings = np.eye(3, dtype=np.float32)
        metrics = paired_retrieval_metrics(
            embeddings, embeddings, ("a", "b", "c"), ("a", "b", "c")
        )
        self.assertEqual(metrics["recall_at_1"], 1.0)
        self.assertEqual(metrics["mrr"], 1.0)

    def test_cached_cam_order_is_remapped_to_semantic_front_back(self):
        features = np.arange(4 * 2, dtype=np.float32).reshape(1, 4, 1, 2)
        tokenized = tokenize_exact_pattern(_label())
        example = ExactRetrievalExample(
            sample_id="sample",
            category="top",
            split="train",
            label_path="labels.json",
            pattern_path="pattern.png",
            feature_index=0,
            pattern_tokens=tokenized.tokens,
            edge_ids=tokenized.edge_ids,
            topology_signature=tokenized.topology_signature,
        )
        corpus = ExactRetrievalCorpus(
            examples=(example,),
            view_features=features,
            max_edges=len(tokenized.tokens),
            feature_cache_path="features.npz",
            missing_feature_sample_ids=(),
        )
        batch = make_retrieval_batch(corpus, [0])
        np.testing.assert_array_equal(batch["view_features"][0, 0], features[0, 1])
        np.testing.assert_array_equal(batch["view_features"][0, 1], features[0, 0])


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is not installed")
class ExactRetrievalModelTests(unittest.TestCase):
    def test_dual_encoder_output_is_normalized(self):
        import torch

        config = {
            "spatial_feature_dim": 8,
            "max_spatial_tokens": 5,
            "max_edges": 4,
            "hidden_dim": 16,
            "embedding_dim": 8,
            "num_heads": 4,
            "pattern_layers": 1,
            "view_layers": 1,
            "pool_queries_per_view": 2,
            "dropout": 0.0,
        }
        model = build_crossmodal_retrieval_model(config).eval()
        with torch.inference_mode():
            output = model(
                torch.randn(2, 4, 5, 8),
                torch.randn(2, 4, len(EDGE_FEATURE_NAMES)),
                torch.ones(2, 4, dtype=torch.bool),
            )
        self.assertEqual(tuple(output["image_embedding"].shape), (2, 8))
        self.assertTrue(
            torch.allclose(output["image_embedding"].norm(dim=-1), torch.ones(2), atol=1e-5)
        )
        self.assertTrue(
            torch.allclose(output["pattern_embedding"].norm(dim=-1), torch.ones(2), atol=1e-5)
        )


if __name__ == "__main__":
    unittest.main()
