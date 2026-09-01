from __future__ import annotations

import unittest

import numpy as np

from benchmark.gcdv2_exact.pattern_dsl_learning import CATEGORIES
from benchmark.gcdv2_exact.visual_dsl_retrieval import SPLIT_TO_INDEX
from benchmark.scripts.evaluate_gcdv2_raw_fpn_vs_dsl_retrieval import (
    evaluate_embedding_ranking,
    evaluate_saved_dsl_predictions,
    raw_fpn_mean_embeddings,
)


class RawFPNRetrievalAblationTests(unittest.TestCase):
    def test_raw_pooling_returns_unit_garment_vectors(self) -> None:
        values = np.arange(3 * 4 * 5 * 6, dtype=np.float32).reshape(3, 4, 5, 6) + 1
        embedded = raw_fpn_mean_embeddings(values, batch_size=2)
        self.assertEqual(embedded.shape, (3, 6))
        np.testing.assert_allclose(np.linalg.norm(embedded, axis=1), 1.0, atol=1e-6)

    def test_embedding_ranking_uses_train_only_bank(self) -> None:
        embeddings = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.99, 0.01, 0.0],
                [0.01, 0.99, 0.0],
            ],
            dtype=np.float32,
        )
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        splits = np.asarray(
            [
                SPLIT_TO_INDEX["train"],
                SPLIT_TO_INDEX["train"],
                SPLIT_TO_INDEX["test"],
                SPLIT_TO_INDEX["test"],
            ]
        )
        categories = np.asarray([0, 1, 0, 1])
        topologies = np.asarray(["a", "b", "a", "b"])
        metrics, order, train, test = evaluate_embedding_ranking(
            embeddings,
            splits=splits,
            categories=categories,
            topologies=topologies,
            top_k=2,
        )
        self.assertEqual(train.tolist(), [0, 1])
        self.assertEqual(test.tolist(), [2, 3])
        self.assertEqual(order[:, 0].tolist(), [0, 1])
        self.assertEqual(metrics["category_match_at_1"], 1.0)
        self.assertEqual(metrics["exact_topology_compatibility_at_1"], 1.0)

    def test_saved_dsl_predictions_must_be_test_to_train_only(self) -> None:
        sample_ids = np.asarray(["train_top", "train_pants", "test_top", "test_pants"])
        splits = np.asarray(
            [
                SPLIT_TO_INDEX["train"],
                SPLIT_TO_INDEX["train"],
                SPLIT_TO_INDEX["test"],
                SPLIT_TO_INDEX["test"],
            ]
        )
        categories = np.asarray(
            [CATEGORIES.index("top"), CATEGORIES.index("pants"), CATEGORIES.index("top"), CATEGORIES.index("pants")]
        )
        topologies = np.asarray(["a", "b", "a", "b"])
        rows = [
            {
                "sample_id": "test_top",
                "top_train_bank_sample_ids": ["train_top", "train_pants"],
            },
            {
                "sample_id": "test_pants",
                "top_train_bank_sample_ids": ["train_pants", "train_top"],
            },
        ]
        metrics = evaluate_saved_dsl_predictions(
            rows,
            sample_ids=sample_ids,
            splits=splits,
            categories=categories,
            topologies=topologies,
            top_k=2,
        )
        self.assertEqual(metrics["category_match_at_1"], 1.0)
        self.assertEqual(metrics["exact_topology_compatibility_at_1"], 1.0)

        rows[0]["top_train_bank_sample_ids"] = ["test_top", "train_pants"]
        with self.assertRaises(ValueError):
            evaluate_saved_dsl_predictions(
                rows,
                sample_ids=sample_ids,
                splits=splits,
                categories=categories,
                topologies=topologies,
                top_k=2,
            )


if __name__ == "__main__":
    unittest.main()
