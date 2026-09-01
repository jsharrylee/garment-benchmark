from __future__ import annotations

import unittest

import numpy as np

from benchmark.scripts.render_gcdv2_crossmodal_retrieval_review import (
    recompute_retrieval_contract,
    select_stratified_predictions,
)


class RetrievalReviewTests(unittest.TestCase):
    def test_selection_is_deterministic_and_has_requested_mix(self):
        predictions = [
            {
                "sample_id": f"{category}_{index}",
                "target_category": category,
            }
            for category in ("top", "skirt", "pants")
            for index in range(10)
        ]
        first = select_stratified_predictions(predictions, seed=9)
        second = select_stratified_predictions(list(reversed(predictions)), seed=9)
        self.assertEqual(first, second)
        self.assertEqual(sum(row["target_category"] == "top" for row in first), 4)
        self.assertEqual(sum(row["target_category"] == "skirt" for row in first), 3)
        self.assertEqual(sum(row["target_category"] == "pants" for row in first), 3)

    def test_recomputation_uses_unfiltered_global_train_argmax(self):
        sample_ids = ("query", "other_test", "wrong_category", "same_category")
        lookup = {sample_id: index for index, sample_id in enumerate(sample_ids)}
        image = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        pattern = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.99, 0.01], [0.8, 0.2]], dtype=np.float32)
        split = {
            "query": "test",
            "other_test": "test",
            "wrong_category": "train",
            "same_category": "train",
        }
        prediction = {
            "sample_id": "query",
            "paired_gallery_target_rank": 1,
            "retrieved_sample_id": "wrong_category",
            "similarity": 0.99,
        }
        result = recompute_retrieval_contract(
            prediction,
            embedding_lookup=lookup,
            image_embeddings=image,
            pattern_embeddings=pattern,
            split_lookup=split,
        )
        self.assertEqual(result["raw_global_top1_recomputed"], "wrong_category")
        self.assertTrue(result["raw_global_top1_matches_saved"])
        self.assertFalse(result["category_filter"])
        self.assertFalse(result["topology_filter"])


if __name__ == "__main__":
    unittest.main()

