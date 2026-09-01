from __future__ import annotations

import unittest

import numpy as np

from benchmark.gcdv2_exact.fourview_dsl_bridge import (
    NEURAL_GEOMETRY_ARRAY_KEYS,
    RetrievalCatalogEntry,
    adapt_aligned_dsl_prediction,
    aligned_dsl_retrieval_catalog,
    metadata_lookup,
    neural_geometry_input,
    retrieval_catalog_from_arrays,
    select_train_bank_anchor,
    summarize_bridge_records,
)


class FourViewPatternDSLBridgeTests(unittest.TestCase):
    def test_aligned_prediction_adapter_preserves_unfiltered_ranking(self):
        adapted = adapt_aligned_dsl_prediction(
            {
                "sample_id": "query",
                "retrieved_sample_id": "anchor-a",
                "target_category": 0,
                "retrieved_category": 1,
                "category_match": False,
                "exact_closed_cycle_primitive_topology_match": False,
                "top_train_bank_sample_ids": ["anchor-a", "anchor-b"],
            }
        )
        self.assertEqual(adapted["retrieved_sample_id"], "anchor-a")
        self.assertEqual(
            [row["sample_id"] for row in adapted["top_train_bank"]],
            ["anchor-a", "anchor-b"],
        )
        self.assertIsNone(adapted["similarity"])
        self.assertFalse(adapted["category_match"])

    def test_aligned_catalog_uses_authoritative_dsl_split_and_topology(self):
        arrays = {
            "categories": np.asarray([0, 0], np.int8),
            "splits": np.asarray([0, 2], np.int8),
            "edge_commands": np.asarray([[[0, 1]], [[0, 1]]], np.int8),
            "edge_valid": np.asarray([[[True, True]], [[True, True]]]),
            "panel_valid": np.asarray([[True], [True]]),
        }
        rows = [
            {"sample_id": "train", "split": "train"},
            {"sample_id": "test", "split": "test"},
        ]
        catalog = aligned_dsl_retrieval_catalog(rows, arrays)
        self.assertEqual(catalog["train"].split, "train")
        self.assertEqual(catalog["test"].split, "test")
        self.assertEqual(
            catalog["train"].topology_signature,
            catalog["test"].topology_signature,
        )

    def test_anchor_selection_rejects_target_and_nontrain_without_filters(self):
        catalog = {
            "query": RetrievalCatalogEntry("query", "top", "test", "target-topology"),
            "validation": RetrievalCatalogEntry("validation", "top", "validation", "x"),
            "train": RetrievalCatalogEntry("train", "pants", "train", "y"),
        }
        lookup = metadata_lookup([{"sample_id": "train", "split": "test"}])
        prediction = {
            "sample_id": "query",
            "retrieved_sample_id": "query",
            "similarity": 0.9,
            "top_train_bank": [
                {"sample_id": "validation", "similarity": 0.8},
                {"sample_id": "train", "similarity": 0.7},
            ],
        }
        selected = select_train_bank_anchor(
            prediction, retrieval_catalog=catalog, dsl_lookup=lookup
        )
        self.assertEqual(selected.anchor_sample_id, "train")
        self.assertEqual(selected.anchor_rank, 3)
        self.assertFalse(selected.used_saved_top1)
        self.assertEqual(
            [row["reason"] for row in selected.rejected_candidates],
            ["query_target_id_forbidden", "not_retrieval_train:validation"],
        )

    def test_neural_input_whitelist_excludes_source_labels_and_stitches(self):
        arrays = {
            key: np.zeros((1, 1), np.float32) for key in NEURAL_GEOMETRY_ARRAY_KEYS
        }
        arrays.update(
            {
                "stitch_pairs": np.ones((1, 1, 4)),
                "edge_roles": np.ones((1, 1)),
                "landmarks": np.ones((1, 1)),
            }
        )
        selected = neural_geometry_input(arrays, 0)
        self.assertEqual(tuple(selected), NEURAL_GEOMETRY_ARRAY_KEYS)
        self.assertNotIn("stitch_pairs", selected)
        self.assertNotIn("edge_roles", selected)
        self.assertNotIn("landmarks", selected)

    def test_catalog_requires_parallel_arrays(self):
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            retrieval_catalog_from_arrays(["a"], ["top"], ["train", "test"], ["x"])

    def test_summary_keeps_retrieval_and_symbolic_claims_separate(self):
        record = {
            "query": {"category": "top"},
            "retrieval": {
                "category_match": True,
                "topology_compatible": False,
                "saved_category_contract_agrees": True,
                "saved_topology_contract_agrees": True,
            },
            "retrieved_anchor": {
                "used_saved_top1": True,
                "similarity": 0.75,
                "dsl_corpus_split": "validation",
                "rejected_candidates": [],
            },
            "neural_proposer": {"anchor_category_agreement": True},
            "symbolic_projection": {
                "valid": True,
                "raw_grammar_violations": 2,
                "projected_grammar_violations": 0,
                "changed_role_edges": 2,
                "predicted_seam_pair_count": 3,
                "derived_landmark_count": 2,
                "derived_landmark_counts": {"FNP": 1, "SP": 1},
            },
        }
        result = summarize_bridge_records([record])
        self.assertEqual(result["retrieval"]["category_match_rate"], 1.0)
        self.assertEqual(result["retrieval"]["exact_topology_compatibility_rate"], 0.0)
        self.assertEqual(result["retrieval"]["by_query_category"]["top"]["count"], 1)
        self.assertEqual(result["symbolic_projection"]["valid_rate"], 1.0)
        self.assertEqual(result["symbolic_projection"]["predicted_seam_facts"], 3)
        self.assertFalse(result["anchor_dsl"]["source_stitches_consumed"])


if __name__ == "__main__":
    unittest.main()
