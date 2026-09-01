from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from benchmark.gcdv2_exact.residual_learning import (
    ExactGeometryRecord,
    RetrievedResidualPair,
    batch_residual_pairs,
    build_retrieved_residual_model,
    build_visual_retrieval_pairs,
    deterministic_topology_split,
    materialize_prediction,
    reorder_cached_fpn_views,
    topology_hash,
)


def _label(offset: float = 0.0) -> dict:
    vertices = [[offset + 0.0, 0.0], [offset + 10.0, 0.0], [offset + 10.0, 12.0]]
    edges = []
    for index, endpoints in enumerate(((0, 1), (1, 2), (2, 0))):
        start, end = (vertices[value] for value in endpoints)
        edges.append(
            {
                "edge_index": index,
                "endpoints": list(endpoints),
                "start_cm": start,
                "end_cm": end,
                "curve": {
                    "type": "line",
                    "source_type": "line",
                    "source_params": None,
                    "controls_cm": [],
                },
                "length_cm": float(np.linalg.norm(np.asarray(end) - np.asarray(start))),
                "chord_direction_deg": 0.0,
            }
        )
    return {
        "schema_version": "gcdv2-exact-pair-1.0",
        "sample_id": "fixture",
        "category": "top",
        "panels": [
            {
                "panel_id": "front",
                "source_order_index": 0,
                "source_label": "body",
                "vertices_cm": vertices,
                "edges": edges,
            }
        ],
        "stitches": [],
    }


def _record(identifier: str, label_path: Path, value: float) -> ExactGeometryRecord:
    label = _label(value)
    vertices = np.asarray(label["panels"][0]["vertices_cm"], dtype=np.float32)
    return ExactGeometryRecord(
        sample_id=identifier,
        category="top",
        label_path=label_path,
        pattern_path=Path("unused.png"),
        topology_hash=topology_hash(label),
        topology={},
        panel_ids=("front",),
        vertices_cm=vertices,
        vertex_panel_indices=np.zeros(3, dtype=np.int64),
        vertex_local_indices=np.arange(3, dtype=np.int64),
        edges=np.asarray([[0, 1], [1, 2], [2, 0]], dtype=np.int64),
        edge_panel_indices=np.zeros(3, dtype=np.int64),
        edge_local_indices=np.arange(3, dtype=np.int64),
        curve_types=np.zeros(3, dtype=np.int64),
        curve_parameters_cm=np.zeros((3, 5), dtype=np.float32),
        curve_parameter_mask=np.zeros((3, 5), dtype=bool),
        spatial_features=np.broadcast_to(
            np.asarray([1.0, value + 1.0, 0.25, 0.5, 0.75, 0.1, 0.2, 0.3], dtype=np.float32),
            (4, 3, 8),
        ).copy(),
    )


class RetrievedResidualTests(unittest.TestCase):
    def test_topology_signature_ignores_geometry_but_not_curve_type(self) -> None:
        first = _label(0.0)
        moved = _label(19.0)
        self.assertEqual(topology_hash(first), topology_hash(moved))
        moved["panels"][0]["edges"][0]["curve"]["type"] = "quadratic_bezier"
        self.assertNotEqual(topology_hash(first), topology_hash(moved))

    def test_cached_views_are_reordered_to_semantic_front_back(self) -> None:
        values = np.arange(4 * 2 * 3).reshape(4, 2, 3)
        reordered = reorder_cached_fpn_views(values)
        np.testing.assert_array_equal(reordered[0], values[1])
        np.testing.assert_array_equal(reordered[1], values[0])
        np.testing.assert_array_equal(reordered[2], values[2])
        np.testing.assert_array_equal(reordered[3], values[3])

    def test_pair_builder_never_uses_self_or_heldout_anchor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.json"
            path.write_text(json.dumps(_label()), encoding="utf-8")
            records = tuple(_record(f"sample-{index}", path, float(index)) for index in range(6))
            split = deterministic_topology_split(records, seed=11)
            pairs, audit = build_visual_retrieval_pairs(records, split)
            self.assertTrue(pairs)
            self.assertGreater(audit["coverage"], 0.0)
            for pair in pairs:
                self.assertNotEqual(pair.target.sample_id, pair.anchor.sample_id)
                self.assertEqual(split[pair.anchor.sample_id], "train")
                self.assertEqual(pair.target.topology_hash, pair.anchor.topology_hash)

    def test_materialized_edges_share_one_corrected_vertex(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.json"
            label = _label()
            path.write_text(json.dumps(label), encoding="utf-8")
            target = _record("target", path, 0.0)
            anchor = _record("anchor", path, 1.0)
            pair = RetrievedResidualPair(target, anchor, "test", 0.9)
            vertices = target.vertices_cm.copy()
            vertices[1] = [12.5, 3.0]
            result = materialize_prediction(pair, vertices, np.zeros((3, 5), np.float32))
            edge_0, edge_1 = result["panels"][0]["edges"][:2]
            self.assertEqual(edge_0["end_cm"], edge_1["start_cm"])
            self.assertEqual(edge_0["end_cm"], [12.5, 3.0])

    def test_model_forward_preserves_padded_shapes(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.json"
            path.write_text(json.dumps(_label()), encoding="utf-8")
            target = _record("target", path, 0.0)
            anchor = _record("anchor", path, 1.0)
            pair = RetrievedResidualPair(target, anchor, "test", 0.9)
            raw = batch_residual_pairs([pair], maximum_vertices=5, maximum_edges=5)
            model = build_retrieved_residual_model(
                {
                    "width": 16,
                    "heads": 4,
                    "decoder_layers": 1,
                    "dropout": 0.0,
                    "visual_feature_dimension": 8,
                    "maximum_visual_tokens_per_view": 3,
                    "maximum_panels": 2,
                    "maximum_local_vertices": 5,
                    "maximum_local_edges": 5,
                }
            ).eval()
            kwargs = {
                key: torch.from_numpy(value)
                for key, value in raw.items()
                if isinstance(value, np.ndarray)
            }
            output = model(
                visual_features=kwargs["visual_features"].float(),
                anchor_vertices=kwargs["anchor_vertices"].float(),
                vertex_mask=kwargs["vertex_mask"].bool(),
                vertex_panel_indices=kwargs["vertex_panel_indices"].long(),
                vertex_local_indices=kwargs["vertex_local_indices"].long(),
                anchor_curve_parameters=kwargs["anchor_curve_parameters"].float(),
                edge_mask=kwargs["edge_mask"].bool(),
                edge_vertices=kwargs["edge_vertices"].long(),
                edge_panel_indices=kwargs["edge_panel_indices"].long(),
                edge_local_indices=kwargs["edge_local_indices"].long(),
                curve_types=kwargs["curve_types"].long(),
                category=kwargs["category"].long(),
            )
            self.assertEqual(tuple(output["predicted_vertices"].shape), (1, 5, 2))
            self.assertEqual(tuple(output["predicted_curve_parameters"].shape), (1, 5, 5))
            # Zero-initialized heads make the initial model exactly the anchor.
            torch.testing.assert_close(
                output["predicted_vertices"], kwargs["anchor_vertices"].float()
            )


if __name__ == "__main__":
    unittest.main()
