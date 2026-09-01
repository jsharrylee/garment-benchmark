from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from benchmark.drafting_semantics.schema import EDGE_ROLES
from benchmark.gcdv2_exact.pattern_dsl import compile_formal_graph
from benchmark.scripts.render_gcdv2_pattern_dsl_review import (
    build_reviews,
    map_projected_seams,
    project_proposer_outputs,
    render_program_review,
    select_representative_test_rows,
)
from benchmark.tests.test_pattern_dsl import _graph, _roles


class PatternDSLReviewTests(unittest.TestCase):
    def test_representative_selection_is_category_covering_and_deterministic(self) -> None:
        rows = [
            {"sample_id": "top_7", "garment_category": "top", "split": "test", "panel_count": 7, "stitch_count": 7},
            {"sample_id": "pants_9", "garment_category": "pants", "split": "test", "panel_count": 9, "stitch_count": 9},
            {"sample_id": "train_only", "garment_category": "pants", "split": "train", "panel_count": 1, "stitch_count": 1},
            {"sample_id": "top_1", "garment_category": "top", "split": "test", "panel_count": 1, "stitch_count": 1},
            {"sample_id": "skirt_4", "garment_category": "skirt", "split": "test", "panel_count": 4, "stitch_count": 4},
            {"sample_id": "top_5", "garment_category": "top", "split": "test", "panel_count": 5, "stitch_count": 5},
            {"sample_id": "pants_3", "garment_category": "pants", "split": "test", "panel_count": 3, "stitch_count": 3},
            {"sample_id": "top_3", "garment_category": "top", "split": "test", "panel_count": 3, "stitch_count": 3},
        ]
        first = select_representative_test_rows(rows, 3)
        second = select_representative_test_rows(list(reversed(rows)), 3)
        self.assertEqual(first, second)
        self.assertEqual(
            [value["sample_id"] for value in first],
            ["pants_3", "skirt_4", "top_3"],
        )
        self.assertEqual({value["garment_category"] for value in first}, {"pants", "skirt", "top"})

    def test_review_outputs_are_deterministic_and_show_relations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            garment = {
                "sample_id": "sample",
                "garment_category": "top",
                "split": "test",
                "panels": [{"formal_graph": _graph()}],
                "stitch_constraints": [
                    {
                        "constraint_id": "side_pair",
                        "sides": [
                            {"panel_uid": "sample:front", "edge_id": "e4", "length_cm": 4.0},
                            {"panel_uid": "sample:front", "edge_id": "e7", "length_cm": 1.0},
                        ],
                        "source_annotations": [],
                    }
                ],
            }
            garment_path = root / "sample.json"
            garment_path.write_text(json.dumps(garment), encoding="utf-8")
            index_path = root / "index.jsonl"
            index_path.write_text(
                json.dumps(
                    {
                        "sample_id": "sample",
                        "garment_category": "top",
                        "split": "test",
                        "panel_count": 1,
                        "stitch_count": 1,
                        "garment_record_path": garment_path.as_posix(),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            roles = {"sample": {"sample:front": _roles("center_front")}}
            first = build_reviews(index_path, root / "first", count=1, roles=roles, samples_per_curve=17)
            second = build_reviews(index_path, root / "second", count=1, roles=roles, samples_per_curve=17)
            first_record, second_record = first["records"][0], second["records"][0]
            self.assertTrue(first_record["symbolic_validation"]["valid"])
            self.assertEqual(first_record["derived_landmark_count"], 3)
            self.assertEqual(first_record["seam_rendering"]["source"], "garment_record")
            self.assertFalse(first_record["seam_rendering"]["source_seams_replaced"])
            for artifact in ("dsl", "svg", "png"):
                self.assertEqual(
                    first_record["artifacts"][artifact]["sha256"],
                    second_record["artifacts"][artifact]["sha256"],
                )
                path = root / "first" / first_record["artifacts"][artifact]["file"]
                self.assertGreater(path.stat().st_size, 100)
            dsl = (root / "first" / first_record["artifacts"]["dsl"]["file"]).read_text(encoding="utf-8")
            svg = (root / "first" / first_record["artifacts"]["svg"]["file"]).read_text(encoding="utf-8")
            self.assertNotIn("xy_cm", dsl)
            for token in ("NEXT ", "SHARED_ENDPOINT ", "SEWN_TO ", "LANDMARK "):
                self.assertIn(token, dsl)
            for label in ("NEXT", "SHARED", "FNP", "e1 Q", "SEWN side_pair"):
                self.assertIn(label, svg)
            self.assertIn("SOURCE SEWN side_pair", svg)
            self.assertNotIn("PREDICTED SEWN", svg)
            self.assertTrue((root / "first" / "manifest.json").is_file())

    def test_mock_proposer_is_projected_and_mapped_to_formal_edge_ids(self) -> None:
        """Exercise the checkpoint-independent neural-to-symbolic review seam."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph = _graph()
            graph_path = root / "formal_graph.json"
            graph_path.write_text(json.dumps(graph), encoding="utf-8")
            metadata = {
                "sample_id": "sample",
                "category": "top",
                "split": "test",
                "panels": [
                    {
                        "panel_id": "front",
                        "panel_uid": "sample:front",
                        "formal_graph_path": graph_path.as_posix(),
                        "edge_count": 8,
                    }
                ],
            }
            valid = np.ones((1, 8), dtype=bool)
            logits = np.full((1, 8, len(EDGE_ROLES)), -20.0, dtype=np.float32)
            expected = _roles("center_front")
            for edge_index in range(8):
                logits[0, edge_index, EDGE_ROLES.index(expected[f"e{edge_index}"])] = 20.0
            seam_scores = np.zeros((8, 8), dtype=np.float32)
            seam_scores[4, 7] = seam_scores[7, 4] = 0.99
            allowed = np.ones((len(EDGE_ROLES), len(EDGE_ROLES)), dtype=bool)

            mapped, projection = project_proposer_outputs(
                metadata,
                logits,
                seam_scores,
                valid,
                allowed,
                seam_threshold=0.5,
            )

            self.assertTrue(projection.valid)
            self.assertEqual(mapped, {"sample:front": expected})
            self.assertEqual(
                {(value.base_name, value.vertex_index) for value in projection.landmarks},
                {("FNP", 1), ("SNP", 2), ("SP", 3)},
            )
            mapped_seams = map_projected_seams(metadata, projection, valid)
            self.assertEqual(len(mapped_seams), 1)
            self.assertEqual(
                {
                    mapped_seams[0]["first_edge_id"],
                    mapped_seams[0]["second_edge_id"],
                },
                {"e4", "e7"},
            )

            garment = {
                "sample_id": "sample",
                "garment_category": "top",
                "split": "test",
                "panels": [{"formal_graph": graph}],
                "stitch_constraints": [
                    {
                        "constraint_id": "source_only_pair",
                        "sides": [
                            {
                                "panel_uid": "sample:front",
                                "edge_id": "e5",
                                "length_cm": 1.0,
                            },
                            {
                                "panel_uid": "sample:front",
                                "edge_id": "e6",
                                "length_cm": 1.0,
                            },
                        ],
                        "source_annotations": [],
                    }
                ],
            }
            garment_path = root / "garment.json"
            garment_path.write_text(json.dumps(garment), encoding="utf-8")
            index_path = root / "index.jsonl"
            index_path.write_text(
                json.dumps(
                    {
                        "sample_id": "sample",
                        "garment_category": "top",
                        "split": "test",
                        "panel_count": 1,
                        "stitch_count": 1,
                        "garment_record_path": garment_path.as_posix(),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            projection_payload = projection.to_dict()
            projection_payload["mapped_seams"] = mapped_seams
            manifest = build_reviews(
                index_path,
                root / "review",
                count=1,
                roles={"sample": mapped},
                projection_reports={"sample": projection_payload},
                semantic_role_source="checkpoint_symbolic_projection",
                samples_per_curve=17,
            )
            result = manifest["records"][0]
            self.assertEqual(result["semantic_role_source"], "checkpoint_symbolic_projection")
            self.assertEqual(result["derived_landmark_count"], 3)
            self.assertTrue(result["proposal_projection"]["valid"])
            self.assertIn("symbolic_projection", result["artifacts"])
            self.assertEqual(
                result["seam_rendering"],
                {
                    "source": "checkpoint_symbolic_projection",
                    "source_seam_count": 1,
                    "rendered_seam_count": 1,
                    "source_seams_replaced": True,
                },
            )
            dsl = (root / "review" / result["artifacts"]["dsl"]["file"]).read_text(
                encoding="utf-8"
            )
            svg = (root / "review" / result["artifacts"]["svg"]["file"]).read_text(
                encoding="utf-8"
            )
            self.assertIn("predicted_seam_000", dsl)
            self.assertNotIn("source_only_pair", dsl)
            self.assertIn("PREDICTED SEWN predicted_seam_000", svg)
            self.assertNotIn("SOURCE SEWN", svg)
            for label in ("center_front", "neckline", "shoulder", "armhole", "FNP", "SNP", "SP"):
                self.assertIn(label, svg)

    def test_footer_distinguishes_roles_without_applicable_landmarks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "roles_without_landmarks.svg"
            program = compile_formal_graph(
                _graph(), edge_roles={f"e{index}": "other" for index in range(8)}
            )
            render_program_review(program, destination, samples_per_curve=17)
            svg = destination.read_text(encoding="utf-8")
            self.assertIn(
                "ROLE supplied; no applicable FNP/BNP/SNP/SP junction", svg
            )
            self.assertNotIn("semantic ROLE input absent", svg)


if __name__ == "__main__":
    unittest.main()
