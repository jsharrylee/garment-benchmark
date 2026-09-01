from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from benchmark.retrieval.corpus import PatternRecord, build_gcd_ts_record, infer_garment_category
from benchmark.retrieval.features import multiview_descriptor
from benchmark.retrieval.index import PatternIndex, QueryEvidence
from benchmark.retrieval.garmentcode import convert_garmentcode_specification
from benchmark.retrieval.anchor_bank import load_procedural_anchors, rank_dataset_anchors
from benchmark.pattern_pipeline.validation import validate_pattern


def _mask(path: Path, inset: int) -> None:
    image = Image.new("L", (64, 64), 0)
    ImageDraw.Draw(image).rectangle((inset, 8, 63 - inset, 55), fill=255)
    image.save(path)


def _record(sample: str, category: str, panel_count: int, edge_count: int, descriptor: tuple[float, ...]) -> PatternRecord:
    return PatternRecord(
        sample_id=sample,
        category=category,
        panel_names=tuple(f"panel_{index}" for index in range(panel_count)),
        panel_count=panel_count,
        edge_count=edge_count,
        mean_edges_per_panel=edge_count / panel_count,
        semantic_counts={},
        visual_descriptor=descriptor,
        source_pattern_sha256=sample * 8,
        view_sha256=(sample, sample, sample, sample),
    )


class RetrievalTests(unittest.TestCase):
    def test_dataset_anchor_ranking_uses_soft_structure_and_complexity(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "catalog.json"
            records = [
                {"sample_id": "simple", "category": "pants", "panel_count": 4, "edge_count": 20, "stitch_count": 8, "split": "training", "specification_sha256": "0" * 64},
                {"sample_id": "complex", "category": "pants", "panel_count": 14, "edge_count": 70, "stitch_count": 30, "split": "training", "specification_sha256": "1" * 64},
            ]
            path.write_text(json.dumps({"dataset": "GCDv2", "license": "CC BY 4.0", "records": records}), encoding="utf-8")
            low_confidence = rank_dataset_anchors(
                path,
                category="pants",
                reweaver_panel_count=14,
                reweaver_edge_count=70,
                reweaver_reliability=0.01,
                garment_particles_panel_count=14,
                garment_particles_edge_count=70,
                garment_particles_reliability=0.01,
                top_k=2,
            )
            self.assertEqual(low_confidence[0]["sample_id"], "simple")
            high_confidence = rank_dataset_anchors(
                path,
                category="pants",
                reweaver_panel_count=14,
                reweaver_edge_count=70,
                reweaver_reliability=1.0,
                garment_particles_panel_count=14,
                garment_particles_edge_count=70,
                garment_particles_reliability=1.0,
                top_k=2,
            )
            # Genuinely high-confidence agreeing evidence may promote a complex
            # garment, while the measured low-confidence target run stays simple.
            self.assertEqual(high_confidence[0]["sample_id"], "complex")

    def test_procedural_anchor_bank_supplies_missing_category(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bank.json"
            path.write_text(
                json.dumps(
                    {
                        "source": "official/source",
                        "source_commit": "abc123",
                        "source_code_license": "MIT",
                        "records": [
                            {
                                "anchor_id": "pants",
                                "category": "pants",
                                "panel_count": 6,
                                "stitch_count": 24,
                                "specification_sha256": "0" * 64,
                            },
                            {
                                "anchor_id": "shirt",
                                "category": "top",
                                "panel_count": 8,
                                "stitch_count": 16,
                                "specification_sha256": "1" * 64,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            anchors = load_procedural_anchors(path, category="pants")
            self.assertEqual([anchor.anchor_id for anchor in anchors], ["pants"])
            self.assertEqual(anchors[0].source_code_license, "MIT")

    def test_category_inference(self):
        self.assertEqual(infer_garment_category(["skirt_front", "wb_front"]), "skirt")
        self.assertEqual(infer_garment_category(["left_ftorso", "right_sleeve_f"]), "top")
        self.assertEqual(infer_garment_category(["left_ftorso", "sl_left_cuff_skirt_f"]), "top")
        self.assertEqual(infer_garment_category(["left_ftorso", "skirt_front"]), "dress")
        self.assertEqual(infer_garment_category(["pants_front", "pants_back"]), "pants")

    def test_multiview_descriptor_requires_four_distinct_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = []
            for index in range(4):
                path = root / f"view_{index}.png"
                _mask(path, 5 + index)
                paths.append(path)
            descriptor = multiview_descriptor(paths)
            self.assertEqual(len(descriptor), 84)
            with self.assertRaises(ValueError):
                multiview_descriptor(paths[:3])

    def test_gcd_record_preserves_panel_semantics(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            pattern = root / "demo_2d_panel.json"
            pattern.write_text(
                json.dumps(
                    {
                        "panel_order": ["skirt_front", "skirt_back", "wb_front", "wb_back"],
                        "panels": {str(index): {"edge_points": [[[0, 0], [1, 0]]] * 4} for index in range(4)},
                    }
                ),
                encoding="utf-8",
            )
            views = []
            for index in range(4):
                path = root / f"view_{index}.png"
                _mask(path, 8)
                views.append(path)
            record = build_gcd_ts_record(pattern, views)
            self.assertEqual(record.category, "skirt")
            self.assertEqual(record.panel_count, 4)
            self.assertEqual(record.edge_count, 16)
            self.assertEqual(record.semantic_counts["waistband"], 2)

    def test_model_hints_change_ranking_and_missing_category_is_rejected(self):
        descriptor = tuple([0.5] * 84)
        index = PatternIndex(
            [
                _record("simple", "top", 4, 16, descriptor),
                _record("complex", "top", 14, 72, descriptor),
            ]
        )
        query = QueryEvidence(
            descriptor,
            category="top",
            reweaver_panel_count=15,
            reweaver_edge_count=70,
            reweaver_reliability=1.0,
            garment_particles_panel_count=14,
            garment_particles_edge_count=72,
            garment_particles_reliability=1.0,
        )
        result = index.search(query)
        self.assertEqual(result.candidates[0].sample_id, "complex")
        rejected = index.search(QueryEvidence(descriptor, category="pants"))
        self.assertEqual(rejected.decision, "NO_SUITABLE_ANCHOR")
        self.assertEqual(rejected.eligible_size, 0)

    def test_garmentcode_anchor_conversion_preserves_stitch_graph(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "anchor_specification.json"
            panel = {
                "translation": [0, 0, 0],
                "rotation": [0, 0, 0],
                "vertices": [[0, 0], [2, 0], [2, 2], [0, 2]],
                "edges": [
                    {"endpoints": [0, 1]},
                    {"endpoints": [1, 2], "label": "lower_interface"},
                    {"endpoints": [2, 3]},
                    {"endpoints": [3, 0]},
                ],
            }
            path.write_text(
                json.dumps(
                    {
                        "pattern": {
                            "panels": {"front": panel, "back": panel},
                            "stitches": [[{"panel": "front", "edge": 1}, {"panel": "back", "edge": 3}, "right_wrong"]],
                        }
                    }
                ),
                encoding="utf-8",
            )
            document = convert_garmentcode_specification(path, anchor_id="test_anchor", source_license="CC BY 4.0")
            self.assertEqual(len(document.panels), 2)
            self.assertEqual(len(document.stitches), 1)
            self.assertTrue(document.annotations["template_retrieval"])
            self.assertEqual(document.provenance["source_license"], "CC BY 4.0")
            self.assertEqual(len(document.annotations["edge_labels"]), 2)
            self.assertEqual(document.annotations["source_stitch_tags"], {"stitch_0": ["right_wrong"]})
            self.assertTrue(validate_pattern(document).accepted)


if __name__ == "__main__":
    unittest.main()
