from __future__ import annotations

import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from benchmark.drafting_semantics.basic_blocks import (
    DESIGN_BOUNDS,
    MEASUREMENT_BOUNDS,
    PROVENANCE_STATUS,
    SCHEMA_VERSION,
    BasicBlock,
    BasicBlockCorpus,
    build_basic_block,
    generate_corpus,
    generate_variations,
    load_corpus_json,
    write_corpus_json,
)
from benchmark.pattern_pipeline.validation import validate_pattern


class BasicBlockTests(unittest.TestCase):
    def test_defaults_are_valid_but_explicitly_not_industrial_truth(self) -> None:
        for category in ("tshirt", "pants", "skirt"):
            with self.subTest(category=category):
                block = build_basic_block(category)
                block.validate()
                self.assertEqual(block.provenance.status, PROVENANCE_STATUS)
                self.assertEqual(block.provenance.expert_review, "PENDING")
                self.assertFalse(block.provenance.industrial_pattern_claim)
                with self.assertRaisesRegex(ValueError, "industrial"):
                    replace(
                        block,
                        provenance=replace(block.provenance, industrial_pattern_claim=True),
                    ).validate()

    def test_tshirt_has_conservative_neck_and_stable_named_semantics(self) -> None:
        block = build_basic_block("tshirt")
        front = block.panel("front")
        back = block.panel("back")
        self.assertGreater(front.landmark("FNP").xy_cm[1], back.landmark("BNP").xy_cm[1])
        self.assertLessEqual(front.landmark("FNP").xy_cm[1], 10.5)
        self.assertGreater(front.landmark("SP").xy_cm[0], front.landmark("SNP").xy_cm[0])
        self.assertTrue(front.symmetry.cut_on_fold)
        self.assertEqual(front.symmetry.axis, "center_front")
        self.assertEqual(
            front.path("side_seam").landmark_sequence,
            ("SIDE_HEM", "WAIST_SIDE", "UNDERARM"),
        )
        self.assertEqual(front.path("armhole").geometry_kind, "cubic_bezier")
        self.assertFalse(front.darts)

    def test_tshirt_sleeve_cap_is_fitted_to_each_armhole(self) -> None:
        for block in generate_variations("tshirt", 64, seed=20260828):
            audit = block.panel("sleeve").metadata
            front_ratio = audit["front_cap_length_cm"] / audit["front_armhole_length_cm"]
            back_ratio = audit["back_cap_length_cm"] / audit["back_armhole_length_cm"]
            self.assertLessEqual(abs(front_ratio - 1.0), 0.12)
            self.assertLessEqual(abs(back_ratio - 1.0), 0.12)

    def test_pants_use_one_outseam_and_inseam_with_named_construction_levels(self) -> None:
        block = build_basic_block("pants")
        for panel_id in ("front_pants", "back_pants"):
            panel = block.panel(panel_id)
            self.assertEqual(
                panel.path("outseam").landmark_sequence,
                ("SIDE_WAIST", "SIDE_HIP", "SIDE_KNEE", "SIDE_HEM"),
            )
            self.assertEqual(
                panel.path("inseam").landmark_sequence,
                ("INSEAM_HEM", "INSEAM_KNEE", "CROTCH_POINT"),
            )
            self.assertEqual(
                {line.name for line in panel.reference_lines},
                {"WL", "HL", "CL", "KL", "GRAIN"},
            )
            self.assertEqual(len(panel.darts), 1)
            self.assertEqual(panel.darts[0].apex_landmark, "WAIST_DART_APEX")

        front = block.panel("front_pants")
        back = block.panel("back_pants")
        requested_knee = block.design["knee_circumference_cm"] + block.design["knee_ease_cm"]
        requested_hem = block.design["hem_circumference_cm"]
        drafted_knee = sum(
            abs(panel.landmark("SIDE_KNEE").xy_cm[0] - panel.landmark("INSEAM_KNEE").xy_cm[0])
            for panel in (front, back)
        )
        drafted_hem = sum(
            abs(panel.landmark("SIDE_HEM").xy_cm[0] - panel.landmark("INSEAM_HEM").xy_cm[0])
            for panel in (front, back)
        )
        self.assertTrue(math.isclose(drafted_knee, requested_knee, abs_tol=1e-6))
        self.assertTrue(math.isclose(drafted_hem, requested_hem, abs_tol=1e-6))

    def test_skirt_has_two_archetypes_and_only_evidenced_provisional_slit_cue(self) -> None:
        block = build_basic_block("skirt")
        self.assertEqual({panel.id for panel in block.panels}, {"front_skirt", "back_skirt"})
        front = block.panel("front_skirt")
        back = block.panel("back_skirt")
        self.assertTrue(front.symmetry.cut_on_fold)
        self.assertFalse(back.symmetry.cut_on_fold)
        self.assertIn("SLIT_END", {item.name for item in back.landmarks})
        self.assertIn("slit", {item.name for item in back.paths})
        serialized = block.to_json().lower()
        self.assertNotIn("zipper", serialized)
        self.assertNotIn("notch", serialized)
        self.assertEqual(len(front.darts), 1)
        self.assertEqual(len(back.darts), 1)

    def test_skirt_center_hip_is_a_real_boundary_landmark_on_hl(self) -> None:
        block = build_basic_block("skirt")
        for panel_id, point_name, center_path in (
            ("front_skirt", "CF_HIP", "center_front"),
            ("back_skirt", "CB_HIP", "center_back"),
        ):
            panel = block.panel(panel_id)
            center_hip = panel.landmark(point_name)
            center_waist = panel.landmark(
                "CF_WAIST" if panel_id == "front_skirt" else "CB_WAIST"
            )
            hip_line = next(line for line in panel.reference_lines if line.name == "HL")
            self.assertNotEqual(center_hip.xy_cm[1], center_waist.xy_cm[1])
            self.assertAlmostEqual(center_hip.xy_cm[1], hip_line.start_cm[1])
            self.assertAlmostEqual(center_hip.xy_cm[1], hip_line.end_cm[1])
            self.assertIn(point_name, panel.path(center_path).landmark_sequence)

    def test_pattern_document_annotations_match_teacher_student_queries(self) -> None:
        expected = {
            "tshirt": {
                "landmarks": {"FNP", "BNP", "SNP_front", "SNP_back", "SP_front", "SP_back"},
                "paths": {"front_neckline", "back_neckline", "front_armhole", "back_armhole", "sleeve_head"},
                "reference_lines": {"front_BL", "back_BL", "front_WL", "back_WL", "front_HL", "back_HL"},
            },
            "pants": {
                "landmarks": {"CF_waist", "CB_waist", "front_center_hip", "back_center_hip", "front_crotch_point", "back_crotch_point", "front_knee_in", "front_hem_out", "front_dart_apex", "back_dart_apex"},
                "paths": {"front_waistline", "back_waistline", "side_seam", "inseam", "front_crotch_curve", "back_crotch_curve", "front_dart_leg", "back_dart_leg"},
                "reference_lines": {"front_WL", "back_WL", "front_HL", "back_HL", "front_KL", "back_KL", "front_CL", "back_CL", "front_GRAIN", "back_GRAIN"},
            },
            "skirt": {
                "landmarks": {"front_center_waist", "back_center_waist", "front_center_hip", "back_center_hip", "front_side_waist", "back_side_hip", "front_dart_apex", "back_dart_apex", "slit_end"},
                "paths": {"waistline", "side_seam", "center_seam", "hemline", "front_dart_leg", "back_dart_leg", "slit"},
                "reference_lines": {"front_WL", "back_WL", "front_HL", "back_HL", "front_GRAIN", "back_GRAIN"},
            },
        }
        for category, required in expected.items():
            with self.subTest(category=category):
                document = build_basic_block(category).to_pattern_document(curve_samples=8)
                annotations = document.annotations
                self.assertTrue(required["landmarks"] <= set(annotations["semantic_landmarks"]))
                self.assertTrue(required["paths"] <= set(annotations["semantic_paths"]))
                self.assertTrue(
                    required["reference_lines"]
                    <= set(annotations["semantic_reference_lines"])
                )
                self.assertEqual(annotations["provenance_status"], PROVENANCE_STATUS)
                self.assertTrue(
                    annotations["semantic_query_adapter"]["no_zipper_notch_or_seam_allowance_claim"]
                )
                if category == "skirt":
                    self.assertIsNone(annotations["semantic_query_presence"]["closure"])
                    self.assertIsNone(annotations["semantic_query_presence"]["closure_end"])
                for panel in document.panels:
                    for index, edge in enumerate(panel.edges):
                        following = panel.edges[(index + 1) % len(panel.edges)]
                        self.assertEqual(edge.points[-1], following.points[0])

    def test_bounds_reject_unknown_and_implausible_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown"):
            build_basic_block("tshirt", design={"mystery": 1.0})
        with self.assertRaisesRegex(ValueError, "outside"):
            build_basic_block("tshirt", design={"front_neck_depth_cm": 20.0})
        with self.assertRaisesRegex(ValueError, "between 20 and 34"):
            build_basic_block("pants", measurements={"outseam_cm": 112.0, "inseam_cm": 68.0})

    def test_variations_are_deterministic_unique_and_inside_declared_bounds(self) -> None:
        first = generate_variations("pants", 12, seed=77)
        second = generate_variations("pants", 12, seed=77)
        third = generate_variations("pants", 12, seed=78)
        self.assertEqual(
            [record.to_json(indent=None) for record in first],
            [record.to_json(indent=None) for record in second],
        )
        self.assertNotEqual(first[0].to_json(indent=None), third[0].to_json(indent=None))
        self.assertEqual(len({record.sample_id for record in first}), len(first))
        for record in first:
            record.validate()
            self.assertTrue(validate_pattern(record.to_pattern_document()).accepted)
            for name, value in record.measurements.items():
                bound = MEASUREMENT_BOUNDS[record.category][name]
                self.assertGreaterEqual(value, bound.low)
                self.assertLessEqual(value, bound.high)

        tshirts = generate_variations("tshirt", 64, seed=81)
        self.assertTrue(all(item.measurements["waist_cm"] <= item.measurements["bust_cm"] + 4.0 for item in tshirts))
        self.assertTrue(all(item.measurements["hip_cm"] >= item.measurements["waist_cm"] + 4.0 for item in tshirts))
        for category in ("pants", "skirt"):
            self.assertTrue(
                all(
                    item.measurements["waist_cm"] <= item.measurements["hip_cm"] - 12.0
                    for item in generate_variations(category, 64, seed=81)
                )
            )
            for name, value in record.design.items():
                bound = DESIGN_BOUNDS[record.category][name]
                self.assertGreaterEqual(value, bound.low)
                self.assertLessEqual(value, bound.high)

    def test_json_round_trip_is_lossless_for_record_and_corpus(self) -> None:
        record = build_basic_block("skirt", sample_id="round_trip")
        restored = BasicBlock.from_json(record.to_json())
        self.assertEqual(record, restored)

        corpus = generate_corpus({"tshirt": 2, "pants": 2, "skirt": 2}, seed=19)
        restored_corpus = BasicBlockCorpus.from_dict(json.loads(corpus.to_json()))
        self.assertEqual(corpus, restored_corpus)
        with tempfile.TemporaryDirectory() as folder:
            path = write_corpus_json(corpus, Path(folder) / "corpus.json")
            self.assertEqual(load_corpus_json(path), corpus)

    def test_v2_serialized_blocks_fail_loudly_after_center_hip_schema_change(self) -> None:
        self.assertEqual(SCHEMA_VERSION, "basic-garment-blocks/v3")
        payload = build_basic_block("skirt").to_dict()
        payload["schema_version"] = "basic-garment-blocks/v2"
        with self.assertRaisesRegex(ValueError, "basic-garment-blocks/v3"):
            BasicBlock.from_dict(payload)


if __name__ == "__main__":
    unittest.main()
