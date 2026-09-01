import json
import unittest
from dataclasses import replace

from benchmark.drafting_semantics.basic_blocks import SCHEMA_VERSION
from benchmark.drafting_semantics.basic_semantic_targets import (
    semantic_target_from_pattern_document,
)
from benchmark.drafting_semantics.tshirt_parametric_decoder import (
    PARAMETER_NAMES,
    TShirtDraftParameters,
    TShirtParametricDraftingDecoder,
    audit_sampled_tshirt_document_sleeve_constraint,
    decode_tshirt_pattern,
)
from benchmark.pattern_pipeline.semantic_editing import semantic_annotation_entries
from benchmark.pattern_pipeline.validation import validate_pattern


class TShirtParametricDecoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.graph = decode_tshirt_pattern(pattern_id="decoder_test")

    def test_parameter_schema_is_bounded_vector_and_projects_residuals(self) -> None:
        parameters = TShirtDraftParameters()
        self.assertEqual(len(parameters.to_vector()), len(PARAMETER_NAMES))
        self.assertEqual(TShirtDraftParameters.schema()["parameterOrder"], list(PARAMETER_NAMES))
        for value, low, high in zip(
            parameters.to_vector(),
            parameters.lower_bounds(),
            parameters.upper_bounds(),
        ):
            self.assertLessEqual(low, value)
            self.assertLessEqual(value, high)

        projected = TShirtDraftParameters.from_vector(
            [-1e6] * len(PARAMETER_NAMES), project=True
        )
        self.assertEqual(projected.to_vector(), projected.lower_bounds())
        residual = parameters.with_residual({"front_neck_depth_cm": 0.7})
        self.assertAlmostEqual(
            residual.front_neck_depth_cm,
            parameters.front_neck_depth_cm + 0.7,
        )
        with self.assertRaisesRegex(ValueError, "unknown T-shirt parameter"):
            parameters.with_residual({"front_neck_dept_cm": 1.0})

    def test_decoder_reuses_basic_block_v3_geometry(self) -> None:
        graph = self.graph
        self.assertEqual(graph.archetype_block.schema_version, SCHEMA_VERSION)
        source_armhole = graph.archetype_block.panel("front").path("armhole")
        decoded_armhole = graph.path("front_armhole#right").segments[0]
        self.assertEqual(decoded_armhole.kind, "cubic_bezier")
        self.assertEqual(
            decoded_armhole.control_points_cm[1:3],
            source_armhole.control_points_cm,
        )
        for path_id in (
            "front_neckline#right",
            "back_neckline#right",
            "front_armhole#right",
            "back_armhole#right",
            "front_sleeve_head#right",
            "back_sleeve_head#right",
        ):
            self.assertEqual(graph.path(path_id).segments[0].kind, "cubic_bezier")

    def test_adjacent_paths_reference_the_same_landmarks(self) -> None:
        graph = self.graph
        neckline = graph.path("front_neckline#right")
        shoulder = graph.path("front_shoulder#right")
        armhole = graph.path("front_armhole#right")
        self.assertEqual(neckline.start_landmark_id, shoulder.end_landmark_id)
        self.assertEqual(shoulder.start_landmark_id, armhole.end_landmark_id)
        self.assertEqual(
            graph.landmark(neckline.start_landmark_id).semantic_name,
            "SNP",
        )
        self.assertEqual(
            graph.landmark(armhole.end_landmark_id).semantic_name,
            "SP",
        )

    def test_left_instances_are_exact_reflections_not_independent_predictions(self) -> None:
        graph = self.graph
        relation = next(
            item
            for item in graph.symmetry_relations
            if item.id == "front_left_right_symmetry"
        )
        source_id, target_id = next(
            pair for pair in relation.path_pairs if pair[0] == "front_armhole#right"
        )
        source = graph.path(source_id).sampled_points(17)
        target = graph.path(target_id).sampled_points(17)
        expected = tuple((-x, y) for x, y in reversed(source))
        for actual, mirrored in zip(target, expected):
            self.assertAlmostEqual(actual[0], mirrored[0], places=12)
            self.assertAlmostEqual(actual[1], mirrored[1], places=12)

    def test_sleeve_head_equation_is_solved_numerically_after_parameter_change(self) -> None:
        baseline = self.graph
        changed = TShirtParametricDraftingDecoder().decode(
            replace(TShirtDraftParameters(), sleeve_ease_cm=2.0),
            pattern_id="more_ease",
        )
        for graph in (baseline, changed):
            receipt = graph.sleeve_head_constraint
            armhole_length = sum(
                graph.path(path_id).length_cm()
                for path_id in receipt.armhole_path_ids
            )
            sleeve_length = sum(
                graph.path(path_id).length_cm()
                for path_id in receipt.sleeve_head_path_ids
            )
            self.assertAlmostEqual(
                sleeve_length,
                armhole_length + receipt.sleeve_ease_cm,
                delta=receipt.tolerance_cm,
            )
            self.assertTrue(receipt.converged)
            self.assertGreater(receipt.iterations, 0)
        self.assertGreater(
            changed.sleeve_head_constraint.solved_cap_height_cm,
            baseline.sleeve_head_constraint.solved_cap_height_cm,
        )
        self.assertEqual(
            changed.path("front_armhole#right").segments,
            baseline.path("front_armhole#right").segments,
        )

    def test_pattern_document_is_valid_and_keeps_exact_and_legacy_queries(self) -> None:
        document = self.graph.to_pattern_document()
        report = validate_pattern(document)
        self.assertTrue(report.accepted, report.to_dict())
        self.assertEqual(report.metrics["panel_count"], 4)
        self.assertEqual(report.metrics["stitch_count"], 10)
        self.assertEqual(
            semantic_annotation_entries(document, "path", "front_armhole#left")[0][
                "panel_id"
            ],
            "front",
        )
        self.assertEqual(
            semantic_annotation_entries(document, "path", "front_armhole")[0][
                "edge_ids"
            ],
            ["front_armhole#right"],
        )
        self.assertIn("front_armhole#left", document.annotations["parametric_path_geometry"])
        self.assertFalse(document.provenance["raw_control_points_predicted"])
        semantic_target = semantic_target_from_pattern_document(
            document,
            category="tshirt",
            source="parametric_decoder_test",
            provenance_status=document.annotations["provenance_status"],
            source_y_axis_down=True,
        )
        semantic_target.validate()
        sampled = audit_sampled_tshirt_document_sleeve_constraint(
            document,
            sleeve_ease_cm=self.graph.sleeve_head_constraint.sleeve_ease_cm,
            samples_per_cubic=self.graph.samples_per_cubic,
        )
        self.assertTrue(sampled.passed, sampled.to_dict())
        self.assertGreater(sampled.tolerance_cm, self.graph.sleeve_head_constraint.tolerance_cm)
        json.dumps(self.graph.to_dict())


if __name__ == "__main__":
    unittest.main()
