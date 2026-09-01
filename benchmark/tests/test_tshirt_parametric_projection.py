from __future__ import annotations

import unittest

import numpy as np

from benchmark.drafting_semantics.basic_blocks import DESIGN_BOUNDS
from benchmark.drafting_semantics.basic_semantic_targets import (
    semantic_target_from_basic_block,
)
from benchmark.drafting_semantics.tshirt_parametric_projection import (
    TSHIRT_DRAFT_PARAMETER_NAMES,
    TShirtProjectionConfig,
    audit_tshirt_constraints,
    decode_tshirt_parameters,
    default_normalized_parameters,
    design_from_normalized,
    fit_tshirt_drafting_parameters,
    normalized_from_design,
)
from benchmark.pattern_pipeline.four_view_semantic_inference import (
    StaticSemanticPrediction,
)
from benchmark.pattern_pipeline.parametric_drafting_inference import (
    decode_static_tshirt_prediction,
)


class TShirtParametricProjectionTests(unittest.TestCase):
    def test_parameter_schema_round_trip_uses_existing_bounds(self) -> None:
        default = default_normalized_parameters()
        self.assertEqual(default.shape, (len(TSHIRT_DRAFT_PARAMETER_NAMES),))
        design = design_from_normalized(default)
        self.assertEqual(tuple(design), TSHIRT_DRAFT_PARAMETER_NAMES)
        np.testing.assert_allclose(normalized_from_design(design), default, atol=1e-12)
        for name, value in design.items():
            self.assertEqual(value, DESIGN_BOUNDS["tshirt"][name].default)

    def test_decoder_couples_shared_points_symmetry_and_sleeve_ease(self) -> None:
        for fraction in (0.0, 0.2, 0.5, 0.8, 1.0):
            with self.subTest(fraction=fraction):
                values = np.full(len(TSHIRT_DRAFT_PARAMETER_NAMES), fraction)
                block = decode_tshirt_parameters(values, sample_id=f"edge_{fraction}")
                audit = audit_tshirt_constraints(block)
                self.assertTrue(audit.passed, audit.to_dict())
                self.assertAlmostEqual(audit.total_sleeve_cap_ease_ratio, 1.01, places=4)
                document = block.to_pattern_document(curve_samples=24)
                for panel in document.panels:
                    for index, edge in enumerate(panel.edges):
                        following = panel.edges[(index + 1) % len(panel.edges)]
                        self.assertEqual(edge.points[-1], following.points[0])

    def test_inverse_projection_improves_a_known_decoder_observation(self) -> None:
        known = np.asarray(
            [0.18, 0.72, 0.31, 0.64, 0.22, 0.76, 0.35, 0.67, 0.28, 0.73, 0.42, 0.61],
            dtype=np.float64,
        )
        self.assertEqual(len(known), len(TSHIRT_DRAFT_PARAMETER_NAMES))
        target = semantic_target_from_basic_block(
            decode_tshirt_parameters(known, sample_id="known"), curve_samples=24
        )
        confidence = target.query_applicability.astype(np.float64)
        result = fit_tshirt_drafting_parameters(
            target.coordinates,
            target.presence,
            confidence,
            query_mask=target.query_applicability,
            config=TShirtProjectionConfig(
                confidence_threshold=0.5,
                prior_strength=1e-6,
                max_function_evaluations=600,
            ),
            sample_id="known_fit",
        )
        self.assertTrue(result.optimizer_success, result.optimizer_message)
        self.assertTrue(result.constraint_audit.passed)
        self.assertLess(result.final_data_loss, result.initial_data_loss * 0.15)

    def test_fit_rejects_empty_or_malformed_observations(self) -> None:
        query_count = semantic_target_from_basic_block(
            decode_tshirt_parameters(default_normalized_parameters())
        ).coordinates.shape[0]
        with self.assertRaisesRegex(ValueError, "shape"):
            fit_tshirt_drafting_parameters(
                np.zeros((query_count, 2)),
                np.ones(query_count),
                np.ones(query_count),
            )
        with self.assertRaisesRegex(ValueError, "no confident"):
            fit_tshirt_drafting_parameters(
                np.zeros((query_count, 8)),
                np.zeros(query_count),
                np.zeros(query_count),
            )

    def test_static_visual_contract_decodes_without_a_pattern_input(self) -> None:
        known = np.linspace(0.2, 0.75, len(TSHIRT_DRAFT_PARAMETER_NAMES))
        target = semantic_target_from_basic_block(
            decode_tshirt_parameters(known, sample_id="visual_contract")
        )
        prediction = StaticSemanticPrediction(
            category="tshirt",
            presence_probability=target.presence.astype(np.float64),
            coordinates=target.coordinates.astype(np.float64),
            coordinate_confidence=target.query_applicability.astype(np.float64),
            query_mask=target.query_applicability,
            coordinate_mask=target.coordinate_mask,
            confidence_receipt={"status": "UNIT_TEST_SYNTHETIC"},
        )
        result = decode_static_tshirt_prediction(
            prediction,
            config=TShirtProjectionConfig(prior_strength=1e-6),
            output_id="visual_contract_output",
        )
        self.assertEqual(result.receipt["status"], "APPLIED_VALIDATED_PARAMETRIC_DRAFT")
        self.assertEqual(
            result.receipt["input_contract"]["target_pattern_used_for_fit"],
            "NOT_ATTESTED",
        )
        self.assertFalse(result.receipt["input_contract"]["visual_origin_attested"])
        self.assertTrue(result.validation.accepted)
        self.assertTrue(result.projection.constraint_audit.passed)
        self.assertEqual(
            [panel.id for panel in result.document.panels],
            ["front", "back", "sleeve#right", "sleeve#left"],
        )
        self.assertEqual(len(result.document.stitches), 10)
        self.assertTrue(
            result.receipt["physical_pattern_graph"][
                "all_path_ids_are_instance_aware"
            ]
        )
        exact = result.graph.sleeve_head_constraint
        self.assertTrue(exact.converged)
        self.assertLessEqual(abs(exact.residual_cm), exact.tolerance_cm)
        sampled = result.receipt["physical_pattern_graph"][
            "sampled_document_sleeve_head_constraint"
        ]
        self.assertTrue(sampled["passed"])
        self.assertGreater(sampled["tolerance_cm"], exact.tolerance_cm)
        self.assertAlmostEqual(
            exact.sleeve_ease_cm,
            (exact.front_armhole_length_cm + exact.back_armhole_length_cm) * 0.01,
            places=10,
        )


if __name__ == "__main__":
    unittest.main()
