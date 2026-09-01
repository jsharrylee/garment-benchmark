from __future__ import annotations

import unittest

import numpy as np

from benchmark.drafting_semantics.basic_blocks import build_basic_block
from benchmark.drafting_semantics.basic_semantic_targets import (
    semantic_target_from_basic_block,
    semantic_target_from_pattern_document,
)
from benchmark.drafting_semantics.semantic_teacher_student import (
    MAX_COORDINATE_DIM,
    SEMANTIC_QUERY_INDEX,
    SEMANTIC_QUERY_INVENTORY,
)
from benchmark.pattern_pipeline.semantic_editing import apply_semantic_residual
from benchmark.pattern_pipeline.semantic_residual_planning import (
    ResidualPlanningConfig,
    build_semantic_residual_plan,
)
from benchmark.pattern_pipeline.validation import validate_pattern


def _empty_targets():
    queries = len(SEMANTIC_QUERY_INVENTORY)
    anchor_coordinates = np.full((queries, MAX_COORDINATE_DIM), np.nan, dtype=np.float64)
    predicted_coordinates = anchor_coordinates.copy()
    anchor_presence = np.zeros(queries, dtype=np.float64)
    predicted_presence = np.zeros(queries, dtype=np.float64)
    predicted_confidence = np.zeros(queries, dtype=np.float64)
    return (
        anchor_coordinates,
        anchor_presence,
        predicted_coordinates,
        predicted_presence,
        predicted_confidence,
    )


def _put(
    targets,
    key: str,
    anchor_values,
    predicted_values,
    *,
    anchor_presence: float = 1.0,
    predicted_presence: float = 1.0,
    confidence: float = 0.95,
) -> None:
    anchor_coordinates, anchor_flags, predicted_coordinates, predicted_flags, scores = targets
    index = SEMANTIC_QUERY_INDEX[key]
    anchor_values = tuple(float(value) for value in anchor_values)
    predicted_values = tuple(float(value) for value in predicted_values)
    anchor_coordinates[index, : len(anchor_values)] = anchor_values
    predicted_coordinates[index, : len(predicted_values)] = predicted_values
    anchor_flags[index] = anchor_presence
    predicted_flags[index] = predicted_presence
    scores[index] = confidence


def _topology(document):
    return (
        len(document.panels),
        tuple(len(panel.edges) for panel in document.panels),
        len(document.stitches),
        tuple((stitch.side_a.panel_id, stitch.side_a.edge_id, stitch.side_b.panel_id, stitch.side_b.edge_id) for stitch in document.stitches),
    )


class SemanticResidualPlanningTests(unittest.TestCase):
    def test_reference_line_predictions_are_diagnostic_and_never_edit_boundaries(self) -> None:
        block = build_basic_block("pants")
        anchor = semantic_target_from_basic_block(block)
        predicted = anchor.coordinates.copy()
        line = SEMANTIC_QUERY_INDEX["pants:reference_line:front_KL"]
        predicted[line, 1] += 0.20
        predicted[line, 3] += 0.20
        plan = build_semantic_residual_plan(
            "pants",
            anchor.coordinates,
            anchor.presence,
            predicted,
            anchor.presence,
            np.ones_like(anchor.presence),
            block,
        )
        self.assertFalse(plan.landmark_residuals)
        self.assertFalse(plan.path_residuals)
        self.assertFalse(plan.gated_queries)

    def test_skirt_center_hip_is_edge_backed_and_editable(self) -> None:
        anchor = build_basic_block("skirt")
        target = build_basic_block("skirt", design={"hip_depth_cm": 22.0})
        anchor_target = semantic_target_from_basic_block(anchor)
        target_semantics = semantic_target_from_basic_block(target)
        plan = build_semantic_residual_plan(
            "skirt",
            anchor_target.coordinates,
            anchor_target.presence,
            target_semantics.coordinates,
            target_semantics.presence,
            np.ones_like(target_semantics.presence),
            anchor,
        )
        self.assertIn("front_center_hip", plan.landmark_residuals)
        self.assertIn("back_center_hip", plan.landmark_residuals)
        edited = apply_semantic_residual(anchor.to_pattern_document(), plan)
        self.assertTrue(validate_pattern(edited).accepted)

    def test_canonical_upward_landmark_move_is_negative_y_in_basic_block(self) -> None:
        block = build_basic_block("tshirt")
        targets = _empty_targets()
        _put(
            targets,
            "tshirt:landmark:FNP",
            (0.10, 0.20),
            (0.10, 0.30),
        )
        plan = build_semantic_residual_plan("tshirt", *targets, block)
        self.assertLess(plan.landmark_residuals["FNP"].dy_cm, 0.0)

    def test_tshirt_exact_queries_and_sleeve_armhole_compatibility(self) -> None:
        block = build_basic_block("tshirt")
        targets = _empty_targets()
        _put(targets, "tshirt:landmark:FNP", (0.10, 0.12), (0.10, 0.06))
        _put(targets, "tshirt:landmark:SP_front", (0.34, 0.08), (0.37, 0.07))
        _put(
            targets,
            "tshirt:path:front_neckline",
            (0.10, 0.12, 0.26, 0.02, 0.20, 0.035, 0.10, 0.28),
            (0.10, 0.06, 0.28, 0.02, 0.22, 0.050, 0.08, 0.34),
        )
        _put(
            targets,
            "tshirt:path:front_armhole",
            (0.34, 0.08, 0.46, 0.34, 0.20, 0.035, 0.18, 0.44),
            (0.37, 0.07, 0.48, 0.35, 0.24, 0.045, 0.16, 0.50),
        )
        _put(
            targets,
            "tshirt:path:back_armhole",
            (0.34, 0.08, 0.46, 0.34, 0.20, 0.030, 0.16, 0.40),
            (0.35, 0.08, 0.47, 0.34, 0.24, 0.037, 0.14, 0.45),
        )
        # Deliberately incompatible: predicted sleeve arc 0.80 versus total
        # armhole arc 0.48.  The planner must spend a bounded compatibility
        # correction on cap shape instead of accepting that ratio literally.
        _put(
            targets,
            "tshirt:path:sleeve_head",
            (0.20, 0.22, 0.80, 0.22, 0.40, 0.10, -0.35, 0.35),
            (0.20, 0.22, 0.80, 0.22, 0.80, 0.10, -0.35, 0.35),
        )
        plan = build_semantic_residual_plan("tshirt", *targets, block)
        self.assertEqual(set(plan.landmark_residuals), {"FNP", "SP_front"})
        self.assertTrue(
            {"front_neckline", "front_armhole", "back_armhole", "sleeve_head"}
            <= set(plan.path_residuals)
        )
        self.assertLessEqual(
            np.hypot(
                plan.landmark_residuals["FNP"].dx_cm,
                plan.landmark_residuals["FNP"].dy_cm,
            ),
            ResidualPlanningConfig().max_landmark_displacement_cm + 1e-8,
        )
        sleeve = plan.path_residuals["sleeve_head"]
        self.assertGreaterEqual(sleeve.normal_scale, ResidualPlanningConfig().min_normal_scale)
        self.assertLessEqual(sleeve.normal_scale, ResidualPlanningConfig().max_normal_scale)
        # Without the arc compatibility correction the raw path estimate is
        # driven upward by the 2x sleeve arc.  The correction reins it back.
        self.assertLess(sleeve.normal_scale, 1.20)

        source = block.to_pattern_document(curve_samples=8)
        edited = apply_semantic_residual(source, plan)
        self.assertEqual(_topology(edited), _topology(source))
        self.assertTrue(edited.annotations["semantic_edit_receipt"]["topology_preserved"])

    def test_pants_knee_hem_crotch_inseam_and_outseam_semantics(self) -> None:
        block = build_basic_block("pants")
        targets = _empty_targets()
        for name, anchor_xy, predicted_xy in (
            ("front_knee_in", (0.22, 0.62), (0.20, 0.64)),
            ("front_knee_out", (0.58, 0.62), (0.61, 0.64)),
            ("front_hem_in", (0.25, 0.94), (0.23, 0.96)),
            ("front_hem_out", (0.55, 0.94), (0.58, 0.96)),
            ("front_crotch_point", (0.08, 0.31), (0.06, 0.33)),
            ("back_crotch_point", (0.04, 0.31), (0.01, 0.34)),
        ):
            _put(targets, f"pants:landmark:{name}", anchor_xy, predicted_xy)
        _put(
            targets,
            "pants:path:side_seam",
            (0.62, 0.05, 0.58, 0.96, 0.93, 0.012, 0.48, 0.50),
            (0.64, 0.05, 0.61, 0.98, 0.96, 0.018, 0.47, 0.51),
        )
        _put(
            targets,
            "pants:path:inseam",
            (0.23, 0.96, 0.08, 0.31, 0.68, -0.025, -0.48, -0.42),
            (0.21, 0.98, 0.06, 0.33, 0.70, -0.032, -0.47, -0.39),
        )
        _put(
            targets,
            "pants:path:front_crotch_curve",
            (0.08, 0.31, 0.30, 0.20, 0.27, 0.055, -0.25, 0.18),
            (0.06, 0.33, 0.30, 0.20, 0.30, 0.067, -0.30, 0.22),
        )
        plan = build_semantic_residual_plan("pants", *targets, block)
        self.assertTrue(
            {
                "front_knee_in",
                "front_knee_out",
                "front_hem_in",
                "front_hem_out",
                "front_crotch_point",
                "back_crotch_point",
            }
            <= set(plan.landmark_residuals)
        )
        # The shared exact query name is side_seam; on the pants anchor it maps
        # directly to both front/back `outseam` VectorPaths.
        self.assertIn("front_crotch_curve", plan.path_residuals)
        self.assertNotIn("side_seam", plan.path_residuals)
        self.assertNotIn("inseam", plan.path_residuals)
        self.assertEqual(
            plan.gated_queries["path:side_seam"],
            "unsupported_one_to_many_query:2_instances",
        )
        self.assertEqual(
            plan.gated_queries["path:inseam"],
            "unsupported_one_to_many_query:2_instances",
        )
        self.assertNotIn("outseam", plan.path_residuals)

        source = block.to_pattern_document(curve_samples=8)
        edited = apply_semantic_residual(source, plan)
        self.assertEqual(_topology(edited), _topology(source))

    def test_skirt_absent_or_low_confidence_dart_slit_is_never_fabricated(self) -> None:
        block = build_basic_block("skirt")
        targets = _empty_targets()
        path = (0.20, 0.90, 0.20, 0.65, 0.25, 0.0, -0.50, -0.50)
        _put(
            targets,
            "skirt:path:slit",
            path,
            (0.20, 0.90, 0.20, 0.58, 0.32, 0.0, -0.50, -0.50),
            anchor_presence=0.0,
            predicted_presence=1.0,
        )
        _put(
            targets,
            "skirt:landmark:slit_end",
            (0.20, 0.65),
            (0.20, 0.58),
            anchor_presence=0.0,
            predicted_presence=1.0,
        )
        _put(
            targets,
            "skirt:path:dart_leg",
            (0.35, 0.03, 0.35, 0.18, 0.16, 0.01, 0.50, 0.50),
            (0.33, 0.03, 0.33, 0.22, 0.20, 0.01, 0.50, 0.50),
            confidence=0.20,
        )
        _put(
            targets,
            "skirt:path:closure",
            path,
            (0.20, 0.90, 0.20, 0.55, 0.35, 0.0, -0.50, -0.50),
        )
        plan = build_semantic_residual_plan("skirt", *targets, block)
        self.assertNotIn("slit", plan.path_residuals)
        self.assertNotIn("slit_end", plan.landmark_residuals)
        self.assertNotIn("dart_leg", plan.path_residuals)
        self.assertNotIn("closure", plan.path_residuals)

    def test_bad_shapes_and_logits_fail_closed(self) -> None:
        block = build_basic_block("tshirt")
        targets = list(_empty_targets())
        targets[1][0] = 2.0
        with self.assertRaisesRegex(ValueError, "not logits"):
            build_semantic_residual_plan("tshirt", *targets, block)

    def test_oracle_basic_block_targets_reduce_full_semantic_error(self) -> None:
        cases = (
            ("tshirt", {"front_neck_depth_cm": 9.5}),
            ("pants", {"knee_circumference_cm": 48.0}),
            ("skirt", {"vent_length_cm": 23.0}),
        )
        for category, design in cases:
            with self.subTest(category=category):
                anchor = build_basic_block(category)
                target = build_basic_block(category, design=design)
                anchor_target = semantic_target_from_basic_block(anchor)
                oracle_target = semantic_target_from_basic_block(target)
                plan = build_semantic_residual_plan(
                    category,
                    anchor_target.coordinates,
                    anchor_target.presence,
                    oracle_target.coordinates,
                    oracle_target.presence,
                    np.ones_like(oracle_target.presence),
                    anchor,
                )
                source = anchor.to_pattern_document()
                edited = apply_semantic_residual(source, plan)
                edited_target = semantic_target_from_pattern_document(
                    edited,
                    category=category,
                    source="oracle_semantic_edit_integration_test",
                    provenance_status="PROVISIONAL_EXPERT_REVIEW",
                    source_y_axis_down=True,
                )
                mask = (
                    anchor_target.coordinate_mask
                    & oracle_target.coordinate_mask
                    & edited_target.coordinate_mask
                )
                before = float(
                    np.mean(
                        np.abs(
                            anchor_target.coordinates[mask]
                            - oracle_target.coordinates[mask]
                        )
                    )
                )
                after = float(
                    np.mean(
                        np.abs(
                            edited_target.coordinates[mask]
                            - oracle_target.coordinates[mask]
                        )
                    )
                )
                self.assertGreater(before, 0.0)
                self.assertLess(after, before)
                self.assertEqual(_topology(edited), _topology(source))
                self.assertTrue(validate_pattern(edited).accepted)

    def test_role_specific_dart_target_edits_front_only(self) -> None:
        for category in ("pants", "skirt"):
            with self.subTest(category=category):
                anchor = build_basic_block(category)
                target = build_basic_block(
                    category, design={"front_dart_length_cm": 10.5}
                )
                anchor_target = semantic_target_from_basic_block(anchor)
                oracle_target = semantic_target_from_basic_block(target)
                plan = build_semantic_residual_plan(
                    category,
                    anchor_target.coordinates,
                    anchor_target.presence,
                    oracle_target.coordinates,
                    oracle_target.presence,
                    np.ones_like(oracle_target.presence),
                    anchor,
                )
                self.assertIn("front_dart_apex", plan.landmark_residuals)
                self.assertIn("front_dart_leg", plan.path_residuals)
                self.assertNotIn("back_dart_apex", plan.landmark_residuals)
                self.assertNotIn("back_dart_leg", plan.path_residuals)
                self.assertNotIn("dart_apex", plan.landmark_residuals)
                self.assertNotIn("dart_leg", plan.path_residuals)

                source = anchor.to_pattern_document()
                back_id = "back_pants" if category == "pants" else "back_skirt"
                back_before = next(panel for panel in source.panels if panel.id == back_id)
                edited = apply_semantic_residual(source, plan)
                back_after = next(panel for panel in edited.panels if panel.id == back_id)
                self.assertEqual(back_after, back_before)


if __name__ == "__main__":
    unittest.main()
