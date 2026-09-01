from __future__ import annotations

import unittest
from dataclasses import replace

from benchmark.drafting_semantics.basic_blocks import build_basic_block
from benchmark.pattern_pipeline.schema import Edge, Panel, PatternDocument
from benchmark.pattern_pipeline.semantic_editing import (
    LandmarkResidual,
    PathResidual,
    SemanticResidualPlan,
    apply_semantic_residual,
)
from benchmark.pattern_pipeline.validation import validate_pattern


def _anchor() -> PatternDocument:
    panel = Panel(
        id="front",
        edges=(
            Edge("neck", ((0.0, 10.0), (5.0, 8.0), (10.0, 10.0))),
            Edge("side", ((10.0, 10.0), (10.0, 0.0))),
            Edge("hem", ((10.0, 0.0), (0.0, 0.0))),
            Edge("center", ((0.0, 0.0), (0.0, 10.0))),
        ),
    )
    return PatternDocument(
        pattern_id="basic_tshirt",
        generator="provisional expert block",
        panels=(panel,),
        stitches=(),
        annotations={
            "semantic_landmarks": {
                "front/FNP": [{"panel_id": "front", "edge_id": "neck", "point_index": 0}],
            },
            "semantic_paths": {
                "front/neckline": [{"panel_id": "front", "edge_ids": ["neck"]}],
            },
        },
    )


class SemanticEditingTests(unittest.TestCase):
    def test_landmark_and_path_residual_preserve_topology_and_close_loop(self):
        source = _anchor()
        plan = SemanticResidualPlan(
            category="tshirt",
            landmark_residuals={"front/FNP": LandmarkResidual(0.0, 1.0, influence_radius_cm=4.0)},
            path_residuals={"front/neckline": PathResidual(chord_scale=0.9, normal_scale=1.2)},
        )
        edited = apply_semantic_residual(source, plan)
        self.assertEqual(len(edited.panels), len(source.panels))
        self.assertEqual(len(edited.panels[0].edges), len(source.panels[0].edges))
        self.assertEqual(len(edited.stitches), len(source.stitches))
        for index, edge in enumerate(edited.panels[0].edges):
            following = edited.panels[0].edges[(index + 1) % len(edited.panels[0].edges)]
            self.assertEqual(edge.points[-1], following.points[0])
        self.assertNotEqual(edited.panels[0].edges[0].points, source.panels[0].edges[0].points)
        receipt = edited.annotations["semantic_edit_receipt"]
        self.assertTrue(receipt["topology_preserved"])
        self.assertEqual(receipt["applied_landmarks"], ["front/FNP"])
        self.assertEqual(receipt["applied_paths"], ["front/neckline"])

    def test_missing_semantic_is_fail_closed_or_audited_skip(self):
        source = _anchor()
        plan = SemanticResidualPlan(
            category="tshirt",
            landmark_residuals={"front/SP": LandmarkResidual(1.0, 0.0)},
        )
        with self.assertRaises(ValueError):
            apply_semantic_residual(source, plan)
        edited = apply_semantic_residual(source, plan, strict=False)
        self.assertEqual(edited.annotations["semantic_edit_receipt"]["skipped"], ["landmark:front/SP"])

    def test_implausible_path_scale_is_rejected(self):
        with self.assertRaises(ValueError):
            SemanticResidualPlan(
                category="pants",
                path_residuals={"front/inseam": PathResidual(chord_scale=2.0)},
            ).validate()

    def test_disconnected_waist_segments_are_transformed_independently(self):
        source = build_basic_block("pants").to_pattern_document(curve_samples=8)
        before = tuple(len(panel.edges) for panel in source.panels)
        plan = SemanticResidualPlan(
            category="pants",
            path_residuals={"front_waistline": PathResidual(chord_scale=0.9)},
        )
        edited = apply_semantic_residual(source, plan)
        self.assertEqual(tuple(len(panel.edges) for panel in edited.panels), before)
        self.assertTrue(validate_pattern(edited).accepted)
        self.assertEqual(
            edited.annotations["semantic_edit_receipt"]["applied_paths"],
            ["front_waistline"],
        )

    def test_path_endpoint_edit_is_propagated_without_half_attenuation(self):
        source = _anchor()
        # This isolated propagation case intentionally has no named landmark
        # ownership at the endpoint.  Named endpoints are tested separately
        # below and must be changed through their landmark residual instead.
        source = replace(
            source,
            annotations={
                **source.annotations,
                "semantic_landmarks": {},
            },
        )
        edited = apply_semantic_residual(
            source,
            SemanticResidualPlan(
                category="tshirt",
                path_residuals={
                    "front/neckline": PathResidual(chord_scale=0.8)
                },
            ),
        )
        # A 10 cm chord scaled to 8 cm moves its left endpoint by exactly
        # +1 cm.  The old final-loop averaging reduced this to +0.5 cm.
        self.assertAlmostEqual(edited.panels[0].edges[0].points[0][0], 1.0, places=7)
        self.assertEqual(
            edited.panels[0].edges[0].points[0],
            edited.panels[0].edges[-1].points[-1],
        )

    def test_landmark_owns_endpoint_when_path_also_predicts_translation(self):
        source = _anchor()
        edited = apply_semantic_residual(
            source,
            SemanticResidualPlan(
                category="tshirt",
                landmark_residuals={
                    "front/FNP": LandmarkResidual(
                        0.0, 1.0, influence_radius_cm=4.0
                    )
                },
                path_residuals={
                    "front/neckline": PathResidual(normal_offset_cm=1.0)
                },
            ),
        )
        # The path shape may change between landmarks, but it cannot add its
        # endpoint translation on top of the landmark's exact +1 cm target.
        self.assertEqual(edited.panels[0].edges[0].points[0], (0.0, 11.0))
        self.assertEqual(edited.panels[0].edges[-1].points[-1], (0.0, 11.0))

    def test_unidentified_multiple_instances_fail_closed(self):
        source = build_basic_block("pants").to_pattern_document(curve_samples=8)
        annotations = dict(source.annotations)
        semantic_paths = dict(annotations["semantic_paths"])
        semantic_paths["ambiguous_dart"] = [
            {
                "panel_id": "front_pants",
                "edge_ids": ["dart_leg_inner", "dart_leg_outer"],
            },
            {
                "panel_id": "back_pants",
                "edge_ids": ["dart_leg_inner", "dart_leg_outer"],
            },
        ]
        annotations["semantic_paths"] = semantic_paths
        source = replace(source, annotations=annotations)
        plan = SemanticResidualPlan(
            category="pants",
            path_residuals={"ambiguous_dart": PathResidual(chord_scale=0.95)},
        )
        with self.assertRaisesRegex(ValueError, "physical instances"):
            apply_semantic_residual(source, plan)
        edited = apply_semantic_residual(source, plan, strict=False)
        self.assertEqual(edited.panels, source.panels)
        self.assertEqual(
            edited.annotations["semantic_edit_receipt"]["multiplicity_gated"],
            ["path:ambiguous_dart:unsupported_multiplicity:2"],
        )

    def test_distinct_front_dart_query_does_not_touch_back_dart(self):
        source = build_basic_block("pants").to_pattern_document(curve_samples=8)
        annotations = dict(source.annotations)
        semantic_paths = dict(annotations["semantic_paths"])
        semantic_paths["front_dart_leg"] = [
            {
                "panel_id": "front_pants",
                "edge_ids": ["dart_leg_inner", "dart_leg_outer"],
            }
        ]
        annotations["semantic_paths"] = semantic_paths
        source = replace(source, annotations=annotations)
        back_before = next(panel for panel in source.panels if panel.id == "back_pants")
        edited = apply_semantic_residual(
            source,
            SemanticResidualPlan(
                category="pants",
                path_residuals={
                    "front_dart_leg": PathResidual(normal_scale=0.9)
                },
            ),
        )
        back_after = next(panel for panel in edited.panels if panel.id == "back_pants")
        self.assertEqual(back_after, back_before)
        self.assertEqual(
            edited.annotations["semantic_edit_receipt"]["applied_paths"],
            ["front_dart_leg"],
        )


if __name__ == "__main__":
    unittest.main()
