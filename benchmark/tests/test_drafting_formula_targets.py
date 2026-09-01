from __future__ import annotations

import unittest

from benchmark.drafting_semantics.drafting_formula_targets import (
    build_drafting_formula_targets,
    build_sleeve_armhole_relation,
)
from benchmark.drafting_semantics.tshirt_schema import (
    ConstructionOperation,
    CurveGeometry,
    TShirtTraceRecord,
    TracedEdge,
    TracedPanel,
    TracedPoint,
)


def _single_curve_panel(
    panel_id: str,
    panel_role: str,
    edge_role: str,
    start: tuple[float, float],
    end: tuple[float, float],
    controls: tuple[tuple[float, float], tuple[float, float]],
    start_name: str | None,
    end_name: str | None,
) -> TracedPanel:
    operation_id = f"op.{panel_id}"
    points = (
        TracedPoint(
            id=f"{panel_id}.start",
            panel_id=panel_id,
            xy_cm=start,
            formula="author formula start",
            canonical_name=start_name,
            source_name="source_start",
            operation_id=operation_id,
        ),
        TracedPoint(
            id=f"{panel_id}.end",
            panel_id=panel_id,
            xy_cm=end,
            formula="author formula end",
            canonical_name=end_name,
            source_name="source_end",
            operation_id=operation_id,
        ),
    )
    edge = TracedEdge(
        id=f"{panel_id}.edge",
        panel_id=panel_id,
        start_point_id=points[0].id,
        end_point_id=points[1].id,
        semantic_role=edge_role,
        geometry=CurveGeometry(
            kind="cubic_bezier",
            start_cm=start,
            end_cm=end,
            control_points_cm=controls,
        ),
        formula=f"fixture {edge_role} cubic formula",
        operation_id=operation_id,
        provenance={"measurement_inputs": {"design.depth": 0.35}},
    )
    return TracedPanel(
        id=panel_id,
        semantic_role=panel_role,
        points=points,
        edges=(edge,),
        operation_id=operation_id,
    )


class DraftingFormulaTargetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.panels = (
            _single_curve_panel(
                "front_neck", "front", "neckline", (0.0, 0.0), (4.0, 3.0),
                ((0.8, 0.0), (3.5, 2.0)), "FNP", "SNP",
            ),
            _single_curve_panel(
                "front_arm", "front", "armhole", (7.0, 1.0), (8.0, 8.0),
                ((7.2, 2.0), (8.7, 6.0)), "SP", None,
            ),
            _single_curve_panel(
                "sleeve", "sleeve", "sleeve_head", (-5.0, 0.0), (5.0, 0.0),
                ((-3.0, -6.0), (3.0, -6.0)), None, None,
            ),
        )

    def test_path_local_landmarks_tangents_controls_and_semantic_dimensions_are_masked(self):
        targets = build_drafting_formula_targets(
            self.panels,
            source_kind="garmentcode_creation_trace",
            formula_parameters={"front.neckline": {"design.neck_width": 0.4}},
        )
        by_role = {target.semantic_role: target for target in targets}
        self.assertEqual(set(by_role), {"neckline", "armhole", "sleeve_head"})

        neckline = by_role["neckline"]
        self.assertEqual(neckline.panel_role, "front")
        self.assertEqual(neckline.endpoint_names, ("FNP", "SNP"))
        self.assertEqual(neckline.endpoint_name_mask, (True, True))
        self.assertAlmostEqual(neckline.semantic_values["neckline_width_cm"], 4.0)
        self.assertAlmostEqual(neckline.semantic_values["neckline_depth_cm"], 3.0)
        self.assertEqual(neckline.segments[0].normalized_start, (0.0, 0.0))
        self.assertAlmostEqual(neckline.segments[0].normalized_end[0], 1.0)
        self.assertAlmostEqual(neckline.segments[0].normalized_end[1], 0.0)
        self.assertEqual(neckline.segments[0].bezier_control_mask, (True, True))
        self.assertEqual(neckline.endpoint_tangent_mask, (True, True))
        self.assertTrue(neckline.source_parameter_mask["design.neck_width"])
        self.assertTrue(neckline.source_parameter_mask["design.depth"])
        self.assertFalse(neckline.provenance["role_inferred_from_shape"])

        armhole = by_role["armhole"]
        self.assertEqual(armhole.endpoint_names[0], "SP")
        self.assertAlmostEqual(armhole.semantic_values["armhole_depth_cm"], 7.0)
        sleeve = by_role["sleeve_head"]
        self.assertGreater(sleeve.semantic_values["sleeve_cap_height_cm"], 0.0)

    def test_seam_relation_and_schema_round_trip_preserve_evidence_boundary(self):
        targets = build_drafting_formula_targets(self.panels, source_kind="garmentcode_creation_trace")
        relations = build_sleeve_armhole_relation(targets, source_kind="garmentcode_creation_trace")
        self.assertEqual(len(relations), 1)
        relation = relations[0]
        self.assertTrue(relation.value_mask["ease_ratio"])
        self.assertEqual(relation.provenance["pairing_scope"], "aggregate_per_record")
        self.assertFalse(relation.provenance["exact_sleeve_segment_to_front_back_pairing_asserted"])

        operations = tuple(
            ConstructionOperation(
                id=f"op.{panel.id}",
                order=index,
                operation="create_fixture_curve",
                outputs=tuple(edge.id for edge in panel.edges),
            )
            for index, panel in enumerate(self.panels)
        )
        record = TShirtTraceRecord(
            sample_id="formula-fixture",
            split="test",
            source={"name": "fixture"},
            body={"bust": 90.0},
            design={"fixture": True},
            provenance={"fixture": True},
            panels=self.panels,
            operations=operations,
            drafting_formula_targets=targets,
            drafting_seam_relations=relations,
        )
        record.validate()
        restored = TShirtTraceRecord.from_dict(record.to_dict())
        self.assertEqual(restored.drafting_formula_targets, targets)
        self.assertEqual(restored.drafting_seam_relations, relations)
        self.assertEqual(restored.schema_version, "tshirt-construction-trace-1.1")

    def test_arc_through_point_is_not_mislabeled_as_a_bezier_control(self):
        panel_id = "arc_neck"
        points = (
            TracedPoint(
                id=f"{panel_id}.fnp", panel_id=panel_id, xy_cm=(0.0, 0.0),
                formula="FNP", canonical_name="FNP", operation_id="op.arc",
            ),
            TracedPoint(
                id=f"{panel_id}.snp", panel_id=panel_id, xy_cm=(2.0, 0.0),
                formula="SNP", canonical_name="SNP", operation_id="op.arc",
            ),
        )
        panel = TracedPanel(
            id=panel_id,
            semantic_role="front",
            points=points,
            edges=(
                TracedEdge(
                    id=f"{panel_id}.edge",
                    panel_id=panel_id,
                    start_point_id=points[0].id,
                    end_point_id=points[1].id,
                    semantic_role="neckline",
                    geometry=CurveGeometry(
                        kind="arc",
                        start_cm=(0.0, 0.0),
                        end_cm=(2.0, 0.0),
                        control_points_cm=((1.0, -1.0),),
                        center_cm=(1.0, 0.0),
                        radius_cm=1.0,
                        parameters={"source_depth_cm": 1.0},
                    ),
                    formula="circle through FNP, depth point, SNP",
                    operation_id="op.arc",
                ),
            ),
        )
        target = build_drafting_formula_targets(
            (panel,), source_kind="garmentcode_creation_trace"
        )[0]
        self.assertEqual(target.segments[0].bezier_control_mask, (False, False))
        self.assertEqual(target.segments[0].normalized_bezier_controls, ((0.0, 0.0), (0.0, 0.0)))
        self.assertEqual(target.segments[0].source_parameters["geometry.source_depth_cm"], 1.0)
        self.assertGreater(target.scalar_values["depth_cm"], 0.99)


if __name__ == "__main__":
    unittest.main()
