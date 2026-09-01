from __future__ import annotations

from dataclasses import replace
import math
import unittest

import numpy as np

from benchmark.drafting_semantics.basic_blocks import build_basic_block
from benchmark.drafting_semantics.basic_semantic_targets import (
    common_basic_category,
    filter_common_basic_records,
    is_common_basic_pants,
    is_common_basic_skirt,
    is_common_basic_tshirt,
    semantic_target_from_basic_block,
    semantic_target_from_drafting_record,
    semantic_target_from_pattern_document,
    stack_semantic_targets,
)
from benchmark.drafting_semantics.schema import (
    DraftingSemanticRecord,
    EdgeAnnotation,
    Landmark,
    PanelAnnotation,
    ReferenceLine,
)
from benchmark.drafting_semantics.semantic_teacher_student import (
    MAX_COORDINATE_DIM,
    SEMANTIC_QUERY_INDEX,
    SEMANTIC_QUERY_INVENTORY,
)


def _edge(
    panel_id: str,
    index: int,
    vertices: tuple[tuple[float, float], ...],
    endpoints: tuple[int, int],
    role: str,
    curvature_type: str = "line",
) -> EdgeAnnotation:
    start, end = vertices[endpoints[0]], vertices[endpoints[1]]
    return EdgeAnnotation(
        id=f"{panel_id}.edge_{index}",
        index=index,
        endpoints=endpoints,
        start_cm=start,
        end_cm=end,
        curvature_type=curvature_type,
        role=role,
        stitched=False,
        self_stitched=False,
        length_cm=math.dist(start, end),
        evidence="observed_source",
        confidence=1.0,
    )


def _panel(
    panel_id: str,
    role: str,
    vertices: tuple[tuple[float, float], ...],
    roles: tuple[str, ...],
    landmarks: tuple[tuple[str, int], ...] = (),
) -> PanelAnnotation:
    edges = tuple(
        _edge(panel_id, index, vertices, (index, (index + 1) % len(vertices)), edge_role)
        for index, edge_role in enumerate(roles)
    )
    points = tuple(
        Landmark(
            name=name,
            panel_id=panel_id,
            xy_cm=vertices[index],
            evidence="observed_source",
            confidence=1.0,
            vertex_index=index,
        )
        for name, index in landmarks
    )
    return PanelAnnotation(panel_id, role, vertices, edges, points, ())


def _record(
    sample_id: str,
    panels: tuple[PanelAnnotation, ...],
    *,
    upper_type: str | None,
    bottom: str | None,
) -> DraftingSemanticRecord:
    result = DraftingSemanticRecord(
        sample_id=sample_id,
        split="training",
        panels=panels,
        darts=(),
        measurements={},
        construction_steps=(),
        body_condition_cm={},
        program={"upper_type": upper_type, "design_values": {"meta.bottom": bottom}},
        provenance={"source": "synthetic-unit-test"},
    )
    result.validate()
    return result


def _rich_tshirt_record() -> DraftingSemanticRecord:
    front_vertices = (
        (0.0, 0.0),
        (6.0, 0.0),
        (6.0, 6.0),
        (4.0, 10.0),
        (3.0, 10.0),
        (2.0, 9.0),
        (0.0, 10.0),
    )
    front = _panel(
        "front",
        "front_bodice",
        front_vertices,
        ("hemline", "side_seam", "armhole", "shoulder", "neckline", "neckline", "center_front"),
        (("FNP", 6), ("SNP", 4), ("SP", 3)),
    )
    back_vertices = ((0.0, 0.0), (5.5, 0.0), (5.5, 6.0), (4.0, 10.0), (3.0, 10.0), (0.0, 9.5))
    back = _panel(
        "back",
        "back_bodice",
        back_vertices,
        ("hemline", "side_seam", "armhole", "shoulder", "neckline", "center_back"),
        (("BNP", 5), ("SNP", 4), ("SP", 3)),
    )
    sleeve_vertices = ((0.0, 0.0), (4.0, 0.0), (3.0, 5.0), (1.0, 5.0))
    sleeve = _panel(
        "sleeve",
        "sleeve",
        sleeve_vertices,
        ("sleeve_hem", "sleeve_underarm", "sleeve_head", "sleeve_underarm"),
    )
    return _record("rich_tshirt", (front, back, sleeve), upper_type="Shirt", bottom=None)


def _simple_skirt_record() -> DraftingSemanticRecord:
    vertices = ((0.0, 0.0), (10.0, 0.0), (10.0, 20.0), (0.0, 20.0))
    front = _panel(
        "skirt_front",
        "front_skirt",
        vertices,
        ("hemline", "side_seam", "waistline", "side_seam"),
    )
    back = _panel(
        "skirt_back",
        "back_skirt",
        vertices,
        ("hemline", "side_seam", "waistline", "side_seam"),
    )
    return _record("plain_skirt", (front, back), upper_type=None, bottom="Skirt2")


class BasicSemanticTargetTests(unittest.TestCase):
    def test_merged_line_path_emits_declared_geometry(self) -> None:
        target = semantic_target_from_drafting_record(_rich_tshirt_record())
        index = SEMANTIC_QUERY_INDEX["tshirt:path:front_neckline"]
        self.assertTrue(target.query_applicability[index])
        self.assertEqual(target.presence[index], 1.0)
        self.assertTrue(target.coordinate_mask[index, :8].all())
        names = SEMANTIC_QUERY_INVENTORY[index].coordinate_names
        values = dict(zip(names, target.coordinates[index, : len(names)]))
        self.assertAlmostEqual(float(values["start_u"]), 0.0, places=6)
        self.assertAlmostEqual(float(values["start_v"]), 1.0, places=6)
        self.assertAlmostEqual(float(values["end_u"]), 0.5, places=6)
        self.assertAlmostEqual(float(values["end_v"]), 1.0, places=6)
        self.assertGreater(float(values["arc_length_norm"]), 0.3)
        self.assertNotEqual(float(values["signed_depth_norm"]), 0.0)

    def test_unencoded_skirt_slit_and_closure_are_unknown(self) -> None:
        target = semantic_target_from_drafting_record(_simple_skirt_record())
        for key in ("skirt:path:slit", "skirt:path:closure", "skirt:landmark:closure_end"):
            index = SEMANTIC_QUERY_INDEX[key]
            self.assertFalse(target.query_applicability[index])
            self.assertEqual(target.presence[index], 0.0)
            self.assertFalse(target.coordinate_mask[index].any())
        panel = SEMANTIC_QUERY_INDEX["skirt:panel:skirt_panel"]
        self.assertTrue(target.query_applicability[panel])
        self.assertEqual(target.presence[panel], 1.0)

    def test_provisional_blocks_have_explicit_full_category_contract(self) -> None:
        deprecated = {
            "pants:path:dart_leg",
            "pants:landmark:dart_apex",
            "skirt:path:dart_leg",
            "skirt:path:closure",
            "skirt:landmark:center_waist",
            "skirt:landmark:side_waist",
            "skirt:landmark:side_hip",
            "skirt:landmark:hem_center",
            "skirt:landmark:hem_side",
            "skirt:landmark:dart_apex",
            "skirt:landmark:dart_leg_left",
            "skirt:landmark:dart_leg_right",
            "skirt:landmark:closure_end",
        }
        for category in ("tshirt", "pants", "skirt"):
            target = semantic_target_from_basic_block(build_basic_block(category), curve_samples=12)
            category_indices = [
                index for index, query in enumerate(SEMANTIC_QUERY_INVENTORY) if query.category == category
            ]
            for index in category_indices:
                query = SEMANTIC_QUERY_INVENTORY[index]
                self.assertEqual(
                    bool(target.query_applicability[index]), query.key not in deprecated
                )
            self.assertEqual(target.source, "provisional_common_basic_block")
            self.assertEqual(target.provenance_status, "PROVISIONAL_EXPERT_REVIEW")
            path_indices = [
                index
                for index, query in enumerate(SEMANTIC_QUERY_INVENTORY)
                if query.category == category and query.kind == "path" and target.presence[index] > 0.5
            ]
            for index in path_indices:
                self.assertEqual(int(target.coordinate_mask[index].sum()), 8)

    def test_reference_lines_are_normalized_fixed_queries(self) -> None:
        for category, query_keys in (
            (
                "tshirt",
                ("front_BL", "back_BL", "front_WL", "back_WL", "front_HL", "back_HL"),
            ),
            (
                "pants",
                ("front_WL", "back_WL", "front_HL", "back_HL", "front_KL", "back_KL", "front_CL", "back_CL", "front_GRAIN", "back_GRAIN"),
            ),
            (
                "skirt",
                ("front_WL", "back_WL", "front_HL", "back_HL", "front_GRAIN", "back_GRAIN"),
            ),
        ):
            target = semantic_target_from_basic_block(build_basic_block(category))
            for name in query_keys:
                index = SEMANTIC_QUERY_INDEX[f"{category}:reference_line:{name}"]
                self.assertTrue(target.query_applicability[index])
                self.assertEqual(float(target.presence[index]), 1.0)
                self.assertTrue(target.coordinate_mask[index, :4].all())
                self.assertFalse(target.coordinate_mask[index, 4:].any())
                self.assertTrue(np.isfinite(target.coordinates[index, :4]).all())

    def test_stored_but_ineligible_reference_line_is_present_and_masked(self) -> None:
        record = _rich_tshirt_record()
        panels = []
        for panel in record.panels:
            if panel.role not in {"front_bodice", "back_bodice"}:
                panels.append(panel)
                continue
            line = ReferenceLine(
                name="HL",
                panel_id=panel.id,
                points_cm=((0.0, 2.0), (4.0, 2.0)),
                evidence="synthetic_unvalidated",
                confidence=0.5,
                training_eligible=False,
            )
            panels.append(replace(panel, reference_lines=(line,)))
        target = semantic_target_from_drafting_record(
            replace(record, panels=tuple(panels))
        )
        for name in ("front_HL", "back_HL"):
            index = SEMANTIC_QUERY_INDEX[f"tshirt:reference_line:{name}"]
            self.assertEqual(float(target.presence[index]), 1.0)
            self.assertFalse(target.query_applicability[index])
            self.assertFalse(target.coordinate_mask[index].any())
            self.assertIn("not training eligible", target.evidence[index])

    def test_skirt_center_hip_landmark_lies_on_reference_hl(self) -> None:
        target = semantic_target_from_basic_block(build_basic_block("skirt"))
        for prefix in ("front", "back"):
            waist = SEMANTIC_QUERY_INDEX[f"skirt:landmark:{prefix}_center_waist"]
            hip = SEMANTIC_QUERY_INDEX[f"skirt:landmark:{prefix}_center_hip"]
            line = SEMANTIC_QUERY_INDEX[f"skirt:reference_line:{prefix}_HL"]
            hip_v = float(target.coordinates[hip, 1])
            self.assertNotEqual(hip_v, float(target.coordinates[waist, 1]))
            self.assertAlmostEqual(hip_v, float(target.coordinates[line, 1]))
            self.assertAlmostEqual(hip_v, float(target.coordinates[line, 3]))

    def test_role_specific_darts_replace_legacy_combined_queries(self) -> None:
        for category in ("pants", "skirt"):
            target = semantic_target_from_basic_block(build_basic_block(category))
            for prefix in ("front", "back"):
                for kind, suffix in (
                    ("path", "dart_leg"),
                    ("landmark", "dart_apex"),
                    ("landmark", "dart_leg_left"),
                    ("landmark", "dart_leg_right"),
                ):
                    index = SEMANTIC_QUERY_INDEX[f"{category}:{kind}:{prefix}_{suffix}"]
                    self.assertTrue(target.query_applicability[index])
                    self.assertEqual(float(target.presence[index]), 1.0)
                    self.assertTrue(target.coordinate_mask[index].any())
            legacy = SEMANTIC_QUERY_INDEX[f"{category}:path:dart_leg"]
            self.assertFalse(target.query_applicability[legacy])

    def test_pattern_document_presence_supports_explicit_unknown(self) -> None:
        block = build_basic_block("tshirt", sample_id="tri_state")
        document = block.to_pattern_document(curve_samples=12)
        annotations = dict(document.annotations)
        presence = dict(annotations["semantic_query_presence"])
        presence["neckband_attachment"] = "UNKNOWN"
        annotations["semantic_query_presence"] = presence
        updated = replace(document, annotations=annotations)
        target = semantic_target_from_pattern_document(
            updated,
            category="tshirt",
            source="unit_test",
            provenance_status="PROVISIONAL_EXPERT_REVIEW",
            source_y_axis_down=True,
        )
        index = SEMANTIC_QUERY_INDEX["tshirt:path:neckband_attachment"]
        self.assertFalse(target.query_applicability[index])
        self.assertIn("UNKNOWN", target.evidence[index])

    def test_y_up_and_y_down_sources_share_top_above_hem_convention(self) -> None:
        targets = (
            semantic_target_from_drafting_record(_rich_tshirt_record()),
            semantic_target_from_basic_block(build_basic_block("tshirt"), curve_samples=12),
        )
        fnp = SEMANTIC_QUERY_INDEX["tshirt:landmark:FNP"]
        hem = SEMANTIC_QUERY_INDEX["tshirt:path:front_hemline"]
        for target in targets:
            fnp_v = float(target.coordinates[fnp, 1])
            hem_v = float(np.nanmean(target.coordinates[hem, (1, 3)]))
            self.assertGreater(fnp_v, hem_v)

    def test_frozen_common_filters_match_exact_topologies(self) -> None:
        rectangle = ((0.0, 0.0), (4.0, 0.0), (4.0, 8.0), (0.0, 8.0))

        def panel(panel_id: str, role: str) -> PanelAnnotation:
            return _panel(panel_id, role, rectangle, ("hemline", "side_seam", "waistline", "other"))

        tshirt = _record(
            "basic_top",
            (
                panel("f0", "front_bodice"),
                panel("f1", "front_bodice"),
                panel("b0", "back_bodice"),
                panel("b1", "back_bodice"),
                *(panel(f"s{index}", "sleeve") for index in range(4)),
            ),
            upper_type="Shirt",
            bottom=None,
        )
        pants = _record(
            "basic_pants",
            tuple(panel(f"pf{index}", "front_pants") for index in range(2))
            + tuple(panel(f"pb{index}", "back_pants") for index in range(2)),
            upper_type=None,
            bottom="Pants",
        )
        skirt = _simple_skirt_record()
        self.assertTrue(is_common_basic_tshirt(tshirt))
        self.assertTrue(is_common_basic_pants(pants))
        self.assertTrue(is_common_basic_skirt(skirt))
        self.assertEqual(common_basic_category(tshirt), "tshirt")
        self.assertEqual(common_basic_category(pants), "pants")
        self.assertEqual(common_basic_category(skirt), "skirt")
        self.assertEqual(
            tuple(record.sample_id for record in filter_common_basic_records((tshirt, pants, skirt), "pants")),
            ("basic_pants",),
        )
        extra = replace(tshirt, panels=tshirt.panels + (panel("collar", "collar"),))
        self.assertFalse(is_common_basic_tshirt(extra))

    def test_batch_fields_match_training_contract(self) -> None:
        targets = (
            semantic_target_from_drafting_record(_rich_tshirt_record()),
            semantic_target_from_basic_block(build_basic_block("pants"), curve_samples=8),
        )
        batch = stack_semantic_targets(targets)
        count = len(SEMANTIC_QUERY_INVENTORY)
        self.assertEqual(batch["category_ids"].shape, (2,))
        self.assertEqual(batch["query_mask"].shape, (2, count))
        self.assertEqual(batch["presence_targets"].shape, (2, count))
        self.assertEqual(batch["coordinate_targets"].shape, (2, count, MAX_COORDINATE_DIM))
        self.assertEqual(batch["coordinate_mask"].shape, (2, count, MAX_COORDINATE_DIM))
        self.assertEqual(batch["category_ids"].dtype, np.int64)
        self.assertEqual(batch["coordinate_targets"].dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
