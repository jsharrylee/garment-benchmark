from __future__ import annotations

import json
from pathlib import Path
import unittest

from benchmark.drafting_semantics.lower_body_semantics import (
    PANTS_FRONT_LANDMARKS,
    PANTS_REFERENCE_LINES,
    SKIRT_LANDMARKS,
    SKIRT_REFERENCE_LINES,
    extract_lower_body_semantics,
)
from benchmark.drafting_semantics.schema import (
    Dart,
    DraftingSemanticRecord,
    EdgeAnnotation,
    Landmark,
    PanelAnnotation,
)


def _edge(
    panel_id: str,
    vertices: tuple[tuple[float, float], ...],
    index: int,
    start: int,
    end: int,
    role: str,
    *,
    edge_id: str | None = None,
    stitched: bool | None = None,
) -> EdgeAnnotation:
    first, second = vertices[start], vertices[end]
    return EdgeAnnotation(
        id=edge_id or f"{panel_id}.edge_{index}",
        index=index,
        endpoints=(start, end),
        start_cm=first,
        end_cm=second,
        curvature_type="line",
        role=role,
        stitched=(role not in {"hemline"}) if stitched is None else stitched,
        self_stitched=role == "dart_leg",
        length_cm=((second[0] - first[0]) ** 2 + (second[1] - first[1]) ** 2) ** 0.5,
        evidence="observed_source",
        confidence=1.0,
    )


def _pants_panel() -> tuple[PanelAnnotation, Dart]:
    panel_id = "front_pants"
    # Clockwise boundary: side waist -> dart -> centre waist -> crotch ->
    # inseam hem -> side hem.  The centre/crotch side deliberately protrudes.
    vertices = (
        (0.0, 100.0),
        (10.0, 100.0),
        (11.0, 88.0),
        (12.0, 100.0),
        (30.0, 100.0),
        (36.0, 70.0),
        (25.0, 0.0),
        (5.0, 0.0),
    )
    roles = (
        (0, 1, "waistline"),
        (1, 2, "dart_leg"),
        (2, 3, "dart_leg"),
        (3, 4, "waistline"),
        (4, 5, "crotch_curve"),
        (5, 6, "inseam"),
        (6, 7, "hemline"),
        (7, 0, "outseam"),
    )
    edges = tuple(
        _edge(panel_id, vertices, index, start, end, role)
        for index, (start, end, role) in enumerate(roles)
    )
    panel = PanelAnnotation(panel_id, "front_pants", vertices, edges)
    dart = Dart(
        panel_id,
        "waist_dart",
        (f"{panel_id}.edge_1", f"{panel_id}.edge_2"),
        vertices[2],
        (vertices[1], vertices[3]),
        2.0,
        12.0,
        "observed_source",
        1.0,
    )
    return panel, dart


def _skirt_panel(*, slit: bool = False, role: str = "front_skirt") -> tuple[PanelAnnotation, Dart]:
    panel_id = "skirt_a"
    vertices = (
        (-20.0, 50.0),
        (-5.0, 50.0),
        (-4.0, 40.0),
        (-3.0, 50.0),
        (20.0, 50.0),
        (25.0, 0.0),
        (-25.0, 0.0),
    )
    roles = (
        (0, 1, "waistline"),
        (1, 2, "dart_leg"),
        (2, 3, "dart_leg"),
        (3, 4, "waistline"),
        (4, 5, "side_seam"),
        (5, 6, "hemline"),
        (6, 0, "side_seam"),
    )
    edges = tuple(
        _edge(panel_id, vertices, index, start, end, edge_role)
        for index, (start, end, edge_role) in enumerate(roles)
    )
    landmarks = (
        (Landmark("SLIT_TOP", panel_id, (0.0, 12.0), "observed_source", 1.0),)
        if slit
        else ()
    )
    panel = PanelAnnotation(panel_id, role, vertices, edges, landmarks=landmarks)
    dart = Dart(
        panel_id,
        "waist_dart",
        (f"{panel_id}.edge_1", f"{panel_id}.edge_2"),
        vertices[2],
        (vertices[1], vertices[3]),
        2.0,
        10.0,
    )
    return panel, dart


def _skirt_with_geometric_slit() -> PanelAnnotation:
    panel_id = "skirt_vent"
    vertices = (
        (-20.0, 50.0),
        (20.0, 50.0),
        (25.0, 0.0),
        (2.0, 0.0),
        (0.0, 15.0),
        (-2.0, 0.0),
        (-25.0, 0.0),
    )
    definitions = (
        (0, 1, "waistline", True),
        (1, 2, "side_seam", True),
        (2, 3, "hemline", False),
        (3, 4, "other", False),
        (4, 5, "other", False),
        (5, 6, "hemline", False),
        (6, 0, "side_seam", True),
    )
    edges = tuple(
        _edge(panel_id, vertices, index, start, end, role, stitched=stitched)
        for index, (start, end, role, stitched) in enumerate(definitions)
    )
    return PanelAnnotation(panel_id, "back_skirt", vertices, edges)


def _record(
    *panels: PanelAnnotation,
    darts: tuple[Dart, ...] = (),
    hip_depth: float | None = 20.0,
    production: dict | None = None,
) -> DraftingSemanticRecord:
    measurements = {} if hip_depth is None else {"garmentcode_waist_to_hip_cm": hip_depth}
    return DraftingSemanticRecord(
        sample_id="synthetic_lower_body",
        split="test",
        panels=tuple(panels),
        darts=darts,
        measurements=measurements,
        construction_steps=(),
        body_condition_cm={},
        program={},
        provenance={"dataset": "synthetic unit test"},
        production_annotations=production
        or {
            "source_notches": {"available": False, "reason": "not encoded"},
            "source_zippers": {"available": False, "reason": "not encoded"},
        },
    )


class LowerBodySemanticTests(unittest.TestCase):
    def test_v2_manifest_matches_lower_body_ontology_and_knee_evidence_boundary(self):
        path = (
            Path(__file__).resolve().parents[2]
            / "data"
            / "manifests"
            / "lower_body_semantic_targets.json"
        )
        manifest = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["schema_version"],
            "lower-body-semantic-target-manifest/v2",
        )
        self.assertEqual(manifest["ontology_version"], "lower-body-landmarks-1.1")
        self.assertEqual(
            manifest["source_records_sha256"],
            "2f19f80e39a11f41f8b5ddcb5486d8f7062880fa8ae669c3a4fb853cb1f316a9",
        )
        self.assertEqual(manifest["record_count_audited"], 2937)
        self.assertEqual(manifest["pants"]["panel_count"], 1844)
        self.assertEqual(
            manifest["pants"]["reference_line_inventory"],
            list(PANTS_REFERENCE_LINES),
        )
        self.assertEqual(
            manifest["pants"]["reference_line_counts"]["CL"],
            {"available": 1844, "training_eligible": 1844},
        )
        self.assertEqual(
            manifest["pants"]["reference_line_counts"]["KNEE_LINE"],
            {"available": 1844, "training_eligible": 0},
        )
        raw_knee = manifest["training_policy"]["base_lower_body_extractor"][
            "KNEE_LINE"
        ]
        self.assertEqual(raw_knee["evidence"], "synthetic_unvalidated")
        self.assertFalse(raw_knee["training_eligible"])
        teacher = manifest["training_policy"]["separate_teacher_adapter"]
        self.assertEqual(
            teacher["KNEE_LINE_to_KL_status"],
            "PROVISIONAL_CONVENTIONAL_HALF_LEG_KNEE_LINE",
        )
        self.assertEqual(manifest["skirt"]["panel_count"], 7707)
        self.assertEqual(
            manifest["skirt"]["reference_line_counts"],
            {
                "WL": {"available": 4793, "training_eligible": 4793},
                "HL": {"available": 3435, "training_eligible": 3435},
                "GRAIN": {"available": 4793, "training_eligible": 4793},
            },
        )

    def test_pants_landmarks_lines_and_dart_are_geometry_evidenced(self):
        panel, dart = _pants_panel()
        output = extract_lower_body_semantics(_record(panel, darts=(dart,)))
        self.assertEqual(len(output.panels), 1)
        semantic = output.panels[0]

        self.assertEqual(tuple(item.name for item in semantic.landmarks), PANTS_FRONT_LANDMARKS)
        self.assertEqual(tuple(item.name for item in semantic.reference_lines), PANTS_REFERENCE_LINES)
        self.assertEqual(semantic.landmark("CF_WAIST").xy_cm, (30.0, 100.0))
        self.assertEqual(semantic.landmark("SIDE_WAIST").xy_cm, (0.0, 100.0))
        self.assertEqual(semantic.landmark("CROTCH_POINT").xy_cm, (36.0, 70.0))
        self.assertAlmostEqual(semantic.landmark("CF_HIP").xy_cm[1], 80.0)
        self.assertAlmostEqual(semantic.landmark("SIDE_HIP").xy_cm[1], 80.0)
        self.assertTrue(semantic.reference_line("WL").intersects_panel)
        self.assertTrue(semantic.reference_line("HL").training_eligible)

        # A conventional half-leg knee is exposed for review, but not promoted
        # to observed ground truth.
        self.assertEqual(semantic.reference_line("KNEE_LINE").evidence, "synthetic_unvalidated")
        self.assertFalse(semantic.reference_line("KNEE_LINE").training_eligible)
        self.assertFalse(semantic.landmark("KNEE_SIDE").training_eligible)

        self.assertEqual(len(semantic.darts), 1)
        self.assertEqual(semantic.darts[0].apex_cm, (11.0, 88.0))
        self.assertEqual(semantic.darts[0].leg_a_cm, (10.0, 100.0))
        self.assertEqual(semantic.darts[0].leg_b_cm, (12.0, 100.0))
        self.assertEqual(semantic.darts[0].intake_cm, 2.0)
        self.assertTrue(semantic.feature("dart").available)

    def test_missing_hip_measurement_is_explicit_not_fabricated(self):
        panel, dart = _pants_panel()
        semantic = extract_lower_body_semantics(_record(panel, darts=(dart,), hip_depth=None)).panels[0]
        for name in ("CF_HIP", "SIDE_HIP"):
            point = semantic.landmark(name)
            self.assertFalse(point.available)
            self.assertIsNone(point.xy_cm)
            self.assertEqual(point.evidence, "unavailable")
            self.assertIn("waist-to-hip", point.reason)
        self.assertFalse(semantic.reference_line("HL").available)

    def test_pants_centre_side_fallback_is_translation_invariant(self):
        panel, dart = _pants_panel()
        translated_vertices = tuple((x - 200.0, y) for x, y in panel.vertices_cm)
        translated_edges = []
        for edge in panel.edges:
            role = edge.role if edge.role in {"waistline", "hemline", "dart_leg"} else "other"
            translated_edges.append(
                _edge(
                    panel.id,
                    translated_vertices,
                    edge.index,
                    edge.endpoints[0],
                    edge.endpoints[1],
                    role,
                    stitched=edge.stitched,
                )
            )
        translated = PanelAnnotation(panel.id, panel.role, translated_vertices, tuple(translated_edges))
        translated_dart = Dart(
            dart.panel_id,
            dart.kind,
            dart.leg_edge_ids,
            (dart.apex_cm[0] - 200.0, dart.apex_cm[1]),
            tuple((point[0] - 200.0, point[1]) for point in dart.base_cm),
            dart.intake_cm,
            dart.depth_cm,
        )
        semantic = extract_lower_body_semantics(
            _record(translated, darts=(translated_dart,))
        ).panels[0]
        self.assertEqual(semantic.landmark("CF_WAIST").xy_cm, (-170.0, 100.0))
        self.assertEqual(semantic.landmark("SIDE_WAIST").xy_cm, (-200.0, 100.0))
        self.assertEqual(semantic.landmark("CF_WAIST").confidence, 0.72)

    def test_skirt_role_is_exchangeable_and_front_back_name_is_not_a_target(self):
        # No dart or visible closure cue: source front/back names are retained
        # only as metadata and canonical targets are identical.
        front, _ = _skirt_panel(role="front_skirt")
        back, _ = _skirt_panel(role="back_skirt")
        back = PanelAnnotation("skirt_b", back.role, back.vertices_cm, tuple(
            EdgeAnnotation(**{**edge.__dict__, "id": edge.id.replace("skirt_a", "skirt_b")})
            for edge in back.edges
        ))
        output = extract_lower_body_semantics(_record(front, back, darts=()))
        self.assertEqual([item.canonical_role for item in output.panels], ["skirt_panel", "skirt_panel"])
        self.assertTrue(all(item.front_back_exchangeable for item in output.panels))
        self.assertEqual(tuple(item.name for item in output.panels[0].landmarks), SKIRT_LANDMARKS)
        self.assertEqual(tuple(item.name for item in output.panels[0].reference_lines), SKIRT_REFERENCE_LINES)
        self.assertEqual(output.panels[0].landmark("WAIST_LEFT").xy_cm, (-20.0, 50.0))
        self.assertEqual(output.panels[0].landmark("HEM_RIGHT").xy_cm, (25.0, 0.0))

    def test_skirt_dart_and_explicit_slit_are_preserved_but_not_invented(self):
        plain, dart = _skirt_panel(slit=False)
        plain_semantic = extract_lower_body_semantics(_record(plain, darts=(dart,))).panels[0]
        self.assertFalse(plain_semantic.landmark("SLIT_TOP").available)
        self.assertFalse(plain_semantic.feature("slit").available)
        self.assertFalse(plain_semantic.feature("zipper").available)
        self.assertFalse(plain_semantic.feature("notches").available)
        self.assertFalse(plain_semantic.front_back_exchangeable)  # dart is a visible cue

        slit, dart = _skirt_panel(slit=True)
        slit_semantic = extract_lower_body_semantics(_record(slit, darts=(dart,))).panels[0]
        self.assertTrue(slit_semantic.landmark("SLIT_TOP").available)
        self.assertEqual(slit_semantic.landmark("SLIT_TOP").xy_cm, (0.0, 12.0))
        self.assertTrue(slit_semantic.feature("slit").available)
        # Only the observed top is available; hem endpoints are not guessed.
        self.assertFalse(slit_semantic.landmark("SLIT_HEM_LEFT").available)
        self.assertFalse(slit_semantic.landmark("SLIT_HEM_RIGHT").available)

    def test_observable_unstitched_hem_vent_yields_all_slit_landmarks(self):
        panel = _skirt_with_geometric_slit()
        semantic = extract_lower_body_semantics(_record(panel, darts=())).panels[0]
        self.assertEqual(semantic.landmark("SLIT_TOP").xy_cm, (0.0, 15.0))
        self.assertEqual(semantic.landmark("SLIT_HEM_LEFT").xy_cm, (-2.0, 0.0))
        self.assertEqual(semantic.landmark("SLIT_HEM_RIGHT").xy_cm, (2.0, 0.0))
        self.assertEqual(semantic.landmark("SLIT_TOP").evidence, "derived_topology")
        self.assertTrue(semantic.feature("slit").available)
        self.assertFalse(semantic.front_back_exchangeable)

    def test_extraction_is_deterministic_and_serializable(self):
        panel, dart = _pants_panel()
        record = _record(panel, darts=(dart,))
        first = extract_lower_body_semantics(record)
        second = extract_lower_body_semantics(record)
        self.assertEqual(first, second)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.to_dict()["ontology_version"], "lower-body-landmarks-1.1")


if __name__ == "__main__":
    unittest.main()
