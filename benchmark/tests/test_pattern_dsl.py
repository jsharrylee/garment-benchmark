from __future__ import annotations

from dataclasses import asdict, replace
import math
import unittest

from benchmark.gcdv2_exact.pattern_dsl import (
    CurveCommand,
    LandmarkCommand,
    NextCommand,
    PatternDSLParseError,
    PatternProgram,
    SewnToCommand,
    compile_formal_graph,
    compile_garment_record,
    parse_pattern_dsl,
    verify_pattern_dsl,
)
from benchmark.pattern_pipeline.validation import validate_pattern


def _angle(first: tuple[float, float], second: tuple[float, float]) -> float:
    return math.degrees(math.atan2(second[1] - first[1], second[0] - first[0]))


def _graph(
    panel_uid: str = "sample:front",
    *,
    surface: str = "front",
    transform=lambda point: point,
    length_scale: float = 1.0,
    rotation_deg: float = 0.0,
) -> dict:
    source_points = (
        (0.0, 0.0),
        (0.0, 4.0),
        (1.0, 5.0),
        (3.0, 5.0),
        (4.0, 4.0),
        (4.0, 0.0),
        (3.0, -1.0),
        (0.0, -1.0),
    )
    points = tuple(transform(value) for value in source_points)
    primitives = (
        ("line", {}),
        ("quadratic_bezier", {"relative_controls_chord_frame": [[0.45, 0.15]]}),
        ("line", {}),
        (
            "cubic_bezier",
            {"relative_controls_chord_frame": [[0.2, 0.25], [0.8, 0.25]]},
        ),
        ("line", {}),
        ("line", {}),
        (
            "circular_arc",
            {
                "radius_cm": 2.0 * length_scale,
                "large_arc": False,
                "sweep_y_up": True,
            },
        ),
        ("line", {}),
    )
    curves = []
    for index, (primitive, parameters) in enumerate(primitives):
        start, end = points[index], points[(index + 1) % len(points)]
        direction = _angle(start, end)
        curves.append(
            {
                "edge_id": f"e{index}",
                "source_edge_id": f"source.edge_{index}",
                "source_edge_index": index,
                "start_point_id": f"p{index}",
                "end_point_id": f"p{(index + 1) % len(points)}",
                "primitive": primitive,
                "parameters": parameters,
                "length_cm": math.dist(start, end) * (1.12 if primitive != "line" else 1.0),
                "chord_direction_deg_y_up": direction,
                "start_tangent_deg_y_up": direction + rotation_deg * 0.0,
                "end_tangent_deg_y_up": direction + rotation_deg * 0.0,
            }
        )
    relations = []
    for index in range(len(curves)):
        following = (index + 1) % len(curves)
        relations.extend(
            (
                {"predicate": "NEXT", "arguments": [f"e{index}", f"e{following}"]},
                {
                    "predicate": "SHARED_ENDPOINT",
                    "arguments": [f"e{index}", f"e{following}", f"p{following}"],
                },
            )
        )
    source_panel = panel_uid.split(":", 1)[-1]
    return {
        "schema_version": "synthetic-formal-graph/v1",
        "panel_uid": panel_uid,
        "source_panel_id": source_panel,
        "garment_category": "top",
        "weak_role": {
            "part": "bodice",
            "surface": surface,
            "side": "unspecified",
        },
        "points": [
            {"point_id": f"p{index}", "xy_cm": list(point)}
            for index, point in enumerate(points)
        ],
        "curves": curves,
        "relations": relations,
    }


def _roles(center: str) -> dict[str, str]:
    return {
        "e0": center,
        "e1": "neckline",
        "e2": "shoulder",
        "e3": "armhole",
        "e4": "side_seam",
        "e5": "waistline",
        "e6": "other",
        "e7": "other",
    }


class PatternDSLTests(unittest.TestCase):
    def test_compile_uses_all_curve_ops_and_coordinate_free_invariants(self) -> None:
        baseline = compile_formal_graph(_graph(), edge_roles=_roles("center_front"))

        angle = math.radians(37.0)
        scale = 2.75

        def transformed(point):
            x, y = point[0] * scale, point[1] * scale
            return (
                x * math.cos(angle) - y * math.sin(angle) + 91.0,
                x * math.sin(angle) + y * math.cos(angle) - 43.0,
            )

        moved = compile_formal_graph(
            _graph(
                transform=transformed,
                length_scale=scale,
                rotation_deg=37.0,
            ),
            edge_roles=_roles("center_front"),
        )
        baseline_edges = [value for value in baseline.commands if isinstance(value, CurveCommand)]
        moved_edges = [value for value in moved.commands if isinstance(value, CurveCommand)]
        self.assertEqual({value.op for value in baseline_edges}, {"L", "Q", "C", "A"})
        for first, second in zip(baseline_edges, moved_edges, strict=True):
            self.assertEqual(first.op, second.op)
            self.assertAlmostEqual(first.length_ratio, second.length_ratio, places=10)
            self.assertAlmostEqual(first.chord_ratio, second.chord_ratio, places=10)
            self.assertAlmostEqual(first.turn_sin, second.turn_sin, places=10)
            self.assertAlmostEqual(first.turn_cos, second.turn_cos, places=10)
            self.assertEqual(first.controls_chord_frame, second.controls_chord_frame)
            if first.op == "A":
                self.assertAlmostEqual(
                    first.arc_radius_over_chord,
                    second.arc_radius_over_chord,
                    places=10,
                )
        encoded = baseline.serialize()
        self.assertNotIn("xy_cm", encoded)
        self.assertNotIn("image", encoded.lower())
        self.assertIn("Q ", encoded)
        self.assertIn("C ", encoded)
        self.assertIn("A ", encoded)

    def test_serializer_parser_round_trip_and_semantic_landmarks(self) -> None:
        program = compile_formal_graph(_graph(), edge_roles=_roles("center_front"))
        restored = parse_pattern_dsl(program.serialize())
        self.assertEqual(restored, program)
        report = restored.verify()
        self.assertTrue(report.valid, report.to_dict())
        landmarks = {
            (value.base_name, value.point_id)
            for value in report.derived_landmarks
        }
        self.assertEqual(
            landmarks,
            {("FNP", "p1"), ("SNP", "p2"), ("SP", "p3")},
        )
        self.assertTrue(
            any(isinstance(value, LandmarkCommand) for value in restored.commands)
        )
        for token in (
            "PANEL ",
            "M ",
            "L ",
            "Z ",
            "ROLE ",
            "NEXT ",
            "SHARED_ENDPOINT ",
            "LANDMARK ",
        ):
            self.assertIn(token, restored.serialize())

    def test_garment_compile_adds_seams_and_front_back_landmarks(self) -> None:
        front = _graph("sample:front", surface="front")
        back = _graph("sample:back", surface="back")
        record = {
            "sample_id": "sample",
            "garment_category": "top",
            "panels": [
                {"formal_graph": front},
                {"formal_graph": back},
            ],
            "stitch_constraints": [
                {
                    "constraint_id": "side",
                    "sides": [
                        {"panel_uid": "sample:front", "edge_id": "e4", "length_cm": 4.0},
                        {"panel_uid": "sample:back", "edge_id": "e4", "length_cm": 4.2},
                    ],
                    "source_annotations": ["right_wrong"],
                }
            ],
        }
        program = compile_garment_record(
            record,
            edge_roles={
                "sample:front": _roles("center_front"),
                "sample:back": _roles("center_back"),
            },
        )
        report = verify_pattern_dsl(program)
        self.assertTrue(report.valid, report.to_dict())
        self.assertEqual(report.metrics["panel_count"], 2)
        self.assertEqual(report.metrics["seam_count"], 1)
        names = {value.base_name for value in report.derived_landmarks}
        self.assertEqual(names, {"FNP", "BNP", "SNP", "SP"})
        self.assertIn("SEWN_TO ", program.serialize())

    def test_verifier_rejects_bad_next_seam_and_landmark_relations(self) -> None:
        valid = compile_formal_graph(_graph(), edge_roles=_roles("center_front"))
        commands = list(valid.commands)
        next_index = next(
            index
            for index, value in enumerate(commands)
            if isinstance(value, NextCommand) and value.first_edge_id == "e0"
        )
        commands[next_index] = replace(commands[next_index], second_edge_id="e3")
        commands.extend(
            (
                SewnToCommand("bad_seam", "sample:front", "missing", "sample:front", "e4", 1.0),
                LandmarkCommand("sample:front", "FNP", "p4", derived=False),
            )
        )
        report = PatternProgram(tuple(commands)).verify()
        self.assertFalse(report.valid)
        codes = {value.code for value in report.issues}
        self.assertIn("NEXT_ENDPOINT_MISMATCH", codes)
        self.assertIn("UNKNOWN_SEAM_EDGE", codes)
        self.assertIn("SEMANTIC_JUNCTION_MISMATCH", codes)

    def test_materialization_keeps_analytic_payload_and_valid_structure(self) -> None:
        program = compile_formal_graph(_graph(), edge_roles=_roles("center_front"))
        document = program.to_pattern_document(samples_per_curve=41)
        report = validate_pattern(document)
        self.assertTrue(report.accepted, report.to_dict())
        analytic = document.annotations["analytic_edge_geometry"]
        self.assertEqual(analytic["sample:front/e1"]["curve"]["type"], "quadratic_bezier")
        self.assertEqual(len(analytic["sample:front/e3"]["curve"]["controls_cm"]), 2)
        self.assertEqual(analytic["sample:front/e6"]["curve"]["type"], "circular_arc")
        self.assertIn("radius_cm", analytic["sample:front/e6"]["curve"]["arc"])
        self.assertFalse(document.provenance["absolute_source_coordinates_serialized"])
        self.assertIn("FNP", document.annotations["semantic_landmarks"])

    def test_parser_rejects_absolute_coordinate_payload(self) -> None:
        with self.assertRaises(PatternDSLParseError):
            parse_pattern_dsl(
                '# gcd-pattern-dsl/v1\n'
                'M {"panel_id":"p","point_id":"v0","xy_cm":[1,2]}\n'
            )


if __name__ == "__main__":
    unittest.main()
