from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from benchmark.gcdv2_exact.pattern_dsl import (
    CloseCommand,
    PatternDSLError,
    PatternProgram,
    compile_garment_record,
)
from benchmark.gcdv2_exact.pattern_dsl_svg import (
    SVG_NAMESPACE,
    SvgExportOptions,
    compile_pattern_svg,
    write_pattern_svg,
)
from benchmark.pattern_pipeline.schema import PatternDocument


def _direction(first: tuple[float, float], second: tuple[float, float]) -> float:
    return math.degrees(math.atan2(second[1] - first[1], second[0] - first[0]))


def _formal_graph(panel_uid: str, surface: str) -> dict:
    points = ((0.0, 0.0), (4.0, 0.0), (5.0, 2.0), (4.0, 5.0), (0.0, 5.0), (-1.0, 2.0))
    primitive_payloads = (
        ("line", {}),
        ("quadratic_bezier", {"relative_controls_chord_frame": [[0.48, 0.18]]}),
        (
            "cubic_bezier",
            {"relative_controls_chord_frame": [[0.2, 0.22], [0.8, 0.18]]},
        ),
        (
            "circular_arc",
            {"radius_cm": 2.8, "large_arc": False, "sweep_y_up": True},
        ),
        ("line", {}),
        ("line", {}),
    )
    curves = []
    relations = []
    for index, (primitive, parameters) in enumerate(primitive_payloads):
        start = points[index]
        end = points[(index + 1) % len(points)]
        direction = _direction(start, end)
        curves.append(
            {
                "edge_id": f"e{index}",
                "source_edge_id": f"source.e{index}",
                "source_edge_index": index,
                "start_point_id": f"p{index}",
                "end_point_id": f"p{(index + 1) % len(points)}",
                "primitive": primitive,
                "parameters": parameters,
                "length_cm": math.dist(start, end) * (1.08 if primitive != "line" else 1.0),
                "chord_direction_deg_y_up": direction,
                "start_tangent_deg_y_up": direction,
                "end_tangent_deg_y_up": direction,
            }
        )
        following = (index + 1) % len(points)
        relations.extend(
            (
                {"predicate": "NEXT", "arguments": [f"e{index}", f"e{following}"]},
                {
                    "predicate": "SHARED_ENDPOINT",
                    "arguments": [f"e{index}", f"e{following}", f"p{following}"],
                },
            )
        )
    return {
        "panel_uid": panel_uid,
        "source_panel_id": panel_uid.split(":", 1)[-1],
        "garment_category": "top",
        "weak_role": {"part": "bodice", "surface": surface, "side": "full"},
        "points": [
            {"point_id": f"p{index}", "xy_cm": list(point)}
            for index, point in enumerate(points)
        ],
        "curves": curves,
        "relations": relations,
    }


def _program() -> PatternProgram:
    front = _formal_graph("svg-test:front", "front")
    back = _formal_graph("svg-test:back", "back")
    roles = {
        "e0": "center_front",
        "e1": "neckline",
        "e2": "shoulder",
        "e3": "armhole",
        "e4": "side_seam",
        "e5": "waistline",
    }
    back_roles = dict(roles)
    back_roles["e0"] = "center_back"
    return compile_garment_record(
        {
            "sample_id": "svg-test",
            "garment_category": "top",
            "panels": [{"formal_graph": front}, {"formal_graph": back}],
            "stitch_constraints": [
                {
                    "constraint_id": "side_pair",
                    "sides": [
                        {"panel_uid": "svg-test:front", "edge_id": "e4", "length_cm": 3.0},
                        {"panel_uid": "svg-test:back", "edge_id": "e4", "length_cm": 3.0},
                    ],
                }
            ],
        },
        edge_roles={"svg-test:front": roles, "svg-test:back": back_roles},
    )


def _metadata(root: ET.Element) -> dict:
    node = root.find(f"{{{SVG_NAMESPACE}}}metadata")
    if node is None or node.text is None:
        raise AssertionError("SVG metadata is missing")
    return json.loads(node.text)


class PatternDSLSvgTests(unittest.TestCase):
    def test_native_commands_metadata_and_panel_separation(self) -> None:
        program = _program()
        options = SvgExportOptions(
            include_semantic_facts=True,
            include_provenance=True,
        )
        svg = compile_pattern_svg(program, options=options)
        self.assertEqual(svg, compile_pattern_svg(program, options=options))
        self.assertNotIn("data:image", svg)
        root = ET.fromstring(svg)
        boundaries = root.findall(f".//{{{SVG_NAMESPACE}}}path[@class='panel-boundary']")
        self.assertEqual(len(boundaries), 2)
        for boundary in boundaries:
            path = boundary.attrib["d"]
            for opcode in ("M ", "L ", "Q ", "C ", "A ", "Z"):
                self.assertIn(opcode, path)
        metadata = _metadata(root)
        self.assertTrue(metadata["verification"]["valid"])
        self.assertFalse(metadata["geometry_contract"]["raster_or_contour_dependency"])
        self.assertEqual(metadata["svg_command_counts"]["M"], 2)
        self.assertEqual(metadata["svg_command_counts"]["Z"], 2)
        self.assertGreater(len(metadata["proof_facts"]), 0)
        boxes = [value["svg_bbox_cm_y_down"] for value in metadata["layout"]["panels"]]
        first, second = boxes
        overlaps = not (
            first[2] <= second[0]
            or second[2] <= first[0]
            or first[3] <= second[1]
            or second[3] <= first[1]
        )
        self.assertFalse(overlaps)

    def test_default_is_geometry_only_and_leakage_safe(self) -> None:
        svg = compile_pattern_svg(_program())
        for forbidden in (
            "svg-test",
            "source_panel_id",
            "source_edge_index",
            "center_front",
            "center_back",
            "neckline",
            "side_pair",
            "bodice",
            "waistline",
        ):
            self.assertNotIn(forbidden, svg)
        root = ET.fromstring(svg)
        metadata = _metadata(root)
        self.assertTrue(metadata["geometry_contract"]["default_export_is_label_free"])
        self.assertNotIn("proof_facts", metadata)
        self.assertNotIn("semantic_verification", metadata)
        self.assertNotIn("provenance", metadata)
        self.assertEqual(
            [value["panel_id"] for value in metadata["layout"]["panels"]],
            ["panel_000", "panel_001"],
        )
        boundaries = root.findall(f".//{{{SVG_NAMESPACE}}}path[@class='panel-boundary']")
        self.assertEqual({value.attrib["fill"] for value in boundaries}, {"#d9dee8"})

    def test_optional_overlays_keep_semantics_as_separate_group(self) -> None:
        clean = compile_pattern_svg(_program())
        reviewed = compile_pattern_svg(
            _program(),
            options=SvgExportOptions(
                include_overlays=True,
                include_semantic_facts=True,
                include_provenance=True,
            ),
        )
        self.assertNotIn('id="semantic-overlays"', clean)
        self.assertIn('id="semantic-overlays"', reviewed)
        self.assertIn("center_front", reviewed)
        self.assertIn("FNP", reviewed)
        self.assertIn("SNP", reviewed)
        self.assertIn("SP", reviewed)
        self.assertIn("SEWN_TO side_pair", reviewed)

    def test_dsl_backed_document_compiles_identically(self) -> None:
        program = _program()
        document = program.to_pattern_document()
        options = SvgExportOptions(include_overlays=True, decimals=5)
        self.assertEqual(
            compile_pattern_svg(program, options=options),
            compile_pattern_svg(document, options=options),
        )

    def test_rejects_invalid_program_and_unprovenanced_document(self) -> None:
        program = _program()
        invalid = PatternProgram(
            tuple(value for value in program.commands if not isinstance(value, CloseCommand))
        )
        with self.assertRaises(PatternDSLError):
            compile_pattern_svg(invalid)
        document = PatternDocument("plain", "test", (), ())
        with self.assertRaises(PatternDSLError):
            compile_pattern_svg(document)

    def test_writer_is_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "a.svg"
            second = Path(directory) / "b.svg"
            write_pattern_svg(_program(), first)
            write_pattern_svg(_program(), second)
            self.assertEqual(first.read_bytes(), second.read_bytes())


if __name__ == "__main__":
    unittest.main()
