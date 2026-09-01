from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import inspect
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import yaml

from benchmark.drafting_semantics.freesewing_teagan import teagan_json_to_trace
from benchmark.drafting_semantics.runtime_trace import RuntimeTraceRecorder
from benchmark.drafting_semantics.tshirt_corpus import (
    build_body_cases,
    build_design_cases,
    build_sample_plan,
    plan_digest,
    split_counts,
)
from benchmark.drafting_semantics.tshirt_learning import panel_example
from benchmark.drafting_semantics.tshirt_garmentcode import _trace_operations
from benchmark.drafting_semantics.tshirt_schema import (
    ConstructionOperation,
    CurveGeometry,
    DartTrace,
    TShirtTraceRecord,
    TracedEdge,
    TracedPanel,
    TracedPoint,
)


EXPECTED_SPLIT_COUNTS = {
    "iid_test": 128,
    "iid_validation": 128,
    "test_body": 256,
    "test_design": 256,
    "test_double": 64,
    "train": 768,
    "unseen_body": 256,
    "unseen_design": 256,
    "unseen_double": 64,
    "validation_body": 128,
    "validation_design": 256,
    "validation_double": 32,
}


def _numeric_digest(values: dict[str, object]) -> str:
    """Hash numeric content without allowing an ID/family to hide duplicates."""

    payload = json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_six_body_presets(root: Path) -> None:
    body_root = root / "assets" / "bodies"
    body_root.mkdir(parents=True)
    for index in range(6):
        body = {
            "height": 154.0 + index * 5.0,
            "head_l": 21.0 + index * 0.2,
            "bust": 78.0 + index * 5.0,
            "underbust": 70.0 + index * 4.0,
            "waist": 62.0 + index * 4.0,
            "hips": 86.0 + index * 5.0,
            "leg_circ": 48.0 + index * 2.0,
            "back_width": 32.0 + index * 1.2,
            "waist_back_width": 27.0 + index,
            "hip_back_width": 35.0 + index * 1.2,
            "bust_points": 16.0 + index * 0.5,
            "bum_points": 18.0 + index * 0.5,
            "shoulder_w": 29.0 + index,
            "arm_pose_angle": 15.0,
            "hip_inclination": 5.0,
            "shoulder_incl": 12.0,
        }
        (body_root / f"body_{index}.yaml").write_text(
            yaml.safe_dump({"body": body}, sort_keys=True), encoding="utf-8"
        )


def _triangle_panel(operation_id: str = "op.curves") -> TracedPanel:
    points = (
        TracedPoint(
            id="front.p0",
            panel_id="front",
            xy_cm=(0.0, 0.0),
            formula="origin",
            canonical_name="FNP",
            operation_id="op.points",
        ),
        TracedPoint(
            id="front.p1",
            panel_id="front",
            xy_cm=(2.0, 3.0),
            formula="bust / 40",
            canonical_name="SNP",
            measurement_inputs={"bust": 80.0},
            operation_id="op.points",
        ),
        TracedPoint(
            id="front.p2",
            panel_id="front",
            xy_cm=(4.0, 0.0),
            formula="width",
            canonical_name="SP",
            operation_id="op.points",
        ),
    )
    roles = ("neckline", "dart_leg", "dart_leg")
    edges = tuple(
        TracedEdge(
            id=f"front.e{index}",
            panel_id="front",
            start_point_id=points[index].id,
            end_point_id=points[(index + 1) % 3].id,
            semantic_role=role,
            geometry=CurveGeometry(
                kind="line",
                start_cm=points[index].xy_cm,
                end_cm=points[(index + 1) % 3].xy_cm,
            ),
            dependencies=(points[index].id, points[(index + 1) % 3].id),
            operation_id=operation_id,
        )
        for index, role in enumerate(roles)
    )
    return TracedPanel(
        id="front",
        semantic_role="front",
        points=points,
        edges=edges,
        operation_id=operation_id,
    )


def _operations() -> tuple[ConstructionOperation, ...]:
    # Deliberately serialize these out of topological/order sequence.  The DAG,
    # rather than tuple position or the numeric trace order, must define causality.
    return (
        ConstructionOperation(
            id="op.curves",
            order=0,
            operation="create_curves",
            dependencies=("op.points",),
            outputs=("front.e0", "front.e1", "front.e2"),
        ),
        ConstructionOperation(
            id="op.base",
            order=5,
            operation="read_measurements",
            outputs=("bust",),
        ),
        ConstructionOperation(
            id="op.points",
            order=1,
            operation="create_named_points",
            dependencies=("op.base",),
            inputs=("bust",),
            outputs=("front.p0", "front.p1", "front.p2"),
        ),
    )


def _trace_record(
    *,
    darts: tuple[DartTrace, ...] = (),
    metadata: dict | None = None,
    operations: tuple[ConstructionOperation, ...] | None = None,
) -> TShirtTraceRecord:
    return TShirtTraceRecord(
        sample_id="fixture",
        split="test",
        source={"name": "fixture"},
        body={"bust": 80.0},
        design={"ease": 1.0},
        provenance={"fixture": True},
        panels=(_triangle_panel(),),
        operations=operations or _operations(),
        darts=darts,
        metadata=metadata or {},
    )


def _point(x: float, y: float, name: str) -> dict:
    return {"x_mm": x, "y_mm": y, "point_refs": [name], "source_name": name}


def _closed_path(points: list[dict], count: int) -> dict:
    if len(points) != count:
        raise AssertionError("fixture topology mismatch")
    return {
        "operations": [
            {"type": "line", "index": index, "from": points[index], "to": points[(index + 1) % count]}
            for index in range(count)
        ]
    }


def _teagan_fixture() -> dict:
    front = [
        _point(0, 0, "frontHemFold"),
        _point(500, 0, "frontHemSide"),
        _point(500, -700, "frontArmLow"),
        _point(450, -850, "frontArmMid"),
        _point(350, -950, "frontSP"),
        _point(150, -1000, "frontSNP"),
        _point(0, -900, "frontFNP"),
    ]
    back = [
        _point(0, 0, "backHemFold"),
        _point(510, 0, "backHemSide"),
        _point(510, -710, "backArmLow"),
        _point(455, -850, "backArmMid"),
        _point(350, -950, "backSP"),
        _point(150, -990, "backSNP"),
        _point(0, -940, "backBNP"),
    ]
    sleeve = [
        _point(-300, 0, "sleeveHemBack"),
        _point(-360, -500, "sleeveUnderarmBack"),
        _point(-300, -650, "sleeveHeadBack1"),
        _point(-150, -760, "sleeveHeadBack2"),
        _point(0, -800, "sleeveHeadTop"),
        _point(150, -760, "sleeveHeadFront2"),
        _point(300, -650, "sleeveHeadFront1"),
        _point(360, -500, "sleeveUnderarmFront"),
    ]

    def landmark(part: str, coordinate: dict, source_name: str) -> dict:
        return {
            "part": part,
            "coordinate": coordinate,
            "source_point_name": source_name,
            "evidence": "author_named_point",
        }

    def level(measurement: str, y: float, status: str = "EXACT") -> dict:
        instances = []
        for part, width in (("teagan.front", 500), ("teagan.back", 510)):
            instances.append(
                {
                    "part": part,
                    "coordinate": _point(width / 2, y, f"{part}.{measurement}"),
                    "source_point_name": measurement,
                    "evidence": "author_named_point",
                }
            )
        return {
            "source_measurement": measurement,
            "status": status,
            "meaning": f"fixture {measurement} level",
            "instances": instances,
        }

    def annotation(part: str, path_name: str, start: tuple[float, float], end: tuple[float, float]) -> dict:
        return {
            "part": part,
            "path_name": path_name,
            "path": {
                "operations": [
                    {
                        "type": "line",
                        "index": 0,
                        "from": _point(*start, f"{path_name}.start"),
                        "to": _point(*end, f"{path_name}.end"),
                    }
                ]
            },
        }

    return {
        "schema_version": "teagan-test-fixture-1",
        "source": {
            "design_package": "@freesewing/teagan",
            "design_version": "4.10.1",
            "repository": "https://codeberg.org/freesewing/freesewing",
            "source_code_license_spdx": "MIT",
        },
        "input": {
            "model": "fixture-adult",
            "resolved_measurements_mm": {"chestCircumference": 960.0, "waistToHips": 200.0, "hpsToWaist": 420.0},
            "resolved_options": {"sa": 10.0, "stretch": 0.2},
        },
        "parts": {
            "teagan.front": {"paths": {"seam": _closed_path(front, 7)}},
            "teagan.back": {"paths": {"seam": _closed_path(back, 7)}},
            "teagan.sleeve": {"paths": {"seam": _closed_path(sleeve, 8)}},
        },
        "canonical_semantics": {
            "landmarks": {
                "FNP": [landmark("teagan.front", front[6], "frontFNP")],
                "BNP": [landmark("teagan.back", back[6], "backBNP")],
                "SNP": [
                    landmark("teagan.front", front[5], "frontSNP"),
                    landmark("teagan.back", back[5], "backSNP"),
                ],
                "SP": [
                    landmark("teagan.front", front[4], "frontSP"),
                    landmark("teagan.back", back[4], "backSP"),
                ],
                "BP": [],
            },
            "horizontal_levels": {
                "BL": level("chestCircumference", -650, "APPROXIMATE_PROXY"),
                "WL": level("hpsToWaist", -420),
                "HL": level("waistToHips", -620),
            },
            "darts": {"status": "ABSENT_BY_DESIGN"},
        },
        "production_semantics": {
            "notches": {
                "items": [
                    {
                        "part": part,
                        "anchor": coordinate,
                        "snippet_name": f"notch{index}",
                        "notch_type": "single",
                    }
                    for index, (part, coordinate) in enumerate(
                        (
                            ("teagan.front", front[3]),
                            ("teagan.back", back[3]),
                            ("teagan.sleeve", sleeve[2]),
                            ("teagan.sleeve", sleeve[6]),
                        )
                    )
                ]
            },
            "grainlines": {
                "items": [annotation("teagan.sleeve", "grainline", (0, -50), (0, -700))]
            },
            "cut_on_fold": {
                "items": [
                    annotation("teagan.front", "frontCutOnFold", (0, -50), (0, -850)),
                    annotation("teagan.back", "backCutOnFold", (0, -50), (0, -900)),
                ]
            },
            "seam_allowance": {"requested_mm": 10.0},
        },
    }


class TShirtCorpusPlanTests(unittest.TestCase):
    def test_split_cardinalities_are_exact_and_sample_ids_are_globally_unique(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _write_six_body_presets(root)
            first = build_sample_plan(root)
            second = build_sample_plan(root)

        self.assertEqual(len(first), 2592)
        self.assertEqual(split_counts(first), EXPECTED_SPLIT_COUNTS)
        self.assertEqual(len({sample.sample_id for sample in first}), len(first))
        self.assertEqual(first, second)
        self.assertEqual(plan_digest(first), plan_digest(second))

        # This expectation uses the six small presets created by this test,
        # so it pins the numeric plan without depending on an external clone.
        self.assertEqual(plan_digest(first), "ac99944b9b170a2d381ee4d6e47361c4b2166b1cfa740e58426954b8cc59c591")

    def test_ood_designs_are_controlled_supported_block_extrapolations(self):
        designs = build_design_cases()
        training = [design for design in designs if design.family == "train_design"]
        ood = [design for design in designs if design.family == "ood_design"]
        train_bounds = {
            name: (
                min(design.values[name] for design in training),
                max(design.values[name] for design in training),
            )
            for name in training[0].values
        }
        # Ranges declared by the official GarmentCode t-shirt.yaml for the
        # active numeric variables in this bounded recipe.
        supported_bounds = {
            "shirt_width": (1.0, 1.3),
            "shirt_flare": (0.7, 1.6),
            "shirt_length": (0.5, 3.5),
            "neck_width": (-0.5, 1.0),
            "front_neck_depth": (0.3, 2.0),
            "back_neck_depth": (0.0, 2.0),
            "sleeve_length": (0.1, 1.15),
            "armhole_depth": (0.0, 2.0),
            "sleeve_end_width": (0.2, 2.0),
            "sleeve_angle": (10.0, 50.0),
            "armhole_smoothing": (0.1, 0.4),
        }
        expected_blocks = (
            ("below", {"shirt_width", "shirt_flare", "shirt_length"}),
            ("above", {"shirt_width", "shirt_flare", "shirt_length"}),
            ("above", {"neck_width", "front_neck_depth", "back_neck_depth"}),
            (
                "above",
                {
                    "sleeve_length",
                    "armhole_depth",
                    "sleeve_end_width",
                    "sleeve_angle",
                    "armhole_smoothing",
                },
            ),
        )

        self.assertEqual(len(ood), 4)
        for design, (direction, block) in zip(ood, expected_blocks):
            for name, value in design.values.items():
                self.assertGreaterEqual(value, supported_bounds[name][0], f"{design.id}:{name}")
                self.assertLessEqual(value, supported_bounds[name][1], f"{design.id}:{name}")
                train_low, train_high = train_bounds[name]
                if name in block:
                    if direction == "below":
                        self.assertLess(value, train_low, f"{design.id}:{name}")
                    else:
                        self.assertGreater(value, train_high, f"{design.id}:{name}")
                else:
                    self.assertGreaterEqual(value, train_low, f"{design.id}:{name}")
                    self.assertLessEqual(value, train_high, f"{design.id}:{name}")

    def test_numeric_body_and_design_cases_are_unique_across_holdout_families(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _write_six_body_presets(root)
            bodies = build_body_cases(root)
            designs = build_design_cases()
            plan = build_sample_plan(root)

        body_hashes = defaultdict(set)
        for body in bodies:
            digest = _numeric_digest(body.values)
            self.assertNotIn(digest, body_hashes[body.family], body.id)
            body_hashes[body.family].add(digest)
        self.assertEqual(len(body_hashes["ood_body"]), 16)
        self.assertEqual(sum(len(values) for values in body_hashes.values()), len(bodies))
        for first_index, first in enumerate(body_hashes):
            for second in list(body_hashes)[first_index + 1 :]:
                self.assertTrue(body_hashes[first].isdisjoint(body_hashes[second]), f"{first} duplicates {second}")

        design_hashes = defaultdict(set)
        for design in designs:
            digest = _numeric_digest(design.values)
            self.assertNotIn(digest, design_hashes[design.family], design.id)
            design_hashes[design.family].add(digest)
        self.assertEqual(sum(len(values) for values in design_hashes.values()), len(designs))
        for first_index, first in enumerate(design_hashes):
            for second in list(design_hashes)[first_index + 1 :]:
                self.assertTrue(
                    design_hashes[first].isdisjoint(design_hashes[second]),
                    f"{first} duplicates {second}",
                )

        # IDs are intentionally excluded: every body/design numeric pair must
        # itself be unique across all 2,592 samples.
        sample_hashes = {
            _numeric_digest({"body": sample.body.values, "design": sample.design.values}) for sample in plan
        }
        self.assertEqual(len(sample_hashes), len(plan))

    def test_body_and_design_holdouts_are_disjoint_by_family_and_identifier(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _write_six_body_presets(root)
            plan = build_sample_plan(root)

        expected_families = {
            "train": ("central", "train_design"),
            "iid_validation": ("central", "train_design"),
            "iid_test": ("central", "train_design"),
            "validation_body": ("validation_body", "train_design"),
            "validation_design": ("central", "validation_design"),
            "validation_double": ("validation_body", "validation_design"),
            "test_body": ("test_body", "train_design"),
            "test_design": ("central", "test_design"),
            "test_double": ("test_body", "test_design"),
            "unseen_body": ("ood_body", "train_design"),
            "unseen_design": ("central", "ood_design"),
            "unseen_double": ("ood_body", "ood_design"),
        }
        observed = defaultdict(set)
        body_ids = defaultdict(set)
        design_ids = defaultdict(set)
        for sample in plan:
            observed[sample.split].add((sample.body.family, sample.design.family))
            body_ids[sample.body.family].add(sample.body.id)
            design_ids[sample.design.family].add(sample.design.id)
        self.assertEqual(
            dict(observed), {split: {families} for split, families in expected_families.items()}
        )
        for groups in (body_ids, design_ids):
            families = list(groups)
            for index, first in enumerate(families):
                for second in families[index + 1 :]:
                    self.assertTrue(groups[first].isdisjoint(groups[second]), f"{first} leaks into {second}")

        iid = [sample for sample in plan if sample.split in {"train", "iid_validation", "iid_test"}]
        self.assertEqual(len(iid), 64 * 16)
        self.assertEqual(Counter(sample.split for sample in iid), Counter({"train": 768, "iid_validation": 128, "iid_test": 128}))


class TShirtTraceSchemaTests(unittest.TestCase):
    def test_runtime_operation_adapter_rejects_unknown_dependency(self):
        events = (
            {
                "id": "op_1",
                "order": 0,
                "operation": "fixture",
                "dependencies": ("missing_op",),
                "status": "ok",
            },
        )
        with self.assertRaisesRegex(ValueError, "unknown dependencies"):
            _trace_operations(events)

    def test_runtime_operation_adapter_only_attaches_inputs_to_roots(self):
        events = (
            {"id": "op_1", "order": 0, "operation": "root", "dependencies": (), "status": "ok"},
            {"id": "op_2", "order": 1, "operation": "child", "dependencies": ("op_1",), "status": "ok"},
        )
        operations = {operation.id: operation for operation in _trace_operations(events)}
        self.assertEqual(operations["runtime.op_1"].dependencies, ("recipe.inputs",))
        self.assertEqual(operations["runtime.op_2"].dependencies, ("runtime.op_1",))

    def test_topological_order_follows_dependencies_not_serialization_or_trace_order(self):
        record = _trace_record()
        record.validate()
        self.assertEqual(
            [operation.id for operation in record.topological_operations()],
            ["op.base", "op.points", "op.curves"],
        )

    def test_operation_dag_rejects_missing_dependencies_and_cycles(self):
        missing = replace(
            _operations()[0],
            dependencies=("op.does_not_exist",),
        )
        with self.assertRaisesRegex(ValueError, "missing dependency"):
            replace(_trace_record(), operations=(missing, *_operations()[1:])).validate()

        cycle = (
            ConstructionOperation(id="op.curves", order=0, operation="curves", dependencies=("op.points",)),
            ConstructionOperation(id="op.base", order=5, operation="base", dependencies=("op.curves",)),
            ConstructionOperation(id="op.points", order=1, operation="points", dependencies=("op.base",)),
        )
        with self.assertRaisesRegex(ValueError, "contains a cycle"):
            replace(_trace_record(), operations=cycle).validate()

    def test_non_applicable_dart_record_is_not_misread_as_applicable(self):
        dart = DartTrace(
            id="front.no_dart",
            panel_id="front",
            kind="none",
            applicable=False,
            applicability_reason="basic T-shirt is dartless",
        )
        record = _trace_record(darts=(dart,))
        record.validate()
        self.assertEqual(panel_example(record, record.panels[0]).dart_applicability, "NOT_APPLICABLE")

    def test_applicable_dart_and_explicit_recipe_override_are_distinguished(self):
        dart = DartTrace(
            id="front.waist_dart",
            panel_id="front",
            kind="waist_dart",
            applicable=True,
            applicability_reason="fitted fixture",
            apex_point_id="front.p1",
            leg_edge_ids=("front.e1", "front.e2"),
            affected_edge_ids=("front.e1", "front.e2"),
        )
        record = _trace_record(darts=(dart,))
        record.validate()
        self.assertEqual(panel_example(record, record.panels[0]).dart_applicability, "APPLICABLE")

        overridden = replace(record, metadata={"dart_applicability": "EXCLUDED_FROM_BASIC_TSHIRT_TASK"})
        self.assertEqual(
            panel_example(overridden, overridden.panels[0]).dart_applicability,
            "EXCLUDED_FROM_BASIC_TSHIRT_TASK",
        )


class FreeSewingTeaganAdapterTests(unittest.TestCase):
    def test_embedded_named_output_fixture_converts_without_node_or_downloads(self):
        record = teagan_json_to_trace(_teagan_fixture(), sample_id="teagan-fixture")

        self.assertEqual(record.split, "unseen_source")
        self.assertEqual([panel.semantic_role for panel in record.panels], ["front", "back", "sleeve"])
        self.assertEqual(len(record.operations), 3)
        self.assertTrue(all(not operation.training_eligible for operation in record.operations))
        self.assertTrue(all(operation.domain == "freesewing_named_output" for operation in record.operations))
        self.assertEqual(len(record.named_paths), 3)
        self.assertEqual(len(record.notches), 4)
        self.assertEqual(len(record.grainlines), 3)
        self.assertEqual(len(record.seam_allowances), 3)
        self.assertEqual(record.metadata["creation_time_operation_DAG"], "NOT_AVAILABLE_FROM_PACKAGE_OUTPUT")
        self.assertEqual(record.metadata["dart_applicability"], "NOT_APPLICABLE")
        self.assertTrue(all(not dart.applicable for dart in record.darts))
        self.assertEqual(
            Counter(target.semantic_role for target in record.drafting_formula_targets),
            Counter({"neckline": 2, "armhole": 2, "sleeve_head": 1}),
        )
        self.assertTrue(record.drafting_seam_relations)
        self.assertTrue(all(
            target.evidence == "author_named_completed_path_and_public_source_formula"
            for target in record.drafting_formula_targets
        ))
        self.assertTrue(all(
            target.provenance["source_kind"] == "freesewing_named_output"
            for target in record.drafting_formula_targets
        ))
        self.assertTrue(all(
            "creation-time operation DAG unavailable" in target.provenance["evidence_boundary"]
            for target in record.drafting_formula_targets
        ))

        canonical = {
            point.canonical_name
            for panel in record.panels
            for point in panel.points
            if point.canonical_name is not None
        }
        self.assertEqual(canonical, {"FNP", "BNP", "SNP", "SP"})
        bl_lines = [line for line in record.reference_lines if line.canonical_name == "BL"]
        self.assertTrue(bl_lines)
        self.assertTrue(all(not line.training_eligible for line in bl_lines))

        for panel, allowance in zip(record.panels, record.seam_allowances):
            self.assertEqual(set(allowance.width_by_edge_cm), {edge.id for edge in panel.edges})
            if panel.semantic_role in {"front", "back"}:
                fold_edge = next(
                    edge for edge in panel.edges if edge.semantic_role in {"center_front", "center_back"}
                )
                self.assertEqual(allowance.width_by_edge_cm[fold_edge.id], 0.0)


class RuntimeTraceRecorderTests(unittest.TestCase):
    @staticmethod
    def _fake_targets():
        class Edge:
            def __init__(self, start, end):
                self.start = start
                self.end = end
                self.label = ""

            def length(self):
                return sum((float(b) - float(a)) ** 2 for a, b in zip(self.start, self.end)) ** 0.5

            def subdivide_param(self, fractions, connect_internal_verts=True):
                fraction = float(fractions[0])
                middle = [
                    float(self.start[index])
                    + fraction * (float(self.end[index]) - float(self.start[index]))
                    for index in range(2)
                ]
                return EdgeSequence((Edge(self.start, middle), Edge(middle, self.end)))

            def subdivide_len(self, fractions, connect_internal_verts=True):
                return self.subdivide_param(fractions, connect_internal_verts)

        class EdgeSequence:
            def __init__(self, edges=()):
                self.edges = list(edges)

            def length(self):
                return sum(edge.length() for edge in self.edges)

            def isChained(self):
                return all(
                    self.edges[index - 1].end is self.edges[index].start
                    for index in range(1, len(self.edges))
                )

            def isLoop(self):
                return bool(self.edges) and self.edges[-1].end is self.edges[0].start

            def close_loop(self):
                if self.edges and not self.isLoop():
                    self.edges.append(Edge(self.edges[-1].end, self.edges[0].start))
                # Deliberately return None: the recorder must inspect mutated
                # parameters rather than rely on a helper return value.
                return None

        class EdgeSeqFactory:
            @staticmethod
            def from_verts(*verts, loop=False):
                sequence = EdgeSequence(
                    Edge(verts[index - 1], verts[index]) for index in range(1, len(verts))
                )
                if loop:
                    sequence.close_loop()
                return sequence

        class CurveEdgeFactory:
            @staticmethod
            def curve_from_tangents(start, end, **kwargs):
                return Edge(start, end)

        class Panel:
            def __init__(self):
                self.name = "fake_panel"
                self.edges = EdgeSequence()
                self.interfaces = {}

            def add_dart(self, dart_shape, edge, offset, **kwargs):
                return dart_shape, edge

        def cut_corner(target_shape, target_interface, verbose=False):
            return target_shape.edges[0].subdivide_param((0.5, 0.5)), target_interface

        ops = SimpleNamespace(
            cut_corner=cut_corner,
            even_armhole_openings=lambda front_opening, back_opening, **kwargs: (
                front_opening,
                back_opening,
            ),
        )
        pyg = SimpleNamespace(
            ops=ops,
            EdgeSeqFactory=EdgeSeqFactory,
            CurveEdgeFactory=CurveEdgeFactory,
            EdgeSequence=EdgeSequence,
            Edge=Edge,
        )

        def armhole_curve(*args, **kwargs):
            return EdgeSeqFactory.from_verts([0.0, 0.0], [1.0, 1.0], [2.0, 0.0], loop=True)

        sleeves = SimpleNamespace(ArmholeCurve=armhole_curve)
        collars = SimpleNamespace(
            CircleNeckHalf=lambda depth, width, **kwargs: EdgeSeqFactory.from_verts(
                [0.0, 0.0], [width / 2.0, -depth]
            )
        )
        return pyg, sleeves, collars, Panel, EdgeSeqFactory, EdgeSequence

    def test_nested_dependencies_mutated_geometry_enrichment_and_restoration(self):
        pyg, sleeves, collars, panel_cls, edge_factory, edge_sequence = self._fake_targets()
        original_factory = inspect.getattr_static(edge_factory, "from_verts")
        original_close = inspect.getattr_static(edge_sequence, "close_loop")
        marker = []
        holder = {}

        def enrich(event):
            event["semantic_test"] = {
                "active": holder["recorder"].active,
                "marker": holder["recorder"].object_token(marker, "Adapter Marker"),
            }

        recorder = RuntimeTraceRecorder(
            pyg=pyg,
            sleeves=sleeves,
            collars=collars,
            panel_cls=panel_cls,
            event_enricher=enrich,
        )
        holder["recorder"] = recorder
        with recorder:
            result = sleeves.ArmholeCurve(1.0, 2.0, 0.0)

        self.assertEqual(len(result.edges), 3)
        by_operation = {event["operation"]: event for event in recorder.events}
        armhole = by_operation["ArmholeCurve"]
        from_verts = by_operation["EdgeSeqFactory.from_verts"]
        close_loop = by_operation["EdgeSequence.close_loop"]
        self.assertIn(armhole["id"], from_verts["dependencies"])
        self.assertIn(from_verts["id"], close_loop["dependencies"])
        self.assertEqual(len(from_verts["created_primitives"]), 2)
        self.assertEqual(len(close_loop["created_primitives"]), 1)
        self.assertEqual(
            close_loop["created_primitives"][0]["observed_via"],
            ["mutated_parameter:self"],
        )
        self.assertEqual(armhole["created_primitives"], [])
        self.assertTrue(all(event["semantic_test"]["active"] for event in recorder.events))
        self.assertTrue(
            all(event["semantic_test"]["marker"].startswith("adapter_marker_") for event in recorder.events)
        )
        self.assertIs(inspect.getattr_static(edge_factory, "from_verts"), original_factory)
        self.assertIs(inspect.getattr_static(edge_sequence, "close_loop"), original_close)

    def test_event_enricher_failure_is_recorded_without_changing_helper_result(self):
        pyg, sleeves, collars, panel_cls, _, _ = self._fake_targets()

        def fail(_event):
            raise ValueError("semantic enrichment failed")

        recorder = RuntimeTraceRecorder(
            pyg=pyg,
            sleeves=sleeves,
            collars=collars,
            panel_cls=panel_cls,
            event_enricher=fail,
        )
        with recorder:
            result = collars.CircleNeckHalf(2.0, 8.0)

        self.assertEqual(len(result.edges), 1)
        self.assertTrue(recorder.events)
        self.assertTrue(all(event["status"] == "ok" for event in recorder.events))
        self.assertTrue(all("enrichment_error" in event for event in recorder.events))
        self.assertTrue(all("semantic enrichment failed" in event["enrichment_error"]["message"] for event in recorder.events))

    def test_subdivision_tracks_parent_input_and_claims_semantics_once(self):
        pyg, sleeves, collars, panel_cls, edge_factory, _ = self._fake_targets()

        def enrich(event):
            event["semantic_primitives"] = (
                [
                    {"edge_token": primitive["edge_token"], "semantic_role": "armhole"}
                    for primitive in event.get("created_primitives", ())
                ]
                if event["operation"] == "Edge.subdivide_param"
                else []
            )

        recorder = RuntimeTraceRecorder(
            pyg=pyg,
            sleeves=sleeves,
            collars=collars,
            panel_cls=panel_cls,
            event_enricher=enrich,
        )
        with recorder:
            source = edge_factory.from_verts([0.0, 0.0], [2.0, 0.0])
            result, _ = pyg.ops.cut_corner(source, source)

        self.assertEqual(len(result.edges), 2)
        factory = next(event for event in recorder.events if event["operation"] == "EdgeSeqFactory.from_verts")
        corner = next(event for event in recorder.events if event["operation"] == "cut_corner")
        subdivision = next(event for event in recorder.events if event["operation"] == "Edge.subdivide_param")
        self.assertIn(corner["id"], subdivision["dependencies"])
        self.assertIn(factory["id"], subdivision["dependencies"])
        self.assertEqual(subdivision["parent_event_id"], corner["id"])
        self.assertEqual(len(subdivision["created_primitives"]), 2)
        self.assertEqual(subdivision["creation_attribution"], "this_event_after_semantic_enrichment")
        self.assertEqual(corner["created_primitives"], [])
        self.assertEqual(
            {item["edge_token"] for item in subdivision["semantic_primitives"]},
            {item["edge_token"] for item in subdivision["created_primitives"]},
        )


if __name__ == "__main__":
    unittest.main()
