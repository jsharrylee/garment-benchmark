from __future__ import annotations

import gzip
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from benchmark.drafting_semantics.tshirt_schema import (
    ConstructionOperation,
    CurveGeometry,
    TShirtTraceRecord,
    TracedEdge,
    TracedPanel,
    TracedPoint,
)
from benchmark.scripts.augment_tshirt_formula_targets import augment_record, migrate_corpus


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _panel(
    panel_id: str,
    panel_role: str,
    edge_role: str,
    start: tuple[float, float],
    end: tuple[float, float],
    controls: tuple[tuple[float, float], tuple[float, float]],
    canonical: tuple[str | None, str | None],
) -> tuple[TracedPanel, ConstructionOperation]:
    operation_id = f"runtime.{panel_id}"
    token = f"edge_token:{panel_id}"
    points = (
        TracedPoint(
            id=f"{panel_id}.start",
            panel_id=panel_id,
            xy_cm=start,
            formula="creation formula start",
            canonical_name=canonical[0],
            source_name="named_start",
            operation_id=operation_id,
            evidence="creation_event_binding",
        ),
        TracedPoint(
            id=f"{panel_id}.end",
            panel_id=panel_id,
            xy_cm=end,
            formula="creation formula end",
            canonical_name=canonical[1],
            source_name="named_end",
            operation_id=operation_id,
            evidence="creation_event_binding",
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
        formula=f"creation formula {edge_role}",
        operation_id=operation_id,
        evidence="creation_event_binding",
        provenance={
            "runtime_object_token": token,
            "measurement_inputs": {"design.fixture": 0.5},
        },
    )
    panel = TracedPanel(
        id=panel_id,
        semantic_role=panel_role,
        points=points,
        edges=(edge,),
        operation_id=operation_id,
    )
    operation = ConstructionOperation(
        id=operation_id,
        order=0,
        operation="create_fixture_curve",
        outputs=(edge.id,),
        parameters={"created_primitives": [{"edge_token": token}]},
    )
    return panel, operation


def _legacy_record() -> TShirtTraceRecord:
    specs = (
        ("front_neck", "front", "neckline", (0.0, 0.0), (4.0, 3.0), ((1.0, 0.0), (3.0, 2.0)), ("FNP", "SNP")),
        ("front_arm", "front", "armhole", (7.0, 1.0), (8.0, 8.0), ((7.2, 2.0), (8.7, 6.0)), ("SP", None)),
        ("sleeve", "sleeve", "sleeve_head", (-5.0, 0.0), (5.0, 0.0), ((-3.0, -6.0), (3.0, -6.0)), (None, None)),
    )
    values = [_panel(*spec) for spec in specs]
    operations = tuple(replace(operation, order=index) for index, (_, operation) in enumerate(values))
    return TShirtTraceRecord(
        sample_id="legacy-fixture",
        split="train",
        source={"name": "GarmentCode fixture"},
        body={"bust": 90.0},
        design={"fixture": 0.5},
        provenance={"legacy": True},
        panels=tuple(panel for panel, _ in values),
        operations=operations,
        metadata={
            "creation_semantic_contract": {
                "completed_panel_topology_used_for_semantic_labels": False,
                "canonical_points_verified_as_operation_outputs": True,
                "semantic_edges_verified_as_operation_outputs": True,
                "all_operations_reachable_from_recipe_inputs": True,
            }
        },
        schema_version="tshirt-construction-trace-1.0",
    )


class FormulaTargetMigrationTests(unittest.TestCase):
    def test_streaming_migration_preserves_source_and_is_byte_deterministic(self):
        record = _legacy_record()
        legacy = record.to_dict()
        legacy.pop("drafting_formula_targets")
        legacy.pop("drafting_seam_relations")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "legacy.jsonl.gz"
            first = root / "first.jsonl.gz"
            second = root / "second.jsonl.gz"
            with gzip.open(source, "wt", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(legacy, sort_keys=True, separators=(",", ":")) + "\n")
            source_hash = _sha256(source)
            first_manifest = migrate_corpus(source, first, expected_count=1)
            second_manifest = migrate_corpus(source, second, expected_count=1)

            self.assertEqual(_sha256(source), source_hash)
            self.assertEqual(_sha256(first), _sha256(second))
            self.assertEqual(first_manifest["output_uncompressed_canonical_rows_sha256"],
                             second_manifest["output_uncompressed_canonical_rows_sha256"])
            self.assertTrue(first_manifest["source_preserved"])
            self.assertTrue(first_manifest["audit"]["runtime_operation_output_membership_verified"])
            self.assertFalse(first_manifest["audit"]["completed_topology_role_inference_used"])
            self.assertEqual(first_manifest["target_count"], 3)
            self.assertEqual(first_manifest["seam_relation_count"], 1)

            with gzip.open(first, "rt", encoding="utf-8") as stream:
                migrated = TShirtTraceRecord.from_dict(json.loads(next(stream)))
            self.assertEqual(migrated.schema_version, "tshirt-construction-trace-1.1")
            self.assertEqual(len(migrated.drafting_formula_targets), 3)
            self.assertEqual(migrated.provenance, record.provenance)
            self.assertEqual(migrated.panels, record.panels)
            self.assertEqual(migrated.operations, record.operations)

    def test_creation_evidence_drift_fails_closed(self):
        record = _legacy_record()
        bad_edge = replace(record.panels[0].edges[0], evidence="post_hoc_shape_label")
        bad_panel = replace(record.panels[0], edges=(bad_edge,))
        bad_record = replace(record, panels=(bad_panel, *record.panels[1:]))
        with self.assertRaisesRegex(ValueError, "not eligible creation-event evidence"):
            augment_record(bad_record)

    def test_augmented_input_is_rejected_instead_of_rewritten(self):
        migrated = augment_record(_legacy_record())
        with self.assertRaisesRegex(ValueError, "already formula-augmented"):
            augment_record(migrated)


if __name__ == "__main__":
    unittest.main()
