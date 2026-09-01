from __future__ import annotations

from dataclasses import replace
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from benchmark.drafting_semantics.dataset import (
    EDGE_FEATURE_DIM,
    balanced_class_weights,
    edge_features,
    padded_batch,
    panel_examples,
    read_records,
    reindex_panel_example,
)
from benchmark.drafting_semantics.decoding import (
    decode_darts,
    decode_named_landmarks,
    decode_path_measurements,
    landmark_error_summary,
)
from benchmark.drafting_semantics.garmentcode import annotate_garmentcode_sample
from benchmark.drafting_semantics.model import DEFAULT_MODEL_CONFIG, build_model
from benchmark.drafting_semantics.schema import (
    EDGE_ROLES,
    DraftingSemanticRecord,
    ReferenceLine,
)


def _edge(start: int, end: int, *, label: str | None = None, curve: float | None = None) -> dict:
    value: dict = {"endpoints": [start, end]}
    if label is not None:
        value["label"] = label
    if curve is not None:
        value["curvature"] = {"type": "quadratic", "params": [[0.5, curve]]}
    return value


def _front_panel() -> dict:
    # Boundary order: waist, dart legs, waist, side, armhole, shoulder,
    # neckline, centre.  This is intentionally close to a GarmentCode fitted
    # bodice while remaining small enough that expected topology is explicit.
    return {
        "vertices": [
            [0.0, 0.0],
            [0.8, 0.0],
            [1.0, 1.0],
            [1.2, 0.0],
            [3.0, 0.0],
            [3.0, 3.0],
            [2.7, 4.0],
            [1.0, 4.5],
            [0.0, 4.0],
        ],
        "edges": [
            _edge(0, 1),
            _edge(1, 2),
            _edge(2, 3),
            _edge(3, 4),
            _edge(4, 5),
            _edge(5, 6, label="armhole", curve=-0.12),
            _edge(6, 7),
            _edge(7, 8, label="collar", curve=0.15),
            _edge(8, 0),
        ],
    }


def _back_panel() -> dict:
    return {
        "vertices": [[0.0, 0.0], [3.0, 0.0], [3.0, 3.0], [2.7, 4.0], [1.0, 4.5], [0.0, 4.0]],
        "edges": [
            _edge(0, 1),
            _edge(1, 2),
            _edge(2, 3, label="armhole", curve=-0.12),
            _edge(3, 4),
            _edge(4, 5, label="collar", curve=0.10),
            _edge(5, 0),
        ],
    }


def _stitch(first_panel: str, first_edge: int, second_panel: str, second_edge: int) -> list[dict]:
    return [{"panel": first_panel, "edge": first_edge}, {"panel": second_panel, "edge": second_edge}]


def _write_garmentcode_fixture(root: Path) -> tuple[Path, Path, Path]:
    panels = {
        "left_ftorso": _front_panel(),
        "right_ftorso": _front_panel(),
        "left_btorso": _back_panel(),
        "right_btorso": _back_panel(),
    }
    stitches = [
        _stitch("left_ftorso", 1, "left_ftorso", 2),
        _stitch("left_ftorso", 8, "right_ftorso", 8),
        _stitch("left_btorso", 5, "right_btorso", 5),
        _stitch("left_ftorso", 4, "left_btorso", 1),
        _stitch("right_ftorso", 4, "right_btorso", 1),
        _stitch("left_ftorso", 6, "left_btorso", 3),
        _stitch("right_ftorso", 6, "right_btorso", 3),
    ]
    specification = root / "fixture_specification.json"
    body = root / "fixture_body_measurements.yaml"
    design = root / "fixture_design_params.yaml"
    specification.write_text(json.dumps({"pattern": {"panels": panels, "stitches": stitches}}), encoding="utf-8")
    body.write_text(
        yaml.safe_dump(
            {
                "body": {
                    "waist_line": 40.0,
                    "hips_line": 20.0,
                    "_bust_line": 25.0,
                    "back_width": 42.0,
                    "shoulder_w": 38.0,
                    "_shoulder_incl": 20.0,
                    "neck_w": 18.0,
                    "armscye_depth": 13.0,
                    "bust_points": 18.0,
                    "bust": 96.0,
                    "waist": 76.0,
                    "hips": 100.0,
                }
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    design.write_text(
        yaml.safe_dump(
            {
                "design": {
                    "meta": {"upper": {"v": "FittedShirt"}},
                    "shirt": {"length": {"v": 0.8}, "collar": {"style": {"v": "round"}}},
                }
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return specification, body, design


class DraftingSemanticsTests(unittest.TestCase):
    def _record(self, root: Path, *, production_marks: bool = False) -> DraftingSemanticRecord:
        specification, body, design = _write_garmentcode_fixture(root)
        return annotate_garmentcode_sample(
            specification,
            body,
            design,
            split="training",
            synthesize_production_marks=production_marks,
        )

    def test_garmentcode_annotation_recovers_boundary_roles_points_lines_and_dart(self):
        with tempfile.TemporaryDirectory() as folder:
            record = self._record(Path(folder))

        record.validate()
        panels = {panel.id: panel for panel in record.panels}
        front = panels["left_ftorso"]
        self.assertEqual(front.role, "front_bodice")
        self.assertEqual(
            [edge.role for edge in front.edges],
            ["waistline", "dart_leg", "dart_leg", "waistline", "side_seam", "armhole", "shoulder", "neckline", "center_front"],
        )
        self.assertGreater(front.edges[7].length_cm, np.linalg.norm(np.asarray(front.edges[7].end_cm) - front.edges[7].start_cm))

        landmarks = {item.name: item for item in front.landmarks}
        self.assertEqual(landmarks["FNP"].xy_cm, (0.0, 4.0))
        self.assertEqual(landmarks["SNP"].xy_cm, (1.0, 4.5))
        self.assertEqual(landmarks["SP"].xy_cm, (2.7, 4.0))
        self.assertEqual(landmarks["BP"].xy_cm, (9.0, 15.0))
        self.assertFalse(landmarks["BP"].training_eligible)

        lines = {item.name: item for item in front.reference_lines}
        self.assertEqual(lines["WL"].points_cm[0][1], 0.0)
        self.assertEqual(lines["BL"].points_cm[0][1], 15.0)
        self.assertEqual(lines["HL"].points_cm[0][1], -20.0)
        self.assertFalse(lines["HL"].intersects_panel)
        self.assertFalse(lines["HL"].training_eligible)

        self.assertEqual(len(record.darts), 1)
        dart = record.darts[0]
        self.assertEqual(dart.kind, "waist_dart")
        self.assertAlmostEqual(dart.intake_cm, 0.4)
        self.assertAlmostEqual(dart.depth_cm, 1.0)
        self.assertEqual(len(record.construction_steps), 7)
        self.assertEqual(record.measurements["garmentcode_back_waist_length_cm"], 40.0)
        self.assertEqual(record.measurements["garmentcode_waist_to_hip_cm"], 20.0)

    def test_semantic_record_jsonl_round_trip_preserves_typed_structure(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            record = self._record(root)
            path = root / "records.jsonl"
            path.write_text(json.dumps(record.to_dict()) + "\n", encoding="utf-8")
            restored = read_records(path)[0]

        self.assertEqual(restored, record)
        self.assertIsInstance(restored.panels[0].edges[0].endpoints, tuple)
        self.assertIsInstance(restored.panels[0].landmarks[0].xy_cm, tuple)
        self.assertIsInstance(restored.darts[0].leg_edge_ids, tuple)
        self.assertIsInstance(restored.construction_steps[0].inputs, tuple)

    def test_reference_line_validation_accepts_current_intersection_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            record = self._record(Path(folder))
        record.validate()
        front = next(panel for panel in record.panels if panel.id == "left_ftorso")
        by_name = {line.name: line for line in front.reference_lines}
        self.assertTrue(by_name["BL"].intersects_panel)
        self.assertTrue(by_name["BL"].training_eligible)
        # A mathematically defined line outside the panel is stored evidence,
        # but is correctly excluded from coordinate training.
        self.assertFalse(by_name["HL"].intersects_panel)
        self.assertFalse(by_name["HL"].training_eligible)

    def test_reference_line_validation_rejects_bad_geometry_and_evidence(self):
        valid = ReferenceLine(
            name="WL",
            panel_id="panel",
            points_cm=((0.0, 0.0), (10.0, 0.0)),
            evidence="derived_topology",
            confidence=0.9,
        )
        valid.validate(expected_panel_id="panel")
        cases = (
            (replace(valid, panel_id="other"), "panel mismatch"),
            (replace(valid, points_cm=((0.0, 0.0), (float("nan"), 1.0))), "finite"),
            (replace(valid, points_cm=((1.0, 1.0), (1.0, 1.0))), "nondegenerate"),
            (replace(valid, evidence="guess"), "evidence"),
            (replace(valid, evidence="unavailable", training_eligible=False), "unavailable"),
            (replace(valid, confidence=1.01), "confidence"),
            (replace(valid, intersects_panel="yes"), "intersects_panel"),
            (replace(valid, training_eligible="yes"), "training_eligible"),
            (
                replace(valid, intersects_panel=False, training_eligible=True),
                "without intersecting",
            ),
            (
                replace(valid, evidence="synthetic_unvalidated", training_eligible=True),
                "synthetic_unvalidated",
            ),
        )
        for line, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    line.validate(expected_panel_id="panel")

    def test_record_validation_enforces_reference_line_ownership_and_uniqueness(self):
        with tempfile.TemporaryDirectory() as folder:
            record = self._record(Path(folder))
        front = next(panel for panel in record.panels if panel.id == "left_ftorso")
        line = front.reference_lines[0]

        def record_with(lines):
            return replace(
                record,
                panels=tuple(
                    replace(panel, reference_lines=tuple(lines))
                    if panel.id == front.id
                    else panel
                    for panel in record.panels
                ),
            )

        with self.assertRaisesRegex(ValueError, "panel mismatch"):
            record_with((replace(line, panel_id="right_ftorso"),)).validate()
        with self.assertRaisesRegex(ValueError, "duplicate reference line"):
            record_with((line, replace(line))).validate()

    def test_feature_encoder_masks_stitch_leakage_by_default_and_batches(self):
        with tempfile.TemporaryDirectory() as folder:
            record = self._record(Path(folder))
        front = next(panel for panel in record.panels if panel.id == "left_ftorso")
        hidden, targets = edge_features(front)
        exposed, _ = edge_features(front, include_stitch_features=True)
        self.assertEqual(hidden.shape, (9, EDGE_FEATURE_DIM))
        self.assertEqual(targets.shape, (9,))
        np.testing.assert_allclose(hidden[:, 13:15], 0.0)
        np.testing.assert_allclose(exposed[1, 13:15], [1.0, 1.0])
        np.testing.assert_allclose(exposed[4, 13:15], [1.0, 0.0])
        self.assertTrue(np.isfinite(hidden).all())

        examples = panel_examples((record,), splits={"training"})
        self.assertEqual(len(examples), 4)
        features, batch_targets, valid, roles = padded_batch(examples[:2], maximum_edges=12)
        self.assertEqual(features.shape, (2, 12, EDGE_FEATURE_DIM))
        self.assertEqual(batch_targets.shape, (2, 12))
        self.assertEqual(valid.sum(axis=1).tolist(), [9, 9])
        self.assertTrue(np.all(batch_targets[~valid] == -100))
        self.assertEqual(roles.shape, (2,))

        weights = balanced_class_weights(examples)
        self.assertEqual(weights.shape, (len(EDGE_ROLES),))
        self.assertGreater(weights[EDGE_ROLES.index("hemline")], weights[EDGE_ROLES.index("waistline")])

    def test_landmark_decoder_is_strict_and_reports_missing_predictions(self):
        with tempfile.TemporaryDirectory() as folder:
            record = self._record(Path(folder))
        front = next(panel for panel in record.panels if panel.id == "left_ftorso")
        roles = [edge.role for edge in front.edges]
        decoded = decode_named_landmarks(front, roles)
        self.assertEqual(decoded, {"FNP": (0.0, 4.0), "SNP": (1.0, 4.5), "SP": (2.7, 4.0)})
        predicted_darts = decode_darts(front, roles)
        self.assertEqual(len(predicted_darts), 1)
        self.assertAlmostEqual(predicted_darts[0]["intake_cm"], 0.4)
        self.assertAlmostEqual(predicted_darts[0]["depth_cm"], 1.0)
        path_measurements = decode_path_measurements(front, roles)
        self.assertGreater(path_measurements["neckline_arc_length_cm"], np.linalg.norm(np.asarray(front.edges[7].end_cm) - front.edges[7].start_cm))
        self.assertAlmostEqual(path_measurements["shoulder_path_length_cm"], front.edges[6].length_cm)
        # argmax predictions arrive here as NumPy integer scalars during the
        # real training evaluator, not as Python ints.
        numeric_roles = np.asarray([EDGE_ROLES.index(role) for role in roles], dtype=np.int64)
        self.assertEqual(decode_named_landmarks(front, numeric_roles), decoded)
        self.assertEqual(
            landmark_error_summary(front, roles),
            {"target_count": 3, "decoded_count": 3, "exact_count": 3, "normalized_distance_sum": 0.0},
        )
        missing = ["other" if role == "shoulder" else role for role in roles]
        self.assertEqual(decode_named_landmarks(front, missing), {"FNP": (0.0, 4.0)})
        summary = landmark_error_summary(front, missing)
        self.assertEqual(summary["decoded_count"], 1)
        with self.assertRaises(ValueError):
            decode_named_landmarks(front, roles[:-1])

    def test_boundary_reindexing_preserves_targets_and_canonical_edge_mapping(self):
        with tempfile.TemporaryDirectory() as folder:
            record = self._record(Path(folder))
        example = panel_examples((record,))[0]
        transformed = reindex_panel_example(example, shift=3, reverse=True)
        self.assertCountEqual(transformed.targets.tolist(), example.targets.tolist())
        canonical = np.empty_like(transformed.targets)
        canonical[transformed.edge_indices] = transformed.targets
        np.testing.assert_array_equal(canonical, example.targets)
        np.testing.assert_allclose(transformed.features[:, 15] ** 2 + transformed.features[:, 16] ** 2, 1.0, atol=1e-6)

    def test_synthetic_production_marks_are_explicitly_non_training_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            record = self._record(Path(folder), production_marks=True)
        production = record.production_annotations
        self.assertFalse(production["source_notches"]["available"])
        self.assertFalse(production["source_grainlines"]["available"])
        self.assertFalse(production["source_seam_allowances"]["available"])
        self.assertTrue(production["synthetic_grainlines"])
        self.assertTrue(all(not item["training_eligible"] for item in production["synthetic_grainlines"]))
        self.assertTrue(all(not item["training_eligible"] for item in production["synthetic_notches"]))
        self.assertFalse(production["synthetic_seam_allowance_cm"]["training_eligible"])

    def test_small_transformer_emits_one_role_distribution_per_edge(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is optional in the lightweight unit-test environment")

        with tempfile.TemporaryDirectory() as folder:
            record = self._record(Path(folder))
        examples = panel_examples((record,))[:2]
        features, _, valid, roles = padded_batch(examples, maximum_edges=12)
        config = dict(DEFAULT_MODEL_CONFIG)
        config.update({"width": 32, "heads": 4, "layers": 1, "feedforward_multiplier": 2, "dropout": 0.0})
        model = build_model(config).eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(features), torch.from_numpy(valid), torch.from_numpy(roles))
        self.assertEqual(tuple(logits.shape), (2, 12, len(EDGE_ROLES)))
        self.assertTrue(torch.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()
