from __future__ import annotations

import tempfile
import unittest
import gzip
from dataclasses import replace
from pathlib import Path

import numpy as np

from benchmark.drafting_semantics.tshirt_learning import (
    DEFAULT_TSHIRT_MODEL_CONFIG,
    EDGE_FEATURE_DIM,
    EDGE_ROLES,
    LANDMARK_NAMES,
    PANEL_ROLES,
    BodyFeatureSpec,
    BoundaryAugmentation,
    augment_panel_example,
    build_tshirt_model,
    deterministic_split,
    evaluate_model,
    padded_batch,
    panel_example,
    panel_geometry_features,
    read_tshirt_records,
    _classification_metrics,
    _landmark_metrics,
)
from benchmark.drafting_semantics.tshirt_schema import (
    ConstructionOperation,
    CurveGeometry,
    TShirtTraceRecord,
    TracedEdge,
    TracedPanel,
    TracedPoint,
)
from benchmark.scripts.train_tshirt_semantics import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_RECORDS_PATH,
    _guard_hash_split_reallocation,
    _loss,
    _validation_selection_key,
    train,
)
from benchmark.scripts.combine_tshirt_semantic_records import training_projection


def _panel(panel_id: str = "front", role: str = "front") -> TracedPanel:
    names = ("FNP", "SNP", "SP", None, None, None) if role == "front" else ("BNP", "SNP", "SP", None, None, None)
    coordinates = ((0.0, 4.0), (1.0, 4.5), (3.0, 4.0), (3.5, 3.0), (3.2, 0.0), (0.0, 0.0))
    points = tuple(
        TracedPoint(
            id=f"{panel_id}:p{index}",
            panel_id=panel_id,
            xy_cm=xy,
            formula="synthetic fixture",
            canonical_name=name,
            source_name=None if name else f"corner_{index}",
        )
        for index, (name, xy) in enumerate(zip(names, coordinates))
    )
    roles = (
        "neckline",
        "shoulder",
        "armhole",
        "side_seam",
        "hemline",
        "center_front" if role == "front" else "center_back",
    )
    edges = []
    for index, edge_role in enumerate(roles):
        start = points[index]
        end = points[(index + 1) % len(points)]
        geometry = CurveGeometry(
            kind="quadratic_bezier" if edge_role in {"neckline", "armhole"} else "line",
            start_cm=start.xy_cm,
            end_cm=end.xy_cm,
            control_points_cm=(
                ((start.xy_cm[0] + end.xy_cm[0]) / 2.0, (start.xy_cm[1] + end.xy_cm[1]) / 2.0 - 0.2),
            )
            if edge_role in {"neckline", "armhole"}
            else (),
        )
        edges.append(
            TracedEdge(
                id=f"{panel_id}:e{index}",
                panel_id=panel_id,
                start_point_id=start.id,
                end_point_id=end.id,
                semantic_role=edge_role,
                geometry=geometry,
            )
        )
    return TracedPanel(id=panel_id, semantic_role=role, points=points, edges=tuple(edges))


def _record(sample_id: str = "sample-1", split: str = "train", source: str = "GarmentCode") -> TShirtTraceRecord:
    record = TShirtTraceRecord(
        sample_id=sample_id,
        split=split,
        source={"name": source},
        body={"bust": 92.0, "waist_line": 41.0, "shoulder_w": 38.0},
        design={"width": 1.0},
        provenance={"fixture": True},
        panels=(_panel("front", "front"), _panel("back", "back")),
        operations=(ConstructionOperation(id="op:base", order=0, operation="create_fixture"),),
        metadata={"dart_applicability": "NOT_APPLICABLE"},
    )
    record.validate()
    return record


class TShirtLearningTests(unittest.TestCase):
    def test_geometry_features_exclude_labels_and_serialization_phase(self):
        panel = _panel()
        changed_edges = tuple(replace(edge, semantic_role="other", source_name="leaky truth") for edge in panel.edges)
        changed = replace(panel, semantic_role="back", source_name="another source panel", edges=changed_edges)
        first, lengths, _, _ = panel_geometry_features(panel)
        second, changed_lengths, _, _ = panel_geometry_features(changed)
        np.testing.assert_allclose(first, second)
        np.testing.assert_allclose(lengths, changed_lengths)
        self.assertEqual(first.shape, (6, EDGE_FEATURE_DIM))
        self.assertTrue(np.isfinite(first).all())

    def test_clockwise_wrapped_arc_normalization_uses_directed_sweep(self):
        panel = _panel()
        arc = replace(
            panel.edges[0],
            geometry=CurveGeometry(
                kind="arc",
                start_cm=(0.984807753, 0.173648178),
                end_cm=(0.984807753, -0.173648178),
                center_cm=(0.0, 0.0),
                radius_cm=1.0,
                start_angle_degrees=10.0,
                end_angle_degrees=350.0,
                clockwise=True,
            ),
        )
        changed = replace(panel, edges=(arc, *panel.edges[1:]))
        features, lengths, _, _ = panel_geometry_features(changed)
        self.assertTrue(np.isfinite(features).all())
        self.assertAlmostEqual(float(lengths[0]), np.deg2rad(20.0), places=5)

    def test_batch_refuses_to_silently_truncate_panel_edges(self):
        example = panel_example(_record(), _panel())
        with self.assertRaisesRegex(ValueError, "would truncate panel geometry"):
            padded_batch((example,), maximum_edges=5)

    def test_shift_reverse_rotation_and_scale_preserve_edge_truth_mapping(self):
        example = panel_example(_record(), _panel())
        transformed = augment_panel_example(
            example,
            BoundaryAugmentation(shift=2, reverse=True, rotation_degrees=73.0, scale=1.15),
        )
        original = dict(zip(example.edge_ids, example.edge_targets.tolist()))
        moved = dict(zip(transformed.edge_ids, transformed.edge_targets.tolist()))
        self.assertEqual(moved, original)
        np.testing.assert_allclose(
            transformed.features[:, 6],
            np.asarray([example.features[list(example.edge_ids).index(edge_id), 6] for edge_id in transformed.edge_ids]) * 1.15,
            atol=1e-6,
        )
        self.assertCountEqual(transformed.edge_targets.tolist(), example.edge_targets.tolist())

    def test_body_conditioning_uses_training_statistics_and_presence_bits(self):
        first = _record("one")
        second = replace(_record("two"), body={"bust": 100.0, "waist_line": 43.0})
        spec = BodyFeatureSpec.fit((first, second))
        encoded = spec.encode({"bust": 96.0})
        self.assertEqual(encoded.shape, (spec.feature_dim,))
        bust = spec.names.index("bust")
        shoulder = spec.names.index("shoulder_w")
        self.assertAlmostEqual(float(encoded[bust]), 0.0, places=6)
        self.assertEqual(float(encoded[len(spec.names) + bust]), 1.0)
        self.assertEqual(float(encoded[len(spec.names) + shoulder]), 0.0)

        try:
            import torch
        except ImportError:
            return
        examples = tuple(panel_example(first, panel, body_spec=spec) for panel in first.panels)
        batch = padded_batch(examples, maximum_edges=10)
        config = dict(DEFAULT_TSHIRT_MODEL_CONFIG)
        config.update(
            {
                "width": 16,
                "heads": 2,
                "layers": 1,
                "feedforward_multiplier": 2,
                "dropout": 0.0,
                "maximum_edges": 10,
                "mode": "pattern+body",
            }
        )
        model = build_tshirt_model(config, body_feature_dim=spec.feature_dim).eval()
        with torch.no_grad():
            output = model(
                torch.from_numpy(batch["features"]),
                torch.from_numpy(batch["valid_mask"]),
                torch.from_numpy(batch["body_features"]),
            )
        self.assertEqual(tuple(output["panel_logits"].shape), (2, len(PANEL_ROLES)))

    def test_jsonl_round_trip_and_hash_split_are_deterministic(self):
        record = _record()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "records.jsonl"
            path.write_text(__import__("json").dumps(record.to_dict()) + "\n", encoding="utf-8")
            restored = read_tshirt_records(path)
        self.assertEqual(restored, (record,))
        self.assertEqual(deterministic_split("same-id", seed=9), deterministic_split("same-id", seed=9))
        self.assertIn(deterministic_split("different-id", seed=9), {"train", "validation", "test"})

    def test_training_projection_preserves_geometry_targets_and_validates(self):
        original = _record()
        projected = TShirtTraceRecord.from_dict(training_projection(original.to_dict()))
        projected.validate()
        self.assertTrue(projected.provenance["training_projection"])
        self.assertEqual(len(projected.operations), 1)
        self.assertFalse(projected.named_paths)
        original_examples = tuple(panel_example(original, panel) for panel in original.panels)
        projected_examples = tuple(panel_example(projected, panel) for panel in projected.panels)
        for first, second in zip(original_examples, projected_examples):
            np.testing.assert_allclose(first.features, second.features)
            np.testing.assert_array_equal(first.edge_targets, second.edge_targets)
            np.testing.assert_array_equal(first.landmark_exists, second.landmark_exists)
            np.testing.assert_allclose(first.landmark_xy_normalized, second.landmark_xy_normalized)

    def test_gzip_reader_and_cli_defaults_match_generated_corpus(self):
        import json

        record = _record()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "records.jsonl.gz"
            with gzip.open(path, "wt", encoding="utf-8") as stream:
                stream.write(json.dumps(record.to_dict()) + "\n")
            self.assertEqual(read_tshirt_records(path), (record,))
        self.assertEqual(DEFAULT_RECORDS_PATH.as_posix(), "artifacts/drafting_semantics/tshirt_traces.jsonl.gz")
        self.assertEqual(DEFAULT_CONFIG_PATH.as_posix(), "benchmark/configs/tshirt_creation_trace_training.json")

    def test_true_edge_length_weighting_applies_to_tp_fp_and_fn(self):
        metrics = _classification_metrics(
            predictions=np.asarray([0, 1, 1], dtype=np.int64),
            targets=np.asarray([0, 0, 1], dtype=np.int64),
            weights=np.asarray([1.0, 100.0, 1.0], dtype=np.float64),
            names=("first", "second"),
        )
        # Count-F1 is 2/3 for each role, but the 100 cm error must dominate the
        # actual length-weighted confusion matrix.
        self.assertAlmostEqual(metrics["macro_f1_supported_semantics"], 2.0 / 3.0)
        expected = 2.0 / 102.0
        self.assertAlmostEqual(metrics["per_role"]["first"]["length_weighted_f1"], expected)
        self.assertAlmostEqual(metrics["per_role"]["second"]["length_weighted_f1"], expected)
        self.assertAlmostEqual(metrics["length_weighted_macro_f1_supported_semantics"], expected)

    def test_landmark_location_reports_conditional_and_detection_aware_scores(self):
        probability = np.asarray([[0.1, 0.1, 0.1, 0.1]], dtype=np.float32)
        truth = np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        coordinates = np.zeros((1, 4, 2), dtype=np.float32)
        mask = np.asarray([[True, False, False, False]])
        metrics = _landmark_metrics(probability, coordinates, truth, coordinates, mask, np.asarray([10.0]))
        self.assertEqual(metrics["gt_positive_conditional_pck_panel_span_1pct"], 1.0)
        self.assertEqual(metrics["detection_aware_success_pck_panel_span_1pct"], 0.0)
        self.assertEqual(metrics["detection_aware_mean_error_cm_one_panel_span_miss_penalty"], 10.0)
        self.assertNotIn("point_mean_euclidean_error_cm", metrics)

    def test_checkpoint_objectives_prioritize_the_declared_validation_task(self):
        first = {
            "edge_semantics": {"length_weighted_macro_f1_supported_semantics": 0.99},
            "landmarks": {
                "existence_micro_f1": 0.9,
                "detection_aware_success_pck_panel_span_2pct": 0.1,
                "gt_positive_conditional_median_euclidean_error_cm": 2.0,
            },
        }
        second = {
            "edge_semantics": {"length_weighted_macro_f1_supported_semantics": 0.95},
            "landmarks": {
                "existence_micro_f1": 0.95,
                "detection_aware_success_pck_panel_span_2pct": 0.4,
                "gt_positive_conditional_median_euclidean_error_cm": 1.0,
            },
        }
        self.assertGreater(_validation_selection_key(first, "edge-primary"), _validation_selection_key(second, "edge-primary"))
        self.assertGreater(_validation_selection_key(second, "landmark-primary"), _validation_selection_key(first, "landmark-primary"))

    def test_hash_split_refuses_explicit_or_external_unseen_source(self):
        records = (
            _record("train", "train", "GarmentCode"),
            _record("external", "unseen_source", "FreeSewing"),
        )
        with self.assertRaisesRegex(ValueError, "frozen cross-source"):
            _guard_hash_split_reallocation(records, {"train"})

    def test_training_fails_instead_of_silently_selecting_without_validation(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch is optional in the lightweight test environment")
        import json

        config = json.loads(json.dumps(DEFAULT_TSHIRT_MODEL_CONFIG))
        config.update({"epochs": 1, "width": 16, "heads": 2, "layers": 1})
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            records_path = root / "records.jsonl"
            records_path.write_text(json.dumps(_record("train-only").to_dict()) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no validation panels"):
                train(
                    records_path=records_path,
                    checkpoint_path=root / "unused.pt",
                    metrics_path=root / "unused.json",
                    config=config,
                    train_splits={"train"},
                    validation_splits={"iid_validation"},
                    split_mode="record",
                    device_name="cpu",
                )

    def test_small_set_transformer_emits_all_heads_and_evaluates(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is optional in the lightweight test environment")
        record = _record()
        examples = tuple(panel_example(record, panel) for panel in record.panels)
        batch = padded_batch(examples, maximum_edges=10)
        config = dict(DEFAULT_TSHIRT_MODEL_CONFIG)
        config.update(
            {
                "width": 32,
                "heads": 4,
                "layers": 1,
                "feedforward_multiplier": 2,
                "dropout": 0.0,
                "maximum_edges": 10,
                "batch_size": 2,
                "mode": "pattern-only",
            }
        )
        model = build_tshirt_model(config).eval()
        with torch.no_grad():
            output = model(torch.from_numpy(batch["features"]), torch.from_numpy(batch["valid_mask"]), torch.from_numpy(batch["body_features"]))
        self.assertEqual(tuple(output["edge_logits"].shape), (2, 10, len(EDGE_ROLES)))
        self.assertEqual(tuple(output["panel_logits"].shape), (2, len(PANEL_ROLES)))
        self.assertEqual(tuple(output["landmark_existence_logits"].shape), (2, len(LANDMARK_NAMES)))
        self.assertEqual(tuple(output["landmark_xy_normalized"].shape), (2, len(LANDMARK_NAMES), 2))
        metrics = evaluate_model(model, examples, config, torch.device("cpu"))
        self.assertEqual(metrics["panel_count"], 2)
        self.assertEqual(metrics["landmarks"]["gt_positive_conditional_location_target_count"], 6)
        self.assertEqual(metrics["construction_dag"]["score"], None)
        self.assertEqual(metrics["dart_false_positive"]["ground_truth_applicability"], "NOT_APPLICABLE_BASIC_TSHIRT")

    def test_flattened_edge_loss_matches_nd_cross_entropy(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is optional in the lightweight test environment")
        generator = torch.Generator().manual_seed(7)
        logits = torch.randn(2, 5, len(EDGE_ROLES), generator=generator)
        targets = torch.tensor([[0, 1, 2, -100, -100], [3, 4, 5, 6, -100]])
        outputs = {
            "edge_logits": logits,
            "panel_logits": torch.randn(2, len(PANEL_ROLES), generator=generator),
            "landmark_existence_logits": torch.randn(2, len(LANDMARK_NAMES), generator=generator),
            "landmark_xy_normalized": torch.randn(2, len(LANDMARK_NAMES), 2, generator=generator),
        }
        tensors = {
            "edge_targets": targets,
            "panel_targets": torch.tensor([0, 1]),
            "landmark_exists": torch.zeros(2, len(LANDMARK_NAMES)),
            "landmark_xy_normalized": torch.zeros(2, len(LANDMARK_NAMES), 2),
            "landmark_coordinate_mask": torch.zeros(2, len(LANDMARK_NAMES), dtype=torch.bool),
        }
        weights = torch.ones(len(EDGE_ROLES))
        _, components = _loss(outputs, tensors, weights, DEFAULT_TSHIRT_MODEL_CONFIG)
        expected = torch.nn.functional.cross_entropy(logits.transpose(1, 2), targets, ignore_index=-100)
        self.assertAlmostEqual(components["edge"], float(expected), places=6)

    def test_one_epoch_training_smoke_reports_record_splits_and_unseen_source(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch is optional in the lightweight test environment")
        import json

        records = (
            _record("train-1", "train", "GarmentCode"),
            _record("train-2", "train", "GarmentCode"),
            _record("validation-1", "validation", "GarmentCode"),
            _record("test-1", "test", "GarmentCode"),
            _record("unseen-1", "unseen_source", "FreeSewing"),
        )
        config = json.loads(json.dumps(DEFAULT_TSHIRT_MODEL_CONFIG))
        config.update(
            {
                "width": 16,
                "heads": 2,
                "layers": 1,
                "feedforward_multiplier": 2,
                "dropout": 0.0,
                "maximum_edges": 10,
                "batch_size": 4,
                "epochs": 1,
                "mode": "pattern-only",
            }
        )
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            records_path = root / "records.jsonl"
            records_path.write_text(
                "".join(json.dumps(record.to_dict()) + "\n" for record in records), encoding="utf-8"
            )
            result = train(
                records_path=records_path,
                checkpoint_path=root / "model.pt",
                metrics_path=root / "metrics.json",
                config=config,
                train_splits={"train"},
                validation_splits={"validation"},
                split_mode="record",
                device_name="cpu",
            )
            self.assertTrue((root / "model.pt").is_file())
            self.assertTrue((root / "metrics.json").is_file())
            checkpoint = torch.load(root / "model.pt", map_location="cpu", weights_only=True)
            self.assertEqual(checkpoint["selected_epoch"], 1)
            self.assertIn("validation", checkpoint["selection_metric"])
        self.assertEqual(result["status"], "TRAINED_BASIC_TSHIRT_SEMANTIC_BASELINE")
        self.assertIn("test", result["evaluation"]["by_split"])
        self.assertIn("unseen_source", result["evaluation"]["by_split"])
        self.assertIn("FreeSewing", result["evaluation"]["by_source"])
        self.assertFalse(result["manifest_safe"]["feature_contract"]["includes_source_label"])
        self.assertEqual(result["selected_epoch"], 1)
        self.assertEqual(result["checkpoint_policy"], "BEST_VALIDATION_EDGE_PRIMARY_LEXICOGRAPHIC")


if __name__ == "__main__":
    unittest.main()
