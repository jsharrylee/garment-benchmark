from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from benchmark.drafting_semantics.basic_blocks import build_basic_block
from benchmark.drafting_semantics.basic_semantic_targets import (
    semantic_target_from_basic_block,
    semantic_target_from_pattern_document,
)
from benchmark.drafting_semantics.semantic_teacher_student import (
    MAX_COORDINATE_DIM,
    SEMANTIC_QUERY_INDEX,
    SEMANTIC_QUERY_INVENTORY,
    SEMANTIC_QUERY_KEYS,
    ModalityContractError,
    build_four_view_semantic_student,
    category_query_mask,
    query_coordinate_mask,
)
from benchmark.pattern_pipeline.four_view_semantic_inference import (
    CALIBRATION_SCHEMA_VERSION,
    CANONICAL_VIEW_ORDER,
    EDIT_CALIBRATION_SCHEMA_VERSION,
    FourViewFeatureBundle,
    StaticSemanticPrediction,
    _scaled_residual_plan,
    _select_semantic_projection_candidate,
    infer_provisional_basic_pattern,
    load_four_view_student_checkpoint,
    load_precomputed_four_view_features,
    predict_static_semantic_queries,
)
from benchmark.pattern_pipeline.semantic_editing import (
    LandmarkResidual,
    PathResidual,
    SemanticResidualPlan,
    apply_semantic_residual,
)
from benchmark.pattern_pipeline.validation import validate_pattern


def _config() -> dict[str, object]:
    return {
        "edge_feature_dim": 6,
        "spatial_feature_dim": 8,
        "global_feature_dim": 10,
        "width": 16,
        "token_dim": 8,
        "heads": 4,
        "encoder_layers": 1,
        "decoder_layers": 1,
        "feedforward_multiplier": 2,
        "dropout": 0.0,
        "max_views": 4,
        "panel_role_count": 0,
        "edge_role_count": 0,
    }


def _write_checkpoint(
    path: Path,
    *,
    coordinate_bias: np.ndarray | None = None,
    reliable_query: str | None = None,
    include_calibration: bool = True,
    edit_query: str | None = None,
    retained_query: str | None = None,
) -> None:
    import torch

    model = build_four_view_semantic_student(_config())
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.presence_head[-1].bias.fill_(9.0)
        if coordinate_bias is not None:
            model.coordinate_head[-1].bias.copy_(
                torch.as_tensor(coordinate_bias, dtype=torch.float32)
            )
    payload: dict[str, object] = {
        "schema_version": "test-student/v1",
        "stage": "four_view_student",
        "model_state": model.state_dict(),
        "model_config": _config(),
        "visual_feature_mode": "spatial",
        "query_keys": list(SEMANTIC_QUERY_KEYS),
        "inference_contract": "four_view_features_plus_category_only_no_pattern_graph",
    }
    if include_calibration:
        per_query = {reliable_query: 0.95} if reliable_query is not None else {}
        payload["coordinate_confidence_calibration"] = {
            "schema_version": CALIBRATION_SCHEMA_VERSION,
            "method": "validation_per_query_reliability",
            "per_query": per_query,
            "fallback": "FAIL_CLOSED",
        }
    if edit_query is not None or retained_query is not None:
        per_query = {}
        if edit_query is not None:
            per_query[edit_query] = {
                "allow_student_edit": True,
                "anchor_retention_weight": 0.0,
            }
        if retained_query is not None:
            per_query[retained_query] = {
                "allow_student_edit": False,
                "anchor_retention_weight": 0.9,
            }
        payload["semantic_edit_calibration"] = {
            "schema_version": EDIT_CALIBRATION_SCHEMA_VERSION,
            "method": "student_vs_default_anchor_validation_mae",
            "per_query": per_query,
            "fallback": "FAIL_CLOSED",
        }
    torch.save(payload, path)


def _features() -> FourViewFeatureBundle:
    return FourViewFeatureBundle(
        CANONICAL_VIEW_ORDER,
        spatial_features=np.zeros((4, 3, 8), dtype=np.float32),
    )


def _static_prediction_for_target(
    category: str,
    coordinates: np.ndarray,
    *,
    reliable_query: str,
) -> StaticSemanticPrediction:
    query_mask = np.asarray(category_query_mask(category), dtype=bool)
    coordinate_mask = np.asarray(query_coordinate_mask(), dtype=bool)
    coordinate_mask &= query_mask[:, None]
    confidence = np.zeros(len(SEMANTIC_QUERY_KEYS), dtype=np.float64)
    confidence[SEMANTIC_QUERY_INDEX[reliable_query]] = 0.95
    return StaticSemanticPrediction(
        category=category,
        presence_probability=np.where(query_mask, 0.99, 0.0),
        coordinates=np.asarray(coordinates, dtype=np.float64),
        coordinate_confidence=confidence,
        query_mask=query_mask,
        coordinate_mask=coordinate_mask,
        confidence_receipt={"status": "TEST_VALIDATION_RELIABILITY"},
    )


class FourViewSemanticInferenceTests(unittest.TestCase):
    def test_feature_loader_requires_canonical_explicit_order_and_rejects_pattern_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "explicitly ordered"):
            FourViewFeatureBundle(
                ("back", "front", "left", "right"),
                spatial_features=np.zeros((4, 2, 8), dtype=np.float32),
            )
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            missing_order = root / "missing_order.npz"
            np.savez(missing_order, spatial_features=np.zeros((4, 2, 8), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "does not embed view names"):
                load_precomputed_four_view_features(missing_order)

            reversed_order = root / "reversed_order.npz"
            np.savez(
                reversed_order,
                view_names=np.asarray(("back", "front", "left", "right")),
                spatial_features=np.zeros((4, 2, 8), dtype=np.float32),
            )
            with self.assertRaisesRegex(ValueError, "explicitly ordered"):
                load_precomputed_four_view_features(reversed_order)

            forbidden = root / "forbidden.npz"
            np.savez(
                forbidden,
                view_names=np.asarray(CANONICAL_VIEW_ORDER),
                spatial_features=np.zeros((4, 2, 8), dtype=np.float32),
                pattern_graph=np.zeros((1,), dtype=np.float32),
            )
            with self.assertRaises(ModalityContractError):
                load_precomputed_four_view_features(forbidden)

            legacy = root / "legacy.npz"
            np.savez(
                legacy,
                sample_ids=np.asarray(("first", "second")),
                features=np.zeros((2, 4, 2, 8), dtype=np.float32),
            )
            selected = load_precomputed_four_view_features(
                legacy,
                sample_id="second",
                generic_feature_kind="spatial",
                declared_view_order=CANONICAL_VIEW_ORDER,
            )
            self.assertEqual(selected.spatial_features.shape, (4, 2, 8))

    def test_loaded_student_rejects_any_pattern_input(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            checkpoint = Path(raw) / "student.pt"
            _write_checkpoint(checkpoint)
            loaded = load_four_view_student_checkpoint(checkpoint, device="cpu")
            with self.assertRaises(ModalityContractError):
                predict_static_semantic_queries(
                    loaded,
                    _features(),
                    category="tshirt",
                    pattern_input={"panels": []},
                )
            with self.assertRaises(ModalityContractError):
                infer_provisional_basic_pattern(
                    loaded,
                    _features(),
                    category="tshirt",
                    pattern_input=build_basic_block("tshirt").to_pattern_document(),
                )

    def test_prediction_to_planner_to_editor_bridge_for_all_categories(self) -> None:
        cases = (
            ("tshirt", "FNP"),
            ("pants", "front_knee_in"),
            ("skirt", "slit_end"),
        )
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            for category, landmark in cases:
                with self.subTest(category=category):
                    key = f"{category}:landmark:{landmark}"
                    anchor_target = semantic_target_from_basic_block(
                        build_basic_block(
                            category,
                            sample_id=f"{category}_provisional_inference_anchor",
                            metadata={"anchor_selection": "deterministic_category_default"},
                        ),
                        curve_samples=24,
                    )
                    index = SEMANTIC_QUERY_INDEX[key]
                    self.assertTrue(anchor_target.query_applicability[index])
                    bias = np.zeros(MAX_COORDINATE_DIM, dtype=np.float32)
                    bias[:2] = anchor_target.coordinates[index, :2]
                    bias[0] += 0.002
                    bias[1] -= 0.001
                    checkpoint = root / f"{category}_student.pt"
                    _write_checkpoint(
                        checkpoint,
                        coordinate_bias=bias,
                        reliable_query=key,
                    )
                    loaded = load_four_view_student_checkpoint(checkpoint, device="cpu")
                    result = infer_provisional_basic_pattern(
                        loaded,
                        _features(),
                        category=category,
                    )
                    self.assertEqual(result.receipt["status"], "APPLIED_VALIDATED")
                    self.assertIn(landmark, result.plan.landmark_residuals)
                    projection = result.receipt["semantic_projection"]
                    self.assertEqual(
                        projection["status"], "SELECTED_STRICT_IMPROVEMENT"
                    )
                    self.assertTrue(projection["strictly_improved"])
                    self.assertLess(
                        projection["selected_loss"], projection["anchor_loss"]
                    )
                    self.assertTrue(validate_pattern(result.document).accepted)
                    self.assertEqual(
                        result.receipt["topology"]["before_sha256"],
                        result.receipt["topology"]["after_sha256"],
                    )
                    rows = result.receipt["static_query_predictions"]
                    self.assertEqual(len(rows), len(SEMANTIC_QUERY_INVENTORY))
                    for row, query in zip(rows, SEMANTIC_QUERY_INVENTORY):
                        self.assertEqual(
                            tuple(row["predicted_coordinates"]), query.coordinate_names
                        )
                        self.assertEqual(
                            row["coordinate_channels_from"],
                            "STATIC_QUERY_SCHEMA_NO_GROUND_TRUTH_MASK",
                        )

                    pattern_path = root / f"{category}_pattern.json"
                    receipt_path = root / f"{category}_receipt.json"
                    result.save(pattern_path, receipt_path)
                    pattern_text = pattern_path.read_text(encoding="utf-8")
                    receipt_text = receipt_path.read_text(encoding="utf-8")
                    self.assertNotIn(str(checkpoint), pattern_text)
                    self.assertNotIn(str(checkpoint), receipt_text)
                    self.assertFalse(json.loads(receipt_text)["output"]["contains_source_paths"])

    def test_missing_validation_calibration_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            checkpoint = Path(raw) / "student.pt"
            _write_checkpoint(checkpoint, include_calibration=False)
            loaded = load_four_view_student_checkpoint(checkpoint, device="cpu")
            self.assertEqual(
                loaded.confidence_receipt["status"],
                "FAIL_CLOSED_NO_VALIDATION_CALIBRATION",
            )
            result = infer_provisional_basic_pattern(
                loaded, _features(), category="tshirt"
            )
            self.assertEqual(
                result.receipt["status"],
                "NO_ELIGIBLE_RESIDUALS_VALIDATED_ANCHOR",
            )
            self.assertFalse(result.plan.landmark_residuals)
            self.assertFalse(result.plan.path_residuals)
            self.assertTrue(np.all(result.prediction.coordinate_confidence == 0.0))

    def test_validation_edit_selector_gates_student_and_exposes_anchor_retention(self) -> None:
        editable = "skirt:landmark:front_side_hip"
        retained = "skirt:landmark:front_center_waist"
        with tempfile.TemporaryDirectory() as raw:
            checkpoint = Path(raw) / "student.pt"
            _write_checkpoint(
                checkpoint,
                reliable_query=editable,
                edit_query=editable,
                retained_query=retained,
            )
            # Retention also needs a coordinate reliability estimate.  Add it
            # to the otherwise minimal test checkpoint calibration.
            import torch

            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            payload["coordinate_confidence_calibration"]["per_query"][retained] = 0.95
            torch.save(payload, checkpoint)
            loaded = load_four_view_student_checkpoint(checkpoint, device="cpu")
            prediction = predict_static_semantic_queries(
                loaded, _features(), category="skirt"
            )
            editable_index = SEMANTIC_QUERY_INDEX[editable]
            retained_index = SEMANTIC_QUERY_INDEX[retained]
            unlisted_index = SEMANTIC_QUERY_INDEX["skirt:landmark:front_hem_center"]
            self.assertGreater(prediction.coordinate_confidence[editable_index], 0.9)
            self.assertGreater(prediction.coordinate_confidence[retained_index], 0.9)
            self.assertEqual(prediction.coordinate_confidence[unlisted_index], 0.0)
            self.assertGreater(
                prediction.edit_coordinate_confidence[editable_index], 0.9
            )
            self.assertEqual(
                prediction.edit_coordinate_confidence[retained_index], 0.0
            )
            self.assertEqual(
                prediction.edit_coordinate_confidence[unlisted_index], 0.0
            )
            self.assertGreater(
                prediction.anchor_retention_confidence[retained_index], 0.8
            )
            self.assertEqual(
                loaded.confidence_receipt["semantic_edit_selector"]["status"],
                "VALIDATION_EDIT_SELECTOR_AVAILABLE",
            )

    def test_projection_line_search_selects_reduced_validated_scale(self) -> None:
        category = "tshirt"
        query_key = "tshirt:landmark:FNP"
        query_index = SEMANTIC_QUERY_INDEX[query_key]
        anchor = build_basic_block(category).to_pattern_document(curve_samples=24)
        requested = SemanticResidualPlan(
            category=category,
            landmark_residuals={
                "FNP": LandmarkResidual(
                    dx_cm=4.0,
                    dy_cm=0.0,
                    influence_radius_cm=8.0,
                    confidence=0.95,
                )
            },
            path_residuals={
                "front_neckline": PathResidual(
                    chord_scale=1.08,
                    normal_scale=1.12,
                    normal_offset_cm=0.4,
                    confidence=0.95,
                )
            },
        )
        quarter_plan = _scaled_residual_plan(requested, 0.25)
        self.assertAlmostEqual(
            quarter_plan.path_residuals["front_neckline"].chord_scale,
            1.02,
        )
        desired = apply_semantic_residual(anchor, quarter_plan)
        desired_target = semantic_target_from_pattern_document(
            desired,
            category=category,
            source="test_desired_projection",
            provenance_status="PROVISIONAL_EXPERT_REVIEW",
            source_y_axis_down=True,
        )
        anchor_target = semantic_target_from_pattern_document(
            anchor,
            category=category,
            source="test_anchor_projection",
            provenance_status="PROVISIONAL_EXPERT_REVIEW",
            source_y_axis_down=True,
        )
        predicted_coordinates = anchor_target.coordinates.copy()
        predicted_coordinates[query_index, :2] = desired_target.coordinates[
            query_index, :2
        ]
        prediction = _static_prediction_for_target(
            category,
            predicted_coordinates,
            reliable_query=query_key,
        )
        selection = _select_semantic_projection_candidate(
            anchor,
            requested,
            prediction,
            category=category,
            confidence_threshold=0.55,
        )
        self.assertEqual(selection.selected_scale, 0.25)
        self.assertIsNone(selection.rejection_reason)
        self.assertIsNotNone(selection.validation)
        self.assertTrue(selection.validation.accepted)
        self.assertLess(selection.selected_loss, selection.anchor_loss)
        self.assertEqual(
            len(selection.plan.landmark_residuals),
            len(requested.landmark_residuals),
        )

    def test_projection_rejects_every_edit_when_anchor_is_closer(self) -> None:
        category = "tshirt"
        query_key = "tshirt:landmark:FNP"
        anchor = build_basic_block(category).to_pattern_document(curve_samples=24)
        anchor_target = semantic_target_from_pattern_document(
            anchor,
            category=category,
            source="test_anchor_projection",
            provenance_status="PROVISIONAL_EXPERT_REVIEW",
            source_y_axis_down=True,
        )
        prediction = _static_prediction_for_target(
            category,
            anchor_target.coordinates.copy(),
            reliable_query=query_key,
        )
        requested = SemanticResidualPlan(
            category=category,
            landmark_residuals={
                "FNP": LandmarkResidual(
                    dx_cm=3.0,
                    dy_cm=0.0,
                    influence_radius_cm=8.0,
                    confidence=0.95,
                )
            },
        )
        selection = _select_semantic_projection_candidate(
            anchor,
            requested,
            prediction,
            category=category,
            confidence_threshold=0.55,
        )
        self.assertEqual(selection.selected_scale, 0.0)
        self.assertEqual(
            selection.rejection_reason,
            "NO_CANDIDATE_IMPROVED_SEMANTIC_PROJECTION_LOSS",
        )
        self.assertEqual(selection.anchor_loss, 0.0)
        self.assertEqual(selection.selected_loss, 0.0)
        self.assertFalse(selection.plan.landmark_residuals)
        self.assertFalse(selection.plan.path_residuals)
        self.assertEqual(selection.document, anchor)
        self.assertTrue(all(row["validation_accepted"] for row in selection.candidates))

    def test_projection_rejects_collateral_change_to_anchor_preferred_query(self) -> None:
        category = "tshirt"
        edit_key = "tshirt:landmark:FNP"
        retained_key = "tshirt:path:front_neckline"
        edit_index = SEMANTIC_QUERY_INDEX[edit_key]
        retained_index = SEMANTIC_QUERY_INDEX[retained_key]
        anchor = build_basic_block(category).to_pattern_document(curve_samples=24)
        requested = SemanticResidualPlan(
            category=category,
            landmark_residuals={
                "FNP": LandmarkResidual(
                    dx_cm=3.0,
                    dy_cm=0.0,
                    influence_radius_cm=8.0,
                    confidence=0.95,
                )
            },
        )
        full_candidate = apply_semantic_residual(anchor, requested)
        anchor_target = semantic_target_from_pattern_document(
            anchor,
            category=category,
            source="test_anchor_retention",
            provenance_status="PROVISIONAL_EXPERT_REVIEW",
            source_y_axis_down=True,
        )
        candidate_target = semantic_target_from_pattern_document(
            full_candidate,
            category=category,
            source="test_student_target",
            provenance_status="PROVISIONAL_EXPERT_REVIEW",
            source_y_axis_down=True,
        )
        query_mask = np.asarray(category_query_mask(category), dtype=bool)
        coordinate_mask = np.asarray(query_coordinate_mask(), dtype=bool)
        coordinate_mask &= query_mask[:, None]
        predicted = anchor_target.coordinates.copy()
        predicted[edit_index, :2] = candidate_target.coordinates[edit_index, :2]
        confidence = np.zeros(len(SEMANTIC_QUERY_KEYS), dtype=np.float64)
        confidence[edit_index] = 0.95
        retention = np.zeros_like(confidence)
        retention[retained_index] = 0.95
        prediction = StaticSemanticPrediction(
            category=category,
            presence_probability=np.where(query_mask, 0.99, 0.0),
            coordinates=predicted,
            coordinate_confidence=confidence,
            query_mask=query_mask,
            coordinate_mask=coordinate_mask,
            confidence_receipt={"status": "TEST_VALIDATION_SELECTOR"},
            anchor_retention_confidence=retention,
        )
        selection = _select_semantic_projection_candidate(
            anchor,
            requested,
            prediction,
            category=category,
            confidence_threshold=0.55,
        )
        self.assertEqual(selection.selected_scale, 0.0)
        self.assertEqual(
            selection.rejection_reason,
            "NO_CANDIDATE_IMPROVED_SEMANTIC_PROJECTION_LOSS",
        )
        self.assertEqual(selection.document, anchor)


if __name__ == "__main__":
    unittest.main()
