from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmark.drafting_semantics.counterfactual_pairs import file_sha256
from benchmark.drafting_semantics.pattern_visual_effects import (
    EFFECT_RECEIPT_SCHEMA_VERSION,
    CounterfactualVisualExample,
    ObservableAxisCalibration,
    PatternVisualEffectContractError,
    TSHIRT_DECODER_RESIDUAL_PARAMETER_NAMES,
    TSHIRT_ELEMENT_QUERIES,
    TSHIRT_OBSERVABLE_AXES,
    TSHIRT_OBSERVABLE_AXIS_NAMES,
    TSHIRT_OBSERVABLE_SCHEMA_VERSION,
    TSHIRT_OBSERVABLE_TO_DECODER_RELATIONS,
    TSHIRT_SEMANTIC_PARAMETERS,
    TSHIRT_SEMANTIC_PARAMETER_NAMES,
    adapt_observable_residuals_to_decoder,
    assert_base_group_split_integrity,
    build_pattern_inverse_residual,
    build_pattern_visual_effect_bridge,
    decode_tshirt_observable_residual,
    decode_tshirt_semantic_residual,
    load_pattern_only_counterfactual_manifest,
    validate_effect_render_receipt,
    validate_inverse_input_contract,
)


_HASH = "a" * 64


def _clean_pair(pair_id: str = "pair", *, baseline_hash: str = _HASH) -> dict:
    return {
        "pair_id": pair_id,
        "source": "unit",
        "intervention_parameter": "body_length_cm",
        "observable_axis": "body_length_cm",
        "baseline_value": 64.0,
        "intervention_value": 66.0,
        "baseline_canonical_pattern": "patterns/base.json",
        "intervention_canonical_pattern": f"patterns/{pair_id}.json",
        "baseline_pattern_sha256": baseline_hash,
        "intervention_pattern_sha256": ("b" if pair_id == "pair" else "c") * 64,
        "unchanged_state_fingerprint": "d" * 64,
        "expected_affected_elements": ["center_front", "center_back"],
        "ground_truth_semantic_delta": {"curves": {"front/center_front": {}}},
        "contract_validation": "PASS",
        "pattern_geometry_changed": True,
        "topology_stable": True,
        "semantic_delta_coverage": {"status": "FULL"},
        "pattern_only": True,
        "render_status": "PENDING_VALIDATED_SIMULATOR",
    }


class PatternVisualEffectSchemaTests(unittest.TestCase):
    def test_observable_and_decoder_schemas_are_distinct_and_fixed(self):
        self.assertEqual(
            TSHIRT_OBSERVABLE_AXIS_NAMES,
            (
                "neck_width_cm",
                "front_neck_depth_cm",
                "shoulder_slope_deg",
                "armhole_depth_cm",
                "body_length_cm",
                "sleeve_cap_height_cm",
                "sleeve_length_cm",
                "sleeve_width_cm",
            ),
        )
        self.assertEqual(len(TSHIRT_OBSERVABLE_AXES), 8)
        self.assertEqual(len(TSHIRT_SEMANTIC_PARAMETERS), 8)
        self.assertEqual(len(set(TSHIRT_SEMANTIC_PARAMETER_NAMES)), 8)
        self.assertEqual(
            TSHIRT_SEMANTIC_PARAMETER_NAMES,
            (
                "chest_ease_cm",
                "body_length_cm",
                "neck_width_cm",
                "front_neck_depth_cm",
                "shoulder_drop_cm",
                "armhole_depth_cm",
                "sleeve_length_cm",
                "sleeve_ease_cm",
            ),
        )
        self.assertEqual(len({item.name for item in TSHIRT_ELEMENT_QUERIES}), len(TSHIRT_ELEMENT_QUERIES))
        self.assertEqual(len(TSHIRT_ELEMENT_QUERIES), 15)
        required = {
            "front_neckline",
            "back_neckline",
            "front_armhole",
            "back_armhole",
            "sleeve_head",
            "front_side_seam",
            "back_side_seam",
        }
        self.assertTrue(required <= {item.name for item in TSHIRT_ELEMENT_QUERIES})
        all_elements = {item.name for item in TSHIRT_ELEMENT_QUERIES}
        for parameter in TSHIRT_SEMANTIC_PARAMETERS:
            self.assertTrue(set(parameter.affected_elements) <= all_elements)
        for axis in TSHIRT_OBSERVABLE_AXES:
            self.assertTrue(set(axis.affected_surface_elements) <= all_elements)
        self.assertNotEqual(TSHIRT_OBSERVABLE_AXIS_NAMES, TSHIRT_DECODER_RESIDUAL_PARAMETER_NAMES)
        relations = {item.observable_axis: item for item in TSHIRT_OBSERVABLE_TO_DECODER_RELATIONS}
        self.assertEqual(relations["shoulder_slope_deg"].decoder_parameters, ("shoulder_drop_cm",))
        self.assertEqual(
            relations["sleeve_cap_height_cm"].decoder_parameters,
            ("sleeve_ease_cm", "armhole_depth_cm"),
        )
        self.assertEqual(
            relations["sleeve_width_cm"].decoder_parameters,
            ("bicep_ease_cm", "sleeve_hem_reduction_cm"),
        )
        self.assertTrue(all("IDENTITY" not in item.conversion_kind for item in relations.values()))
        expected_decoder_targets = tuple(
            dict.fromkeys(
                target
                for axis in TSHIRT_OBSERVABLE_AXIS_NAMES
                for target in relations[axis].decoder_parameters
            )
        )
        self.assertEqual(TSHIRT_DECODER_RESIDUAL_PARAMETER_NAMES, expected_decoder_targets)

    def test_pattern_only_loader_derives_base_group_and_maps_semantic_parameter(self):
        with tempfile.TemporaryDirectory() as value:
            path = Path(value) / "manifest.json"
            first = _clean_pair("first")
            second = _clean_pair("second")
            first.pop("observable_axis")
            first["intervention_parameter"] = "shirt.length"
            payload = {
                "pattern_only": True,
                "true_four_view_pair_count": 0,
                "records": [first, second],
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            examples = load_pattern_only_counterfactual_manifest(
                path, split_assignments={"first": "train", "second": "train"}
            )
        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[0].observable_axis, "body_length_cm")
        self.assertEqual(examples[0].semantic_parameter, "body_length_cm")
        self.assertEqual(examples[0].base_group_id, examples[1].base_group_id)
        self.assertEqual(examples[0].source_delta, 2.0)

    def test_base_group_cannot_cross_splits(self):
        with tempfile.TemporaryDirectory() as value:
            path = Path(value) / "manifest.json"
            payload = {
                "pattern_only": True,
                "true_four_view_pair_count": 0,
                "records": [_clean_pair("first"), _clean_pair("second")],
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(PatternVisualEffectContractError, "crosses splits"):
                load_pattern_only_counterfactual_manifest(
                    path, split_assignments={"first": "train", "second": "test"}
                )

    def test_direct_base_group_integrity_check_rejects_leakage(self):
        common = dict(
            pair_id="a",
            base_group_id="base",
            source="unit",
            source_parameter="length",
            observable_axis="body_length_cm",
            baseline_value=1.0,
            intervention_value=2.0,
            baseline_pattern_path="a",
            intervention_pattern_path="b",
            baseline_pattern_sha256="a" * 64,
            intervention_pattern_sha256="b" * 64,
            unchanged_state_fingerprint="c" * 64,
            expected_elements=(),
            semantic_delta={},
        )
        first = CounterfactualVisualExample(split="train", **common)
        second = CounterfactualVisualExample(split="test", **{**common, "pair_id": "b"})
        with self.assertRaisesRegex(PatternVisualEffectContractError, "crosses splits"):
            assert_base_group_split_integrity((first, second))

    def test_inverse_contract_rejects_target_dsl_and_unknown_payload(self):
        clean = {
            "target_views": object(),
            "anchor_elements": object(),
            "anchor_observables": object(),
            "element_valid": object(),
        }
        receipt = validate_inverse_input_contract(clean)
        self.assertFalse(receipt["target_dsl_used"])
        with self.assertRaisesRegex(PatternVisualEffectContractError, "forbidden"):
            validate_inverse_input_contract({**clean, "target_dsl": object()})
        with self.assertRaisesRegex(PatternVisualEffectContractError, "missing"):
            validate_inverse_input_contract({"target_views": object()})

    def test_observable_adapter_never_assumes_same_name_identity(self):
        with self.assertRaisesRegex(PatternVisualEffectContractError, "calibration is missing"):
            adapt_observable_residuals_to_decoder(
                {"neck_width_cm": 1.0}, calibrations={}
            )
        result = adapt_observable_residuals_to_decoder(
            {"neck_width_cm": 1.0},
            calibrations={
                "neck_width_cm": {
                    "decoder_coefficients": {"neck_width_cm": 0.5},
                    "calibration_id": "local-linear-neck-v1",
                    "scope_id": "gcdv2-tshirt-neutral-body",
                    "evidence": "finite-difference decoder sweep",
                }
            },
        )
        self.assertEqual(result.decoder_residuals_cm, {"neck_width_cm": 0.5})
        self.assertFalse(result.receipt["identity_assumption_used"])
        self.assertEqual(result.receipt["status"], "PASS_EXPLICIT_CALIBRATION")

    def test_observable_adapter_converts_degrees_with_explicit_context(self):
        result = adapt_observable_residuals_to_decoder(
            {"shoulder_slope_deg": 2.0},
            calibrations={
                "shoulder_slope_deg": ObservableAxisCalibration(
                    observable_axis="shoulder_slope_deg",
                    decoder_coefficients=(("shoulder_drop_cm", 0.1),),
                    calibration_id="shoulder-run-13cm-v1",
                    scope_id="decoder-default-body",
                    evidence="tan-angle local linearization at the default shoulder run",
                )
            },
        )
        self.assertAlmostEqual(result.decoder_residuals_cm["shoulder_drop_cm"], 0.2)

    def test_multi_parameter_calibration_is_complete_and_target_restricted(self):
        with self.assertRaisesRegex(PatternVisualEffectContractError, "omits required"):
            ObservableAxisCalibration(
                observable_axis="sleeve_width_cm",
                decoder_coefficients=(("bicep_ease_cm", 0.5),),
                calibration_id="bad",
                scope_id="unit",
                evidence="incomplete fit",
            )
        with self.assertRaisesRegex(PatternVisualEffectContractError, "unsupported"):
            ObservableAxisCalibration(
                observable_axis="neck_width_cm",
                decoder_coefficients=(("body_length_cm", 1.0),),
                calibration_id="bad",
                scope_id="unit",
                evidence="wrong target",
            )


class EffectReceiptTests(unittest.TestCase):
    @staticmethod
    def _write(root: Path, name: str, *, image: bool = False, vertex_count: int | None = None) -> dict:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode("utf-8"))
        result = {"path": name, "sha256": file_sha256(path)}
        if image:
            result["image_size"] = [32, 48]
        if vertex_count is not None:
            result["vertex_count"] = vertex_count
        return result

    def _receipt(self, root: Path) -> tuple[dict, dict]:
        pair = _clean_pair()
        members = {}
        camera_intrinsics = "1" * 64
        camera_extrinsics = {
            view: str(index + 2) * 64
            for index, view in enumerate(("front", "back", "left", "right"))
        }
        for member in ("baseline", "intervention"):
            views = {}
            for view in ("front", "back", "left", "right"):
                primary = self._write(root, f"{member}/{view}/rgba.bin", image=True)
                primary.update(
                    {
                        "camera_intrinsics_sha256": camera_intrinsics,
                        "camera_extrinsics_sha256": camera_extrinsics[view],
                        "passes": {
                            name: self._write(
                                root, f"{member}/{view}/{name}.bin", image=True
                            )
                            for name in ("silhouette", "depth", "normal", "panel_id")
                        },
                    }
                )
                views[view] = primary
            members[member] = {
                "mesh": self._write(root, f"{member}/mesh.bin", vertex_count=17),
                "views": views,
            }
        effects = {
            view: {
                name: self._write(root, f"effects/{view}/{name}.bin", image=True)
                for name in ("flow", "effect_mask")
            }
            for view in ("front", "back", "left", "right")
        }
        receipt = {
            "schema_version": EFFECT_RECEIPT_SCHEMA_VERSION,
            "pair_id": pair["pair_id"],
            "unchanged_state_fingerprint": pair["unchanged_state_fingerprint"],
            "simulator_fidelity": {
                "status": "PASS",
                "profile_id": "validated-simulator",
                "version": "1.0",
                "reference_receipt_sha256": "e" * 64,
            },
            "fixed_state": {
                "body_sha256": "1" * 64,
                "material_sha256": "2" * 64,
                "pose_sha256": "3" * 64,
                "camera_rig_sha256": "4" * 64,
                "simulator_sha256": "5" * 64,
            },
            "correspondence": {
                "status": "PASS",
                "topology_stable": True,
                "vertex_count": 17,
                "panel_uv": self._write(root, "correspondence/panel_uv.bin"),
                "vertex_map": self._write(root, "correspondence/vertex_map.bin"),
            },
            "members": members,
            "effects": effects,
        }
        return pair, receipt

    def test_effect_receipt_requires_all_causal_evidence(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            pair, receipt = self._receipt(root)
            result = validate_effect_render_receipt(pair, receipt, root=root)
            self.assertEqual(result["status"], "PASS_VALIDATED_CAUSAL_EFFECT_RENDER")
            self.assertFalse(result["target_dsl_used_for_inverse"])
            receipt["simulator_fidelity"]["status"] = "UNKNOWN"
            with self.assertRaisesRegex(PatternVisualEffectContractError, "fidelity"):
                validate_effect_render_receipt(pair, receipt, root=root)

    def test_effect_receipt_rejects_camera_drift_and_missing_flow(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            pair, receipt = self._receipt(root)
            receipt["members"]["intervention"]["views"]["front"][
                "camera_extrinsics_sha256"
            ] = "f" * 64
            with self.assertRaisesRegex(PatternVisualEffectContractError, "camera differs"):
                validate_effect_render_receipt(pair, receipt, root=root)
            pair, receipt = self._receipt(root)
            del receipt["effects"]["left"]["flow"]
            with self.assertRaisesRegex(PatternVisualEffectContractError, "file descriptor"):
                validate_effect_render_receipt(pair, receipt, root=root)


class PatternVisualEffectModelTests(unittest.TestCase):
    def test_forward_effect_bridge_shapes(self):
        import torch

        torch.manual_seed(5)
        query_count = len(TSHIRT_ELEMENT_QUERIES)
        observable_count = len(TSHIRT_OBSERVABLE_AXES)
        model = build_pattern_visual_effect_bridge(
            {
                "view_dim": 12,
                "element_dim": 10,
                "hidden_dim": 16,
                "heads": 4,
                "layers": 1,
                "max_spatial_tokens": 7,
                "dropout": 0.0,
            }
        ).eval()
        views = torch.randn(2, 4, 5, 12)
        baseline = torch.randn(2, query_count, 10)
        intervention = baseline + 0.1 * torch.randn_like(baseline)
        valid = torch.ones(2, query_count, dtype=torch.bool)
        with torch.no_grad():
            output = model(
                views,
                baseline,
                intervention,
                valid,
                capture_attention=True,
            )
        self.assertEqual(output["effect_map_logits"].shape, (2, query_count, 4, 5))
        self.assertEqual(output["visual_delta_prediction"].shape, (2, 4, 5, 12))
        self.assertEqual(output["intervention_logits"].shape, (2, observable_count))
        self.assertEqual(output["observable_delta"].shape, (2, observable_count))
        self.assertEqual(model.output_schema_version, TSHIRT_OBSERVABLE_SCHEMA_VERSION)
        self.assertEqual(output["cross_attention"].shape, (2, 4, query_count, 4, 5))
        self.assertTrue(all(torch.isfinite(value).all() for value in output.values()))

    def test_inverse_residual_payload_is_target_dsl_free_and_masked(self):
        import torch

        torch.manual_seed(7)
        query_count = len(TSHIRT_ELEMENT_QUERIES)
        observable_count = len(TSHIRT_OBSERVABLE_AXES)
        model = build_pattern_inverse_residual(
            {
                "view_dim": 12,
                "element_dim": 10,
                "hidden_dim": 16,
                "heads": 4,
                "layers": 1,
                "max_spatial_tokens": 7,
                "dropout": 0.0,
            }
        ).eval()
        mask = torch.ones(2, observable_count, dtype=torch.bool)
        mask[:, -2:] = False
        payload = {
            "target_views": torch.randn(2, 4, 5, 12),
            "anchor_elements": torch.randn(2, query_count, 10),
            "anchor_observables": torch.zeros(2, observable_count),
            "element_valid": torch.ones(2, query_count, dtype=torch.bool),
            "observable_mask": mask,
        }
        with torch.no_grad():
            output = model.forward_payload(payload, capture_attention=True)
        self.assertEqual(output["observable_delta"].shape, (2, observable_count))
        torch.testing.assert_close(
            output["observable_delta"][:, -2:], torch.zeros(2, 2)
        )
        self.assertEqual(model.output_schema_version, TSHIRT_OBSERVABLE_SCHEMA_VERSION)
        self.assertEqual(output["cross_attention"].shape, (2, 4, query_count, 4, 5))
        with self.assertRaisesRegex(PatternVisualEffectContractError, "forbidden"):
            model.forward_payload({**payload, "target_dsl": torch.zeros(1)})

    def test_decoder_residual_helper_projects_and_preserves_constraints(self):
        from benchmark.drafting_semantics.tshirt_parametric_decoder import (
            TShirtDraftParameters,
        )

        anchor = TShirtDraftParameters()
        result = decode_tshirt_semantic_residual(
            anchor,
            {
                "body_length_cm": 2.0,
                "front_neck_depth_cm": 1.0,
                "sleeve_ease_cm": 10.0,
            },
            pattern_id="unit_residual",
        )
        self.assertEqual(result.parameters.body_length_cm, 66.0)
        self.assertEqual(result.parameters.front_neck_depth_cm, 9.2)
        self.assertEqual(result.parameters.sleeve_ease_cm, 4.0)
        self.assertIn("sleeve_ease_cm", result.receipt["box_projected_parameter_names"])
        self.assertTrue(result.receipt["constraint"]["converged"])
        self.assertFalse(result.receipt["input_contract"]["target_dsl_used"])
        result.graph.validate()
        with self.assertRaisesRegex(PatternVisualEffectContractError, "unknown semantic"):
            decode_tshirt_semantic_residual(anchor, {"raw_control_point": 1.0})

    def test_observable_decode_uses_adapter_before_solver(self):
        from benchmark.drafting_semantics.tshirt_parametric_decoder import (
            TShirtDraftParameters,
        )

        calibrations = {
            "sleeve_cap_height_cm": {
                "decoder_coefficients": {
                    "sleeve_ease_cm": 0.25,
                    "armhole_depth_cm": 0.10,
                },
                "calibration_id": "cap-local-jacobian-v1",
                "scope_id": "decoder-default-body",
                "evidence": "finite-difference constrained decoder sweep",
            },
            "sleeve_width_cm": {
                "decoder_coefficients": {
                    "bicep_ease_cm": 0.50,
                    "sleeve_hem_reduction_cm": -0.20,
                },
                "calibration_id": "width-local-jacobian-v1",
                "scope_id": "decoder-default-body",
                "evidence": "finite-difference constrained decoder sweep",
            },
        }
        result = decode_tshirt_observable_residual(
            TShirtDraftParameters(),
            {"sleeve_cap_height_cm": 2.0, "sleeve_width_cm": 2.0},
            calibrations=calibrations,
            pattern_id="unit_observable_residual",
        )
        self.assertEqual(result.parameters.sleeve_ease_cm, 1.0)
        self.assertEqual(result.parameters.armhole_depth_cm, 21.7)
        self.assertEqual(result.parameters.bicep_ease_cm, 7.0)
        self.assertEqual(result.parameters.sleeve_hem_reduction_cm, 2.1)
        self.assertTrue(result.receipt["constraint"]["converged"])
        self.assertFalse(result.receipt["adapter"]["identity_assumption_used"])
        result.graph.validate()


if __name__ == "__main__":
    unittest.main()
