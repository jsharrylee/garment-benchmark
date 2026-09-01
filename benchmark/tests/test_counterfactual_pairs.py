from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from benchmark.drafting_semantics.counterfactual_pairs import (
    CounterfactualContractError,
    assert_single_intervention,
    counterfactual_training_eligibility,
    fixed_state_fingerprint,
    file_sha256,
    flatten_garmentcode_values,
    freesewing_topology_signature,
    garmentcode_topology_signature,
    pair_contract,
    semantic_delta_coverage,
    semantic_ground_truth_delta,
    set_garmentcode_value,
    validate_four_view_receipt,
)
from benchmark.scripts.audit_drafting_counterfactual_training_corpus import audit_manifest


class CounterfactualPairTests(unittest.TestCase):
    def setUp(self) -> None:
        self.design = {
            "shirt": {
                "length": {"v": 1.2, "range": [0.5, 3.5]},
                "width": {"v": 1.05, "range": [1.0, 1.3]},
            },
            "sleeve": {"length": {"v": 0.3, "range": [0.1, 1.15]}},
        }

    def test_garmentcode_flatten_and_single_value_update(self) -> None:
        baseline = flatten_garmentcode_values(self.design)
        variant = copy.deepcopy(self.design)
        set_garmentcode_value(variant, "shirt.length", 1.6)
        changed = flatten_garmentcode_values(variant)
        self.assertEqual(
            baseline,
            {"shirt.length": 1.2, "shirt.width": 1.05, "sleeve.length": 0.3},
        )
        self.assertEqual(
            assert_single_intervention(baseline, changed, "shirt.length"),
            {"baseline": 1.2, "intervention": 1.6},
        )

    def test_two_changes_are_rejected(self) -> None:
        baseline = flatten_garmentcode_values(self.design)
        changed = dict(baseline, **{"shirt.length": 1.6, "shirt.width": 1.2})
        with self.assertRaisesRegex(CounterfactualContractError, "observed"):
            assert_single_intervention(baseline, changed, "shirt.length")

    def test_unknown_parameter_is_rejected(self) -> None:
        with self.assertRaises(KeyError):
            set_garmentcode_value(self.design, "shirt.missing", 1.0)

    def test_fixed_state_hash_is_order_independent_and_sensitive(self) -> None:
        first = fixed_state_fingerprint(
            body={"bust": 90, "waist": 70},
            material={"mass": 0.2, "damping": 3},
            simulator={"seed": 7},
            cameras={"front": [0, -1, 0], "back": [0, 1, 0]},
        )
        reordered = fixed_state_fingerprint(
            body={"waist": 70, "bust": 90},
            material={"damping": 3, "mass": 0.2},
            simulator={"seed": 7},
            cameras={"back": [0, 1, 0], "front": [0, -1, 0]},
        )
        changed = fixed_state_fingerprint(
            body={"bust": 91, "waist": 70},
            material={"mass": 0.2, "damping": 3},
            simulator={"seed": 7},
            cameras={"front": [0, -1, 0], "back": [0, 1, 0]},
        )
        self.assertEqual(first, reordered)
        self.assertNotEqual(first, changed)

    def test_garmentcode_topology_signature_ignores_geometry(self) -> None:
        baseline = {
            "pattern": {
                "panels": {
                    "front": {"vertices": [[0, 0]], "edges": [{}, {}]},
                    "back": {"vertices": [[5, 8]], "edges": [{}, {}, {}]},
                },
                "stitches": [[{}, {}]],
            }
        }
        moved = copy.deepcopy(baseline)
        moved["pattern"]["panels"]["front"]["vertices"] = [[99, -11]]
        self.assertEqual(
            garmentcode_topology_signature(baseline),
            garmentcode_topology_signature(moved),
        )

    def test_freesewing_topology_excludes_hidden_source_parts(self) -> None:
        raw = {
            "parts": {
                "visible": {
                    "hidden": False,
                    "paths": {
                        "seam": {
                            "operations": [
                                {"type": "move"},
                                {"type": "line"},
                                {"type": "curve"},
                                {"type": "close"},
                            ]
                        }
                    },
                },
                "source": {
                    "hidden": True,
                    "paths": {"seam": {"operations": [{"type": "line"}]}},
                },
            }
        }
        self.assertEqual(
            freesewing_topology_signature(raw),
            {"panel_count": 1, "panel_edge_counts": {"visible": 2}},
        )

    def test_pair_contract_rejects_state_drift(self) -> None:
        with self.assertRaisesRegex(CounterfactualContractError, "state differs"):
            pair_contract(
                pair_id="example",
                source="test",
                expected_parameter="length",
                baseline_inputs={"length": 1.0},
                intervention_inputs={"length": 2.0},
                baseline_state_fingerprint="a",
                intervention_state_fingerprint="b",
                baseline_topology={"panel_count": 1},
                intervention_topology={"panel_count": 1},
            )

    def test_control_only_curve_delta_is_retained_and_covered(self) -> None:
        baseline_curve = {
            "primitive_count": 1,
            "total_length_cm": 10.0,
            "total_chord_cm": 9.0,
            "curvature_types": ["quadratic"],
            "primitive_geometry": [
                {
                    "source_curvature": {
                        "type": "quadratic",
                        "params": [[0.3, 0.5]],
                    }
                }
            ],
        }
        intervention_curve = copy.deepcopy(baseline_curve)
        intervention_curve["primitive_geometry"][0]["source_curvature"]["params"] = [
            [0.7, 0.5]
        ]
        delta = semantic_ground_truth_delta(
            {"landmarks": {}, "curves": {"front/panel/neckline": baseline_curve}},
            {
                "landmarks": {},
                "curves": {"front/panel/neckline": intervention_curve},
            },
        )
        self.assertEqual(delta["changed_curve_group_count"], 1)
        self.assertEqual(delta["changed_control_geometry_group_count"], 1)
        self.assertTrue(
            delta["curves"]["front/panel/neckline"]["control_geometry_changed"]
        )
        coverage = semantic_delta_coverage(
            delta, ["neckline", "neckline_control_point"]
        )
        self.assertEqual(coverage["status"], "FULL")

    def test_semantic_coverage_does_not_hide_missing_expected_effect(self) -> None:
        coverage = semantic_delta_coverage(
            {
                "landmarks": {},
                "curves": {},
                "changed_landmark_group_count": 0,
                "changed_curve_group_count": 0,
            },
            ["neckline"],
        )
        self.assertEqual(coverage["status"], "NONE")
        self.assertEqual(coverage["missing_expected_elements"], ["neckline"])

    def test_four_view_receipt_requires_distinct_files_and_matching_state(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            members = {}
            for member in ("baseline", "intervention"):
                views = {}
                for view in ("front", "back", "left", "right"):
                    path = root / f"{member}_{view}.png"
                    path.write_bytes(f"{member}/{view}".encode("ascii"))
                    views[view] = {
                        "path": path.name,
                        "sha256": file_sha256(path),
                        "image_size": [384, 384],
                    }
                members[member] = {"views": views}
            record = {
                "pair_id": "pair",
                "unchanged_state_fingerprint": "fixed",
            }
            receipt = {
                "pair_id": "pair",
                "unchanged_state_fingerprint": "fixed",
                "members": members,
            }
            result = validate_four_view_receipt(record, receipt, root=root)
            self.assertEqual(result["render_status"], "VALIDATED")
            receipt["unchanged_state_fingerprint"] = "drifted"
            with self.assertRaisesRegex(CounterfactualContractError, "fingerprint"):
                validate_four_view_receipt(record, receipt, root=root)

    def test_training_eligibility_is_strict_conjunction(self) -> None:
        clean = {
            "contract_validation": "PASS",
            "pattern_geometry_changed": True,
            "topology_stable": True,
            "semantic_delta_coverage": {"status": "FULL"},
        }
        self.assertTrue(counterfactual_training_eligibility(clean)["training_eligible"])
        for field, value, reason in (
            ("contract_validation", "FAIL", "INPUT_CONTRACT_NOT_PASS"),
            ("pattern_geometry_changed", False, "PATTERN_GEOMETRY_UNCHANGED"),
            ("topology_stable", False, "TOPOLOGY_CHANGED"),
        ):
            record = {**clean, field: value}
            result = counterfactual_training_eligibility(record)
            self.assertFalse(result["training_eligible"])
            self.assertIn(reason, result["quarantine_reasons"])
        result = counterfactual_training_eligibility(
            {**clean, "semantic_delta_coverage": {"status": "PARTIAL"}}
        )
        self.assertEqual(
            result["quarantine_reasons"], ["SEMANTIC_DELTA_COVERAGE_NOT_FULL"]
        )

    def test_audit_preserves_source_and_aggregates_overlapping_reasons(self) -> None:
        base = {
            "source": "unit",
            "intervention_parameter": "length",
            "contract_validation": "PASS",
            "pattern_geometry_changed": True,
            "topology_stable": True,
            "semantic_delta_coverage": {"status": "FULL"},
        }
        source = {
            "records": [
                {**base, "pair_id": "clean"},
                {**base, "pair_id": "topology", "topology_stable": False},
                {
                    **base,
                    "pair_id": "two_reasons",
                    "topology_stable": False,
                    "semantic_delta_coverage": {"status": "PARTIAL"},
                },
            ]
        }
        before = copy.deepcopy(source)
        result = audit_manifest(source)
        self.assertEqual(source, before)
        self.assertEqual(result["summary"]["accepted_count"], 1)
        self.assertEqual(result["summary"]["quarantined_count"], 2)
        self.assertEqual(result["summary"]["accepted_source_counts"], {"unit": 1})
        self.assertEqual(result["summary"]["quarantined_source_counts"], {"unit": 2})
        self.assertEqual(
            result["summary"]["quarantine_reason_counts"],
            {"SEMANTIC_DELTA_COVERAGE_NOT_FULL": 1, "TOPOLOGY_CHANGED": 2},
        )
        self.assertTrue(result["accepted"][0]["training_eligible"])
        self.assertFalse(result["quarantined"][0]["training_eligible"])


if __name__ == "__main__":
    unittest.main()
