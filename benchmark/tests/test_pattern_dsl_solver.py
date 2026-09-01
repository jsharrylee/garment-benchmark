from __future__ import annotations

import unittest

import numpy as np

from benchmark.drafting_semantics.schema import EDGE_ROLES
from benchmark.gcdv2_exact.pattern_dsl_solver import (
    check_cyclic_role_grammar,
    derive_landmark_facts,
    project_cyclic_role_sequence,
    project_cyclic_roles,
    solve_symmetric_seam_matching,
    symbolic_project_and_verify,
)


class PatternDSLSolverTests(unittest.TestCase):
    def test_exact_cyclic_viterbi_repairs_wraparound_and_internal_transition(self) -> None:
        # Identity transitions are legal, as is the directed 0 -> 1 -> 2 -> 0 cycle.
        allowed = np.eye(3, dtype=bool)
        allowed[0, 1] = allowed[1, 2] = allowed[2, 0] = True
        logits = np.asarray(
            [
                [5.0, 0.0, 0.0],
                [0.0, 5.0, 4.0],
                [0.0, 5.0, 4.0],
            ]
        )
        raw = logits.argmax(-1)
        self.assertEqual(raw.tolist(), [0, 1, 1])
        projected, feasible, score = project_cyclic_role_sequence(logits, allowed)
        self.assertTrue(feasible)
        self.assertEqual(projected.tolist(), [0, 1, 2])
        self.assertAlmostEqual(score, 14.0)

    def test_infeasible_grammar_is_explicit_and_preserves_raw_prediction(self) -> None:
        logits = np.asarray([[2.0, 1.0], [1.0, 2.0]])
        projected, feasible, _ = project_cyclic_role_sequence(
            logits, np.zeros((2, 2), dtype=bool)
        )
        self.assertFalse(feasible)
        self.assertEqual(projected.tolist(), logits.argmax(-1).tolist())

    def test_grammar_check_includes_last_to_first_boundary(self) -> None:
        allowed = np.eye(3, dtype=bool)
        allowed[0, 1] = allowed[1, 2] = True
        roles = np.asarray([[0, 1, 2, -1]])
        valid = np.asarray([[True, True, True, False]])
        violations = check_cyclic_role_grammar(roles, valid, allowed)
        self.assertEqual(len(violations), 1)
        self.assertEqual((violations[0].first_edge_index, violations[0].second_edge_index), (2, 0))

    def test_panel_projection_supports_noncontiguous_storage_mask(self) -> None:
        allowed = np.ones((2, 2), dtype=bool)
        logits = np.zeros((1, 5, 2), dtype=np.float64)
        logits[0, 0, 0] = logits[0, 2, 1] = logits[0, 4, 0] = 3.0
        valid = np.asarray([[True, False, True, False, True]])
        result = project_cyclic_roles(logits, valid, allowed)
        self.assertEqual(result.projected_roles[0].tolist(), [0, -1, 1, -1, 0])
        self.assertEqual(result.panels[0].edge_indices, (0, 2, 4))

    def test_blossom_finds_better_matching_than_greedy_and_allows_same_panel(self) -> None:
        valid = np.ones((1, 4), dtype=bool)
        scores = np.zeros((4, 4), dtype=np.float64)
        # Greedy 0-1 then 2-3 gives 11. Blossom should choose 0-2 and 1-3 = 18.
        for first, second, value in (
            (0, 1, 10.0),
            (0, 2, 9.0),
            (1, 3, 9.0),
            (2, 3, 1.0),
        ):
            scores[first, second] = scores[second, first] = value
        result = solve_symmetric_seam_matching(scores, valid, threshold=0.5)
        pairs = {
            (value.first.edge_index, value.second.edge_index) for value in result.pairs
        }
        self.assertEqual(pairs, {(0, 2), (1, 3)})
        self.assertEqual(result.maximum_degree, 1)
        self.assertTrue(all(value.first.panel_index == value.second.panel_index == 0 for value in result.pairs))

    def test_seam_direction_scores_are_symmetrized(self) -> None:
        valid = np.asarray([[True, True]])
        scores = np.asarray([[0.0, 0.9], [0.5, 0.0]])
        result = solve_symmetric_seam_matching(scores, valid, threshold=0.65)
        self.assertEqual(len(result.pairs), 1)
        self.assertAlmostEqual(result.pairs[0].score, 0.7)
        self.assertAlmostEqual(result.symmetry_max_abs_error, 0.4)

    def test_rejected_seam_candidates_retain_symbolic_reason(self) -> None:
        valid = np.asarray([[True, True, True]])
        scores = np.zeros((3, 3), dtype=np.float64)
        scores[0, 1] = scores[1, 0] = 0.9
        scores[0, 2] = scores[2, 0] = 0.8
        scores[1, 2] = scores[2, 1] = 0.1
        scores[2, 2] = 1.0
        result = solve_symmetric_seam_matching(scores, valid, threshold=0.5)
        self.assertEqual(len(result.pairs), 1)
        self.assertEqual(result.rejected_by_reason["MATCHING_ENDPOINT_ALREADY_USED"], 1)
        self.assertEqual(result.rejected_by_reason["SELF_PAIR_FORBIDDEN"], 1)

    def test_landmarks_are_derived_from_incident_roles_at_same_vertex(self) -> None:
        role = {name: EDGE_ROLES.index(name) for name in EDGE_ROLES}
        roles = np.asarray(
            [[
                role["center_front"],
                role["neckline"],
                role["shoulder"],
                role["armhole"],
                role["side_seam"],
            ]]
        )
        valid = np.ones_like(roles, dtype=bool)
        landmarks = derive_landmark_facts(roles, valid)
        by_name = {value.name: value for value in landmarks}
        self.assertEqual(by_name["FNP"].vertex_index, 1)
        self.assertEqual(by_name["SNP"].vertex_index, 2)
        self.assertEqual(by_name["SP"].vertex_index, 3)
        self.assertNotIn("BNP", by_name)
        self.assertEqual(by_name["SP"].previous_edge_index, 2)
        self.assertEqual(by_name["SP"].following_edge_index, 3)

    def test_structured_report_contains_checked_proof_facts(self) -> None:
        role = {name: EDGE_ROLES.index(name) for name in EDGE_ROLES}
        role_ids = [role["center_front"], role["neckline"], role["shoulder"], role["armhole"]]
        logits = np.full((1, 4, len(EDGE_ROLES)), -5.0)
        for edge, value in enumerate(role_ids):
            logits[0, edge, value] = 5.0
        allowed = np.ones((len(EDGE_ROLES), len(EDGE_ROLES)), dtype=bool)
        valid = np.ones((1, 4), dtype=bool)
        seams = np.zeros((4, 4), dtype=np.float64)
        seams[0, 2] = seams[2, 0] = 0.9
        report = symbolic_project_and_verify(
            logits, seams, valid, allowed, seam_threshold=0.5
        )
        self.assertTrue(report.valid)
        self.assertEqual(report.metrics["seam_maximum_degree"], 1)
        facts = report.facts()
        self.assertIn("LANDMARK(p0,v1,FNP)", facts)
        self.assertIn("SEWN_TO(p0:e0,p0:e2)", facts)
        payload = report.to_dict()
        self.assertTrue(payload["valid"])
        self.assertIn("claim_boundary", payload)
        landmark_proof = next(
            value for value in report.derivations
            if value.conclusion == "LANDMARK(p0,v1,FNP)"
        )
        self.assertEqual(landmark_proof.rule, "SEMANTIC_JUNCTION")
        self.assertIn("ROLE(p0,e0,center_front)", landmark_proof.premises)
        self.assertIn("ROLE(p0,e1,neckline)", landmark_proof.premises)
        self.assertIn("SHARED_ENDPOINT(p0:e0,p0:e1,v1)", landmark_proof.premises)


if __name__ == "__main__":
    unittest.main()
