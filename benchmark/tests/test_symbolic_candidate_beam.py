from __future__ import annotations

import unittest

from benchmark.gcdv2_exact.symbolic_candidate_beam import (
    BeamWeights,
    SymbolicCandidate,
    candidate_beam_metrics,
    rank_symbolic_beam,
    select_validation_weights,
)


def _candidate(
    sample_id: str,
    rank: int,
    similarity: float,
    *,
    visual=(0.8, 0.1, 0.1),
    pattern=(0.8, 0.1, 0.1),
    role=0.9,
    repair=0.0,
    valid=True,
    topology="a",
    category=0,
) -> SymbolicCandidate:
    return SymbolicCandidate(
        sample_id=sample_id,
        retrieval_rank=rank,
        similarity=similarity,
        visual_category_probabilities=visual,
        dsl_category_probabilities=pattern,
        projected_role_confidence=role,
        repair_fraction=repair,
        symbolic_valid=valid,
        topology_signature=topology,
        evaluation_category=category,
        panel_command_cycles=(("L", "C", "L"),),
        projected_role_cycles=(("side_seam", "armhole", "shoulder"),),
        seam_pair_count=1,
        landmark_count=1,
        raw_grammar_violations=0,
        projected_grammar_violations=0,
    )


class SymbolicCandidateBeamTest(unittest.TestCase):
    def test_pure_similarity_preserves_retrieval_top1(self) -> None:
        beam = [_candidate("a", 1, 0.8), _candidate("b", 2, 0.7)]
        ranked = rank_symbolic_beam(beam, BeamWeights())
        self.assertEqual(ranked[0].candidate.sample_id, "a")

    def test_invalid_program_is_hard_rejected(self) -> None:
        beam = [
            _candidate("invalid", 1, 0.99, valid=False),
            _candidate("valid", 2, 0.60, valid=True),
        ]
        ranked = rank_symbolic_beam(beam, BeamWeights())
        self.assertEqual(ranked[0].candidate.sample_id, "valid")

    def test_learned_category_agreement_can_rerank_without_target(self) -> None:
        beam = [
            _candidate("wrong", 1, 0.80, pattern=(0.1, 0.8, 0.1)),
            _candidate("agree", 2, 0.78, pattern=(0.9, 0.05, 0.05)),
        ]
        ranked = rank_symbolic_beam(
            beam, BeamWeights(similarity=1.0, category_agreement=0.2)
        )
        self.assertEqual(ranked[0].candidate.sample_id, "agree")

    def test_evaluation_targets_do_not_enter_ranking(self) -> None:
        # evaluation_category and topology can change without changing scores.
        first = _candidate("a", 1, 0.8, topology="target", category=0)
        second = _candidate("b", 2, 0.7, topology="other", category=1)
        ranking = rank_symbolic_beam([first, second], BeamWeights())
        altered = [
            _candidate("a", 1, 0.8, topology="other", category=2),
            _candidate("b", 2, 0.7, topology="target", category=0),
        ]
        reranking = rank_symbolic_beam(altered, BeamWeights())
        self.assertEqual(
            [value.candidate.sample_id for value in ranking],
            [value.candidate.sample_id for value in reranking],
        )

    def test_metrics_separate_selected_result_from_oracle_coverage(self) -> None:
        ranked = rank_symbolic_beam(
            [
                _candidate("a", 1, 0.8, topology="wrong"),
                _candidate("b", 2, 0.7, topology="target"),
            ],
            BeamWeights(),
        )
        metrics = candidate_beam_metrics([ranked], [0], ["target"])
        self.assertEqual(metrics["selected_exact_primitive_topology_rate"], 0.0)
        self.assertEqual(metrics["beam_exact_primitive_topology_coverage_rate"], 1.0)

    def test_validation_grid_prefers_topology_improvement(self) -> None:
        beam = [
            _candidate("similar", 1, 0.80, pattern=(0.1, 0.8, 0.1), topology="wrong"),
            _candidate("agree", 2, 0.79, pattern=(0.9, 0.05, 0.05), topology="target"),
        ]
        weights, metrics, table = select_validation_weights(
            [beam],
            [0],
            ["target"],
            [BeamWeights(), BeamWeights(category_agreement=0.2)],
        )
        self.assertEqual(weights.category_agreement, 0.2)
        self.assertEqual(metrics["selected_exact_primitive_topology_rate"], 1.0)
        self.assertEqual(len(table), 2)


if __name__ == "__main__":
    unittest.main()
