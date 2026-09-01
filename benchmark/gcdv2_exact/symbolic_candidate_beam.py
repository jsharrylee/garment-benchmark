"""Leakage-safe symbolic candidate beams for four-view Pattern DSL retrieval.

The current visual model is a retriever, not a command decoder.  Consequently
the only defensible discrete topology proposals are complete train-bank
``L/Q/C/A`` programs.  Each retrieved program is treated as one hypothesis in
the beam.  The neural Pattern-DSL proposer supplies category/edge-role/seam
scores and the exact solver turns those scores into grammar-valid facts.

Candidate scoring uses learned outputs available at inference only:

* four-view/DSL cosine similarity;
* agreement between the visual and DSL category posteriors;
* posterior confidence of the grammar-projected edge roles; and
* the fraction of roles that did not require symbolic repair.

Held-out category and topology labels are deliberately absent from
``rank_symbolic_beam``.  They may be used by callers after selection for
evaluation, or on a validation split to choose the four scalar weights.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class BeamWeights:
    similarity: float = 1.0
    category_agreement: float = 0.0
    projected_role_confidence: float = 0.0
    no_repair_fraction: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "similarity": float(self.similarity),
            "category_agreement": float(self.category_agreement),
            "projected_role_confidence": float(self.projected_role_confidence),
            "no_repair_fraction": float(self.no_repair_fraction),
        }


@dataclass(frozen=True)
class SymbolicCandidate:
    sample_id: str
    retrieval_rank: int
    similarity: float
    visual_category_probabilities: tuple[float, ...]
    dsl_category_probabilities: tuple[float, ...]
    projected_role_confidence: float
    repair_fraction: float
    symbolic_valid: bool
    topology_signature: str
    evaluation_category: int
    panel_command_cycles: tuple[tuple[str, ...], ...]
    projected_role_cycles: tuple[tuple[str, ...], ...]
    seam_pair_count: int
    landmark_count: int
    raw_grammar_violations: int
    projected_grammar_violations: int

    @property
    def category_agreement(self) -> float:
        visual = np.asarray(self.visual_category_probabilities, dtype=np.float64)
        pattern = np.asarray(self.dsl_category_probabilities, dtype=np.float64)
        if visual.shape != pattern.shape or visual.ndim != 1:
            raise ValueError("category posteriors must be equal one-dimensional vectors")
        return float(np.dot(visual, pattern))

    @property
    def no_repair_fraction(self) -> float:
        return float(np.clip(1.0 - self.repair_fraction, 0.0, 1.0))


@dataclass(frozen=True)
class RankedSymbolicCandidate:
    candidate: SymbolicCandidate
    score: float
    component_scores: Mapping[str, float]


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def score_symbolic_candidate(
    candidate: SymbolicCandidate, weights: BeamWeights
) -> tuple[float, dict[str, float]]:
    """Score one already-verified hypothesis without target information."""

    if not candidate.symbolic_valid:
        return -math.inf, {
            "similarity": -math.inf,
            "category_agreement": 0.0,
            "projected_role_confidence": 0.0,
            "no_repair_fraction": 0.0,
        }
    values = {
        "similarity": _finite(candidate.similarity, "similarity"),
        "category_agreement": _finite(candidate.category_agreement, "category_agreement"),
        "projected_role_confidence": _finite(
            candidate.projected_role_confidence, "projected_role_confidence"
        ),
        "no_repair_fraction": _finite(candidate.no_repair_fraction, "no_repair_fraction"),
    }
    components = {
        name: values[name] * float(getattr(weights, name)) for name in values
    }
    return float(sum(components.values())), components


def rank_symbolic_beam(
    candidates: Sequence[SymbolicCandidate], weights: BeamWeights
) -> tuple[RankedSymbolicCandidate, ...]:
    """Rank a beam with a stable, label-free inference rule.

    A caller may pass candidates in retrieval order.  Equal scores retain that
    preference through ``retrieval_rank`` and finally use ``sample_id`` for
    deterministic results.
    """

    output: list[RankedSymbolicCandidate] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.sample_id in seen:
            continue
        seen.add(candidate.sample_id)
        score, components = score_symbolic_candidate(candidate, weights)
        output.append(RankedSymbolicCandidate(candidate, score, components))
    output.sort(
        key=lambda value: (
            -value.score,
            value.candidate.retrieval_rank,
            value.candidate.sample_id,
        )
    )
    return tuple(output)


def candidate_beam_metrics(
    ranked_beams: Sequence[Sequence[RankedSymbolicCandidate]],
    target_categories: Sequence[int],
    target_topologies: Sequence[str],
) -> dict[str, float | int]:
    """Evaluate frozen selections; target values are never passed to ranking."""

    if not (len(ranked_beams) == len(target_categories) == len(target_topologies)):
        raise ValueError("beam and evaluation target counts differ")
    count = len(ranked_beams)
    selected_category = 0
    selected_topology = 0
    beam_category = 0
    beam_topology = 0
    valid = 0
    unique_topologies: list[int] = []
    for beam, target_category, target_topology in zip(
        ranked_beams, target_categories, target_topologies, strict=True
    ):
        if not beam:
            unique_topologies.append(0)
            continue
        winner = beam[0].candidate
        selected_category += int(winner.evaluation_category == int(target_category))
        selected_topology += int(winner.topology_signature == str(target_topology))
        beam_category += int(
            any(value.candidate.evaluation_category == int(target_category) for value in beam)
        )
        beam_topology += int(
            any(value.candidate.topology_signature == str(target_topology) for value in beam)
        )
        valid += sum(int(value.candidate.symbolic_valid) for value in beam)
        unique_topologies.append(len({value.candidate.topology_signature for value in beam}))
    candidate_count = sum(len(beam) for beam in ranked_beams)
    divisor = max(count, 1)
    return {
        "query_count": count,
        "selected_category_match_rate": selected_category / divisor,
        "selected_exact_primitive_topology_rate": selected_topology / divisor,
        "beam_category_coverage_rate": beam_category / divisor,
        "beam_exact_primitive_topology_coverage_rate": beam_topology / divisor,
        "symbolic_valid_candidate_rate": valid / max(candidate_count, 1),
        "mean_unique_topology_hypotheses": float(np.mean(unique_topologies)) if count else 0.0,
    }


def select_validation_weights(
    beams: Sequence[Sequence[SymbolicCandidate]],
    target_categories: Sequence[int],
    target_topologies: Sequence[str],
    candidates: Iterable[BeamWeights],
) -> tuple[BeamWeights, dict[str, float | int], list[dict[str, object]]]:
    """Freeze scalar reranking weights on validation-only evaluation labels."""

    rows: list[tuple[BeamWeights, dict[str, float | int]]] = []
    for weights in candidates:
        ranked = [rank_symbolic_beam(beam, weights) for beam in beams]
        metrics = candidate_beam_metrics(ranked, target_categories, target_topologies)
        rows.append((weights, metrics))
    if not rows:
        raise ValueError("weight grid is empty")

    def objective(row: tuple[BeamWeights, Mapping[str, float | int]]) -> tuple[float, ...]:
        weights, metrics = row
        # Topology is the primary target.  Category is a secondary diagnostic.
        # Lower auxiliary norm wins exact ties and makes the pure retrieval
        # baseline preferred when no validation improvement is demonstrated.
        auxiliary_norm = (
            abs(weights.category_agreement)
            + abs(weights.projected_role_confidence)
            + abs(weights.no_repair_fraction)
        )
        return (
            float(metrics["selected_exact_primitive_topology_rate"]),
            float(metrics["selected_category_match_rate"]),
            -auxiliary_norm,
            -abs(weights.category_agreement),
            -abs(weights.projected_role_confidence),
            -abs(weights.no_repair_fraction),
        )

    best_weights, best_metrics = max(rows, key=objective)
    table = [
        {"weights": weights.as_dict(), "metrics": dict(metrics)} for weights, metrics in rows
    ]
    return best_weights, best_metrics, table


__all__ = [
    "BeamWeights",
    "RankedSymbolicCandidate",
    "SymbolicCandidate",
    "candidate_beam_metrics",
    "rank_symbolic_beam",
    "score_symbolic_candidate",
    "select_validation_weights",
]
