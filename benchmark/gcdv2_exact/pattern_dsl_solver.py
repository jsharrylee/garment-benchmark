"""Symbolic projection for neural Pattern-DSL propositions.

The neural model in :mod:`pattern_dsl_learning` proposes edge roles and seam
relations.  This module is the deterministic half of the AlphaGeometry-style
pipeline: it turns those scores into a small set of facts which satisfy the
known pattern topology.

The solver deliberately does *not* move a point or invent a curve.  Exact
``M/L/Q/C/A/Z`` geometry is an observed premise.  It only projects semantic
propositions onto three invariants:

* the semantic roles around every panel form a permitted cyclic sequence;
* ``SEWN_TO`` is symmetric and every edge has at most one mate; and
* FNP/BNP/SNP/SP are consequences of two semantic edges sharing a vertex.

Same-panel sewing is valid (darts are a common example), so the seam solver
only forbids an edge from being paired with itself.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from benchmark.drafting_semantics.schema import EDGE_ROLES
from benchmark.gcdv2_exact.pattern_dsl_learning import LANDMARK_NAMES


LANDMARK_ROLE_NAMES: Mapping[str, frozenset[str]] = {
    "FNP": frozenset(("center_front", "neckline")),
    "BNP": frozenset(("center_back", "neckline")),
    "SNP": frozenset(("neckline", "shoulder")),
    "SP": frozenset(("shoulder", "armhole")),
}


@dataclass(frozen=True, order=True)
class EdgeReference:
    panel_index: int
    edge_index: int
    flat_index: int


@dataclass(frozen=True)
class GrammarViolation:
    panel_index: int
    first_edge_index: int
    second_edge_index: int
    first_role: int
    second_role: int


@dataclass(frozen=True)
class PanelRoleProjection:
    panel_index: int
    edge_indices: tuple[int, ...]
    raw_roles: tuple[int, ...]
    projected_roles: tuple[int, ...]
    feasible: bool
    selected_logit_sum: float
    changed_edges: int
    raw_violation_count: int
    projected_violation_count: int


@dataclass(frozen=True)
class CyclicRoleProjection:
    """Result of exact cyclic Viterbi projection for every valid panel."""

    raw_roles: np.ndarray
    projected_roles: np.ndarray
    panels: tuple[PanelRoleProjection, ...]
    raw_violations: tuple[GrammarViolation, ...]
    projected_violations: tuple[GrammarViolation, ...]

    @property
    def feasible(self) -> bool:
        return all(panel.feasible for panel in self.panels)

    @property
    def changed_edges(self) -> int:
        return sum(panel.changed_edges for panel in self.panels)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feasible": self.feasible,
            "changed_edges": self.changed_edges,
            "raw_roles": self.raw_roles.tolist(),
            "projected_roles": self.projected_roles.tolist(),
            "panels": [asdict(value) for value in self.panels],
            "raw_violations": [asdict(value) for value in self.raw_violations],
            "projected_violations": [asdict(value) for value in self.projected_violations],
        }


@dataclass(frozen=True)
class SeamPair:
    first: EdgeReference
    second: EdgeReference
    score: float


@dataclass(frozen=True)
class RejectedSeam:
    first: EdgeReference
    second: EdgeReference
    score: float
    reason: str


@dataclass(frozen=True)
class SeamMatchingReport:
    pairs: tuple[SeamPair, ...]
    candidate_count: int
    valid_edge_count: int
    unmatched_edge_count: int
    maximum_degree: int
    symmetry_max_abs_error: float
    threshold: float
    top_k_per_edge: int | None
    rejected_candidate_count: int
    rejected_by_reason: Mapping[str, int]
    rejected_examples: tuple[RejectedSeam, ...]

    @property
    def valid(self) -> bool:
        return self.maximum_degree <= 1


@dataclass(frozen=True)
class LandmarkFact:
    panel_index: int
    vertex_index: int
    name: str
    base_name: str
    previous_edge_index: int
    following_edge_index: int
    incident_roles: tuple[str, str]


@dataclass(frozen=True)
class ProjectionIssue:
    severity: str
    code: str
    subject: str
    message: str


@dataclass(frozen=True)
class ProofStep:
    """One accepted deduction or one explicitly rejected proposition."""

    status: str
    conclusion: str
    premises: tuple[str, ...]
    rule: str
    reason: str | None = None


@dataclass(frozen=True)
class SymbolicProjectionReport:
    """Machine-readable proof/check output for one garment prediction."""

    valid: bool
    roles: CyclicRoleProjection
    seams: SeamMatchingReport
    landmarks: tuple[LandmarkFact, ...]
    issues: tuple[ProjectionIssue, ...]
    derivations: tuple[ProofStep, ...]
    metrics: Mapping[str, int | float | bool]
    role_names: tuple[str, ...]

    def facts(self) -> tuple[str, ...]:
        """Return compact symbolic facts suitable for logs and DSL reports."""

        facts: list[str] = []
        for panel in self.roles.panels:
            count = len(panel.edge_indices)
            for local, (edge, role) in enumerate(
                zip(panel.edge_indices, panel.projected_roles, strict=True)
            ):
                facts.append(f"ROLE(p{panel.panel_index},e{edge},{self.role_names[role]})")
                if count:
                    following = panel.edge_indices[(local + 1) % count]
                    facts.append(f"NEXT(p{panel.panel_index},e{edge},e{following})")
        for pair in self.seams.pairs:
            facts.append(
                "SEWN_TO("
                f"p{pair.first.panel_index}:e{pair.first.edge_index},"
                f"p{pair.second.panel_index}:e{pair.second.edge_index})"
            )
        for landmark in self.landmarks:
            facts.append(
                f"LANDMARK(p{landmark.panel_index},v{landmark.vertex_index},{landmark.name})"
            )
        return tuple(facts)

    def to_dict(self) -> dict[str, Any]:
        seam_values = asdict(self.seams)
        return {
            "valid": self.valid,
            "roles": self.roles.to_dict(),
            "seams": seam_values,
            "landmarks": [asdict(value) for value in self.landmarks],
            "issues": [asdict(value) for value in self.issues],
            "derivations": [asdict(value) for value in self.derivations],
            "metrics": dict(self.metrics),
            "facts": list(self.facts()),
            "claim_boundary": (
                "Symbolic consistency of neural propositions over observed exact geometry; "
                "not expert-pattern correctness."
            ),
        }


def _validate_allowed(allowed_transitions: np.ndarray, role_count: int) -> np.ndarray:
    allowed = np.asarray(allowed_transitions, dtype=bool)
    if allowed.shape != (role_count, role_count):
        raise ValueError(
            f"allowed_transitions must have shape {(role_count, role_count)}, got {allowed.shape}"
        )
    return allowed


def project_cyclic_role_sequence(
    role_logits: np.ndarray,
    allowed_transitions: np.ndarray,
    *,
    transition_bonus: float = 0.0,
) -> tuple[np.ndarray, bool, float]:
    """Find the maximum-score role sequence whose final edge wraps to edge 0.

    This is exact cyclic Viterbi, not greedy repair.  The first role is
    enumerated and dynamic programming solves the remaining chain.  When the
    supplied grammar admits no cycle, raw argmax roles are returned with
    ``feasible=False`` so callers never silently lose a prediction.
    """

    scores = np.asarray(role_logits, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError(f"role_logits must be [edges, roles], got {scores.shape}")
    edge_count, role_count = scores.shape
    if edge_count == 0:
        return np.empty(0, dtype=np.int64), True, 0.0
    if np.isnan(scores).any() or np.isposinf(scores).any():
        raise ValueError("role_logits contain NaN or +inf")
    allowed = _validate_allowed(allowed_transitions, role_count)
    bonus = float(transition_bonus)
    if not math.isfinite(bonus):
        raise ValueError("transition_bonus must be finite")

    best_score = -np.inf
    best_roles: np.ndarray | None = None
    for start in range(role_count):
        if not np.isfinite(scores[0, start]):
            continue
        if edge_count == 1:
            if allowed[start, start]:
                value = float(scores[0, start] + bonus)
                if value > best_score:
                    best_score = value
                    best_roles = np.asarray([start], dtype=np.int64)
            continue

        dynamic = np.full(role_count, -np.inf, dtype=np.float64)
        dynamic[start] = scores[0, start]
        backpointers: list[np.ndarray] = []
        for position in range(1, edge_count):
            candidates = dynamic[:, None] + scores[position][None, :]
            candidates = candidates + np.where(allowed, bonus, -np.inf)
            previous = np.argmax(candidates, axis=0)
            dynamic = candidates[previous, np.arange(role_count)]
            backpointers.append(previous.astype(np.int64))
        closed = dynamic + np.where(allowed[:, start], bonus, -np.inf)
        end = int(np.argmax(closed))
        value = float(closed[end])
        if not np.isfinite(value) or value <= best_score:
            continue
        decoded = np.empty(edge_count, dtype=np.int64)
        decoded[-1] = end
        for position in range(edge_count - 1, 0, -1):
            decoded[position - 1] = backpointers[position - 1][decoded[position]]
        if decoded[0] != start:  # defensive assertion for corrupted backtracking
            raise RuntimeError("cyclic Viterbi backtracking lost its fixed start state")
        best_score, best_roles = value, decoded

    if best_roles is None:
        raw = np.argmax(scores, axis=-1).astype(np.int64)
        return raw, False, float(scores[np.arange(edge_count), raw].sum())
    return best_roles, True, best_score


def check_cyclic_role_grammar(
    role_ids: np.ndarray,
    edge_valid: np.ndarray,
    allowed_transitions: np.ndarray,
) -> tuple[GrammarViolation, ...]:
    roles = np.asarray(role_ids, dtype=np.int64)
    valid = np.asarray(edge_valid, dtype=bool)
    if roles.shape != valid.shape or roles.ndim != 2:
        raise ValueError("role_ids and edge_valid must have identical [panels, edges] shapes")
    allowed = _validate_allowed(allowed_transitions, int(allowed_transitions.shape[0]))
    output: list[GrammarViolation] = []
    for panel_index in range(roles.shape[0]):
        edge_indices = np.flatnonzero(valid[panel_index])
        if not len(edge_indices):
            continue
        values = roles[panel_index, edge_indices]
        if np.any(values < 0) or np.any(values >= len(allowed)):
            raise ValueError("valid edges contain an out-of-range role id")
        for local, first in enumerate(edge_indices):
            second = int(edge_indices[(local + 1) % len(edge_indices)])
            first_role, second_role = int(roles[panel_index, first]), int(roles[panel_index, second])
            if not allowed[first_role, second_role]:
                output.append(
                    GrammarViolation(
                        panel_index,
                        int(first),
                        second,
                        first_role,
                        second_role,
                    )
                )
    return tuple(output)


def project_cyclic_roles(
    role_logits: np.ndarray,
    edge_valid: np.ndarray,
    allowed_transitions: np.ndarray,
    *,
    transition_bonus: float = 0.0,
) -> CyclicRoleProjection:
    scores = np.asarray(role_logits, dtype=np.float64)
    valid = np.asarray(edge_valid, dtype=bool)
    if scores.ndim != 3 or scores.shape[:2] != valid.shape:
        raise ValueError("role_logits must be [panels, edges, roles] and match edge_valid")
    allowed = _validate_allowed(allowed_transitions, scores.shape[-1])
    raw = np.full(valid.shape, -1, dtype=np.int64)
    projected = np.full(valid.shape, -1, dtype=np.int64)
    raw[valid] = scores[valid].argmax(-1)
    panels: list[PanelRoleProjection] = []
    for panel_index in range(valid.shape[0]):
        edge_indices = np.flatnonzero(valid[panel_index])
        if not len(edge_indices):
            continue
        selected, feasible, score = project_cyclic_role_sequence(
            scores[panel_index, edge_indices], allowed, transition_bonus=transition_bonus
        )
        projected[panel_index, edge_indices] = selected
        panel_valid = np.ones((1, len(edge_indices)), dtype=bool)
        raw_local = raw[panel_index, edge_indices][None]
        projected_local = selected[None]
        raw_bad = check_cyclic_role_grammar(raw_local, panel_valid, allowed)
        projected_bad = check_cyclic_role_grammar(projected_local, panel_valid, allowed)
        panels.append(
            PanelRoleProjection(
                panel_index=panel_index,
                edge_indices=tuple(int(value) for value in edge_indices),
                raw_roles=tuple(int(value) for value in raw_local[0]),
                projected_roles=tuple(int(value) for value in selected),
                feasible=feasible,
                selected_logit_sum=score,
                changed_edges=int((raw_local[0] != selected).sum()),
                raw_violation_count=len(raw_bad),
                projected_violation_count=len(projected_bad),
            )
        )
    return CyclicRoleProjection(
        raw_roles=raw,
        projected_roles=projected,
        panels=tuple(panels),
        raw_violations=check_cyclic_role_grammar(raw, valid, allowed),
        projected_violations=check_cyclic_role_grammar(projected, valid, allowed),
    )


def _square_scores(values: np.ndarray, edge_valid: np.ndarray) -> np.ndarray:
    scores = np.asarray(values, dtype=np.float64)
    panel_count, edge_count = edge_valid.shape
    flat_count = panel_count * edge_count
    if scores.shape == (panel_count, edge_count, panel_count, edge_count):
        scores = scores.reshape(flat_count, flat_count)
    if scores.shape != (flat_count, flat_count):
        raise ValueError(
            "seam_scores must be [panels*edges, panels*edges] or "
            "[panels, edges, panels, edges]"
        )
    if np.isnan(scores).any():
        raise ValueError("seam_scores contain NaN")
    return scores


def solve_symmetric_seam_matching(
    seam_scores: np.ndarray,
    edge_valid: np.ndarray,
    *,
    threshold: float = 0.5,
    top_k_per_edge: int | None = None,
    pair_allowed: np.ndarray | None = None,
    maximum_rejection_examples: int = 64,
) -> SeamMatchingReport:
    """Maximum-weight general-graph seam matching (Blossom algorithm).

    Candidate scores are averaged in both directions before matching.  This
    makes ``SEWN_TO(a,b) == SEWN_TO(b,a)`` explicit even when a caller supplies
    asymmetric neural scores.  No panel inequality is imposed, hence dart legs
    and other same-panel seams remain eligible.
    """

    import networkx as nx

    valid = np.asarray(edge_valid, dtype=bool)
    if valid.ndim != 2:
        raise ValueError("edge_valid must be [panels, edges]")
    raw = _square_scores(seam_scores, valid)
    if not math.isfinite(float(threshold)):
        raise ValueError("threshold must be finite")
    if top_k_per_edge is not None and top_k_per_edge <= 0:
        raise ValueError("top_k_per_edge must be positive or None")
    if maximum_rejection_examples < 0:
        raise ValueError("maximum_rejection_examples must be nonnegative")
    symmetric = 0.5 * (raw + raw.T)
    finite_delta = np.abs(raw - raw.T)
    finite_delta = finite_delta[np.isfinite(finite_delta)]
    symmetry_error = float(finite_delta.max()) if len(finite_delta) else 0.0
    flat_valid = valid.reshape(-1)
    valid_indices = np.flatnonzero(flat_valid)
    allowed = np.ones_like(symmetric, dtype=bool)
    if pair_allowed is not None:
        allowed_raw = _square_scores(np.asarray(pair_allowed, dtype=np.float64), valid)
        allowed = allowed_raw.astype(bool) & allowed_raw.T.astype(bool)
    np.fill_diagonal(allowed, False)

    above_threshold = (symmetric >= float(threshold)) & np.isfinite(symmetric)
    valid_pair = flat_valid[:, None] & flat_valid[None, :]
    threshold_valid = np.triu(above_threshold & valid_pair, 1)
    disallowed_mask = threshold_valid & ~allowed
    candidate_mask = threshold_valid & allowed
    before_top_k = candidate_mask.copy()
    if top_k_per_edge is not None:
        keep = np.zeros_like(candidate_mask)
        full = candidate_mask | candidate_mask.T
        for first in valid_indices:
            choices = np.flatnonzero(full[first])
            if len(choices) > top_k_per_edge:
                order = np.argsort(symmetric[first, choices], kind="stable")[-top_k_per_edge:]
                choices = choices[order]
            keep[first, choices] = True
        candidate_mask &= keep | keep.T

    top_k_rejected_mask = before_top_k & ~candidate_mask

    graph = nx.Graph()
    graph.add_nodes_from(int(value) for value in valid_indices)
    first_values, second_values = np.nonzero(candidate_mask)
    for first, second in zip(first_values, second_values, strict=True):
        graph.add_edge(int(first), int(second), weight=float(symmetric[first, second]))
    raw_matching = nx.max_weight_matching(graph, maxcardinality=False, weight="weight")
    panel_count, edge_count = valid.shape
    pairs: list[SeamPair] = []
    degrees: dict[int, int] = {}
    for raw_first, raw_second in raw_matching:
        first, second = sorted((int(raw_first), int(raw_second)))
        degrees[first] = degrees.get(first, 0) + 1
        degrees[second] = degrees.get(second, 0) + 1
        pairs.append(
            SeamPair(
                EdgeReference(first // edge_count, first % edge_count, first),
                EdgeReference(second // edge_count, second % edge_count, second),
                float(symmetric[first, second]),
            )
        )
    pairs.sort(key=lambda value: (value.first.flat_index, value.second.flat_index))
    selected_keys = {
        (value.first.flat_index, value.second.flat_index) for value in pairs
    }
    selected_edges = {
        value for pair in selected_keys for value in pair
    }
    rejection_counts: dict[str, int] = {}
    rejection_examples: list[RejectedSeam] = []

    def reject(first: int, second: int, reason: str) -> None:
        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
        if len(rejection_examples) >= maximum_rejection_examples:
            return
        rejection_examples.append(
            RejectedSeam(
                EdgeReference(first // edge_count, first % edge_count, first),
                EdgeReference(second // edge_count, second % edge_count, second),
                float(symmetric[first, second]),
                reason,
            )
        )

    for first, second in zip(*np.nonzero(disallowed_mask), strict=True):
        reject(int(first), int(second), "PAIR_CONSTRAINT_FORBIDS")
    for first, second in zip(*np.nonzero(top_k_rejected_mask), strict=True):
        reject(int(first), int(second), "TOP_K_PRUNED")
    for first, second in zip(*np.nonzero(candidate_mask), strict=True):
        key = (int(first), int(second))
        if key in selected_keys:
            continue
        reason = (
            "MATCHING_ENDPOINT_ALREADY_USED"
            if first in selected_edges or second in selected_edges
            else "NON_POSITIVE_GLOBAL_GAIN"
        )
        reject(int(first), int(second), reason)
    # Diagonal scores can look attractive to a neural proposer, but an edge
    # cannot be sewn to itself.  Same-panel *distinct* edges remain permitted.
    for edge in valid_indices:
        if above_threshold[edge, edge]:
            reject(int(edge), int(edge), "SELF_PAIR_FORBIDDEN")
    maximum_degree = max(degrees.values(), default=0)
    return SeamMatchingReport(
        pairs=tuple(pairs),
        candidate_count=int(candidate_mask.sum()),
        valid_edge_count=int(flat_valid.sum()),
        unmatched_edge_count=int(flat_valid.sum()) - 2 * len(pairs),
        maximum_degree=maximum_degree,
        symmetry_max_abs_error=symmetry_error,
        threshold=float(threshold),
        top_k_per_edge=top_k_per_edge,
        rejected_candidate_count=sum(rejection_counts.values()),
        rejected_by_reason=dict(sorted(rejection_counts.items())),
        rejected_examples=tuple(rejection_examples),
    )


def _proof_steps(
    roles: CyclicRoleProjection,
    seams: SeamMatchingReport,
    landmarks: Sequence[LandmarkFact],
    role_names: Sequence[str],
) -> tuple[ProofStep, ...]:
    names = tuple(role_names)
    output: list[ProofStep] = []
    projected_bad = {
        (value.panel_index, value.first_edge_index, value.second_edge_index): value
        for value in roles.projected_violations
    }
    for panel in roles.panels:
        count = len(panel.edge_indices)
        for local, edge in enumerate(panel.edge_indices):
            following = panel.edge_indices[(local + 1) % count]
            first_role = panel.projected_roles[local]
            second_role = panel.projected_roles[(local + 1) % count]
            key = (panel.panel_index, edge, following)
            if key in projected_bad:
                output.append(
                    ProofStep(
                        "rejected",
                        f"PROJECTED_NEXT_ROLE(p{panel.panel_index}:e{edge},p{panel.panel_index}:e{following})",
                        (
                            f"ROLE(p{panel.panel_index},e{edge},{names[first_role]})",
                            f"ROLE(p{panel.panel_index},e{following},{names[second_role]})",
                        ),
                        "CYCLIC_GRAMMAR_CHECK",
                        f"GRAMMAR_FORBIDS({names[first_role]},{names[second_role]})",
                    )
                )
            else:
                output.append(
                    ProofStep(
                        "accepted",
                        f"ALLOWED_NEXT_ROLE(p{panel.panel_index}:e{edge},p{panel.panel_index}:e{following})",
                        (
                            f"ROLE(p{panel.panel_index},e{edge},{names[first_role]})",
                            f"ROLE(p{panel.panel_index},e{following},{names[second_role]})",
                            f"GRAMMAR_ALLOWS({names[first_role]},{names[second_role]})",
                        ),
                        "CYCLIC_VITERBI",
                    )
                )
    for violation in roles.raw_violations:
        output.append(
            ProofStep(
                "rejected",
                f"RAW_NEXT_ROLE(p{violation.panel_index}:e{violation.first_edge_index},p{violation.panel_index}:e{violation.second_edge_index})",
                (
                    f"ROLE_RAW(p{violation.panel_index},e{violation.first_edge_index},{names[violation.first_role]})",
                    f"ROLE_RAW(p{violation.panel_index},e{violation.second_edge_index},{names[violation.second_role]})",
                ),
                "CYCLIC_GRAMMAR_CHECK",
                f"GRAMMAR_FORBIDS({names[violation.first_role]},{names[violation.second_role]})",
            )
        )
    for pair in seams.pairs:
        first = f"p{pair.first.panel_index}:e{pair.first.edge_index}"
        second = f"p{pair.second.panel_index}:e{pair.second.edge_index}"
        output.append(
            ProofStep(
                "accepted",
                f"SEWN_TO({first},{second})",
                (
                    f"SYMMETRIC_SCORE({first},{second})={pair.score:.8g}",
                    f"SCORE_AT_LEAST_THRESHOLD({pair.score:.8g},{seams.threshold:.8g})",
                    f"DEGREE_AT_MOST_ONE({first})",
                    f"DEGREE_AT_MOST_ONE({second})",
                ),
                "MAX_WEIGHT_GENERAL_MATCHING",
            )
        )
    for rejected in seams.rejected_examples:
        first = f"p{rejected.first.panel_index}:e{rejected.first.edge_index}"
        second = f"p{rejected.second.panel_index}:e{rejected.second.edge_index}"
        output.append(
            ProofStep(
                "rejected",
                f"SEWN_TO({first},{second})",
                (f"SYMMETRIC_SCORE({first},{second})={rejected.score:.8g}",),
                "SEAM_CONSTRAINT_CHECK",
                rejected.reason,
            )
        )
    for landmark in landmarks:
        previous_role, following_role = landmark.incident_roles
        output.append(
            ProofStep(
                "accepted",
                f"LANDMARK(p{landmark.panel_index},v{landmark.vertex_index},{landmark.name})",
                (
                    f"ROLE(p{landmark.panel_index},e{landmark.previous_edge_index},{previous_role})",
                    f"ROLE(p{landmark.panel_index},e{landmark.following_edge_index},{following_role})",
                    "SHARED_ENDPOINT("
                    f"p{landmark.panel_index}:e{landmark.previous_edge_index},"
                    f"p{landmark.panel_index}:e{landmark.following_edge_index},"
                    f"v{landmark.vertex_index})",
                ),
                "SEMANTIC_JUNCTION",
            )
        )
    return tuple(output)


def derive_landmark_facts(
    role_ids: np.ndarray,
    edge_valid: np.ndarray,
    *,
    role_names: Sequence[str] = EDGE_ROLES,
) -> tuple[LandmarkFact, ...]:
    """Derive named junctions; coordinates are never independently regressed."""

    roles = np.asarray(role_ids, dtype=np.int64)
    valid = np.asarray(edge_valid, dtype=bool)
    if roles.shape != valid.shape or roles.ndim != 2:
        raise ValueError("role_ids and edge_valid must have identical [panels, edges] shapes")
    names = tuple(str(value) for value in role_names)
    required = set().union(*LANDMARK_ROLE_NAMES.values())
    if not required.issubset(names):
        missing = sorted(required - set(names))
        raise ValueError(f"role_names are missing landmark roles: {missing}")

    provisional: list[tuple[int, int, str, int, int, tuple[str, str]]] = []
    for panel_index in range(roles.shape[0]):
        edge_indices = np.flatnonzero(valid[panel_index])
        if not len(edge_indices):
            continue
        for local, following in enumerate(edge_indices):
            previous = int(edge_indices[(local - 1) % len(edge_indices)])
            previous_role = int(roles[panel_index, previous])
            following_role = int(roles[panel_index, following])
            if not (0 <= previous_role < len(names) and 0 <= following_role < len(names)):
                raise ValueError("valid edges contain an out-of-range role id")
            incident = (names[previous_role], names[following_role])
            incident_set = frozenset(incident)
            for landmark_name in LANDMARK_NAMES:
                if incident_set == LANDMARK_ROLE_NAMES[landmark_name]:
                    # In an ordered closed boundary, vertex i is the endpoint
                    # of edge i-1 and startpoint of edge i.
                    provisional.append(
                        (
                            panel_index,
                            int(following),
                            landmark_name,
                            previous,
                            int(following),
                            incident,
                        )
                    )

    totals: dict[tuple[int, str], int] = {}
    for panel, _, name, *_ in provisional:
        totals[(panel, name)] = totals.get((panel, name), 0) + 1
    seen: dict[tuple[int, str], int] = {}
    output: list[LandmarkFact] = []
    for panel, vertex, base_name, previous, following, incident in provisional:
        key = (panel, base_name)
        instance = seen.get(key, 0)
        seen[key] = instance + 1
        resolved = base_name if totals[key] == 1 else f"{base_name}#{instance}"
        output.append(
            LandmarkFact(
                panel_index=panel,
                vertex_index=vertex,
                name=resolved,
                base_name=base_name,
                previous_edge_index=previous,
                following_edge_index=following,
                incident_roles=incident,
            )
        )
    return tuple(output)


def symbolic_project_and_verify(
    role_logits: np.ndarray,
    seam_scores: np.ndarray,
    edge_valid: np.ndarray,
    allowed_transitions: np.ndarray,
    *,
    role_names: Sequence[str] = EDGE_ROLES,
    seam_threshold: float = 0.5,
    seam_top_k_per_edge: int | None = None,
    seam_pair_allowed: np.ndarray | None = None,
    transition_bonus: float = 0.0,
) -> SymbolicProjectionReport:
    """Project one garment's neural propositions and emit proof-like facts."""

    names = tuple(str(value) for value in role_names)
    scores = np.asarray(role_logits)
    if scores.shape[-1] != len(names):
        raise ValueError("role_names length does not match role_logits")
    roles = project_cyclic_roles(
        scores,
        edge_valid,
        allowed_transitions,
        transition_bonus=transition_bonus,
    )
    seams = solve_symmetric_seam_matching(
        seam_scores,
        edge_valid,
        threshold=seam_threshold,
        top_k_per_edge=seam_top_k_per_edge,
        pair_allowed=seam_pair_allowed,
    )
    landmarks = derive_landmark_facts(
        roles.projected_roles,
        edge_valid,
        role_names=names,
    )
    issues: list[ProjectionIssue] = []
    for panel in roles.panels:
        if not panel.feasible:
            issues.append(
                ProjectionIssue(
                    "error",
                    "NO_FEASIBLE_ROLE_CYCLE",
                    f"panel:{panel.panel_index}",
                    "No role sequence satisfies the supplied cyclic grammar.",
                )
            )
    if roles.projected_violations:
        issues.append(
            ProjectionIssue(
                "error",
                "PROJECTED_GRAMMAR_VIOLATION",
                "roles",
                f"{len(roles.projected_violations)} forbidden cyclic transitions remain.",
            )
        )
    if roles.raw_violations:
        issues.append(
            ProjectionIssue(
                "warning",
                "RAW_GRAMMAR_REPAIRED",
                "roles",
                f"Projected {len(roles.raw_violations)} forbidden raw transitions.",
            )
        )
    if not seams.valid:
        issues.append(
            ProjectionIssue(
                "error",
                "SEAM_EDGE_REUSED",
                "seams",
                f"Maximum seam degree is {seams.maximum_degree}; expected at most one.",
            )
        )
    metrics: dict[str, int | float | bool] = {
        "role_projection_feasible": roles.feasible,
        "raw_grammar_violations": len(roles.raw_violations),
        "projected_grammar_violations": len(roles.projected_violations),
        "role_edges_changed": roles.changed_edges,
        "seam_candidates": seams.candidate_count,
        "seam_pairs": len(seams.pairs),
        "seam_maximum_degree": seams.maximum_degree,
        "derived_landmarks": len(landmarks),
        "rejected_seam_candidates": seams.rejected_candidate_count,
    }
    valid = not any(issue.severity == "error" for issue in issues)
    derivations = _proof_steps(roles, seams, landmarks, names)
    return SymbolicProjectionReport(
        valid,
        roles,
        seams,
        landmarks,
        tuple(issues),
        derivations,
        metrics,
        names,
    )


__all__ = [
    "CyclicRoleProjection",
    "EdgeReference",
    "GrammarViolation",
    "LANDMARK_ROLE_NAMES",
    "LandmarkFact",
    "PanelRoleProjection",
    "ProjectionIssue",
    "ProofStep",
    "RejectedSeam",
    "SeamMatchingReport",
    "SeamPair",
    "SymbolicProjectionReport",
    "check_cyclic_role_grammar",
    "derive_landmark_facts",
    "project_cyclic_role_sequence",
    "project_cyclic_roles",
    "solve_symmetric_seam_matching",
    "symbolic_project_and_verify",
]
