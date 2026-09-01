"""Project four-view semantic evidence into a constrained T-shirt draft.

The visual student predicts a table of named pattern elements.  Those values
are useful evidence, but they are not a sewable pattern because shared
landmarks and mating seams are decoded independently.  This module supplies
the missing inverse bridge::

    frozen four-view semantic table
        -> bounded drafting-parameter residual
        -> :func:`basic_blocks.build_basic_block`
        -> shared landmarks, Bezier curves, and sleeve/armhole constraints

Only the twelve T-shirt *design* parameters are optimized.  Body measurements
remain the category prior because the current GCD visual lane uses one neutral
body and cannot identify body measurement versus garment ease.  Fitting never
receives source-pattern targets; an oracle fit may be run separately by an
evaluator and must be labelled as a representational ceiling.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from benchmark.drafting_semantics.basic_blocks import (
    DESIGN_BOUNDS,
    BasicBlock,
    build_basic_block,
)
from benchmark.drafting_semantics.basic_semantic_targets import (
    BasicSemanticTarget,
    semantic_target_from_basic_block,
)
from benchmark.drafting_semantics.semantic_teacher_student import (
    MAX_COORDINATE_DIM,
    SEMANTIC_QUERY_INVENTORY,
    category_query_mask,
)
from benchmark.drafting_semantics.tshirt_parametric_decoder import (
    CanonicalPatternGraph,
    TShirtDraftParameters,
    TShirtParametricDraftingDecoder,
)


TSHIRT_DRAFT_PARAMETER_NAMES = tuple(DESIGN_BOUNDS["tshirt"])
"""Stable order of the bounded decoder parameters."""


@dataclass(frozen=True)
class TShirtProjectionConfig:
    """Predeclared fitting policy for the deterministic inverse decoder."""

    confidence_threshold: float = 0.50
    presence_threshold: float = 0.50
    prior_strength: float = 0.02
    max_function_evaluations: int = 800
    curve_samples: int = 24
    robust_loss: str = "soft_l1"
    robust_scale: float = 0.05

    def validate(self) -> None:
        for name in ("confidence_threshold", "presence_threshold"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not math.isfinite(self.prior_strength) or self.prior_strength < 0.0:
            raise ValueError("prior_strength must be finite and non-negative")
        if self.max_function_evaluations <= 0:
            raise ValueError("max_function_evaluations must be positive")
        if self.curve_samples < 8:
            raise ValueError("curve_samples must be at least 8")
        if self.robust_loss not in {"linear", "soft_l1", "huber"}:
            raise ValueError("unsupported robust_loss")
        if not math.isfinite(self.robust_scale) or self.robust_scale <= 0.0:
            raise ValueError("robust_scale must be finite and positive")


@dataclass(frozen=True)
class TShirtConstraintAudit:
    """Explicit postconditions that are stronger than generic validation."""

    front_armhole_length_cm: float
    back_armhole_length_cm: float
    front_sleeve_cap_length_cm: float
    back_sleeve_cap_length_cm: float
    total_sleeve_cap_ease_ratio: float
    front_mate_ratio: float
    back_mate_ratio: float
    requested_total_ease_ratio: float
    total_ease_absolute_error: float
    per_mate_within_declared_tolerance: bool
    shared_landmarks_are_single_source: bool
    symmetry_contract_present: bool

    @property
    def passed(self) -> bool:
        return bool(
            self.total_ease_absolute_error <= 5e-4
            and self.per_mate_within_declared_tolerance
            and self.shared_landmarks_are_single_source
            and self.symmetry_contract_present
        )

    def to_dict(self) -> dict[str, float | bool]:
        return {
            "front_armhole_length_cm": self.front_armhole_length_cm,
            "back_armhole_length_cm": self.back_armhole_length_cm,
            "front_sleeve_cap_length_cm": self.front_sleeve_cap_length_cm,
            "back_sleeve_cap_length_cm": self.back_sleeve_cap_length_cm,
            "total_sleeve_cap_ease_ratio": self.total_sleeve_cap_ease_ratio,
            "front_mate_ratio": self.front_mate_ratio,
            "back_mate_ratio": self.back_mate_ratio,
            "requested_total_ease_ratio": self.requested_total_ease_ratio,
            "total_ease_absolute_error": self.total_ease_absolute_error,
            "per_mate_within_declared_tolerance": self.per_mate_within_declared_tolerance,
            "shared_landmarks_are_single_source": self.shared_landmarks_are_single_source,
            "symmetry_contract_present": self.symmetry_contract_present,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class TShirtParametricProjection:
    """One fitted, validated decoder result and its reproducibility receipt."""

    block: BasicBlock
    normalized_parameters: tuple[float, ...]
    design_parameters_cm: Mapping[str, float]
    initial_data_loss: float
    final_data_loss: float
    objective_residual_rms: float
    selected_coordinate_count: int
    selected_query_keys: tuple[str, ...]
    optimizer_success: bool
    optimizer_status: int
    optimizer_message: str
    function_evaluations: int
    constraint_audit: TShirtConstraintAudit

    def to_receipt(self) -> dict[str, Any]:
        return {
            "schema_version": "tshirt-parametric-projection/v1",
            "parameter_order": list(TSHIRT_DRAFT_PARAMETER_NAMES),
            "normalized_parameters": list(self.normalized_parameters),
            "design_parameters_cm": dict(self.design_parameters_cm),
            "initial_data_loss": self.initial_data_loss,
            "final_data_loss": self.final_data_loss,
            "relative_data_loss_change_percent": (
                100.0
                * (self.final_data_loss - self.initial_data_loss)
                / max(self.initial_data_loss, 1e-12)
            ),
            "objective_residual_rms": self.objective_residual_rms,
            "selected_coordinate_count": self.selected_coordinate_count,
            "selected_query_keys": list(self.selected_query_keys),
            "optimizer": {
                "success": self.optimizer_success,
                "status": self.optimizer_status,
                "message": self.optimizer_message,
                "function_evaluations": self.function_evaluations,
            },
            "constraint_audit": self.constraint_audit.to_dict(),
            "claim_boundary": (
                "The fitted values are bounded decoder parameters inferred from a "
                "semantic observation, not recovered anthropometric measurements or "
                "industrial drafting truth."
            ),
        }


def default_normalized_parameters() -> np.ndarray:
    """Return the category-prior design vector in unit-bound coordinates."""

    values = []
    for name in TSHIRT_DRAFT_PARAMETER_NAMES:
        bound = DESIGN_BOUNDS["tshirt"][name]
        values.append((bound.default - bound.low) / (bound.high - bound.low))
    return np.asarray(values, dtype=np.float64)


def design_from_normalized(values: Sequence[float]) -> dict[str, float]:
    """Decode a unit-bound residual vector into named centimetre parameters."""

    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (len(TSHIRT_DRAFT_PARAMETER_NAMES),):
        raise ValueError(
            f"normalized T-shirt parameter vector must have shape "
            f"({len(TSHIRT_DRAFT_PARAMETER_NAMES)},)"
        )
    if not np.isfinite(vector).all() or bool(np.any(vector < 0.0)) or bool(np.any(vector > 1.0)):
        raise ValueError("normalized T-shirt parameters must be finite and inside [0, 1]")
    output = {}
    for name, value in zip(TSHIRT_DRAFT_PARAMETER_NAMES, vector):
        bound = DESIGN_BOUNDS["tshirt"][name]
        output[name] = float(bound.low + float(value) * (bound.high - bound.low))
    return output


def normalized_from_design(values: Mapping[str, float]) -> np.ndarray:
    """Encode a complete named design mapping into the stable unit vector."""

    if set(values) != set(TSHIRT_DRAFT_PARAMETER_NAMES):
        missing = sorted(set(TSHIRT_DRAFT_PARAMETER_NAMES) - set(values))
        extra = sorted(set(values) - set(TSHIRT_DRAFT_PARAMETER_NAMES))
        raise ValueError(f"design parameter schema mismatch; missing={missing}, extra={extra}")
    output = []
    for name in TSHIRT_DRAFT_PARAMETER_NAMES:
        bound = DESIGN_BOUNDS["tshirt"][name]
        value = float(values[name])
        bound.validate(value, name)
        output.append((value - bound.low) / (bound.high - bound.low))
    return np.asarray(output, dtype=np.float64)


def decode_tshirt_parameters(
    normalized_parameters: Sequence[float],
    *,
    sample_id: str = "tshirt_parametric_decoder",
) -> BasicBlock:
    """Run the existing shared-point drafting decoder and seam solver."""

    block = build_basic_block(
        "tshirt",
        design=design_from_normalized(normalized_parameters),
        sample_id=sample_id,
        metadata={
            "decoder": "bounded_tshirt_design_parameters_to_basic_block_v1",
            "body_prior": "fixed_category_default_measurements",
            "parameter_residual_source": "semantic_projection",
        },
    )
    block.validate()
    return block


def _as_coordinate_table(values: Any, name: str) -> np.ndarray:
    output = np.asarray(values, dtype=np.float64)
    expected = (len(SEMANTIC_QUERY_INVENTORY), MAX_COORDINATE_DIM)
    if output.shape != expected:
        raise ValueError(f"{name} must have shape {expected}")
    return output


def _as_query_vector(values: Any, name: str) -> np.ndarray:
    output = np.asarray(values, dtype=np.float64)
    expected = (len(SEMANTIC_QUERY_INVENTORY),)
    if output.shape != expected:
        raise ValueError(f"{name} must have shape {expected}")
    return output


def _angle_delta(value: np.ndarray) -> np.ndarray:
    """Wrap angle/pi residuals to the shortest direction (period two)."""

    return (value + 1.0) % 2.0 - 1.0


def _coordinate_delta(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    delta = np.asarray(first - second, dtype=np.float64)
    for index, query in enumerate(SEMANTIC_QUERY_INVENTORY):
        for channel, coordinate_name in enumerate(query.coordinate_names):
            if "tangent_angle_norm" in coordinate_name:
                delta[index, channel] = _angle_delta(delta[index, channel])
    return delta


def _weighted_data_residual(
    target: BasicSemanticTarget,
    observation: np.ndarray,
    support: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    delta = _coordinate_delta(target.coordinates, observation)
    return (delta * weights)[support]


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if len(values) else 0.0


def fit_tshirt_drafting_parameters(
    predicted_coordinates: Any,
    presence_probability: Any,
    coordinate_confidence: Any,
    *,
    query_mask: Any | None = None,
    config: TShirtProjectionConfig | None = None,
    sample_id: str = "tshirt_parametric_projection",
) -> TShirtParametricProjection:
    """Fit decoder parameters using only a semantic observation.

    ``predicted_coordinates`` may be a visual-student prediction or a target
    table for a separately labelled oracle-capacity audit.  The function has
    no target/mask argument and therefore cannot accidentally use hidden test
    truth while fitting the image-only lane.
    """

    resolved = config or TShirtProjectionConfig()
    resolved.validate()
    observation = _as_coordinate_table(predicted_coordinates, "predicted_coordinates")
    presence = _as_query_vector(presence_probability, "presence_probability")
    confidence = _as_query_vector(coordinate_confidence, "coordinate_confidence")
    if not np.isfinite(presence).all() or not np.isfinite(confidence).all():
        raise ValueError("presence and confidence must contain only finite values")
    if bool(np.any(presence < 0.0)) or bool(np.any(presence > 1.0)):
        raise ValueError("presence probabilities must be in [0, 1]")
    if bool(np.any(confidence < 0.0)) or bool(np.any(confidence > 1.0)):
        raise ValueError("coordinate confidence must be in [0, 1]")

    category_mask = np.asarray(category_query_mask("tshirt"), dtype=np.bool_)
    if query_mask is not None:
        supplied = np.asarray(query_mask, dtype=np.bool_)
        if supplied.shape != category_mask.shape:
            raise ValueError(f"query_mask must have shape {category_mask.shape}")
        category_mask &= supplied
    anchor_parameters = default_normalized_parameters()
    anchor_target = semantic_target_from_basic_block(
        decode_tshirt_parameters(anchor_parameters, sample_id=f"{sample_id}_anchor"),
        curve_samples=resolved.curve_samples,
    )
    selected_queries = (
        category_mask
        & anchor_target.query_applicability
        & (anchor_target.presence > 0.5)
        & (presence >= resolved.presence_threshold)
        & (confidence >= resolved.confidence_threshold)
    )
    support = (
        anchor_target.coordinate_mask
        & selected_queries[:, None]
        & np.isfinite(observation)
    )
    if not bool(support.any()):
        raise ValueError("no confident T-shirt semantic coordinates are available for fitting")
    weights = np.sqrt(np.clip(confidence, 0.0, 1.0))[:, None]
    weights = np.broadcast_to(weights, support.shape)
    initial_data_residual = _weighted_data_residual(
        anchor_target, observation, support, weights
    )

    def residual(vector: np.ndarray) -> np.ndarray:
        block = decode_tshirt_parameters(vector, sample_id=sample_id)
        semantic = semantic_target_from_basic_block(
            block, curve_samples=resolved.curve_samples
        )
        data = _weighted_data_residual(semantic, observation, support, weights)
        if resolved.prior_strength <= 0.0:
            return data
        prior = math.sqrt(resolved.prior_strength) * (vector - anchor_parameters)
        return np.concatenate((data, prior))

    # SciPy is already part of the benchmark runtime used by the canonical
    # pattern and sewing pipeline.  Import it lazily so schema-only consumers
    # can still import this module in the lightweight preprocessing venv.
    try:
        from scipy.optimize import least_squares
    except ImportError as exc:  # pragma: no cover - exercised in lightweight envs
        raise RuntimeError(
            "T-shirt parametric fitting requires the benchmark runtime with SciPy"
        ) from exc

    result = least_squares(
        residual,
        anchor_parameters,
        bounds=(np.zeros_like(anchor_parameters), np.ones_like(anchor_parameters)),
        loss=resolved.robust_loss,
        f_scale=resolved.robust_scale,
        max_nfev=resolved.max_function_evaluations,
        x_scale="jac",
        # BasicBlock serializes centimetres at six decimal places.  SciPy's
        # default machine-epsilon step can therefore observe an all-zero
        # numerical Jacobian even though the drafting map is responsive.
        diff_step=1e-3,
        xtol=1e-9,
        ftol=1e-9,
        gtol=1e-9,
    )
    fitted = np.clip(np.asarray(result.x, dtype=np.float64), 0.0, 1.0)
    block = decode_tshirt_parameters(fitted, sample_id=sample_id)
    final_target = semantic_target_from_basic_block(
        block, curve_samples=resolved.curve_samples
    )
    final_data_residual = _weighted_data_residual(
        final_target, observation, support, weights
    )
    audit = audit_tshirt_constraints(block)
    if not audit.passed:
        raise RuntimeError("parametric decoder violated its hard T-shirt constraints")
    selected_keys = tuple(
        query.key
        for query, selected in zip(SEMANTIC_QUERY_INVENTORY, selected_queries)
        if bool(selected)
    )
    objective = residual(fitted)
    return TShirtParametricProjection(
        block=block,
        normalized_parameters=tuple(float(value) for value in fitted),
        design_parameters_cm=block.design,
        initial_data_loss=_rms(initial_data_residual),
        final_data_loss=_rms(final_data_residual),
        objective_residual_rms=_rms(objective),
        selected_coordinate_count=int(support.sum()),
        selected_query_keys=selected_keys,
        optimizer_success=bool(result.success),
        optimizer_status=int(result.status),
        optimizer_message=str(result.message),
        function_evaluations=int(result.nfev),
        constraint_audit=audit,
    )


def audit_tshirt_constraints(block: BasicBlock) -> TShirtConstraintAudit:
    """Audit the coupled decoder rather than relying on generic warnings."""

    if block.category != "tshirt":
        raise ValueError("T-shirt constraint audit requires a T-shirt block")
    block.validate()
    front = block.panel("front")
    back = block.panel("back")
    sleeve = block.panel("sleeve")
    values = sleeve.metadata
    front_armhole = float(values["front_armhole_length_cm"])
    back_armhole = float(values["back_armhole_length_cm"])
    front_cap = float(values["front_cap_length_cm"])
    back_cap = float(values["back_cap_length_cm"])
    armhole_total = front_armhole + back_armhole
    cap_total = front_cap + back_cap
    requested = 1.01
    total_ratio = cap_total / armhole_total
    front_ratio = front_cap / front_armhole
    back_ratio = back_cap / back_armhole
    shared = all(
        path.landmark_sequence[0] in {item.name for item in panel.landmarks}
        and path.landmark_sequence[-1] in {item.name for item in panel.landmarks}
        for panel in (front, back, sleeve)
        for path in panel.paths
    )
    symmetry = bool(
        front.symmetry.cut_on_fold
        and back.symmetry.cut_on_fold
        and sleeve.symmetry.kind == "mirrored_pair"
        and sleeve.metadata.get("front_back_cap_halves_are_not_interchangeable")
    )
    return TShirtConstraintAudit(
        front_armhole_length_cm=front_armhole,
        back_armhole_length_cm=back_armhole,
        front_sleeve_cap_length_cm=front_cap,
        back_sleeve_cap_length_cm=back_cap,
        total_sleeve_cap_ease_ratio=total_ratio,
        front_mate_ratio=front_ratio,
        back_mate_ratio=back_ratio,
        requested_total_ease_ratio=requested,
        total_ease_absolute_error=abs(total_ratio - requested),
        per_mate_within_declared_tolerance=bool(
            abs(front_ratio - 1.0) <= 0.12 + 1e-6
            and abs(back_ratio - 1.0) <= 0.12 + 1e-6
        ),
        shared_landmarks_are_single_source=shared,
        symmetry_contract_present=symmetry,
    )


def materialize_tshirt_projection_graph(
    projection: TShirtParametricProjection,
    *,
    pattern_id: str = "tshirt_parametric_projection",
    samples_per_cubic: int = 33,
) -> CanonicalPatternGraph:
    """Materialize a fitted archetype as an explicit physical pattern graph.

    The inverse fit uses BasicBlock's positive-x semantic archetype because that
    is the frame used by the frozen teacher/student targets.  Final export is a
    separate operation: left/right instances are reflected exactly, shared
    landmarks are referenced by id, and sleeve-cap height is re-solved against
    the additive seam equation.  Converting BasicBlock's declared 1% ease to a
    centimetre value preserves the fitted archetype's intended sleeve relation.
    """

    audit = projection.constraint_audit
    decoder = TShirtParametricDraftingDecoder(samples_per_cubic=samples_per_cubic)
    base_parameters = TShirtDraftParameters.from_mapping(
        projection.design_parameters_cm
    )
    # BasicBlock metadata is rounded for portable JSON.  Probe the exact cubic
    # graph instead so a rounded receipt cannot leak into the hard constraint.
    probe = decoder.decode(base_parameters, pattern_id=f"{pattern_id}_ease_probe")
    exact_armhole_total_cm = (
        probe.sleeve_head_constraint.front_armhole_length_cm
        + probe.sleeve_head_constraint.back_armhole_length_cm
    )
    additive_ease_cm = exact_armhole_total_cm * (
        audit.requested_total_ease_ratio - 1.0
    )
    parameters = TShirtDraftParameters.from_mapping(
        {"sleeve_ease_cm": additive_ease_cm},
        base=base_parameters,
    )
    graph = decoder.decode(parameters, pattern_id=pattern_id)
    graph.validate()
    return graph


__all__ = [
    "TSHIRT_DRAFT_PARAMETER_NAMES",
    "TShirtConstraintAudit",
    "TShirtParametricProjection",
    "TShirtProjectionConfig",
    "audit_tshirt_constraints",
    "decode_tshirt_parameters",
    "default_normalized_parameters",
    "design_from_normalized",
    "fit_tshirt_drafting_parameters",
    "materialize_tshirt_projection_graph",
    "normalized_from_design",
]
