"""Causal bridge contracts for Pattern DSL and paired four-view evidence.

This module is deliberately additive.  It consumes cached Pattern DSL element
tokens and cached four-view FPN tokens without changing either pretrained
encoder.  Pattern-only counterfactuals can supervise intervention/geometry
heads, while visual-effect heads require a fail-closed render receipt.

The inverse model accepts target *images/features* and an anchor pattern.  It
never accepts the target Pattern DSL; :func:`validate_inverse_input_contract`
enforces that boundary for mapping-based callers.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from numbers import Real
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from benchmark.drafting_semantics.counterfactual_pairs import (
    CounterfactualContractError,
    counterfactual_training_eligibility,
    file_sha256,
    validate_four_view_receipt,
)
from benchmark.drafting_semantics.gcdv2_surface_correspondence import (
    SCHEMA_VERSION as GCDV2_SURFACE_SCHEMA_VERSION,
    TSHIRT_PARAMETER_NAMES as _SURFACE_PARAMETER_NAMES,
)


EFFECT_RECEIPT_SCHEMA_VERSION = "pattern-visual-effect-render-receipt/v1"
INVERSE_INPUT_CONTRACT_VERSION = "pattern-visual-inverse-input/v1"
TSHIRT_OBSERVABLE_SCHEMA_VERSION = "tshirt-observable-pattern-axes/v1"
TSHIRT_DECODER_RESIDUAL_SCHEMA_VERSION = "tshirt-decoder-native-residual/v1"
TSHIRT_OBSERVABLE_ADAPTER_SCHEMA_VERSION = "tshirt-observable-to-decoder-adapter/v1"


class PatternVisualEffectContractError(ValueError):
    """Raised when a causal-data or leakage boundary is not satisfied."""


@dataclass(frozen=True)
class ObservableAxis:
    """A quantity measured from completed exact 2D pattern geometry.

    These axes are supervision emitted by ``gcdv2_surface_correspondence``.
    They are not necessarily the author-facing or decoder-native variables
    that caused the measured geometry.
    """

    name: str
    unit: str
    definition: str
    affected_surface_elements: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name or self.unit not in {"cm", "deg"}:
            raise ValueError("observable axis requires a name and cm/deg unit")
        if not self.definition or not self.affected_surface_elements:
            raise ValueError(f"{self.name} requires a definition and affected elements")


_OBSERVABLE_DEFINITIONS = {
    "neck_width_cm": "front half-pattern neckline endpoint X separation",
    "front_neck_depth_cm": "front half-pattern neckline endpoint Y separation",
    "shoulder_slope_deg": "mean absolute front/back shoulder-edge angle from horizontal",
    "armhole_depth_cm": "front/back armhole path endpoint Y span",
    "body_length_cm": "mean complete front/back bodice panel vertical extent",
    "sleeve_cap_height_cm": "half-sleeve path maximum chord-normal height",
    "sleeve_length_cm": "mean half-sleeve panel longitudinal extent",
    "sleeve_width_cm": "sum of the two non-stitched opening edges of one sleeve",
}
_LOCAL_OBSERVABLE_AXIS_NAMES = (
    "neck_width_cm",
    "front_neck_depth_cm",
    "shoulder_slope_deg",
    "armhole_depth_cm",
    "body_length_cm",
    "sleeve_cap_height_cm",
    "sleeve_length_cm",
    "sleeve_width_cm",
)
if tuple(_SURFACE_PARAMETER_NAMES) != _LOCAL_OBSERVABLE_AXIS_NAMES:
    raise RuntimeError(
        "GCDv2 surface observable schema drifted from the visual-effect bridge"
    )

# This is deliberately bridge-owned.  The surface-correspondence report may
# use a different visualization vocabulary; training queries are the 15
# front/back-separated elements consumed by this module.
_OBSERVABLE_AFFECTED_QUERIES = {
    "neck_width_cm": (
        "front_neckline", "back_neckline", "front_shoulder", "back_shoulder",
    ),
    "front_neck_depth_cm": ("front_neckline", "front_center"),
    "shoulder_slope_deg": ("front_shoulder", "back_shoulder"),
    "armhole_depth_cm": (
        "front_armhole", "back_armhole", "front_side_seam", "back_side_seam",
    ),
    "body_length_cm": (
        "front_center", "back_center", "front_side_seam", "back_side_seam",
        "front_hemline", "back_hemline",
    ),
    "sleeve_cap_height_cm": ("sleeve_head", "front_armhole", "back_armhole"),
    "sleeve_length_cm": ("sleeve_underarm", "sleeve_hem"),
    "sleeve_width_cm": ("sleeve_underarm", "sleeve_hem"),
}
TSHIRT_OBSERVABLE_AXES: tuple[ObservableAxis, ...] = tuple(
    ObservableAxis(
        name,
        "deg" if name.endswith("_deg") else "cm",
        _OBSERVABLE_DEFINITIONS[name],
        _OBSERVABLE_AFFECTED_QUERIES[name],
    )
    for name in _LOCAL_OBSERVABLE_AXIS_NAMES
)
TSHIRT_OBSERVABLE_AXIS_NAMES = tuple(item.name for item in TSHIRT_OBSERVABLE_AXES)


@dataclass(frozen=True)
class SemanticParameter:
    """One decoder-native residual parameter, not a measured surface axis."""

    name: str
    unit: str
    minimum: float
    maximum: float
    default: float
    affected_elements: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name or self.unit != "cm":
            raise ValueError("semantic parameter requires a name and centimetre unit")
        if not self.minimum < self.maximum:
            raise ValueError(f"invalid bounds for {self.name}")
        if not self.minimum <= self.default <= self.maximum:
            raise ValueError(f"default outside bounds for {self.name}")
        if not self.affected_elements:
            raise ValueError(f"{self.name} requires at least one affected element")


# A compact first-stage schema.  Every name is directly consumed by the
# existing TShirtDraftParameters decoder; no raw vertex or Bezier control is a
# regression target here.
TSHIRT_SEMANTIC_PARAMETERS: tuple[SemanticParameter, ...] = (
    SemanticParameter(
        "chest_ease_cm", "cm", 6.0, 16.0, 10.0,
        ("front_side_seam", "back_side_seam", "front_armhole", "back_armhole"),
    ),
    SemanticParameter(
        "body_length_cm", "cm", 58.0, 72.0, 64.0,
        ("front_center", "back_center", "front_side_seam", "back_side_seam", "front_hemline", "back_hemline"),
    ),
    SemanticParameter(
        "neck_width_cm", "cm", 6.8, 9.0, 7.6,
        ("front_neckline", "back_neckline", "front_shoulder", "back_shoulder"),
    ),
    SemanticParameter(
        "front_neck_depth_cm", "cm", 6.5, 10.5, 8.2,
        ("front_neckline", "front_center"),
    ),
    SemanticParameter(
        "shoulder_drop_cm", "cm", 1.5, 3.5, 2.4,
        ("front_shoulder", "back_shoulder", "front_armhole", "back_armhole"),
    ),
    SemanticParameter(
        "armhole_depth_cm", "cm", 18.5, 25.5, 21.5,
        ("front_armhole", "back_armhole", "front_side_seam", "back_side_seam", "sleeve_head"),
    ),
    SemanticParameter(
        "sleeve_length_cm", "cm", 16.0, 28.0, 21.0,
        ("sleeve_underarm", "sleeve_hem"),
    ),
    SemanticParameter(
        "sleeve_ease_cm", "cm", 0.0, 4.0, 0.5,
        ("front_armhole", "back_armhole", "sleeve_head"),
    ),
)
TSHIRT_SEMANTIC_PARAMETER_NAMES = tuple(item.name for item in TSHIRT_SEMANTIC_PARAMETERS)

# This is the legacy compact decoder-head schema.  It remains exported for
# old checkpoints, but is neither the GCDv2 observable schema nor the complete
# target vocabulary of the observable-to-decoder adapter below.


@dataclass(frozen=True)
class ObservableDecoderRelation:
    observable_axis: str
    decoder_parameters: tuple[str, ...]
    conversion_kind: str
    explanation: str

    def __post_init__(self) -> None:
        if self.observable_axis not in TSHIRT_OBSERVABLE_AXIS_NAMES:
            raise ValueError(f"unknown observable axis: {self.observable_axis}")
        if not self.decoder_parameters or not self.conversion_kind or not self.explanation:
            raise ValueError(f"incomplete adapter relation for {self.observable_axis}")


# Same spelling does not imply numeric identity.  For example, the GCDv2
# observable ``neck_width_cm`` is an endpoint separation, while the decoder's
# value is an input to its own drafting formulas.  Every relation therefore
# requires an evidenced calibration.  Sleeve-cap height and sleeve width are
# intentionally one-to-many mappings.
TSHIRT_OBSERVABLE_TO_DECODER_RELATIONS: tuple[ObservableDecoderRelation, ...] = (
    ObservableDecoderRelation(
        "neck_width_cm", ("neck_width_cm",), "CALIBRATION_REQUIRED",
        "measured endpoint separation must be calibrated to the decoder input",
    ),
    ObservableDecoderRelation(
        "front_neck_depth_cm", ("front_neck_depth_cm",), "CALIBRATION_REQUIRED",
        "measured neckline depth and decoder design depth share units but not definitions",
    ),
    ObservableDecoderRelation(
        "shoulder_slope_deg", ("shoulder_drop_cm",), "GEOMETRIC_CONTEXT_REQUIRED",
        "degrees convert to vertical drop only with the decoder shoulder run/body context",
    ),
    ObservableDecoderRelation(
        "armhole_depth_cm", ("armhole_depth_cm",), "CALIBRATION_REQUIRED",
        "measured endpoint span must be calibrated to the decoder drafting input",
    ),
    ObservableDecoderRelation(
        "body_length_cm", ("body_length_cm",), "CALIBRATION_REQUIRED",
        "measured center-boundary length is not assumed equal to the decoder input",
    ),
    ObservableDecoderRelation(
        "sleeve_cap_height_cm", ("sleeve_ease_cm", "armhole_depth_cm"), "SOLVER_INVERSE_REQUIRED",
        "cap height is derived by the sleeve-head/armhole solver and has no direct decoder knob",
    ),
    ObservableDecoderRelation(
        "sleeve_length_cm", ("sleeve_length_cm",), "CALIBRATION_REQUIRED",
        "measured panel extent must be calibrated to the decoder sleeve-length input",
    ),
    ObservableDecoderRelation(
        "sleeve_width_cm", ("bicep_ease_cm", "sleeve_hem_reduction_cm"), "MULTI_PARAMETER_CALIBRATION_REQUIRED",
        "one opening-width observation cannot identify bicep ease and hem reduction without calibration",
    ),
)
_OBSERVABLE_RELATION_BY_NAME = {
    item.observable_axis: item for item in TSHIRT_OBSERVABLE_TO_DECODER_RELATIONS
}
TSHIRT_ADAPTER_DECODER_PARAMETER_NAMES = tuple(
    dict.fromkeys(
        parameter
        for relation in TSHIRT_OBSERVABLE_TO_DECODER_RELATIONS
        for parameter in relation.decoder_parameters
    )
)
_DECODER_PARAMETER_METADATA = {
    item.name: item for item in TSHIRT_SEMANTIC_PARAMETERS
}
_DECODER_PARAMETER_METADATA.update(
    {
        "bicep_ease_cm": SemanticParameter(
            "bicep_ease_cm", "cm", 4.0, 10.0, 6.0,
            ("sleeve_head", "sleeve_underarm", "sleeve_hem"),
        ),
        "sleeve_hem_reduction_cm": SemanticParameter(
            "sleeve_hem_reduction_cm", "cm", 1.0, 5.0, 2.5,
            ("sleeve_underarm", "sleeve_hem"),
        ),
    }
)
TSHIRT_DECODER_RESIDUAL_PARAMETER_NAMES = TSHIRT_ADAPTER_DECODER_PARAMETER_NAMES
TSHIRT_DECODER_RESIDUAL_PARAMETERS = tuple(
    _DECODER_PARAMETER_METADATA[name]
    for name in TSHIRT_DECODER_RESIDUAL_PARAMETER_NAMES
)


@dataclass(frozen=True)
class ObservableAxisCalibration:
    """Evidenced local linearization from one observable delta to decoder deltas."""

    observable_axis: str
    decoder_coefficients: tuple[tuple[str, float], ...]
    calibration_id: str
    scope_id: str
    evidence: str

    @classmethod
    def from_mapping(
        cls,
        observable_axis: str,
        value: Mapping[str, Any],
    ) -> "ObservableAxisCalibration":
        if not isinstance(value, Mapping):
            raise PatternVisualEffectContractError(
                f"{observable_axis}: calibration must be a mapping"
            )
        coefficients = value.get("decoder_coefficients")
        if not isinstance(coefficients, Mapping):
            raise PatternVisualEffectContractError(
                f"{observable_axis}: decoder_coefficients mapping is required"
            )
        return cls(
            observable_axis=observable_axis,
            decoder_coefficients=tuple(
                (str(name), _finite_number(coefficient, f"{observable_axis}.{name}"))
                for name, coefficient in sorted(coefficients.items())
            ),
            calibration_id=_required_text(value.get("calibration_id"), f"{observable_axis}.calibration_id"),
            scope_id=_required_text(value.get("scope_id"), f"{observable_axis}.scope_id"),
            evidence=_required_text(value.get("evidence"), f"{observable_axis}.evidence"),
        )

    def __post_init__(self) -> None:
        relation = _OBSERVABLE_RELATION_BY_NAME.get(self.observable_axis)
        if relation is None:
            raise PatternVisualEffectContractError(
                f"unknown calibrated observable axis {self.observable_axis!r}"
            )
        if not self.decoder_coefficients:
            raise PatternVisualEffectContractError(
                f"{self.observable_axis}: at least one decoder coefficient is required"
            )
        names = tuple(name for name, _ in self.decoder_coefficients)
        if len(names) != len(set(names)):
            raise PatternVisualEffectContractError(
                f"{self.observable_axis}: duplicate decoder coefficient"
            )
        unsupported = sorted(set(names) - set(relation.decoder_parameters))
        if unsupported:
            raise PatternVisualEffectContractError(
                f"{self.observable_axis}: calibration targets unsupported decoder parameters: "
                + ", ".join(unsupported)
            )
        missing = sorted(set(relation.decoder_parameters) - set(names))
        if missing:
            raise PatternVisualEffectContractError(
                f"{self.observable_axis}: calibration omits required decoder parameters: "
                + ", ".join(missing)
            )
        for name, value in self.decoder_coefficients:
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
                raise PatternVisualEffectContractError(
                    f"{self.observable_axis}.{name} coefficient must be finite"
                )
        _required_text(self.calibration_id, f"{self.observable_axis}.calibration_id")
        _required_text(self.scope_id, f"{self.observable_axis}.scope_id")
        _required_text(self.evidence, f"{self.observable_axis}.evidence")


@dataclass(frozen=True)
class ObservableResidualAdapterResult:
    observable_residuals: Mapping[str, float]
    decoder_residuals_cm: Mapping[str, float]
    unresolved_axes: tuple[str, ...]
    receipt: Mapping[str, Any]


@dataclass(frozen=True)
class ElementQuery:
    name: str
    panel_roles: tuple[str, ...]
    edge_roles: tuple[str, ...]
    symmetric_instances: bool = True

    def __post_init__(self) -> None:
        if not self.name or not self.panel_roles or not self.edge_roles:
            raise ValueError("element query requires name, panel roles, and edge roles")


TSHIRT_ELEMENT_QUERIES: tuple[ElementQuery, ...] = (
    ElementQuery("front_neckline", ("front_bodice",), ("neckline",)),
    ElementQuery("back_neckline", ("back_bodice",), ("neckline",)),
    ElementQuery("front_shoulder", ("front_bodice",), ("shoulder",)),
    ElementQuery("back_shoulder", ("back_bodice",), ("shoulder",)),
    ElementQuery("front_armhole", ("front_bodice",), ("armhole",)),
    ElementQuery("back_armhole", ("back_bodice",), ("armhole",)),
    ElementQuery("front_center", ("front_bodice",), ("center_front",), False),
    ElementQuery("back_center", ("back_bodice",), ("center_back",), False),
    ElementQuery("front_side_seam", ("front_bodice",), ("side_seam",)),
    ElementQuery("back_side_seam", ("back_bodice",), ("side_seam",)),
    ElementQuery("front_hemline", ("front_bodice",), ("hemline", "waistline")),
    ElementQuery("back_hemline", ("back_bodice",), ("hemline", "waistline")),
    ElementQuery("sleeve_head", ("sleeve",), ("sleeve_head",)),
    ElementQuery("sleeve_underarm", ("sleeve",), ("sleeve_underarm",)),
    ElementQuery("sleeve_hem", ("sleeve",), ("sleeve_hem",)),
)
TSHIRT_ELEMENT_QUERY_NAMES = tuple(item.name for item in TSHIRT_ELEMENT_QUERIES)


_SOURCE_PARAMETER_ALIASES: dict[str, str] = {
    "shirt.length": "body_length_cm",
    "lengthBonus": "body_length_cm",
    "collar.width": "neck_width_cm",
    "necklineWidth": "neck_width_cm",
    "collar.fc_depth": "front_neck_depth_cm",
    "necklineDepth": "front_neck_depth_cm",
    "sleeve.length": "sleeve_length_cm",
    "sleeveLength": "sleeve_length_cm",
}


@dataclass(frozen=True)
class CounterfactualVisualExample:
    """One pattern-only intervention, optionally awaiting visual evidence."""

    pair_id: str
    base_group_id: str
    split: str
    source: str
    source_parameter: str
    observable_axis: str | None
    baseline_value: float
    intervention_value: float
    baseline_pattern_path: str
    intervention_pattern_path: str
    baseline_pattern_sha256: str
    intervention_pattern_sha256: str
    unchanged_state_fingerprint: str
    expected_elements: tuple[str, ...]
    semantic_delta: Mapping[str, Any]
    pattern_only: bool = True
    render_status: str = "PENDING_VALIDATED_SIMULATOR"

    @property
    def source_delta(self) -> float:
        return self.intervention_value - self.baseline_value

    @property
    def semantic_parameter(self) -> str | None:
        """Deprecated read alias; this value is a measured observable axis."""

        return self.observable_axis


_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_VALID_SPLITS = frozenset({"train", "validation", "test", "unassigned"})


def _required_text(value: Any, label: str) -> str:
    result = "" if value is None else str(value).strip()
    if not result:
        raise PatternVisualEffectContractError(f"{label} is required")
    return result


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise PatternVisualEffectContractError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise PatternVisualEffectContractError(f"{label} must be finite")
    return result


def adapt_observable_residuals_to_decoder(
    observable_residual: Mapping[str, Any] | Sequence[Any],
    *,
    calibrations: Mapping[str, ObservableAxisCalibration | Mapping[str, Any]],
    observable_valid: Mapping[str, bool] | Sequence[bool] | None = None,
    require_complete: bool = True,
) -> ObservableResidualAdapterResult:
    """Map measured-geometry deltas to decoder inputs with explicit evidence.

    This function has no identity fallback.  Even when an observable and a
    decoder input share a spelling/unit, callers must supply a scoped
    calibration.  That prevents, for example, an observed neckline endpoint
    separation from silently being treated as the decoder's drafting knob.
    """

    if not isinstance(calibrations, Mapping):
        raise PatternVisualEffectContractError("calibrations must be a mapping")
    if isinstance(observable_residual, Mapping):
        unknown = sorted(set(observable_residual) - set(TSHIRT_OBSERVABLE_AXIS_NAMES))
        if unknown:
            raise PatternVisualEffectContractError(
                "unknown observable residuals: " + ", ".join(unknown)
            )
        values = {
            name: _finite_number(value, f"observable_residual.{name}")
            for name, value in observable_residual.items()
        }
    else:
        if (
            isinstance(observable_residual, (str, bytes))
            or len(observable_residual) != len(TSHIRT_OBSERVABLE_AXIS_NAMES)
        ):
            raise PatternVisualEffectContractError(
                "observable residual vector must contain "
                f"{len(TSHIRT_OBSERVABLE_AXIS_NAMES)} values"
            )
        values = {
            name: _finite_number(value, f"observable_residual.{name}")
            for name, value in zip(TSHIRT_OBSERVABLE_AXIS_NAMES, observable_residual)
        }

    if not values:
        raise PatternVisualEffectContractError("at least one observable residual is required")
    if observable_valid is None:
        mask = {name: True for name in values}
    elif isinstance(observable_valid, Mapping):
        unknown = sorted(set(observable_valid) - set(TSHIRT_OBSERVABLE_AXIS_NAMES))
        if unknown:
            raise PatternVisualEffectContractError(
                "unknown observable masks: " + ", ".join(unknown)
            )
        mask = {name: bool(observable_valid.get(name, False)) for name in values}
    else:
        if (
            isinstance(observable_valid, (str, bytes))
            or len(observable_valid) != len(TSHIRT_OBSERVABLE_AXIS_NAMES)
        ):
            raise PatternVisualEffectContractError(
                "observable mask must contain "
                f"{len(TSHIRT_OBSERVABLE_AXIS_NAMES)} values"
            )
        full_mask = dict(
            zip(TSHIRT_OBSERVABLE_AXIS_NAMES, (bool(value) for value in observable_valid))
        )
        mask = {name: full_mask[name] for name in values}

    unknown_calibrations = sorted(set(calibrations) - set(TSHIRT_OBSERVABLE_AXIS_NAMES))
    if unknown_calibrations:
        raise PatternVisualEffectContractError(
            "calibrations contain unknown observable axes: "
            + ", ".join(unknown_calibrations)
        )

    active = tuple(name for name in TSHIRT_OBSERVABLE_AXIS_NAMES if name in values and mask[name])
    resolved_calibrations: dict[str, ObservableAxisCalibration] = {}
    unresolved: list[str] = []
    for name in active:
        source = calibrations.get(name)
        if source is None:
            unresolved.append(name)
            continue
        calibration = (
            source
            if isinstance(source, ObservableAxisCalibration)
            else ObservableAxisCalibration.from_mapping(name, source)
        )
        if calibration.observable_axis != name:
            raise PatternVisualEffectContractError(
                f"calibration key {name!r} contains axis {calibration.observable_axis!r}"
            )
        resolved_calibrations[name] = calibration
    if unresolved and require_complete:
        raise PatternVisualEffectContractError(
            "explicit observable-to-decoder calibration is missing for: "
            + ", ".join(unresolved)
        )

    decoder_residuals = {
        name: 0.0 for name in TSHIRT_ADAPTER_DECODER_PARAMETER_NAMES
    }
    for name, calibration in resolved_calibrations.items():
        delta = values[name]
        for decoder_name, coefficient in calibration.decoder_coefficients:
            decoder_residuals[decoder_name] += delta * coefficient
    decoder_residuals = {
        name: value for name, value in decoder_residuals.items() if value != 0.0
    }
    calibration_receipts = {
        name: {
            "conversion_kind": _OBSERVABLE_RELATION_BY_NAME[name].conversion_kind,
            "observable_unit": next(
                item.unit for item in TSHIRT_OBSERVABLE_AXES if item.name == name
            ),
            "decoder_unit": "cm",
            "decoder_coefficients": dict(calibration.decoder_coefficients),
            "calibration_id": calibration.calibration_id,
            "scope_id": calibration.scope_id,
            "evidence": calibration.evidence,
        }
        for name, calibration in resolved_calibrations.items()
    }
    return ObservableResidualAdapterResult(
        observable_residuals={name: values[name] for name in active},
        decoder_residuals_cm=decoder_residuals,
        unresolved_axes=tuple(unresolved),
        receipt={
            "schema_version": TSHIRT_OBSERVABLE_ADAPTER_SCHEMA_VERSION,
            "status": (
                "PASS_EXPLICIT_CALIBRATION"
                if not unresolved
                else "PARTIAL_UNRESOLVED_CALIBRATION"
            ),
            "source_schema_version": GCDV2_SURFACE_SCHEMA_VERSION,
            "observable_schema_version": TSHIRT_OBSERVABLE_SCHEMA_VERSION,
            "decoder_schema_version": TSHIRT_DECODER_RESIDUAL_SCHEMA_VERSION,
            "identity_assumption_used": False,
            "active_observable_axes": list(active),
            "masked_observable_axes": sorted(name for name in values if not mask[name]),
            "unresolved_observable_axes": list(unresolved),
            "calibrations": calibration_receipts,
        },
    )


def _sha256(value: Any, label: str) -> str:
    result = str(value).lower()
    if not _HEX_64.fullmatch(result):
        raise PatternVisualEffectContractError(f"{label} must be a lowercase SHA-256")
    return result


def _base_group_id(row: Mapping[str, Any]) -> str:
    explicit = str(row.get("base_group_id", "")).strip()
    if explicit:
        return explicit
    source = _required_text(row.get("source"), "source")
    baseline_hash = _sha256(row.get("baseline_pattern_sha256"), "baseline_pattern_sha256")
    return f"{source}:{baseline_hash}"


def assert_base_group_split_integrity(
    examples: Sequence[CounterfactualVisualExample],
) -> dict[str, str]:
    """Ensure all interventions from one baseline live in one split."""

    observed: dict[str, str] = {}
    for item in examples:
        if item.split not in _VALID_SPLITS:
            raise PatternVisualEffectContractError(
                f"{item.pair_id}: unsupported split {item.split!r}"
            )
        previous = observed.setdefault(item.base_group_id, item.split)
        if previous != item.split:
            raise PatternVisualEffectContractError(
                f"base_group {item.base_group_id!r} crosses splits: "
                f"{previous!r} vs {item.split!r}"
            )
    return observed


def load_pattern_only_counterfactual_manifest(
    path: str | Path,
    *,
    split_assignments: Mapping[str, str] | None = None,
    eligible_only: bool = True,
) -> tuple[CounterfactualVisualExample, ...]:
    """Load exact 2D interventions without pretending that renders exist.

    ``split_assignments`` may address either ``base_group_id`` or ``pair_id``.
    The group-level integrity check still rejects pair-wise assignments that
    would leak one baseline across train and test.
    """

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not isinstance(payload.get("records"), list):
        raise PatternVisualEffectContractError("counterfactual manifest requires records[]")
    if payload.get("pattern_only") is not True:
        raise PatternVisualEffectContractError("loader accepts pattern-only manifests only")
    if int(payload.get("true_four_view_pair_count", 0)) != 0:
        raise PatternVisualEffectContractError(
            "pattern-only manifest cannot claim validated four-view pairs"
        )

    assignments = dict(split_assignments or {})
    output: list[CounterfactualVisualExample] = []
    for row_index, value in enumerate(payload["records"]):
        if not isinstance(value, Mapping):
            raise PatternVisualEffectContractError(f"records[{row_index}] must be an object")
        if eligible_only and not counterfactual_training_eligibility(value)["training_eligible"]:
            continue
        pair_id = _required_text(value.get("pair_id"), f"records[{row_index}].pair_id")
        if value.get("pattern_only") is not True:
            raise PatternVisualEffectContractError(f"{pair_id}: expected pattern_only=true")
        render_status = _required_text(value.get("render_status"), f"{pair_id}.render_status")
        if render_status in {"VALIDATED", "PASS_VALIDATED_FOUR_VIEW_RECEIPT"}:
            raise PatternVisualEffectContractError(
                f"{pair_id}: validated visual data belongs in the render-pair loader"
            )
        group = _base_group_id(value)
        split = str(
            value.get(
                "split",
                assignments.get(group, assignments.get(pair_id, "unassigned")),
            )
        )
        source_parameter = _required_text(
            value.get("intervention_parameter"), f"{pair_id}.intervention_parameter"
        )
        observable_axis = value.get("observable_axis", value.get("semantic_parameter"))
        if observable_axis is None:
            observable_axis = _SOURCE_PARAMETER_ALIASES.get(source_parameter)
        elif observable_axis not in TSHIRT_OBSERVABLE_AXIS_NAMES:
            raise PatternVisualEffectContractError(
                f"{pair_id}: unknown observable axis {observable_axis!r}"
            )
        output.append(
            CounterfactualVisualExample(
                pair_id=pair_id,
                base_group_id=group,
                split=split,
                source=_required_text(value.get("source"), f"{pair_id}.source"),
                source_parameter=source_parameter,
                observable_axis=observable_axis,
                baseline_value=_finite_number(value.get("baseline_value"), f"{pair_id}.baseline_value"),
                intervention_value=_finite_number(value.get("intervention_value"), f"{pair_id}.intervention_value"),
                baseline_pattern_path=_required_text(value.get("baseline_canonical_pattern"), f"{pair_id}.baseline pattern"),
                intervention_pattern_path=_required_text(value.get("intervention_canonical_pattern"), f"{pair_id}.intervention pattern"),
                baseline_pattern_sha256=_sha256(value.get("baseline_pattern_sha256"), f"{pair_id}.baseline hash"),
                intervention_pattern_sha256=_sha256(value.get("intervention_pattern_sha256"), f"{pair_id}.intervention hash"),
                unchanged_state_fingerprint=_sha256(value.get("unchanged_state_fingerprint"), f"{pair_id}.state fingerprint"),
                expected_elements=tuple(str(item) for item in value.get("expected_affected_elements", ())),
                semantic_delta=dict(value.get("ground_truth_semantic_delta", {})),
                render_status=render_status,
            )
        )
    if not output:
        raise PatternVisualEffectContractError("manifest contains no eligible pattern-only pairs")
    assert_base_group_split_integrity(output)
    return tuple(output)


def _resolve_inside(root: Path, relative: Any, label: str) -> Path:
    root = root.resolve()
    path = (root / _required_text(relative, f"{label}.path")).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise PatternVisualEffectContractError(f"{label}.path escapes receipt root") from error
    if not path.is_file():
        raise PatternVisualEffectContractError(f"missing {label}: {path}")
    return path


def _validate_file_descriptor(
    descriptor: Any,
    *,
    root: Path,
    label: str,
    seen_paths: set[Path],
    expected_size: tuple[int, int] | None = None,
    require_size: bool = False,
) -> dict[str, Any]:
    if not isinstance(descriptor, Mapping):
        raise PatternVisualEffectContractError(f"{label} must be a file descriptor")
    path = _resolve_inside(root, descriptor.get("path"), label)
    if path in seen_paths:
        raise PatternVisualEffectContractError(f"render receipt reuses file: {path}")
    seen_paths.add(path)
    expected_hash = _sha256(descriptor.get("sha256"), f"{label}.sha256")
    if file_sha256(path) != expected_hash:
        raise PatternVisualEffectContractError(f"{label} SHA-256 mismatch")
    size_value = descriptor.get("image_size")
    size: tuple[int, int] | None = None
    if size_value is not None:
        if (
            not isinstance(size_value, Sequence)
            or isinstance(size_value, (str, bytes))
            or len(size_value) != 2
        ):
            raise PatternVisualEffectContractError(f"{label}.image_size must contain two values")
        size = tuple(int(item) for item in size_value)
        if min(size) <= 0:
            raise PatternVisualEffectContractError(f"{label}.image_size must be positive")
    if require_size and size is None:
        raise PatternVisualEffectContractError(f"{label}.image_size is required")
    if expected_size is not None and size != expected_size:
        raise PatternVisualEffectContractError(f"{label}.image_size differs from RGBA view")
    output: dict[str, Any] = {"sha256": expected_hash}
    if size is not None:
        output["image_size"] = list(size)
    if "vertex_count" in descriptor:
        output["vertex_count"] = int(descriptor["vertex_count"])
    return output


def validate_effect_render_receipt(
    pair_record: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    root: str | Path,
) -> dict[str, Any]:
    """Validate all evidence required for a true 2D-change→3D-change pair.

    A normal four-view receipt is insufficient: the effect lane also requires
    simulator fidelity, fixed cameras, 2D/3D vertex correspondence, auxiliary
    render passes, screen-space flow, and an effect mask.  Any omitted field
    fails closed.
    """

    eligibility = counterfactual_training_eligibility(pair_record)
    if not eligibility["training_eligible"]:
        raise PatternVisualEffectContractError(
            "counterfactual pair is not pattern-training eligible: "
            + ",".join(eligibility["quarantine_reasons"])
        )
    if receipt.get("schema_version") != EFFECT_RECEIPT_SCHEMA_VERSION:
        raise PatternVisualEffectContractError("effect render receipt schema mismatch")
    resolved_root = Path(root)
    try:
        basic = validate_four_view_receipt(pair_record, receipt, root=resolved_root)
    except (CounterfactualContractError, KeyError, TypeError, ValueError) as error:
        raise PatternVisualEffectContractError(str(error)) from error

    fidelity = receipt.get("simulator_fidelity")
    if not isinstance(fidelity, Mapping) or fidelity.get("status") != "PASS":
        raise PatternVisualEffectContractError("simulator fidelity must explicitly PASS")
    profile_id = _required_text(fidelity.get("profile_id"), "simulator_fidelity.profile_id")
    version = _required_text(fidelity.get("version"), "simulator_fidelity.version")
    reference_hash = _sha256(
        fidelity.get("reference_receipt_sha256"),
        "simulator_fidelity.reference_receipt_sha256",
    )

    fixed = receipt.get("fixed_state")
    required_fixed = (
        "body_sha256",
        "material_sha256",
        "pose_sha256",
        "camera_rig_sha256",
        "simulator_sha256",
    )
    if not isinstance(fixed, Mapping):
        raise PatternVisualEffectContractError("fixed_state is required")
    fixed_hashes = {name: _sha256(fixed.get(name), f"fixed_state.{name}") for name in required_fixed}

    correspondence = receipt.get("correspondence")
    if (
        not isinstance(correspondence, Mapping)
        or correspondence.get("status") != "PASS"
        or correspondence.get("topology_stable") is not True
    ):
        raise PatternVisualEffectContractError(
            "2D/3D correspondence must PASS with stable topology"
        )
    vertex_count = int(correspondence.get("vertex_count", 0))
    if vertex_count <= 0:
        raise PatternVisualEffectContractError("correspondence.vertex_count must be positive")

    seen_paths: set[Path] = set()
    correspondence_files = {
        name: _validate_file_descriptor(
            correspondence.get(name),
            root=resolved_root,
            label=f"correspondence.{name}",
            seen_paths=seen_paths,
        )
        for name in ("panel_uv", "vertex_map")
    }

    camera_hashes: dict[str, tuple[str, str]] = {}
    member_summary: dict[str, Any] = {}
    for member in ("baseline", "intervention"):
        source_member = receipt.get("members", {}).get(member)
        if not isinstance(source_member, Mapping):
            raise PatternVisualEffectContractError(f"missing receipt member {member}")
        mesh = _validate_file_descriptor(
            source_member.get("mesh"),
            root=resolved_root,
            label=f"{member}.mesh",
            seen_paths=seen_paths,
        )
        if mesh.get("vertex_count") != vertex_count:
            raise PatternVisualEffectContractError(
                f"{member}.mesh vertex_count differs from correspondence"
            )
        views_summary: dict[str, Any] = {}
        for view in ("front", "back", "left", "right"):
            item = source_member.get("views", {}).get(view)
            if not isinstance(item, Mapping):
                raise PatternVisualEffectContractError(f"missing {member}/{view} view")
            size = tuple(int(value) for value in item.get("image_size", ()))
            primary = _validate_file_descriptor(
                item,
                root=resolved_root,
                label=f"{member}.{view}.rgba",
                seen_paths=seen_paths,
                require_size=True,
            )
            intrinsics = _sha256(
                item.get("camera_intrinsics_sha256"),
                f"{member}.{view}.camera_intrinsics_sha256",
            )
            extrinsics = _sha256(
                item.get("camera_extrinsics_sha256"),
                f"{member}.{view}.camera_extrinsics_sha256",
            )
            observed_camera = (intrinsics, extrinsics)
            if view in camera_hashes and camera_hashes[view] != observed_camera:
                raise PatternVisualEffectContractError(
                    f"baseline/intervention camera differs for {view}"
                )
            camera_hashes[view] = observed_camera
            passes = item.get("passes")
            if not isinstance(passes, Mapping):
                raise PatternVisualEffectContractError(f"{member}.{view}.passes is required")
            pass_summary = {
                name: _validate_file_descriptor(
                    passes.get(name),
                    root=resolved_root,
                    label=f"{member}.{view}.{name}",
                    seen_paths=seen_paths,
                    expected_size=size,
                    require_size=True,
                )
                for name in ("silhouette", "depth", "normal", "panel_id")
            }
            views_summary[view] = {
                "rgba": primary,
                "passes": pass_summary,
                "camera_intrinsics_sha256": intrinsics,
                "camera_extrinsics_sha256": extrinsics,
            }
        member_summary[member] = {"mesh": mesh, "views": views_summary}

    effects = receipt.get("effects")
    if not isinstance(effects, Mapping) or set(effects) != {"front", "back", "left", "right"}:
        raise PatternVisualEffectContractError("effects must contain exactly four views")
    effect_summary: dict[str, Any] = {}
    for view in ("front", "back", "left", "right"):
        item = effects[view]
        expected_size = tuple(member_summary["baseline"]["views"][view]["rgba"]["image_size"])
        effect_summary[view] = {
            name: _validate_file_descriptor(
                item.get(name) if isinstance(item, Mapping) else None,
                root=resolved_root,
                label=f"effects.{view}.{name}",
                seen_paths=seen_paths,
                expected_size=expected_size,
                require_size=True,
            )
            for name in ("flow", "effect_mask")
        }

    return {
        "schema_version": EFFECT_RECEIPT_SCHEMA_VERSION,
        "status": "PASS_VALIDATED_CAUSAL_EFFECT_RENDER",
        "pair_id": basic["pair_id"],
        "pattern_only": False,
        "target_dsl_used_for_inverse": False,
        "unchanged_state_fingerprint": basic["unchanged_state_fingerprint"],
        "simulator_fidelity": {
            "status": "PASS",
            "profile_id": profile_id,
            "version": version,
            "reference_receipt_sha256": reference_hash,
        },
        "fixed_state": fixed_hashes,
        "correspondence": {
            "status": "PASS",
            "topology_stable": True,
            "vertex_count": vertex_count,
            **correspondence_files,
        },
        "members": member_summary,
        "effects": effect_summary,
    }


INVERSE_REQUIRED_INPUT_KEYS = frozenset(
    {"target_views", "anchor_elements", "anchor_observables", "element_valid"}
)
INVERSE_OPTIONAL_INPUT_KEYS = frozenset(
    {"anchor_views", "observable_mask", "conditions", "view_valid"}
)
INVERSE_ALLOWED_INPUT_KEYS = INVERSE_REQUIRED_INPUT_KEYS | INVERSE_OPTIONAL_INPUT_KEYS


def validate_inverse_input_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Reject target Pattern DSL/geometry from the inverse inference boundary."""

    if not isinstance(payload, Mapping):
        raise PatternVisualEffectContractError("inverse input must be a mapping")
    keys = set(payload)
    missing = sorted(INVERSE_REQUIRED_INPUT_KEYS - keys)
    unknown = sorted(keys - INVERSE_ALLOWED_INPUT_KEYS)
    if missing:
        raise PatternVisualEffectContractError(
            "inverse input is missing: " + ", ".join(missing)
        )
    if unknown:
        raise PatternVisualEffectContractError(
            "inverse input contains forbidden/unknown keys: " + ", ".join(unknown)
        )
    return {
        "schema_version": INVERSE_INPUT_CONTRACT_VERSION,
        "status": "PASS_TARGET_DSL_ABSENT",
        "modality": "TARGET_FOUR_VIEW_PLUS_ANCHOR_PATTERN",
        "target_dsl_used": False,
        "accepted_keys": sorted(keys),
    }


def build_pattern_visual_effect_bridge(config: Mapping[str, Any] | None = None):
    """Build a forward exact-pattern delta→visual-effect/observable bridge.

    Torch is imported lazily so dataset/receipt tooling works in lightweight
    environments that do not have the training runtime installed.
    """

    import torch
    from torch import nn

    settings = dict(config or {})
    if "parameter_count" in settings:
        raise ValueError("parameter_count is ambiguous; use the fixed observable schema")
    view_dim = int(settings.get("view_dim", 256))
    element_dim = int(settings.get("element_dim", 128))
    hidden = int(settings.get("hidden_dim", 128))
    heads = int(settings.get("heads", 4))
    layers = int(settings.get("layers", 2))
    dropout = float(settings.get("dropout", 0.1))
    max_tokens = int(settings.get("max_spatial_tokens", 85))
    query_count = int(settings.get("element_count", len(TSHIRT_ELEMENT_QUERIES)))
    observable_count = int(settings.get("observable_count", len(TSHIRT_OBSERVABLE_AXES)))
    condition_dim = int(settings.get("condition_dim", 0))
    if hidden <= 0 or heads <= 0 or hidden % heads:
        raise ValueError("hidden_dim must be positive and divisible by heads")
    if query_count != len(TSHIRT_ELEMENT_QUERIES):
        raise ValueError("element_count must match the fixed 15-query element schema")
    if observable_count != len(TSHIRT_OBSERVABLE_AXES):
        raise ValueError("observable_count must match the fixed eight-axis schema")

    class PatternVisualEffectBridge(nn.Module):
        output_schema_version = TSHIRT_OBSERVABLE_SCHEMA_VERSION

        def __init__(self) -> None:
            super().__init__()
            self.view_project = nn.Linear(view_dim, hidden)
            self.element_project = nn.Linear(element_dim, hidden)
            self.view_embedding = nn.Parameter(torch.zeros(1, 4, 1, hidden))
            self.spatial_embedding = nn.Parameter(torch.zeros(1, 1, max_tokens, hidden))
            self.element_embedding = nn.Parameter(torch.zeros(1, query_count, hidden))
            layer = nn.TransformerEncoderLayer(
                hidden, heads, hidden * 4, dropout, "gelu", batch_first=True, norm_first=True
            )
            self.view_encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            element_layer = nn.TransformerEncoderLayer(
                hidden, heads, hidden * 3, dropout, "gelu", batch_first=True, norm_first=True
            )
            self.element_encoder = nn.TransformerEncoder(
                element_layer, max(1, layers - 1), enable_nested_tensor=False
            )
            self.cross_attention = nn.MultiheadAttention(
                hidden, heads, dropout=dropout, batch_first=True
            )
            self.map_query = nn.Linear(hidden, hidden, bias=False)
            self.map_key = nn.Linear(hidden, hidden, bias=False)
            self.effect_value = nn.Linear(hidden, hidden)
            self.visual_delta_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, view_dim))
            self.affected_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
            self.intervention_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, observable_count))
            self.delta_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, observable_count))
            self.log_variance_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, observable_count))
            self.condition = nn.Linear(condition_dim, hidden) if condition_dim else None
            for value in (self.view_embedding, self.spatial_embedding, self.element_embedding):
                nn.init.normal_(value, std=0.02)

        def forward(
            self,
            baseline_views,
            baseline_elements,
            intervention_elements,
            element_valid,
            *,
            view_valid=None,
            conditions=None,
            capture_attention: bool = False,
        ):
            if baseline_views.ndim != 4 or baseline_views.shape[1] != 4 or baseline_views.shape[-1] != view_dim:
                raise ValueError("baseline_views must have shape [B,4,T,view_dim]")
            batch, views, tokens, _ = baseline_views.shape
            if tokens > max_tokens:
                raise ValueError("too many spatial tokens")
            expected_elements = (batch, query_count, element_dim)
            if tuple(baseline_elements.shape) != expected_elements or tuple(intervention_elements.shape) != expected_elements:
                raise ValueError(f"element tensors must have shape {expected_elements}")
            if tuple(element_valid.shape) != (batch, query_count) or not bool(element_valid.any(-1).all()):
                raise ValueError("element_valid must keep at least one query per sample")
            if view_valid is None:
                view_valid = torch.ones((batch, views), dtype=torch.bool, device=baseline_views.device)
            if tuple(view_valid.shape) != (batch, views) or not bool(view_valid.any(-1).all()):
                raise ValueError("view_valid must keep at least one view per sample")
            memory = self.view_project(baseline_views)
            memory = memory + self.view_embedding + self.spatial_embedding[:, :, :tokens]
            memory = memory.reshape(batch, views * tokens, hidden)
            memory_padding = ~view_valid[:, :, None].expand(-1, -1, tokens).reshape(batch, -1)
            memory = self.view_encoder(memory, src_key_padding_mask=memory_padding)

            delta = self.element_project(intervention_elements - baseline_elements)
            delta = delta + self.element_embedding
            if self.condition is not None:
                if conditions is None or tuple(conditions.shape) != (batch, condition_dim):
                    raise ValueError(f"conditions must have shape {(batch, condition_dim)}")
                delta = delta + self.condition(conditions)[:, None]
            elif conditions is not None:
                raise ValueError("conditions supplied but condition_dim is zero")
            delta = self.element_encoder(delta, src_key_padding_mask=~element_valid.bool())
            attended, attention = self.cross_attention(
                delta,
                memory,
                memory,
                key_padding_mask=memory_padding,
                need_weights=capture_attention,
                average_attn_weights=False,
            )
            fused = delta + attended
            map_logits = torch.einsum(
                "bqh,bkh->bqk", self.map_query(fused), self.map_key(memory)
            ) / math.sqrt(hidden)
            map_logits = map_logits.reshape(batch, query_count, views, tokens)
            map_logits = map_logits.masked_fill(~element_valid[:, :, None, None], -1.0e4)
            map_logits = map_logits.masked_fill(~view_valid[:, None, :, None], -1.0e4)
            contribution = self.effect_value(fused)
            weights = torch.softmax(map_logits, dim=1)
            visual_hidden = torch.einsum("bqvt,bqh->bvth", weights, contribution)
            valid_float = element_valid[..., None].to(fused.dtype)
            pooled = (fused * valid_float).sum(1) / valid_float.sum(1).clamp_min(1)
            output = {
                "affected_element_logits": self.affected_head(fused).squeeze(-1),
                "effect_map_logits": map_logits,
                "visual_delta_prediction": self.visual_delta_head(visual_hidden),
                "intervention_logits": self.intervention_head(pooled),
                "observable_delta": self.delta_head(pooled),
                "observable_log_variance": self.log_variance_head(pooled).clamp(-8.0, 8.0),
            }
            if capture_attention:
                output["cross_attention"] = attention.reshape(
                    batch, heads, query_count, views, tokens
                )
            return output

    return PatternVisualEffectBridge()


def build_pattern_inverse_residual(config: Mapping[str, Any] | None = None):
    """Build a target-four-view + anchor-elements observable residual model."""

    import torch
    from torch import nn

    settings = dict(config or {})
    if "parameter_count" in settings:
        raise ValueError("parameter_count is ambiguous; use the fixed observable schema")
    view_dim = int(settings.get("view_dim", 256))
    element_dim = int(settings.get("element_dim", 128))
    hidden = int(settings.get("hidden_dim", 128))
    heads = int(settings.get("heads", 4))
    layers = int(settings.get("layers", 2))
    dropout = float(settings.get("dropout", 0.1))
    max_tokens = int(settings.get("max_spatial_tokens", 85))
    query_count = int(settings.get("element_count", len(TSHIRT_ELEMENT_QUERIES)))
    observable_count = int(settings.get("observable_count", len(TSHIRT_OBSERVABLE_AXES)))
    condition_dim = int(settings.get("condition_dim", 0))
    if hidden <= 0 or heads <= 0 or hidden % heads:
        raise ValueError("hidden_dim must be positive and divisible by heads")
    if query_count != len(TSHIRT_ELEMENT_QUERIES):
        raise ValueError("element_count must match the fixed 15-query element schema")
    if observable_count != len(TSHIRT_OBSERVABLE_AXES):
        raise ValueError("observable_count must match the fixed eight-axis schema")

    class PatternInverseResidual(nn.Module):
        output_schema_version = TSHIRT_OBSERVABLE_SCHEMA_VERSION

        def __init__(self) -> None:
            super().__init__()
            self.target_view_project = nn.Linear(view_dim, hidden)
            self.anchor_view_delta_project = nn.Linear(view_dim, hidden)
            self.element_project = nn.Linear(element_dim, hidden)
            self.observable_project = nn.Linear(observable_count, hidden)
            self.view_embedding = nn.Parameter(torch.zeros(1, 4, 1, hidden))
            self.spatial_embedding = nn.Parameter(torch.zeros(1, 1, max_tokens, hidden))
            self.element_embedding = nn.Parameter(torch.zeros(1, query_count, hidden))
            layer = nn.TransformerEncoderLayer(
                hidden, heads, hidden * 4, dropout, "gelu", batch_first=True, norm_first=True
            )
            self.view_encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.element_encoder = nn.TransformerEncoder(layer, max(1, layers - 1), enable_nested_tensor=False)
            self.cross_attention = nn.MultiheadAttention(
                hidden, heads, dropout=dropout, batch_first=True
            )
            self.residual_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, observable_count))
            self.log_variance_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, observable_count))
            self.affected_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
            self.abstention_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
            self.condition = nn.Linear(condition_dim, hidden) if condition_dim else None
            for value in (self.view_embedding, self.spatial_embedding, self.element_embedding):
                nn.init.normal_(value, std=0.02)

        def forward(
            self,
            target_views,
            anchor_elements,
            anchor_observables,
            element_valid,
            *,
            anchor_views=None,
            observable_mask=None,
            conditions=None,
            view_valid=None,
            capture_attention: bool = False,
        ):
            if target_views.ndim != 4 or target_views.shape[1] != 4 or target_views.shape[-1] != view_dim:
                raise ValueError("target_views must have shape [B,4,T,view_dim]")
            batch, views, tokens, _ = target_views.shape
            if tokens > max_tokens:
                raise ValueError("too many spatial tokens")
            if tuple(anchor_elements.shape) != (batch, query_count, element_dim):
                raise ValueError("anchor_elements shape mismatch")
            if tuple(anchor_observables.shape) != (batch, observable_count):
                raise ValueError("anchor_observables shape mismatch")
            if tuple(element_valid.shape) != (batch, query_count) or not bool(element_valid.any(-1).all()):
                raise ValueError("element_valid must keep at least one query per sample")
            if view_valid is None:
                view_valid = torch.ones((batch, views), dtype=torch.bool, device=target_views.device)
            if tuple(view_valid.shape) != (batch, views) or not bool(view_valid.any(-1).all()):
                raise ValueError("view_valid must keep at least one view per sample")
            memory = self.target_view_project(target_views)
            if anchor_views is not None:
                if tuple(anchor_views.shape) != tuple(target_views.shape):
                    raise ValueError("anchor_views must match target_views")
                memory = memory + self.anchor_view_delta_project(target_views - anchor_views)
            memory = memory + self.view_embedding + self.spatial_embedding[:, :, :tokens]
            memory = memory.reshape(batch, views * tokens, hidden)
            memory_padding = ~view_valid[:, :, None].expand(-1, -1, tokens).reshape(batch, -1)
            memory = self.view_encoder(memory, src_key_padding_mask=memory_padding)

            queries = self.element_project(anchor_elements) + self.element_embedding
            if self.condition is not None:
                if conditions is None or tuple(conditions.shape) != (batch, condition_dim):
                    raise ValueError(f"conditions must have shape {(batch, condition_dim)}")
                queries = queries + self.condition(conditions)[:, None]
            elif conditions is not None:
                raise ValueError("conditions supplied but condition_dim is zero")
            queries = self.element_encoder(queries, src_key_padding_mask=~element_valid.bool())
            attended, attention = self.cross_attention(
                queries,
                memory,
                memory,
                key_padding_mask=memory_padding,
                need_weights=capture_attention,
                average_attn_weights=False,
            )
            fused = queries + attended
            valid_float = element_valid[..., None].to(fused.dtype)
            pooled = (fused * valid_float).sum(1) / valid_float.sum(1).clamp_min(1)
            pooled = pooled + self.observable_project(anchor_observables)
            raw_residual = self.residual_head(pooled)
            if observable_mask is None:
                observable_mask = torch.ones_like(raw_residual, dtype=torch.bool)
            if tuple(observable_mask.shape) != tuple(raw_residual.shape):
                raise ValueError("observable_mask shape mismatch")
            residual = torch.where(observable_mask.bool(), raw_residual, torch.zeros_like(raw_residual))
            output = {
                "raw_observable_delta": raw_residual,
                "observable_delta": residual,
                "observable_log_variance": self.log_variance_head(pooled).clamp(-8.0, 8.0),
                "affected_element_logits": self.affected_head(fused).squeeze(-1),
                "abstention_logit": self.abstention_head(pooled).squeeze(-1),
            }
            if capture_attention:
                output["cross_attention"] = attention.reshape(
                    batch, heads, query_count, views, tokens
                )
            return output

        def forward_payload(self, payload: Mapping[str, Any], *, capture_attention: bool = False):
            validate_inverse_input_contract(payload)
            return self(**payload, capture_attention=capture_attention)

    return PatternInverseResidual()


@dataclass(frozen=True)
class TShirtResidualDecodeResult:
    parameters: Any
    graph: Any
    receipt: Mapping[str, Any]


def decode_tshirt_semantic_residual(
    anchor_parameters: Any,
    residual: Mapping[str, Any] | Sequence[Any],
    *,
    parameter_mask: Mapping[str, bool] | Sequence[bool] | None = None,
    pattern_id: str = "pattern_visual_inverse_tshirt",
) -> TShirtResidualDecodeResult:
    """Project an 8-D semantic residual through the existing constrained decoder."""

    from benchmark.drafting_semantics.tshirt_parametric_decoder import (
        TShirtDraftParameters,
        decode_tshirt_pattern,
    )

    if isinstance(anchor_parameters, TShirtDraftParameters):
        anchor = anchor_parameters
    elif isinstance(anchor_parameters, Mapping):
        anchor = TShirtDraftParameters.from_mapping(anchor_parameters)
    else:
        raise PatternVisualEffectContractError(
            "anchor_parameters must be TShirtDraftParameters or a mapping"
        )
    if isinstance(residual, Mapping):
        unknown = sorted(set(residual) - set(TSHIRT_SEMANTIC_PARAMETER_NAMES))
        if unknown:
            raise PatternVisualEffectContractError(
                "unknown semantic residuals: " + ", ".join(unknown)
            )
        values = {name: _finite_number(value, f"residual.{name}") for name, value in residual.items()}
    else:
        if isinstance(residual, (str, bytes)) or len(residual) != len(TSHIRT_SEMANTIC_PARAMETERS):
            raise PatternVisualEffectContractError(
                f"semantic residual vector must contain {len(TSHIRT_SEMANTIC_PARAMETERS)} values"
            )
        values = {
            name: _finite_number(value, f"residual.{name}")
            for name, value in zip(TSHIRT_SEMANTIC_PARAMETER_NAMES, residual)
        }
    if parameter_mask is None:
        mask = {name: True for name in values}
    elif isinstance(parameter_mask, Mapping):
        unknown = sorted(set(parameter_mask) - set(TSHIRT_SEMANTIC_PARAMETER_NAMES))
        if unknown:
            raise PatternVisualEffectContractError(
                "unknown semantic parameter masks: " + ", ".join(unknown)
            )
        mask = {name: bool(parameter_mask.get(name, False)) for name in values}
    else:
        if isinstance(parameter_mask, (str, bytes)) or len(parameter_mask) != len(TSHIRT_SEMANTIC_PARAMETERS):
            raise PatternVisualEffectContractError(
                f"parameter mask must contain {len(TSHIRT_SEMANTIC_PARAMETERS)} values"
            )
        all_mask = dict(zip(TSHIRT_SEMANTIC_PARAMETER_NAMES, (bool(value) for value in parameter_mask)))
        mask = {name: all_mask[name] for name in values}
    applied = {name: value for name, value in values.items() if mask[name]}
    resolved = anchor.with_residual(applied, project=True)
    graph = decode_tshirt_pattern(resolved, pattern_id=_required_text(pattern_id, "pattern_id"))
    graph.validate()
    projected = []
    anchor_values = anchor.to_dict()
    resolved_values = resolved.to_dict()
    for name, value in applied.items():
        if not math.isclose(anchor_values[name] + value, resolved_values[name], rel_tol=0.0, abs_tol=1e-9):
            projected.append(name)
    return TShirtResidualDecodeResult(
        parameters=resolved,
        graph=graph,
        receipt={
            "schema_version": "pattern-visual-tshirt-residual-decode/v1",
            "status": "PASS_CONSTRAINED_TSHIRT_DECODE",
            "input_contract": {
                "target_dsl_used": False,
                "anchor_pattern_parameters_used": True,
                "semantic_parameter_order": list(TSHIRT_SEMANTIC_PARAMETER_NAMES),
            },
            "applied_residual_cm": applied,
            "masked_parameter_names": sorted(name for name in values if not mask[name]),
            "box_projected_parameter_names": projected,
            "constraint": {
                "name": graph.sleeve_head_constraint.id,
                "converged": graph.sleeve_head_constraint.converged,
                "residual_cm": graph.sleeve_head_constraint.residual_cm,
                "tolerance_cm": graph.sleeve_head_constraint.tolerance_cm,
            },
        },
    )


def decode_tshirt_observable_residual(
    anchor_parameters: Any,
    observable_residual: Mapping[str, Any] | Sequence[Any],
    *,
    calibrations: Mapping[str, ObservableAxisCalibration | Mapping[str, Any]],
    observable_valid: Mapping[str, bool] | Sequence[bool] | None = None,
    pattern_id: str = "pattern_visual_observable_inverse_tshirt",
) -> TShirtResidualDecodeResult:
    """Calibrate observable deltas, then run the constrained T-shirt decoder.

    The learned inverse head predicts the eight completed-pattern observables.
    This helper is the explicit boundary where those values become native
    decoder residuals.  Missing calibration evidence fails before drafting.
    """

    from benchmark.drafting_semantics.tshirt_parametric_decoder import (
        TShirtDraftParameters,
        decode_tshirt_pattern,
    )

    if isinstance(anchor_parameters, TShirtDraftParameters):
        anchor = anchor_parameters
    elif isinstance(anchor_parameters, Mapping):
        anchor = TShirtDraftParameters.from_mapping(anchor_parameters)
    else:
        raise PatternVisualEffectContractError(
            "anchor_parameters must be TShirtDraftParameters or a mapping"
        )
    adapter = adapt_observable_residuals_to_decoder(
        observable_residual,
        calibrations=calibrations,
        observable_valid=observable_valid,
        require_complete=True,
    )
    resolved = anchor.with_residual(adapter.decoder_residuals_cm, project=True)
    graph = decode_tshirt_pattern(resolved, pattern_id=_required_text(pattern_id, "pattern_id"))
    graph.validate()
    projected = []
    anchor_values = anchor.to_dict()
    resolved_values = resolved.to_dict()
    for name, value in adapter.decoder_residuals_cm.items():
        if not math.isclose(
            anchor_values[name] + value,
            resolved_values[name],
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            projected.append(name)
    return TShirtResidualDecodeResult(
        parameters=resolved,
        graph=graph,
        receipt={
            "schema_version": "pattern-visual-tshirt-observable-decode/v1",
            "status": "PASS_CALIBRATED_CONSTRAINED_TSHIRT_DECODE",
            "input_contract": {
                "target_dsl_used": False,
                "anchor_pattern_parameters_used": True,
                "model_output_schema": TSHIRT_OBSERVABLE_SCHEMA_VERSION,
                "decoder_input_schema": TSHIRT_DECODER_RESIDUAL_SCHEMA_VERSION,
            },
            "adapter": adapter.receipt,
            "applied_decoder_residual_cm": dict(adapter.decoder_residuals_cm),
            "box_projected_parameter_names": projected,
            "constraint": {
                "name": graph.sleeve_head_constraint.id,
                "converged": graph.sleeve_head_constraint.converged,
                "residual_cm": graph.sleeve_head_constraint.residual_cm,
                "tolerance_cm": graph.sleeve_head_constraint.tolerance_cm,
            },
        },
    )


__all__ = [
    "GCDV2_SURFACE_SCHEMA_VERSION",
    "CounterfactualVisualExample",
    "EFFECT_RECEIPT_SCHEMA_VERSION",
    "ElementQuery",
    "INVERSE_ALLOWED_INPUT_KEYS",
    "INVERSE_INPUT_CONTRACT_VERSION",
    "ObservableAxis",
    "ObservableAxisCalibration",
    "ObservableDecoderRelation",
    "ObservableResidualAdapterResult",
    "PatternVisualEffectContractError",
    "SemanticParameter",
    "TSHIRT_ADAPTER_DECODER_PARAMETER_NAMES",
    "TSHIRT_DECODER_RESIDUAL_PARAMETERS",
    "TSHIRT_DECODER_RESIDUAL_PARAMETER_NAMES",
    "TSHIRT_ELEMENT_QUERIES",
    "TSHIRT_ELEMENT_QUERY_NAMES",
    "TSHIRT_OBSERVABLE_ADAPTER_SCHEMA_VERSION",
    "TSHIRT_OBSERVABLE_AXES",
    "TSHIRT_OBSERVABLE_AXIS_NAMES",
    "TSHIRT_OBSERVABLE_SCHEMA_VERSION",
    "TSHIRT_OBSERVABLE_TO_DECODER_RELATIONS",
    "TSHIRT_DECODER_RESIDUAL_SCHEMA_VERSION",
    "TSHIRT_SEMANTIC_PARAMETERS",
    "TSHIRT_SEMANTIC_PARAMETER_NAMES",
    "TShirtResidualDecodeResult",
    "adapt_observable_residuals_to_decoder",
    "assert_base_group_split_integrity",
    "build_pattern_inverse_residual",
    "build_pattern_visual_effect_bridge",
    "decode_tshirt_semantic_residual",
    "decode_tshirt_observable_residual",
    "load_pattern_only_counterfactual_manifest",
    "validate_effect_render_receipt",
    "validate_inverse_input_contract",
]
