"""Fail-closed four-view semantic inference into provisional basic patterns.

This module is the inference-only bridge for the common T-shirt, trousers,
and skirt pilot.  It accepts frozen *visual feature tensors*, never a pattern
graph, and converts the student's fixed semantic query table into bounded
residual edits on a deterministic ``PROVISIONAL_EXPERT_REVIEW`` BasicBlock.

The privacy contract is deliberately narrow.  Receipts contain tensor hashes,
shapes, numeric predictions, model hashes, and validation findings.  They do
not contain checkpoint paths, feature archive paths, source image paths, or
image pixels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from benchmark.drafting_semantics.basic_blocks import PROVENANCE_STATUS, build_basic_block
from benchmark.drafting_semantics.basic_semantic_targets import (
    semantic_target_from_basic_block,
    semantic_target_from_pattern_document,
)
from benchmark.drafting_semantics.semantic_teacher_student import (
    CATEGORY_NAMES,
    MAX_COORDINATE_DIM,
    SEMANTIC_QUERY_INVENTORY,
    SEMANTIC_QUERY_KEYS,
    ModalityContractError,
    build_four_view_semantic_student,
    category_query_mask,
    infer_four_view_semantics,
    query_coordinate_mask,
)
from benchmark.pattern_pipeline.schema import PatternDocument
from benchmark.pattern_pipeline.semantic_editing import (
    LandmarkResidual,
    PathResidual,
    SemanticResidualPlan,
    apply_semantic_residual,
)
from benchmark.pattern_pipeline.semantic_residual_planning import (
    ResidualPlanningConfig,
    build_semantic_residual_plan,
)
from benchmark.pattern_pipeline.validation import ValidationReport, validate_pattern


CANONICAL_VIEW_ORDER = ("front", "back", "left", "right")
CHECKPOINT_CALIBRATION_FIELD = "coordinate_confidence_calibration"
CALIBRATION_SCHEMA_VERSION = "semantic-coordinate-confidence/v1"
CHECKPOINT_EDIT_CALIBRATION_FIELD = "semantic_edit_calibration"
EDIT_CALIBRATION_SCHEMA_VERSION = "semantic-edit-validation-gate/v1"
INFERENCE_SCHEMA_VERSION = "four-view-basic-pattern-inference/v3-validation-edit-guard"
SEMANTIC_PROJECTION_SCALES = (1.0, 0.75, 0.5, 0.25)
_FORBIDDEN_FEATURE_FIELDS = {
    "pattern",
    "pattern_graph",
    "pattern_features",
    "edge_features",
    "panel_features",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_view_order(values: Sequence[Any]) -> tuple[str, ...]:
    output = []
    for value in values:
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        output.append(str(value).strip().lower())
    return tuple(output)


@dataclass(frozen=True)
class FourViewFeatureBundle:
    """One sample's explicitly ordered visual features.

    ``global_features`` is ``[4, channels]`` and ``spatial_features`` is
    ``[4, patches, channels]``.  The view order is data, not an implicit
    convention: a bundle without the exact canonical order is rejected.
    """

    view_order: tuple[str, ...]
    global_features: np.ndarray | None = None
    spatial_features: np.ndarray | None = None

    def __post_init__(self) -> None:
        order = _normalized_view_order(self.view_order)
        object.__setattr__(self, "view_order", order)
        if order != CANONICAL_VIEW_ORDER:
            raise ValueError(
                "four-view features must be explicitly ordered "
                f"{CANONICAL_VIEW_ORDER}, received {order}"
            )
        if self.global_features is None and self.spatial_features is None:
            raise ValueError("at least one precomputed visual feature stream is required")
        if self.global_features is not None:
            values = np.ascontiguousarray(self.global_features, dtype=np.float32)
            if values.ndim != 2 or values.shape[0] != 4 or values.shape[1] <= 0:
                raise ValueError("global_features must have shape [4, channels]")
            if not np.isfinite(values).all():
                raise ValueError("global_features must contain only finite values")
            object.__setattr__(self, "global_features", values)
        if self.spatial_features is not None:
            values = np.ascontiguousarray(self.spatial_features, dtype=np.float32)
            if (
                values.ndim != 3
                or values.shape[0] != 4
                or values.shape[1] <= 0
                or values.shape[2] <= 0
            ):
                raise ValueError("spatial_features must have shape [4, patches, channels]")
            if not np.isfinite(values).all():
                raise ValueError("spatial_features must contain only finite values")
            object.__setattr__(self, "spatial_features", values)

    @property
    def tensor_sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update("|".join(self.view_order).encode("utf-8"))
        for name, values in (
            ("global", self.global_features),
            ("spatial", self.spatial_features),
        ):
            if values is None:
                continue
            digest.update(name.encode("ascii"))
            digest.update(str(values.shape).encode("ascii"))
            digest.update(str(values.dtype).encode("ascii"))
            digest.update(np.ascontiguousarray(values).tobytes())
        return digest.hexdigest()

    def shape_receipt(self) -> dict[str, list[int] | None]:
        return {
            "global": list(self.global_features.shape) if self.global_features is not None else None,
            "spatial": list(self.spatial_features.shape) if self.spatial_features is not None else None,
        }


def _archive_sample_index(
    archive: Any,
    batch_size: int,
    sample_id: str | None,
) -> int:
    if batch_size == 1 and sample_id is None:
        return 0
    if sample_id is None:
        raise ValueError("a batched feature archive requires an explicit sample_id")
    if "sample_ids" not in archive.files:
        raise ValueError("a batched feature archive requires a sample_ids array")
    identifiers = _normalized_view_order(np.asarray(archive["sample_ids"]).reshape(-1))
    # Sample IDs are identifiers rather than view names; normalize only
    # whitespace/case consistently for lookup and never copy them to receipts.
    requested = str(sample_id).strip().lower()
    matches = [index for index, value in enumerate(identifiers) if value == requested]
    if len(matches) != 1:
        raise ValueError("sample_id must select exactly one row in the feature archive")
    if len(identifiers) != batch_size:
        raise ValueError("sample_ids length does not match feature batch size")
    return matches[0]


def load_precomputed_four_view_features(
    path: str | Path,
    *,
    sample_id: str | None = None,
    generic_feature_kind: str | None = None,
    declared_view_order: Sequence[str] | None = None,
) -> FourViewFeatureBundle:
    """Load one strict four-view sample from a local ``.npz`` archive.

    Preferred archives contain ``global_features`` and/or
    ``spatial_features`` plus ``view_names``.  Existing extractor archives
    with a generic ``features`` array are supported only when the caller
    declares whether it is ``global`` or ``spatial``.  If ``view_names`` is
    absent, the caller must explicitly attest the view order.
    """

    resolved = Path(path)
    with np.load(resolved, allow_pickle=False) as archive:
        fields = set(archive.files)
        forbidden = fields & _FORBIDDEN_FEATURE_FIELDS
        if forbidden:
            raise ModalityContractError(
                "visual feature archive contains forbidden pattern fields: "
                + ", ".join(sorted(forbidden))
            )
        if "features" in fields:
            if generic_feature_kind not in {"global", "spatial"}:
                raise ValueError(
                    "generic `features` archives require generic_feature_kind="
                    "'global' or 'spatial'"
                )
            if {"global_features", "spatial_features"} & fields:
                raise ValueError("generic and named feature arrays cannot be mixed")
            named = {f"{generic_feature_kind}_features": np.asarray(archive["features"])}
        else:
            named = {
                key: np.asarray(archive[key])
                for key in ("global_features", "spatial_features")
                if key in fields
            }
        if not named:
            raise ValueError("feature archive contains no recognized visual tensor")

        batch_sizes: list[int] = []
        for key, values in named.items():
            unbatched_ndim = 2 if key == "global_features" else 3
            if values.ndim == unbatched_ndim:
                if values.shape[0] != 4:
                    raise ValueError(f"{key} must contain exactly four ordered views")
            elif values.ndim == unbatched_ndim + 1:
                if values.shape[1] != 4:
                    raise ValueError(f"batched {key} must have four views on axis 1")
                batch_sizes.append(int(values.shape[0]))
            else:
                raise ValueError(f"{key} has an unsupported rank {values.ndim}")
        if batch_sizes and any(value != batch_sizes[0] for value in batch_sizes):
            raise ValueError("visual feature streams have different batch sizes")
        if batch_sizes and len(batch_sizes) != len(named):
            raise ValueError("cannot mix batched and unbatched feature streams")
        row = _archive_sample_index(archive, batch_sizes[0], sample_id) if batch_sizes else None
        selected = {
            key: values[row] if row is not None else values
            for key, values in named.items()
        }

        embedded_order: tuple[str, ...] | None = None
        order_key = "view_names" if "view_names" in fields else (
            "view_order" if "view_order" in fields else None
        )
        if order_key is not None:
            raw_order = np.asarray(archive[order_key])
            if raw_order.ndim == 2:
                if row is None:
                    if raw_order.shape[0] != 1:
                        raise ValueError("batched view order requires batched features")
                    raw_order = raw_order[0]
                else:
                    if raw_order.shape[0] != batch_sizes[0]:
                        raise ValueError("view order batch does not match feature batch")
                    raw_order = raw_order[row]
            if raw_order.ndim != 1 or len(raw_order) != 4:
                raise ValueError("view order must contain exactly four names")
            embedded_order = _normalized_view_order(raw_order)

        declared = (
            _normalized_view_order(declared_view_order)
            if declared_view_order is not None
            else None
        )
        if embedded_order is None and declared is None:
            raise ValueError(
                "feature archive does not embed view names; declare exact order "
                "front back left right"
            )
        if embedded_order is not None and declared is not None and embedded_order != declared:
            raise ValueError("embedded and declared view orders disagree")
        order = embedded_order or declared
        assert order is not None
        return FourViewFeatureBundle(
            view_order=order,
            global_features=selected.get("global_features"),
            spatial_features=selected.get("spatial_features"),
        )


@dataclass(frozen=True)
class LoadedFourViewStudent:
    model: Any
    device: Any
    checkpoint_sha256: str
    model_config: Mapping[str, Any]
    visual_feature_mode: str
    reliability: np.ndarray
    edit_gate: np.ndarray
    anchor_retention: np.ndarray
    confidence_receipt: Mapping[str, Any]


def _reliability_value(value: Any, query_key: str) -> float:
    if isinstance(value, Mapping):
        if "reliability" not in value:
            raise ValueError(f"calibration entry {query_key!r} lacks reliability")
        value = value["reliability"]
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"calibration reliability for {query_key!r} must be in [0, 1]")
    return result


def _checkpoint_reliability(
    payload: Mapping[str, Any],
    *,
    uncalibrated_confidence: float | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    raw = payload.get(CHECKPOINT_CALIBRATION_FIELD)
    if raw is None:
        if uncalibrated_confidence is None:
            return np.zeros(len(SEMANTIC_QUERY_KEYS), dtype=np.float64), {
                "status": "FAIL_CLOSED_NO_VALIDATION_CALIBRATION",
                "field": CHECKPOINT_CALIBRATION_FIELD,
                "fallback": "FAIL_CLOSED",
                "calibrated_query_count": 0,
            }
        value = float(uncalibrated_confidence)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("uncalibrated_confidence must be in [0, 1]")
        return np.full(len(SEMANTIC_QUERY_KEYS), value, dtype=np.float64), {
            "status": "UNCALIBRATED_EXPLICIT_CONSTANT_FALLBACK",
            "field": CHECKPOINT_CALIBRATION_FIELD,
            "fallback": "EXPLICIT_CONSTANT",
            "constant_reliability": value,
            "calibrated_query_count": 0,
            "warning": "not empirical calibration; private technical evaluation only",
        }
    if not isinstance(raw, Mapping):
        raise ValueError(f"checkpoint {CHECKPOINT_CALIBRATION_FIELD} must be a mapping")
    if raw.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
        raise ValueError(
            f"checkpoint calibration schema must be {CALIBRATION_SCHEMA_VERSION}"
        )
    if raw.get("method") != "validation_per_query_reliability":
        raise ValueError("unsupported coordinate confidence calibration method")
    if raw.get("fallback", "FAIL_CLOSED") != "FAIL_CLOSED":
        raise ValueError("checkpoint calibration must declare FAIL_CLOSED fallback")
    if "query_keys" in raw and tuple(raw["query_keys"]) != SEMANTIC_QUERY_KEYS:
        raise ValueError("checkpoint calibration query order does not match static schema")
    values = raw.get("per_query")
    reliability = np.zeros(len(SEMANTIC_QUERY_KEYS), dtype=np.float64)
    if isinstance(values, Mapping):
        unknown = set(str(key) for key in values) - set(SEMANTIC_QUERY_KEYS)
        if unknown:
            raise ValueError(f"calibration contains unknown query keys: {sorted(unknown)}")
        populated = 0
        for index, key in enumerate(SEMANTIC_QUERY_KEYS):
            if key in values:
                reliability[index] = _reliability_value(values[key], key)
                populated += 1
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        if len(values) != len(SEMANTIC_QUERY_KEYS):
            raise ValueError("calibration per_query sequence has the wrong length")
        reliability = np.asarray(
            [_reliability_value(value, key) for key, value in zip(SEMANTIC_QUERY_KEYS, values)],
            dtype=np.float64,
        )
        populated = len(values)
    else:
        raise ValueError("checkpoint calibration requires per_query mapping or sequence")
    return reliability, {
        "status": "VALIDATION_RELIABILITY_AVAILABLE",
        "field": CHECKPOINT_CALIBRATION_FIELD,
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "method": "validation_per_query_reliability",
        "fallback": "FAIL_CLOSED",
        "calibrated_query_count": int(populated),
        "missing_queries_fail_closed": int(len(SEMANTIC_QUERY_KEYS) - populated),
        "interpretation": "empirical same-domain validation reliability, not probabilistic calibration",
    }


def _checkpoint_edit_calibration(
    payload: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load a validation-only selector between the student and anchor prior.

    Old checkpoints retain their previous behaviour.  Once the field exists,
    missing queries fail closed: they cannot request an edit and do not add an
    anchor-retention term to the projection objective.
    """

    raw = payload.get(CHECKPOINT_EDIT_CALIBRATION_FIELD)
    count = len(SEMANTIC_QUERY_KEYS)
    if raw is None:
        return (
            np.ones(count, dtype=np.float64),
            np.zeros(count, dtype=np.float64),
            {
                "status": "LEGACY_NO_VALIDATION_EDIT_CALIBRATION",
                "field": CHECKPOINT_EDIT_CALIBRATION_FIELD,
                "fallback": "PRESERVE_PREVIOUS_RELIABILITY_ONLY_BEHAVIOUR",
                "warning": (
                    "no validation comparison against the deterministic anchor; "
                    "student reliability alone does not establish edit utility"
                ),
            },
        )
    if not isinstance(raw, Mapping):
        raise ValueError(
            f"checkpoint {CHECKPOINT_EDIT_CALIBRATION_FIELD} must be a mapping"
        )
    if raw.get("schema_version") != EDIT_CALIBRATION_SCHEMA_VERSION:
        raise ValueError(
            "checkpoint semantic edit calibration schema must be "
            f"{EDIT_CALIBRATION_SCHEMA_VERSION}"
        )
    if raw.get("method") != "student_vs_default_anchor_validation_mae":
        raise ValueError("unsupported semantic edit calibration method")
    if raw.get("fallback", "FAIL_CLOSED") != "FAIL_CLOSED":
        raise ValueError("semantic edit calibration must declare FAIL_CLOSED fallback")
    if "query_keys" in raw and tuple(raw["query_keys"]) != SEMANTIC_QUERY_KEYS:
        raise ValueError("semantic edit calibration query order does not match static schema")
    values = raw.get("per_query")
    if not isinstance(values, Mapping):
        raise ValueError("semantic edit calibration requires a per_query mapping")
    unknown = set(str(key) for key in values) - set(SEMANTIC_QUERY_KEYS)
    if unknown:
        raise ValueError(
            f"semantic edit calibration contains unknown query keys: {sorted(unknown)}"
        )
    edit_gate = np.zeros(count, dtype=np.float64)
    anchor_retention = np.zeros(count, dtype=np.float64)
    populated = 0
    editable = 0
    protected = 0
    for index, key in enumerate(SEMANTIC_QUERY_KEYS):
        entry = values.get(key)
        if entry is None:
            continue
        if not isinstance(entry, Mapping):
            raise ValueError(f"semantic edit calibration entry {key!r} must be a mapping")
        allow = entry.get("allow_student_edit")
        if not isinstance(allow, bool):
            raise ValueError(
                f"semantic edit calibration entry {key!r} lacks boolean allow_student_edit"
            )
        retention = float(entry.get("anchor_retention_weight", 0.0))
        if not math.isfinite(retention) or not 0.0 <= retention <= 1.0:
            raise ValueError(
                f"anchor retention weight for {key!r} must be in [0, 1]"
            )
        edit_gate[index] = 1.0 if allow else 0.0
        anchor_retention[index] = retention
        populated += 1
        editable += int(allow)
        protected += int(retention > 0.0)
    return edit_gate, anchor_retention, {
        "status": "VALIDATION_EDIT_SELECTOR_AVAILABLE",
        "field": CHECKPOINT_EDIT_CALIBRATION_FIELD,
        "schema_version": EDIT_CALIBRATION_SCHEMA_VERSION,
        "method": "student_vs_default_anchor_validation_mae",
        "fallback": "FAIL_CLOSED",
        "calibrated_query_count": populated,
        "student_edit_query_count": editable,
        "anchor_retention_query_count": protected,
        "missing_queries_fail_closed": count - populated,
        "interpretation": (
            "same-generator validation selection between the visual student and "
            "the deterministic provisional anchor; not ground-truth CAD confidence"
        ),
    }


def load_four_view_student_checkpoint(
    path: str | Path,
    *,
    device: str = "auto",
    uncalibrated_confidence: float | None = None,
) -> LoadedFourViewStudent:
    """Load and verify an inference-only student checkpoint."""

    import torch

    resolved_path = Path(path)
    resolved_device = torch.device(
        "cuda"
        if device == "auto" and torch.cuda.is_available()
        else ("cpu" if device == "auto" else device)
    )
    payload = torch.load(resolved_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("student checkpoint must contain a mapping")
    if payload.get("stage") != "four_view_student":
        raise ValueError("checkpoint is not a four-view student checkpoint")
    if tuple(payload.get("query_keys", ())) != SEMANTIC_QUERY_KEYS:
        raise ValueError("checkpoint query keys do not match the static query schema")
    if payload.get("inference_contract") not in {
        None,
        "four_view_features_plus_category_only_no_pattern_graph",
    }:
        raise ValueError("checkpoint inference contract permits unsupported inputs")
    mode = str(payload.get("visual_feature_mode", ""))
    if mode not in {"global", "spatial", "global+spatial"}:
        raise ValueError("checkpoint has an unsupported visual_feature_mode")
    config = payload.get("model_config")
    if not isinstance(config, Mapping):
        raise ValueError("checkpoint lacks model_config")
    if int(config.get("max_views", 0)) < 4:
        raise ValueError("student checkpoint cannot encode all four canonical views")
    state = payload.get("model_state")
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint lacks model_state")
    reliability, confidence_receipt = _checkpoint_reliability(
        payload, uncalibrated_confidence=uncalibrated_confidence
    )
    edit_gate, anchor_retention, edit_receipt = _checkpoint_edit_calibration(payload)
    model = build_four_view_semantic_student(config)
    model.load_state_dict(state, strict=True)
    model.to(resolved_device).eval()
    return LoadedFourViewStudent(
        model=model,
        device=resolved_device,
        checkpoint_sha256=_sha256_file(resolved_path),
        model_config=dict(config),
        visual_feature_mode=mode,
        reliability=reliability,
        edit_gate=edit_gate,
        anchor_retention=anchor_retention,
        confidence_receipt={
            **confidence_receipt,
            "semantic_edit_selector": edit_receipt,
        },
    )


@dataclass(frozen=True)
class StaticSemanticPrediction:
    category: str
    presence_probability: np.ndarray
    coordinates: np.ndarray
    coordinate_confidence: np.ndarray
    query_mask: np.ndarray
    coordinate_mask: np.ndarray
    confidence_receipt: Mapping[str, Any]
    edit_coordinate_confidence: np.ndarray | None = None
    anchor_retention_confidence: np.ndarray | None = None

    def rows(self) -> list[dict[str, Any]]:
        output = []
        for index, query in enumerate(SEMANTIC_QUERY_INVENTORY):
            names = query.coordinate_names
            output.append(
                {
                    "query_key": query.key,
                    "category": query.category,
                    "kind": query.kind,
                    "name": query.name,
                    "category_applicable": bool(self.query_mask[index]),
                    "presence_probability": float(self.presence_probability[index]),
                    "coordinate_confidence": float(self.coordinate_confidence[index]),
                    "edit_coordinate_confidence": float(
                        self.coordinate_confidence[index]
                        if self.edit_coordinate_confidence is None
                        else self.edit_coordinate_confidence[index]
                    ),
                    "anchor_retention_confidence": float(
                        0.0
                        if self.anchor_retention_confidence is None
                        else self.anchor_retention_confidence[index]
                    ),
                    "predicted_coordinates": {
                        name: float(self.coordinates[index, channel])
                        for channel, name in enumerate(names)
                    },
                    "coordinate_channels_from": "STATIC_QUERY_SCHEMA_NO_GROUND_TRUTH_MASK",
                }
            )
        return output


def _validate_features_for_checkpoint(
    loaded: LoadedFourViewStudent,
    bundle: FourViewFeatureBundle,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    mode = loaded.visual_feature_mode
    global_values = bundle.global_features if mode in {"global", "global+spatial"} else None
    spatial_values = bundle.spatial_features if mode in {"spatial", "global+spatial"} else None
    if mode in {"global", "global+spatial"} and global_values is None:
        raise ValueError("checkpoint requires global four-view features")
    if mode in {"spatial", "global+spatial"} and spatial_values is None:
        raise ValueError("checkpoint requires spatial four-view features")
    if global_values is not None and global_values.shape[-1] != int(
        loaded.model_config["global_feature_dim"]
    ):
        raise ValueError("global feature dimension does not match checkpoint")
    if spatial_values is not None and spatial_values.shape[-1] != int(
        loaded.model_config["spatial_feature_dim"]
    ):
        raise ValueError("spatial feature dimension does not match checkpoint")
    return global_values, spatial_values


def predict_static_semantic_queries(
    loaded: LoadedFourViewStudent,
    bundle: FourViewFeatureBundle,
    *,
    category: str,
    pattern_input: Any | None = None,
) -> StaticSemanticPrediction:
    """Predict the complete static query schema from visual features only."""

    import torch

    if pattern_input is not None:
        raise ModalityContractError(
            "pattern input is forbidden in four-view student inference"
        )
    if category not in CATEGORY_NAMES:
        raise ValueError(f"unsupported category: {category!r}")
    global_values, spatial_values = _validate_features_for_checkpoint(loaded, bundle)
    category_id = CATEGORY_NAMES.index(category)
    output = infer_four_view_semantics(
        loaded.model,
        category_ids=torch.as_tensor([category_id], device=loaded.device, dtype=torch.long),
        global_features=(
            torch.as_tensor(global_values[None], device=loaded.device, dtype=torch.float32)
            if global_values is not None
            else None
        ),
        spatial_features=(
            torch.as_tensor(spatial_values[None], device=loaded.device, dtype=torch.float32)
            if spatial_values is not None
            else None
        ),
        view_valid=torch.ones((1, 4), device=loaded.device, dtype=torch.bool),
        pattern_graph=None,
    )
    logits = output["presence_logits"][0].detach().float().cpu().numpy().astype(np.float64)
    coordinates = output["coordinates"][0].detach().float().cpu().numpy().astype(np.float64)
    if logits.shape != (len(SEMANTIC_QUERY_KEYS),):
        raise RuntimeError("student returned an invalid presence table")
    if coordinates.shape != (len(SEMANTIC_QUERY_KEYS), MAX_COORDINATE_DIM):
        raise RuntimeError("student returned an invalid coordinate table")
    if not np.isfinite(logits).all() or not np.isfinite(coordinates).all():
        raise RuntimeError("student returned non-finite semantic predictions")
    expected_query_mask = np.asarray(category_query_mask(category), dtype=bool)
    expected_coordinate_mask = np.asarray(query_coordinate_mask(), dtype=bool)
    returned_query_mask = output["query_mask"][0].detach().cpu().numpy().astype(bool)
    returned_coordinate_mask = output["coordinate_mask"][0].detach().cpu().numpy().astype(bool)
    expected_coordinate_mask = expected_coordinate_mask & expected_query_mask[:, None]
    if not np.array_equal(returned_query_mask, expected_query_mask):
        raise RuntimeError("student query mask does not match the static category schema")
    if not np.array_equal(returned_coordinate_mask, expected_coordinate_mask):
        raise RuntimeError("student coordinate mask does not match the static query schema")
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
    confidence = probability * loaded.reliability
    confidence = np.where(expected_query_mask, confidence, 0.0)
    edit_confidence = confidence * loaded.edit_gate
    # The checkpoint value is already validation-reliability weighted.
    anchor_retention = loaded.anchor_retention.copy()
    anchor_retention = np.where(expected_query_mask, anchor_retention, 0.0)
    return StaticSemanticPrediction(
        category=category,
        presence_probability=probability,
        coordinates=coordinates,
        coordinate_confidence=confidence,
        query_mask=expected_query_mask,
        coordinate_mask=expected_coordinate_mask,
        confidence_receipt=dict(loaded.confidence_receipt),
        edit_coordinate_confidence=edit_confidence,
        anchor_retention_confidence=anchor_retention,
    )


def _topology_signature(document: PatternDocument) -> dict[str, Any]:
    return {
        "panels": [
            {
                "id": panel.id,
                "edges": [
                    {"id": edge.id, "point_count": len(edge.points)}
                    for edge in panel.edges
                ],
            }
            for panel in document.panels
        ],
        "stitches": [
            {
                "id": stitch.id,
                "side_a": [
                    stitch.side_a.panel_id,
                    stitch.side_a.edge_id,
                    bool(stitch.side_a.reversed),
                ],
                "side_b": [
                    stitch.side_b.panel_id,
                    stitch.side_b.edge_id,
                    bool(stitch.side_b.reversed),
                ],
            }
            for stitch in document.stitches
        ],
    }


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _plan_receipt(plan: SemanticResidualPlan) -> dict[str, Any]:
    return {
        "category": plan.category,
        "source": plan.source,
        "landmark_residuals": {
            name: asdict(value) for name, value in sorted(plan.landmark_residuals.items())
        },
        "path_residuals": {
            name: asdict(value) for name, value in sorted(plan.path_residuals.items())
        },
        "gated_queries": dict(sorted(plan.gated_queries.items())),
    }


def _scaled_residual_plan(
    plan: SemanticResidualPlan,
    scale: float,
) -> SemanticResidualPlan:
    """Interpolate a residual plan towards the identity edit.

    Landmark displacements and path offsets scale linearly. Multiplicative
    path terms interpolate around one, so every scale in ``[0, 1]`` remains
    inside the bounds already validated by :class:`PathResidual`.
    """

    value = float(scale)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("semantic projection scale must be finite and in [0, 1]")
    landmarks = {
        name: LandmarkResidual(
            dx_cm=residual.dx_cm * value,
            dy_cm=residual.dy_cm * value,
            influence_radius_cm=residual.influence_radius_cm,
            confidence=residual.confidence,
        )
        for name, residual in plan.landmark_residuals.items()
    }
    paths = {
        name: PathResidual(
            chord_scale=1.0 + value * (residual.chord_scale - 1.0),
            normal_scale=1.0 + value * (residual.normal_scale - 1.0),
            normal_offset_cm=value * residual.normal_offset_cm,
            confidence=residual.confidence,
        )
        for name, residual in plan.path_residuals.items()
    }
    if value == 0.0:
        landmarks = {}
        paths = {}
    scaled = SemanticResidualPlan(
        category=plan.category,
        landmark_residuals=landmarks,
        path_residuals=paths,
        gated_queries=plan.gated_queries,
        source=plan.source,
        schema_version=plan.schema_version,
    )
    scaled.validate()
    return scaled


def _projection_coordinate_loss(
    target: Any,
    prediction: StaticSemanticPrediction,
    *,
    confidence_threshold: float,
    anchor_reference: Any | None = None,
) -> tuple[float | None, int, float]:
    """Return the validation-selected student/anchor semantic objective.

    Queries for which the visual student beat the deterministic default on
    validation are scored against the student's coordinates.  Queries for
    which the anchor was better are scored against the unchanged anchor and
    therefore penalize collateral deformation.  The selector is frozen in
    the checkpoint and no test or inference ground truth is inspected.
    """

    threshold = float(confidence_threshold)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("projection confidence threshold must be in [0, 1]")
    edit_confidence = prediction.edit_coordinate_confidence
    if edit_confidence is None:
        edit_confidence = prediction.coordinate_confidence
    else:
        edit_confidence = np.asarray(edit_confidence, dtype=np.float64)
        if edit_confidence.shape != prediction.coordinate_confidence.shape:
            raise ValueError("edit coordinate confidence has the wrong shape")
        if not np.all(np.isfinite(edit_confidence)) or np.any(edit_confidence < 0.0):
            raise ValueError("edit coordinate confidence must be finite and non-negative")
    query_confident = edit_confidence >= threshold
    finite = np.isfinite(target.coordinates) & np.isfinite(prediction.coordinates)
    student_mask = (
        target.coordinate_mask
        & prediction.coordinate_mask
        & prediction.query_mask[:, None]
        & query_confident[:, None]
        & finite
    )
    retention_values = prediction.anchor_retention_confidence
    if retention_values is None:
        retention_values = np.zeros_like(prediction.coordinate_confidence)
    else:
        retention_values = np.asarray(retention_values, dtype=np.float64)
        if retention_values.shape != prediction.coordinate_confidence.shape:
            raise ValueError("anchor retention confidence has the wrong shape")
        if not np.all(np.isfinite(retention_values)) or np.any(retention_values < 0.0):
            raise ValueError("anchor retention confidence must be finite and non-negative")
    retention_mask = np.zeros_like(student_mask)
    if anchor_reference is not None:
        retention_finite = np.isfinite(target.coordinates) & np.isfinite(
            anchor_reference.coordinates
        )
        retention_mask = (
            target.coordinate_mask
            & anchor_reference.coordinate_mask
            & prediction.coordinate_mask
            & prediction.query_mask[:, None]
            & (retention_values > 0.0)[:, None]
            & retention_finite
        )
    # Calibration entries are intentionally exclusive.  Reject overlap rather
    # than silently double-counting a query against incompatible targets.
    if bool(np.any(student_mask & retention_mask)):
        raise ValueError("student-edit and anchor-retention scoring masks overlap")
    count = int(student_mask.sum() + retention_mask.sum())
    if count == 0:
        return None, 0, 0.0
    student_weights = np.broadcast_to(
        edit_confidence[:, None], target.coordinates.shape
    )
    retention_weights = np.broadcast_to(
        retention_values[:, None], target.coordinates.shape
    )
    weight_sum = float(
        student_weights[student_mask].sum()
        + retention_weights[retention_mask].sum()
    )
    if not math.isfinite(weight_sum) or weight_sum <= 0.0:
        return None, count, weight_sum
    student_error = np.abs(target.coordinates - prediction.coordinates)
    weighted_error = float(
        np.dot(student_error[student_mask], student_weights[student_mask])
    )
    if anchor_reference is not None and bool(retention_mask.any()):
        retention_error = np.abs(target.coordinates - anchor_reference.coordinates)
        weighted_error += float(
            np.dot(
                retention_error[retention_mask],
                retention_weights[retention_mask],
            )
        )
    loss = weighted_error / weight_sum
    if not math.isfinite(loss):
        return None, count, weight_sum
    return loss, count, weight_sum


@dataclass(frozen=True)
class _SemanticProjectionSelection:
    document: PatternDocument
    plan: SemanticResidualPlan
    validation: ValidationReport | None
    selected_scale: float
    anchor_loss: float | None
    selected_loss: float | None
    scoring_coordinate_count: int
    scoring_weight_sum: float
    candidates: tuple[Mapping[str, Any], ...]
    rejection_reason: str | None


def _select_semantic_projection_candidate(
    anchor: PatternDocument,
    requested_plan: SemanticResidualPlan,
    prediction: StaticSemanticPrediction,
    *,
    category: str,
    confidence_threshold: float,
    scales: Sequence[float] = SEMANTIC_PROJECTION_SCALES,
) -> _SemanticProjectionSelection:
    """Choose only a validated edit that improves student self-consistency.

    This is not an accuracy oracle: no ground-truth pattern is inspected. A
    candidate must preserve exact topology, pass validation, and strictly
    reduce confidence-weighted semantic-coordinate loss versus the anchor.
    """

    if prediction.category != category or requested_plan.category != category:
        raise ValueError("semantic projection category mismatch")
    requested_plan.validate()
    resolved_scales = tuple(float(value) for value in scales)
    if not resolved_scales:
        raise ValueError("semantic projection requires at least one non-zero scale")
    if any(
        not math.isfinite(value) or not 0.0 < value <= 1.0
        for value in resolved_scales
    ):
        raise ValueError("semantic projection scales must be finite and in (0, 1]")
    if len(set(resolved_scales)) != len(resolved_scales):
        raise ValueError("semantic projection scales must be unique")

    source_frame = anchor.annotations.get("semantic_coordinate_frame", {})
    if not isinstance(source_frame, Mapping):
        raise ValueError("semantic_coordinate_frame annotation must be a mapping")
    source_y_axis_down = source_frame.get("source_y_axis_down", False)
    if not isinstance(source_y_axis_down, bool):
        raise ValueError("semantic coordinate axis flag must be boolean")
    anchor_target = semantic_target_from_pattern_document(
        anchor,
        category=category,
        source="provisional_inference_anchor_projection",
        provenance_status=PROVENANCE_STATUS,
        source_y_axis_down=source_y_axis_down,
    )
    anchor_loss, coordinate_count, weight_sum = _projection_coordinate_loss(
        anchor_target,
        prediction,
        confidence_threshold=confidence_threshold,
        anchor_reference=anchor_target,
    )
    empty_plan = _scaled_residual_plan(requested_plan, 0.0)
    if anchor_loss is None:
        return _SemanticProjectionSelection(
            document=anchor,
            plan=empty_plan,
            validation=None,
            selected_scale=0.0,
            anchor_loss=None,
            selected_loss=None,
            scoring_coordinate_count=coordinate_count,
            scoring_weight_sum=weight_sum,
            candidates=(),
            rejection_reason="NO_HIGH_CONFIDENCE_SEMANTIC_COORDINATES",
        )

    before_topology = _topology_signature(anchor)
    rows: list[dict[str, Any]] = []
    best_document: PatternDocument | None = None
    best_plan: SemanticResidualPlan | None = None
    best_validation: ValidationReport | None = None
    best_scale = 0.0
    best_loss = anchor_loss
    strict_improvement = max(1e-10, abs(anchor_loss) * 1e-7)
    for scale in resolved_scales:
        scaled_plan = _scaled_residual_plan(requested_plan, scale)
        row: dict[str, Any] = {
            "scale": scale,
            "status": "REJECTED",
            "loss": None,
            "improvement_from_anchor": None,
            "validation_accepted": False,
            "topology_preserved": False,
        }
        try:
            candidate = apply_semantic_residual(anchor, scaled_plan)
            topology_preserved = _topology_signature(candidate) == before_topology
            row["topology_preserved"] = topology_preserved
            validation = validate_pattern(candidate)
            row["validation_accepted"] = bool(validation.accepted)
            if not topology_preserved:
                raise RuntimeError("semantic edit changed the exact topology signature")
            if not validation.accepted:
                raise RuntimeError("semantic edit failed PatternDocument validation")
            candidate_target = semantic_target_from_pattern_document(
                candidate,
                category=category,
                source="provisional_inference_candidate_projection",
                provenance_status=PROVENANCE_STATUS,
                source_y_axis_down=source_y_axis_down,
            )
            candidate_loss, candidate_count, candidate_weight_sum = (
                _projection_coordinate_loss(
                    candidate_target,
                    prediction,
                    confidence_threshold=confidence_threshold,
                    anchor_reference=anchor_target,
                )
            )
            if (
                candidate_loss is None
                or candidate_count != coordinate_count
                or not math.isclose(
                    candidate_weight_sum, weight_sum, rel_tol=1e-12, abs_tol=1e-12
                )
            ):
                raise RuntimeError("candidate changed semantic projection scoring support")
            improvement = anchor_loss - candidate_loss
            row["loss"] = candidate_loss
            row["improvement_from_anchor"] = improvement
            if candidate_loss < anchor_loss - strict_improvement:
                row["status"] = "ELIGIBLE_IMPROVEMENT"
                if (
                    best_document is None
                    or candidate_loss < best_loss - strict_improvement
                    or (
                        math.isclose(
                            candidate_loss,
                            best_loss,
                            rel_tol=0.0,
                            abs_tol=strict_improvement,
                        )
                        and scale < best_scale
                    )
                ):
                    best_document = candidate
                    best_plan = scaled_plan
                    best_validation = validation
                    best_scale = scale
                    best_loss = candidate_loss
            else:
                row["status"] = "REJECTED_NO_STRICT_IMPROVEMENT"
        except (ValueError, RuntimeError) as exc:
            row["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        rows.append(row)

    if best_document is None or best_plan is None:
        valid_count = sum(
            bool(row["validation_accepted"] and row["topology_preserved"])
            for row in rows
        )
        reason = (
            "NO_VALID_TOPOLOGY_PRESERVING_CANDIDATE"
            if valid_count == 0
            else "NO_CANDIDATE_IMPROVED_SEMANTIC_PROJECTION_LOSS"
        )
        return _SemanticProjectionSelection(
            document=anchor,
            plan=empty_plan,
            validation=None,
            selected_scale=0.0,
            anchor_loss=anchor_loss,
            selected_loss=anchor_loss,
            scoring_coordinate_count=coordinate_count,
            scoring_weight_sum=weight_sum,
            candidates=tuple(rows),
            rejection_reason=reason,
        )
    return _SemanticProjectionSelection(
        document=best_document,
        plan=best_plan,
        validation=best_validation,
        selected_scale=best_scale,
        anchor_loss=anchor_loss,
        selected_loss=best_loss,
        scoring_coordinate_count=coordinate_count,
        scoring_weight_sum=weight_sum,
        candidates=tuple(rows),
        rejection_reason=None,
    )


@dataclass(frozen=True)
class FourViewPatternInferenceResult:
    document: PatternDocument
    prediction: StaticSemanticPrediction
    plan: SemanticResidualPlan
    receipt: Mapping[str, Any]

    def save(self, pattern_path: str | Path, receipt_path: str | Path) -> None:
        """Write only the vector pattern and text/numeric receipt."""

        self.document.write_json(Path(pattern_path))
        resolved_receipt = Path(receipt_path)
        resolved_receipt.parent.mkdir(parents=True, exist_ok=True)
        resolved_receipt.write_text(
            json.dumps(self.receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def infer_provisional_basic_pattern(
    loaded: LoadedFourViewStudent,
    bundle: FourViewFeatureBundle,
    *,
    category: str,
    curve_samples: int = 24,
    planning_config: ResidualPlanningConfig | None = None,
    pattern_input: Any | None = None,
) -> FourViewPatternInferenceResult:
    """Run prediction -> anchor -> residual planning -> edit -> validation.

    An invalid edit is never serialized as if it passed.  The bridge returns
    the unchanged, independently validated provisional anchor and records the
    failed edit attempt in that case.
    """

    if pattern_input is not None:
        raise ModalityContractError(
            "pattern input is forbidden; the inference anchor is selected internally"
        )
    if curve_samples < 4:
        raise ValueError("curve_samples must be at least four")
    prediction = predict_static_semantic_queries(
        loaded, bundle, category=category, pattern_input=None
    )
    block = build_basic_block(
        category,
        sample_id=f"{category}_provisional_inference_anchor",
        metadata={"anchor_selection": "deterministic_category_default"},
    )
    if block.provenance.status != PROVENANCE_STATUS:
        raise RuntimeError("inference anchor lost its provisional-review status")
    anchor = block.to_pattern_document(curve_samples=curve_samples)
    anchor_validation = validate_pattern(anchor)
    if not anchor_validation.accepted:
        raise RuntimeError("selected provisional BasicBlock anchor failed validation")
    anchor_target = semantic_target_from_basic_block(block, curve_samples=curve_samples)
    anchor_presence = np.where(
        anchor_target.query_applicability,
        anchor_target.presence,
        0.0,
    )
    resolved_planning_config = planning_config or ResidualPlanningConfig()
    resolved_planning_config.validate()
    requested_plan = build_semantic_residual_plan(
        category,
        anchor_target.coordinates,
        anchor_presence,
        prediction.coordinates,
        prediction.presence_probability,
        (
            prediction.coordinate_confidence
            if prediction.edit_coordinate_confidence is None
            else prediction.edit_coordinate_confidence
        ),
        anchor,
        config=resolved_planning_config,
    )
    before_topology = _topology_signature(anchor)
    edit_failure: dict[str, str] | None = None
    candidate_validation: ValidationReport | None = None
    projection: _SemanticProjectionSelection | None = None
    if requested_plan.landmark_residuals or requested_plan.path_residuals:
        try:
            projection = _select_semantic_projection_candidate(
                anchor,
                requested_plan,
                prediction,
                category=category,
                confidence_threshold=resolved_planning_config.confidence_threshold,
            )
            selected = projection.document
            plan = projection.plan
            candidate_validation = projection.validation
            if projection.selected_scale > 0.0:
                status = "APPLIED_VALIDATED"
            else:
                status = (
                    "EDIT_REJECTED_SEMANTIC_PROJECTION_"
                    "FALLBACK_TO_VALIDATED_ANCHOR"
                )
                edit_failure = {
                    "type": "SemanticProjectionRejection",
                    "message": str(projection.rejection_reason),
                }
        except (ValueError, RuntimeError) as exc:
            selected = anchor
            plan = _scaled_residual_plan(requested_plan, 0.0)
            status = "EDIT_REJECTED_FALLBACK_TO_VALIDATED_ANCHOR"
            edit_failure = {"type": type(exc).__name__, "message": str(exc)}
    else:
        selected = anchor
        plan = requested_plan
        status = "NO_ELIGIBLE_RESIDUALS_VALIDATED_ANCHOR"

    final_topology = _topology_signature(selected)
    final_validation = validate_pattern(selected)
    if final_topology != before_topology or not final_validation.accepted:
        raise RuntimeError("fail-closed inference output is not a valid topology-preserved anchor")

    plan_data = _plan_receipt(plan)
    requested_plan_data = _plan_receipt(requested_plan)
    feature_digest = bundle.tensor_sha256
    inference_id = hashlib.sha256(
        f"{category}:{loaded.checkpoint_sha256}:{feature_digest}".encode("ascii")
    ).hexdigest()[:16]
    compact_annotation = {
        "schema_version": INFERENCE_SCHEMA_VERSION,
        "status": status,
        "checkpoint_sha256": loaded.checkpoint_sha256,
        "feature_tensor_sha256": feature_digest,
        "view_order": list(CANONICAL_VIEW_ORDER),
        "confidence_status": prediction.confidence_receipt["status"],
        "landmark_edit_count": len(plan.landmark_residuals),
        "path_edit_count": len(plan.path_residuals),
        "semantic_projection_scale": (
            projection.selected_scale if projection is not None else 0.0
        ),
        "topology_preserved": True,
        "validation_accepted": True,
        "source_images_embedded": False,
        "source_paths_embedded": False,
    }
    document = replace(
        selected,
        pattern_id=f"{category}_four_view_semantic_{inference_id}",
        generator="provisional BasicBlock + four-view semantic residual inference",
        provenance={
            **selected.provenance,
            "status": PROVENANCE_STATUS,
            "four_view_student_checkpoint_sha256": loaded.checkpoint_sha256,
            "feature_tensor_sha256": feature_digest,
            "industrial_pattern_claim": False,
        },
        annotations={
            **selected.annotations,
            "four_view_semantic_inference": compact_annotation,
        },
    )
    receipt: dict[str, Any] = {
        "schema_version": INFERENCE_SCHEMA_VERSION,
        "status": status,
        "category": category,
        "input_contract": {
            "modality": "FOUR_PRECOMPUTED_VISUAL_FEATURES_ONLY",
            "view_order": list(CANONICAL_VIEW_ORDER),
            "feature_shapes": bundle.shape_receipt(),
            "feature_tensor_sha256": feature_digest,
            "pattern_input": "REJECTED",
            "source_images_embedded": False,
            "source_paths_embedded": False,
        },
        "checkpoint": {
            "sha256": loaded.checkpoint_sha256,
            "stage": "four_view_student",
            "visual_feature_mode": loaded.visual_feature_mode,
            "query_schema": "STATIC_INVENTORY_NO_GROUND_TRUTH_MASK",
        },
        "coordinate_confidence": dict(prediction.confidence_receipt),
        "anchor": {
            "selection": "DETERMINISTIC_CATEGORY_DEFAULT_BASIC_BLOCK",
            "provenance_status": block.provenance.status,
            "industrial_pattern_claim": block.provenance.industrial_pattern_claim,
            "measurements_cm": dict(sorted(block.measurements.items())),
            "design_parameters": dict(sorted(block.design.items())),
        },
        "static_query_predictions": prediction.rows(),
        "residual_plan": plan_data,
        "requested_residual_plan": requested_plan_data,
        "semantic_projection": {
            "status": (
                "SELECTED_STRICT_IMPROVEMENT"
                if projection is not None and projection.selected_scale > 0.0
                else (
                    "REJECTED_FAIL_CLOSED"
                    if projection is not None
                    else "NOT_RUN_NO_ELIGIBLE_RESIDUALS"
                )
            ),
            "objective": (
                "validation_selected_confidence_weighted_semantic_coordinate_mae_"
                "to_student_or_anchor"
            ),
            "uses_ground_truth": False,
            "confidence_threshold": resolved_planning_config.confidence_threshold,
            "attempted_scales": list(SEMANTIC_PROJECTION_SCALES),
            "selected_scale": (
                projection.selected_scale if projection is not None else 0.0
            ),
            "anchor_loss": projection.anchor_loss if projection is not None else None,
            "selected_loss": (
                projection.selected_loss if projection is not None else None
            ),
            "strictly_improved": bool(
                projection is not None
                and projection.selected_scale > 0.0
                and projection.anchor_loss is not None
                and projection.selected_loss is not None
                and projection.selected_loss < projection.anchor_loss
            ),
            "scoring_coordinate_count": (
                projection.scoring_coordinate_count if projection is not None else 0
            ),
            "scoring_weight_sum": (
                projection.scoring_weight_sum if projection is not None else 0.0
            ),
            "rejection_reason": (
                projection.rejection_reason if projection is not None else None
            ),
            "candidates": (
                list(projection.candidates) if projection is not None else []
            ),
            "interpretation": (
                "validation-selected self-consistency safeguard: student-preferred "
                "queries move toward the visual estimate while anchor-preferred queries "
                "penalize collateral deformation; not ground-truth pattern accuracy"
            ),
        },
        "edit_failure": edit_failure,
        "validation": {
            "anchor": anchor_validation.to_dict(),
            "candidate": candidate_validation.to_dict() if candidate_validation is not None else None,
            "final": final_validation.to_dict(),
        },
        "topology": {
            "preserved": True,
            "before_sha256": _json_sha256(before_topology),
            "after_sha256": _json_sha256(final_topology),
            "signature": final_topology,
        },
        "output": {
            "pattern_id": document.pattern_id,
            "contains_source_images": False,
            "contains_source_paths": False,
            "permission_boundary": (
                "provisional private technical output; not industrial pattern truth"
            ),
        },
    }
    return FourViewPatternInferenceResult(document, prediction, plan, receipt)


__all__ = [
    "CALIBRATION_SCHEMA_VERSION",
    "CANONICAL_VIEW_ORDER",
    "CHECKPOINT_CALIBRATION_FIELD",
    "CHECKPOINT_EDIT_CALIBRATION_FIELD",
    "EDIT_CALIBRATION_SCHEMA_VERSION",
    "FourViewFeatureBundle",
    "FourViewPatternInferenceResult",
    "INFERENCE_SCHEMA_VERSION",
    "LoadedFourViewStudent",
    "StaticSemanticPrediction",
    "infer_provisional_basic_pattern",
    "load_four_view_student_checkpoint",
    "load_precomputed_four_view_features",
    "predict_static_semantic_queries",
]
