"""Learning utilities for construction-traced basic T-shirt patterns.

This module intentionally does *not* consume source names, panel names,
operation names, formulas, or trace labels as model inputs.  The network sees
only numerical 2D boundary geometry and, in the opt-in ``pattern+body`` mode,
standardized body measurements.  The source and split fields are retained for
stratified evaluation only.

The boundary encoder has no serialization-index or positional feature.  A
Transformer without positional encodings therefore treats the edge list as a
set, while deterministic cyclic-shift, winding-reversal, rotation, and scale
augmentations exercise the intended invariances/equivariances explicitly.
"""

from __future__ import annotations

import hashlib
import gzip
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .tshirt_schema import (
    CANONICAL_EDGE_ROLES,
    CANONICAL_PANEL_ROLES,
    TShirtTraceRecord,
    TracedEdge,
    TracedPanel,
)


LANDMARK_NAMES = ("FNP", "BNP", "SNP", "SP")
"""Construction landmarks learned by the T-shirt baseline.

BP is deliberately excluded: neither the basic GarmentCode T-shirt recipe nor
FreeSewing Teagan defines a bust point.  Keeping BP in the trace schema as
``unavailable`` is preferable to inventing a false target.
"""

PANEL_ROLES = (*CANONICAL_PANEL_ROLES, "other")
EDGE_ROLES = (*CANONICAL_EDGE_ROLES, "other")
CURVE_KINDS = ("line", "quadratic_bezier", "cubic_bezier", "bezier", "arc", "other")

# Layout is documented so augmentation can transform the numerical geometry
# without having to reconstruct source objects.
FEATURE_SLICES = {
    "start": slice(0, 2),
    "end": slice(2, 4),
    "delta": slice(4, 6),
    "length": 6,
    "chord": 7,
    "direction_sin": 8,
    "direction_cos": 9,
    "control_1": slice(10, 12),
    "control_2": slice(12, 14),
    "control_count": 14,
    "arc_center": slice(15, 17),
    "arc_radius": 17,
    "arc_direction": 18,
    "curve_kind": slice(19, 25),
}
EDGE_FEATURE_DIM = 25


DEFAULT_TSHIRT_MODEL_CONFIG: dict[str, Any] = {
    "width": 128,
    "heads": 4,
    "layers": 4,
    "feedforward_multiplier": 3,
    "dropout": 0.1,
    "maximum_edges": 48,
    "batch_size": 96,
    "epochs": 30,
    "learning_rate": 3e-4,
    "weight_decay": 1e-3,
    "seed": 2027,
    "mode": "pattern-only",
    "augmentation": {
        "cyclic_shift": True,
        "reverse_probability": 0.5,
        "rotation_degrees": 180.0,
        "minimum_scale": 0.80,
        "maximum_scale": 1.20,
    },
    "loss_weights": {
        "edge": 1.0,
        "panel": 0.5,
        "landmark_existence": 0.5,
        "landmark_coordinate": 2.0,
    },
}


_PANEL_ALIASES = {
    "front": "front",
    "front_bodice": "front",
    "front_torso": "front",
    "ftorso": "front",
    "back": "back",
    "back_bodice": "back",
    "back_torso": "back",
    "btorso": "back",
    "sleeve": "sleeve",
    "front_sleeve": "sleeve",
    "back_sleeve": "sleeve",
    "neckband": "neckband",
    "collar": "neckband",
}

_EDGE_ALIASES = {
    "neck": "neckline",
    "collar": "neckline",
    "collar_interface": "neckline",
    "shoulder_seam": "shoulder",
    "armscye": "armhole",
    "side": "side_seam",
    "centerfront": "center_front",
    "centre_front": "center_front",
    "centerback": "center_back",
    "centre_back": "center_back",
    "bottom": "hemline",
    "hem": "hemline",
    "sleeve_cap": "sleeve_head",
    "sleeve_cap_seam": "sleeve_head",
    "underarm": "sleeve_underarm",
    "cuff": "sleeve_hem",
    "neckband": "neckband_attachment",
}


def _slug(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def canonical_panel_role(value: Any) -> str:
    normalized = _slug(value)
    if normalized in PANEL_ROLES:
        return normalized
    return _PANEL_ALIASES.get(normalized, "other")


def canonical_edge_role(value: Any) -> str:
    normalized = _slug(value)
    if normalized in EDGE_ROLES:
        return normalized
    return _EDGE_ALIASES.get(normalized, "other")


def source_value(record: TShirtTraceRecord) -> str:
    """Return a stable, human-readable source group without using it as input."""

    for key in ("name", "source", "project", "generator", "id"):
        raw = record.source.get(key)
        if raw is not None and str(raw).strip():
            return str(raw).strip()
    payload = json.dumps(record.source, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "source-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def deterministic_split(
    sample_id: str,
    *,
    seed: int = 2027,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> str:
    """Hash a stable sample id into train/validation/test without RNG state."""

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between zero and one")
    if not 0.0 <= validation_fraction < 1.0 - train_fraction:
        raise ValueError("validation_fraction leaves no test partition")
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).digest()
    unit = int.from_bytes(digest[:8], "big") / float(2**64)
    if unit < train_fraction:
        return "train"
    if unit < train_fraction + validation_fraction:
        return "validation"
    return "test"


def read_tshirt_records(path: str | Path, *, validate: bool = True) -> tuple[TShirtTraceRecord, ...]:
    """Read one trace, a JSON array/object, or JSONL traces."""

    source_path = Path(path)
    if source_path.suffix.lower() == ".gz":
        with gzip.open(source_path, "rt", encoding="utf-8") as stream:
            values = [json.loads(line) for line in stream if line.strip()]
    elif source_path.suffix.lower() == ".jsonl":
        values = [json.loads(line) for line in source_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        raw = json.loads(source_path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            values = raw
        elif isinstance(raw, Mapping) and "records" in raw:
            values = raw["records"]
        else:
            values = [raw]
    records = tuple(TShirtTraceRecord.from_dict(value) for value in values)
    if validate:
        for record in records:
            record.validate()
    return records


@dataclass(frozen=True)
class BodyFeatureSpec:
    """Training-split-only standardization for optional body conditioning."""

    names: tuple[str, ...] = ()
    means: tuple[float, ...] = ()
    standard_deviations: tuple[float, ...] = ()

    @property
    def feature_dim(self) -> int:
        # A presence bit for every measurement prevents missing=mean ambiguity.
        return 2 * len(self.names)

    @classmethod
    def fit(
        cls,
        records: Iterable[TShirtTraceRecord],
        *,
        names: Sequence[str] | None = None,
    ) -> "BodyFeatureSpec":
        values = tuple(records)
        selected = tuple(names) if names is not None else tuple(sorted({key for record in values for key in record.body}))
        means: list[float] = []
        deviations: list[float] = []
        for name in selected:
            observed = np.asarray(
                [float(record.body[name]) for record in values if name in record.body and math.isfinite(float(record.body[name]))],
                dtype=np.float64,
            )
            if not len(observed):
                means.append(0.0)
                deviations.append(1.0)
            else:
                means.append(float(observed.mean()))
                standard_deviation = float(observed.std())
                deviations.append(standard_deviation if standard_deviation > 1e-6 else 1.0)
        return cls(selected, tuple(means), tuple(deviations))

    def encode(self, body: Mapping[str, float]) -> np.ndarray:
        if not self.names:
            return np.zeros((0,), dtype=np.float32)
        standardized = np.zeros(len(self.names), dtype=np.float32)
        present = np.zeros(len(self.names), dtype=np.float32)
        for index, (name, mean, deviation) in enumerate(zip(self.names, self.means, self.standard_deviations)):
            raw = body.get(name)
            if raw is not None and math.isfinite(float(raw)):
                standardized[index] = float(np.clip((float(raw) - mean) / deviation, -6.0, 6.0))
                present[index] = 1.0
        return np.concatenate((standardized, present)).astype(np.float32, copy=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "names": list(self.names),
            "means": list(self.means),
            "standard_deviations": list(self.standard_deviations),
            "feature_dim": self.feature_dim,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BodyFeatureSpec":
        return cls(
            names=tuple(str(item) for item in value.get("names", ())),
            means=tuple(float(item) for item in value.get("means", ())),
            standard_deviations=tuple(float(item) for item in value.get("standard_deviations", ())),
        )


@dataclass(frozen=True)
class PanelExample:
    sample_id: str
    split: str
    source: str
    panel_id: str
    features: np.ndarray
    edge_targets: np.ndarray
    edge_lengths_cm: np.ndarray
    edge_ids: tuple[str, ...]
    panel_target: int
    landmark_exists: np.ndarray
    landmark_xy_normalized: np.ndarray
    landmark_coordinate_mask: np.ndarray
    normalization_center_cm: np.ndarray
    normalization_scale_cm: float
    body_features: np.ndarray
    dart_applicability: str


def _curve_kind(kind: str) -> str:
    normalized = _slug(kind)
    aliases = {
        "quadratic": "quadratic_bezier",
        "cubic": "cubic_bezier",
        "curve": "bezier",
        "curveedge": "bezier",
        "circle": "arc",
        "circular_arc": "arc",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in CURVE_KINDS else "other"


def _bezier_points(control_polygon: np.ndarray, samples: int = 33) -> np.ndarray:
    output = []
    for t in np.linspace(0.0, 1.0, samples, dtype=np.float64):
        current = control_polygon.astype(np.float64, copy=True)
        for level in range(1, len(current)):
            current[: len(current) - level] = (
                (1.0 - t) * current[: len(current) - level] + t * current[1 : len(current) - level + 1]
            )
        output.append(current[0].copy())
    return np.asarray(output, dtype=np.float64)


def _arc_sweep_radians(geometry: CurveGeometry) -> float:
    """Return the directed sweep encoded by start/end angles and winding."""

    if geometry.start_angle_degrees is None or geometry.end_angle_degrees is None:
        raise ValueError("arc sweep requires start and end angles")
    raw = math.radians(float(geometry.end_angle_degrees) - float(geometry.start_angle_degrees))
    if geometry.clockwise is True and raw > 0.0:
        raw -= 2.0 * math.pi
    elif geometry.clockwise is False and raw < 0.0:
        raw += 2.0 * math.pi
    return raw


def edge_length_cm(edge: TracedEdge) -> float:
    """Numerically approximate line, Bezier, or arc length in trace units (cm)."""

    geometry = edge.geometry
    start = np.asarray(geometry.start_cm, dtype=np.float64)
    end = np.asarray(geometry.end_cm, dtype=np.float64)
    kind = _curve_kind(geometry.kind)
    if kind == "line":
        return float(np.linalg.norm(end - start))
    if kind in {"quadratic_bezier", "cubic_bezier", "bezier"} and geometry.control_points_cm:
        polygon = np.asarray((geometry.start_cm, *geometry.control_points_cm, geometry.end_cm), dtype=np.float64)
        points = _bezier_points(polygon)
        return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    if kind == "arc" and geometry.radius_cm is not None:
        if geometry.start_angle_degrees is not None and geometry.end_angle_degrees is not None:
            return abs(_arc_sweep_radians(geometry)) * float(geometry.radius_cm)
        if geometry.center_cm is not None:
            center = np.asarray(geometry.center_cm, dtype=np.float64)
            first = start - center
            second = end - center
            cosine = float(np.dot(first, second) / max(np.linalg.norm(first) * np.linalg.norm(second), 1e-12))
            return math.acos(float(np.clip(cosine, -1.0, 1.0))) * float(geometry.radius_cm)
    if geometry.control_points_cm:
        points = np.asarray((geometry.start_cm, *geometry.control_points_cm, geometry.end_cm), dtype=np.float64)
        return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    return float(np.linalg.norm(end - start))


def _panel_normalization(panel: TracedPanel) -> tuple[np.ndarray, float]:
    sampled: list[np.ndarray] = []
    for edge in panel.edges:
        geometry = edge.geometry
        kind = _curve_kind(geometry.kind)
        if kind in {"quadratic_bezier", "cubic_bezier", "bezier"} and geometry.control_points_cm:
            sampled.append(
                _bezier_points(
                    np.asarray(
                        (geometry.start_cm, *geometry.control_points_cm, geometry.end_cm), dtype=np.float64
                    )
                )
            )
        elif (
            kind == "arc"
            and geometry.center_cm is not None
            and geometry.radius_cm is not None
            and geometry.start_angle_degrees is not None
            and geometry.end_angle_degrees is not None
        ):
            start = math.radians(float(geometry.start_angle_degrees))
            sampled_angles = start + np.linspace(0.0, _arc_sweep_radians(geometry), 33, dtype=np.float64)
            center = np.asarray(geometry.center_cm, dtype=np.float64)
            sampled.append(
                center[None, :]
                + float(geometry.radius_cm)
                * np.stack((np.cos(sampled_angles), np.sin(sampled_angles)), axis=1)
            )
        else:
            sampled.append(np.asarray((geometry.start_cm, geometry.end_cm), dtype=np.float64))
    if sampled:
        values = np.concatenate(sampled, axis=0)
    else:
        values = np.asarray([point.xy_cm for point in panel.points], dtype=np.float64)
    minimum = values.min(axis=0)
    maximum = values.max(axis=0)
    center = (minimum + maximum) / 2.0
    scale = float(max(*(maximum - minimum), 1e-6))
    return center.astype(np.float32), scale


def _normalized_xy(value: Sequence[float], center: np.ndarray, scale: float) -> np.ndarray:
    return (np.asarray(value, dtype=np.float32) - center) / float(scale)


def panel_geometry_features(panel: TracedPanel) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Encode geometry only; labels and serialization phase never enter features."""

    center, scale = _panel_normalization(panel)
    features = np.zeros((len(panel.edges), EDGE_FEATURE_DIM), dtype=np.float32)
    lengths = np.zeros(len(panel.edges), dtype=np.float32)
    for index, edge in enumerate(panel.edges):
        geometry = edge.geometry
        start = _normalized_xy(geometry.start_cm, center, scale)
        end = _normalized_xy(geometry.end_cm, center, scale)
        delta = end - start
        chord = float(np.linalg.norm(delta))
        angle = math.atan2(float(delta[1]), float(delta[0]))
        controls = [_normalized_xy(point, center, scale) for point in geometry.control_points_cm[:2]]
        arc_center = (
            _normalized_xy(geometry.center_cm, center, scale)
            if geometry.center_cm is not None
            else np.zeros(2, dtype=np.float32)
        )
        kind = _curve_kind(geometry.kind)
        length = edge_length_cm(edge)
        lengths[index] = length
        row = features[index]
        row[FEATURE_SLICES["start"]] = start
        row[FEATURE_SLICES["end"]] = end
        row[FEATURE_SLICES["delta"]] = delta
        row[FEATURE_SLICES["length"]] = length / scale
        row[FEATURE_SLICES["chord"]] = chord
        row[FEATURE_SLICES["direction_sin"]] = math.sin(angle)
        row[FEATURE_SLICES["direction_cos"]] = math.cos(angle)
        if controls:
            row[FEATURE_SLICES["control_1"]] = controls[0]
        if len(controls) > 1:
            row[FEATURE_SLICES["control_2"]] = controls[1]
        row[FEATURE_SLICES["control_count"]] = min(len(geometry.control_points_cm), 2) / 2.0
        row[FEATURE_SLICES["arc_center"]] = arc_center
        row[FEATURE_SLICES["arc_radius"]] = (
            float(geometry.radius_cm) / scale if geometry.radius_cm is not None else 0.0
        )
        row[FEATURE_SLICES["arc_direction"]] = (
            -1.0 if geometry.clockwise is True else (1.0 if geometry.clockwise is False else 0.0)
        )
        row[FEATURE_SLICES["curve_kind"].start + CURVE_KINDS.index(kind)] = 1.0
    return features, lengths, center, scale


def panel_example(
    record: TShirtTraceRecord,
    panel: TracedPanel,
    *,
    body_spec: BodyFeatureSpec | None = None,
) -> PanelExample:
    features, lengths, center, scale = panel_geometry_features(panel)
    edge_targets = np.asarray(
        [EDGE_ROLES.index(canonical_edge_role(edge.semantic_role)) if edge.training_eligible else -100 for edge in panel.edges],
        dtype=np.int64,
    )
    exists = np.zeros(len(LANDMARK_NAMES), dtype=np.float32)
    coordinates = np.zeros((len(LANDMARK_NAMES), 2), dtype=np.float32)
    coordinate_mask = np.zeros(len(LANDMARK_NAMES), dtype=bool)
    candidates: dict[str, list[Any]] = {name: [] for name in LANDMARK_NAMES}
    for point in panel.points:
        name = str(point.canonical_name or "").upper()
        if name in candidates and point.training_eligible:
            candidates[name].append(point)
    for index, name in enumerate(LANDMARK_NAMES):
        if candidates[name]:
            # Duplicates are unusual but creation traces carry confidence; use
            # the best evidence deterministically and report duplicates upstream.
            point = sorted(candidates[name], key=lambda item: (-float(item.confidence), item.id))[0]
            exists[index] = 1.0
            coordinates[index] = _normalized_xy(point.xy_cm, center, scale)
            coordinate_mask[index] = True
    applicable_dart = any(bool(dart.applicable) for dart in record.darts)
    applicability = str(
        record.metadata.get(
            "dart_applicability",
            "APPLICABLE" if applicable_dart else "NOT_APPLICABLE",
        )
    )
    return PanelExample(
        sample_id=record.sample_id,
        split=record.split,
        source=source_value(record),
        panel_id=panel.id,
        features=features,
        edge_targets=edge_targets,
        edge_lengths_cm=lengths,
        edge_ids=tuple(edge.id for edge in panel.edges),
        panel_target=PANEL_ROLES.index(canonical_panel_role(panel.semantic_role)),
        landmark_exists=exists,
        landmark_xy_normalized=coordinates,
        landmark_coordinate_mask=coordinate_mask,
        normalization_center_cm=center,
        normalization_scale_cm=scale,
        body_features=(body_spec or BodyFeatureSpec()).encode(record.body),
        dart_applicability=applicability,
    )


def panel_examples(
    records: Iterable[TShirtTraceRecord],
    *,
    splits: set[str] | None = None,
    sources: set[str] | None = None,
    body_spec: BodyFeatureSpec | None = None,
) -> tuple[PanelExample, ...]:
    output: list[PanelExample] = []
    for record in records:
        record_source = source_value(record)
        if splits is not None and record.split not in splits:
            continue
        if sources is not None and record_source not in sources:
            continue
        for panel in record.panels:
            if panel.edges:
                output.append(panel_example(record, panel, body_spec=body_spec))
    return tuple(output)


@dataclass(frozen=True)
class BoundaryAugmentation:
    shift: int = 0
    reverse: bool = False
    rotation_degrees: float = 0.0
    scale: float = 1.0


def augment_panel_example(example: PanelExample, augmentation: BoundaryAugmentation) -> PanelExample:
    """Apply label-preserving boundary and coordinate transformations."""

    count = len(example.features)
    if count == 0:
        return example
    features = example.features.copy()
    targets = example.edge_targets.copy()
    lengths = example.edge_lengths_cm.copy()
    edge_ids = np.asarray(example.edge_ids, dtype=object)

    if augmentation.reverse:
        order = np.arange(count - 1, -1, -1)
        features = features[order]
        targets = targets[order]
        lengths = lengths[order]
        edge_ids = edge_ids[order]
        start = features[:, FEATURE_SLICES["start"]].copy()
        features[:, FEATURE_SLICES["start"]] = features[:, FEATURE_SLICES["end"]]
        features[:, FEATURE_SLICES["end"]] = start
        features[:, FEATURE_SLICES["delta"]] *= -1.0
        features[:, FEATURE_SLICES["direction_sin"]] *= -1.0
        features[:, FEATURE_SLICES["direction_cos"]] *= -1.0
        two_controls = features[:, FEATURE_SLICES["control_count"]] > 0.75
        first = features[two_controls, FEATURE_SLICES["control_1"]].copy()
        features[two_controls, FEATURE_SLICES["control_1"]] = features[two_controls, FEATURE_SLICES["control_2"]]
        features[two_controls, FEATURE_SLICES["control_2"]] = first
        features[:, FEATURE_SLICES["arc_direction"]] *= -1.0

    shift = int(augmentation.shift) % count
    if shift:
        features = np.roll(features, -shift, axis=0)
        targets = np.roll(targets, -shift, axis=0)
        lengths = np.roll(lengths, -shift, axis=0)
        edge_ids = np.roll(edge_ids, -shift, axis=0)

    theta = math.radians(float(augmentation.rotation_degrees))
    cosine, sine = math.cos(theta), math.sin(theta)
    rotation = np.asarray(((cosine, -sine), (sine, cosine)), dtype=np.float32)
    factor = float(augmentation.scale)
    if not math.isfinite(factor) or factor <= 0.0:
        raise ValueError("augmentation scale must be finite and positive")
    for key in ("start", "end", "delta", "control_1", "control_2", "arc_center"):
        section = FEATURE_SLICES[key]
        features[:, section] = (features[:, section] @ rotation.T) * factor
    features[:, FEATURE_SLICES["length"]] *= factor
    features[:, FEATURE_SLICES["chord"]] *= factor
    features[:, FEATURE_SLICES["arc_radius"]] *= factor
    direction = np.stack(
        (features[:, FEATURE_SLICES["direction_cos"]], features[:, FEATURE_SLICES["direction_sin"]]), axis=1
    )
    direction = direction @ rotation.T
    features[:, FEATURE_SLICES["direction_cos"]] = direction[:, 0]
    features[:, FEATURE_SLICES["direction_sin"]] = direction[:, 1]
    landmark_xy = (example.landmark_xy_normalized @ rotation.T) * factor
    return replace(
        example,
        features=features,
        edge_targets=targets,
        edge_lengths_cm=lengths,
        edge_ids=tuple(str(item) for item in edge_ids.tolist()),
        landmark_xy_normalized=landmark_xy.astype(np.float32, copy=False),
    )


def random_augmentation(example: PanelExample, generator: np.random.Generator, config: Mapping[str, Any]) -> PanelExample:
    count = max(len(example.features), 1)
    minimum_scale = float(config.get("minimum_scale", 0.8))
    maximum_scale = float(config.get("maximum_scale", 1.2))
    if minimum_scale <= 0.0 or maximum_scale < minimum_scale:
        raise ValueError("invalid augmentation scale range")
    augmentation = BoundaryAugmentation(
        shift=int(generator.integers(count)) if bool(config.get("cyclic_shift", True)) else 0,
        reverse=bool(generator.random() < float(config.get("reverse_probability", 0.5))),
        rotation_degrees=float(generator.uniform(-float(config.get("rotation_degrees", 180.0)), float(config.get("rotation_degrees", 180.0)))),
        scale=float(generator.uniform(minimum_scale, maximum_scale)),
    )
    return augment_panel_example(example, augmentation)


def padded_batch(examples: Sequence[PanelExample], maximum_edges: int) -> dict[str, Any]:
    oversized = [(example.panel_id, len(example.features)) for example in examples if len(example.features) > maximum_edges]
    if oversized:
        raise ValueError(
            f"maximum_edges={maximum_edges} would truncate panel geometry: {oversized}"
        )
    batch_size = len(examples)
    body_dim = max((len(example.body_features) for example in examples), default=0)
    features = np.zeros((batch_size, maximum_edges, EDGE_FEATURE_DIM), dtype=np.float32)
    edge_targets = np.full((batch_size, maximum_edges), -100, dtype=np.int64)
    edge_lengths = np.zeros((batch_size, maximum_edges), dtype=np.float32)
    valid = np.zeros((batch_size, maximum_edges), dtype=bool)
    panels = np.zeros(batch_size, dtype=np.int64)
    landmark_exists = np.zeros((batch_size, len(LANDMARK_NAMES)), dtype=np.float32)
    landmark_xy = np.zeros((batch_size, len(LANDMARK_NAMES), 2), dtype=np.float32)
    landmark_mask = np.zeros((batch_size, len(LANDMARK_NAMES)), dtype=bool)
    scales = np.zeros(batch_size, dtype=np.float32)
    body = np.zeros((batch_size, body_dim), dtype=np.float32)
    for row, example in enumerate(examples):
        count = len(example.features)
        features[row, :count] = example.features[:count]
        edge_targets[row, :count] = example.edge_targets[:count]
        edge_lengths[row, :count] = example.edge_lengths_cm[:count]
        valid[row, :count] = True
        panels[row] = example.panel_target
        landmark_exists[row] = example.landmark_exists
        landmark_xy[row] = example.landmark_xy_normalized
        landmark_mask[row] = example.landmark_coordinate_mask
        scales[row] = example.normalization_scale_cm
        if len(example.body_features):
            body[row, : len(example.body_features)] = example.body_features
    return {
        "features": features,
        "edge_targets": edge_targets,
        "edge_lengths_cm": edge_lengths,
        "valid_mask": valid,
        "panel_targets": panels,
        "landmark_exists": landmark_exists,
        "landmark_xy_normalized": landmark_xy,
        "landmark_coordinate_mask": landmark_mask,
        "normalization_scale_cm": scales,
        "body_features": body,
        "sample_ids": tuple(example.sample_id for example in examples),
        "splits": tuple(example.split for example in examples),
        "sources": tuple(example.source for example in examples),
    }


def decode_structural_semantics(
    features: np.ndarray,
    edge_role_ids: Sequence[int],
    *,
    tolerance: float = 1e-4,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Derive panel role and canonical junctions from predicted edge roles.

    A front/back/sleeve label is redundant once center-front, center-back, or
    sleeve-head edges are known. Likewise FNP/BNP/SNP/SP are topological
    junctions between semantic paths, so a separate coordinate regressor should
    not override an exact shared endpoint. The decoder supports a semantic path
    split across multiple curve segments.
    """

    if len(features) != len(edge_role_ids):
        raise ValueError("edge_role_ids must have one item per feature row")
    roles = tuple(EDGE_ROLES[int(value)] for value in edge_role_ids)
    role_set = set(roles)
    if "center_front" in role_set:
        panel_role = PANEL_ROLES.index("front")
    elif "center_back" in role_set:
        panel_role = PANEL_ROLES.index("back")
    elif role_set & {"sleeve_head", "sleeve_hem", "sleeve_underarm"}:
        panel_role = PANEL_ROLES.index("sleeve")
    elif "neckband_inner" in role_set or "neckband_outer" in role_set:
        panel_role = PANEL_ROLES.index("neckband")
    else:
        panel_role = PANEL_ROLES.index("other")

    endpoints = [
        (
            np.asarray(row[FEATURE_SLICES["start"]], dtype=np.float32),
            np.asarray(row[FEATURE_SLICES["end"]], dtype=np.float32),
        )
        for row in features
    ]

    def shared(first_role: str, second_role: str) -> np.ndarray | None:
        candidates: list[tuple[float, np.ndarray]] = []
        for first_index, first in enumerate(roles):
            if first != first_role:
                continue
            for second_index, second in enumerate(roles):
                if second != second_role:
                    continue
                for first_point in endpoints[first_index]:
                    for second_point in endpoints[second_index]:
                        distance = float(np.linalg.norm(first_point - second_point))
                        candidates.append((distance, (first_point + second_point) / 2.0))
        if not candidates:
            return None
        distance, point = min(candidates, key=lambda item: item[0])
        return point if distance <= tolerance else None

    requests = {
        "FNP": ("neckline", "center_front"),
        "BNP": ("neckline", "center_back"),
        "SNP": ("neckline", "shoulder"),
        "SP": ("shoulder", "armhole"),
    }
    exists = np.zeros(len(LANDMARK_NAMES), dtype=np.float32)
    coordinates = np.zeros((len(LANDMARK_NAMES), 2), dtype=np.float32)
    for index, name in enumerate(LANDMARK_NAMES):
        point = shared(*requests[name])
        if point is not None:
            exists[index] = 1.0
            coordinates[index] = point
    return panel_role, exists, coordinates


def balanced_edge_weights(examples: Iterable[PanelExample]) -> np.ndarray:
    counts = np.ones(len(EDGE_ROLES), dtype=np.float64)
    for example in examples:
        valid = example.edge_targets >= 0
        counts += np.bincount(example.edge_targets[valid], minlength=len(EDGE_ROLES))
    weights = 1.0 / np.sqrt(counts)
    weights /= weights.mean()
    return weights.astype(np.float32)


def build_tshirt_model(config: Mapping[str, Any], *, body_feature_dim: int = 0):
    """Build a GPU-friendly set Transformer with four supervised heads."""

    import torch

    if str(config.get("mode", "pattern-only")) not in {"pattern-only", "pattern+body"}:
        raise ValueError("mode must be pattern-only or pattern+body")
    if str(config.get("mode", "pattern-only")) == "pattern-only" and body_feature_dim:
        raise ValueError("pattern-only mode must not receive body features")

    class TShirtSemanticNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            width = int(config["width"])
            self.feature_projection = torch.nn.Linear(EDGE_FEATURE_DIM, width)
            self.class_token = torch.nn.Parameter(torch.zeros(1, 1, width))
            self.body_projection = torch.nn.Linear(body_feature_dim, width) if body_feature_dim else None
            layer = torch.nn.TransformerEncoderLayer(
                d_model=width,
                nhead=int(config["heads"]),
                dim_feedforward=width * int(config["feedforward_multiplier"]),
                dropout=float(config["dropout"]),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = torch.nn.TransformerEncoder(
                layer,
                num_layers=int(config["layers"]),
                enable_nested_tensor=False,
            )
            self.edge_head = torch.nn.Sequential(
                torch.nn.LayerNorm(width), torch.nn.Linear(width, width), torch.nn.GELU(), torch.nn.Linear(width, len(EDGE_ROLES))
            )
            self.panel_head = torch.nn.Sequential(torch.nn.LayerNorm(width), torch.nn.Linear(width, len(PANEL_ROLES)))
            self.landmark_existence_head = torch.nn.Sequential(
                torch.nn.LayerNorm(width), torch.nn.Linear(width, len(LANDMARK_NAMES))
            )
            self.landmark_coordinate_head = torch.nn.Sequential(
                torch.nn.LayerNorm(width), torch.nn.Linear(width, width), torch.nn.GELU(), torch.nn.Linear(width, 2 * len(LANDMARK_NAMES))
            )
            torch.nn.init.normal_(self.class_token, std=0.02)

        def forward(self, features, valid_mask, body_features=None):
            edge_hidden = self.feature_projection(features)
            batch_size = edge_hidden.shape[0]
            class_hidden = self.class_token.expand(batch_size, -1, -1)
            if self.body_projection is not None:
                if body_features is None:
                    raise ValueError("pattern+body model requires body_features")
                conditioning = self.body_projection(body_features)[:, None, :]
                class_hidden = class_hidden + conditioning
                edge_hidden = edge_hidden + conditioning
            hidden = torch.cat((class_hidden, edge_hidden), dim=1)
            class_valid = torch.ones((batch_size, 1), dtype=torch.bool, device=valid_mask.device)
            encoder_mask = torch.cat((class_valid, valid_mask), dim=1)
            hidden = self.encoder(hidden, src_key_padding_mask=~encoder_mask)
            pooled = hidden[:, 0]
            return {
                "edge_logits": self.edge_head(hidden[:, 1:]),
                "panel_logits": self.panel_head(pooled),
                "landmark_existence_logits": self.landmark_existence_head(pooled),
                "landmark_xy_normalized": self.landmark_coordinate_head(pooled).reshape(
                    batch_size, len(LANDMARK_NAMES), 2
                ),
            }

    return TShirtSemanticNet()


def _classification_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    names: Sequence[str],
) -> dict[str, Any]:
    valid = targets >= 0
    predictions = predictions[valid]
    targets = targets[valid]
    weights = weights[valid]
    per_role: dict[str, Any] = {}
    semantic_f1: list[float] = []
    all_f1: list[float] = []
    length_semantic_f1: list[float] = []
    length_all_f1: list[float] = []
    for index, name in enumerate(names):
        true_positive = int(np.sum((predictions == index) & (targets == index)))
        false_positive = int(np.sum((predictions == index) & (targets != index)))
        false_negative = int(np.sum((predictions != index) & (targets == index)))
        support = int(np.sum(targets == index))
        support_length = float(weights[targets == index].sum())
        true_positive_length = float(weights[(predictions == index) & (targets == index)].sum())
        false_positive_length = float(weights[(predictions == index) & (targets != index)].sum())
        false_negative_length = float(weights[(predictions != index) & (targets == index)].sum())
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        length_precision = true_positive_length / max(true_positive_length + false_positive_length, 1e-12)
        length_recall = true_positive_length / max(true_positive_length + false_negative_length, 1e-12)
        length_f1 = 2.0 * length_precision * length_recall / max(length_precision + length_recall, 1e-12)
        per_role[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support_edges": support,
            "support_length_cm": support_length,
            "length_weighted_true_positive_cm": true_positive_length,
            "length_weighted_false_positive_cm": false_positive_length,
            "length_weighted_false_negative_cm": false_negative_length,
            "length_weighted_precision": length_precision,
            "length_weighted_recall": length_recall,
            "length_weighted_f1": length_f1,
        }
        if support:
            all_f1.append(f1)
            length_all_f1.append(length_f1)
            if name != "other":
                semantic_f1.append(f1)
                length_semantic_f1.append(length_f1)
    supported = [
        (per_role[name]["length_weighted_f1"], per_role[name]["support_length_cm"])
        for name in names
        if per_role[name]["support_edges"]
    ]
    length_sum = sum(item[1] for item in supported)
    return {
        "edge_count": int(len(targets)),
        "accuracy": float(np.mean(predictions == targets)) if len(targets) else 0.0,
        "length_weighted_accuracy": (
            float(weights[predictions == targets].sum() / max(weights.sum(), 1e-12)) if len(targets) else 0.0
        ),
        "macro_f1_supported_semantics": float(np.mean(semantic_f1)) if semantic_f1 else 0.0,
        "macro_f1_all_supported": float(np.mean(all_f1)) if all_f1 else 0.0,
        "length_weighted_macro_f1_supported_semantics": (
            float(np.mean(length_semantic_f1)) if length_semantic_f1 else 0.0
        ),
        "length_weighted_macro_f1_all_supported": float(np.mean(length_all_f1)) if length_all_f1 else 0.0,
        "length_weighted_f1": (
            float(sum(f1 * support_length for f1, support_length in supported) / max(length_sum, 1e-12))
            if supported
            else 0.0
        ),
        "length_weighted_f1_aggregation": (
            "ground-truth-length-support-weighted mean of per-role F1 values whose TP/FP/FN are each weighted by edge length"
        ),
        "per_role": per_role,
    }


def _landmark_metrics(
    existence_probabilities: np.ndarray,
    coordinate_predictions: np.ndarray,
    existence_targets: np.ndarray,
    coordinate_targets: np.ndarray,
    coordinate_masks: np.ndarray,
    panel_scales_cm: np.ndarray,
) -> dict[str, Any]:
    binary = existence_probabilities >= 0.5
    truth = existence_targets >= 0.5
    per_landmark: dict[str, Any] = {}
    true_positive_total = false_positive_total = false_negative_total = 0
    errors_cm: list[float] = []
    normalized_errors: list[float] = []
    detection_aware_errors_cm: list[float] = []
    detected_flags: list[bool] = []
    for index, name in enumerate(LANDMARK_NAMES):
        true_positive = int(np.sum(binary[:, index] & truth[:, index]))
        false_positive = int(np.sum(binary[:, index] & ~truth[:, index]))
        false_negative = int(np.sum(~binary[:, index] & truth[:, index]))
        true_positive_total += true_positive
        false_positive_total += false_positive
        false_negative_total += false_negative
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        mask = coordinate_masks[:, index]
        normalized = np.linalg.norm(coordinate_predictions[mask, index] - coordinate_targets[mask, index], axis=1)
        errors = normalized * panel_scales_cm[mask]
        errors_cm.extend(errors.tolist())
        normalized_errors.extend(normalized.tolist())
        detected = binary[mask, index]
        detected_flags.extend(detected.tolist())
        # A missed landmark is assigned one panel span.  This keeps the unit
        # interpretable in centimetres and prevents a perfect coordinate head
        # from hiding a failed existence decision.
        detection_aware_errors_cm.extend(
            np.where(detected, errors, panel_scales_cm[mask]).astype(np.float64).tolist()
        )
        per_landmark[name] = {
            "existence_precision": precision,
            "existence_recall": recall,
            "existence_f1": f1,
            "positive_support": int(truth[:, index].sum()),
            "coordinate_target_count": int(mask.sum()),
            "gt_positive_conditional_mean_euclidean_error_cm": float(errors.mean()) if len(errors) else None,
            "detection_aware_success_pck_panel_span_2pct": (
                float(np.mean(detected & (normalized <= 0.02))) if len(normalized) else None
            ),
        }
    precision = true_positive_total / max(true_positive_total + false_positive_total, 1)
    recall = true_positive_total / max(true_positive_total + false_negative_total, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    errors_array = np.asarray(errors_cm, dtype=np.float64)
    normalized_array = np.asarray(normalized_errors, dtype=np.float64)
    detected_array = np.asarray(detected_flags, dtype=bool)
    detection_aware_errors_array = np.asarray(detection_aware_errors_cm, dtype=np.float64)
    return {
        "existence_micro_precision": precision,
        "existence_micro_recall": recall,
        "existence_micro_f1": f1,
        "gt_positive_conditional_location_target_count": int(len(errors_array)),
        "gt_positive_conditional_mean_euclidean_error_cm": float(errors_array.mean()) if len(errors_array) else None,
        "gt_positive_conditional_median_euclidean_error_cm": (
            float(np.median(errors_array)) if len(errors_array) else None
        ),
        "gt_positive_conditional_pck_panel_span_1pct": (
            float(np.mean(normalized_array <= 0.01)) if len(normalized_array) else None
        ),
        "gt_positive_conditional_pck_panel_span_2pct": (
            float(np.mean(normalized_array <= 0.02)) if len(normalized_array) else None
        ),
        "gt_positive_conditional_pck_panel_span_5pct": (
            float(np.mean(normalized_array <= 0.05)) if len(normalized_array) else None
        ),
        "detection_aware_success_pck_panel_span_1pct": (
            float(np.mean(detected_array & (normalized_array <= 0.01))) if len(normalized_array) else None
        ),
        "detection_aware_success_pck_panel_span_2pct": (
            float(np.mean(detected_array & (normalized_array <= 0.02))) if len(normalized_array) else None
        ),
        "detection_aware_success_pck_panel_span_5pct": (
            float(np.mean(detected_array & (normalized_array <= 0.05))) if len(normalized_array) else None
        ),
        "detection_aware_mean_error_cm_one_panel_span_miss_penalty": (
            float(detection_aware_errors_array.mean()) if len(detection_aware_errors_array) else None
        ),
        "location_metric_scope": (
            "gt_positive_conditional metrics ignore the existence decision; detection_aware metrics require a positive existence prediction"
        ),
        "per_landmark": per_landmark,
    }


def evaluate_model(model, examples: Sequence[PanelExample], config: Mapping[str, Any], device) -> dict[str, Any]:
    """Evaluate any collection, irrespective of its split label or source."""

    import torch

    if not examples:
        return {"panel_count": 0, "status": "NO_EXAMPLES"}
    model.eval()
    edge_predictions: list[np.ndarray] = []
    edge_targets: list[np.ndarray] = []
    edge_lengths: list[np.ndarray] = []
    panel_predictions: list[np.ndarray] = []
    panel_targets: list[np.ndarray] = []
    structural_panel_predictions: list[np.ndarray] = []
    structural_existence_predictions: list[np.ndarray] = []
    structural_coordinate_predictions: list[np.ndarray] = []
    existence_probabilities: list[np.ndarray] = []
    existence_targets: list[np.ndarray] = []
    coordinate_predictions: list[np.ndarray] = []
    coordinate_targets: list[np.ndarray] = []
    coordinate_masks: list[np.ndarray] = []
    scales: list[np.ndarray] = []
    dart_by_sample: dict[str, bool] = {example.sample_id: False for example in examples}
    maximum_edges = int(config["maximum_edges"])
    batch_size = int(config.get("batch_size", 96))
    dart_index = EDGE_ROLES.index("dart_leg")
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch_examples = examples[start : start + batch_size]
            batch = padded_batch(batch_examples, maximum_edges)
            tensors = {
                key: torch.from_numpy(batch[key]).to(device)
                for key in ("features", "valid_mask", "body_features")
            }
            outputs = model(tensors["features"], tensors["valid_mask"], tensors["body_features"])
            predicted_edges = outputs["edge_logits"].argmax(dim=-1).cpu().numpy()
            valid = batch["valid_mask"] & (batch["edge_targets"] >= 0)
            edge_predictions.append(predicted_edges[valid])
            edge_targets.append(batch["edge_targets"][valid])
            edge_lengths.append(batch["edge_lengths_cm"][valid])
            panel_predictions.append(outputs["panel_logits"].argmax(dim=-1).cpu().numpy())
            panel_targets.append(batch["panel_targets"])
            existence_probabilities.append(torch.sigmoid(outputs["landmark_existence_logits"]).cpu().numpy())
            existence_targets.append(batch["landmark_exists"])
            coordinate_predictions.append(outputs["landmark_xy_normalized"].cpu().numpy())
            coordinate_targets.append(batch["landmark_xy_normalized"])
            coordinate_masks.append(batch["landmark_coordinate_mask"])
            scales.append(batch["normalization_scale_cm"])
            for row, example in enumerate(batch_examples):
                count = min(len(example.edge_targets), maximum_edges)
                structural_panel, structural_exists, structural_xy = decode_structural_semantics(
                    example.features[:count], predicted_edges[row, :count]
                )
                structural_panel_predictions.append(np.asarray([structural_panel], dtype=np.int64))
                structural_existence_predictions.append(structural_exists[None, :])
                structural_coordinate_predictions.append(structural_xy[None, :, :])
                if np.any(predicted_edges[row, :count] == dart_index):
                    dart_by_sample[example.sample_id] = True
    predicted_panels = np.concatenate(panel_predictions)
    target_panels = np.concatenate(panel_targets)
    edge_result = _classification_metrics(
        np.concatenate(edge_predictions), np.concatenate(edge_targets), np.concatenate(edge_lengths), EDGE_ROLES
    )
    landmarks = _landmark_metrics(
        np.concatenate(existence_probabilities),
        np.concatenate(coordinate_predictions),
        np.concatenate(existence_targets),
        np.concatenate(coordinate_targets),
        np.concatenate(coordinate_masks),
        np.concatenate(scales),
    )
    structural_panels = np.concatenate(structural_panel_predictions)
    structural_landmarks = _landmark_metrics(
        np.concatenate(structural_existence_predictions),
        np.concatenate(structural_coordinate_predictions),
        np.concatenate(existence_targets),
        np.concatenate(coordinate_targets),
        np.concatenate(coordinate_masks),
        np.concatenate(scales),
    )
    false_positive_garments = sum(dart_by_sample.values())
    return {
        "status": "EVALUATED",
        "panel_count": len(examples),
        "garment_count": len(dart_by_sample),
        "edge_semantics": edge_result,
        "panel_role": {
            "accuracy": float(np.mean(predicted_panels == target_panels)),
            "count": int(len(target_panels)),
            "confusion": {
                truth_name: {
                    predicted_name: int(np.sum((target_panels == truth_index) & (predicted_panels == predicted_index)))
                    for predicted_index, predicted_name in enumerate(PANEL_ROLES)
                }
                for truth_index, truth_name in enumerate(PANEL_ROLES)
            },
        },
        "landmarks": landmarks,
        "structural_decoding": {
            "basis": "predicted semantic edge roles plus exact shared endpoints; no source-specific coordinates",
            "panel_role_accuracy": float(np.mean(structural_panels == target_panels)),
            "landmarks": structural_landmarks,
        },
        "dart_false_positive": {
            "ground_truth_applicability": "NOT_APPLICABLE_BASIC_TSHIRT",
            "detection_basis": "any edge predicted as dart_leg; there is no fabricated dart target or separate dart head",
            "false_positive_garment_count": false_positive_garments,
            "false_positive_garment_rate": false_positive_garments / max(len(dart_by_sample), 1),
        },
        "construction_dag": {
            "score": None,
            "status": "NOT_A_GENERALIZATION_METRIC",
            "reason": "A same-recipe operation DAG is fixed generator provenance; predicting it would measure recipe memorization.",
        },
    }


def evaluate_by_split_and_source(
    model,
    examples: Sequence[PanelExample],
    config: Mapping[str, Any],
    device,
) -> dict[str, Any]:
    """Report overall and arbitrary record-provided split/source groups."""

    result: dict[str, Any] = {"overall": evaluate_model(model, examples, config, device)}
    result["by_split"] = {
        split: evaluate_model(model, tuple(item for item in examples if item.split == split), config, device)
        for split in sorted({item.split for item in examples})
    }
    result["by_source"] = {
        source: evaluate_model(model, tuple(item for item in examples if item.source == source), config, device)
        for source in sorted({item.source for item in examples})
    }
    result["by_split_and_source"] = {
        f"{split}::{source}": evaluate_model(
            model,
            tuple(item for item in examples if item.split == split and item.source == source),
            config,
            device,
        )
        for split, source in sorted({(item.split, item.source) for item in examples})
    }
    return result


def dataset_audit(records: Sequence[TShirtTraceRecord], examples: Sequence[PanelExample]) -> dict[str, Any]:
    """Compact, path-free counts suitable for a tracked textual manifest."""

    duplicate_points = 0
    for record in records:
        for panel in record.panels:
            names = [str(point.canonical_name or "").upper() for point in panel.points if point.training_eligible]
            duplicate_points += sum(max(names.count(name) - 1, 0) for name in LANDMARK_NAMES)
    sample_splits: dict[str, set[str]] = {}
    for record in records:
        sample_splits.setdefault(record.sample_id, set()).add(record.split)
    return {
        "record_count": len(records),
        "panel_count": len(examples),
        "edge_count": int(sum(len(example.edge_targets) for example in examples)),
        "split_record_counts": {
            split: sum(record.split == split for record in records) for split in sorted({record.split for record in records})
        },
        "source_record_counts": {
            source: sum(source_value(record) == source for record in records)
            for source in sorted({source_value(record) for record in records})
        },
        "panel_role_counts": {
            role: sum(example.panel_target == index for example in examples) for index, role in enumerate(PANEL_ROLES)
        },
        "duplicate_training_landmark_candidates": duplicate_points,
        "sample_ids_present_in_multiple_splits": sum(len(splits) > 1 for splits in sample_splits.values()),
        "feature_contract": {
            "geometry_feature_dimension": EDGE_FEATURE_DIM,
            "includes_serialization_phase": False,
            "includes_positional_encoding": False,
            "includes_source_label": False,
            "includes_panel_or_edge_truth_as_input": False,
            "includes_trace_operation_or_formula": False,
        },
    }


__all__ = [
    "LANDMARK_NAMES",
    "PANEL_ROLES",
    "EDGE_ROLES",
    "CURVE_KINDS",
    "EDGE_FEATURE_DIM",
    "DEFAULT_TSHIRT_MODEL_CONFIG",
    "BodyFeatureSpec",
    "PanelExample",
    "BoundaryAugmentation",
    "canonical_panel_role",
    "canonical_edge_role",
    "decode_structural_semantics",
    "source_value",
    "deterministic_split",
    "read_tshirt_records",
    "edge_length_cm",
    "panel_geometry_features",
    "panel_example",
    "panel_examples",
    "augment_panel_example",
    "random_augmentation",
    "padded_batch",
    "balanced_edge_weights",
    "build_tshirt_model",
    "evaluate_model",
    "evaluate_by_split_and_source",
    "dataset_audit",
]
