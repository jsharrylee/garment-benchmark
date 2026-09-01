"""Semantically constrained residual edits for validated basic pattern anchors.

The editor is deliberately topology preserving.  A visual model predicts
named landmark and path residuals; this module applies those residuals to an
already valid category anchor.  It never invents panels, edges, stitches, or
unobserved production marks.

Anchors expose editable geometry through ``PatternDocument.annotations``::

    {
      "semantic_landmarks": {
        "front/FNP": [
          {"panel_id": "front", "edge_id": "neckline", "point_index": 0}
        ]
      },
      "semantic_paths": {
        "front/neckline": [
          {"panel_id": "front", "edge_ids": ["neckline"]}
        ]
      }
    }

Landmark moves use a compact smooth influence inside the owning panel.  Path
edits operate in the path's chord-local frame, which makes length/depth
residuals independent of panel rotation.  Panel-loop endpoints are reconciled
after every edit so the operation cannot introduce an open boundary merely
through duplicate endpoint storage.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .schema import Edge, Panel, PatternDocument


@dataclass(frozen=True)
class LandmarkResidual:
    """A named 2D displacement in anchor centimetres."""

    dx_cm: float
    dy_cm: float
    influence_radius_cm: float = 8.0
    confidence: float = 1.0

    def validate(self) -> None:
        values = (self.dx_cm, self.dy_cm, self.influence_radius_cm, self.confidence)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("landmark residual values must be finite")
        if self.influence_radius_cm <= 0.0:
            raise ValueError("landmark influence radius must be positive")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("landmark confidence must be in [0, 1]")


@dataclass(frozen=True)
class PathResidual:
    """A topology-preserving edit in a semantic path's chord-local frame."""

    chord_scale: float = 1.0
    normal_scale: float = 1.0
    normal_offset_cm: float = 0.0
    confidence: float = 1.0

    def validate(self) -> None:
        values = (self.chord_scale, self.normal_scale, self.normal_offset_cm, self.confidence)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("path residual values must be finite")
        if not 0.5 <= self.chord_scale <= 1.5:
            raise ValueError("path chord scale must be in [0.5, 1.5]")
        if not 0.25 <= self.normal_scale <= 2.0:
            raise ValueError("path normal scale must be in [0.25, 2.0]")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("path confidence must be in [0, 1]")


@dataclass(frozen=True)
class SemanticResidualPlan:
    category: str
    landmark_residuals: Mapping[str, LandmarkResidual] = field(default_factory=dict)
    path_residuals: Mapping[str, PathResidual] = field(default_factory=dict)
    gated_queries: Mapping[str, str] = field(default_factory=dict)
    source: str = "four_view_semantic_student"
    schema_version: str = "semantic-pattern-residual/v1"

    def validate(self) -> None:
        if self.category not in {"tshirt", "pants", "skirt"}:
            raise ValueError(f"unsupported semantic edit category: {self.category!r}")
        for residual in self.landmark_residuals.values():
            residual.validate()
        for residual in self.path_residuals.values():
            residual.validate()
        for name, reason in self.gated_queries.items():
            if not str(name).strip() or not str(reason).strip():
                raise ValueError("gated semantic queries require non-empty names and reasons")


_ROLE_SPECIFIC_DART_LANDMARKS = (
    "front_dart_apex",
    "back_dart_apex",
    "front_dart_leg_left",
    "front_dart_leg_right",
    "back_dart_leg_left",
    "back_dart_leg_right",
)


def _raw_annotation_entries(
    document: PatternDocument, group: str, name: str
) -> list[dict[str, Any]]:
    raw_group = document.annotations.get(group, {})
    if not isinstance(raw_group, Mapping):
        raise ValueError(f"annotation group {group!r} must be a mapping")
    raw = raw_group.get(name, ())
    if isinstance(raw, Mapping):
        raw = (raw,)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError(f"semantic annotation {group}/{name} must be a sequence")
    return [dict(item) for item in raw]


def _role_specific_dart_path_entries(
    document: PatternDocument, name: str
) -> list[dict[str, Any]]:
    if name not in {"front_dart_leg", "back_dart_leg"}:
        return []
    prefix = name.split("_", 1)[0]
    legacy = _raw_annotation_entries(document, "semantic_paths", "dart_leg")
    return [
        entry
        for entry in legacy
        if prefix in str(entry.get("panel_id", "")).lower()
    ]


def _role_specific_dart_landmark_entries(
    document: PatternDocument, name: str
) -> list[dict[str, Any]]:
    if name not in _ROLE_SPECIFIC_DART_LANDMARKS:
        return []
    prefix = name.split("_", 1)[0]
    path_entries = semantic_annotation_entries(
        document, "path", f"{prefix}_dart_leg"
    )
    if len(path_entries) != 1:
        return []
    path_entry = path_entries[0]
    panel_id = str(path_entry["panel_id"])
    panels = {panel.id: panel for panel in document.panels}
    if panel_id not in panels:
        raise ValueError(f"dart annotation references unknown panel {panel_id!r}")
    edges = {edge.id: edge for edge in panels[panel_id].edges}
    occurrences: list[tuple[np.ndarray, str, int]] = []
    for raw_edge_id in path_entry.get("edge_ids", ()):
        edge_id = str(raw_edge_id)
        if edge_id not in edges:
            raise ValueError(f"dart annotation references unknown edge {panel_id}/{edge_id}")
        points = edges[edge_id].points
        if len(points) < 2:
            raise ValueError(f"dart edge has fewer than two points: {panel_id}/{edge_id}")
        occurrences.extend(
            (
                np.asarray(points[index], dtype=np.float64),
                edge_id,
                index,
            )
            for index in (0, len(points) - 1)
        )
    groups: list[list[tuple[np.ndarray, str, int]]] = []
    for occurrence in occurrences:
        for group in groups:
            if float(np.linalg.norm(occurrence[0] - group[0][0])) <= 1e-6:
                group.append(occurrence)
                break
        else:
            groups.append([occurrence])
    shared = [group for group in groups if len(group) > 1]
    terminals = [group for group in groups if len(group) == 1]
    if len(shared) != 1 or len(terminals) < 2:
        raise ValueError(f"dart topology is not a two-leg V: {panel_id}")
    if name.endswith("_dart_apex"):
        selected = shared[0][0]
    else:
        terminals.sort(key=lambda group: (float(group[0][0][0]), float(group[0][0][1])))
        selected = terminals[0][0] if name.endswith("_left") else terminals[-1][0]
    return [
        {
            "panel_id": panel_id,
            "edge_id": selected[1],
            "point_index": selected[2],
            "source_landmark": name,
            "derived_from": f"{prefix}_dart_leg_topology",
        }
    ]


def semantic_annotation_entries(
    document: PatternDocument, kind: str, name: str
) -> list[dict[str, Any]]:
    """Resolve exact annotations plus safe front/back dart specialization.

    The source basic-block adapter historically exposed one ``dart_leg`` query
    with front and back entries.  A training query cannot safely broadcast one
    residual to both.  Role-specific query names are therefore resolved to one
    physical panel here while the ambiguous legacy name remains multi-instance
    and fail-closed.
    """

    groups = {
        "landmark": "semantic_landmarks",
        "path": "semantic_paths",
        "panel": "semantic_panels",
    }
    if kind not in groups:
        raise ValueError(f"unsupported semantic annotation kind: {kind!r}")
    exact = _raw_annotation_entries(document, groups[kind], name)
    if exact:
        return exact
    if kind == "path":
        return _role_specific_dart_path_entries(document, name)
    if kind == "landmark":
        return _role_specific_dart_landmark_entries(document, name)
    return []


def _annotation_entries(document: PatternDocument, group: str, name: str) -> list[dict[str, Any]]:
    kind = {
        "semantic_landmarks": "landmark",
        "semantic_paths": "path",
        "semantic_panels": "panel",
    }.get(group)
    if kind is None:
        raise ValueError(f"unsupported semantic annotation group: {group!r}")
    return semantic_annotation_entries(document, kind, name)


def _edge_index(panel: Panel) -> dict[str, int]:
    return {edge.id: index for index, edge in enumerate(panel.edges)}


def _landmark_xy(panel: Panel, entry: Mapping[str, Any]) -> np.ndarray:
    edge_id = str(entry["edge_id"])
    edges = {edge.id: edge for edge in panel.edges}
    if edge_id not in edges:
        raise ValueError(f"unknown landmark edge {panel.id}/{edge_id}")
    points = edges[edge_id].points
    index = int(entry.get("point_index", 0))
    if index < 0:
        index += len(points)
    if not 0 <= index < len(points):
        raise ValueError(f"landmark point index out of range: {panel.id}/{edge_id}/{index}")
    return np.asarray(points[index], dtype=np.float64)


def _smooth_weight(distance: np.ndarray, radius: float) -> np.ndarray:
    normalized = np.clip(distance / max(float(radius), 1e-8), 0.0, 1.0)
    # Compact cubic smoothstep: exactly zero outside the stated influence.
    return np.where(normalized < 1.0, 1.0 - normalized * normalized * (3.0 - 2.0 * normalized), 0.0)


def _apply_landmark_to_panel(
    panel: Panel,
    anchor: np.ndarray,
    residual: LandmarkResidual,
) -> Panel:
    displacement = np.asarray((residual.dx_cm, residual.dy_cm), dtype=np.float64) * float(residual.confidence)
    edges = []
    for edge in panel.edges:
        points = np.asarray(edge.points, dtype=np.float64)
        distances = np.linalg.norm(points - anchor[None, :], axis=1)
        weights = _smooth_weight(distances, residual.influence_radius_cm)[:, None]
        moved = points + weights * displacement[None, :]
        edges.append(replace(edge, points=tuple((float(x), float(y)) for x, y in moved)))
    return replace(panel, edges=tuple(edges))


@dataclass(frozen=True)
class _PathComponent:
    points: np.ndarray
    # Every dense point can occur in more than one edge at a shared vertex.
    occurrences: tuple[tuple[tuple[int, int], ...], ...]


def _endpoint_key(point: np.ndarray, bounds: tuple[float, float, float, float], y_down: bool) -> tuple[float, float]:
    min_x, min_y, max_x, max_y = bounds
    width = max(max_x - min_x, 1e-8)
    height = max(max_y - min_y, 1e-8)
    u = (float(point[0]) - min_x) / width
    raw_v = (float(point[1]) - min_y) / height
    return u, 1.0 - raw_v if y_down else raw_v


def _panel_bounds(panel: Panel) -> tuple[float, float, float, float]:
    values = np.asarray(
        [point for edge in panel.edges for point in edge.points], dtype=np.float64
    )
    if values.ndim != 2 or values.shape[1:] != (2,) or not len(values):
        raise ValueError(f"panel {panel.id!r} has no 2D path geometry")
    minimum, maximum = values.min(axis=0), values.max(axis=0)
    return float(minimum[0]), float(minimum[1]), float(maximum[0]), float(maximum[1])


def _ordered_path_components(
    panel: Panel,
    edge_ids: Sequence[str],
    *,
    source_y_axis_down: bool,
    tolerance: float = 1e-6,
) -> tuple[_PathComponent, ...]:
    """Return continuous components without joining dart-separated segments.

    Semantic paths such as a waistline can legitimately contain ``waist_inner``
    and ``waist_outer`` with two dart legs between them.  They are one semantic
    measurement but not one connected polyline.  Each component is therefore
    transformed independently.  Connected paths (for example the two sleeve
    cap halves) retain the shared point and its edge occurrences.
    """

    lookup = _edge_index(panel)
    bounds = _panel_bounds(panel)
    components: list[_PathComponent] = []
    current_points: list[np.ndarray] = []
    current_occurrences: list[list[tuple[int, int]]] = []

    def finish() -> None:
        nonlocal current_points, current_occurrences
        if not current_points:
            return
        points = np.asarray(current_points, dtype=np.float64)
        occurrences = tuple(tuple(values) for values in current_occurrences)
        if _endpoint_key(points[-1], bounds, source_y_axis_down) < _endpoint_key(
            points[0], bounds, source_y_axis_down
        ):
            points = points[::-1].copy()
            occurrences = tuple(reversed(occurrences))
        components.append(_PathComponent(points=points, occurrences=occurrences))
        current_points = []
        current_occurrences = []

    for raw_id in edge_ids:
        edge_id = str(raw_id)
        if edge_id not in lookup:
            raise ValueError(f"unknown semantic path edge {panel.id}/{edge_id}")
        edge_index = lookup[edge_id]
        values = np.asarray(panel.edges[edge_index].points, dtype=np.float64)
        if len(values) < 2:
            raise ValueError(f"semantic path edge has fewer than two points: {panel.id}/{edge_id}")
        indices = list(range(len(values)))
        if current_points:
            distance_start = float(np.linalg.norm(values[0] - current_points[-1]))
            distance_end = float(np.linalg.norm(values[-1] - current_points[-1]))
            closest = min(distance_start, distance_end)
            if closest > tolerance:
                finish()
            elif distance_end < distance_start:
                values = values[::-1]
                indices.reverse()
        if not current_points:
            current_points = [point.copy() for point in values]
            current_occurrences = [[(edge_index, index)] for index in indices]
            continue
        # The new edge is continuous with the preceding component.  Store its
        # first occurrence on the existing shared dense point.
        current_occurrences[-1].append((edge_index, indices[0]))
        for point, point_index in zip(values[1:], indices[1:]):
            current_points.append(point.copy())
            current_occurrences.append([(edge_index, point_index)])
    finish()
    return tuple(components)


def _transform_path(
    points: np.ndarray,
    residual: PathResidual,
    *,
    protected_points: Sequence[np.ndarray] = (),
    tolerance: float = 1e-6,
) -> np.ndarray:
    start, end = points[0], points[-1]
    chord = end - start
    length = float(np.linalg.norm(chord))
    if length <= 1e-8:
        raise ValueError("semantic path chord is degenerate")
    tangent = chord / length
    normal = np.asarray((-tangent[1], tangent[0]), dtype=np.float64)
    center = 0.5 * (start + end)
    local = points - center[None, :]
    u = local @ tangent
    v = local @ normal
    confidence = float(residual.confidence)
    chord_scale = 1.0 + confidence * (float(residual.chord_scale) - 1.0)
    normal_scale = 1.0 + confidence * (float(residual.normal_scale) - 1.0)
    offset = confidence * float(residual.normal_offset_cm)
    transformed = center[None, :] + (u * chord_scale)[:, None] * tangent + (v * normal_scale + offset)[:, None] * normal

    if protected_points:
        protected = np.zeros(len(points), dtype=np.bool_)
        for value in protected_points:
            protected |= np.linalg.norm(points - np.asarray(value, dtype=np.float64), axis=1) <= tolerance
        if bool(protected.any()):
            # Remove the affine displacement induced at protected named
            # landmarks and interpolate that correction along arc length.
            # The landmark head owns those coordinates; the path head retains
            # only shape change between them.
            segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
            positions = np.concatenate(([0.0], np.cumsum(segment_lengths)))
            if positions[-1] <= 1e-8:
                positions = np.linspace(0.0, 1.0, len(points))
            else:
                positions /= positions[-1]
            displacement = transformed - points
            correction_positions: dict[float, np.ndarray] = {
                0.0: np.zeros(2, dtype=np.float64),
                1.0: np.zeros(2, dtype=np.float64),
            }
            for index in np.flatnonzero(protected):
                correction_positions[float(positions[index])] = displacement[index]
            ordered = sorted(correction_positions.items())
            xp = np.asarray([item[0] for item in ordered], dtype=np.float64)
            correction = np.column_stack(
                [
                    np.interp(positions, xp, np.asarray([item[1][axis] for item in ordered]))
                    for axis in range(2)
                ]
            )
            transformed = transformed - correction
            transformed[protected] = points[protected]
    return transformed


def _force_landmark_target(
    original: Panel,
    edited: Panel,
    original_anchor: np.ndarray,
    target: np.ndarray,
    *,
    tolerance: float = 1e-6,
) -> Panel:
    edges = []
    for original_edge, edited_edge in zip(original.edges, edited.edges):
        original_points = np.asarray(original_edge.points, dtype=np.float64)
        edited_points = np.asarray(edited_edge.points, dtype=np.float64)
        selected = np.linalg.norm(original_points - original_anchor[None, :], axis=1) <= tolerance
        edited_points[selected] = target
        edges.append(
            replace(
                edited_edge,
                points=tuple((float(x), float(y)) for x, y in edited_points),
            )
        )
    return replace(edited, edges=tuple(edges))


def _endpoint_groups(panel: Panel, *, tolerance: float = 1e-6) -> tuple[tuple[tuple[int, int], ...], ...]:
    occurrences = [
        (edge_index, point_index)
        for edge_index, edge in enumerate(panel.edges)
        for point_index in ({0, len(edge.points) - 1} if edge.points else set())
    ]
    groups: list[list[tuple[int, int]]] = []
    for occurrence in occurrences:
        edge_index, point_index = occurrence
        point = np.asarray(panel.edges[edge_index].points[point_index], dtype=np.float64)
        for group in groups:
            first_edge, first_point = group[0]
            reference = np.asarray(panel.edges[first_edge].points[first_point], dtype=np.float64)
            if float(np.linalg.norm(point - reference)) <= tolerance:
                group.append(occurrence)
                break
        else:
            groups.append([occurrence])
    return tuple(tuple(group) for group in groups)


def _apply_path_proposals(
    panel: Panel,
    proposals: Mapping[tuple[int, int], Sequence[np.ndarray]],
    protected_points: Sequence[np.ndarray],
    *,
    tolerance: float = 1e-6,
) -> Panel:
    values = [np.asarray(edge.points, dtype=np.float64).copy() for edge in panel.edges]

    def protected(point: np.ndarray) -> bool:
        return any(float(np.linalg.norm(point - item)) <= tolerance for item in protected_points)

    for occurrence, candidates in proposals.items():
        edge_index, point_index = occurrence
        if not protected(values[edge_index][point_index]):
            values[edge_index][point_index] = np.mean(np.asarray(candidates), axis=0)

    # Boundary vertices are duplicate storage for one topological point.  A
    # transformed proposal is authoritative and is propagated to every
    # occurrence; unchanged neighbours are never averaged in, which previously
    # halved intended endpoint edits.
    for group in _endpoint_groups(panel, tolerance=tolerance):
        originals = [values[edge_index][point_index] for edge_index, point_index in group]
        if any(protected(point) for point in originals):
            common = np.mean(np.asarray(originals), axis=0)
        else:
            candidates = [
                candidate
                for occurrence in group
                for candidate in proposals.get(occurrence, ())
            ]
            common = (
                np.mean(np.asarray(candidates), axis=0)
                if candidates
                else np.mean(np.asarray(originals), axis=0)
            )
        for edge_index, point_index in group:
            values[edge_index][point_index] = common

    edges = tuple(
        replace(edge, points=tuple((float(x), float(y)) for x, y in points))
        for edge, points in zip(panel.edges, values)
    )
    return replace(panel, edges=edges)


def _assert_connectivity_preserved(before: Panel, after: Panel, *, tolerance: float = 1e-6) -> None:
    if len(before.edges) != len(after.edges):
        raise RuntimeError(f"semantic edit changed edge count for panel {before.id}")
    for group in _endpoint_groups(before, tolerance=tolerance):
        if len(group) < 2:
            continue
        points = [
            np.asarray(after.edges[edge_index].points[point_index], dtype=np.float64)
            for edge_index, point_index in group
        ]
        if any(float(np.linalg.norm(point - points[0])) > tolerance for point in points[1:]):
            raise RuntimeError(f"semantic edit opened a boundary vertex in panel {before.id}")


def apply_semantic_residual(
    document: PatternDocument,
    plan: SemanticResidualPlan,
    *,
    strict: bool = True,
) -> PatternDocument:
    """Apply an auditable residual plan while preserving topology and stitches."""

    plan.validate()
    original_panels = {panel.id: panel for panel in document.panels}
    panels = dict(original_panels)
    applied_landmarks: list[str] = []
    applied_paths: list[str] = []
    skipped: list[str] = []
    multiplicity_gated: list[str] = []
    protected_points: dict[str, list[np.ndarray]] = {
        panel.id: [] for panel in document.panels
    }
    landmark_targets: dict[
        str, dict[tuple[float, float], tuple[np.ndarray, np.ndarray]]
    ] = {
        panel.id: {} for panel in document.panels
    }

    # Every named landmark is a geometric constraint, even when its predicted
    # residual is zero.  Smooth edits around FNP, for example, must not drag an
    # unchanged SNP merely because it lies inside the influence radius.
    raw_landmark_group = document.annotations.get("semantic_landmarks", {})
    if not isinstance(raw_landmark_group, Mapping):
        raise ValueError("semantic_landmarks must be a mapping")
    for query_name in raw_landmark_group:
        for entry in _annotation_entries(document, "semantic_landmarks", str(query_name)):
            panel_id = str(entry["panel_id"])
            if panel_id not in original_panels:
                raise ValueError(f"unknown landmark panel {panel_id!r}")
            anchor = _landmark_xy(original_panels[panel_id], entry)
            key = (round(float(anchor[0]), 8), round(float(anchor[1]), 8))
            landmark_targets[panel_id][key] = (anchor, anchor.copy())
    for query_name in _ROLE_SPECIFIC_DART_LANDMARKS:
        for entry in semantic_annotation_entries(document, "landmark", query_name):
            panel_id = str(entry["panel_id"])
            anchor = _landmark_xy(original_panels[panel_id], entry)
            key = (round(float(anchor[0]), 8), round(float(anchor[1]), 8))
            landmark_targets[panel_id][key] = (anchor, anchor.copy())

    # Resolve all annotations before mutating geometry.  A single query tensor
    # does not identify which of several physical instances it describes, so
    # broadcasting it to every dart/panel is unsafe.  The planner normally
    # gates these queries; direct callers receive the same fail-closed policy.
    landmark_work: list[tuple[str, LandmarkResidual, dict[str, Any]]] = []
    path_work: list[tuple[str, PathResidual, dict[str, Any]]] = []

    def resolve(
        group: str,
        name: str,
        residual: LandmarkResidual | PathResidual,
        destination: list[Any],
    ) -> None:
        entries = _annotation_entries(document, group, name)
        kind = "landmark" if group == "semantic_landmarks" else "path"
        if not entries:
            if strict:
                raise ValueError(f"anchor has no semantic {kind} {name!r}")
            skipped.append(f"{kind}:{name}")
            return
        if len(entries) != 1:
            message = f"{kind}:{name}:unsupported_multiplicity:{len(entries)}"
            if strict:
                raise ValueError(
                    f"semantic {kind} {name!r} has {len(entries)} physical instances; "
                    "the query does not identify an instance"
                )
            multiplicity_gated.append(message)
            return
        destination.append((name, residual, entries[0]))

    for name, residual in plan.landmark_residuals.items():
        resolve("semantic_landmarks", name, residual, landmark_work)
    for name, residual in plan.path_residuals.items():
        resolve("semantic_paths", name, residual, path_work)

    for name, residual, entry in landmark_work:
        panel_id = str(entry["panel_id"])
        if panel_id not in panels:
            raise ValueError(f"unknown landmark panel {panel_id!r}")
        original_anchor = _landmark_xy(original_panels[panel_id], entry)
        target = original_anchor + np.asarray(
            (residual.dx_cm, residual.dy_cm), dtype=np.float64
        ) * float(residual.confidence)
        panels[panel_id] = _apply_landmark_to_panel(
            panels[panel_id], original_anchor, residual
        )
        key = (round(float(original_anchor[0]), 8), round(float(original_anchor[1]), 8))
        landmark_targets[panel_id][key] = (original_anchor, target)
        applied_landmarks.append(name)

    # Overlapping smooth landmark fields must not perturb the named targets
    # themselves.  Restore every named topological occurrence to its exact
    # intended coordinate before path-shape edits.
    for panel_id, targets in landmark_targets.items():
        for original_anchor, target in targets.values():
            panels[panel_id] = _force_landmark_target(
                original_panels[panel_id],
                panels[panel_id],
                original_anchor,
                target,
            )
            protected_points[panel_id].append(target)

    source_frame = document.annotations.get("semantic_coordinate_frame", {})
    if source_frame is None:
        source_frame = {}
    if not isinstance(source_frame, Mapping):
        raise ValueError("semantic_coordinate_frame must be a mapping when present")
    source_y_axis_down = bool(source_frame.get("source_y_axis_down", False))

    # Every path proposal is computed from the same post-landmark base.  This
    # prevents sequential adjacent path queries from applying endpoint edits
    # twice.  Shared vertex proposals are merged only after all paths finish.
    path_base = dict(panels)
    proposals: dict[str, dict[tuple[int, int], list[np.ndarray]]] = {
        panel.id: {} for panel in document.panels
    }
    for name, residual, entry in path_work:
        panel_id = str(entry["panel_id"])
        if panel_id not in path_base:
            raise ValueError(f"unknown semantic path panel {panel_id!r}")
        edge_ids = tuple(str(value) for value in entry["edge_ids"])
        components = _ordered_path_components(
            path_base[panel_id],
            edge_ids,
            source_y_axis_down=source_y_axis_down,
        )
        for component in components:
            transformed = _transform_path(
                component.points,
                residual,
                protected_points=protected_points[panel_id],
            )
            for value, occurrences in zip(transformed, component.occurrences):
                for occurrence in occurrences:
                    proposals[panel_id].setdefault(occurrence, []).append(value)
        applied_paths.append(name)

    for panel_id, values in proposals.items():
        panels[panel_id] = _apply_path_proposals(
            path_base[panel_id], values, protected_points[panel_id]
        )
        _assert_connectivity_preserved(original_panels[panel_id], panels[panel_id])

    edited_panels = tuple(panels[panel.id] for panel in document.panels)
    receipt = {
        "schema_version": plan.schema_version,
        "category": plan.category,
        "source": plan.source,
        "applied_landmarks": sorted(applied_landmarks),
        "applied_paths": sorted(applied_paths),
        "skipped": sorted(skipped),
        "multiplicity_gated": sorted(multiplicity_gated),
        "planner_gated_queries": dict(sorted(plan.gated_queries.items())),
        "topology_preserved": True,
        "panel_count_before_after": [len(document.panels), len(edited_panels)],
        "edge_count_before_after": [
            sum(len(panel.edges) for panel in document.panels),
            sum(len(panel.edges) for panel in edited_panels),
        ],
        "stitch_count_before_after": [len(document.stitches), len(document.stitches)],
    }
    return replace(
        document,
        pattern_id=f"{document.pattern_id}_semantic_edit",
        generator=f"{document.generator} + semantic residual editor",
        panels=edited_panels,
        annotations={
            **document.annotations,
            "refinement_status": "semantic_residual_applied",
            "semantic_edit_receipt": receipt,
        },
    )


__all__ = [
    "LandmarkResidual",
    "PathResidual",
    "SemanticResidualPlan",
    "apply_semantic_residual",
    "semantic_annotation_entries",
]
