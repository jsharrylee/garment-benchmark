"""Exact GCDv2 pattern-element to simulated-surface correspondence.

GarmentCodeData v2 stores the pre-simulation box mesh and the draped mesh with
the same face/UV topology.  The PLY conversion expands a small number of
vertices at UV seams, while ``*_sim_segmentation.txt`` and
``*_vertex_labels.yaml`` retain the original OBJ vertex indexing.  This module
collapses only exact UV duplicates, verifies the original vertex count against
the segmentation, and then carries source semantic edges through the shared
vertex order into the simulated mesh.

This is an observational correspondence contract for already released GCDv2
samples.  It is not a counterfactual cloth simulator and must not be used to
claim that an edited pattern caused a rendered change.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .basic_semantic_targets import resolved_common_basic_edge_roles
from .schema import DraftingSemanticRecord


SCHEMA_VERSION = "gcdv2-tshirt-surface-correspondence/v2"
VIEW_NAMES = ("front", "back", "left", "right")
FPN_GRID_SIZES = (8, 4, 2, 1)

ELEMENT_NAMES = (
    "front_neckline",
    "back_neckline",
    "front_shoulder",
    "back_shoulder",
    "front_armhole",
    "back_armhole",
    "front_side_seam",
    "back_side_seam",
    "front_center",
    "back_center",
    "front_hemline",
    "back_hemline",
    "sleeve_head",
    "sleeve_underarm",
    "sleeve_hem",
)

# ``sleeve_width_cm`` is explicitly the sum of the two non-stitched opening
# edges of one physical sleeve.  It is not a biceps circumference estimate.
TSHIRT_PARAMETER_NAMES = (
    "neck_width_cm",
    "front_neck_depth_cm",
    "shoulder_slope_deg",
    "armhole_depth_cm",
    "body_length_cm",
    "sleeve_cap_height_cm",
    "sleeve_length_cm",
    "sleeve_width_cm",
)

PARAMETER_TO_ELEMENTS = {
    "neck_width_cm": (
        "front_neckline",
        "back_neckline",
        "front_shoulder",
        "back_shoulder",
    ),
    "front_neck_depth_cm": ("front_neckline", "front_center"),
    "shoulder_slope_deg": ("front_shoulder", "back_shoulder"),
    "armhole_depth_cm": (
        "front_armhole",
        "back_armhole",
        "front_side_seam",
        "back_side_seam",
    ),
    "body_length_cm": (
        "front_center",
        "back_center",
        "front_side_seam",
        "back_side_seam",
        "front_hemline",
        "back_hemline",
    ),
    "sleeve_cap_height_cm": ("sleeve_head", "front_armhole", "back_armhole"),
    "sleeve_length_cm": ("sleeve_underarm", "sleeve_hem"),
    "sleeve_width_cm": ("sleeve_hem", "sleeve_underarm"),
}


@dataclass(frozen=True)
class SharedTopologyMesh:
    """Original-index box/sim vertices recovered from UV-expanded PLY files."""

    box_vertices_cm: np.ndarray
    sim_vertices_cm: np.ndarray
    segmentation: tuple[str, ...]
    raw_vertex_count: int
    original_vertex_count: int
    face_count: int


@dataclass(frozen=True)
class ElementProjection:
    element_name: str
    vertex_indices: np.ndarray
    normalized_xy: np.ndarray  # [4, maximum_points, 2], NaN padded by caller
    visible: np.ndarray


@dataclass(frozen=True)
class TShirtSurfaceExample:
    sample_id: str
    split: str
    parameter_values: np.ndarray
    parameter_valid: np.ndarray
    element_heatmaps: np.ndarray
    element_valid: np.ndarray
    element_vertex_counts: np.ndarray
    audit: Mapping[str, Any]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_segmentation(path: Path) -> tuple[str, ...]:
    return tuple(
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def collapse_uv_expanded_vertices(
    vertices: np.ndarray,
    *,
    decimals: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """Preserve first-occurrence order while collapsing identical XYZ rows.

    Trimesh's OBJ-to-PLY conversion duplicates a geometric vertex when it has
    multiple texture coordinates.  The conversion emits duplicates adjacent to
    the original face traversal while retaining the original vertex encounter
    order.  Returning the first raw index lets the same collapse be applied to
    the paired simulated vertex array.
    """

    values = np.asarray(vertices, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("vertices must have shape [N,3]")
    first_indices: list[int] = []
    seen: dict[tuple[float, float, float], int] = {}
    for index, row in enumerate(np.round(values, decimals=decimals)):
        key = (float(row[0]), float(row[1]), float(row[2]))
        if key not in seen:
            seen[key] = len(first_indices)
            first_indices.append(index)
    raw = np.asarray(first_indices, dtype=np.int64)
    return values[raw], raw


def load_shared_topology_mesh(
    boxmesh_path: Path,
    simmesh_path: Path,
    segmentation_path: Path,
) -> SharedTopologyMesh:
    """Load and verify the GCDv2 boxmesh-to-sim shared vertex contract."""

    import trimesh

    box = trimesh.load_mesh(Path(boxmesh_path), process=False)
    sim = trimesh.load_mesh(Path(simmesh_path), process=False)
    box_vertices = np.asarray(box.vertices, dtype=np.float64)
    sim_vertices = np.asarray(sim.vertices, dtype=np.float64)
    box_faces = np.asarray(box.faces, dtype=np.int64)
    sim_faces = np.asarray(sim.faces, dtype=np.int64)
    if box_vertices.shape != sim_vertices.shape:
        raise ValueError("boxmesh and sim vertex arrays differ")
    if box_faces.shape != sim_faces.shape or not np.array_equal(box_faces, sim_faces):
        raise ValueError("boxmesh and sim face topology differs")
    box_unique, first_raw = collapse_uv_expanded_vertices(box_vertices)
    segmentation = _read_segmentation(segmentation_path)
    if len(box_unique) != len(segmentation):
        raise ValueError(
            "UV-collapse/original segmentation count mismatch: "
            f"{len(box_unique)} != {len(segmentation)}"
        )
    sim_unique = sim_vertices[first_raw]
    return SharedTopologyMesh(
        box_vertices_cm=box_unique.astype(np.float32),
        sim_vertices_cm=sim_unique.astype(np.float32),
        segmentation=segmentation,
        raw_vertex_count=int(len(box_vertices)),
        original_vertex_count=int(len(box_unique)),
        face_count=int(len(box_faces)),
    )


def _rotation_matrix_xyz_degrees(values: Sequence[float]) -> np.ndarray:
    x, y, z = (math.radians(float(value)) for value in values)
    rx = np.asarray(
        ((1.0, 0.0, 0.0), (0.0, math.cos(x), -math.sin(x)), (0.0, math.sin(x), math.cos(x)))
    )
    ry = np.asarray(
        ((math.cos(y), 0.0, math.sin(y)), (0.0, 1.0, 0.0), (-math.sin(y), 0.0, math.cos(y)))
    )
    rz = np.asarray(
        ((math.cos(z), -math.sin(z), 0.0), (math.sin(z), math.cos(z), 0.0), (0.0, 0.0, 1.0))
    )
    return rz @ ry @ rx


def _relative_control(start: np.ndarray, end: np.ndarray, relative: Sequence[float]) -> np.ndarray:
    edge = end - start
    perpendicular = np.asarray((-edge[1], edge[0]), dtype=np.float64)
    return start + float(relative[0]) * edge + float(relative[1]) * perpendicular


def sample_specification_edge(
    panel: Mapping[str, Any],
    edge_index: int,
    *,
    samples: int = 513,
) -> np.ndarray:
    """Sample one source line/Bezier/arc in panel-local centimetres."""

    edge = panel["edges"][int(edge_index)]
    vertices = np.asarray(panel["vertices"], dtype=np.float64)
    start, end = vertices[np.asarray(edge["endpoints"], dtype=np.int64)]
    t = np.linspace(0.0, 1.0, int(samples), dtype=np.float64)[:, None]
    curvature = edge.get("curvature")
    if curvature is None:
        return (1.0 - t) * start + t * end
    if isinstance(curvature, list):
        controls = [curvature]
        kind = "quadratic"
    else:
        kind = str(curvature.get("type"))
        controls = curvature.get("params", [])
    if kind == "quadratic":
        control = _relative_control(start, end, controls[0])
        return (1.0 - t) ** 2 * start + 2.0 * (1.0 - t) * t * control + t**2 * end
    if kind == "cubic":
        first = _relative_control(start, end, controls[0])
        second = _relative_control(start, end, controls[1])
        return (
            (1.0 - t) ** 3 * start
            + 3.0 * (1.0 - t) ** 2 * t * first
            + 3.0 * (1.0 - t) * t**2 * second
            + t**3 * end
        )
    if kind == "circle":
        from svgpathtools import Arc

        radius, large_arc, sweep = controls
        curve = Arc(
            complex(*start),
            complex(float(radius), float(radius)),
            rotation=0.0,
            large_arc=bool(large_arc),
            sweep=bool(sweep),
            end=complex(*end),
        )
        return np.asarray(
            ((curve.point(float(value)).real, curve.point(float(value)).imag) for value in t[:, 0]),
            dtype=np.float64,
        )
    raise ValueError(f"unsupported source curvature: {kind}")


def transform_panel_points(panel: Mapping[str, Any], points: np.ndarray) -> np.ndarray:
    local = np.column_stack((np.asarray(points, dtype=np.float64), np.zeros(len(points))))
    rotation = _rotation_matrix_xyz_degrees(panel.get("rotation", (0.0, 0.0, 0.0)))
    translation = np.asarray(panel.get("translation", (0.0, 0.0, 0.0)), dtype=np.float64)
    return local @ rotation.T + translation


def _stitch_lookup(specification: Mapping[str, Any]) -> dict[tuple[str, int], tuple[int, ...]]:
    output: dict[tuple[str, int], list[int]] = {}
    for stitch_index, pair in enumerate(specification["pattern"].get("stitches", [])):
        for side in pair:
            output.setdefault((str(side["panel"]), int(side["edge"])), []).append(stitch_index)
    return {key: tuple(value) for key, value in output.items()}


def _load_vertex_labels(path: Path) -> dict[str, np.ndarray]:
    import yaml

    value = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return {str(key): np.asarray(indices, dtype=np.int64) for key, indices in value.items()}


def source_edge_vertex_indices(
    *,
    specification: Mapping[str, Any],
    panel_id: str,
    edge_index: int,
    mesh: SharedTopologyMesh,
    vertex_labels: Mapping[str, np.ndarray],
    maximum_curve_distance_cm: float = 0.08,
) -> tuple[np.ndarray, str]:
    """Resolve one source edge to original mesh vertex indices with evidence."""

    panel = specification["pattern"]["panels"][panel_id]
    edge = panel["edges"][int(edge_index)]
    label = edge.get("label")
    if label and str(label) in vertex_labels:
        indices = np.asarray(vertex_labels[str(label)], dtype=np.int64)
        if len(indices) and int(indices.max()) < mesh.original_vertex_count:
            return np.unique(indices), "source_vertex_label"

    stitch_ids = _stitch_lookup(specification).get((panel_id, int(edge_index)), ())
    if stitch_ids:
        wanted = {f"stitch_{value}" for value in stitch_ids}
        indices = [
            index
            for index, row in enumerate(mesh.segmentation)
            if wanted.intersection(row.split(","))
        ]
        if indices:
            # GarmentCode's stitched boundary vertices are shared by the two
            # sewn pattern edges in 3D.  The same stitch membership is thus the
            # exact surface locus for both semantic sides (for example front
            # and back shoulder), even though their 2D source curves differ.
            return np.asarray(indices, dtype=np.int64), "source_stitch_membership_shared"

    candidates = np.asarray(
        [index for index, row in enumerate(mesh.segmentation) if row == panel_id],
        dtype=np.int64,
    )
    if not len(candidates):
        return np.zeros(0, dtype=np.int64), "unresolved_no_panel_vertices"
    sampled = transform_panel_points(panel, sample_specification_edge(panel, edge_index))
    try:
        from scipy.spatial import cKDTree

        distance, _ = cKDTree(sampled).query(mesh.box_vertices_cm[candidates], k=1)
    except ImportError:
        difference = mesh.box_vertices_cm[candidates, None, :] - sampled[None, :, :]
        distance = np.sqrt(np.min(np.sum(difference * difference, axis=-1), axis=1))
    matched = candidates[np.asarray(distance) <= float(maximum_curve_distance_cm)]
    return np.unique(matched), "boxmesh_analytic_curve_match"


def _element_name(panel_role: str, edge_role: str, *, stitched: bool) -> str | None:
    if edge_role == "neckline" and panel_role == "front_bodice":
        return "front_neckline"
    if edge_role == "neckline" and panel_role == "back_bodice":
        return "back_neckline"
    if edge_role == "shoulder" and panel_role == "front_bodice":
        return "front_shoulder"
    if edge_role == "shoulder" and panel_role == "back_bodice":
        return "back_shoulder"
    if edge_role == "armhole" and panel_role == "front_bodice":
        return "front_armhole"
    if edge_role == "armhole" and panel_role == "back_bodice":
        return "back_armhole"
    if edge_role == "side_seam" and panel_role == "front_bodice":
        return "front_side_seam"
    if edge_role == "side_seam" and panel_role == "back_bodice":
        return "back_side_seam"
    if edge_role == "center_front" and panel_role == "front_bodice":
        return "front_center"
    if edge_role == "center_back" and panel_role == "back_bodice":
        return "back_center"
    if edge_role in {"waistline", "hemline"} and panel_role == "front_bodice":
        return "front_hemline"
    if edge_role in {"waistline", "hemline"} and panel_role == "back_bodice":
        return "back_hemline"
    if edge_role == "sleeve_head" and panel_role == "sleeve":
        return "sleeve_head"
    if edge_role == "sleeve_underarm" and panel_role == "sleeve":
        return "sleeve_underarm"
    if edge_role == "other" and panel_role == "sleeve" and not stitched:
        return "sleeve_hem"
    return None


def build_element_vertex_index(
    semantic_record: Mapping[str, Any],
    specification: Mapping[str, Any],
    mesh: SharedTopologyMesh,
    vertex_labels_path: Path,
    *,
    resolved_roles: Mapping[str, str | None] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    labels = _load_vertex_labels(vertex_labels_path)
    grouped: dict[str, list[np.ndarray]] = {name: [] for name in ELEMENT_NAMES}
    evidence_counts: dict[str, int] = {}
    unresolved: list[str] = []
    for panel in semantic_record["panels"]:
        panel_id = str(panel["id"])
        panel_role = str(panel["role"])
        for edge in panel["edges"]:
            raw_role = str(edge["role"])
            edge_role = (
                resolved_roles.get(str(edge["id"]), raw_role)
                if resolved_roles is not None
                else raw_role
            )
            # The common-basic resolver intentionally masks generic ``other``
            # edges for primitive-role supervision.  In this exact surface
            # contract an unstitched sleeve ``other`` edge is the observed
            # sleeve opening/hem, so preserve that source evidence.
            if (
                edge_role is None
                and panel_role == "sleeve"
                and raw_role == "other"
                and not bool(edge.get("stitched", False))
            ):
                edge_role = raw_role
            if edge_role is None:
                continue
            element = _element_name(
                panel_role,
                str(edge_role),
                stitched=bool(edge.get("stitched", False)),
            )
            if element is None:
                continue
            indices, evidence = source_edge_vertex_indices(
                specification=specification,
                panel_id=panel_id,
                edge_index=int(edge["index"]),
                mesh=mesh,
                vertex_labels=labels,
            )
            evidence_counts[evidence] = evidence_counts.get(evidence, 0) + 1
            if len(indices):
                grouped[element].append(indices)
            else:
                unresolved.append(str(edge["id"]))
    result = {
        name: np.unique(np.concatenate(values)).astype(np.int64)
        if values
        else np.zeros(0, dtype=np.int64)
        for name, values in grouped.items()
    }
    return result, {
        "evidence_counts": evidence_counts,
        "unresolved_edge_ids": unresolved,
        "element_vertex_counts": {key: int(len(value)) for key, value in result.items()},
    }


def project_gcdv2_orthographic(
    vertices_cm: np.ndarray,
    *,
    bounds_vertices_cm: np.ndarray | None = None,
    margin_scale: float = 1.10,
) -> tuple[np.ndarray, np.ndarray]:
    """Match the fixed orthographic geometry used by the Blender renderer.

    Returns normalized image coordinates and a larger-is-closer depth score in
    semantic view order ``front, back, left, right``.
    """

    values = np.asarray(vertices_cm, dtype=np.float64)
    bounds = values if bounds_vertices_cm is None else np.asarray(bounds_vertices_cm, dtype=np.float64)
    minimum = bounds.min(axis=0)
    maximum = bounds.max(axis=0)
    center = (minimum + maximum) * 0.5
    scale = max(float((maximum - minimum).max()) * float(margin_scale), 1e-8)
    x, y, z = values.T
    coordinates = np.stack(
        (
            np.column_stack((-x, y)),
            np.column_stack((x, y)),
            np.column_stack((-z, y)),
            np.column_stack((z, y)),
        )
    )
    center_coordinates = np.asarray(
        ((-center[0], center[1]), (center[0], center[1]), (-center[2], center[1]), (center[2], center[1])),
        dtype=np.float64,
    )
    normalized = (coordinates - center_coordinates[:, None, :]) / scale + 0.5
    normalized[..., 1] = 1.0 - normalized[..., 1]
    depth = np.stack((z, -z, -x, x))
    return normalized.astype(np.float32), depth.astype(np.float32)


def visible_projected_vertices(
    normalized_xy: np.ndarray,
    depth_score: np.ndarray,
    candidate_indices: np.ndarray,
    *,
    resolution: int = 384,
    depth_tolerance_cm: float = 2.0,
) -> np.ndarray:
    """Approximate per-view visibility with a vertex z-buffer."""

    xy = np.asarray(normalized_xy, dtype=np.float64)
    depth = np.asarray(depth_score, dtype=np.float64)
    candidates = np.asarray(candidate_indices, dtype=np.int64)
    output = np.zeros((len(VIEW_NAMES), len(candidates)), dtype=bool)
    if not len(candidates):
        return output
    for view in range(len(VIEW_NAMES)):
        pixels = np.rint(xy[view] * (resolution - 1)).astype(np.int64)
        inside = np.all((pixels >= 0) & (pixels < resolution), axis=1)
        zbuffer = np.full((resolution, resolution), -np.inf, dtype=np.float32)
        np.maximum.at(
            zbuffer,
            (pixels[inside, 1], pixels[inside, 0]),
            depth[view, inside],
        )
        target_pixels = pixels[candidates]
        target_inside = np.all((target_pixels >= 0) & (target_pixels < resolution), axis=1)
        for local_index in np.flatnonzero(target_inside):
            px, py = target_pixels[local_index]
            x0, x1 = max(0, px - 1), min(resolution, px + 2)
            y0, y1 = max(0, py - 1), min(resolution, py + 2)
            nearest = float(np.max(zbuffer[y0:y1, x0:x1]))
            output[view, local_index] = depth[view, candidates[local_index]] >= nearest - depth_tolerance_cm
    return output


def fpn_token_layout(grid_sizes: Sequence[int] = FPN_GRID_SIZES) -> np.ndarray:
    rows: list[tuple[float, float, float]] = []
    for level, grid in enumerate(grid_sizes):
        for y in range(int(grid)):
            for x in range(int(grid)):
                rows.append(((x + 0.5) / grid, (y + 0.5) / grid, float(level)))
    return np.asarray(rows, dtype=np.float32)


def element_heatmaps_from_vertices(
    sim_vertices_cm: np.ndarray,
    element_indices: Mapping[str, np.ndarray],
    *,
    grid_sizes: Sequence[int] = FPN_GRID_SIZES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xy, depth = project_gcdv2_orthographic(sim_vertices_cm)
    layout = fpn_token_layout(grid_sizes)
    heatmaps = np.zeros((len(ELEMENT_NAMES), len(VIEW_NAMES), len(layout)), dtype=np.float32)
    valid = np.zeros(len(ELEMENT_NAMES), dtype=bool)
    counts = np.zeros(len(ELEMENT_NAMES), dtype=np.int32)
    level_sigmas = np.asarray([0.72 / float(grid_sizes[int(level)]) for level in layout[:, 2]])
    for element_index, name in enumerate(ELEMENT_NAMES):
        indices = np.asarray(element_indices.get(name, ()), dtype=np.int64)
        counts[element_index] = len(indices)
        if not len(indices):
            continue
        visible = visible_projected_vertices(xy, depth, indices)
        for view in range(len(VIEW_NAMES)):
            points = xy[view, indices[visible[view]]]
            if not len(points):
                continue
            distance_squared = np.min(
                np.sum((layout[:, None, :2] - points[None, :, :]) ** 2, axis=-1),
                axis=1,
            )
            heatmaps[element_index, view] = np.exp(
                -distance_squared / (2.0 * np.square(level_sigmas))
            )
        valid[element_index] = bool(np.any(heatmaps[element_index] > 0.05))
    return heatmaps, valid, counts


def _group_edges(
    record: Mapping[str, Any],
    panel_role: str,
    role: str,
    resolved_roles: Mapping[str, str | None] | None = None,
) -> list[Mapping[str, Any]]:
    return [
        edge
        for panel in record["panels"]
        if str(panel["role"]) == panel_role
        for edge in panel["edges"]
        if (
            resolved_roles.get(str(edge.get("id", "")), str(edge["role"]))
            if resolved_roles is not None
            else str(edge["role"])
        )
        == role
    ]


def _axis_extent(edges: Sequence[Mapping[str, Any]]) -> tuple[float, float]:
    points = [point for edge in edges for point in (edge["start_cm"], edge["end_cm"])]
    if not points:
        return math.nan, math.nan
    values = np.asarray(points, dtype=np.float64)
    return float(np.ptp(values[:, 0])), float(np.ptp(values[:, 1]))


def _mean_edge_angle(edges: Sequence[Mapping[str, Any]]) -> float:
    values = []
    for edge in edges:
        start = np.asarray(edge["start_cm"], dtype=np.float64)
        end = np.asarray(edge["end_cm"], dtype=np.float64)
        delta = end - start
        values.append(math.degrees(math.atan2(abs(float(delta[1])), abs(float(delta[0])) + 1e-8)))
    return float(np.mean(values)) if values else math.nan


def _panel_source_edges(
    semantic_record: Mapping[str, Any],
    specification: Mapping[str, Any],
    *,
    panel_role: str,
    edge_role: str,
) -> list[np.ndarray]:
    output: list[np.ndarray] = []
    panels = specification["pattern"]["panels"]
    for panel in semantic_record["panels"]:
        if str(panel["role"]) != panel_role:
            continue
        source_panel = panels[str(panel["id"])]
        matching = [edge for edge in panel["edges"] if str(edge["role"]) == edge_role]
        if not matching:
            continue
        pieces = [sample_specification_edge(source_panel, int(edge["index"])) for edge in matching]
        output.append(np.concatenate(pieces, axis=0))
    return output


def _cap_height(points: np.ndarray) -> float:
    if len(points) < 3:
        return math.nan
    start, end = points[0], points[-1]
    chord = end - start
    length = float(np.linalg.norm(chord))
    if length <= 1e-8:
        return math.nan
    cross = np.abs(chord[0] * (start[1] - points[:, 1]) - chord[1] * (start[0] - points[:, 0]))
    return float(np.max(cross) / length)


def extract_tshirt_physical_parameters(
    semantic_record: Mapping[str, Any],
    specification: Mapping[str, Any],
    *,
    resolved_roles: Mapping[str, str | None] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    """Derive eight panel-intrinsic physical quantities from exact 2D truth."""

    front_neck = _group_edges(
        semantic_record, "front_bodice", "neckline", resolved_roles
    )
    shoulder = _group_edges(
        semantic_record, "back_bodice", "shoulder", resolved_roles
    ) + _group_edges(
        semantic_record, "front_bodice", "shoulder", resolved_roles
    )
    armhole = _group_edges(
        semantic_record, "back_bodice", "armhole", resolved_roles
    ) + _group_edges(
        semantic_record, "front_bodice", "armhole", resolved_roles
    )
    neck_width, neck_depth = _axis_extent(front_neck)
    _, armhole_depth = _axis_extent(armhole)
    # A drafting path may be split into several primitive edges at reference
    # lines or construction junctions.  Averaging primitive lengths therefore
    # underestimates the garment length.  Measure each complete bodice panel's
    # vertical extent and then average symmetric front/back instances.
    body_lengths = [
        float(np.ptp(np.asarray(panel["vertices_cm"], dtype=np.float64)[:, 1]))
        for panel in semantic_record["panels"]
        if str(panel["role"]) in {"front_bodice", "back_bodice"}
        and len(panel.get("vertices_cm", ())) >= 2
    ]

    sleeve_caps = _panel_source_edges(
        semantic_record, specification, panel_role="sleeve", edge_role="sleeve_head"
    )
    cap_values = [_cap_height(points) for points in sleeve_caps]
    cap_values = [value for value in cap_values if math.isfinite(value)]

    sleeve_length_values: list[float] = []
    sleeve_width_values: dict[str, list[float]] = {"left": [], "right": []}
    for panel in semantic_record["panels"]:
        if str(panel["role"]) != "sleeve":
            continue
        vertices = np.asarray(panel["vertices_cm"], dtype=np.float64)
        sleeve_length_values.append(float(np.ptp(vertices[:, 0])))
        opening = [
            edge
            for edge in panel["edges"]
            if str(edge["role"]) == "other" and not bool(edge.get("stitched", False))
        ]
        side = "left" if str(panel["id"]).startswith("left_") else "right"
        sleeve_width_values[side].extend(float(edge["length_cm"]) for edge in opening)
    side_widths = [sum(values) for values in sleeve_width_values.values() if values]

    values = np.asarray(
        (
            neck_width,
            neck_depth,
            _mean_edge_angle(shoulder),
            armhole_depth,
            float(np.mean(body_lengths)) if body_lengths else math.nan,
            float(np.mean(cap_values)) if cap_values else math.nan,
            float(np.mean(sleeve_length_values)) if sleeve_length_values else math.nan,
            float(np.mean(side_widths)) if side_widths else math.nan,
        ),
        dtype=np.float32,
    )
    valid = np.isfinite(values)
    definitions = {
        "neck_width_cm": "front half-pattern neckline endpoint X separation",
        "front_neck_depth_cm": "front half-pattern neckline endpoint Y separation",
        "shoulder_slope_deg": "mean absolute shoulder-edge angle from horizontal",
        "armhole_depth_cm": "front/back armhole path endpoint Y span",
        "body_length_cm": "mean complete front/back bodice panel vertical extent",
        "sleeve_cap_height_cm": "GCD half-sleeve path maximum chord-normal height",
        "sleeve_length_cm": "mean GCD half-sleeve panel X extent",
        "sleeve_width_cm": "sum of front/back opening-edge lengths for one sleeve",
    }
    return values, valid, definitions


def is_simple_tshirt_record(record: Mapping[str, Any], category: str) -> bool:
    if category != "top":
        return False
    design = record.get("program", {}).get("design_values", {})
    return bool(
        record.get("program", {}).get("upper_type") in {"Shirt", "FittedShirt"}
        and design.get("collar.component.style") is None
        and design.get("sleeve.cuff.type") is None
        and not design.get("sleeve.sleeveless", False)
        and not design.get("shirt.strapless", False)
    )


def build_tshirt_surface_example(
    *,
    index_row: Mapping[str, Any],
    semantic_record: Mapping[str, Any],
    raw_root: Path,
) -> TShirtSurfaceExample:
    sample_id = str(index_row["sample_id"])
    sample_root = Path(raw_root) / sample_id
    prefix = sample_root / sample_id
    specification = json.loads(
        prefix.with_name(f"{sample_id}_specification.json").read_text(encoding="utf-8")
    )
    mesh = load_shared_topology_mesh(
        prefix.with_name(f"{sample_id}_boxmesh.ply"),
        prefix.with_name(f"{sample_id}_sim.ply"),
        prefix.with_name(f"{sample_id}_sim_segmentation.txt"),
    )
    typed_record = DraftingSemanticRecord.from_dict(dict(semantic_record))
    resolved_roles = resolved_common_basic_edge_roles(typed_record)
    resolved_role_overrides = {
        str(edge["id"]): {
            "raw": str(edge["role"]),
            "resolved": resolved_roles[str(edge["id"])],
        }
        for panel in semantic_record["panels"]
        for edge in panel["edges"]
        if resolved_roles.get(str(edge["id"])) not in {None, str(edge["role"])}
    }
    elements, element_audit = build_element_vertex_index(
        semantic_record,
        specification,
        mesh,
        prefix.with_name(f"{sample_id}_vertex_labels.yaml"),
        resolved_roles=resolved_roles,
    )
    heatmaps, element_valid, counts = element_heatmaps_from_vertices(
        mesh.sim_vertices_cm, elements
    )
    parameters, parameter_valid, definitions = extract_tshirt_physical_parameters(
        semantic_record, specification, resolved_roles=resolved_roles
    )
    return TShirtSurfaceExample(
        sample_id=sample_id,
        split=str(semantic_record["split"]),
        parameter_values=parameters,
        parameter_valid=parameter_valid,
        element_heatmaps=heatmaps,
        element_valid=element_valid,
        element_vertex_counts=counts,
        audit={
            "schema_version": SCHEMA_VERSION,
            "source": "GarmentCodeData v2 official boxmesh/sim shared topology",
            "causal_claim": False,
            "raw_vertex_count": mesh.raw_vertex_count,
            "original_vertex_count": mesh.original_vertex_count,
            "face_count": mesh.face_count,
            "element_mapping": element_audit,
            "semantic_role_resolver": {
                "schema": "resolved_common_basic_edge_roles",
                "override_count": len(resolved_role_overrides),
                "overrides": resolved_role_overrides,
            },
            "parameter_definitions": definitions,
        },
    )


def build_tshirt_visual_correspondence_model(
    *,
    feature_dim: int = 256,
    width: int = 128,
    heads: int = 4,
    layers: int = 2,
    dropout: float = 0.1,
):
    """Build a small frozen-FPN student with element-location and value heads."""

    import torch

    class TShirtVisualCorrespondence(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.feature_projection = torch.nn.Sequential(
                torch.nn.LayerNorm(feature_dim),
                torch.nn.Linear(feature_dim, width),
                torch.nn.GELU(),
            )
            self.view_embedding = torch.nn.Embedding(len(VIEW_NAMES), width)
            self.spatial_embedding = torch.nn.Parameter(
                torch.zeros(len(VIEW_NAMES), sum(value * value for value in FPN_GRID_SIZES), width)
            )
            self.element_queries = torch.nn.Parameter(torch.zeros(len(ELEMENT_NAMES), width))
            decoder_layer = torch.nn.TransformerDecoderLayer(
                width,
                heads,
                width * 3,
                dropout,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
            self.decoder = torch.nn.TransformerDecoder(decoder_layer, layers)
            self.location_query = torch.nn.Linear(width, width, bias=False)
            self.location_key = torch.nn.Linear(width, width, bias=False)
            self.parameter_queries = torch.nn.Parameter(torch.zeros(len(TSHIRT_PARAMETER_NAMES), width))
            self.parameter_attention = torch.nn.MultiheadAttention(width, heads, batch_first=True)
            self.parameter_head = torch.nn.Sequential(
                torch.nn.LayerNorm(width), torch.nn.Linear(width, width), torch.nn.GELU(), torch.nn.Linear(width, 2)
            )
            torch.nn.init.normal_(self.element_queries, std=0.02)
            torch.nn.init.normal_(self.parameter_queries, std=0.02)
            torch.nn.init.normal_(self.spatial_embedding, std=0.01)

        def forward(self, spatial_features):
            if spatial_features.ndim != 4 or tuple(spatial_features.shape[1:3]) != (
                len(VIEW_NAMES),
                sum(value * value for value in FPN_GRID_SIZES),
            ):
                raise ValueError("spatial_features must have shape [B,4,85,256]")
            batch = spatial_features.shape[0]
            memory = self.feature_projection(spatial_features)
            view_ids = torch.arange(len(VIEW_NAMES), device=memory.device)
            memory = memory + self.view_embedding(view_ids)[None, :, None] + self.spatial_embedding[None]
            flat_memory = memory.reshape(batch, -1, memory.shape[-1])
            queries = self.element_queries[None].expand(batch, -1, -1)
            elements = self.decoder(queries, flat_memory)
            location = torch.matmul(
                self.location_query(elements),
                self.location_key(flat_memory).transpose(1, 2),
            ) / math.sqrt(float(elements.shape[-1]))
            location = location.reshape(batch, len(ELEMENT_NAMES), len(VIEW_NAMES), -1)
            parameter_queries = self.parameter_queries[None].expand(batch, -1, -1)
            parameter_hidden, parameter_attention = self.parameter_attention(
                parameter_queries, elements, elements, need_weights=True, average_attn_weights=False
            )
            distribution = self.parameter_head(parameter_hidden)
            return {
                "element_location_logits": location,
                "element_hidden": elements,
                "parameter_mean": distribution[..., 0],
                "parameter_log_variance": distribution[..., 1].clamp(-6.0, 5.0),
                "parameter_element_attention": parameter_attention,
            }

    return TShirtVisualCorrespondence()


__all__ = [
    "ELEMENT_NAMES",
    "FPN_GRID_SIZES",
    "PARAMETER_TO_ELEMENTS",
    "SCHEMA_VERSION",
    "TSHIRT_PARAMETER_NAMES",
    "TShirtSurfaceExample",
    "VIEW_NAMES",
    "build_element_vertex_index",
    "build_tshirt_surface_example",
    "build_tshirt_visual_correspondence_model",
    "collapse_uv_expanded_vertices",
    "element_heatmaps_from_vertices",
    "extract_tshirt_physical_parameters",
    "fpn_token_layout",
    "is_simple_tshirt_record",
    "load_shared_topology_mesh",
    "project_gcdv2_orthographic",
    "sample_specification_edge",
    "source_edge_vertex_indices",
    "visible_projected_vertices",
]
