from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from benchmark.adapters.reweaver import sha256

from .schema import Edge, Panel, PatternDocument, Placement, Stitch, StitchSide


@dataclass(frozen=True)
class OrderedEdge:
    source_index: int
    flipped: bool
    points: np.ndarray


def order_edges(edges: np.ndarray) -> tuple[OrderedEdge, ...]:
    """Find the exact minimum-gap cycle while preserving generated edge identities."""
    edges = np.asarray(edges, dtype=float)
    if edges.ndim != 3 or edges.shape[0] == 0 or edges.shape[-1] != 2:
        raise ValueError("edge array must have shape [edge, sample, 2]")
    best_path: tuple[tuple[int, bool], ...] | None = None
    best_cost = float("inf")
    all_mask = (1 << len(edges)) - 1

    def oriented(index: int, flipped: bool) -> np.ndarray:
        return edges[index][::-1] if flipped else edges[index]

    # A cycle can always be rotated to start at edge 0, so one fixed identity is exact.
    for start in (0,):
        for flip_start in (False, True):
            states: dict[tuple[int, int, bool], tuple[float, tuple[tuple[int, bool], ...]]] = {
                (1 << start, start, flip_start): (0.0, ((start, flip_start),))
            }
            for _ in range(1, len(edges)):
                next_states: dict[tuple[int, int, bool], tuple[float, tuple[tuple[int, bool], ...]]] = {}
                for (mask, last, last_flip), (cost, path) in states.items():
                    end = oriented(last, last_flip)[-1]
                    for index in range(len(edges)):
                        if mask & (1 << index):
                            continue
                        for flipped in (False, True):
                            step = float(np.linalg.norm(end - oriented(index, flipped)[0]))
                            key = (mask | (1 << index), index, flipped)
                            candidate = (cost + step, path + ((index, flipped),))
                            if key not in next_states or candidate < next_states[key]:
                                next_states[key] = candidate
                states = next_states
            start_point = oriented(start, flip_start)[0]
            for (mask, last, last_flip), (cost, path) in states.items():
                if mask != all_mask:
                    continue
                cycle_cost = cost + float(np.linalg.norm(oriented(last, last_flip)[-1] - start_point))
                if (cycle_cost, path) < (best_cost, best_path or ()):
                    best_cost, best_path = cycle_cost, path
    assert best_path is not None
    return tuple(OrderedEdge(index, flipped, oriented(index, flipped)) for index, flipped in best_path)


def _placement(points: np.ndarray) -> Placement | None:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3 or not np.isfinite(points).all():
        return None
    origin = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - origin, full_matrices=False)
    x_axis, y_axis = vh[0], vh[1]
    normal = np.cross(x_axis, y_axis)
    if normal[1] < 0:
        y_axis, normal = -y_axis, -normal
    to_tuple = lambda value: tuple(float(item) for item in value)
    return Placement(to_tuple(origin), to_tuple(x_axis), to_tuple(y_axis), to_tuple(normal))


def convert_reweaver_npz(path: Path, *, validity_threshold: float = 0.5) -> PatternDocument:
    archive = np.load(path, allow_pickle=True)
    flatten = archive["flatten_pred"].item()
    connectivity = np.asarray(archive["patch_curve_connectivity"], dtype=bool)
    similarities = np.asarray(archive["patch_curve_similarity"], dtype=float)
    patch_conf = np.asarray(archive["patch_valid_prob"], dtype=float)
    curve_conf = np.asarray(archive["curve_valid_prob"], dtype=float)
    patch_points = np.asarray(archive["patch_points"], dtype=float)
    panels: list[Panel] = []
    curve_sides: dict[int, list[StitchSide]] = {}

    for panel_index in sorted(flatten):
        if panel_index >= len(patch_conf) or patch_conf[panel_index] < validity_threshold:
            continue
        generated = flatten[panel_index]
        scale = float(np.asarray(generated.get("scale_pred", [1.0])).reshape(-1)[0])
        connected_curves = np.flatnonzero(connectivity[panel_index]).tolist()
        raw_edges = np.asarray(generated["edge_points"], dtype=float) * scale
        if len(connected_curves) != len(raw_edges):
            raise ValueError(f"panel {panel_index}: {len(raw_edges)} edges but {len(connected_curves)} connected curves")
        edges: list[Edge] = []
        for ordered_index, ordered in enumerate(order_edges(raw_edges)):
            curve_id = int(connected_curves[ordered.source_index])
            edge_id = f"panel_{panel_index}.edge_{ordered_index}"
            edge = Edge(
                id=edge_id,
                points=tuple((float(x), float(y)) for x, y in ordered.points),
                source_curve_id=curve_id,
                confidence=float(min(curve_conf[curve_id], similarities[panel_index, curve_id])),
            )
            edges.append(edge)
            curve_sides.setdefault(curve_id, []).append(StitchSide(f"panel_{panel_index}", edge_id, ordered.flipped))
        panels.append(
            Panel(
                id=f"panel_{panel_index}",
                edges=tuple(edges),
                placement=_placement(patch_points[panel_index]),
                source_panel_id=int(panel_index),
                confidence=float(patch_conf[panel_index]),
            )
        )

    stitches = []
    unresolved = {}
    for curve_id, sides in sorted(curve_sides.items()):
        if len(sides) == 2:
            stitches.append(
                Stitch(
                    id=f"stitch_{curve_id}",
                    side_a=sides[0],
                    side_b=sides[1],
                    source_curve_id=curve_id,
                    confidence=float(curve_conf[curve_id]),
                )
            )
        elif len(sides) > 2:
            unresolved[str(curve_id)] = len(sides)
    return PatternDocument(
        pattern_id=path.stem,
        generator="ReWeaver",
        panels=tuple(panels),
        stitches=tuple(stitches),
        provenance={"source_artifact_sha256": sha256(path), "source_format": "reweaver_npz"},
        annotations={
            "topology": "model_generated_variable",
            "template_retrieval": False,
            "unresolved_curve_degrees": unresolved,
        },
    )
