from __future__ import annotations

from pathlib import Path
import math

import numpy as np

from benchmark.adapters.garment_particles import sha256

from .schema import Edge, Panel, PatternDocument, Placement, Stitch, StitchSide


def _relative_point(start: np.ndarray, end: np.ndarray, relative: np.ndarray) -> np.ndarray:
    vector = end - start
    return start + relative[0] * vector + relative[1] * np.array([-vector[1], vector[0]])


def sample_generated_edge(start: np.ndarray, vector: np.ndarray, samples: int = 30) -> np.ndarray:
    end = start + vector[:2]
    t = np.linspace(0.0, 1.0, samples)[:, None]
    if vector[6] > 0.5:
        point = _relative_point(start, end, vector[2:4])
        ax, ay = start
        bx, by = point
        cx, cy = end
        determinant = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(determinant) < 1e-9:
            return (1 - t) * start + t * end
        center = np.array(
            [
                ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / determinant,
                ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / determinant,
            ]
        )
        angles = [math.atan2(*(value - center)[::-1]) for value in (start, point, end)]
        ccw = (angles[2] - angles[0]) % (2.0 * math.pi)
        contains_point = (angles[1] - angles[0]) % (2.0 * math.pi) <= ccw
        delta = ccw if contains_point else -((angles[0] - angles[2]) % (2.0 * math.pi))
        theta = angles[0] + np.linspace(0.0, delta, samples)
        radius = float(np.linalg.norm(start - center))
        return center + radius * np.column_stack((np.cos(theta), np.sin(theta)))
    if np.allclose(vector[2:6], 0.0, atol=1e-3):
        return (1 - t) * start + t * end
    control_1 = _relative_point(start, end, vector[2:4])
    control_2 = _relative_point(start, end, vector[4:6])
    return (1 - t) ** 3 * start + 3 * (1 - t) ** 2 * t * control_1 + 3 * (1 - t) * t**2 * control_2 + t**3 * end


def convert_garment_particles_npz(path: Path) -> PatternDocument:
    archive = np.load(path)
    edges = np.asarray(archive["edges"], dtype=float)
    valid = np.asarray(archive["edge_valid_mask"], dtype=bool)
    translations = np.asarray(archive["panel_translations"], dtype=float)
    rotations = np.asarray(archive["panel_rotations"], dtype=float)
    pairs = np.asarray(archive["stitch_pairs"], dtype=int).reshape(-1, 4)
    panels: list[Panel] = []
    sides: dict[tuple[int, int], StitchSide] = {}

    for panel_index in np.flatnonzero(valid.any(axis=1)):
        cursor = np.zeros(2, dtype=float)
        generated_edges: list[Edge] = []
        for edge_index in np.flatnonzero(valid[panel_index]):
            vector = edges[panel_index, edge_index]
            points = sample_generated_edge(cursor, vector)
            edge_id = f"panel_{panel_index}.edge_{edge_index}"
            generated_edges.append(
                Edge(
                    edge_id,
                    tuple((float(point[0]), float(point[1])) for point in points),
                    source_curve_id=int(edge_index),
                )
            )
            sides[(int(panel_index), int(edge_index))] = StitchSide(f"panel_{panel_index}", edge_id)
            cursor += vector[:2]
        origin = tuple(float(value) for value in translations[panel_index])
        panels.append(
            Panel(
                f"panel_{panel_index}",
                tuple(generated_edges),
                Placement(origin, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), "predicted_panel_transform"),
                source_panel_id=int(panel_index),
            )
        )

    stitches = []
    for stitch_index, (panel_a, edge_a, panel_b, edge_b) in enumerate(pairs):
        first, second = sides[(int(panel_a), int(edge_a))], sides[(int(panel_b), int(edge_b))]
        stitches.append(Stitch(f"stitch_{stitch_index}", first, second))
    return PatternDocument(
        pattern_id=path.parent.name,
        generator="Garment Particles",
        panels=tuple(panels),
        stitches=tuple(stitches),
        provenance={"source_artifact_sha256": sha256(path), "source_format": "garment_particles_npz"},
        annotations={
            "topology": "model_generated_variable",
            "template_retrieval": False,
            "panel_rotations": rotations[valid.any(axis=1)].tolist(),
            "unit_interpretation": "upstream GarmentCode training coordinate treated as centimetres",
            "stitch_direction": "not_predicted",
        },
    )
