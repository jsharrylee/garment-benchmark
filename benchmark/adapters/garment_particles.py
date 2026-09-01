from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


GARMENT_PARTICLES_COMMIT = "328e9735a46b7638cf6f0348a85c5a146b16cd69"
MODEL_REVISION = "2c463e6cc48cb3263d1e0e1e052a3ec0fc760db6"
N_POINTS = 8196
N_PANELS = 37
N_CURVES = 38
N_EDGE_PARAMS = 15
PCD_MEAN = np.array([1.02373453, 0.53175375, 0.0537376513, 86.1824935, -1.33205849, 0.0])
PCD_STD = np.array([0.51201646, 0.13567009, 17.43921292, 36.19954752, 9.72246543, 1.0])
UV_MIN = np.array([-150.0, -80.0])
UV_MAX = np.array([150.0, 220.0])
TRANSFORM_MEAN = np.array([145.16962989, 86.01913585, -0.0125378371, 113.507532, 2.63046369, 0.0, 0.0, 0.0])
TRANSFORM_STD = np.array([157.87811579, 42.82879761, 26.06867645, 32.42920198, 22.29905009, 1.0, 1.0, 1.0])
EDGE_MEAN = np.array([0.0, 0.0, 0.0989319516, -0.0017836434, 0.1501257727, -0.0017253332, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0537376513, 86.1824935, -1.33205849])
EDGE_STD = np.array([21.5768255296, 18.1631555269, 0.1818133799, 0.1216707505, 0.2674056122, 0.1216692602, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 17.43921292, 36.19954752, 9.72246543])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_predictions(particles_normalized: np.ndarray, edge_normalized: np.ndarray) -> dict[str, np.ndarray]:
    particles = np.asarray(particles_normalized, dtype=np.float64) * PCD_STD + PCD_MEAN
    particles[:, :2] = particles[:, :2] * (UV_MAX - UV_MIN) + UV_MIN
    raw = np.asarray(edge_normalized, dtype=np.float64).reshape(N_PANELS, N_CURVES, N_EDGE_PARAMS)
    metadata = raw[:, 0, :8] * TRANSFORM_STD + TRANSFORM_MEAN
    normalized_edges = raw[:, 1:, :]
    valid = normalized_edges[..., 7] > 0.5
    edges = normalized_edges * EDGE_STD + EDGE_MEAN
    attachment_bits = edges[..., 8:11] > 0.5
    attachment_types = np.sum(attachment_bits.astype(np.int8) * np.array([4, 2, 1]), axis=-1)
    stitch_flags = edges[..., 11] > 0.5
    return {
        "particles_normalized": np.asarray(particles_normalized),
        "particles": particles,
        "edge_raw_normalized": raw,
        "edges": edges[..., :7],
        "edge_valid_mask": valid,
        "panel_shifts": metadata[:, :2],
        "panel_translations": metadata[:, 2:5],
        "panel_rotations": metadata[:, 5:8],
        "attachment_types": attachment_types,
        "stitch_flags": stitch_flags,
        "stitch_tags": edges[..., 12:15],
    }


def stitch_pairs(stitch_flags: np.ndarray, stitch_tags: np.ndarray, valid: np.ndarray) -> np.ndarray:
    import networkx as nx

    candidates = [tuple(map(int, item)) for item in np.argwhere(stitch_flags & valid)]
    graph = nx.Graph()
    for left, first in enumerate(candidates):
        for second in candidates[left + 1 :]:
            graph.add_edge(first, second, weight=float(np.linalg.norm(stitch_tags[first] - stitch_tags[second])))
    matching = nx.min_weight_matching(graph)
    pairs = [[first[0], first[1], second[0], second[1]] for first, second in matching]
    return np.asarray(pairs, dtype=np.int16).reshape(-1, 4)


def summarize_output(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        return {"valid": False, "failure": "EMPTY_OUTPUT"}
    archive = np.load(path)
    required = {"particles", "edges", "edge_valid_mask", "panel_translations", "panel_rotations", "stitch_pairs"}
    missing = sorted(required - set(archive.files))
    if missing:
        return {"valid": False, "failure": "INVALID_PATTERN_STRUCTURE", "missing_keys": missing}
    for key in required - {"edge_valid_mask"}:
        if archive[key].size == 0 and key != "stitch_pairs":
            return {"valid": False, "failure": "EMPTY_OUTPUT", "key": key}
        if not np.isfinite(archive[key]).all():
            return {"valid": False, "failure": "NONFINITE_OUTPUT", "key": key}
    valid_mask = archive["edge_valid_mask"].astype(bool)
    panel_count = int(valid_mask.any(axis=1).sum())
    edge_count = int(valid_mask.sum())
    particles = archive["particles"]
    if panel_count <= 0 or edge_count <= 0 or particles.shape != (N_POINTS, 6):
        return {"valid": False, "failure": "INVALID_PATTERN_STRUCTURE", "panel_count": panel_count, "edge_count": edge_count}
    pairs = archive["stitch_pairs"]
    if pairs.size:
        if pairs.ndim != 2 or pairs.shape[1] != 4 or np.any(pairs < 0) or np.any(pairs[:, [0, 2]] >= N_PANELS) or np.any(pairs[:, [1, 3]] >= N_CURVES - 1):
            return {"valid": False, "failure": "INVALID_STITCH_REFERENCE"}
        referenced_valid = valid_mask[pairs[:, 0], pairs[:, 1]] & valid_mask[pairs[:, 2], pairs[:, 3]]
        if not referenced_valid.all():
            return {"valid": False, "failure": "INVALID_STITCH_REFERENCE"}
    closure_gaps = np.array(
        [np.linalg.norm(archive["edges"][panel, valid_mask[panel], :2].sum(axis=0)) for panel in np.flatnonzero(valid_mask.any(axis=1))]
    )
    return {
        "valid": True,
        "failure": None,
        "particle_count": int(len(particles)),
        "boundary_particle_count": int(np.count_nonzero(particles[:, -1] >= 0.5)),
        "panel_count": panel_count,
        "edge_count": edge_count,
        "stitch_pair_count": int(len(pairs)),
        "panel_closure_gap_mean": float(closure_gaps.mean()),
        "panel_closure_gap_max": float(closure_gaps.max()),
        "translation_bbox_min": archive["panel_translations"][valid_mask.any(axis=1)].min(axis=0).tolist(),
        "translation_bbox_max": archive["panel_translations"][valid_mask.any(axis=1)].max(axis=0).tolist(),
        "particle_bbox_min": particles[:, 2:5].min(axis=0).tolist(),
        "particle_bbox_max": particles[:, 2:5].max(axis=0).tolist(),
        "output_sha256": sha256(path),
    }


def write_ascii_ply(path: Path, particles: np.ndarray) -> None:
    xyz = np.asarray(particles)[:, 2:5]
    boundary = np.asarray(particles)[:, -1] >= 0.5
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {len(xyz)}\n")
        stream.write("property float x\nproperty float y\nproperty float z\n")
        stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for point, is_boundary in zip(xyz, boundary, strict=True):
            color = (220, 50, 47) if is_boundary else (38, 139, 210)
            stream.write(f"{point[0]:.8g} {point[1]:.8g} {point[2]:.8g} {color[0]} {color[1]} {color[2]}\n")


def write_summary(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
