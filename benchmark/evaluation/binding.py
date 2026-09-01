from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compare_reweaver_outputs(baseline_path: Path, candidate_path: Path, *, tolerance: float = 1e-6) -> dict:
    baseline = np.load(baseline_path, allow_pickle=True)
    candidate = np.load(candidate_path, allow_pickle=True)
    base_patch = np.asarray(baseline["patch_points"], dtype=np.float64)
    cand_patch = np.asarray(candidate["patch_points"], dtype=np.float64)
    base_curve = np.asarray(baseline["curve_points"], dtype=np.float64)
    cand_curve = np.asarray(candidate["curve_points"], dtype=np.float64)

    patch_rms = None
    curve_rms = None
    if base_patch.shape == cand_patch.shape:
        patch_rms = float(np.sqrt(np.mean(np.square(base_patch - cand_patch))))
    if base_curve.shape == cand_curve.shape:
        curve_rms = float(np.sqrt(np.mean(np.square(base_curve - cand_curve))))

    base_panels = int(np.count_nonzero(baseline["patch_valid_prob"] >= 0.5))
    cand_panels = int(np.count_nonzero(candidate["patch_valid_prob"] >= 0.5))
    base_curves = int(np.count_nonzero(baseline["curve_valid_prob"] >= 0.5))
    cand_curves = int(np.count_nonzero(candidate["curve_valid_prob"] >= 0.5))
    changed = (
        sha256(baseline_path) != sha256(candidate_path)
        and (
            base_panels != cand_panels
            or base_curves != cand_curves
            or (patch_rms is not None and patch_rms > tolerance)
            or (curve_rms is not None and curve_rms > tolerance)
        )
    )
    return {
        "valid": changed,
        "failure": None if changed else "OUTPUT_NOT_BOUND_TO_INPUT",
        "tolerance": tolerance,
        "baseline_sha256": sha256(baseline_path),
        "candidate_sha256": sha256(candidate_path),
        "panel_counts": [base_panels, cand_panels],
        "curve_counts": [base_curves, cand_curves],
        "patch_rms": patch_rms,
        "curve_rms": curve_rms,
    }


def compare_garment_particles_outputs(
    baseline_path: Path,
    candidate_path: Path,
    *,
    tolerance: float = 1e-6,
) -> dict:
    baseline = np.load(baseline_path)
    candidate = np.load(candidate_path)
    base_particles = np.asarray(baseline["particles"], dtype=np.float64)
    cand_particles = np.asarray(candidate["particles"], dtype=np.float64)
    base_edges = np.asarray(baseline["edge_raw_normalized"], dtype=np.float64)
    cand_edges = np.asarray(candidate["edge_raw_normalized"], dtype=np.float64)
    particle_rms = float(np.sqrt(np.mean(np.square(base_particles - cand_particles))))
    edge_rms = float(np.sqrt(np.mean(np.square(base_edges - cand_edges))))
    base_mask = np.asarray(baseline["edge_valid_mask"], dtype=bool)
    cand_mask = np.asarray(candidate["edge_valid_mask"], dtype=bool)
    base_panels = int(base_mask.any(axis=1).sum())
    cand_panels = int(cand_mask.any(axis=1).sum())
    base_edges_count = int(base_mask.sum())
    cand_edges_count = int(cand_mask.sum())
    changed = sha256(baseline_path) != sha256(candidate_path) and (
        particle_rms > tolerance
        or edge_rms > tolerance
        or base_panels != cand_panels
        or base_edges_count != cand_edges_count
    )
    return {
        "valid": changed,
        "failure": None if changed else "OUTPUT_NOT_BOUND_TO_INPUT",
        "tolerance": tolerance,
        "baseline_sha256": sha256(baseline_path),
        "candidate_sha256": sha256(candidate_path),
        "panel_counts": [base_panels, cand_panels],
        "edge_counts": [base_edges_count, cand_edges_count],
        "particle_rms": particle_rms,
        "edge_rms": edge_rms,
    }
