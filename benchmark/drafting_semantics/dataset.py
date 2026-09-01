from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .schema import EDGE_ROLES, PANEL_ROLES, DraftingSemanticRecord, PanelAnnotation


CURVATURE_TYPES = ("line", "quadratic", "cubic", "circle")
EDGE_FEATURE_DIM = 17


@dataclass(frozen=True)
class PanelExample:
    sample_id: str
    split: str
    panel_id: str
    panel_role_id: int
    panel: PanelAnnotation
    features: np.ndarray
    targets: np.ndarray
    edge_indices: np.ndarray


def read_records(path: Path) -> tuple[DraftingSemanticRecord, ...]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        raw = json.loads(path.read_text(encoding="utf-8"))
        values = raw if isinstance(raw, list) else raw.get("records", [raw])
    return tuple(DraftingSemanticRecord.from_dict(value) for value in values)


def edge_features(panel: PanelAnnotation, *, include_stitch_features: bool = False) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(panel.vertices_cm, dtype=np.float32)
    center = vertices.mean(axis=0)
    span = np.ptp(vertices, axis=0)
    scale = float(max(span.max(), 1e-6))
    features = np.zeros((len(panel.edges), EDGE_FEATURE_DIM), dtype=np.float32)
    targets = np.zeros(len(panel.edges), dtype=np.int64)
    for index, edge in enumerate(panel.edges):
        start = (np.asarray(edge.start_cm, dtype=np.float32) - center) / scale
        end = (np.asarray(edge.end_cm, dtype=np.float32) - center) / scale
        delta = end - start
        angle = math.atan2(float(delta[1]), float(delta[0]))
        phase = 2.0 * math.pi * index / max(len(panel.edges), 1)
        curvature = np.zeros(4, dtype=np.float32)
        curvature[CURVATURE_TYPES.index(edge.curvature_type) if edge.curvature_type in CURVATURE_TYPES else 0] = 1.0
        stitched = float(edge.stitched) if include_stitch_features else 0.0
        self_stitched = float(edge.self_stitched) if include_stitch_features else 0.0
        features[index] = np.asarray(
            [
                start[0],
                start[1],
                end[0],
                end[1],
                delta[0],
                delta[1],
                edge.length_cm / scale,
                math.sin(angle),
                math.cos(angle),
                *curvature,
                stitched,
                self_stitched,
                math.sin(phase),
                math.cos(phase),
            ],
            dtype=np.float32,
        )
        targets[index] = EDGE_ROLES.index(edge.role)
    return features, targets


def panel_examples(
    records: Iterable[DraftingSemanticRecord],
    *,
    splits: set[str] | None = None,
    bodice_only: bool = True,
    include_stitch_features: bool = False,
) -> tuple[PanelExample, ...]:
    output = []
    for record in records:
        if splits is not None and record.split not in splits:
            continue
        for panel in record.panels:
            if bodice_only and panel.role not in {"front_bodice", "back_bodice"}:
                continue
            features, targets = edge_features(panel, include_stitch_features=include_stitch_features)
            if not len(features):
                continue
            output.append(
                PanelExample(
                    sample_id=record.sample_id,
                    split=record.split,
                    panel_id=panel.id,
                    panel_role_id=PANEL_ROLES.index(panel.role),
                    panel=panel,
                    features=features,
                    targets=targets,
                    edge_indices=np.arange(len(targets), dtype=np.int64),
                )
            )
    return tuple(output)


def reindex_panel_example(example: PanelExample, *, shift: int = 0, reverse: bool = False) -> PanelExample:
    """Change the arbitrary boundary start/direction without changing semantics.

    Vector-pattern formats disagree on which edge is serialized first and on
    clockwise versus counter-clockwise winding.  Treating those choices as
    data augmentation is necessary before evaluating across CAD sources.
    """

    count = len(example.targets)
    if not count:
        return example
    order = np.arange(count, dtype=np.int64)
    if reverse:
        order = order[::-1]
    order = np.roll(order, -(int(shift) % count))
    features = example.features[order].copy()
    targets = example.targets[order].copy()
    edge_indices = example.edge_indices[order].copy()
    if reverse:
        start = features[:, 0:2].copy()
        features[:, 0:2] = features[:, 2:4]
        features[:, 2:4] = start
        features[:, 4:6] *= -1.0
        features[:, 7:9] *= -1.0
    phase = 2.0 * math.pi * np.arange(count, dtype=np.float32) / count
    features[:, 15] = np.sin(phase)
    features[:, 16] = np.cos(phase)
    return PanelExample(
        sample_id=example.sample_id,
        split=example.split,
        panel_id=example.panel_id,
        panel_role_id=example.panel_role_id,
        panel=example.panel,
        features=features,
        targets=targets,
        edge_indices=edge_indices,
    )


def augment_boundary_serialization(
    examples: Iterable[PanelExample], *, variants: int, seed: int
) -> tuple[PanelExample, ...]:
    """Add deterministic random start-edge/winding variants for training."""

    values = tuple(examples)
    if variants <= 0:
        return values
    generator = np.random.default_rng(seed)
    output = list(values)
    for example in values:
        for variant in range(variants):
            output.append(
                reindex_panel_example(
                    example,
                    shift=int(generator.integers(max(len(example.targets), 1))),
                    reverse=bool((variant + int(generator.integers(2))) % 2),
                )
            )
    return tuple(output)


def padded_batch(examples: Iterable[PanelExample], maximum_edges: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = tuple(examples)
    features = np.zeros((len(values), maximum_edges, EDGE_FEATURE_DIM), dtype=np.float32)
    targets = np.full((len(values), maximum_edges), -100, dtype=np.int64)
    valid = np.zeros((len(values), maximum_edges), dtype=bool)
    panel_roles = np.zeros(len(values), dtype=np.int64)
    for row, example in enumerate(values):
        count = min(len(example.features), maximum_edges)
        features[row, :count] = example.features[:count]
        targets[row, :count] = example.targets[:count]
        valid[row, :count] = True
        panel_roles[row] = example.panel_role_id
    return features, targets, valid, panel_roles


def balanced_class_weights(examples: Iterable[PanelExample]) -> np.ndarray:
    counts = np.ones(len(EDGE_ROLES), dtype=np.float64)
    for example in examples:
        counts += np.bincount(example.targets, minlength=len(EDGE_ROLES))
    weights = 1.0 / np.sqrt(counts)
    weights /= weights.mean()
    return weights.astype(np.float32)
