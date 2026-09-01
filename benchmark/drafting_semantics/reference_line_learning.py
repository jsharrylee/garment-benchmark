from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .dataset import EDGE_FEATURE_DIM, edge_features
from .schema import PANEL_ROLES, DraftingSemanticRecord, PanelAnnotation


REFERENCE_LINES = ("BL", "WL")


@dataclass(frozen=True)
class ReferenceLineExample:
    sample_id: str
    split: str
    panel: PanelAnnotation
    features: np.ndarray
    panel_role_id: int
    target_normalized_y: np.ndarray
    center_y_cm: float
    scale_cm: float


def reference_line_examples(records: Iterable[DraftingSemanticRecord], splits: set[str]) -> tuple[ReferenceLineExample, ...]:
    output = []
    for record in records:
        if record.split not in splits:
            continue
        for panel in record.panels:
            if panel.role not in {"front_bodice", "back_bodice"}:
                continue
            eligible = {line.name: line for line in panel.reference_lines if line.training_eligible}
            if not all(name in eligible for name in REFERENCE_LINES):
                continue
            vertices = np.asarray(panel.vertices_cm, np.float32)
            center_y = float(vertices[:, 1].mean())
            scale = float(max(np.ptp(vertices, axis=0).max(), 1e-6))
            features, _ = edge_features(panel, include_stitch_features=False)
            targets = np.asarray([(eligible[name].points_cm[0][1] - center_y) / scale for name in REFERENCE_LINES], np.float32)
            output.append(ReferenceLineExample(record.sample_id, record.split, panel, features, PANEL_ROLES.index(panel.role), targets, center_y, scale))
    return tuple(output)


def collate_reference_lines(examples: Iterable[ReferenceLineExample], maximum_edges: int = 40):
    import torch

    values = tuple(examples)
    features = np.zeros((len(values), maximum_edges, EDGE_FEATURE_DIM), np.float32)
    valid = np.zeros((len(values), maximum_edges), bool)
    roles = np.zeros(len(values), np.int64)
    targets = np.zeros((len(values), len(REFERENCE_LINES)), np.float32)
    centers = np.zeros(len(values), np.float32)
    scales = np.zeros(len(values), np.float32)
    for row, example in enumerate(values):
        count = min(len(example.features), maximum_edges)
        features[row, :count] = example.features[:count]
        valid[row, :count] = True
        roles[row] = example.panel_role_id
        targets[row] = example.target_normalized_y
        centers[row] = example.center_y_cm
        scales[row] = example.scale_cm
    return {
        "features": torch.from_numpy(features),
        "valid": torch.from_numpy(valid),
        "panel_roles": torch.from_numpy(roles),
        "targets": torch.from_numpy(targets),
        "centers_cm": torch.from_numpy(centers),
        "scales_cm": torch.from_numpy(scales),
    }


def build_reference_line_model(width: int = 96, heads: int = 4, layers: int = 2):
    import torch

    class ReferenceLineTransformer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.project = torch.nn.Linear(EDGE_FEATURE_DIM, width)
            self.role = torch.nn.Embedding(len(PANEL_ROLES), width)
            layer = torch.nn.TransformerEncoderLayer(width, heads, width * 3, 0.1, batch_first=True, norm_first=True, activation="gelu")
            self.encoder = torch.nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.head = torch.nn.Sequential(torch.nn.LayerNorm(width), torch.nn.Linear(width, width), torch.nn.GELU(), torch.nn.Linear(width, len(REFERENCE_LINES)))

        def forward(self, features, valid, panel_roles):
            hidden = self.project(features) + self.role(panel_roles)[:, None]
            hidden = self.encoder(hidden, src_key_padding_mask=~valid)
            weight = valid[:, :, None].to(hidden.dtype)
            pooled = (hidden * weight).sum(1) / weight.sum(1).clamp_min(1)
            return self.head(pooled)

    return ReferenceLineTransformer()


__all__ = ["REFERENCE_LINES", "build_reference_line_model", "collate_reference_lines", "reference_line_examples"]
