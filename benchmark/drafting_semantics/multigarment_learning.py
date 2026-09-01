"""Unified multi-garment vector-pattern learning utilities.

The representation is deliberately hierarchical: primitive curves are encoded
inside panels, then panel tokens communicate at garment level.  This preserves
cross-panel physical length ratios (for example sleeve-cap to armhole) while
keeping source names, operation ids, and target labels out of model inputs.
"""

from __future__ import annotations

import gzip
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .dataset import CURVATURE_TYPES, edge_features as gcd_edge_features
from .schema import DraftingSemanticRecord, EDGE_ROLES, PANEL_ROLES, PanelAnnotation
from .tshirt_learning import (
    CURVE_KINDS,
    FEATURE_SLICES,
    panel_geometry_features,
    read_tshirt_records,
)
from .tshirt_schema import TShirtTraceRecord, TracedPanel


GARMENT_ROLES = ("top", "pants", "skirt", "dress", "jumpsuit")
MULTIGARMENT_EDGE_ROLES = tuple(EDGE_ROLES)
MULTIGARMENT_PANEL_ROLES = tuple(PANEL_ROLES)

# 17 source-agnostic local geometry values plus two garment-wide scale values.
LOCAL_EDGE_FEATURE_DIM = 17
EDGE_FEATURE_DIM = 19


@dataclass(frozen=True)
class MultiPanelExample:
    panel_id: str
    panel_target: int
    features: np.ndarray
    edge_targets: np.ndarray
    edge_lengths_cm: np.ndarray
    edge_ids: tuple[str, ...]
    panel_scale_cm: float


@dataclass(frozen=True)
class MultiGarmentExample:
    sample_id: str
    split: str
    source: str
    garment_target: int
    panels: tuple[MultiPanelExample, ...]


def _category_from_record(record: DraftingSemanticRecord) -> str:
    upper = record.program.get("upper_type")
    bottom = record.program.get("design_values", {}).get("meta.bottom")
    if upper and bottom == "Pants":
        return "jumpsuit"
    if upper and bottom:
        return "dress"
    if upper:
        return "top"
    if bottom == "Pants":
        return "pants"
    return "skirt"


def _canonical_split(value: str) -> str:
    normalized = str(value).lower()
    if normalized in {"training", "train"}:
        return "train"
    if normalized == "validation" or (
        "validation" in normalized and "test" not in normalized
    ):
        return "validation"
    if normalized == "test" or "test" in normalized:
        return "test"
    return "auxiliary"


def _scale_of_vertices(vertices: Sequence[Sequence[float]]) -> float:
    values = np.asarray(vertices, dtype=np.float32)
    if not len(values):
        return 1.0
    return float(max(np.ptp(values, axis=0).max(), 1e-6))


def _extend_features(local: np.ndarray, lengths_cm: np.ndarray, panel_scale: float, garment_scale: float) -> np.ndarray:
    output = np.zeros((len(local), EDGE_FEATURE_DIM), dtype=np.float32)
    output[:, :LOCAL_EDGE_FEATURE_DIM] = local
    output[:, 17] = lengths_cm / max(garment_scale, 1e-6)
    output[:, 18] = panel_scale / max(garment_scale, 1e-6)
    return output


def _gcd_panel(panel: PanelAnnotation, garment_scale: float) -> MultiPanelExample:
    local, raw_targets = gcd_edge_features(panel, include_stitch_features=False)
    lengths = np.asarray([edge.length_cm for edge in panel.edges], dtype=np.float32)
    panel_scale = _scale_of_vertices(panel.vertices_cm)
    targets = raw_targets.copy()
    # An unlabeled boundary is not a professionally verified `other` edge.
    # Mask it so the network is not rewarded for learning missing annotations.
    for index, edge in enumerate(panel.edges):
        if edge.role == "other":
            targets[index] = -100
    return MultiPanelExample(
        panel_id=panel.id,
        panel_target=MULTIGARMENT_PANEL_ROLES.index(panel.role),
        features=_extend_features(local, lengths, panel_scale, garment_scale),
        edge_targets=targets.astype(np.int64),
        edge_lengths_cm=lengths,
        edge_ids=tuple(edge.id for edge in panel.edges),
        panel_scale_cm=panel_scale,
    )


def _teagan_local_features(panel: TracedPanel) -> tuple[np.ndarray, np.ndarray, float]:
    rich, lengths, _, scale = panel_geometry_features(panel)
    local = np.zeros((len(rich), LOCAL_EDGE_FEATURE_DIM), dtype=np.float32)
    local[:, 0:6] = rich[:, 0:6]
    local[:, 6] = rich[:, FEATURE_SLICES["length"]]
    local[:, 7] = rich[:, FEATURE_SLICES["direction_sin"]]
    local[:, 8] = rich[:, FEATURE_SLICES["direction_cos"]]
    for row in range(len(rich)):
        source_kind = int(np.argmax(rich[row, FEATURE_SLICES["curve_kind"]]))
        kind = CURVE_KINDS[source_kind]
        target_kind = {
            "line": "line",
            "quadratic_bezier": "quadratic",
            "cubic_bezier": "cubic",
            "bezier": "cubic",
            "arc": "circle",
            "other": "line",
        }[kind]
        local[row, 9 + CURVATURE_TYPES.index(target_kind)] = 1.0
        phase = 2.0 * math.pi * row / max(len(rich), 1)
        local[row, 15] = math.sin(phase)
        local[row, 16] = math.cos(phase)
    return local, lengths.astype(np.float32), float(scale)


_TEAGAN_PANEL_ROLE = {"front": "front_bodice", "back": "back_bodice", "sleeve": "sleeve", "neckband": "collar"}


def _teagan_panel(panel: TracedPanel, garment_scale: float) -> MultiPanelExample:
    local, lengths, panel_scale = _teagan_local_features(panel)
    edge_targets = []
    for edge in panel.edges:
        role = str(edge.semantic_role)
        edge_targets.append(MULTIGARMENT_EDGE_ROLES.index(role) if role in MULTIGARMENT_EDGE_ROLES else -100)
    role = _TEAGAN_PANEL_ROLE.get(str(panel.semantic_role), str(panel.semantic_role))
    return MultiPanelExample(
        panel_id=panel.id,
        panel_target=MULTIGARMENT_PANEL_ROLES.index(role) if role in MULTIGARMENT_PANEL_ROLES else 0,
        features=_extend_features(local, lengths, panel_scale, garment_scale),
        edge_targets=np.asarray(edge_targets, dtype=np.int64),
        edge_lengths_cm=lengths,
        edge_ids=tuple(edge.id for edge in panel.edges),
        panel_scale_cm=panel_scale,
    )


def read_gcd_multigarment_examples(path: Path) -> tuple[MultiGarmentExample, ...]:
    examples: list[MultiGarmentExample] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = DraftingSemanticRecord.from_dict(json.loads(line))
            garment_scale = max((_scale_of_vertices(panel.vertices_cm) for panel in record.panels), default=1.0)
            panels = tuple(_gcd_panel(panel, garment_scale) for panel in record.panels if panel.edges)
            if panels:
                category = _category_from_record(record)
                examples.append(
                    MultiGarmentExample(
                        sample_id=record.sample_id,
                        split=_canonical_split(record.split),
                        source="garmentcode_v2",
                        garment_target=GARMENT_ROLES.index(category),
                        panels=panels,
                    )
                )
    return tuple(examples)


def read_teagan_multigarment_examples(path: Path) -> tuple[MultiGarmentExample, ...]:
    output: list[MultiGarmentExample] = []
    for record in read_tshirt_records(path):
        scales = []
        for panel in record.panels:
            _, _, _, scale = panel_geometry_features(panel)
            scales.append(scale)
        garment_scale = max(scales, default=1.0)
        panels = tuple(_teagan_panel(panel, garment_scale) for panel in record.panels if panel.edges)
        output.append(
            MultiGarmentExample(
                sample_id=record.sample_id,
                split=_canonical_split(record.split),
                source="freesewing_teagan_4.10.1",
                garment_target=GARMENT_ROLES.index("top"),
                panels=panels,
            )
        )
    return tuple(output)


def randomize_boundary_serialization(example: MultiGarmentExample, generator: np.random.Generator) -> MultiGarmentExample:
    panels = []
    for panel in example.panels:
        count = len(panel.features)
        if count <= 1:
            panels.append(panel)
            continue
        reverse = bool(generator.integers(2))
        shift = int(generator.integers(count))
        order = np.arange(count - 1, -1, -1) if reverse else np.arange(count)
        order = np.roll(order, -shift)
        features = panel.features[order].copy()
        targets = panel.edge_targets[order].copy()
        lengths = panel.edge_lengths_cm[order].copy()
        edge_ids = np.asarray(panel.edge_ids, dtype=object)[order]
        if reverse:
            start = features[:, 0:2].copy()
            features[:, 0:2] = features[:, 2:4]
            features[:, 2:4] = start
            features[:, 4:6] *= -1.0
            features[:, 7:9] *= -1.0
        phase = 2.0 * math.pi * np.arange(count, dtype=np.float32) / count
        features[:, 15] = np.sin(phase)
        features[:, 16] = np.cos(phase)
        panels.append(
            replace(
                panel,
                features=features,
                edge_targets=targets,
                edge_lengths_cm=lengths,
                edge_ids=tuple(str(value) for value in edge_ids.tolist()),
            )
        )
    return replace(example, panels=tuple(panels))


def padded_garment_batch(
    examples: Sequence[MultiGarmentExample], *, maximum_panels: int, maximum_edges: int
) -> dict[str, Any]:
    oversized_panels = [(item.sample_id, len(item.panels)) for item in examples if len(item.panels) > maximum_panels]
    oversized_edges = [
        (item.sample_id, panel.panel_id, len(panel.features))
        for item in examples
        for panel in item.panels
        if len(panel.features) > maximum_edges
    ]
    if oversized_panels or oversized_edges:
        raise ValueError(f"padding limits would truncate patterns: panels={oversized_panels[:3]} edges={oversized_edges[:3]}")
    batch = len(examples)
    features = np.zeros((batch, maximum_panels, maximum_edges, EDGE_FEATURE_DIM), dtype=np.float32)
    edge_targets = np.full((batch, maximum_panels, maximum_edges), -100, dtype=np.int64)
    edge_valid = np.zeros((batch, maximum_panels, maximum_edges), dtype=bool)
    panel_targets = np.full((batch, maximum_panels), -100, dtype=np.int64)
    panel_valid = np.zeros((batch, maximum_panels), dtype=bool)
    same_path = np.zeros((batch, maximum_panels, maximum_edges), dtype=np.float32)
    same_path_mask = np.zeros((batch, maximum_panels, maximum_edges), dtype=bool)
    panel_presence = np.zeros((batch, len(MULTIGARMENT_PANEL_ROLES)), dtype=np.float32)
    garment_targets = np.zeros(batch, dtype=np.int64)
    seam_ratio = np.zeros(batch, dtype=np.float32)
    seam_ratio_mask = np.zeros(batch, dtype=bool)
    for row, example in enumerate(examples):
        garment_targets[row] = example.garment_target
        armhole_length = 0.0
        sleeve_head_length = 0.0
        for panel_index, panel in enumerate(example.panels):
            count = len(panel.features)
            features[row, panel_index, :count] = panel.features
            edge_targets[row, panel_index, :count] = panel.edge_targets
            edge_valid[row, panel_index, :count] = True
            panel_targets[row, panel_index] = panel.panel_target
            panel_valid[row, panel_index] = True
            panel_presence[row, panel.panel_target] = 1.0
            for edge_index in range(count):
                following = (edge_index + 1) % count
                first = int(panel.edge_targets[edge_index])
                second = int(panel.edge_targets[following])
                if first >= 0 and second >= 0:
                    same_path[row, panel_index, edge_index] = float(first == second)
                    same_path_mask[row, panel_index, edge_index] = True
                if first >= 0:
                    role = MULTIGARMENT_EDGE_ROLES[first]
                    if role == "armhole":
                        armhole_length += float(panel.edge_lengths_cm[edge_index])
                    elif role == "sleeve_head":
                        sleeve_head_length += float(panel.edge_lengths_cm[edge_index])
        if armhole_length > 1e-6 and sleeve_head_length > 1e-6:
            seam_ratio[row] = sleeve_head_length / armhole_length
            seam_ratio_mask[row] = True
    return {
        "features": features,
        "edge_targets": edge_targets,
        "edge_valid": edge_valid,
        "panel_targets": panel_targets,
        "panel_valid": panel_valid,
        "garment_targets": garment_targets,
        "same_path_targets": same_path,
        "same_path_mask": same_path_mask,
        "panel_presence_targets": panel_presence,
        "seam_ratio_targets": seam_ratio,
        "seam_ratio_mask": seam_ratio_mask,
        "sample_ids": tuple(item.sample_id for item in examples),
        "sources": tuple(item.source for item in examples),
    }


class InspectableEncoderLayer:
    """Factory namespace so torch remains an optional import for data tools."""

    @staticmethod
    def build(width: int, heads: int, feedforward: int, dropout: float):
        import torch

        class Layer(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm1 = torch.nn.LayerNorm(width)
                self.attention = torch.nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
                self.norm2 = torch.nn.LayerNorm(width)
                self.feedforward = torch.nn.Sequential(
                    torch.nn.Linear(width, feedforward),
                    torch.nn.GELU(),
                    torch.nn.Dropout(dropout),
                    torch.nn.Linear(feedforward, width),
                )
                self.dropout = torch.nn.Dropout(dropout)

            def forward(self, hidden, padding_mask, capture_attention: bool = False):
                normalized = self.norm1(hidden)
                attended, weights = self.attention(
                    normalized,
                    normalized,
                    normalized,
                    key_padding_mask=padding_mask,
                    need_weights=capture_attention,
                    average_attn_weights=False,
                )
                hidden = hidden + self.dropout(attended)
                hidden = hidden + self.dropout(self.feedforward(self.norm2(hidden)))
                return hidden, weights if capture_attention else None

        return Layer()


def build_multigarment_model(config: Mapping[str, Any]):
    import torch

    class GarmentGraphTransformer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            width = int(config["width"])
            heads = int(config["heads"])
            feedforward = width * int(config["feedforward_multiplier"])
            dropout = float(config["dropout"])
            self.feature_projection = torch.nn.Linear(EDGE_FEATURE_DIM, width)
            self.local_class = torch.nn.Parameter(torch.zeros(1, 1, width))
            self.garment_class = torch.nn.Parameter(torch.zeros(1, 1, width))
            self.local_layers = torch.nn.ModuleList(
                InspectableEncoderLayer.build(width, heads, feedforward, dropout)
                for _ in range(int(config["local_layers"]))
            )
            self.global_layers = torch.nn.ModuleList(
                InspectableEncoderLayer.build(width, heads, feedforward, dropout)
                for _ in range(int(config["global_layers"]))
            )
            self.edge_head = torch.nn.Sequential(
                torch.nn.LayerNorm(width), torch.nn.Linear(width, width), torch.nn.GELU(), torch.nn.Linear(width, len(MULTIGARMENT_EDGE_ROLES))
            )
            self.panel_head = torch.nn.Sequential(
                torch.nn.LayerNorm(width), torch.nn.Linear(width, len(MULTIGARMENT_PANEL_ROLES))
            )
            self.garment_head = torch.nn.Sequential(torch.nn.LayerNorm(width), torch.nn.Linear(width, len(GARMENT_ROLES)))
            self.panel_presence_head = torch.nn.Sequential(
                torch.nn.LayerNorm(width), torch.nn.Linear(width, len(MULTIGARMENT_PANEL_ROLES))
            )
            self.same_path_head = torch.nn.Sequential(
                torch.nn.LayerNorm(width * 2), torch.nn.Linear(width * 2, width), torch.nn.GELU(), torch.nn.Linear(width, 1)
            )
            self.seam_ratio_head = torch.nn.Sequential(
                torch.nn.LayerNorm(width), torch.nn.Linear(width, width // 2), torch.nn.GELU(), torch.nn.Linear(width // 2, 1)
            )
            torch.nn.init.normal_(self.local_class, std=0.02)
            torch.nn.init.normal_(self.garment_class, std=0.02)

        def forward(self, features, edge_valid, panel_valid, *, capture_attention: bool = False):
            batch, panels, edges, _ = features.shape
            hidden = self.feature_projection(features).reshape(batch * panels, edges, -1)
            local_class = self.local_class.expand(batch * panels, -1, -1)
            hidden = torch.cat((local_class, hidden), dim=1)
            flattened_edge_valid = edge_valid.reshape(batch * panels, edges)
            local_padding = torch.cat(
                (torch.zeros((batch * panels, 1), dtype=torch.bool, device=features.device), ~flattened_edge_valid),
                dim=1,
            )
            local_attention = []
            for layer in self.local_layers:
                hidden, weights = layer(hidden, local_padding, capture_attention)
                if capture_attention:
                    local_attention.append(weights)
            panel_hidden = hidden[:, 0].reshape(batch, panels, -1)
            edge_hidden = hidden[:, 1:].reshape(batch, panels, edges, -1)

            garment_class = self.garment_class.expand(batch, -1, -1)
            global_hidden = torch.cat((garment_class, panel_hidden), dim=1)
            global_padding = torch.cat(
                (torch.zeros((batch, 1), dtype=torch.bool, device=features.device), ~panel_valid), dim=1
            )
            global_attention = []
            for layer in self.global_layers:
                global_hidden, weights = layer(global_hidden, global_padding, capture_attention)
                if capture_attention:
                    global_attention.append(weights)
            garment_hidden = global_hidden[:, 0]
            contextual_panels = global_hidden[:, 1:]
            contextual_edges = edge_hidden + contextual_panels[:, :, None, :]
            following = torch.roll(contextual_edges, shifts=-1, dims=2)
            same_path_input = torch.cat((contextual_edges, following), dim=-1)
            return {
                "edge_logits": self.edge_head(contextual_edges),
                "panel_logits": self.panel_head(contextual_panels),
                "garment_logits": self.garment_head(garment_hidden),
                "panel_presence_logits": self.panel_presence_head(garment_hidden),
                "same_path_logits": self.same_path_head(same_path_input).squeeze(-1),
                "seam_ratio": self.seam_ratio_head(garment_hidden).squeeze(-1),
                "local_attention": local_attention,
                "global_attention": global_attention,
            }

    return GarmentGraphTransformer()


__all__ = [
    "EDGE_FEATURE_DIM",
    "GARMENT_ROLES",
    "MULTIGARMENT_EDGE_ROLES",
    "MULTIGARMENT_PANEL_ROLES",
    "MultiGarmentExample",
    "MultiPanelExample",
    "build_multigarment_model",
    "padded_garment_batch",
    "randomize_boundary_serialization",
    "read_gcd_multigarment_examples",
    "read_teagan_multigarment_examples",
]
