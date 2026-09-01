from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np

from benchmark.gcdv2_exact.intrinsic_graph_learning import (
    _cyclic_source_direction,
    intrinsic_segment_features,
    nearest_contour_indices,
    segment_between,
)
from .dataset import read_records


MAXIMUM_VISIBLE_EDGES = 36
MERGED_EDGE_ROLES = ("other", "neckline", "shoulder", "armhole", "center_front", "center_back", "side_seam", "waistline", "dart_leg")
PANEL_ROLES = ("front_bodice", "back_bodice")
LANDMARK_NAMES = ("FNP", "BNP", "SNP", "SP")


def _merged_role(values: Sequence[str]) -> str:
    semantic = {value for value in values if value != "other" and value in MERGED_EDGE_ROLES}
    return next(iter(semantic)) if len(semantic) == 1 else "other"


def _source_adjacency_roles(graph: Mapping[str, Any], panel) -> dict[frozenset[int], str]:
    visible = [index for index, point in enumerate(graph["points"]) if point["visual_supervision_eligible"]]
    point_count = len(graph["points"])
    output = {}
    for local, start in enumerate(visible):
        end = visible[(local + 1) % len(visible)]
        roles = []
        current = start
        while current != end:
            source_edge = int(graph["curves"][current]["source_edge_index"])
            roles.append(panel.edges[source_edge].role)
            current = (current + 1) % point_count
            if len(roles) > point_count:
                raise ValueError("semantic visible adjacency failed to close")
        output[frozenset((start, end))] = _merged_role(roles)
    return output


def _merged_role_between(graph: Mapping[str, Any], panel, start: int, end: int, direction: int) -> str:
    roles = []
    current = start
    point_count = len(graph["points"])
    while current != end:
        curve_index = current if direction > 0 else (current - 1) % point_count
        source_edge = int(graph["curves"][curve_index]["source_edge_index"])
        roles.append(panel.edges[source_edge].role)
        current = (current + direction) % point_count
        if len(roles) > point_count:
            raise ValueError("semantic source path failed to close")
    return _merged_role(roles)


def build_merged_visible_arrays(
    panel_rows: Sequence[Mapping[str, Any]], predicted_contours: np.ndarray, records_path: Path
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    row_lookup = {str(row["panel_uid"]): (index, row) for index, row in enumerate(panel_rows)}
    records = read_records(records_path)
    features_rows, role_rows, valid_rows, vertex_rows = [], [], [], []
    panel_roles, splits, panel_indices = [], [], []
    landmark_uv_rows, landmark_mask_rows, metadata = [], [], []
    split_lookup = {"training": 0, "validation": 1, "test": 2}
    for record in records:
        if record.split not in split_lookup:
            continue
        for panel in record.panels:
            if panel.role not in PANEL_ROLES:
                continue
            key = f"{record.sample_id}:{panel.id}"
            if key not in row_lookup:
                continue
            panel_index, row = row_lookup[key]
            contour = np.asarray(predicted_contours[panel_index], np.float32)
            graph = json.loads(Path(row["formal_graph_path"]).read_text(encoding="utf-8"))
            visible = [(index, point) for index, point in enumerate(graph["points"]) if point["visual_supervision_eligible"]]
            indices = nearest_contour_indices(contour, [point["uv"] for _, point in visible])
            order = np.argsort(indices)
            indices = indices[order]
            source_indices = [visible[value][0] for value in order]
            keep = np.concatenate(([True], np.diff(indices) > 0))
            indices = indices[keep]
            source_indices = [value for value, accepted in zip(source_indices, keep, strict=True) if accepted]
            if len(indices) < 3 or len(indices) > MAXIMUM_VISIBLE_EDGES:
                continue
            visible_graph_indices = [index for index, point in enumerate(graph["points"]) if point["visual_supervision_eligible"]]
            source_direction = _cyclic_source_direction(source_indices, visible_graph_indices)
            panel_features = np.zeros((MAXIMUM_VISIBLE_EDGES, 32, 8), np.float16)
            panel_targets = np.full(MAXIMUM_VISIBLE_EDGES, -1, np.int8)
            panel_valid = np.zeros(MAXIMUM_VISIBLE_EDGES, bool)
            panel_vertices = np.zeros((MAXIMUM_VISIBLE_EDGES, 2), np.float32)
            for local, start in enumerate(indices):
                end = int(indices[(local + 1) % len(indices)])
                segment = segment_between(contour, int(start), end)
                segment_features, _ = intrinsic_segment_features(segment)
                panel_features[local] = segment_features.astype(np.float16)
                role = _merged_role_between(
                    graph, panel, source_indices[local], source_indices[(local + 1) % len(indices)], source_direction
                )
                panel_targets[local] = MERGED_EDGE_ROLES.index(role)
                panel_valid[local] = True
                panel_vertices[local] = contour[int(start)]
            point_by_source_vertex = {int(point["source_vertex_index"]): point for point in graph["points"]}
            landmark_uv = np.zeros((len(LANDMARK_NAMES), 2), np.float32)
            landmark_mask = np.zeros(len(LANDMARK_NAMES), bool)
            for landmark in panel.landmarks:
                if landmark.name not in LANDMARK_NAMES or not landmark.training_eligible or landmark.vertex_index is None:
                    continue
                point = point_by_source_vertex.get(int(landmark.vertex_index))
                if point is None or not point["visual_supervision_eligible"]:
                    continue
                landmark_index = LANDMARK_NAMES.index(landmark.name)
                landmark_uv[landmark_index] = point["uv"]
                landmark_mask[landmark_index] = True
            features_rows.append(panel_features)
            role_rows.append(panel_targets)
            valid_rows.append(panel_valid)
            vertex_rows.append(panel_vertices)
            panel_roles.append(PANEL_ROLES.index(panel.role))
            splits.append(split_lookup[record.split])
            panel_indices.append(panel_index)
            landmark_uv_rows.append(landmark_uv)
            landmark_mask_rows.append(landmark_mask)
            metadata.append({"sample_id": record.sample_id, "panel_id": panel.id, "panel_uid": key, "panel_role": panel.role, "split": record.split, "source_panel_index": panel_index})
    arrays = {
        "segment_features": np.asarray(features_rows, np.float16),
        "edge_roles": np.asarray(role_rows, np.int8),
        "valid_edges": np.asarray(valid_rows, bool),
        "vertices_uv": np.asarray(vertex_rows, np.float32),
        "panel_roles": np.asarray(panel_roles, np.int8),
        "splits": np.asarray(splits, np.int8),
        "source_panel_indices": np.asarray(panel_indices, np.int32),
        "landmark_uv": np.asarray(landmark_uv_rows, np.float32),
        "landmark_mask": np.asarray(landmark_mask_rows, bool),
    }
    return arrays, metadata


class MergedVisibleDataset:
    def __init__(self, arrays: Mapping[str, np.ndarray], indices: Sequence[int], *, augment: bool) -> None:
        self.arrays, self.indices, self.augment = arrays, np.asarray(indices), augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        source = int(self.indices[index])
        valid_count = int(self.arrays["valid_edges"][source].sum())
        features = self.arrays["segment_features"][source, :valid_count].astype(np.float32)
        roles = self.arrays["edge_roles"][source, :valid_count].astype(np.int64)
        vertices = self.arrays["vertices_uv"][source, :valid_count].astype(np.float32)
        if self.augment and random.random() < 0.5:
            # Reverse graph direction and re-express each segment in its new
            # unit-chord frame; role identity is direction invariant.
            features = np.asarray([intrinsic_segment_features(value[::-1, :2])[0] for value in features[::-1]], np.float32)
            roles = roles[::-1].copy()
            vertices = np.roll(vertices[::-1], -1, axis=0).copy()
        if self.augment:
            shift = random.randrange(valid_count)
            features, roles, vertices = np.roll(features, shift, axis=0), np.roll(roles, shift), np.roll(vertices, shift, axis=0)
        padded_features = np.zeros((MAXIMUM_VISIBLE_EDGES, 32, 8), np.float32)
        padded_roles = np.full(MAXIMUM_VISIBLE_EDGES, -100, np.int64)
        valid = np.zeros(MAXIMUM_VISIBLE_EDGES, bool)
        padded_vertices = np.zeros((MAXIMUM_VISIBLE_EDGES, 2), np.float32)
        padded_features[:valid_count], padded_roles[:valid_count], valid[:valid_count], padded_vertices[:valid_count] = features, roles, True, vertices
        return {
            "features": padded_features, "roles": padded_roles, "valid": valid, "vertices_uv": padded_vertices,
            "panel_role": int(self.arrays["panel_roles"][source]), "landmark_uv": self.arrays["landmark_uv"][source].astype(np.float32),
            "landmark_mask": self.arrays["landmark_mask"][source], "source": source,
        }


def build_merged_semantic_model(width: int = 128, heads: int = 4, segment_layers: int = 1, graph_layers: int = 3):
    import torch

    class MergedVisibleSemanticNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.point_project = torch.nn.Linear(8, width)
            point_layer = torch.nn.TransformerEncoderLayer(width, heads, width * 3, 0.1, batch_first=True, norm_first=True, activation="gelu")
            self.segment_encoder = torch.nn.TransformerEncoder(point_layer, segment_layers, enable_nested_tensor=False)
            self.panel_role = torch.nn.Embedding(len(PANEL_ROLES), width)
            graph_layer = torch.nn.TransformerEncoderLayer(width, heads, width * 3, 0.1, batch_first=True, norm_first=True, activation="gelu")
            self.graph_encoder = torch.nn.TransformerEncoder(graph_layer, graph_layers, enable_nested_tensor=False)
            self.head = torch.nn.Sequential(torch.nn.LayerNorm(width), torch.nn.Linear(width, width), torch.nn.GELU(), torch.nn.Linear(width, len(MERGED_EDGE_ROLES)))

        def forward(self, features, valid, panel_roles):
            batch, edges, points = features.shape[:3]
            segment = self.segment_encoder(self.point_project(features.reshape(batch * edges, points, 8))).mean(1).reshape(batch, edges, -1)
            segment = segment + self.panel_role(panel_roles)[:, None]
            graph = self.graph_encoder(segment, src_key_padding_mask=~valid)
            return self.head(graph)

    return MergedVisibleSemanticNet()


def decode_landmarks(roles: Sequence[int], vertices_uv: np.ndarray, panel_role: int) -> dict[str, np.ndarray]:
    names = [MERGED_EDGE_ROLES[int(value)] for value in roles]
    requests = [(("neckline", "center_front" if panel_role == 0 else "center_back"), "FNP" if panel_role == 0 else "BNP"), (("neckline", "shoulder"), "SNP"), (("shoulder", "armhole"), "SP")]
    output = {}
    for (first, second), landmark in requests:
        for vertex in range(len(names)):
            adjacent = {names[(vertex - 1) % len(names)], names[vertex]}
            if adjacent == {first, second}:
                output[landmark] = np.asarray(vertices_uv[vertex], np.float32)
                break
    return output


__all__ = [
    "LANDMARK_NAMES", "MAXIMUM_VISIBLE_EDGES", "MERGED_EDGE_ROLES", "MergedVisibleDataset", "PANEL_ROLES",
    "build_merged_semantic_model", "build_merged_visible_arrays", "decode_landmarks",
]
