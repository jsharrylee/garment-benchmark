"""Topology-preserving retrieved-pattern residual learning for GCDv2.

The module implements the final stage of the retrieval-first inverse-pattern
experiment:

``four-view FPN tokens + an actually retrieved compatible pattern``
    -> ``per-vertex and native-curve-parameter corrections``.

Two details are deliberately non-negotiable:

* an anchor is selected using *visual features only* from the training bank;
  target geometry is never consulted by retrieval, and the target itself is
  always excluded;
* shared edge endpoints are represented by one panel vertex.  The network
  predicts one delta for that vertex and edge endpoints are derived from the
  corrected vertex array.  It therefore cannot tear two incident edges apart.

Curve type, endpoint incidence, stitches, panel inventory, and circular-arc
flags are part of the topology signature.  They are copied from the anchor
instead of being reclassified.  Only Bezier controls and arc radius are
regressed when the fixed native type has such parameters.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from benchmark.gcdv2_exact.geometry import CURVE_TYPES, load_exact_label, sample_curve


CATEGORY_NAMES = ("top", "skirt", "pants")
CATEGORY_TO_INDEX = {name: index for index, name in enumerate(CATEGORY_NAMES)}
CURVE_TO_INDEX = {name: index for index, name in enumerate(CURVE_TYPES)}
COORDINATE_SCALE_CM = 100.0
CURVE_PARAMETER_DIM = 5

# The existing cache was extracted in filename order CAM000..CAM003.  The
# corrected semantic contract established by the exact-pair dataset is
# front=CAM001, back=CAM000, left=CAM002, right=CAM003.
FPN_CACHE_TO_SEMANTIC_VIEW_ORDER = (1, 0, 2, 3)
SEMANTIC_VIEW_NAMES = ("front", "back", "left", "right")


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _ordered_panels(label: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return sorted(label["panels"], key=lambda panel: int(panel["source_order_index"]))


def topology_contract(label: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact structure that must agree before residual editing.

    Geometry values (coordinates, lengths, controls, radius) are intentionally
    absent.  Panel identity/order, endpoint incidence, native curve type/arc
    branch, and stitch incidence are retained so flattened slots align without
    a learned or heuristic correspondence.
    """

    panels = []
    for panel in _ordered_panels(label):
        edges = []
        for edge in sorted(panel["edges"], key=lambda value: int(value["edge_index"])):
            curve = edge["curve"]
            arc = curve.get("arc", {})
            edges.append(
                {
                    "edge_index": int(edge["edge_index"]),
                    "endpoints": [int(value) for value in edge["endpoints"]],
                    "curve_type": str(curve["type"]),
                    "arc_large": bool(arc.get("large_arc", False)),
                    "arc_right": bool(arc.get("right", False)),
                    "arc_sweep_y_up": bool(arc.get("sweep_y_up", False)),
                }
            )
        panels.append(
            {
                "panel_id": str(panel["panel_id"]),
                "source_order_index": int(panel["source_order_index"]),
                "source_label": panel.get("source_label"),
                "vertex_count": len(panel["vertices_cm"]),
                "edges": edges,
            }
        )
    stitches = []
    for pair in label.get("stitches", []):
        normalized_pair = []
        for side in pair:
            if isinstance(side, Mapping) and "panel" in side and "edge" in side:
                normalized_pair.append(
                    {"panel": str(side["panel"]), "edge": int(side["edge"])}
                )
            else:
                # A small number of official samples attach a source marker
                # such as ``right_wrong`` as a third stitch item.  It remains
                # part of the compatibility signature without pretending to be
                # a sewable side.
                normalized_pair.append({"source_metadata": side})
        stitches.append(normalized_pair)
    return {
        "category": str(label["category"]),
        "panels": panels,
        "stitches": stitches,
    }


def topology_hash(label: Mapping[str, Any]) -> str:
    return _stable_hash(topology_contract(label))


def reorder_cached_fpn_views(features: np.ndarray) -> np.ndarray:
    """Map cached CAM000..CAM003 features to front/back/left/right."""

    array = np.asarray(features)
    if array.ndim < 2 or array.shape[-4 if array.ndim >= 4 else 0] == 0:
        raise ValueError(f"invalid FPN feature shape: {array.shape}")
    # Accepted shapes are [4,T,D] and [N,4,T,D].
    view_axis = 0 if array.ndim == 3 else 1 if array.ndim == 4 else None
    if view_axis is None or array.shape[view_axis] != 4:
        raise ValueError(f"expected four cached views, got {array.shape}")
    return np.take(array, FPN_CACHE_TO_SEMANTIC_VIEW_ORDER, axis=view_axis)


def _curve_parameters(edge: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Encode native geometry without conflating type with parameters.

    Slots are ``control_1.xy, control_2.xy, radius``.  A mask records which
    entries have geometric meaning for the fixed curve type.
    """

    values = np.zeros(CURVE_PARAMETER_DIM, dtype=np.float32)
    mask = np.zeros(CURVE_PARAMETER_DIM, dtype=bool)
    curve = edge["curve"]
    kind = str(curve["type"])
    controls = curve.get("controls_cm", [])
    if kind == "quadratic_bezier":
        values[:2] = np.asarray(controls[0], dtype=np.float32)
        mask[:2] = True
    elif kind == "cubic_bezier":
        values[:4] = np.asarray(controls[:2], dtype=np.float32).reshape(-1)
        mask[:4] = True
    elif kind == "circular_arc":
        values[4] = float(curve["arc"]["radius_cm"])
        mask[4] = True
    elif kind != "line":
        raise ValueError(f"unsupported exact curve type: {kind!r}")
    return values, mask


@dataclass(frozen=True)
class ExactGeometryRecord:
    sample_id: str
    category: str
    label_path: Path
    pattern_path: Path
    topology_hash: str
    topology: Mapping[str, Any]
    panel_ids: tuple[str, ...]
    vertices_cm: np.ndarray
    vertex_panel_indices: np.ndarray
    vertex_local_indices: np.ndarray
    edges: np.ndarray
    edge_panel_indices: np.ndarray
    edge_local_indices: np.ndarray
    curve_types: np.ndarray
    curve_parameters_cm: np.ndarray
    curve_parameter_mask: np.ndarray
    spatial_features: np.ndarray

    @property
    def visual_descriptor(self) -> np.ndarray:
        value = self.spatial_features.astype(np.float32).mean(axis=(0, 1))
        norm = float(np.linalg.norm(value))
        return value / max(norm, 1e-12)


@dataclass(frozen=True)
class RetrievedResidualPair:
    target: ExactGeometryRecord
    anchor: ExactGeometryRecord
    split: str
    visual_cosine_similarity: float

    def validate(self, training_ids: set[str]) -> None:
        if self.target.sample_id == self.anchor.sample_id:
            raise ValueError("retrieval selected the target itself")
        if self.anchor.sample_id not in training_ids:
            raise ValueError("anchor is not in the training bank")
        if self.target.topology_hash != self.anchor.topology_hash:
            raise ValueError("anchor topology is incompatible with target")
        if self.target.vertices_cm.shape != self.anchor.vertices_cm.shape:
            raise ValueError("compatible topology has misaligned vertex slots")
        if not np.array_equal(self.target.edges, self.anchor.edges):
            raise ValueError("compatible topology has different endpoint incidence")
        if not np.array_equal(self.target.curve_types, self.anchor.curve_types):
            raise ValueError("compatible topology has different curve types")


def _record_from_label(
    row: Mapping[str, Any], label: Mapping[str, Any], spatial_features: np.ndarray
) -> ExactGeometryRecord:
    vertices: list[list[float]] = []
    vertex_panel: list[int] = []
    vertex_local: list[int] = []
    edges: list[list[int]] = []
    edge_panel: list[int] = []
    edge_local: list[int] = []
    curve_types: list[int] = []
    curve_parameters: list[np.ndarray] = []
    curve_masks: list[np.ndarray] = []
    panel_ids = []
    for panel_index, panel in enumerate(_ordered_panels(label)):
        panel_ids.append(str(panel["panel_id"]))
        offset = len(vertices)
        current_vertices = np.asarray(panel["vertices_cm"], dtype=np.float32)
        vertices.extend(current_vertices.tolist())
        vertex_panel.extend([panel_index] * len(current_vertices))
        vertex_local.extend(range(len(current_vertices)))
        for edge in sorted(panel["edges"], key=lambda value: int(value["edge_index"])):
            start, end = (int(value) for value in edge["endpoints"])
            edges.append([offset + start, offset + end])
            edge_panel.append(panel_index)
            edge_local.append(int(edge["edge_index"]))
            curve_types.append(CURVE_TO_INDEX[str(edge["curve"]["type"])])
            values, mask = _curve_parameters(edge)
            curve_parameters.append(values)
            curve_masks.append(mask)
    return ExactGeometryRecord(
        sample_id=str(row["sample_id"]),
        category=str(row["category"]),
        label_path=Path(row["label_path"]),
        pattern_path=Path(row["pattern_path"]),
        topology_hash=topology_hash(label),
        topology=topology_contract(label),
        panel_ids=tuple(panel_ids),
        vertices_cm=np.asarray(vertices, dtype=np.float32),
        vertex_panel_indices=np.asarray(vertex_panel, dtype=np.int64),
        vertex_local_indices=np.asarray(vertex_local, dtype=np.int64),
        edges=np.asarray(edges, dtype=np.int64),
        edge_panel_indices=np.asarray(edge_panel, dtype=np.int64),
        edge_local_indices=np.asarray(edge_local, dtype=np.int64),
        curve_types=np.asarray(curve_types, dtype=np.int64),
        curve_parameters_cm=np.asarray(curve_parameters, dtype=np.float32),
        curve_parameter_mask=np.asarray(curve_masks, dtype=bool),
        spatial_features=np.asarray(spatial_features),
    )


def read_exact_geometry_records(
    index_path: Path,
    feature_path: Path,
    *,
    maximum_records: int | None = None,
    strict_features: bool = False,
) -> tuple[ExactGeometryRecord, ...]:
    """Read exact labels and the existing four-view FPN cache.

    The cache is explicitly reordered to the corrected semantic view order.
    Missing feature rows are excluded (or rejected with ``strict_features``);
    they are never silently replaced with zeros, which would turn the
    experiment into an anchor-only model.  The caller must report the resulting
    feature coverage.
    """

    rows = [
        json.loads(line)
        for line in Path(index_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if maximum_records is not None:
        rows = rows[: int(maximum_records)]
    archive = np.load(feature_path, allow_pickle=False)
    feature_ids = [str(value) for value in archive["sample_ids"].tolist()]
    feature_values = archive["features"]
    lookup = {sample_id: index for index, sample_id in enumerate(feature_ids)}
    records = []
    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id not in lookup:
            if strict_features:
                raise KeyError(f"four-view FPN cache has no row for {sample_id}")
            continue
        label = load_exact_label(Path(row["label_path"]))
        if label.get("validation", {}).get("status") != "PASS":
            continue
        features = reorder_cached_fpn_views(feature_values[lookup[sample_id]])
        records.append(_record_from_label(row, label, features))
    return tuple(records)


def deterministic_topology_split(
    records: Sequence[ExactGeometryRecord], *, seed: int = 20260829
) -> dict[str, str]:
    """Create a deterministic topology-stratified train/validation/test split.

    Every pairable held-out topology receives at least one training-bank item.
    Singleton topologies remain in ``train`` but are reported as uncovered and
    never used as residual targets because no non-self compatible anchor exists.
    """

    groups: dict[str, list[ExactGeometryRecord]] = defaultdict(list)
    for record in records:
        groups[record.topology_hash].append(record)
    result: dict[str, str] = {}
    for signature, group in sorted(groups.items()):
        ordered = sorted(
            group,
            key=lambda item: _stable_hash(
                {"seed": int(seed), "topology": signature, "sample_id": item.sample_id}
            ),
        )
        count = len(ordered)
        if count == 1:
            train_count, validation_count = 1, 0
        elif count == 2:
            train_count, validation_count = 1, 0
        elif count == 3:
            train_count, validation_count = 1, 1
        else:
            train_count = max(2, int(round(0.70 * count)))
            train_count = min(train_count, count - 2)
            validation_count = max(1, int(round(0.15 * count)))
            validation_count = min(validation_count, count - train_count - 1)
        for index, record in enumerate(ordered):
            if index < train_count:
                split = "train"
            elif index < train_count + validation_count:
                split = "validation"
            else:
                split = "test"
            result[record.sample_id] = split
    return result


def load_crossmodal_embedding_bank(
    path: Path,
) -> tuple[dict[str, str], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load Stage-2 split assignments and normalized image/pattern embeddings."""

    archive = np.load(Path(path), allow_pickle=False)
    sample_ids = [str(value) for value in archive["sample_ids"].tolist()]
    splits = [str(value) for value in archive["splits"].tolist()]
    image = np.asarray(archive["image_embeddings"], dtype=np.float32)
    pattern = np.asarray(archive["pattern_embeddings"], dtype=np.float32)
    if not (len(sample_ids) == len(splits) == len(image) == len(pattern)):
        raise ValueError("cross-modal embedding arrays have inconsistent lengths")
    return (
        dict(zip(sample_ids, splits)),
        {sample_id: image[index] for index, sample_id in enumerate(sample_ids)},
        {sample_id: pattern[index] for index, sample_id in enumerate(sample_ids)},
    )


def build_visual_retrieval_pairs(
    records: Sequence[ExactGeometryRecord],
    split_by_id: Mapping[str, str],
    *,
    crossmodal_image_embeddings: Mapping[str, np.ndarray] | None = None,
    crossmodal_pattern_embeddings: Mapping[str, np.ndarray] | None = None,
) -> tuple[tuple[RetrievedResidualPair, ...], dict[str, Any]]:
    """Retrieve nearest non-self, same-topology anchors from the train bank.

    With Stage-2 embeddings, the score is four-view-image to 2D-pattern
    similarity, so the residual stage consumes the actual preceding model's
    output.  Mean-FPN cosine is retained only as an explicit fallback.
    """

    use_crossmodal = (
        crossmodal_image_embeddings is not None
        and crossmodal_pattern_embeddings is not None
    )

    groups: dict[str, list[ExactGeometryRecord]] = defaultdict(list)
    for record in records:
        groups[record.topology_hash].append(record)
    training_ids = {
        record.sample_id
        for record in records
        if split_by_id.get(record.sample_id) == "train"
    }
    pairs: list[RetrievedResidualPair] = []
    uncovered = []
    split_target_counts: dict[str, int] = defaultdict(int)
    split_pair_counts: dict[str, int] = defaultdict(int)
    for target in records:
        split = str(split_by_id[target.sample_id])
        split_target_counts[split] += 1
        candidates = [
            value
            for value in groups[target.topology_hash]
            if value.sample_id in training_ids and value.sample_id != target.sample_id
        ]
        if not candidates:
            uncovered.append(
                {
                    "sample_id": target.sample_id,
                    "category": target.category,
                    "split": split,
                    "topology_hash": target.topology_hash,
                    "reason": "NO_NON_SELF_SAME_TOPOLOGY_TRAIN_ANCHOR",
                }
            )
            continue
        if use_crossmodal:
            if target.sample_id not in crossmodal_image_embeddings:
                raise KeyError(f"Stage-2 image embedding missing for {target.sample_id}")
            descriptor = crossmodal_image_embeddings[target.sample_id]
            missing = [
                candidate.sample_id
                for candidate in candidates
                if candidate.sample_id not in crossmodal_pattern_embeddings
            ]
            if missing:
                raise KeyError(f"Stage-2 pattern embedding missing for {missing[0]}")
            scored = [
                (
                    float(np.dot(descriptor, crossmodal_pattern_embeddings[candidate.sample_id])),
                    candidate.sample_id,
                    candidate,
                )
                for candidate in candidates
            ]
        else:
            descriptor = target.visual_descriptor
            scored = [
                (float(np.dot(descriptor, candidate.visual_descriptor)), candidate.sample_id, candidate)
                for candidate in candidates
            ]
        similarity, _, anchor = max(scored, key=lambda item: (item[0], item[1]))
        pair = RetrievedResidualPair(target, anchor, split, similarity)
        pair.validate(training_ids)
        pairs.append(pair)
        split_pair_counts[split] += 1
    audit = {
        "selection": (
            "highest Stage-2 four-view-image to 2D-pattern embedding cosine within exact topology"
            if use_crossmodal
            else "highest cosine similarity of mean four-view FPN descriptors within exact topology"
        ),
        "uses_trained_stage2_crossmodal_retrieval": use_crossmodal,
        "selection_uses_target_geometry": False,
        "selection_uses_target_topology_contract": True,
        "deployment_requirement": "a preceding retrieval/topology classifier must supply a compatible topology; this evaluation uses the exact target topology as a compatibility gate",
        "anchor_bank_split": "train",
        "target_self_excluded": True,
        "same_topology_required": True,
        "total_records": len(records),
        "topology_group_count": len(groups),
        "non_singleton_topology_records": sum(
            len(group) for group in groups.values() if len(group) > 1
        ),
        "target_counts": dict(split_target_counts),
        "pair_counts": dict(split_pair_counts),
        "covered_records": len(pairs),
        "coverage": len(pairs) / max(len(records), 1),
        "uncovered_count": len(uncovered),
        "uncovered": uncovered,
    }
    return tuple(pairs), audit


def batch_residual_pairs(
    pairs: Sequence[RetrievedResidualPair],
    *,
    maximum_vertices: int,
    maximum_edges: int,
    coordinate_scale_cm: float = COORDINATE_SCALE_CM,
) -> dict[str, Any]:
    """Pad aligned topology pairs into numpy tensors."""

    batch_size = len(pairs)
    if not batch_size:
        raise ValueError("cannot batch an empty pair sequence")
    token_shape = pairs[0].target.spatial_features.shape
    visual = np.empty((batch_size, *token_shape), dtype=np.float32)
    anchor_vertices = np.zeros((batch_size, maximum_vertices, 2), dtype=np.float32)
    target_vertices = np.zeros_like(anchor_vertices)
    vertex_mask = np.zeros((batch_size, maximum_vertices), dtype=bool)
    vertex_panel = np.zeros((batch_size, maximum_vertices), dtype=np.int64)
    vertex_local = np.zeros((batch_size, maximum_vertices), dtype=np.int64)
    anchor_curve = np.zeros((batch_size, maximum_edges, CURVE_PARAMETER_DIM), dtype=np.float32)
    target_curve = np.zeros_like(anchor_curve)
    curve_parameter_mask = np.zeros_like(anchor_curve, dtype=bool)
    edge_mask = np.zeros((batch_size, maximum_edges), dtype=bool)
    edge_vertices = np.zeros((batch_size, maximum_edges, 2), dtype=np.int64)
    edge_panel = np.zeros((batch_size, maximum_edges), dtype=np.int64)
    edge_local = np.zeros((batch_size, maximum_edges), dtype=np.int64)
    curve_types = np.zeros((batch_size, maximum_edges), dtype=np.int64)
    category = np.zeros(batch_size, dtype=np.int64)
    for batch_index, pair in enumerate(pairs):
        target, anchor = pair.target, pair.anchor
        vertex_count, edge_count = len(target.vertices_cm), len(target.edges)
        if vertex_count > maximum_vertices or edge_count > maximum_edges:
            raise ValueError("batch maxima are smaller than a record")
        if target.spatial_features.shape != token_shape:
            raise ValueError("inconsistent FPN token shapes")
        visual[batch_index] = target.spatial_features.astype(np.float32)
        anchor_vertices[batch_index, :vertex_count] = anchor.vertices_cm / coordinate_scale_cm
        target_vertices[batch_index, :vertex_count] = target.vertices_cm / coordinate_scale_cm
        vertex_mask[batch_index, :vertex_count] = True
        vertex_panel[batch_index, :vertex_count] = target.vertex_panel_indices
        vertex_local[batch_index, :vertex_count] = target.vertex_local_indices
        anchor_curve[batch_index, :edge_count] = anchor.curve_parameters_cm / coordinate_scale_cm
        target_curve[batch_index, :edge_count] = target.curve_parameters_cm / coordinate_scale_cm
        curve_parameter_mask[batch_index, :edge_count] = target.curve_parameter_mask
        edge_mask[batch_index, :edge_count] = True
        edge_vertices[batch_index, :edge_count] = target.edges
        edge_panel[batch_index, :edge_count] = target.edge_panel_indices
        edge_local[batch_index, :edge_count] = target.edge_local_indices
        curve_types[batch_index, :edge_count] = target.curve_types
        category[batch_index] = CATEGORY_TO_INDEX[target.category]
    return {
        "visual_features": visual,
        "anchor_vertices": anchor_vertices,
        "target_vertices": target_vertices,
        "vertex_mask": vertex_mask,
        "vertex_panel_indices": vertex_panel,
        "vertex_local_indices": vertex_local,
        "anchor_curve_parameters": anchor_curve,
        "target_curve_parameters": target_curve,
        "curve_parameter_mask": curve_parameter_mask,
        "edge_mask": edge_mask,
        "edge_vertices": edge_vertices,
        "edge_panel_indices": edge_panel,
        "edge_local_indices": edge_local,
        "curve_types": curve_types,
        "category": category,
        "target_ids": [pair.target.sample_id for pair in pairs],
        "anchor_ids": [pair.anchor.sample_id for pair in pairs],
    }


def build_retrieved_residual_model(config: Mapping[str, Any]):
    import torch

    width = int(config.get("width", 128))
    heads = int(config.get("heads", 4))
    layers = int(config.get("decoder_layers", 2))
    dropout = float(config.get("dropout", 0.10))
    visual_dimension = int(config.get("visual_feature_dimension", 256))
    maximum_visual_tokens = int(config.get("maximum_visual_tokens_per_view", 85))
    maximum_panels = int(config["maximum_panels"])
    maximum_local_vertices = int(config["maximum_local_vertices"])
    maximum_local_edges = int(config["maximum_local_edges"])

    class RetrievedPatternResidualTransformer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.visual_projection = torch.nn.Linear(visual_dimension, width)
            self.view_embedding = torch.nn.Embedding(4, width)
            self.visual_position_embedding = torch.nn.Embedding(maximum_visual_tokens, width)
            self.category_embedding = torch.nn.Embedding(len(CATEGORY_NAMES), width)
            self.panel_embedding = torch.nn.Embedding(maximum_panels, width)
            self.vertex_embedding = torch.nn.Embedding(maximum_local_vertices, width)
            self.edge_embedding = torch.nn.Embedding(maximum_local_edges, width)
            self.curve_type_embedding = torch.nn.Embedding(len(CURVE_TYPES), width)
            self.vertex_input = torch.nn.Linear(2, width)
            self.edge_input = torch.nn.Linear(9, width)
            decoder_layer = lambda: torch.nn.TransformerDecoderLayer(
                d_model=width,
                nhead=heads,
                dim_feedforward=width * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.vertex_decoder = torch.nn.TransformerDecoder(decoder_layer(), layers)
            self.edge_decoder = torch.nn.TransformerDecoder(decoder_layer(), layers)
            self.vertex_delta = torch.nn.Sequential(
                torch.nn.LayerNorm(width), torch.nn.Linear(width, width), torch.nn.GELU(), torch.nn.Linear(width, 2)
            )
            self.curve_delta = torch.nn.Sequential(
                torch.nn.LayerNorm(width), torch.nn.Linear(width, width), torch.nn.GELU(), torch.nn.Linear(width, CURVE_PARAMETER_DIM)
            )
            # Epoch zero exactly reproduces the retrieved anchor.  This makes
            # "did learning improve over retrieval?" a literal checkpoint gate.
            torch.nn.init.zeros_(self.vertex_delta[-1].weight)
            torch.nn.init.zeros_(self.vertex_delta[-1].bias)
            torch.nn.init.zeros_(self.curve_delta[-1].weight)
            torch.nn.init.zeros_(self.curve_delta[-1].bias)

        def forward(
            self,
            *,
            visual_features,
            anchor_vertices,
            vertex_mask,
            vertex_panel_indices,
            vertex_local_indices,
            anchor_curve_parameters,
            edge_mask,
            edge_vertices,
            edge_panel_indices,
            edge_local_indices,
            curve_types,
            category,
        ):
            batch, views, tokens, _ = visual_features.shape
            if views != 4 or tokens > maximum_visual_tokens:
                raise ValueError(f"unsupported visual token shape: {visual_features.shape}")
            memory = self.visual_projection(visual_features)
            view_ids = torch.arange(views, device=memory.device).view(1, views, 1)
            token_ids = torch.arange(tokens, device=memory.device).view(1, 1, tokens)
            memory = memory + self.view_embedding(view_ids) + self.visual_position_embedding(token_ids)
            memory = memory.reshape(batch, views * tokens, width)
            category_token = self.category_embedding(category).unsqueeze(1)
            vertex_query = (
                self.vertex_input(anchor_vertices)
                + self.panel_embedding(vertex_panel_indices)
                + self.vertex_embedding(vertex_local_indices)
                + category_token
            )
            decoded_vertices = self.vertex_decoder(
                vertex_query,
                memory,
                tgt_key_padding_mask=~vertex_mask,
            )
            predicted_vertices = anchor_vertices + self.vertex_delta(decoded_vertices)
            gather_start = edge_vertices[..., 0].unsqueeze(-1).expand(-1, -1, 2)
            gather_end = edge_vertices[..., 1].unsqueeze(-1).expand(-1, -1, 2)
            anchor_start = torch.gather(anchor_vertices, 1, gather_start)
            anchor_end = torch.gather(anchor_vertices, 1, gather_end)
            edge_values = torch.cat((anchor_start, anchor_end, anchor_curve_parameters), dim=-1)
            edge_query = (
                self.edge_input(edge_values)
                + self.panel_embedding(edge_panel_indices)
                + self.edge_embedding(edge_local_indices)
                + self.curve_type_embedding(curve_types)
                + category_token
            )
            decoded_edges = self.edge_decoder(
                edge_query,
                memory,
                tgt_key_padding_mask=~edge_mask,
            )
            predicted_curve = anchor_curve_parameters + self.curve_delta(decoded_edges)
            return {
                "predicted_vertices": predicted_vertices,
                "predicted_curve_parameters": predicted_curve,
                "vertex_delta": predicted_vertices - anchor_vertices,
                "curve_parameter_delta": predicted_curve - anchor_curve_parameters,
            }

    return RetrievedPatternResidualTransformer()


def residual_loss(
    output: Mapping[str, Any],
    batch: Mapping[str, Any],
    *,
    vertex_weight: float = 1.0,
    curve_weight: float = 0.50,
    chord_length_weight: float = 0.20,
    direction_weight: float = 0.10,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    vertex_mask = batch["vertex_mask"]
    curve_mask = batch["curve_parameter_mask"]
    predicted_vertices = output["predicted_vertices"]
    target_vertices = batch["target_vertices"]
    vertex_error = functional.smooth_l1_loss(
        predicted_vertices, target_vertices, reduction="none", beta=0.02
    )
    vertex = (vertex_error * vertex_mask.unsqueeze(-1)).sum() / vertex_mask.sum().clamp_min(1)
    curve_error = functional.smooth_l1_loss(
        output["predicted_curve_parameters"],
        batch["target_curve_parameters"],
        reduction="none",
        beta=0.02,
    )
    curve = (curve_error * curve_mask).sum() / curve_mask.sum().clamp_min(1)
    edge_indices = batch["edge_vertices"]
    start_index = edge_indices[..., 0].unsqueeze(-1).expand(-1, -1, 2)
    end_index = edge_indices[..., 1].unsqueeze(-1).expand(-1, -1, 2)
    predicted_vector = torch.gather(predicted_vertices, 1, end_index) - torch.gather(
        predicted_vertices, 1, start_index
    )
    target_vector = torch.gather(target_vertices, 1, end_index) - torch.gather(
        target_vertices, 1, start_index
    )
    edge_mask = batch["edge_mask"]
    predicted_length = torch.linalg.vector_norm(predicted_vector, dim=-1)
    target_length = torch.linalg.vector_norm(target_vector, dim=-1)
    chord_length = (
        torch.abs(predicted_length - target_length) * edge_mask
    ).sum() / edge_mask.sum().clamp_min(1)
    cosine = functional.cosine_similarity(predicted_vector, target_vector, dim=-1, eps=1e-8)
    direction = ((1.0 - cosine) * edge_mask).sum() / edge_mask.sum().clamp_min(1)
    total = (
        float(vertex_weight) * vertex
        + float(curve_weight) * curve
        + float(chord_length_weight) * chord_length
        + float(direction_weight) * direction
    )
    return {
        "loss": total,
        "vertex": vertex,
        "curve_parameter": curve,
        "chord_length": chord_length,
        "direction": direction,
    }


def _decode_curve(
    template_edge: Mapping[str, Any],
    parameters_cm: Sequence[float],
    *,
    start: Sequence[float] | None = None,
    end: Sequence[float] | None = None,
) -> dict[str, Any]:
    curve = copy.deepcopy(template_edge["curve"])
    values = np.asarray(parameters_cm, dtype=float)
    kind = str(curve["type"])
    if kind == "quadratic_bezier":
        curve["controls_cm"] = [values[:2].tolist()]
    elif kind == "cubic_bezier":
        curve["controls_cm"] = [values[:2].tolist(), values[2:4].tolist()]
    elif kind == "circular_arc":
        minimum = 1e-4
        if start is not None and end is not None:
            minimum = max(minimum, 0.5 * math.dist(start, end) + 1e-4)
        curve["arc"]["radius_cm"] = max(float(abs(values[4])), minimum)
    return curve


def _curve_length_and_direction(
    start: Sequence[float], end: Sequence[float], curve: Mapping[str, Any]
) -> tuple[float, float]:
    points = sample_curve(start, end, curve, samples=129)
    length = float(sum(math.dist(first, second) for first, second in zip(points, points[1:])))
    direction = float(math.degrees(math.atan2(float(end[1]) - float(start[1]), float(end[0]) - float(start[0]))))
    return length, direction


def materialize_prediction(
    pair: RetrievedResidualPair,
    predicted_vertices_cm: np.ndarray,
    predicted_curve_parameters_cm: np.ndarray,
) -> dict[str, Any]:
    """Create an overlay-ready prediction while deriving endpoints from vertices."""

    target_label = load_exact_label(pair.target.label_path)
    predicted_vertices_cm = np.asarray(predicted_vertices_cm, dtype=float)
    predicted_curve_parameters_cm = np.asarray(predicted_curve_parameters_cm, dtype=float)
    panels = []
    vertex_offset = 0
    edge_offset = 0
    for panel in _ordered_panels(target_label):
        vertex_count = len(panel["vertices_cm"])
        edge_count = len(panel["edges"])
        vertices = predicted_vertices_cm[vertex_offset : vertex_offset + vertex_count]
        edges = []
        for local_edge, template in enumerate(
            sorted(panel["edges"], key=lambda value: int(value["edge_index"]))
        ):
            start_index, end_index = (int(value) for value in template["endpoints"])
            start, end = vertices[start_index], vertices[end_index]
            curve = _decode_curve(
                template,
                predicted_curve_parameters_cm[edge_offset + local_edge],
                start=start,
                end=end,
            )
            length, direction = _curve_length_and_direction(start, end, curve)
            edges.append(
                {
                    "edge_index": int(template["edge_index"]),
                    "endpoints": [start_index, end_index],
                    "start_cm": start.tolist(),
                    "end_cm": end.tolist(),
                    "curve": curve,
                    "length_cm": length,
                    "chord_direction_deg": direction,
                }
            )
        panels.append(
            {
                "panel_id": str(panel["panel_id"]),
                "source_order_index": int(panel["source_order_index"]),
                "vertices_cm": vertices.tolist(),
                "edges": edges,
            }
        )
        vertex_offset += vertex_count
        edge_offset += edge_count
    return {
        "schema_version": "gcdv2-retrieved-residual-prediction-1.0",
        "sample_id": pair.target.sample_id,
        "anchor_id": pair.anchor.sample_id,
        "category": pair.target.category,
        "split": pair.split,
        "topology_hash": pair.target.topology_hash,
        "visual_cosine_similarity": pair.visual_cosine_similarity,
        "topology_frozen": {
            "panel_inventory": True,
            "edge_endpoint_incidence": True,
            "curve_types": True,
            "stitches": True,
            "circular_arc_branch_flags": True,
        },
        "shared_vertex_contract": "edge start/end coordinates are derived from one corrected panel vertex array",
        "target_label_path": str(pair.target.label_path.as_posix()),
        "anchor_label_path": str(pair.anchor.label_path.as_posix()),
        "panels": panels,
    }


def _angular_error(first: float, second: float) -> float:
    return abs((float(first) - float(second) + 180.0) % 360.0 - 180.0)


def geometry_metrics(
    pairs: Sequence[RetrievedResidualPair],
    predicted_vertices_cm: Sequence[np.ndarray],
    predicted_curve_parameters_cm: Sequence[np.ndarray],
) -> dict[str, Any]:
    """Compare retrieval baseline and learned edit using identical supports."""

    vertex_baseline_sq: list[float] = []
    vertex_edited_sq: list[float] = []
    curve_baseline_sq: list[float] = []
    curve_edited_sq: list[float] = []
    length_baseline: list[float] = []
    length_edited: list[float] = []
    direction_baseline: list[float] = []
    direction_edited: list[float] = []
    per_category: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for pair, predicted_vertices, predicted_parameters in zip(
        pairs, predicted_vertices_cm, predicted_curve_parameters_cm
    ):
        target, anchor = pair.target, pair.anchor
        predicted_vertices = np.asarray(predicted_vertices, dtype=float)
        predicted_parameters = np.asarray(predicted_parameters, dtype=float)
        baseline_vertex = np.square(anchor.vertices_cm - target.vertices_cm).sum(axis=1)
        edited_vertex = np.square(predicted_vertices - target.vertices_cm).sum(axis=1)
        vertex_baseline_sq.extend(baseline_vertex.tolist())
        vertex_edited_sq.extend(edited_vertex.tolist())
        mask = target.curve_parameter_mask
        curve_baseline_sq.extend(
            np.square(anchor.curve_parameters_cm[mask] - target.curve_parameters_cm[mask]).tolist()
        )
        curve_edited_sq.extend(
            np.square(predicted_parameters[mask] - target.curve_parameters_cm[mask]).tolist()
        )
        target_label = load_exact_label(target.label_path)
        flat_templates = [
            edge
            for panel in _ordered_panels(target_label)
            for edge in sorted(panel["edges"], key=lambda value: int(value["edge_index"]))
        ]
        for edge_index, (start_index, end_index) in enumerate(target.edges):
            template = flat_templates[edge_index]
            target_curve = _decode_curve(
                template,
                target.curve_parameters_cm[edge_index],
                start=target.vertices_cm[start_index],
                end=target.vertices_cm[end_index],
            )
            anchor_curve = _decode_curve(
                template,
                anchor.curve_parameters_cm[edge_index],
                start=anchor.vertices_cm[start_index],
                end=anchor.vertices_cm[end_index],
            )
            edited_curve = _decode_curve(
                template,
                predicted_parameters[edge_index],
                start=predicted_vertices[start_index],
                end=predicted_vertices[end_index],
            )
            target_length, target_direction = _curve_length_and_direction(
                target.vertices_cm[start_index], target.vertices_cm[end_index], target_curve
            )
            anchor_length, anchor_direction = _curve_length_and_direction(
                anchor.vertices_cm[start_index], anchor.vertices_cm[end_index], anchor_curve
            )
            edited_length, edited_direction = _curve_length_and_direction(
                predicted_vertices[start_index], predicted_vertices[end_index], edited_curve
            )
            length_baseline.append(abs(anchor_length - target_length))
            length_edited.append(abs(edited_length - target_length))
            direction_baseline.append(_angular_error(anchor_direction, target_direction))
            direction_edited.append(_angular_error(edited_direction, target_direction))
        per_category[target.category]["baseline_vertex_sq"].extend(baseline_vertex.tolist())
        per_category[target.category]["edited_vertex_sq"].extend(edited_vertex.tolist())

    def rmse(values: Sequence[float]) -> float | None:
        return float(math.sqrt(float(np.mean(values)))) if values else None

    def mae(values: Sequence[float]) -> float | None:
        return float(np.mean(values)) if values else None

    baseline_vertex_rmse = rmse(vertex_baseline_sq)
    edited_vertex_rmse = rmse(vertex_edited_sq)
    payload = {
        "pair_count": len(pairs),
        "vertex": {
            "baseline_anchor_rmse_cm": baseline_vertex_rmse,
            "edited_rmse_cm": edited_vertex_rmse,
            "relative_improvement": (
                (baseline_vertex_rmse - edited_vertex_rmse) / baseline_vertex_rmse
                if baseline_vertex_rmse not in (None, 0.0) and edited_vertex_rmse is not None
                else None
            ),
        },
        "curve_parameters": {
            "baseline_anchor_rmse_cm": rmse(curve_baseline_sq),
            "edited_rmse_cm": rmse(curve_edited_sq),
        },
        "edge_length": {
            "baseline_anchor_mae_cm": mae(length_baseline),
            "edited_mae_cm": mae(length_edited),
        },
        "edge_direction": {
            "baseline_anchor_mae_deg": mae(direction_baseline),
            "edited_mae_deg": mae(direction_edited),
        },
        "curve_type": {
            "baseline_accuracy": 1.0 if pairs else None,
            "edited_accuracy": 1.0 if pairs else None,
            "policy": "copied from exact-topology anchor; not learned",
        },
        "shared_vertex_endpoint_consistency": 1.0 if pairs else None,
        "per_category_vertex": {},
    }
    for category, values in sorted(per_category.items()):
        payload["per_category_vertex"][category] = {
            "baseline_anchor_rmse_cm": rmse(values["baseline_vertex_sq"]),
            "edited_rmse_cm": rmse(values["edited_vertex_sq"]),
        }
    return payload


def project_anchor_direction_constraints(
    anchor_vertices_cm: np.ndarray,
    predicted_vertices_cm: np.ndarray,
    edges: np.ndarray,
    *,
    weight: float,
) -> np.ndarray:
    """Least-squares projection that discourages rotating retrieved edges.

    The data term keeps every shared vertex near the network prediction.  For
    each edge, a perpendicular linear constraint asks its corrected vector to
    remain parallel to the retrieved anchor vector.  Incidence is untouched,
    so all adjacent edges still meet at exactly one shared vertex.
    """

    anchor = np.asarray(anchor_vertices_cm, dtype=np.float64)
    predicted = np.asarray(predicted_vertices_cm, dtype=np.float64)
    incidence = np.asarray(edges, dtype=np.int64)
    if anchor.shape != predicted.shape or anchor.ndim != 2 or anchor.shape[1] != 2:
        raise ValueError("anchor/predicted vertices must share shape [N,2]")
    if weight <= 0.0 or not len(incidence):
        return predicted.astype(np.float32)
    count = len(anchor)
    rows = []
    targets = []
    # Interleaved x/y unknowns.  Identity observations retain absolute panel
    # position and make the system full rank.
    for vertex in range(count):
        for axis in range(2):
            row = np.zeros(count * 2, dtype=np.float64)
            row[2 * vertex + axis] = 1.0
            rows.append(row)
            targets.append(float(predicted[vertex, axis]))
    scale = math.sqrt(float(weight))
    for start, end in incidence:
        vector = anchor[int(end)] - anchor[int(start)]
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-8:
            continue
        perpendicular = np.asarray((-vector[1], vector[0]), dtype=np.float64) / norm
        row = np.zeros(count * 2, dtype=np.float64)
        row[2 * int(end) : 2 * int(end) + 2] = scale * perpendicular
        row[2 * int(start) : 2 * int(start) + 2] = -scale * perpendicular
        rows.append(row)
        targets.append(0.0)
    solution, _, _, _ = np.linalg.lstsq(np.stack(rows), np.asarray(targets), rcond=None)
    return solution.reshape(count, 2).astype(np.float32)


__all__ = [
    "CATEGORY_NAMES",
    "COORDINATE_SCALE_CM",
    "CURVE_PARAMETER_DIM",
    "FPN_CACHE_TO_SEMANTIC_VIEW_ORDER",
    "SEMANTIC_VIEW_NAMES",
    "ExactGeometryRecord",
    "RetrievedResidualPair",
    "batch_residual_pairs",
    "build_retrieved_residual_model",
    "build_visual_retrieval_pairs",
    "deterministic_topology_split",
    "geometry_metrics",
    "load_crossmodal_embedding_bank",
    "materialize_prediction",
    "project_anchor_direction_constraints",
    "read_exact_geometry_records",
    "reorder_cached_fpn_views",
    "residual_loss",
    "topology_contract",
    "topology_hash",
]
