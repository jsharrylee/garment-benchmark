from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


CATEGORIES = ("top", "skirt", "pants")
PRIMITIVE_TYPES = ("line", "quadratic_bezier", "cubic_bezier", "circular_arc")
MAXIMUM_PANELS = 22
MAXIMUM_EDGES = 134
IMAGE_SIZE = 1024
# The 16/8/4/2 dense-token ablation did not improve localization (its held-out
# endpoint MAE was 110.7 px versus 104.4 px here), so the smaller pyramid is
# retained and the set-assignment formulation is addressed instead.
SPATIAL_GRID_SIZES = (8, 4, 2, 1)


@dataclass(frozen=True)
class PatternExample:
    sample_id: str
    category: str
    pattern_path: Path
    label_path: Path
    family_id: str
    panel_boxes: np.ndarray
    panel_refs: tuple[dict[str, Any], ...]
    edge_geometry: np.ndarray
    edge_types: np.ndarray
    edge_refs: tuple[dict[str, Any], ...]
    packing_scale_px_per_cm: float
    canvas_size_px: int
    spatial_features: np.ndarray | None = None


def _canonical_endpoint_order(
    start: Sequence[float], end: Sequence[float]
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Orient an edge by image-space (y, x), independent of source serialization."""

    left = np.asarray(start, dtype=np.float32)
    right = np.asarray(end, dtype=np.float32)
    swap = (float(right[1]), float(right[0])) < (float(left[1]), float(left[0]))
    return (right, left, True) if swap else (left, right, False)


def _panel_key(panel: Mapping[str, Any]) -> tuple[Any, ...]:
    x0, y0, x1, y1 = (float(value) for value in panel["packed_bbox_px"])
    type_counts = Counter(str(edge["curve"]["type"]) for edge in panel["edges"])
    signature = tuple(type_counts.get(name, 0) for name in PRIMITIVE_TYPES)
    # Packed location is observable in pattern.png.  The remaining fields make
    # ties deterministic without relying on JSON panel order.
    return (round(y0, 5), round(x0, 5), round(y1 - y0, 5), round(x1 - x0, 5), len(panel["edges"]), signature, str(panel["panel_id"]))


def _edge_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    start, end = row["start"], row["end"]
    midpoint = 0.5 * (start + end)
    return (
        round(float(midpoint[1]), 6),
        round(float(midpoint[0]), 6),
        round(float(start[1]), 6),
        round(float(start[0]), 6),
        round(float(end[1]), 6),
        round(float(end[0]), 6),
        int(row["type_index"]),
        str(row["source_edge_id"]),
    )


def topology_family_id(label: Mapping[str, Any]) -> str:
    """A panel-name/order invariant topology family for disjoint splitting."""

    panel_signatures = []
    for panel in label["panels"]:
        counts = Counter(str(edge["curve"]["type"]) for edge in panel["edges"])
        degrees = Counter()
        for edge in panel["edges"]:
            first, second = (int(value) for value in edge["endpoints"])
            degrees[first] += 1
            degrees[second] += 1
        panel_signatures.append(
            (
                len(panel["vertices_cm"]),
                len(panel["edges"]),
                tuple(counts.get(name, 0) for name in PRIMITIVE_TYPES),
                tuple(sorted(degrees.values())),
            )
        )
    payload = json.dumps(
        [str(label["category"]), sorted(panel_signatures)],
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


def target_from_label(
    label: Mapping[str, Any],
    *,
    label_path: Path,
    pattern_path: Path | None = None,
) -> PatternExample:
    """Create packed-image targets while retaining exact-cm truth in the sidecar.

    The learning target deliberately contains no absolute centimetre regression.
    GCDv2's display PNGs use a different ``scale_px_per_cm`` per sample, so an
    image-only network cannot infer physical scale.  ``labels.json`` remains the
    lossless physical source of truth; the scale is carried only as evaluation
    metadata for optional post-hoc conversion.
    """

    if label.get("schema_version") != "gcdv2-exact-pair-1.0":
        raise ValueError(f"unsupported exact label schema: {label.get('schema_version')!r}")
    category = str(label["category"])
    if category not in CATEGORIES:
        raise ValueError(f"unsupported category: {category!r}")
    canvas_sizes = {
        int(round(float(transform["canvas_size_px"][0])))
        for transform in label["packing"].values()
    }
    scales = {
        round(float(transform["scale_px_per_cm"]), 10)
        for transform in label["packing"].values()
    }
    if len(canvas_sizes) != 1 or len(scales) != 1:
        raise ValueError("a sample must use one common canvas size and packing scale")
    canvas_size = next(iter(canvas_sizes))
    scale = next(iter(scales))

    panel_boxes: list[list[float]] = []
    panel_refs: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    for panel_slot, panel in enumerate(sorted(label["panels"], key=_panel_key)):
        x0, y0, x1, y1 = (float(value) / canvas_size for value in panel["packed_bbox_px"])
        panel_boxes.append([x0, y0, x1, y1])
        panel_refs.append(
            {
                "panel_slot": panel_slot,
                "source_panel_id": str(panel["panel_id"]),
                "source_order_index": int(panel["source_order_index"]),
            }
        )
        current = []
        for edge in panel["edges"]:
            start, end, swapped = _canonical_endpoint_order(
                edge["packed_start_uv"], edge["packed_end_uv"]
            )
            delta = end - start
            chord = float(np.linalg.norm(delta))
            direction = delta / max(chord, 1e-8)
            current.append(
                {
                    "start": start,
                    "end": end,
                    "length_fraction": float(edge["length_cm"]) * scale / canvas_size,
                    # Geometry uses image coordinates: +x right, +y down.
                    "direction": direction,
                    "type_index": PRIMITIVE_TYPES.index(str(edge["curve"]["type"])),
                    "source_panel_id": str(panel["panel_id"]),
                    "source_edge_id": str(edge["edge_id"]),
                    "source_edge_index": int(edge["edge_index"]),
                    "source_orientation_swapped": bool(swapped),
                    "truth_length_cm": float(edge["length_cm"]),
                    "source_endpoints": [int(value) for value in edge["endpoints"]],
                    "panel_slot": panel_slot,
                }
            )
        edge_rows.extend(sorted(current, key=_edge_key))
    if len(panel_boxes) > MAXIMUM_PANELS or len(edge_rows) > MAXIMUM_EDGES:
        raise ValueError(
            f"sample exceeds fixed contract: {len(panel_boxes)} panels, {len(edge_rows)} edges"
        )
    geometry = np.asarray(
        [
            [
                *row["start"].tolist(),
                *row["end"].tolist(),
                row["length_fraction"],
                float(row["direction"][1]),  # sin(theta) in image coordinates
                float(row["direction"][0]),  # cos(theta)
            ]
            for row in edge_rows
        ],
        dtype=np.float32,
    ).reshape((-1, 7))
    edge_types = np.asarray([row["type_index"] for row in edge_rows], dtype=np.int64)
    refs = tuple(
        {
            key: value
            for key, value in row.items()
            if key
            not in {"start", "end", "direction", "type_index", "length_fraction"}
        }
        for row in edge_rows
    )
    path = Path(pattern_path or label["pattern_image"])
    return PatternExample(
        sample_id=str(label["sample_id"]),
        category=category,
        pattern_path=path,
        label_path=Path(label_path),
        family_id=topology_family_id(label),
        panel_boxes=np.asarray(panel_boxes, dtype=np.float32).reshape((-1, 4)),
        panel_refs=tuple(panel_refs),
        edge_geometry=geometry,
        edge_types=edge_types,
        edge_refs=refs,
        packing_scale_px_per_cm=float(scale),
        canvas_size_px=int(canvas_size),
    )


def read_pattern_examples(
    index_path: Path,
    *,
    feature_path: Path | None = None,
    limit: int | None = None,
) -> tuple[PatternExample, ...]:
    feature_lookup: dict[str, np.ndarray] = {}
    if feature_path is not None:
        with np.load(feature_path, allow_pickle=False) as cache:
            ids = [str(value) for value in cache["sample_ids"]]
            values = np.asarray(cache["spatial_features"])
            if len(ids) != len(values):
                raise ValueError("feature cache sample_ids/features length mismatch")
            feature_lookup = {sample_id: values[index] for index, sample_id in enumerate(ids)}
    rows = []
    with Path(index_path).open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    rows.sort(key=lambda row: (str(row["category"]), str(row["sample_id"])))
    if limit is not None:
        rows = rows[: int(limit)]
    output = []
    for row in rows:
        sample_id = str(row["sample_id"])
        if feature_path is not None and sample_id not in feature_lookup:
            continue
        label_path = Path(row["label_path"])
        label = json.loads(label_path.read_text(encoding="utf-8"))
        example = target_from_label(
            label,
            label_path=label_path,
            pattern_path=Path(row["pattern_path"]),
        )
        if feature_path is not None:
            example = PatternExample(
                **{
                    **example.__dict__,
                    "spatial_features": np.asarray(feature_lookup[sample_id], dtype=np.float32),
                }
            )
        output.append(example)
    if feature_path is not None and not output:
        raise ValueError("feature cache has no sample IDs present in the exact-pair index")
    return tuple(output)


def family_disjoint_split(
    examples: Sequence[PatternExample],
    *,
    seed: int = 20260829,
    validation_fraction: float = 0.10,
    test_fraction: float = 0.10,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Assign complete topology families to splits, independently per category."""

    assignments: dict[str, str] = {}
    audit: dict[str, Any] = {"strategy": "category_stratified_topology_family_disjoint", "categories": {}}
    for category in CATEGORIES:
        current = [item for item in examples if item.category == category]
        groups: dict[str, list[PatternExample]] = defaultdict(list)
        for item in current:
            groups[item.family_id].append(item)
        ordered = sorted(
            groups.items(),
            key=lambda item: hashlib.sha256(f"{seed}:{category}:{item[0]}".encode()).hexdigest(),
        )
        targets = {
            "test": max(1, round(len(current) * test_fraction)),
            "validation": max(1, round(len(current) * validation_fraction)),
        }
        counts = Counter()
        for family, members in ordered:
            # Put a whole family in the split with the largest remaining
            # validation/test deficit; otherwise retain it for training.
            deficits = {
                split: targets[split] - counts[split]
                for split in ("test", "validation")
            }
            split = max(deficits, key=lambda name: (deficits[name], name))
            if deficits[split] <= 0:
                split = "train"
            for member in members:
                assignments[member.sample_id] = split
            counts[split] += len(members)
        category_families = {
            split: sorted(
                {
                    item.family_id
                    for item in current
                    if assignments.get(item.sample_id) == split
                }
            )
            for split in ("train", "validation", "test")
        }
        overlap = sorted(
            (set(category_families["train"]) & set(category_families["validation"]))
            | (set(category_families["train"]) & set(category_families["test"]))
            | (set(category_families["validation"]) & set(category_families["test"]))
        )
        audit["categories"][category] = {
            "samples": dict(sorted(counts.items())),
            "family_counts": {key: len(value) for key, value in category_families.items()},
            "family_overlap": overlap,
        }
    audit["family_disjoint"] = all(
        not row["family_overlap"] for row in audit["categories"].values()
    )
    return assignments, audit


def padded_pattern_batch(examples: Sequence[PatternExample]) -> dict[str, Any]:
    batch = len(examples)
    edge_geometry = np.zeros((batch, MAXIMUM_EDGES, 7), dtype=np.float32)
    edge_types = np.full((batch, MAXIMUM_EDGES), -100, dtype=np.int64)
    edge_valid = np.zeros((batch, MAXIMUM_EDGES), dtype=bool)
    panel_boxes = np.zeros((batch, MAXIMUM_PANELS, 4), dtype=np.float32)
    panel_valid = np.zeros((batch, MAXIMUM_PANELS), dtype=bool)
    for row, example in enumerate(examples):
        edges, panels = len(example.edge_geometry), len(example.panel_boxes)
        edge_geometry[row, :edges] = example.edge_geometry
        edge_types[row, :edges] = example.edge_types
        edge_valid[row, :edges] = True
        panel_boxes[row, :panels] = example.panel_boxes
        panel_valid[row, :panels] = True
    output: dict[str, Any] = {
        "sample_ids": tuple(item.sample_id for item in examples),
        "categories": np.asarray([CATEGORIES.index(item.category) for item in examples], dtype=np.int64),
        "edge_geometry": edge_geometry,
        "edge_types": edge_types,
        "edge_valid": edge_valid,
        "panel_boxes": panel_boxes,
        "panel_valid": panel_valid,
    }
    if all(item.spatial_features is not None for item in examples):
        output["spatial_features"] = np.stack([item.spatial_features for item in examples]).astype(np.float32)
    elif any(item.spatial_features is not None for item in examples):
        raise ValueError("a batch cannot mix feature-backed and image-only examples")
    return output


def spatial_token_metadata(grid_sizes: Sequence[int] = SPATIAL_GRID_SIZES) -> np.ndarray:
    rows = []
    for level, grid in enumerate(grid_sizes):
        for y in range(int(grid)):
            for x in range(int(grid)):
                rows.append(
                    [
                        (x + 0.5) / grid,
                        (y + 0.5) / grid,
                        level / max(len(grid_sizes) - 1, 1),
                        1.0 / grid,
                    ]
                )
    return np.asarray(rows, dtype=np.float32)


def canonicalize_edge_geometry_torch(geometry):
    """Canonicalize predicted edge orientation by packed image (y, x)."""

    import torch

    start, end = geometry[..., 0:2], geometry[..., 2:4]
    swap = (end[..., 1] < start[..., 1]) | (
        (torch.abs(end[..., 1] - start[..., 1]) < 1e-7) & (end[..., 0] < start[..., 0])
    )
    selected_start = torch.where(swap[..., None], end, start)
    selected_end = torch.where(swap[..., None], start, end)
    direction = geometry[..., 5:7]
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    direction = torch.where(swap[..., None], -direction, direction)
    return torch.cat((selected_start, selected_end, geometry[..., 4:5], direction), dim=-1)


def build_pattern_parser_model(config: Mapping[str, Any] | None = None):
    """Build a DETR-style set parser over frozen single-view FPN tokens."""

    import torch

    values = {
        "feature_dim": 256,
        "width": 256,
        "heads": 8,
        "encoder_layers": 2,
        "decoder_layers": 3,
        "feedforward_multiplier": 4,
        "dropout": 0.1,
        **dict(config or {}),
    }
    metadata = torch.from_numpy(spatial_token_metadata())

    class PatternSetParser(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            width = int(values["width"])
            heads = int(values["heads"])
            feedforward = width * int(values["feedforward_multiplier"])
            dropout = float(values["dropout"])
            self.feature_projection = torch.nn.Linear(int(values["feature_dim"]), width)
            self.position_projection = torch.nn.Sequential(
                torch.nn.Linear(4, width), torch.nn.GELU(), torch.nn.Linear(width, width)
            )
            self.register_buffer("spatial_metadata", metadata)
            encoder = torch.nn.TransformerEncoderLayer(
                width,
                heads,
                dim_feedforward=feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = torch.nn.TransformerEncoder(
                encoder,
                num_layers=int(values["encoder_layers"]),
                enable_nested_tensor=False,
            )
            decoder = torch.nn.TransformerDecoderLayer(
                width,
                heads,
                dim_feedforward=feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.edge_decoder = torch.nn.TransformerDecoder(
                decoder, num_layers=int(values["decoder_layers"])
            )
            self.panel_decoder = torch.nn.TransformerDecoder(
                decoder, num_layers=max(1, int(values["decoder_layers"]) - 1)
            )
            self.edge_queries = torch.nn.Parameter(torch.empty(1, MAXIMUM_EDGES, width))
            self.panel_queries = torch.nn.Parameter(torch.empty(1, MAXIMUM_PANELS, width))
            self.category_head = torch.nn.Sequential(
                torch.nn.LayerNorm(width), torch.nn.Linear(width, len(CATEGORIES))
            )
            self.edge_presence_head = torch.nn.Linear(width, 1)
            self.edge_type_head = torch.nn.Linear(width, len(PRIMITIVE_TYPES))
            self.edge_geometry_head = torch.nn.Sequential(
                torch.nn.LayerNorm(width),
                torch.nn.Linear(width, width),
                torch.nn.GELU(),
                torch.nn.Linear(width, 7),
            )
            self.panel_presence_head = torch.nn.Linear(width, 1)
            self.panel_bbox_head = torch.nn.Sequential(
                torch.nn.LayerNorm(width), torch.nn.Linear(width, 4)
            )
            torch.nn.init.normal_(self.edge_queries, std=0.02)
            torch.nn.init.normal_(self.panel_queries, std=0.02)

        def forward(self, spatial_features):
            if spatial_features.ndim != 3 or spatial_features.shape[1:] != (
                len(self.spatial_metadata),
                int(values["feature_dim"]),
            ):
                raise ValueError(
                    f"spatial_features must have shape [batch, {len(self.spatial_metadata)}, {values['feature_dim']}]"
                )
            memory = self.feature_projection(spatial_features)
            memory = memory + self.position_projection(self.spatial_metadata)[None]
            memory = self.encoder(memory)
            edge_hidden = self.edge_decoder(
                self.edge_queries.expand(len(memory), -1, -1), memory
            )
            panel_hidden = self.panel_decoder(
                self.panel_queries.expand(len(memory), -1, -1), memory
            )
            raw_edge_geometry = self.edge_geometry_head(edge_hidden)
            endpoints = raw_edge_geometry[..., :4].sigmoid()
            # A long circular arc can exceed one canvas width even though its
            # endpoints and panel bbox lie inside the image.  Softplus keeps
            # rendered length positive without the incorrect [0, 1] cap.
            length = torch.nn.functional.softplus(raw_edge_geometry[..., 4:5])
            direction = raw_edge_geometry[..., 5:7]
            direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            edge_geometry = canonicalize_edge_geometry_torch(
                torch.cat((endpoints, length, direction), dim=-1)
            )
            raw_bbox = self.panel_bbox_head(panel_hidden).sigmoid()
            minimum = torch.minimum(raw_bbox[..., :2], raw_bbox[..., 2:])
            maximum = torch.maximum(raw_bbox[..., :2], raw_bbox[..., 2:])
            return {
                "category_logits": self.category_head(memory.mean(dim=1)),
                "edge_presence_logits": self.edge_presence_head(edge_hidden).squeeze(-1),
                "edge_type_logits": self.edge_type_head(edge_hidden),
                "edge_geometry": edge_geometry,
                "panel_presence_logits": self.panel_presence_head(panel_hidden).squeeze(-1),
                "panel_boxes": torch.cat((minimum, maximum), dim=-1),
            }

    model = PatternSetParser()
    model.config = values
    return model


def hungarian_matches(output: Mapping[str, Any], batch: Mapping[str, Any]):
    """Batch cost construction, then CPU Hungarian assignment per sample.

    The original implementation copied one cost matrix from GPU to CPU for
    every sample, forcing dozens of CUDA synchronizations per batch.  Here all
    padded costs cross the device boundary once; SciPy still solves the exact
    rectangular assignment only over each sample's valid targets.
    """

    import torch

    edge_probability = output["edge_presence_logits"].detach().sigmoid()
    type_probability = output["edge_type_logits"].detach().softmax(dim=-1)
    predicted_geometry = output["edge_geometry"].detach()
    predicted_panels = output["panel_boxes"].detach()
    panel_probability = output["panel_presence_logits"].detach().sigmoid()

    endpoint_cost = torch.cdist(
        predicted_geometry[..., :4], batch["edge_geometry"][..., :4], p=1
    ) / 4.0
    length_cost = torch.cdist(
        predicted_geometry[..., 4:5], batch["edge_geometry"][..., 4:5], p=1
    )
    direction_cost = 1.0 - torch.einsum(
        "bqd,bnd->bqn", predicted_geometry[..., 5:7], batch["edge_geometry"][..., 5:7]
    )
    safe_types = batch["edge_types"].clamp_min(0)
    primitive_cost = -torch.gather(
        type_probability,
        2,
        safe_types[:, None, :].expand(-1, type_probability.shape[1], -1),
    )
    edge_cost = (
        3.0 * endpoint_cost
        + length_cost
        + 0.5 * direction_cost
        + primitive_cost
        - 0.25 * edge_probability[..., None]
    )
    edge_cost = edge_cost.masked_fill(~batch["edge_valid"][:, None, :], 1e6)
    panel_cost = torch.cdist(predicted_panels, batch["panel_boxes"], p=1) / 4.0
    panel_cost = panel_cost - 0.25 * panel_probability[..., None]
    panel_cost = panel_cost.masked_fill(~batch["panel_valid"][:, None, :], 1e6)
    edge_cost_cpu = edge_cost.float().cpu().numpy()
    panel_cost_cpu = panel_cost.float().cpu().numpy()
    edge_counts = batch["edge_valid"].sum(dim=1).cpu().tolist()
    panel_counts = batch["panel_valid"].sum(dim=1).cpu().tolist()

    edge_matches = []
    panel_matches = []
    for row, (edge_count, panel_count) in enumerate(zip(edge_counts, panel_counts)):
        edge_count = int(edge_count)
        if edge_count:
            query, target = linear_sum_assignment(edge_cost_cpu[row, :, :edge_count])
            edge_matches.append(
                (
                    torch.as_tensor(query, device=edge_probability.device, dtype=torch.long),
                    torch.as_tensor(target, device=edge_probability.device, dtype=torch.long),
                )
            )
        else:
            empty = torch.empty(0, device=edge_probability.device, dtype=torch.long)
            edge_matches.append((empty, empty))
        panel_count = int(panel_count)
        if panel_count:
            query, target = linear_sum_assignment(panel_cost_cpu[row, :, :panel_count])
            panel_matches.append(
                (
                    torch.as_tensor(query, device=edge_probability.device, dtype=torch.long),
                    torch.as_tensor(target, device=edge_probability.device, dtype=torch.long),
                )
            )
        else:
            empty = torch.empty(0, device=edge_probability.device, dtype=torch.long)
            panel_matches.append((empty, empty))
    return edge_matches, panel_matches


def pattern_parser_loss(
    output: Mapping[str, Any],
    batch: Mapping[str, Any],
    *,
    category_weights=None,
    primitive_weights=None,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    edge_matches, panel_matches = hungarian_matches(output, batch)
    edge_presence = torch.zeros_like(output["edge_presence_logits"])
    panel_presence = torch.zeros_like(output["panel_presence_logits"])
    primitive_losses = []
    endpoint_losses = []
    length_losses = []
    direction_losses = []
    panel_box_losses = []
    for row, (query, target) in enumerate(edge_matches):
        edge_presence[row, query] = 1.0
        if len(query):
            primitive_losses.append(
                functional.cross_entropy(
                    output["edge_type_logits"][row, query],
                    batch["edge_types"][row, target],
                    weight=primitive_weights,
                )
            )
            prediction = output["edge_geometry"][row, query]
            expected = batch["edge_geometry"][row, target]
            endpoint_losses.append(functional.smooth_l1_loss(prediction[:, :4], expected[:, :4]))
            length_losses.append(functional.smooth_l1_loss(prediction[:, 4], expected[:, 4]))
            direction_losses.append((1.0 - (prediction[:, 5:7] * expected[:, 5:7]).sum(dim=-1)).mean())
    for row, (query, target) in enumerate(panel_matches):
        panel_presence[row, query] = 1.0
        if len(query):
            panel_box_losses.append(
                functional.smooth_l1_loss(
                    output["panel_boxes"][row, query], batch["panel_boxes"][row, target]
                )
            )
    zero = output["category_logits"].sum() * 0.0
    mean = lambda rows: torch.stack(rows).mean() if rows else zero
    components = {
        "category_loss": functional.cross_entropy(
            output["category_logits"], batch["categories"], weight=category_weights
        ),
        "edge_presence_loss": functional.binary_cross_entropy_with_logits(
            output["edge_presence_logits"], edge_presence
        ),
        "primitive_loss": mean(primitive_losses),
        "endpoint_loss": mean(endpoint_losses),
        "length_loss": mean(length_losses),
        "direction_loss": mean(direction_losses),
        "panel_presence_loss": functional.binary_cross_entropy_with_logits(
            output["panel_presence_logits"], panel_presence
        ),
        "panel_bbox_loss": mean(panel_box_losses),
    }
    components["loss"] = (
        components["category_loss"]
        + components["edge_presence_loss"]
        + components["primitive_loss"]
        + 5.0 * components["endpoint_loss"]
        + 2.0 * components["length_loss"]
        + 0.5 * components["direction_loss"]
        + 0.5 * components["panel_presence_loss"]
        + 2.0 * components["panel_bbox_loss"]
    )
    components["edge_matches"] = edge_matches
    components["panel_matches"] = panel_matches
    return components


def ordered_pattern_parser_loss(
    output: Mapping[str, Any],
    batch: Mapping[str, Any],
    *,
    category_weights=None,
    primitive_weights=None,
) -> dict[str, Any]:
    """Fast training loss on the observable canonical packed-geometry order.

    ``target_from_label`` sorts panels by packed top/left position and edges by
    their canonical image-space geometry.  This is independent of source JSON
    serialization, yet avoids solving 156 Hungarian assignments per gradient
    step.  Evaluation still uses Hungarian set matching, so reported metrics
    do not receive credit merely for reproducing this training order.
    """

    import torch
    import torch.nn.functional as functional

    edge_valid = batch["edge_valid"]
    panel_valid = batch["panel_valid"]
    zero = output["category_logits"].sum() * 0.0
    if edge_valid.any():
        primitive = functional.cross_entropy(
            output["edge_type_logits"][edge_valid],
            batch["edge_types"][edge_valid],
            weight=primitive_weights,
        )
        prediction = output["edge_geometry"][edge_valid]
        expected = batch["edge_geometry"][edge_valid]
        endpoint = functional.smooth_l1_loss(prediction[:, :4], expected[:, :4])
        length = functional.smooth_l1_loss(prediction[:, 4], expected[:, 4])
        direction = (1.0 - (prediction[:, 5:7] * expected[:, 5:7]).sum(dim=-1)).mean()
    else:
        primitive = endpoint = length = direction = zero
    panel_box = (
        functional.smooth_l1_loss(output["panel_boxes"][panel_valid], batch["panel_boxes"][panel_valid])
        if panel_valid.any()
        else zero
    )
    components = {
        "category_loss": functional.cross_entropy(
            output["category_logits"], batch["categories"], weight=category_weights
        ),
        "edge_presence_loss": functional.binary_cross_entropy_with_logits(
            output["edge_presence_logits"], edge_valid.float()
        ),
        "primitive_loss": primitive,
        "endpoint_loss": endpoint,
        "length_loss": length,
        "direction_loss": direction,
        "panel_presence_loss": functional.binary_cross_entropy_with_logits(
            output["panel_presence_logits"], panel_valid.float()
        ),
        "panel_bbox_loss": panel_box,
    }
    components["loss"] = (
        components["category_loss"]
        + components["edge_presence_loss"]
        + components["primitive_loss"]
        + 5.0 * components["endpoint_loss"]
        + 2.0 * components["length_loss"]
        + 0.5 * components["direction_loss"]
        + 0.5 * components["panel_presence_loss"]
        + 2.0 * components["panel_bbox_loss"]
    )
    return components


def classification_metrics(predicted: np.ndarray, expected: np.ndarray, names: Sequence[str]) -> dict[str, Any]:
    per_class = {}
    f1s = []
    for index, name in enumerate(names):
        tp = int(np.sum((predicted == index) & (expected == index)))
        fp = int(np.sum((predicted == index) & (expected != index)))
        fn = int(np.sum((predicted != index) & (expected == index)))
        support = int(np.sum(expected == index))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        if support:
            f1s.append(f1)
    return {
        "accuracy": float(np.mean(predicted == expected)) if len(expected) else 0.0,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "per_class": per_class,
    }


__all__ = [
    "CATEGORIES",
    "IMAGE_SIZE",
    "MAXIMUM_EDGES",
    "MAXIMUM_PANELS",
    "PRIMITIVE_TYPES",
    "PatternExample",
    "build_pattern_parser_model",
    "canonicalize_edge_geometry_torch",
    "classification_metrics",
    "family_disjoint_split",
    "hungarian_matches",
    "padded_pattern_batch",
    "pattern_parser_loss",
    "ordered_pattern_parser_loss",
    "read_pattern_examples",
    "spatial_token_metadata",
    "target_from_label",
    "topology_family_id",
]
