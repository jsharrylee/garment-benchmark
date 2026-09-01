from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image


CATEGORIES = ("pants", "skirt", "top")
PARTS = (
    "bodice",
    "collar",
    "hood",
    "pants_cuff",
    "pants_leg",
    "skirt_insert",
    "skirt_panel",
    "sleeve",
    "sleeve_cuff",
    "waistband",
)
SURFACES = ("back", "front", "unspecified")
SIDES = ("left", "right", "unspecified")
CURVE_TYPES = ("circular_arc", "cubic_bezier", "line", "quadratic_bezier")
MAXIMUM_EDGES = 36
IMAGE_SIZE = 128


@dataclass(frozen=True)
class PanelExample:
    row: Mapping[str, Any]
    target: Mapping[str, Any]


@dataclass(frozen=True)
class GarmentExample:
    sample_id: str
    category: str
    panels: tuple[PanelExample, ...]


def read_garments(index_path: Path) -> tuple[list[GarmentExample], tuple[str, ...]]:
    rows = [
        json.loads(line)
        for line in Path(index_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    source_ids = tuple(sorted({str(row["source_panel_id"]) for row in rows}))
    grouped: defaultdict[str, list[PanelExample]] = defaultdict(list)
    categories: dict[str, str] = {}
    for row in rows:
        target = json.loads(Path(row["target_path"]).read_text(encoding="utf-8"))
        sample_id = str(row["sample_id"])
        grouped[sample_id].append(PanelExample(row=row, target=target))
        categories[sample_id] = str(row["garment_category"])
    garments = [
        GarmentExample(
            sample_id=sample_id,
            category=categories[sample_id],
            panels=tuple(
                sorted(grouped[sample_id], key=lambda panel: int(panel.row["source_panel_order_index"]))
            ),
        )
        for sample_id in sorted(grouped)
    ]
    return garments, source_ids


def garment_disjoint_split(
    garments: Sequence[GarmentExample], *, seed: int = 20260829
) -> tuple[dict[str, str], dict[str, Any]]:
    by_category: defaultdict[str, list[GarmentExample]] = defaultdict(list)
    for garment in garments:
        by_category[garment.category].append(garment)
    rng = random.Random(seed)
    assignments: dict[str, str] = {}
    counts: defaultdict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
    for category in CATEGORIES:
        current = sorted(by_category[category], key=lambda value: value.sample_id)
        rng.shuffle(current)
        validation_count = max(1, round(len(current) * 0.1))
        test_count = max(1, round(len(current) * 0.1))
        for index, garment in enumerate(current):
            if index < validation_count:
                split = "validation"
            elif index < validation_count + test_count:
                split = "test"
            else:
                split = "train"
            assignments[garment.sample_id] = split
            counts[split][category] += 1
    split_ids = {
        split: {sample_id for sample_id, assigned in assignments.items() if assigned == split}
        for split in ("train", "validation", "test")
    }
    audit = {
        "seed": seed,
        "strategy": "category-stratified garment-ID-disjoint 80/10/10",
        "counts": {split: dict(sorted(values.items())) for split, values in counts.items()},
        "garment_disjoint": not (
            split_ids["train"] & split_ids["validation"]
            or split_ids["train"] & split_ids["test"]
            or split_ids["validation"] & split_ids["test"]
        ),
    }
    return assignments, audit


def _angle_vector(degrees: float) -> tuple[float, float]:
    radians = math.radians(float(degrees))
    return math.sin(radians), math.cos(radians)


def panel_targets(panel: PanelExample, source_id_to_index: Mapping[str, int]) -> dict[str, np.ndarray | int | float]:
    row, target = panel.row, panel.target
    geometry = target["geometry"]
    count = int(geometry["boundary_vertex_count"])
    vertices = np.zeros((MAXIMUM_EDGES, 2), np.float32)
    edge_types = np.full(MAXIMUM_EDGES, -1, np.int64)
    lengths = np.zeros(MAXIMUM_EDGES, np.float32)
    directions = np.zeros((MAXIMUM_EDGES, 2), np.float32)
    tangents = np.zeros((MAXIMUM_EDGES, 4), np.float32)
    controls = np.zeros((MAXIMUM_EDGES, 4), np.float32)
    control_masks = np.zeros((MAXIMUM_EDGES, 4), np.float32)
    arc_radius = np.zeros(MAXIMUM_EDGES, np.float32)
    arc_flags = np.zeros((MAXIMUM_EDGES, 2), np.float32)
    arc_mask = np.zeros(MAXIMUM_EDGES, np.float32)
    # Source images are 1024 and model images are 128; normalized UV is unchanged.
    vertices[:count] = np.asarray(
        [vertex["image_xy_px"] for vertex in geometry["vertices"]], np.float32
    ) / 1024.0
    for index, edge in enumerate(geometry["edges"]):
        kind = str(edge["curve_type"])
        edge_types[index] = CURVE_TYPES.index(kind)
        lengths[index] = float(edge["length_cm"]) / 100.0
        directions[index] = _angle_vector(float(edge["chord_direction_deg_y_up"]))
        tangents[index, :2] = _angle_vector(float(edge["start_tangent_deg_y_up"]))
        tangents[index, 2:] = _angle_vector(float(edge["end_tangent_deg_y_up"]))
        relative = edge["curve_parameters"].get("relative_controls_chord_frame", [])
        for control_index, control in enumerate(relative[:2]):
            controls[index, control_index * 2 : control_index * 2 + 2] = control
            control_masks[index, control_index * 2 : control_index * 2 + 2] = 1.0
        if kind == "circular_arc":
            arc_mask[index] = 1.0
            arc_radius[index] = float(edge["curve_parameters"]["radius_cm"]) / 100.0
            arc_flags[index] = (
                float(edge["curve_parameters"]["large_arc"]),
                float(edge["curve_parameters"]["sweep_y_up"]),
            )
    role = target["role_labels"]
    return {
        "source_id": source_id_to_index[str(row["source_panel_id"])],
        "part": PARTS.index(str(role["part"])),
        "surface": SURFACES.index(str(role["surface"])),
        "side": SIDES.index(str(role["side"])),
        "count": count,
        "vertices": vertices,
        "edge_types": edge_types,
        "lengths": lengths,
        "directions": directions,
        "tangents": tangents,
        "controls": controls,
        "control_masks": control_masks,
        "arc_radius": arc_radius,
        "arc_flags": arc_flags,
        "arc_mask": arc_mask,
        "cm_per_pixel": float(row["panel_image_cm_per_pixel"]),
    }


class GarmentPanelDataset:
    def __init__(
        self,
        garments: Sequence[GarmentExample],
        source_ids: Sequence[str],
        *,
        shuffle_panels: bool,
    ) -> None:
        self.garments = tuple(garments)
        self.source_ids = tuple(source_ids)
        self.source_id_to_index = {value: index for index, value in enumerate(self.source_ids)}
        self.shuffle_panels = shuffle_panels

    def __len__(self) -> int:
        return len(self.garments)

    def __getitem__(self, index: int) -> dict[str, Any]:
        garment = self.garments[index]
        panels = list(garment.panels)
        if self.shuffle_panels:
            random.shuffle(panels)
        images, scales, targets, panel_uids = [], [], [], []
        for panel in panels:
            with Image.open(panel.row["panel_image_path"]) as image:
                image = image.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
                images.append(np.asarray(image, np.float32)[None] / 255.0)
            target = panel_targets(panel, self.source_id_to_index)
            scales.append([math.log(max(float(target["cm_per_pixel"]), 1e-8))])
            targets.append(target)
            panel_uids.append(str(panel.row["panel_uid"]))
        return {
            "sample_id": garment.sample_id,
            "category": CATEGORIES.index(garment.category),
            "images": np.stack(images),
            "scales": np.asarray(scales, np.float32),
            "targets": targets,
            "panel_uids": panel_uids,
        }


def collate_garments(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    import torch

    batch_size = len(items)
    maximum_panels = max(len(item["targets"]) for item in items)
    arrays: dict[str, np.ndarray] = {
        "images": np.zeros((batch_size, maximum_panels, 1, IMAGE_SIZE, IMAGE_SIZE), np.float32),
        "scales": np.zeros((batch_size, maximum_panels, 1), np.float32),
        "panel_mask": np.zeros((batch_size, maximum_panels), np.bool_),
        "category": np.zeros(batch_size, np.int64),
        "source_id": np.zeros((batch_size, maximum_panels), np.int64),
        "part": np.zeros((batch_size, maximum_panels), np.int64),
        "surface": np.zeros((batch_size, maximum_panels), np.int64),
        "side": np.zeros((batch_size, maximum_panels), np.int64),
        "count": np.zeros((batch_size, maximum_panels), np.int64),
        "vertices": np.zeros((batch_size, maximum_panels, MAXIMUM_EDGES, 2), np.float32),
        "edge_types": np.full((batch_size, maximum_panels, MAXIMUM_EDGES), -1, np.int64),
        "lengths": np.zeros((batch_size, maximum_panels, MAXIMUM_EDGES), np.float32),
        "directions": np.zeros((batch_size, maximum_panels, MAXIMUM_EDGES, 2), np.float32),
        "tangents": np.zeros((batch_size, maximum_panels, MAXIMUM_EDGES, 4), np.float32),
        "controls": np.zeros((batch_size, maximum_panels, MAXIMUM_EDGES, 4), np.float32),
        "control_masks": np.zeros((batch_size, maximum_panels, MAXIMUM_EDGES, 4), np.float32),
        "arc_radius": np.zeros((batch_size, maximum_panels, MAXIMUM_EDGES), np.float32),
        "arc_flags": np.zeros((batch_size, maximum_panels, MAXIMUM_EDGES, 2), np.float32),
        "arc_mask": np.zeros((batch_size, maximum_panels, MAXIMUM_EDGES), np.float32),
    }
    panel_uids = []
    sample_ids = []
    for batch_index, item in enumerate(items):
        panel_count = len(item["targets"])
        arrays["images"][batch_index, :panel_count] = item["images"]
        arrays["scales"][batch_index, :panel_count] = item["scales"]
        arrays["panel_mask"][batch_index, :panel_count] = True
        arrays["category"][batch_index] = item["category"]
        for panel_index, target in enumerate(item["targets"]):
            for key in arrays:
                if key in {"images", "scales", "panel_mask", "category"}:
                    continue
                arrays[key][batch_index, panel_index] = target[key]
        panel_uids.append(item["panel_uids"])
        sample_ids.append(item["sample_id"])
    result = {key: torch.from_numpy(value) for key, value in arrays.items()}
    result["panel_uids"] = panel_uids
    result["sample_ids"] = sample_ids
    return result


def build_model(source_id_count: int, config: Mapping[str, Any]):
    import torch
    from torch import nn
    import torch.nn.functional as F

    width = int(config.get("width", 128))
    heads = int(config.get("heads", 4))
    set_layers = int(config.get("set_layers", 2))
    graph_layers = int(config.get("graph_layers", 2))
    dropout = float(config.get("dropout", 0.1))

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 32, 5, 2, 2), nn.GroupNorm(4, 32), nn.GELU(),
                nn.Conv2d(32, 64, 3, 2, 1), nn.GroupNorm(8, 64), nn.GELU(),
                nn.Conv2d(64, 96, 3, 2, 1), nn.GroupNorm(8, 96), nn.GELU(),
                nn.Conv2d(96, width, 3, 2, 1), nn.GroupNorm(8, width), nn.GELU(),
            )
            self.spatial_position = nn.Parameter(torch.randn(8 * 8, width) * 0.02)
            self.scale_embed = nn.Sequential(nn.Linear(1, width), nn.GELU(), nn.Linear(width, width))
            set_layer = nn.TransformerEncoderLayer(width, heads, width * 4, dropout, batch_first=True, norm_first=True)
            self.set_encoder = nn.TransformerEncoder(set_layer, set_layers)
            graph_layer = nn.TransformerDecoderLayer(width, heads, width * 4, dropout, batch_first=True, norm_first=True)
            self.graph_decoder = nn.TransformerDecoder(graph_layer, graph_layers)
            self.sequence_queries = nn.Parameter(torch.randn(MAXIMUM_EDGES, width) * 0.02)
            self.category_head = nn.Linear(width, len(CATEGORIES))
            self.source_head = nn.Linear(width, source_id_count)
            self.part_head = nn.Linear(width, len(PARTS))
            self.surface_head = nn.Linear(width, len(SURFACES))
            self.side_head = nn.Linear(width, len(SIDES))
            self.count_head = nn.Linear(width, MAXIMUM_EDGES + 1)
            self.vertex_head = nn.Linear(width, 2)
            self.edge_type_head = nn.Linear(width, len(CURVE_TYPES))
            self.length_head = nn.Linear(width, 1)
            self.direction_head = nn.Linear(width, 2)
            self.tangent_head = nn.Linear(width, 4)
            self.control_head = nn.Linear(width, 4)
            self.arc_radius_head = nn.Linear(width, 1)
            self.arc_flag_head = nn.Linear(width, 2)

        def forward(self, images, scales, panel_mask):
            batch, panels = images.shape[:2]
            spatial = self.encoder(images.reshape(batch * panels, 1, IMAGE_SIZE, IMAGE_SIZE))
            spatial = spatial.flatten(2).transpose(1, 2)
            spatial = spatial + self.spatial_position.unsqueeze(0)
            pooled = spatial.mean(dim=1).reshape(batch, panels, width)
            pooled = pooled + self.scale_embed(scales)
            context = self.set_encoder(pooled, src_key_padding_mask=~panel_mask)
            valid = panel_mask.unsqueeze(-1).to(context.dtype)
            garment = (context * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
            memory = torch.cat((context.reshape(batch * panels, 1, width), spatial), dim=1)
            queries = self.sequence_queries.unsqueeze(0).expand(batch * panels, -1, -1)
            decoded = self.graph_decoder(queries, memory).reshape(batch, panels, MAXIMUM_EDGES, width)
            direction = F.normalize(self.direction_head(decoded), dim=-1)
            tangent_raw = self.tangent_head(decoded).reshape(batch, panels, MAXIMUM_EDGES, 2, 2)
            tangents = F.normalize(tangent_raw, dim=-1).reshape(batch, panels, MAXIMUM_EDGES, 4)
            return {
                "category_logits": self.category_head(garment),
                "source_logits": self.source_head(context),
                "part_logits": self.part_head(context),
                "surface_logits": self.surface_head(context),
                "side_logits": self.side_head(context),
                "count_logits": self.count_head(context),
                "vertices": self.vertex_head(decoded).sigmoid(),
                "edge_type_logits": self.edge_type_head(decoded),
                "lengths": F.softplus(self.length_head(decoded).squeeze(-1)),
                "directions": direction,
                "tangents": tangents,
                "controls": self.control_head(decoded),
                "arc_radius": F.softplus(self.arc_radius_head(decoded).squeeze(-1)),
                "arc_flag_logits": self.arc_flag_head(decoded),
            }

    return Model()


def model_loss(output: Mapping[str, Any], batch: Mapping[str, Any]) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    panel_mask = batch["panel_mask"]
    edge_indices = torch.arange(MAXIMUM_EDGES, device=panel_mask.device)[None, None, :]
    edge_mask = panel_mask[:, :, None] & (edge_indices < batch["count"][:, :, None])
    panel_denominator = panel_mask.sum().clamp_min(1)
    edge_denominator = edge_mask.sum().clamp_min(1)

    def panel_ce(key: str, target: str) -> torch.Tensor:
        values = F.cross_entropy(output[key].transpose(1, 2), batch[target], reduction="none")
        return (values * panel_mask).sum() / panel_denominator

    category = F.cross_entropy(output["category_logits"], batch["category"])
    source = panel_ce("source_logits", "source_id")
    part = panel_ce("part_logits", "part")
    surface = panel_ce("surface_logits", "surface")
    side = panel_ce("side_logits", "side")
    count = panel_ce("count_logits", "count")
    vertex = ((output["vertices"] - batch["vertices"]).abs().sum(dim=-1) * edge_mask).sum() / (edge_denominator * 2)
    edge_type_raw = F.cross_entropy(
        output["edge_type_logits"].reshape(-1, len(CURVE_TYPES)),
        batch["edge_types"].reshape(-1),
        ignore_index=-1,
        reduction="none",
    ).reshape_as(batch["edge_types"])
    edge_type = (edge_type_raw * edge_mask).sum() / edge_denominator
    length = ((output["lengths"] - batch["lengths"]).abs() * edge_mask).sum() / edge_denominator
    direction = ((1.0 - (output["directions"] * batch["directions"]).sum(dim=-1)) * edge_mask).sum() / edge_denominator
    tangent_pred = output["tangents"].reshape(*output["tangents"].shape[:-1], 2, 2)
    tangent_true = batch["tangents"].reshape(*batch["tangents"].shape[:-1], 2, 2)
    tangent = (
        (1.0 - (tangent_pred * tangent_true).sum(dim=-1)).sum(dim=-1) * edge_mask
    ).sum() / (edge_denominator * 2)
    control_denominator = batch["control_masks"].sum().clamp_min(1)
    control = (
        (output["controls"] - batch["controls"]).abs() * batch["control_masks"]
    ).sum() / control_denominator
    arc_denominator = batch["arc_mask"].sum().clamp_min(1)
    arc_radius = (
        (output["arc_radius"] - batch["arc_radius"]).abs() * batch["arc_mask"]
    ).sum() / arc_denominator
    arc_flags = (
        F.binary_cross_entropy_with_logits(output["arc_flag_logits"], batch["arc_flags"], reduction="none").sum(dim=-1)
        * batch["arc_mask"]
    ).sum() / (arc_denominator * 2)
    total = (
        category + source + 0.6 * (part + surface + side) + count
        + 5.0 * vertex + edge_type + 2.0 * length + 0.5 * direction
        + 0.5 * tangent + control + 0.5 * arc_radius + 0.5 * arc_flags
    )
    return {
        "loss": total,
        "category": category,
        "source": source,
        "part": part,
        "surface": surface,
        "side": side,
        "count": count,
        "vertex": vertex,
        "edge_type": edge_type,
        "length": length,
        "direction": direction,
        "tangent": tangent,
        "control": control,
        "arc_radius": arc_radius,
        "arc_flags": arc_flags,
    }


__all__ = [
    "CATEGORIES",
    "CURVE_TYPES",
    "GarmentPanelDataset",
    "IMAGE_SIZE",
    "MAXIMUM_EDGES",
    "PARTS",
    "SIDES",
    "SURFACES",
    "build_model",
    "collate_garments",
    "garment_disjoint_split",
    "model_loss",
    "panel_targets",
    "read_garments",
]
