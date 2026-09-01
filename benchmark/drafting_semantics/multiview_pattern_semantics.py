"""Four-view visual-feature to 2D pattern-structure baseline.

Each view token is either a deterministic silhouette descriptor or a frozen
image-backbone embedding computed from one real GCDv2 orthographic RGBA
render.  The target is read from that sample's paired canonical vector
pattern.  This is the bounded semantic inverse stage; it does not claim
pixel-to-spline recovery.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .multigarment_learning import (
    GARMENT_ROLES,
    MULTIGARMENT_EDGE_ROLES,
    MULTIGARMENT_PANEL_ROLES,
    InspectableEncoderLayer,
)
from .schema import DraftingSemanticRecord
from .semantic_paths import merge_predicted_semantic_paths


VIEW_NAMES = ("front", "back", "left", "right")
PANEL_COUNT_NAMES = tuple(role for role in MULTIGARMENT_PANEL_ROLES if role != "other")
SEMANTIC_COUNT_NAMES = tuple(role for role in MULTIGARMENT_EDGE_ROLES if role != "other")
PATTERN_TARGET_NAMES = (
    "panel_count",
    "edge_count",
    *(f"panel:{role}" for role in PANEL_COUNT_NAMES),
    *(f"path:{role}" for role in SEMANTIC_COUNT_NAMES),
)
VIEW_FEATURE_DIM = 21
DEFAULT_OFFICIAL_SPLIT_PREFIX = "garments_5000_0/default_body"


@dataclass(frozen=True)
class MultiviewPatternExample:
    sample_id: str
    split: str
    category_target: int
    view_features: np.ndarray
    pattern_target: np.ndarray
    view_paths: tuple[str, ...]
    pattern_path: str


@dataclass(frozen=True)
class TargetStandardizer:
    means: tuple[float, ...]
    standard_deviations: tuple[float, ...]

    @classmethod
    def fit(cls, examples: Sequence[MultiviewPatternExample]) -> "TargetStandardizer":
        values = np.stack([item.pattern_target for item in examples], axis=0)
        means = values.mean(axis=0)
        deviations = values.std(axis=0)
        deviations = np.where(deviations < 1e-6, 1.0, deviations)
        return cls(tuple(float(value) for value in means), tuple(float(value) for value in deviations))

    def encode(self, values: np.ndarray) -> np.ndarray:
        return (values - np.asarray(self.means, dtype=np.float32)) / np.asarray(self.standard_deviations, dtype=np.float32)

    def decode(self, values: np.ndarray) -> np.ndarray:
        return values * np.asarray(self.standard_deviations, dtype=np.float32) + np.asarray(self.means, dtype=np.float32)


def _split_lookup(
    path: Path,
    split_prefix: str | None = DEFAULT_OFFICIAL_SPLIT_PREFIX,
) -> dict[str, str]:
    """Resolve official split entries within one explicitly selected archive batch.

    GarmentCodeData reuses some sample basenames across its 5k archives.  The
    local benchmark index is batch 0, so collapsing the complete official split
    by basename before filtering can silently let a later archive overwrite the
    right assignment.
    """

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    lookup = {}
    prefix = str(split_prefix).replace("\\", "/").strip("/") if split_prefix else None
    for split, values in payload.items():
        for value in values:
            normalized = str(value).replace("\\", "/").strip("/")
            if prefix and not normalized.startswith(f"{prefix}/"):
                continue
            sample_id = normalized.rsplit("/", 1)[-1]
            resolved = "train" if split == "training" else split
            existing = lookup.get(sample_id)
            if existing is not None and existing != resolved:
                raise ValueError(
                    f"ambiguous split for {sample_id!r} within prefix {split_prefix!r}"
                )
            lookup[sample_id] = resolved
    return lookup


def _semantic_targets(path: Path) -> dict[str, np.ndarray]:
    output = {}
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = DraftingSemanticRecord.from_dict(json.loads(line))
            panel_counts = {role: 0 for role in PANEL_COUNT_NAMES}
            path_counts = {role: 0 for role in SEMANTIC_COUNT_NAMES}
            edge_count = 0
            for panel in record.panels:
                if panel.role in panel_counts:
                    panel_counts[panel.role] += 1
                edge_count += len(panel.edges)
                roles = tuple(edge.role for edge in panel.edges)
                for path_value in merge_predicted_semantic_paths(roles):
                    # `other` is unknown/unlabeled in this corpus and is not a
                    # negative or a semantic target.
                    if path_value.role in path_counts:
                        path_counts[path_value.role] += 1
            output[record.sample_id] = np.asarray(
                [
                    float(len(record.panels)),
                    float(edge_count),
                    *(float(panel_counts[name]) for name in PANEL_COUNT_NAMES),
                    *(float(path_counts[name]) for name in SEMANTIC_COUNT_NAMES),
                ],
                dtype=np.float32,
            )
    return output


def read_multiview_pattern_examples(
    index_path: Path,
    split_path: Path,
    semantic_records_path: Path,
    precomputed_features_path: Path | None = None,
    *,
    split_prefix: str | None = DEFAULT_OFFICIAL_SPLIT_PREFIX,
) -> tuple[MultiviewPatternExample, ...]:
    payload = json.loads(Path(index_path).read_text(encoding="utf-8"))
    split = _split_lookup(split_path, split_prefix=split_prefix)
    semantic_targets = _semantic_targets(semantic_records_path)
    precomputed: dict[str, np.ndarray] = {}
    if precomputed_features_path is not None:
        archive = np.load(precomputed_features_path, allow_pickle=False)
        ids = [str(value) for value in archive["sample_ids"].tolist()]
        values = archive["features"].astype(np.float32)
        if values.ndim != 3 or values.shape[1] != len(VIEW_NAMES):
            raise ValueError(f"invalid precomputed feature shape: {values.shape}")
        precomputed = dict(zip(ids, values))
    output = []
    for row in payload["records"]:
        sample_id = str(row["sample_id"])
        if precomputed:
            if sample_id not in precomputed:
                continue
            view_features = precomputed[sample_id]
        else:
            descriptor = np.asarray(row["visual_descriptor"], dtype=np.float32)
            if descriptor.size != len(VIEW_NAMES) * VIEW_FEATURE_DIM:
                raise ValueError(f"{row['sample_id']} has visual descriptor length {descriptor.size}")
            view_features = descriptor.reshape(len(VIEW_NAMES), VIEW_FEATURE_DIM)
        category = str(row["category"])
        target = semantic_targets.get(str(row["sample_id"]))
        if target is None:
            continue
        output.append(
            MultiviewPatternExample(
                sample_id=sample_id,
                split=split.get(sample_id, "auxiliary"),
                category_target=GARMENT_ROLES.index(category),
                view_features=view_features,
                pattern_target=target,
                view_paths=tuple(str(value) for value in row["source_views"]),
                pattern_path=str(row["source_pattern"]),
            )
        )
    return tuple(output)


def multiview_batch(examples: Sequence[MultiviewPatternExample], standardizer: TargetStandardizer) -> dict[str, Any]:
    raw_targets = np.stack([item.pattern_target for item in examples], axis=0)
    return {
        "view_features": np.stack([item.view_features for item in examples], axis=0),
        "category_targets": np.asarray([item.category_target for item in examples], dtype=np.int64),
        "pattern_targets": standardizer.encode(raw_targets).astype(np.float32),
        "raw_pattern_targets": raw_targets,
        "sample_ids": tuple(item.sample_id for item in examples),
    }


def build_multiview_pattern_model(config: Mapping[str, Any]):
    import torch

    class MultiviewPatternTransformer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            width = int(config["width"])
            heads = int(config["heads"])
            feedforward = width * int(config["feedforward_multiplier"])
            dropout = float(config["dropout"])
            self.view_projection = torch.nn.Linear(int(config.get("view_feature_dim", VIEW_FEATURE_DIM)), width)
            self.view_embedding = torch.nn.Parameter(torch.zeros(1, len(VIEW_NAMES), width))
            self.class_token = torch.nn.Parameter(torch.zeros(1, 1, width))
            self.layers = torch.nn.ModuleList(
                InspectableEncoderLayer.build(width, heads, feedforward, dropout)
                for _ in range(int(config["layers"]))
            )
            self.category_head = torch.nn.Sequential(torch.nn.LayerNorm(width), torch.nn.Linear(width, len(GARMENT_ROLES)))
            self.pattern_head = torch.nn.Sequential(
                torch.nn.LayerNorm(width), torch.nn.Linear(width, width), torch.nn.GELU(), torch.nn.Linear(width, len(PATTERN_TARGET_NAMES))
            )
            embedding = int(config["contrastive_dimension"])
            self.image_projection = torch.nn.Sequential(torch.nn.LayerNorm(width), torch.nn.Linear(width, embedding))
            self.pattern_projection = torch.nn.Sequential(
                torch.nn.Linear(len(PATTERN_TARGET_NAMES), width), torch.nn.GELU(), torch.nn.Linear(width, embedding)
            )
            torch.nn.init.normal_(self.view_embedding, std=0.02)
            torch.nn.init.normal_(self.class_token, std=0.02)

        def forward(self, view_features, *, pattern_targets=None, view_valid=None, capture_attention: bool = False):
            batch = view_features.shape[0]
            hidden = self.view_projection(view_features) + self.view_embedding
            hidden = torch.cat((self.class_token.expand(batch, -1, -1), hidden), dim=1)
            if view_valid is None:
                view_valid = torch.ones((batch, len(VIEW_NAMES)), dtype=torch.bool, device=view_features.device)
            padding = torch.cat((torch.zeros((batch, 1), dtype=torch.bool, device=view_features.device), ~view_valid), dim=1)
            attention = []
            for layer in self.layers:
                hidden, weights = layer(hidden, padding, capture_attention)
                if capture_attention:
                    attention.append(weights)
            pooled = hidden[:, 0]
            result = {
                "category_logits": self.category_head(pooled),
                "pattern_prediction": self.pattern_head(pooled),
                "image_embedding": torch.nn.functional.normalize(self.image_projection(pooled), dim=-1),
                "attention": attention,
            }
            if pattern_targets is not None:
                result["pattern_embedding"] = torch.nn.functional.normalize(self.pattern_projection(pattern_targets), dim=-1)
            return result

    return MultiviewPatternTransformer()


__all__ = [
    "DEFAULT_OFFICIAL_SPLIT_PREFIX",
    "PATTERN_TARGET_NAMES",
    "PANEL_COUNT_NAMES",
    "SEMANTIC_COUNT_NAMES",
    "TargetStandardizer",
    "VIEW_FEATURE_DIM",
    "VIEW_NAMES",
    "MultiviewPatternExample",
    "build_multiview_pattern_model",
    "multiview_batch",
    "read_multiview_pattern_examples",
]
