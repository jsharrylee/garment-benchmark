"""Four-view retrieval into the coordinate-free Garment Pattern DSL space.

The DSL teacher is the frozen ``PatternDSLTransformer``.  It consumes only
analytic command kinds and intrinsic edge geometry; semantic roles, seams,
absolute coordinates, and source identifiers are targets or metadata, never
neural inputs.  Its per-panel hidden states become the fixed pattern-side
tokens for a small contrastive adapter.

The visual side consumes the already frozen ResNet50-FPN cache.  The trainable
model therefore learns only the bridge between image evidence and the DSL
teacher's geometry space.  This module intentionally does not claim to decode
a complete SVG/DSL program from pixels.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from benchmark.gcdv2_exact.pattern_dsl_learning import (
    EDGE_FEATURE_SCHEMA_TANGENT_GAP_V1,
    MASK_COMMAND,
    build_pattern_dsl_model,
)


SCHEMA_VERSION = "gcdv2-visual-pattern-dsl-retrieval-1.0"
SPLIT_NAMES = ("train", "validation", "test")
SPLIT_TO_INDEX = {name: index for index, name in enumerate(SPLIT_NAMES)}
SEMANTIC_VIEW_NAMES = ("front", "back", "left", "right")
FPN_CACHE_TO_SEMANTIC_VIEW_ORDER = (1, 0, 2, 3)


@dataclass(frozen=True)
class VisualDSLCorpus:
    sample_ids: np.ndarray
    categories: np.ndarray
    splits: np.ndarray
    view_features: np.ndarray
    dsl_panel_tokens: np.ndarray
    panel_valid: np.ndarray
    topology_signatures: np.ndarray
    dsl_indices: np.ndarray
    feature_indices: np.ndarray

    def indices(self, split: str) -> np.ndarray:
        return np.flatnonzero(self.splits == SPLIT_TO_INDEX[split]).astype(np.int64)


def read_metadata(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def topology_signature_from_program(
    category: int,
    commands: np.ndarray,
    edge_valid: np.ndarray,
    panel_valid: np.ndarray,
) -> str:
    """Return an evaluation-only exact primitive/cycle topology signature.

    Panel ordering is canonicalized by sorting cyclic command strings.  A
    rotation and reversed rotation of a closed panel are equivalent.  This is
    deliberately not passed to either neural encoder.
    """

    import hashlib

    panel_programs: list[tuple[int, ...]] = []
    for panel in np.flatnonzero(panel_valid):
        count = int(edge_valid[panel].sum())
        values = tuple(int(value) for value in commands[panel, :count])
        if not values:
            continue
        rotations = [values[index:] + values[:index] for index in range(len(values))]
        reverse = tuple(reversed(values))
        rotations.extend(reverse[index:] + reverse[:index] for index in range(len(reverse)))
        panel_programs.append(min(rotations))
    payload = json.dumps(
        {"category": int(category), "closed_panel_commands": sorted(panel_programs)},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def extract_frozen_dsl_panel_tokens(
    arrays: Mapping[str, np.ndarray],
    indices: Sequence[int],
    checkpoint_path: Path,
    *,
    device,
    batch_size: int = 16,
) -> np.ndarray:
    """Run the pretrained DSL encoder once and pool valid edges per panel."""

    import torch

    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    array_schema = str(
        np.asarray(
            arrays.get("edge_feature_schema", EDGE_FEATURE_SCHEMA_TANGENT_GAP_V1)
        ).item()
    )
    checkpoint_schema = str(
        checkpoint.get("edge_feature_schema", EDGE_FEATURE_SCHEMA_TANGENT_GAP_V1)
    )
    if array_schema != checkpoint_schema:
        raise ValueError(
            "Pattern DSL feature/checkpoint schema mismatch: "
            f"{array_schema!r} != {checkpoint_schema!r}"
        )
    width = int(checkpoint.get("width", 128))
    model = build_pattern_dsl_model(width=width)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.requires_grad_(False).eval().to(device)
    source_indices = np.asarray(indices, dtype=np.int64)
    result = np.zeros((len(source_indices), arrays["panel_valid"].shape[1], width), np.float16)
    with torch.inference_mode():
        for start in range(0, len(source_indices), batch_size):
            current = source_indices[start : start + batch_size]
            features = torch.from_numpy(
                arrays["edge_features"][current].astype(np.float32)
            ).to(device)
            commands = torch.from_numpy(
                arrays["edge_commands"][current].astype(np.int64)
            ).to(device)
            edge_valid = torch.from_numpy(arrays["edge_valid"][current]).to(device)
            panel_valid = torch.from_numpy(arrays["panel_valid"][current]).to(device)
            # Padded commands in old caches are already MASK_COMMAND; enforce
            # the contract explicitly so no accidental negative/source value
            # can enter the embedding table.
            commands = torch.where(
                edge_valid,
                commands,
                torch.full_like(commands, MASK_COMMAND),
            )
            with torch.amp.autocast(
                device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                output = model(features, commands, edge_valid, panel_valid)
                hidden = output["edge_hidden"]
                weights = edge_valid[..., None].to(hidden.dtype)
                panel = (hidden * weights).sum(2) / weights.sum(2).clamp_min(1)
            result[start : start + len(current)] = panel.float().cpu().numpy().astype(np.float16)
    return result


def build_visual_dsl_corpus(
    programs_path: Path,
    metadata_path: Path,
    features_path: Path,
    dsl_checkpoint_path: Path,
    *,
    device,
    extraction_batch_size: int = 16,
    cached_panel_tokens_path: Path | None = None,
) -> VisualDSLCorpus:
    """Intersect FPN samples with the authoritative DSL split and encode DSL."""

    arrays = np.load(Path(programs_path), allow_pickle=False)
    metadata = read_metadata(metadata_path)
    if len(metadata) != len(arrays["splits"]):
        raise ValueError("DSL metadata/program counts differ")
    for index, row in enumerate(metadata):
        expected = SPLIT_TO_INDEX[str(row["split"])]
        if int(arrays["splits"][index]) != expected:
            raise ValueError(f"DSL split mismatch at {row['sample_id']}")

    cache = np.load(Path(features_path), allow_pickle=False, mmap_mode="r")
    feature_ids = [str(value) for value in cache["sample_ids"]]
    feature_lookup = {sample_id: index for index, sample_id in enumerate(feature_ids)}
    dsl_indices = np.asarray(
        [index for index, row in enumerate(metadata) if str(row["sample_id"]) in feature_lookup],
        dtype=np.int64,
    )
    if not len(dsl_indices):
        raise ValueError("DSL and FPN caches have no sample intersection")
    sample_ids = np.asarray([str(metadata[index]["sample_id"]) for index in dsl_indices])
    feature_indices = np.asarray([feature_lookup[value] for value in sample_ids], dtype=np.int64)

    panel_tokens: np.ndarray | None = None
    cached_path = Path(cached_panel_tokens_path) if cached_panel_tokens_path else None
    if cached_path is not None and cached_path.is_file():
        cached = np.load(cached_path, allow_pickle=False)
        if np.array_equal(cached["sample_ids"].astype(str), sample_ids):
            panel_tokens = cached["panel_tokens"]
    if panel_tokens is None:
        panel_tokens = extract_frozen_dsl_panel_tokens(
            arrays,
            dsl_indices,
            dsl_checkpoint_path,
            device=device,
            batch_size=extraction_batch_size,
        )
        if cached_path is not None:
            cached_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cached_path, sample_ids=sample_ids, panel_tokens=panel_tokens)

    topologies = np.asarray(
        [
            topology_signature_from_program(
                int(arrays["categories"][source]),
                arrays["edge_commands"][source],
                arrays["edge_valid"][source],
                arrays["panel_valid"][source],
            )
            for source in dsl_indices
        ]
    )
    return VisualDSLCorpus(
        sample_ids=sample_ids,
        categories=arrays["categories"][dsl_indices].astype(np.int64),
        splits=arrays["splits"][dsl_indices].astype(np.int64),
        view_features=cache["features"],
        dsl_panel_tokens=np.asarray(panel_tokens),
        panel_valid=arrays["panel_valid"][dsl_indices],
        topology_signatures=topologies,
        dsl_indices=dsl_indices,
        feature_indices=feature_indices,
    )


def make_visual_dsl_batch(corpus: VisualDSLCorpus, indices: Sequence[int]) -> dict[str, np.ndarray]:
    current = np.asarray(indices, dtype=np.int64)
    feature_rows = corpus.feature_indices[current]
    views = np.asarray(corpus.view_features[feature_rows], dtype=np.float32)
    views = views[:, list(FPN_CACHE_TO_SEMANTIC_VIEW_ORDER)]
    return {
        "views": views,
        "panel_tokens": corpus.dsl_panel_tokens[current].astype(np.float32),
        "panel_valid": corpus.panel_valid[current],
        "categories": corpus.categories[current],
    }


def build_visual_dsl_retrieval_model(config: Mapping[str, Any]):
    """Build a compact FPN visual encoder plus a DSL-panel adapter."""

    import torch
    from torch import nn
    from torch.nn import functional as F

    spatial_dim = int(config.get("spatial_dim", 256))
    dsl_dim = int(config.get("dsl_dim", 128))
    hidden = int(config.get("hidden_dim", 128))
    embedding = int(config.get("embedding_dim", 128))
    heads = int(config.get("heads", 4))
    visual_layers = int(config.get("visual_layers", 2))
    pattern_layers = int(config.get("pattern_layers", 2))
    max_spatial_tokens = int(config.get("max_spatial_tokens", 85))
    max_panels = int(config.get("max_panels", 22))
    pool_queries = int(config.get("pool_queries_per_view", 4))
    dropout = float(config.get("dropout", 0.1))

    class VisualEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(spatial_dim)
            self.project = nn.Linear(spatial_dim, hidden)
            self.spatial_position = nn.Parameter(torch.empty(1, max_spatial_tokens, hidden))
            self.pool_queries = nn.Parameter(torch.empty(1, pool_queries, hidden))
            self.pool_attention = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
            self.view_embedding = nn.Parameter(torch.empty(1, 4, 1, hidden))
            self.pool_embedding = nn.Parameter(torch.empty(1, 1, pool_queries, hidden))
            self.cls = nn.Parameter(torch.empty(1, 1, hidden))
            layer = nn.TransformerEncoderLayer(
                hidden, heads, hidden * 4, dropout, activation="gelu", batch_first=True, norm_first=True
            )
            self.encoder = nn.TransformerEncoder(layer, visual_layers, norm=nn.LayerNorm(hidden))
            self.output = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, embedding))
            self.category = nn.Linear(hidden, 3)
            for parameter in (
                self.spatial_position,
                self.pool_queries,
                self.view_embedding,
                self.pool_embedding,
                self.cls,
            ):
                nn.init.trunc_normal_(parameter, std=0.02)

        def forward(self, values):
            if values.ndim != 4 or values.shape[1] != 4:
                raise ValueError("views must have shape [B,4,T,D]")
            batch, views, token_count, _ = values.shape
            if token_count > max_spatial_tokens:
                raise ValueError("too many FPN tokens")
            hidden_values = self.project(self.norm(values))
            hidden_values = hidden_values + self.spatial_position[:, :token_count].unsqueeze(1)
            hidden_values = hidden_values.reshape(batch * views, token_count, hidden)
            queries = self.pool_queries.expand(batch * views, -1, -1)
            pooled, _ = self.pool_attention(queries, hidden_values, hidden_values, need_weights=False)
            pooled = pooled.reshape(batch, views, pool_queries, hidden)
            pooled = pooled + self.view_embedding + self.pool_embedding
            cls = self.cls.expand(batch, -1, -1)
            sequence = torch.cat((cls, pooled.reshape(batch, views * pool_queries, hidden)), dim=1)
            encoded = self.encoder(sequence)[:, 0]
            return F.normalize(self.output(encoded), dim=-1), self.category(encoded)

    class PatternAdapter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(dsl_dim)
            self.project = nn.Linear(dsl_dim, hidden)
            self.cls = nn.Parameter(torch.empty(1, 1, hidden))
            layer = nn.TransformerEncoderLayer(
                hidden, heads, hidden * 3, dropout, activation="gelu", batch_first=True, norm_first=True
            )
            self.encoder = nn.TransformerEncoder(layer, pattern_layers, norm=nn.LayerNorm(hidden))
            self.output = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, embedding))
            nn.init.trunc_normal_(self.cls, std=0.02)

        def forward(self, values, valid):
            if values.ndim != 3 or valid.shape != values.shape[:2]:
                raise ValueError("DSL panel tokens/mask must be [B,P,D] and [B,P]")
            if values.shape[1] > max_panels:
                raise ValueError("too many DSL panels")
            projected = self.project(self.norm(values))
            cls = self.cls.expand(values.shape[0], -1, -1)
            padding = torch.cat(
                (torch.zeros((len(values), 1), dtype=torch.bool, device=values.device), ~valid.bool()),
                dim=1,
            )
            encoded = self.encoder(torch.cat((cls, projected), dim=1), src_key_padding_mask=padding)
            return F.normalize(self.output(encoded[:, 0]), dim=-1)

    class VisualDSLRetrieval(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.visual_encoder = VisualEncoder()
            self.pattern_adapter = PatternAdapter()
            temperature = float(config.get("temperature", 0.07))
            self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temperature)))

        def forward(self, views, panel_tokens, panel_valid):
            visual, category_logits = self.visual_encoder(views)
            pattern = self.pattern_adapter(panel_tokens, panel_valid)
            return {"visual_embedding": visual, "pattern_embedding": pattern, "category_logits": category_logits}

        def logits(self, visual, pattern):
            return self.logit_scale.exp().clamp(max=100.0) * visual @ pattern.T

    return VisualDSLRetrieval()


def bidirectional_infonce(model, visual, pattern):
    import torch
    from torch.nn import functional as F

    logits = model.logits(visual, pattern)
    targets = torch.arange(len(logits), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets))


def paired_retrieval_metrics(query: np.ndarray, gallery: np.ndarray) -> dict[str, float | int]:
    similarity = np.asarray(query, np.float32) @ np.asarray(gallery, np.float32).T
    order = np.argsort(-similarity, axis=1, kind="stable")
    ranks = np.asarray(
        [int(np.flatnonzero(order[index] == index)[0]) + 1 for index in range(len(order))],
        dtype=np.int64,
    )
    return {
        "count": int(len(ranks)),
        "recall_at_1": float(np.mean(ranks <= 1)),
        "recall_at_5": float(np.mean(ranks <= 5)),
        "recall_at_10": float(np.mean(ranks <= 10)),
        "mrr": float(np.mean(1.0 / ranks)),
        "median_rank": float(np.median(ranks)),
    }


def train_bank_retrieval_metrics(
    query: np.ndarray,
    bank: np.ndarray,
    query_categories: np.ndarray,
    bank_categories: np.ndarray,
    query_topologies: np.ndarray,
    bank_topologies: np.ndarray,
    *,
    top_k: int = 10,
) -> tuple[dict[str, Any], np.ndarray]:
    """Evaluate test images against a train-only, target-absent DSL bank."""

    similarity = np.asarray(query, np.float32) @ np.asarray(bank, np.float32).T
    order = np.argsort(-similarity, axis=1, kind="stable")
    winner = order[:, 0]
    category = np.asarray(query_categories) == np.asarray(bank_categories)[winner]
    topology = np.asarray(query_topologies) == np.asarray(bank_topologies)[winner]
    k = min(int(top_k), order.shape[1])
    category_at_k = np.asarray(
        [np.any(np.asarray(bank_categories)[row[:k]] == query_categories[index]) for index, row in enumerate(order)]
    )
    topology_at_k = np.asarray(
        [np.any(np.asarray(bank_topologies)[row[:k]] == query_topologies[index]) for index, row in enumerate(order)]
    )
    return (
        {
            "count": int(len(query)),
            "exact_target_present": False,
            "category_match_at_1": float(category.mean()),
            f"category_match_at_{k}": float(category_at_k.mean()),
            "exact_topology_compatibility_at_1": float(topology.mean()),
            f"exact_topology_compatibility_at_{k}": float(topology_at_k.mean()),
        },
        order,
    )


__all__ = [
    "FPN_CACHE_TO_SEMANTIC_VIEW_ORDER",
    "SCHEMA_VERSION",
    "SEMANTIC_VIEW_NAMES",
    "SPLIT_NAMES",
    "VisualDSLCorpus",
    "bidirectional_infonce",
    "build_visual_dsl_corpus",
    "build_visual_dsl_retrieval_model",
    "extract_frozen_dsl_panel_tokens",
    "make_visual_dsl_batch",
    "paired_retrieval_metrics",
    "topology_signature_from_program",
    "train_bank_retrieval_metrics",
]
