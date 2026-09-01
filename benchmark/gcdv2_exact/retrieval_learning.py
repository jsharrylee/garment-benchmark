"""Cross-modal retrieval between four rendered views and exact GCDv2 patterns.

The pattern branch deliberately consumes the lossless geometry labels produced
by :mod:`benchmark.gcdv2_exact.geometry`.  It is not the historical 29-value
pattern summary.  Every source edge becomes a token containing its endpoints,
native curve kind and controls, metric length, directions/tangents, panel
scale, and exact ordered connectivity.  Initial 3D placement is intentionally
excluded so the pattern branch remains a strictly 2D geometry encoder.

Torch is imported lazily so the geometry and evaluation helpers remain usable
in lightweight preprocessing environments.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "gcdv2-exact-crossmodal-retrieval-1.0"
EXACT_LABEL_SCHEMA_VERSION = "gcdv2-exact-pair-1.0"
CATEGORY_NAMES = ("top", "skirt", "pants")
CURVE_TYPES = ("line", "quadratic_bezier", "cubic_bezier", "circular_arc")
# The historical cache was extracted in filename order CAM000..CAM003.  Source
# geometry inspection established that CAM001 is semantic front and CAM000 is
# semantic back, so every batch is explicitly remapped here.
FPN_CACHE_TO_SEMANTIC_VIEW_ORDER = (1, 0, 2, 3)
SEMANTIC_VIEW_NAMES = ("front", "back", "left", "right")

# All values are dimensionless.  Distances are divided by the largest panel
# extent in that garment, coordinates by their own panel bbox, and angles are
# represented as sine/cosine pairs.
EDGE_FEATURE_NAMES = (
    "panel_index_fraction",
    "edge_index_fraction",
    "panel_vertex_count_fraction",
    "start_vertex_index_fraction",
    "end_vertex_index_fraction",
    "panel_edge_count_fraction",
    "panel_width_over_sample",
    "panel_height_over_sample",
    "panel_log_aspect_tanh",
    "start_u_in_panel",
    "start_v_in_panel",
    "end_u_in_panel",
    "end_v_in_panel",
    "chord_over_sample",
    "curve_length_over_sample",
    "chord_sin",
    "chord_cos",
    "start_tangent_sin",
    "start_tangent_cos",
    "end_tangent_sin",
    "end_tangent_cos",
    "curve_is_line",
    "curve_is_quadratic_bezier",
    "curve_is_cubic_bezier",
    "curve_is_circular_arc",
    "control_count_over_two",
    "control_1_u_in_panel",
    "control_1_v_in_panel",
    "control_1_present",
    "control_2_u_in_panel",
    "control_2_v_in_panel",
    "control_2_present",
    "arc_radius_over_sample",
    "arc_large_flag",
    "arc_sweep_y_up_flag",
)
EDGE_FEATURE_INDEX = {name: index for index, name in enumerate(EDGE_FEATURE_NAMES)}

# The train-bank distance measures normalized 2D geometry rather than initial
# 3D placement or incidental source ordering.
GEOMETRY_DISTANCE_FEATURES = tuple(
    name
    for name in EDGE_FEATURE_NAMES
    if name
    not in {
        "panel_index_fraction",
        "edge_index_fraction",
        "panel_vertex_count_fraction",
        "start_vertex_index_fraction",
        "end_vertex_index_fraction",
        "panel_edge_count_fraction",
    }
)
GEOMETRY_DISTANCE_INDICES = np.asarray(
    [EDGE_FEATURE_INDEX[name] for name in GEOMETRY_DISTANCE_FEATURES], dtype=np.int64
)


@dataclass(frozen=True)
class TokenizedExactPattern:
    tokens: np.ndarray
    edge_ids: tuple[str, ...]
    topology_signature: str
    topology_payload: tuple[tuple[int, tuple[tuple[int, int, int], ...]], ...]


@dataclass(frozen=True)
class ExactRetrievalExample:
    sample_id: str
    category: str
    split: str
    label_path: str
    pattern_path: str
    feature_index: int
    pattern_tokens: np.ndarray
    edge_ids: tuple[str, ...]
    topology_signature: str


@dataclass(frozen=True)
class ExactRetrievalCorpus:
    examples: tuple[ExactRetrievalExample, ...]
    view_features: np.ndarray
    max_edges: int
    feature_cache_path: str
    missing_feature_sample_ids: tuple[str, ...]


def _angle_pair(degrees: float) -> tuple[float, float]:
    radians = math.radians(float(degrees))
    return math.sin(radians), math.cos(radians)


def _safe_extent(value: float) -> float:
    return max(abs(float(value)), 1e-6)


def _panel_uv(point: Sequence[float], bbox: Sequence[float]) -> tuple[float, float]:
    width = _safe_extent(float(bbox[2]) - float(bbox[0]))
    height = _safe_extent(float(bbox[3]) - float(bbox[1]))
    return (
        (float(point[0]) - float(bbox[0])) / width,
        (float(point[1]) - float(bbox[1])) / height,
    )


def topology_payload(label: Mapping[str, Any]) -> tuple[tuple[int, tuple[tuple[int, int, int], ...]], ...]:
    """Return exact ordered vertex/connectivity/curve topology."""

    rows = []
    for panel in label["panels"]:
        edges = tuple(
            (
                int(edge["endpoints"][0]),
                int(edge["endpoints"][1]),
                CURVE_TYPES.index(str(edge["curve"]["type"])),
            )
            for edge in panel["edges"]
        )
        rows.append((len(panel["vertices_cm"]), edges))
    return tuple(rows)


def topology_signature(label: Mapping[str, Any]) -> str:
    payload = {
        "category": str(label["category"]),
        "panels": topology_payload(label),
    }
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def tokenize_exact_pattern(label: Mapping[str, Any]) -> TokenizedExactPattern:
    """Convert one exact label to a variable-length sequence of edge tokens."""

    panels = list(label["panels"])
    if not panels:
        raise ValueError("an exact pattern must contain at least one panel")
    panel_extents = []
    for panel in panels:
        bbox = panel["local_curve_bbox_cm"]
        panel_extents.extend((float(bbox[2]) - float(bbox[0]), float(bbox[3]) - float(bbox[1])))
    sample_scale = max(max(abs(value) for value in panel_extents), 1e-6)
    rows: list[list[float]] = []
    edge_ids: list[str] = []
    panel_denominator = max(len(panels) - 1, 1)
    for panel_index, panel in enumerate(panels):
        edges = list(panel["edges"])
        bbox = panel["local_curve_bbox_cm"]
        width = _safe_extent(float(bbox[2]) - float(bbox[0]))
        height = _safe_extent(float(bbox[3]) - float(bbox[1]))
        edge_denominator = max(len(edges) - 1, 1)
        vertex_count = len(panel["vertices_cm"])
        vertex_denominator = max(vertex_count - 1, 1)
        for edge_index, edge in enumerate(edges):
            start_vertex_index = int(edge["endpoints"][0])
            end_vertex_index = int(edge["endpoints"][1])
            start_u, start_v = _panel_uv(edge["start_cm"], bbox)
            end_u, end_v = _panel_uv(edge["end_cm"], bbox)
            chord = math.dist(edge["start_cm"], edge["end_cm"])
            chord_sin, chord_cos = _angle_pair(edge["chord_direction_deg"])
            start_sin, start_cos = _angle_pair(edge["start_tangent_deg"])
            end_sin, end_cos = _angle_pair(edge["end_tangent_deg"])
            curve = edge["curve"]
            kind = str(curve["type"])
            if kind not in CURVE_TYPES:
                raise ValueError(f"unsupported exact curve kind: {kind!r}")
            one_hot = [float(kind == value) for value in CURVE_TYPES]
            controls = list(curve.get("controls_cm", ()))
            control_values: list[float] = []
            for control_index in range(2):
                if control_index < len(controls):
                    control_u, control_v = _panel_uv(controls[control_index], bbox)
                    control_values.extend((control_u, control_v, 1.0))
                else:
                    control_values.extend((0.0, 0.0, 0.0))
            arc = curve.get("arc", {})
            row = [
                panel_index / panel_denominator,
                edge_index / edge_denominator,
                math.log1p(vertex_count) / math.log(33.0),
                start_vertex_index / vertex_denominator,
                end_vertex_index / vertex_denominator,
                math.log1p(len(edges)) / math.log(33.0),
                width / sample_scale,
                height / sample_scale,
                math.tanh(math.log(width / height)),
                start_u,
                start_v,
                end_u,
                end_v,
                chord / sample_scale,
                float(edge["length_cm"]) / sample_scale,
                chord_sin,
                chord_cos,
                start_sin,
                start_cos,
                end_sin,
                end_cos,
                *one_hot,
                min(len(controls), 2) / 2.0,
                *control_values,
                float(arc.get("radius_cm", 0.0)) / sample_scale,
                float(bool(arc.get("large_arc", False))),
                float(bool(arc.get("sweep_y_up", False))),
            ]
            if len(row) != len(EDGE_FEATURE_NAMES):
                raise AssertionError(f"edge token width {len(row)} != {len(EDGE_FEATURE_NAMES)}")
            if not np.isfinite(row).all():
                raise ValueError(f"non-finite token values in {edge['edge_id']}")
            rows.append(row)
            edge_ids.append(str(edge["edge_id"]))
    if not rows:
        raise ValueError("an exact pattern must contain at least one edge")
    payload = topology_payload(label)
    return TokenizedExactPattern(
        tokens=np.asarray(rows, dtype=np.float32),
        edge_ids=tuple(edge_ids),
        topology_signature=topology_signature(label),
        topology_payload=payload,
    )


def deterministic_stratified_split(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int = 20260829,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> dict[str, str]:
    """Create a stable category-stratified train/validation/test assignment."""

    if validation_fraction < 0 or test_fraction < 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("validation/test fractions must be nonnegative and sum to less than one")
    grouped: dict[str, list[str]] = {}
    for row in records:
        grouped.setdefault(str(row["category"]), []).append(str(row["sample_id"]))
    result: dict[str, str] = {}
    for category, sample_ids in sorted(grouped.items()):
        ordered = sorted(
            set(sample_ids),
            key=lambda sample_id: hashlib.sha256(
                f"{seed}:{category}:{sample_id}".encode("utf-8")
            ).digest(),
        )
        count = len(ordered)
        if count >= 3:
            validation_count = max(1, int(round(count * validation_fraction)))
            test_count = max(1, int(round(count * test_fraction)))
            if validation_count + test_count >= count:
                validation_count = test_count = 1
        else:
            validation_count = 0
            test_count = 0
        for index, sample_id in enumerate(ordered):
            if index < test_count:
                split = "test"
            elif index < test_count + validation_count:
                split = "validation"
            else:
                split = "train"
            result[sample_id] = split
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _resolve_project_path(raw_path: str | Path, index_path: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    if path.is_file():
        return path
    # The default index lives at <root>/artifacts/gcdv2_exact_pairs_v1/index.jsonl.
    candidates = [index_path.parent / path, index_path.parents[2] / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return path


def load_exact_retrieval_corpus(
    index_path: Path,
    feature_cache_path: Path,
    *,
    seed: int = 20260829,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    limit: int | None = None,
) -> ExactRetrievalCorpus:
    """Load exact labels and map them to the already-extracted four-view FPN cache."""

    index_path = Path(index_path)
    rows = _read_jsonl(index_path)
    rows = [row for row in rows if row.get("validation", {}).get("status") == "PASS"]
    cache = np.load(Path(feature_cache_path), allow_pickle=False)
    sample_ids = [str(value) for value in cache["sample_ids"]]
    feature_lookup = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    features = cache["features"]
    if features.ndim != 4 or features.shape[1] != 4:
        raise ValueError(f"expected [N,4,T,D] spatial features, got {features.shape}")
    missing = tuple(
        str(row["sample_id"])
        for row in rows
        if str(row["sample_id"]) not in feature_lookup
    )
    # The existing cache predates the corrected exact converter.  Its 1765
    # intersecting samples are usable as-is; the remaining exact samples need
    # a later feature-extraction pass and are reported rather than silently
    # entering a split without image features.
    rows = [row for row in rows if str(row["sample_id"]) in feature_lookup]
    if limit is not None and limit < len(rows):
        # A category-balanced deterministic subset is useful for smoke tests.
        by_category: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_category.setdefault(str(row["category"]), []).append(row)
        selected = []
        categories = sorted(by_category)
        quota = max(1, int(math.ceil(limit / max(len(categories), 1))))
        for category in categories:
            ordered = sorted(
                by_category[category],
                key=lambda row: hashlib.sha256(
                    f"limit:{seed}:{row['sample_id']}".encode("utf-8")
                ).digest(),
            )
            selected.extend(ordered[:quota])
        rows = sorted(selected, key=lambda row: str(row["sample_id"]))[:limit]
    split_lookup = deterministic_stratified_split(
        rows,
        seed=seed,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )
    examples: list[ExactRetrievalExample] = []
    max_edges = 0
    for row in rows:
        sample_id = str(row["sample_id"])
        label_path = _resolve_project_path(row["label_path"], index_path)
        label = json.loads(label_path.read_text(encoding="utf-8"))
        if label.get("schema_version") != EXACT_LABEL_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported exact label schema for {sample_id}: {label.get('schema_version')!r}"
            )
        tokenized = tokenize_exact_pattern(label)
        max_edges = max(max_edges, len(tokenized.tokens))
        examples.append(
            ExactRetrievalExample(
                sample_id=sample_id,
                category=str(row["category"]),
                split=split_lookup[sample_id],
                label_path=str(label_path),
                pattern_path=str(_resolve_project_path(row["pattern_path"], index_path)),
                feature_index=feature_lookup[sample_id],
                pattern_tokens=tokenized.tokens,
                edge_ids=tokenized.edge_ids,
                topology_signature=tokenized.topology_signature,
            )
        )
    if not examples:
        raise ValueError("no exact retrieval examples were loaded")
    return ExactRetrievalCorpus(
        examples=tuple(examples),
        view_features=features,
        max_edges=max_edges,
        feature_cache_path=str(feature_cache_path),
        missing_feature_sample_ids=missing,
    )


def make_retrieval_batch(
    corpus: ExactRetrievalCorpus,
    example_indices: Sequence[int],
) -> dict[str, np.ndarray]:
    selected = [corpus.examples[int(index)] for index in example_indices]
    views = np.stack(
        [
            corpus.view_features[item.feature_index][list(FPN_CACHE_TO_SEMANTIC_VIEW_ORDER)]
            for item in selected
        ]
    ).astype(np.float32, copy=False)
    tokens = np.zeros(
        (len(selected), corpus.max_edges, len(EDGE_FEATURE_NAMES)), dtype=np.float32
    )
    mask = np.zeros((len(selected), corpus.max_edges), dtype=bool)
    for batch_index, item in enumerate(selected):
        count = len(item.pattern_tokens)
        tokens[batch_index, :count] = item.pattern_tokens
        mask[batch_index, :count] = True
    return {"view_features": views, "pattern_tokens": tokens, "pattern_mask": mask}


def build_crossmodal_retrieval_model(config: Mapping[str, Any]):
    """Build the learned exact-pattern/FPN dual encoder."""

    import torch
    from torch import nn
    from torch.nn import functional as F

    hidden = int(config.get("hidden_dim", 192))
    heads = int(config.get("num_heads", 6))
    dropout = float(config.get("dropout", 0.1))
    embedding_dim = int(config.get("embedding_dim", 128))
    spatial_dim = int(config.get("spatial_feature_dim", 256))
    max_spatial_tokens = int(config.get("max_spatial_tokens", 85))
    max_edges = int(config["max_edges"])
    pool_queries = int(config.get("pool_queries_per_view", 4))

    class FourViewSpatialEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_norm = nn.LayerNorm(spatial_dim)
            self.input_projection = nn.Linear(spatial_dim, hidden)
            self.spatial_position = nn.Parameter(torch.empty(1, max_spatial_tokens, hidden))
            self.pool_queries = nn.Parameter(torch.empty(1, pool_queries, hidden))
            self.pool_attention = nn.MultiheadAttention(
                hidden, heads, dropout=dropout, batch_first=True
            )
            self.view_embedding = nn.Parameter(torch.empty(1, 4, 1, hidden))
            self.pool_embedding = nn.Parameter(torch.empty(1, 1, pool_queries, hidden))
            self.cls = nn.Parameter(torch.empty(1, 1, hidden))
            layer = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=heads,
                dim_feedforward=hidden * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                layer, num_layers=int(config.get("view_layers", 2)), norm=nn.LayerNorm(hidden)
            )
            self.output = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, embedding_dim))
            nn.init.trunc_normal_(self.spatial_position, std=0.02)
            nn.init.trunc_normal_(self.pool_queries, std=0.02)
            nn.init.trunc_normal_(self.view_embedding, std=0.02)
            nn.init.trunc_normal_(self.pool_embedding, std=0.02)
            nn.init.trunc_normal_(self.cls, std=0.02)

        def forward(self, values):
            if values.ndim != 4 or values.shape[1] != 4:
                raise ValueError("four-view features must have shape [B,4,T,D]")
            batch, views, spatial_tokens, _ = values.shape
            if spatial_tokens > max_spatial_tokens:
                raise ValueError(
                    f"{spatial_tokens} spatial tokens exceed configured maximum {max_spatial_tokens}"
                )
            projected = self.input_projection(self.input_norm(values))
            projected = projected + self.spatial_position[:, :spatial_tokens].unsqueeze(1)
            projected = projected.reshape(batch * views, spatial_tokens, hidden)
            queries = self.pool_queries.expand(batch * views, -1, -1)
            pooled, _ = self.pool_attention(queries, projected, projected, need_weights=False)
            pooled = pooled.reshape(batch, views, pool_queries, hidden)
            pooled = pooled + self.view_embedding + self.pool_embedding
            pooled = pooled.reshape(batch, views * pool_queries, hidden)
            cls = self.cls.expand(batch, -1, -1)
            encoded = self.transformer(torch.cat((cls, pooled), dim=1))
            return F.normalize(self.output(encoded[:, 0]), dim=-1)

    class ExactPatternEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_norm = nn.LayerNorm(len(EDGE_FEATURE_NAMES))
            self.input_projection = nn.Linear(len(EDGE_FEATURE_NAMES), hidden)
            self.edge_position = nn.Parameter(torch.empty(1, max_edges, hidden))
            self.cls = nn.Parameter(torch.empty(1, 1, hidden))
            layer = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=heads,
                dim_feedforward=hidden * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                layer,
                num_layers=int(config.get("pattern_layers", 3)),
                norm=nn.LayerNorm(hidden),
            )
            self.output = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, embedding_dim))
            nn.init.trunc_normal_(self.edge_position, std=0.02)
            nn.init.trunc_normal_(self.cls, std=0.02)

        def forward(self, values, valid_mask):
            if values.ndim != 3 or valid_mask.shape != values.shape[:2]:
                raise ValueError("pattern tokens/mask must have shapes [B,E,D] and [B,E]")
            count = values.shape[1]
            if count > max_edges:
                raise ValueError(f"{count} edge tokens exceed configured maximum {max_edges}")
            encoded = self.input_projection(self.input_norm(values)) + self.edge_position[:, :count]
            cls = self.cls.expand(values.shape[0], -1, -1)
            encoded = torch.cat((cls, encoded), dim=1)
            padding = torch.cat(
                (
                    torch.zeros((values.shape[0], 1), dtype=torch.bool, device=values.device),
                    ~valid_mask.bool(),
                ),
                dim=1,
            )
            encoded = self.transformer(encoded, src_key_padding_mask=padding)
            return F.normalize(self.output(encoded[:, 0]), dim=-1)

    class ExactCrossModalRetrieval(nn.Module):
        def __init__(self):
            super().__init__()
            self.image_encoder = FourViewSpatialEncoder()
            self.pattern_encoder = ExactPatternEncoder()
            temperature = float(config.get("temperature", 0.07))
            self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temperature)))

        def forward(self, view_features, pattern_tokens, pattern_mask):
            return {
                "image_embedding": self.image_encoder(view_features),
                "pattern_embedding": self.pattern_encoder(pattern_tokens, pattern_mask),
            }

        def similarity_logits(self, image_embedding, pattern_embedding):
            scale = self.logit_scale.exp().clamp(max=100.0)
            return scale * image_embedding @ pattern_embedding.T

    return ExactCrossModalRetrieval()


def bidirectional_infonce(model, image_embedding, pattern_embedding):
    import torch
    from torch.nn import functional as F

    logits = model.similarity_logits(image_embedding, pattern_embedding)
    target = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))


def paired_retrieval_metrics(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    query_ids: Sequence[str],
    gallery_ids: Sequence[str],
    *,
    return_rankings: bool = False,
) -> dict[str, Any]:
    """Evaluate exact paired retrieval when every target is present."""

    query = np.asarray(query_embeddings, dtype=np.float32)
    gallery = np.asarray(gallery_embeddings, dtype=np.float32)
    if query.ndim != 2 or gallery.ndim != 2 or query.shape[1] != gallery.shape[1]:
        raise ValueError("query and gallery embeddings must be compatible matrices")
    if len(query_ids) != len(query) or len(gallery_ids) != len(gallery):
        raise ValueError("embedding/id counts differ")
    gallery_lookup = {sample_id: index for index, sample_id in enumerate(gallery_ids)}
    missing = [sample_id for sample_id in query_ids if sample_id not in gallery_lookup]
    if missing:
        raise ValueError(f"paired targets missing from gallery: {missing[:3]}")
    similarity = query @ gallery.T
    ordering = np.argsort(-similarity, axis=1, kind="stable")
    ranks = []
    for query_index, sample_id in enumerate(query_ids):
        target_index = gallery_lookup[sample_id]
        ranks.append(int(np.flatnonzero(ordering[query_index] == target_index)[0]) + 1)
    ranks_array = np.asarray(ranks, dtype=np.int64)
    result: dict[str, Any] = {
        "count": len(ranks),
        "recall_at_1": float(np.mean(ranks_array <= 1)),
        "recall_at_5": float(np.mean(ranks_array <= 5)),
        "recall_at_10": float(np.mean(ranks_array <= 10)),
        "mrr": float(np.mean(1.0 / ranks_array)),
        "median_rank": float(np.median(ranks_array)),
    }
    if return_rankings:
        result["ranks"] = ranks
        result["ordering"] = ordering
        result["similarity"] = similarity
    return result


def normalized_geometry_distance(
    first_tokens: np.ndarray,
    second_tokens: np.ndarray,
    *,
    topology_compatible: bool,
) -> float:
    """Compare exact, dimensionless edge geometry.

    Compatible topologies use ordered edge RMSE.  Incompatible topologies use
    a symmetric edge-token Chamfer distance plus an explicit edge-count term.
    """

    first = np.asarray(first_tokens, dtype=np.float32)[:, GEOMETRY_DISTANCE_INDICES]
    second = np.asarray(second_tokens, dtype=np.float32)[:, GEOMETRY_DISTANCE_INDICES]
    if not len(first) or not len(second):
        return float("inf")
    if topology_compatible and first.shape == second.shape:
        return float(np.sqrt(np.mean((first - second) ** 2)))
    pairwise = np.sqrt(np.mean((first[:, None, :] - second[None, :, :]) ** 2, axis=-1))
    chamfer = 0.5 * (float(pairwise.min(axis=1).mean()) + float(pairwise.min(axis=0).mean()))
    count_penalty = abs(len(first) - len(second)) / max(len(first), len(second))
    return float(chamfer + count_penalty)


def train_bank_retrieval(
    query_embeddings: np.ndarray,
    query_examples: Sequence[ExactRetrievalExample],
    bank_embeddings: np.ndarray,
    bank_examples: Sequence[ExactRetrievalExample],
    *,
    top_k: int = 10,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate unseen image queries against a train-only pattern bank."""

    if not len(query_examples) or not len(bank_examples):
        raise ValueError("query and train bank must both be nonempty")
    query_ids = {item.sample_id for item in query_examples}
    bank_ids = {item.sample_id for item in bank_examples}
    overlap = query_ids & bank_ids
    if overlap:
        raise ValueError(f"train bank contains held-out query IDs: {sorted(overlap)[:3]}")
    similarity = np.asarray(query_embeddings, dtype=np.float32) @ np.asarray(
        bank_embeddings, dtype=np.float32
    ).T
    ordering = np.argsort(-similarity, axis=1, kind="stable")
    rows = []
    distances = []
    category_matches = []
    topology_matches = []
    for index, query_example in enumerate(query_examples):
        winner_index = int(ordering[index, 0])
        winner = bank_examples[winner_index]
        topology_match = query_example.topology_signature == winner.topology_signature
        geometry_distance = normalized_geometry_distance(
            query_example.pattern_tokens,
            winner.pattern_tokens,
            topology_compatible=topology_match,
        )
        category_match = query_example.category == winner.category
        category_matches.append(category_match)
        topology_matches.append(topology_match)
        distances.append(geometry_distance)
        rows.append(
            {
                "sample_id": query_example.sample_id,
                "target_category": query_example.category,
                "retrieved_sample_id": winner.sample_id,
                "retrieved_category": winner.category,
                "similarity": float(similarity[index, winner_index]),
                "category_match": bool(category_match),
                "topology_compatible": bool(topology_match),
                "normalized_geometry_distance": float(geometry_distance),
                "top_train_bank": [
                    {
                        "sample_id": bank_examples[int(value)].sample_id,
                        "category": bank_examples[int(value)].category,
                        "similarity": float(similarity[index, int(value)]),
                    }
                    for value in ordering[index, : min(top_k, len(bank_examples))]
                ],
            }
        )
    metrics = {
        "count": len(rows),
        "exact_target_present": False,
        "category_match_rate": float(np.mean(category_matches)),
        "topology_compatibility_rate": float(np.mean(topology_matches)),
        "normalized_geometry_distance_mean": float(np.mean(distances)),
        "normalized_geometry_distance_median": float(np.median(distances)),
    }
    return metrics, rows
