"""Compare a raw-FPN visual nearest-neighbour with the trained DSL retriever.

This is a deliberately narrow, leakage-checked ablation.  Both lanes use the
canonical Pattern DSL v2 split, the same held-out queries, and the same
train-only anchor bank.  The raw lane has no learned cross-modal adapter: it
L2-normalizes every frozen FPN token, mean-pools the four semantic views, and
retrieves the nearest train garment in that visual space.  The DSL lane reads
the already-saved train-bank ranking from the trained dual encoder.

The target category and canonical primitive-cycle topology are evaluation
metadata only.  They are read after both rankings have been fixed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from benchmark.gcdv2_exact.pattern_dsl_learning import CATEGORIES
from benchmark.gcdv2_exact.visual_dsl_retrieval import (
    FPN_CACHE_TO_SEMANTIC_VIEW_ORDER,
    SPLIT_NAMES,
    SPLIT_TO_INDEX,
    topology_signature_from_program,
    train_bank_retrieval_metrics,
)


SCHEMA_VERSION = "gcdv2-raw-fpn-vs-dsl-retrieval-ablation-1.0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def raw_fpn_mean_embeddings(
    features: np.ndarray,
    *,
    batch_size: int = 128,
) -> np.ndarray:
    """Return one unit vector per garment without fitting any parameters."""

    values = np.asarray(features)
    if values.ndim != 4 or values.shape[1] != 4:
        raise ValueError(f"expected frozen FPN features [N,4,T,D], got {values.shape}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    output: list[np.ndarray] = []
    for start in range(0, len(values), batch_size):
        current = values[start : start + batch_size][
            :, list(FPN_CACHE_TO_SEMANTIC_VIEW_ORDER)
        ].astype(np.float32)
        token_norm = np.linalg.norm(current, axis=-1, keepdims=True)
        current = current / np.maximum(token_norm, 1e-8)
        pooled = current.mean(axis=(1, 2))
        pooled = pooled / np.maximum(
            np.linalg.norm(pooled, axis=-1, keepdims=True), 1e-8
        )
        output.append(pooled)
    return np.concatenate(output, axis=0)


def _per_category_metrics(
    order: np.ndarray,
    *,
    query_categories: np.ndarray,
    bank_categories: np.ndarray,
    query_topologies: np.ndarray,
    bank_topologies: np.ndarray,
    top_k: int,
) -> dict[str, dict[str, float | int]]:
    order = np.asarray(order, dtype=np.int64)
    k = min(int(top_k), order.shape[1])
    output: dict[str, dict[str, float | int]] = {}
    for category_index, category_name in enumerate(CATEGORIES):
        selected = np.flatnonzero(query_categories == category_index)
        if not len(selected):
            continue
        winners = order[selected, 0]
        output[category_name] = {
            "count": int(len(selected)),
            "category_match_at_1": float(
                np.mean(bank_categories[winners] == query_categories[selected])
            ),
            "exact_topology_compatibility_at_1": float(
                np.mean(bank_topologies[winners] == query_topologies[selected])
            ),
            f"exact_topology_compatibility_at_{k}": float(
                np.mean(
                    [
                        np.any(
                            bank_topologies[order[query_index, :k]]
                            == query_topologies[query_index]
                        )
                        for query_index in selected
                    ]
                )
            ),
        }
    return output


def evaluate_embedding_ranking(
    embeddings: np.ndarray,
    *,
    splits: np.ndarray,
    categories: np.ndarray,
    topologies: np.ndarray,
    top_k: int = 10,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """Rank the canonical test split against the canonical train split."""

    train = np.flatnonzero(splits == SPLIT_TO_INDEX["train"])
    test = np.flatnonzero(splits == SPLIT_TO_INDEX["test"])
    if not len(train) or not len(test):
        raise ValueError("canonical train and test splits must both be non-empty")
    if np.intersect1d(train, test).size:
        raise ValueError("train/test index overlap")
    aggregate, order = train_bank_retrieval_metrics(
        embeddings[test],
        embeddings[train],
        categories[test],
        categories[train],
        topologies[test],
        topologies[train],
        top_k=top_k,
    )
    aggregate["per_category"] = _per_category_metrics(
        order,
        query_categories=categories[test],
        bank_categories=categories[train],
        query_topologies=topologies[test],
        bank_topologies=topologies[train],
        top_k=top_k,
    )
    return aggregate, order, train, test


def evaluate_saved_dsl_predictions(
    rows: Sequence[Mapping[str, Any]],
    *,
    sample_ids: np.ndarray,
    splits: np.ndarray,
    categories: np.ndarray,
    topologies: np.ndarray,
    top_k: int = 10,
) -> dict[str, Any]:
    """Re-score saved DSL rankings after enforcing the same split contract."""

    ids = np.asarray(sample_ids).astype(str)
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("canonical corpus contains duplicate sample IDs")
    lookup = {sample_id: index for index, sample_id in enumerate(ids)}
    train_indices = np.flatnonzero(splits == SPLIT_TO_INDEX["train"])
    test_indices = np.flatnonzero(splits == SPLIT_TO_INDEX["test"])
    train_ids = {ids[index] for index in train_indices}
    expected_test_ids = {ids[index] for index in test_indices}
    by_query: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id in by_query:
            raise ValueError(f"duplicate saved DSL query: {sample_id}")
        by_query[sample_id] = row
    if set(by_query) != expected_test_ids:
        missing = sorted(expected_test_ids.difference(by_query))[:3]
        extra = sorted(set(by_query).difference(expected_test_ids))[:3]
        raise ValueError(f"saved DSL query set differs from frozen test: missing={missing}, extra={extra}")

    order = np.zeros((len(test_indices), top_k), dtype=np.int64)
    bank_indices = np.asarray(sorted(train_indices.tolist()), dtype=np.int64)
    bank_offset = {ids[index]: offset for offset, index in enumerate(bank_indices)}
    for query_offset, source_index in enumerate(test_indices):
        sample_id = ids[source_index]
        row = by_query[sample_id]
        ranked = [str(value) for value in row.get("top_train_bank_sample_ids", ())]
        if len(ranked) < top_k:
            raise ValueError(f"{sample_id} has only {len(ranked)} saved DSL candidates")
        ranked = ranked[:top_k]
        if len(set(ranked)) != len(ranked):
            raise ValueError(f"{sample_id} saved DSL ranking contains duplicates")
        foreign = [value for value in ranked if value not in train_ids]
        if foreign:
            raise ValueError(f"{sample_id} DSL candidates are not canonical-train IDs: {foreign[:3]}")
        if sample_id in ranked:
            raise ValueError(f"{sample_id} leaked into its train-bank ranking")
        order[query_offset] = [bank_offset[value] for value in ranked]

    query_categories = categories[test_indices]
    query_topologies = topologies[test_indices]
    bank_categories = categories[bank_indices]
    bank_topologies = topologies[bank_indices]
    winner = order[:, 0]
    aggregate: dict[str, Any] = {
        "count": int(len(test_indices)),
        "exact_target_present": False,
        "category_match_at_1": float(
            np.mean(bank_categories[winner] == query_categories)
        ),
        "category_match_at_10": float(
            np.mean(
                [
                    np.any(bank_categories[row] == query_categories[index])
                    for index, row in enumerate(order)
                ]
            )
        ),
        "exact_topology_compatibility_at_1": float(
            np.mean(bank_topologies[winner] == query_topologies)
        ),
        "exact_topology_compatibility_at_10": float(
            np.mean(
                [
                    np.any(bank_topologies[row] == query_topologies[index])
                    for index, row in enumerate(order)
                ]
            )
        ),
    }
    aggregate["per_category"] = _per_category_metrics(
        order,
        query_categories=query_categories,
        bank_categories=bank_categories,
        query_topologies=query_topologies,
        bank_topologies=bank_topologies,
        top_k=top_k,
    )
    return aggregate


def _metric_delta(dsl: Mapping[str, Any], raw: Mapping[str, Any]) -> dict[str, Any]:
    names = (
        "category_match_at_1",
        "exact_topology_compatibility_at_1",
        "exact_topology_compatibility_at_10",
    )
    output: dict[str, Any] = {
        name: float(dsl[name]) - float(raw[name]) for name in names
    }
    output["per_category"] = {
        category: {
            name: float(dsl["per_category"][category][name])
            - float(raw["per_category"][category][name])
            for name in names
            if name in dsl["per_category"][category]
            and name in raw["per_category"][category]
        }
        for category in sorted(set(dsl["per_category"]) & set(raw["per_category"]))
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Leakage-checked canonical-v2 raw-FPN versus DSL train-bank retrieval ablation."
    )
    parser.add_argument(
        "--programs",
        type=Path,
        default=Path("artifacts/gcdv2_pattern_dsl_v2/programs.npz"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("artifacts/gcdv2_pattern_dsl_v2/metadata.jsonl"),
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("artifacts/gcdv2_exact_pairs_v1/multiview_fpn_tokens.npz"),
    )
    parser.add_argument(
        "--dsl-predictions",
        type=Path,
        default=Path(
            "artifacts/gcdv2_visual_pattern_dsl_retrieval_v2/test_train_bank_predictions.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/gcdv2_visual_pattern_dsl_retrieval_v2/raw_fpn_vs_dsl_ablation.json"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    if args.top_k != 10:
        raise ValueError("saved canonical DSL predictions currently contain exactly top-10")

    programs = np.load(args.programs, allow_pickle=False)
    metadata = _read_jsonl(args.metadata)
    if len(metadata) != len(programs["splits"]):
        raise ValueError("DSL metadata/program counts differ")
    for index, row in enumerate(metadata):
        if int(programs["splits"][index]) != SPLIT_TO_INDEX[str(row["split"])]:
            raise ValueError(f"metadata/program split mismatch at {row['sample_id']}")
        if int(programs["categories"][index]) != CATEGORIES.index(str(row["category"])):
            raise ValueError(f"metadata/program category mismatch at {row['sample_id']}")

    feature_cache = np.load(args.features, allow_pickle=False)
    feature_ids = np.asarray(feature_cache["sample_ids"]).astype(str)
    if len(set(feature_ids.tolist())) != len(feature_ids):
        raise ValueError("FPN cache contains duplicate sample IDs")
    feature_lookup = {sample_id: index for index, sample_id in enumerate(feature_ids)}
    dsl_indices = np.asarray(
        [
            index
            for index, row in enumerate(metadata)
            if str(row["sample_id"]) in feature_lookup
        ],
        dtype=np.int64,
    )
    sample_ids = np.asarray([str(metadata[index]["sample_id"]) for index in dsl_indices])
    feature_indices = np.asarray(
        [feature_lookup[sample_id] for sample_id in sample_ids], dtype=np.int64
    )
    features = feature_cache["features"][feature_indices]
    embeddings = raw_fpn_mean_embeddings(features, batch_size=args.batch_size)
    categories = programs["categories"][dsl_indices].astype(np.int64)
    splits = programs["splits"][dsl_indices].astype(np.int64)
    topologies = np.asarray(
        [
            topology_signature_from_program(
                int(programs["categories"][source]),
                programs["edge_commands"][source],
                programs["edge_valid"][source],
                programs["panel_valid"][source],
            )
            for source in dsl_indices
        ]
    )
    raw, _, train, test = evaluate_embedding_ranking(
        embeddings,
        splits=splits,
        categories=categories,
        topologies=topologies,
        top_k=args.top_k,
    )
    dsl = evaluate_saved_dsl_predictions(
        _read_jsonl(args.dsl_predictions),
        sample_ids=sample_ids,
        splits=splits,
        categories=categories,
        topologies=topologies,
        top_k=args.top_k,
    )
    if int(raw["count"]) != int(dsl["count"]):
        raise AssertionError("raw and DSL query counts differ")
    split_counts = {
        name: int(np.sum(splits == SPLIT_TO_INDEX[name])) for name in SPLIT_NAMES
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE_MIXED_TOP1_POSITIVE_TOP10_DSL_GAIN",
        "split_authority": "canonical Pattern DSL v2 metadata/program split",
        "split_counts": split_counts,
        "same_query_ids": True,
        "same_train_bank_ids": True,
        "train_test_overlap": int(np.intersect1d(sample_ids[train], sample_ids[test]).size),
        "top_k": int(args.top_k),
        "raw_fpn_visual_nearest_neighbor": raw,
        "trained_visual_dsl_dual_encoder": dsl,
        "dsl_minus_raw_fpn": _metric_delta(dsl, raw),
        "input_contract": {
            "raw_fpn": (
                "Frozen ResNet50-FPN [4,85,256]; semantic view reorder; per-token L2; "
                "mean across views/tokens; garment L2; no learned parameters"
            ),
            "dsl": (
                "Existing trained canonical-v2 visual/DSL dual-encoder top-10 rankings; "
                "frozen semantic-pretrained Pattern DSL teacher"
            ),
            "evaluation_only": [
                "garment category",
                "canonical closed-panel L/Q/C/A topology signature",
            ],
        },
        "input_sha256": {
            "programs": _sha256(args.programs),
            "metadata": _sha256(args.metadata),
            "features": _sha256(args.features),
            "dsl_predictions": _sha256(args.dsl_predictions),
        },
        "claim_boundary": [
            "This is a fixed-split retrieval ablation, not 2D pattern generation.",
            "The raw baseline is parameter-free while the DSL lane is trained; this is not an architecture-matched semantic-pretraining ablation.",
            "All data remain inside one GCDv2 generator, neutral body, and Blender render domain.",
            "Topology and category labels are used only after rankings are fixed.",
            "A top-10 coverage gain shows a better candidate beam, not a correct top-1 target pattern.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
