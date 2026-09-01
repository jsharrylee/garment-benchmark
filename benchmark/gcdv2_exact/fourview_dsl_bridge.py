"""Bridge four-view retrieval to the neural-symbolic Pattern DSL.

This module intentionally implements *retrieval plus interpretation*, not
image-to-pattern generation.  A four-view query first selects a pattern from
the cross-modal retrieval model's train bank.  The selected anchor's observed
vector geometry is then passed to the Pattern-DSL proposer and symbolic
projector.

Only geometry inputs used by :class:`PatternDSLTransformer` are exposed here.
In particular, source ``stitch_pairs`` are neither read by the bridge model nor
copied to its review records.  Every emitted ``SEWN_TO`` fact is a neural
proposal accepted by the symbolic matching constraint.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import json
import math

import numpy as np

from benchmark.gcdv2_exact.pattern_dsl_learning import CATEGORIES, CURVE_COMMANDS
from benchmark.gcdv2_exact.pattern_dsl_solver import SymbolicProjectionReport


NEURAL_GEOMETRY_ARRAY_KEYS = (
    "edge_features",
    "edge_commands",
    "edge_valid",
    "panel_valid",
)


@dataclass(frozen=True)
class RetrievalCatalogEntry:
    sample_id: str
    category: str
    split: str
    topology_signature: str


@dataclass(frozen=True)
class AnchorSelection:
    query_sample_id: str
    anchor_sample_id: str
    anchor_rank: int
    similarity: float | None
    used_saved_top1: bool
    rejected_candidates: tuple[Mapping[str, Any], ...]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def retrieval_catalog_from_arrays(
    sample_ids: Sequence[Any],
    categories: Sequence[Any],
    splits: Sequence[Any],
    topology_signatures: Sequence[Any],
) -> dict[str, RetrievalCatalogEntry]:
    lengths = {len(sample_ids), len(categories), len(splits), len(topology_signatures)}
    if len(lengths) != 1:
        raise ValueError("retrieval catalogue arrays must have equal lengths")
    output: dict[str, RetrievalCatalogEntry] = {}
    for sample_id, category, split, topology in zip(
        sample_ids, categories, splits, topology_signatures, strict=True
    ):
        key = str(sample_id)
        if key in output:
            raise ValueError(f"duplicate retrieval sample id: {key}")
        output[key] = RetrievalCatalogEntry(
            sample_id=key,
            category=str(category),
            split=str(split),
            topology_signature=str(topology),
        )
    return output


def load_retrieval_catalog(path: Path) -> dict[str, RetrievalCatalogEntry]:
    """Load only retrieval metadata; embeddings are deliberately untouched."""

    archive = np.load(Path(path), allow_pickle=False)
    return retrieval_catalog_from_arrays(
        archive["sample_ids"],
        archive["categories"],
        archive["splits"],
        archive["topology_signatures"],
    )


def metadata_lookup(rows: Sequence[Mapping[str, Any]]) -> dict[str, tuple[int, Mapping[str, Any]]]:
    output: dict[str, tuple[int, Mapping[str, Any]]] = {}
    for source_index, row in enumerate(rows):
        sample_id = str(row["sample_id"])
        if sample_id in output:
            raise ValueError(f"duplicate Pattern-DSL sample id: {sample_id}")
        output[sample_id] = (source_index, row)
    return output


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _candidate_rows(prediction: Mapping[str, Any]) -> list[tuple[str, float | None]]:
    rows: list[tuple[str, float | None]] = []
    if prediction.get("retrieved_sample_id") is not None:
        rows.append(
            (
                str(prediction["retrieved_sample_id"]),
                _optional_float(prediction.get("similarity")),
            )
        )
    for candidate in prediction.get("top_train_bank", ()):
        rows.append((str(candidate["sample_id"]), _optional_float(candidate.get("similarity"))))
    # The saved top-1 is generally repeated as top_train_bank[0].  Stable
    # de-duplication keeps the rank meaningful while retaining fallbacks.
    seen: set[str] = set()
    return [value for value in rows if not (value[0] in seen or seen.add(value[0]))]


def adapt_aligned_dsl_prediction(row: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize the DSL-split-aligned visual retriever's compact JSON schema."""

    if "top_train_bank_sample_ids" not in row:
        raise ValueError("aligned prediction is missing top_train_bank_sample_ids")
    candidates = [str(value) for value in row["top_train_bank_sample_ids"]]
    return {
        "sample_id": str(row["sample_id"]),
        "retrieved_sample_id": str(row["retrieved_sample_id"]),
        # The aligned prediction artifact intentionally stores ranking, not
        # cosine scores.  Keep absence explicit instead of manufacturing 0.0.
        "similarity": None,
        "top_train_bank": [
            {"sample_id": sample_id, "similarity": None} for sample_id in candidates
        ],
        "category_match": bool(row["category_match"]),
        "topology_compatible": bool(
            row["exact_closed_cycle_primitive_topology_match"]
        ),
        "source_prediction_schema": "gcdv2-visual-pattern-dsl-retrieval-1.0",
    }


def aligned_dsl_retrieval_catalog(
    metadata_rows: Sequence[Mapping[str, Any]], arrays: Mapping[str, np.ndarray]
) -> dict[str, RetrievalCatalogEntry]:
    """Build retrieval metadata from the authoritative Pattern-DSL corpus."""

    from benchmark.gcdv2_exact.visual_dsl_retrieval import (
        SPLIT_NAMES,
        topology_signature_from_program,
    )

    count = len(metadata_rows)
    for key in ("categories", "splits", "edge_commands", "edge_valid", "panel_valid"):
        if len(arrays[key]) != count:
            raise ValueError(f"Pattern-DSL metadata/{key} counts differ")
    output: dict[str, RetrievalCatalogEntry] = {}
    for index, row in enumerate(metadata_rows):
        sample_id = str(row["sample_id"])
        category_index = int(arrays["categories"][index])
        split_index = int(arrays["splits"][index])
        split = SPLIT_NAMES[split_index]
        if str(row["split"]) != split:
            raise ValueError(f"Pattern-DSL split mismatch at {sample_id}")
        if sample_id in output:
            raise ValueError(f"duplicate Pattern-DSL catalogue ID: {sample_id}")
        output[sample_id] = RetrievalCatalogEntry(
            sample_id=sample_id,
            category=CATEGORIES[category_index],
            split=split,
            topology_signature=topology_signature_from_program(
                category_index,
                arrays["edge_commands"][index],
                arrays["edge_valid"][index],
                arrays["panel_valid"][index],
            ),
        )
    return output


def select_train_bank_anchor(
    prediction: Mapping[str, Any],
    *,
    retrieval_catalog: Mapping[str, RetrievalCatalogEntry],
    dsl_lookup: Mapping[str, tuple[int, Mapping[str, Any]]],
) -> AnchorSelection:
    """Select the first usable raw train-bank candidate without semantic filters."""

    query = str(prediction["sample_id"])
    rejected: list[Mapping[str, Any]] = []
    for rank, (candidate, similarity) in enumerate(_candidate_rows(prediction), start=1):
        reason: str | None = None
        if candidate == query:
            reason = "query_target_id_forbidden"
        elif candidate not in retrieval_catalog:
            reason = "missing_retrieval_catalog_entry"
        elif retrieval_catalog[candidate].split != "train":
            reason = f"not_retrieval_train:{retrieval_catalog[candidate].split}"
        elif candidate not in dsl_lookup:
            reason = "missing_pattern_dsl_geometry"
        if reason is not None:
            rejected.append({"sample_id": candidate, "rank": rank, "reason": reason})
            continue
        return AnchorSelection(
            query_sample_id=query,
            anchor_sample_id=candidate,
            anchor_rank=rank,
            similarity=similarity,
            used_saved_top1=candidate == str(prediction.get("retrieved_sample_id", "")),
            rejected_candidates=tuple(rejected),
        )
    raise ValueError(f"no usable train-bank Pattern-DSL anchor for query {query}")


def neural_geometry_input(
    arrays: Mapping[str, np.ndarray], source_index: int
) -> dict[str, np.ndarray]:
    """Return the complete and exclusive neural input for one anchor.

    Keeping this whitelist explicit is a leakage guard: ``stitch_pairs``,
    semantic labels, source IDs and absolute coordinates cannot silently enter
    the model call.
    """

    return {key: np.asarray(arrays[key][source_index]) for key in NEURAL_GEOMETRY_ARRAY_KEYS}


def split_symbolic_facts(report: SymbolicProjectionReport) -> dict[str, list[str]]:
    buckets = {"role": [], "next": [], "seam": [], "landmark": []}
    for fact in report.facts():
        if fact.startswith("ROLE("):
            buckets["role"].append(fact)
        elif fact.startswith("NEXT("):
            buckets["next"].append(fact)
        elif fact.startswith("SEWN_TO("):
            buckets["seam"].append(fact)
        elif fact.startswith("LANDMARK("):
            buckets["landmark"].append(fact)
        else:  # pragma: no cover - fail loudly if the solver vocabulary grows
            raise ValueError(f"unknown symbolic fact: {fact}")
    return buckets


def make_bridge_record(
    *,
    prediction: Mapping[str, Any],
    selection: AnchorSelection,
    retrieval_catalog: Mapping[str, RetrievalCatalogEntry],
    dsl_metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    source_index: int,
    predicted_category_index: int,
    report: SymbolicProjectionReport,
) -> dict[str, Any]:
    query = retrieval_catalog[selection.query_sample_id]
    anchor = retrieval_catalog[selection.anchor_sample_id]
    edge_valid = np.asarray(arrays["edge_valid"][source_index], dtype=bool)
    panel_valid = np.asarray(arrays["panel_valid"][source_index], dtype=bool)
    commands = np.asarray(arrays["edge_commands"][source_index])
    command_counts = Counter(
        CURVE_COMMANDS[int(command)] for command in commands[edge_valid]
    )
    facts = split_symbolic_facts(report)
    landmark_counts = Counter(value.base_name for value in report.landmarks)
    predicted_category = CATEGORIES[int(predicted_category_index)]
    saved_category_match = prediction.get("category_match")
    saved_topology_match = prediction.get("topology_compatible")
    category_match = query.category == anchor.category
    topology_match = query.topology_signature == anchor.topology_signature
    return {
        "query": {
            "sample_id": query.sample_id,
            "split": query.split,
            "category": query.category,
            "topology_signature": query.topology_signature,
            "input": "four_view_frozen_fpn_embedding",
        },
        "retrieved_anchor": {
            "sample_id": anchor.sample_id,
            "retrieval_split": anchor.split,
            "dsl_corpus_split": str(dsl_metadata["split"]),
            "category": anchor.category,
            "topology_signature": anchor.topology_signature,
            "rank_after_availability_check": selection.anchor_rank,
            "similarity": selection.similarity,
            "used_saved_top1": selection.used_saved_top1,
            "rejected_candidates": list(selection.rejected_candidates),
        },
        "retrieval": {
            "category_match": category_match,
            "topology_compatible": topology_match,
            "saved_category_contract_agrees": (
                saved_category_match is None or bool(saved_category_match) == category_match
            ),
            "saved_topology_contract_agrees": (
                saved_topology_match is None or bool(saved_topology_match) == topology_match
            ),
            "category_filter_used": False,
            "topology_filter_used": False,
            "exact_target_id_available_to_anchor_bank": False,
        },
        "anchor_observed_geometry": {
            "panel_count": int(panel_valid.sum()),
            "edge_count": int(edge_valid.sum()),
            "svg_command_counts": dict(sorted(command_counts.items())),
            "neural_input_array_keys": list(NEURAL_GEOMETRY_ARRAY_KEYS),
            "absolute_xy_used": False,
            "source_stitches_used": False,
            "source_semantic_labels_used": False,
        },
        "neural_proposer": {
            "predicted_anchor_category": predicted_category,
            "anchor_category_agreement": predicted_category == anchor.category,
        },
        "symbolic_projection": {
            "valid": report.valid,
            "raw_grammar_violations": len(report.roles.raw_violations),
            "projected_grammar_violations": len(report.roles.projected_violations),
            "changed_role_edges": report.roles.changed_edges,
            "predicted_seam_pair_count": len(report.seams.pairs),
            "seam_maximum_degree": report.seams.maximum_degree,
            "derived_landmark_count": len(report.landmarks),
            "derived_landmark_counts": dict(sorted(landmark_counts.items())),
            "facts": facts,
        },
        "claim_boundary": (
            "4-view query -> retrieved train-bank anchor -> anchor geometry DSL facts. "
            "This record is not target-pattern DSL generation or target-pattern reconstruction."
        ),
    }


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / max(denominator, 1)


def summarize_bridge_records(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    dsl_splits: Counter[str] = Counter()
    landmark_counts: Counter[str] = Counter()
    rejected_reasons: Counter[str] = Counter()
    by_category: dict[str, Counter[str]] = {}
    totals = Counter()
    similarities: list[float] = []
    for row in rows:
        retrieval = row["retrieval"]
        query_category = str(row.get("query", {}).get("category", "unknown"))
        category_totals = by_category.setdefault(query_category, Counter())
        anchor = row["retrieved_anchor"]
        symbolic = row["symbolic_projection"]
        proposer = row["neural_proposer"]
        totals["category_match"] += int(retrieval["category_match"])
        totals["topology_match"] += int(retrieval["topology_compatible"])
        category_totals["count"] += 1
        category_totals["category_match"] += int(retrieval["category_match"])
        category_totals["topology_match"] += int(retrieval["topology_compatible"])
        totals["saved_category_agrees"] += int(retrieval["saved_category_contract_agrees"])
        totals["saved_topology_agrees"] += int(retrieval["saved_topology_contract_agrees"])
        totals["saved_top1"] += int(anchor["used_saved_top1"])
        totals["symbolic_valid"] += int(symbolic["valid"])
        totals["anchor_category_agreement"] += int(proposer["anchor_category_agreement"])
        totals["raw_grammar"] += int(symbolic["raw_grammar_violations"])
        totals["projected_grammar"] += int(symbolic["projected_grammar_violations"])
        totals["changed_role_edges"] += int(symbolic["changed_role_edges"])
        totals["predicted_seams"] += int(symbolic["predicted_seam_pair_count"])
        totals["landmarks"] += int(symbolic["derived_landmark_count"])
        dsl_splits[str(anchor["dsl_corpus_split"])] += 1
        landmark_counts.update(symbolic["derived_landmark_counts"])
        for candidate in anchor["rejected_candidates"]:
            rejected_reasons[str(candidate["reason"])] += 1
        similarity = _optional_float(anchor["similarity"])
        if similarity is not None:
            similarities.append(similarity)
    count = len(rows)
    return {
        "status": "PASS_FOURVIEW_TO_RETRIEVED_ANCHOR_PATTERN_DSL_BRIDGE",
        "sample_count": count,
        "retrieval": {
            "category_match_rate": _ratio(totals["category_match"], count),
            "exact_topology_compatibility_rate": _ratio(totals["topology_match"], count),
            "mean_similarity": float(np.mean(similarities)) if similarities else None,
            "saved_top1_used_rate": _ratio(totals["saved_top1"], count),
            "saved_category_contract_agreement_rate": _ratio(
                totals["saved_category_agrees"], count
            ),
            "saved_topology_contract_agreement_rate": _ratio(
                totals["saved_topology_agrees"], count
            ),
            "rejected_fallback_candidates": dict(sorted(rejected_reasons.items())),
            "category_filter_used": False,
            "topology_filter_used": False,
            "by_query_category": {
                category: {
                    "count": values["count"],
                    "category_match_rate": _ratio(values["category_match"], values["count"]),
                    "exact_topology_compatibility_rate": _ratio(
                        values["topology_match"], values["count"]
                    ),
                }
                for category, values in sorted(by_category.items())
            },
        },
        "anchor_dsl": {
            "retrieval_anchor_split": "train",
            "dsl_corpus_split_distribution": dict(sorted(dsl_splits.items())),
            "neural_input_array_keys": list(NEURAL_GEOMETRY_ARRAY_KEYS),
            "source_stitches_consumed": False,
            "source_semantic_labels_consumed": False,
            "absolute_xy_consumed": False,
            "proposer_anchor_category_accuracy": _ratio(
                totals["anchor_category_agreement"], count
            ),
        },
        "symbolic_projection": {
            "valid_rate": _ratio(totals["symbolic_valid"], count),
            "raw_grammar_violations": totals["raw_grammar"],
            "projected_grammar_violations": totals["projected_grammar"],
            "changed_role_edges": totals["changed_role_edges"],
            "predicted_seam_facts": totals["predicted_seams"],
            "predicted_seam_facts_per_sample": _ratio(totals["predicted_seams"], count),
            "derived_landmark_facts": totals["landmarks"],
            "derived_landmark_counts": dict(sorted(landmark_counts.items())),
        },
        "claim_boundary": [
            "The four-view encoder retrieves a train-bank anchor; it does not generate the held-out target DSL.",
            "Pattern-DSL inference interprets the retrieved anchor's observed vector geometry, not target geometry.",
            "SEWN_TO facts are neural proposals after one-mate symbolic projection; no source stitch pairs are consumed or scored.",
            "Topology compatibility is exact train-anchor versus held-out-query signature equality, not geometric edit quality.",
            "No category or topology filter changes the retrieval ranking.",
        ],
    }


__all__ = [
    "AnchorSelection",
    "NEURAL_GEOMETRY_ARRAY_KEYS",
    "RetrievalCatalogEntry",
    "adapt_aligned_dsl_prediction",
    "aligned_dsl_retrieval_catalog",
    "load_retrieval_catalog",
    "make_bridge_record",
    "metadata_lookup",
    "neural_geometry_input",
    "read_jsonl",
    "retrieval_catalog_from_arrays",
    "select_train_bank_anchor",
    "split_symbolic_facts",
    "summarize_bridge_records",
]
