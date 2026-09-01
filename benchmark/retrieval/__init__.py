"""Retrieval-anchored sewing-pattern selection primitives."""

from .anchor_bank import ProceduralAnchor, load_procedural_anchors, rank_dataset_anchors
from .corpus import PatternRecord, build_gcd_ts_record, infer_garment_category
from .index import PatternIndex, QueryEvidence, RetrievalResult

__all__ = [
    "PatternIndex",
    "PatternRecord",
    "ProceduralAnchor",
    "QueryEvidence",
    "RetrievalResult",
    "build_gcd_ts_record",
    "infer_garment_category",
    "load_procedural_anchors",
    "rank_dataset_anchors",
]
