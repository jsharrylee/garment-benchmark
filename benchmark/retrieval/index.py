from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .corpus import PatternRecord
from .features import normalized_l1_similarity


@dataclass(frozen=True)
class QueryEvidence:
    visual_descriptor: tuple[float, ...]
    category: str | None = None
    reweaver_panel_count: int | None = None
    reweaver_edge_count: int | None = None
    reweaver_reliability: float = 0.0
    garment_particles_panel_count: int | None = None
    garment_particles_edge_count: int | None = None
    garment_particles_stitch_count: int | None = None
    garment_particles_reliability: float = 0.0

    @classmethod
    def from_files(
        cls,
        visual_descriptor: tuple[float, ...],
        *,
        category: str | None = None,
        reweaver_summary: Path | None = None,
        garment_particles_summary: Path | None = None,
    ) -> "QueryEvidence":
        rw: dict[str, Any] = {}
        gp: dict[str, Any] = {}
        if reweaver_summary:
            rw = json.loads(Path(reweaver_summary).read_text(encoding="utf-8"))
        if garment_particles_summary:
            gp = json.loads(Path(garment_particles_summary).read_text(encoding="utf-8"))
        rw_gap = rw.get("mean_boundary_closure_gap")
        gp_gap = gp.get("panel_closure_gap_max")
        rw_reliability = math.exp(-max(0.0, float(rw_gap)) / 0.05) if rw_gap is not None else 0.0
        gp_reliability = math.exp(-max(0.0, float(gp_gap)) / 1.0) if gp_gap is not None else 0.0
        return cls(
            visual_descriptor=visual_descriptor,
            category=category,
            reweaver_panel_count=rw.get("panel_count"),
            reweaver_edge_count=rw.get("curve_count"),
            reweaver_reliability=rw_reliability,
            garment_particles_panel_count=gp.get("panel_count"),
            garment_particles_edge_count=gp.get("edge_count"),
            garment_particles_stitch_count=gp.get("stitch_pair_count"),
            garment_particles_reliability=gp_reliability,
        )


@dataclass(frozen=True)
class RetrievalCandidate:
    sample_id: str
    category: str
    score: float
    visual_similarity: float
    reweaver_similarity: float | None
    garment_particles_similarity: float | None
    complexity_prior: float
    panel_count: int
    edge_count: int
    panel_names: tuple[str, ...]
    source_pattern_sha256: str


@dataclass(frozen=True)
class RetrievalResult:
    decision: str
    requested_category: str | None
    candidates: tuple[RetrievalCandidate, ...]
    corpus_size: int
    eligible_size: int
    final_acceptance: str = "REQUIRES_SIMULATION_RERANK"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _count_similarity(panel_count: int, edge_count: int, record: PatternRecord) -> float:
    panel_scale = max(2.0, 0.5 * record.panel_count)
    edge_scale = max(4.0, 0.5 * record.edge_count)
    panel_score = math.exp(-abs(panel_count - record.panel_count) / panel_scale)
    edge_score = math.exp(-abs(edge_count - record.edge_count) / edge_scale)
    return 0.65 * panel_score + 0.35 * edge_score


class PatternIndex:
    def __init__(self, records: list[PatternRecord] | tuple[PatternRecord, ...]):
        self.records = tuple(records)
        if not self.records:
            raise ValueError("pattern index requires at least one record")
        lengths = {len(record.visual_descriptor) for record in self.records}
        if len(lengths) != 1:
            raise ValueError("all visual descriptors must have equal length")

    @classmethod
    def read_json(cls, path: Path) -> "PatternIndex":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([PatternRecord.from_dict(item) for item in raw["records"]])

    def write_json(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"schema_version": "2.0", "mode": "retrieval_anchored_v2", "records": [record.to_dict() for record in self.records]}, indent=2)
            + "\n",
            encoding="utf-8",
            newline="\n",
        )

    def search(self, query: QueryEvidence, *, top_k: int = 5, minimum_score: float = 0.45) -> RetrievalResult:
        eligible = [record for record in self.records if query.category is None or record.category == query.category]
        if not eligible:
            return RetrievalResult("NO_SUITABLE_ANCHOR", query.category, (), len(self.records), 0, "BLOCKED_NO_CATEGORY_COVERAGE")
        candidates: list[RetrievalCandidate] = []
        for record in eligible:
            visual = normalized_l1_similarity(query.visual_descriptor, record.visual_descriptor)
            rw = None
            if query.reweaver_panel_count is not None and query.reweaver_edge_count is not None:
                rw = _count_similarity(query.reweaver_panel_count, query.reweaver_edge_count, record)
            gp = None
            if query.garment_particles_panel_count is not None and query.garment_particles_edge_count is not None:
                gp = _count_similarity(query.garment_particles_panel_count, query.garment_particles_edge_count, record)
            weighted: list[tuple[float, float]] = []
            if rw is not None and query.reweaver_reliability > 0.0:
                weighted.append((rw, query.reweaver_reliability))
            if gp is not None and query.garment_particles_reliability > 0.0:
                weighted.append((gp, query.garment_particles_reliability))
            if weighted:
                total_reliability = sum(weight for _, weight in weighted)
                raw_structural = sum(value * weight for value, weight in weighted) / total_reliability
                confidence = min(1.0, total_reliability / len(weighted))
                structural = 0.5 + confidence * (raw_structural - 0.5)
            else:
                structural = 0.5
            complexity = 1.0 / (1.0 + 0.03 * record.panel_count)
            score = 0.65 * visual + 0.25 * structural + 0.10 * complexity
            candidates.append(
                RetrievalCandidate(
                    sample_id=record.sample_id,
                    category=record.category,
                    score=score,
                    visual_similarity=visual,
                    reweaver_similarity=rw,
                    garment_particles_similarity=gp,
                    complexity_prior=complexity,
                    panel_count=record.panel_count,
                    edge_count=record.edge_count,
                    panel_names=record.panel_names,
                    source_pattern_sha256=record.source_pattern_sha256,
                )
            )
        candidates.sort(key=lambda item: (-item.score, item.panel_count, item.sample_id))
        selected = tuple(candidates[: max(1, top_k)])
        decision = "ANCHOR_CANDIDATES_AVAILABLE" if selected[0].score >= minimum_score else "NO_SUITABLE_ANCHOR"
        final = "REQUIRES_SIMULATION_RERANK" if decision == "ANCHOR_CANDIDATES_AVAILABLE" else "BLOCKED_LOW_RETRIEVAL_CONFIDENCE"
        return RetrievalResult(decision, query.category, selected, len(self.records), len(eligible), final)
