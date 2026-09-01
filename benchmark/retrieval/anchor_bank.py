from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProceduralAnchor:
    anchor_id: str
    category: str
    panel_count: int
    stitch_count: int
    specification_sha256: str
    source: str
    source_commit: str
    source_code_license: str


def load_procedural_anchors(path: Path, *, category: str | None = None) -> tuple[ProceduralAnchor, ...]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    anchors = tuple(
        ProceduralAnchor(
            anchor_id=record["anchor_id"],
            category=record["category"],
            panel_count=int(record["panel_count"]),
            stitch_count=int(record["stitch_count"]),
            specification_sha256=record["specification_sha256"],
            source=raw["source"],
            source_commit=raw["source_commit"],
            source_code_license=raw["source_code_license"],
        )
        for record in raw["records"]
        if category is None or record["category"] == category
    )
    return tuple(sorted(anchors, key=lambda item: (item.panel_count, item.stitch_count, item.anchor_id)))


def rank_dataset_anchors(
    path: Path,
    *,
    category: str | None,
    reweaver_panel_count: int | None,
    reweaver_edge_count: int | None,
    reweaver_reliability: float,
    garment_particles_panel_count: int | None,
    garment_particles_edge_count: int | None,
    garment_particles_reliability: float,
    top_k: int,
) -> tuple[dict, ...]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    canonical_panel_caps = {"top": 10, "pants": 8, "shorts": 8, "skirt": 8, "dress": 14, "jumpsuit": 16}
    ranked = []
    for record in raw["records"]:
        if category is not None and record["category"] != category:
            continue
        if record.get("render_quality", "PASS") != "PASS":
            continue
        observations = []
        for panel_count, edge_count, reliability in (
            (reweaver_panel_count, reweaver_edge_count, reweaver_reliability),
            (garment_particles_panel_count, garment_particles_edge_count, garment_particles_reliability),
        ):
            if panel_count is None or edge_count is None or reliability <= 0:
                continue
            panel_score = math.exp(-abs(panel_count - record["panel_count"]) / max(2.0, 0.5 * record["panel_count"]))
            edge_score = math.exp(-abs(edge_count - record["edge_count"]) / max(4.0, 0.5 * record["edge_count"]))
            observations.append((0.65 * panel_score + 0.35 * edge_score, reliability))
        if observations:
            total = sum(weight for _, weight in observations)
            raw_structural = sum(score * weight for score, weight in observations) / total
            confidence = min(1.0, total / len(observations))
            structural = 0.5 + confidence * (raw_structural - 0.5)
        else:
            structural = 0.5
        complexity = 1.0 / (1.0 + 0.08 * record["panel_count"])
        cap = canonical_panel_caps.get(record["category"], 12)
        canonical_topology = math.exp(-max(0, record["panel_count"] - cap) / 2.0)
        paired_split = 0.0 if record["split"] == "not_in_official_paired_split" else 1.0
        ranked.append(
            {
                "sample_id": record["sample_id"],
                "category": record["category"],
                "score": 0.52 * structural + 0.25 * complexity + 0.20 * canonical_topology + 0.03 * paired_split,
                "structural_hint_score": structural,
                "complexity_prior": complexity,
                "canonical_topology_prior": canonical_topology,
                "paired_split_prior": paired_split,
                "panel_count": record["panel_count"],
                "edge_count": record["edge_count"],
                "stitch_count": record["stitch_count"],
                "split": record["split"],
                "specification_sha256": record["specification_sha256"],
                "source_dataset": raw["dataset"],
                "source_license": raw["license"],
                "selection_scope": "category_soft_topology_and_canonical_complexity_without_visual_descriptor",
            }
        )
    ranked.sort(key=lambda item: (-item["score"], item["panel_count"], item["sample_id"]))
    return tuple(ranked[: max(1, top_k)])
