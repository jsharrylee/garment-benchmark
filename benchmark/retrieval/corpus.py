from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .features import multiview_descriptor


SEMANTIC_TOKENS = ("torso", "sleeve", "cuff", "skirt", "waistband", "hood", "collar", "pants", "shorts")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def infer_garment_category(panel_names: list[str] | tuple[str, ...]) -> str:
    joined = " ".join(panel_names).lower()
    has_torso = "torso" in joined
    # GarmentCode also uses names such as cuff_skirt for flared sleeve cuffs.
    # Only actual skirt panels should promote a garment to skirt/dress.
    has_skirt = any(name.lower().startswith("skirt_") or name.lower() in {"skirt", "skirt_front", "skirt_back"} for name in panel_names)
    if ("pants" in joined or "pant_" in joined or "leg_" in joined) and has_torso:
        return "jumpsuit"
    if "pants" in joined or "pant_" in joined or "leg_" in joined:
        return "pants"
    if "short" in joined:
        return "shorts"
    if has_torso and has_skirt:
        return "dress"
    if has_skirt:
        return "skirt"
    if "hood" in joined or "collar" in joined:
        return "top"
    if has_torso or "sleeve" in joined:
        return "top"
    return "unknown"


@dataclass(frozen=True)
class PatternRecord:
    sample_id: str
    category: str
    panel_names: tuple[str, ...]
    panel_count: int
    edge_count: int
    mean_edges_per_panel: float
    semantic_counts: dict[str, int]
    visual_descriptor: tuple[float, ...]
    source_pattern_sha256: str
    view_sha256: tuple[str, ...]
    source_pattern: str | None = None
    source_views: tuple[str, ...] = ()
    source_dataset: str = "ReWeaver-GCD-TS"
    source_license: str = "CC BY-NC 4.0"

    def to_dict(self, *, include_local_paths: bool = True) -> dict[str, Any]:
        result = asdict(self)
        if not include_local_paths:
            result.pop("source_pattern", None)
            result.pop("source_views", None)
            result.pop("visual_descriptor", None)
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PatternRecord":
        return cls(
            sample_id=value["sample_id"],
            category=value["category"],
            panel_names=tuple(value["panel_names"]),
            panel_count=int(value["panel_count"]),
            edge_count=int(value["edge_count"]),
            mean_edges_per_panel=float(value["mean_edges_per_panel"]),
            semantic_counts={str(key): int(count) for key, count in value["semantic_counts"].items()},
            visual_descriptor=tuple(float(item) for item in value["visual_descriptor"]),
            source_pattern_sha256=value["source_pattern_sha256"],
            view_sha256=tuple(value["view_sha256"]),
            source_pattern=value.get("source_pattern"),
            source_views=tuple(value.get("source_views", [])),
            source_dataset=value.get("source_dataset", "ReWeaver-GCD-TS"),
            source_license=value.get("source_license", "CC BY-NC 4.0"),
        )


def build_gcd_ts_record(pattern_path: Path, view_paths: list[Path] | tuple[Path, ...]) -> PatternRecord:
    pattern_path = Path(pattern_path)
    views = tuple(sorted((Path(path) for path in view_paths), key=lambda value: value.name.lower()))
    raw = json.loads(pattern_path.read_text(encoding="utf-8"))
    panel_names = tuple(str(name) for name in raw["panel_order"])
    panels = raw["panels"]
    edge_counts = [len(panels[str(index)]["edge_points"]) for index in range(len(panel_names))]
    semantic_counts = {
        token: sum(token in panel_name.lower() or (token == "waistband" and panel_name.lower().startswith("wb_")) for panel_name in panel_names)
        for token in SEMANTIC_TOKENS
    }
    return PatternRecord(
        sample_id=pattern_path.stem.replace("_2d_panel", ""),
        category=infer_garment_category(panel_names),
        panel_names=panel_names,
        panel_count=len(panel_names),
        edge_count=sum(edge_counts),
        mean_edges_per_panel=float(sum(edge_counts) / max(1, len(edge_counts))),
        semantic_counts=semantic_counts,
        visual_descriptor=multiview_descriptor(views),
        source_pattern_sha256=sha256(pattern_path),
        view_sha256=tuple(sha256(path) for path in views),
        source_pattern=str(pattern_path),
        source_views=tuple(str(path) for path in views),
    )
