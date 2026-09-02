from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


Point2 = tuple[float, float]
Point3 = tuple[float, float, float]


@dataclass(frozen=True)
class Edge:
    id: str
    points: tuple[Point2, ...]
    source_curve_id: int | None = None
    confidence: float | None = None


@dataclass(frozen=True)
class Placement:
    origin: Point3
    x_axis: Point3
    y_axis: Point3
    normal: Point3
    method: str = "predicted_surface_pca"


@dataclass(frozen=True)
class Panel:
    id: str
    edges: tuple[Edge, ...]
    placement: Placement | None = None
    source_panel_id: int | None = None
    confidence: float | None = None


@dataclass(frozen=True)
class StitchSide:
    panel_id: str
    edge_id: str
    reversed: bool = False


@dataclass(frozen=True)
class Stitch:
    id: str
    side_a: StitchSide
    side_b: StitchSide
    source_curve_id: int | None = None
    confidence: float | None = None


@dataclass(frozen=True)
class PatternDocument:
    pattern_id: str
    generator: str
    panels: tuple[Panel, ...]
    stitches: tuple[Stitch, ...]
    units: str = "cm"
    schema_version: str = "1.0"
    provenance: dict[str, Any] = field(default_factory=dict)
    annotations: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PatternDocument":
        panels = []
        for raw_panel in value["panels"]:
            placement = raw_panel.get("placement")
            panels.append(
                Panel(
                    id=raw_panel["id"],
                    edges=tuple(
                        Edge(
                            id=edge["id"],
                            points=tuple(tuple(float(v) for v in point) for point in edge["points"]),
                            source_curve_id=edge.get("source_curve_id"),
                            confidence=edge.get("confidence"),
                        )
                        for edge in raw_panel["edges"]
                    ),
                    placement=Placement(**placement) if placement else None,
                    source_panel_id=raw_panel.get("source_panel_id"),
                    confidence=raw_panel.get("confidence"),
                )
            )
        stitches = tuple(
            Stitch(
                id=item["id"],
                side_a=StitchSide(**item["side_a"]),
                side_b=StitchSide(**item["side_b"]),
                source_curve_id=item.get("source_curve_id"),
                confidence=item.get("confidence"),
            )
            for item in value.get("stitches", [])
        )
        return cls(
            pattern_id=value["pattern_id"],
            generator=value["generator"],
            panels=tuple(panels),
            stitches=stitches,
            units=value.get("units", "cm"),
            schema_version=value.get("schema_version", "1.0"),
            provenance=value.get("provenance", {}),
            annotations=value.get("annotations", {}),
        )

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    @classmethod
    def read_json(cls, path: Path) -> "PatternDocument":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
