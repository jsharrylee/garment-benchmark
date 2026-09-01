from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import svgpathtools as svgpath
from scipy.spatial.transform import Rotation

from benchmark.pattern_pipeline.schema import Edge, Panel, PatternDocument, Placement, Stitch, StitchSide


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_point(start: np.ndarray, end: np.ndarray, relative: list[float]) -> np.ndarray:
    vector = end - start
    return start + float(relative[0]) * vector + float(relative[1]) * np.array([-vector[1], vector[0]])


def _sample_edge(vertices: np.ndarray, edge: dict, samples: int = 30) -> np.ndarray:
    start = vertices[int(edge["endpoints"][0])]
    end = vertices[int(edge["endpoints"][1])]
    t = np.linspace(0.0, 1.0, samples)[:, None]
    curvature = edge.get("curvature")
    if not curvature:
        return (1.0 - t) * start + t * end
    if isinstance(curvature, list):
        kind, params = "quadratic", [curvature]
    else:
        kind, params = curvature["type"], curvature["params"]
    if kind == "quadratic":
        control = _relative_point(start, end, params[0])
        return (1.0 - t) ** 2 * start + 2.0 * (1.0 - t) * t * control + t**2 * end
    if kind == "cubic":
        first = _relative_point(start, end, params[0])
        second = _relative_point(start, end, params[1])
        return (1.0 - t) ** 3 * start + 3.0 * (1.0 - t) ** 2 * t * first + 3.0 * (1.0 - t) * t**2 * second + t**3 * end
    if kind == "circle":
        radius, large_arc, right = params
        arc = svgpath.Arc(
            complex(*start),
            complex(float(radius), float(radius)),
            rotation=0,
            large_arc=bool(large_arc),
            sweep=not bool(right),
            end=complex(*end),
        )
        return np.asarray([[arc.point(float(value)).real, arc.point(float(value)).imag] for value in np.linspace(0.0, 1.0, samples)])
    raise ValueError(f"unsupported GarmentCode curvature type: {kind}")


def convert_garmentcode_specification(
    path: Path,
    *,
    anchor_id: str | None = None,
    panel_mesh_spacing_cm: float = 0.0,
    source_license: str = "MIT",
) -> PatternDocument:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    pattern = raw["pattern"]
    panels = []
    labels = {}
    edge_labels = {}
    for panel_name, source in pattern["panels"].items():
        vertices = np.asarray(source["vertices"], dtype=float)
        edges = tuple(
            Edge(
                id=f"{panel_name}.edge_{index}",
                points=tuple((float(point[0]), float(point[1])) for point in _sample_edge(vertices, source_edge)),
                source_curve_id=index,
                confidence=1.0,
            )
            for index, source_edge in enumerate(source["edges"])
        )
        matrix = Rotation.from_euler("XYZ", source.get("rotation", [0.0, 0.0, 0.0]), degrees=True).as_matrix()
        origin = tuple(float(value) for value in source.get("translation", [0.0, 0.0, 0.0]))
        x_axis = tuple(float(value) for value in matrix @ np.array([1.0, 0.0, 0.0]))
        y_axis = tuple(float(value) for value in matrix @ np.array([0.0, 1.0, 0.0]))
        normal = tuple(float(value) for value in matrix @ np.array([0.0, 0.0, 1.0]))
        panels.append(Panel(panel_name, edges, Placement(origin, x_axis, y_axis, normal, "garmentcode_initial_placement"), confidence=1.0))
        labels[panel_name] = source.get("label")
        for index, source_edge in enumerate(source["edges"]):
            if source_edge.get("label"):
                edge_labels[f"{panel_name}/{panel_name}.edge_{index}"] = source_edge["label"]

    panel_lookup = {panel.id: panel for panel in panels}

    def world_endpoints(panel_id: str, edge_index: int) -> tuple[np.ndarray, np.ndarray]:
        panel = panel_lookup[panel_id]
        edge = panel.edges[edge_index]
        placement = panel.placement
        if placement is None:
            return np.asarray((*edge.points[0], 0.0)), np.asarray((*edge.points[-1], 0.0))
        origin = np.asarray(placement.origin, dtype=float)
        x_axis = np.asarray(placement.x_axis, dtype=float)
        y_axis = np.asarray(placement.y_axis, dtype=float)
        first = origin + edge.points[0][0] * x_axis + edge.points[0][1] * y_axis
        last = origin + edge.points[-1][0] * x_axis + edge.points[-1][1] * y_axis
        return first, last

    stitches = []
    stitch_tags = {}
    stitch_orientation = {}
    for index, pair in enumerate(pattern.get("stitches", [])):
        if len(pair) < 2:
            raise ValueError(f"stitch {index} contains fewer than two sides")
        first, second = pair[:2]
        if len(pair) > 2:
            stitch_tags[f"stitch_{index}"] = pair[2:]
        first_endpoints = world_endpoints(first["panel"], int(first["edge"]))
        second_endpoints = world_endpoints(second["panel"], int(second["edge"]))
        direct_cost = float(np.linalg.norm(first_endpoints[0] - second_endpoints[0]) + np.linalg.norm(first_endpoints[1] - second_endpoints[1]))
        reversed_cost = float(np.linalg.norm(first_endpoints[0] - second_endpoints[1]) + np.linalg.norm(first_endpoints[1] - second_endpoints[0]))
        reverse_second = reversed_cost < direct_cost
        stitch_orientation[f"stitch_{index}"] = {
            "second_side_reversed": reverse_second,
            "direct_endpoint_cost_cm": direct_cost,
            "reversed_endpoint_cost_cm": reversed_cost,
        }
        stitches.append(
            Stitch(
                id=f"stitch_{index}",
                side_a=StitchSide(first["panel"], f"{first['panel']}.edge_{int(first['edge'])}"),
                side_b=StitchSide(second["panel"], f"{second['panel']}.edge_{int(second['edge'])}", reversed=reverse_second),
                confidence=1.0,
            )
        )
    return PatternDocument(
        pattern_id=anchor_id or path.stem.replace("_specification", ""),
        generator="GarmentCode retrieved anchor",
        panels=tuple(panels),
        stitches=tuple(stitches),
        provenance={
            "source_artifact_sha256": _sha256(path),
            "source_format": "garmentcode_specification_json",
            "source_license": source_license,
        },
        annotations={
            "topology": "retrieved_structured_anchor",
            "template_retrieval": True,
            # GarmentCode translations are already body-relative initial placements.
            # Recentring them destroys the waist/collar height before cloth simulation.
            "preserve_absolute_placement": True,
            "panel_mesh_spacing_cm": float(panel_mesh_spacing_cm),
            "panel_labels": labels,
            "edge_labels": edge_labels,
            "source_stitch_tags": stitch_tags,
            "stitch_orientation": stitch_orientation,
            "pin_semantics": ["lower_interface", "collar"],
            "refinement_status": "unrefined_anchor",
        },
    )
