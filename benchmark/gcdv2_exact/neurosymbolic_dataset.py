from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

from benchmark.gcdv2_exact.geometry import sample_curve


SCHEMA_VERSION = "gcdv2-neurosymbolic-panel-1.0"
VISUAL_SIZE = 128
CONTOUR_SAMPLES = 256
VISIBLE_CORNER_THRESHOLD_DEG = 12.0


def _wrap_angle(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def _curve(edge: Mapping[str, Any]) -> dict[str, Any]:
    curve_type = str(edge["curve_type"])
    curve: dict[str, Any] = {
        "type": curve_type,
        "controls_cm": [list(value) for value in edge["centered_controls_cm"]],
    }
    if curve_type == "circular_arc":
        parameters = edge["curve_parameters"]
        curve["arc"] = {
            "radius_cm": float(parameters["radius_cm"]),
            "large_arc": bool(parameters["large_arc"]),
            "sweep_y_up": bool(parameters["sweep_y_up"]),
            "right": bool(parameters["sweep_y_up"]),
        }
    return curve


def _uniform_closed_contour(target: Mapping[str, Any], count: int = CONTOUR_SAMPLES) -> np.ndarray:
    vertices = [value["centered_xy_cm"] for value in target["geometry"]["vertices"]]
    dense: list[tuple[float, float]] = []
    for edge in target["geometry"]["edges"]:
        start = vertices[int(edge["start_vertex_index"])]
        end = vertices[int(edge["end_vertex_index"])]
        samples = max(24, min(256, int(math.ceil(float(edge["length_cm"]) * 4))))
        points = sample_curve(start, end, _curve(edge), samples=samples)
        dense.extend((float(x), float(y)) for x, y in points[:-1])
    dense.append(dense[0])
    values = np.asarray(dense, np.float64)
    lengths = np.linalg.norm(np.diff(values, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    positions = np.linspace(0.0, cumulative[-1], count, endpoint=False)
    result = np.empty((count, 2), np.float32)
    for axis in range(2):
        result[:, axis] = np.interp(positions, cumulative, values[:, axis])
    contract = target["input_contract"]["normalized_panel_image"]
    scale = float(contract["pixels_per_cm"])
    canvas = float(target["input_contract"]["canvas_size_px"][0])
    uv = np.empty_like(result)
    uv[:, 0] = (canvas * 0.5 + result[:, 0] * scale) / canvas
    uv[:, 1] = (canvas * 0.5 - result[:, 1] * scale) / canvas
    return uv


def junction_records(target: Mapping[str, Any]) -> list[dict[str, Any]]:
    vertices = target["geometry"]["vertices"]
    edges = target["geometry"]["edges"]
    result = []
    for index, vertex in enumerate(vertices):
        incoming = edges[(index - 1) % len(edges)]
        outgoing = edges[index]
        tangent_change = abs(
            _wrap_angle(
                float(outgoing["start_tangent_deg_y_up"])
                - float(incoming["end_tangent_deg_y_up"])
            )
        )
        visible = tangent_change >= VISIBLE_CORNER_THRESHOLD_DEG
        result.append(
            {
                "point_id": f"p{index}",
                "source_vertex_index": int(vertex["source_vertex_index"]),
                "xy_cm": list(vertex["centered_xy_cm"]),
                "uv": [float(value) / 1024.0 for value in vertex["image_xy_px"]],
                "incoming_edge_id": f"e{(index - 1) % len(edges)}",
                "outgoing_edge_id": f"e{index}",
                "tangent_change_deg": tangent_change,
                "observability": "VISIBLE_CORNER" if visible else "LATENT_SMOOTH_SOURCE_SUBDIVISION",
                "visual_supervision_eligible": visible,
            }
        )
    return result


def build_visual_truth(target: Mapping[str, Any], panel_image_path: Path) -> dict[str, np.ndarray]:
    with Image.open(panel_image_path) as image:
        mask = np.asarray(
            image.convert("L").resize((VISUAL_SIZE, VISUAL_SIZE), Image.Resampling.LANCZOS),
            dtype=np.uint8,
        )
    binary = mask >= 128
    cm_per_source_pixel = float(
        target["input_contract"]["normalized_panel_image"]["cm_per_pixel"]
    )
    cm_per_visual_pixel = cm_per_source_pixel * (1024.0 / VISUAL_SIZE)
    signed_distance_cm = (
        distance_transform_edt(binary) - distance_transform_edt(~binary)
    ) * cm_per_visual_pixel
    heatmap = np.zeros((VISUAL_SIZE, VISUAL_SIZE), np.float32)
    yy, xx = np.mgrid[:VISUAL_SIZE, :VISUAL_SIZE]
    visible_uv = []
    for junction in junction_records(target):
        if not junction["visual_supervision_eligible"]:
            continue
        u, v = junction["uv"]
        visible_uv.append((u, v))
        x, y = u * VISUAL_SIZE, v * VISUAL_SIZE
        heatmap = np.maximum(heatmap, np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 2.0**2)))
    return {
        "mask_u8": mask,
        "sdf_cm_f16": signed_distance_cm.astype(np.float16),
        "dense_contour_uv_f32": _uniform_closed_contour(target),
        "visible_junction_heatmap_f16": heatmap.astype(np.float16),
        "visible_junction_uv_f32": np.asarray(visible_uv, np.float32).reshape(-1, 2),
        "cm_per_visual_pixel_f32": np.asarray([cm_per_visual_pixel], np.float32),
    }


def formal_graph(target: Mapping[str, Any]) -> dict[str, Any]:
    junctions = junction_records(target)
    edges = []
    operation_tokens = []
    for junction in junctions:
        operation_tokens.append(
            {
                "op": "CREATE_POINT",
                "args": [junction["point_id"]],
                "attributes": {
                    "xy_cm": junction["xy_cm"],
                    "observability": junction["observability"],
                },
            }
        )
    for index, source in enumerate(target["geometry"]["edges"]):
        edge = {
            "edge_id": f"e{index}",
            "source_edge_index": int(source["source_edge_index"]),
            "source_edge_id": str(source["source_edge_id"]),
            "start_point_id": f"p{index}",
            "end_point_id": f"p{(index + 1) % len(junctions)}",
            "primitive": str(source["curve_type"]),
            "parameters": source["curve_parameters"],
            "centered_controls_cm": source["centered_controls_cm"],
            "length_cm": float(source["length_cm"]),
            "chord_direction_deg_y_up": float(source["chord_direction_deg_y_up"]),
            "start_tangent_deg_y_up": float(source["start_tangent_deg_y_up"]),
            "end_tangent_deg_y_up": float(source["end_tangent_deg_y_up"]),
        }
        edges.append(edge)
        operation_tokens.append(
            {
                "op": "CREATE_CURVE",
                "args": [edge["edge_id"], edge["primitive"], edge["start_point_id"], edge["end_point_id"]],
                "attributes": {
                    "parameters": edge["parameters"],
                    "length_cm": edge["length_cm"],
                },
            }
        )
    relations = []
    for index in range(len(edges)):
        relations.append(
            {
                "predicate": "NEXT",
                "arguments": [f"e{index}", f"e{(index + 1) % len(edges)}"],
            }
        )
        relations.append(
            {
                "predicate": "SHARED_ENDPOINT",
                "arguments": [f"e{index}", f"e{(index + 1) % len(edges)}", f"p{(index + 1) % len(edges)}"],
            }
        )
    operation_tokens.append(
        {"op": "CLOSE_CYCLE", "args": ["panel_boundary"], "attributes": {"edge_count": len(edges)}}
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "panel_uid": target["panel_uid"],
        "garment_category": target["garment_category"],
        "source_panel_id": target["source"]["panel_id"],
        "weak_role": target["role_labels"],
        "points": junctions,
        "curves": edges,
        "relations": relations,
        "global_constraints": [
            {"predicate": "SINGLE_CONNECTED_COMPONENT", "arguments": ["panel_boundary"]},
            {"predicate": "CLOSED_CYCLE", "arguments": ["panel_boundary"]},
            {"predicate": "DEGREE_EQUALS", "arguments": ["every_boundary_point", 2]},
        ],
        "serialization_equivalence": {
            "cyclic_rotations_are_same_shape": True,
            "reversal_is_same_undirected_shape": True,
            "directed_source_cycle_retained_for_exact_evaluation": True,
            "canonical_start_is_not_a_semantic_landmark": True,
        },
        "operation_tokens": operation_tokens,
        "supervision_partition": {
            "visual": [
                "mask",
                "signed_distance_cm",
                "dense_contour_uv",
                "visible_corner_heatmap",
            ],
            "source_formal": [
                "latent_smooth_subdivision_points",
                "exact_curve_primitive_and_parameters",
                "ordered_incidence_graph",
                "exact_lengths_and_tangents",
            ],
        },
    }


def stitch_constraints(
    sample_label: Mapping[str, Any], panel_targets: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    mappings = {
        panel_id: {
            int(edge["source_edge_index"]): edge
            for edge in target["geometry"]["edges"]
        }
        for panel_id, target in panel_targets.items()
    }
    result = []
    for stitch_index, pair in enumerate(sample_label.get("stitches", [])):
        source_sides = [value for value in pair if isinstance(value, Mapping)]
        source_annotations = [value for value in pair if not isinstance(value, Mapping)]
        if len(source_sides) != 2:
            raise ValueError(f"stitch {stitch_index} does not contain exactly two edge references")
        sides = []
        for side in source_sides:
            panel_id = str(side["panel"])
            source_edge_index = int(side["edge"])
            edge = mappings[panel_id][source_edge_index]
            sides.append(
                {
                    "panel_uid": f"{sample_label['sample_id']}:{panel_id}",
                    "edge_id": f"e{int(edge['edge_index'])}",
                    "source_edge_index": source_edge_index,
                    "length_cm": float(edge["length_cm"]),
                }
            )
        result.append(
            {
                "constraint_id": f"stitch_{stitch_index}",
                "predicate": "SEWN_TO",
                "sides": sides,
                "absolute_length_difference_cm": abs(sides[0]["length_cm"] - sides[1]["length_cm"]),
                "source_annotations": source_annotations,
            }
        )
    return result


__all__ = [
    "CONTOUR_SAMPLES",
    "SCHEMA_VERSION",
    "VISIBLE_CORNER_THRESHOLD_DEG",
    "VISUAL_SIZE",
    "build_visual_truth",
    "formal_graph",
    "junction_records",
    "stitch_constraints",
]
