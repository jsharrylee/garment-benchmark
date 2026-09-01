"""Run inside Blender and estimate reproducible body measurements from a mesh."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--blend", type=Path, required=True)
    parser.add_argument("--object", default="Human")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(values)


def world_vertices(obj: bpy.types.Object) -> list[Vector]:
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        return [evaluated.matrix_world @ vertex.co for vertex in mesh.vertices]
    finally:
        evaluated.to_mesh_clear()


def slice_measure(vertices: list[Vector], level: float, height: float, center_x: float) -> tuple[float, float, float]:
    band = max(0.006, height * 0.008)
    torso_limit = height * 0.15
    points = [point for point in vertices if abs(point.z - level) <= band and abs(point.x - center_x) <= torso_limit]
    if len(points) < 8:
        points = sorted(vertices, key=lambda point: abs(point.z - level))[: max(8, len(vertices) // 250)]
    def robust_span(values: list[float]) -> float:
        ordered = sorted(values)
        lower = ordered[int(0.02 * (len(ordered) - 1))]
        upper = ordered[int(0.98 * (len(ordered) - 1))]
        return upper - lower

    width = robust_span([point.x for point in points])
    depth = robust_span([point.y for point in points])
    a, b = max(width, 1e-4) * 0.5, max(depth, 1e-4) * 0.5
    circumference = math.pi * (3.0 * (a + b) - math.sqrt(max(0.0, (3.0 * a + b) * (a + 3.0 * b))))
    return width * 100.0, depth * 100.0, circumference * 100.0


def main() -> None:
    args = parse_args()
    bpy.ops.wm.open_mainfile(filepath=str(args.blend))
    obj = bpy.data.objects.get(args.object)
    if obj is None or obj.type != "MESH":
        raise RuntimeError(f"mesh object {args.object!r} not found")
    vertices = world_vertices(obj)
    minimum_z = min(point.z for point in vertices)
    maximum_z = max(point.z for point in vertices)
    minimum_x = min(point.x for point in vertices)
    maximum_x = max(point.x for point in vertices)
    height = maximum_z - minimum_z
    center_x = (minimum_x + maximum_x) * 0.5
    # MakeHuman's standing mesh has the hip, natural waist and bust close to
    # 54%, 61% and 70% of stature.  Lower fractions cut through separated legs.
    levels = {"hips": 0.54, "waist": 0.61, "bust": 0.70}
    result = {
        "schema_version": "1.0",
        "method": "mesh_slice_ellipse_estimate",
        "source_object": args.object,
        "body": {"height": height * 100.0, "leg_length": height * 0.49 * 100.0},
        "bounds_m": {"minimum": [minimum_x, min(point.y for point in vertices), minimum_z], "maximum": [maximum_x, max(point.y for point in vertices), maximum_z]},
        "slice_levels": {},
    }
    for name, fraction in levels.items():
        z = minimum_z + fraction * height
        width, depth, circumference = slice_measure(vertices, z, height, center_x)
        result["body"][name] = circumference
        result["slice_levels"][name] = {"z_m": z, "width_cm": width, "depth_cm": depth, "circumference_cm": circumference}
    result["body"]["hips"] = max(result["body"]["hips"], result["body"]["waist"] * 1.10)
    result["body"]["bust"] = max(result["body"]["bust"], result["body"]["waist"] * 1.15)
    result["anthropometric_constraints"] = {
        "hips_minimum_to_waist": 1.10,
        "bust_minimum_to_waist": 1.15,
        "reason": "suppress arm-pose and sparse-slice underestimation",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
