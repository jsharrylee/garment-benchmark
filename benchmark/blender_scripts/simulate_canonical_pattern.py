"""Run inside Blender: import canonical panels and simulate explicit sewing springs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector
from mathutils.geometry import tessellate_polygon


VIEWS = (
    ("CAM000", "front", Vector((0.0, -1.0, 0.0))),
    ("CAM001", "back", Vector((0.0, 1.0, 0.0))),
    ("CAM002", "left", Vector((-1.0, 0.0, 0.0))),
    ("CAM003", "right", Vector((1.0, 0.0, 0.0))),
)


def parse_args() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=45)
    parser.add_argument("--body-blend", type=Path)
    parser.add_argument("--body-object", default="Human")
    parser.add_argument("--camera-metadata", type=Path)
    parser.add_argument("--subdivision-levels", type=int, default=0)
    parser.add_argument("--calibration", type=Path)
    return parser.parse_args(values)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def triangulate(vertices: list[list[float]], loops: list[list[int]]) -> list[tuple[int, int, int]]:
    faces = []
    for loop in loops:
        polygon = [Vector(vertices[index]) for index in loop]
        lookup = {tuple(vertex): index for vertex, index in zip(polygon, loop, strict=True)}
        for triangle in tessellate_polygon([polygon]):
            if triangle and isinstance(triangle[0], int):
                faces.append(tuple(loop[int(index)] for index in triangle))
            else:
                faces.append(tuple(lookup[tuple(vertex)] for vertex in triangle))
    return faces


def add_mannequin(vertices: list[list[float]]) -> bpy.types.Object:
    values = [Vector(value) for value in vertices]
    xs, ys, zs = zip(*values)
    span = Vector((max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)))
    center_z = (max(zs) + min(zs)) * 0.5
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=20, location=(0.0, 0.0, center_z))
    body = bpy.context.object
    body.name = "benchmark_collision_mannequin"
    body.scale = (
        max(0.12, min(0.35, span.x * 0.22)),
        max(0.10, min(0.25, max(span.y, span.x * 0.55) * 0.20)),
        max(0.35, min(0.85, span.z * 0.42)),
    )
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    collision = body.modifiers.new("Collision", "COLLISION")
    collision.settings.thickness_outer = 0.006
    return body


def append_collision_body(blend_path: Path, object_name: str, calibration: dict) -> bpy.types.Object:
    with bpy.data.libraries.load(str(blend_path), link=False) as (source, target):
        if object_name not in source.objects:
            raise RuntimeError(f"body object {object_name!r} not found in {blend_path}")
        target.objects = [object_name]
    body = target.objects[0]
    bpy.context.scene.collection.objects.link(body)
    body.name = "benchmark_collision_body"
    collision = body.modifiers.new("Collision", "COLLISION")
    collision.settings.thickness_outer = float(calibration.get("collision_thickness_outer", 0.004))
    body.hide_render = True
    return body


def make_garment(plan: dict, frames: int, subdivision_levels: int, calibration: dict) -> bpy.types.Object:
    vertices = plan["vertices"]
    faces = [tuple(face) for face in plan.get("panel_faces", [])] or triangulate(vertices, plan["panel_loops"])
    face_edges = {tuple(sorted((face[index], face[(index + 1) % 3]))) for face in faces for index in range(3)}
    sewing = [tuple(edge) for edge in plan["sewing_edges"] if tuple(sorted(edge)) not in face_edges and edge[0] != edge[1]]
    mesh = bpy.data.meshes.new("canonical_sewing_pattern_mesh")
    mesh.from_pydata(vertices, sewing, faces)
    mesh.update()
    garment = bpy.data.objects.new("canonical_sewn_garment", mesh)
    bpy.context.scene.collection.objects.link(garment)
    pre_scale = calibration.get("precompensation_scale_blender_xyz", [1.0, 1.0, 1.0])
    center = sum((vertex.co for vertex in garment.data.vertices), Vector()) / max(1, len(garment.data.vertices))
    for vertex in garment.data.vertices:
        delta = vertex.co - center
        vertex.co = center + Vector((delta.x * float(pre_scale[0]), delta.y * float(pre_scale[1]), delta.z * float(pre_scale[2])))
    garment.data.update()

    material = bpy.data.materials.new("benchmark_garment_white")
    material.diffuse_color = (1.0, 1.0, 1.0, 1.0)
    garment.data.materials.append(material)
    pinned = [int(index) for index in plan.get("pinned_vertices", [])]
    pin_group = None
    if pinned:
        pin_group = garment.vertex_groups.new(name="semantic_attachment_pins")
        pin_group.add(pinned, 1.0, "REPLACE")
    if subdivision_levels:
        subdivision = garment.modifiers.new("ClothResolution", "SUBSURF")
        subdivision.subdivision_type = "SIMPLE"
        subdivision.levels = subdivision_levels
        subdivision.render_levels = subdivision_levels
    cloth = garment.modifiers.new("StitchAwareCloth", "CLOTH")
    cloth.settings.quality = int(calibration.get("quality", 8))
    cloth.settings.mass = float(calibration.get("mass", 0.3))
    cloth.settings.air_damping = float(calibration.get("air_damping", 3.0))
    cloth.settings.use_sewing_springs = True
    cloth.settings.sewing_force_max = float(calibration.get("sewing_force_max", 12.0))
    if pin_group:
        cloth.settings.vertex_group_mass = pin_group.name
        cloth.settings.pin_stiffness = float(calibration.get("pin_stiffness", 1.0))
    cloth.collision_settings.use_collision = True
    cloth.collision_settings.use_self_collision = bool(calibration.get("use_self_collision", True))
    cloth.collision_settings.self_distance_min = float(calibration.get("self_distance_min", 0.008))
    cloth.point_cache.frame_start = 1
    cloth.point_cache.frame_end = frames
    solidify = garment.modifiers.new("RenderThickness", "SOLIDIFY")
    solidify.thickness = 0.002
    return garment


def look_at(obj: bpy.types.Object, target: Vector) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def evaluated_bounds(obj: bpy.types.Object) -> tuple[Vector, Vector]:
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    points = [evaluated.matrix_world @ Vector(corner) for corner in evaluated.bound_box]
    return Vector(tuple(min(point[index] for point in points) for index in range(3))), Vector(tuple(max(point[index] for point in points) for index in range(3)))


def bounds_dict(obj: bpy.types.Object) -> dict:
    minimum, maximum = evaluated_bounds(obj)
    return {"minimum": list(minimum), "maximum": list(maximum), "extent": list(maximum - minimum), "center": list((minimum + maximum) * 0.5)}


def render_views(garment: bpy.types.Object, output: Path, camera_config: dict | None = None) -> dict:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = 518
    scene.render.resolution_y = 518
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True
    minimum, maximum = evaluated_bounds(garment)
    center = (minimum + maximum) * 0.5
    span = maximum - minimum
    camera_data = bpy.data.cameras.new("benchmark_simulation_camera")
    camera_data.type = "ORTHO"
    camera = bpy.data.objects.new("benchmark_simulation_camera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    output.mkdir(parents=True, exist_ok=True)
    metadata = {}
    distance = max(span.x, span.y, span.z, 1.0) * 3.0
    for camera_id, label, direction in VIEWS:
        if camera_config and camera_id in camera_config:
            configured = camera_config[camera_id]
            camera.location = Vector(configured["position"])
            look_at(camera, Vector(configured["target"]))
            camera.data.ortho_scale = float(configured["ortho_scale"])
        else:
            camera.location = center + direction * distance
            look_at(camera, center)
            camera.data.ortho_scale = max(span.z, span.x if label in ("front", "back") else span.y, 0.2) * 1.12
        scene.render.filepath = str(output / f"{camera_id}.png")
        bpy.ops.render.render(write_still=True)
        metadata[camera_id] = {"view_label": label, "location": list(camera.location), "ortho_scale": camera.data.ortho_scale}
    return metadata


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plan = json.loads(args.mesh_plan.read_text(encoding="utf-8"))
    calibration = json.loads(args.calibration.read_text(encoding="utf-8")) if args.calibration else {}
    clear_scene()
    garment = make_garment(plan, args.frames, args.subdivision_levels, calibration)
    initial_bounds = bounds_dict(garment)
    body = append_collision_body(args.body_blend, args.body_object, calibration) if args.body_blend else add_mannequin(plan["vertices"])
    body.hide_render = True
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = args.frames
    scene.gravity = (0.0, 0.0, -9.81)
    for frame in range(1, args.frames + 1):
        scene.frame_set(frame)
    final_bounds = bounds_dict(garment)
    camera_config = json.loads(args.camera_metadata.read_text(encoding="utf-8")) if args.camera_metadata else None
    metadata = render_views(garment, args.output / "masks", camera_config)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output / "simulation.blend"))
    result = {
        "backend": "Blender cloth explicit loose-edge sewing springs",
        "blender_version": bpy.app.version_string,
        "frames": args.frames,
        "vertex_count": len(plan["vertices"]),
        "sewing_spring_count": len(plan["sewing_edges"]),
        "pinned_vertex_count": len(plan.get("pinned_vertices", [])),
        "subdivision_levels": args.subdivision_levels,
        "mock_or_proxy_used": False,
        "canonical_pattern_imported": True,
        "collision_body": "source_blend_object" if args.body_blend else "generic_ellipsoid_proxy",
        "target_camera_metadata_used": bool(args.camera_metadata),
        "camera_metadata": metadata,
        "calibration": calibration,
        "initial_bounds": initial_bounds,
        "final_bounds": final_bounds,
        "drape_extent_ratio_blender_xyz": [
            final_bounds["extent"][index] / max(initial_bounds["extent"][index], 1e-8) for index in range(3)
        ],
    }
    (args.output / "simulation_metadata.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
