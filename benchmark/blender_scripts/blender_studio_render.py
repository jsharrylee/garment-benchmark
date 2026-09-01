"""Run inside Blender: turn an official production character into benchmark views."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector


CAMERAS = (
    ("CAM000", "front", Vector((0.0, -1.0, 0.0))),
    ("CAM001", "back", Vector((0.0, 1.0, 0.0))),
    ("CAM002", "left", Vector((-1.0, 0.0, 0.0))),
    ("CAM003", "right", Vector((1.0, 0.0, 0.0))),
)


def parse_args() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(values)


def recursive_objects(collection_name: str) -> set[bpy.types.Object]:
    root = bpy.data.collections.get(collection_name)
    if root is None:
        raise RuntimeError(f"collection not found: {collection_name}")
    found = set(root.objects)
    for child in root.children_recursive:
        found.update(child.objects)
    return found


def select_render_objects(sample: dict) -> tuple[list[bpy.types.Object], list[bpy.types.Object], list[bpy.types.Object]]:
    prefixes = tuple(sample.get("excluded_name_prefixes", ()))
    fragments = tuple(sample.get("excluded_name_fragments", ()))

    def accepted(obj: bpy.types.Object) -> bool:
        return (
            obj.type == "MESH"
            and not obj.hide_render
            and not obj.name.startswith(prefixes)
            and not any(token in obj.name for token in fragments)
        )

    rgb_candidates: set[bpy.types.Object] = set()
    for collection_name in sample["rgb_collection_roots"]:
        rgb_candidates.update(recursive_objects(collection_name))
    garments: set[bpy.types.Object] = set()
    for collection_name in sample["garment_collections"]:
        garments.update(recursive_objects(collection_name))
    rgb_candidates.update(garments)
    rgb = sorted((obj for obj in rgb_candidates if accepted(obj)), key=lambda obj: obj.name)
    garment_meshes = sorted((obj for obj in garments if accepted(obj)), key=lambda obj: obj.name)
    bodies = [obj for obj in rgb if obj not in set(garment_meshes)]
    if not rgb or not garment_meshes or not bodies:
        raise RuntimeError(f"invalid separation: rgb={len(rgb)}, garments={len(garment_meshes)}, body={len(bodies)}")
    return rgb, bodies, garment_meshes


def world_bounds(objects: list[bpy.types.Object]) -> tuple[Vector, Vector]:
    points = [obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box]
    return (
        Vector(tuple(min(point[i] for point in points) for i in range(3))),
        Vector(tuple(max(point[i] for point in points) for i in range(3))),
    )


def look_at(obj: bpy.types.Object, target: Vector) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def configure_scene(rgb_objects: list[bpy.types.Object]) -> tuple[bpy.types.Object, Vector, float, float, float]:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "MATERIAL"
    scene.display.shading.show_shadows = True
    scene.display.shading.show_cavity = True
    scene.render.resolution_x = 518
    scene.render.resolution_y = 518
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = True

    for obj in bpy.data.objects:
        if hasattr(obj, "hide_render"):
            obj.hide_render = obj not in set(rgb_objects)
        if obj.type == "MESH":
            for modifier in obj.modifiers:
                modifier.show_render = False

    minimum, maximum = world_bounds(rgb_objects)
    center = (minimum + maximum) * 0.5
    width, depth, height = maximum.x - minimum.x, maximum.y - minimum.y, maximum.z - minimum.z
    distance = max(width, depth, height) * 3.0
    scale_front = max(height, width) * 1.10
    scale_side = max(height, depth) * 1.10

    camera_data = bpy.data.cameras.new("benchmark_ortho_camera")
    camera_data.type = "ORTHO"
    camera = bpy.data.objects.new("benchmark_ortho_camera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.hide_render = False

    for index, location in enumerate(((4.0, -4.0, maximum.z + 3.0), (-4.0, -2.0, center.z + 1.0))):
        light_data = bpy.data.lights.new(f"benchmark_area_{index}", "AREA")
        light_data.energy = 1000.0 if index == 0 else 650.0
        light_data.shape = "DISK"
        light_data.size = 5.0
        light = bpy.data.objects.new(light_data.name, light_data)
        scene.collection.objects.link(light)
        light.location = location
        light.hide_render = False
        look_at(light, center)
    return camera, center, max(distance, 3.0), max(scale_front, 0.1), max(scale_side, 0.1)


def make_mask_material() -> bpy.types.Material:
    material = bpy.data.materials.new("benchmark_mask_white")
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    principled.inputs["Roughness"].default_value = 1.0
    return material


def render_views(output: Path, rgb_objects: list[bpy.types.Object], garments: list[bpy.types.Object]) -> dict:
    rgb_dir, mask_dir = output / "rgb", output / "masks"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    camera, center, distance, scale_front, scale_side = configure_scene(rgb_objects)
    scene = bpy.context.scene
    view_layer = scene.view_layers[0]
    mask_material = make_mask_material()
    rgb_set, garment_set = set(rgb_objects), set(garments)
    metadata = {}

    for camera_id, label, direction in CAMERAS:
        camera.location = center + direction * distance
        camera.data.ortho_scale = scale_front if label in {"front", "back"} else scale_side
        look_at(camera, center)
        for obj in bpy.data.objects:
            if hasattr(obj, "hide_render"):
                obj.hide_render = obj not in rgb_set and obj not in {camera}
        view_layer.material_override = None
        scene.render.filepath = str((rgb_dir / f"{camera_id}.png").resolve())
        bpy.ops.render.render(write_still=True)

        for obj in bpy.data.objects:
            if hasattr(obj, "hide_render"):
                obj.hide_render = obj not in garment_set and obj not in {camera}
        view_layer.material_override = mask_material
        scene.render.filepath = str((mask_dir / f"{camera_id}.png").resolve())
        bpy.ops.render.render(write_still=True)
        view_layer.material_override = None
        metadata[camera_id] = {
            "view_label": label,
            "projection": "orthographic",
            "position": [float(value) for value in camera.location],
            "target": [float(value) for value in center],
            "ortho_scale": float(camera.data.ortho_scale),
            "world_to_camera": [float(value) for row in camera.matrix_world.inverted() for value in row],
        }
    return metadata


def mesh_record(obj: bpy.types.Object) -> dict:
    return {"name": obj.name, "vertices": len(obj.data.vertices), "polygons": len(obj.data.polygons)}


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    sample = next(item for item in config["samples"] if item["sample_id"] == args.sample)
    args.output.mkdir(parents=True, exist_ok=True)
    rgb, bodies, garments = select_render_objects(sample)
    cameras = render_views(args.output, rgb, garments)
    (args.output / "camera_metadata.json").write_text(json.dumps(cameras, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "sample_id": sample["sample_id"],
        "source": "Blender Studio",
        "source_asset": "Einar",
        "source_page": config["source"]["official_page"],
        "license": config["source"]["license"],
        "credit": config["source"]["credit"],
        "garment_collections": sample["garment_collections"],
        "garment_objects": [obj.name for obj in garments],
        "body_objects": [obj.name for obj in bodies],
        "object_separation": True,
        "mesh_records": [mesh_record(obj) for obj in rgb],
        "pattern_ground_truth_available": False,
        "use_role": "multiview_visual_input_only",
    }
    (args.output / "character_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("BLENDER_STUDIO_SAMPLE_COMPLETE", sample["sample_id"], len(rgb), len(garments), len(bodies))


if __name__ == "__main__":
    main()
