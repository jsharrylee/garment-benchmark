"""Run inside Blender: generate deterministic MPFB characters and render four views."""

from __future__ import annotations

import argparse
import json
import math
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
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(values)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in (bpy.data.meshes, bpy.data.curves, bpy.data.materials, bpy.data.cameras, bpy.data.lights):
        for item in list(block):
            if item.users == 0:
                block.remove(item)


def configure_mpfb(sample: dict) -> None:
    from bl_ext.user_default.mpfb.ui.new_human.randomize.randomizeproperties import (
        RANDOMIZE_PROPERTIES,
        scene_to_spec,
        spec_to_scene,
    )

    scene = bpy.context.scene
    spec = scene_to_spec(scene)
    phenotype = spec["phenotype"]
    phenotype["discrete_gender"] = True
    phenotype["discrete_age"] = True
    phenotype["attributes"]["gender"]["allowed"] = [sample["gender"]]
    phenotype["attributes"]["age"]["allowed"] = ["young"]
    phenotype["attributes"]["height"]["deviation"] = 0.25
    phenotype["attributes"]["weight"]["deviation"] = 0.35
    phenotype["attributes"]["muscle"]["deviation"] = 0.25

    assets = spec["assets"]
    assets["skin"]["enabled"] = True
    assets["skin"]["match_gender"] = True
    assets["hair"]["enabled"] = False
    assets["eyes"]["mode"] = "DONOTADD"
    for part in ("eyebrows", "eyelashes", "teeth", "tongue"):
        assets[part]["enabled"] = False
    for slot in assets["clothes"].values():
        slot["enabled"] = False
    full_body = assets["clothes"]["full_body"]
    full_body.update(
        {
            "enabled": True,
            "chance": 100,
            "pack": "makehuman_system_assets",
            "include_any": sample["garment"],
            "include_female": "",
            "include_male": "",
            "exclude": "",
        }
    )
    spec["creation"].update(
        {
            "rig": "NONE",
            "detailed_helpers": False,
            "extra_vertex_groups": False,
            "mask_helpers": True,
            "add_subdiv_modifier": False,
            "subdiv_render_levels": 0,
        }
    )
    spec["details"]["enabled"] = True
    spec_to_scene(spec, scene)
    RANDOMIZE_PROPERTIES.set_value("seed", int(sample["seed"]), entity_reference=scene)
    RANDOMIZE_PROPERTIES.set_value("new_random_seed", False, entity_reference=scene)


def create_character(sample: dict) -> tuple[list[bpy.types.Object], list[bpy.types.Object], list[bpy.types.Object]]:
    before = set(bpy.data.objects)
    configure_mpfb(sample)
    result = bpy.ops.mpfb.create_random_human()
    if "FINISHED" not in result:
        raise RuntimeError(f"MPFB generation failed: {result}")
    created = [obj for obj in bpy.data.objects if obj not in before]
    meshes = [obj for obj in created if obj.type == "MESH"]
    garment_token = sample["garment"].lower()
    garments = [obj for obj in meshes if garment_token in obj.name.lower()]
    if not garments:
        custom = {obj.name: {key: str(obj[key]) for key in obj.keys()} for obj in meshes}
        raise RuntimeError(f"MPFB garment object {sample['garment']} not found: {custom}")
    bodies = [obj for obj in meshes if obj not in garments]
    if not bodies:
        raise RuntimeError("MPFB produced no separate body mesh")
    return created, bodies, garments


def world_bounds(objects: list[bpy.types.Object]) -> tuple[Vector, Vector]:
    points = [obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box if obj.type == "MESH"]
    if not points:
        raise RuntimeError("character has no renderable bounds")
    return Vector(tuple(min(point[i] for point in points) for i in range(3))), Vector(tuple(max(point[i] for point in points) for i in range(3)))


def look_at(camera: bpy.types.Object, target: Vector) -> None:
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()


def make_mask_material() -> bpy.types.Material:
    material = bpy.data.materials.new("benchmark_mask_white")
    material.diffuse_color = (1.0, 1.0, 1.0, 1.0)
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    principled.inputs["Roughness"].default_value = 1.0
    return material


def setup_render(meshes: list[bpy.types.Object]) -> tuple[bpy.types.Object, Vector, float, float, float]:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = 518
    scene.render.resolution_y = 518
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = True
    scene.world.color = (0.055, 0.055, 0.055)

    minimum, maximum = world_bounds(meshes)
    center = (minimum + maximum) * 0.5
    width = maximum.x - minimum.x
    depth = maximum.y - minimum.y
    height = maximum.z - minimum.z
    scale_front = max(height, width) * 1.10
    scale_side = max(height, depth) * 1.10
    distance = max(width, depth, height) * 3.0

    camera_data = bpy.data.cameras.new("benchmark_ortho_camera")
    camera_data.type = "ORTHO"
    camera = bpy.data.objects.new("benchmark_ortho_camera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera

    for index, location in enumerate(((4.0, -4.0, maximum.z + 3.0), (-4.0, -2.0, center.z + 1.0))):
        data = bpy.data.lights.new(f"benchmark_area_{index}", "AREA")
        data.energy = 900.0 if index == 0 else 500.0
        data.shape = "DISK"
        data.size = 5.0
        light = bpy.data.objects.new(data.name, data)
        scene.collection.objects.link(light)
        light.location = location
        look_at(light, center)
    return camera, center, max(distance, 3.0), max(scale_front, 0.1), max(scale_side, 0.1)


def render_views(sample_dir: Path, meshes: list[bpy.types.Object], garments: list[bpy.types.Object]) -> dict:
    rgb_dir, mask_dir = sample_dir / "rgb", sample_dir / "masks"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    camera, center, distance, scale_front, scale_side = setup_render(meshes)
    scene = bpy.context.scene
    view_layer = scene.view_layers[0]
    mask_material = make_mask_material()
    metadata = {}
    garment_set = set(garments)

    for camera_id, label, direction in CAMERAS:
        camera.location = center + direction * distance
        camera.data.ortho_scale = scale_front if label in {"front", "back"} else scale_side
        look_at(camera, center)
        for obj in meshes:
            obj.hide_render = False
        view_layer.material_override = None
        scene.render.filepath = str((rgb_dir / f"{camera_id}.png").resolve())
        bpy.ops.render.render(write_still=True)

        for obj in meshes:
            obj.hide_render = obj not in garment_set
        view_layer.material_override = mask_material
        scene.render.filepath = str((mask_dir / f"{camera_id}.png").resolve())
        bpy.ops.render.render(write_still=True)
        view_layer.material_override = None
        for obj in meshes:
            obj.hide_render = False

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
    return {
        "name": obj.name,
        "vertices": len(obj.data.vertices),
        "polygons": len(obj.data.polygons),
        "parent": obj.parent.name if obj.parent else None,
    }


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    for sample in config["samples"]:
        clear_scene()
        created, bodies, garments = create_character(sample)
        meshes = [obj for obj in created if obj.type == "MESH"]
        sample_dir = args.output / sample["sample_id"]
        camera_metadata = render_views(sample_dir, meshes, garments)
        (sample_dir / "camera_metadata.json").write_text(json.dumps(camera_metadata, indent=2) + "\n", encoding="utf-8")
        manifest = {
            "sample_id": sample["sample_id"],
            "source": "MPFB",
            "source_asset": sample["garment"],
            "seed": sample["seed"],
            "gender_constraint": sample["gender"],
            "garment_objects": [obj.name for obj in garments],
            "body_objects": [obj.name for obj in bodies],
            "object_separation": True,
            "mesh_records": [mesh_record(obj) for obj in meshes],
            "pattern_ground_truth_available": False,
            "use_role": "multiview_visual_input_only",
        }
        (sample_dir / "character_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        bpy.ops.wm.save_as_mainfile(filepath=str((sample_dir / "source.blend").resolve()), check_existing=False)
        print("MPFB_SAMPLE_COMPLETE", sample["sample_id"], [obj.name for obj in garments])


if __name__ == "__main__":
    main()
