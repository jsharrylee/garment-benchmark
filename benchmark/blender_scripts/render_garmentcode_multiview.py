"""Batch-render GarmentCodeData simulated meshes from orthogonal cameras."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector


VIEWS = {
    # GCDv2's source +Z depth becomes Blender +Y below.  Front panels are
    # placed on the +Y side, so the semantic front camera must also sit at
    # +Y and look toward the origin.  The legacy receipt had these two labels
    # reversed even though the rendered CAM files themselves were usable.
    "front": ("CAM001", Vector((0.0, 1.0, 0.0))),
    "back": ("CAM000", Vector((0.0, -1.0, 0.0))),
    "left": ("CAM002", Vector((-1.0, 0.0, 0.0))),
    "right": ("CAM003", Vector((1.0, 0.0, 0.0))),
}


def parse_args() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=384)
    return parser.parse_args(values)


def configure_scene(resolution: int) -> tuple[bpy.types.Object, bpy.types.Material]:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "MATERIAL"
    scene.display.shading.show_shadows = True
    scene.display.shading.show_cavity = True
    camera_data = bpy.data.cameras.new("gcdv2_orthographic_camera")
    camera_data.type = "ORTHO"
    camera = bpy.data.objects.new("gcdv2_orthographic_camera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    material = bpy.data.materials.new("gcdv2_garment_material")
    material.diffuse_color = (0.53, 0.22, 0.72, 1.0)
    return camera, material


def look_at(camera: bpy.types.Object, target: Vector) -> None:
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()


def import_garment(path: Path, material: bpy.types.Material) -> bpy.types.Object:
    before = set(bpy.data.objects)
    bpy.ops.wm.ply_import(filepath=str(path))
    imported = [obj for obj in bpy.data.objects if obj not in before and obj.type == "MESH"]
    if len(imported) != 1:
        raise RuntimeError(f"expected one mesh from {path}, got {len(imported)}")
    garment = imported[0]
    garment.name = f"reference_{path.parent.name}"
    # Dataset: X horizontal, Y vertical, Z depth in centimetres.
    for vertex in garment.data.vertices:
        x, y, z = vertex.co
        vertex.co = (x * 0.01, z * 0.01, y * 0.01)
    garment.data.update()
    garment.data.materials.append(material)
    garment.color = material.diffuse_color
    return garment


def bounds(obj: bpy.types.Object) -> tuple[Vector, Vector]:
    # ``Object.bound_box`` can remain cached at the pre-transform PLY scale.
    points = [obj.matrix_world @ vertex.co for vertex in obj.data.vertices]
    return (
        Vector(tuple(min(point[index] for point in points) for index in range(3))),
        Vector(tuple(max(point[index] for point in points) for index in range(3))),
    )


def render_job(job: dict, camera: bpy.types.Object, material: bpy.types.Material) -> dict:
    garment = import_garment(Path(job["garment_ply"]), material)
    minimum, maximum = bounds(garment)
    center = (minimum + maximum) * 0.5
    span = maximum - minimum
    distance = max(span.x, span.y, span.z, 1.0) * 3.0
    ortho_scale = max(span.z, span.x, span.y, 0.2) * 1.10
    output = Path(job["output"])
    output.mkdir(parents=True, exist_ok=True)
    metadata = {}
    for label in job["views"]:
        camera_id, direction = VIEWS[label]
        destination = output / f"{camera_id}.png"
        if destination.is_file() and job.get("resume", True):
            metadata[camera_id] = {"view_label": label, "status": "EXISTING"}
            continue
        camera.location = center + direction * distance
        look_at(camera, center)
        camera.data.ortho_scale = ortho_scale
        bpy.context.scene.render.filepath = str(destination)
        bpy.ops.render.render(write_still=True)
        metadata[camera_id] = {
            "view_label": label,
            "status": "RENDERED",
            "position": list(camera.location),
            "target": list(center),
            "ortho_scale": ortho_scale,
        }
    receipt = {
        "sample_id": job["sample_id"],
        "source": "GarmentCodeData v2 simulated garment PLY",
        "coordinate_transform": "dataset_xyz_cm_to_blender_xzy_m",
        "garment_only": True,
        "camera_metadata": metadata,
    }
    (output / "rerender_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    bpy.data.objects.remove(garment, do_unlink=True)
    return receipt


def main() -> None:
    args = parse_args()
    jobs = json.loads(args.jobs.read_text(encoding="utf-8"))["jobs"]
    camera, material = configure_scene(args.resolution)
    failures = []
    for index, job in enumerate(jobs, start=1):
        try:
            render_job(job, camera, material)
        except Exception as error:  # keep a full-batch render resumable
            failures.append({"sample_id": job["sample_id"], "error": f"{type(error).__name__}: {error}"})
        if index == 1 or index % 100 == 0 or index == len(jobs):
            print(json.dumps({"rendered_samples": index, "total": len(jobs), "failures": len(failures)}), flush=True)
    result = {"status": "PASS" if not failures else "PARTIAL", "sample_count": len(jobs), "failures": failures}
    Path(json.loads(args.jobs.read_text(encoding="utf-8"))["receipt"]).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
