from __future__ import annotations

import json
import shutil
from pathlib import Path

from benchmark.adapters.reweaver import VIEW_ORDER, sha256


VIEW_LABELS = ("front", "back", "left", "right")


def prepare_inference_bundle(path: Path) -> dict:
    from PIL import Image

    from benchmark.adapters.reweaver import validate_input_directory
    from benchmark.visualization.contact_sheet import create_contact_sheet

    prepared = path / "reweaver" / "render_output" / "rgb"
    garment_particles = path / "garment_particles"
    prepared.mkdir(parents=True, exist_ok=True)
    garment_particles.mkdir(parents=True, exist_ok=True)
    for camera in VIEW_ORDER:
        with Image.open(path / "rgb" / f"{camera}.png") as image:
            rgba = image.convert("RGBA")
            white = Image.new("RGB", rgba.size, "white")
            white.paste(rgba.convert("RGB"), mask=rgba.getchannel("A"))
            white.save(prepared / f"{camera}.png")
        mask_path = path / "masks" / f"{camera}.png"
        with Image.open(mask_path) as image:
            alpha = image.getchannel("A") if "A" in image.getbands() else image.convert("L")
            alpha.point(lambda value: 255 if value else 0).save(mask_path)
    shutil.copy2(prepared / "CAM000.png", garment_particles / "input.png")
    create_contact_sheet(
        [prepared / f"{camera}.png" for camera in VIEW_ORDER],
        path / "reweaver_input_contact_sheet.jpg",
        list(VIEW_LABELS),
        cell=(518, 518),
    )
    create_contact_sheet(
        [path / "masks" / f"{camera}.png" for camera in VIEW_ORDER],
        path / "garment_mask_contact_sheet.jpg",
        list(VIEW_LABELS),
        cell=(518, 518),
    )
    return validate_input_directory(prepared)


def prepare_layer_bundles(path: Path, *, split_ratio: float) -> dict:
    """Create upper/lower model conditions without pretending proxy UVs are patterns."""
    from PIL import Image, ImageChops

    from benchmark.adapters.reweaver import validate_input_directory
    from benchmark.visualization.contact_sheet import create_contact_sheet

    if not 0.2 <= split_ratio <= 0.8:
        raise ValueError("split_ratio must be between 0.2 and 0.8")
    records = {"upper": {}, "lower": {}}
    for layer in records:
        (path / "layers" / layer / "reweaver" / "render_output" / "rgb").mkdir(parents=True, exist_ok=True)
        (path / "layers" / layer / "masks").mkdir(parents=True, exist_ok=True)
        (path / "layers" / layer / "garment_particles" / "views").mkdir(parents=True, exist_ok=True)
        (path / "layers" / layer / "garment_particles").mkdir(parents=True, exist_ok=True)

    for camera in VIEW_ORDER:
        source_path = path / "reweaver" / "render_output" / "rgb" / f"{camera}.png"
        full_mask_path = path / "masks" / f"{camera}.png"
        with Image.open(source_path) as source_image, Image.open(full_mask_path) as mask_image:
            source = source_image.convert("RGB")
            full_mask = mask_image.convert("L").point(lambda value: 255 if value else 0)
        bbox = full_mask.getbbox()
        if bbox is None:
            raise ValueError(f"empty garment mask: {full_mask_path}")
        split_y = int(round(bbox[1] + split_ratio * (bbox[3] - bbox[1])))
        upper_gate = Image.new("L", full_mask.size, 0)
        lower_gate = Image.new("L", full_mask.size, 0)
        upper_gate.paste(255, (0, 0, full_mask.width, split_y))
        lower_gate.paste(255, (0, split_y, full_mask.width, full_mask.height))
        layer_masks = {
            "upper": ImageChops.multiply(full_mask, upper_gate),
            "lower": ImageChops.multiply(full_mask, lower_gate),
        }
        for layer, target_mask in layer_masks.items():
            other_mask = ImageChops.subtract(full_mask, target_mask)
            faded = Image.blend(source, Image.new("RGB", source.size, "white"), 0.82)
            emphasized = source.copy()
            emphasized.paste(faded, mask=other_mask)
            layer_root = path / "layers" / layer
            prepared_path = layer_root / "reweaver" / "render_output" / "rgb" / f"{camera}.png"
            view_path = layer_root / "garment_particles" / "views" / f"{camera}.png"
            emphasized.save(prepared_path)
            emphasized.save(view_path)
            target_mask.save(layer_root / "masks" / f"{camera}.png")
            records[layer][camera] = {"split_y": split_y, "mask_bbox": list(target_mask.getbbox() or ())}

    for layer in records:
        layer_root = path / "layers" / layer
        shutil.copy2(
            layer_root / "garment_particles" / "views" / "CAM000.png",
            layer_root / "garment_particles" / "input.png",
        )
        create_contact_sheet(
            [layer_root / "garment_particles" / "views" / f"{camera}.png" for camera in VIEW_ORDER],
            layer_root / "four_view_condition.jpg",
            list(VIEW_LABELS),
            cell=(518, 518),
        )
        records[layer]["input_validation"] = validate_input_directory(
            layer_root / "reweaver" / "render_output" / "rgb"
        )
    manifest = {
        "method": "anthropometric_horizontal_layer_emphasis",
        "split_ratio": split_ratio,
        "other_layer_fade_to_white": 0.82,
        "source_pattern_inferred_from_uv": False,
        "layers": records,
    }
    (path / "layers" / "layer_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def validate_blender_character_bundle(path: Path) -> dict:
    from PIL import Image

    manifest_path = path / "character_manifest.json"
    camera_path = path / "camera_metadata.json"
    required = [manifest_path, camera_path]
    required += [path / "rgb" / f"{camera}.png" for camera in VIEW_ORDER]
    required += [path / "masks" / f"{camera}.png" for camera in VIEW_ORDER]
    required += [path / "reweaver" / "render_output" / "rgb" / f"{camera}.png" for camera in VIEW_ORDER]
    required += [path / "garment_particles" / "input.png"]
    missing = [item.name for item in required if not item.is_file() or item.stat().st_size == 0]
    if missing:
        return {"valid": False, "failure": "MISSING_RENDER_BUNDLE_FILE", "missing": missing}

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cameras = json.loads(camera_path.read_text(encoding="utf-8"))
    dimensions = []
    nonempty_masks = []
    hashes = []
    for camera in VIEW_ORDER:
        rgb_path = path / "rgb" / f"{camera}.png"
        mask_path = path / "masks" / f"{camera}.png"
        with Image.open(rgb_path) as image:
            image.verify()
        with Image.open(rgb_path) as image:
            dimensions.append([image.width, image.height])
        with Image.open(mask_path) as image:
            alpha = image.getchannel("A") if "A" in image.getbands() else image.convert("L")
            nonempty_masks.append(alpha.getbbox() is not None)
        hashes.append(sha256(rgb_path))

    labels = [cameras.get(camera, {}).get("view_label") for camera in VIEW_ORDER]
    garment_objects = manifest.get("garment_objects", [])
    body_objects = manifest.get("body_objects", [])
    valid = (
        dimensions == [[518, 518]] * 4
        and all(nonempty_masks)
        and len(set(hashes)) == 4
        and labels == list(VIEW_LABELS)
        and bool(garment_objects)
        and bool(body_objects)
        and not set(garment_objects).intersection(body_objects)
    )
    return {
        "valid": valid,
        "failure": None if valid else "INVALID_BLENDER_CHARACTER_BUNDLE",
        "dimensions": dimensions,
        "view_labels": labels,
        "distinct_rgb_views": len(set(hashes)),
        "nonempty_masks": nonempty_masks,
        "garment_objects": garment_objects,
        "body_objects": body_objects,
        "source_asset": manifest.get("source_asset"),
        "reweaver_prepared": all((path / "reweaver" / "render_output" / "rgb" / f"{camera}.png").is_file() for camera in VIEW_ORDER),
        "garment_particles_prepared": (path / "garment_particles" / "input.png").is_file(),
    }
