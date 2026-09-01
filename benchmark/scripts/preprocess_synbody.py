from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark.preprocessing.masking import TorchvisionPersonSegmenter, grabcut_mask, mask_statistics
from benchmark.preprocessing.normalization import contain_square
from benchmark.preprocessing.view_selection import resolve_selected_views, validate_selected_views
from benchmark.visualization.contact_sheet import create_contact_sheet


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def process_sample(root: Path, output_root: Path, sample: dict, *, mask_backend: str, segmenter=None) -> dict:
    selected = resolve_selected_views(root, sample)
    validate_selected_views(selected)
    sample_dir = output_root / sample["sample_id"]
    originals = sample_dir / "original"
    masks = sample_dir / "mask"
    masked = sample_dir / "masked"
    normalized = sample_dir / "reweaver" / "render_output" / "rgb"
    gp_dir = sample_dir / "garment_particles"
    for path in (originals, masks, masked, normalized, gp_dir):
        path.mkdir(parents=True, exist_ok=True)

    records = []
    for view in selected:
        original_path = originals / f"{view.camera}.jpeg"
        shutil.copy2(view.source, original_path)
        with Image.open(view.source) as source_image:
            rgb_image = source_image.convert("RGB")
        rgb = np.asarray(rgb_image)
        if mask_backend == "maskrcnn":
            mask_array = segmenter(rgb, view.bbox_xyxy)
            mask_method = "torchvision_maskrcnn_resnet50_fpn_v2_coco_v1"
        else:
            mask_array = grabcut_mask(rgb, view.bbox_xyxy)
            mask_method = "opencv_grabcut_bbox_seed_v2"
        mask_image = Image.fromarray(mask_array, mode="L")
        mask_path = masks / f"{view.camera}.png"
        mask_image.save(mask_path)
        masked_image = Image.new("RGB", rgb_image.size, "white")
        masked_image.paste(rgb_image, mask=mask_image)
        masked_path = masked / f"{view.camera}.png"
        masked_image.save(masked_path)
        normalized_image, transform = contain_square(rgb_image, mask_image)
        normalized_path = normalized / f"{view.camera}.png"
        normalized_image.save(normalized_path)
        stats = mask_statistics(mask_array)
        if stats["foreground_fraction"] < 0.001:
            raise ValueError(f"MASK_FAILURE: {sample['sample_id']} {view.camera} is nearly empty")
        records.append({
            "camera": view.camera,
            "source_relative": str(view.source.relative_to(root)),
            "source_sha256": sha256(view.source),
            "tracked_bbox_xyxy": list(view.bbox_xyxy),
            "mask_method": mask_method,
            "mask": stats,
            "transform": transform,
            "normalized_sha256": sha256(normalized_path),
        })

    representative = sample["representative_view"]
    representative_path = normalized / f"{representative}.png"
    gp_path = gp_dir / "input.png"
    shutil.copy2(representative_path, gp_path)

    labels = [item.camera for item in selected]
    create_contact_sheet([originals / f"{label}.jpeg" for label in labels], sample_dir / "original_views.jpg", labels)
    create_contact_sheet([masks / f"{label}.png" for label in labels], sample_dir / "foreground_masks.jpg", labels)
    create_contact_sheet([masked / f"{label}.png" for label in labels], sample_dir / "masked_views.jpg", labels)
    create_contact_sheet([normalized / f"{label}.png" for label in labels], sample_dir / "normalized_inputs.jpg", labels, cell=(518, 518))

    return {
        "sample_id": sample["sample_id"],
        "source_dataset": "SynBody S100K_rgb_part_1",
        "scene": sample["scene"],
        "sequence": sample["sequence"],
        "frame": sample["frame"],
        "camera_order": labels,
        "representative_view": representative,
        "representative_input_sha256": sha256(gp_path),
        "official_mask_status": "NOT_PRESENT_IN_LOCAL_RGB_PART",
        "views": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("benchmark/configs/synbody_samples.json"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/processed/synbody"))
    parser.add_argument("--mask-backend", choices=("grabcut", "maskrcnn"), default="grabcut")
    parser.add_argument("--mask-device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    root = args.root or Path(os.environ[config["source_root_env"]])
    segmenter = TorchvisionPersonSegmenter(args.mask_device) if args.mask_backend == "maskrcnn" else None
    manifests = [process_sample(root, args.output, sample, mask_backend=args.mask_backend, segmenter=segmenter) for sample in config["samples"]]
    manifest = {
        "schema_version": 1,
        "adapter_version": "synbody-preprocess-v2",
        "mask_backend": args.mask_backend,
        "source_root_env": config["source_root_env"],
        "samples": manifests,
    }
    manifest_path = args.output / "sample_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", "samples": len(manifests), "manifest": str(manifest_path)}))


if __name__ == "__main__":
    main()
