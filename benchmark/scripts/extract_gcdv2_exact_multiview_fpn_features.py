from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark.drafting_semantics.multiview_curve_parameters import (
    build_local_maskrcnn_fpn_backbone,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as image:
        value = np.asarray(
            image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR),
            dtype=np.float32,
        ) / 255.0
    return np.transpose(value, (2, 0, 1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract complete four-view FPN features for every exact GCDv2 pair.")
    parser.add_argument("--index", type=Path, default=Path("artifacts/gcdv2_exact_pairs_v1/index.jsonl"))
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("cache/torch/hub/checkpoints/maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/gcdv2_exact_pairs_v1/multiview_fpn_tokens.npz"),
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Garments per batch; four images each.")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    import torch
    import torch.nn.functional as functional

    rows = [json.loads(line) for line in args.index.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows.sort(key=lambda value: str(value["sample_id"]))
    if args.limit is not None:
        rows = rows[: int(args.limit)]
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    backbone = build_local_maskrcnn_fpn_backbone(args.weights).to(device).eval()
    mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)
    # Keep the historical cache order so all existing exact-model loaders can
    # apply their audited [1,0,2,3] mapping to semantic front/back/left/right.
    legacy_from_semantic = (1, 0, 2, 3)
    levels = ("0", "1", "2", "3")
    grids = (8, 4, 2, 1)
    output = np.empty((len(rows), 4, sum(grid * grid for grid in grids), 256), dtype=np.float16)
    started = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(rows), int(args.batch_size)):
            current = rows[offset : offset + int(args.batch_size)]
            images = []
            for row in current:
                label = json.loads(Path(row["label_path"]).read_text(encoding="utf-8"))
                semantic_paths = [Path(value["path"]) for value in label["views"]]
                if len(semantic_paths) != 4 or not all(path.is_file() for path in semantic_paths):
                    raise FileNotFoundError(f"incomplete four-view pair: {row['sample_id']}")
                images.extend(_image(semantic_paths[index], int(args.image_size)) for index in legacy_from_semantic)
            tensor = (torch.from_numpy(np.stack(images)).to(device) - mean) / std
            with torch.amp.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                pyramid = backbone(tensor)
                tokens = []
                for level, grid in zip(levels, grids):
                    pooled = functional.adaptive_avg_pool2d(pyramid[level], (grid, grid))
                    tokens.append(pooled.flatten(2).transpose(1, 2))
                value = torch.cat(tokens, dim=1).reshape(len(current), 4, -1, 256)
            output[offset : offset + len(current)] = value.float().cpu().numpy().astype(np.float16)
            completed = offset + len(current)
            if completed == len(rows) or completed % 200 == 0:
                print(json.dumps({"garments": completed, "total": len(rows)}), flush=True)

    metadata = {
        "schema_version": "gcdv2-exact-four-view-fpn-1.0",
        "source_index": args.index.as_posix(),
        "source_index_sha256": _sha256(args.index),
        "sample_count": len(rows),
        "feature_shape": list(output.shape),
        "feature_dtype": str(output.dtype),
        "cache_view_order": ["CAM000_back", "CAM001_front", "CAM002_left", "CAM003_right"],
        "semantic_reorder": [1, 0, 2, 3],
        "semantic_view_order_after_reorder": ["front", "back", "left", "right"],
        "input_image_size": int(args.image_size),
        "fpn_levels": list(levels),
        "fpn_grid_sizes": list(grids),
        "backbone": "MaskRCNN ResNet50-FPN v2 COCO; frozen; explicit local checkpoint",
        "weights_sha256": _sha256(args.weights),
        "network_download": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        sample_ids=np.asarray([str(row["sample_id"]) for row in rows]),
        features=output,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(json.dumps({**metadata, "output": args.output.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
