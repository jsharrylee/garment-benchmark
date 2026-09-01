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
from benchmark.gcdv2_exact.pattern_learning import SPATIAL_GRID_SIZES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_image(path: Path, size: int) -> np.ndarray:
    # Resize before tensor conversion: the source is 1024 square and retaining
    # it in a training batch wastes both host and device memory.
    with Image.open(path) as image:
        rgb = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        value = np.asarray(rgb, dtype=np.float32) / 255.0
    return np.transpose(value, (2, 0, 1))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract frozen local Mask R-CNN ResNet50-FPN tokens from clean GCDv2 pattern PNGs."
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("artifacts/gcdv2_exact_pairs_v1/index.jsonl"),
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("cache/torch/hub/checkpoints/maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/gcdv2_exact_pairs_v1/pattern_fpn_tokens.npz"),
    )
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    import torch
    import torch.nn.functional as functional

    rows = [
        json.loads(line)
        for line in args.index.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows.sort(key=lambda row: (str(row["category"]), str(row["sample_id"])))
    if args.limit is not None:
        rows = rows[: int(args.limit)]
    if not rows:
        raise SystemExit("exact-pair index contains no rows")
    missing = [row["pattern_path"] for row in rows if not Path(row["pattern_path"]).is_file()]
    if missing:
        raise FileNotFoundError(f"missing clean pattern image: {missing[0]}")
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    backbone = build_local_maskrcnn_fpn_backbone(args.weights).to(device).eval()
    mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)
    features = []
    started = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(rows), int(args.batch_size)):
            current = rows[offset : offset + int(args.batch_size)]
            batch = np.stack(
                [_load_image(Path(row["pattern_path"]), int(args.image_size)) for row in current]
            )
            images = torch.from_numpy(batch).to(device)
            images = (images - mean) / std
            with torch.amp.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                pyramid = backbone(images)
                tokens = []
                for level, grid in zip(("0", "1", "2", "3"), SPATIAL_GRID_SIZES):
                    if level not in pyramid:
                        raise KeyError(f"FPN output is missing level {level!r}")
                    pooled = functional.adaptive_avg_pool2d(pyramid[level], (grid, grid))
                    tokens.append(pooled.flatten(2).transpose(1, 2))
                value = torch.cat(tokens, dim=1).float().cpu().numpy().astype(np.float16)
            features.append(value)
            completed = min(offset + len(current), len(rows))
            if completed == len(rows) or completed % 120 == 0:
                print(f"features {completed}/{len(rows)}")
    stacked = np.concatenate(features, axis=0)
    metadata = {
        "schema_version": "gcdv2-pattern-fpn-1.0",
        "source_index": str(args.index.as_posix()),
        "source_index_sha256": _sha256(args.index),
        "weights": str(args.weights.as_posix()),
        "weights_sha256": _sha256(args.weights),
        "backbone": "MaskRCNN ResNet50-FPN v2 COCO, frozen, local checkpoint only",
        "input_contract": "clean pattern.png RGB only; no labels, scale, category, or sidecar supplied to backbone",
        "image_size": int(args.image_size),
        "fpn_levels": ["0", "1", "2", "3"],
        "grid_sizes": list(SPATIAL_GRID_SIZES),
        "tokens_per_image": int(stacked.shape[1]),
        "feature_dimension": int(stacked.shape[2]),
        "storage_dtype": "float16",
        "sample_count": len(rows),
        "device": str(device),
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        sample_ids=np.asarray([str(row["sample_id"]) for row in rows]),
        spatial_features=stacked,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(json.dumps({**metadata, "output": str(args.output.as_posix())}, indent=2))


if __name__ == "__main__":
    main()
