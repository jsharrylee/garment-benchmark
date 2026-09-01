from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmark.drafting_semantics.multiview_curve_parameters import (
    build_local_maskrcnn_fpn_backbone,
    build_spatial_curve_model,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract pooled spatial ResNet-50-FPN tokens from the four actual GCDv2 views."
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("artifacts/retrieval_v2/garmentcode_v2_batch_0_multiview_index.json"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("benchmark/configs/multiview_curve_parameters_fpn.json"),
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path.home()
        / ".cache/torch/hub/checkpoints/maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/drafting_semantics/multiview_curve_parameters/resnet50_fpn_tokens.npz"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/drafting_semantics/multiview_curve_parameters/resnet50_fpn_tokens_manifest.json"),
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Garments per batch (four images each).")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    import torch
    from PIL import Image
    from torchvision.transforms import v2

    if not args.weights.is_file():
        raise SystemExit(
            f"local pretrained weights not found: {args.weights}; no network download is attempted"
        )
    config = json.loads(args.config.read_text(encoding="utf-8"))
    payload = json.loads(args.index.read_text(encoding="utf-8"))
    records = payload["records"]
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    backbone = build_local_maskrcnn_fpn_backbone(args.weights)
    model = build_spatial_curve_model(config, backbone=backbone).to(device).eval()
    extractor = model.extractor
    if extractor is None:
        raise RuntimeError("the image extractor was not installed")
    transform = v2.Compose(
        [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
    )
    token_count = int(model.tokens_per_view)
    feature_dim = int(config["spatial_feature_dim"])
    output = np.empty((len(records), 4, token_count, feature_dim), dtype=np.float16)
    with torch.inference_mode():
        for start in range(0, len(records), args.batch_size):
            current = records[start : start + args.batch_size]
            images = []
            for row in current:
                if len(row["source_views"]) != 4:
                    raise ValueError(f"{row['sample_id']} does not have exactly four views")
                for raw_path in row["source_views"]:
                    with Image.open(raw_path) as image:
                        images.append(transform(image.convert("RGB")))
            # Source renders have one fixed size; extractor performs the
            # configured resize and ImageNet normalization.
            batch = torch.stack(images).to(device)
            with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                tokens = extractor(batch)
            tokens = tokens.reshape(len(current), 4, token_count, feature_dim)
            output[start : start + len(current)] = tokens.float().cpu().numpy().astype(np.float16)
            if start == 0 or (start // args.batch_size) % 25 == 0:
                print(json.dumps({"garments": start + len(current), "total": len(records)}), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Uncompressed NPZ avoids a long, memory-heavy compression pass.  The
    # artifact is ignored and can be regenerated from the manifest contract.
    np.savez(
        args.output,
        sample_ids=np.asarray([row["sample_id"] for row in records]),
        features=output,
    )
    manifest = {
        "schema_version": "gcdv2-four-view-resnet50-fpn-spatial-features-1.0",
        "record_count": len(records),
        "image_count": len(records) * 4,
        "feature_shape": list(output.shape),
        "feature_dtype": str(output.dtype),
        "pyramid_levels": list(config["pyramid_levels"]),
        "pyramid_grid_sizes": list(config["pyramid_grid_sizes"]),
        "tokens_per_view": token_count,
        "backbone": "torchvision Mask R-CNN v2 ResNet-50-FPN loaded from an explicit local checkpoint",
        "weights_sha256": _sha256(args.weights),
        "index_sha256": _sha256(args.index),
        "output_sha256": _sha256(args.output),
        "network_download": False,
    }
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
