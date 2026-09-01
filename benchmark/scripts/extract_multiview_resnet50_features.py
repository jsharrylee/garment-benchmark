from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frozen ResNet-50 features from all four real GCDv2 views.")
    parser.add_argument("--index", type=Path, default=Path("artifacts/retrieval_v2/garmentcode_v2_batch_0_multiview_index.json"))
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path.home() / ".cache/torch/hub/checkpoints/maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/drafting_semantics/multiview_pattern_semantics/resnet50_features.npz"))
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/drafting_semantics/multiview_pattern_semantics/resnet50_features_manifest.json"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    import torch
    from PIL import Image
    from torchvision.models import resnet50
    from torchvision.transforms import v2

    if not args.weights.is_file():
        raise SystemExit(f"local pretrained weights not found: {args.weights}; no network download is attempted")
    payload = json.loads(args.index.read_text(encoding="utf-8"))
    records = payload["records"]
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    state = torch.load(args.weights, map_location="cpu", weights_only=True)
    backbone = {key.removeprefix("backbone.body."): value for key, value in state.items() if key.startswith("backbone.body.")}
    model = resnet50(weights=None)
    missing, unexpected = model.load_state_dict(backbone, strict=False)
    allowed_missing = {"fc.weight", "fc.bias"}
    if set(missing) != allowed_missing or unexpected:
        raise RuntimeError(f"unexpected ResNet load mismatch: missing={missing} unexpected={unexpected}")
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    transform = v2.Compose(
        [
            v2.Resize((224, 224), antialias=True),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    all_paths = [(record["sample_id"], view) for record in records for view in record["source_views"]]
    output = np.zeros((len(records) * 4, 2048), dtype=np.float16)
    with torch.inference_mode():
        for start in range(0, len(all_paths), args.batch_size):
            images = []
            for _, raw_path in all_paths[start : start + args.batch_size]:
                with Image.open(raw_path) as image:
                    images.append(transform(image.convert("RGB")))
            batch = torch.stack(images).to(device)
            with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                features = model(batch)
            output[start : start + len(images)] = features.float().cpu().numpy().astype(np.float16)
            if start == 0 or (start // args.batch_size) % 40 == 0:
                print(json.dumps({"images": min(start + len(images), len(all_paths)), "total": len(all_paths)}), flush=True)
    output = output.reshape(len(records), 4, 2048)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        sample_ids=np.asarray([record["sample_id"] for record in records]),
        features=output,
    )
    manifest = {
        "schema_version": "gcdv2-four-view-resnet50-features-1.0",
        "record_count": len(records),
        "image_count": len(all_paths),
        "feature_shape": list(output.shape),
        "feature_dtype": str(output.dtype),
        "backbone": "torchvision ResNet-50 initialized from local Mask R-CNN v2 COCO checkpoint backbone",
        "weights_sha256": _sha256(args.weights),
        "index_sha256": _sha256(args.index),
        "output_sha256": _sha256(args.output),
        "network_download": False,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
