from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from benchmark.gcdv2_exact.neurosymbolic_learning import build_visual_model
from benchmark.gcdv2_exact.predicted_contours import contour_from_probability, symmetric_chamfer


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract 256-point contours from learned GCDv2 mask/SDF predictions.")
    parser.add_argument("--index", type=Path, default=Path("artifacts/gcdv2_neurosymbolic_v1/panel_index.jsonl"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/gcdv2_neurosymbolic/visual_geometry.pt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/gcdv2_predicted_contours_v1/predicted_contours.npz"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/gcdv2_predicted_contours_v1.json"))
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    import torch

    rows = [json.loads(line) for line in args.index.read_text(encoding="utf-8").splitlines() if line.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = build_visual_model(int(checkpoint["base_width"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    contours = np.zeros((len(rows), 256, 2), np.float32)
    valid = np.zeros(len(rows), bool)
    chamfer = np.full(len(rows), np.nan, np.float32)
    failures = []
    started = time.perf_counter()
    for start in range(0, len(rows), args.batch_size):
        current = rows[start : start + args.batch_size]
        images = []
        truth = []
        for row in current:
            with Image.open(row["input_panel_image"]) as image:
                images.append(np.asarray(image.convert("L").resize((128, 128), Image.Resampling.LANCZOS), np.float32)[None] / 255.0)
            with np.load(row["visual_truth_path"]) as visual:
                truth.append(visual["dense_contour_uv_f32"].astype(np.float32))
        with torch.no_grad(), torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            probability = model(torch.from_numpy(np.stack(images)).to(device))["mask_logits"].sigmoid().float().cpu().numpy()[:, 0]
        for local, (prediction, expected, row) in enumerate(zip(probability, truth, current, strict=True)):
            target_index = start + local
            try:
                contour = contour_from_probability(prediction)
                contours[target_index] = contour
                valid[target_index] = True
                chamfer[target_index] = symmetric_chamfer(contour, expected)
            except Exception as error:
                failures.append({"panel_uid": row["panel_uid"], "error": str(error)})
        if start % (args.batch_size * 20) == 0:
            print(json.dumps({"processed": min(start + len(current), len(rows)), "total": len(rows)}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, contours=contours, valid=valid, symmetric_chamfer_uv=chamfer)
    split_metrics = {}
    for split in ("train", "validation", "test"):
        indices = np.asarray([index for index, row in enumerate(rows) if row["split"] == split and valid[index]])
        split_metrics[split] = {"count": int(len(indices)), "mean_symmetric_chamfer_uv": float(np.nanmean(chamfer[indices]))}
    manifest = {
        "schema_version": "gcdv2-predicted-contours-1.0",
        "status": "PASS" if not failures else "FAIL",
        "panel_count": len(rows),
        "valid_count": int(valid.sum()),
        "failure_count": len(failures),
        "failures": failures[:100],
        "split_metrics": split_metrics,
        "seconds": time.perf_counter() - started,
        "source": "learned visual_geometry.pt mask head; largest external contour; equal-arclength 256 samples",
        "artifact": args.output.as_posix(),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
