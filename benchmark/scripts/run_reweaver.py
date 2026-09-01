from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark.adapters.reweaver import IMAGE_MEAN, IMAGE_STD, REWEAVER_COMMIT, VIEW_ORDER, sha256, summarize_output, write_summary


def detach_numpy(value):
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, dict):
        return {key: detach_numpy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [detach_numpy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(detach_numpy(item) for item in value)
    return value


def load_images(files: list[Path]):
    arrays = []
    for path in files:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32).transpose(2, 0, 1) / 255.0
        arrays.append(rgb)
    stacked = np.stack(arrays)
    return (stacked - IMAGE_MEAN) / IMAGE_STD


def resolve_input_files(input_root: Path, sample_id: str, layout: str) -> list[Path]:
    if layout == "prepared":
        input_dir = input_root / sample_id / "reweaver" / "render_output" / "rgb"
        files = [input_dir / f"{camera}.png" for camera in VIEW_ORDER]
    elif layout == "gcd-ts-tileable":
        input_dir = input_root / sample_id / "render_output" / "rgb"
        files = sorted(input_dir.glob("view_*.png"))
    else:
        raise ValueError(f"unknown input layout: {layout}")
    if len(files) != 4 or any(not path.is_file() for path in files):
        raise ValueError(f"INPUT_ADAPTER: expected exactly four input views in {input_dir}")
    dimensions = []
    for path in files:
        with Image.open(path) as image:
            dimensions.append([image.width, image.height])
    if dimensions != [[518, 518]] * 4:
        raise ValueError(f"INPUT_ADAPTER: expected 518x518 views, got {dimensions}")
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("external/ReWeaver-Code"))
    parser.add_argument("--input-root", type=Path, default=Path("data/processed/synbody"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints/ReWeaver/tileable"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/reweaver"))
    parser.add_argument("--samples", nargs="+", default=["synbody_cyan_jacket", "synbody_patterned_shirt"])
    parser.add_argument("--input-layout", choices=("prepared", "gcd-ts-tileable"), default="prepared")
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo.resolve()))
    import torch
    from config import ComplexStitchConfig, FlattenConfig, ImageEncoderConfig
    from models.complex_stitch import ComplexStitchModel
    from models.flatten import FlattenModel
    from vggtencoder.aggregator import Aggregator

    if not torch.cuda.is_available():
        raise RuntimeError("MODEL_ENVIRONMENT: CUDA is unavailable")
    device = torch.device("cuda:0")
    torch.manual_seed(20260825)
    torch.cuda.manual_seed_all(20260825)
    torch.backends.cudnn.deterministic = True

    complex_cfg = ComplexStitchConfig(d_model=768, topo_embed_dim=768, curve_avg_count=36, patch_avg_count=9)
    flatten_cfg = FlattenConfig(scale_loss_coef=1e-3, edge_classify_loss_coef=1, edge_geometry_loss_coef=300)
    encoder_cfg = ImageEncoderConfig()
    checkpoint_paths = {
        "complex_stitch": args.checkpoint_root / "complex_stitch.pth",
        "flatten": args.checkpoint_root / "flatten.pth",
        "img_encoder": args.checkpoint_root / "img_encoder.pth",
    }
    missing = [str(path) for path in checkpoint_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"CHECKPOINT_MISSING: {missing}")
    checkpoint_hashes = {name: sha256(path) for name, path in checkpoint_paths.items()}
    config_snapshot = {
        "repository_commit": REWEAVER_COMMIT,
        "checkpoint_repository_revision": "1112e56f58acfe744733b8a9964a9b9f3bc9669d",
        "checkpoint_variant": "tileable",
        "checkpoint_hashes": checkpoint_hashes,
        "view_order": list(VIEW_ORDER),
        "input_resolution": [518, 518],
        "image_mean": IMAGE_MEAN.reshape(-1).tolist(),
        "image_std": IMAGE_STD.reshape(-1).tolist(),
        "seed": 20260825,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "condition_path": "normalized RGB -> Aggregator image tokens -> ComplexStitch -> FlattenModel",
        "ground_truth_panel_json": "not loaded",
        "camera_parameters": "not consumed by official TestDataSet_GCD or model forward path",
    }

    started = time.perf_counter()
    encoder = Aggregator(encoder_cfg).to(device)
    complex_model = ComplexStitchModel(complex_cfg).to(device)
    flatten_model = FlattenModel(flatten_cfg).to(device)
    encoder.load_state_dict(torch.load(checkpoint_paths["img_encoder"], map_location=device, weights_only=True))
    complex_model.load_state_dict(torch.load(checkpoint_paths["complex_stitch"], map_location=device, weights_only=True))
    flatten_model.load_state_dict(torch.load(checkpoint_paths["flatten"], map_location=device, weights_only=True))
    encoder.eval()
    complex_model.eval()
    flatten_model.eval()
    load_seconds = time.perf_counter() - started

    run_records = []
    for sample_id in args.samples:
        input_files = resolve_input_files(args.input_root, sample_id, args.input_layout)
        input_validation = {
            "valid": True,
            "dimensions": [[518, 518]] * 4,
            "sha256": [sha256(path) for path in input_files],
        }
        tensor = torch.from_numpy(load_images(input_files)).unsqueeze(0).to(device)
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        with torch.inference_mode():
            image_tokens, _ = encoder(tensor)
            batch, views, tokens, dimension = image_tokens.shape
            image_tokens = image_tokens.reshape(batch, views * tokens, dimension)
            curve_predictions, patch_predictions, curve_features, patch_features = complex_model(image_tokens)
            scaled = complex_model.get_scaled_points(patch_features)
            patch_predictions["pred_patch_points_scaled"] = scaled["pred_patch_points_scaled"]
            prediction = flatten_model.infer(curve_predictions, patch_predictions, curve_features, patch_features, names=[sample_id])[0]
        torch.cuda.synchronize(device)
        runtime = time.perf_counter() - start
        save_dict = detach_numpy(prediction)
        save_dict["patch_points_scaled"] = np.array(save_dict["patch_points_scaled"], dtype=object)
        sample_output = args.output_root / sample_id
        sample_output.mkdir(parents=True, exist_ok=True)
        (sample_output / "config.json").write_text(json.dumps(config_snapshot, indent=2), encoding="utf-8")
        npz_path = sample_output / f"{sample_id}.npz"
        np.savez_compressed(npz_path, **save_dict)
        summary = summarize_output(npz_path)
        summary.update({
            "sample_id": sample_id,
            "runtime_seconds": runtime,
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
            "input_hashes": input_validation["sha256"],
            "checkpoint_hashes": checkpoint_hashes,
            "repository_commit": REWEAVER_COMMIT,
        })
        write_summary(sample_output / "summary.json", summary)
        run_records.append(summary)

    print(json.dumps({"status": "PASS" if all(item["valid"] for item in run_records) else "FAILED_VALIDATION", "model_load_seconds": load_seconds, "samples": run_records}))


if __name__ == "__main__":
    main()
