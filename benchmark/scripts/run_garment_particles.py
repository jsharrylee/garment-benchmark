from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from benchmark.adapters.flash_attn_compat import install_flash_attn_compat, install_missing_image_block_v2
from benchmark.adapters.garment_particles import (
    GARMENT_PARTICLES_COMMIT,
    MODEL_REVISION,
    N_CURVES,
    N_EDGE_PARAMS,
    N_PANELS,
    N_POINTS,
    parse_predictions,
    sha256,
    stitch_pairs,
    summarize_output,
    write_ascii_ply,
    write_summary,
)
from benchmark.adapters.reweaver import VIEW_ORDER, validate_input_directory


def tensor_sha256(tensor) -> str:
    array = tensor.detach().cpu().float().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def load_dcp(model, checkpoint: Path) -> dict:
    import pathlib

    if os.name == "nt":
        pathlib.PosixPath = pathlib.WindowsPath
    import torch.distributed.checkpoint as dcp
    from utils.train_utils import check_checkpoint_compatibility

    missing, unexpected, common = check_checkpoint_compatibility(str(checkpoint), model, app_name="app.ema")
    model_state = model.state_dict()
    common_state = {key.removeprefix("app.ema."): model_state[key.removeprefix("app.ema.")] for key in common}
    state = {"app": {"ema": common_state}}
    dcp.load(state, checkpoint_id=str(checkpoint), no_dist=True)
    loaded_missing, loaded_unexpected = model.load_state_dict(state["app"]["ema"], strict=False)
    missing_prefixes = Counter(key.split(".", 1)[0] for key in loaded_missing)
    unsupported_missing = [key for key in loaded_missing if not key.startswith(("image_encoder.", "text_encoder."))]
    if unsupported_missing:
        raise RuntimeError(f"CHECKPOINT_MISSING: non-encoder model keys absent: {unsupported_missing[:10]}")
    common_parameters = sum(model_state[key.removeprefix("app.ema.")].numel() for key in common)
    total_parameters = sum(value.numel() for value in model_state.values())
    return {
        "metadata_missing_key_count": len(missing),
        "metadata_unexpected_key_count": len(unexpected),
        "load_missing_key_count": len(loaded_missing),
        "load_unexpected_key_count": len(loaded_unexpected),
        "missing_prefix_counts": dict(sorted(missing_prefixes.items())),
        "common_parameter_fraction": common_parameters / total_parameters,
    }


def build_sampler(config, steps: int):
    from transport import Sampler, Transport

    transport = Transport(
        model_type=config.transport.model_type,
        path_type=config.transport.path_type,
        loss_type=config.transport.loss_type,
        train_eps=config.transport.train_eps,
        sample_eps=config.transport.sample_eps,
        use_lognorm=config.transport.use_lognorm,
    )
    return Sampler(transport).sample_ode(
        sampling_method=config.sample.sampling_method,
        num_steps=steps,
        atol=config.sample.atol,
        rtol=config.sample.rtol,
        reverse=config.sample.reverse,
        timestep_shift=config.sample.timestep_shift,
    )


def instantiate_bfloat16(config):
    import hydra
    import torch

    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        return hydra.utils.instantiate(config)
    finally:
        torch.set_default_dtype(previous)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run official Garment Particles image PGF and edge recovery without GCDv2 target leakage.")
    parser.add_argument("--repo", type=Path, default=Path("external/GarmentParticles"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints/GarmentParticles"))
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input", type=Path)
    inputs.add_argument("--input-dir", type=Path, help="CAM000..CAM003 directory for mean image-token fusion")
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/garment_particles"))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()
    if args.steps <= 0:
        raise ValueError("num sampling steps must be positive")

    project_root = Path.cwd().resolve()
    os.environ.setdefault("HF_HOME", str((project_root / "cache" / "huggingface").resolve()))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    source_root = (args.repo / "src").resolve()
    sys.path.insert(0, str(source_root))
    install_flash_attn_compat()
    install_missing_image_block_v2()

    import hydra
    import torch
    from omegaconf import OmegaConf
    from transformers import AutoImageProcessor, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("MODEL_ENVIRONMENT: CUDA is unavailable")
    device = torch.device("cuda:0")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    pgf_checkpoint = args.checkpoint_root / "pgf_image"
    edge_checkpoint = args.checkpoint_root / "edge"
    for checkpoint in (pgf_checkpoint, edge_checkpoint):
        if not (checkpoint / ".metadata").is_file():
            raise FileNotFoundError(f"CHECKPOINT_MISSING: {checkpoint}")

    pgf_config = OmegaConf.load(source_root / "configs" / "pretrained" / "pgf_image.yaml")
    edge_config = OmegaConf.load(source_root / "configs" / "pretrained" / "edge.yaml")
    output_dir = args.output_root / args.sample_id
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.input_dir:
        validation = validate_input_directory(args.input_dir)
        if not validation["valid"]:
            raise ValueError(f"invalid four-view input: {validation}")
        input_paths = [args.input_dir / f"{camera}.png" for camera in VIEW_ORDER]
        condition_mode = "four_view_mean_image_token_fusion"
    else:
        input_paths = [args.input]
        condition_mode = "single_front_image"
    input_rgbs = []
    for input_path in input_paths:
        with Image.open(input_path) as image:
            input_rgbs.append(image.convert("RGB"))
    processor = AutoImageProcessor.from_pretrained(pgf_config.model.image_encoder_name, use_fast=False)
    tokenizer = AutoTokenizer.from_pretrained(pgf_config.model.text_encoder_name, use_fast=False)
    pixel_values_cpu = processor(input_rgbs, return_tensors="pt")["pixel_values"]
    text = tokenizer("", max_length=77, padding="max_length", truncation=True, return_tensors="pt")
    text_tokens_cpu = text["input_ids"]
    text_mask_cpu = text["attention_mask"].bool()

    pgf_started = time.perf_counter()
    print("PGF_PREPARE_MODEL", flush=True)
    pgf_model = instantiate_bfloat16(pgf_config.model)
    pgf_load = load_dcp(pgf_model, pgf_checkpoint)
    pgf_model.eval()
    with torch.inference_mode():
        pgf_model.image_encoder.to(device)
        per_view_hidden = pgf_model.image_encoder(
            pixel_values_cpu.to(device=device, dtype=torch.bfloat16)
        ).last_hidden_state.detach().cpu()
        image_hidden = per_view_hidden.mean(dim=0, keepdim=True)
        pgf_model.image_encoder.to("cpu")
        torch.cuda.empty_cache()
        pgf_model.text_encoder.to(device)
        text_hidden = pgf_model.text_encoder(text_tokens_cpu.to(device), attention_mask=text_mask_cpu.to(device)).last_hidden_state.detach().cpu()
        pgf_model.text_encoder.to("cpu")
        torch.cuda.empty_cache()
    image_embedding_hash = tensor_sha256(image_hidden)

    class StaticEncoder(torch.nn.Module):
        def __init__(self, hidden):
            super().__init__()
            self.register_buffer("hidden", hidden)

        def forward(self, *_args, **_kwargs):
            return SimpleNamespace(last_hidden_state=self.hidden)

    pgf_model.image_encoder = StaticEncoder(image_hidden)
    pgf_model.text_encoder = StaticEncoder(text_hidden)
    pgf_model = pgf_model.to(device=device, dtype=torch.bfloat16)
    pgf_model.eval()
    pixel_values = pixel_values_cpu.mean(dim=0, keepdim=True).unsqueeze(1).to(device=device, dtype=torch.bfloat16)
    text_tokens = text_tokens_cpu.to(device)
    text_mask = text_mask_cpu.to(device)
    point_mask = torch.ones((1, N_POINTS), dtype=torch.bool, device=device)
    particle_noise = torch.randn((1, N_POINTS, 6), device=device, dtype=torch.bfloat16)
    pgf_sampler = build_sampler(pgf_config, args.steps)
    torch.cuda.reset_peak_memory_stats(device)
    sample_started = time.perf_counter()
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        particle_steps = pgf_sampler(
            particle_noise,
            pgf_model.forward_with_mask,
            mask=point_mask,
            pixel_values=pixel_values,
            text_tokens=text_tokens,
            text_attn_mask=text_mask,
            image_path=[str(input_paths[0])],
        )
    torch.cuda.synchronize(device)
    pgf_sample_seconds = time.perf_counter() - sample_started
    pgf_peak = int(torch.cuda.max_memory_allocated(device))
    particles_normalized = particle_steps[-1].float().cpu().numpy()[0]
    del particle_steps, particle_noise, pgf_sampler, pgf_model, image_hidden, text_hidden
    gc.collect()
    torch.cuda.empty_cache()
    pgf_total_seconds = time.perf_counter() - pgf_started

    edge_started = time.perf_counter()
    edge_config.model._target_ = "models.lightningdit_edge_model.LightningCrossAttnDiTV3EdgeModel"
    edge_config.model.backend = "flash-attn"
    print("EDGE_PREPARE_MODEL", flush=True)
    edge_model = instantiate_bfloat16(edge_config.model)
    edge_load = load_dcp(edge_model, edge_checkpoint)
    edge_model = edge_model.to(device=device, dtype=torch.bfloat16).eval()
    edge_sampler = build_sampler(edge_config, args.steps)
    generated_context = torch.from_numpy(particles_normalized).unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    point_mask = torch.ones((1, N_POINTS), dtype=torch.bool, device=device)
    edge_noise = torch.randn((1, N_PANELS * N_CURVES, N_EDGE_PARAMS), device=device, dtype=torch.bfloat16)
    torch.cuda.reset_peak_memory_stats(device)
    sample_started = time.perf_counter()
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        edge_steps = edge_sampler(
            edge_noise,
            edge_model.forward,
            panel_points=generated_context,
            panel_points_mask=point_mask,
            panel_indices=None,
        )
    torch.cuda.synchronize(device)
    edge_sample_seconds = time.perf_counter() - sample_started
    edge_peak = int(torch.cuda.max_memory_allocated(device))
    edge_normalized = edge_steps[-1].float().cpu().numpy()[0]
    edge_total_seconds = time.perf_counter() - edge_started

    parsed = parse_predictions(particles_normalized, edge_normalized)
    pairs = stitch_pairs(parsed["stitch_flags"], parsed["stitch_tags"], parsed["edge_valid_mask"])
    prediction_path = output_dir / "prediction.npz"
    np.savez_compressed(prediction_path, **parsed, stitch_pairs=pairs)
    write_ascii_ply(output_dir / "particles.ply", parsed["particles"])
    summary = summarize_output(prediction_path)
    input_hashes = [sha256(path) for path in input_paths]
    combined_input_hash = (
        input_hashes[0]
        if len(input_hashes) == 1
        else hashlib.sha256("".join(input_hashes).encode("ascii")).hexdigest()
    )
    summary.update({
        "sample_id": args.sample_id,
        "input_sha256": combined_input_hash,
        "input_sha256_per_view": input_hashes,
        "condition_mode": condition_mode,
        "condition_view_count": len(input_paths),
        "image_embedding_sha256": image_embedding_hash,
        "repository_commit": GARMENT_PARTICLES_COMMIT,
        "checkpoint_revision": MODEL_REVISION,
        "sampling_steps_per_stage": args.steps,
        "seed": args.seed,
        "pgf_load": pgf_load,
        "edge_load": edge_load,
        "pgf_sample_seconds": pgf_sample_seconds,
        "edge_sample_seconds": edge_sample_seconds,
        "pgf_total_seconds": pgf_total_seconds,
        "edge_total_seconds": edge_total_seconds,
        "pgf_peak_vram_bytes": pgf_peak,
        "edge_peak_vram_bytes": edge_peak,
        "attention_backend": "PyTorch SDPA compatibility for FlashAttention API (PGF and non-varlen edge)",
        "point_mask": "fixed all-valid 8196; no GCDv2 sample metadata",
        "image_dropout": 0,
        "text_condition": "empty",
    })
    write_summary(output_dir / "summary.json", summary)
    config = {
        "repository_commit": GARMENT_PARTICLES_COMMIT,
        "checkpoint_revision": MODEL_REVISION,
        "pgf_checkpoint": "pgf_image",
        "edge_checkpoint": "edge",
        "sampling_steps_per_stage": args.steps,
        "seed": args.seed,
        "input_sha256": summary["input_sha256"],
        "input_sha256_per_view": input_hashes,
        "condition_mode": condition_mode,
        "condition_view_count": len(input_paths),
        "image_embedding_sha256": image_embedding_hash,
        "n_points": N_POINTS,
        "n_panels": N_PANELS,
        "n_curves_including_metadata_row": N_CURVES,
        "n_edge_params": N_EDGE_PARAMS,
        "ground_truth_dataset_files_loaded": False,
    }
    write_summary(output_dir / "config.json", config)
    print(json.dumps({"status": "PASS" if summary["valid"] else "FAILED_VALIDATION", "summary": summary}))
    if not summary["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
