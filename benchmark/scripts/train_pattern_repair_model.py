from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from benchmark.pattern_pipeline.geometry import boundary_points
from benchmark.pattern_pipeline.schema import PatternDocument
from benchmark.pattern_pipeline.validation import validate_pattern
from benchmark.pattern_repair.data import (
    corrupt_loop,
    generate_clean_loop,
    loop_features,
    normalize_loop,
    strict_self_intersections,
    synthetic_batch,
)
from benchmark.pattern_repair.model import DEFAULT_MODEL_CONFIG, build_model


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "benchmark" / "configs" / "pattern_repair_model.json"


def discover_clean_loops(root: Path, maximum_nodes: int, *, trusted_canonical_pool: bool = False) -> tuple[np.ndarray, ...]:
    loops = []
    for path in root.glob("**/*.json"):
        try:
            document = PatternDocument.read_json(path)
        except (KeyError, ValueError, json.JSONDecodeError):
            continue
        if not trusted_canonical_pool and not validate_pattern(document).accepted:
            continue
        for panel in document.panels:
            values = np.asarray(boundary_points(panel), dtype=np.float32)
            if len(values) > 1 and np.allclose(values[0], values[-1]):
                values = values[:-1]
            if 8 <= len(values) <= maximum_nodes:
                loops.append(normalize_loop(values)[0])
    return tuple(loops)


def repair_loss(predicted, target, mask):
    import torch

    weights = mask.unsqueeze(-1)
    coordinate = torch.nn.functional.smooth_l1_loss(predicted[weights.expand_as(predicted)], target[weights.expand_as(target)])
    adjacent_mask = (mask[:, 1:] & mask[:, :-1]).unsqueeze(-1)
    predicted_edges = predicted[:, 1:] - predicted[:, :-1]
    target_edges = target[:, 1:] - target[:, :-1]
    edge = torch.nn.functional.smooth_l1_loss(
        predicted_edges[adjacent_mask.expand_as(predicted_edges)],
        target_edges[adjacent_mask.expand_as(target_edges)],
    )
    return coordinate + 0.35 * edge, coordinate, edge


def evaluate(model, device: str, *, samples: int, seed: int, maximum_nodes: int) -> dict:
    import torch

    rng = np.random.default_rng(seed)
    before_intersections = 0
    after_intersections = 0
    eligible = 0
    resolved = 0
    squared_error = []
    model.eval()
    for _ in range(samples):
        clean = generate_clean_loop(rng, 16, min(maximum_nodes, 128))
        pair = corrupt_loop(clean, rng)
        count = len(clean)
        features = np.zeros((1, maximum_nodes, 10), dtype=np.float32)
        valid = np.zeros((1, maximum_nodes), dtype=bool)
        features[0, :count] = loop_features(pair.corrupted)
        valid[0, :count] = True
        with torch.inference_mode():
            predicted = model(torch.from_numpy(features).to(device), torch.from_numpy(valid).to(device))[0, :count].float().cpu().numpy()
        before = strict_self_intersections(pair.corrupted)
        after = strict_self_intersections(predicted)
        before_intersections += before
        after_intersections += after
        if before:
            eligible += 1
            resolved += int(after == 0)
        squared_error.append(float(np.mean((predicted - pair.clean) ** 2)))
    return {
        "samples": samples,
        "corrupted_samples_with_intersections": eligible,
        "fully_resolved_intersection_rate": resolved / eligible if eligible else 0.0,
        "intersection_count_before": before_intersections,
        "intersection_count_after": after_intersections,
        "intersection_reduction": 1.0 - after_intersections / before_intersections if before_intersections else 0.0,
        "coordinate_rmse": float(np.sqrt(np.mean(squared_error))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train lightweight PatternRepairNet on synthetic topology corruptions")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=ROOT / "checkpoints" / "pattern_repair" / "pattern_repair_net.pt")
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--clean-pattern-root", type=Path, default=ROOT / "artifacts" / "pattern_pipeline")
    parser.add_argument("--validation-samples", type=int, default=256)
    parser.add_argument("--clean-pool-probability", type=float, default=0.3)
    parser.add_argument("--trust-canonical-pool", action="store_true", help="Skip repeat validation for an accepted-only canonical pool.")
    args = parser.parse_args()

    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    configured = json.loads(args.config.read_text(encoding="utf-8"))
    config = {**DEFAULT_MODEL_CONFIG, **configured.get("model", {})}
    model = build_model(config).to(args.device)
    model.repair_config = config
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=args.device.startswith("cuda"))
    rng = np.random.default_rng(args.seed)
    clean_pool = discover_clean_loops(
        args.clean_pattern_root,
        int(config["maximum_nodes"]),
        trusted_canonical_pool=args.trust_canonical_pool,
    )
    started = time.perf_counter()
    running = []
    model.train()
    for step in range(1, args.steps + 1):
        features, target, mask = synthetic_batch(
            rng,
            args.batch_size,
            maximum_nodes=int(config["maximum_nodes"]),
            minimum_nodes=16,
            clean_pool=clean_pool,
            clean_pool_probability=args.clean_pool_probability,
        )
        feature_tensor = torch.from_numpy(features).to(args.device)
        target_tensor = torch.from_numpy(target).to(args.device)
        mask_tensor = torch.from_numpy(mask).to(args.device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
            predicted = model(feature_tensor, mask_tensor)
            loss, coordinate_loss, edge_loss = repair_loss(predicted, target_tensor, mask_tensor)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        running.append(float(loss.detach().cpu()))
        if step == 1 or step % 100 == 0:
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": float(loss.detach().cpu()),
                        "coordinate_loss": float(coordinate_loss.detach().cpu()),
                        "edge_loss": float(edge_loss.detach().cpu()),
                        "mean_loss_100": float(np.mean(running[-100:])),
                    }
                ),
                flush=True,
            )
    metrics = evaluate(
        model,
        args.device,
        samples=args.validation_samples,
        seed=args.seed + 1,
        maximum_nodes=int(config["maximum_nodes"]),
    )
    metrics.update(
        {
            "training_steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "training_seconds": time.perf_counter() - started,
            "clean_canonical_panel_count": len(clean_pool),
            "clean_pool_probability": args.clean_pool_probability,
            "clean_pattern_source": "official_canonical_pool_plus_synthetic_corruptions" if clean_pool else "synthetic_only",
            "synthetic_corruption_training": True,
            "template_retrieval": False,
            "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        }
    )
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "model_config": config, "metrics": metrics}, args.output)
    metrics_path = args.output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "checkpoint": str(args.output), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
