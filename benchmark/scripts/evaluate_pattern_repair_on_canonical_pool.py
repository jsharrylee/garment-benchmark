from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmark.pattern_repair.data import corrupt_loop, loop_features, strict_self_intersections
from benchmark.pattern_repair.model import build_model
from benchmark.scripts.train_pattern_repair_model import discover_clean_loops


def evaluate_checkpoint(checkpoint: Path, clean_pool: tuple[np.ndarray, ...], device: str, seed: int) -> dict:
    import torch

    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    config = payload["model_config"]
    model = build_model(config).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    rng = np.random.default_rng(seed)
    before_total = 0
    after_total = 0
    eligible = 0
    resolved = 0
    squared_error = []
    maximum_nodes = int(config["maximum_nodes"])
    for clean in clean_pool:
        pair = corrupt_loop(clean, rng)
        count = len(clean)
        features = np.zeros((1, maximum_nodes, 10), dtype=np.float32)
        mask = np.zeros((1, maximum_nodes), dtype=bool)
        features[0, :count] = loop_features(pair.corrupted)
        mask[0, :count] = True
        with torch.inference_mode():
            predicted = model(torch.from_numpy(features).to(device), torch.from_numpy(mask).to(device))[0, :count].float().cpu().numpy()
        before = strict_self_intersections(pair.corrupted)
        after = strict_self_intersections(predicted)
        before_total += before
        after_total += after
        if before:
            eligible += 1
            resolved += int(after == 0)
        squared_error.append(float(np.mean((predicted - clean) ** 2)))
    return {
        "checkpoint": checkpoint.name,
        "official_clean_panel_count": len(clean_pool),
        "corrupted_panels_with_intersections": eligible,
        "fully_resolved_intersection_rate": resolved / eligible if eligible else 0.0,
        "intersection_count_before": before_total,
        "intersection_count_after": after_total,
        "intersection_reduction": 1.0 - after_total / before_total if before_total else 0.0,
        "coordinate_rmse": float(np.sqrt(np.mean(squared_error))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare repair checkpoints on corrupted official canonical garment panels.")
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--clean-pattern-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trust-canonical-pool", action="store_true")
    parser.add_argument("--maximum-panels", type=int, default=0, help="Deterministically subsample a large official pool for bounded evaluation.")
    args = parser.parse_args()
    clean_pool = discover_clean_loops(args.clean_pattern_root, 256, trusted_canonical_pool=args.trust_canonical_pool)
    if not clean_pool:
        raise SystemExit("no valid canonical panel loops found")
    full_pool_count = len(clean_pool)
    if args.maximum_panels and len(clean_pool) > args.maximum_panels:
        indices = np.linspace(0, len(clean_pool) - 1, args.maximum_panels, dtype=int)
        clean_pool = tuple(clean_pool[int(index)] for index in indices)
    rows = [evaluate_checkpoint(path, clean_pool, args.device, args.seed) for path in args.checkpoints]
    payload = {
        "evaluation": "official_garmentcode_v2_panel_corruption",
        "full_eligible_clean_panel_count": full_pool_count,
        "evaluated_panel_count": len(clean_pool),
        "deterministic_even_subsample": len(clean_pool) != full_pool_count,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
