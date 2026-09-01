from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark.pattern_pipeline.export import export_bundle
from benchmark.pattern_pipeline.schema import PatternDocument
from benchmark.pattern_repair.application import repair_document
from benchmark.pattern_repair.checkpoint import load_repair_model


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply learned topology-preserving repair to one canonical generated pattern")
    parser.add_argument("pattern", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "pattern_repair" / "pattern_repair_net.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--strengths", nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--maximum-passes", type=int, default=3)
    args = parser.parse_args()
    if any(not 0.0 < strength <= 1.0 for strength in args.strengths):
        raise ValueError("repair strengths must be in (0, 1]")
    model, checkpoint = load_repair_model(args.checkpoint.resolve(), args.device)
    document = PatternDocument.read_json(args.pattern.resolve())
    repaired, receipt = repair_document(
        model,
        document,
        args.device,
        tuple(args.strengths),
        maximum_passes=args.maximum_passes,
    )
    args.output = args.output.resolve()
    paths = export_bundle(repaired, args.output)
    receipt.update(
        {
            "checkpoint_metrics": checkpoint.get("metrics", {}),
            "artifacts": {name: path.name for name, path in paths.items()},
        }
    )
    (args.output / "repair_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
