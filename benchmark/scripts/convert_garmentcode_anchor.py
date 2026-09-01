from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark.pattern_pipeline.validation import validate_pattern
from benchmark.retrieval.garmentcode import convert_garmentcode_specification


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert an official GarmentCode anchor to the canonical stitch-aware schema.")
    parser.add_argument("specification", type=Path)
    parser.add_argument("--anchor-id")
    parser.add_argument("--panel-mesh-spacing-cm", type=float, default=0.0)
    parser.add_argument("--source-license", default="MIT")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    document = convert_garmentcode_specification(
        args.specification,
        anchor_id=args.anchor_id,
        panel_mesh_spacing_cm=args.panel_mesh_spacing_cm,
        source_license=args.source_license,
    )
    report = validate_pattern(document)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    document.write_json(args.output)
    receipt = {
        "pattern_id": document.pattern_id,
        "mode": "retrieval_anchored_v2",
        "template_retrieval": True,
        "panel_mesh_spacing_cm": args.panel_mesh_spacing_cm,
        "structural_validation": report.to_dict(),
        "simulation_status": "READY" if report.accepted else "BLOCKED_STRUCTURAL_VALIDATION",
    }
    receipt_path = args.output.with_name(args.output.stem + "_conversion_receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
