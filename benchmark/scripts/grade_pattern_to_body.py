from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from benchmark.pattern_pipeline.grading import grade_pattern
from benchmark.pattern_pipeline.schema import PatternDocument
from benchmark.pattern_pipeline.validation import validate_pattern


def _body_measurements(path: Path) -> dict[str, float]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    value = value.get("body", value)
    return {str(key): float(item) for key, item in value.items() if isinstance(item, (int, float))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade a retrieved GarmentCode pattern to target body measurements.")
    parser.add_argument("pattern", type=Path)
    parser.add_argument("--category", required=True, choices=["top", "pants", "shorts", "skirt", "dress", "jumpsuit"])
    parser.add_argument("--source-measurements", type=Path, required=True)
    parser.add_argument("--target-measurements", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panel-mesh-spacing-cm", type=float, default=2.5)
    args = parser.parse_args()

    source = _body_measurements(args.source_measurements)
    target = _body_measurements(args.target_measurements)
    graded = grade_pattern(
        PatternDocument.read_json(args.pattern),
        category=args.category,
        source_measurements=source,
        target_measurements=target,
        panel_mesh_spacing_cm=args.panel_mesh_spacing_cm,
    )
    validation = validate_pattern(graded)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    graded.write_json(args.output)
    print(json.dumps({"pattern": graded.pattern_id, "validation": validation.to_dict(), "grading": graded.annotations["body_grading"]}, indent=2))


if __name__ == "__main__":
    main()
