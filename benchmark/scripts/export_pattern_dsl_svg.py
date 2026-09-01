"""Compile a Pattern DSL program or DSL-backed PatternDocument to SVG."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from benchmark.gcdv2_exact.pattern_dsl import parse_pattern_dsl
from benchmark.gcdv2_exact.pattern_dsl_svg import SvgExportOptions, write_pattern_svg
from benchmark.pattern_pipeline.schema import PatternDocument


def _load_source(path: Path, input_format: str):
    resolved = input_format
    if resolved == "auto":
        resolved = "dsl" if path.suffix.lower() in {".dsl", ".pattern"} else "json"
    if resolved == "dsl":
        return parse_pattern_dsl(path.read_text(encoding="utf-8"))
    return PatternDocument.read_json(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compile verified Pattern DSL M/L/Q/C/A/Z geometry to deterministic SVG."
        )
    )
    parser.add_argument("input", type=Path, help="Pattern DSL or DSL-backed PatternDocument JSON")
    parser.add_argument("output", type=Path, help="Destination .svg")
    parser.add_argument("--input-format", choices=("auto", "dsl", "json"), default="auto")
    parser.add_argument("--overlays", action="store_true", help="Draw semantic/proof overlays")
    parser.add_argument(
        "--semantic-metadata",
        action="store_true",
        help="Embed ROLE/NEXT/SHARED_ENDPOINT/SEWN_TO/LANDMARK proof facts",
    )
    parser.add_argument(
        "--provenance-debug",
        action="store_true",
        help="Embed source identifiers, weak roles, and the complete source DSL",
    )
    parser.add_argument("--no-metadata", action="store_true", help="Omit machine-readable metadata")
    parser.add_argument("--gap-cm", type=float, default=3.0)
    parser.add_argument("--padding-cm", type=float, default=1.5)
    parser.add_argument("--max-columns", type=int, default=4)
    parser.add_argument("--decimals", type=int, default=6)
    args = parser.parse_args()
    source = _load_source(args.input, args.input_format)
    options = SvgExportOptions(
        gap_cm=args.gap_cm,
        padding_cm=args.padding_cm,
        max_columns=args.max_columns,
        decimals=args.decimals,
        include_overlays=args.overlays,
        include_metadata=not args.no_metadata,
        include_semantic_facts=args.semantic_metadata,
        include_provenance=args.provenance_debug,
    )
    destination = write_pattern_svg(source, args.output, options=options)
    print(
        json.dumps(
            {
                "input": args.input.as_posix(),
                "output": destination.as_posix(),
                "sha256": _sha256(destination),
                "semantic_overlays": options.include_overlays,
                "semantic_metadata": options.include_semantic_facts,
                "provenance_debug": options.include_provenance,
                "metadata": options.include_metadata,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
