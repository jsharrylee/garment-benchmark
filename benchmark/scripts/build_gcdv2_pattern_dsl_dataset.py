from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmark.drafting_semantics.schema import EDGE_ROLES, PANEL_ROLES
from benchmark.gcdv2_exact.pattern_dsl_learning import (
    CATEGORIES,
    CURVE_COMMANDS,
    EDGE_FEATURE_SCHEMA_DSL_CHORD_TURN_V2,
    EDGE_FEATURE_SCHEMA_TANGENT_GAP_V1,
    build_program_arrays,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile exact GCDv2 geometry facts and proposed Pattern-DSL facts.")
    parser.add_argument("--garment-index", type=Path, default=Path("artifacts/gcdv2_neurosymbolic_v1/garment_index.jsonl"))
    parser.add_argument("--records", type=Path, default=Path("artifacts/drafting_semantics/gcdv2_multigarment/records.jsonl"))
    parser.add_argument(
        "--feature-schema",
        choices=("tangent-gap-v1", "dsl-chord-turn-v2"),
        default="tangent-gap-v1",
        help="Keep v1 checkpoints reproducible or build the canonical PatternProgram v2 tensor corpus.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    schema_lookup = {
        "tangent-gap-v1": EDGE_FEATURE_SCHEMA_TANGENT_GAP_V1,
        "dsl-chord-turn-v2": EDGE_FEATURE_SCHEMA_DSL_CHORD_TURN_V2,
    }
    feature_schema = schema_lookup[args.feature_schema]
    version = "v1" if args.feature_schema == "tangent-gap-v1" else "v2"
    args.output = args.output or Path(f"artifacts/gcdv2_pattern_dsl_{version}/programs.npz")
    args.metadata = args.metadata or Path(f"artifacts/gcdv2_pattern_dsl_{version}/metadata.jsonl")
    args.manifest = args.manifest or Path(f"data/manifests/gcdv2_pattern_dsl_{version}.json")

    rows = [json.loads(line) for line in args.garment_index.read_text(encoding="utf-8").splitlines() if line.strip()]
    arrays, metadata = build_program_arrays(
        rows,
        args.records,
        feature_schema=feature_schema,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(
        "".join(json.dumps(row) + "\n" for row in metadata),
        encoding="utf-8",
        newline="\n",
    )
    split_names = {0: "train", 1: "validation", 2: "test"}
    manifest = {
        "schema_version": f"gcdv2-pattern-dsl-{version[1:]}.0",
        "status": "PASS",
        "representation": "exact M/L/Q/C/A/Z-style geometry problem facts plus masked semantic/seam proposition targets",
        "edge_feature_schema": feature_schema,
        "checkpoint_compatibility": (
            "existing unified_transformer.pt"
            if args.feature_schema == "tangent-gap-v1"
            else "requires regenerated corpus and newly trained v2 checkpoint"
        ),
        "neural_input_excludes": ["raster image", "absolute x/y", "source panel id", "source edge id", "semantic role", "stitch target"],
        "garment_count": len(rows),
        "panel_count": int(arrays["panel_valid"].sum()),
        "edge_count": int(arrays["edge_valid"].sum()),
        "stitch_count": int(arrays["stitch_valid"].sum()),
        "semantic_panel_count": int((arrays["panel_roles"] >= 0).sum()),
        "semantic_edge_count": int((arrays["edge_roles"] >= 0).sum()),
        "landmark_count": int((arrays["landmarks"] >= 0).sum()),
        "split_counts": {name: int((arrays["splits"] == code).sum()) for code, name in split_names.items()},
        "vocabulary": {
            "curve_commands": CURVE_COMMANDS,
            "garment_roles": CATEGORIES,
            "panel_roles": PANEL_ROLES,
            "edge_roles": EDGE_ROLES,
            "relations": ["MEMBER", "NEXT", "SHARED_ENDPOINT", "SEWN_TO", "LANDMARK"],
        },
        "split_authority": "gcdv2_neurosymbolic garment-ID-disjoint split; semantic-record split ignored",
        "artifacts": {"arrays": args.output.as_posix(), "metadata": args.metadata.as_posix()},
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
