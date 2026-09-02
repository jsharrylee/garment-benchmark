from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmark.gcdv2_exact.intrinsic_graph_learning import build_intrinsic_arrays


def main() -> None:
    parser = argparse.ArgumentParser(description="Build translation/rotation/scale-invariant visible graph supervision.")
    parser.add_argument("--index", type=Path, default=Path("artifacts/gcdv2_neurosymbolic_v1/panel_index.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/gcdv2_intrinsic_graph_v1/intrinsic_graph.npz"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/gcdv2_intrinsic_graph_v1.json"))
    parser.add_argument(
        "--predicted-contours",
        type=Path,
        help="Optional NPZ containing image-model contour predictions in panel-index order.",
    )
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.index.read_text(encoding="utf-8").splitlines() if line.strip()]
    contours_override = None
    if args.predicted_contours is not None:
        contour_data = np.load(args.predicted_contours)
        contours_override = contour_data["contours"]
        if len(contours_override) != len(rows):
            raise ValueError(
                f"predicted contour count {len(contours_override)} does not match panel index {len(rows)}"
            )
    arrays = build_intrinsic_arrays(rows, contours_override=contours_override)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    split_names = {0: "train", 1: "validation", 2: "test"}
    manifest = {
        "schema_version": "gcdv2-intrinsic-visible-graph-1.0",
        "status": "PASS",
        "panel_count": len(arrays["contours"]),
        "segment_count": len(arrays["segment_features"]),
        "panel_split_counts": {split_names[value]: int((arrays["panel_splits"] == value).sum()) for value in split_names},
        "segment_split_counts": {split_names[value]: int((arrays["segment_splits"] == value).sum()) for value in split_names},
        "primitive_counts": {name: int((arrays["segment_primitives"] == index).sum()) for index, name in enumerate(("line", "quadratic_bezier", "cubic_bezier", "circular_arc"))},
        "input_contract": "cyclic multi-scale length/turn/chord ratios only; no absolute x/y supplied to the corner model",
        "contour_source": "learned_image_mask_sdf" if contours_override is not None else "source_vector_ground_truth",
        "predicted_contour_artifact": args.predicted_contours.as_posix() if args.predicted_contours else None,
        "segment_contract": "unit chord-frame samples, tangent/curvature/step features, relative cubic controls and ratios",
        "artifact": args.output.as_posix(),
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
