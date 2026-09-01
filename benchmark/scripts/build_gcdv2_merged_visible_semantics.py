from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmark.drafting_semantics.merged_visible_learning import MERGED_EDGE_ROLES, build_merged_visible_arrays


def main() -> None:
    parser = argparse.ArgumentParser(description="Build merged-edge semantics on learned predicted contours.")
    parser.add_argument("--panel-index", type=Path, default=Path("artifacts/gcdv2_neurosymbolic_v1/panel_index.jsonl"))
    parser.add_argument("--predicted-contours", type=Path, default=Path("artifacts/gcdv2_predicted_contours_v1/predicted_contours.npz"))
    parser.add_argument("--records", type=Path, default=Path("artifacts/drafting_semantics/gcdv2_batch0/records.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/gcdv2_merged_visible_semantics_v1/merged_semantics.npz"))
    parser.add_argument("--metadata", type=Path, default=Path("artifacts/gcdv2_merged_visible_semantics_v1/metadata.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/gcdv2_merged_visible_semantics_v1.json"))
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.panel_index.read_text(encoding="utf-8").splitlines() if line.strip()]
    with np.load(args.predicted_contours) as predicted:
        if not predicted["valid"].all():
            raise SystemExit("predicted contour artifact contains invalid panels")
        arrays, metadata = build_merged_visible_arrays(rows, predicted["contours"], args.records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    args.metadata.write_text("".join(json.dumps(row) + "\n" for row in metadata), encoding="utf-8")
    split_names = {0: "train", 1: "validation", 2: "test"}
    manifest = {
        "schema_version": "gcdv2-merged-visible-semantics-1.0", "status": "PASS", "panel_count": len(metadata),
        "split_counts": {name: int((arrays["splits"] == code).sum()) for code, name in split_names.items()},
        "edge_count": int(arrays["valid_edges"].sum()),
        "role_counts": {role: int((arrays["edge_roles"][arrays["valid_edges"]] == index).sum()) for index, role in enumerate(MERGED_EDGE_ROLES)},
        "landmark_counts": {name: int(arrays["landmark_mask"][:, index].sum()) for index, name in enumerate(("FNP", "BNP", "SNP", "SP"))},
        "input_contract": "learned-mask predicted contour -> merged visible segments -> unit-chord intrinsic features",
        "panel_role_conditioning": "source front/back bodice role is supplied; not inferred in this phase",
        "artifact": args.output.as_posix(),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
