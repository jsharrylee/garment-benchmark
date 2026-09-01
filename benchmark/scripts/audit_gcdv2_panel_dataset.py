from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit exact one-panel GCDv2 image/vector pairs.")
    parser.add_argument("--index", type=Path, default=Path("artifacts/gcdv2_exact_panels_v1/index.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/gcdv2_exact_panels_v1/audit.json"))
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.index.read_text(encoding="utf-8").splitlines() if line]
    failures: list[dict[str, str]] = []
    categories: Counter[str] = Counter()
    curve_types: Counter[str] = Counter()
    vertices = edges = 0
    for index, row in enumerate(rows, 1):
        try:
            target = json.loads(Path(row["target_path"]).read_text(encoding="utf-8"))
            geometry = target["geometry"]
            target_vertices = geometry["vertices"]
            target_edges = geometry["edges"]
            count = len(target_vertices)
            if count < 3 or len(target_edges) != count:
                raise ValueError("not one closed vertex/edge cycle")
            if geometry["boundary_sequence"] != list(range(count)):
                raise ValueError("boundary sequence is not canonical")
            for edge_index, edge in enumerate(target_edges):
                if edge["start_vertex_index"] != edge_index:
                    raise ValueError(f"edge {edge_index} start incidence mismatch")
                if edge["end_vertex_index"] != (edge_index + 1) % count:
                    raise ValueError(f"edge {edge_index} end incidence mismatch")
                if not math.isfinite(float(edge["length_cm"])) or float(edge["length_cm"]) <= 0:
                    raise ValueError(f"edge {edge_index} invalid length")
                curve_types[str(edge["curve_type"])] += 1
            contract = target["input_contract"]
            scale = float(contract["normalized_panel_image"]["pixels_per_cm"])
            origin = contract["origin_px"]
            for vertex in target_vertices:
                x, y = vertex["centered_xy_cm"]
                expected = (origin[0] + x * scale, origin[1] - y * scale)
                actual = vertex["image_xy_px"]
                if max(abs(expected[0] - actual[0]), abs(expected[1] - actual[1])) > 1e-7:
                    raise ValueError("image/metric coordinate mismatch")
            for key in ("panel_image_path", "metric_panel_image_path"):
                path = Path(row[key])
                if not path.is_file():
                    raise FileNotFoundError(path)
                with Image.open(path) as image:
                    if image.size != tuple(contract["canvas_size_px"]):
                        raise ValueError(f"{key} has size {image.size}")
            if target["panel_uid"] != row["panel_uid"]:
                raise ValueError("index/target UID mismatch")
            categories[str(row["garment_category"])] += 1
            vertices += count
            edges += len(target_edges)
        except Exception as error:
            failures.append({"panel_uid": str(row.get("panel_uid")), "error": f"{type(error).__name__}: {error}"})
        if index % 2000 == 0 or index == len(rows):
            print(json.dumps({"audited": index, "total": len(rows), "failures": len(failures)}), flush=True)
    result = {
        "status": "PASS" if not failures else "FAIL",
        "panel_count": len(rows),
        "category_panel_counts": dict(sorted(categories.items())),
        "vertex_count": vertices,
        "edge_count": edges,
        "curve_type_counts": dict(sorted(curve_types.items())),
        "single_closed_cycle_count": len(rows) - len(failures),
        "image_target_pair_count": len(rows) - len(failures),
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
