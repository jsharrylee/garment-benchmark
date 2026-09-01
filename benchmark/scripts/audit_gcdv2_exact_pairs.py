from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_curve(edge: dict[str, Any]) -> tuple[str, Any]:
    curvature = edge.get("curvature")
    if not curvature:
        return "line", None
    if isinstance(curvature, list):
        return "quadratic", curvature
    return str(curvature.get("type", "line")), curvature.get("params", [])


def _maximum_abs(first: Any, second: Any) -> float:
    if first is None or second is None:
        return 0.0 if first is second else math.inf
    if isinstance(first, bool) or isinstance(second, bool):
        return 0.0 if bool(first) == bool(second) else math.inf
    if isinstance(first, (int, float)) and isinstance(second, (int, float)):
        return abs(float(first) - float(second))
    if isinstance(first, (list, tuple)) and isinstance(second, (list, tuple)):
        if len(first) != len(second):
            return math.inf
        return max((_maximum_abs(a, b) for a, b in zip(first, second)), default=0.0)
    return 0.0 if first == second else math.inf


def main() -> None:
    parser = argparse.ArgumentParser(description="Independently audit exact GCDv2 label round trips and paired views.")
    parser.add_argument("--index", type=Path, default=Path("artifacts/gcdv2_exact_pairs_v1/index.jsonl"))
    parser.add_argument("--source", type=Path, default=Path("data/processed/garmentcode_v2/batch_0_full"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/gcdv2_exact_pairs_v1/audit.json"))
    parser.add_argument("--allow-missing-views", action="store_true")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.index.read_text(encoding="utf-8").splitlines() if line.strip()]
    failures: list[dict[str, Any]] = []
    counts = Counter()
    maximum_vertex_error = 0.0
    maximum_curve_parameter_error = 0.0
    for number, row in enumerate(rows, start=1):
        sample_id = row["sample_id"]
        label_path = Path(row["label_path"])
        label = json.loads(label_path.read_text(encoding="utf-8"))
        source_path = args.source / sample_id / f"{sample_id}_specification.json"
        source = json.loads(source_path.read_text(encoding="utf-8"))["pattern"]
        local_failures: list[str] = []
        if label["source_specification_sha256"] != _sha256(source_path):
            local_failures.append("source_sha256")
        panel_order = [str(value) for value in source.get("panel_order", tuple(source["panels"]))]
        if panel_order != [panel["panel_id"] for panel in label["panels"]]:
            local_failures.append("panel_order")
        if len(source.get("stitches", [])) != len(label.get("stitches", [])):
            local_failures.append("stitch_count")
        for panel in label["panels"]:
            raw = source["panels"][panel["panel_id"]]
            vertex_error = _maximum_abs(raw["vertices"], panel["vertices_cm"])
            maximum_vertex_error = max(maximum_vertex_error, vertex_error)
            if vertex_error > 1e-9:
                local_failures.append(f"vertices:{panel['panel_id']}")
            if len(raw["edges"]) != len(panel["edges"]):
                local_failures.append(f"edge_count:{panel['panel_id']}")
                continue
            for source_edge, exact_edge in zip(raw["edges"], panel["edges"]):
                if list(source_edge["endpoints"]) != list(exact_edge["endpoints"]):
                    local_failures.append(f"endpoints:{exact_edge['edge_id']}")
                source_type, source_params = _source_curve(source_edge)
                if source_type != exact_edge["curve"]["source_type"]:
                    local_failures.append(f"curve_type:{exact_edge['edge_id']}")
                curve_error = _maximum_abs(source_params, exact_edge["curve"]["source_params"])
                maximum_curve_parameter_error = max(maximum_curve_parameter_error, curve_error)
                if curve_error > 1e-9:
                    local_failures.append(f"curve_params:{exact_edge['edge_id']}")
                counts[exact_edge["curve"]["type"]] += 1
        pattern_path = Path(label["pattern_image"])
        if not pattern_path.is_file():
            local_failures.append("pattern_missing")
        else:
            with Image.open(pattern_path) as image:
                if image.size != (1024, 1024):
                    local_failures.append(f"pattern_size:{image.size}")
        missing = [view["path"] for view in label["views"] if not Path(view["path"]).is_file()]
        if missing and not args.allow_missing_views:
            local_failures.append(f"missing_views:{len(missing)}")
        for view in label["views"]:
            path = Path(view["path"])
            if path.is_file() and view.get("sha256") != _sha256(path):
                local_failures.append(f"view_sha256:{view['view_label']}")
        if local_failures:
            failures.append({"sample_id": sample_id, "failures": local_failures})
        if number == 1 or number % 250 == 0 or number == len(rows):
            print(json.dumps({"audited": number, "total": len(rows), "failed": len(failures)}), flush=True)

    payload = {
        "schema_version": "gcdv2-exact-audit-1.0",
        "index": args.index.as_posix(),
        "record_count": len(rows),
        "pass_count": len(rows) - len(failures),
        "failure_count": len(failures),
        "maximum_source_vertex_roundtrip_error_cm": maximum_vertex_error,
        "maximum_source_curve_parameter_roundtrip_error": maximum_curve_parameter_error,
        "curve_type_counts": dict(sorted(counts.items())),
        "view_policy": "missing_allowed" if args.allow_missing_views else "four_required",
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
