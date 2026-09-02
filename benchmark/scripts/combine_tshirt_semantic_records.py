from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def _lines(path: Path):
    if path.suffix.lower() == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            yield from stream
    else:
        with path.open("r", encoding="utf-8") as stream:
            yield from stream


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def training_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    """Drop unscored provenance payloads while preserving every model target.

    Raw creation traces and FreeSewing production semantics remain in their
    source artifacts.  This projection is an ignored, disposable input for the
    geometry baseline; it is not a replacement archival format.
    """

    panels = []
    for panel in value.get("panels", ()):
        points = [
            {
                key: point[key]
                for key in (
                    "id",
                    "panel_id",
                    "xy_cm",
                    "formula",
                    "canonical_name",
                    "source_name",
                    "domain",
                    "evidence",
                    "training_eligible",
                    "confidence",
                )
                if key in point
            }
            for point in panel.get("points", ())
        ]
        edges = [
            {
                key: edge[key]
                for key in (
                    "id",
                    "panel_id",
                    "start_point_id",
                    "end_point_id",
                    "semantic_role",
                    "geometry",
                    "domain",
                    "evidence",
                    "training_eligible",
                    "confidence",
                )
                if key in edge
            }
            for edge in panel.get("edges", ())
        ]
        panels.append(
            {
                "id": panel["id"],
                "semantic_role": panel["semantic_role"],
                "points": points,
                "edges": edges,
                "metadata": {"training_projection": True},
            }
        )
    darts = [
        {
            key: dart[key]
            for key in (
                "id",
                "panel_id",
                "kind",
                "applicable",
                "applicability_reason",
                "domain",
                "evidence",
                "training_eligible",
                "confidence",
            )
            if key in dart
        }
        for dart in value.get("darts", ())
    ]
    return {
        "schema_version": value.get("schema_version", "tshirt-construction-trace-1.0"),
        "sample_id": value["sample_id"],
        "split": value["split"],
        "source": value["source"],
        "body": value["body"],
        "design": value.get("design", {}),
        "provenance": {
            "training_projection": True,
            "excluded_domains": [
                "construction_operation_payloads",
                "reference_lines",
                "named_paths",
                "notches",
                "grainlines",
                "seam_allowances",
            ],
        },
        "panels": panels,
        "operations": [
            {
                "id": "training_projection",
                "order": 0,
                "operation": "project_geometry_targets_for_training",
                "status": "completed",
                "training_eligible": False,
            }
        ],
        "darts": darts,
        "metadata": {
            "dart_applicability": value.get("metadata", {}).get("dart_applicability", "UNKNOWN"),
            "training_projection": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine traced T-shirt sources without changing records.")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--training-projection",
        action="store_true",
        help="Keep model geometry/targets only; source trace artifacts remain unchanged.",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    total = 0
    with gzip.open(args.output, "wt", encoding="utf-8", newline="\n", compresslevel=6) as output:
        for path in args.inputs:
            count = 0
            for line in _lines(path):
                if not line.strip():
                    continue
                # Parse once so malformed or non-JSONL input cannot silently
                # enter a frozen evaluation bundle.
                value = json.loads(line)
                if args.training_projection:
                    output.write(
                        json.dumps(
                            training_projection(value),
                            sort_keys=True,
                            ensure_ascii=False,
                            allow_nan=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                else:
                    output.write(line.rstrip("\r\n") + "\n")
                count += 1
                total += 1
            counts[path.name] = count
    manifest = {
        "schema_version": "combined-tshirt-semantics-1.0",
        "record_count": total,
        "input_record_counts": counts,
        "artifact_sha256": _sha256(args.output),
        "artifact_bytes": args.output.stat().st_size,
        "mutation_policy": (
            "model geometry and targets projected; raw source artifacts unchanged"
            if args.training_projection
            else "records concatenated verbatim after JSON validation"
        ),
        "training_projection": bool(args.training_projection),
    }
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
