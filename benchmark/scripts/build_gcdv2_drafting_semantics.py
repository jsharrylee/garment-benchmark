from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import yaml

from benchmark.drafting_semantics.garmentcode import annotate_garmentcode_sample


def _sample_id(path: Path) -> str:
    return path.stem.removesuffix("_specification")


def _official_splits(path: Path) -> dict[str, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    output = {}
    for split, values in raw.items():
        for value in values:
            output[Path(value).name] = split
    return output


def _design_scope(path: Path) -> tuple[str | None, bool]:
    design = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("design", {})
    upper = design.get("meta", {}).get("upper", {}).get("v")
    asymmetric = bool(design.get("left", {}).get("enable_asym", {}).get("v", False))
    return upper, asymmetric


def main() -> None:
    parser = argparse.ArgumentParser(description="Build evidence-graded drafting semantics from extracted GarmentCodeData v2.")
    parser.add_argument("--input", type=Path, default=Path("data/processed/garmentcode_v2/batch_0_full"))
    parser.add_argument("--official-split", type=Path, default=Path("data/raw/garmentcode_v2/metadata/official_split.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/drafting_semantics/gcdv2_batch0/records.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/gcdv2_drafting_semantics.json"))
    parser.add_argument("--include-asymmetric", action="store_true")
    parser.add_argument("--include-production-synthetic", action="store_true")
    parser.add_argument(
        "--scope",
        choices=("symmetric-bodice", "all-garments"),
        default="symmetric-bodice",
        help="Preserve the historical bodice lane by default; all-garments also keeps lower-only designs.",
    )
    parser.add_argument(
        "--sample-index",
        type=Path,
        help="Optional JSON index whose records define the exact accepted sample-id set.",
    )
    parser.add_argument("--max-records", type=int, default=0)
    args = parser.parse_args()

    split_lookup = _official_splits(args.official_split)
    specifications = sorted(args.input.rglob("*_specification.json"))
    if not specifications:
        raise SystemExit(f"no GarmentCode specifications found below {args.input}")
    accepted_ids: set[str] | None = None
    category_by_id: dict[str, str] = {}
    if args.sample_index is not None:
        index_payload = json.loads(args.sample_index.read_text(encoding="utf-8"))
        index_records = index_payload.get("records", ())
        accepted_ids = {str(row["sample_id"]) for row in index_records}
        category_by_id = {str(row["sample_id"]): str(row.get("category", "unknown")) for row in index_records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = 0
    split_counts: Counter[str] = Counter()
    upper_counts: Counter[str] = Counter()
    edge_roles: Counter[str] = Counter()
    edge_evidence: Counter[str] = Counter()
    panel_roles: Counter[str] = Counter()
    landmark_counts: Counter[str] = Counter()
    landmark_evidence: Counter[str] = Counter()
    reference_line_evidence: Counter[str] = Counter()
    dart_counts: Counter[str] = Counter()
    dart_evidence: Counter[str] = Counter()
    construction_evidence: Counter[str] = Counter()
    body_hashes: set[str] = set()
    rejected_scope: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    with args.output.open("w", encoding="utf-8") as stream:
        for specification in specifications:
            sample_id = _sample_id(specification)
            if accepted_ids is not None and sample_id not in accepted_ids:
                rejected_scope["outside_sample_index"] += 1
                continue
            design = specification.with_name(f"{sample_id}_design_params.yaml")
            body = specification.with_name(f"{sample_id}_body_measurements.yaml")
            if not design.is_file() or not body.is_file():
                rejected_scope["missing_sidecar"] += 1
                continue
            upper, asymmetric = _design_scope(design)
            if args.scope == "symmetric-bodice" and upper not in {"FittedShirt", "Shirt"}:
                rejected_scope["no_supported_upper"] += 1
                continue
            if args.scope == "symmetric-bodice" and asymmetric and not args.include_asymmetric:
                rejected_scope["asymmetric_excluded"] += 1
                continue
            record = annotate_garmentcode_sample(
                specification,
                body,
                design,
                split=split_lookup.get(sample_id, "not_in_official_paired_split"),
                synthesize_production_marks=args.include_production_synthetic,
            )
            if args.scope == "symmetric-bodice" and (
                not any(panel.role == "front_bodice" for panel in record.panels)
                or not any(panel.role == "back_bodice" for panel in record.panels)
            ):
                rejected_scope["missing_bodice_panels"] += 1
                continue
            stream.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
            records += 1
            split_counts[record.split] += 1
            upper_counts[str(record.program.get("upper_type"))] += 1
            category_counts[category_by_id.get(sample_id, "unknown")] += 1
            body_hashes.add(record.provenance["body_measurements_sha256"])
            for panel in record.panels:
                panel_roles[panel.role] += 1
                edge_roles.update(edge.role for edge in panel.edges)
                edge_evidence.update(edge.evidence for edge in panel.edges)
                landmark_counts.update(item.name for item in panel.landmarks)
                landmark_evidence.update(item.evidence for item in panel.landmarks)
                reference_line_evidence.update(item.evidence for item in panel.reference_lines)
            dart_counts.update(dart.kind for dart in record.darts)
            dart_evidence.update(dart.evidence for dart in record.darts)
            construction_evidence.update(step.evidence for step in record.construction_steps)
            if records % 250 == 0:
                print(json.dumps({"records": records, "last_sample": sample_id}), flush=True)
            if args.max_records and records >= args.max_records:
                break

    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    output_display = args.output.as_posix()
    manifest = {
        "schema_version": "drafting-semantic-manifest-1.0",
        "dataset": "GarmentCodeData v2",
        "doi": "https://doi.org/10.3929/ethz-b-000690432",
        "license": "CC BY 4.0",
        "scope": (
            "all_garments_v1"
            if args.scope == "all-garments"
            else ("symmetric_bodice_v1" if not args.include_asymmetric else "bodice_v1_with_asymmetry")
        ),
        "record_count": records,
        "split_counts": dict(sorted(split_counts.items())),
        "upper_type_counts": dict(sorted(upper_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "panel_role_counts": dict(sorted(panel_roles.items())),
        "edge_role_counts": dict(sorted(edge_roles.items())),
        "edge_evidence_counts": dict(sorted(edge_evidence.items())),
        "landmark_counts": dict(sorted(landmark_counts.items())),
        "landmark_evidence_counts": dict(sorted(landmark_evidence.items())),
        "reference_line_evidence_counts": dict(sorted(reference_line_evidence.items())),
        "dart_counts": dict(sorted(dart_counts.items())),
        "dart_evidence_counts": dict(sorted(dart_evidence.items())),
        "construction_evidence_counts": dict(sorted(construction_evidence.items())),
        "unique_body_measurement_hashes": len(body_hashes),
        "body_measurement_prediction": "DISABLED_CONSTANT_TARGET" if len(body_hashes) <= 1 else "ELIGIBLE_AFTER_VARIANCE_AUDIT",
        "rejected_scope_counts": dict(sorted(rejected_scope.items())),
        "records_artifact": output_display,
        "records_sha256": digest,
        "production_synthetic_marks_included": args.include_production_synthetic,
        "production_synthetic_training_eligible": False,
        "sample_index": None if args.sample_index is None else args.sample_index.as_posix(),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
