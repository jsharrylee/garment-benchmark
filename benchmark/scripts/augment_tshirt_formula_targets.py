"""Deterministically augment the immutable v1.0 T-shirt trace corpus.

The migration does not redraft garments and does not infer roles from final
panel topology.  It consumes only the pre-existing creation-event-labeled
edges, adds formula-shaped targets and an aggregate sleeve/armhole relation,
and writes a separate gzip artifact.  The original corpus is hash-checked
before and after migration and is never opened for writing.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator, TextIO

from benchmark.drafting_semantics.drafting_formula_targets import (
    build_drafting_formula_targets,
    build_sleeve_armhole_relation,
)
from benchmark.drafting_semantics.tshirt_schema import TShirtTraceRecord


MIGRATION_VERSION = "tshirt-formula-target-augmentation-1.0"
DEFAULT_INPUT = Path("artifacts/drafting_semantics/tshirt_traces.jsonl.gz")
DEFAULT_OUTPUT = Path("artifacts/drafting_semantics/tshirt_traces_formula_augmented.jsonl.gz")
DEFAULT_MANIFEST = Path("data/manifests/tshirt_construction_formula_augmented.json")

_PRESERVED_FIELDS = (
    "sample_id",
    "split",
    "source",
    "body",
    "design",
    "provenance",
    "panels",
    "operations",
    "recipe_id",
    "garment_type",
    "reference_lines",
    "darts",
    "named_paths",
    "notches",
    "grainlines",
    "seam_allowances",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} JSONL row must be an object")
            yield line_number, value


def _deterministic_gzip_text(path: Path, *, compresslevel: int = 6) -> tuple[TextIO, Any]:
    """Return a text writer and owning ExitStack-like tuple for mtime=0 gzip."""

    binary = path.open("wb")
    compressed = gzip.GzipFile(filename="", mode="wb", fileobj=binary, compresslevel=compresslevel, mtime=0)
    text = io.TextIOWrapper(compressed, encoding="utf-8", newline="\n")
    return text, (text, compressed, binary)


def _close_writer(owners: Any) -> None:
    text, compressed, binary = owners
    try:
        text.close()
    finally:
        if not compressed.closed:
            compressed.close()
        if not binary.closed:
            binary.close()


def _edge_operation_output_verified(record: TShirtTraceRecord, target: Any) -> None:
    operations = {operation.id: operation for operation in record.operations}
    edges = {edge.id: edge for panel in record.panels for edge in panel.edges}
    for edge_id in target.edge_ids:
        edge = edges[edge_id]
        if not edge.training_eligible or edge.evidence != "creation_event_binding":
            raise ValueError(
                f"{record.sample_id}:{target.id}:{edge.id} is not eligible creation-event evidence"
            )
        if edge.operation_id is None or not edge.operation_id.startswith("runtime."):
            raise ValueError(f"{record.sample_id}:{target.id}:{edge.id} lacks a runtime creator")
        operation = operations.get(edge.operation_id)
        if operation is None:
            raise ValueError(f"{record.sample_id}:{target.id}:{edge.id} creator is missing")
        created_tokens = {
            str(item.get("edge_token"))
            for item in operation.parameters.get("created_primitives", ())
            if isinstance(item, dict)
        }
        token = str(edge.provenance.get("runtime_object_token"))
        if token not in created_tokens:
            raise ValueError(
                f"{record.sample_id}:{target.id}:{edge.id} is not an output of {edge.operation_id}"
            )


def _audit_original_contract(record: TShirtTraceRecord) -> None:
    contract = record.metadata.get("creation_semantic_contract", {})
    required = {
        "completed_panel_topology_used_for_semantic_labels": False,
        "canonical_points_verified_as_operation_outputs": True,
        "semantic_edges_verified_as_operation_outputs": True,
        "all_operations_reachable_from_recipe_inputs": True,
    }
    for name, expected in required.items():
        if contract.get(name) is not expected:
            raise ValueError(
                f"{record.sample_id} creation contract {name}={contract.get(name)!r}, expected {expected!r}"
            )


def augment_record(record: TShirtTraceRecord) -> TShirtTraceRecord:
    """Add targets to one legacy record and fail closed on evidence drift."""

    if record.drafting_formula_targets or record.drafting_seam_relations:
        raise ValueError(f"{record.sample_id} is already formula-augmented")
    _audit_original_contract(record)
    targets = build_drafting_formula_targets(
        record.panels, source_kind="garmentcode_creation_trace"
    )
    relations = build_sleeve_armhole_relation(
        targets, source_kind="garmentcode_creation_trace"
    )
    if not targets or not relations:
        raise ValueError(f"{record.sample_id} produced no formula targets/seam relation")
    for target in targets:
        if (
            not target.training_eligible
            or target.evidence != "creation_event_formula_and_live_geometry"
            or target.provenance.get("source_kind") != "garmentcode_creation_trace"
            or target.provenance.get("role_inferred_from_shape") is not False
        ):
            raise ValueError(f"{record.sample_id}:{target.id} fails target evidence policy")
        _edge_operation_output_verified(record, target)
    for relation in relations:
        if not relation.training_eligible or relation.provenance.get("post_hoc_role_inference") is not False:
            raise ValueError(f"{record.sample_id}:{relation.id} fails seam evidence policy")

    metadata = {
        **record.metadata,
        "drafting_formula_target_migration": {
            "migration_version": MIGRATION_VERSION,
            "source_schema_version": record.schema_version,
            "role_source": "pre-labeled creation-event edges only",
            "completed_topology_role_inference": False,
            "geometry_source": "preserved live pre-assembly edge geometry",
        },
    }
    migrated = replace(
        record,
        drafting_formula_targets=targets,
        drafting_seam_relations=relations,
        metadata=metadata,
        schema_version="tshirt-construction-trace-1.1",
    )
    migrated.validate()
    for name in _PRESERVED_FIELDS:
        if getattr(migrated, name) != getattr(record, name):
            raise ValueError(f"{record.sample_id} migration changed preserved field: {name}")
    for name, value in record.metadata.items():
        if migrated.metadata.get(name) != value:
            raise ValueError(f"{record.sample_id} migration changed metadata field: {name}")
    return migrated


def _numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def migrate_corpus(
    input_path: Path,
    output_path: Path,
    *,
    expected_count: int | None = 2592,
    compresslevel: int = 6,
) -> dict[str, Any]:
    input_path = Path(input_path)
    output_path = Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output must be different; the source corpus is immutable")
    source_hash_before = _sha256(input_path)
    source_bytes_before = input_path.stat().st_size
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    if temporary.exists():
        temporary.unlink()

    counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    source_schema_counts: Counter[str] = Counter()
    target_role_counts: Counter[str] = Counter()
    panel_role_target_counts: Counter[str] = Counter()
    evidence_counts: Counter[str] = Counter()
    geometry_kind_counts: Counter[str] = Counter()
    control_mask_counts: Counter[str] = Counter()
    target_count_distribution: Counter[int] = Counter()
    relation_count_distribution: Counter[int] = Counter()
    semantic_value_support: Counter[str] = Counter()
    source_parameter_support: Counter[str] = Counter()
    relation_values: dict[str, list[float]] = defaultdict(list)
    content_digest = hashlib.sha256()
    sample_ids: set[str] = set()
    writer = None
    owners = None
    try:
        writer, owners = _deterministic_gzip_text(temporary, compresslevel=compresslevel)
        for line_number, raw in _rows(input_path):
            if "drafting_formula_targets" in raw or "drafting_seam_relations" in raw:
                raise ValueError(f"{input_path}:{line_number} is not a legacy unaugmented row")
            record = TShirtTraceRecord.from_dict(raw)
            if record.sample_id in sample_ids:
                raise ValueError(f"duplicate sample id: {record.sample_id}")
            sample_ids.add(record.sample_id)
            migrated = augment_record(record)
            row = json.dumps(
                migrated.to_dict(), sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
            )
            writer.write(row + "\n")
            content_digest.update(row.encode("utf-8") + b"\n")

            counts["records"] += 1
            split_counts[migrated.split] += 1
            source_schema_counts[record.schema_version] += 1
            target_count_distribution[len(migrated.drafting_formula_targets)] += 1
            relation_count_distribution[len(migrated.drafting_seam_relations)] += 1
            for target in migrated.drafting_formula_targets:
                counts["targets"] += 1
                counts["eligible_targets"] += int(target.training_eligible)
                target_role_counts[target.semantic_role] += 1
                panel_role_target_counts[f"{target.panel_role}:{target.semantic_role}"] += 1
                evidence_counts[target.evidence] += 1
                semantic_value_support.update(name for name, mask in target.semantic_mask.items() if mask)
                source_parameter_support.update(
                    name for name, mask in target.source_parameter_mask.items() if mask
                )
                for segment in target.segments:
                    counts["segments"] += 1
                    geometry_kind_counts[segment.geometry_kind] += 1
                    control_mask_counts[str(tuple(segment.bezier_control_mask))] += 1
            for relation in migrated.drafting_seam_relations:
                counts["relations"] += 1
                counts["eligible_relations"] += int(relation.training_eligible)
                for name, value in relation.values.items():
                    if relation.value_mask[name]:
                        relation_values[name].append(float(value))
        _close_writer(owners)
        writer = owners = None
        if expected_count is not None and counts["records"] != expected_count:
            raise ValueError(f"expected {expected_count} records, migrated {counts['records']}")
        os.replace(temporary, output_path)
    except Exception:
        if owners is not None:
            _close_writer(owners)
        if temporary.exists():
            temporary.unlink()
        raise

    source_hash_after = _sha256(input_path)
    if source_hash_after != source_hash_before or input_path.stat().st_size != source_bytes_before:
        raise RuntimeError("immutable source corpus changed during migration")
    return {
        "schema_version": "tshirt-formula-augmentation-manifest-1.0",
        "migration_version": MIGRATION_VERSION,
        "source_artifact": input_path.as_posix(),
        "source_artifact_bytes": source_bytes_before,
        "source_artifact_sha256_before": source_hash_before,
        "source_artifact_sha256_after": source_hash_after,
        "source_preserved": True,
        "source_schema_counts": dict(sorted(source_schema_counts.items())),
        "output_artifact": output_path.as_posix(),
        "output_artifact_bytes": output_path.stat().st_size,
        "output_artifact_sha256": _sha256(output_path),
        "output_uncompressed_canonical_rows_sha256": content_digest.hexdigest(),
        "record_count": counts["records"],
        "unique_sample_id_count": len(sample_ids),
        "split_counts": dict(sorted(split_counts.items())),
        "target_count": counts["targets"],
        "eligible_target_count": counts["eligible_targets"],
        "target_count_distribution_per_record": {
            str(count): frequency for count, frequency in sorted(target_count_distribution.items())
        },
        "target_role_counts": dict(sorted(target_role_counts.items())),
        "panel_role_target_counts": dict(sorted(panel_role_target_counts.items())),
        "target_evidence_counts": dict(sorted(evidence_counts.items())),
        "segment_count": counts["segments"],
        "segment_geometry_kind_counts": dict(sorted(geometry_kind_counts.items())),
        "segment_bezier_control_mask_counts": dict(sorted(control_mask_counts.items())),
        "semantic_value_support_counts": dict(sorted(semantic_value_support.items())),
        "source_formula_parameter_support_counts": dict(sorted(source_parameter_support.items())),
        "seam_relation_count": counts["relations"],
        "eligible_seam_relation_count": counts["eligible_relations"],
        "seam_relation_count_distribution_per_record": {
            str(count): frequency for count, frequency in sorted(relation_count_distribution.items())
        },
        "seam_relation_value_summary": {
            name: _numeric_summary(values) for name, values in sorted(relation_values.items())
        },
        "audit": {
            "preserved_fields_checked_per_record": list(_PRESERVED_FIELDS),
            "existing_metadata_preserved": True,
            "creation_edge_evidence_required": "creation_event_binding",
            "runtime_operation_output_membership_verified": True,
            "all_targets_training_eligible": counts["targets"] == counts["eligible_targets"],
            "all_relations_training_eligible": counts["relations"] == counts["eligible_relations"],
            "completed_topology_role_inference_used": False,
        },
        "limitations": {
            "expert_validation": "PENDING",
            "freesewing_included": False,
            "formula_target_scope": "neckline, armhole, sleeve_head only",
            "garment_scope": "bounded GarmentCode Shirt(fitted=False) basic T-shirt recipe",
            "sleeve_topology": "GarmentCode left/right front/back half-sleeve generator pieces",
            "seam_relation": "aggregate per record; exact sleeve segment to front/back pairing not asserted",
            "semantic_dimensions": "panel-axis endpoint measures and chord-local cap height; not yet expert-approved block definitions",
            "counterfactual_3d_renders": "not part of this migration",
        },
        "compression": {"format": "gzip", "level": compresslevel, "mtime": 0, "embedded_filename": ""},
    }


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Augment legacy creation traces with masked drafting-formula targets."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--expected-count", type=int, default=2592)
    parser.add_argument("--compresslevel", type=int, default=6)
    args = parser.parse_args()
    manifest = migrate_corpus(
        args.input,
        args.output,
        expected_count=args.expected_count,
        compresslevel=args.compresslevel,
    )
    _write_manifest(args.manifest, manifest)
    print(json.dumps(manifest, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
