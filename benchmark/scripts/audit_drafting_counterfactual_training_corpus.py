from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from benchmark.drafting_semantics.counterfactual_pairs import (
    TRAINING_ELIGIBILITY_RULE,
    canonical_json_sha256,
    counterfactual_training_eligibility,
    file_sha256,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = (
    ROOT
    / "artifacts"
    / "drafting_semantics"
    / "counterfactual_pairs"
    / "corpus32_manifest.json"
)


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def audit_manifest(source: dict[str, Any]) -> dict[str, Any]:
    records = source.get("records")
    if not isinstance(records, list):
        raise ValueError("source counterfactual manifest requires a records array")
    ids = [record.get("pair_id") for record in records]
    if any(not isinstance(value, str) or not value for value in ids):
        raise ValueError("every source record requires a pair_id")
    if len(ids) != len(set(ids)):
        raise ValueError("source pair_id values must be unique")

    accepted: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    compact_rows: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    accepted_source_counts: Counter[str] = Counter()
    quarantined_source_counts: Counter[str] = Counter()
    for record in records:
        eligibility = counterfactual_training_eligibility(record)
        enriched = {
            **record,
            **eligibility,
            "source_evidence_record_sha256": canonical_json_sha256(record),
        }
        source_name = str(record.get("source", "unknown"))
        source_counts[source_name] += 1
        if eligibility["training_eligible"]:
            accepted.append(enriched)
            accepted_source_counts[source_name] += 1
        else:
            quarantined.append(enriched)
            quarantined_source_counts[source_name] += 1
            reason_counts.update(eligibility["quarantine_reasons"])
        compact_rows.append(
            {
                "pair_id": record["pair_id"],
                "source": source_name,
                "intervention_parameter": record.get("intervention_parameter"),
                "training_eligible": eligibility["training_eligible"],
                "quarantine_reasons": eligibility["quarantine_reasons"],
                "contract_validation": record.get("contract_validation"),
                "pattern_geometry_changed": record.get("pattern_geometry_changed"),
                "topology_stable": record.get("topology_stable"),
                "semantic_delta_coverage": record.get("semantic_delta_coverage", {}).get("status"),
                "source_evidence_record_sha256": enriched["source_evidence_record_sha256"],
            }
        )
    return {
        "accepted": accepted,
        "quarantined": quarantined,
        "audit_rows": compact_rows,
        "summary": {
            "source_record_count": len(records),
            "accepted_count": len(accepted),
            "quarantined_count": len(quarantined),
            "source_counts": dict(sorted(source_counts.items())),
            "accepted_source_counts": dict(sorted(accepted_source_counts.items())),
            "quarantined_source_counts": dict(sorted(quarantined_source_counts.items())),
            "quarantine_reason_counts": dict(sorted(reason_counts.items())),
            "all_source_records_accounted_for": len(accepted) + len(quarantined) == len(records),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit and filter controlled pattern-only counterfactual pairs without mutating source evidence."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--training-output", type=Path)
    parser.add_argument("--quarantine-output", type=Path)
    parser.add_argument("--audit-output", type=Path)
    args = parser.parse_args()
    source_path = args.source.resolve()
    stem = source_path.stem.removesuffix("_manifest")
    output_directory = source_path.parent
    training_output = (
        args.training_output.resolve()
        if args.training_output
        else output_directory / f"{stem}_training_manifest.json"
    )
    quarantine_output = (
        args.quarantine_output.resolve()
        if args.quarantine_output
        else output_directory / f"{stem}_quarantine_manifest.json"
    )
    audit_output = (
        args.audit_output.resolve()
        if args.audit_output
        else output_directory / f"{stem}_training_audit.json"
    )
    source = json.loads(source_path.read_text(encoding="utf-8"))
    result = audit_manifest(source)
    common = {
        "source_manifest": _display_path(source_path),
        "source_manifest_sha256": file_sha256(source_path),
        "source_manifest_schema_version": source.get("schema_version"),
        "source_manifest_status": source.get("status"),
        "training_eligibility_rule": TRAINING_ELIGIBILITY_RULE,
        "evidence_policy": "source manifest is immutable; copied records retain source_evidence_record_sha256",
        "pattern_only": True,
        "render_status": source.get("render_status", "PENDING_VALIDATED_SIMULATOR"),
        **result["summary"],
    }
    # Avoid a misleading top-level ``source_counts`` value in filtered
    # manifests: the common summary describes the immutable 448-record source,
    # while each output must describe only the records it actually contains.
    common["source_manifest_source_counts"] = common.pop("source_counts")
    training_manifest = {
        "schema_version": "drafting-counterfactual-training-manifest/v1",
        "status": "PASS_FILTERED_PATTERN_ONLY_TRAINING_CORPUS",
        **common,
        "record_count": result["summary"]["accepted_count"],
        "source_counts": result["summary"]["accepted_source_counts"],
        "records": result["accepted"],
    }
    quarantine_manifest = {
        "schema_version": "drafting-counterfactual-quarantine-manifest/v1",
        "status": "QUARANTINED_NOT_TRAINING_ELIGIBLE",
        **common,
        "record_count": result["summary"]["quarantined_count"],
        "source_counts": result["summary"]["quarantined_source_counts"],
        "records": result["quarantined"],
    }
    audit = {
        "schema_version": "drafting-counterfactual-training-audit/v1",
        "status": "PASS_AUDIT_COMPLETE",
        **common,
        "training_manifest": _display_path(training_output),
        "quarantine_manifest": _display_path(quarantine_output),
        "records": result["audit_rows"],
    }
    _write(training_output, training_manifest)
    _write(quarantine_output, quarantine_manifest)
    _write(audit_output, audit)
    print(
        json.dumps(
            {
                "source": result["summary"]["source_record_count"],
                "accepted": result["summary"]["accepted_count"],
                "quarantined": result["summary"]["quarantined_count"],
                "reasons": result["summary"]["quarantine_reason_counts"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
