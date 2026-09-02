from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from benchmark.drafting_semantics.dataset import read_records
from benchmark.drafting_semantics.decoding import decode_named_landmarks


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit rule-derived drafting landmarks against their edge-junction definitions.")
    parser.add_argument("--records", type=Path, default=Path("artifacts/drafting_semantics/gcdv2_batch0/records.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/manifests/gcdv2_semantic_landmark_audit.json"))
    args = parser.parse_args()

    records = read_records(args.records)
    failures = []
    counts = Counter()
    split_ids: defaultdict[str, set[str]] = defaultdict(set)
    for record in records:
        split_ids[record.split].add(record.sample_id)
        for panel in record.panels:
            if panel.role not in {"front_bodice", "back_bodice"}:
                continue
            counts[f"panel:{panel.role}"] += 1
            roles = [edge.role for edge in panel.edges]
            decoded = decode_named_landmarks(panel, roles)
            expected = {
                item.name: tuple(item.xy_cm)
                for item in panel.landmarks
                if item.training_eligible and item.name in {"FNP", "BNP", "SNP", "SP"}
            }
            for name in expected:
                counts[f"landmark:{name}"] += 1
            if decoded != expected:
                failures.append({"sample_id": record.sample_id, "panel_id": panel.id, "expected": expected, "decoded": decoded})
    official = {key: value for key, value in split_ids.items() if key in {"training", "validation", "test"}}
    overlap = sorted(
        (official.get("training", set()) & official.get("validation", set()))
        | (official.get("training", set()) & official.get("test", set()))
        | (official.get("validation", set()) & official.get("test", set()))
    )
    result = {
        "status": "PASS" if not failures and not overlap else "FAIL",
        "record_count": len(records),
        "counts": dict(sorted(counts.items())),
        "official_split_sample_counts": {key: len(value) for key, value in sorted(official.items())},
        "official_split_overlap": overlap,
        "junction_rule_failures": failures[:100],
        "failure_count": len(failures),
        "evidence_boundary": "Names are derived from source edge labels plus topology; they are not independent expert annotations.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
