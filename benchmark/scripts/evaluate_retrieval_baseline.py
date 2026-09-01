from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark.retrieval.index import PatternIndex, QueryEvidence


def main() -> None:
    parser = argparse.ArgumentParser(description="Leave-one-out category retrieval on the bounded official corpus.")
    parser.add_argument("--index", type=Path, default=Path("artifacts/retrieval_v2/gcd_ts_8_index.json"))
    parser.add_argument("--summary-root", type=Path, default=Path("artifacts/reweaver_official_gcd_ts"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/retrieval_v2/leave_one_out.json"))
    args = parser.parse_args()

    complete = PatternIndex.read_json(args.index)
    rows = []
    eligible_rows = []
    for target in complete.records:
        remaining = [record for record in complete.records if record.sample_id != target.sample_id]
        relevant = [record for record in remaining if record.category == target.category]
        summary = args.summary_root / target.sample_id / "summary.json"
        query = QueryEvidence.from_files(
            target.visual_descriptor,
            category=None,
            reweaver_summary=summary if summary.exists() else None,
        )
        result = PatternIndex(remaining).search(query, top_k=5, minimum_score=0.0)
        ranks = [index + 1 for index, item in enumerate(result.candidates) if item.category == target.category]
        row = {
            "sample_id": target.sample_id,
            "category": target.category,
            "category_has_other_anchor": bool(relevant),
            "first_category_match_rank": ranks[0] if ranks else None,
            "top_ids": [item.sample_id for item in result.candidates],
            "top_categories": [item.category for item in result.candidates],
        }
        rows.append(row)
        if relevant:
            eligible_rows.append(row)
    metrics = {
        "eligible_queries": len(eligible_rows),
        "recall_at_1": sum(row["first_category_match_rank"] == 1 for row in eligible_rows) / max(1, len(eligible_rows)),
        "recall_at_3": sum(row["first_category_match_rank"] is not None and row["first_category_match_rank"] <= 3 for row in eligible_rows) / max(1, len(eligible_rows)),
        "recall_at_5": sum(row["first_category_match_rank"] is not None and row["first_category_match_rank"] <= 5 for row in eligible_rows) / max(1, len(eligible_rows)),
    }
    payload = {"mode": "retrieval_anchored_v2", "evaluation": "leave_one_out_category", "metrics": metrics, "samples": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
