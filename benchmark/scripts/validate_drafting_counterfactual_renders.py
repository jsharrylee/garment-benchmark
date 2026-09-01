from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from benchmark.drafting_semantics.counterfactual_pairs import (
    CounterfactualContractError,
    validate_four_view_receipt,
)


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate future fixed-state four-view receipts; missing receipts remain pending."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data" / "manifests" / "drafting_counterfactual_pairs.json",
    )
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "artifacts"
        / "drafting_semantics"
        / "counterfactual_pairs"
        / "render_validation.json",
    )
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.resolve().read_text(encoding="utf-8"))
    results = []
    for record in manifest["records"]:
        receipt_path = args.receipts.resolve() / f"{record['pair_id']}.json"
        if not receipt_path.is_file():
            results.append(
                {
                    "pair_id": record["pair_id"],
                    "status": "PENDING_VALIDATED_SIMULATOR",
                    "pattern_only": True,
                }
            )
            continue
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            results.append(validate_four_view_receipt(record, receipt, root=ROOT))
        except (CounterfactualContractError, KeyError, TypeError, ValueError) as error:
            results.append(
                {
                    "pair_id": record["pair_id"],
                    "status": "FAILED_RENDER_RECEIPT_VALIDATION",
                    "pattern_only": True,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    counts = dict(sorted(Counter(item["status"] for item in results).items()))
    output = {
        "schema_version": "drafting-counterfactual-render-validation/v1",
        "source_manifest": args.manifest.resolve().relative_to(ROOT).as_posix(),
        "status_counts": counts,
        "validated_pair_count": counts.get("PASS_VALIDATED_FOUR_VIEW_RECEIPT", 0),
        "pair_count": len(results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status_counts": counts}, sort_keys=True))
    if args.require_complete and counts.get("PASS_VALIDATED_FOUR_VIEW_RECEIPT", 0) != len(results):
        raise SystemExit(2)


if __name__ == "__main__":
    main()

