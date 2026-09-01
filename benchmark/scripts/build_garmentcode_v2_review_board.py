from __future__ import annotations

import json
from pathlib import Path

from benchmark.visualization.contact_sheet import create_contact_sheet


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    manifest = json.loads((ROOT / "data/manifests/garmentcode_v2_batch_0_canonical_subset.json").read_text(encoding="utf-8"))
    source = ROOT / "data/processed/garmentcode_v2/batch_0_quality_subset"
    chosen = {}
    for record in manifest["records"]:
        if record["structural_validation"]["accepted"] and record["category"] not in chosen:
            chosen[record["category"]] = record["sample_id"]
    paths = []
    labels = []
    for category, sample_id in sorted(chosen.items()):
        sample = source / sample_id
        for suffix, label in (("pattern.png", "pattern"), ("render_front.png", "front drape"), ("render_back.png", "back drape")):
            paths.append(sample / f"{sample_id}_{suffix}")
            labels.append(f"{category} · {sample_id} · {label}")
    output = ROOT / "artifacts/retrieval_v2/review_boards/garmentcode_v2_batch_0.jpg"
    create_contact_sheet(paths, output, labels, cell=(360, 360), columns=3)
    print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
