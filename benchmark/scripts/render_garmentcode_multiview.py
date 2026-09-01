from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BLENDER = ROOT / "external" / "blender-4.5.12-windows-x64" / "blender.exe"
BLENDER_SCRIPT = ROOT / "benchmark" / "blender_scripts" / "render_garmentcode_multiview.py"


def build_jobs(dataset: Path, output: Path, catalog: Path, views: list[str], limit: int, resume: bool) -> list[dict]:
    quality = {}
    if catalog.is_file():
        quality = {row["sample_id"]: row.get("render_quality") for row in json.loads(catalog.read_text(encoding="utf-8"))["records"]}
    jobs = []
    for sample in sorted(path for path in dataset.iterdir() if path.is_dir() and path.name.startswith("rand_")):
        if quality and quality.get(sample.name) != "PASS":
            continue
        garment = sample / f"{sample.name}_sim.ply"
        if not garment.is_file():
            continue
        jobs.append(
            {
                "sample_id": sample.name,
                "garment_ply": str(garment.resolve()),
                "output": str((output / sample.name).resolve()),
                "views": views,
                "resume": resume,
            }
        )
        if limit and len(jobs) >= limit:
            break
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerender official GCDv2 simulated meshes as orthogonal garment-only views.")
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/garmentcode_v2/batch_0_full"))
    parser.add_argument("--catalog", type=Path, default=Path("artifacts/retrieval_v2/garmentcode_v2_batch_0_catalog.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/retrieval_v2/garmentcode_v2_batch_0_multiview"))
    parser.add_argument(
        "--views",
        nargs="+",
        choices=["front", "back", "left", "right"],
        default=["front", "back", "left", "right"],
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--blender", type=Path, default=DEFAULT_BLENDER)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(args.dataset, args.output, args.catalog, args.views, args.limit, not args.no_resume)
    jobs_path = args.output / "_render_jobs.json"
    receipt = args.output / "_render_receipt.json"
    jobs_path.write_text(json.dumps({"jobs": jobs, "receipt": str(receipt.resolve())}, indent=2) + "\n", encoding="utf-8")
    completed = subprocess.run(
        [str(args.blender.resolve()), "--background", "--python", str(BLENDER_SCRIPT), "--", "--jobs", str(jobs_path.resolve()), "--resolution", str(args.resolution)],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if not receipt.is_file():
        raise RuntimeError(f"Blender produced no receipt. Tail:\n{completed.stdout[-4000:]}")
    print(receipt.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
