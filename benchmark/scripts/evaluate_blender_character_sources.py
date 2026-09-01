from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from benchmark.evaluation.generative_routing import assess_generated_patterns
from benchmark.visualization.contact_sheet import create_contact_sheet


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "benchmark" / "configs" / "blender_character_evaluation.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "blender_character_evaluation"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def stage_metrics(receipt: dict, stage_name: str) -> dict:
    stage = next(stage for stage in receipt["stages"] if stage["stage"] == stage_name)
    if stage_name == "REPAIR_STRUCTURE":
        return stage["after"]["metrics"]
    return stage["metrics"]


def build_board(root: Path, output: Path, sample: dict, record: dict) -> None:
    bundle = root / sample["bundle"]
    reweaver = root / sample["reweaver"]
    particles = root / sample["garment_particles"]
    stats_path = output.parent / f"{sample['sample_id']}_statistics.png"
    stats = Image.new("RGB", (720, 640), "white")
    draw = ImageDraw.Draw(stats)
    lines = [
        sample["sample_id"],
        f"source: {sample['source']}",
        "",
        "ReWeaver (4 views)",
        f"panels / errors: {record['reweaver']['panel_count']} / {record['reweaver']['errors_after_repair']}",
        f"status: {record['assessment']['reweaver']['structural_status']}",
        "",
        "Garment Particles (front)",
        f"panels / edges: {record['garment_particles']['panel_count']} / {record['garment_particles']['edge_count']}",
        f"stitch pairs: {record['garment_particles']['stitch_pair_count']}",
        f"canonical status: {record['assessment']['garment_particles']['structural_status']}",
        f"closure mean / max: {record['garment_particles']['panel_closure_gap_mean']:.3f} / {record['garment_particles']['panel_closure_gap_max']:.3f}",
        "",
        f"draft route: {record['assessment']['primary_generated_draft']}",
        f"technical: {record['assessment']['technical_status']}",
        "manufacturing: NOT VALIDATED",
        "template retrieval: NEVER",
    ]
    draw.multiline_text((30, 30), "\n".join(lines), fill="black", spacing=12)
    stats.save(stats_path)
    paths = [
        bundle / "reweaver_input_contact_sheet.jpg",
        bundle / "garment_particles" / "input.png",
        reweaver / "pattern.png",
        reweaver / "geometry.png",
        particles / "pattern.png",
        particles / "geometry.png",
        stats_path,
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    create_contact_sheet(
        paths,
        output,
        ["Four-view input", "Selected front", "ReWeaver 2D", "ReWeaver 3D", "Garment Particles 2D", "Garment Particles 3D", "Decision"],
        cell=(420, 340),
        columns=4,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare generated pattern lanes on Blender character sources")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config = load_json(args.config)
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for sample in config["samples"]:
        reweaver_summary = load_json(ROOT / sample["reweaver"] / "summary.json")
        receipt = load_json(ROOT / sample["pattern_pipeline"] / "pipeline_receipt.json")
        particles = load_json(ROOT / sample["garment_particles"] / "summary.json")
        particles_receipt = load_json(ROOT / sample["garment_particles_pattern_pipeline"] / "pipeline_receipt.json")
        repaired = stage_metrics(receipt, "REPAIR_STRUCTURE")
        record = {
            "sample_id": sample["sample_id"],
            "source": sample["source"],
            "reweaver": {
                "panel_count": reweaver_summary["panel_count"],
                "curve_count": reweaver_summary["curve_count"],
                "errors_after_repair": repaired["error_count"],
                "warnings_after_repair": repaired["warning_count"],
                "max_closure_gap_cm": repaired["max_closure_gap_cm"],
                "mean_closure_gap_cm": repaired["mean_closure_gap_cm"],
                "mean_seam_length_mismatch": repaired["mean_seam_length_mismatch"],
            },
            "garment_particles": {
                key: particles[key]
                for key in (
                    "valid",
                    "panel_count",
                    "edge_count",
                    "stitch_pair_count",
                    "panel_closure_gap_mean",
                    "panel_closure_gap_max",
                    "particle_count",
                )
            },
            "assessment": assess_generated_patterns(receipt, particles, particles_receipt),
        }
        board = args.output / f"{sample['sample_id']}.jpg"
        build_board(ROOT, board, sample, record)
        record["local_review_board"] = str(board.relative_to(ROOT))
        records.append(record)
    result = {"generation_contract": config["generation_contract"], "samples": records}
    (args.output / "evaluation.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
