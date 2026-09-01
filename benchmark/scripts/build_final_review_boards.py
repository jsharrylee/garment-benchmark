from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from benchmark.visualization.contact_sheet import create_contact_sheet


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ignored side-by-side benchmark review boards.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--samples", nargs="+", default=["synbody_cyan_jacket", "synbody_patterned_shirt"])
    args = parser.parse_args()
    root = args.root.resolve()
    review_root = root / "artifacts" / "review_boards"
    results = []
    for sample in args.samples:
        reweaver_summary = json.loads((root / "artifacts" / "reweaver" / sample / "summary.json").read_text(encoding="utf-8"))
        particles_summary = json.loads((root / "artifacts" / "garment_particles" / sample / "summary.json").read_text(encoding="utf-8"))
        stats_path = review_root / f"{sample}_statistics.png"
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats = Image.new("RGB", (720, 640), "white")
        draw = ImageDraw.Draw(stats)
        lines = [
            sample,
            "",
            "ReWeaver",
            f"panels: {reweaver_summary['panel_count']}",
            f"curves: {reweaver_summary['curve_count']}",
            f"runtime: {reweaver_summary['runtime_seconds']:.2f} s",
            f"peak VRAM: {reweaver_summary['peak_vram_bytes'] / 1e9:.2f} GB",
            "",
            "Garment Particles",
            f"particles: {particles_summary['particle_count']}",
            f"panels / edges: {particles_summary['panel_count']} / {particles_summary['edge_count']}",
            f"stitch pairs: {particles_summary['stitch_pair_count']}",
            f"closure mean / max: {particles_summary['panel_closure_gap_mean']:.3f} / {particles_summary['panel_closure_gap_max']:.3f}",
            f"PGF / edge: {particles_summary['pgf_sample_seconds']:.2f} / {particles_summary['edge_sample_seconds']:.2f} s",
            f"peak VRAM: {max(particles_summary['pgf_peak_vram_bytes'], particles_summary['edge_peak_vram_bytes']) / 1e9:.2f} GB",
        ]
        draw.multiline_text((30, 30), "\n".join(lines), fill="black", spacing=12)
        stats.save(stats_path)
        paths = [
            root / "data" / "processed" / "synbody" / sample / "original_views.jpg",
            root / "data" / "processed" / "synbody" / sample / "normalized_inputs.jpg",
            root / "data" / "processed" / "synbody" / sample / "garment_particles" / "input.png",
            root / "artifacts" / "reweaver" / sample / "pattern.png",
            root / "artifacts" / "reweaver" / sample / "geometry.png",
            root / "artifacts" / "garment_particles" / sample / "pattern.png",
            root / "artifacts" / "garment_particles" / sample / "geometry.png",
            stats_path,
        ]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(missing)
        output = review_root / f"{sample}.jpg"
        create_contact_sheet(
            paths,
            output,
            ["Original four views", "Processed four views", "Selected front", "ReWeaver 2D", "ReWeaver 3D", "Garment Particles 2D", "Garment Particles 3D", "Statistics"],
            cell=(360, 320),
        )
        results.append({"sample_id": sample, "bytes": output.stat().st_size, "output": str(output.relative_to(root))})
    (review_root / "index.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", "boards": results}))


if __name__ == "__main__":
    main()
