from __future__ import annotations

import argparse
from pathlib import Path

from benchmark.visualization.contact_sheet import create_contact_sheet


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an ignored visual comparison of layered generative results")
    parser.add_argument("--sample-id", default="mpfb_female_sportsuit")
    parser.add_argument("--bundle", type=Path, default=ROOT / "data" / "processed" / "mpfb" / "mpfb_female_sportsuit")
    parser.add_argument("--upper-candidate", default="mpfb_female_sportsuit_upper_seed_20260826")
    parser.add_argument("--lower-candidate", default="mpfb_female_sportsuit_lower_seed_20260826")
    args = parser.parse_args()
    reweaver = ROOT / "artifacts" / "reweaver_layered" / args.sample_id
    particles = ROOT / "artifacts" / "garment_particles_layered"
    output = ROOT / "artifacts" / "layered_generative_pipeline" / args.sample_id / "review_board.jpg"
    images = [
        args.bundle / "layers" / "upper" / "four_view_condition.jpg",
        reweaver / "upper" / "pattern.png",
        particles / args.upper_candidate / "pattern.png",
        particles / args.upper_candidate / "geometry.png",
        args.bundle / "layers" / "lower" / "four_view_condition.jpg",
        reweaver / "lower" / "pattern.png",
        particles / args.lower_candidate / "pattern.png",
        particles / args.lower_candidate / "geometry.png",
    ]
    missing = [str(path) for path in images if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    create_contact_sheet(
        images,
        output,
        [
            "Upper 4-view condition",
            "Upper ReWeaver (63 errors)",
            "Upper GP selected (8 errors)",
            "Upper generated particles",
            "Lower 4-view condition",
            "Lower ReWeaver (26 errors)",
            "Lower GP selected (7 errors)",
            "Lower generated particles",
        ],
        cell=(480, 360),
        columns=4,
    )
    print(output)


if __name__ == "__main__":
    main()
