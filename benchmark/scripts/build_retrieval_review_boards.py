from __future__ import annotations

from pathlib import Path

from benchmark.visualization.contact_sheet import create_contact_sheet


ROOT = Path(__file__).resolve().parents[2]
VIEWS = ("CAM000", "CAM001", "CAM002", "CAM003")


def build(layer: str, simulation: str, output_name: str) -> Path:
    target = ROOT / "data" / "processed" / "mpfb" / "mpfb_female_sportsuit" / "layers" / layer / "masks"
    simulated = ROOT / "artifacts" / "retrieval_v2" / simulation / "masks"
    paths = [target / f"{view}.png" for view in VIEWS] + [simulated / f"{view}.png" for view in VIEWS]
    labels = [f"target {view}" for view in VIEWS] + [f"retrieved anchor {view}" for view in VIEWS]
    output = ROOT / "artifacts" / "retrieval_v2" / "review_boards" / output_name
    create_contact_sheet(paths, output, labels, cell=(320, 320), columns=4)
    return output


def main() -> None:
    outputs = (
        build("lower", "simulation_dense/pants_straight", "pants_best.jpg"),
        build("upper", "simulation_absolute/tshirt_short_sleeve", "tshirt_best.jpg"),
    )
    for output in outputs:
        print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
