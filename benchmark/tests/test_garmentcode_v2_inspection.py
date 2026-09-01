from __future__ import annotations

import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from benchmark.scripts.inspect_garmentcode_v2_batch import extract_all, extract_selected, inspect, render_quality, safe_relative_path, select_balanced


def add_bytes(bundle: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    bundle.addfile(info, io.BytesIO(payload))


def specification(panel_names: list[str]) -> bytes:
    panels = {
        name: {
            "vertices": [[0, 0], [1, 0], [1, 1], [0, 1]],
            "edges": [{"endpoints": [0, 1]}, {"endpoints": [1, 2]}, {"endpoints": [2, 3]}, {"endpoints": [3, 0]}],
        }
        for name in panel_names
    }
    return json.dumps({"pattern": {"panels": panels, "stitches": []}}).encode()


def rendered_garment() -> bytes:
    image = Image.new("RGB", (64, 64), "white")
    ImageDraw.Draw(image).rectangle((20, 20, 44, 48), fill=(180, 80, 120))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


class GarmentCodeV2InspectionTests(unittest.TestCase):
    def test_scan_category_selection_and_safe_extraction(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root / "data.tar.gz"
            with tarfile.open(archive, "w:gz") as bundle:
                add_bytes(bundle, "rand_TOP/rand_TOP_specification.json", specification(["left_ftorso", "right_ftorso"]))
                add_bytes(bundle, "rand_TOP/rand_TOP_sim.obj", b"v 0 0 0\n")
                add_bytes(bundle, "rand_TOP/rand_TOP_render_front.png", rendered_garment())
                add_bytes(bundle, "rand_PANTS/rand_PANTS_specification.json", specification(["pant_f_l", "pant_b_l"]))
                add_bytes(bundle, "rand_PANTS/rand_PANTS_sim.obj", b"v 0 0 0\n")
                add_bytes(bundle, "rand_PANTS/rand_PANTS_render_front.png", rendered_garment())
            split = root / "split.json"
            split.write_text(
                json.dumps(
                    {
                        "training": ["garments_5000_0/default_body/rand_TOP"],
                        "validation": ["garments_5000_0/default_body/rand_PANTS"],
                        "test": [],
                    }
                ),
                encoding="utf-8",
            )
            summary, records = inspect(archive, split)
            self.assertEqual(summary["specification_count"], 2)
            self.assertEqual(summary["category_counts"], {"pants": 1, "top": 1})
            selected = select_balanced(records, 1)
            output = root / "selected"
            files, size = extract_selected(archive, output, selected)
            self.assertEqual(files, 6)
            self.assertGreater(size, 0)
            self.assertTrue((output / "rand_TOP" / "rand_TOP_specification.json").is_file())

    def test_unsafe_member_is_rejected(self):
        with self.assertRaises(ValueError):
            safe_relative_path("../escape.txt")

    def test_render_quality_rejects_fallen_skirt(self):
        self.assertEqual(render_quality("skirt", {"colored_fraction": 0.02, "centroid_y": 0.93}), "REJECT_IMPLAUSIBLE_VERTICAL_PLACEMENT")
        self.assertEqual(render_quality("pants", {"colored_fraction": 0.02, "centroid_y": 0.65}), "PASS")

    def test_extract_all_writes_completion_receipt(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root / "data.tar.gz"
            with tarfile.open(archive, "w:gz") as bundle:
                add_bytes(bundle, "./rand_A/rand_A_specification.json", b"{}")
                add_bytes(bundle, "./rand_A/rand_A_render_front.png", b"png")
            output = root / "full"
            files, size = extract_all(archive, output)
            self.assertEqual(files, 2)
            self.assertEqual(size, 5)
            receipt = json.loads((output / "_extraction_receipt.json").read_text(encoding="utf-8"))
            self.assertEqual(receipt["status"], "PASS_COMPLETE_SAFE_EXTRACTION")
            self.assertEqual(receipt["file_count"], 2)


if __name__ == "__main__":
    unittest.main()
