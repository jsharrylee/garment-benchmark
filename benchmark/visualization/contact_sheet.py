from __future__ import annotations

from math import ceil
from pathlib import Path

from PIL import Image, ImageDraw


def create_contact_sheet(paths: list[Path], output: Path, labels: list[str], *, cell=(480, 270), columns: int | None = None) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = columns or len(paths)
    if columns < 1:
        raise ValueError("columns must be positive")
    rows = ceil(len(paths) / columns)
    sheet = Image.new("RGB", (cell[0] * columns, (cell[1] + 34) * rows), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (path, label) in enumerate(zip(paths, labels, strict=True)):
        with Image.open(path) as source:
            image = source.convert("RGB")
            image.thumbnail(cell)
            column, row = index % columns, index // columns
            x = column * cell[0] + (cell[0] - image.width) // 2
            y = row * (cell[1] + 34) + (cell[1] - image.height) // 2
            sheet.paste(image, (x, y))
        draw.text((column * cell[0] + 8, row * (cell[1] + 34) + cell[1] + 8), label, fill="black")
    sheet.save(output, quality=90)
