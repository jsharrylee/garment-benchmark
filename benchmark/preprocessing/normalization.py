from __future__ import annotations

from PIL import Image


def contain_square(image: Image.Image, mask: Image.Image, *, size: int = 518, padding_ratio: float = 0.12) -> tuple[Image.Image, dict]:
    bbox = mask.getbbox()
    if bbox is None:
        raise ValueError("Cannot normalize an empty foreground mask")
    x0, y0, x1, y1 = bbox
    width, height = x1 - x0, y1 - y0
    pad = round(max(width, height) * padding_ratio)
    side = max(width, height) + 2 * pad
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    crop = (round(cx - side / 2), round(cy - side / 2), round(cx + side / 2), round(cy + side / 2))

    rgba = image.convert("RGBA")
    rgba.putalpha(mask)
    cropped = rgba.crop(crop)
    subject = cropped.resize((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    offset = ((size - subject.width) // 2, (size - subject.height) // 2)
    canvas.alpha_composite(subject, offset)
    result = Image.new("RGB", (size, size), "white")
    result.paste(canvas.convert("RGB"))
    return result, {
        "mask_bbox_xyxy": list(bbox),
        "square_crop_xyxy": list(crop),
        "resize_from": [cropped.width, cropped.height],
        "offset_xy": list(offset),
        "final_resolution": [size, size],
    }
