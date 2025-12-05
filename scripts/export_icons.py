"""
Export PNG app icons for Technic using Pillow (no native cairo dependency).

Run:
    python scripts/export_icons.py
"""

from __future__ import annotations

import pathlib
from typing import Tuple

from PIL import Image, ImageDraw

OUTDIR = pathlib.Path("assets/brand/png")
SIZES = [64, 128, 180, 256, 512]
BG = (2, 4, 13, 255)  # dark backdrop
COLOR1 = (182, 255, 59, 255)  # neon green
COLOR2 = (94, 234, 212, 255)  # aqua accent


def make_gradient(size: int) -> Image.Image:
    """Create a simple linear gradient from COLOR1 to COLOR2."""
    grad = Image.linear_gradient("L").resize((size, size))
    base1 = Image.new("RGBA", (size, size), COLOR1)
    base2 = Image.new("RGBA", (size, size), COLOR2)
    return Image.composite(base2, base1, grad)


def draw_symbol(size: int) -> Image.Image:
    """
    Draw the Technic symbol procedurally:
    - Rounded tile with gradient, rotated -6deg
    - Inner white stroke
    - Small highlight bar
    """
    tile_size = int(size * 0.72)
    corner = int(tile_size * 0.25)

    tile = make_gradient(tile_size)
    mask = Image.new("L", (tile_size, tile_size), 0)
    draw_mask = ImageDraw.Draw(mask)
    draw_mask.rounded_rectangle(
        [(0, 0), (tile_size, tile_size)], radius=corner, fill=255
    )
    tile.putalpha(mask)

    # Inner stroke
    stroke = Image.new("RGBA", (tile_size, tile_size), (0, 0, 0, 0))
    draw_stroke = ImageDraw.Draw(stroke)
    stroke_width = max(2, tile_size // 16)
    inset = stroke_width * 2
    draw_stroke.rounded_rectangle(
        [(inset, inset), (tile_size - inset, tile_size - inset)],
        radius=corner // 1.1,
        outline=(255, 255, 255, 220),
        width=stroke_width,
    )

    # Highlight bar
    hl = Image.new("RGBA", (tile_size, tile_size), (0, 0, 0, 0))
    draw_hl = ImageDraw.Draw(hl)
    hl_w = max(4, tile_size // 12)
    hl_h = max(10, tile_size // 8)
    draw_hl.rounded_rectangle(
        [
            (tile_size - hl_w - inset, inset),
            (tile_size - inset, inset + hl_h),
        ],
        radius=hl_w // 2,
        fill=(255, 255, 255, 55),
    )

    # Combine and rotate
    symbol = Image.new("RGBA", (tile_size, tile_size), (0, 0, 0, 0))
    symbol = Image.alpha_composite(symbol, tile)
    symbol = Image.alpha_composite(symbol, stroke)
    symbol = Image.alpha_composite(symbol, hl)
    symbol = symbol.rotate(-6, resample=Image.BICUBIC, expand=True)
    return symbol


def export():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for size in SIZES:
        canvas = Image.new("RGBA", (size, size), BG)
        symbol = draw_symbol(size)
        sx, sy = symbol.size
        pos = ((size - sx) // 2, (size - sy) // 2)
        canvas.alpha_composite(symbol, dest=pos)
        dest = OUTDIR / f"icon_{size}.png"
        canvas.save(dest)
        print(f"Wrote {dest}")


if __name__ == "__main__":
    export()
