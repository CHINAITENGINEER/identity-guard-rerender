from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    tiles: list[Image.Image] = []
    for path in args.inputs:
        rgba = Image.open(path).convert("RGBA")
        rgba.thumbnail((640, 640), Image.Resampling.LANCZOS)
        checker = Image.new("RGB", rgba.size, (238, 238, 238))
        draw = ImageDraw.Draw(checker)
        step = 24
        for y in range(0, rgba.height, step):
            for x in range(0, rgba.width, step):
                if ((x // step) + (y // step)) % 2:
                    draw.rectangle((x, y, x + step - 1, y + step - 1), fill=(198, 198, 198))
        checker.paste(rgba.convert("RGB"), mask=rgba.getchannel("A"))
        tiles.append(checker)

    width = sum(tile.width for tile in tiles)
    height = max(tile.height for tile in tiles)
    sheet = Image.new("RGB", (width, height), "white")
    offset = 0
    for tile in tiles:
        sheet.paste(tile, (offset, (height - tile.height) // 2))
        offset += tile.width
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)


if __name__ == "__main__":
    main()
