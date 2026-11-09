from __future__ import annotations

"""Build a geometrically equivalent reduced-resolution compositor job.

This is an iteration aid only.  It never changes the delivery job or source
RGBA files; the final candidate must still be rendered from the original
full-resolution manifest.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from strict_compositor.image_io import load_background, load_mask, save_png


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_job")
    parser.add_argument("output_dir")
    parser.add_argument("--scale", type=float, default=0.35)
    args = parser.parse_args()

    scale = float(args.scale)
    if not 0.10 <= scale <= 0.75:
        parser.error("--scale must be in [0.10, 0.75]")

    source_job = Path(args.source_job).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with source_job.open(encoding="utf-8") as stream:
        job = json.load(stream)

    background = load_background(job["background"])
    preview_width = max(1, int(round(background.shape[1] * scale)))
    preview_height = max(1, int(round(background.shape[0] * scale)))
    preview = cv2.resize(
        background,
        (preview_width, preview_height),
        interpolation=cv2.INTER_AREA,
    )
    preview_path = output_dir / "background_preview.png"
    save_png(preview_path, preview)
    job["background"] = str(preview_path)

    # The support evidence is pixel-space data even though the polygon is
    # normalised.  Leaving a full-resolution mask attached to a reduced job
    # makes the contract loader resize it implicitly with the wrong geometry
    # and can falsely report that a valid product base is off the table.
    support_mask_path = job.get("support_mask")
    if support_mask_path:
        support_mask = load_mask(
            support_mask_path,
            (background.shape[1], background.shape[0]),
            allow_resize=True,
        )
        support_preview = cv2.resize(
            support_mask.astype(np.uint8) * 255,
            (preview_width, preview_height),
            interpolation=cv2.INTER_NEAREST,
        )
        preview_support_path = output_dir / "support_mask_preview.png"
        save_png(preview_support_path, support_preview)
        job["support_mask"] = str(preview_support_path)

    destination_scale = np.asarray(
        [[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]],
        np.float64,
    )
    for product in job["products"]:
        homography = np.asarray(product["homography"], np.float64)
        product["homography"] = (destination_scale @ homography).tolist()

    preview_job = output_dir / "job.json"
    with preview_job.open("w", encoding="utf-8") as stream:
        json.dump(job, stream, ensure_ascii=False, indent=2)
    print(preview_job)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
