"""Run a reproducible, offline pixel-contract demo with generated artwork.

No source photographs, model weights, credentials or network calls are needed.
The intentionally bad full-frame candidate demonstrates local hard recovery;
it is not intended to demonstrate photorealistic rendering quality.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strict_compositor.contracts import Job
from strict_compositor.image_io import save_png
from strict_compositor.pipeline import StrictCompositor
from strict_compositor.validate import audit_background


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def generate_assets(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    width, height = 640, 420
    yy, xx = np.indices((height, width), dtype=np.float32)
    background = np.empty((height, width, 3), dtype=np.uint8)
    background[:] = [225, 231, 234]
    tabletop = yy >= 176
    for channel, base in enumerate((190, 159, 122)):
        shade = base + (xx / width) * 9 - (yy / height) * 12
        background[..., channel][tabletop] = shade[tabletop].astype(np.uint8)
    scene = Image.fromarray(background)
    draw = ImageDraw.Draw(scene)
    draw.rectangle((575, 180, 630, 230), fill=(60, 70, 78))
    draw.text((15, 22), "SYNTHETIC INPUT - NO EXTERNAL ASSETS", fill=(45, 55, 65))
    background = np.asarray(scene).copy()
    save_png(root / "background.png", background)

    can = Image.new("RGBA", (96, 144), (0, 0, 0, 0))
    draw = ImageDraw.Draw(can)
    draw.rounded_rectangle((12, 5, 84, 140), radius=10, fill=(38, 99, 157, 255))
    draw.rectangle((13, 51, 83, 89), fill=(241, 238, 222, 255))
    draw.text((21, 64), "KEEP 01", fill=(20, 33, 48, 255))
    can.save(root / "can_rgba.png")
    identity = Image.new("L", can.size, 0)
    ImageDraw.Draw(identity).rectangle((13, 51, 83, 89), fill=255)
    identity.save(root / "can_identity.png")

    glass = Image.new("RGBA", (96, 144), (0, 0, 0, 0))
    draw = ImageDraw.Draw(glass)
    draw.rounded_rectangle((9, 8, 70, 139), radius=10, fill=(190, 204, 212, 125), outline=(225, 236, 240, 230), width=3)
    draw.ellipse((60, 35, 94, 100), outline=(221, 231, 236, 220), width=5)
    draw.line((17, 21, 17, 119), fill=(240, 246, 248, 180), width=3)
    draw.line((18, 135, 63, 135), fill=(230, 236, 240, 245), width=4)
    glass.save(root / "glass_rgba.png")

    box = Image.new("RGBA", (96, 144), (0, 0, 0, 0))
    draw = ImageDraw.Draw(box)
    draw.rectangle((9, 10, 87, 140), fill=(187, 104, 57, 255))
    draw.rectangle((15, 58, 81, 88), fill=(244, 229, 203, 255))
    draw.text((22, 67), "KEEP 02", fill=(69, 40, 23, 255))
    box.save(root / "box_rgba.png")

    return {
        "background": "background.png",
        "support_polygon": [[0, 0.42], [1, 0.42], [1, 0.98], [0, 0.98]],
        "protected_polygons": [[[0.895, 0.42], [0.99, 0.42], [0.99, 0.56], [0.895, 0.56]]],
        "lighting": {
            "shadow_direction": [1, 0.35],
            "shadow_length_ratio": 0.12,
            "surface_material": "matte",
        },
        "products": [
            {"id": "printed_can", "rgba": "can_rgba.png", "homography": [[1, 0, 95], [0, 1, 195], [0, 0, 1]], "z_order": 1, "material_mode": "opaque", "semantic_category": "printed_can", "identity_status": "detected", "identity_mask": "can_identity.png", "rerender_policy": "never"},
            {"id": "clear_glass", "rgba": "glass_rgba.png", "homography": [[1, 0, 265], [0, 1, 195], [0, 0, 1]], "z_order": 2, "material_mode": "clear_glass", "material_confidence": 1.0, "transmission_strength": 0.90, "semantic_category": "glass_pitcher", "identity_status": "confirmed_absent", "rerender_policy": "transparent_optical"},
            {"id": "printed_box", "rgba": "box_rgba.png", "homography": [[1, 0, 435], [0, 1, 195], [0, 0, 1]], "z_order": 3, "material_mode": "opaque", "semantic_category": "printed_package", "rerender_policy": "never"},
        ],
    }


def run_demo(root: Path) -> dict:
    job_data = generate_assets(root)
    write_json(root / "strict_job.json", job_data)
    strict_result = StrictCompositor().compose(
        Job.from_dict(job_data, root), root / "strict", strict_audit=False,
    )

    # Deliberately alter the entire frame. The recovery stage must reject all
    # candidate pixels outside the precomputed glass interior.
    candidate = strict_result.final.copy()
    candidate = np.clip(
        candidate.astype(np.int16) + np.asarray([-28, 12, 42]), 0, 255,
    ).astype(np.uint8)
    save_png(root / "untrusted_candidate.png", candidate)
    hybrid_job = dict(job_data)
    hybrid_job["hybrid_render"] = {
        "enabled": True,
        "provider": "manual_candidate",
        "candidate": "untrusted_candidate.png",
    }
    write_json(root / "hybrid_job.json", hybrid_job)
    hybrid_result = StrictCompositor().compose(
        Job.from_dict(hybrid_job, root), root / "hybrid", strict_audit=False,
    )
    strict_base = np.asarray(Image.open(root / "hybrid" / "strict_base.png").convert("RGB"))
    allow = np.asarray(Image.open(root / "hybrid" / "hybrid_rerender_mask.png").convert("L")) > 0
    guard = np.asarray(Image.open(root / "hybrid" / "hybrid_identity_guard.png").convert("L")) > 0
    effective = allow & ~guard
    delta = np.abs(hybrid_result.final.astype(np.int16) - strict_base.astype(np.int16))
    changed = np.any(delta > 0, axis=2)
    original_background = np.asarray(Image.open(root / "background.png").convert("RGB"))
    authorized = np.asarray(Image.open(root / "hybrid" / "authorized_mask.png").convert("L")) > 0
    background_audit = audit_background(original_background, hybrid_result.final, authorized)
    report = {
        "input_origin": "generated geometric artwork; no external assets",
        "image_size": [int(strict_base.shape[1]), int(strict_base.shape[0])],
        "eligible_pixels": int(np.count_nonzero(effective)),
        "changed_pixels_inside": int(np.count_nonzero(changed & effective)),
        "identity_changed_pixels": int(np.count_nonzero(changed & guard)),
        "outside_changed_pixels": int(np.count_nonzero(changed & ~effective)),
        "max_outside_delta": int(delta[~effective].max()) if np.any(~effective) else 0,
        "background_pixel_audit": background_audit.to_dict(),
        "visual_acceptance": "not_evaluated",
        "batch_unlock": False,
    }
    if not report["eligible_pixels"] or not report["changed_pixels_inside"]:
        raise RuntimeError("Demo did not exercise candidate recovery.")
    if any(report[key] != 0 for key in ("identity_changed_pixels", "outside_changed_pixels", "max_outside_delta")):
        raise RuntimeError("Local pixel-recovery invariant failed.")
    if background_audit.verdict != "pass":
        raise RuntimeError("Background pixel audit failed.")
    write_json(root / "pixel_contract.json", report)

    panels = [
        (strict_base, "1. Strict base"),
        (candidate, "2. Untrusted candidate: whole frame modified"),
        (np.repeat(effective[..., None], 3, axis=2).astype(np.uint8) * 255, "3. Effective editable mask"),
        (hybrid_result.final, "4. Recovered: only editable pixels adopted"),
    ]
    width, height = strict_base.shape[1], strict_base.shape[0]
    comparison = Image.new("RGB", (width * 2, (height + 30) * 2), "white")
    painter = ImageDraw.Draw(comparison)
    for index, (array, title) in enumerate(panels):
        x, y = (index % 2) * width, (index // 2) * (height + 30)
        painter.text((x + 10, y + 8), title, fill="black")
        comparison.paste(Image.fromarray(array), (x, y + 30))
    comparison.save(root / "comparison.png")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "output" / "synthetic")
    args = parser.parse_args()
    report = run_demo(args.out.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
