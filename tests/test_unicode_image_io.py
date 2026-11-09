from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from strict_compositor.image_io import save_png


def test_lossless_images_round_trip_in_unicode_paths() -> None:
    """A checkout under a Chinese Windows folder must keep exact pixel data."""
    rng = np.random.default_rng(2026)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "商品示例" / "透明图层"
        for channels in (1, 3, 4):
            shape = (19, 23) if channels == 1 else (19, 23, channels)
            pixels = rng.integers(0, 256, shape, dtype=np.uint8)
            path = root / f"测试_{channels}.png"
            save_png(path, pixels)
            with Image.open(path) as decoded:
                np.testing.assert_array_equal(np.asarray(decoded), pixels)
