from __future__ import annotations

import unittest

import numpy as np

from strict_compositor.cutout import refine_green_product_shadows


class GreenShadowRefinementTests(unittest.TestCase):
    def test_removes_lateral_brown_shadow_but_keeps_central_base(self) -> None:
        rgba = np.zeros((120, 100, 4), np.uint8)
        rgba[5:55, 10:90, :3] = [55, 105, 35]
        rgba[5:55, 10:90, 3] = 255
        rgba[50:100, 38:62, :3] = [25, 20, 15]
        rgba[50:100, 38:62, 3] = 255
        rgba[95:116, 28:72, :3] = [85, 55, 30]
        rgba[95:116, 28:72, 3] = 255
        rgba[58:85, 8:25, :3] = [75, 48, 30]
        rgba[58:85, 8:25, 3] = 255

        refined, changed = refine_green_product_shadows(rgba)

        self.assertTrue(changed)
        self.assertEqual(int(refined[70, 14, 3]), 0)
        self.assertGreater(int(refined[105, 50, 3]), 240)

    def test_non_green_product_is_untouched(self) -> None:
        rgba = np.zeros((80, 80, 4), np.uint8)
        rgba[10:70, 20:60, :3] = [220, 60, 40]
        rgba[10:70, 20:60, 3] = 255

        refined, changed = refine_green_product_shadows(rgba)

        self.assertFalse(changed)
        self.assertTrue(np.array_equal(refined, rgba))
