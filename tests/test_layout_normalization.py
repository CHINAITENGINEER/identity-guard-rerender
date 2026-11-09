from __future__ import annotations

import unittest

from strict_compositor.placement_solver import normalized_display_width


class LayoutNormalizationTests(unittest.TestCase):
    def test_landscape_x_width_uses_height_over_width(self) -> None:
        self.assertAlmostEqual(
            normalized_display_width(0.15, 1.0, (1500, 1000)),
            0.10,
            places=7,
        )

    def test_resolution_invariant_for_same_aspect(self) -> None:
        first = normalized_display_width(0.23, 0.72, (1500, 1000))
        second = normalized_display_width(0.23, 0.72, (6000, 4000))
        self.assertAlmostEqual(first, second, places=9)


if __name__ == "__main__":
    unittest.main()
