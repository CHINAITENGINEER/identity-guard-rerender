from __future__ import annotations

import unittest

import numpy as np

from strict_compositor.ui import _color_gain


class SurfaceAwareColorGainTests(unittest.TestCase):
    @staticmethod
    def _neutral_product(
        rgb: tuple[int, int, int] = (164, 168, 178),
    ) -> np.ndarray:
        rgba = np.zeros((80, 40, 4), np.uint8)
        rgba[..., :3] = np.asarray(rgb, np.uint8)
        rgba[..., 3] = 255
        # Small source detail must not affect where the target illumination is
        # sampled.  _color_gain returns one bounded low-frequency triplet; the
        # downstream harmonizer preserves this high-frequency residual.
        rgba[30:36, 8:32, :3] = (44, 48, 56)
        return rgba

    @staticmethod
    def _placement() -> dict[str, float | str]:
        return {
            "id": "product_1",
            "center_x": 0.5,
            "baseline_y": 0.78,
            "height_ratio": 0.50,
        }

    @staticmethod
    def _lower_support() -> list[list[float]]:
        return [[0.0, 0.58], [1.0, 0.58], [1.0, 1.0], [0.0, 1.0]]

    def test_transparent_receiver_preserves_exposure_and_uses_environment_colour(self) -> None:
        background = np.zeros((200, 240, 3), np.uint8)
        # Warm scene environment behind the future product.
        background[:116, :] = (224, 194, 154)
        # A smoked glass/under-table region is not an illuminant probe.
        background[116:, :] = (30, 39, 55)
        rgba = self._neutral_product()

        local_gain = _color_gain(
            rgba,
            background,
            self._placement(),
            self._lower_support(),
            surface_material="matte",
        )
        optical_gain = _color_gain(
            rgba,
            background,
            self._placement(),
            self._lower_support(),
            surface_material="transparent_glass",
        )

        # Receiver brightness is not a calibrated exposure reference.  Neither
        # route may darken a correctly photographed product simply because the
        # local tabletop is dark; the optical route may apply only a restrained
        # warm/cool adaptation from the full-height environment.
        self.assertTrue(all(0.949 <= value <= 1.051 for value in local_gain))
        self.assertTrue(all(0.949 <= value <= 1.051 for value in optical_gain))
        self.assertGreater(optical_gain[0], optical_gain[2] + 0.04)

    def test_mirror_receiver_ignores_bright_contact_reflection(self) -> None:
        background = np.full((200, 240, 3), (156, 156, 156), np.uint8)
        # A narrow mirror highlight at the contact point must not make the
        # entire product brighter.  The full-height environment is moderate.
        background[132:190, 82:158] = (255, 255, 255)
        gain = _color_gain(
            self._neutral_product((148, 148, 148)),
            background,
            self._placement(),
            self._lower_support(),
            surface_material="mirror",
        )

        self.assertLess(max(gain), 1.04)
        self.assertGreater(min(gain), 0.78)

    def test_surface_adaptation_remains_strictly_bounded(self) -> None:
        bright_warm = np.full((160, 180, 3), (255, 220, 170), np.uint8)
        gains = _color_gain(
            self._neutral_product((32, 34, 38)),
            bright_warm,
            self._placement(),
            None,
            surface_material="transparent_glass",
            adaptation_strength=1.5,
        )

        self.assertTrue(all(0.78 <= value <= 1.13 for value in gains))


if __name__ == "__main__":
    unittest.main()
