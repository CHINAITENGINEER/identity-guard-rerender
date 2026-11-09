import unittest

import cv2
import numpy as np

from strict_compositor.contracts import Lighting
from strict_compositor.harmonization import low_frequency_harmonize
from strict_compositor.render import render_product


class LowFrequencyHarmonizationTests(unittest.TestCase):
    def test_neutral_adjustment_is_byte_exact_for_colour_and_gloss(self) -> None:
        rng = np.random.default_rng(23)
        rgba = rng.integers(0, 256, (96, 84, 4), dtype=np.uint8)
        rgba[..., 3] = 0
        rgba[9:89, 11:73, 3] = 255
        # A compact photographed specular lobe must not be shifted or clipped.
        rgba[28:34, 38:45, :3] = [251, 238, 218]

        output = low_frequency_harmonize(
            rgba,
            (1.0, 1.0, 1.0),
            chroma_retention=1.0,
            shadow_direction=(0.8, 0.6),
            relight_strength=0.0,
        )

        np.testing.assert_array_equal(output, rgba)

    def test_opaque_render_does_not_repaint_source_highlights(self) -> None:
        rgba = np.zeros((72, 72, 4), np.uint8)
        rgba[8:65, 13:59, :3] = [112, 28, 43]
        rgba[8:65, 13:59, 3] = 255
        # Deliberately asymmetric gloss: a guessed light gradient would alter
        # this central identity patch even though the RGB gain is neutral.
        rgba[25:31, 25:46, :3] = [226, 135, 145]
        background = np.full((72, 72, 3), [196, 174, 136], np.uint8)
        support = np.ones((72, 72), bool)
        protected = np.zeros((72, 72), bool)
        lighting = Lighting(
            direction=(0.8, 0.6),
            shadow_length_ratio=0.08,
            reflection_enabled=False,
        )

        rendered = render_product(
            rgba,
            np.eye(3, dtype=np.float64),
            (72, 72),
            background,
            support,
            protected,
            lighting,
            rgb_gain=(1.0, 1.0, 1.0),
            material_mode="opaque",
        )

        np.testing.assert_array_equal(
            rendered.rgb[18:43, 20:52], rgba[18:43, 20:52, :3],
        )

    def test_preserves_alpha_transparent_rgb_and_fine_texture(self) -> None:
        height, width = 180, 220
        yy, xx = np.indices((height, width))
        rgba = np.zeros((height, width, 4), np.uint8)
        rgba[..., :3] = (13, 29, 47)
        body = ((xx - 110) ** 2 / 88**2 + (yy - 92) ** 2 / 70**2) <= 1
        checker = ((xx // 2 + yy // 2) % 2) * 34 - 17
        base = np.stack(
            [118 + checker, 92 + checker, 63 + checker], axis=-1,
        )
        rgba[..., :3][body] = np.clip(base[body], 0, 255).astype(np.uint8)
        rgba[..., 3][body] = 255

        output = low_frequency_harmonize(
            rgba,
            (0.82, 0.88, 1.04),
            shadow_direction=(0.6, 0.8),
        )

        self.assertTrue(np.array_equal(output[..., 3], rgba[..., 3]))
        self.assertTrue(np.array_equal(output[..., :3][~body], rgba[..., :3][~body]))
        eroded = cv2.erode(body.astype(np.uint8), np.ones((21, 21), np.uint8)) > 0
        source_hp = rgba[..., :3].astype(np.float32) - cv2.GaussianBlur(
            rgba[..., :3].astype(np.float32), (0, 0), 2.0,
        )
        output_hp = output[..., :3].astype(np.float32) - cv2.GaussianBlur(
            output[..., :3].astype(np.float32), (0, 0), 2.0,
        )
        hp_error = float(np.sqrt(np.mean((output_hp[eroded] - source_hp[eroded]) ** 2)))
        self.assertLess(hp_error, 0.8)
        self.assertGreater(float(np.mean(np.abs(
            output[..., :3][eroded].astype(np.float32)
            - rgba[..., :3][eroded].astype(np.float32)
        ))), 3.0)

    def test_rejects_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            low_frequency_harmonize(np.zeros((4, 4, 3), np.uint8))
        with self.assertRaises(ValueError):
            low_frequency_harmonize(
                np.zeros((4, 4, 4), np.uint8), (1.0, float("nan"), 1.0),
            )


if __name__ == "__main__":
    unittest.main()
