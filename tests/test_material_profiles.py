from __future__ import annotations

import unittest

import cv2
import numpy as np

from strict_compositor.material_profiles import (
    canonical_material_mode,
    is_opaque,
    is_transmissive,
    material_profile,
    stabilise_opaque_alpha,
)
from strict_compositor.materials import reinforce_transmissive_readability


class MaterialProfileTests(unittest.TestCase):
    def test_aliases_route_to_the_expected_commercial_profiles(self) -> None:
        aliases = {
            " solid ": "opaque",
            "MATTE": "opaque_matte",
            "ceramic": "opaque_glossy",
            "glossy_dielectric": "opaque_glossy",
            "metallic": "metal",
            "paper": "wood_paper_fabric",
            "wood": "wood_paper_fabric",
            "textile": "wood_paper_fabric",
            "clear_glass": "clear_glass",
            "mixed_transparent": "mixed_transparent",
            "unrecognised-material": "opaque",
        }

        for supplied, expected in aliases.items():
            with self.subTest(supplied=supplied):
                self.assertEqual(canonical_material_mode(supplied), expected)
                self.assertEqual(
                    material_profile(supplied), material_profile(expected)
                )

        self.assertTrue(is_transmissive("clear_glass"))
        self.assertGreaterEqual(material_profile("clear_glass").contact_gain, 0.75)
        self.assertTrue(is_transmissive("mixed_transparent"))
        self.assertFalse(is_transmissive("ceramic"))
        self.assertTrue(is_opaque("metallic"))
        self.assertTrue(is_opaque("paper"))

        # The route must have a visible effect on rendering policy rather than
        # reducing all opaque products to the same legacy profile.
        self.assertGreater(
            material_profile("metal").reflection_gain,
            material_profile("opaque_matte").reflection_gain,
        )
        self.assertLess(
            material_profile("paper").reflection_gain,
            material_profile("ceramic").reflection_gain,
        )

    def test_opaque_alpha_strengthens_only_the_core_and_preserves_holes(self) -> None:
        alpha = np.zeros((25, 25), np.uint8)
        alpha[3:22, 3:22] = 112
        # Genuine negative space in a handle/ring must remain transparent.
        alpha[10:15, 10:15] = 0
        original = alpha.copy()

        strengthened = stabilise_opaque_alpha(alpha, "ceramic")

        self.assertTrue(np.array_equal(alpha, original))
        self.assertTrue(np.all(strengthened[10:15, 10:15] == 0))
        self.assertTrue(np.all(strengthened[:3, :] == 0))
        self.assertTrue(np.all(strengthened[:, :3] == 0))
        # The one-pixel silhouette band remains anti-aliased rather than being
        # expanded or snapped to an opaque rectangle.
        self.assertEqual(int(strengthened[3, 7]), int(original[3, 7]))
        self.assertEqual(int(strengthened[9, 12]), int(original[9, 12]))
        # A point well inside the measured solid region is the only area that
        # should receive meaningful coverage reinforcement.
        self.assertGreater(int(strengthened[7, 7]), int(original[7, 7]))
        self.assertTrue(np.all(strengthened[original == 0] == 0))

        # Transmissive mattes are optical evidence and must never be hardened by
        # the opaque-material helper.
        self.assertTrue(
            np.array_equal(
                stabilise_opaque_alpha(original, "clear_glass"), original
            )
        )

    def test_clear_glass_readability_is_local_and_never_a_white_body_fill(self) -> None:
        height = width = 128
        alpha = np.zeros((height, width), np.uint8)
        alpha[20:108, 28:100] = 255

        source = np.full((height, width, 3), 80, np.uint8)
        source[20:108, 28:100] = 70
        for x in (38, 50, 62, 74, 86):
            source[20:108, x:x + 2] = 150

        target = np.full((height, width, 3), 42, np.uint8)
        reconstructed = target.copy()
        source_before = source.copy()
        reconstructed_before = reconstructed.copy()

        enhanced = reinforce_transmissive_readability(
            reconstructed,
            source,
            target,
            alpha,
            "clear_glass",
        )

        active = alpha > 4
        delta = (
            enhanced.astype(np.int16) - reconstructed_before.astype(np.int16)
        )
        luma_delta = delta[..., 0]

        self.assertTrue(np.array_equal(source, source_before))
        self.assertTrue(np.array_equal(reconstructed, reconstructed_before))
        self.assertTrue(
            np.array_equal(enhanced[~active], reconstructed_before[~active])
        )
        # The visibility pass is a bounded contrast treatment.  The current
        # catalog-view contract permits a sparse +14 highlight so a clear
        # flute survives browser downsampling, while the area/mean gates below
        # still forbid a broad white or milky alpha-shaped body fill.
        self.assertLessEqual(int(luma_delta[active].max()), 14)
        # Repeated fluting legitimately occupies more area than a single rim;
        # still require that most of the transparent body remains untouched.
        self.assertLess(float(np.mean(luma_delta[active] > 8)), 0.22)
        self.assertLess(float(enhanced[active].mean()), 55.0)
        self.assertLess(int(enhanced[active].max()), 80)

        # Nevertheless, a measured rim and a measured vertical flute remain
        # more legible than a flat interior sample at browser viewing scale.
        self.assertGreater(
            int(enhanced[60, 28, 0]), int(enhanced[60, 70, 0])
        )
        self.assertGreater(
            int(enhanced[60, 74, 0]), int(enhanced[60, 70, 0])
        )

    def test_weak_upper_ribs_survive_a_strong_base_and_display_downscale(self) -> None:
        height, width = 240, 160
        alpha = np.zeros((height, width), np.uint8)
        alpha[18:222, 30:130] = 255

        # The upper flutes are intentionally subtle while the thick base is
        # much stronger.  A whole-object percentile used to let the base hide
        # the upper structure completely.
        source = np.full((height, width, 3), 112, np.uint8)
        source[18:222, 30:130] = 108
        for x in (42, 56, 70, 84, 98, 112):
            source[30:168, x:x + 3] = 121
        for x in range(34, 128, 8):
            source[190:218, x:x + 4] = 190

        target = np.full((height, width, 3), 208, np.uint8)
        reconstructed = target.copy()
        enhanced = reinforce_transmissive_readability(
            reconstructed, source, target, alpha, "clear_glass",
        )

        self.assertTrue(np.array_equal(enhanced[alpha == 0], target[alpha == 0]))
        delta = enhanced.astype(np.int16) - target.astype(np.int16)
        active = alpha > 4
        self.assertLessEqual(int(delta[active].max()), 14)
        # On a bright receiver, measured negative flute faces need materially
        # more headroom than highlights or the vessel collapses into pale line
        # art after display downsampling.  The wider range remains sparse and
        # source-high-pass-gated; the median/body checks below still reject a
        # broad smoked fill.
        self.assertGreaterEqual(int(delta[active].min()), -39)
        self.assertLessEqual(float(np.median(np.abs(delta[..., 0][active]))), 2.0)

        small = cv2.resize(enhanced, (width // 4, height // 4), interpolation=cv2.INTER_AREA)
        small_target = cv2.resize(target, (width // 4, height // 4), interpolation=cv2.INTER_AREA)
        small_alpha = cv2.resize(alpha, (width // 4, height // 4), interpolation=cv2.INTER_AREA)
        yy = np.indices(small_alpha.shape)[0]
        upper = (small_alpha > 220) & (yy >= 8) & (yy < 42)
        display_difference = np.abs(
            small[..., 0].astype(np.int16) - small_target[..., 0].astype(np.int16)
        )[upper]
        self.assertGreaterEqual(float(np.percentile(display_difference, 82.0)), 3.0)

    def test_bright_receiver_preserves_alternating_dark_glass_flutes(self) -> None:
        """A white tabletop must not erase the negative half of a rib phase."""
        height, width = 220, 160
        alpha = np.zeros((height, width), np.uint8)
        alpha[18:204, 28:132] = 255

        source = np.full((height, width, 3), 176, np.uint8)
        source[18:204, 28:132] = 174
        for rib_x in range(38, 124, 12):
            source[28:194, rib_x:rib_x + 3] = 226
            source[28:194, rib_x + 5:rib_x + 8] = 112

        target = np.full((height, width, 3), [232, 229, 221], np.uint8)
        enhanced = reinforce_transmissive_readability(
            target, source, target, alpha, "clear_glass",
        )
        delta = enhanced[..., 0].astype(np.int16) - target[..., 0].astype(np.int16)
        core = np.zeros_like(alpha, bool)
        core[36:186, 36:124] = True

        # The receiver remains the dominant body appearance; visibility comes
        # from localized, alternating optical faces rather than a grey fill.
        self.assertLess(abs(float(np.mean(delta[core]))), 3.5)
        self.assertLess(float(np.mean(np.abs(delta[core]) > 18)), 0.24)
        self.assertLessEqual(int(np.percentile(delta[core], 5)), -24)
        self.assertGreaterEqual(int(np.percentile(delta[core], 95)), 9)

        preview = cv2.resize(
            enhanced, (width // 4, height // 4), interpolation=cv2.INTER_AREA,
        )
        preview_target = cv2.resize(
            target, (width // 4, height // 4), interpolation=cv2.INTER_AREA,
        )
        preview_delta = (
            preview[..., 0].astype(np.int16)
            - preview_target[..., 0].astype(np.int16)
        )
        preview_core = preview_delta[9:46, 9:31]
        self.assertGreater(
            float(np.percentile(preview_core, 90)
                  - np.percentile(preview_core, 10)),
            12.0,
        )
        self.assertTrue(np.array_equal(enhanced[alpha == 0], target[alpha == 0]))

    def test_commercial_visibility_scales_readability_without_expanding_alpha(
        self,
    ) -> None:
        height = width = 140
        alpha = np.zeros((height, width), np.uint8)
        cv2.ellipse(alpha, (78, 70), (48, 55), 0, 0, 360, 255, -1)
        cv2.ellipse(alpha, (78, 70), (25, 32), 0, 0, 360, 0, -1)
        source = np.full((height, width, 3), 100, np.uint8)
        for rib_x in range(40, 116, 9):
            source[20:122, rib_x:rib_x + 2] = 145
        target = np.full((height, width, 3), 180, np.uint8)

        low = reinforce_transmissive_readability(
            target,
            source,
            target,
            alpha,
            "clear_glass",
            visibility_strength=0.20,
        )
        high = reinforce_transmissive_readability(
            target,
            source,
            target,
            alpha,
            "clear_glass",
            visibility_strength=1.0,
        )
        active = alpha > 4
        low_delta = np.abs(
            low[..., 0].astype(np.int16) - target[..., 0].astype(np.int16)
        )
        high_delta = np.abs(
            high[..., 0].astype(np.int16) - target[..., 0].astype(np.int16)
        )

        self.assertGreater(
            float(np.mean(high_delta[active])),
            float(np.mean(low_delta[active])) + 0.8,
        )
        self.assertTrue(np.array_equal(high[~active], target[~active]))
        self.assertTrue(np.array_equal(low[~active], target[~active]))
        # The enclosed centre is a genuine hole, not a weakly visible glass
        # region; changing commercial visibility cannot authorise any pixels.
        self.assertTrue(np.array_equal(high[55:85, 65:91], target[55:85, 65:91]))

        with self.assertRaisesRegex(ValueError, "visibility_strength"):
            reinforce_transmissive_readability(
                target,
                source,
                target,
                alpha,
                "clear_glass",
                visibility_strength=-0.1,
            )


if __name__ == "__main__":
    unittest.main()
