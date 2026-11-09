from __future__ import annotations

import unittest

import numpy as np

from strict_compositor.contracts import Lighting
from strict_compositor.materials import (
    _clear_glass_refract_target,
    reconstruct_transmissive_product,
)
from strict_compositor.identity import warp_identity_mask
from strict_compositor.render import (
    cast_shadow,
    prepare_source_identity_detail,
    reinforce_scaled_identity_detail,
    reflection,
    render_product,
    transform_rgba,
)


class OpticalMaterialTests(unittest.TestCase):
    def test_scaled_identity_detail_is_bounded_and_mask_local(self) -> None:
        rgb = np.full((90, 120, 3), 92, np.uint8)
        alpha = np.zeros((90, 120), np.uint8)
        alpha[12:80, 18:104] = 255
        identity = np.zeros_like(alpha)
        identity[28:68, 32:92] = 255
        rgb[34:62:8, 38:86] = 184
        rgb[38:66:8, 38:86] = 34
        homography = np.asarray(
            [[0.26, 0.0, 0.0], [0.0, 0.26, 0.0], [0.0, 0.0, 1.0]],
            np.float64,
        )

        enhanced = reinforce_scaled_identity_detail(
            rgb, alpha, homography, identity,
        )

        delta = enhanced.astype(np.int16) - rgb.astype(np.int16)
        self.assertTrue(np.array_equal(enhanced[identity == 0], rgb[identity == 0]))
        self.assertLessEqual(int(np.abs(delta).max()), 14)
        before = float(rgb[34:62:8, 38:86, 0].mean() - rgb[38:66:8, 38:86, 0].mean())
        after = float(
            enhanced[34:62:8, 38:86, 0].mean()
            - enhanced[38:66:8, 38:86, 0].mean()
        )
        self.assertGreater(after, before)

    def test_source_identity_detail_is_local_and_survives_downscale(self) -> None:
        rgba = np.zeros((240, 180, 4), np.uint8)
        rgba[12:228, 18:162, :3] = 88
        rgba[12:228, 18:162, 3] = 255
        identity = np.zeros((240, 180), np.uint8)
        identity[62:186, 34:146] = 255
        for y in range(76, 176, 18):
            rgba[y:y + 5, 44:136, :3] = 205
            rgba[y + 7:y + 11, 44:136, :3] = 28
        matrix = np.asarray(
            [[0.22, 0.0, 0.0], [0.0, 0.22, 0.0], [0.0, 0.0, 1.0]],
            np.float64,
        )
        prepared = prepare_source_identity_detail(rgba, matrix, identity)
        delta = prepared[..., :3].astype(np.int16) - rgba[..., :3].astype(np.int16)
        self.assertTrue(np.array_equal(prepared[identity == 0], rgba[identity == 0]))
        self.assertLessEqual(int(np.abs(delta).max()), 24)

        before, _ = transform_rgba(rgba, matrix, (60, 50))
        after, _ = transform_rgba(prepared, matrix, (60, 50))
        before_luma = before.astype(np.float32).mean(axis=2)
        after_luma = after.astype(np.float32).mean(axis=2)
        self.assertGreater(float(after_luma.std()), float(before_luma.std()))

    def test_explicit_identity_survives_clear_glass_readability_pass(self) -> None:
        height, width = 150, 170
        rgb = np.zeros((height, width, 3), np.uint8)
        alpha = np.zeros((height, width), np.uint8)
        alpha[12:138, 30:140] = 190
        rgb[12:138, 30:140] = [28, 34, 42]
        # An opaque printed identity patch inside an otherwise clear vessel.
        identity = np.zeros((height, width), np.uint8)
        identity[62:88, 52:118] = 255
        alpha[62:88, 52:118] = 255
        rgb[62:88, 52:118] = [24, 32, 205]
        rgb[68:72, 60:110] = [242, 235, 218]
        target = np.full_like(rgb, [198, 174, 132])

        result = reconstruct_transmissive_product(
            rgb,
            alpha,
            target,
            "clear_glass",
            0.92,
            0.96,
            identity,
        )

        protected = identity > 0
        np.testing.assert_array_equal(result.rgb[protected], rgb[protected])
        self.assertIsNotNone(result.optical_matte)
        assert result.optical_matte is not None
        x0, y0, x1, y1 = result.optical_matte.bbox
        protected_roi = protected[y0:y1, x0:x1]
        np.testing.assert_allclose(
            result.optical_matte.opaque_identity_mask[protected_roi],
            result.optical_matte.coverage_alpha[protected_roi],
        )
        np.testing.assert_allclose(
            result.optical_matte.shadow_transmittance[protected_roi],
            0.0,
        )
        self.assertTrue(np.any(result.rgb[(alpha > 0) & ~protected] != rgb[(alpha > 0) & ~protected]))

    def test_render_product_warps_source_identity_and_restores_strict_baseline(
        self,
    ) -> None:
        source = np.zeros((92, 74, 4), np.uint8)
        source[5:87, 8:66, :3] = [32, 38, 45]
        source[5:87, 8:66, 3] = 188
        source_identity = np.zeros(source.shape[:2], np.uint8)
        source_identity[40:58, 18:56] = 255
        source[40:58, 18:56, :3] = [160, 24, 52]
        source[40:58, 18:56, 3] = 255
        source[45:49, 23:51, :3] = [245, 232, 210]

        shape = (170, 210)
        homography = np.asarray(
            [[1.18, 0.0, 54.0], [0.0, 1.18, 31.0], [0.0, 0.0, 1.0]],
            np.float64,
        )
        background = np.full((*shape, 3), [202, 176, 126], np.uint8)
        support = np.ones(shape, bool)
        protected = np.zeros(shape, bool)
        lighting = Lighting(
            direction=(0.8, 0.6),
            shadow_length_ratio=0.0,
            cast_shadow_mode="ambient_only",
        )

        baseline_rgb, baseline_alpha = transform_rgba(
            source, homography, shape, "clear_glass",
        )
        placed_identity = warp_identity_mask(
            source_identity, homography, shape, baseline_alpha,
        ) > 0
        rendered = render_product(
            source,
            homography,
            shape,
            background,
            support,
            protected,
            lighting,
            material_mode="clear_glass",
            material_confidence=0.96,
            transmission_strength=0.92,
            source_identity_mask=source_identity,
        )

        np.testing.assert_array_equal(
            rendered.rgb[placed_identity], baseline_rgb[placed_identity],
        )
        self.assertIsNotNone(rendered.optical_matte)
        assert rendered.optical_matte is not None
        self.assertTrue(np.any(rendered.optical_matte.opaque_identity_mask > 0))
        interior = (rendered.alpha > 32) & ~placed_identity
        self.assertTrue(np.any(rendered.rgb[interior] != baseline_rgb[interior]))

        with self.assertRaisesRegex(ValueError, "source_identity_mask"):
            render_product(
                source,
                homography,
                shape,
                background,
                support,
                protected,
                lighting,
                material_mode="clear_glass",
                material_confidence=0.96,
                transmission_strength=0.92,
                source_identity_mask=np.zeros((10, 10), np.uint8),
            )

    def test_clear_glass_uses_target_low_frequency_and_keeps_alpha(self) -> None:
        height, width = 120, 140
        rgb = np.zeros((height, width, 3), np.uint8)
        alpha = np.zeros((height, width), np.uint8)
        alpha[16:108, 36:104] = 255
        # Trapped source-room darkness plus an intrinsic fine glass rib.
        rgb[16:108, 36:104] = [9, 11, 14]
        rgb[16:108, 68:71] = [220, 225, 230]
        target = np.full((height, width, 3), [194, 166, 122], np.uint8)

        result = reconstruct_transmissive_product(
            rgb, alpha, target, "clear_glass", 0.92, 0.96,
        )

        self.assertTrue(np.array_equal(result.coverage_alpha, alpha))
        self.assertGreater(float(result.rgb[60, 52].mean()), 95.0)
        # The narrow original rib remains distinctly brighter than its local
        # target-aware transparent interior.  The contrast is bounded because
        # Fresnel reflection is now energy-conserving rather than an additive
        # white-light term.
        self.assertGreater(
            float(result.rgb[60, 69].mean()) - float(result.rgb[60, 61].mean()),
            14.0,
        )
        self.assertGreater(result.source_environment_replacement_ratio, 0.55)

    def test_semantically_clear_glass_at_point_seven_uses_verified_optical_route(
        self,
    ) -> None:
        """A07-class evidence must not silently fall back to source-room RGB."""
        height, width = 132, 148
        rgb = np.full((height, width, 3), [54, 49, 43], np.uint8)
        alpha = np.zeros((height, width), np.uint8)
        alpha[12:122, 30:112] = 232
        for x in range(38, 108, 10):
            rgb[12:122, x:x + 2] = [205, 208, 212]
        target = np.full_like(rgb, [194, 168, 126])

        result = reconstruct_transmissive_product(
            rgb,
            alpha,
            target,
            "clear_glass",
            transmission_strength=0.97,
            material_confidence=0.70,
        )

        self.assertIsNotNone(result.optical_matte)
        assert result.optical_matte is not None
        upstream = result.optical_matte.provenance["upstream"]
        self.assertEqual(upstream["material_route"], "verified_optical_matte")
        self.assertEqual(
            upstream["material_route_reason"],
            "clear_glass_confidence_and_transmission",
        )
        self.assertEqual(upstream["commercial_visibility"], 1.0)
        self.assertGreater(result.source_environment_replacement_ratio, 0.80)

        # A genuinely weak classification remains on the conservative fallback
        # unless the semantic router explicitly verifies it.
        fallback = reconstruct_transmissive_product(
            rgb,
            alpha,
            target,
            "clear_glass",
            transmission_strength=0.97,
            material_confidence=0.64,
        )
        self.assertIsNone(fallback.optical_matte)

        explicit = reconstruct_transmissive_product(
            rgb,
            alpha,
            target,
            "clear_glass",
            transmission_strength=0.97,
            material_confidence=0.55,
            material_route_verified=True,
            commercial_visibility=0.90,
        )
        self.assertIsNotNone(explicit.optical_matte)
        assert explicit.optical_matte is not None
        self.assertEqual(
            explicit.optical_matte.provenance["upstream"]["material_route_reason"],
            "explicit_semantic_verification",
        )
        self.assertEqual(
            explicit.optical_matte.provenance["upstream"]["commercial_visibility"],
            0.90,
        )

    def test_opaque_mode_is_bit_exact(self) -> None:
        rng = np.random.default_rng(4)
        rgb = rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
        alpha = rng.integers(0, 256, (48, 64), dtype=np.uint8)
        target = rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
        result = reconstruct_transmissive_product(
            rgb, alpha, target, "opaque", 0.9, 1.0,
        )
        self.assertTrue(np.array_equal(result.rgb, rgb))
        self.assertTrue(np.array_equal(result.coverage_alpha, alpha))
        self.assertTrue(np.array_equal(result.shadow_occlusion_alpha, alpha))

    def test_clear_glass_cast_occlusion_is_lower_than_coverage(self) -> None:
        rgb = np.full((80, 80, 3), 80, np.uint8)
        alpha = np.zeros((80, 80), np.uint8)
        alpha[10:72, 20:60] = 255
        target = np.full_like(rgb, 170)
        result = reconstruct_transmissive_product(
            rgb, alpha, target, "clear_glass", 0.9, 1.0,
        )
        self.assertLess(
            int(result.shadow_occlusion_alpha[30:55, 30:50].mean()),
            80,
        )
        self.assertGreater(
            int(result.shadow_occlusion_alpha[10:14, 20:60].mean()),
            int(result.shadow_occlusion_alpha[30:55, 30:50].mean()),
        )

    def test_clear_glass_replaces_source_band_and_boundary_matte(self) -> None:
        rgb = np.zeros((140, 120, 3), np.uint8)
        alpha = np.zeros((140, 120), np.uint8)
        alpha[12:128, 24:96] = 255
        rgb[12:128, 24:96] = [176, 168, 154]
        # A broad source-room shelf/table line must not survive as intrinsic
        # black material.  Narrow positive ribs are actual product detail.
        rgb[68:98, 24:96] = [7, 9, 12]
        rgb[12:128, 42:45] = [225, 231, 237]
        rgb[12:128, 70:73] = [218, 225, 232]
        # Simulate a contaminated bright source matte at the silhouette.
        rgb[12:128, 24:27] = [248, 244, 235]
        target = np.full_like(rgb, [188, 156, 112])

        result = reconstruct_transmissive_product(
            rgb, alpha, target, "clear_glass", 0.90, 0.92,
        )

        clean_luma = float(result.rgb[45, 55].mean())
        band_luma = float(result.rgb[82, 55].mean())
        self.assertLess(clean_luma - band_luma, 26.0)
        self.assertGreater(band_luma, 126.0)
        target_luma = float(target[45, 25].mean())
        self.assertLess(
            abs(float(result.rgb[45, 25].mean()) - target_luma),
            abs(float(rgb[45, 25].mean()) - target_luma),
        )
        self.assertGreater(result.source_environment_replacement_ratio, 0.82)
        # A verified neutral clear vessel must also reject low-frequency matte
        # colour immediately inside the contour; the measured high-frequency
        # rim remains source-derived through the separate residual path.
        self.assertLess(
            abs(float(result.rgb[45, 25].mean()) - target_luma),
            28.0,
        )

    def test_high_confidence_clear_glass_has_visible_but_weak_reflection(self) -> None:
        rgb = np.full((96, 96, 3), 146, np.uint8)
        alpha = np.zeros((96, 96), np.uint8)
        alpha[8:88, 20:76] = 255
        target = np.full_like(rgb, 184)

        result = reconstruct_transmissive_product(
            rgb, alpha, target, "clear_glass", 0.72, 0.94,
        )

        interior_reflection = float(
            result.reflection_detail_alpha[40:60, 38:58].mean()
        )
        self.assertGreater(interior_reflection, 85.0)
        # The matte is multiplied by the receiver's reflection opacity later;
        # a roughly half-strength detail mask remains a faint glass reflection
        # in the final image while surviving a 4K browser downsample.
        self.assertLess(interior_reflection, 150.0)
        self.assertGreater(result.source_environment_replacement_ratio, 0.94)

    def test_clear_glass_keeps_vertical_ribs_but_rejects_horizontal_source_shelf(self) -> None:
        height, width = 180, 150
        rgb = np.zeros((height, width, 3), np.uint8)
        alpha = np.zeros((height, width), np.uint8)
        alpha[12:168, 24:126] = 255
        rgb[12:168, 24:126] = [178, 161, 132]
        # A dark horizontal source shelf crosses the body.  Bright thin vertical
        # ribs are intrinsic glass detail and must remain more visible than the
        # shelf after target-aware reconstruction.
        rgb[78:118, 24:126] = [12, 12, 12]
        for x in (42, 58, 76, 94, 112):
            rgb[12:168, x:x + 2] = [238, 240, 245]
        target = np.full_like(rgb, [202, 178, 146])

        result = reconstruct_transmissive_product(
            rgb, alpha, target, "clear_glass", 0.94, 0.98,
        )

        shelf = float(result.rgb[96, 68].mean())
        nearby_clear = float(result.rgb[54, 68].mean())
        rib = float(result.rgb[54, 76].mean())
        self.assertLess(nearby_clear - shelf, 22.0)
        self.assertGreater(rib - nearby_clear, 18.0)

    def test_clear_glass_rejects_local_diagonal_source_band(self) -> None:
        """A refracted old-room boundary is not persistent glass identity."""
        height, width = 220, 170
        rgb = np.zeros((height, width, 3), np.uint8)
        alpha = np.zeros((height, width), np.uint8)
        alpha[14:206, 28:142] = 255
        rgb[14:206, 28:142] = [172, 160, 142]
        # A broad diagonal wall/table boundary is local in Y but varies across
        # X, so a horizontal high-pass alone would incorrectly retain it.
        for x in range(28, 142):
            centre = 78 + int(round((x - 28) * 0.32))
            rgb[centre - 12:centre + 13, x] = [18, 20, 22]
        # True flutes persist through most of the vessel height.
        for x in (44, 62, 80, 98, 116, 134):
            rgb[14:206, x:x + 2] = [232, 236, 240]
        target = np.full_like(rgb, [202, 180, 150])

        result = reconstruct_transmissive_product(
            rgb, alpha, target, "clear_glass", 0.97, 0.98,
        )

        # Away from a true flute, the old diagonal band must collapse toward
        # the same transmitted target seen above it.
        x = 72
        band_y = 78 + int(round((x - 28) * 0.32))
        clear_luma = float(result.rgb[46, x].mean())
        band_luma = float(result.rgb[band_y, x].mean())
        self.assertLess(clear_luma - band_luma, 18.0)
        # The vertically coherent measured flute remains legible.
        rib_luma = float(result.rgb[46, 80].mean())
        nearby_luma = float(result.rgb[46, 86].mean())
        self.assertGreater(rib_luma - nearby_luma, 12.0)

    def test_clear_glass_remaps_coloured_source_fringe_to_neutral_thin_highlight(self) -> None:
        """A source-room gold rim is not an intrinsic colour of clear glass."""
        height, width = 150, 138
        rgb = np.zeros((height, width, 3), np.uint8)
        alpha = np.zeros((height, width), np.uint8)
        alpha[14:136, 28:110] = 255
        rgb[14:136, 28:110] = [118, 122, 126]
        # Deliberately coloured, several-pixel source matte plus a thin bright
        # centre rib.  Geometry and brightness should remain, but its hue must
        # not be pasted into a neutral target scene.
        rgb[14:136, 28:33] = [252, 176, 38]
        rgb[14:136, 68:70] = [236, 188, 66]
        target = np.full_like(rgb, [154, 158, 162])

        result = reconstruct_transmissive_product(
            rgb, alpha, target, "clear_glass", 0.97, 0.96,
        )

        interior = result.rgb[72, 52].astype(np.float32)
        edge = result.rgb[72, 28].astype(np.float32)
        rib = result.rgb[72, 68].astype(np.float32)
        edge_excess = edge - interior
        rib_excess = rib - interior
        # Both measured features remain visibly brighter than the transmitted
        # target, while their excess is essentially neutral rather than the
        # orange RGB ratio of the source room.
        # On a uniform target an energy-conserving clear interface is subtle;
        # stronger contrast comes from the target environment band rather than
        # a fabricated additive white halo.
        self.assertGreater(float(edge_excess.mean()), 4.0)
        self.assertGreater(float(rib_excess.mean()), 12.0)
        self.assertLess(float(np.ptp(edge_excess)), 18.0)
        self.assertLess(float(np.ptp(rib_excess)), 18.0)
        # The broad five-pixel matte must collapse toward a thinner highlight:
        # one step inward is materially weaker than the actual alpha contour.
        inner_edge_excess = (
            result.rgb[72, 32].astype(np.float32) - interior
        )
        self.assertLess(
            float(inner_edge_excess.mean()),
            float(edge_excess.mean()) * 0.82,
        )

    def test_clear_glass_retains_dark_vertical_ribs_without_source_shelf(self) -> None:
        height, width = 180, 150
        rgb = np.zeros((height, width, 3), np.uint8)
        alpha = np.zeros((height, width), np.uint8)
        alpha[12:168, 24:126] = 255
        rgb[12:168, 24:126] = [188, 170, 145]
        # The source has a dark room shelf across the lower body and dark
        # fluting at a few X coordinates.  The reconstruction must remove the
        # row-constant shelf while retaining the physically measured rib contrast.
        rgb[102:138, 24:126] = [10, 10, 10]
        for x in (38, 54, 72, 90, 108):
            rgb[12:168, x:x + 3] = [42, 44, 48]
        target = np.full_like(rgb, [214, 183, 142])

        result = reconstruct_transmissive_product(
            rgb, alpha, target, "clear_glass", 0.94, 0.98,
        )

        clear = float(result.rgb[58, 64].mean())
        dark_rib = float(result.rgb[58, 72].mean())
        shelf = float(result.rgb[120, 64].mean())
        self.assertGreater(clear - dark_rib, 10.0)
        self.assertLess(clear - shelf, 24.0)

    def test_clear_glass_target_refraction_is_subtle_and_stays_inside_coverage(self) -> None:
        height, width = 120, 140
        target = np.empty((height, width, 3), np.float32)
        x_gradient = np.arange(width, dtype=np.float32)[None, :]
        target[..., 0] = 72.0 + x_gradient
        target[..., 1] = 92.0 + x_gradient * 0.72
        target[..., 2] = 106.0 + x_gradient * 0.48
        # A real tabletop edge crossing the future vessel gives the refraction
        # something target-derived to bend; no source pixels are involved.
        target[66:, :] = [76, 96, 92]
        target[:, 66:70] = [35, 40, 45]
        alpha = np.zeros((height, width), np.uint8)
        alpha[14:108, 38:102] = 255
        boundary_keep = np.zeros((height, width), np.float32)
        boundary_keep[14:108, 38:102] = 0.12
        boundary_keep[14:18, 38:102] = 1.0
        boundary_keep[104:108, 38:102] = 1.0
        boundary_keep[14:108, 38:42] = 1.0
        boundary_keep[14:108, 98:102] = 1.0

        refracted = _clear_glass_refract_target(target, alpha, boundary_keep)

        self.assertTrue(np.array_equal(refracted[alpha == 0], target[alpha == 0]))
        # The center stays on the optical axis while the inner side of the
        # target edge moves by a few pixels, proving this is not a flat alpha
        # overlay.  The bounded distortion remains far below the 64px vessel
        # width.
        self.assertLess(
            float(np.abs(refracted[60, 68] - target[60, 68]).mean()),
            0.25,
        )
        self.assertGreater(
            float(np.abs(refracted[60, 52] - target[60, 52]).mean()),
            0.5,
        )

    def test_clear_glass_rib_phase_bends_only_target_and_keeps_outside_exact(self) -> None:
        height, width = 124, 156
        target = np.empty((height, width, 3), np.float32)
        gradient = np.arange(width, dtype=np.float32)[None, :]
        target[..., 0] = 62.0 + gradient * 1.20
        target[..., 1] = 78.0 + gradient * 0.80
        target[..., 2] = 94.0 + gradient * 0.45
        alpha = np.zeros((height, width), np.uint8)
        alpha[14:112, 38:118] = 255
        boundary_keep = np.zeros((height, width), np.float32)
        boundary_keep[14:112, 38:118] = 0.10
        boundary_keep[14:18, 38:118] = 1.0
        boundary_keep[108:112, 38:118] = 1.0
        boundary_keep[14:112, 38:42] = 1.0
        boundary_keep[14:112, 114:118] = 1.0
        # Alternating vertical phases model measured fluting.  These are an
        # optical displacement cue only; target RGB remains the sole sampled
        # scene content.
        ribs = np.zeros((height, width), np.float32)
        for x, value in ((52, -19.0), (61, 17.0), (73, -21.0), (84, 19.0), (98, -18.0)):
            ribs[18:108, x:x + 2] = value

        smooth = _clear_glass_refract_target(target, alpha, boundary_keep)
        fluted = _clear_glass_refract_target(target, alpha, boundary_keep, ribs)

        self.assertTrue(np.array_equal(fluted[alpha == 0], target[alpha == 0]))
        self.assertGreater(
            float(np.abs(fluted[44:96, 48:108] - smooth[44:96, 48:108]).mean()),
            0.06,
        )
        # The measured rib cue is intentionally bounded: it may reveal glass
        # mass but cannot become a large invented warp.
        self.assertLess(
            float(np.abs(fluted - smooth).max()), 12.0,
        )

    def test_clear_glass_rib_phase_remains_visible_on_a_quiet_target(self) -> None:
        """Fluting must not disappear merely because the wall is low-detail.

        The measured rib values are geometry / optical-field evidence only.
        The resulting RGB remains a bounded modulation of the target colour,
        not a leak of the photographed source room.
        """
        height, width = 112, 144
        target = np.full((height, width, 3), [188.0, 151.0, 108.0], np.float32)
        alpha = np.zeros((height, width), np.uint8)
        alpha[12:102, 36:108] = 255
        boundary_keep = np.zeros((height, width), np.float32)
        boundary_keep[12:102, 36:108] = 0.10
        boundary_keep[12:16, 36:108] = 1.0
        boundary_keep[98:102, 36:108] = 1.0
        boundary_keep[12:102, 36:40] = 1.0
        boundary_keep[12:102, 104:108] = 1.0
        ribs = np.zeros((height, width), np.float32)
        ribs[18:96, 57:60] = 22.0
        ribs[18:96, 75:78] = -22.0

        smooth = _clear_glass_refract_target(target, alpha, boundary_keep)
        fluted = _clear_glass_refract_target(target, alpha, boundary_keep, ribs)

        self.assertTrue(np.array_equal(fluted[alpha == 0], target[alpha == 0]))
        self.assertGreater(
            float(abs(fluted[54, 58].mean() - smooth[54, 58].mean())),
            3.5,
        )
        self.assertLess(
            float(np.abs(fluted - smooth).max()),
            22.0,
        )
        # The modulation is achromatic: it scales target RGB instead of
        # importing the source room's black/brown colour into the vessel.
        expected_ratio = target[54, 58, 1] / target[54, 58, 0]
        observed_ratio = fluted[54, 58, 1] / fluted[54, 58, 0]
        self.assertAlmostEqual(float(observed_ratio), float(expected_ratio), places=4)

    def test_clear_glass_handle_does_not_shift_cylindrical_body_axis(self) -> None:
        """A wide handle must not pull the vessel's optical axis sideways."""
        height, width = 140, 180
        target = np.empty((height, width, 3), np.float32)
        gradient = np.arange(width, dtype=np.float32)[None, :]
        target[..., 0] = 50.0 + gradient
        target[..., 1] = 70.0 + gradient * 0.8
        target[..., 2] = 90.0 + gradient * 0.6
        alpha = np.zeros((height, width), np.uint8)
        alpha[16:128, 38:102] = 255
        # A U-shaped handle extends far to the right but occupies much less
        # vertical height than the principal body.
        alpha[42:48, 102:154] = 255
        alpha[96:102, 102:154] = 255
        alpha[42:102, 148:154] = 255
        boundary_keep = np.zeros((height, width), np.float32)
        boundary_keep[alpha > 0] = 0.08
        boundary_keep[16:20, 38:102] = 1.0
        boundary_keep[124:128, 38:102] = 1.0
        boundary_keep[16:128, 38:42] = 1.0
        boundary_keep[16:128, 98:102] = 1.0

        refracted = _clear_glass_refract_target(target, alpha, boundary_keep)

        # The main body midpoint x=70 remains the optical axis even though the
        # complete silhouette bbox midpoint would be near x=96.
        self.assertLess(
            float(np.abs(refracted[70, 70] - target[70, 70]).mean()),
            0.30,
        )
        self.assertGreater(
            float(np.abs(refracted[70, 55] - target[70, 55]).mean()),
            0.45,
        )

    def test_mixed_transparent_does_not_wash_high_detail_label(self) -> None:
        rgb = np.full((120, 100, 3), [132, 84, 35], np.uint8)
        alpha = np.zeros((120, 100), np.uint8)
        alpha[8:112, 18:82] = 255
        # Opaque dark label with bright fine typography.
        rgb[44:92, 24:76] = [20, 18, 16]
        rgb[52:56, 30:70] = [238, 232, 219]
        rgb[66:70, 30:70] = [238, 232, 219]
        target = np.full_like(rgb, [205, 176, 128])

        result = reconstruct_transmissive_product(
            rgb, alpha, target, "mixed_transparent", 0.48, 0.92,
        )

        np.testing.assert_allclose(
            result.rgb[52:56, 30:70].astype(np.int16),
            rgb[52:56, 30:70].astype(np.int16),
            atol=3,
        )
        np.testing.assert_allclose(
            result.rgb[58:64, 30:70].astype(np.int16),
            rgb[58:64, 30:70].astype(np.int16),
            atol=3,
        )

    def test_ambient_only_disables_long_cast_but_not_contract(self) -> None:
        alpha = np.zeros((100, 100), np.uint8)
        alpha[20:82, 35:65] = 255
        support = np.ones((100, 100), bool)
        protected = np.zeros_like(support)
        lighting = Lighting.from_dict({
            "shadow_direction": [0.8, 0.6],
            "shadow_length_ratio": 0.08,
            "shadow_opacity": 0.0,
            "cast_shadow_mode": "ambient_only",
        })
        self.assertFalse(np.any(cast_shadow(alpha, support, lighting, protected)))

    def test_reflection_colour_reads_target_background(self) -> None:
        rgb = np.zeros((120, 140, 3), np.uint8)
        alpha = np.zeros((120, 140), np.uint8)
        rgb[20:82, 50:90] = [110, 110, 110]
        alpha[20:82, 50:90] = 255
        support = np.zeros((120, 140), bool)
        support[78:118, 10:130] = True
        protected = np.zeros_like(support)
        lighting = Lighting.from_dict({
            "shadow_direction": [0.8, 0.6],
            "shadow_length_ratio": 0.04,
            "reflection_enabled": True,
            "reflection_opacity": 0.30,
            "reflection_roughness": 0.20,
        })
        warm = np.full_like(rgb, [210, 155, 90])
        cool = np.full_like(rgb, [70, 130, 220])
        warm_rgb, warm_alpha = reflection(
            rgb, alpha, warm, support, protected, lighting,
        )
        cool_rgb, cool_alpha = reflection(
            rgb, alpha, cool, support, protected, lighting,
        )
        visible = (warm_alpha > 2) & (cool_alpha > 2)
        self.assertTrue(np.any(visible))
        self.assertGreater(
            float(np.abs(
                warm_rgb[visible].astype(np.float32)
                - cool_rgb[visible].astype(np.float32)
            ).mean()),
            10.0,
        )
        self.assertFalse(np.any(warm_alpha[~support]))

    def test_glossy_receiver_uses_bounded_catalogue_reflection_lift(self) -> None:
        rgb = np.zeros((150, 170, 3), np.uint8)
        alpha = np.zeros((150, 170), np.uint8)
        rgb[20:102, 56:112] = [174, 62, 38]
        alpha[20:102, 56:112] = 255
        support = np.ones((150, 170), bool)
        protected = np.zeros_like(support)
        background = np.full_like(rgb, [116, 105, 84])
        common = {
            "shadow_direction": [0.8, 0.6],
            "reflection_enabled": True,
            "reflection_opacity": 0.28,
            "reflection_roughness": 0.14,
        }
        matte = Lighting.from_dict({**common, "surface_reflectivity": 0.05})
        glossy = Lighting.from_dict({**common, "surface_reflectivity": 0.82})

        _, matte_alpha = reflection(
            rgb, alpha, background, support, protected, matte,
            material_mode="opaque_glossy",
        )
        _, glossy_alpha = reflection(
            rgb, alpha, background, support, protected, glossy,
            material_mode="opaque_glossy",
        )

        self.assertGreater(
            float(glossy_alpha.sum()),
            float(matte_alpha.sum()) * 1.20,
        )
        # The beauty lift remains a translucent effect layer, never an opaque
        # upside-down product copy.
        self.assertLessEqual(int(glossy_alpha.max()), 112)
        self.assertFalse(np.any(glossy_alpha[~support]))

    def test_clear_glass_reflection_preserves_more_optical_contrast(self) -> None:
        rgb = np.full((120, 140, 3), [122, 116, 102], np.uint8)
        alpha = np.zeros((120, 140), np.uint8)
        alpha[18:88, 42:98] = 128
        # Target-derived rib contrast in the reconstructed clear vessel.
        rgb[18:88, 58:62] = [166, 158, 138]
        rgb[18:88, 76:80] = [82, 78, 70]
        support = np.zeros((120, 140), bool)
        support[84:118, 8:132] = True
        protected = np.zeros_like(support)
        background = np.full_like(rgb, [120, 114, 100])
        lighting = Lighting.from_dict({
            "shadow_direction": [0.8, 0.6],
            "reflection_enabled": True,
            "reflection_opacity": 0.34,
            "reflection_roughness": 0.15,
        })

        opaque_rgb, opaque_alpha = reflection(
            rgb, alpha, background, support, protected, lighting,
            material_mode="opaque",
        )
        clear_rgb, clear_alpha = reflection(
            rgb, alpha, background, support, protected, lighting,
            material_mode="clear_glass",
        )
        visible = (opaque_alpha > 2) & (clear_alpha > 2)
        receiver = np.asarray([120, 114, 100], np.float32)
        opaque_contrast = float(np.mean(np.abs(
            opaque_rgb[visible].astype(np.float32) - receiver,
        )))
        clear_contrast = float(np.mean(np.abs(
            clear_rgb[visible].astype(np.float32) - receiver,
        )))
        self.assertGreater(clear_contrast, opaque_contrast * 1.35)

    def test_clear_glass_reflection_cancels_transmitted_backdrop(self) -> None:
        """Changing the wall through glass must not copy that wall to the table."""
        shape = (128, 156)
        alpha = np.zeros(shape, np.uint8)
        alpha[16:91, 48:108] = 150
        support = np.zeros(shape, bool)
        support[88:126, 10:146] = True
        protected = np.zeros(shape, bool)
        lighting = Lighting.from_dict({
            "shadow_direction": [0.75, 0.66],
            "reflection_enabled": True,
            "reflection_opacity": 0.35,
            "reflection_roughness": 0.12,
        })

        cool_background = np.full((*shape, 3), [72, 116, 172], np.uint8)
        warm_background = np.full((*shape, 3), [186, 142, 82], np.uint8)
        cool_glass = cool_background.copy()
        warm_glass = warm_background.copy()
        # The same intrinsic optical ribs are applied to two different rooms.
        cool_glass[16:91, 64:68] = np.clip(
            cool_glass[16:91, 64:68].astype(np.int16) + 20, 0, 255,
        )
        warm_glass[16:91, 64:68] = np.clip(
            warm_glass[16:91, 64:68].astype(np.int16) + 20, 0, 255,
        )
        cool_glass[16:91, 86:90] = np.clip(
            cool_glass[16:91, 86:90].astype(np.int16) - 18, 0, 255,
        )
        warm_glass[16:91, 86:90] = np.clip(
            warm_glass[16:91, 86:90].astype(np.int16) - 18, 0, 255,
        )

        cool_rgb, cool_alpha = reflection(
            cool_glass, alpha, cool_background, support, protected, lighting,
            material_mode="clear_glass",
        )
        warm_rgb, warm_alpha = reflection(
            warm_glass, alpha, warm_background, support, protected, lighting,
            material_mode="clear_glass",
        )
        visible = (cool_alpha > 3) & (warm_alpha > 3)
        self.assertTrue(np.any(visible))
        cool_residual = (
            cool_rgb[visible].astype(np.float32)
            - cool_background[visible].astype(np.float32)
        )
        warm_residual = (
            warm_rgb[visible].astype(np.float32)
            - warm_background[visible].astype(np.float32)
        )
        # Gamma makes the two encoded residuals non-identical, but the broad
        # target colour itself must cancel rather than dominate the difference.
        self.assertLess(
            float(np.mean(np.abs(cool_residual - warm_residual))),
            12.0,
        )

    def test_reflection_mirrors_hanging_parts_about_shared_contact_plane(self) -> None:
        """A raised handle must not acquire its own false tabletop contact."""
        rgb = np.zeros((150, 160, 3), np.uint8)
        alpha = np.zeros((150, 160), np.uint8)
        # Main vessel reaches the physical receiver at y=94.
        rgb[24:95, 48:82] = [118, 136, 152]
        alpha[24:95, 48:82] = 255
        # A side handle hangs well above that receiver.
        rgb[42:68, 82:112] = [142, 156, 170]
        alpha[42:68, 82:112] = 255
        support = np.ones((150, 160), bool)
        protected = np.zeros_like(support)
        lighting = Lighting.from_dict({
            "shadow_direction": [0.75, 0.66],
            "shadow_length_ratio": 0.06,
            "reflection_enabled": True,
            "reflection_opacity": 0.35,
            "reflection_roughness": 0.08,
        })

        _, reflected_alpha = reflection(
            rgb,
            alpha,
            np.full_like(rgb, [172, 148, 116]),
            support,
            protected,
            lighting,
        )

        # Nothing can reflect on the far side of the shared contact plane.
        self.assertFalse(np.any(reflected_alpha[:94]))
        # The vessel base is attached immediately below the receiver contact.
        self.assertTrue(np.any(reflected_alpha[94:99, 52:78] > 2))
        # The elevated handle appears later in the mirror image, not attached
        # directly to its own lower silhouette at y=67.
        self.assertFalse(np.any(reflected_alpha[94:101, 86:108] > 2))
        self.assertTrue(np.any(reflected_alpha[103:122, 86:108] > 2))

    def test_reflection_fades_before_receiver_edge_instead_of_forming_wedge(self) -> None:
        rgb = np.zeros((160, 160, 3), np.uint8)
        alpha = np.zeros((160, 160), np.uint8)
        rgb[24:100, 48:104] = [174, 98, 46]
        alpha[24:100, 48:104] = 255
        support = np.ones((160, 160), bool)
        support[:, 96:] = False  # table edge clips part of the mirror corridor
        lighting = Lighting.from_dict({
            "shadow_direction": [0.75, 0.66],
            "shadow_length_ratio": 0.06,
            "reflection_enabled": True,
            "reflection_opacity": 0.34,
            "reflection_roughness": 0.14,
        })

        _, reflected_alpha = reflection(
            rgb,
            alpha,
            np.full_like(rgb, [138, 122, 94]),
            support,
            np.zeros_like(support),
            lighting,
            material_mode="mixed_transparent",
        )

        self.assertFalse(np.any(reflected_alpha[:, 96:]))
        central_peak = int(reflected_alpha[100:116, 58:78].max())
        clipped_peak = int(reflected_alpha[100:116, 95:96].max())
        self.assertGreater(central_peak, 0)
        self.assertLess(clipped_peak, central_peak * 0.35)

    def test_glossy_reflection_reaches_real_receiver_edge(self) -> None:
        rgb = np.zeros((150, 180, 3), np.uint8)
        alpha = np.zeros((150, 180), np.uint8)
        rgb[20:100, 60:120] = [178, 68, 42]
        alpha[20:100, 60:120] = 255
        support = np.zeros((150, 180), bool)
        support[96:128, 10:170] = True
        lighting = Lighting.from_dict({
            "shadow_direction": [0.75, 0.66],
            "reflection_enabled": True,
            "reflection_opacity": 0.32,
            "surface_reflectivity": 0.88,
            "reflection_roughness": 0.10,
        })

        _, reflected_alpha = reflection(
            rgb,
            alpha,
            np.full_like(rgb, [122, 108, 82]),
            support,
            np.zeros_like(support),
            lighting,
            material_mode="opaque_glossy",
        )

        # The mirror remains visible immediately before the real table edge;
        # it is not globally truncated at a low shared percentile.
        self.assertTrue(np.any(reflected_alpha[122:128, 70:110] > 2))
        self.assertFalse(np.any(reflected_alpha[128:, :] > 0))

    def test_protected_occluder_only_masks_itself_and_reflection_resumes(self) -> None:
        rgb = np.zeros((190, 180, 3), np.uint8)
        alpha = np.zeros((190, 180), np.uint8)
        rgb[20:112, 62:118] = [178, 68, 42]
        alpha[20:112, 62:118] = 255
        support = np.zeros((190, 180), bool)
        support[108:185, 8:172] = True
        protected = np.zeros_like(support)
        # A narrow tabletop fitting/occluder crosses the reflection corridor.
        protected[136:140, 70:110] = True
        lighting = Lighting.from_dict({
            "shadow_direction": [0.75, 0.66],
            "reflection_enabled": True,
            "reflection_opacity": 0.34,
            "surface_reflectivity": 0.88,
            "reflection_roughness": 0.10,
        })

        _, reflected_alpha = reflection(
            rgb,
            alpha,
            np.full_like(rgb, [122, 108, 82]),
            support,
            protected,
            lighting,
            material_mode="opaque_glossy",
        )

        self.assertFalse(np.any(reflected_alpha[136:140, 70:110]))
        self.assertTrue(np.any(reflected_alpha[143:156, 76:104] > 2))
        self.assertFalse(np.any(reflected_alpha[~support]))

    def test_reflection_central_body_is_contiguous_to_feathered_terminal(self) -> None:
        rgb = np.zeros((210, 190, 3), np.uint8)
        alpha = np.zeros((210, 190), np.uint8)
        rgb[22:116, 64:122] = [164, 58, 40]
        alpha[22:116, 64:122] = 255
        support = np.zeros((210, 190), bool)
        support[112:204, 6:184] = True
        lighting = Lighting.from_dict({
            "shadow_direction": [0.75, 0.66],
            "reflection_enabled": True,
            "reflection_opacity": 0.30,
            "surface_reflectivity": 0.88,
            "reflection_roughness": 0.10,
        })

        _, reflected_alpha = reflection(
            rgb,
            alpha,
            np.full_like(rgb, [126, 112, 88]),
            support,
            np.zeros_like(support),
            lighting,
            material_mode="opaque_glossy",
        )
        profile = np.percentile(reflected_alpha[:, 72:114], 75, axis=1)
        active_rows = np.flatnonzero(profile > 2)
        self.assertGreater(len(active_rows), 0)
        self.assertLessEqual(abs(int(active_rows[0]) - 115), 1)
        interior = profile[int(active_rows[0]):int(active_rows[-1]) + 1]
        self.assertLessEqual(int(np.count_nonzero(interior <= 2)), 1)
        self.assertGreaterEqual(
            int(active_rows[-1] - active_rows[0] + 1),
            int(round((116 - 22) * 0.28)),
        )
        self.assertFalse(np.any(reflected_alpha[~support]))


if __name__ == "__main__":
    unittest.main()
