from __future__ import annotations

import unittest

import cv2
import numpy as np

from strict_compositor.optical_matte import (
    OpticalMatte,
    estimate_clear_glass_matte,
    evaluate_optical_matte,
    evaluate_optical_matte_roi_linear,
    resolve_optical_matte_roi,
    resolve_optical_matte_roi_linear,
)


class OpticalMatteTests(unittest.TestCase):
    @staticmethod
    def _clear_source(
        height: int = 170,
        width: int = 150,
    ) -> tuple[np.ndarray, np.ndarray]:
        rgb = np.zeros((height, width, 3), np.uint8)
        alpha = np.zeros((height, width), np.uint8)
        alpha[14:156, 28:122] = 255
        rgb[14:156, 28:122] = [184, 151, 91]
        return rgb, alpha

    def test_old_horizontal_band_is_removed_and_vertical_rib_is_retained(self) -> None:
        rgb, alpha = self._clear_source()
        # A source-room shelf cuts across the entire glass body.  It should be
        # part of the row-wise environment estimate, not product identity.
        rgb[76:112, 28:122] = [7, 11, 18]
        # This thin rib persists through the object's height and is therefore
        # valid, transferable optical evidence.
        rgb[14:156, 72:75] = [245, 246, 248]
        target = np.full_like(rgb, [165, 179, 195])

        matte = estimate_clear_glass_matte(rgb, alpha)
        result = evaluate_optical_matte(matte, target)

        clear_above = float(result[52, 64].mean())
        old_band = float(result[94, 64].mean())
        rib = float(result[52, 73].mean())
        self.assertLess(abs(clear_above - old_band), 5.0)
        self.assertGreater(rib - clear_above, 8.0)
        self.assertEqual(
            matte.provenance["source_environment_model"],
            "alpha_normalized_row_lowpass",
        )

    def test_broad_vertical_source_reflection_is_not_a_transferable_rib(self) -> None:
        rgb, alpha = self._clear_source()
        # A doorway/photographer reflection can remain vertically coherent but
        # is much wider than a real flute.  It must not become a dark column in
        # the destination glass.  The adjacent thin line is genuine topology
        # and must remain readable.
        rgb[14:156, 44:72] = [18, 22, 29]
        rgb[14:156, 94:97] = [245, 246, 248]
        target = np.full_like(rgb, [165, 179, 195])

        matte = estimate_clear_glass_matte(rgb, alpha)
        result = evaluate_optical_matte(matte, target)

        clear = float(result[70, 82].mean())
        broad_band_centre = float(result[70, 58].mean())
        thin_rib = float(result[70, 95].mean())
        self.assertLess(abs(clear - broad_band_centre), 3.0)
        self.assertGreater(thin_rib - clear, 10.0)
        self.assertGreater(float(matte.provenance["rib_phase_scale"]), 0.0)

    def test_target_background_colour_dominates_instead_of_source_room(self) -> None:
        rgb, alpha = self._clear_source()
        # Deliberately implausible saturated source room colour.
        rgb[14:156, 28:122] = [245, 24, 187]
        matte = estimate_clear_glass_matte(rgb, alpha)
        warm_target = np.full_like(rgb, [211, 151, 76])
        cool_target = np.full_like(rgb, [69, 142, 218])

        warm = evaluate_optical_matte(matte, warm_target)
        cool = evaluate_optical_matte(matte, cool_target)
        warm_pixel = warm[70, 62].astype(np.int16)
        cool_pixel = cool[70, 62].astype(np.int16)

        self.assertGreater(int(warm_pixel[0] - warm_pixel[2]), 90)
        self.assertGreater(int(cool_pixel[2] - cool_pixel[0]), 100)
        # Changing only the target must have a large, predictable effect.
        self.assertGreater(float(np.abs(warm_pixel - cool_pixel).mean()), 60.0)

    def test_bbox_alpha_and_pixels_outside_coverage_are_exact(self) -> None:
        height, width = 96, 128
        rgb = np.full((height, width, 3), [110, 130, 150], np.uint8)
        alpha = np.zeros((height, width), np.uint8)
        yy, xx = np.indices((height, width))
        disc = (xx - 64) ** 2 + (yy - 47) ** 2 <= 24 ** 2
        alpha[disc] = 255
        target = np.arange(height * width * 3, dtype=np.uint32)
        target = (target.reshape(height, width, 3) % 251).astype(np.uint8)

        matte = estimate_clear_glass_matte(rgb, alpha)
        result = evaluate_optical_matte(matte, target)

        self.assertEqual(matte.bbox, (40, 23, 89, 72))
        expected_alpha = alpha[23:72, 40:89].astype(np.float32) / 255.0
        self.assertTrue(np.array_equal(matte.coverage_alpha, expected_alpha))
        self.assertTrue(np.array_equal(result[~disc], target[~disc]))
        # The low-level API exposes the exact linear-RGB ROI contract.
        linear_roi = evaluate_optical_matte_roi_linear(matte, target)
        self.assertEqual(linear_roi.shape, (49, 49, 3))
        self.assertEqual(linear_roi.dtype, np.float32)
        resolved_roi = resolve_optical_matte_roi(matte, target)
        self.assertEqual(resolved_roi.shape, (49, 49, 3))
        self.assertEqual(resolved_roi.dtype, np.uint8)

    def test_partial_coverage_is_not_applied_twice(self) -> None:
        rgb = np.full((72, 84, 3), [235, 42, 176], np.uint8)
        alpha = np.zeros((72, 84), np.uint8)
        alpha[12:60, 18:66] = 128
        target = np.full_like(rgb, [96, 142, 188])

        matte = estimate_clear_glass_matte(rgb, alpha)
        covered = matte.coverage_alpha > 0.0
        # Transmission is conditional material response.  Coverage belongs to
        # the later alpha-over operation and must not add another (1-alpha)
        # term that can push an antialiased edge toward 2x brightness.
        self.assertLessEqual(
            float(matte.transmission_rgb[covered].max()),
            1.39,
        )
        self.assertLessEqual(
            float(matte.specular_rgb[covered].max()),
            0.720001,
        )
        # A half-covered clear edge loses only half of the conditional cast
        # light, so its absolute shadow transmittance stays close to one.
        self.assertGreater(
            float(matte.shadow_transmittance[covered].mean()),
            0.94,
        )
        result = evaluate_optical_matte(matte, target)
        self.assertTrue(np.array_equal(result[alpha == 0], target[alpha == 0]))

    def test_partial_coverage_fresnel_is_conditional_not_alpha_squared(self) -> None:
        """Thin glass topology keeps the material response before alpha-over."""
        height, width = 92, 112
        rgb = np.full((height, width, 3), [182, 178, 170], np.uint8)
        alpha = np.zeros((height, width), np.uint8)
        # A half-covered, handle-like ring exercises the antialiased topology
        # that previously vanished when coverage was multiplied into Fresnel
        # and then multiplied a second time by the evaluator.
        cv2.ellipse(alpha, (56, 46), (30, 35), 0, 0, 360, 128, 5)
        target = np.full_like(rgb, [180, 166, 142])

        matte = estimate_clear_glass_matte(rgb, alpha)
        covered = matte.coverage_alpha > 0.0
        conditional_fresnel = np.mean(matte.specular_rgb, axis=2)

        self.assertGreater(float(conditional_fresnel[covered].max()), 0.24)
        # It remains a bounded conditional dielectric response, not an opaque
        # white carrier painted into every partially covered pixel.
        self.assertLessEqual(float(conditional_fresnel[covered].max()), 0.720001)
        result = evaluate_optical_matte(matte, target)
        self.assertTrue(np.array_equal(result[alpha == 0], target[alpha == 0]))

    def test_coloured_source_fringe_becomes_achromatic_fresnel(self) -> None:
        rgb, alpha = self._clear_source(150, 138)
        # Wide orange source matte around a nominally clear product.
        rgb[14:136, 28:34] = [255, 142, 12]
        rgb[14:136, 116:122] = [28, 78, 255]
        target = np.full_like(rgb, [157, 157, 157])

        matte = estimate_clear_glass_matte(rgb, alpha)
        result = evaluate_optical_matte(matte, target)
        interior = result[70, 62].astype(np.float32)
        left_edge = result[70, 28].astype(np.float32)
        right_edge = result[70, 121].astype(np.float32)

        self.assertLess(float(np.ptp(left_edge - interior)), 3.0)
        self.assertLess(float(np.ptp(right_edge - interior)), 3.0)
        self.assertGreater(float((left_edge - interior).mean()), 2.0)

    def test_explicit_opaque_identity_is_the_only_source_rgb_reuse(self) -> None:
        rgb, alpha = self._clear_source(110, 110)
        rgb[42:58, 40:70] = [18, 32, 210]
        opaque = np.zeros((110, 110), np.uint8)
        opaque[42:58, 40:70] = 255
        target = np.full_like(rgb, [185, 175, 160])

        matte = estimate_clear_glass_matte(
            rgb,
            alpha,
            opaque_identity_mask=opaque,
        )
        result = evaluate_optical_matte(matte, target)

        self.assertTrue(np.array_equal(result[48, 54], rgb[48, 54]))
        self.assertEqual(float(matte.shadow_transmittance[48 - 14, 54 - 28]), 0.0)
        self.assertEqual(matte.provenance["source_rgb_reuse"], "opaque_identity_mask_only")

    def test_neutral_fresnel_energy_uses_target_illuminant_colour(self) -> None:
        shape = (4, 4)
        matte = OpticalMatte(
            coverage_alpha=np.ones(shape, np.float32),
            opaque_identity_mask=np.zeros(shape, np.float32),
            transmission_rgb=np.ones((*shape, 3), np.float32),
            refraction_flow=np.zeros((*shape, 2), np.float32),
            specular_rgb=np.full((*shape, 3), 0.10, np.float32),
            shadow_transmittance=np.ones(shape, np.float32),
            bbox=(1, 1, 5, 5),
            confidence=0.9,
            provenance={"algorithm": "target-illuminant-test"},
        )
        target = np.empty((6, 6, 3), np.float32)
        target[...] = [0.60, 0.25, 0.08]

        resolved = resolve_optical_matte_roi_linear(
            matte,
            target,
            target_is_linear=True,
        )
        excess = resolved[2, 2] - target[2, 2]
        self.assertGreater(float(excess[0]), float(excess[2]) * 1.35)
        self.assertGreater(float(excess[1]), float(excess[2]) * 1.10)

    def test_fresnel_does_not_add_a_body_wide_warm_veil(self) -> None:
        """Moderate reflection energy must not behave like glass emission."""
        shape = (18, 18)
        matte = OpticalMatte(
            coverage_alpha=np.ones(shape, np.float32),
            opaque_identity_mask=np.zeros(shape, np.float32),
            transmission_rgb=np.ones((*shape, 3), np.float32),
            refraction_flow=np.zeros((*shape, 2), np.float32),
            specular_rgb=np.full((*shape, 3), 0.25, np.float32),
            shadow_transmittance=np.ones(shape, np.float32),
            bbox=(1, 1, 19, 19),
            confidence=0.9,
            provenance={"algorithm": "warm-veil-regression"},
        )
        target = np.empty((20, 20, 3), np.float32)
        target[...] = [0.58, 0.31, 0.12]

        resolved = resolve_optical_matte_roi_linear(
            matte,
            target,
            target_is_linear=True,
        )
        mean_excess = float(np.mean(resolved - target[1:19, 1:19]))

        # The target-coloured micro-glint may be non-zero, but it cannot replace
        # the energy-conserving reflection with a broad luminous beige layer.
        self.assertGreater(mean_excess, 0.0)
        self.assertLess(mean_excess, 0.012)

    def test_compact_glass_detail_is_stronger_in_the_thick_base(self) -> None:
        """The same compact topology reads more strongly through a thick base."""
        height, width = 180, 120
        rgb = np.full((height, width, 3), 175, np.uint8)
        alpha = np.zeros((height, width), np.uint8)
        alpha[10:165, 20:100] = 255
        for rib_x in range(28, 92, 8):
            rgb[35:60, rib_x:rib_x + 2] = 230
            rgb[35:60, rib_x + 3:rib_x + 5] = 120
            rgb[132:157, rib_x:rib_x + 2] = 230
            rgb[132:157, rib_x + 3:rib_x + 5] = 120

        matte = estimate_clear_glass_matte(rgb, alpha)
        fresnel = matte.specular_rgb[..., 0]
        upper_detail = float(np.mean(fresnel[25:50, 8:72]))
        base_detail = float(np.mean(fresnel[122:147, 8:72]))

        self.assertGreater(base_detail, upper_detail + 0.035)

    def test_fresnel_reflection_comes_from_target_environment_band(self) -> None:
        height, width = 180, 320
        rgb = np.full((height, width, 3), [180, 180, 180], np.uint8)
        alpha = np.zeros((height, width), np.uint8)
        alpha[18:164, 110:210] = 255
        # The product stands on a quiet neutral patch while the target room
        # around it contains two different coloured structures.  A convex
        # interface should compress those target colours into opposite rims;
        # no source-room colour may be needed to create the cue.
        target = np.full_like(rgb, [142, 142, 142])
        target[:, :105] = [205, 72, 48]
        target[:, 215:] = [50, 91, 210]

        matte = estimate_clear_glass_matte(rgb, alpha)
        result = evaluate_optical_matte(matte, target)
        left_rim = result[90, 110].astype(np.int16)
        right_rim = result[90, 209].astype(np.int16)

        # Reflection mapping is mirrored across the cylindrical axis.
        self.assertGreater(int(left_rim[2] - left_rim[0]), 2)
        self.assertGreater(int(right_rim[0] - right_rim[2]), 2)

    def test_clear_glass_remains_legible_in_browser_preview_without_dark_fill_or_halo(
        self,
    ) -> None:
        """Protect perceptual glass cues at the actual review scale.

        A full-resolution dielectric edge can be numerically non-zero yet
        disappear when a 4K/7K result is fitted into the browser.  This fixture
        models a clear, vertically fluted pitcher on a quiet warm wall, then
        measures the deterministic result after a 27% area downsample.  The
        anti-cheat assertions reject two tempting substitutes for real optical
        structure: uniformly darkening the vessel and painting a wide halo.
        """
        height, width = 360, 300
        yy, xx = np.indices((height, width))

        # Low-contrast target: a smooth warm wall without high-frequency
        # texture that could be mistaken for product detail.
        target_base = np.asarray([188.0, 180.0, 166.0], np.float32)
        target_gradient = (
            (xx - width * 0.5) / width * 4.0
            + (yy - height * 0.5) / height * 2.0
        )[..., None]
        target = target_base + target_gradient * np.asarray(
            [1.0, 0.8, 0.55],
            np.float32,
        )
        target = np.clip(np.round(target), 0, 255).astype(np.uint8)

        alpha = np.zeros((height, width), np.uint8)
        cv2.rectangle(alpha, (70, 55), (220, 310), 255, -1)
        cv2.ellipse(alpha, (145, 55), (75, 24), 0, 180, 360, 255, -1)
        cv2.ellipse(alpha, (145, 310), (75, 22), 0, 0, 180, 255, -1)
        # A measured handle is deliberately included: contour legibility must
        # not depend on testing a convenient rectangular silhouette.
        cv2.ellipse(alpha, (229, 160), (43, 72), 0, -78, 78, 255, 18)

        source = np.full((height, width, 3), [181, 177, 169], np.uint8)
        # Alternating vertically persistent source residuals are transferable
        # fluting evidence.  Their hue is neutral, so the test does not reward
        # leaking the photographed source room into the target scene.
        for rib_x in range(78, 214, 14):
            source[45:322, rib_x - 2:rib_x + 2] = [231, 230, 226]
            source[45:322, rib_x + 4:rib_x + 7] = [143, 145, 147]

        matte = estimate_clear_glass_matte(source, alpha)
        result = evaluate_optical_matte(matte, target)

        preview_scale = 0.27
        preview_size = (
            int(round(width * preview_scale)),
            int(round(height * preview_scale)),
        )
        preview_result = cv2.resize(
            result,
            preview_size,
            interpolation=cv2.INTER_AREA,
        )
        preview_target = cv2.resize(
            target,
            preview_size,
            interpolation=cv2.INTER_AREA,
        )
        preview_alpha = cv2.resize(
            alpha.astype(np.float32) / 255.0,
            preview_size,
            interpolation=cv2.INTER_AREA,
        )

        rgb_delta = np.mean(
            np.abs(
                preview_result.astype(np.float32)
                - preview_target.astype(np.float32)
            ),
            axis=2,
        )
        luma = np.asarray([0.2126, 0.7152, 0.0722], np.float32)
        signed_luma_delta = (
            preview_result.astype(np.float32)
            - preview_target.astype(np.float32)
        ) @ luma

        solid = preview_alpha > 0.50
        eroded = cv2.erode(
            solid.astype(np.uint8),
            np.ones((5, 5), np.uint8),
            iterations=1,
        ).astype(bool)
        contour = solid & ~eroded
        self.assertGreaterEqual(int(np.count_nonzero(contour)), 500)
        self.assertGreater(
            float(np.mean(rgb_delta[contour])),
            3.5,
            "the measured silhouette vanished at browser-preview scale",
        )

        # Measure vertical fluting only in a five-preview-pixel-deep body core,
        # excluding the stronger Fresnel boundary and the side handle.  A
        # broad X trend is removed before measuring the alternating ribs.
        solid_interior = (preview_alpha > 0.95).astype(np.uint8)
        distance = cv2.distanceTransform(solid_interior, cv2.DIST_L2, 5)
        core = distance >= 5.0
        column_signal: list[float] = []
        for column in range(preview_size[0]):
            column_core = core[:, column]
            if int(np.count_nonzero(column_core)) >= 12:
                column_signal.append(
                    float(np.mean(signed_luma_delta[column_core, column])),
                )
        self.assertGreaterEqual(len(column_signal), 24)
        column_signal_array = np.asarray(column_signal, np.float32)
        broad_signal = cv2.GaussianBlur(
            column_signal_array.reshape(1, -1),
            (0, 0),
            sigmaX=2.4,
            borderType=cv2.BORDER_REPLICATE,
        )[0]
        rib_signal = column_signal_array - broad_signal
        rib_span = float(
            np.percentile(rib_signal, 90) - np.percentile(rib_signal, 10),
        )
        self.assertGreater(
            rib_span,
            12.0,
            "vertical fluting did not survive the preview downsample",
        )

        # Legibility may come from localized positive/negative optical cues,
        # but not from filling the entire transparent body with a dark plate.
        self.assertLess(
            abs(float(np.mean(signed_luma_delta[core]))),
            2.5,
            "glass preview was made visible by a broad brightness shift",
        )
        self.assertLess(
            float(np.mean(np.abs(signed_luma_delta[core]) > 18.0)),
            0.20,
            "too much of the clear interior was converted into opaque contrast",
        )

        # Area resampling can mix the true silhouette into its immediate
        # display pixel.  Starting two pixels farther out, a 4-pixel-wide ring
        # must still be exact target imagery; otherwise this is a wide halo.
        active = preview_alpha > 0.01
        inner = cv2.dilate(
            active.astype(np.uint8),
            np.ones((3, 3), np.uint8),
            iterations=1,
        ).astype(bool)
        outer = cv2.dilate(
            active.astype(np.uint8),
            np.ones((9, 9), np.uint8),
            iterations=1,
        ).astype(bool)
        exterior_ring = outer & ~inner
        self.assertGreaterEqual(int(np.count_nonzero(exterior_ring)), 500)
        self.assertEqual(float(np.max(rgb_delta[exterior_ring])), 0.0)

    def test_side_handle_gets_tubular_refraction_without_filling_its_hole(
        self,
    ) -> None:
        """An alpha-valid loop handle participates in optics continuously."""
        height, width = 240, 270
        alpha = np.zeros((height, width), np.uint8)
        # Draw an annular handle first, then overlap a tall vessel body.  The
        # remaining enclosed negative space is the only evidence the estimator
        # needs; no pitcher-specific rectangle or semantic label is supplied.
        cv2.ellipse(alpha, (160, 112), (62, 78), 0, 0, 360, 255, -1)
        cv2.ellipse(alpha, (160, 112), (39, 55), 0, 0, 360, 0, -1)
        cv2.rectangle(alpha, (38, 25), (133, 213), 255, -1)

        source = np.full(
            (height, width, 3), [178, 175, 168], np.uint8,
        )
        for rib_x in range(48, 128, 12):
            source[25:214, rib_x:rib_x + 2] = 225
            source[25:214, rib_x + 4:rib_x + 6] = 140
        yy, xx = np.indices((height, width), dtype=np.float32)
        target = np.empty((height, width, 3), np.float32)
        target[..., 0] = 145.0 + xx * 0.20
        target[..., 1] = 155.0 + yy * 0.10
        target[..., 2] = 165.0 + xx * 0.08
        target = np.clip(np.round(target), 0, 255).astype(np.uint8)

        matte = estimate_clear_glass_matte(source, alpha)
        result = evaluate_optical_matte(matte, target)
        x0, y0, x1, y1 = matte.bbox
        local_alpha = alpha[y0:y1, x0:x1]
        local_y, local_x = np.indices(local_alpha.shape)
        flow_energy = np.linalg.norm(matte.refraction_flow, axis=2)
        handle = (local_alpha > 4) & (local_x > 99)

        self.assertEqual(
            matte.provenance["tubular_appendages"]["count"], 1,
        )
        self.assertGreater(
            float(np.percentile(flow_energy[handle], 90.0)), 2.5,
            "the measured handle retained alpha but received no refraction",
        )
        # Both attachment zones retain non-zero flow, preventing the two
        # disconnected strokes seen in the A07 browser preview.
        upper_joint = handle & (local_y >= 20) & (local_y < 78)
        lower_joint = handle & (local_y >= 112) & (local_y < 185)
        self.assertGreater(
            float(np.percentile(flow_energy[upper_joint], 75.0)), 1.0,
        )
        self.assertGreater(
            float(np.percentile(flow_energy[lower_joint], 75.0)), 1.0,
        )

        # The topology inference is optical only.  It cannot expand alpha or
        # paint the enclosed hole white, and the evaluator must leave every
        # zero-alpha pixel byte-identical to the target scene.
        hole = (local_alpha == 0) & (local_x > 95)
        self.assertEqual(float(np.max(flow_energy[hole])), 0.0)
        self.assertEqual(float(np.max(matte.specular_rgb[..., 0][hole])), 0.0)
        self.assertTrue(np.array_equal(result[alpha == 0], target[alpha == 0]))
        self.assertTrue(np.array_equal(
            matte.coverage_alpha,
            local_alpha.astype(np.float32) / 255.0,
        ))

    def test_transmission_and_visibility_parameters_have_continuous_effects(
        self,
    ) -> None:
        height, width = 160, 140
        alpha = np.zeros((height, width), np.uint8)
        alpha[10:150, 20:120] = 255
        source = np.full((height, width, 3), 178, np.uint8)

        low = estimate_clear_glass_matte(
            source,
            alpha,
            transmission_strength=0.62,
            commercial_visibility=0.25,
        )
        high = estimate_clear_glass_matte(
            source,
            alpha,
            transmission_strength=0.97,
            commercial_visibility=1.0,
        )
        core = cv2.distanceTransform(
            (high.coverage_alpha > 0.99).astype(np.uint8),
            cv2.DIST_L2,
            5,
        ) > 12.0
        active = high.coverage_alpha > 0.50

        self.assertGreater(
            float(np.mean(high.transmission_rgb[..., 0][core])),
            float(np.mean(low.transmission_rgb[..., 0][core])) + 0.006,
        )
        self.assertGreater(
            float(np.percentile(
                np.linalg.norm(high.refraction_flow, axis=2)[active], 90.0,
            )),
            float(np.percentile(
                np.linalg.norm(low.refraction_flow, axis=2)[active], 90.0,
            )) + 2.0,
        )
        self.assertGreater(
            float(np.mean(high.specular_rgb[..., 0][active])),
            float(np.mean(low.specular_rgb[..., 0][active])) + 0.005,
        )
        self.assertTrue(np.array_equal(
            high.coverage_alpha, low.coverage_alpha,
        ))
        self.assertEqual(
            high.provenance["transmission_strength"], 0.97,
        )
        self.assertEqual(
            low.provenance["commercial_visibility"], 0.25,
        )

        with self.assertRaisesRegex(ValueError, "transmission_strength"):
            estimate_clear_glass_matte(
                source, alpha, transmission_strength=float("nan"),
            )
        with self.assertRaisesRegex(ValueError, "commercial_visibility"):
            estimate_clear_glass_matte(
                source, alpha, commercial_visibility=1.1,
            )

    def test_matte_is_validated_and_arrays_are_read_only(self) -> None:
        rgb, alpha = self._clear_source(90, 100)
        matte = estimate_clear_glass_matte(rgb, alpha, confidence=0.83)

        self.assertAlmostEqual(matte.confidence, 0.83)
        self.assertFalse(matte.coverage_alpha.flags.writeable)
        self.assertFalse(matte.refraction_flow.flags.writeable)
        with self.assertRaises(ValueError):
            matte.coverage_alpha[0, 0] = 0.5

        shape = (2, 3)
        with self.assertRaises(ValueError):
            OpticalMatte(
                coverage_alpha=np.ones(shape, np.float32),
                opaque_identity_mask=np.zeros(shape, np.float32),
                transmission_rgb=np.ones((*shape, 3), np.float32),
                refraction_flow=np.zeros((*shape, 2), np.float32),
                specular_rgb=np.zeros((*shape, 3), np.float32),
                shadow_transmittance=np.ones(shape, np.float32),
                bbox=(0, 0, 4, 2),  # Width disagrees with every ROI tensor.
                confidence=0.8,
                provenance={"algorithm": "test"},
            )

    def test_invalid_inputs_fail_before_processing(self) -> None:
        rgb = np.zeros((20, 30, 3), np.uint8)
        alpha = np.zeros((20, 30), np.uint8)
        with self.assertRaisesRegex(ValueError, "no non-zero"):
            estimate_clear_glass_matte(rgb, alpha)
        with self.assertRaisesRegex(ValueError, "match"):
            estimate_clear_glass_matte(rgb, np.zeros((19, 30), np.uint8))

        alpha[4:16, 8:22] = 255
        matte = estimate_clear_glass_matte(rgb, alpha)
        with self.assertRaisesRegex(ValueError, "outside target"):
            evaluate_optical_matte(matte, np.zeros((10, 10, 3), np.uint8))
        with self.assertRaisesRegex(TypeError, "linear target"):
            evaluate_optical_matte(matte, rgb, target_is_linear=True)


if __name__ == "__main__":
    unittest.main()
