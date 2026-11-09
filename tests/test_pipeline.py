from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from strict_compositor.contracts import Job, Lighting
from strict_compositor.image_io import save_png
from strict_compositor.pipeline import StrictCompositor
from strict_compositor.quality import gate_visual_score
from strict_compositor.render import (
    _cast_shadow_flat,
    _clear_glass_contact_shadow,
    _continuous_base_profile,
    _foot_narrowness,
    _grounding_profile,
    _lower_silhouette_occlusion,
    _narrow_body_receiver_occlusion,
    _receiver_baseline_occlusion,
    cast_shadow,
    contact_profile,
    contact_shadow,
    alpha_over,
    darken,
    decontaminate_edge_color,
    render_product,
    transform_rgba,
)
from strict_compositor.validate import audit_background


class StrictPipelineTests(unittest.TestCase):
    def test_receiver_compositing_uses_linear_light(self) -> None:
        middle = np.full((1, 1, 3), 128, np.uint8)
        half = np.full((1, 1), 128, np.uint8)
        shadowed = darken(middle, half)
        # 50% light loss at encoded 128 is about sRGB 92, not gamma-space 64.
        self.assertTrue(np.all(np.abs(shadowed.astype(int) - 92) <= 1))

        black = np.zeros((1, 1, 3), np.uint8)
        white = np.full((1, 1, 3), 255, np.uint8)
        blended = alpha_over(black, white, half)
        # Half linear radiance encodes near sRGB 188.
        self.assertTrue(np.all(np.abs(blended.astype(int) - 188) <= 1))

    def test_linear_compositing_preserves_exact_alpha_endpoints(self) -> None:
        base = np.asarray([[[17, 83, 201], [40, 50, 60]]], np.uint8)
        foreground = np.asarray([[[220, 91, 12], [210, 190, 170]]], np.uint8)
        alpha = np.asarray([[0, 255]], np.uint8)
        result = alpha_over(base, foreground, alpha)
        np.testing.assert_array_equal(result[0, 0], base[0, 0])
        np.testing.assert_array_equal(result[0, 1], foreground[0, 1])

    def test_rgba_resampling_does_not_bleed_transparent_background_colour(self) -> None:
        rgba = np.zeros((80, 80, 4), np.uint8)
        rgba[..., :3] = [0, 255, 0]  # source background stored under alpha zero
        rgba[16:64, 16:64, :3] = [220, 30, 20]
        rgba[16:64, 16:64, 3] = 255
        matrix = np.asarray(
            [[0.43, 0.0, 8.25], [0.0, 0.43, 7.75], [0.0, 0.0, 1.0]],
            np.float64,
        )

        rgb, alpha = transform_rgba(rgba, matrix, (64, 64))

        edge = (alpha > 4) & (alpha < 250)
        self.assertTrue(edge.any())
        self.assertGreater(float(rgb[..., 0][edge].mean()), 170.0)
        self.assertLess(float(rgb[..., 1][edge].mean()), 80.0)

    def test_source_matte_colour_is_removed_only_from_translucent_fringe(self) -> None:
        rgba = np.zeros((40, 40, 4), np.uint8)
        rgba[..., :3] = [10, 240, 20]
        rgba[10:30, 10:30, :3] = [220, 35, 25]
        rgba[10:30, 10:30, 3] = 255
        rgba[8:32, 8:32, 3] = np.maximum(rgba[8:32, 8:32, 3], 96)
        original_opaque = rgba[10:30, 10:30, :3].copy()

        cleaned = decontaminate_edge_color(rgba)

        fringe = (rgba[..., 3] > 0) & (rgba[..., 3] < 250)
        self.assertLess(float(cleaned[..., 1][fringe].mean()), float(rgba[..., 1][fringe].mean()))
        self.assertGreater(float(cleaned[..., 0][fringe].mean()), float(rgba[..., 0][fringe].mean()))
        np.testing.assert_array_equal(cleaned[10:30, 10:30, :3], original_opaque)
        np.testing.assert_array_equal(cleaned[..., 3], rgba[..., 3])

    def test_hard_opaque_side_matte_is_cleaned_without_touching_top_rim(self) -> None:
        rgba = np.zeros((64, 52, 4), np.uint8)
        rgba[8:58, 10:42, :3] = [112, 28, 40]
        rgba[8:58, 10:42, 3] = 255
        # Intrinsic metallic top rim must remain byte exact.
        rgba[8:12, 10:42, :3] = [205, 202, 194]
        original_top = rgba[8:12, 10:42, :3].copy()
        # Simulated source-wall pixels incorrectly marked fully opaque on both
        # vertical silhouettes.
        rgba[14:54, 10:12, :3] = [234, 220, 184]
        rgba[14:54, 40:42, :3] = [234, 220, 184]

        cleaned = decontaminate_edge_color(
            rgba, clean_opaque_sides=True,
        )

        self.assertLess(
            float(cleaned[20:48, 10:12, :3].mean()),
            float(rgba[20:48, 10:12, :3].mean()) - 18.0,
        )
        np.testing.assert_array_equal(cleaned[8:12, 10:42, :3], original_top)
        np.testing.assert_array_equal(cleaned[..., 3], rgba[..., 3])

    def test_background_is_bit_exact_outside_authorized_region(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            background = np.full((100, 160, 3), [40, 80, 120], dtype=np.uint8)
            product = np.zeros((30, 20, 4), dtype=np.uint8)
            product[..., :3] = [230, 80, 30]
            product[4:27, 3:17, 3] = 255
            save_png(root / "background.png", background)
            save_png(root / "product.png", product)
            job_data = {
                "background": "background.png",
                "support_polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
                "lighting": {"shadow_direction": [1, 0.4], "shadow_length_ratio": 0.08},
                "products": [{"id": "p", "rgba": "product.png", "homography": [[1, 0, 60], [0, 1, 45], [0, 0, 1]], "z_order": 0}],
            }
            job = Job.from_dict(job_data, root)
            outcome = StrictCompositor().compose(job, root / "out")
            self.assertEqual(outcome.audit.verdict, "pass")
            self.assertEqual(outcome.audit.changed_pixels, 0)
            self.assertEqual(outcome.final.shape, background.shape)
            report = json.loads((root / "out" / "audit.json").read_text(encoding="utf-8"))
            self.assertEqual(
                report["physical_audit"]["schema_version"],
                "physical-evidence-v1",
            )
            self.assertIn("grounding", report["physical_audit"]["subscores"])
            self.assertEqual(
                report["verdict"],
                "pending_visual_review"
                if report["physical_audit"]["verdict"] == "pass"
                else "reject",
            )

    def test_debug_mode_skips_only_terminal_audits_and_keeps_layers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            background = np.full((100, 160, 3), [40, 80, 120], dtype=np.uint8)
            product = np.zeros((30, 20, 4), dtype=np.uint8)
            product[..., :3] = [230, 80, 30]
            product[4:27, 3:17, 3] = 255
            save_png(root / "background.png", background)
            save_png(root / "product.png", product)
            job = Job.from_dict({
                "background": "background.png",
                "support_polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
                "lighting": {
                    "shadow_direction": [1, 0.4],
                    "shadow_length_ratio": 0.08,
                },
                "products": [{
                    "id": "p",
                    "rgba": "product.png",
                    "homography": [[1, 0, 60], [0, 1, 45], [0, 0, 1]],
                    "z_order": 0,
                }],
            }, root)

            outcome = StrictCompositor().compose(
                job, root / "out", strict_audit=False,
            )
            report = json.loads(
                (root / "out" / "audit.json").read_text(encoding="utf-8")
            )

            self.assertEqual(outcome.audit.verdict, "skipped_debug")
            self.assertEqual(report["audit_mode"], "debug_fast")
            self.assertEqual(report["physical_audit"]["verdict"], "skipped_debug")
            self.assertEqual(report["verdict"], "debug_preview")
            self.assertTrue(report["diagnostics_exported"])
            self.assertTrue((root / "out" / "authorized_mask.png").is_file())
            self.assertTrue((root / "out" / "support_mask.png").is_file())
            self.assertTrue(any((root / "out" / "layers").glob("*.png")))

    def test_product_in_protected_area_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            background = np.zeros((80, 80, 3), dtype=np.uint8)
            product = np.zeros((20, 20, 4), dtype=np.uint8)
            product[..., :3] = 255
            product[..., 3] = 255
            protected = np.zeros((80, 80), dtype=np.uint8)
            protected[40:60, 40:60] = 255
            save_png(root / "background.png", background)
            save_png(root / "product.png", product)
            save_png(root / "protected.png", protected)
            job = Job.from_dict({
                "background": "background.png", "support_polygon": [[0, 0], [1, 0], [1, 1], [0, 1]], "protected_mask": "protected.png",
                "lighting": {"shadow_direction": [1, 1], "shadow_length_ratio": 0.1},
                "products": [{"id": "p", "rgba": "product.png", "homography": [[1, 0, 45], [0, 1, 45], [0, 0, 1]]}],
            }, root)
            with self.assertRaisesRegex(ValueError, "protected"):
                StrictCompositor().compose(job, root / "out")

    def test_only_base_not_whole_body_must_be_inside_support(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            background = np.full((120, 160, 3), [150, 145, 130], dtype=np.uint8)
            product = np.zeros((80, 100, 4), dtype=np.uint8)
            product[..., :3] = [70, 140, 70]
            product[4:35, 2:98, 3] = 255
            product[30:73, 44:56, 3] = 255
            product[70:80, 39:61, 3] = 255
            save_png(root / "background.png", background)
            save_png(root / "product.png", product)
            job = Job.from_dict({
                "background": "background.png",
                "support_polygon": [[0.43, 0.45], [0.57, 0.45], [0.57, 0.90], [0.43, 0.90]],
                "lighting": {
                    "shadow_direction": [1, 0.4],
                    "shadow_length_ratio": 0.045,
                    "shadow_opacity": 0.24,
                    "shadow_softness_ratio": 0.15,
                },
                "products": [{
                    "id": "p",
                    "rgba": "product.png",
                    "homography": [[1, 0, 30], [0, 1, 20], [0, 0, 1]],
                    "z_order": 0,
                }],
            }, root)
            outcome = StrictCompositor().compose(job, root / "out")
            self.assertEqual(outcome.audit.verdict, "pass")

    def test_audit_rejects_non_authorized_change(self) -> None:
        background = np.zeros((3, 3, 3), dtype=np.uint8)
        result = background.copy()
        result[0, 0] = 1
        audit = audit_background(background, result, np.zeros((3, 3), dtype=bool))
        self.assertEqual((audit.verdict, audit.changed_pixels, audit.max_delta), ("reject", 1, 1))

    def test_visual_gate_is_not_bypassed(self) -> None:
        self.assertEqual(gate_visual_score(89.9, "pass").verdict, "reject")
        self.assertEqual(gate_visual_score(90, "pass").verdict, "pass")
        self.assertEqual(gate_visual_score(99, "reject").verdict, "reject")

    def test_total_shadow_length_grows_with_rendered_product_height(self) -> None:
        lighting = Lighting.from_dict({
            "shadow_direction": [0, 1],
            "shadow_length_ratio": 0.08,
            "shadow_opacity": 0.34,
            "shadow_softness_ratio": 0.08,
        })
        short = np.zeros((180, 120), np.uint8)
        tall = np.zeros_like(short)
        short[80:111, 48:72] = 255
        tall[40:111, 48:72] = 255
        short_shadow = _cast_shadow_flat(short, lighting)
        tall_shadow = _cast_shadow_flat(tall, lighting)
        short_rows = np.flatnonzero(short_shadow > 2)
        tall_rows = np.flatnonzero(tall_shadow > 2)
        self.assertGreater(int(tall_rows.max()), int(short_rows.max()))

    def test_contact_ao_retains_dense_core_after_softening(self) -> None:
        alpha = np.zeros((180, 160), np.uint8)
        alpha[30:150, 55:105] = 255
        alpha[145:155, 45:115] = 255
        support = np.ones_like(alpha, bool)
        protected = np.zeros_like(alpha, bool)
        ao = contact_shadow(alpha, support, protected)
        self.assertGreaterEqual(int(ao.max()), 170)
        self.assertGreater(np.count_nonzero(ao > 8), 300)

    def test_clear_glass_contact_is_narrow_continuous_and_footprint_bound(self) -> None:
        alpha = np.zeros((240, 320), np.uint8)
        # Tall clear vessel with a narrow, slightly stepped real base.  The
        # wide body must not be converted into a generic receiver-side oval.
        alpha[30:185, 70:250] = 96
        alpha[180:205, 125:195] = 210
        for x in range(125, 195):
            bottom = 204 + ((x - 125) // 24)
            alpha[200:bottom + 1, x] = 230
        optical = np.zeros_like(alpha)
        optical[alpha > 4] = 74
        support = np.ones_like(alpha, bool)
        protected = np.zeros_like(alpha, bool)
        lighting = Lighting.from_dict({
            "shadow_direction": [0.8, 0.2],
            "cast_shadow_mode": "ambient_only",
            "surface_material": "transparent_glass",
            # Exercise the production upper bound: even a strong scene-level
            # request cannot turn the transparent foot into a black sticker.
            "contact_ao_strength": 1.50,
        })

        ao = _clear_glass_contact_shadow(
            alpha, optical, support, protected, lighting,
        )
        footprint = contact_profile(alpha) > 4
        footprint_x = footprint.any(axis=0)
        ao_x = (ao > 0).any(axis=0)

        # AO columns are a subset of the measured physical base columns.  The
        # broad transparent body and the space beside the base stay untouched.
        self.assertFalse(np.any(ao_x & ~footprint_x))
        self.assertEqual(int(ao[:, :120].max()), 0)
        self.assertEqual(int(ao[:, 200:].max()), 0)
        # A single real base remains continuous, strong enough to survive
        # preview downsampling, and only a few receiver pixels deep.
        active_x = np.flatnonzero(footprint_x)
        per_column = (ao[:, active_x] > 36).any(axis=0)
        self.assertGreater(np.count_nonzero(per_column), len(active_x) * 0.97)
        self.assertGreaterEqual(int(ao.max()), 100)
        self.assertLessEqual(int(ao.max()), 178)
        active_rows = np.flatnonzero((ao > 0).any(axis=1))
        self.assertLessEqual(int(active_rows.max() - active_rows.min() + 1), 8)
        # The contact layer never darkens any visible product pixel.
        self.assertEqual(int(ao[alpha > 4].max()), 0)

    def test_clear_glass_contact_preserves_foot_gap_and_authorization_masks(self) -> None:
        alpha = np.zeros((180, 260), np.uint8)
        alpha[30:140, 55:205] = 92
        alpha[135:160, 72:102] = 220
        alpha[135:160, 158:188] = 220
        optical = np.zeros_like(alpha)
        optical[alpha > 4] = 68
        support = np.ones_like(alpha, bool)
        support[:, :210] = True
        support[:, 210:] = False
        protected = np.zeros_like(alpha, bool)
        protected[160:170, 78:88] = True
        lighting = Lighting.from_dict({
            "shadow_direction": [0.0, 1.0],
            "surface_material": "transparent_glass",
        })

        ao = _clear_glass_contact_shadow(
            alpha, optical, support, protected, lighting,
        )

        self.assertGreater(int(ao[:, 72:102].max()), 0)
        self.assertGreater(int(ao[:, 158:188].max()), 0)
        self.assertEqual(int(ao[:, 112:148].max()), 0)
        self.assertEqual(int(ao[~support].max()), 0)
        self.assertEqual(int(ao[protected].max()), 0)

    def test_render_product_routes_clear_glass_table_contact_to_narrow_seam(self) -> None:
        height, width = 180, 260
        rgba = np.zeros((height, width, 4), np.uint8)
        rgba[25:140, 55:205, :3] = 180
        rgba[25:140, 55:205, 3] = 96
        rgba[135:160, 105:155, :3] = 145
        rgba[135:160, 105:155, 3] = 220
        background = np.full((height, width, 3), 180, np.uint8)
        support = np.ones((height, width), bool)
        protected = np.zeros_like(support)
        lighting = Lighting.from_dict({
            "shadow_direction": [0.0, 1.0],
            "cast_shadow_mode": "ambient_only",
            "direct_light_fraction": 0.05,
            "surface_material": "transparent_glass",
        })

        layer = render_product(
            rgba,
            np.eye(3),
            (height, width),
            background,
            support,
            protected,
            lighting,
            material_mode="clear_glass",
            material_confidence=0.95,
            transmission_strength=0.95,
        )

        footprint_x = (contact_profile(layer.alpha) > 4).any(axis=0)
        contact_x = (layer.contact_shadow > 0).any(axis=0)
        self.assertTrue(np.any(contact_x))
        self.assertFalse(np.any(contact_x & ~footprint_x))
        self.assertEqual(
            int(layer.contact_shadow[layer.alpha > 4].max()), 0,
        )
        self.assertLessEqual(int(layer.contact_shadow.max()), 178)

    def test_lit_contact_shadow_exposes_attached_seam_on_shadow_side(self) -> None:
        from strict_compositor.contracts import Lighting

        alpha = np.zeros((180, 180), np.uint8)
        alpha[30:150, 55:125] = 255
        support = np.ones_like(alpha, bool)
        protected = np.zeros_like(alpha, bool)
        lighting = Lighting(
            direction=(0.6, 0.8),
            shadow_length_ratio=0.08,
            shadow_opacity=0.3,
            shadow_softness_ratio=0.1,
        )
        ao = contact_shadow(alpha, support, protected, radius=12, lighting=lighting)
        self.assertGreater(int(ao[153:160, 70:120].max()), 30)
        # Several consecutive rows must remain visible after the object body
        # covers its source contact line; a single anti-aliased row disappears
        # when a 4K result is fitted to the browser canvas.
        visible_rows = np.flatnonzero((ao[151:163, 70:120] > 24).any(axis=1))
        self.assertGreaterEqual(len(visible_rows), 3)

    def test_contact_ao_energy_tracks_scene_shadow_opacity(self) -> None:
        alpha = np.zeros((180, 180), np.uint8)
        alpha[30:150, 55:125] = 255
        support = np.ones_like(alpha, bool)
        protected = np.zeros_like(alpha, bool)
        dim = Lighting.from_dict({
            "shadow_direction": [0.6, 0.8],
            "shadow_opacity": 0.18,
        })
        strong = Lighting.from_dict({
            "shadow_direction": [0.6, 0.8],
            "shadow_opacity": 0.38,
        })

        dim_ao = contact_shadow(alpha, support, protected, lighting=dim)
        strong_ao = contact_shadow(alpha, support, protected, lighting=strong)

        self.assertLess(int(dim_ao.max()), int(strong_ao.max()))
        self.assertLessEqual(int(dim_ao.max()), 120)

    def test_contact_ao_has_receiver_side_seam_independent_of_light_vector(self) -> None:
        alpha = np.zeros((180, 180), np.uint8)
        alpha[30:150, 55:125] = 255
        support = np.ones_like(alpha, bool)
        protected = np.zeros_like(alpha, bool)
        horizontal_light = Lighting.from_dict({
            "shadow_direction": [1.0, 0.0],
            "shadow_opacity": 0.30,
        })

        ao = contact_shadow(
            alpha, support, protected, radius=12, lighting=horizontal_light,
        )

        # Cast direction is horizontal, but ambient contact occlusion still
        # appears directly below the physical base where it meets the plane.
        self.assertGreater(int(ao[151:158, 70:120].max()), 90)
        self.assertGreaterEqual(
            len(np.flatnonzero((ao[151:159, 70:120] > 40).any(axis=1))),
            3,
        )

    def test_ambient_only_contact_ao_is_direction_invariant(self) -> None:
        alpha = np.zeros((180, 180), np.uint8)
        alpha[30:150, 55:125] = 255
        support = np.ones_like(alpha, bool)
        protected = np.zeros_like(alpha, bool)
        common = {
            "cast_shadow_mode": "ambient_only",
            "direct_light_fraction": 0.05,
            "shadow_opacity": 0.23,
        }
        left = Lighting.from_dict({
            **common, "shadow_direction": [-0.8, 0.2],
        })
        right = Lighting.from_dict({
            **common, "shadow_direction": [0.6, -0.7],
        })

        left_ao = contact_shadow(alpha, support, protected, lighting=left)
        right_ao = contact_shadow(alpha, support, protected, lighting=right)
        np.testing.assert_array_equal(left_ao, right_ao)

    def test_ambient_only_contact_ao_has_a_scale_bounded_receiver_tail(self) -> None:
        alpha = np.zeros((320, 360), np.uint8)
        yy, xx = np.indices(alpha.shape)
        body = ((xx - 180) / 92.0) ** 2 + ((yy - 155) / 105.0) ** 2 <= 1.0
        alpha[body] = 255
        alpha[254:270, 160:200] = 255
        support = np.ones_like(alpha, bool)
        protected = np.zeros_like(alpha, bool)
        lighting = Lighting.from_dict({
            "shadow_direction": [0.4, 0.9],
            "cast_shadow_mode": "ambient_only",
            "direct_light_fraction": 0.04,
            "shadow_opacity": 0.0,
            "contact_ao_strength": 1.12,
        })

        ao = contact_shadow(alpha, support, protected, lighting=lighting)
        strict = contact_profile(alpha)
        baseline = int(np.argwhere(strict > 4)[:, 0].max())
        occupied = np.argwhere(alpha > 4)
        object_height = int(occupied[:, 0].max() - occupied[:, 0].min() + 1)
        allowed_tail = max(8, int(round(object_height * 0.035)))
        active_rows = np.flatnonzero(ao.max(axis=1) > 0)

        self.assertLessEqual(int(active_rows.max()), baseline + allowed_tail)
        self.assertGreater(int(ao[baseline:baseline + 5].max()), 80)

    def test_round_body_seals_only_its_alpha_derived_narrow_foot(self) -> None:
        alpha = np.zeros((220, 280), np.uint8)
        yy, xx = np.indices(alpha.shape)
        alpha[
            ((xx - 140) / 88.0) ** 2 + ((yy - 104) / 68.0) ** 2 <= 1.0
        ] = 255

        strict = contact_profile(alpha)
        wide = _continuous_base_profile(alpha)
        receiver = _receiver_baseline_occlusion(alpha, strict, wide)
        body_columns = np.flatnonzero((alpha >= 32).max(axis=0))
        receiver_columns = np.flatnonzero(receiver.max(axis=0) > 4)

        # A curved belly is not promoted to a wide generic oval.  The sealed
        # receiver geometry is constrained to the coherent lowest alpha band.
        self.assertEqual(np.count_nonzero(wide), 0)
        self.assertLess(len(receiver_columns), len(body_columns) * 0.40)
        self.assertGreater(len(receiver_columns), 8)
        self.assertEqual(int(receiver[:, 70:100].max()), 0)
        self.assertEqual(int(receiver[:, 180:210].max()), 0)

        support = np.ones_like(alpha, bool)
        protected = np.zeros_like(alpha, bool)
        ao = contact_shadow(alpha, support, protected, radius=10)
        contact_y = int(np.argwhere(alpha >= 32)[:, 0].max())
        attached = ao[
            contact_y + 1:contact_y + 5,
            receiver_columns.min():receiver_columns.max() + 1,
        ]
        self.assertGreater(int(attached.max()), 100)
        self.assertGreater(
            np.count_nonzero(attached.max(axis=0) > 45),
            len(receiver_columns) * 0.80,
        )

    def test_piecewise_sloped_wide_base_has_continuous_measured_seam(self) -> None:
        alpha = np.zeros((240, 320), np.uint8)
        expected_bottoms: dict[int, int] = {}
        for x in range(60, 260):
            # A real cut-out edge is rarely one perfect Hough line: retain a
            # shallow perspective slope plus deterministic one-pixel steps.
            bottom = 174 + int(round((x - 60) * 8 / 199)) + ((x // 17) % 2)
            alpha[45:bottom + 1, x] = 255
            expected_bottoms[x] = bottom

        wide = _continuous_base_profile(alpha)
        wide_columns = np.flatnonzero(wide.max(axis=0) > 4)
        self.assertGreaterEqual(len(wide_columns), 180)
        self.assertLessEqual(int(wide_columns.min()), 66)
        self.assertGreaterEqual(int(wide_columns.max()), 253)

        support = np.ones_like(alpha, bool)
        protected = np.zeros_like(alpha, bool)
        lighting = Lighting.from_dict({
            "shadow_direction": [1.0, 0.0],
            "shadow_opacity": 0.30,
        })
        ao = contact_shadow(
            alpha, support, protected, radius=10, lighting=lighting,
        )
        attached = np.asarray([
            int(ao[bottom + 1:bottom + 5, x].max())
            for x, bottom in expected_bottoms.items()
            if 66 <= x <= 253
        ])
        # Nearly the full measured base must retain a receiver-side seam even
        # when the light vector is horizontal.  This is contact AO, not cast.
        self.assertGreater(np.count_nonzero(attached > 45), len(attached) * 0.95)
        self.assertGreater(int(np.percentile(attached, 10)), 70)

    def test_two_narrow_feet_do_not_create_a_filled_between_feet_oval(self) -> None:
        alpha = np.zeros((200, 260), np.uint8)
        alpha[35:145, 55:205] = 255
        alpha[140:170, 75:100] = 255
        alpha[140:170, 160:185] = 255

        strict = contact_profile(alpha)
        receiver = _receiver_baseline_occlusion(
            alpha, strict, _continuous_base_profile(alpha),
        )

        self.assertGreater(int(receiver[:, 75:100].max()), 0)
        self.assertGreater(int(receiver[:, 160:185].max()), 0)
        self.assertEqual(int(receiver[:, 110:150].max()), 0)

    def test_round_lower_body_generates_wider_soft_occlusion_than_foot(self) -> None:
        alpha = np.zeros((180, 220), np.uint8)
        yy, xx = np.indices(alpha.shape)
        body = ((xx - 110) / 70.0) ** 2 + ((yy - 95) / 58.0) ** 2 <= 1.0
        alpha[body] = 255
        alpha[150:158, 98:122] = 255  # narrow physical foot

        lower = _lower_silhouette_occlusion(alpha)
        columns = np.flatnonzero(lower.max(axis=0) > 0.08)
        rows = np.flatnonzero(lower.max(axis=1) > 0.08)

        self.assertGreater(int(columns.max() - columns.min()), 50)
        self.assertLess(int(columns.max() - columns.min()), 150)
        self.assertGreater(int(rows.max() - rows.min()), 8)

    def test_round_narrow_foot_gets_alpha_derived_receiver_underside(self) -> None:
        alpha = np.zeros((220, 280), np.uint8)
        yy, xx = np.indices(alpha.shape)
        body = ((xx - 140) / 88.0) ** 2 + ((yy - 108) / 68.0) ** 2 <= 1.0
        alpha[body] = 255
        alpha[170:181, 126:154] = 255

        strict = contact_profile(alpha)
        underside = _narrow_body_receiver_occlusion(alpha, strict)
        foot_columns = np.flatnonzero(strict.max(axis=0) > 4)
        underside_columns = np.flatnonzero(underside.max(axis=0) > 0.025)
        baseline = int(np.percentile(np.argwhere(strict > 4)[:, 0], 90))

        self.assertGreater(len(underside_columns), len(foot_columns) * 1.8)
        self.assertLess(len(underside_columns), 176)
        self.assertGreater(float(underside[baseline + 1:baseline + 7].max()), 0.08)
        # The real curved lower silhouette must connect continuously to the
        # foot baseline; a baseline-only stripe leaves a visible daylight
        # crescent and makes a round vessel look suspended.
        body_bottom = int(np.flatnonzero(alpha[:, 90] >= 32)[-1])
        self.assertGreater(
            float(underside[body_bottom + 1:baseline + 1, 90].max()),
            0.01,
        )
        self.assertEqual(float(underside[:, :35].max()), 0.0)
        self.assertEqual(float(underside[:, 245:].max()), 0.0)

    def test_wide_base_receiver_seam_is_dense_but_not_a_tall_dark_band(self) -> None:
        alpha = np.zeros((220, 280), np.uint8)
        alpha[35:176, 60:220] = 255
        support = np.ones_like(alpha, bool)
        protected = np.zeros_like(alpha, bool)
        lighting = Lighting.from_dict({
            "shadow_direction": [1.0, 0.0],
            "shadow_opacity": 0.30,
        })

        ao = contact_shadow(alpha, support, protected, radius=12, lighting=lighting)
        near = float(np.percentile(ao[176:179, 80:200], 50))
        far = float(np.percentile(ao[182:187, 80:200], 50))

        self.assertGreater(near, 65.0)
        self.assertLess(far, near * 0.72)

    def test_round_body_cast_shadow_has_dense_near_attachment_wider_than_foot(self) -> None:
        alpha = np.zeros((220, 260), np.uint8)
        yy, xx = np.indices(alpha.shape)
        body = ((xx - 125) / 78.0) ** 2 + ((yy - 105) / 65.0) ** 2 <= 1.0
        alpha[body] = 255
        alpha[166:178, 111:139] = 255
        lighting = Lighting.from_dict({
            "shadow_direction": [0.6, 0.8],
            "shadow_length_ratio": 0.04,
            "shadow_opacity": 0.23,
            "shadow_softness_ratio": 0.16,
        })

        cast = _cast_shadow_flat(alpha, lighting)
        near = cast[178:194]
        columns = np.flatnonzero(near.max(axis=0) > 24)

        self.assertGreater(int(columns.max() - columns.min()), 45)
        self.assertGreater(int(near.max()), 65)

    def test_seated_object_keeps_visible_contact_below_secondary_soles(self) -> None:
        """A tiny rear-most foot must not make two visible soles look airborne."""
        alpha = np.zeros((240, 300), np.uint8)
        alpha[35:160, 105:205] = 255       # body
        alpha[135:183, 58:126] = 255       # left forward shoe
        alpha[140:189, 132:198] = 255      # right forward shoe
        alpha[145:202, 190:222] = 255      # narrow rear-most support
        support = np.ones_like(alpha, bool)
        protected = np.zeros_like(alpha, bool)
        lighting = Lighting.from_dict({
            "shadow_direction": [-0.55, 0.83],
            "shadow_length_ratio": 0.04,
            "shadow_opacity": 0.25,
            "shadow_softness_ratio": 0.15,
            "shadow_tail_strength": 0.58,
        })

        ao = contact_shadow(alpha, support, protected, radius=12, lighting=lighting)
        cast = _cast_shadow_flat(alpha, lighting)

        # These receiver-side samples begin immediately below the two measured
        # shoe contours, which sit above the object's narrow global baseline.
        self.assertGreater(int(ao[183:189, 70:118].max()), 80)
        self.assertGreater(int(ao[189:195, 144:190].max()), 80)
        self.assertGreater(int(cast[183:202, 65:125].max()), 55)
        self.assertGreater(int(cast[189:207, 135:198].max()), 55)

    def test_sloped_wide_plinth_is_not_misclassified_as_a_narrow_foot(self) -> None:
        alpha = np.zeros((220, 260), np.uint8)
        alpha[45:165, 70:190] = 255
        # A six-pixel slope means the lowest 1% band sees only the downhill
        # corner, while Hough-reviewed grounding still finds the complete base.
        for x in range(70, 190):
            bottom = 166 + int(round((x - 70) * 6 / 119))
            alpha[164:bottom + 1, x] = 255

        strict = contact_profile(alpha)
        grounding = _grounding_profile(alpha)

        strict_x = np.flatnonzero(strict.max(axis=0) > 4)
        ground_x = np.flatnonzero(grounding.max(axis=0) > 4)
        self.assertLess(len(strict_x), len(ground_x))
        self.assertLess(_foot_narrowness(alpha, strict, grounding), 0.15)

    def test_true_small_round_foot_retains_high_continuous_narrowness(self) -> None:
        alpha = np.zeros((220, 260), np.uint8)
        yy, xx = np.indices(alpha.shape)
        alpha[((xx - 130) / 78.0) ** 2 + ((yy - 105) / 62.0) ** 2 <= 1.0] = 255
        alpha[165:178, 112:148] = 255

        self.assertGreater(_foot_narrowness(alpha), 0.60)

    def test_cast_shadow_tapers_before_support_boundary(self) -> None:
        alpha = np.zeros((220, 260), np.uint8)
        alpha[70:170, 95:145] = 255
        support = np.zeros_like(alpha, bool)
        support[20:205, 20:230] = True
        protected = np.zeros_like(support)
        lighting = Lighting.from_dict({
            "shadow_direction": [0.8, 0.6],
            "shadow_length_ratio": 0.08,
            "shadow_opacity": 0.34,
            "shadow_softness_ratio": 0.16,
        })

        cast = cast_shadow(alpha, support, lighting, protected)

        self.assertEqual(int(cast[~support].max()), 0)
        boundary_peak = int(cast[202:205, 20:230].max())
        interior_peak = int(cast[180:198, 20:230].max())
        self.assertGreater(interior_peak, 0)
        self.assertLess(boundary_peak, interior_peak)

    def test_physical_contact_ignores_round_belly_and_low_alpha_tail(self) -> None:
        alpha = np.zeros((220, 260), np.uint8)
        yy, xx = np.indices(alpha.shape)
        body = ((xx - 125) / 78.0) ** 2 + ((yy - 105) / 65.0) ** 2 <= 1.0
        alpha[body] = 255
        alpha[166:178, 111:139] = 255
        alpha[180, 30:230] = 8  # resampling/matte tail, not a physical foot

        profile = contact_profile(alpha)
        columns = np.flatnonzero(profile.max(axis=0) > 0)

        self.assertGreaterEqual(int(columns.min()), 108)
        self.assertLessEqual(int(columns.max()), 142)
        self.assertEqual(int(profile[180].max()), 0)


if __name__ == "__main__":
    unittest.main()
