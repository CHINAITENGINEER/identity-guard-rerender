from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from strict_compositor.hybrid import (
    HybridRenderError,
    load_registered_candidate,
    recover_hybrid_candidate,
)
from strict_compositor.identity import (
    build_identity_masks,
    conservative_source_identity_mask,
    warp_identity_mask,
)
from strict_compositor.contracts import Job
from strict_compositor.image_io import save_png
from strict_compositor.pipeline import StrictCompositor


class IdentityRoutingTests(unittest.TestCase):
    def alpha(self) -> np.ndarray:
        alpha = np.zeros((120, 100), np.uint8)
        alpha[10:112, 20:82] = 255
        alpha[45:72, 56:68] = 0  # handle/cavity boundary must remain guarded
        return alpha

    def test_unknown_clear_pitcher_opens_only_interior(self) -> None:
        alpha = self.alpha()
        bundle = build_identity_masks(
            alpha,
            material_mode="clear_glass",
            semantic_category="glass_pitcher",
            physical_type="rigid_vessel",
        )
        self.assertEqual(bundle.route, "regional_rerender")
        self.assertTrue(bundle.rerender_allow_mask[60, 40])
        self.assertFalse(bundle.rerender_allow_mask[10, 40])
        self.assertFalse(bundle.rerender_allow_mask[58, 55])
        self.assertFalse(bundle.rerender_allow_mask[108, 40])
        self.assertTrue(bundle.structure_guard_mask[108, 40])
        self.assertGreater(bundle.provenance["base_structure_guard_pixels"], 0)
        self.assertFalse(np.any(bundle.rerender_allow_mask & bundle.structure_guard_mask))

    def test_packaged_can_is_entirely_strict(self) -> None:
        alpha = self.alpha()
        bundle = build_identity_masks(
            alpha,
            material_mode="opaque_matte",
            semantic_category="aluminum_beverage_can",
            physical_type="rigid_vessel",
        )
        self.assertEqual(bundle.route, "strict_product")
        self.assertFalse(np.any(bundle.rerender_allow_mask))
        self.assertTrue(np.array_equal(bundle.hard_restore_mask, alpha > 4))

    def test_opaque_can_source_guard_is_full_silhouette(self) -> None:
        rgba = np.zeros((140, 90, 4), np.uint8)
        rgba[10:132, 18:72, :3] = (75, 30, 40)
        rgba[10:132, 18:72, 3] = 255
        mask, reason = conservative_source_identity_mask(
            rgba,
            material_mode="metal",
            semantic_category="aluminum_can",
        )
        self.assertEqual(reason, "opaque_full_silhouette_identity_guard_v2")
        self.assertIsNotNone(mask)
        self.assertTrue(np.array_equal(mask > 0, rgba[..., 3] > 4))

    def test_mixed_bottle_partitions_label_without_locking_all_glass(self) -> None:
        rgba = np.zeros((180, 100, 4), np.uint8)
        rgba[8:174, 22:78, :3] = (184, 135, 62)  # amber contents
        rgba[8:174, 22:78, 3] = 255
        rgba[64:146, 26:74, :3] = (18, 18, 17)   # opaque label
        rgba[76:134:9, 32:68, :3] = 235          # bright printed strokes
        rgba[8:28, 35:65, :3] = 12               # cap
        mask, reason = conservative_source_identity_mask(
            rgba,
            material_mode="mixed_transparent",
            semantic_category="whiskey_bottle",
        )
        self.assertEqual(reason, "material_partition_identity_guard_v2")
        self.assertIsNotNone(mask)
        body = rgba[..., 3] > 4
        self.assertTrue(mask[92, 50] > 0)          # label is strict
        self.assertTrue(mask[16, 50] > 0)          # closure is strict
        self.assertLess(np.count_nonzero(mask), np.count_nonzero(body))
        bundle = build_identity_masks(
            rgba[..., 3],
            material_mode="mixed_transparent",
            semantic_category="whiskey_bottle",
            placed_identity_mask=mask,
        )
        self.assertEqual(bundle.route, "regional_rerender")
        self.assertTrue(np.any(bundle.rerender_allow_mask))

    def test_detected_text_blocks_even_clear_glass(self) -> None:
        bundle = build_identity_masks(
            self.alpha(),
            material_mode="clear_glass",
            semantic_category="glass_pitcher",
            identity_status="detected",
        )
        self.assertEqual(bundle.route, "strict_product")
        self.assertFalse(np.any(bundle.rerender_allow_mask))

    def test_regional_identity_mask_stays_hard_restored(self) -> None:
        alpha = self.alpha()
        identity = np.zeros_like(alpha)
        identity[68:86, 32:62] = 255
        bundle = build_identity_masks(
            alpha,
            material_mode="clear_glass",
            semantic_category="glass_pitcher",
            identity_status="detected",
            rerender_policy="regional",
            placed_identity_mask=identity,
        )
        self.assertEqual(bundle.route, "regional_rerender")
        self.assertTrue(bundle.hard_restore_mask[74, 40])
        self.assertFalse(bundle.rerender_allow_mask[74, 40])
        self.assertTrue(bundle.rerender_allow_mask[35, 40])

    def test_regional_detected_without_mask_fails_closed(self) -> None:
        alpha = self.alpha()
        bundle = build_identity_masks(
            alpha,
            material_mode="clear_glass",
            semantic_category="glass_pitcher",
            identity_status="detected",
            rerender_policy="regional",
        )
        self.assertEqual(bundle.route, "strict_product")
        self.assertFalse(np.any(bundle.rerender_allow_mask))
        self.assertTrue(np.array_equal(bundle.hard_restore_mask, alpha > 4))
        self.assertEqual(
            bundle.provenance["route_reason"],
            "regional_identity_mask_missing",
        )

    def test_warped_identity_never_leaks_outside_alpha(self) -> None:
        source = np.zeros((30, 30), np.uint8)
        source[6:24, 8:22] = 255
        alpha = np.zeros((80, 90), np.uint8)
        alpha[20:60, 30:70] = 255
        matrix = np.asarray([[1.5, 0, 20], [0, 1.5, 10], [0, 0, 1]], np.float64)
        warped = warp_identity_mask(source, matrix, (80, 90), alpha)
        self.assertFalse(np.any((warped > 0) & (alpha <= 4)))


class HybridRecoveryTests(unittest.TestCase):
    def test_clear_glass_optical_polish_restores_only_missing_sparse_glints(self) -> None:
        base = np.full((120, 140, 3), 96, np.uint8)
        candidate = np.full_like(base, 96)
        allow = np.zeros((120, 140), bool)
        allow[12:108, 18:122] = True
        guard = np.zeros_like(allow)
        # Target-derived deterministic interface evidence: a narrow rim/glint,
        # not a broad product-wide brightness change.
        base[32:34, 38:104] = 150
        optical = allow.copy()

        result, report = recover_hybrid_candidate(
            base,
            candidate,
            allow,
            guard,
            optical_polish_mask=optical,
        )

        self.assertGreater(report["optical_polish_pixels"], 0)
        self.assertLessEqual(report["optical_polish_max_delta"], 14.0)
        self.assertEqual(
            report["optical_polish_mode"],
            "sparse_interface_residual_v2",
        )
        self.assertEqual(
            report["recovery_profile"],
            "clear_optical_interface",
        )
        self.assertLessEqual(report["recovery_feather_pixels"], 8.0)
        self.assertGreater(
            int(result[30:36, 38:104].max()),
            96,
        )
        self.assertTrue(np.array_equal(result[~allow], base[~allow]))

    def test_candidate_changes_only_allowed_pixels(self) -> None:
        base = np.full((80, 100, 3), 40, np.uint8)
        candidate = np.full_like(base, 220)
        allow = np.zeros((80, 100), bool)
        allow[15:68, 20:85] = True
        guard = np.zeros_like(allow)
        guard[32:48, 45:68] = True
        result, report = recover_hybrid_candidate(base, candidate, allow, guard)
        self.assertEqual(report["status"], "applied")
        self.assertEqual(report["identity_changed_pixels"], 0)
        self.assertEqual(report["outside_changed_pixels"], 0)
        self.assertTrue(np.array_equal(result[guard], base[guard]))
        self.assertTrue(np.array_equal(result[~allow], base[~allow]))
        self.assertGreater(float(result[40, 30].mean()), 100.0)

    def test_no_eligible_region_returns_byte_exact_base(self) -> None:
        base = np.arange(60 * 80 * 3, dtype=np.uint8).reshape(60, 80, 3)
        result, report = recover_hybrid_candidate(
            base, np.zeros_like(base), np.zeros((60, 80), bool),
            np.zeros((60, 80), bool),
        )
        self.assertEqual(report["status"], "strict_fallback")
        self.assertTrue(np.array_equal(result, base))

    def test_large_hybrid_region_uses_scale_aware_smooth_edge_recovery(self) -> None:
        base = np.full((420, 360, 3), 60, np.uint8)
        candidate = np.full_like(base, 220)
        allow = np.zeros((420, 360), bool)
        allow[35:390, 45:315] = True
        guard = np.zeros_like(allow)
        result, report = recover_hybrid_candidate(
            base, candidate, allow, guard,
        )
        self.assertGreater(report["recovery_feather_pixels"], 5.0)
        # The first interior pixel stays visually tied to the deterministic
        # rim, while the central optical region still adopts the candidate.
        self.assertLess(int(result[36, 180, 0]), 90)
        self.assertGreater(int(result[210, 180, 0]), 200)
        self.assertTrue(np.array_equal(result[~allow], base[~allow]))

    def test_aspect_compatible_candidate_is_resized_but_wrong_aspect_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compatible = root / "compatible.png"
            wrong = root / "wrong.png"
            Image.fromarray(np.full((100, 150, 3), 127, np.uint8)).save(compatible)
            Image.fromarray(np.full((100, 120, 3), 127, np.uint8)).save(wrong)
            image, report = load_registered_candidate(compatible, (200, 300))
            self.assertEqual(image.shape, (200, 300, 3))
            self.assertEqual(report["registration"], "aspect_compatible_resize")
            with self.assertRaises(HybridRenderError):
                load_registered_candidate(wrong, (200, 300))

    def test_pipeline_applies_candidate_only_to_clear_pitcher_interior(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            background = np.full((140, 190, 3), [180, 150, 110], np.uint8)
            product = np.zeros((80, 55, 4), np.uint8)
            product[5:75, 5:50, :3] = [195, 190, 180]
            product[5:75, 5:50, 3] = 150
            product[68:77, 10:45, 3] = 235
            candidate = np.full_like(background, [35, 145, 225])
            save_png(root / "background.png", background)
            save_png(root / "product.png", product)
            save_png(root / "candidate.png", candidate)
            job = Job.from_dict({
                "background": "background.png",
                "support_polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
                "lighting": {
                    "shadow_direction": [1, 0.2],
                    "shadow_length_ratio": 0.0,
                    "cast_shadow_mode": "ambient_only",
                    "surface_material": "transparent_glass",
                },
                "products": [{
                    "id": "pitcher",
                    "rgba": "product.png",
                    "homography": [[1, 0, 70], [0, 1, 35], [0, 0, 1]],
                    "material_mode": "clear_glass",
                    "material_confidence": 0.9,
                    "transmission_strength": 0.92,
                    "semantic_category": "glass_pitcher",
                }],
                "hybrid_render": {
                    "enabled": True,
                    "provider": "manual_candidate",
                    "candidate": "candidate.png",
                },
            }, root)
            outcome = StrictCompositor().compose(
                job, root / "out", strict_audit=False,
            )
            report = __import__("json").loads(
                (root / "out" / "audit.json").read_text(encoding="utf-8")
            )
            hybrid = report["hybrid_render"]
            self.assertEqual(hybrid["status"], "applied")
            self.assertEqual(hybrid["outside_changed_pixels"], 0)
            self.assertEqual(hybrid["identity_changed_pixels"], 0)
            strict = np.asarray(Image.open(root / "out" / "strict_base.png").convert("RGB"))
            allow = np.asarray(Image.open(root / "out" / "hybrid_rerender_mask.png").convert("L")) > 0
            self.assertTrue(np.array_equal(outcome.final[~allow], strict[~allow]))
            self.assertTrue(np.any(outcome.final[allow] != strict[allow]))


if __name__ == "__main__":
    unittest.main()
