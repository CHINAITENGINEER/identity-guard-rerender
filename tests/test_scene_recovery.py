from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from strict_compositor.scene import (
    _clean_surface,
    _clip_recovered_plane_at_front_edge,
    _dominant_receiver_instance,
    _elliptical_glass_support,
    _lighting_proxy,
    _protected_from_semantics,
    _semantic_seed_needs_central_recovery,
    _support_geometry_is_suspicious,
    analyze_scene,
)


class SceneRecoveryTests(unittest.TestCase):
    def test_dominant_receiver_separates_two_tables_joined_by_narrow_bridge(self) -> None:
        mask = np.zeros((300, 500), np.uint8)
        # Main lower-central receiver.
        cv2.rectangle(mask, (115, 135), (455, 265), 1, -1)
        # A smaller rear-left white table accidentally joined in projection.
        cv2.rectangle(mask, (18, 45), (190, 115), 1, -1)
        cv2.line(mask, (174, 112), (136, 142), 1, 5)

        selected = _dominant_receiver_instance(mask)

        self.assertGreater(float(selected[150:250, 140:440].mean()), 0.92)
        self.assertEqual(int(selected[55:100, 30:170].max()), 0)
        self.assertEqual(int(selected[105:130, 155:185].max()), 0)

    def test_dominant_receiver_ignores_small_tail_created_by_opening(self) -> None:
        mask = np.zeros((260, 420), np.uint8)
        cv2.rectangle(mask, (45, 70), (375, 205), 1, -1)
        # A narrow support tail is not a second substantial receiver instance.
        cv2.rectangle(mask, (198, 205), (212, 252), 1, -1)

        selected = _dominant_receiver_instance(mask)

        np.testing.assert_array_equal(selected, mask)

    def test_clean_surface_reconnects_occlusion_split_single_table(self) -> None:
        mask = np.zeros((240, 420), np.uint8)
        cv2.rectangle(mask, (30, 70), (196, 190), 1, -1)
        cv2.rectangle(mask, (204, 70), (390, 190), 1, -1)

        cleaned = _clean_surface(mask)

        self.assertGreater(float(cleaned[90:170, 50:370].mean()), 0.90)
        self.assertGreater(int(cleaned[120, 198:206].max()), 0)

    def test_clean_surface_keeps_only_dominant_narrow_bridge_lobe(self) -> None:
        mask = np.zeros((300, 500), np.uint8)
        cv2.rectangle(mask, (105, 138), (460, 265), 1, -1)
        cv2.rectangle(mask, (20, 42), (185, 112), 1, -1)
        cv2.line(mask, (170, 108), (128, 145), 1, 5)

        cleaned = _clean_surface(mask)

        self.assertGreater(float(cleaned[160:245, 135:440].mean()), 0.86)
        self.assertEqual(int(cleaned[50:100, 30:170].max()), 0)

    def test_scene_analysis_publishes_only_dominant_joined_receiver(self) -> None:
        height, width = 300, 500
        image = np.full((height, width, 3), 214, np.uint8)
        joined = np.zeros((height, width), np.uint8)
        cv2.rectangle(joined, (105, 138), (460, 265), 1, -1)
        cv2.rectangle(joined, (20, 42), (185, 112), 1, -1)
        cv2.line(joined, (170, 108), (128, 145), 1, 5)
        relative_depth = np.linspace(
            0.2, 0.8, height, dtype=np.float32,
        )[:, None].repeat(width, axis=1)
        relative_normal = np.zeros((height, width, 3), np.float32)
        relative_normal[..., 1] = 1.0

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            background = root / "synthetic_two_white_tables.png"
            cv2.imwrite(
                str(background),
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            )
            no_glass = np.zeros_like(joined)
            with (
                patch(
                    "strict_compositor.scene._elliptical_glass_support",
                    return_value=(no_glass, 0.0),
                ),
                patch(
                    "strict_compositor.scene._semantic_support",
                    return_value=(joined, [("table", joined)]),
                ),
                patch(
                    "strict_compositor.scene._horizontal_from_depth",
                    return_value=(joined, relative_depth, relative_normal),
                ),
                patch(
                    "strict_compositor.scene._fit_support_depth_plane",
                    return_value=joined,
                ),
            ):
                analysis = analyze_scene(background, root / "cache", deep=True)

            published = cv2.imread(
                analysis.support_mask_path,
                cv2.IMREAD_GRAYSCALE,
            )
            self.assertIsNotNone(published)
            assert published is not None
            self.assertGreater(float((published[160:245, 135:440] > 0).mean()), 0.86)
            self.assertEqual(int(published[50:100, 30:170].max()), 0)
            self.assertEqual(analysis.method, "semantic-depth-plane")

    def test_frame_dominating_wall_mask_is_not_receiver_geometry(self) -> None:
        mask = np.ones((240, 360), np.uint8)
        mask[145:215, 80:285] = 0

        self.assertTrue(_support_geometry_is_suspicious(mask))

    def test_visible_rim_and_legs_recover_round_glass_table(self) -> None:
        image = np.full((480, 720, 3), (196, 181, 145), np.uint8)
        image = np.clip(
            image.astype(np.int16)
            + np.linspace(12, -8, 720, dtype=np.int16)[None, :, None],
            0,
            255,
        ).astype(np.uint8)
        centre, axes = (345, 318), (190, 75)
        tinted = image.copy()
        cv2.ellipse(tinted, centre, axes, 0, 0, 360, (112, 105, 88), -1)
        image = cv2.addWeighted(tinted, 0.52, image, 0.48, 0)
        cv2.ellipse(image, centre, axes, 0, 0, 360, (44, 43, 39), 4)
        cv2.line(image, (326, 389), (300, 479), (35, 35, 32), 11)
        cv2.line(image, (365, 389), (397, 479), (35, 35, 32), 11)
        cv2.ellipse(image, centre, (48, 25), 0, 0, 360, (62, 60, 55), -1)

        support, confidence = _elliptical_glass_support(image)

        ys, xs = np.nonzero(support)
        self.assertGreater(confidence, 0.85)
        self.assertGreater(float(support.mean()), 0.08)
        self.assertAlmostEqual(float(xs.mean() / image.shape[1]), 0.48, delta=0.06)
        self.assertAlmostEqual(float(ys.mean() / image.shape[0]), 0.66, delta=0.06)
        self.assertFalse(_support_geometry_is_suspicious(support))

    def test_round_soft_cushion_without_pedestal_is_not_glass_table(self) -> None:
        image = np.full((480, 720, 3), (204, 174, 170), np.uint8)
        cv2.ellipse(image, (345, 318), (190, 75), 0, 0, 360, (181, 112, 126), -1)
        cv2.ellipse(image, (345, 318), (190, 75), 0, 0, 360, (135, 76, 92), 4)

        support, confidence = _elliptical_glass_support(image)

        self.assertFalse(support.any())
        self.assertLess(confidence, 0.78)

    def test_automatic_indoor_shadow_remains_visible_at_browser_scale(self) -> None:
        image = np.full((240, 360, 3), 190, np.uint8)
        image[:, :, 0] += np.linspace(0, 35, 360, dtype=np.uint8)[None, :]
        support = np.zeros((240, 360), bool)
        support[35:220, 20:340] = True

        _, length, opacity, softness = _lighting_proxy(image, support)

        self.assertGreaterEqual(length, 0.038)
        self.assertGreaterEqual(opacity, 0.22)
        self.assertGreaterEqual(softness, 0.12)

    def test_small_low_semantic_table_seed_is_suspicious(self) -> None:
        mask = np.zeros((200, 300), np.uint8)
        mask[155:190, 40:160] = 1

        self.assertTrue(_semantic_seed_needs_central_recovery(mask))

    def test_normal_central_table_seed_is_not_suspicious(self) -> None:
        mask = np.zeros((200, 300), np.uint8)
        mask[70:160, 20:280] = 1

        self.assertFalse(_semantic_seed_needs_central_recovery(mask))

    def test_recovered_plane_stops_at_strong_front_edge(self) -> None:
        image = np.full((200, 300, 3), 155, np.uint8)
        image[132:, :] = 25
        plane = np.zeros((200, 300), np.uint8)
        plane[:165, 10:295] = 1

        clipped = _clip_recovered_plane_at_front_edge(image, plane)
        rows = np.flatnonzero(clipped.any(axis=1))

        self.assertLessEqual(int(rows.max()), 133)
        self.assertGreater(float(clipped.mean()), 0.50)

    def test_windowpane_misclassification_is_not_a_protected_object(self) -> None:
        support = np.zeros((100, 160), np.uint8)
        support[15:75, 10:150] = 1
        pane = np.zeros_like(support)
        pane[20:70, 20:145] = 1

        protected = _protected_from_semantics([("windowpane", pane)], support)

        self.assertEqual(protected, [])


if __name__ == "__main__":
    unittest.main()
