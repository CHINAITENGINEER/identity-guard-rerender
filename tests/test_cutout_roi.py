from __future__ import annotations

import numpy as np
import cv2
import unittest

from strict_compositor.cutout import (
    _semantic_refinement_preservation,
    _select_rembg_candidate,
    alpha_quality,
    crop_normalized_roi,
    remove_attached_bottom_surface_residue,
    solidify_semantic_opaque_panel,
)
from strict_compositor.errors import ContractError


class CutoutRoiTests(unittest.TestCase):

    def test_compact_opaque_figurine_fills_only_micro_internal_hole(self) -> None:
        rgba = np.zeros((240, 300, 4), np.uint8)
        cv2.ellipse(rgba, (150, 125), (112, 88), 0, 0, 360, (75, 150, 62, 255), -1)
        # A highlight-sized false transparency and a much larger structural
        # opening: only the former may be restored.
        alpha = np.ascontiguousarray(rgba[..., 3])
        cv2.circle(alpha, (125, 95), 3, 0, -1)
        cv2.circle(alpha, (180, 135), 24, 0, -1)
        rgba[..., 3] = alpha
        original_rgb = rgba[..., :3].copy()

        repaired, details = solidify_semantic_opaque_panel(
            rgba, "ceramic fish figurine",
        )

        self.assertTrue(details["applied"])
        self.assertEqual(
            details["geometry_model"], "bounded_enclosed_microhole_fill",
        )
        self.assertGreater(int(repaired[95, 125, 3]), 250)
        self.assertEqual(int(repaired[135, 180, 3]), 0)
        self.assertTrue(np.array_equal(repaired[..., :3], original_rgb))
        self.assertFalse(details["requires_transparent_audit_border"])
    def test_fragmented_clock_face_is_solidified_from_source_rgb(self) -> None:
        rgba = np.zeros((500, 260, 4), np.uint8)
        yy, xx = np.indices(rgba.shape[:2])
        rgba[..., :3] = np.stack(
            ((xx + 40) % 255, (yy + 60) % 255, np.full_like(xx, 120)),
            axis=2,
        ).astype(np.uint8)
        alpha = np.zeros(rgba.shape[:2], np.uint8)
        roof = np.array([[130, 8], [18, 150], [242, 150]], np.int32)
        cv2.fillConvexPoly(alpha, roof, 255)
        cv2.rectangle(alpha, (18, 145), (242, 485), 255, thickness=12)
        # Mimic a segmented printed face: detached glyph islands plus frame.
        cv2.circle(alpha, (130, 250), 14, 255, thickness=cv2.FILLED)
        cv2.rectangle(alpha, (85, 470), (175, 482), 255, thickness=cv2.FILLED)
        rgba[..., 3] = alpha
        source_rgb = rgba[..., :3].copy()

        repaired, details = solidify_semantic_opaque_panel(
            rgba, "house-shaped wooden desk clock",
        )

        self.assertTrue(details["applied"])
        self.assertGreater(repaired[300, 130, 3], 250)
        np.testing.assert_array_equal(repaired[..., :3], source_rgb)

    def test_house_clock_repair_restores_missing_lower_corner(self) -> None:
        rgba = np.zeros((600, 300, 4), np.uint8)
        yy, xx = np.indices(rgba.shape[:2])
        rgba[..., 0] = (xx * 5 + yy) % 251
        rgba[..., 1] = (yy * 3) % 253
        rgba[..., 2] = 171
        alpha = np.zeros(rgba.shape[:2], np.uint8)
        roof = np.asarray([[150, 8], [20, 185], [280, 185]], np.int32)
        cv2.fillConvexPoly(alpha, roof, 255)
        cv2.rectangle(alpha, (20, 175), (280, 320), 255, thickness=cv2.FILLED)
        # Fragmented printed face plus an incomplete lower rail whose missing
        # right corner used to make the global hull cut diagonally upward.
        cv2.rectangle(alpha, (20, 315), (35, 565), 255, thickness=cv2.FILLED)
        cv2.rectangle(alpha, (265, 315), (280, 430), 255, thickness=cv2.FILLED)
        cv2.rectangle(alpha, (20, 560), (225, 585), 255, thickness=cv2.FILLED)
        cv2.circle(alpha, (150, 390), 12, 255, thickness=cv2.FILLED)
        rgba[..., 3] = alpha
        source_rgb = rgba[..., :3].copy()

        repaired, details = solidify_semantic_opaque_panel(
            rgba, "house-shaped wooden desk clock",
        )

        self.assertTrue(details["applied"])
        self.assertEqual(
            details["geometry_model"],
            "house_clock_roof_hull_plus_vertical_cabinet",
        )
        self.assertGreater(repaired[570, 270, 3], 245)
        np.testing.assert_array_equal(repaired[..., :3], source_rgb)

    def test_fragmented_opaque_birdhouse_face_is_solidified(self) -> None:
        rgba = np.zeros((640, 420, 4), np.uint8)
        yy, xx = np.indices(rgba.shape[:2])
        rgba[..., 0] = (xx * 3 + 31) % 251
        rgba[..., 1] = (yy * 5 + 43) % 253
        rgba[..., 2] = 167
        alpha = np.zeros(rgba.shape[:2], np.uint8)
        shell = np.asarray(
            [[210, 8], [18, 240], [76, 620], [344, 620], [402, 240]],
            np.int32,
        )
        cv2.polylines(alpha, [shell], True, 255, thickness=22)
        # Mimic a segmenter that retained only printed decoration islands and
        # the outer wooden rails while dropping the pale, physically opaque
        # face of the birdhouse.
        cv2.circle(alpha, (210, 330), 26, 255, thickness=cv2.FILLED)
        cv2.rectangle(alpha, (170, 500), (250, 530), 255, thickness=cv2.FILLED)
        for y in range(270, 560, 24):
            cv2.line(alpha, (108, y), (312, y), 255, thickness=7)
        rgba[..., 3] = alpha
        source_rgb = rgba[..., :3].copy()

        repaired, details = solidify_semantic_opaque_panel(
            rgba, "decorative wooden birdhouse with painted floral design",
        )

        self.assertTrue(details["applied"])
        self.assertEqual(
            details["geometry_model"],
            "birdhouse_dominant_external_contour",
        )
        self.assertEqual(details["repair_expansion_verdict"], "pass")
        self.assertLess(details["max_expansion_ratio"], 3.5)
        self.assertLess(details["repair_confidence_score"], 98.0)
        self.assertGreater(repaired[420, 210, 3], 250)
        np.testing.assert_array_equal(repaired[..., :3], source_rgb)

    def test_transparent_or_handled_subject_is_never_solidified(self) -> None:
        rgba = np.zeros((200, 200, 4), np.uint8)
        alpha = np.zeros(rgba.shape[:2], np.uint8)
        cv2.circle(alpha, (100, 100), 80, 255, thickness=8)
        rgba[..., 3] = alpha
        unchanged, details = solidify_semantic_opaque_panel(
            rgba, "transparent glass clock with handle",
        )
        self.assertFalse(details["applied"])
        np.testing.assert_array_equal(unchanged, rgba)

    def test_birdhouse_repair_preserves_under_eave_negative_space(self) -> None:
        rgba = np.zeros((620, 420, 4), np.uint8)
        rgba[..., :3] = [176, 158, 112]
        alpha = np.zeros(rgba.shape[:2], np.uint8)
        outline = np.asarray([
            [210, 8], [18, 205], [72, 205], [92, 565],
            [328, 565], [348, 205], [402, 205],
        ], np.int32)
        cv2.polylines(alpha, [outline], True, 255, thickness=18)
        cv2.circle(alpha, (210, 330), 22, 255, thickness=cv2.FILLED)
        for y in range(270, 545, 24):
            cv2.line(alpha, (112, y), (308, y), 255, thickness=7)
        rgba[..., 3] = alpha

        repaired, details = solidify_semantic_opaque_panel(
            rgba, "painted wooden birdhouse",
        )

        self.assertTrue(details["applied"])
        self.assertGreater(repaired[360, 210, 3], 245)
        # The roof tips are real, but the air directly below them is not part
        # of the cabinet.  A row envelope used to expose source background here.
        self.assertEqual(int(repaired[310, 45, 3]), 0)
        self.assertEqual(int(repaired[310, 375, 3]), 0)

    def test_solid_figurine_with_large_internal_holes_is_rejected(self) -> None:
        rgba = np.zeros((260, 360, 4), np.uint8)
        rgba[..., :3] = [202, 116, 48]
        alpha = np.zeros(rgba.shape[:2], np.uint8)
        cv2.ellipse(
            alpha, (180, 145), (145, 92), 0, 0, 360, 255,
            thickness=cv2.FILLED,
        )
        for x, y, radius in ((115, 105, 23), (170, 130, 29), (220, 92, 20)):
            cv2.circle(alpha, (x, y), radius, 0, thickness=cv2.FILLED)
        rgba[..., 3] = alpha

        solid = alpha_quality(rgba, "painted ceramic tiger figurine")
        handled = alpha_quality(rgba, "decorative handled wire basket")

        self.assertEqual(solid["verdict"], "reject")
        self.assertTrue(solid["opaque_topology_suspect"])
        self.assertGreater(solid["internal_hole_fraction"], 0.012)
        self.assertFalse(handled["opaque_topology_suspect"])

    def test_semantic_candidate_selection_avoids_perforated_sam2(self) -> None:
        rgb = np.full((260, 360, 3), [202, 116, 48], np.uint8)
        complete = np.zeros((260, 360), np.uint8)
        cv2.ellipse(
            complete, (180, 145), (145, 92), 0, 0, 360, 255,
            thickness=cv2.FILLED,
        )
        perforated = complete.copy()
        for x, y, radius in ((115, 105, 23), (170, 130, 29), (220, 92, 20)):
            cv2.circle(perforated, (x, y), radius, 0, thickness=cv2.FILLED)

        selected, details = _select_rembg_candidate(
            {
                "u2net": np.dstack((rgb, complete)),
                "sam2-box-recovery": np.dstack((rgb, perforated)),
            },
            subject_label="painted ceramic tiger figurine",
        )

        self.assertEqual(selected, "u2net")
        self.assertEqual(
            details["candidate_metrics"]["sam2-box-recovery"]["quality"]["verdict"],
            "reject",
        )

    def test_low_saturation_attached_bottom_surface_tail_is_removed(self) -> None:
        rgba = np.zeros((500, 320, 4), np.uint8)
        rgba[..., :3] = [65, 170, 105]
        alpha = np.zeros(rgba.shape[:2], np.uint8)
        cv2.rectangle(alpha, (70, 35), (250, 445), 255, cv2.FILLED)
        tail = np.asarray([[145, 446], [175, 446], [160, 492]], np.int32)
        cv2.fillConvexPoly(alpha, tail, 255)
        rgba[..., 3] = alpha
        tail_mask = np.zeros(rgba.shape[:2], np.uint8)
        cv2.fillConvexPoly(tail_mask, tail, 255)
        rgba[tail_mask > 0, :3] = [132, 136, 132]
        source_rgb = rgba[..., :3].copy()

        repaired, details = remove_attached_bottom_surface_residue(
            rgba, "porcelain seated girl figurine",
        )

        self.assertTrue(details["applied"])
        self.assertEqual(details["verdict"], "pass")
        self.assertEqual(int(repaired[480, 160, 3]), 0)
        self.assertGreater(int(repaired[430, 160, 3]), 250)
        np.testing.assert_array_equal(repaired[..., :3], source_rgb)

    def test_semantic_refinement_rejects_round_rim_and_stand_deletion(self) -> None:
        """A coarse semantic envelope must not amputate product structure."""
        original = np.zeros((512, 512), np.uint8)
        cv2.circle(original, (256, 190), 138, 255, thickness=cv2.FILLED)
        cv2.rectangle(original, (246, 320), (266, 420), 255, thickness=cv2.FILLED)
        cv2.rectangle(original, (190, 410), (322, 452), 255, thickness=cv2.FILLED)

        # Simulate a low-resolution semantic allow-list: the round body is
        # shrunken on every side and the dark support is classified as table.
        refined = cv2.erode(
            original,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)),
        )
        refined[338:, :] = 0

        audit = _semantic_refinement_preservation(original, refined)

        self.assertFalse(audit["preserves_subject"])
        self.assertIn(
            "widespread_subject_boundary_erosion", audit["reasons"],
        )
        self.assertIn(
            "bottom_support_structure_removed", audit["reasons"],
        )
        self.assertLess(audit["bottom_support_retention"], 0.48)

    def test_semantic_refinement_allows_local_one_sided_shadow_cleanup(self) -> None:
        """A material but localized attached source shadow may be removed."""
        original = np.zeros((420, 560), np.uint8)
        cv2.rectangle(original, (185, 60), (385, 360), 255, thickness=cv2.FILLED)
        cv2.ellipse(
            original, (125, 275), (105, 46), -12, 0, 360, 255,
            thickness=cv2.FILLED,
        )
        cv2.rectangle(original, (150, 258), (205, 305), 255, thickness=cv2.FILLED)
        refined = np.zeros_like(original)
        cv2.rectangle(refined, (185, 60), (385, 360), 255, thickness=cv2.FILLED)

        audit = _semantic_refinement_preservation(original, refined)

        self.assertLess(audit["retained_ratio"], 0.94)
        self.assertTrue(audit["preserves_subject"])
        self.assertEqual(audit["reasons"], [])
        self.assertLess(audit["boundary_damage_coverage"], 0.50)

    def test_semantic_refinement_allows_narrow_boundary_polish(self) -> None:
        original = np.zeros((360, 360), np.uint8)
        cv2.circle(original, (180, 180), 126, 255, thickness=cv2.FILLED)
        refined = cv2.erode(
            original,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )

        audit = _semantic_refinement_preservation(original, refined)

        self.assertGreater(audit["retained_ratio"], 0.94)
        self.assertTrue(audit["preserves_subject"])

    def test_ensemble_does_not_fill_a_handle_or_strap_hole(self) -> None:
        height, width = 180, 180
        rgb = np.full((height, width, 3), 140, np.uint8)
        correct = np.zeros((height, width), np.uint8)
        correct[82:162, 38:142] = 255
        cv2.line(correct, (45, 88), (90, 18), 255, 9)
        cv2.line(correct, (90, 18), (135, 88), 255, 9)
        incorrect = correct.copy()
        triangle = np.asarray([[45, 88], [90, 18], [135, 88]], np.int32)
        cv2.fillPoly(incorrect, [triangle], 255)
        candidates = {
            "hole_correct": np.dstack((rgb, correct)),
            "hole_filled": np.dstack((rgb, incorrect)),
        }

        selected, details = _select_rembg_candidate(candidates)

        self.assertEqual(selected, "hole_correct")
        pair = details["pair_metrics"][0]
        self.assertGreater(
            pair["hole_filled_added_inside_hole_correct_holes"], 0.80,
        )

    def test_crop_normalized_roi_sorts_clips_and_preserves_pixels(self) -> None:
        image = np.arange(100 * 200 * 3, dtype=np.uint32).reshape(100, 200, 3)
        image = (image % 251).astype(np.uint8)

        crop, rect = crop_normalized_roi(image, [0.75, 0.80, 0.25, 0.20])

        self.assertEqual(rect, (50, 20, 150, 80))
        self.assertTrue(np.array_equal(crop, image[20:80, 50:150]))

    def test_crop_normalized_roi_rejects_tiny_box(self) -> None:
        image = np.zeros((100, 200, 3), np.uint8)

        with self.assertRaisesRegex(ContractError, "too small"):
            crop_normalized_roi(image, [0.50, 0.50, 0.51, 0.51])

    def test_alpha_quality_rejects_background_attached_to_opposite_edges(self) -> None:
        rgba = np.zeros((120, 240, 4), np.uint8)
        rgba[..., :3] = [120, 90, 55]
        rgba[20:100, :, 3] = 255
        # The actual product has become a transparent hole in a retained wall.
        rgba[48:72, 35:205, 3] = 0

        quality = alpha_quality(rgba)

        self.assertEqual(quality["verdict"], "reject")
        self.assertTrue(quality["opposite_edge_attachment"])

    def test_alpha_quality_allows_long_object_with_clear_crop_margin(self) -> None:
        rgba = np.zeros((120, 480, 4), np.uint8)
        rgba[..., :3] = [210, 190, 150]
        rgba[40:70, 12:468, 3] = 255
        rgba[68:112, 100:130, 3] = 255
        rgba[68:112, 330:360, 3] = 255

        quality = alpha_quality(rgba)

        self.assertNotEqual(quality["verdict"], "reject")
        self.assertFalse(quality["opposite_edge_attachment"])

    def test_alpha_quality_rejects_two_large_distant_products(self) -> None:
        rgba = np.zeros((300, 520, 4), np.uint8)
        rgba[..., :3] = [145, 112, 82]
        alpha = np.zeros((300, 520), np.uint8)
        cv2.circle(alpha, (155, 145), 100, 255, thickness=cv2.FILLED)
        cv2.rectangle(
            alpha, (360, 52), (490, 268), 255,
            thickness=cv2.FILLED,
        )
        rgba[..., 3] = alpha

        quality = alpha_quality(rgba)

        self.assertEqual(quality["verdict"], "reject")
        self.assertTrue(quality["multi_object_leakage"])
        self.assertGreaterEqual(quality["second_component_share"], 0.15)

    def test_alpha_quality_allows_nearby_detached_small_detail(self) -> None:
        rgba = np.zeros((240, 320, 4), np.uint8)
        rgba[..., :3] = [185, 165, 120]
        alpha = np.zeros((240, 320), np.uint8)
        cv2.rectangle(
            alpha, (80, 40), (220, 220), 255,
            thickness=cv2.FILLED,
        )
        cv2.rectangle(
            alpha, (120, 18), (180, 34), 255,
            thickness=cv2.FILLED,
        )
        rgba[..., 3] = alpha

        quality = alpha_quality(rgba)

        self.assertFalse(quality["multi_object_leakage"])

    def test_ensemble_prefers_plausible_superset_over_clean_fragment(self) -> None:
        fragment = np.zeros((120, 480, 4), np.uint8)
        complete = np.zeros_like(fragment)
        fragment[..., :3] = complete[..., :3] = [190, 170, 120]
        fragment[35:62, 190:465, 3] = 255
        fragment[60:110, 250:360, 3] = 255
        complete[35:62, 15:465, 3] = 255
        complete[60:110, 250:360, 3] = 255

        selected, details = _select_rembg_candidate({
            "fragment_model": fragment,
            "complete_model": complete,
        })

        self.assertEqual(selected, "complete_model")
        self.assertLess(details["quality"]["score"], 100.0)
        self.assertEqual(
            details["quality"]["score_kind"],
            "automatic_confidence_not_ground_truth",
        )

    def test_promptable_witness_recovers_appendage_despite_small_net_area(self) -> None:
        rgb = np.full((180, 220, 3), 175, np.uint8)
        base = np.zeros((180, 220), np.uint8)
        base[30:130, 30:130] = 255
        base[60:70, 130:150] = 255
        promptable = base.copy()
        # Recover 350 appendage pixels but disagree on 300 boundary pixels.
        # Net growth is below the old 2% threshold even though the recovered
        # structure is material and almost all of the conventional mask is
        # preserved.
        promptable[60:70, 150:185] = 255
        promptable[127:130, 30:130] = 0

        selected, details = _select_rembg_candidate({
            "u2net": np.dstack((rgb, base)),
            "sam2-box-recovery": np.dstack((rgb, promptable)),
        })

        self.assertEqual(selected, "sam2-box-recovery")
        self.assertIn(
            "promptable_structural_recovery",
            details["selection_reason"],
        )
