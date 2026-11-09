from __future__ import annotations

import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

import strict_compositor.placement_solver as placement_solver_module
from strict_compositor.image_io import save_png
from strict_compositor.ui import (
    LayoutUI,
    ProductUI,
    _choose_advice,
    _consensus_incompatible_viewpoints,
    _layout_surface_diagnostics,
    _prefer_replanned_layout,
    auto_layout,
)
from strict_compositor.minimax_advisor import MiniMaxAdvice, ProductAdvice
from strict_compositor.placement_solver import (
    _refine_background_readability,
    _restore_triangle_after_readability,
    background_readability_cost_map,
    legalize_placements,
    product_geometry,
    solve_aesthetic_layout,
)


ROOT = Path(__file__).resolve().parents[1]
TEST_TMP = ROOT / "ui_workspace" / "test_tmp"
SUPPORT = [[0.05, 0.15], [0.95, 0.15], [0.95, 0.90], [0.05, 0.90]]


class LayoutCandidateSelectionTests(unittest.TestCase):
    def test_pass_replan_beats_higher_scoring_rejected_seed(self) -> None:
        seed = {"layout_verdict": "reject", "layout_score": 96.0}
        local = {"layout_verdict": "pass", "layout_score": 90.0}
        self.assertTrue(_prefer_replanned_layout(seed, local))

    def test_rejected_replan_never_replaces_passing_seed(self) -> None:
        seed = {"layout_verdict": "pass", "layout_score": 90.0}
        local = {"layout_verdict": "reject", "layout_score": 99.0}
        self.assertFalse(_prefer_replanned_layout(seed, local))

    def test_score_breaks_tie_within_same_verdict_class(self) -> None:
        seed = {"layout_verdict": "pass", "layout_score": 91.0}
        local = {"layout_verdict": "pass", "layout_score": 93.0}
        self.assertTrue(_prefer_replanned_layout(seed, local))

    def test_compact_replan_beats_higher_scoring_sparse_seed(self) -> None:
        seed = {
            "layout_verdict": "pass",
            "layout_score": 97.0,
            "maximum_normalized_gap": 0.91,
        }
        local = {
            "layout_verdict": "pass",
            "layout_score": 92.0,
            "maximum_normalized_gap": 0.48,
        }
        self.assertTrue(_prefer_replanned_layout(seed, local))

    def test_sparse_replan_never_replaces_compact_passing_seed(self) -> None:
        seed = {
            "layout_verdict": "pass",
            "layout_score": 92.0,
            "maximum_normalized_gap": 0.48,
        }
        local = {
            "layout_verdict": "pass",
            "layout_score": 99.0,
            "maximum_normalized_gap": 0.91,
        }
        self.assertFalse(_prefer_replanned_layout(seed, local))

    def test_small_score_gain_cannot_buy_semantic_scale_damage(self) -> None:
        seed = {
            "layout_verdict": "reject",
            "layout_score": 84.0,
            "maximum_normalized_gap": 0.50,
            "semantic_scale_score": 100.0,
        }
        local = {
            "layout_verdict": "reject",
            "layout_score": 88.0,
            "maximum_normalized_gap": 0.62,
            "semantic_scale_score": 86.0,
        }
        self.assertFalse(_prefer_replanned_layout(seed, local))


def _advice(product: ProductAdvice, confidence: float) -> MiniMaxAdvice:
    return MiniMaxAdvice(
        support_polygon=SUPPORT,
        protected_polygons=[],
        protected_labels=[],
        shadow_direction=[0.6, 0.8],
        light_direction=[-0.6, -0.8],
        reflectivity=0.2,
        reflection_recommended=False,
        route="strict_composite",
        confidence=confidence,
        products=[product],
        warnings=[],
        model="MiniMax-M3",
    )


def _rgba() -> np.ndarray:
    image = np.zeros((80, 50, 4), np.uint8)
    image[3:78, 3:47, :3] = [170, 90, 45]
    image[3:78, 3:47, 3] = 255
    return image


def _placement(path: Path, product_id: str, x: float, y: float) -> dict[str, object]:
    return {
        "id": product_id,
        "source": str(path),
        "rgba": str(path),
        "center_x": x,
        "baseline_y": y,
        "height_ratio": 0.16,
        "z_order": 1,
        "placement_mode": "upright",
        "physical_type": "rigid_object",
    }


class LayoutSurfaceDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        TEST_TMP.mkdir(parents=True, exist_ok=True)

    def test_retry_can_correct_one_products_pose_without_replacing_scene(self) -> None:
        first = ProductAdvice(
            product_id="dish",
            center_x=0.4,
            bottom_y=0.6,
            height_ratio=0.1,
            z_order=2,
            footprint_width=0.18,
            source_light_compatibility=0.6,
            placement_mode="upright",
            physical_type="decorative_object",
            semantic_category="sculptural_bowl",
            photographed_pose="display_stand",
            pose_confidence=0.65,
            size_confidence=0.75,
            viewpoint_compatibility=0.75,
        )
        retry = replace(
            first,
            placement_mode="surface_flat",
            photographed_pose="lying_flat",
            pose_confidence=0.88,
        )
        chosen = _choose_advice([
            _advice(first, 0.72),
            _advice(retry, 0.62),
        ])
        self.assertEqual(chosen.confidence, 0.72)
        self.assertEqual(chosen.products[0].placement_mode, "surface_flat")
        self.assertEqual(chosen.products[0].photographed_pose, "lying_flat")
        self.assertEqual(chosen.products[0].pose_confidence, 0.88)

    def test_viewpoint_disagreement_defers_to_local_geometry(self) -> None:
        first = ProductAdvice(
            product_id="fish",
            center_x=0.4,
            bottom_y=0.6,
            height_ratio=0.1,
            z_order=1,
            footprint_width=0.18,
            source_light_compatibility=0.6,
            placement_mode="surface_flat",
            physical_type="rigid_sculpture",
            semantic_category="ceramic_fish_figurine",
            photographed_pose="lying_flat",
            pose_confidence=0.85,
            viewpoint_compatibility=0.40,
        )
        retry = replace(first, viewpoint_compatibility=0.55)
        attempts = [_advice(first, 0.55), _advice(retry, 0.50)]
        chosen = _choose_advice(attempts)
        self.assertEqual(
            _consensus_incompatible_viewpoints(attempts, chosen), [],
        )

    def test_viewpoint_consensus_still_fails_closed(self) -> None:
        first = ProductAdvice(
            product_id="unsafe",
            center_x=0.4,
            bottom_y=0.6,
            height_ratio=0.1,
            z_order=1,
            footprint_width=0.18,
            source_light_compatibility=0.6,
            viewpoint_compatibility=0.30,
        )
        retry = replace(first, viewpoint_compatibility=0.42)
        attempts = [_advice(first, 0.55), _advice(retry, 0.52)]
        chosen = _choose_advice(attempts)
        self.assertEqual(
            _consensus_incompatible_viewpoints(attempts, chosen), ["unsafe"],
        )

    def test_sloped_front_row_is_rejected_in_surface_coordinates(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as temporary:
            rgba_path = Path(temporary) / "product.png"
            save_png(rgba_path, _rgba())
            products = [
                _placement(rgba_path, f"p{index}", x, y)
                for index, (x, y) in enumerate(
                    ((0.25, 0.86), (0.50, 0.86), (0.75, 0.86)), 1
                )
            ]
            result = _layout_surface_diagnostics(
                (600, 400), SUPPORT, [], products,
            )
            self.assertEqual(result["layout_verdict"], "reject")
            self.assertLess(result["layout_score"], 90.0)
            self.assertIn("too_close_to_front_edge", result["layout_reasons"])
            self.assertGreater(result["surface_depth_median"], 0.90)

    def test_interior_triangle_passes_surface_gate(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as temporary:
            rgba_path = Path(temporary) / "product.png"
            save_png(rgba_path, _rgba())
            products = [
                _placement(rgba_path, f"p{index}", x, y)
                for index, (x, y) in enumerate(
                    ((0.30, 0.48), (0.50, 0.57), (0.70, 0.44)), 1
                )
            ]
            result = _layout_surface_diagnostics(
                (600, 400), SUPPORT, [], products,
            )
            self.assertEqual(result["layout_verdict"], "pass")
            self.assertGreaterEqual(result["layout_score"], 90.0)
            self.assertLess(result["surface_depth_max"], 0.72)
            self.assertGreater(result["surface_depth_range"], 0.08)

    def test_adjacent_width_normalized_gap_distinguishes_styled_and_sparse_groups(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as temporary:
            rgba_path = Path(temporary) / "product.png"
            save_png(rgba_path, _rgba())

            def group(centers: tuple[float, float, float]) -> list[dict[str, object]]:
                result = [
                    _placement(rgba_path, f"p{index}", x, y)
                    for index, (x, y) in enumerate(
                        zip(centers, (0.48, 0.57, 0.44)), 1,
                    )
                ]
                for item in result:
                    item["height_ratio"] = 0.24
                return result

            styled = _layout_surface_diagnostics(
                (600, 400), SUPPORT, [], group((0.364, 0.500, 0.636)),
            )
            sparse = _layout_surface_diagnostics(
                (600, 400), SUPPORT, [], group((0.250, 0.500, 0.750)),
            )

            self.assertGreaterEqual(styled["median_normalized_gap"], 0.35)
            self.assertLessEqual(styled["maximum_normalized_gap"], 0.55)
            self.assertNotIn("object_gaps_too_sparse", styled["layout_reasons"])
            self.assertGreater(sparse["maximum_normalized_gap"], 0.68)
            self.assertIn("object_gaps_too_sparse", sparse["layout_reasons"])
            self.assertLess(
                sparse["composition_score"], styled["composition_score"],
            )

    def test_equal_three_column_catalogue_row_is_not_aesthetic_pass(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as temporary:
            rgba_path = Path(temporary) / "product.png"
            save_png(rgba_path, _rgba())
            products = [
                _placement(rgba_path, f"p{index}", x, 0.54)
                for index, x in enumerate((0.28, 0.50, 0.72), 1)
            ]
            result = _layout_surface_diagnostics(
                (600, 400), SUPPORT, [], products,
            )
            self.assertEqual(result["layout_verdict"], "reject")
            self.assertLess(result["composition_score"], 90.0)
            self.assertIn("flat_catalogue_row", result["layout_reasons"])

    def test_receiver_depth_without_visible_base_stagger_is_rejected(self) -> None:
        shallow_support = [
            [0.16, 0.61], [0.84, 0.61], [0.82, 0.82], [0.18, 0.82],
        ]
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as temporary:
            rgba_path = Path(temporary) / "product.png"
            save_png(rgba_path, _rgba())
            # These bases differ enough in receiver-normalised coordinates to
            # evade the old flat-row threshold, but the projected separation
            # is under one fifth of the displayed object height and still
            # reads as a ruler line.
            products = [
                _placement(rgba_path, f"p{index}", x, y)
                for index, (x, y) in enumerate(
                    ((0.30, 0.69), (0.50, 0.71), (0.70, 0.72)), 1
                )
            ]
            result = _layout_surface_diagnostics(
                (600, 400), shallow_support, [], products,
            )
            self.assertEqual(result["layout_verdict"], "reject")
            self.assertGreater(result["surface_depth_range"], 0.08)
            self.assertLess(result["visible_staging_ratio"], 0.20)
            self.assertIn("visually_flat_staging", result["layout_reasons"])

    def test_joint_solver_projects_visible_triangle_on_shallow_receiver(self) -> None:
        shallow_support = [
            [0.28, 0.61], [0.72, 0.61], [0.71, 0.82], [0.29, 0.82],
        ]
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as temporary:
            rgba_path = Path(temporary) / "product.png"
            save_png(rgba_path, _rgba())
            products = []
            for index, (height_cm, salience) in enumerate(
                ((25.0, 0.72), (14.5, 0.38), (22.0, 0.78)), 1,
            ):
                item = _placement(rgba_path, f"p{index}", 0.5, 0.70)
                item.update({
                    "estimated_real_height_cm": height_cm,
                    "size_confidence": 0.88,
                    "receiver_width_or_diameter_cm": 55.0,
                    "receiver_size_confidence": 0.72,
                    "visual_salience": salience,
                })
                products.append(item)

            result, metrics = solve_aesthetic_layout(
                (900, 600), shallow_support, [], products,
                shadow_length_ratio=0.0,
            )

            baselines = np.asarray(
                [float(item["baseline_y"]) for item in result], np.float64,
            )
            heights = np.asarray(
                [float(item["height_ratio"]) for item in result], np.float64,
            )
            self.assertGreaterEqual(
                float(np.ptp(baselines) / np.median(heights)), 0.20,
            )
            self.assertLessEqual(
                float(np.ptp(baselines) / np.median(heights)), 0.38,
            )
            self.assertGreaterEqual(metrics["surface_depth_range"], 0.12)
            # A 0.22-height projected triangle on this deliberately shallow
            # receiver maps to about 0.34 receiver-local depth.  Keep the
            # solver's independent anti-scatter ceiling (0.36) authoritative
            # instead of forcing a UI-passable visual triangle back below it.
            self.assertLessEqual(metrics["surface_depth_range"], 0.36)
            self.assertGreaterEqual(
                # Same minimum enforced by both triangle-recovery paths.
                metrics["front_clearance_min_heights"], 0.30,
            )

    def test_perspective_staging_does_not_overinflate_small_foreground_bridge(self) -> None:
        shallow_support = [
            [0.28, 0.61], [0.72, 0.61], [0.71, 0.82], [0.29, 0.82],
        ]
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as temporary:
            rgba_path = Path(temporary) / "product.png"
            save_png(rgba_path, _rgba())
            products = []
            semantic_heights = (25.0, 14.5, 22.0)
            for index, (height_cm, salience) in enumerate(
                zip(semantic_heights, (0.72, 0.38, 0.78)), 1,
            ):
                item = _placement(rgba_path, f"p{index}", 0.5, 0.70)
                item.update({
                    "estimated_real_height_cm": height_cm,
                    "size_confidence": 0.88,
                    "receiver_width_or_diameter_cm": 55.0,
                    "receiver_size_confidence": 0.72,
                    "visual_salience": salience,
                })
                products.append(item)

            result, _ = solve_aesthetic_layout(
                (900, 600), shallow_support, [], products,
                shadow_length_ratio=0.0,
            )

            heights = [float(item["height_ratio"]) for item in result]
            # Camera depth may gently modulate apparent scale, but it must not
            # erase more than five percent of a high-confidence physical-size
            # relationship merely to dramatise the staging triangle.
            self.assertGreaterEqual(
                heights[0] / heights[1],
                (semantic_heights[0] / semantic_heights[1]) * 0.95,
            )
            self.assertGreaterEqual(
                heights[2] / heights[1],
                (semantic_heights[2] / semantic_heights[1]) * 0.95,
            )

    def test_confident_real_dimensions_reject_tiny_sticker_scale(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as temporary:
            rgba_path = Path(temporary) / "product.png"
            save_png(rgba_path, _rgba())
            products = []
            for index, (x, y, real_height) in enumerate(
                ((0.30, 0.48, 25.0), (0.50, 0.57, 16.0), (0.70, 0.44, 24.0)), 1
            ):
                item = _placement(rgba_path, f"p{index}", x, y)
                item.update({
                    "height_ratio": 0.09,
                    "estimated_real_height_cm": real_height,
                    "size_confidence": 0.88,
                    "receiver_width_or_diameter_cm": 72.0,
                    "receiver_size_confidence": 0.90,
                })
                products.append(item)
            result = _layout_surface_diagnostics(
                (600, 400), SUPPORT, [], products,
            )
            self.assertEqual(result["layout_verdict"], "reject")
            self.assertLess(result["semantic_scale_score"], 90.0)
            self.assertIn("semantic_scale_too_small", result["layout_reasons"])

    def test_auto_layout_replans_rejected_minimax_seed(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as temporary:
            folder = Path(temporary)
            background = folder / "background.png"
            rgba_path = folder / "product.png"
            support_mask_path = folder / "support_mask.png"
            save_png(background, np.full((400, 600, 3), 180, np.uint8))
            save_png(rgba_path, _rgba())
            exact_support = np.zeros((400, 600), np.uint8)
            exact_support[100:369, 25:576] = 255
            save_png(support_mask_path, exact_support)
            source_products = [
                ProductUI(
                    **_placement(rgba_path, f"p{index}", x, 0.65)
                )
                for index, x in enumerate((0.30, 0.50, 0.70), 1)
            ]
            bad_seed = [
                _placement(rgba_path, f"p{index}", x, 0.86)
                for index, x in enumerate((0.30, 0.50, 0.70), 1)
            ]
            good_local = [
                _placement(rgba_path, f"p{index}", x, y)
                for index, (x, y) in enumerate(
                    ((0.30, 0.48), (0.50, 0.57), (0.70, 0.44)), 1
                )
            ]
            advice = MiniMaxAdvice(
                support_polygon=SUPPORT,
                protected_polygons=[],
                protected_labels=[],
                shadow_direction=[0.6, 0.8],
                light_direction=[-0.6, -0.8],
                reflectivity=0.2,
                reflection_recommended=False,
                route="strict_composite",
                confidence=0.65,
                products=[
                    ProductAdvice(
                        product_id=f"p{index}",
                        center_x=x,
                        bottom_y=0.86,
                        height_ratio=0.16,
                        z_order=index,
                        footprint_width=0.12,
                        source_light_compatibility=0.8,
                    )
                    for index, x in enumerate((0.30, 0.50, 0.70), 1)
                ],
                warnings=[],
                model="MiniMax-M3",
                request_id="layout-test",
            )
            payload = LayoutUI(
                background=str(background),
                products=source_products,
                support_polygon=SUPPORT,
                support_mask_path=str(support_mask_path),
            )
            with (
                patch("strict_compositor.ui.minimax_status", return_value={"configured": True}),
                patch("strict_compositor.ui.request_minimax_advice", return_value=advice),
                patch("strict_compositor.ui.apply_layout_advice", return_value=bad_seed),
                patch(
                    "strict_compositor.ui.refine_seeded_composition",
                    return_value=(bad_seed, ["repair 1", "repair 2"]),
                ) as seed_refiner,
                patch(
                    "strict_compositor.ui.solve_aesthetic_layout",
                    return_value=(good_local, {
                        "layout_score": 96.0,
                        "usable_surface_ratio": 0.6,
                        "automatic_repairs": 0.0,
                    }),
                ) as local_solver,
            ):
                result = auto_layout(payload)

            local_solver.assert_called_once()
            self.assertEqual(
                seed_refiner.call_args.kwargs["background_rgb"].shape,
                (400, 600, 3),
            )
            self.assertEqual(
                local_solver.call_args.kwargs["background_rgb"].shape,
                (400, 600, 3),
            )
            self.assertTrue(
                np.array_equal(
                    local_solver.call_args.kwargs["support_mask"],
                    exact_support > 0,
                )
            )
            self.assertTrue(
                np.array_equal(
                    seed_refiner.call_args.kwargs["support_mask"],
                    exact_support > 0,
                )
            )
            self.assertEqual(result["layout_verdict"], "pass")
            self.assertGreaterEqual(result["metrics"]["layout_score"], 90.0)
            self.assertTrue(result["metrics"]["local_joint_replan"])
            self.assertLess(result["metrics"]["advisor_seed_score"], 90.0)
            self.assertEqual(result["advisor"]["status"], "seed_replanned_local")
            self.assertEqual(result["products"], good_local)

    def test_auto_layout_replans_passing_but_visibly_sparse_minimax_seed(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as temporary:
            folder = Path(temporary)
            background = folder / "background.png"
            rgba_path = folder / "product.png"
            save_png(background, np.full((400, 600, 3), 180, np.uint8))
            save_png(rgba_path, _rgba())

            source_products: list[ProductUI] = []
            sparse_seed: list[dict[str, object]] = []
            compact_local: list[dict[str, object]] = []
            sparse_centers = (0.250, 0.500, 0.750)
            compact_centers = (0.364, 0.500, 0.636)
            baselines = (0.48, 0.57, 0.44)
            for index, (sparse_x, compact_x, baseline) in enumerate(
                zip(sparse_centers, compact_centers, baselines), 1,
            ):
                source = _placement(
                    rgba_path, f"p{index}", sparse_x, baseline,
                )
                source["height_ratio"] = 0.24
                source_products.append(ProductUI(**source))
                sparse_seed.append(dict(source))
                compact = dict(source)
                compact["center_x"] = compact_x
                compact_local.append(compact)

            advice = MiniMaxAdvice(
                support_polygon=SUPPORT,
                protected_polygons=[],
                protected_labels=[],
                shadow_direction=[0.6, 0.8],
                light_direction=[-0.6, -0.8],
                reflectivity=0.2,
                reflection_recommended=False,
                route="strict_composite",
                confidence=0.82,
                products=[
                    ProductAdvice(
                        product_id=f"p{index}",
                        center_x=x,
                        bottom_y=baseline,
                        height_ratio=0.24,
                        z_order=index,
                        footprint_width=0.12,
                        source_light_compatibility=0.8,
                        estimated_real_height_cm=20.0,
                        size_confidence=0.80,
                        viewpoint_compatibility=0.80,
                    )
                    for index, (x, baseline) in enumerate(
                        zip(sparse_centers, baselines), 1,
                    )
                ],
                warnings=[],
                model="MiniMax-M3",
                request_id="sparse-seed-test",
            )
            payload = LayoutUI(
                background=str(background),
                products=source_products,
                support_polygon=SUPPORT,
            )
            with (
                patch(
                    "strict_compositor.ui.minimax_status",
                    return_value={"configured": True},
                ),
                patch(
                    "strict_compositor.ui.request_minimax_advice",
                    return_value=advice,
                ),
                patch(
                    "strict_compositor.ui.apply_layout_advice",
                    return_value=sparse_seed,
                ),
                patch(
                    "strict_compositor.ui.refine_seeded_composition",
                    return_value=(sparse_seed, []),
                ),
                patch(
                    "strict_compositor.ui.solve_aesthetic_layout",
                    return_value=(compact_local, {
                        "layout_score": 91.0,
                        "usable_surface_ratio": 0.6,
                        "automatic_repairs": 0.0,
                    }),
                ) as local_solver,
            ):
                result = auto_layout(payload)

            local_solver.assert_not_called()
            self.assertEqual(result["advisor"]["status"], "seed_compacted")
            self.assertEqual(result["advisor"]["seed_layout_verdict"], "pass")
            self.assertGreater(
                result["advisor"]["seed_maximum_normalized_gap"], 0.68,
            )
            self.assertLessEqual(
                result["metrics"]["maximum_normalized_gap"], 0.55,
            )
            self.assertTrue(result["metrics"]["semantic_seed_compacted"])
            self.assertEqual(
                [item["height_ratio"] for item in result["products"]],
                [item["height_ratio"] for item in sparse_seed],
            )

    def test_joint_solver_uses_product_scaled_negative_space_on_large_support(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as temporary:
            folder = Path(temporary)
            rgba_path = folder / "product.png"
            save_png(rgba_path, _rgba())
            products = []
            for index, (height_cm, salience) in enumerate(
                ((25.0, 0.72), (14.5, 0.45), (22.0, 0.78)), 1,
            ):
                item = _placement(rgba_path, f"p{index}", 0.5, 0.55)
                item.update({
                    "estimated_real_height_cm": height_cm,
                    "size_confidence": 0.82,
                    "receiver_width_or_diameter_cm": 70.0,
                    "receiver_size_confidence": 0.70,
                    "visual_salience": salience,
                })
                products.append(item)

            result, metrics = solve_aesthetic_layout(
                (900, 600), SUPPORT, [], products,
                shadow_direction=(0.7, 0.6),
                shadow_length_ratio=0.0,
            )

            centers = sorted(float(item["center_x"]) for item in result)
            self.assertGreaterEqual(centers[-1] - centers[0], 0.16)
            self.assertLessEqual(centers[-1] - centers[0], 0.38)
            self.assertGreaterEqual(metrics["entity_gap_to_median_width"], 0.18)
            self.assertLessEqual(metrics["entity_gap_to_median_width"], 0.62)
            self.assertGreater(metrics["layout_score"], 90.0)

    def test_readability_map_marks_dark_highlight_and_texture_not_quiet_surface(self) -> None:
        background = np.full((160, 240, 3), 180, np.uint8)
        background[30:70, 20:55] = 10
        background[30:70, 80:115] = 255
        yy, xx = np.indices((40, 40))
        checker = np.where(((xx // 2 + yy // 2) % 2)[..., None], 70, 230)
        background[30:70, 140:180] = checker
        support = np.ones((160, 240), bool)

        cost = background_readability_cost_map(
            background, (160, 240), support,
        )

        quiet = float(cost[100:130, 20:55].mean())
        self.assertGreater(float(cost[30:70, 20:55].mean()), quiet + 0.25)
        self.assertGreater(float(cost[30:70, 80:115].mean()), quiet + 0.20)
        self.assertGreater(float(cost[30:70, 140:180].mean()), quiet + 0.25)

    def test_clear_glass_is_moved_inward_when_table_edge_splits_its_body(self) -> None:
        """A quiet scene still needs a geometric receiver-boundary guard."""
        rear_edge_support = [
            [0.05, 0.30], [0.95, 0.30], [0.95, 0.95], [0.05, 0.95],
        ]
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as temporary:
            folder = Path(temporary)
            rgba_path = folder / "glass.png"
            save_png(rgba_path, _rgba())
            product = _placement(rgba_path, "glass", 0.5, 0.50)
            product.update({
                "height_ratio": 0.34,
                "material_mode": "clear_glass",
                "material_confidence": 0.98,
            })
            quiet_background = np.full((400, 600, 3), 180, np.uint8)

            result, metrics = _refine_background_readability(
                (600, 400),
                rear_edge_support,
                [],
                [product],
                quiet_background,
                (0.7, 0.7),
                0.0,
                0.15,
            )

            self.assertGreater(float(result[0]["baseline_y"]), 0.60)
            self.assertAlmostEqual(float(result[0]["center_x"]), 0.5, places=8)
            self.assertGreater(
                metrics["background_boundary_seam_improvement"], 0.60,
            )
            self.assertLess(
                metrics["background_boundary_seam_cost"], 0.05,
            )
            self.assertGreater(
                metrics["background_readability_depth_shift_max"], 0.10,
            )

    def test_opaque_product_is_not_moved_for_table_edge_behind_body(self) -> None:
        rear_edge_support = [
            [0.05, 0.30], [0.95, 0.30], [0.95, 0.95], [0.05, 0.95],
        ]
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as temporary:
            folder = Path(temporary)
            rgba_path = folder / "opaque.png"
            save_png(rgba_path, _rgba())
            product = _placement(rgba_path, "opaque", 0.5, 0.50)
            product.update({
                "height_ratio": 0.34,
                "material_mode": "opaque",
                "material_confidence": 0.99,
            })
            quiet_background = np.full((400, 600, 3), 180, np.uint8)

            result, metrics = _refine_background_readability(
                (600, 400),
                rear_edge_support,
                [],
                [product],
                quiet_background,
                (0.7, 0.7),
                0.0,
                0.15,
            )

            self.assertEqual(result, [product])
            self.assertEqual(metrics["background_boundary_seam_cost"], 0.0)
            self.assertEqual(
                metrics["background_readability_depth_shift_max"], 0.0,
            )

    def test_auto_layout_moves_clear_glass_off_severe_dark_structure(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as temporary:
            folder = Path(temporary)
            rgba_path = folder / "glass.png"
            save_png(rgba_path, _rgba())
            product = _placement(rgba_path, "glass", 0.5, 0.60)
            product.update({
                "material_mode": "clear_glass",
                "material_confidence": 0.95,
            })
            background = np.full((400, 600, 3), 180, np.uint8)
            # A synthetic table bracket directly behind the nominal centre.
            # An opaque body hides this structure completely; with no clutter
            # in its contact/reflection corridor the layout must stay put.
            background[150:215, 270:330] = 8

            result, metrics = solve_aesthetic_layout(
                (600, 400),
                SUPPORT,
                [],
                [product],
                shadow_length_ratio=0.0,
                background_rgb=background,
            )

            self.assertGreater(
                abs(float(result[0]["center_x"]) - 0.5), 0.06,
            )
            self.assertGreater(
                metrics["background_readability_improvement"], 0.08,
            )
            self.assertLessEqual(
                metrics["background_readability_shift_max"], 0.09,
            )

    def test_opaque_auto_layout_is_not_moved_by_readability_soft_cost(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as temporary:
            folder = Path(temporary)
            rgba_path = folder / "opaque.png"
            save_png(rgba_path, _rgba())
            product = _placement(rgba_path, "opaque", 0.5, 0.60)
            product.update({
                "material_mode": "opaque",
                "material_confidence": 0.98,
            })
            background = np.full((400, 600, 3), 180, np.uint8)
            # The patch is fully hidden by the opaque body and stops well
            # above the contact/reflection corridor.  It must therefore have
            # no influence on the final staging position.
            background[190:215, 270:330] = 8

            without_background, _ = solve_aesthetic_layout(
                (600, 400), SUPPORT, [], [product], shadow_length_ratio=0.0,
            )
            with_background, metrics = solve_aesthetic_layout(
                (600, 400),
                SUPPORT,
                [],
                [product],
                shadow_length_ratio=0.0,
                background_rgb=background,
            )

            self.assertAlmostEqual(
                float(with_background[0]["center_x"]),
                float(without_background[0]["center_x"]),
                places=8,
            )
            self.assertEqual(metrics["background_readability_shift_max"], 0.0)

    def test_opaque_auto_layout_avoids_severe_reflection_corridor_structure(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as temporary:
            folder = Path(temporary)
            rgba_path = folder / "opaque.png"
            save_png(rgba_path, _rgba())
            product = _placement(rgba_path, "opaque", 0.5, 0.60)
            product.update({
                "material_mode": "metal",
                "material_confidence": 0.98,
            })
            background = np.full((400, 600, 3), 180, np.uint8)
            # The object itself covers the receiver above its baseline, but a
            # dark bracket immediately below remains visible through contact
            # and reflection and reads as a false long shadow.
            background[238:286, 270:330] = 8

            without_background, _ = solve_aesthetic_layout(
                (600, 400), SUPPORT, [], [product], shadow_length_ratio=0.0,
            )
            with_background, metrics = solve_aesthetic_layout(
                (600, 400),
                SUPPORT,
                [],
                [product],
                shadow_length_ratio=0.0,
                background_rgb=background,
            )

            self.assertGreater(
                abs(
                    float(with_background[0]["center_x"])
                    - float(without_background[0]["center_x"])
                ),
                0.025,
            )
            self.assertGreater(
                metrics["background_readability_improvement"], 0.025,
            )
            self.assertIn(
                "background_readability_foreground_product_cost", metrics,
            )
            self.assertIn(
                "background_readability_max_product_cost", metrics,
            )

    def test_central_hardware_uses_large_opaque_product_as_visual_bridge(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as temporary:
            folder = Path(temporary)
            rgba_path = folder / "product.png"
            save_png(rgba_path, _rgba())
            specifications = (
                ("clear", "clear_glass", 28.0, 0.78),
                ("small", "metal", 15.0, 0.42),
                ("hero", "mixed_transparent", 27.0, 0.68),
            )
            products = []
            for product_id, material, real_height, salience in specifications:
                item = _placement(rgba_path, product_id, 0.5, 0.60)
                item.update({
                    "material_mode": material,
                    "material_confidence": 0.95,
                    "estimated_real_height_cm": real_height,
                    "size_confidence": 0.92,
                    "receiver_width_or_diameter_cm": 70.0,
                    "receiver_size_confidence": 0.90,
                    "visual_salience": salience,
                })
                products.append(item)
            background = np.full((400, 600, 3), 180, np.uint8)
            # A dark mounting plate occupies the commercial centre slot.
            background[205:310, 250:350] = 8

            result, metrics = solve_aesthetic_layout(
                (600, 400),
                SUPPORT,
                [],
                products,
                shadow_length_ratio=0.0,
                background_rgb=background,
            )

            by_id = {item["id"]: item for item in result}
            self.assertGreater(
                float(by_id["hero"]["center_x"]),
                float(by_id["small"]["center_x"]),
            )
            self.assertLess(
                float(by_id["hero"]["center_x"]),
                float(by_id["clear"]["center_x"]),
            )
            self.assertEqual(metrics["central_occlusion_bridge"], 1.0)
            self.assertGreaterEqual(metrics["central_structure_cost"], 0.34)

    def test_transparent_backdrop_nudge_preserves_three_item_gap_rhythm(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as temporary:
            folder = Path(temporary)
            rgba_path = folder / "product.png"
            save_png(rgba_path, _rgba())
            products = []
            for index, salience in enumerate((0.42, 0.55, 0.82), 1):
                item = _placement(rgba_path, f"p{index}", 0.5, 0.55)
                item.update({
                    "material_mode": "clear_glass" if index == 3 else "opaque",
                    "material_confidence": 0.96,
                    "visual_salience": salience,
                })
                products.append(item)
            background = np.full((400, 600, 3), 180, np.uint8)
            # The visually dominant clear item is staged on the right.  A
            # severe structure behind that slot should trigger a nudge, but
            # never leave one catalogue-sized void in the three-item group.
            background[110:350, 330:480] = 6

            result, metrics = solve_aesthetic_layout(
                (600, 400),
                SUPPORT,
                [],
                products,
                shadow_length_ratio=0.0,
                background_rgb=background,
            )

            geometry = product_geometry(rgba_path)
            display_widths = [
                float(item["height_ratio"])
                * geometry.content_aspect
                * (400.0 / 600.0)
                for item in result
            ]
            intervals = sorted(
                (float(item["center_x"]) - item_width * 0.5,
                 float(item["center_x"]) + item_width * 0.5)
                for item, item_width in zip(result, display_widths)
            )
            gaps = [
                intervals[index + 1][0] - intervals[index][1]
                for index in range(2)
            ]
            ratio = max(gaps) / min(gaps)
            self.assertLessEqual(ratio, 1.90)
            median_width = float(np.median(display_widths))
            self.assertLessEqual(float(np.median(gaps)) / median_width, 0.72)
            self.assertGreater(
                metrics["background_readability_improvement"], 0.04,
            )

    def test_readability_cannot_buy_a_sparse_three_product_group(self) -> None:
        """A cleaner backdrop cannot override commercial group cohesion."""
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as temporary:
            folder = Path(temporary)
            rgba_path = folder / "product.png"
            save_png(rgba_path, _rgba())
            products = []
            for index, x in enumerate((0.42, 0.50, 0.58), 1):
                item = _placement(rgba_path, f"p{index}", x, 0.60)
                item.update({
                    "height_ratio": 0.18,
                    "material_mode": (
                        "clear_glass" if index == 3 else "opaque"
                    ),
                    "material_confidence": 0.98,
                })
                products.append(item)
            background = np.full((400, 600, 3), 180, np.uint8)
            # A severe high-contrast structure invites the clear right item
            # to move by the largest search step.  That alternative is
            # visually cleaner in isolation but would split the product group.
            background[80:360, 330:470] = 4

            result, _ = _refine_background_readability(
                (600, 400), SUPPORT, [], products, background,
                (0.7, 0.7), 0.0, 0.15,
            )

            geometry = product_geometry(rgba_path)
            widths = [
                float(item["height_ratio"])
                * geometry.content_aspect
                * (400.0 / 600.0)
                for item in result
            ]
            intervals = sorted(
                (float(item["center_x"]) - width * 0.5,
                 float(item["center_x"]) + width * 0.5)
                for item, width in zip(result, widths)
            )
            gaps = [
                intervals[index + 1][0] - intervals[index][1]
                for index in range(2)
            ]
            positive = [gap for gap in gaps if gap > 1e-6]
            if positive:
                self.assertLessEqual(
                    float(np.median(positive)) / float(np.median(widths)),
                    0.62,
                )

    def test_transparent_depth_repair_keeps_visible_staging_triangle(self) -> None:
        """A locked clear anchor must not flatten the final three-item group."""
        rear_edge_support = [
            [0.05, 0.50], [0.95, 0.50], [0.95, 0.95], [0.05, 0.95],
        ]
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as temporary:
            folder = Path(temporary)
            rgba_path = folder / "product.png"
            save_png(rgba_path, _rgba())
            products = []
            for index, salience in enumerate((0.42, 0.55, 0.82), 1):
                item = _placement(rgba_path, f"p{index}", 0.5, 0.55)
                item.update({
                    "material_mode": (
                        "clear_glass" if index == 3 else "opaque"
                    ),
                    "material_confidence": 0.96,
                    "visual_salience": salience,
                })
                if index == 1:
                    # This test isolates the clear-anchor depth repair at the
                    # larger presentation scale where that repair is needed.
                    # Mark one opaque item as genuinely label-bearing instead
                    # of relying on the obsolete behaviour that commercially
                    # enlarged every unlabelled three-object group.
                    item.update({
                        "semantic_category": "branded_bottle",
                        "rerender_policy": "never",
                    })
                products.append(item)
            quiet_background = np.full((400, 600, 3), 180, np.uint8)

            without_background, _ = solve_aesthetic_layout(
                (600, 400), rear_edge_support, [], products,
                shadow_length_ratio=0.0,
            )
            result, metrics = solve_aesthetic_layout(
                (600, 400),
                rear_edge_support,
                [],
                products,
                shadow_length_ratio=0.0,
                background_rgb=quiet_background,
            )

            baseline_by_id = {
                item["id"]: float(item["baseline_y"])
                for item in without_background
            }
            by_id = {item["id"]: item for item in result}
            # p3 is the clear right anchor.  Its receiver-local depth repair is
            # retained instead of being undone to recover composition.
            self.assertGreater(
                float(by_id["p3"]["baseline_y"])
                - baseline_by_id["p3"],
                0.05,
            )
            left, bridge, right = by_id["p2"], by_id["p1"], by_id["p3"]
            blend = (
                float(bridge["center_x"]) - float(left["center_x"])
            ) / (
                float(right["center_x"]) - float(left["center_x"])
            )
            anchor_line = (
                float(left["baseline_y"]) * (1.0 - blend)
                + float(right["baseline_y"]) * blend
            )
            median_height = float(np.median([
                float(item["height_ratio"]) for item in result
            ]))
            self.assertGreaterEqual(
                (float(bridge["baseline_y"]) - anchor_line) / median_height,
                0.205,
            )
            self.assertGreaterEqual(
                metrics["front_clearance_min_heights"], 0.30,
            )
            self.assertLess(metrics["background_boundary_seam_cost"], 0.05)
            # Exercise the same diagnostic used by the real UI rather than
            # accepting only the solver's internal recovery threshold.
            ui_diagnostics = _layout_surface_diagnostics(
                (600, 400), rear_edge_support, [], result,
            )
            self.assertGreaterEqual(
                ui_diagnostics["visible_staging_ratio"], 0.20,
            )
            self.assertNotIn(
                "visually_flat_staging",
                ui_diagnostics["layout_reasons"],
            )

    def test_post_readability_triangle_uses_one_full_legalization_at_most(self) -> None:
        """Losing staging trials must not enter the exhaustive repair search."""
        rear_edge_support = [
            [0.05, 0.50], [0.95, 0.50], [0.95, 0.95], [0.05, 0.95],
        ]
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as temporary:
            rgba_path = Path(temporary) / "product.png"
            save_png(rgba_path, _rgba())
            products = []
            for index, (x, y) in enumerate(
                ((0.50, 0.828), (0.32, 0.750), (0.68, 0.841)), 1
            ):
                item = _placement(rgba_path, f"p{index}", x, y)
                item.update({
                    "height_ratio": 0.327,
                    "z_order": index,
                    "material_mode": (
                        "clear_glass" if index == 3 else "opaque"
                    ),
                    "material_confidence": 0.96,
                })
                products.append(item)
            before = [dict(item) for item in products]
            before[2]["baseline_y"] = 0.765

            with patch(
                "strict_compositor.placement_solver.legalize_placements",
                wraps=legalize_placements,
            ) as full_legalize:
                result = _restore_triangle_after_readability(
                    (600, 400),
                    rear_edge_support,
                    [],
                    before,
                    products,
                    [1, 0, 2],
                    (0.75, -0.66),
                    0.0,
                    0.15,
                    None,
                )

            self.assertLessEqual(full_legalize.call_count, 1)
            self.assertAlmostEqual(
                float(result[2]["baseline_y"]), 0.841, places=8,
            )

    def test_readability_search_meets_production_preview_budget(self) -> None:
        """A 7K canvas must stay bounded by the 900px planning preview."""
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as temporary:
            folder = Path(temporary)
            paths: list[Path] = []
            for index in range(3):
                path = folder / f"product_{index}.png"
                save_png(path, _rgba())
                paths.append(path)
            products = []
            for index, (path, x, material) in enumerate(zip(
                paths,
                (0.34, 0.50, 0.66),
                ("clear_glass", "metal", "mixed_transparent"),
            ), 1):
                item = _placement(path, f"p{index}", x, 0.68)
                item.update({
                    "height_ratio": 0.18,
                    "z_order": index,
                    "material_mode": material,
                    "material_confidence": 0.96,
                    "visual_salience": 0.45 + index * 0.12,
                })
                products.append(item)

            # A public, synthetic receiver with several costly structures.  Its
            # pixel dimensions are deliberately smaller than 7K; canvas_size
            # controls the same 900x600 production planning preview while the
            # resize keeps the test fixture lightweight.
            background = np.full((800, 1200, 3), 178, np.uint8)
            background[250:620, 285:390] = 8
            background[300:690, 540:655] = 252
            yy, xx = np.indices((360, 130))
            texture = np.where(((xx // 3 + yy // 3) % 2)[..., None], 45, 225)
            background[270:630, 770:900] = texture

            original_load = placement_solver_module.load_rgba
            original_warp = placement_solver_module._placement_mask
            started = time.perf_counter()
            with (
                patch.object(
                    placement_solver_module, "load_rgba", wraps=original_load,
                ) as rgba_load,
                patch.object(
                    placement_solver_module, "_placement_mask", wraps=original_warp,
                ) as placement_warp,
            ):
                _, metrics = _refine_background_readability(
                    (7008, 4672),
                    SUPPORT,
                    [],
                    products,
                    background,
                    (0.75, -0.66),
                    0.0,
                    0.15,
                )
            elapsed = time.perf_counter() - started

            self.assertLess(elapsed, 15.0)
            self.assertLessEqual(
                metrics["background_readability_trials"], 72.0,
            )
            self.assertGreater(metrics["background_readability_candidates"], 0.0)
            # Three geometry reads plus at most one shared readability read per
            # source.  The old implementation decoded hundreds of PNGs.
            self.assertLessEqual(rgba_load.call_count, 6)
            # Exact validation and scoring share the candidate warp.  The old
            # repair search exceeded 13,000 warps for this class of request.
            self.assertLessEqual(placement_warp.call_count, 220)

    def test_ambient_only_shallow_receiver_requires_only_the_base(self) -> None:
        """No cast-shadow corridor may turn an upright body into a footprint."""
        shallow_support = [
            [0.08, 0.46], [0.92, 0.46], [0.92, 0.52], [0.08, 0.52],
        ]
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as temporary:
            path = Path(temporary) / "upright.png"
            save_png(path, _rgba())
            product = _placement(path, "upright", 0.50, 0.50)
            product["height_ratio"] = 0.28

            result, messages = legalize_placements(
                (900, 600),
                shallow_support,
                [],
                [product],
                automatic_edge_margin_ratio=0.012,
                shadow_direction=None,
                shadow_length_ratio=0.0,
                repair_invalid=False,
            )

            self.assertEqual(messages, [])
            self.assertAlmostEqual(result[0]["center_x"], 0.50, places=8)
            self.assertAlmostEqual(result[0]["baseline_y"], 0.50, places=8)
            self.assertAlmostEqual(result[0]["height_ratio"], 0.28, places=8)

    def test_ultra_shallow_receiver_keeps_visible_three_item_staging(self) -> None:
        """A front-facing top keeps a quantisation-tolerant triangle."""
        shallow_support = [
            [0.08, 0.46], [0.92, 0.46], [0.92, 0.52], [0.08, 0.52],
        ]
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as temporary:
            path = Path(temporary) / "wide_product.png"
            rgba = np.zeros((120, 100, 4), np.uint8)
            rgba[2:118, 2:98, :3] = [170, 90, 45]
            rgba[2:118, 2:98, 3] = 255
            save_png(path, rgba)
            products = []
            for index, salience in enumerate((0.40, 0.70, 0.55), 1):
                item = _placement(path, f"p{index}", 0.50, 0.49)
                item.update({
                    "visual_salience": salience,
                    "material_mode": "opaque",
                })
                products.append(item)

            result, metrics = solve_aesthetic_layout(
                (900, 600),
                shallow_support,
                [],
                products,
                shadow_direction=None,
                shadow_length_ratio=0.0,
                background_rgb=np.full((600, 900, 3), 180, np.uint8),
            )
            diagnostic = _layout_surface_diagnostics(
                (900, 600), shallow_support, [], result,
            )

            self.assertEqual(diagnostic["layout_verdict"], "pass")
            self.assertLess(
                diagnostic["receiver_span_over_height_median"], 0.45,
            )
            self.assertGreaterEqual(
                diagnostic["visible_staging_ratio"] + 0.010,
                diagnostic["required_visible_staging_ratio"],
            )
            self.assertGreaterEqual(
                diagnostic["minimum_front_clearance_over_height"],
                diagnostic["required_front_clearance_over_height"],
            )
            self.assertEqual(metrics["automatic_repairs"], 0.0)

    def test_uncertain_real_sizes_do_not_forge_perspective_inversion(self) -> None:
        """A03-like overlapping size intervals keep a visible shallow triangle."""
        a03_support = [
            [0.1215, 0.4517], [0.8756, 0.4678], [0.9421, 0.5032],
            [0.9407, 0.5225], [0.2545, 0.5118], [0.2452, 0.4785],
            [0.1065, 0.4689],
        ]
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as temporary:
            paths = []
            for index, (height, width) in enumerate(
                ((120, 85), (120, 78), (120, 105)), 1,
            ):
                path = Path(temporary) / f"product_{index}.png"
                rgba = np.zeros((height, width, 4), np.uint8)
                rgba[2:-2, 2:-2, :3] = [170, 90, 45]
                rgba[2:-2, 2:-2, 3] = 255
                save_png(path, rgba)
                paths.append(path)

            specifications = (
                # right diffuser, left mug-with-fruit, centre-front figurine
                (0.6463188, 0.4816667, 0.1940738, 27.0, 20.0, 34.0, 0.644),
                (0.3883521, 0.4816667, 0.1354481, 10.5, 8.0, 15.0, 0.595),
                (0.5091153, 0.5100000, 0.1945572, 21.0, 12.0, 35.0, 0.644),
            )
            products = []
            for index, (path, spec) in enumerate(zip(paths, specifications), 1):
                x, y, ratio, height_cm, low_cm, high_cm, confidence = spec
                item = _placement(path, f"p{index}", x, y)
                item.update({
                    "height_ratio": ratio,
                    "estimated_real_height_cm": height_cm,
                    "estimated_height_cm_min": low_cm,
                    "estimated_height_cm_max": high_cm,
                    "size_confidence": confidence,
                })
                products.append(item)

            result = _layout_surface_diagnostics(
                (7008, 4672), a03_support, [], products,
            )

            self.assertEqual(result["perspective_scale_violations"], 0)
            self.assertNotIn(
                "perspective_scale_inversion", result["layout_reasons"],
            )
            self.assertGreaterEqual(
                result["visible_staging_ratio"],
                result["visible_staging_gate_ratio"],
            )
            self.assertNotIn(
                "visually_flat_staging", result["layout_reasons"],
            )
            self.assertEqual(result["layout_verdict"], "pass")

    def test_narrow_size_intervals_still_reject_proven_inversion(self) -> None:
        """Uncertainty-aware comparison remains fail-closed when ranges agree."""
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as temporary:
            path = Path(temporary) / "product.png"
            save_png(path, _rgba())
            far = _placement(path, "far", 0.36, 0.34)
            far.update({
                "height_ratio": 0.24,
                "estimated_real_height_cm": 20.0,
                "estimated_height_cm_min": 19.0,
                "estimated_height_cm_max": 21.0,
                "size_confidence": 0.90,
            })
            near = _placement(path, "near", 0.64, 0.74)
            near.update({
                "height_ratio": 0.12,
                "estimated_real_height_cm": 20.0,
                "estimated_height_cm_min": 19.0,
                "estimated_height_cm_max": 21.0,
                "size_confidence": 0.90,
            })

            result = _layout_surface_diagnostics(
                (600, 400), SUPPORT, [], [far, near],
            )

            self.assertEqual(result["perspective_scale_violations"], 1)
            self.assertIn(
                "perspective_scale_inversion", result["layout_reasons"],
            )
            self.assertTrue(
                result["perspective_scale_pairs"][0]["proven_inversion"],
            )

    def test_group_repair_preserves_centres_rhythm_while_shrinking(self) -> None:
        """A protected-region conflict is repaired as one styled group."""
        support = [[0.05, 0.20], [0.95, 0.20], [0.95, 0.90], [0.05, 0.90]]
        protected = [[
            [0.70, 0.25], [0.90, 0.25], [0.90, 0.68], [0.70, 0.68],
        ]]
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as temporary:
            path = Path(temporary) / "product.png"
            rgba = np.zeros((120, 80, 4), np.uint8)
            rgba[2:118, 2:78, :3] = [170, 90, 45]
            rgba[2:118, 2:78, 3] = 255
            save_png(path, rgba)
            desired = []
            for index, (center, baseline) in enumerate(
                ((0.28, 0.55), (0.50, 0.62), (0.72, 0.56)), 1,
            ):
                item = _placement(path, f"p{index}", center, baseline)
                item["height_ratio"] = 0.20
                desired.append(item)

            rgba_cache: dict[str, np.ndarray] = {}
            mask_cache: dict[tuple[object, ...], np.ndarray] = {}
            greedy, messages = legalize_placements(
                (900, 600),
                support,
                protected,
                desired,
                automatic_edge_margin_ratio=0.012,
                shadow_direction=None,
                shadow_length_ratio=0.0,
                rgba_cache=rgba_cache,
                placement_mask_cache=mask_cache,
            )
            result, _, metrics = placement_solver_module._coherent_group_repair(
                (900, 600),
                support,
                protected,
                desired,
                greedy,
                messages,
                shadow_direction=None,
                shadow_length_ratio=0.0,
                shadow_softness_ratio=0.15,
                support_mask=None,
                rgba_cache=rgba_cache,
                placement_mask_cache=mask_cache,
            )

            self.assertEqual(metrics["coherent_group_repair_applied"], 1.0)
            self.assertLess(metrics["coherent_group_repair_scale"], 1.0)
            original_gaps = np.diff([item["center_x"] for item in desired])
            repaired_gaps = np.diff([item["center_x"] for item in result])
            np.testing.assert_allclose(repaired_gaps, original_gaps, atol=1e-8)
            scale_ratios = [
                result[index]["height_ratio"] / desired[index]["height_ratio"]
                for index in range(3)
            ]
            np.testing.assert_allclose(
                scale_ratios,
                [metrics["coherent_group_repair_scale"]] * 3,
                atol=1e-8,
            )


if __name__ == "__main__":
    unittest.main()
