from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import strict_compositor.placement_solver as placement_solver_module
from strict_compositor.image_io import save_png
from strict_compositor.placement_solver import (
    _placement_mask,
    _planned_shadow_receiver_ratios,
    _valid_mask,
    compact_sparse_seeded_layout,
    layout_surface_diagnostics,
    legalize_placements,
    product_geometry,
    refine_seeded_composition,
    solve_aesthetic_layout,
)
from strict_compositor.render import _grounding_profile, contact_profile
from strict_compositor.scene import _protected_from_support_holes


def _irregular_product() -> np.ndarray:
    rgba = np.zeros((60, 40, 4), np.uint8)
    rgba[..., :3] = [180, 90, 40]
    rgba[5:56, 11:29, 3] = 255
    rgba[25:41, 29:39, 3] = 255  # hanging side appendage, not a foot
    rgba[55:60, 15:25, 3] = 255
    return rgba


def _wide_body_with_narrow_base() -> np.ndarray:
    rgba = np.zeros((80, 80, 4), np.uint8)
    rgba[..., :3] = [80, 150, 70]
    rgba[4:34, 2:78, 3] = 255
    rgba[30:72, 33:47, 3] = 255
    rgba[70:80, 29:51, 3] = 255
    return rgba


class PlacementSolverTests(unittest.TestCase):
    def test_candle_is_not_misclassified_as_a_labelled_can(self) -> None:
        candle = {
            "semantic_category": "decorative_pillar_candle",
            "physical_type": "rigid_pillar",
            "material_mode": "opaque",
            "rerender_policy": "auto",
            "identity_status": "unknown",
        }
        can = dict(candle, semantic_category="aluminum_can")
        self.assertEqual(
            placement_solver_module._identity_readability_floor(candle, 0.13),
            0.0,
        )
        self.assertGreaterEqual(
            placement_solver_module._identity_readability_floor(can, 0.13),
            0.135,
        )
        self.assertLessEqual(
            placement_solver_module._identity_readability_floor(can, 0.13),
            0.175,
        )

    def test_preview_solver_rejects_overlap_near_final_gate(self) -> None:
        accepted = np.zeros((100, 40), np.uint8)
        accepted[:, 0:10] = 255
        candidate = np.zeros_like(accepted)
        candidate[:, 10:20] = 255
        candidate[0:6, 9] = 255  # 6 / 1000 = 0.006 overlap
        support = np.ones_like(accepted, bool)
        protected = np.zeros_like(accepted, bool)
        self.assertFalse(_valid_mask(candidate, support, protected, [accepted]))
    def test_contact_profile_ignores_hanging_appendages(self) -> None:
        profile = contact_profile(_irregular_product()[..., 3])
        xs = np.flatnonzero(profile.any(axis=0))
        self.assertGreater(len(xs), 0)
        self.assertLessEqual(int(xs.max()), 25)
        self.assertGreaterEqual(int(xs.min()), 14)

    def test_contact_profile_does_not_promote_high_side_edges_to_feet(self) -> None:
        alpha = np.zeros((400, 120), np.uint8)
        alpha[20:120, 0] = 255
        alpha[30:150, 119] = 255
        alpha[380:400, 2:118] = 255

        profile = contact_profile(alpha)
        ys, xs = np.nonzero(profile)

        self.assertGreater(len(xs), 0)
        self.assertGreaterEqual(int(xs.min()), 2)
        self.assertLessEqual(int(xs.max()), 117)
        self.assertGreaterEqual(int(ys.min()), 397)

    def test_grouped_product_detects_secondary_straight_base(self) -> None:
        alpha = np.zeros((120, 100), np.uint8)
        alpha[8:62, 8:92] = 255
        for x in range(8, 92):
            alpha[62:64 + (x - 8) // 14, x] = 255
        alpha[42:112, 43:57] = 255
        alpha[108:118, 38:62] = 255
        primary = contact_profile(alpha)
        grounding = _grounding_profile(alpha)
        primary_rows = np.flatnonzero(primary.any(axis=1))
        grounding_rows = np.flatnonzero(grounding.any(axis=1))
        self.assertGreater(int(primary_rows.min()), 100)
        self.assertLess(int(grounding_rows.min()), 75)
        self.assertGreater(np.count_nonzero(grounding), np.count_nonzero(primary))

    def test_valid_manual_placement_is_preserved_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "product.png"
            save_png(path, _irregular_product())
            desired = {
                "id": "p",
                "source": str(path),
                "rgba": str(path),
                "center_x": 0.42,
                "baseline_y": 0.76,
                "height_ratio": 0.20,
                "z_order": 7,
            }
            repaired, messages = legalize_placements(
                (400, 300),
                [[0.08, 0.40], [0.92, 0.40], [0.92, 0.90], [0.08, 0.90]],
                [],
                [desired],
            )
            self.assertEqual(messages, [])
            self.assertEqual(repaired[0]["center_x"], desired["center_x"])
            self.assertEqual(repaired[0]["baseline_y"], desired["baseline_y"])
            self.assertEqual(repaired[0]["height_ratio"], desired["height_ratio"])

    def test_body_may_cross_support_boundary_when_base_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "wide_product.png"
            save_png(path, _wide_body_with_narrow_base())
            desired = {
                "id": "p",
                "source": str(path),
                "rgba": str(path),
                "center_x": 0.50,
                "baseline_y": 0.78,
                "height_ratio": 0.58,
                "z_order": 1,
            }
            repaired, messages = legalize_placements(
                (400, 300),
                [[0.43, 0.50], [0.57, 0.50], [0.57, 0.86], [0.43, 0.86]],
                [],
                [desired],
            )
            self.assertEqual(messages, [])
            self.assertEqual(repaired[0]["center_x"], desired["center_x"])
            self.assertEqual(repaired[0]["baseline_y"], desired["baseline_y"])
            self.assertEqual(repaired[0]["height_ratio"], desired["height_ratio"])

    def test_exact_support_mask_still_constrains_only_the_contact_footprint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "wide_product.png"
            save_png(path, _wide_body_with_narrow_base())
            desired = {
                "id": "p",
                "source": str(path),
                "rgba": str(path),
                "center_x": 0.50,
                "baseline_y": 0.78,
                "height_ratio": 0.58,
                "z_order": 1,
            }
            exact_support = np.zeros((300, 400), bool)
            exact_support[150:259, 172:229] = True
            repaired, messages = legalize_placements(
                (400, 300),
                [[0.05, 0.20], [0.95, 0.20], [0.95, 0.95], [0.05, 0.95]],
                [],
                [desired],
                support_mask=exact_support,
            )
            self.assertEqual(messages, [])
            self.assertEqual(repaired[0]["center_x"], desired["center_x"])
            self.assertEqual(repaired[0]["baseline_y"], desired["baseline_y"])
            self.assertEqual(repaired[0]["height_ratio"], desired["height_ratio"])

    def test_exact_support_mask_overrides_the_polygon_for_legalization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "product.png"
            save_png(path, _irregular_product())
            desired = {
                "id": "p",
                "source": str(path),
                "rgba": str(path),
                "center_x": 0.50,
                "baseline_y": 0.76,
                "height_ratio": 0.20,
                "z_order": 1,
            }
            exact_support = np.zeros((300, 400), bool)
            exact_support[120:271, 40:161] = True
            repaired, messages = legalize_placements(
                (400, 300),
                [[0.05, 0.20], [0.95, 0.20], [0.95, 0.95], [0.05, 0.95]],
                [],
                [desired],
                support_mask=exact_support,
            )
            self.assertTrue(messages)
            self.assertLess(repaired[0]["center_x"], desired["center_x"])

    def test_automatic_layout_keeps_a_margin_while_manual_layout_does_not(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "edge_product.png"
            save_png(path, _wide_body_with_narrow_base())
            desired = {
                "id": "p",
                "source": str(path),
                "center_x": 0.88,
                "baseline_y": 0.76,
                "height_ratio": 0.25,
                "z_order": 1,
            }
            support = [[0.08, 0.40], [0.92, 0.40], [0.92, 0.90], [0.08, 0.90]]
            manual, _ = legalize_placements((400, 300), support, [], [desired])
            automatic, messages = legalize_placements(
                (400, 300),
                support,
                [],
                [desired],
                automatic_edge_margin_ratio=0.025,
            )
            self.assertEqual(manual[0]["center_x"], desired["center_x"])
            self.assertLess(automatic[0]["center_x"], desired["center_x"])
            self.assertTrue(messages)

    def test_automatic_layout_reserves_height_proportional_shadow_corridor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "shadow_edge_product.png"
            save_png(path, _irregular_product())
            desired = {
                "id": "p",
                "source": str(path),
                "rgba": str(path),
                "center_x": 0.50,
                "baseline_y": 0.86,
                "height_ratio": 0.24,
                "z_order": 1,
            }
            support = [[0.05, 0.25], [0.95, 0.25], [0.95, 0.90], [0.05, 0.90]]
            manual, _ = legalize_placements((500, 320), support, [], [desired])
            automatic, messages = legalize_placements(
                (500, 320),
                support,
                [],
                [desired],
                shadow_direction=[0.0, 1.0],
                shadow_length_ratio=0.08,
                shadow_softness_ratio=0.16,
            )
            self.assertEqual(manual[0]["baseline_y"], desired["baseline_y"])
            self.assertLess(automatic[0]["baseline_y"], desired["baseline_y"])
            self.assertTrue(messages)

            alpha = _placement_mask(
                _irregular_product(), automatic[0], (320, 500),
            )
            support_mask = np.zeros((320, 500), bool)
            support_mask[80:288, 25:475] = True
            hard_ratio, soft_ratio = _planned_shadow_receiver_ratios(
                alpha,
                support_mask,
                np.zeros_like(support_mask),
                [0.0, 1.0],
                0.08,
                0.16,
            )
            self.assertGreaterEqual(hard_ratio, 0.992)
            self.assertGreaterEqual(soft_ratio, 0.965)

    def test_surface_flat_mode_requires_complete_silhouette_support(self) -> None:
        alpha = _wide_body_with_narrow_base()[..., 3]
        support = np.zeros_like(alpha, bool)
        support[68:80, 25:55] = True
        protected = np.zeros_like(support)
        self.assertTrue(_valid_mask(alpha, support, protected, [], "upright"))
        self.assertFalse(
            _valid_mask(alpha, support, protected, [], "surface_flat")
        )

    def test_small_manual_overflow_is_repaired_not_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "product.png"
            save_png(path, _irregular_product())
            desired = {
                "id": "p",
                "source": str(path),
                "rgba": str(path),
                "center_x": 0.50,
                "baseline_y": 0.94,
                "height_ratio": 0.18,
                "z_order": 1,
            }
            repaired, messages = legalize_placements(
                (400, 300),
                [[0.08, 0.40], [0.92, 0.40], [0.92, 0.90], [0.08, 0.90]],
                [],
                [desired],
            )
            self.assertEqual(len(messages), 1)
            self.assertAlmostEqual(repaired[0]["center_x"], desired["center_x"], places=6)
            self.assertLess(abs(repaired[0]["baseline_y"] - desired["baseline_y"]), 0.09)
            self.assertEqual(repaired[0]["height_ratio"], desired["height_ratio"])

    def test_seeded_row_gets_safe_depth_stagger(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = []
            for index in range(3):
                path = Path(temporary) / f"product_{index}.png"
                save_png(path, _irregular_product())
                paths.append(path)
            products = [
                {
                    "id": f"p{index}",
                    "source": str(path),
                    "rgba": str(path),
                    "center_x": 0.25 + index * 0.25,
                    "baseline_y": 0.72,
                    "height_ratio": 0.16,
                    "z_order": index + 1,
                }
                for index, path in enumerate(paths)
            ]
            refined, messages = refine_seeded_composition(
                (600, 400),
                [[0.05, 0.30], [0.95, 0.30], [0.95, 0.92], [0.05, 0.92]],
                [],
                products,
            )
            baselines = [item["baseline_y"] for item in refined]
            self.assertGreaterEqual(max(baselines) - min(baselines), 0.04)
            self.assertTrue(any("前后层次" in message for message in messages))

    def test_sparse_semantic_seed_is_compacted_without_rescaling_or_reordering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            products = []
            for index, source_width in enumerate((70, 42, 55)):
                rgba = np.zeros((100, source_width, 4), np.uint8)
                rgba[..., :3] = [90 + index * 30, 70, 45]
                rgba[3:98, 2:-2, 3] = 255
                path = Path(temporary) / f"semantic_{index}.png"
                save_png(path, rgba)
                products.append({
                    "id": f"p{index}",
                    "source": str(path),
                    "rgba": str(path),
                    "center_x": (0.18, 0.51, 0.82)[index],
                    "baseline_y": (0.58, 0.62, 0.60)[index],
                    "height_ratio": (0.25, 0.14, 0.18)[index],
                    "z_order": index + 1,
                    "estimated_real_height_cm": (28.0, 9.0, 12.0)[index],
                    "size_confidence": 0.85,
                })
            support = [
                [0.04, 0.28], [0.96, 0.28], [0.96, 0.92], [0.04, 0.92],
            ]
            compacted, metrics, messages = compact_sparse_seeded_layout(
                (900, 600), support, [], products,
            )

            self.assertEqual(metrics["seed_compaction_applied"], 1.0)
            self.assertLessEqual(metrics["seed_gap_after_max"], 0.62)
            self.assertEqual(metrics["seed_compaction_scale_change_max"], 0.0)
            self.assertTrue(messages)
            self.assertEqual(
                [item["id"] for item in sorted(
                    compacted, key=lambda item: item["center_x"],
                )],
                ["p0", "p1", "p2"],
            )
            self.assertEqual(
                [item["height_ratio"] for item in compacted],
                [item["height_ratio"] for item in products],
            )

    def test_unlegalizable_minimax_seed_uses_joint_planner_and_keeps_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = []
            for index in range(3):
                path = Path(temporary) / f"invalid_seed_product_{index}.png"
                save_png(path, _irregular_product())
                paths.append(path)
            products = [
                {
                    "id": f"p{index}",
                    "source": str(path),
                    "rgba": str(path),
                    "center_x": 0.25 + index * 0.25,
                    "baseline_y": 0.98,
                    "height_ratio": 0.16,
                    "z_order": index + 1,
                    "semantic_role": f"advisor-role-{index}",
                    "source_light_compatibility": 0.70 + index * 0.05,
                }
                for index, path in enumerate(paths)
            ]
            original_legalize = placement_solver_module.legalize_placements
            call_count = 0

            def reject_seed_then_delegate(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count <= 2:
                    raise ValueError("invalid MiniMax seed")
                return original_legalize(*args, **kwargs)

            support = [[0.05, 0.25], [0.95, 0.25], [0.95, 0.92], [0.05, 0.92]]
            with patch.object(
                placement_solver_module,
                "legalize_placements",
                side_effect=reject_seed_then_delegate,
            ):
                refined, messages = refine_seeded_composition(
                    (600, 400), support, [], products,
                )

            self.assertGreaterEqual(call_count, 3)
            refined_by_id = {item["id"]: item for item in refined}
            for original in products:
                item = refined_by_id[original["id"]]
                self.assertEqual(item["semantic_role"], original["semantic_role"])
                self.assertEqual(
                    item["source_light_compatibility"],
                    original["source_light_compatibility"],
                )
            self.assertIn(
                "MiniMax 种子无法合法化，已本地联合重排",
                messages,
            )

    def test_front_edge_adviser_seed_triggers_joint_surface_replan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = []
            for index in range(3):
                path = Path(temporary) / f"front_product_{index}.png"
                save_png(path, _irregular_product())
                paths.append(path)
            products = [
                {
                    "id": f"p{index}",
                    "source": str(path),
                    "rgba": str(path),
                    "center_x": 0.25 + index * 0.25,
                    "baseline_y": 0.875,
                    "height_ratio": 0.16,
                    "z_order": index + 1,
                }
                for index, path in enumerate(paths)
            ]
            support = [[0.05, 0.25], [0.95, 0.25], [0.95, 0.92], [0.05, 0.92]]
            refined, messages = refine_seeded_composition(
                (600, 400), support, [], products,
            )
            diagnostics = layout_surface_diagnostics(
                (600, 400), support, refined,
            )
            self.assertLessEqual(diagnostics["surface_depth_max"], 0.72)
            self.assertGreaterEqual(
                diagnostics["front_clearance_min_heights"], 0.30,
            )
            self.assertTrue(any("联合重排" in message for message in messages))

    def test_visual_mass_scale_ignores_a_thin_upper_appendage(self) -> None:
        compact = np.zeros((60, 40, 4), np.uint8)
        compact[..., :3] = [140, 90, 50]
        compact[10:58, 5:35, 3] = 255
        tall = np.zeros((100, 40, 4), np.uint8)
        tall[..., :3] = compact[0, 0, :3]
        tall[42:90, 5:35, 3] = 255
        tall[2:42, 19:21, 3] = 255
        with tempfile.TemporaryDirectory() as temporary:
            compact_path = Path(temporary) / "compact.png"
            tall_path = Path(temporary) / "tall.png"
            save_png(compact_path, compact)
            save_png(tall_path, tall)
            compact_geometry = product_geometry(compact_path)
            tall_geometry = product_geometry(tall_path)
        compact_mass_height = (
            compact_geometry.content_height * compact_geometry.visual_mass_fraction
        )
        tall_mass_height = tall_geometry.content_height * tall_geometry.visual_mass_fraction
        self.assertLess(abs(compact_mass_height - tall_mass_height), 8.0)
        self.assertLess(tall_geometry.visual_mass_fraction, 0.70)

    def test_trustworthy_physical_scale_caps_group_readability_boost(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            products = []
            specifications = [
                (130, 26.0, 0.80),
                (100, 16.5, 0.45),
                (120, 22.0, 0.65),
            ]
            for index, (source_height, real_height, salience) in enumerate(
                specifications
            ):
                rgba = np.zeros((source_height, 48, 4), np.uint8)
                rgba[..., :3] = [130, 80, 40]
                rgba[3:-2, 3:-3, 3] = 255
                path = Path(temporary) / f"commercial_{index}.png"
                save_png(path, rgba)
                products.append({
                    "id": f"p{index}",
                    "source": str(path),
                    "rgba": str(path),
                    "estimated_real_height_cm": real_height,
                    "size_confidence": 0.90,
                    # Jointly trustworthy product/receiver dimensions are the
                    # physical scale authority.  The labelled can may keep its
                    # local identity floor, but it must not enlarge all three
                    # products by the old 1.2--2.0 commercial factor.
                    "receiver_width_or_diameter_cm": 150.0,
                    "receiver_size_confidence": 0.90,
                    "visual_salience": salience,
                    "semantic_category": (
                        "aluminum_can" if index == 1 else "decor_object"
                    ),
                    "material_mode": "metal" if index == 1 else "opaque",
                    "rerender_policy": "never" if index == 1 else "auto",
                })

            solved, metrics = solve_aesthetic_layout(
                (900, 600),
                [[0.08, 0.48], [0.92, 0.48], [0.92, 0.92], [0.08, 0.92]],
                [],
                products,
                shadow_length_ratio=0.0,
            )
            geometries = [product_geometry(item["rgba"]) for item in solved]
            visual_masses = np.asarray([
                item["height_ratio"] * geometry.visual_mass_fraction
                for item, geometry in zip(solved, geometries)
            ])
            self.assertLessEqual(
                float(metrics["commercial_readability_scale"]), 1.05,
            )
            self.assertGreaterEqual(
                float(metrics["commercial_physical_evidence_median"]), 0.35,
            )
            self.assertEqual(
                float(metrics["commercial_identity_guard_count"]), 1.0,
            )
            solved_by_id = {item["id"]: item for item in solved}
            self.assertGreaterEqual(solved_by_id["p1"]["height_ratio"], 0.135)
            self.assertGreater(
                solved_by_id["p0"]["height_ratio"],
                solved_by_id["p2"]["height_ratio"],
            )
            self.assertGreater(
                solved_by_id["p2"]["height_ratio"],
                solved_by_id["p1"]["height_ratio"],
            )

            entities = []
            for item, geometry in zip(solved, geometries):
                display_width = (
                    item["height_ratio"]
                    * geometry.content_aspect
                    * 600.0 / 900.0
                )
                entities.append((
                    item["center_x"] - display_width * 0.5,
                    item["center_x"] + display_width * 0.5,
                ))
            entities.sort()
            self.assertGreaterEqual(entities[1][0] - entities[0][1], 0.0)
            self.assertGreaterEqual(entities[2][0] - entities[1][1], 0.0)
            self.assertGreaterEqual(
                float(metrics["entity_gap_to_median_width"]),
                0.30,
            )

    def test_unlabelled_three_product_group_is_not_commercially_enlarged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            products = []
            for index, real_height in enumerate((25.0, 22.0, 24.0)):
                rgba = np.zeros((110 + index * 5, 46, 4), np.uint8)
                rgba[..., :3] = [120, 90, 65]
                rgba[3:-3, 3:-3, 3] = 255
                path = Path(temporary) / f"unlabelled_{index}.png"
                save_png(path, rgba)
                products.append({
                    "id": f"p{index}",
                    "source": str(path),
                    "rgba": str(path),
                    "estimated_real_height_cm": real_height,
                    "size_confidence": 0.75,
                    "receiver_width_or_diameter_cm": 130.0,
                    "receiver_size_confidence": 0.70,
                    "semantic_category": "decorative_sculpture",
                    "physical_type": "rigid_sculpture",
                    "material_mode": "opaque_glossy",
                    "rerender_policy": "auto",
                })

            _, metrics = solve_aesthetic_layout(
                (900, 600),
                [[0.08, 0.42], [0.92, 0.42], [0.92, 0.92], [0.08, 0.92]],
                [],
                products,
                shadow_length_ratio=0.0,
            )

            self.assertEqual(
                float(metrics["commercial_identity_guard_count"]), 0.0,
            )
            self.assertAlmostEqual(
                float(metrics["commercial_readability_scale"]), 1.0,
                places=6,
            )

    def test_single_surface_flat_product_becomes_foreground_bridge(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            products = []
            specifications = (
                ("cup", "upright", 8.0, 9.0, 0.35),
                ("bowl", "surface_flat", 5.0, 22.0, 0.45),
                ("candle", "upright", 20.0, 8.0, 0.75),
            )
            for index, (product_id, mode, height_cm, footprint_cm, salience) in enumerate(
                specifications
            ):
                rgba = np.zeros((110, 90 if mode == "surface_flat" else 46, 4), np.uint8)
                rgba[..., :3] = [145, 95, 60]
                rgba[3:-3, 3:-3, 3] = 255
                path = folder / f"flat_bridge_{index}.png"
                save_png(path, rgba)
                products.append({
                    "id": product_id,
                    "source": str(path),
                    "rgba": str(path),
                    "placement_mode": mode,
                    "estimated_real_height_cm": height_cm,
                    "footprint_width_cm": footprint_cm,
                    "size_confidence": 0.85,
                    "receiver_width_or_diameter_cm": 120.0,
                    "receiver_size_confidence": 0.75,
                    "visual_salience": salience,
                })

            solved, _ = solve_aesthetic_layout(
                (900, 600),
                [[0.06, 0.42], [0.94, 0.42], [0.94, 0.92], [0.06, 0.92]],
                [],
                products,
                shadow_length_ratio=0.0,
            )
            by_id = {str(item["id"]): item for item in solved}
            side_centers = sorted((
                float(by_id["cup"]["center_x"]),
                float(by_id["candle"]["center_x"]),
            ))
            self.assertGreater(float(by_id["bowl"]["center_x"]), side_centers[0])
            self.assertLess(float(by_id["bowl"]["center_x"]), side_centers[1])
            self.assertGreater(
                float(by_id["bowl"]["baseline_y"]),
                min(
                    float(by_id["cup"]["baseline_y"]),
                    float(by_id["candle"]["baseline_y"]),
                ),
            )
            # Measure the foreground bridge against the interpolated line
            # between the two legalised anchors.  A merely larger image-space
            # Y can still form a ruler-straight row on an oblique tabletop.
            blend = (
                float(by_id["bowl"]["center_x"])
                - float(by_id["cup"]["center_x"])
            ) / (
                float(by_id["candle"]["center_x"])
                - float(by_id["cup"]["center_x"])
            )
            anchor_line = (
                float(by_id["cup"]["baseline_y"]) * (1.0 - blend)
                + float(by_id["candle"]["baseline_y"]) * blend
            )
            median_height = float(np.median([
                float(item["height_ratio"]) for item in solved
            ]))
            self.assertGreaterEqual(
                float(by_id["bowl"]["baseline_y"]) - anchor_line,
                median_height * 0.18,
            )

    def test_concave_support_not_misread_as_a_protected_object(self) -> None:
        support = np.zeros((120, 180), np.uint8)
        support[20:100, 15:165] = 1
        support[20:55, 80:165] = 0  # open notch, not an enclosed object
        self.assertEqual(_protected_from_support_holes(support), [])

    def test_true_enclosed_support_hole_is_protected(self) -> None:
        support = np.zeros((120, 180), np.uint8)
        support[15:105, 15:165] = 1
        support[48:74, 72:108] = 0
        protected = _protected_from_support_holes(support)
        self.assertEqual(len(protected), 1)
        self.assertGreaterEqual(len(protected[0]), 3)


if __name__ == "__main__":
    unittest.main()
