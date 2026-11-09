from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import strict_compositor.scene as scene
from strict_compositor.contracts import Job
from strict_compositor.scene import SceneAnalysis, _horizontal_from_depth, analyze_scene


class _FakeDepthEstimator:
    def __init__(self, depth: np.ndarray) -> None:
        self.depth = depth

    def __call__(self, _image: object) -> dict[str, np.ndarray]:
        return {"predicted_depth": self.depth}


class SceneGeometryEvidenceTests(unittest.TestCase):
    def test_depth_inference_returns_normalized_depth_and_unit_normals(self) -> None:
        image = np.full((60, 90, 3), 160, np.uint8)
        predicted = np.linspace(-3.0, 8.0, 24 * 36, dtype=np.float32).reshape(24, 36)

        with patch.object(scene, "_DEPTH_ESTIMATOR", _FakeDepthEstimator(predicted)):
            horizontal, depth, normal = _horizontal_from_depth(image)

        self.assertEqual(horizontal.shape, image.shape[:2])
        self.assertEqual(depth.shape, image.shape[:2])
        self.assertEqual(normal.shape, (*image.shape[:2], 3))
        self.assertEqual(depth.dtype, np.float32)
        self.assertEqual(normal.dtype, np.float32)
        self.assertGreaterEqual(float(depth.min()), 0.0)
        self.assertLessEqual(float(depth.max()), 1.0)
        np.testing.assert_allclose(
            np.linalg.norm(normal, axis=2),
            np.ones(image.shape[:2], np.float32),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_scene_analysis_persists_geometry_evidence_and_reloads_cache(self) -> None:
        height, width = 300, 500
        image = np.full((height, width, 3), 204, np.uint8)
        support = np.zeros((height, width), np.uint8)
        cv2.rectangle(support, (55, 75), (455, 260), 1, -1)
        depth = np.linspace(0.2, 0.8, height, dtype=np.float32)[:, None].repeat(
            width,
            axis=1,
        )
        normal = np.zeros((height, width, 3), np.float32)
        normal[..., 1] = 3.0
        normal[..., 2] = 4.0

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            background = root / "background.png"
            cv2.imwrite(str(background), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cache = root / "cache"
            no_glass = np.zeros_like(support)
            with (
                patch(
                    "strict_compositor.scene._elliptical_glass_support",
                    return_value=(no_glass, 0.0),
                ),
                patch(
                    "strict_compositor.scene._semantic_support",
                    return_value=(support, [("table", support)]),
                ),
                patch(
                    "strict_compositor.scene._horizontal_from_depth",
                    return_value=(support, depth, normal),
                ),
                patch(
                    "strict_compositor.scene._fit_support_depth_plane",
                    return_value=support,
                ),
            ):
                first = analyze_scene(background, cache, deep=True)

            self.assertIsNotNone(first.depth_map_path)
            self.assertIsNotNone(first.normal_map_path)
            assert first.depth_map_path is not None
            assert first.normal_map_path is not None
            saved_depth = np.load(first.depth_map_path, allow_pickle=False)
            saved_normal = np.load(first.normal_map_path, allow_pickle=False)
            self.assertEqual(saved_depth.dtype, np.float32)
            self.assertEqual(saved_normal.dtype, np.float32)
            self.assertGreaterEqual(float(saved_depth.min()), 0.0)
            self.assertLessEqual(float(saved_depth.max()), 1.0)
            np.testing.assert_allclose(
                np.linalg.norm(saved_normal, axis=2),
                1.0,
                rtol=1e-6,
                atol=1e-6,
            )

            second = analyze_scene(background, cache, deep=True)
            self.assertEqual(second.depth_map_path, first.depth_map_path)
            self.assertEqual(second.normal_map_path, first.normal_map_path)

    def test_scene_analysis_accepts_legacy_json_without_geometry_paths(self) -> None:
        current = SceneAnalysis(
            support_polygon=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
            protected_polygons=[],
            support_mask_path="support.png",
            front_edge_y=1.0,
            reflectivity=0.0,
            reflection_recommended=False,
            method="legacy",
            confidence=0.5,
            shadow_direction=[1.0, 1.0],
            shadow_length_ratio=0.1,
            shadow_opacity=0.2,
            shadow_softness_ratio=0.1,
        ).to_dict()
        current.pop("depth_map_path")
        current.pop("normal_map_path")

        restored = SceneAnalysis(**current)

        self.assertIsNone(restored.depth_map_path)
        self.assertIsNone(restored.normal_map_path)

    def test_persisted_geometry_is_registered_to_original_background_size(self) -> None:
        original_h, original_w = 180, 1401
        image = np.full((original_h, original_w, 3), 196, np.uint8)

        def no_glass(current: np.ndarray) -> tuple[np.ndarray, float]:
            return np.zeros(current.shape[:2], np.uint8), 0.0

        def table(current: np.ndarray) -> tuple[np.ndarray, list[tuple[str, np.ndarray]]]:
            mask = np.zeros(current.shape[:2], np.uint8)
            height, width = mask.shape
            cv2.rectangle(mask, (25, 28), (width - 26, height - 34), 1, -1)
            return mask, [("table", mask)]

        def geometry(current: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            mask, _ = table(current)
            height, width = mask.shape
            depth = np.linspace(0.1, 0.9, width, dtype=np.float32)[None, :].repeat(
                height,
                axis=0,
            )
            normal = np.zeros((height, width, 3), np.float32)
            normal[..., 0] = 1.0
            normal[..., 1] = 2.0
            return mask, depth, normal

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            background = root / "wide_background.png"
            cv2.imwrite(str(background), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            with (
                patch(
                    "strict_compositor.scene._elliptical_glass_support",
                    side_effect=no_glass,
                ),
                patch(
                    "strict_compositor.scene._semantic_support",
                    side_effect=table,
                ),
                patch(
                    "strict_compositor.scene._horizontal_from_depth",
                    side_effect=geometry,
                ),
                patch(
                    "strict_compositor.scene._fit_support_depth_plane",
                    side_effect=lambda horizontal, *_args: horizontal,
                ),
            ):
                analysis = analyze_scene(background, root / "cache", deep=True)

            assert analysis.depth_map_path is not None
            assert analysis.normal_map_path is not None
            saved_depth = np.load(analysis.depth_map_path, allow_pickle=False)
            saved_normal = np.load(analysis.normal_map_path, allow_pickle=False)
            self.assertEqual(saved_depth.shape, (original_h, original_w))
            self.assertEqual(saved_normal.shape, (original_h, original_w, 3))
            np.testing.assert_allclose(
                np.linalg.norm(saved_normal, axis=2),
                1.0,
                rtol=1e-6,
                atol=1e-6,
            )

    def test_job_geometry_paths_are_optional_and_resolve_from_root(self) -> None:
        payload = {
            "background": "background.png",
            "support_polygon": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
            "lighting": {"shadow_direction": [1.0, 1.0]},
            "products": [
                {
                    "id": "product_1",
                    "rgba": "product.png",
                    "homography": np.eye(3, dtype=np.float64).tolist(),
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy = Job.from_dict(payload, root)
            self.assertIsNone(legacy.depth_map)
            self.assertIsNone(legacy.normal_map)

            with_geometry = Job.from_dict(
                {
                    **payload,
                    "depth_map": "evidence/depth.npy",
                    "normal_map": "evidence/normal.npy",
                },
                root,
            )
            self.assertEqual(
                with_geometry.depth_map,
                (root / "evidence" / "depth.npy").resolve(),
            )
            self.assertEqual(
                with_geometry.normal_map,
                (root / "evidence" / "normal.npy").resolve(),
            )


if __name__ == "__main__":
    unittest.main()
