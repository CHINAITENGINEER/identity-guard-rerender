from __future__ import annotations

import hashlib
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from strict_compositor.contracts import Lighting
from strict_compositor.image_io import save_png
from strict_compositor.learned_effects import (
    EffectContext,
    LearnedEffectContractError,
    LearnedEffectRequest,
)
from strict_compositor.learned_effects_runninghub import (
    RunningHubLearnedEffectProvider,
    runninghub_learned_effects_status,
)
from strict_compositor.runninghub import RunningHubError, RunningHubResult


def _provider_settings(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "runninghub_configured": True,
        "runninghub_learned_enabled": True,
        "runninghub_learned_mode": "comfyui",
        "runninghub_learned_app_id": "learned-scene-workflow-42",
        "runninghub_learned_source_node_id": "10",
        "runninghub_learned_source_field_name": "source_image",
        "runninghub_learned_product_mask_node_id": "20",
        "runninghub_learned_product_mask_field_name": "product_ids",
        "runninghub_learned_geometry_node_id": "30",
        "runninghub_learned_geometry_field_name": "geometry_rgb",
        "runninghub_learned_normal_node_id": "40",
        "runninghub_learned_normal_field_name": "normal_rgb",
        "runninghub_learned_background_node_id": "50",
        "runninghub_learned_background_field_name": "background_image",
        "runninghub_learned_prompt_node_id": "60",
        "runninghub_learned_prompt_field_name": "positive_prompt",
        "runninghub_learned_shadow_output_node_id": "90",
        "runninghub_learned_gain_output_node_id": "",
        "runninghub_learned_specular_output_node_id": "",
        "runninghub_learned_shadow_encoding": "white_is_shadow",
    }
    values.update(overrides)
    return values


def _three_product_request() -> LearnedEffectRequest:
    height, width = 24, 32
    yy, xx = np.indices((height, width))
    background = np.stack(
        [
            70 + xx,
            90 + yy,
            np.full((height, width), 120),
        ],
        axis=2,
    ).astype(np.uint8)
    support = np.ones((height, width), bool)
    support[:2] = False
    protected = np.zeros((height, width), bool)
    protected[2:5, 27:31] = True
    depth = np.linspace(
        0.0, 1.0, height * width, dtype=np.float32,
    ).reshape(height, width)
    normal = np.zeros((height, width, 3), np.float32)
    normal[..., 2] = 1.0
    lighting = Lighting(
        direction=(0.8, -0.6),
        shadow_length_ratio=0.12,
        cast_shadow_mode="directional",
        surface_material="glass",
    )
    # Deliberately not lexicographically sorted. The cloud product-ID mask and
    # the returned batch must preserve request/z-order, not sort these IDs.
    product_ids = ("product_10", "product_2", "product_1")
    boxes = ((5, 4, 11, 12), (12, 6, 18, 15), (21, 5, 28, 17))
    colours = ((160, 50, 40), (60, 170, 80), (70, 90, 190))
    products: list[EffectContext] = []
    for index, (product_id, box, colour) in enumerate(
        zip(product_ids, boxes, colours, strict=True),
    ):
        x0, y0, x1, y1 = box
        rgb = np.zeros((height, width, 3), np.uint8)
        rgb[y0:y1, x0:x1] = colour
        alpha = np.zeros((height, width), np.uint8)
        alpha[y0:y1, x0:x1] = 255
        identity = np.zeros((height, width), bool)
        identity[y0 + 2:y0 + 4, x0 + 1:x1 - 1] = True
        deterministic_shadow = np.zeros((height, width), np.uint8)
        deterministic_shadow[y1:min(height, y1 + 3), x0:x1 + 2] = (
            21 * (index + 1)
        )
        products.append(EffectContext(
            product_id=product_id,
            background_rgb=background,
            support_mask=support,
            protected_mask=protected,
            product_rgb=rgb,
            product_alpha=alpha,
            identity_guard_mask=identity,
            deterministic_shadow_alpha=deterministic_shadow,
            material_mode=(
                "opaque_glossy" if index == 0 else "mixed_transparent"
            ),
            lighting=lighting,
            depth=depth,
            normal=normal,
            metadata={"request_index": index},
        ))
    return LearnedEffectRequest(
        background_rgb=background,
        support_mask=support,
        protected_mask=protected,
        lighting=lighting,
        products=tuple(products),
        depth=depth,
        normal=normal,
        metadata={"fixture": "three-product-scene"},
    )


def _client_with_shadow_output(
    shadow: np.ndarray,
) -> Mock:
    client = Mock()
    client.upload.side_effect = [
        "cloud/shadowless_scene.png",
        "cloud/product_id_mask.png",
        "cloud/geometry.png",
        "cloud/normal.png",
        "cloud/background.png",
    ]
    client.create_task.return_value = "task-scene-123"
    client.wait.return_value = RunningHubResult(
        task_id="task-scene-123",
        file_url="https://cdn.example.test/shared-shadow.png",
        file_type="png",
        node_id="90",
    )

    def download(_url: str, destination: str | Path) -> Path:
        target = Path(destination)
        save_png(target, shadow)
        return target

    client.download.side_effect = download
    return client


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class RunningHubLearnedEffectProviderTests(unittest.TestCase):
    def test_three_products_use_one_task_and_one_shared_shadow_output(self) -> None:
        request = _three_product_request()
        shadow = np.zeros(request.background_rgb.shape[:2], np.uint8)
        shadow[14:21, 7:28] = 173
        client = _client_with_shadow_output(shadow)

        with tempfile.TemporaryDirectory() as directory:
            artifact_root = Path(directory) / "artifacts"
            with (
                patch(
                    "strict_compositor.learned_effects_runninghub.get_public_settings",
                    return_value=_provider_settings(),
                ) as settings_reader,
                patch(
                    "strict_compositor.learned_effects_runninghub.configured_client",
                    return_value=client,
                ),
            ):
                prediction = RunningHubLearnedEffectProvider(
                    timeout=321,
                    artifact_root=artifact_root,
                ).predict(request)

            # Validation and task creation use one immutable settings snapshot.
            settings_reader.assert_called_once_with()
            shadow_path = prediction.metadata["output_artifacts"]["shadow"]
            self.assertEqual(
                prediction.metadata["output_hashes"]["shadow_sha256"],
                _sha256(shadow_path),
            )

        self.assertEqual(set(prediction.products), {
            "product_10", "product_2", "product_1",
        })
        self.assertEqual(prediction.metadata["task_id"], "task-scene-123")
        self.assertTrue(prediction.metadata["scene_shared_maps"])
        self.assertEqual(client.create_task.call_count, 1)
        client.wait.assert_called_once_with(
            "task-scene-123", output_node_id="90", timeout=321.0,
        )
        client.download.assert_called_once()

        # One decoded receiver-space solution is intentionally shared by all
        # products; the provider must not launch or decode three independent
        # shadows with inconsistent light directions.
        product_predictions = [
            prediction.products[item.product_id] for item in request.products
        ]
        first_shadow = product_predictions[0].shadow_alpha
        self.assertIsNotNone(first_shadow)
        self.assertIs(first_shadow, product_predictions[1].shadow_alpha)
        self.assertIs(first_shadow, product_predictions[2].shadow_alpha)
        assert first_shadow is not None
        np.testing.assert_allclose(first_shadow, shadow.astype(np.float32) / 255.0)
        for item in product_predictions:
            self.assertTrue(item.metadata["scene_shared_maps"])
            self.assertEqual(item.metadata["task_id"], "task-scene-123")

    def test_maps_all_scene_inputs_into_one_explicit_node_list(self) -> None:
        request = _three_product_request()
        shadow = np.full(request.background_rgb.shape[:2], 64, np.uint8)
        client = _client_with_shadow_output(shadow)

        with tempfile.TemporaryDirectory() as directory:
            with (
                patch(
                    "strict_compositor.learned_effects_runninghub.get_public_settings",
                    return_value=_provider_settings(),
                ),
                patch(
                    "strict_compositor.learned_effects_runninghub.configured_client",
                    return_value=client,
                ),
            ):
                RunningHubLearnedEffectProvider(
                    artifact_root=Path(directory),
                ).predict(request)

        upload_names = [
            Path(item.args[0]).name for item in client.upload.call_args_list
        ]
        self.assertEqual(upload_names, [
            "shadowless_scene.png",
            "product_id_mask.png",
            "geometry_depth_support_protected.png",
            "normal_xyz.png",
            "background_original.png",
        ])
        client.create_task.assert_called_once()
        app_id, nodes = client.create_task.call_args.args
        self.assertEqual(app_id, "learned-scene-workflow-42")
        self.assertEqual(client.create_task.call_args.kwargs, {"mode": "comfyui"})
        self.assertEqual(nodes[:5], [
            {
                "nodeId": "10",
                "fieldName": "source_image",
                "fieldValue": "cloud/shadowless_scene.png",
            },
            {
                "nodeId": "20",
                "fieldName": "product_ids",
                "fieldValue": "cloud/product_id_mask.png",
            },
            {
                "nodeId": "30",
                "fieldName": "geometry_rgb",
                "fieldValue": "cloud/geometry.png",
            },
            {
                "nodeId": "40",
                "fieldName": "normal_rgb",
                "fieldValue": "cloud/normal.png",
            },
            {
                "nodeId": "50",
                "fieldName": "background_image",
                "fieldValue": "cloud/background.png",
            },
        ])
        self.assertEqual(nodes[5]["nodeId"], "60")
        self.assertEqual(nodes[5]["fieldName"], "positive_prompt")
        self.assertIn("Output only the configured map images", nodes[5]["fieldValue"])
        self.assertIn('"product_id":"product_10"', nodes[5]["fieldValue"])
        self.assertIn('"product_id":"product_2"', nodes[5]["fieldValue"])
        self.assertIn('"product_id":"product_1"', nodes[5]["fieldValue"])

    def test_bad_shared_shadow_output_rejects_the_whole_batch(self) -> None:
        request = _three_product_request()
        # Arbitrary colour is forbidden because a workflow must not smuggle a
        # rendered RGB candidate through the shared-shadow output node.
        invalid_rgb = np.zeros((*request.background_rgb.shape[:2], 3), np.uint8)
        invalid_rgb[..., 0] = 255
        client = _client_with_shadow_output(invalid_rgb)

        with tempfile.TemporaryDirectory() as directory:
            with (
                patch(
                    "strict_compositor.learned_effects_runninghub.get_public_settings",
                    return_value=_provider_settings(),
                ),
                patch(
                    "strict_compositor.learned_effects_runninghub.configured_client",
                    return_value=client,
                ),
            ):
                with self.assertRaisesRegex(
                    LearnedEffectContractError,
                    "single-channel grayscale PNG; arbitrary RGB is forbidden",
                ):
                    RunningHubLearnedEffectProvider(
                        artifact_root=Path(directory),
                    ).predict(request)

        # There was one scene task, but no partially usable product prediction
        # escaped after its shared output failed validation.
        client.create_task.assert_called_once()
        client.wait.assert_called_once()
        client.download.assert_called_once()

    def test_disabled_or_incomplete_configuration_never_touches_network(self) -> None:
        request = _three_product_request()
        cases = (
            (
                _provider_settings(runninghub_learned_enabled=False),
                "disabled",
            ),
            (
                _provider_settings(runninghub_learned_app_id=""),
                "requires app, source, product-mask, geometry, normal, prompt and shadow-output",
            ),
            (
                _provider_settings(runninghub_learned_source_node_id=""),
                "requires app, source, product-mask, geometry, normal, prompt and shadow-output",
            ),
            (
                _provider_settings(runninghub_learned_product_mask_node_id=""),
                "requires app, source, product-mask, geometry, normal, prompt and shadow-output",
            ),
            (
                _provider_settings(runninghub_learned_geometry_node_id=""),
                "requires app, source, product-mask, geometry, normal, prompt and shadow-output",
            ),
            (
                _provider_settings(runninghub_learned_normal_node_id=""),
                "requires app, source, product-mask, geometry, normal, prompt and shadow-output",
            ),
            (
                _provider_settings(runninghub_learned_prompt_node_id=""),
                "requires app, source, product-mask, geometry, normal, prompt and shadow-output",
            ),
            (
                _provider_settings(runninghub_learned_shadow_output_node_id=""),
                "requires app, source, product-mask, geometry, normal, prompt and shadow-output",
            ),
            (
                _provider_settings(runninghub_configured=False),
                "API key is not configured",
            ),
        )
        for public_settings, message in cases:
            with self.subTest(message=message, settings=public_settings):
                with (
                    patch(
                        "strict_compositor.learned_effects_runninghub.get_public_settings",
                        return_value=public_settings,
                    ),
                    patch(
                        "strict_compositor.learned_effects_runninghub.configured_client",
                    ) as configured,
                ):
                    with self.assertRaisesRegex(RunningHubError, message):
                        RunningHubLearnedEffectProvider().predict(request)
                    configured.assert_not_called()

    def test_v1_forbids_gain_and_specular_nodes_before_network(self) -> None:
        request = _three_product_request()
        cases = (
            _provider_settings(runninghub_learned_gain_output_node_id="91"),
            _provider_settings(runninghub_learned_specular_output_node_id="92"),
            _provider_settings(
                runninghub_learned_gain_output_node_id="91",
                runninghub_learned_specular_output_node_id="92",
            ),
        )
        for public_settings in cases:
            with self.subTest(settings=public_settings):
                with (
                    patch(
                        "strict_compositor.learned_effects_runninghub.get_public_settings",
                        return_value=public_settings,
                    ),
                    patch(
                        "strict_compositor.learned_effects_runninghub.configured_client",
                    ) as configured,
                ):
                    with self.assertRaisesRegex(
                        RunningHubError,
                        "V1 forbids gain/specular output nodes",
                    ):
                        RunningHubLearnedEffectProvider().predict(request)
                    configured.assert_not_called()

    def test_depth_and_normal_are_required_before_any_network_call(self) -> None:
        request = _three_product_request()
        cases = (
            replace(
                request,
                depth=None,
                products=tuple(replace(item, depth=None) for item in request.products),
            ),
            replace(
                request,
                normal=None,
                products=tuple(replace(item, normal=None) for item in request.products),
            ),
        )
        for incomplete in cases:
            with self.subTest(
                depth=incomplete.depth is not None,
                normal=incomplete.normal is not None,
            ):
                with (
                    patch(
                        "strict_compositor.learned_effects_runninghub.get_public_settings",
                        return_value=_provider_settings(),
                    ),
                    patch(
                        "strict_compositor.learned_effects_runninghub.configured_client",
                    ) as configured,
                ):
                    with self.assertRaisesRegex(
                        LearnedEffectContractError,
                        "requires validated depth and normal evidence",
                    ):
                        RunningHubLearnedEffectProvider().predict(incomplete)
                    configured.assert_not_called()

    def test_status_uses_one_snapshot_and_reports_readiness_dimensions(self) -> None:
        settings = _provider_settings()
        with patch(
            "strict_compositor.learned_effects_runninghub.get_public_settings",
            return_value=settings,
        ) as settings_reader:
            status = runninghub_learned_effects_status()

        settings_reader.assert_called_once_with()
        self.assertTrue(status["enabled"])
        self.assertTrue(status["secret_configured"])
        self.assertTrue(status["workflow_configured"])
        self.assertTrue(status["ready"])
        self.assertEqual(status["contract"], "scene-shadow-alpha-v1")

        with patch(
            "strict_compositor.learned_effects_runninghub.get_public_settings",
            return_value=_provider_settings(runninghub_configured=False),
        ):
            missing_secret = runninghub_learned_effects_status()
        self.assertTrue(missing_secret["enabled"])
        self.assertFalse(missing_secret["secret_configured"])
        self.assertTrue(missing_secret["workflow_configured"])
        self.assertFalse(missing_secret["ready"])

        with patch(
            "strict_compositor.learned_effects_runninghub.get_public_settings",
            return_value=_provider_settings(
                runninghub_learned_gain_output_node_id="91",
            ),
        ):
            forbidden_v2 = runninghub_learned_effects_status()
        self.assertFalse(forbidden_v2["workflow_configured"])
        self.assertFalse(forbidden_v2["ready"])

        for invalid_settings in (
            _provider_settings(runninghub_learned_mode="unknown"),
            _provider_settings(runninghub_learned_shadow_encoding="alpha"),
        ):
            with self.subTest(invalid_settings=invalid_settings):
                with patch(
                    "strict_compositor.learned_effects_runninghub.get_public_settings",
                    return_value=invalid_settings,
                ):
                    invalid_workflow = runninghub_learned_effects_status()
                self.assertFalse(invalid_workflow["workflow_configured"])
                self.assertFalse(invalid_workflow["ready"])

    def test_same_aspect_reasonable_output_can_be_explicitly_scaled(self) -> None:
        request = _three_product_request()
        half_resolution = np.full((12, 16), 128, np.uint8)
        client = _client_with_shadow_output(half_resolution)

        with tempfile.TemporaryDirectory() as directory:
            with (
                patch(
                    "strict_compositor.learned_effects_runninghub.get_public_settings",
                    return_value=_provider_settings(),
                ),
                patch(
                    "strict_compositor.learned_effects_runninghub.configured_client",
                    return_value=client,
                ),
            ):
                prediction = RunningHubLearnedEffectProvider(
                    artifact_root=Path(directory),
                ).predict(request)

        shadow = prediction.products["product_10"].shadow_alpha
        self.assertIsNotNone(shadow)
        assert shadow is not None
        self.assertEqual(shadow.shape, request.background_rgb.shape[:2])
        np.testing.assert_allclose(shadow, 128.0 / 255.0, atol=1e-6)

    def test_tiny_or_wrong_aspect_shadow_is_rejected_before_scaling(self) -> None:
        request = _three_product_request()
        cases = (
            (np.full((1, 1), 128, np.uint8), "too small"),
            (
                np.full((12, 12), 128, np.uint8),
                "aspect ratio differs from the scene",
            ),
        )
        for output, message in cases:
            with self.subTest(shape=output.shape):
                client = _client_with_shadow_output(output)
                with tempfile.TemporaryDirectory() as directory:
                    with (
                        patch(
                            "strict_compositor.learned_effects_runninghub.get_public_settings",
                            return_value=_provider_settings(),
                        ),
                        patch(
                            "strict_compositor.learned_effects_runninghub.configured_client",
                            return_value=client,
                        ),
                    ):
                        with self.assertRaisesRegex(
                            LearnedEffectContractError,
                            message,
                        ):
                            RunningHubLearnedEffectProvider(
                                artifact_root=Path(directory),
                            ).predict(request)

    def test_one_pixel_scene_output_is_rejected_even_when_dimensions_match(self) -> None:
        request = _three_product_request()
        one_pixel_request = replace(
            request,
            background_rgb=request.background_rgb[:1, :1],
            support_mask=request.support_mask[:1, :1],
            protected_mask=request.protected_mask[:1, :1],
            depth=request.depth[:1, :1],
            normal=request.normal[:1, :1],
            products=tuple(
                replace(
                    item,
                    background_rgb=item.background_rgb[:1, :1],
                    support_mask=item.support_mask[:1, :1],
                    protected_mask=item.protected_mask[:1, :1],
                    product_rgb=item.product_rgb[:1, :1],
                    product_alpha=item.product_alpha[:1, :1],
                    identity_guard_mask=item.identity_guard_mask[:1, :1],
                    deterministic_shadow_alpha=(
                        item.deterministic_shadow_alpha[:1, :1]
                    ),
                    depth=item.depth[:1, :1],
                    normal=item.normal[:1, :1],
                )
                for item in request.products
            ),
        )
        client = _client_with_shadow_output(np.full((1, 1), 128, np.uint8))
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch(
                    "strict_compositor.learned_effects_runninghub.get_public_settings",
                    return_value=_provider_settings(),
                ),
                patch(
                    "strict_compositor.learned_effects_runninghub.configured_client",
                    return_value=client,
                ),
            ):
                with self.assertRaisesRegex(
                    LearnedEffectContractError,
                    "1-pixel dimensions are forbidden",
                ):
                    RunningHubLearnedEffectProvider(
                        artifact_root=Path(directory),
                    ).predict(one_pixel_request)


if __name__ == "__main__":
    unittest.main()
