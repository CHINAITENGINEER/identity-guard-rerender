from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

import numpy as np
from PIL import Image

from strict_compositor import settings
from strict_compositor.contracts import Job
from strict_compositor.hybrid import HybridRenderRequest, HybridRenderResult
from strict_compositor.hybrid_runninghub import (
    RunningHubHybridProvider,
    runninghub_hybrid_status,
)
from strict_compositor.image_io import save_png
from strict_compositor.pipeline import StrictCompositor
from strict_compositor.runninghub import RunningHubError, RunningHubResult


def _provider_settings(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "runninghub_hybrid_enabled": True,
        "runninghub_hybrid_mode": "comfyui",
        "runninghub_hybrid_app_id": "render-workflow-42",
        "runninghub_hybrid_source_node_id": "10",
        "runninghub_hybrid_source_field_name": "source_image",
        "runninghub_hybrid_mask_node_id": "20",
        "runninghub_hybrid_mask_field_name": "edit_mask",
        "runninghub_hybrid_prompt_node_id": "30",
        "runninghub_hybrid_prompt_field_name": "positive_prompt",
        "runninghub_hybrid_output_node_id": "90",
    }
    values.update(overrides)
    return values


class RunningHubHybridProviderTests(unittest.TestCase):
    def test_maps_source_mask_prompt_and_explicit_output_nodes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "strict.png"
            edit_mask = root / "edit.png"
            protect_mask = root / "protect.png"
            save_png(base, np.full((48, 64, 3), 80, np.uint8))
            save_png(edit_mask, np.full((48, 64), 255, np.uint8))
            save_png(protect_mask, np.zeros((48, 64), np.uint8))

            client = Mock()
            client.upload.side_effect = ["cloud/strict.png", "cloud/edit.png"]
            client.create_task.return_value = "task-123"
            client.wait.return_value = RunningHubResult(
                task_id="task-123",
                file_url="https://cdn.example.test/result.png",
                file_type="png",
                node_id="90",
            )

            def download(_url: str, destination: str | Path) -> Path:
                target = Path(destination)
                save_png(target, np.full((48, 64, 3), 120, np.uint8))
                return target

            client.download.side_effect = download
            request = HybridRenderRequest(
                strict_base=base,
                edit_mask=edit_mask,
                protect_mask=protect_mask,
                prompt="rerender only transparent optics",
                expected_size=(64, 48),
                output_dir=root / "output",
            )

            with (
                patch(
                    "strict_compositor.hybrid_runninghub.get_public_settings",
                    return_value=_provider_settings(),
                ),
                patch(
                    "strict_compositor.hybrid_runninghub.configured_client",
                    return_value=client,
                ),
            ):
                result = RunningHubHybridProvider(timeout=321).render(request)

            self.assertEqual(
                client.upload.call_args_list,
                [call(base), call(edit_mask)],
            )
            client.create_task.assert_called_once_with(
                "render-workflow-42",
                [
                    {
                        "nodeId": "10",
                        "fieldName": "source_image",
                        "fieldValue": "cloud/strict.png",
                    },
                    {
                        "nodeId": "20",
                        "fieldName": "edit_mask",
                        "fieldValue": "cloud/edit.png",
                    },
                    {
                        "nodeId": "30",
                        "fieldName": "positive_prompt",
                        "fieldValue": "rerender only transparent optics",
                    },
                ],
                mode="comfyui",
            )
            client.wait.assert_called_once_with(
                "task-123",
                output_node_id="90",
                timeout=321.0,
            )
            self.assertEqual(result.provider, "runninghub")
            self.assertEqual(result.task_id, "task-123")
            self.assertTrue(result.candidate_path.is_file())
            self.assertEqual(result.provenance["source_node_id"], "10")
            self.assertEqual(result.provenance["mask_node_id"], "20")
            self.assertEqual(result.provenance["output_node_id"], "90")

    def test_disabled_or_incomplete_workflow_fails_before_client_use(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = HybridRenderRequest(
                strict_base=root / "strict.png",
                edit_mask=root / "edit.png",
                protect_mask=root / "protect.png",
                prompt="",
                expected_size=(64, 48),
                output_dir=root,
            )
            cases = (
                (
                    _provider_settings(runninghub_hybrid_enabled=False),
                    "disabled",
                ),
                (
                    _provider_settings(runninghub_hybrid_output_node_id=""),
                    "requires app, source, mask and output node IDs",
                ),
            )
            for public_settings, expected_message in cases:
                with self.subTest(expected_message=expected_message):
                    with (
                        patch(
                            "strict_compositor.hybrid_runninghub.get_public_settings",
                            return_value=public_settings,
                        ),
                        patch(
                            "strict_compositor.hybrid_runninghub.configured_client",
                        ) as configured,
                    ):
                        with self.assertRaisesRegex(
                            RunningHubError, expected_message,
                        ):
                            RunningHubHybridProvider().render(request)
                        configured.assert_not_called()

    def test_status_requires_independent_hybrid_node_mapping(self) -> None:
        cutout_only = _provider_settings(
            runninghub_hybrid_app_id="",
            runninghub_hybrid_source_node_id="",
            runninghub_hybrid_mask_node_id="",
            runninghub_hybrid_output_node_id="",
        )
        cutout_only.update({
            "runninghub_cutout_id": "cutout-app",
            "runninghub_input_node_id": "6",
            "runninghub_output_node_id": "7",
        })
        with patch(
            "strict_compositor.hybrid_runninghub.get_public_settings",
            return_value=cutout_only,
        ):
            status = runninghub_hybrid_status()
        self.assertTrue(status["enabled"])
        self.assertFalse(status["workflow_configured"])
        self.assertEqual(status["provider"], "runninghub")


class HybridSettingsTests(unittest.TestCase):
    def test_cutout_and_hybrid_workflows_round_trip_independently(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "integrations.json"
            payload = {
                "runninghub_enabled": True,
                "runninghub_mode": "ai_app",
                "runninghub_cutout_id": "cutout-app",
                "runninghub_input_node_id": "6",
                "runninghub_input_field_name": "image",
                "runninghub_output_node_id": "7",
                "runninghub_hybrid_enabled": True,
                "runninghub_hybrid_mode": "comfyui",
                "runninghub_hybrid_app_id": "render-workflow",
                "runninghub_hybrid_source_node_id": "10",
                "runninghub_hybrid_source_field_name": "source_image",
                "runninghub_hybrid_mask_node_id": "20",
                "runninghub_hybrid_mask_field_name": "edit_mask",
                "runninghub_hybrid_prompt_node_id": "30",
                "runninghub_hybrid_prompt_field_name": "prompt",
                "runninghub_hybrid_output_node_id": "90",
            }
            with (
                patch.object(settings, "SETTINGS_PATH", path),
                patch.object(settings, "PRIVATE_SETTINGS_DIR", path.parent),
                patch.dict("os.environ", {}, clear=True),
            ):
                public = settings.save_integrations(payload)
                loaded = settings.get_public_settings()

            for key, value in payload.items():
                self.assertEqual(public[key], value)
                self.assertEqual(loaded[key], value)
            self.assertNotEqual(
                public["runninghub_cutout_id"],
                public["runninghub_hybrid_app_id"],
            )
            raw = json.loads(path.read_text(encoding="utf-8"))["public"]
            self.assertEqual(raw["runninghub_cutout_id"], "cutout-app")
            self.assertEqual(raw["runninghub_hybrid_app_id"], "render-workflow")


class _FailingProvider:
    def render(self, _request: HybridRenderRequest) -> HybridRenderResult:
        raise RunningHubError("temporary provider failure")


class _StaticCandidateProvider:
    def __init__(self, candidate: Path) -> None:
        self.candidate = candidate

    def render(self, _request: HybridRenderRequest) -> HybridRenderResult:
        return HybridRenderResult(
            candidate_path=self.candidate,
            provider="runninghub",
            task_id="task-invalid",
        )


class HybridProviderFallbackTests(unittest.TestCase):
    def _job(self, root: Path) -> Job:
        background = np.full((120, 160, 3), [175, 145, 110], np.uint8)
        product = np.zeros((64, 46, 4), np.uint8)
        product[4:60, 5:41, :3] = [205, 200, 190]
        product[4:60, 5:41, 3] = 155
        product[55:62, 8:38, 3] = 235
        save_png(root / "background.png", background)
        save_png(root / "product.png", product)
        return Job.from_dict({
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
                "homography": [[1, 0, 55], [0, 1, 30], [0, 0, 1]],
                "material_mode": "clear_glass",
                "material_confidence": 0.95,
                "transmission_strength": 0.9,
                "semantic_category": "glass_pitcher",
            }],
            "hybrid_render": {
                "enabled": True,
                "provider": "runninghub",
            },
        }, root)

    def _assert_byte_exact_fallback(
        self,
        root: Path,
        provider: object,
        reason_fragment: str,
    ) -> None:
        output = root / "out"
        outcome = StrictCompositor(hybrid_provider=provider).compose(
            self._job(root), output, strict_audit=False,
        )
        strict = np.asarray(Image.open(output / "strict_base.png").convert("RGB"))
        report = json.loads((output / "audit.json").read_text(encoding="utf-8"))
        hybrid = report["hybrid_render"]
        self.assertEqual(hybrid["status"], "strict_fallback")
        self.assertIn(reason_fragment, hybrid["reason"])
        self.assertEqual(hybrid["identity_changed_pixels"], 0)
        self.assertEqual(hybrid["outside_changed_pixels"], 0)
        self.assertEqual(hybrid["max_outside_delta"], 0)
        self.assertTrue(np.array_equal(outcome.final, strict))

    def test_provider_exception_returns_byte_exact_strict_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self._assert_byte_exact_fallback(
                Path(directory), _FailingProvider(), "temporary provider failure",
            )

    def test_unregistered_candidate_returns_byte_exact_strict_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "wrong-aspect.png"
            save_png(candidate, np.full((80, 80, 3), 220, np.uint8))
            self._assert_byte_exact_fallback(
                root,
                _StaticCandidateProvider(candidate),
                "aspect ratio differs",
            )


if __name__ == "__main__":
    unittest.main()
