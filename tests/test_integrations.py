from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import cv2
import numpy as np
import requests

from strict_compositor.runninghub import (
    RunningHubClient,
    RunningHubError,
    alpha_from_flat_background,
)
from strict_compositor import settings


class IntegrationSettingsTests(unittest.TestCase):
    def test_dpapi_round_trip_does_not_store_plaintext(self) -> None:
        if settings.os.name != "nt":
            self.skipTest("Windows DPAPI test")
        encrypted = settings.protect_secret("temporary-unit-test-secret")
        self.assertNotIn("temporary-unit-test-secret", encrypted)
        self.assertEqual(
            settings.unprotect_secret(encrypted),
            "temporary-unit-test-secret",
        )

    def test_public_settings_never_return_api_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "integrations.json"
            with (
                patch.object(settings, "SETTINGS_PATH", path),
                patch.object(settings, "PRIVATE_SETTINGS_DIR", path.parent),
                patch.dict("os.environ", {}, clear=True),
            ):
                result = settings.save_integrations({
                    "minimax_api_key": "mini-secret",
                    "runninghub_api_key": "rh-secret",
                    "runninghub_enabled": True,
                    "runninghub_cutout_id": "123",
                    "runninghub_input_node_id": "1",
                    "minimax_network_mode": "direct",
                })
                self.assertTrue(result["minimax_configured"])
                self.assertTrue(result["runninghub_configured"])
                self.assertEqual(result["minimax_network_mode"], "direct")
                self.assertNotIn("minimax_api_key", result)
                self.assertNotIn("runninghub_api_key", result)
                raw = path.read_text(encoding="utf-8")
                self.assertNotIn("mini-secret", raw)
                self.assertNotIn("rh-secret", raw)


class RunningHubClientTests(unittest.TestCase):
    def test_flat_background_preview_recovers_alpha_without_using_preview_rgb(
        self,
    ) -> None:
        height, width = 96, 128
        source = np.full((height, width, 3), (45, 80, 115), np.uint8)
        source[22:76, 31:101] = (155, 62, 28)
        alpha = np.zeros((height, width), np.uint8)
        cv2.rectangle(alpha, (31, 22), (100, 75), 255, -1)
        alpha = cv2.GaussianBlur(alpha, (0, 0), 1.1)
        weight = alpha.astype(np.float32)[..., None] / 255.0
        preview = np.clip(
            source.astype(np.float32) * weight + 255.0 * (1.0 - weight),
            0,
            255,
        ).astype(np.uint8)
        recovered, details = alpha_from_flat_background(source, preview)
        self.assertLess(float(np.mean(np.abs(recovered.astype(int) - alpha))), 2.0)
        self.assertEqual(
            details["rgb_policy"],
            "prediction_rgb_discarded",
        )

    @patch("strict_compositor.runninghub.requests.post")
    def test_upload_uses_documented_task_resource_endpoint_and_bearer_key(
        self,
        post: Mock,
    ) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "code": 0,
            "message": "success",
            "data": {"fileName": "openapi/probe.png"},
        }
        post.return_value = response
        client = RunningHubClient("unit-test-key")
        client.test_upload()
        self.assertTrue(
            post.call_args.args[0].endswith(
                "/task/openapi/upload"
            )
        )
        self.assertEqual(
            post.call_args.kwargs["headers"]["Authorization"],
            "Bearer unit-test-key",
        )
        self.assertEqual(post.call_args.kwargs["data"]["apiKey"], "unit-test-key")
        self.assertEqual(post.call_args.kwargs["data"]["fileType"], "input")

    @patch("strict_compositor.runninghub.time.sleep")
    @patch("strict_compositor.runninghub.requests.post")
    def test_upload_rewinds_and_retries_connection_reset_only(
        self,
        post: Mock,
        sleep: Mock,
    ) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "code": 0,
            "msg": "success",
            "data": {"fileName": "api/probe.png"},
        }
        observed_payloads: list[bytes] = []

        def request(*args, **kwargs):
            stream = kwargs["files"]["file"][1]
            observed_payloads.append(stream.read())
            if len(observed_payloads) == 1:
                raise requests.ConnectionError(
                    "connection reset by peer (10054)"
                )
            return response

        post.side_effect = request
        client = RunningHubClient("unit-test-key")
        client.test_upload()
        self.assertEqual(post.call_count, 2)
        self.assertEqual(observed_payloads[0], observed_payloads[1])
        self.assertGreater(len(observed_payloads[0]), 0)
        sleep.assert_called_once_with(0.45)

    @patch("strict_compositor.runninghub.requests.get")
    def test_ai_app_probe_discovers_image_node(self, get: Mock) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "code": 0,
            "msg": "success",
            "data": {
                "nodeInfoList": [
                    {
                        "nodeId": "39",
                        "nodeName": "LoadImage",
                        "fieldName": "image",
                        "fieldType": "IMAGE",
                        "description": "Upload Image",
                    }
                ]
            },
        }
        get.return_value = response
        client = RunningHubClient("unit-test-key")
        result = client.test_configuration("app-123")
        self.assertEqual(result["suggested_input"]["nodeId"], "39")
        self.assertTrue(
            get.call_args.args[0].endswith("/api/webapp/apiCallDemo")
        )

    @patch("strict_compositor.runninghub.requests.post")
    def test_create_ai_app_task_uses_documented_endpoint(self, post: Mock) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "code": 0,
            "msg": "success",
            "data": {"taskId": "task-123"},
        }
        post.return_value = response
        client = RunningHubClient("unit-test-key")
        task_id = client.create_task(
            "app-123",
            [{"nodeId": "1", "fieldName": "image", "fieldValue": "api/x.png"}],
        )
        self.assertEqual(task_id, "task-123")
        self.assertTrue(post.call_args.args[0].endswith("/task/openapi/ai-app/run"))
        self.assertEqual(post.call_args.kwargs["json"]["webappId"], "app-123")

    @patch("strict_compositor.runninghub.requests.post")
    def test_queued_output_is_not_reported_as_success(self, post: Mock) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"code": 813, "msg": "queued", "data": None}
        post.return_value = response
        client = RunningHubClient("unit-test-key")
        self.assertIsNone(client.outputs("task-123"))

    @patch("strict_compositor.runninghub.time.sleep")
    @patch("strict_compositor.runninghub.requests.post")
    def test_output_query_retries_connection_reset_only(
        self,
        post: Mock,
        sleep: Mock,
    ) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "code": 0,
            "msg": "success",
            "data": [{
                "nodeId": "9",
                "fileUrl": "https://example.test/result.png",
                "fileType": "png",
            }],
        }
        post.side_effect = [
            requests.ConnectionError("connection reset by peer (10054)"),
            requests.ConnectTimeout("connect timed out"),
            response,
        ]
        client = RunningHubClient("unit-test-key")
        outputs = client.outputs("task-123")
        self.assertEqual(outputs, response.json.return_value["data"])
        self.assertEqual(post.call_count, 3)
        self.assertEqual(
            [call.args[0] for call in sleep.call_args_list],
            [0.45, 0.9],
        )

    @patch("strict_compositor.runninghub.time.sleep")
    @patch("strict_compositor.runninghub.requests.post")
    def test_output_query_does_not_retry_http_or_business_errors(
        self,
        post: Mock,
        sleep: Mock,
    ) -> None:
        http_failure = Mock()
        http_failure.raise_for_status.side_effect = requests.HTTPError(
            "503 service unavailable"
        )
        post.return_value = http_failure
        client = RunningHubClient("unit-test-key")
        with self.assertRaisesRegex(RunningHubError, "503 service unavailable"):
            client.outputs("task-http-error")
        self.assertEqual(post.call_count, 1)
        sleep.assert_not_called()

        post.reset_mock()
        business_failure = Mock()
        business_failure.raise_for_status.return_value = None
        business_failure.json.return_value = {
            "code": 805,
            "msg": "failed",
            "data": {},
        }
        post.return_value = business_failure
        with self.assertRaisesRegex(RunningHubError, r"task failed \(805\)"):
            client.outputs("task-business-error")
        self.assertEqual(post.call_count, 1)
        sleep.assert_not_called()

    @patch("strict_compositor.runninghub.requests.post")
    def test_create_task_transport_failure_is_never_retried(
        self,
        post: Mock,
    ) -> None:
        post.side_effect = requests.ConnectionError(
            "connection reset after submit (10054)"
        )
        client = RunningHubClient("unit-test-key")
        with self.assertRaises(RunningHubError):
            client.create_task(
                "app-123",
                [{
                    "nodeId": "1",
                    "fieldName": "image",
                    "fieldValue": "api/x.png",
                }],
            )
        self.assertEqual(post.call_count, 1)

    @patch("strict_compositor.runninghub.requests.post")
    def test_failed_task_raises_instead_of_becoming_visual_pass(self, post: Mock) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"code": 805, "msg": "failed", "data": {}}
        post.return_value = response
        client = RunningHubClient("unit-test-key")
        with self.assertRaises(RunningHubError):
            client.outputs("task-123")

    @patch("strict_compositor.runninghub.requests.post")
    def test_json_post_redacts_key_echoed_by_server(self, post: Mock) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "code": 401,
            "msg": "invalid credential unit-test-key",
            "data": {},
        }
        post.return_value = response
        client = RunningHubClient("unit-test-key")
        with self.assertRaises(RunningHubError) as raised:
            client.create_task("app-123", [])
        message = str(raised.exception)
        self.assertNotIn("unit-test-key", message)
        self.assertIn("[redacted]", message)

    @patch("strict_compositor.runninghub.requests.get")
    def test_json_get_redacts_key_echoed_by_server(self, get: Mock) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "code": 401,
            "msg": "invalid credential unit-test-key",
            "data": {},
        }
        get.return_value = response
        client = RunningHubClient("unit-test-key")
        with self.assertRaises(RunningHubError) as raised:
            client.test_configuration("app-123")
        message = str(raised.exception)
        self.assertNotIn("unit-test-key", message)
        self.assertIn("[redacted]", message)

    @patch("strict_compositor.runninghub.requests.post")
    def test_upload_redacts_key_echoed_by_server(self, post: Mock) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "code": 401,
            "msg": "invalid credential unit-test-key",
            "data": {},
        }
        post.return_value = response
        client = RunningHubClient("unit-test-key")
        with self.assertRaises(RunningHubError) as raised:
            client.test_upload()
        message = str(raised.exception)
        self.assertNotIn("unit-test-key", message)
        self.assertIn("[redacted]", message)

    @patch("strict_compositor.runninghub.requests.post")
    def test_output_query_redacts_key_echoed_by_server(self, post: Mock) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "code": 805,
            "msg": "task rejected for unit-test-key",
            "data": {},
        }
        post.return_value = response
        client = RunningHubClient("unit-test-key")
        with self.assertRaises(RunningHubError) as raised:
            client.outputs("task-123")
        message = str(raised.exception)
        self.assertNotIn("unit-test-key", message)
        self.assertIn("[redacted]", message)

    @patch("strict_compositor.runninghub.requests.get")
    def test_download_rejects_declared_output_above_limit(
        self,
        get: Mock,
    ) -> None:
        response = MagicMock()
        response.__enter__.return_value = response
        response.raise_for_status.return_value = None
        response.headers = {"Content-Length": "9"}
        response.iter_content.return_value = [b"not-called"]
        get.return_value = response
        client = RunningHubClient("unit-test-key")
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "output.bin"
            with self.assertRaisesRegex(RunningHubError, "download limit"):
                client.download(
                    "https://example.test/output.bin",
                    destination,
                    max_bytes=8,
                )
            self.assertFalse(destination.exists())
        response.iter_content.assert_not_called()

    @patch("strict_compositor.runninghub.requests.get")
    def test_download_rejects_streamed_output_above_limit_and_removes_partial(
        self,
        get: Mock,
    ) -> None:
        response = MagicMock()
        response.__enter__.return_value = response
        response.raise_for_status.return_value = None
        response.headers = {}
        response.iter_content.return_value = [b"1234", b"5678"]
        get.return_value = response
        client = RunningHubClient("unit-test-key")
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "output.bin"
            with self.assertRaisesRegex(RunningHubError, "download limit"):
                client.download(
                    "https://example.test/output.bin",
                    destination,
                    max_bytes=7,
                )
            self.assertFalse(destination.exists())

    @patch("strict_compositor.runninghub.requests.get")
    def test_download_keeps_existing_call_shape_and_writes_under_limit(
        self,
        get: Mock,
    ) -> None:
        response = MagicMock()
        response.__enter__.return_value = response
        response.raise_for_status.return_value = None
        response.headers = {}
        response.iter_content.return_value = [b"1234", b"5678"]
        get.return_value = response
        client = RunningHubClient("unit-test-key")
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "output.bin"
            result = client.download(
                "https://example.test/output.bin",
                destination,
            )
            self.assertEqual(result, destination)
            self.assertEqual(destination.read_bytes(), b"12345678")

    @patch("strict_compositor.runninghub.time.sleep", return_value=None)
    @patch("strict_compositor.runninghub.requests.get")
    def test_download_retries_transient_stream_reset_without_partial_publish(
        self,
        get: Mock,
        _sleep: Mock,
    ) -> None:
        broken = MagicMock()
        broken.__enter__.return_value = broken
        broken.raise_for_status.return_value = None
        broken.headers = {}
        broken.iter_content.side_effect = requests.ConnectionError("reset")
        good = MagicMock()
        good.__enter__.return_value = good
        good.raise_for_status.return_value = None
        good.headers = {}
        good.iter_content.return_value = [b"complete"]
        get.side_effect = [broken, good]
        client = RunningHubClient("unit-test-key")

        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "output.bin"
            result = client.download(
                "https://example.test/output.bin",
                destination,
            )
            self.assertEqual(result, destination)
            self.assertEqual(destination.read_bytes(), b"complete")
            self.assertFalse((Path(temporary) / "output.bin.part").exists())

        self.assertEqual(get.call_count, 2)

    def test_multiple_outputs_require_explicit_output_node(self) -> None:
        client = RunningHubClient("unit-test-key")
        results = [
            {"nodeId": "7", "fileUrl": "https://example.test/a.png", "fileType": "png"},
            {"nodeId": "9", "fileUrl": "https://example.test/b.png", "fileType": "png"},
        ]
        with patch.object(client, "outputs", return_value=results):
            with self.assertRaisesRegex(RunningHubError, "explicit output node"):
                client.wait("task-123", interval=0)

            selected = client.wait("task-123", output_node_id="9", interval=0)
            self.assertEqual(selected.node_id, "9")
            self.assertEqual(selected.file_url, "https://example.test/b.png")


if __name__ == "__main__":
    unittest.main()
