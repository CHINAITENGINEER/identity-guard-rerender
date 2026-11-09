from __future__ import annotations

import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
from PIL import Image, ImageOps

from .cutout import alpha_quality, trim_transparent
from .image_io import load_background, save_png
from .settings import get_public_settings, get_secret


DEFAULT_DOWNLOAD_MAX_BYTES = 256 * 1024 * 1024


class RunningHubError(RuntimeError):
    pass


def alpha_from_flat_background(
    source_rgb: np.ndarray,
    prediction_rgb: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Recover alpha from a foreground composited over a uniform background.

    Some audited cutout apps return a same-size RGB preview on white instead of
    an RGBA PNG. Because the original source RGB is known, alpha can still be
    solved per pixel from ``preview = a * source + (1-a) * background``. The
    preview's RGB is never copied into the product layer.
    """
    if (
        source_rgb.shape != prediction_rgb.shape
        or source_rgb.ndim != 3
        or source_rgb.shape[2] != 3
    ):
        raise RunningHubError("Flat-background alpha recovery requires aligned RGB images")
    height, width = prediction_rgb.shape[:2]
    edge = max(2, min(height, width) // 80)
    border = np.concatenate([
        prediction_rgb[:edge].reshape(-1, 3),
        prediction_rgb[-edge:].reshape(-1, 3),
        prediction_rgb[:, :edge].reshape(-1, 3),
        prediction_rgb[:, -edge:].reshape(-1, 3),
    ])
    background = np.median(border.astype(np.float32), axis=0)
    border_distance = np.linalg.norm(
        border.astype(np.float32) - background[None, :],
        axis=1,
    )
    uniform_share = float(np.mean(border_distance <= 10.0))
    if uniform_share < 0.86:
        raise RunningHubError(
            "RunningHub RGB output has no stable flat background for safe alpha recovery"
        )

    source = source_rgb.astype(np.float32)
    preview = prediction_rgb.astype(np.float32)
    delta_source = background[None, None, :] - source
    delta_preview = background[None, None, :] - preview
    denominator = np.sum(delta_source * delta_source, axis=2)
    numerator = np.sum(delta_source * delta_preview, axis=2)
    alpha = np.zeros((height, width), np.float32)
    identifiable = denominator >= 64.0
    alpha[identifiable] = numerator[identifiable] / denominator[identifiable]
    alpha = np.clip(alpha, 0.0, 1.0)
    preview_distance = np.linalg.norm(delta_preview, axis=2)
    alpha[preview_distance <= 1.5] = 0.0
    alpha = cv2.GaussianBlur(alpha, (0, 0), 0.55)
    alpha_u8 = np.clip(np.round(alpha * 255.0), 0, 255).astype(np.uint8)
    visible = alpha_u8 >= 13
    if not np.any(visible):
        raise RunningHubError("RunningHub RGB output produced an empty recovered alpha")
    identifiable_share = float(np.mean(identifiable[visible]))
    if identifiable_share < 0.92:
        raise RunningHubError(
            "RunningHub flat-background output is ambiguous for this product colour"
        )
    return alpha_u8, {
        "method": "aligned_flat_background_equation",
        "background_rgb": [float(value) for value in background],
        "border_uniform_share": uniform_share,
        "identifiable_foreground_share": identifiable_share,
        "rgb_policy": "prediction_rgb_discarded",
    }


@dataclass(frozen=True)
class RunningHubResult:
    task_id: str
    file_url: str
    file_type: str
    node_id: str


class RunningHubClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://www.runninghub.cn",
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        if not self.api_key:
            raise RunningHubError("RunningHub API key is not configured")
        if not self.base_url.startswith("https://"):
            raise RunningHubError("RunningHub base URL must use HTTPS")

    def _redact(self, value: object) -> str:
        return str(value).replace(self.api_key, "[redacted]")

    def _json_post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = requests.post(
                f"{self.base_url}{endpoint}",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={"apiKey": self.api_key, **payload},
                timeout=(15, self.timeout),
            )
            response.raise_for_status()
            envelope = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise RunningHubError(
                f"RunningHub request failed: {self._redact(exc)}"
            ) from exc
        if int(envelope.get("code", -1)) != 0:
            code = envelope.get("code")
            message = envelope.get("msg") or "unknown RunningHub error"
            raise RunningHubError(
                "RunningHub rejected the request "
                f"({code}): {self._redact(message)}"
            )
        return envelope

    def _json_get(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        try:
            response = requests.get(
                f"{self.base_url}{endpoint}",
                headers={"Authorization": f"Bearer {self.api_key}"},
                params={"apiKey": self.api_key, **params},
                timeout=(15, self.timeout),
            )
            response.raise_for_status()
            envelope = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise RunningHubError(
                f"RunningHub request failed: {self._redact(exc)}"
            ) from exc
        if int(envelope.get("code", -1)) != 0:
            code = envelope.get("code")
            message = envelope.get("msg") or "unknown RunningHub error"
            raise RunningHubError(
                "RunningHub rejected the request "
                f"({code}): {self._redact(message)}"
            )
        return envelope

    def test_configuration(
        self,
        application_id: str,
        mode: str = "ai_app",
    ) -> dict[str, Any]:
        if not application_id.strip():
            raise RunningHubError("Please configure the RunningHub cutout App/Workflow ID")
        if mode == "ai_app":
            envelope = self._json_get(
                "/api/webapp/apiCallDemo",
                {"webappId": application_id.strip()},
            )
        else:
            envelope = self._json_post(
                "/api/openapi/getJsonApiFormat",
                {"workflowId": application_id.strip()},
            )
        data = envelope.get("data")
        if isinstance(data, list):
            nodes = data
        elif isinstance(data, dict):
            nodes = data.get("nodeInfoList", [])
        else:
            nodes = []
        image_nodes = [
            {
                "nodeId": str(node.get("nodeId", "")),
                "fieldName": str(node.get("fieldName", "")),
                "nodeName": str(node.get("nodeName", "")),
                "description": str(node.get("description", "")),
            }
            for node in nodes
            if isinstance(node, dict)
            and (
                str(node.get("fieldType", "")).upper() == "IMAGE"
                or str(node.get("fieldName", "")).lower() == "image"
            )
        ]
        return {
            "ok": True,
            "node_count": len(nodes),
            "image_nodes": image_nodes,
            "suggested_input": image_nodes[0] if image_nodes else None,
        }

    def upload(self, path: str | Path) -> str:
        source = Path(path)
        try:
            with source.open("rb") as stream:
                return self._upload_stream(
                    source.name,
                    stream,
                    "application/octet-stream",
                )
        except (OSError, requests.RequestException, ValueError) as exc:
            raise RunningHubError(
                f"RunningHub upload failed: {self._redact(exc)}"
            ) from exc

    def _upload_stream(
        self,
        file_name: str,
        stream: Any,
        content_type: str,
    ) -> str:
        # This is the upload contract documented for AI applications and
        # ComfyUI workflows.  In contrast, /openapi/v2/media/upload/binary is
        # the Standard Model API media endpoint and is not the authoritative
        # input-resource route for /task/openapi/* jobs.
        endpoint = f"{self.base_url}/task/openapi/upload"
        response: requests.Response | None = None
        last_error: requests.RequestException | None = None
        # The international edge currently has two addresses and may reset a
        # fresh multipart connection with WinError 10054. Uploading the same
        # immutable resource again is safe; task creation is deliberately NOT
        # retried because a lost response there could duplicate a billed job.
        for attempt in range(4):
            try:
                if hasattr(stream, "seek"):
                    stream.seek(0)
                response = requests.post(
                    endpoint,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    data={"apiKey": self.api_key, "fileType": "input"},
                    files={"file": (file_name, stream, content_type)},
                    timeout=(15, max(self.timeout, 180.0)),
                )
                response.raise_for_status()
                break
            except (requests.ConnectionError, requests.ConnectTimeout) as exc:
                last_error = exc
                if attempt >= 3:
                    raise RunningHubError(
                        f"RunningHub upload failed after 4 transient attempts: {self._redact(exc)}"
                    ) from exc
                time.sleep(0.45 * (2 ** attempt))
            except requests.RequestException as exc:
                raise RunningHubError(
                    f"RunningHub upload failed: {self._redact(exc)}"
                ) from exc
        if response is None:
            # Defensive only: every loop path either assigns a response or
            # raises. Keep a redacted, deterministic failure if that changes.
            raise RunningHubError(
                "RunningHub upload failed without an HTTP response"
                + (f": {self._redact(last_error)}" if last_error else "")
            )
        try:
            envelope = response.json()
        except ValueError as exc:
            raise RunningHubError(
                f"RunningHub upload failed: {self._redact(exc)}"
            ) from exc
        if int(envelope.get("code", -1)) not in {0, 200}:
            message = (
                envelope.get("msg")
                or envelope.get("message")
                or "unknown error"
            )
            raise RunningHubError(
                "RunningHub upload rejected: "
                f"{self._redact(message)}"
            )
        data = envelope.get("data") or {}
        file_name = str(data.get("fileName") or data.get("filename") or "")
        if not file_name:
            raise RunningHubError("RunningHub upload returned no fileName")
        return file_name

    def test_upload(self) -> None:
        """Verify that the selected region accepts this key for real jobs."""
        probe = io.BytesIO()
        Image.new("RGB", (2, 2), (127, 127, 127)).save(probe, format="PNG")
        probe.seek(0)
        self._upload_stream("strict_compositor_probe.png", probe, "image/png")

    def create_task(
        self,
        application_id: str,
        node_info_list: list[dict[str, str]],
        mode: str = "ai_app",
    ) -> str:
        if mode == "ai_app":
            endpoint = "/task/openapi/ai-app/run"
            identifier = {"webappId": application_id}
        elif mode == "comfyui":
            endpoint = "/task/openapi/create"
            identifier = {"workflowId": application_id}
        else:
            raise RunningHubError("Unsupported RunningHub task mode")
        envelope = self._json_post(
            endpoint,
            {**identifier, "nodeInfoList": node_info_list},
        )
        task_id = str((envelope.get("data") or {}).get("taskId", ""))
        if not task_id:
            raise RunningHubError("RunningHub did not return a taskId")
        return task_id

    def outputs(self, task_id: str) -> list[dict[str, Any]] | None:
        response: requests.Response | None = None
        # Once a task id exists, asking for its current outputs is idempotent.
        # The international edge can intermittently reset otherwise healthy
        # polling connections (WinError 10054), so retry only transport-level
        # connection failures here.  Do not move this policy into _json_post:
        # create_task uses that helper and retrying a lost create response can
        # submit and bill the same job twice.
        for attempt in range(4):
            try:
                response = requests.post(
                    f"{self.base_url}/task/openapi/outputs",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={"apiKey": self.api_key, "taskId": task_id},
                    timeout=(15, self.timeout),
                )
                response.raise_for_status()
                break
            except (requests.ConnectionError, requests.ConnectTimeout) as exc:
                if attempt >= 3:
                    raise RunningHubError(
                        "RunningHub output query failed after 4 transient "
                        f"attempts: {self._redact(exc)}"
                    ) from exc
                time.sleep(0.45 * (2 ** attempt))
            except requests.RequestException as exc:
                # HTTP failures and other non-connection request failures are
                # authoritative and must never be hidden by polling retries.
                raise RunningHubError(
                    f"RunningHub output query failed: {self._redact(exc)}"
                ) from exc
        if response is None:
            raise RunningHubError(
                "RunningHub output query failed without an HTTP response"
            )
        try:
            envelope = response.json()
        except ValueError as exc:
            # A malformed successful response is not a transient connection
            # failure; fail once so the service issue remains visible.
            raise RunningHubError(
                f"RunningHub output query failed: {self._redact(exc)}"
            ) from exc
        code = int(envelope.get("code", -1))
        if code == 0 and isinstance(envelope.get("data"), list):
            return envelope["data"]
        if code in {804, 813}:
            return None
        message = envelope.get("msg") or "task failed"
        raise RunningHubError(
            f"RunningHub task failed ({code}): {self._redact(message)}"
        )

    def wait(
        self,
        task_id: str,
        output_node_id: str = "",
        timeout: float = 600.0,
        interval: float = 3.0,
    ) -> RunningHubResult:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            results = self.outputs(task_id)
            if results:
                candidates = [
                    item
                    for item in results
                    if not output_node_id
                    or str(item.get("nodeId", "")) == output_node_id
                ]
                if not candidates:
                    raise RunningHubError(
                        f"Task completed, but output node {output_node_id} was not returned"
                    )
                if not output_node_id and len(candidates) != 1:
                    nodes = sorted({str(item.get("nodeId", "")) for item in candidates})
                    raise RunningHubError(
                        "Task returned multiple image outputs; configure an explicit "
                        f"output node before production use (candidates={nodes})"
                    )
                item = candidates[0]
                file_url = str(item.get("fileUrl", ""))
                if not file_url.startswith("https://"):
                    raise RunningHubError("RunningHub returned an unsafe output URL")
                return RunningHubResult(
                    task_id=task_id,
                    file_url=file_url,
                    file_type=str(item.get("fileType", "")),
                    node_id=str(item.get("nodeId", "")),
                )
            time.sleep(interval)
        raise RunningHubError("RunningHub task timed out")

    def download(
        self,
        url: str,
        destination: str | Path,
        max_bytes: int | None = DEFAULT_DOWNLOAD_MAX_BYTES,
    ) -> Path:
        parsed = urlparse(url)
        if parsed.scheme != "https" or not parsed.hostname:
            raise RunningHubError("RunningHub output URL must use HTTPS")
        if max_bytes is not None and max_bytes <= 0:
            raise RunningHubError("RunningHub download max_bytes must be positive")
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(f"{target.name}.part")
        last_transport_error: OSError | requests.RequestException | None = None

        # The task is already complete and the output URL is immutable, so a
        # repeated GET cannot create or bill another cloud job.  The
        # international CDN occasionally resets a large streaming response on
        # Windows (WinError 10054); previously that discarded an otherwise
        # successful RunningHub matte and silently routed the product to a much
        # weaker local fallback.  Download into a sidecar and retry only
        # transport failures, then atomically publish the complete file.
        for attempt in range(4):
            partial.unlink(missing_ok=True)
            try:
                with requests.get(
                    url,
                    stream=True,
                    timeout=(15, max(self.timeout, 180.0)),
                ) as response:
                    response.raise_for_status()
                    content_length = response.headers.get("Content-Length")
                    if max_bytes is not None and content_length:
                        try:
                            declared_bytes = int(content_length)
                        except (TypeError, ValueError):
                            declared_bytes = -1
                        if declared_bytes > max_bytes:
                            raise RunningHubError(
                                "RunningHub output exceeds the configured download "
                                f"limit ({declared_bytes} > {max_bytes} bytes)"
                            )
                    downloaded_bytes = 0
                    with partial.open("wb") as stream:
                        for chunk in response.iter_content(1024 * 1024):
                            if chunk:
                                downloaded_bytes += len(chunk)
                                if (
                                    max_bytes is not None
                                    and downloaded_bytes > max_bytes
                                ):
                                    raise RunningHubError(
                                        "RunningHub output exceeds the configured "
                                        f"download limit ({max_bytes} bytes)"
                                    )
                                stream.write(chunk)
                partial.replace(target)
                return target
            except RunningHubError:
                partial.unlink(missing_ok=True)
                target.unlink(missing_ok=True)
                raise
            except (
                OSError,
                requests.ConnectionError,
                requests.ConnectTimeout,
                requests.ReadTimeout,
                requests.exceptions.ChunkedEncodingError,
            ) as exc:
                last_transport_error = exc
                partial.unlink(missing_ok=True)
                target.unlink(missing_ok=True)
                if attempt < 3:
                    time.sleep(0.45 * (2 ** attempt))
                    continue
            except requests.RequestException as exc:
                partial.unlink(missing_ok=True)
                target.unlink(missing_ok=True)
                raise RunningHubError(
                    f"RunningHub output download failed: {self._redact(exc)}"
                ) from exc

        raise RunningHubError(
            "RunningHub output download failed after 4 transient attempts: "
            f"{self._redact(last_transport_error)}"
        ) from last_transport_error


def runninghub_status() -> dict[str, Any]:
    settings = get_public_settings()
    return {
        "configured": bool(get_secret("runninghub_api_key")),
        "enabled": bool(settings["runninghub_enabled"]),
        "workflow_configured": bool(settings["runninghub_cutout_id"]),
        "mode": settings["runninghub_mode"],
        "base_url": settings["runninghub_base_url"],
    }


def configured_client() -> RunningHubClient:
    settings = get_public_settings()
    return RunningHubClient(
        get_secret("runninghub_api_key"),
        str(settings["runninghub_base_url"]),
    )


def run_configured_cutout(
    source: str | Path,
    destination: str | Path,
    timeout: float = 600.0,
) -> dict[str, Any]:
    """Run the configured matte workflow and keep only its alpha prediction.

    RunningHub never owns the product RGB in the returned deterministic layer.
    Even when the workflow returns a foreground image, this function discards
    its colour channels and rebuilds RGBA from the source pixels.
    """
    settings = get_public_settings()
    if not settings["runninghub_enabled"]:
        raise RunningHubError("RunningHub cutout is disabled")
    application_id = str(settings["runninghub_cutout_id"])
    input_node = str(settings["runninghub_input_node_id"])
    input_field = str(settings["runninghub_input_field_name"] or "image")
    if not application_id or not input_node:
        raise RunningHubError(
            "RunningHub cutout requires an App/Workflow ID and input node ID"
        )
    client = configured_client()
    uploaded_name = client.upload(source)
    task_id = client.create_task(
        application_id,
        [{
            "nodeId": input_node,
            "fieldName": input_field,
            "fieldValue": uploaded_name,
        }],
        mode=str(settings["runninghub_mode"]),
    )
    result = client.wait(
        task_id,
        output_node_id=str(settings["runninghub_output_node_id"]),
        timeout=timeout,
    )
    suffix = str(result.file_type).lower().lstrip(".")
    if suffix not in {"png", "jpg", "jpeg", "webp", "tif", "tiff"}:
        suffix = "png"
    download = (
        Path(destination).parent
        / f"runninghub_{task_id}.{suffix}"
    )
    client.download(result.file_url, download)

    source_rgb = load_background(source)
    with Image.open(download) as raw:
        prediction = np.asarray(ImageOps.exif_transpose(raw))
    source_aspect = source_rgb.shape[1] / max(1, source_rgb.shape[0])
    alpha_aspect = prediction.shape[1] / max(1, prediction.shape[0])
    if abs(source_aspect / max(1e-6, alpha_aspect) - 1.0) > 0.015:
        raise RunningHubError(
            "RunningHub matte changed the source aspect ratio; unsafe alpha rejected"
        )
    if prediction.shape[:2] != source_rgb.shape[:2]:
        prediction = cv2.resize(
            prediction,
            (source_rgb.shape[1], source_rgb.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    alpha_details: dict[str, Any]
    if prediction.ndim == 2:
        alpha = prediction.astype(np.uint8)
        alpha_details = {"method": "grayscale_mask"}
    elif prediction.ndim == 3 and prediction.shape[2] >= 4:
        alpha = prediction[..., 3].astype(np.uint8)
        alpha_details = {"method": "rgba_alpha_channel"}
    elif prediction.ndim == 3 and prediction.shape[2] == 1:
        alpha = prediction[..., 0].astype(np.uint8)
        alpha_details = {"method": "single_channel_mask"}
    elif prediction.ndim == 3 and prediction.shape[2] == 3:
        alpha, alpha_details = alpha_from_flat_background(
            source_rgb,
            prediction[..., :3].astype(np.uint8),
        )
    else:
        raise RunningHubError(
            "The configured cutout workflow returned an unsupported image format"
        )
    # Tighten only the uncertain fringe.  Flat-background workflows often
    # return a two-to-four pixel low-alpha glow that is numerically valid but
    # visible as a pale halo after placement on a darker desk.  Smoothstep
    # preserves the 0/255 interior and all source RGB while narrowing that
    # uncertain transition deterministically.
    normalized = np.clip((alpha.astype(np.float32) - 12.0) / 231.0, 0.0, 1.0)
    normalized = normalized * normalized * (3.0 - 2.0 * normalized)
    alpha = np.rint(normalized * 255.0).astype(np.uint8)
    rgba = np.dstack([source_rgb, alpha])
    rgba = trim_transparent(rgba, padding=5)
    quality = alpha_quality(rgba)
    if quality["verdict"] == "reject":
        raise RunningHubError(
            "RunningHub matte failed deterministic alpha-quality gates"
        )
    # This is an automatic confidence score, not labelled ground truth. Never
    # publish the structural heuristic's theoretical 100 as "perfect".
    structure_score = float(quality["score"])
    evidence = min(
        float(alpha_details.get("border_uniform_share", 1.0)),
        float(alpha_details.get("identifiable_foreground_share", 1.0)),
    )
    completeness_score = float(np.clip(88.0 + 8.0 * evidence, 0.0, 96.0))
    quality["structure_score"] = structure_score
    quality["completeness_score"] = completeness_score
    quality["score"] = min(structure_score, completeness_score)
    quality["score_kind"] = "automatic_confidence_not_ground_truth"
    save_png(destination, rgba)
    return {
        "provider": "runninghub",
        "model": "configured-runninghub-workflow",
        "task_id": task_id,
        "output_node_id": result.node_id,
        "quality": quality,
        "rgb_policy": "source_pixels_only",
        "alpha_source": alpha_details,
    }
