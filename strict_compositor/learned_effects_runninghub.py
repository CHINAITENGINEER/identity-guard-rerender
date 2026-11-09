from __future__ import annotations

"""RunningHub adapter for the restricted scene-level learned-effects contract.

The remote workflow never owns final RGB.  V1 receives diagnostic scene
evidence and may return exactly one receiver-space grayscale shadow map.  The
strict learned-effects boundary validates and clips that map again before it
can replace deterministic cast shadow.
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from .image_io import save_png
from .learned_effects import (
    LearnedEffectBatchPrediction,
    LearnedEffectContractError,
    LearnedEffectPrediction,
    LearnedEffectRequest,
)
from .runninghub import RunningHubError, RunningHubResult, configured_client
from .settings import get_public_settings


_IMAGE_SUFFIXES = {"png"}
_MIN_OUTPUT_EDGE = 8
_MIN_OUTPUT_SCALE = 0.125
_ASPECT_RATIO_TOLERANCE = 0.01


class RunningHubLearnedEffectProvider:
    """One-task scene provider using explicitly mapped workflow nodes.

    The standard workflow input contract is:

    * ``source``: shadowless background plus placed products;
    * ``product_mask``: zero for background, stable non-zero ID per product;
    * ``geometry``: RGB channels encode depth, support and protected masks;
    * ``normal``: required camera-space normal encoded from ``[-1,1]``;
    * ``background``: optional pristine background;
    * ``prompt``: required JSON metadata and hard output instructions.

    A required PNG shadow output is decoded as a single-channel alpha.  V1
    explicitly forbids gain and specular output nodes.  The resulting scene
    map is duplicated into the per-product protocol so the receiver sees one
    shared, coherent shadow solution from a single task.
    """

    name = "runninghub_learned_effects"

    def __init__(
        self,
        timeout: float = 900.0,
        artifact_root: str | Path | None = None,
    ) -> None:
        self.timeout = float(timeout)
        self.artifact_root = (
            Path(artifact_root).resolve()
            if artifact_root is not None
            else (Path.cwd().resolve() / "ui_workspace" / "learned_effects")
        )

    def predict(
        self,
        request: LearnedEffectRequest,
    ) -> LearnedEffectBatchPrediction:
        settings = get_public_settings()
        _validate_workflow_settings(settings)
        if request.depth is None or request.normal is None:
            raise LearnedEffectContractError(
                "RunningHub learned shadow requires validated depth and normal evidence"
            )
        if not request.products:
            raise LearnedEffectContractError(
                "RunningHub learned shadow requires at least one product"
            )

        run = self.artifact_root / f"scene_{time.time_ns()}"
        run.mkdir(parents=True, exist_ok=False)
        inputs, manifest = _write_scene_inputs(request, run)
        client = configured_client()
        nodes: list[dict[str, str]] = []

        def add_image(setting_prefix: str, artifact_name: str, required: bool) -> None:
            node_id = str(settings.get(f"{setting_prefix}_node_id") or "").strip()
            if not node_id:
                if required:
                    raise RunningHubError(
                        f"RunningHub learned workflow is missing {setting_prefix}_node_id"
                    )
                return
            uploaded = client.upload(inputs[artifact_name])
            nodes.append({
                "nodeId": node_id,
                "fieldName": str(
                    settings.get(f"{setting_prefix}_field_name") or "image"
                ).strip(),
                "fieldValue": uploaded,
            })

        add_image("runninghub_learned_source", "source", True)
        add_image("runninghub_learned_product_mask", "product_mask", True)
        add_image("runninghub_learned_geometry", "geometry", True)
        add_image("runninghub_learned_normal", "normal", True)
        add_image("runninghub_learned_background", "background", False)

        prompt_node = str(
            settings.get("runninghub_learned_prompt_node_id") or ""
        ).strip()
        nodes.append({
            "nodeId": prompt_node,
            "fieldName": str(
                settings.get("runninghub_learned_prompt_field_name") or "prompt"
            ).strip(),
            "fieldValue": _workflow_prompt(manifest),
        })

        app_id = str(settings["runninghub_learned_app_id"]).strip()
        mode = str(settings["runninghub_learned_mode"]).strip()
        task_id = client.create_task(app_id, nodes, mode=mode)
        shadow_result = client.wait(
            task_id,
            output_node_id=str(
                settings["runninghub_learned_shadow_output_node_id"]
            ).strip(),
            timeout=self.timeout,
        )
        shadow_path = _download_output(client, shadow_result, run, "shadow")
        shadow = _decode_shadow(
            shadow_path,
            request.background_rgb.shape[:2],
            str(settings["runninghub_learned_shadow_encoding"]),
        )

        shared_metadata = {
            "provider": self.name,
            "task_id": task_id,
            "mode": mode,
            "application_id": app_id,
            "scene_shared_maps": True,
            "shadow_encoding": str(
                settings["runninghub_learned_shadow_encoding"]
            ),
            "input_manifest": str(inputs["manifest"]),
            "input_hashes": {
                name: _file_sha256(path)
                for name, path in inputs.items()
            },
            "output_artifacts": {
                "shadow": str(shadow_path),
            },
            "output_hashes": {
                "shadow_sha256": _file_sha256(shadow_path),
            },
            "output_nodes": {
                "shadow": shadow_result.node_id,
            },
        }
        return LearnedEffectBatchPrediction(
            products={
                context.product_id: LearnedEffectPrediction(
                    shadow_alpha=shadow,
                    illumination_gain=None,
                    specular_residual=None,
                    metadata={
                        "scene_shared_maps": True,
                        "task_id": task_id,
                    },
                )
                for context in request.products
            },
            metadata=shared_metadata,
        )


def _status_from_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        settings.get("runninghub_learned_app_id"),
        settings.get("runninghub_learned_source_node_id"),
        settings.get("runninghub_learned_product_mask_node_id"),
        settings.get("runninghub_learned_geometry_node_id"),
        settings.get("runninghub_learned_normal_node_id"),
        settings.get("runninghub_learned_prompt_node_id"),
        settings.get("runninghub_learned_shadow_output_node_id"),
    )
    enabled = bool(settings.get("runninghub_learned_enabled"))
    secret_configured = bool(settings.get("runninghub_configured"))
    forbidden_v2_outputs = bool(
        str(settings.get("runninghub_learned_gain_output_node_id") or "").strip()
        or str(
            settings.get("runninghub_learned_specular_output_node_id") or ""
        ).strip()
    )
    mode_valid = str(settings.get("runninghub_learned_mode") or "") in {
        "ai_app",
        "comfyui",
    }
    encoding_valid = str(
        settings.get("runninghub_learned_shadow_encoding") or ""
    ) in {
        "white_is_shadow",
        "black_is_shadow",
    }
    workflow_configured = (
        all(bool(str(value or "").strip()) for value in required)
        and not forbidden_v2_outputs
        and mode_valid
        and encoding_valid
    )
    return {
        "enabled": enabled,
        "secret_configured": secret_configured,
        "workflow_configured": workflow_configured,
        "ready": enabled and secret_configured and workflow_configured,
        "mode": str(settings.get("runninghub_learned_mode") or "comfyui"),
        "provider": "runninghub",
        "contract": "scene-shadow-alpha-v1",
    }


def runninghub_learned_effects_status() -> dict[str, Any]:
    # Read settings exactly once so enabled/configured/ready cannot describe
    # different snapshots during a concurrent save.
    return _status_from_settings(get_public_settings())


def _validate_workflow_settings(settings: Mapping[str, Any]) -> None:
    if not bool(settings.get("runninghub_learned_enabled")):
        raise RunningHubError("RunningHub learned effects are disabled")
    required = (
        settings.get("runninghub_learned_app_id"),
        settings.get("runninghub_learned_source_node_id"),
        settings.get("runninghub_learned_product_mask_node_id"),
        settings.get("runninghub_learned_geometry_node_id"),
        settings.get("runninghub_learned_normal_node_id"),
        settings.get("runninghub_learned_prompt_node_id"),
        settings.get("runninghub_learned_shadow_output_node_id"),
    )
    if not all(bool(str(value or "").strip()) for value in required):
        raise RunningHubError(
            "RunningHub learned workflow V1 requires app, source, product-mask, "
            "geometry, normal, prompt and shadow-output node IDs"
        )
    if (
        str(settings.get("runninghub_learned_gain_output_node_id") or "").strip()
        or str(
            settings.get("runninghub_learned_specular_output_node_id") or ""
        ).strip()
    ):
        raise RunningHubError(
            "RunningHub learned workflow V1 forbids gain/specular output nodes"
        )
    if not bool(settings.get("runninghub_configured")):
        raise RunningHubError("RunningHub API key is not configured")
    if str(settings.get("runninghub_learned_mode")) not in {"ai_app", "comfyui"}:
        raise RunningHubError("Unsupported RunningHub learned workflow mode")
    if str(settings.get("runninghub_learned_shadow_encoding")) not in {
        "white_is_shadow", "black_is_shadow",
    }:
        raise RunningHubError("Unsupported RunningHub shadow-map encoding")


def _write_scene_inputs(
    request: LearnedEffectRequest,
    run: Path,
) -> tuple[dict[str, Path], dict[str, Any]]:
    height, width = request.background_rgb.shape[:2]
    source = np.asarray(request.background_rgb, dtype=np.uint8).copy()
    product_mask = np.zeros((height, width), np.uint8)
    product_values: dict[str, int] = {}
    product_count = len(request.products)
    for index, context in enumerate(request.products):
        alpha = np.asarray(context.product_alpha, dtype=np.uint8)
        if alpha.shape != (height, width):
            raise LearnedEffectContractError(
                "product alpha does not match RunningHub scene canvas"
            )
        rgb = np.asarray(context.product_rgb, dtype=np.uint8)
        if rgb.shape != (height, width, 3):
            raise LearnedEffectContractError(
                "product RGB does not match RunningHub scene canvas"
            )
        weight = alpha.astype(np.float32)[..., None] / 255.0
        source = np.clip(
            np.round(rgb.astype(np.float32) * weight + source * (1.0 - weight)),
            0,
            255,
        ).astype(np.uint8)
        value = int(round((index + 1) * 255.0 / (product_count + 1)))
        product_mask[alpha > 0] = value
        product_values[context.product_id] = value

    depth = np.asarray(request.depth, dtype=np.float32)
    normal = np.asarray(request.normal, dtype=np.float32)
    if depth.shape != (height, width) or normal.shape != (height, width, 3):
        raise LearnedEffectContractError(
            "RunningHub scene geometry must already match the final canvas"
        )
    geometry = np.stack([
        np.clip(np.round(depth * 255.0), 0, 255).astype(np.uint8),
        np.where(request.support_mask, 255, 0).astype(np.uint8),
        np.where(request.protected_mask, 255, 0).astype(np.uint8),
    ], axis=2)
    normal_png = np.clip(
        np.round((normal * 0.5 + 0.5) * 255.0), 0, 255,
    ).astype(np.uint8)

    paths = {
        "source": run / "shadowless_scene.png",
        "product_mask": run / "product_id_mask.png",
        "geometry": run / "geometry_depth_support_protected.png",
        "normal": run / "normal_xyz.png",
        "background": run / "background_original.png",
        "manifest": run / "request_manifest.json",
    }
    save_png(paths["source"], source)
    save_png(paths["product_mask"], product_mask)
    save_png(paths["geometry"], geometry)
    save_png(paths["normal"], normal_png)
    save_png(paths["background"], np.asarray(request.background_rgb, np.uint8))
    manifest = {
        "schema_version": "runninghub-learned-scene-v1",
        "canvas": [width, height],
        "encodings": {
            "product_mask": product_values,
            "geometry_rgb": {
                "r": "relative_depth_0_to_1",
                "g": "support_mask",
                "b": "protected_mask",
            },
            "normal_rgb": "xyz_minus1_to1_mapped_to_0to255",
        },
        "lighting": {
            "shadow_direction": list(request.lighting.direction),
            "shadow_length_ratio": request.lighting.shadow_length_ratio,
            "shadow_opacity": request.lighting.shadow_opacity,
            "shadow_softness_ratio": request.lighting.shadow_softness_ratio,
            "cast_shadow_mode": request.lighting.cast_shadow_mode,
            "surface_material": request.lighting.surface_material,
            "surface_reflectivity": request.lighting.surface_reflectivity,
            "direct_light_fraction": request.lighting.direct_light_fraction,
        },
        "products": [
            {
                "product_id": context.product_id,
                "material_mode": context.material_mode,
                "metadata": dict(context.metadata),
            }
            for context in request.products
        ],
    }
    paths["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return paths, manifest


def _workflow_prompt(manifest: Mapping[str, Any]) -> str:
    return (
        "Predict receiver-space cast-shadow alpha for all placed products as "
        "one coherent scene. Use the supplied light direction, depth, support, "
        "protected region, normals and product-ID mask. Output only the "
        "configured map images: no final RGB, no product repainting, no new "
        "objects, and no background edit. Shadow must remain continuous from "
        "each contact point and share one light source. Metadata JSON: "
        + json.dumps(manifest, ensure_ascii=False, separators=(",", ":"))
    )[:12000]


def _download_output(
    client: Any,
    result: RunningHubResult,
    run: Path,
    name: str,
) -> Path:
    suffix = str(result.file_type).lower().lstrip(".")
    if suffix not in _IMAGE_SUFFIXES:
        raise LearnedEffectContractError(
            f"RunningHub {name} output must be a grayscale PNG"
        )
    destination = run / f"{name}_output.{suffix}"
    return Path(client.download(result.file_url, destination)).resolve()


def _read_map_image(path: Path) -> tuple[np.ndarray, float]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise LearnedEffectContractError(
            f"cannot decode RunningHub effect map: {path.name}"
        )
    if image.dtype == np.uint8:
        scale = 255.0
    elif image.dtype == np.uint16:
        scale = 65535.0
    else:
        raise LearnedEffectContractError(
            "RunningHub shadow output must be an 8-bit or 16-bit grayscale PNG"
        )
    return image, scale


def _resize_map(
    array: np.ndarray,
    shape: tuple[int, int],
    *,
    name: str,
) -> np.ndarray:
    source_height, source_width = array.shape[:2]
    target_height, target_width = shape
    if source_height <= 1 or source_width <= 1:
        raise LearnedEffectContractError(
            f"RunningHub {name} output is too small: "
            f"got {source_width}x{source_height}; 1-pixel dimensions are forbidden"
        )
    if array.shape[:2] == shape:
        return array
    minimum_height = max(
        _MIN_OUTPUT_EDGE,
        int(np.ceil(target_height * _MIN_OUTPUT_SCALE)),
    )
    minimum_width = max(
        _MIN_OUTPUT_EDGE,
        int(np.ceil(target_width * _MIN_OUTPUT_SCALE)),
    )
    if source_height < minimum_height or source_width < minimum_width:
        raise LearnedEffectContractError(
            f"RunningHub {name} output is too small for explicit scaling: "
            f"got {source_width}x{source_height}, minimum is "
            f"{minimum_width}x{minimum_height}"
        )
    aspect_error = abs(
        (source_width / source_height) - (target_width / target_height)
    ) / (target_width / target_height)
    if aspect_error > _ASPECT_RATIO_TOLERANCE:
        raise LearnedEffectContractError(
            f"RunningHub {name} output aspect ratio differs from the scene"
        )
    interpolation = (
        cv2.INTER_AREA
        if source_height > target_height or source_width > target_width
        else cv2.INTER_LINEAR
    )
    return cv2.resize(
        array,
        (target_width, target_height),
        interpolation=interpolation,
    )


def _decode_shadow(
    path: Path,
    shape: tuple[int, int],
    encoding: str,
) -> np.ndarray:
    image, scale = _read_map_image(path)
    if image.ndim != 2:
        raise LearnedEffectContractError(
            "shadow output must be a single-channel grayscale PNG; "
            "arbitrary RGB is forbidden"
        )
    normalized = _resize_map(
        image.astype(np.float32) / scale,
        shape,
        name="shadow",
    )
    if encoding == "black_is_shadow":
        normalized = 1.0 - normalized
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
