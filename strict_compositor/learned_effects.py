from __future__ import annotations

"""Strict boundary for replaceable learned lighting effects.

The compositor deliberately does not accept a model-produced product image.
A provider may predict only three narrowly defined signals:

* a receiver-space cast-shadow alpha;
* a scalar, low-frequency illumination gain inside the product; and
* a small additive specular residual inside the product.

Geometry, product alpha, identity pixels and the background remain owned by
the deterministic renderer.  This keeps a future ComfyUI/RunningHub model
replaceable without giving it authority over labels, silhouettes or pixels
outside the authorised masks.
"""

from dataclasses import dataclass, field
import copy
import hashlib
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

import cv2
import numpy as np

from .contracts import Lighting


GAIN_MIN = 0.82
GAIN_MAX = 1.18
SPECULAR_RESIDUAL_LIMIT = 0.08


class LearnedEffectContractError(ValueError):
    """Raised when a provider attempts to exceed its pixel authority."""


@dataclass(frozen=True)
class EffectContext:
    """Full-canvas evidence supplied to a learned-effect provider.

    Arrays are provided for inference, not as writable output buffers.  A
    provider must return a :class:`LearnedEffectPrediction`; returning a full
    RGB image is intentionally not part of this interface.
    """

    product_id: str
    background_rgb: np.ndarray
    support_mask: np.ndarray
    protected_mask: np.ndarray
    product_rgb: np.ndarray
    product_alpha: np.ndarray
    identity_guard_mask: np.ndarray
    deterministic_shadow_alpha: np.ndarray
    material_mode: str
    lighting: Lighting
    depth: np.ndarray | None = None
    normal: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LearnedEffectRequest:
    """One atomic scene request so all products share one lighting solution."""

    background_rgb: np.ndarray
    support_mask: np.ndarray
    protected_mask: np.ndarray
    lighting: Lighting
    products: tuple[EffectContext, ...]
    depth: np.ndarray | None = None
    normal: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LearnedEffectPrediction:
    """Restricted model result in full-canvas coordinates.

    ``shadow_alpha`` may be uint8 ``[0,255]`` or float ``[0,1]``.
    ``illumination_gain`` is a scalar float map in ``[0.82,1.18]``.
    ``specular_residual`` is either a scalar or RGB *linear-light* residual in
    ``[-0.08,0.08]``.  It is not an RGB replacement image.
    """

    shadow_alpha: np.ndarray | None = None
    illumination_gain: np.ndarray | None = None
    specular_residual: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LearnedEffectBatchPrediction:
    """Atomic per-product predictions for one scene."""

    products: Mapping[str, LearnedEffectPrediction]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class LearnedEffectProvider(Protocol):
    """Provider protocol implemented by local, ComfyUI or cloud adapters."""

    name: str

    def predict(self, request: LearnedEffectRequest) -> LearnedEffectBatchPrediction:
        ...


@dataclass(frozen=True)
class SanitizedEffectPrediction:
    shadow_alpha: np.ndarray | None
    illumination_gain: np.ndarray | None
    specular_residual: np.ndarray | None
    report: Mapping[str, Any]


@dataclass(frozen=True)
class AppliedLearnedEffects:
    rgb: np.ndarray
    shadow_alpha: np.ndarray
    prediction: SanitizedEffectPrediction
    report: Mapping[str, Any]


@dataclass(frozen=True)
class LoadedSceneEvidence:
    """Validated, canvas-registered geometric evidence for one scene."""

    depth: np.ndarray | None
    normal: np.ndarray | None
    report: Mapping[str, Any]


def load_scene_evidence(
    depth_path: str | Path | None,
    normal_path: str | Path | None,
    canvas_shape: tuple[int, int],
) -> LoadedSceneEvidence:
    """Load optional normalized depth and unit-normal evidence from NPY files.

    This function is intentionally format- and dtype-strict.  Learned-effect
    adapters receive geometry as numeric tensors, never as pickled Python
    objects or loosely interpreted image files.  Depth is normalized to
    ``[0,1]`` before it reaches this boundary.  Normals use XYZ components in
    ``[-1,1]`` and must already be approximately unit length; interpolation is
    followed by an exact renormalization on the target canvas.
    """

    if (
        not isinstance(canvas_shape, tuple)
        or len(canvas_shape) != 2
        or any(not isinstance(value, (int, np.integer)) for value in canvas_shape)
        or any(int(value) <= 0 for value in canvas_shape)
    ):
        raise LearnedEffectContractError(
            "canvas_shape must be a positive (height, width) tuple"
        )
    target = (int(canvas_shape[0]), int(canvas_shape[1]))
    depth, depth_report = _load_scene_evidence_array(
        depth_path,
        target,
        kind="depth",
    )
    normal, normal_report = _load_scene_evidence_array(
        normal_path,
        target,
        kind="normal",
    )
    return LoadedSceneEvidence(
        depth=depth,
        normal=normal,
        report={
            "schema_version": "learned-scene-evidence-v1",
            "canvas_shape": list(target),
            "depth": depth_report,
            "normal": normal_report,
        },
    )


def _load_scene_evidence_array(
    raw_path: str | Path | None,
    target_shape: tuple[int, int],
    *,
    kind: str,
) -> tuple[np.ndarray | None, Mapping[str, Any]]:
    if raw_path is None or not str(raw_path).strip():
        return None, {"configured": False}
    path = Path(raw_path).expanduser().resolve()
    if path.suffix.lower() != ".npy":
        raise LearnedEffectContractError(
            f"{kind} evidence must be a numeric .npy file"
        )
    if not path.is_file():
        raise LearnedEffectContractError(f"{kind} evidence file does not exist")
    try:
        source = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise LearnedEffectContractError(
            f"failed to load {kind} evidence: {exc}"
        ) from exc
    if not isinstance(source, np.ndarray):
        raise LearnedEffectContractError(f"{kind} evidence must be an ndarray")
    if source.dtype != np.float32:
        raise LearnedEffectContractError(
            f"{kind} evidence must have dtype float32, got {source.dtype}"
        )
    expected_rank = 2 if kind == "depth" else 3
    if source.ndim != expected_rank:
        expected = "HxW" if kind == "depth" else "HxWx3"
        raise LearnedEffectContractError(
            f"{kind} evidence must have shape {expected}, got {source.shape}"
        )
    if kind == "normal" and source.shape[2] != 3:
        raise LearnedEffectContractError(
            f"normal evidence must have shape HxWx3, got {source.shape}"
        )
    if source.shape[0] <= 0 or source.shape[1] <= 0:
        raise LearnedEffectContractError(f"{kind} evidence cannot be empty")
    if not np.isfinite(source).all():
        raise LearnedEffectContractError(
            f"{kind} evidence contains NaN or infinity"
        )

    source = np.ascontiguousarray(source)
    source_range = [float(source.min()), float(source.max())]
    norm_range: list[float] | None = None
    if kind == "depth":
        if source_range[0] < 0.0 or source_range[1] > 1.0:
            raise LearnedEffectContractError(
                "depth evidence must be normalized to [0,1]"
            )
    else:
        if source_range[0] < -1.0 or source_range[1] > 1.0:
            raise LearnedEffectContractError(
                "normal evidence components must remain in [-1,1]"
            )
        lengths = np.linalg.norm(source, axis=2)
        norm_range = [float(lengths.min()), float(lengths.max())]
        if norm_range[0] < 0.90 or norm_range[1] > 1.10:
            raise LearnedEffectContractError(
                "normal evidence vectors must be unit length within 10%"
            )

    source_shape = source.shape
    resized = source_shape[:2] != target_shape
    interpolation_name = "none"
    output = source.copy()
    if resized:
        shrinking = (
            target_shape[0] <= source_shape[0]
            and target_shape[1] <= source_shape[1]
        )
        interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
        interpolation_name = "area" if shrinking else "linear"
        output = cv2.resize(
            source,
            (target_shape[1], target_shape[0]),
            interpolation=interpolation,
        )
        if kind == "normal" and output.ndim == 2:
            # OpenCV may drop a singleton channel, but never a real XYZ map.
            raise LearnedEffectContractError(
                "normal evidence resize did not preserve three channels"
            )

    output = np.ascontiguousarray(output, dtype=np.float32)
    if kind == "depth":
        output = np.clip(output, 0.0, 1.0).astype(np.float32, copy=False)
        output_norm_range = None
    else:
        output_lengths = np.linalg.norm(output, axis=2, keepdims=True)
        if not np.isfinite(output_lengths).all() or np.any(output_lengths <= 1e-6):
            raise LearnedEffectContractError(
                "normal evidence contains a zero vector after resize"
            )
        output = output / output_lengths
        output = np.ascontiguousarray(output, dtype=np.float32)
        normalized_lengths = np.linalg.norm(output, axis=2)
        output_norm_range = [
            float(normalized_lengths.min()),
            float(normalized_lengths.max()),
        ]
    if not np.isfinite(output).all():
        raise LearnedEffectContractError(
            f"{kind} evidence became non-finite after resize"
        )
    output.setflags(write=False)
    return output, {
        "configured": True,
        "path": str(path),
        "file_sha256": _file_sha256(path),
        "source_shape": list(source_shape),
        "canvas_shape": list(output.shape),
        "dtype": str(output.dtype),
        "resized": resized,
        "resize_interpolation": interpolation_name,
        "source_range": source_range,
        "canvas_range": _array_range(output),
        "source_vector_norm_range": norm_range,
        "canvas_vector_norm_range": output_norm_range,
        "canvas_array_sha256": _array_sha256(output),
    }


def _require_canvas_map(
    value: np.ndarray,
    shape: tuple[int, int],
    name: str,
) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape:
        if array.ndim == 3 and array.shape[:2] == shape:
            raise LearnedEffectContractError(
                f"{name} must be a single-channel map; full RGB output is forbidden"
            )
        raise LearnedEffectContractError(
            f"{name} must have shape {shape}, got {array.shape}"
        )
    if not np.issubdtype(array.dtype, np.number):
        raise LearnedEffectContractError(f"{name} must be numeric")
    array = array.astype(np.float32, copy=False)
    if not np.isfinite(array).all():
        raise LearnedEffectContractError(f"{name} contains NaN or infinity")
    return array


def sanitize_effect_prediction(
    prediction: LearnedEffectPrediction,
    context: EffectContext,
) -> SanitizedEffectPrediction:
    """Validate a model result and enforce support/protection boundaries.

    Malformed gain/residual maps are rejected rather than silently coerced.
    Shadow values are converted to uint8 and hard-clipped to the receiver
    support while excluding protected pixels.  Clipping is recorded so model
    leakage remains observable during experimentation.
    """

    if not isinstance(prediction, LearnedEffectPrediction):
        raise LearnedEffectContractError(
            "provider must return LearnedEffectPrediction, not a rendered image"
        )
    shape = context.product_alpha.shape
    if context.background_rgb.shape[:2] != shape:
        raise LearnedEffectContractError("background and product canvas differ")
    for name, value in (
        ("support_mask", context.support_mask),
        ("protected_mask", context.protected_mask),
        ("identity_guard_mask", context.identity_guard_mask),
        ("deterministic_shadow_alpha", context.deterministic_shadow_alpha),
    ):
        if np.asarray(value).shape != shape:
            raise LearnedEffectContractError(f"{name} must match product canvas")

    shadow_u8: np.ndarray | None = None
    shadow_clipped_pixels = 0
    shadow_envelope_pixels = 0
    shadow_outside_envelope_pixels = 0
    shadow_accepted_pixels = 0
    shadow_envelope_fill_ratio = 0.0
    shadow_uniform_fill_detected = False
    if prediction.shadow_alpha is not None:
        raw_shadow = np.asarray(prediction.shadow_alpha)
        shadow = _require_canvas_map(raw_shadow, shape, "shadow_alpha")
        if np.issubdtype(raw_shadow.dtype, np.integer):
            if float(shadow.min()) < 0.0 or float(shadow.max()) > 255.0:
                raise LearnedEffectContractError(
                    "integer shadow_alpha must be in [0,255]"
                )
            shadow = shadow / 255.0
        elif float(shadow.min()) < 0.0 or float(shadow.max()) > 1.0:
            raise LearnedEffectContractError("float shadow_alpha must be in [0,1]")
        requested = shadow > (0.5 / 255.0)
        receiver = np.asarray(context.support_mask, dtype=bool)
        receiver &= ~np.asarray(context.protected_mask, dtype=bool)
        # A learned map is never allowed to define its own background-edit
        # authority.  Build a deliberately generous, model-independent
        # envelope from the deterministic directional projection, then let the
        # network refine alpha only inside that envelope.  This keeps learned
        # shadows useful without allowing an all-white or adversarial output to
        # turn the whole tabletop into an authorised region.
        envelope = _shadow_admissible_envelope(context)
        shadow_envelope_pixels = int(np.count_nonzero(envelope))
        shadow_outside_envelope_pixels = int(
            np.count_nonzero(requested & ~envelope)
        )
        receiver &= envelope
        shadow_clipped_pixels = int(np.count_nonzero(requested & ~receiver))
        accepted_shape = requested & receiver
        shadow_accepted_pixels = int(np.count_nonzero(accepted_shape))
        shadow_envelope_fill_ratio = float(
            shadow_accepted_pixels / max(1, shadow_envelope_pixels)
        )
        baseline_shadow = np.asarray(
            context.deterministic_shadow_alpha,
            dtype=np.uint8,
        )
        baseline_anchor = baseline_shadow > 0
        # A failed/miswired workflow commonly returns a constant white image.
        # Even after geometric clipping that would paint the whole admissible
        # envelope as one hard slab.  Reject that signature instead of merely
        # making it a legal background edit.  Legitimate penumbrae retain a
        # density gradient and normally occupy only a subset of the envelope.
        if shadow_accepted_pixels:
            accepted_values = shadow[accepted_shape]
            shadow_uniform_fill_detected = (
                shadow_envelope_fill_ratio >= 0.94
                and shadow_accepted_pixels
                >= max(64, int(np.count_nonzero(baseline_anchor)) * 1.25)
                and float(np.ptp(accepted_values)) <= (2.0 / 255.0)
            )
        # The model predicts relative density/shape.  Absolute darkness stays
        # local and is bounded by both the declared scene opacity and the
        # deterministic renderer's strongest sample.
        declared_peak = float(
            np.clip(context.lighting.shadow_opacity, 0.0, 1.0)
        )
        deterministic_peak = float(baseline_shadow.max()) / 255.0
        shadow_peak_authority = float(
            np.clip(max(declared_peak, deterministic_peak) * 1.10, 0.0, 0.78)
        )
        shadow = np.where(receiver, shadow * shadow_peak_authority, 0.0)
        if np.any(baseline_anchor) and not np.any(
            (shadow > (0.5 / 255.0)) & baseline_anchor
        ):
            raise LearnedEffectContractError(
                "learned shadow is detached from the deterministic contact projection"
            )
        shadow_u8 = np.clip(np.round(shadow * 255.0), 0, 255).astype(np.uint8)
        if (
            context.lighting.cast_shadow_mode != "directional"
            and np.any(shadow_u8)
        ):
            raise LearnedEffectContractError(
                "non-directional scenes cannot receive a learned cast shadow"
            )

    gain: np.ndarray | None = None
    if prediction.illumination_gain is not None:
        gain = _require_canvas_map(
            prediction.illumination_gain,
            shape,
            "illumination_gain",
        )
        if float(gain.min()) < GAIN_MIN or float(gain.max()) > GAIN_MAX:
            raise LearnedEffectContractError(
                f"illumination_gain must remain in [{GAIN_MIN},{GAIN_MAX}]"
            )
        gain = gain.astype(np.float32, copy=True)

    residual: np.ndarray | None = None
    if prediction.specular_residual is not None:
        residual = np.asarray(prediction.specular_residual)
        if residual.shape == shape:
            residual = residual[..., None]
        elif residual.shape != (*shape, 3):
            raise LearnedEffectContractError(
                "specular_residual must be HxW or HxWx3; full replacement RGB is forbidden"
            )
        if not np.issubdtype(residual.dtype, np.floating):
            raise LearnedEffectContractError(
                "specular_residual must be a bounded floating-point residual"
            )
        residual = residual.astype(np.float32, copy=False)
        if not np.isfinite(residual).all():
            raise LearnedEffectContractError(
                "specular_residual contains NaN or infinity"
            )
        if (
            float(residual.min()) < -SPECULAR_RESIDUAL_LIMIT
            or float(residual.max()) > SPECULAR_RESIDUAL_LIMIT
        ):
            raise LearnedEffectContractError(
                "specular_residual exceeds the bounded linear-light authority"
            )
        if residual.shape[2] == 1:
            residual = np.repeat(residual, 3, axis=2)
        residual = residual.copy()

    report = {
        "schema_version": "learned-effects-v1",
        "shadow_predicted": shadow_u8 is not None,
        "illumination_gain_predicted": gain is not None,
        "specular_residual_predicted": residual is not None,
        "shadow_clipped_pixels": shadow_clipped_pixels,
        "shadow_envelope_pixels": shadow_envelope_pixels,
        "shadow_outside_envelope_pixels": shadow_outside_envelope_pixels,
        "shadow_accepted_pixels": shadow_accepted_pixels,
        "shadow_envelope_fill_ratio": shadow_envelope_fill_ratio,
        "shadow_uniform_fill_detected": shadow_uniform_fill_detected,
        "shadow_peak_authority": (
            shadow_peak_authority
            if prediction.shadow_alpha is not None else None
        ),
        "prediction_hashes": {
            "shadow_alpha": _array_sha256(shadow_u8),
            "illumination_gain": _array_sha256(gain),
            "specular_residual": _array_sha256(residual),
        },
        "prediction_ranges": {
            "shadow_alpha": _array_range(shadow_u8),
            "illumination_gain": _array_range(gain),
            "specular_residual": _array_range(residual),
        },
        "provider_metadata": dict(prediction.metadata),
    }
    return SanitizedEffectPrediction(shadow_u8, gain, residual, report)


def _shadow_admissible_envelope(context: EffectContext) -> np.ndarray:
    """Return a broad but model-independent receiver envelope.

    The deterministic renderer already owns contact, direction and maximum
    reach.  Learned inference is permitted to reshape/soften that solution,
    not to create a new receiver-edit region.  A bounded dilation gives the
    model room to improve penumbra and object-specific contour while remaining
    anchored to the renderer's geometric evidence.
    """

    baseline = np.asarray(context.deterministic_shadow_alpha) > 0
    if not np.any(baseline):
        return np.zeros_like(baseline, dtype=bool)
    body = np.asarray(context.product_alpha) > 0
    occupied = np.argwhere(body)
    if occupied.size:
        y0, x0 = occupied.min(axis=0)
        y1, x1 = occupied.max(axis=0) + 1
        product_extent = max(int(y1 - y0), int(x1 - x0))
    else:
        product_extent = max(1, min(baseline.shape))
    expected_reach = max(
        1.0,
        float(max(baseline.shape))
        * float(max(0.0, context.lighting.shadow_length_ratio)),
    )
    softness = product_extent * float(
        max(0.0, context.lighting.shadow_softness_ratio)
    )
    radius = int(round(max(2.0, expected_reach * 0.22 + softness * 0.45)))
    radius = int(np.clip(radius, 2, max(2, min(baseline.shape) // 10)))
    kernel_size = radius * 2 + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    envelope = cv2.dilate(
        baseline.astype(np.uint8),
        kernel,
        iterations=1,
    ) > 0
    envelope &= np.asarray(context.support_mask, dtype=bool)
    envelope &= ~np.asarray(context.protected_mask, dtype=bool)
    return envelope


def _srgb_u8_to_linear(value: np.ndarray) -> np.ndarray:
    encoded = value.astype(np.float32) / 255.0
    return np.where(
        encoded <= 0.04045,
        encoded / 12.92,
        ((encoded + 0.055) / 1.055) ** 2.4,
    )


def _linear_to_srgb_u8(value: np.ndarray) -> np.ndarray:
    linear = np.clip(value.astype(np.float32), 0.0, 1.0)
    encoded = np.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * np.power(linear, 1.0 / 2.4) - 0.055,
    )
    return np.clip(np.round(encoded * 255.0), 0, 255).astype(np.uint8)


def apply_learned_effect_prediction(
    prediction: LearnedEffectPrediction,
    context: EffectContext,
) -> AppliedLearnedEffects:
    """Apply only authorised model signals, then restore identity exactly."""

    sanitized = sanitize_effect_prediction(prediction, context)
    original_rgb = np.asarray(context.product_rgb, dtype=np.uint8)
    if original_rgb.shape != (*context.product_alpha.shape, 3):
        raise LearnedEffectContractError("product_rgb must be HxWx3 uint8")
    body = np.asarray(context.product_alpha) > 0
    guard = np.asarray(context.identity_guard_mask, dtype=bool) & body
    editable = body & ~guard
    has_gain = (
        sanitized.illumination_gain is not None
        and not bool(np.all(sanitized.illumination_gain == 1.0))
    )
    has_residual = (
        sanitized.specular_residual is not None
        and (
            float(sanitized.specular_residual.min()) != 0.0
            or float(sanitized.specular_residual.max()) != 0.0
        )
    )
    if not has_gain and not has_residual:
        # Do not round-trip through linear light for a shadow-only or neutral
        # prediction.  This is the learned branch's byte-stable fast path.
        output_rgb = original_rgb.copy()
    else:
        output_rgb = _apply_low_frequency_appearance(
            original_rgb,
            np.asarray(context.product_alpha, dtype=np.uint8),
            guard,
            sanitized.illumination_gain if has_gain else None,
            sanitized.specular_residual if has_residual else None,
        )

    shadow = (
        sanitized.shadow_alpha
        if sanitized.shadow_alpha is not None
        else np.asarray(context.deterministic_shadow_alpha, dtype=np.uint8).copy()
    )
    changed = np.any(output_rgb != original_rgb, axis=2)
    report = {
        **dict(sanitized.report),
        "appearance_changed_pixels": int(np.count_nonzero(changed)),
        "identity_restored_pixels": int(np.count_nonzero(guard)),
        "outside_product_changed_pixels": int(np.count_nonzero(changed & ~body)),
        "identity_changed_pixels": int(np.count_nonzero(changed & guard)),
        "shadow_source": (
            "learned_replacement"
            if sanitized.shadow_alpha is not None
            else "deterministic_fallback"
        ),
        "strict_input_hashes": {
            "product_rgb": _array_sha256(context.product_rgb),
            "product_alpha": _array_sha256(context.product_alpha),
            "identity_guard": _array_sha256(context.identity_guard_mask),
            "deterministic_shadow": _array_sha256(
                context.deterministic_shadow_alpha,
            ),
        },
        "applied_hashes": {
            "product_rgb": _array_sha256(output_rgb),
            "shadow_alpha": _array_sha256(shadow),
        },
    }
    return AppliedLearnedEffects(output_rgb, shadow, sanitized, report)


def _apply_low_frequency_appearance(
    original_rgb: np.ndarray,
    alpha_u8: np.ndarray,
    identity_guard: np.ndarray,
    illumination_gain: np.ndarray | None,
    specular_residual: np.ndarray | None,
) -> np.ndarray:
    """Relight the low band while retaining the original high-frequency band."""

    body = alpha_u8 > 0
    occupied = np.argwhere(body)
    if not occupied.size:
        return original_rgb.copy()
    y0, x0 = occupied.min(axis=0)
    y1, x1 = occupied.max(axis=0) + 1
    extent = max(int(y1 - y0), int(x1 - x0))
    sigma = float(np.clip(extent * 0.035, 3.0, 36.0))
    pad = int(np.ceil(sigma * 3.0))
    y0 = max(0, int(y0) - pad)
    x0 = max(0, int(x0) - pad)
    y1 = min(alpha_u8.shape[0], int(y1) + pad)
    x1 = min(alpha_u8.shape[1], int(x1) + pad)

    rgb_roi = original_rgb[y0:y1, x0:x1]
    alpha = alpha_u8[y0:y1, x0:x1].astype(np.float32) / 255.0
    linear = _srgb_u8_to_linear(rgb_roi)
    weight = cv2.GaussianBlur(
        alpha,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REPLICATE,
    )
    numerator = np.stack(
        [
            cv2.GaussianBlur(
                linear[..., channel] * alpha,
                (0, 0),
                sigmaX=sigma,
                sigmaY=sigma,
                borderType=cv2.BORDER_REPLICATE,
            )
            for channel in range(3)
        ],
        axis=-1,
    )
    low = numerator / np.maximum(weight[..., None], 1.0 / 255.0)
    high = linear - low
    target_low = low
    if illumination_gain is not None:
        gain_roi = illumination_gain[y0:y1, x0:x1]
        gain_roi = cv2.GaussianBlur(
            gain_roi,
            (0, 0),
            sigmaX=max(1.0, sigma * 0.55),
            sigmaY=max(1.0, sigma * 0.55),
            borderType=cv2.BORDER_REPLICATE,
        )
        target_low = low * gain_roi[..., None]
    modified = target_low + high
    if specular_residual is not None:
        # Residual means a smooth illumination/highlight correction, never a
        # high-frequency replacement texture.  Low-pass it independently of
        # provider behaviour before applying it to unguarded product pixels.
        residual_roi = specular_residual[y0:y1, x0:x1]
        residual_sigma = max(1.5, sigma * 0.35)
        residual_low = np.stack(
            [
                cv2.GaussianBlur(
                    residual_roi[..., channel],
                    (0, 0),
                    sigmaX=residual_sigma,
                    sigmaY=residual_sigma,
                    borderType=cv2.BORDER_REPLICATE,
                )
                for channel in range(3)
            ],
            axis=-1,
        )
        modified += residual_low

    output = original_rgb.copy()
    converted = _linear_to_srgb_u8(modified)
    editable_roi = (
        body[y0:y1, x0:x1]
        & ~identity_guard[y0:y1, x0:x1]
    )
    output_roi = output[y0:y1, x0:x1]
    output_roi[editable_roi] = converted[editable_roi]
    # Product identity, outline and holes remain exact source-derived pixels.
    output[identity_guard & body] = original_rgb[identity_guard & body]
    output[~body] = original_rgb[~body]
    return output


def isolate_effect_request(request: LearnedEffectRequest) -> LearnedEffectRequest:
    """Copy provider inputs and expose only read-only inference tensors."""

    background = _readonly_copy(request.background_rgb)
    support = _readonly_copy(request.support_mask)
    protected = _readonly_copy(request.protected_mask)
    depth = None if request.depth is None else _readonly_copy(request.depth)
    normal = None if request.normal is None else _readonly_copy(request.normal)
    products = tuple(
        EffectContext(
            product_id=item.product_id,
            # Scene evidence is shared by every product; isolate it once, not
            # once per product.  This avoids hundreds of megabytes of duplicate
            # 7K background copies while preserving mutation safety.
            background_rgb=background,
            support_mask=support,
            protected_mask=protected,
            product_rgb=_readonly_copy(item.product_rgb),
            product_alpha=_readonly_copy(item.product_alpha),
            identity_guard_mask=_readonly_copy(item.identity_guard_mask),
            deterministic_shadow_alpha=_readonly_copy(
                item.deterministic_shadow_alpha,
            ),
            material_mode=item.material_mode,
            lighting=item.lighting,
            depth=depth,
            normal=normal,
            metadata=copy.deepcopy(dict(item.metadata)),
        )
        for item in request.products
    )
    return LearnedEffectRequest(
        background_rgb=background,
        support_mask=support,
        protected_mask=protected,
        lighting=request.lighting,
        products=products,
        depth=depth,
        normal=normal,
        metadata=copy.deepcopy(dict(request.metadata)),
    )


def _readonly_copy(value: np.ndarray) -> np.ndarray:
    output = np.array(value, copy=True, order="C")
    output.setflags(write=False)
    return output


def apply_learned_effect_batch(
    prediction: LearnedEffectBatchPrediction,
    request: LearnedEffectRequest,
) -> tuple[AppliedLearnedEffects, ...]:
    """Validate and apply a complete scene atomically.

    Every requested product must have exactly one prediction.  All products
    are validated before the caller receives any replacement layer, so one bad
    map cannot create a half-learned/half-deterministic scene.
    """

    if not isinstance(prediction, LearnedEffectBatchPrediction):
        raise LearnedEffectContractError(
            "provider must return LearnedEffectBatchPrediction"
        )
    expected = [item.product_id for item in request.products]
    if len(set(expected)) != len(expected):
        raise LearnedEffectContractError("scene request contains duplicate product ids")
    actual = set(prediction.products)
    if actual != set(expected):
        raise LearnedEffectContractError(
            "batch prediction product ids do not exactly match the request"
        )
    # No mutable strict layer is touched here.  If any item raises, the entire
    # list is discarded and the pipeline retains its deterministic tuple.
    return tuple(
        apply_learned_effect_prediction(prediction.products[item.product_id], item)
        for item in request.products
    )


def _array_sha256(value: np.ndarray | None) -> str | None:
    if value is None:
        return None
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _array_range(value: np.ndarray | None) -> list[float] | None:
    if value is None or not np.asarray(value).size:
        return None
    array = np.asarray(value)
    return [float(array.min()), float(array.max())]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class RegisteredNpzEffectProvider:
    """Experimental adapter for offline/ComfyUI-produced effect tensors.

    Each product is registered explicitly to an ``.npz`` file containing any
    of ``shadow_alpha``, ``illumination_gain`` and ``specular_residual``.  The
    explicit mapping avoids directory scans and path-derived product ids.
    """

    name = "registered_npz"

    def __init__(self, registrations: Mapping[str, str | Path]) -> None:
        self._registrations = {
            str(product_id): Path(path).resolve()
            for product_id, path in registrations.items()
        }

    def predict(self, request: LearnedEffectRequest) -> LearnedEffectBatchPrediction:
        predictions: dict[str, LearnedEffectPrediction] = {}
        for context in request.products:
            path = self._registrations.get(context.product_id)
            if path is None:
                raise FileNotFoundError(
                    f"no learned-effect tensor registered for {context.product_id}"
                )
            if path.suffix.lower() != ".npz":
                raise LearnedEffectContractError("registered effect tensor must be .npz")
            with np.load(path, allow_pickle=False) as payload:
                names = set(payload.files)
                unknown = names - {
                    "shadow_alpha", "illumination_gain", "specular_residual",
                }
                if unknown:
                    raise LearnedEffectContractError(
                        f"unsupported learned-effect arrays: {sorted(unknown)}"
                    )
                predictions[context.product_id] = LearnedEffectPrediction(
                    shadow_alpha=(
                        payload["shadow_alpha"] if "shadow_alpha" in names else None
                    ),
                    illumination_gain=(
                        payload["illumination_gain"]
                        if "illumination_gain" in names
                        else None
                    ),
                    specular_residual=(
                        payload["specular_residual"]
                        if "specular_residual" in names
                        else None
                    ),
                    metadata={"source": str(path)},
                )
        return LearnedEffectBatchPrediction(
            products=predictions,
            metadata={"provider": self.name},
        )
