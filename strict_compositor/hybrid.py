from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from .image_io import load_background, save_png


class HybridRenderError(RuntimeError):
    pass


@dataclass(frozen=True)
class HybridRenderRequest:
    strict_base: Path
    edit_mask: Path
    protect_mask: Path
    prompt: str
    expected_size: tuple[int, int]
    output_dir: Path


@dataclass(frozen=True)
class HybridRenderResult:
    candidate_path: Path
    provider: str
    task_id: str | None = None
    provenance: dict[str, Any] | None = None


class HybridCandidateProvider(Protocol):
    def render(self, request: HybridRenderRequest) -> HybridRenderResult: ...


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def load_registered_candidate(
    candidate_path: str | Path,
    expected_shape: tuple[int, int],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a candidate and permit only an aspect-compatible deterministic fit."""
    candidate = load_background(candidate_path)
    target_h, target_w = expected_shape
    source_h, source_w = candidate.shape[:2]
    if (source_h, source_w) == (target_h, target_w):
        return candidate, {"registration": "pixel_registered", "source_dimensions": [source_w, source_h]}
    source_aspect = source_w / max(1.0, float(source_h))
    target_aspect = target_w / max(1.0, float(target_h))
    error = abs(source_aspect - target_aspect) / max(1.0e-6, target_aspect)
    if error > 0.008:
        raise HybridRenderError(
            "hybrid candidate aspect ratio differs from the strict composition"
        )
    resized = cv2.resize(candidate, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    return resized, {
        "registration": "aspect_compatible_resize",
        "source_dimensions": [source_w, source_h],
        "aspect_error": float(error),
    }


def recover_hybrid_candidate(
    strict_base: np.ndarray,
    candidate: np.ndarray,
    rerender_allow_mask: np.ndarray,
    identity_guard_mask: np.ndarray,
    *,
    feather_pixels: int = 5,
    optical_polish_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Copy candidate pixels only where the deterministic contract authorises.

    Every pixel outside ``rerender_allow_mask`` is restored from ``strict_base``
    after blending.  Identity protection is therefore a byte-level invariant,
    not a prompt request to the image model.
    """
    if strict_base.shape != candidate.shape or strict_base.ndim != 3:
        raise HybridRenderError("hybrid candidate dimensions do not match strict base")
    expected = strict_base.shape[:2]
    allow = np.asarray(rerender_allow_mask, dtype=bool)
    guard = np.asarray(identity_guard_mask, dtype=bool)
    if allow.shape != expected or guard.shape != expected:
        raise HybridRenderError("hybrid masks must match strict base")
    optical_polish = (
        np.zeros(expected, bool)
        if optical_polish_mask is None
        else np.asarray(optical_polish_mask, dtype=bool)
    )
    if optical_polish.shape != expected:
        raise HybridRenderError("optical polish mask must match strict base")
    effective = allow & ~guard
    if not np.any(effective):
        return strict_base.copy(), {
            "status": "strict_fallback",
            "reason": "no_eligible_region",
            "effective_pixels": 0,
            "identity_changed_pixels": 0,
            "outside_changed_pixels": 0,
            "max_outside_delta": 0,
        }
    distance = cv2.distanceTransform(effective.astype(np.uint8), cv2.DIST_L2, 5)
    # A fixed five-pixel blend is effectively a hard cut on a 4K glass
    # product.  Model candidates commonly carry a bright one-to-three pixel
    # halo after their lower-resolution output is registered to the delivery
    # canvas.  Derive the recovery width from the actual editable interior so
    # the deterministic rim/base/label guard remains authoritative at every
    # output size.  The ceiling keeps narrow glass handles from disappearing.
    positive_distance = distance[distance > 0.0]
    interior_radius = (
        float(np.percentile(positive_distance, 99.0))
        if positive_distance.size else 0.0
    )
    optical_region = optical_polish & effective
    if np.count_nonzero(optical_region) >= 24:
        # Clear-glass value lives in narrow rim/handle/base interfaces.  The
        # generic 18 px candidate feather suppressed those very highlights at
        # 4K, even though a two-pixel deterministic structure guard already
        # locks the exact silhouette.  Scale this transition to the editable
        # optical radius and keep it bounded; opaque or labelled routes retain
        # the wider generic recovery below.
        adaptive_feather = float(np.clip(
            max(3.0, interior_radius * 0.055),
            3.0,
            8.0,
        ))
        recovery_profile = "clear_optical_interface"
    else:
        adaptive_feather = float(np.clip(
            max(float(feather_pixels), interior_radius * 0.18),
            max(1.0, float(feather_pixels)),
            18.0,
        ))
        recovery_profile = "generic_identity_safe"
    weight = np.clip(distance / adaptive_feather, 0.0, 1.0)
    # Smoothstep has zero derivative at the strict/model boundary.  Compared
    # with a linear ramp this removes the pale contour without making the
    # useful central refraction look washed out.
    weight = weight * weight * (3.0 - 2.0 * weight)
    weight[~effective] = 0.0
    blended = np.round(
        strict_base.astype(np.float32) * (1.0 - weight[..., None])
        + candidate.astype(np.float32) * weight[..., None]
    ).clip(0, 255).astype(np.uint8)

    # A generative optical candidate is good at replacing the old room inside
    # clear glass, but its resize/denoise stage often removes the sparse real
    # interface glints that give a catalogue product depth.  Recover only the
    # *missing positive high-frequency residual* measured in the deterministic,
    # target-derived glass layer, then compress the candidate's own brightest
    # interface residuals a little.  Both terms are zero-mean/high-pass and
    # percentile gated: they cannot become a broad brightness veil, import the
    # old room, or change an opaque/text-bearing product.
    polish = optical_polish & effective & (weight > 0.0)
    polish_pixels = 0
    polish_max_delta = 0.0
    if np.count_nonzero(polish) >= 24:
        luma_weights = np.asarray([0.2126, 0.7152, 0.0722], np.float32)
        strict_float = strict_base.astype(np.float32)
        blended_float = blended.astype(np.float32)
        strict_luma = strict_float @ luma_weights
        blended_luma = blended_float @ luma_weights
        strict_low = cv2.GaussianBlur(
            strict_luma, (0, 0), 1.45, borderType=cv2.BORDER_REPLICATE,
        )
        blended_low = cv2.GaussianBlur(
            blended_luma, (0, 0), 1.45, borderType=cv2.BORDER_REPLICATE,
        )
        strict_high = strict_luma - strict_low
        blended_high = blended_luma - blended_low
        positive_samples = strict_high[polish]
        positive_samples = positive_samples[positive_samples > 0.0]
        threshold = (
            max(2.0, float(np.percentile(positive_samples, 66.0)))
            if positive_samples.size else 2.0
        )
        missing = np.clip(
            strict_high - blended_high - threshold,
            0.0,
            20.0,
        )
        polish_delta = np.clip(missing * 0.68, 0.0, 12.0)

        # Preserve the model's scene-aware specular decisions but make its
        # sparse positive peaks survive candidate resize and final 4K display.
        # This is deliberately relative to each product's own high-frequency
        # distribution, not to pitcher-specific coordinates or a fixed RGB.
        candidate_positive = blended_high[polish]
        candidate_positive = candidate_positive[candidate_positive > 0.0]
        if candidate_positive.size:
            candidate_threshold = max(
                2.5, float(np.percentile(candidate_positive, 84.0)),
            )
            candidate_glint = np.clip(
                blended_high - candidate_threshold,
                0.0,
                18.0,
            )
            polish_delta += np.clip(candidate_glint * 0.32, 0.0, 5.0)
            polish_delta = np.clip(polish_delta, 0.0, 14.0)
        polish_delta *= weight
        active_polish = polish & (polish_delta > 0.25)
        if np.any(active_polish):
            blended_float[active_polish] = np.clip(
                blended_float[active_polish]
                + polish_delta[active_polish, None],
                0.0,
                255.0,
            )
            blended = np.clip(
                np.round(blended_float), 0, 255,
            ).astype(np.uint8)
            polish_pixels = int(np.count_nonzero(active_polish))
            polish_max_delta = float(np.max(polish_delta[active_polish]))
    # Hard recovery is deliberately repeated after the soft blend.
    blended[~effective] = strict_base[~effective]
    blended[guard] = strict_base[guard]

    delta = np.abs(blended.astype(np.int16) - strict_base.astype(np.int16))
    changed = np.any(delta > 0, axis=2)
    outside = ~effective
    identity_changed = int(np.count_nonzero(changed & guard))
    outside_changed = int(np.count_nonzero(changed & outside))
    max_outside = int(delta[outside].max()) if np.any(outside) else 0
    if identity_changed or outside_changed or max_outside:
        raise HybridRenderError("hybrid hard recovery invariant failed")
    return blended, {
        "status": "applied",
        "reason": "candidate_recovered_inside_authorized_regions",
        "effective_pixels": int(np.count_nonzero(effective)),
        "identity_changed_pixels": identity_changed,
        "outside_changed_pixels": outside_changed,
        "max_outside_delta": max_outside,
        "candidate_sha256": _array_sha256(candidate),
        "effective_mask_sha256": _array_sha256(effective.astype(np.uint8)),
        "recovery_feather_pixels": adaptive_feather,
        "recovery_profile": recovery_profile,
        "editable_interior_radius_p99": interior_radius,
        "optical_polish_pixels": polish_pixels,
        "optical_polish_max_delta": polish_max_delta,
        "optical_polish_mode": "sparse_interface_residual_v2",
    }


def write_hybrid_inputs(
    output_dir: Path,
    strict_base: np.ndarray,
    allow_mask: np.ndarray,
    guard_mask: np.ndarray,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    strict_path = output_dir / "strict_base.png"
    allow_path = output_dir / "hybrid_rerender_mask.png"
    guard_path = output_dir / "hybrid_identity_guard.png"
    save_png(strict_path, strict_base)
    save_png(allow_path, np.asarray(allow_mask, np.uint8) * 255)
    save_png(guard_path, np.asarray(guard_mask, np.uint8) * 255)
    return strict_path, allow_path, guard_path
