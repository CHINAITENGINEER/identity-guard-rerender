from __future__ import annotations

"""Perceptual calibration for deterministic effect layers.

The geometric reflection renderer owns shape, attachment, reach and receiver
clipping.  This module owns only the final *visible signal*.  Separating those
responsibilities prevents geometry code from accumulating product-specific
opacity boosts.  A future learned reflection/shadow-alpha model can replace
this calibration boundary without touching product pixels or layout logic.
"""

from dataclasses import dataclass

import numpy as np

from .material_profiles import material_profile


@dataclass(frozen=True)
class ReflectionVisibility:
    signal_percentile: float
    target_signal: float
    alpha_scale: float
    active_pixels: int


def calibrate_reflection_visibility(
    reflection_rgb: np.ndarray,
    reflection_alpha: np.ndarray,
    receiver_rgb: np.ndarray,
    receiver_mask: np.ndarray,
    *,
    surface_reflectivity: float,
    roughness: float,
    reflection_opacity: float,
    material_mode: str,
) -> tuple[np.ndarray, ReflectionVisibility]:
    """Make an existing mirror cue readable without changing its geometry.

    The target is expressed in final composited RGB contrast, not raw Alpha.
    Dark merchandise on a white table therefore needs less coverage than pale
    glass on the same table.  Only already-generated reflection pixels are
    rescaled; no new shape, colour, shadow or product pixel is invented.
    """

    alpha = np.asarray(reflection_alpha, dtype=np.float32)
    receiver = np.asarray(receiver_rgb, dtype=np.float32)
    reflected = np.asarray(reflection_rgb, dtype=np.float32)
    active = (
        np.asarray(receiver_mask, dtype=bool)
        & (alpha >= 1.0)
    )
    active_count = int(np.count_nonzero(active))
    if (
        active_count < 16
        or float(surface_reflectivity) < 0.18
        or float(reflection_opacity) <= 0.0
    ):
        return np.clip(np.round(alpha), 0, 255).astype(np.uint8), ReflectionVisibility(
            0.0, 0.0, 1.0, active_count,
        )

    colour_delta = np.mean(np.abs(reflected - receiver), axis=2)
    # What a viewer actually sees after alpha-over, in encoded RGB levels.
    visible_signal = colour_delta * (alpha / 255.0)
    positive = visible_signal[active]
    measured = float(np.percentile(positive, 70.0)) if positive.size else 0.0

    reflectivity = float(np.clip(surface_reflectivity, 0.0, 1.0))
    rough = float(np.clip(roughness, 0.02, 1.0))
    scene_strength = reflectivity * (1.0 - rough * 0.55)
    profile = material_profile(material_mode)
    material_weight = 0.60 + 0.40 * float(np.clip(profile.reflection_gain, 0.0, 1.0))
    # 2.4 encoded levels is a quiet contact cue; a confirmed polished/mirror
    # receiver rises continuously towards an editorially visible 9.6 levels.
    target = (2.4 + 7.2 * scene_strength) * material_weight
    raw_scale = target / max(measured, 0.35)
    # Never chase visibility by producing an opaque upside-down duplicate.
    scale = float(np.clip(raw_scale, 0.85, 3.20))
    calibrated = np.clip(np.round(alpha * scale), 0, 112).astype(np.uint8)
    calibrated[~active] = 0
    return calibrated, ReflectionVisibility(
        signal_percentile=measured,
        target_signal=float(target),
        alpha_scale=scale,
        active_pixels=active_count,
    )
