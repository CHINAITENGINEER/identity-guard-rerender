from __future__ import annotations

"""Commercial-photography material routing for deterministic compositing.

The renderer used to have one opaque branch and four transmissive branches.
That made a painted metal can, matte paper box and glossy ceramic cup share the
same contact/reflection treatment.  These profiles keep the exact product
pixels authoritative while selecting restrained effect-layer behaviour.

Physics remains a safety constraint.  The primary objective here is that the
product stays readable, colourful and orderly at the final viewing size.
"""

from dataclasses import dataclass

import cv2
import numpy as np


TRANSMISSIVE_MATERIAL_MODES = frozenset({
    "clear_glass", "frosted_glass", "colored_glass", "mixed_transparent",
})

OPAQUE_MATERIAL_MODES = frozenset({
    "opaque", "opaque_matte", "opaque_glossy", "metal",
    "wood_paper_fabric",
})

ALL_PRODUCT_MATERIAL_MODES = OPAQUE_MATERIAL_MODES | TRANSMISSIVE_MATERIAL_MODES


@dataclass(frozen=True)
class MaterialRenderProfile:
    contact_gain: float
    cast_gain: float
    reflection_gain: float
    reflection_compression: float
    reflection_shear_limit: float
    reflection_tint_mix: float
    reflection_falloff: float
    reflection_blur_multiplier: float
    colour_gain_limit: float
    glass_visibility_floor: float = 0.0


_PROFILES: dict[str, MaterialRenderProfile] = {
    # ``opaque`` remains the backwards-compatible conservative default.
    # The conservative fallback keeps enough vertical extent that raised
    # handles/spouts enter the reflection after a visible gap instead of being
    # compressed into a second false contact at the body base.
    "opaque": MaterialRenderProfile(0.92, 0.88, 0.40, 0.28, 0.0, 0.105, 1.78, 1.08, 0.05),
    "opaque_matte": MaterialRenderProfile(0.88, 0.84, 0.16, 0.17, 0.0, 0.13, 2.05, 1.28, 0.04),
    "opaque_glossy": MaterialRenderProfile(0.96, 0.88, 0.78, 0.34, 0.0, 0.050, 1.34, 0.92, 0.04),
    "metal": MaterialRenderProfile(0.98, 0.86, 0.88, 0.34, 0.0, 0.030, 1.30, 0.86, 0.03),
    "wood_paper_fabric": MaterialRenderProfile(0.86, 0.82, 0.10, 0.16, 0.0, 0.12, 1.95, 1.35, 0.035),
    # Clear catalogue glass needs a stronger normal-view readability floor
    # than a physically minimal dielectric simulation.  The additional energy
    # is still restricted to measured rims, handles, ribs and the thick base;
    # the broad body remains target-scene transmission rather than white fill.
    "clear_glass": MaterialRenderProfile(0.82, 0.30, 0.70, 0.34, 0.0, 0.030, 1.36, 0.98, 0.0, 24.0),
    "frosted_glass": MaterialRenderProfile(0.72, 0.38, 0.36, 0.20, 0.0, 0.055, 1.82, 1.25, 0.0, 6.0),
    "colored_glass": MaterialRenderProfile(0.82, 0.58, 0.70, 0.34, 0.0, 0.040, 1.36, 0.98, 0.0, 5.0),
    "mixed_transparent": MaterialRenderProfile(0.88, 0.72, 0.80, 0.35, 0.0, 0.040, 1.32, 0.94, 0.025, 4.0),
}


_ALIASES = {
    "solid": "opaque",
    "solid_opaque": "opaque",
    "matte": "opaque_matte",
    "matte_opaque": "opaque_matte",
    "opaque_glossy_dielectric": "opaque_glossy",
    "glossy_dielectric": "opaque_glossy",
    "ceramic": "opaque_glossy",
    "glossy_plastic": "opaque_glossy",
    "metallic": "metal",
    "paper": "wood_paper_fabric",
    "wood": "wood_paper_fabric",
    "fabric": "wood_paper_fabric",
    "textile": "wood_paper_fabric",
}


def canonical_material_mode(material_mode: str) -> str:
    normalized = str(material_mode or "opaque").strip().lower()
    return _ALIASES.get(normalized, normalized if normalized in ALL_PRODUCT_MATERIAL_MODES else "opaque")


def is_transmissive(material_mode: str) -> bool:
    return canonical_material_mode(material_mode) in TRANSMISSIVE_MATERIAL_MODES


def is_opaque(material_mode: str) -> bool:
    return not is_transmissive(material_mode)


def material_profile(material_mode: str) -> MaterialRenderProfile:
    return _PROFILES[canonical_material_mode(material_mode)]


def stabilise_opaque_alpha(alpha: np.ndarray, material_mode: str) -> np.ndarray:
    """Restore solid coverage lost by a soft segmentation matte.

    Anti-aliased edges and genuine holes are kept.  Only pixels several samples
    inside an explicitly opaque material are strengthened.  This prevents an
    otherwise correct RGB product from becoming pale after alpha compositing.
    """
    if is_transmissive(material_mode):
        return alpha.copy()
    source = np.asarray(alpha, dtype=np.uint8)
    active = source >= 16
    if not np.any(active):
        return source.copy()
    padded = cv2.copyMakeBorder(
        active.astype(np.uint8), 1, 1, 1, 1,
        cv2.BORDER_CONSTANT, value=0,
    )
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
    value = source.astype(np.float32) / 255.0
    core_strength = np.clip((distance - 1.0) / 2.25, 0.0, 1.0)
    confidence = np.clip((source.astype(np.float32) - 28.0) / 72.0, 0.0, 1.0)
    blend = core_strength * confidence
    target = np.maximum(np.power(value, 0.58), 0.94 + confidence * 0.06)
    result = value * (1.0 - blend) + target * blend
    result[~active] = 0.0
    return np.clip(np.round(result * 255.0), 0, 255).astype(np.uint8)
