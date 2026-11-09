from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .errors import ContractError
from .material_profiles import ALL_PRODUCT_MATERIAL_MODES, canonical_material_mode
from .identity import validate_identity_status, validate_rerender_policy


CAST_SHADOW_MODES = {"directional", "ambient_only", "disabled"}
SURFACE_MATERIALS = {
    "unknown", "matte", "satin", "polished_opaque", "mirror",
    "transparent_glass",
}
PRODUCT_MATERIAL_MODES = set(ALL_PRODUCT_MATERIAL_MODES)


@dataclass(frozen=True)
class Lighting:
    direction: tuple[float, float]
    shadow_length_ratio: float
    reflection_enabled: bool = False
    reflection_opacity: float = 0.12
    shadow_opacity: float = 0.34
    shadow_softness_ratio: float = 0.08
    shadow_tail_strength: float = 0.38
    cast_shadow_mode: str = "directional"
    surface_material: str = "unknown"
    surface_reflectivity: float = 0.0
    reflection_roughness: float = 0.55
    direct_light_fraction: float = 1.0
    contact_ao_strength: float = 1.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Lighting":
        direction = data.get("shadow_direction")
        if not isinstance(direction, list) or len(direction) != 2:
            raise ContractError("lighting.shadow_direction must be [x, y]")
        vector = np.asarray(direction, dtype=np.float32)
        if not np.isfinite(vector).all() or np.linalg.norm(vector) < 1e-6:
            raise ContractError("lighting.shadow_direction cannot be zero")
        value = float(data.get("shadow_length_ratio", 0.1))
        if not 0 <= value <= 0.5:
            raise ContractError("lighting.shadow_length_ratio must be in [0, 0.5]")
        opacity = float(data.get("reflection_opacity", 0.12))
        if not 0 <= opacity <= 0.35:
            raise ContractError("lighting.reflection_opacity must be in [0, 0.35]")
        shadow_opacity = float(data.get("shadow_opacity", 0.34))
        softness = float(data.get("shadow_softness_ratio", 0.08))
        if not 0.0 <= shadow_opacity <= 0.75:
            raise ContractError("lighting.shadow_opacity must be in [0,0.75]")
        if not 0.01 <= softness <= 0.25:
            raise ContractError("lighting.shadow_softness_ratio must be in [0.01,0.25]")
        tail = float(data.get("shadow_tail_strength", 0.38))
        if not 0 <= tail <= 0.75:
            raise ContractError("lighting.shadow_tail_strength must be in [0,0.75]")
        cast_shadow_mode = str(data.get("cast_shadow_mode", "directional"))
        if cast_shadow_mode not in CAST_SHADOW_MODES:
            raise ContractError(
                f"lighting.cast_shadow_mode must be one of {sorted(CAST_SHADOW_MODES)}"
            )
        surface_material = str(data.get("surface_material", "unknown"))
        if surface_material not in SURFACE_MATERIALS:
            raise ContractError(
                f"lighting.surface_material must be one of {sorted(SURFACE_MATERIALS)}"
            )
        surface_reflectivity = float(data.get("surface_reflectivity", 0.0))
        reflection_roughness = float(data.get("reflection_roughness", 0.55))
        direct_light_fraction = float(data.get("direct_light_fraction", 1.0))
        contact_ao_strength = float(data.get("contact_ao_strength", 1.0))
        if not 0.0 <= surface_reflectivity <= 1.0:
            raise ContractError("lighting.surface_reflectivity must be in [0,1]")
        if not 0.02 <= reflection_roughness <= 1.0:
            raise ContractError("lighting.reflection_roughness must be in [0.02,1]")
        if not 0.0 <= direct_light_fraction <= 1.0:
            raise ContractError("lighting.direct_light_fraction must be in [0,1]")
        if not 0.0 <= contact_ao_strength <= 1.5:
            raise ContractError("lighting.contact_ao_strength must be in [0,1.5]")
        return cls(
            direction=tuple(vector / np.linalg.norm(vector)),
            shadow_length_ratio=value,
            reflection_enabled=bool(data.get("reflection_enabled", False)),
            reflection_opacity=opacity,
            shadow_opacity=shadow_opacity,
            shadow_softness_ratio=softness,
            shadow_tail_strength=tail,
            cast_shadow_mode=cast_shadow_mode,
            surface_material=surface_material,
            surface_reflectivity=surface_reflectivity,
            reflection_roughness=reflection_roughness,
            direct_light_fraction=direct_light_fraction,
            contact_ao_strength=contact_ao_strength,
        )


@dataclass(frozen=True)
class ProductSpec:
    product_id: str
    rgba: Path
    homography: np.ndarray
    z_order: int
    rgb_gain: tuple[float, float, float]
    placement_mode: str
    material_mode: str = "opaque"
    material_confidence: float = 0.0
    transmission_strength: float = 0.0
    semantic_category: str = ""
    physical_type: str = ""
    identity_status: str = "unknown"
    identity_mask: Path | None = None
    rerender_policy: str = "auto"

    @classmethod
    def from_dict(cls, data: dict[str, Any], root: Path) -> "ProductSpec":
        product_id = str(data.get("id", ""))
        if not product_id:
            raise ContractError("Each product requires an id")
        matrix = np.asarray(data.get("homography"), dtype=np.float64)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-10:
            raise ContractError(f"Product {product_id} has an invalid homography")
        raw_path = data.get("rgba")
        if not isinstance(raw_path, str):
            raise ContractError(f"Product {product_id} requires rgba path")
        raw_gain = np.asarray(data.get("rgb_gain", [1, 1, 1]), dtype=np.float32)
        if raw_gain.shape != (3,) or not np.isfinite(raw_gain).all() or np.any(raw_gain < 0.4) or np.any(raw_gain > 1.4):
            raise ContractError(f"Product {product_id} rgb_gain must contain three values in [0.4,1.4]")
        placement_mode = str(data.get("placement_mode", "upright"))
        if placement_mode not in {"upright", "surface_flat"}:
            raise ContractError(
                f"Product {product_id} placement_mode must be upright or surface_flat"
            )
        material_mode = canonical_material_mode(str(data.get("material_mode", "opaque")))
        if material_mode not in PRODUCT_MATERIAL_MODES:
            raise ContractError(
                f"Product {product_id} material_mode must be one of "
                f"{sorted(PRODUCT_MATERIAL_MODES)}"
            )
        material_confidence = float(data.get("material_confidence", 0.0))
        transmission_strength = float(data.get("transmission_strength", 0.0))
        if not 0.0 <= material_confidence <= 1.0:
            raise ContractError(
                f"Product {product_id} material_confidence must be in [0,1]"
            )
        if not 0.0 <= transmission_strength <= 0.98:
            raise ContractError(
                f"Product {product_id} transmission_strength must be in [0,0.98]"
            )
        try:
            identity_status = validate_identity_status(str(
                data.get("identity_status", "unknown"),
            ))
            rerender_policy = validate_rerender_policy(str(
                data.get("rerender_policy", "auto"),
            ))
        except ValueError as exc:
            raise ContractError(f"Product {product_id}: {exc}") from exc
        raw_identity_mask = data.get("identity_mask")
        if raw_identity_mask is not None and not isinstance(raw_identity_mask, str):
            raise ContractError(f"Product {product_id} identity_mask must be a path or null")
        return cls(
            product_id,
            (root / raw_path).resolve(),
            matrix,
            int(data.get("z_order", 0)),
            tuple(float(v) for v in raw_gain),
            placement_mode,
            material_mode,
            material_confidence,
            transmission_strength,
            str(data.get("semantic_category", "")),
            str(data.get("physical_type", "")),
            identity_status,
            (root / raw_identity_mask).resolve() if raw_identity_mask else None,
            rerender_policy,
        )


@dataclass(frozen=True)
class HybridRenderSpec:
    enabled: bool = False
    provider: str = "manual_candidate"
    candidate: Path | None = None
    prompt: str = ""

    @classmethod
    def from_dict(cls, data: Any, root: Path) -> "HybridRenderSpec":
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise ContractError("hybrid_render must be an object")
        provider = str(data.get("provider", "manual_candidate")).strip().lower()
        if provider not in {"manual_candidate", "runninghub"}:
            raise ContractError("hybrid_render.provider must be manual_candidate or runninghub")
        raw_candidate = data.get("candidate")
        if raw_candidate is not None and not isinstance(raw_candidate, str):
            raise ContractError("hybrid_render.candidate must be a path or null")
        return cls(
            enabled=bool(data.get("enabled", False)),
            provider=provider,
            candidate=(root / raw_candidate).resolve() if raw_candidate else None,
            prompt=str(data.get("prompt", ""))[:4000],
        )


@dataclass(frozen=True)
class Job:
    background: Path
    support_polygon: np.ndarray
    support_mask: Path | None
    protected_mask: Path | None
    protected_polygons: tuple[np.ndarray, ...]
    lighting: Lighting
    products: tuple[ProductSpec, ...]
    hybrid_render: HybridRenderSpec = HybridRenderSpec()
    depth_map: Path | None = None
    normal_map: Path | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], root: Path) -> "Job":
        background = data.get("background")
        if not isinstance(background, str):
            raise ContractError("background is required")
        polygon = np.asarray(data.get("support_polygon"), dtype=np.float32)
        if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2 or not np.isfinite(polygon).all():
            raise ContractError("support_polygon must contain at least 3 normalized [x,y] points")
        if np.any(polygon < 0) or np.any(polygon > 1):
            raise ContractError("support_polygon coordinates must be normalized to [0,1]")
        products = tuple(ProductSpec.from_dict(x, root) for x in data.get("products", []))
        if not 1 <= len(products) <= 3:
            raise ContractError("A job must contain one to three products")
        if len({p.product_id for p in products}) != len(products):
            raise ContractError("Product ids must be unique")
        raw_mask = data.get("protected_mask")
        if raw_mask is not None and not isinstance(raw_mask, str):
            raise ContractError("protected_mask must be a path or null")
        raw_support_mask = data.get("support_mask")
        if raw_support_mask is not None and not isinstance(raw_support_mask, str):
            raise ContractError("support_mask must be a path or null")
        raw_depth_map = data.get("depth_map")
        if raw_depth_map is not None and not isinstance(raw_depth_map, str):
            raise ContractError("depth_map must be a path or null")
        raw_normal_map = data.get("normal_map")
        if raw_normal_map is not None and not isinstance(raw_normal_map, str):
            raise ContractError("normal_map must be a path or null")
        polygons: list[np.ndarray] = []
        for index, item in enumerate(data.get("protected_polygons", [])):
            parsed = np.asarray(item, dtype=np.float32)
            if parsed.ndim != 2 or parsed.shape[0] < 3 or parsed.shape[1] != 2 or not np.isfinite(parsed).all() or np.any(parsed < 0) or np.any(parsed > 1):
                raise ContractError(f"protected_polygons[{index}] must be a normalized polygon")
            polygons.append(parsed)
        return cls(
            (root / background).resolve(),
            polygon,
            (root / raw_support_mask).resolve() if raw_support_mask else None,
            (root / raw_mask).resolve() if raw_mask else None,
            tuple(polygons),
            Lighting.from_dict(data.get("lighting", {})),
            products,
            HybridRenderSpec.from_dict(data.get("hybrid_render"), root),
            (root / raw_depth_map).resolve() if raw_depth_map else None,
            (root / raw_normal_map).resolve() if raw_normal_map else None,
        )
