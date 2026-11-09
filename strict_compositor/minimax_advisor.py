from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .errors import ContractError
from .image_io import save_png
from .material_profiles import (
    ALL_PRODUCT_MATERIAL_MODES,
    TRANSMISSIVE_MATERIAL_MODES,
    canonical_material_mode,
)
from .material_router import resolve_material_route
from .quality import VISUAL_SCORE_WEIGHTS, weighted_visual_score
from .settings import get_public_settings, get_secret


PROMPT_VERSION = "scene-advisor-v16-pose-evidence-and-semantic-aspect"
SUBJECT_ROI_PROMPT_VERSION = "subject-roi-v1-single-complete-physical-product"
DEFAULT_BASE_URL = "https://api.minimax.io"
DEFAULT_MODEL = "MiniMax-M3"
ROUTES = {"strict_composite", "restricted_relight", "reject"}
PLACEMENT_MODES = {"upright", "surface_flat"}
MATERIAL_MODES = set(ALL_PRODUCT_MATERIAL_MODES)
TRANSMISSIVE_PRODUCT_MODES = set(TRANSMISSIVE_MATERIAL_MODES)
SURFACE_MATERIALS = {
    "unknown", "matte", "satin", "polished_opaque", "mirror",
    "transparent_glass",
}
CAST_SHADOW_MODES = {"directional", "ambient_only", "disabled"}


def _enum_alias(value: Any, aliases: dict[str, str]) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return aliases.get(normalized, normalized)


def _product_material(value: Any) -> str:
    normalized = _enum_alias(value, {
        "solid": "opaque",
        "solid_opaque": "opaque",
        "matte": "opaque_matte",
        "matte_opaque": "opaque_matte",
        "glossy": "opaque_glossy",
        "glossy_dielectric": "opaque_glossy",
        "opaque_glossy_dielectric": "opaque_glossy",
        "ceramic": "opaque_glossy",
        "glossy_plastic": "opaque_glossy",
        "metallic": "metal",
        "wood": "wood_paper_fabric",
        "paper": "wood_paper_fabric",
        "fabric": "wood_paper_fabric",
        "textile": "wood_paper_fabric",
        "glass": "clear_glass",
        "clear": "clear_glass",
        "transparent": "clear_glass",
        "transparent_glass": "clear_glass",
        "frosted": "frosted_glass",
        "tinted_glass": "colored_glass",
        "amber_glass": "colored_glass",
        "smoked_glass": "colored_glass",
        "mixed": "mixed_transparent",
        "partly_transparent": "mixed_transparent",
        "partially_transparent": "mixed_transparent",
    })
    if normalized in MATERIAL_MODES:
        return normalized
    # OpenAI-compatible relays occasionally return a descriptive compound
    # label even when the prompt asks for an enum.  Keep this conversion
    # deliberately narrow: only unmistakable material words are accepted.
    tokens = set(re.findall(r"[a-z]+", normalized))
    if "frosted" in tokens or "etched" in tokens:
        return "frosted_glass"
    if tokens & {"colored", "coloured", "tinted", "amber", "smoked"}:
        return "colored_glass"
    if tokens & {"mixed", "partial", "partly"} and tokens & {"glass", "transparent"}:
        return "mixed_transparent"
    if tokens & {"clear", "transparent"} and "glass" in tokens:
        return "clear_glass"
    if tokens & {"metal", "metallic", "aluminum", "aluminium", "steel"}:
        return "metal"
    if tokens & {"ceramic", "porcelain", "lacquer", "glossy"}:
        return "opaque_glossy"
    if tokens & {"wood", "paper", "cardboard", "fabric", "textile"}:
        return "wood_paper_fabric"
    if tokens & {"matte"}:
        return "opaque_matte"
    if tokens & {"opaque", "solid", "plastic"}:
        return "opaque"
    return normalized


def _surface_material(value: Any) -> str:
    normalized = _enum_alias(value, {
        "glass": "transparent_glass",
        "clear_glass": "transparent_glass",
        "glass_table": "transparent_glass",
        "glass_tabletop": "transparent_glass",
        "reflective_glass": "transparent_glass",
        "smoked_glass": "transparent_glass",
        "transparent": "transparent_glass",
        "glossy": "polished_opaque",
        "polished": "polished_opaque",
        "reflective": "polished_opaque",
        "polished_stone": "polished_opaque",
        "matte_opaque": "matte",
        "matte_surface": "matte",
        "satin_opaque": "satin",
        "mirror_glass": "mirror",
    })
    if normalized in SURFACE_MATERIALS:
        return normalized
    tokens = set(re.findall(r"[a-z]+", normalized))
    if "mirror" in tokens:
        return "mirror"
    if "glass" in tokens or "transparent" in tokens:
        return "transparent_glass"
    if tokens & {"glossy", "polished", "reflective"}:
        return "polished_opaque"
    if "matte" in tokens:
        return "matte"
    if "satin" in tokens:
        return "satin"
    return normalized


def _cast_mode(value: Any) -> str:
    return _enum_alias(value, {
        "ambient": "ambient_only",
        "contact_only": "ambient_only",
        "ao_only": "ambient_only",
        "reflection_only": "ambient_only",
        "none": "disabled",
        "off": "disabled",
        "no_shadow": "disabled",
        "hard_directional": "directional",
        "soft_directional": "directional",
    })


def _semantic_material_mode(
    material_mode: str,
    semantic_category: str,
    physical_type: str,
) -> str:
    """Resolve common transparent packages without product-id special cases.

    Multimodal models often call a clear vessel ``colored_glass`` because the
    photographed room is visible through it.  Conversely, a labelled bottle
    containing coloured liquid is not uniformly clear or uniformly coloured:
    it is a mixed optical object.  These rules only act on already-transmissive
    advice and require concrete semantic evidence.
    """

    return resolve_material_route(
        material_mode, semantic_category, physical_type,
    ).resolved_mode


def _normalize_base_url(base_url: str) -> str:
    """Return a stable provider root without changing its routing path."""
    raw = str(base_url).strip()
    parts = urlsplit(raw)
    path = re.sub(r"/{2,}", "/", parts.path).rstrip("/")
    if path.lower().endswith("/v1"):
        path = path[:-3] + "/v1"
    return urlunsplit((
        parts.scheme.lower(),
        parts.netloc.lower(),
        path,
        "",
        "",
    ))


def chat_completions_url(base_url: str) -> str:
    """Accept provider roots and OpenAI-compatible ``.../v1`` roots."""
    base = _normalize_base_url(base_url)
    if base.lower().endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


class MiniMaxNotConfigured(RuntimeError):
    pass


class MiniMaxTransportError(RuntimeError):
    """A retryable connection failure before MiniMax produced a response.

    This is deliberately distinct from a valid low score, an API HTTP error,
    or an invalid model payload.  The caller may resume only the unfinished
    review slot after this error, but must never re-sample a valid judgement.
    """

    def __init__(self, message: str, *, attempted_routes: list[str]) -> None:
        super().__init__(message)
        self.attempted_routes = tuple(attempted_routes)


def _minimax_network_mode(settings: dict[str, Any] | None = None) -> str:
    """Resolve the explicit proxy policy without changing the relay endpoint."""

    settings = settings or get_public_settings()
    value = os.environ.get(
        "MINIMAX_NETWORK_MODE",
        str(settings.get("minimax_network_mode", "auto")),
    ).strip().lower()
    return value if value in {"auto", "system_proxy", "direct"} else "auto"


def _post_minimax(
    url: str,
    *,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: tuple[float, float],
    settings: dict[str, Any] | None = None,
) -> tuple[requests.Response, str]:
    """POST to one approved MiniMax/relay URL with a bounded route fallback.

    ``auto`` attempts the Windows/system proxy first, then the same URL with
    ``trust_env=False`` only if a connection-level request exception occurred.
    A normal HTTP response (including 401/429/500) is returned immediately;
    therefore this cannot hide an auth/quota/model response by routing the key
    elsewhere.  The direct route never changes ``url`` or the provider host.
    """

    mode = _minimax_network_mode(settings)
    routes = (
        ["system_proxy", "direct"] if mode == "auto"
        else [mode]
    )
    failures: list[str] = []
    for route in routes:
        try:
            if route == "system_proxy":
                return requests.post(
                    url, headers=headers, json=payload, timeout=timeout,
                ), route
            session = requests.Session()
            session.trust_env = False
            try:
                response = session.post(
                    url, headers=headers, json=payload, timeout=timeout,
                )
            finally:
                session.close()
            return response, route
        except requests.RequestException as exc:
            failures.append(f"{route}:{type(exc).__name__}")
    raise MiniMaxTransportError(
        "MiniMax transport unavailable via " + ", ".join(failures),
        attempted_routes=routes,
    )


@dataclass(frozen=True)
class ProductAdvice:
    product_id: str
    center_x: float
    bottom_y: float
    height_ratio: float
    z_order: int
    footprint_width: float
    source_light_compatibility: float
    placement_mode: str = "upright"
    physical_type: str = "rigid_object"
    estimated_real_height_cm: float = 20.0
    size_confidence: float = 0.5
    viewpoint_compatibility: float = 0.8
    semantic_category: str = "decor_object"
    estimated_height_cm_min: float = 15.0
    estimated_height_cm_max: float = 30.0
    footprint_width_cm: float = 12.0
    visual_salience: float = 0.5
    material_mode: str = "opaque"
    material_confidence: float = 0.0
    transmission_strength: float = 0.0
    photographed_pose: str = "unknown"
    pose_confidence: float = 0.0


@dataclass(frozen=True)
class MiniMaxAdvice:
    support_polygon: list[list[float]]
    protected_polygons: list[list[list[float]]]
    protected_labels: list[str]
    shadow_direction: list[float]
    light_direction: list[float]
    reflectivity: float
    reflection_recommended: bool
    route: str
    confidence: float
    products: list[ProductAdvice]
    warnings: list[str]
    model: str
    request_id: str | None = None
    surface_material: str = "unknown"
    surface_roughness: float = 0.55
    surface_transmittance: float = 0.0
    cast_shadow_mode: str = "directional"
    direct_light_fraction: float = 0.55
    shadow_confidence: float = 0.0
    reflection_confidence: float = 0.0
    receiver_width_or_diameter_cm: float = 100.0
    receiver_size_confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SubjectROI:
    """One complete physical product box in normalized ``xyxy`` coordinates.

    ``fallback_used`` is intentionally explicit.  A deterministic local box is
    useful as a SAM/RunningHub prompt when MiniMax is unavailable, but it must
    never be presented to the cutout chooser as a verified semantic decision.
    """

    bbox: tuple[float, float, float, float]
    subject_label: str
    confidence: float
    included_parts: tuple[str, ...] = ()
    excluded_objects: tuple[str, ...] = ()
    ambiguous: bool = False
    warnings: tuple[str, ...] = ()
    model: str = DEFAULT_MODEL
    request_id: str | None = None
    fallback_used: bool = False
    source: str = "minimax"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def minimax_status() -> dict[str, Any]:
    settings = get_public_settings()
    environment_configured = bool(os.environ.get("MINIMAX_API_KEY", "").strip())
    return {
        "configured": bool(get_secret("minimax_api_key")),
        "model": os.environ.get(
            "MINIMAX_MODEL",
            str(settings.get("minimax_model", DEFAULT_MODEL)),
        ),
        "base_url": os.environ.get(
            "MINIMAX_BASE_URL",
            str(settings.get("minimax_base_url", DEFAULT_BASE_URL)),
        ),
        "key_source": "environment" if environment_configured else "windows_dpapi",
        "network_mode": _minimax_network_mode(settings),
    }


def test_minimax_connection() -> dict[str, Any]:
    status = minimax_status()
    api_key = get_secret("minimax_api_key")
    if not api_key:
        raise MiniMaxNotConfigured("MiniMax API key is not configured")
    selected_base = str(status["base_url"]).rstrip("/")
    official_bases = {"https://api.minimaxi.com", "https://api.minimax.io"}
    candidate_bases = [selected_base]
    # Never send a relay key to an unrelated provider during a retry.
    if selected_base in official_bases:
        candidate_bases.append(
            "https://api.minimax.io"
            if selected_base.endswith("minimaxi.com")
            else "https://api.minimaxi.com"
        )
    failures: list[str] = []
    for base_url in candidate_bases:
        try:
            response, network_route = _post_minimax(
                chat_completions_url(base_url),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                payload={
                    "model": status["model"],
                    "messages": [{"role": "user", "content": "只回复 OK"}],
                    "thinking": {"type": "disabled"},
                    "temperature": 0,
                    "max_completion_tokens": 32,
                    "stream": False,
                },
                timeout=(15, 60),
                settings=get_public_settings(),
            )
            try:
                envelope = response.json()
            except ValueError:
                envelope = {}
        except MiniMaxTransportError as exc:
            failures.append(f"{base_url}: {exc}")
            continue
        error = envelope.get("error") if isinstance(envelope, dict) else None
        message = (
            str(error.get("message", ""))
            if isinstance(error, dict)
            else str((envelope.get("base_resp") or {}).get("status_msg", ""))
        )
        if response.status_code == 200:
            request_id = envelope.get("id")
            return {
                "ok": True,
                "authenticated": True,
                "usable": True,
                "base_url": base_url,
                "model": str(envelope.get("model", status["model"])),
                "request_id": str(request_id) if request_id else None,
                "network_route": network_route,
            }
        if response.status_code == 429 and (
            "usage limit" in message.lower()
            or "credits" in message.lower()
            or "额度" in message
        ):
            return {
                "ok": False,
                "authenticated": True,
                "usable": False,
                "reason": "quota_exhausted",
                "base_url": base_url,
                "model": str(status["model"]),
                "message": "MiniMax 密钥有效，但 Token Plan 额度已耗尽；充值后即可启用场景顾问。",
            }
        failures.append(
            f"{base_url}: HTTP {response.status_code}"
            + (f" ({message[:120]})" if message else "")
        )
    provider_label = "MiniMax 官方中国站和国际站" if len(candidate_bases) > 1 else "当前中转地址"
    raise RuntimeError(f"{provider_label}不可用：" + "；".join(failures))


def _image_data_url(
    path: str | Path,
    max_edge: int = 1400,
    *,
    lossless: bool = False,
    alpha_matte: tuple[int, int, int] = (245, 245, 245),
) -> str:
    """Encode a disposable analysis proxy; final pixels never use this image."""
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source)
        if "A" in image.getbands():
            rgba = image.convert("RGBA")
            background = Image.new("RGB", rgba.size, alpha_matte)
            background.paste(rgba, mask=rgba.getchannel("A"))
            image = background
        else:
            image = image.convert("RGB")
        scale = min(1.0, float(max_edge) / max(image.size))
        if scale < 1.0:
            image = image.resize(
                (
                    max(1, int(round(image.width * scale))),
                    max(1, int(round(image.height * scale))),
                ),
                Image.Resampling.LANCZOS,
            )
        output = io.BytesIO()
        if lossless:
            image.save(output, format="PNG", optimize=True)
            media_type = "image/png"
        else:
            image.save(output, format="JPEG", quality=88, optimize=True)
            media_type = "image/jpeg"
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def _subject_roi_prompt() -> str:
    return """
You are a product-subject bounding-box API. Analyse the supplied source photo
only; never generate or edit pixels. Return exactly one JSON object and no
markdown, prose, reasoning or think tags.

Select exactly ONE intended main merchandise subject. Prefer the clearly
staged/displayed subject nearest the optical centre of the photograph. Use
sharpness, framing, isolation and deliberate product presentation as evidence;
do not simply select the largest region or nearby furniture. If several objects
are visible, never merge them into one box.

The subject_bbox is normalized [x_min,y_min,x_max,y_max], with top-left=(0,0)
and bottom-right=(1,1). It must tightly enclose the COMPLETE physical product,
including every attached or necessary part: handle, spout, lid, cap, base,
transparent rim, cable belonging to the product, and a visible easel/display
stand that physically supports the displayed product. Add a small 2-3 percent
visual safety margin so thin handles and antialiased edges are not clipped.

Exclude all unrelated objects even when close to or partly behind the product:
telephone, separate vase, plant, book, second product, cabinet, wall, tabletop,
floor, source background, cast shadow, mirror image and tabletop reflection.
A reflection or shadow is never a physical product part. A separate vase in
front of a decorative plate is not part of the plate. A display easel directly
supporting a plate is part of that displayed product.

If the main subject is genuinely ambiguous, still choose the best single
candidate but set ambiguous=true and lower confidence. complete_subject must be
true only when the proposed box contains all visible physical parts without
including a second object.

Return this exact schema:
{
  "subject_bbox": [0.20, 0.10, 0.78, 0.94],
  "subject_label": "specific product category",
  "confidence": 0.90,
  "ambiguous": false,
  "complete_subject": true,
  "included_parts": ["body", "handle", "base"],
  "excluded_objects": ["vase", "tabletop", "reflection"],
  "warnings": []
}
""".strip()


def _subject_bbox(value: Any) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ContractError("subject_bbox must be normalized [x_min,y_min,x_max,y_max]")
    try:
        x0, y0, x1, y1 = (float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ContractError("subject_bbox coordinates must be numeric") from exc
    coordinates = (x0, y0, x1, y1)
    if not all(math.isfinite(item) and 0.0 <= item <= 1.0 for item in coordinates):
        raise ContractError("subject_bbox coordinates must be finite values in [0,1]")
    width = x1 - x0
    height = y1 - y0
    if width < 0.04 or height < 0.04:
        raise ContractError("subject_bbox is empty or implausibly small")
    if width * height < 0.004 or width * height > 0.96:
        raise ContractError("subject_bbox has implausible area")
    return coordinates


def _short_string_list(value: Any, name: str, limit: int = 16) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ContractError(f"{name} must be a list")
    return tuple(str(item).strip()[:100] for item in value[:limit] if str(item).strip())


def validate_subject_roi(data: dict[str, Any]) -> SubjectROI:
    """Validate MiniMax's single-product ROI contract without guessing formats."""

    if not isinstance(data, dict):
        raise ContractError("MiniMax subject ROI must be a JSON object")
    if not bool(data.get("complete_subject", False)):
        raise ContractError("subject ROI does not contain the complete physical product")
    label = str(data.get("subject_label", "")).strip()[:120]
    if not label:
        raise ContractError("subject_label must be present")
    confidence = _bounded(data.get("confidence"), "confidence", 0.0, 1.0)
    return SubjectROI(
        bbox=_subject_bbox(data.get("subject_bbox")),
        subject_label=label,
        confidence=confidence,
        included_parts=_short_string_list(data.get("included_parts", []), "included_parts"),
        excluded_objects=_short_string_list(data.get("excluded_objects", []), "excluded_objects"),
        ambiguous=bool(data.get("ambiguous", False)),
        warnings=_short_string_list(data.get("warnings", []), "warnings", limit=8),
    )


def _fallback_subject_roi(
    source: str | Path,
    *,
    reason: str,
    model: str = DEFAULT_MODEL,
) -> SubjectROI:
    """Return a low-confidence centre-saliency prompt box.

    This is not segmentation and is deliberately marked as a fallback.  It is
    only a bounded prompt for a downstream segmenter, never evidence that the
    cutout is clean.  The saliency estimate favours coherent centre components
    while the expansion protects handles, caps and display stands.
    """

    source_path = Path(source).resolve()
    default_bbox = (0.12, 0.06, 0.88, 0.96)
    try:
        with Image.open(source_path) as opened:
            image = np.asarray(ImageOps.exif_transpose(opened).convert("RGB"))
        original_h, original_w = image.shape[:2]
        scale = min(1.0, 640.0 / max(original_h, original_w))
        if scale < 1.0:
            image = cv2.resize(
                image,
                (max(1, int(round(original_w * scale))), max(1, int(round(original_h * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        height, width = image.shape[:2]
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
        border = max(2, int(round(min(height, width) * 0.035)))
        border_pixels = np.concatenate((
            lab[:border].reshape(-1, 3), lab[-border:].reshape(-1, 3),
            lab[:, :border].reshape(-1, 3), lab[:, -border:].reshape(-1, 3),
        ), axis=0)
        background_colour = np.median(border_pixels, axis=0)
        colour_distance = np.linalg.norm(lab - background_colour[None, None, :], axis=2)
        colour_distance /= max(1.0, float(np.percentile(colour_distance, 95)))
        grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        gx = cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3)
        edges = cv2.magnitude(gx, gy)
        edges = cv2.GaussianBlur(edges, (0, 0), 2.2)
        edges /= max(1e-6, float(np.percentile(edges, 97)))
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        centre_prior = np.exp(
            -0.5 * (((xx - (width - 1) * 0.5) / max(1.0, width * 0.34)) ** 2
                    + ((yy - (height - 1) * 0.51) / max(1.0, height * 0.39)) ** 2)
        )
        evidence = np.clip(0.64 * colour_distance + 0.36 * edges, 0.0, 1.5)
        weighted = cv2.GaussianBlur(evidence * (0.48 + 0.52 * centre_prior), (0, 0), 4.0)
        threshold = max(float(np.percentile(weighted, 72)), 0.10)
        mask = (weighted >= threshold).astype(np.uint8) * 255
        kernel_size = max(3, int(round(min(height, width) * 0.018)) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=1)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        candidates: list[tuple[float, int]] = []
        frame_area = float(width * height)
        for component in range(1, count):
            x, y, box_w, box_h, area = (int(v) for v in stats[component])
            area_ratio = area / frame_area
            if area_ratio < 0.003 or area_ratio > 0.82:
                continue
            cx, cy = centroids[component]
            centre_score = float(np.exp(-2.6 * (((cx / width) - 0.5) ** 2 + ((cy / height) - 0.5) ** 2)))
            component_energy = float(weighted[labels == component].mean())
            extent = area / max(1.0, float(box_w * box_h))
            score = 1.8 * component_energy + 0.95 * centre_score + 0.35 * min(1.0, math.sqrt(area_ratio / 0.14)) + 0.15 * extent
            candidates.append((score, component))
        if candidates:
            _, component = max(candidates)
            x, y, box_w, box_h, _ = (int(v) for v in stats[component])
            # Large expansion is intentional: saliency finds the body mass,
            # while thin transparent handles and display stands have weak
            # evidence and need room in the downstream box prompt.
            pad_x = max(5, int(round(box_w * 0.18)))
            pad_y = max(5, int(round(box_h * 0.14)))
            x0 = max(0, x - pad_x) / width
            y0 = max(0, y - pad_y) / height
            x1 = min(width, x + box_w + pad_x) / width
            y1 = min(height, y + box_h + pad_y) / height
            candidate_bbox = (float(x0), float(y0), float(x1), float(y1))
            if (x1 - x0) * (y1 - y0) >= 0.004:
                default_bbox = candidate_bbox
    except (OSError, ValueError, cv2.error):
        pass
    warning = f"MiniMax subject locator degraded to local centre-saliency: {reason}"[:300]
    return SubjectROI(
        bbox=default_bbox,
        subject_label="unknown_central_product",
        confidence=0.24,
        ambiguous=True,
        warnings=(warning,),
        model=model,
        fallback_used=True,
        source="local_center_saliency",
    )


def _subject_roi_cache_key(
    source: Path,
    model: str,
    base_url: str = DEFAULT_BASE_URL,
) -> str:
    stat = source.stat()
    material = "\n".join((
        SUBJECT_ROI_PROMPT_VERSION,
        model,
        f"provider={chat_completions_url(base_url)}",
        f"{source.resolve()}|{stat.st_size}|{stat.st_mtime_ns}",
    ))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:28]


def request_minimax_subject_roi(
    source: str | Path,
    cache_dir: str | Path | None = None,
    *,
    force_refresh: bool = False,
    allow_fallback: bool = True,
) -> SubjectROI:
    """Locate one complete main product in a possibly cluttered source photo.

    Valid MiniMax JSON is cached by source file identity, model, endpoint and
    prompt version. Invalid/corrupt responses and local fallbacks are never
    cached, so a temporary outage cannot become a permanent semantic decision.
    """

    source_path = Path(source).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    settings = get_public_settings()
    api_key = get_secret("minimax_api_key")
    model = os.environ.get(
        "MINIMAX_MODEL", str(settings.get("minimax_model", DEFAULT_MODEL)),
    )
    base_url = _normalize_base_url(os.environ.get(
        "MINIMAX_BASE_URL", str(settings.get("minimax_base_url", DEFAULT_BASE_URL)),
    ))
    cache_path: Path | None = None
    if cache_dir is not None:
        cache_root = Path(cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_path = cache_root / f"subject_roi_{_subject_roi_cache_key(source_path, model, base_url)}.json"
        if cache_path.exists() and not force_refresh:
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if cached.get("prompt_version") == SUBJECT_ROI_PROMPT_VERSION:
                    located = validate_subject_roi(cached["subject_roi"])
                    return replace(
                        located,
                        model=str(cached.get("model", model)),
                        request_id=cached.get("request_id"),
                    )
            except (OSError, ValueError, KeyError, ContractError):
                # A stale/corrupt cache is a miss. Do not delete it here; the
                # source tree may be under forensic or user-managed retention.
                pass
    if not api_key:
        error = MiniMaxNotConfigured("MINIMAX_API_KEY is not configured")
        if not allow_fallback:
            raise error
        return _fallback_subject_roi(source_path, reason=str(error), model=model)

    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": _subject_roi_prompt()},
                {"type": "text", "text": "SOURCE_PRODUCT_PHOTO follows."},
                {"type": "image_url", "image_url": {"url": _image_data_url(source_path, max_edge=1600)}},
            ],
        }],
        "thinking": {"type": "disabled"},
        "temperature": 0.0,
        "top_p": 0.85,
        "max_completion_tokens": 1800,
        "stream": False,
    }
    timeout = float(os.environ.get("MINIMAX_TIMEOUT_SECONDS", "180"))
    try:
        response, _network_route = _post_minimax(
            chat_completions_url(base_url),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            payload=payload,
            timeout=(15, timeout),
            settings=settings,
        )
        response.raise_for_status()
        envelope = response.json()
        text = envelope["choices"][0]["message"]["content"]
        parsed = extract_json_object(str(text))
        located = validate_subject_roi(parsed)
        located = replace(
            located,
            model=str(envelope.get("model", model)),
            request_id=str(envelope.get("id")) if envelope.get("id") else None,
        )
    except (MiniMaxTransportError, requests.RequestException, KeyError, IndexError, ValueError, ContractError) as exc:
        if not allow_fallback:
            raise
        return _fallback_subject_roi(
            source_path,
            reason=f"{type(exc).__name__}: {str(exc)[:180]}",
            model=model,
        )
    if cache_path is not None:
        cache_path.write_text(json.dumps({
            "prompt_version": SUBJECT_ROI_PROMPT_VERSION,
            "model": located.model,
            "provider_endpoint": chat_completions_url(base_url),
            "request_id": located.request_id,
            "subject_roi": parsed,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    return located


def _polygon_area(points: list[list[float]]) -> float:
    array = np.asarray(points, np.float64)
    x, y = array[:, 0], array[:, 1]
    return float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) * 0.5)


def _polygon(value: Any, name: str, min_area: float = 0.001) -> list[list[float]]:
    if not isinstance(value, list) or not 3 <= len(value) <= 32:
        raise ContractError(f"{name} must contain 3 to 32 normalized points")
    parsed: list[list[float]] = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ContractError(f"{name} points must be [x,y]")
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError) as exc:
            raise ContractError(f"{name} coordinates must be numbers") from exc
        if not math.isfinite(x) or not math.isfinite(y) or not 0 <= x <= 1 or not 0 <= y <= 1:
            raise ContractError(f"{name} coordinates must be finite values in [0,1]")
        parsed.append([x, y])
    if _polygon_area(parsed) < min_area:
        raise ContractError(f"{name} has implausibly small area")
    return parsed


def _direction(value: Any, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ContractError(f"{name} must be [x,y]")
    vector = np.asarray(value, np.float64)
    if not np.isfinite(vector).all() or np.linalg.norm(vector) < 1e-6:
        raise ContractError(f"{name} must be a non-zero finite vector")
    vector /= np.linalg.norm(vector)
    return [float(vector[0]), float(vector[1])]


def _bounded(value: Any, name: str, low: float, high: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{name} must be numeric") from exc
    if not math.isfinite(parsed) or not low <= parsed <= high:
        raise ContractError(f"{name} must be in [{low},{high}]")
    return parsed


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract one JSON object from thinking tags or fenced model output."""
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"```(?:json)?", "", cleaned, flags=re.IGNORECASE).replace("```", "")
    decoder = json.JSONDecoder()
    for index, character in enumerate(cleaned):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ContractError("MiniMax response did not contain a valid JSON object")


def validate_advice(data: dict[str, Any], product_ids: list[str] | None = None) -> MiniMaxAdvice:
    support = _polygon(data.get("support_polygon"), "support_polygon", min_area=0.015)
    raw_protected = data.get("protected_regions", data.get("protected_polygons", []))
    if not isinstance(raw_protected, list):
        raise ContractError("protected_regions must be a list")
    protected: list[list[list[float]]] = []
    labels: list[str] = []
    for index, item in enumerate(raw_protected):
        if isinstance(item, dict):
            polygon = item.get("polygon")
            label = str(item.get("label", f"protected_{index + 1}"))[:80]
        else:
            polygon = item
            label = f"protected_{index + 1}"
        protected.append(_polygon(polygon, f"protected_regions[{index}]", min_area=0.0002))
        labels.append(label)

    lighting = data.get("lighting")
    if not isinstance(lighting, dict):
        raise ContractError("lighting must be an object")
    surface = data.get("surface", {})
    if not isinstance(surface, dict):
        raise ContractError("surface must be an object")
    route = str(data.get("route", "strict_composite"))
    if route not in ROUTES:
        raise ContractError(f"route must be one of {sorted(ROUTES)}")

    expected = product_ids or []
    raw_products = data.get("products", [])
    if not isinstance(raw_products, list):
        raise ContractError("products must be a list")
    products: list[ProductAdvice] = []
    seen: set[str] = set()
    for index, item in enumerate(raw_products):
        if not isinstance(item, dict):
            raise ContractError(f"products[{index}] must be an object")
        product_id = str(item.get("product_id", ""))
        if not product_id or product_id in seen:
            raise ContractError("product advice ids must be present and unique")
        if expected and product_id not in expected:
            raise ContractError(f"Unexpected product_id: {product_id}")
        seen.add(product_id)
        placement_mode = str(item.get("placement_mode", "upright"))
        if placement_mode not in PLACEMENT_MODES:
            raise ContractError(
                f"{product_id}.placement_mode must be one of "
                f"{sorted(PLACEMENT_MODES)}"
            )
        semantic_category = str(item.get("semantic_category", "decor_object"))[:80]
        physical_type = str(item.get("physical_type", "rigid_object"))[:80]
        raw_material_mode = _product_material(item.get("material_mode", "opaque"))
        if raw_material_mode not in MATERIAL_MODES:
            raise ContractError(
                f"{product_id}.material_mode must be one of "
                f"{sorted(MATERIAL_MODES)}"
            )
        material_mode = _semantic_material_mode(
            raw_material_mode, semantic_category, physical_type,
        )
        material_confidence = _bounded(
            item.get("material_confidence", 0.0),
            f"{product_id}.material_confidence", 0.0, 1.0,
        )
        transmission_strength = _bounded(
            item.get("transmission_strength", 0.0),
            f"{product_id}.transmission_strength", 0.0, 0.98,
        )
        semantic_tokens = set(re.findall(
            r"[a-z]+", _enum_alias(f"{semantic_category}_{physical_type}", {}),
        ))
        neutral_open_vessel = bool(semantic_tokens & {
            "pitcher", "jug", "carafe", "decanter", "tumbler", "goblet",
        }) and not bool(semantic_tokens & {
            "smoked", "blue", "green", "red", "purple", "violet", "black",
        })
        if material_mode == "clear_glass" and neutral_open_vessel:
            # A colourless open vessel transmits the target scene, never the
            # photographed source room.  This is an adviser contract rather
            # than a request for weak screen-space alpha: the renderer later
            # rebuilds audited rim, rib, handle and thick-base cues from the
            # optical matte, with its own confidence/readability safeguards.
            transmission_strength = max(0.97, transmission_strength)
        elif material_mode != raw_material_mode and material_mode == "mixed_transparent":
            transmission_strength = max(0.42, transmission_strength)
        estimated_height = _bounded(
            item.get("estimated_real_height_cm", 20.0),
            f"{product_id}.estimated_real_height_cm", 2.0, 300.0,
        )
        footprint_width_cm = _bounded(
            item.get("footprint_width_cm", 12.0),
            f"{product_id}.footprint_width_cm", 1.0, 200.0,
        )
        photographed_pose = str(item.get("photographed_pose", "unknown"))
        allowed_poses = {
            "unknown", "standing_on_base", "display_stand", "lying_flat",
            "unsupported_or_ambiguous",
        }
        if photographed_pose not in allowed_poses:
            raise ContractError(
                f"{product_id}.photographed_pose must be one of {sorted(allowed_poses)}"
            )
        pose_confidence = _bounded(
            item.get("pose_confidence", 0.0),
            f"{product_id}.pose_confidence", 0.0, 1.0,
        )
        if pose_confidence >= 0.60:
            if photographed_pose in {"standing_on_base", "display_stand"}:
                placement_mode = "upright"
            elif photographed_pose == "lying_flat":
                placement_mode = "surface_flat"

        semantic_tokens = set(re.findall(
            r"[a-z]+", _enum_alias(f"{semantic_category}_{physical_type}", {}),
        ))
        display_plate = bool(semantic_tokens & {
            "plate", "platter", "tray", "charger",
        }) and photographed_pose == "display_stand" and pose_confidence >= 0.60
        if display_plate:
            # A front-facing plate on a visible easel is an upright display
            # object.  Its physical image height is its diameter, not its few
            # millimetres of material thickness.  Confusing those two axes was
            # the cause of tiny plate-and-stand stickers in batch results.
            estimated_height = max(estimated_height, footprint_width_cm * 0.86)
            physical_type = "rigid_display_object"
        products.append(ProductAdvice(
            product_id=product_id,
            center_x=_bounded(item.get("center_x"), f"{product_id}.center_x", 0.0, 1.0),
            bottom_y=_bounded(item.get("bottom_y"), f"{product_id}.bottom_y", 0.0, 1.0),
            height_ratio=_bounded(item.get("height_ratio"), f"{product_id}.height_ratio", 0.035, 0.65),
            z_order=int(_bounded(item.get("z_order"), f"{product_id}.z_order", 1, max(1, len(expected) or 3))),
            footprint_width=_bounded(item.get("footprint_width", 0.1), f"{product_id}.footprint_width", 0.005, 0.8),
            source_light_compatibility=_bounded(
                item.get("source_light_compatibility", 0.5),
                f"{product_id}.source_light_compatibility", 0.0, 1.0,
            ),
            placement_mode=placement_mode,
            physical_type=physical_type,
            estimated_real_height_cm=estimated_height,
            size_confidence=_bounded(
                item.get("size_confidence", 0.5),
                f"{product_id}.size_confidence", 0.0, 1.0,
            ),
            viewpoint_compatibility=_bounded(
                item.get("viewpoint_compatibility", 0.8),
                f"{product_id}.viewpoint_compatibility", 0.0, 1.0,
            ),
            semantic_category=semantic_category,
            estimated_height_cm_min=_bounded(
                item.get("estimated_height_cm_min", max(2.0, estimated_height * 0.75)),
                f"{product_id}.estimated_height_cm_min", 2.0, 300.0,
            ),
            estimated_height_cm_max=_bounded(
                item.get("estimated_height_cm_max", min(300.0, estimated_height * 1.30)),
                f"{product_id}.estimated_height_cm_max", 2.0, 300.0,
            ),
            footprint_width_cm=footprint_width_cm,
            visual_salience=_bounded(
                item.get("visual_salience", 0.5),
                f"{product_id}.visual_salience", 0.0, 1.0,
            ),
            material_mode=material_mode,
            material_confidence=material_confidence,
            transmission_strength=transmission_strength,
            photographed_pose=photographed_pose,
            pose_confidence=pose_confidence,
        ))
    if expected and seen != set(expected):
        raise ContractError("MiniMax product advice is incomplete")

    warnings_value = data.get("warnings", [])
    if not isinstance(warnings_value, list):
        raise ContractError("warnings must be a list")
    warnings = [str(item)[:300] for item in warnings_value[:20]]
    reflectivity = _bounded(surface.get("reflectivity", 0.0), "surface.reflectivity", 0.0, 1.0)
    raw_surface_material = surface.get(
        "material_class", surface.get("surface_material", "unknown")
    )
    surface_material = _surface_material(raw_surface_material)
    if surface_material not in SURFACE_MATERIALS:
        raise ContractError(
            f"surface.material_class must be one of {sorted(SURFACE_MATERIALS)}; "
            f"received={str(raw_surface_material)[:80]!r}"
        )
    cast_shadow_mode = _cast_mode(lighting.get("cast_shadow_mode", "directional"))
    if cast_shadow_mode not in CAST_SHADOW_MODES:
        raise ContractError(
            f"lighting.cast_shadow_mode must be one of {sorted(CAST_SHADOW_MODES)}"
        )
    return MiniMaxAdvice(
        support_polygon=support,
        protected_polygons=protected,
        protected_labels=labels,
        shadow_direction=_direction(lighting.get("shadow_direction"), "lighting.shadow_direction"),
        light_direction=_direction(lighting.get("light_direction"), "lighting.light_direction"),
        reflectivity=reflectivity,
        reflection_recommended=bool(surface.get("reflection_recommended", reflectivity >= 0.54)),
        route=route,
        confidence=_bounded(data.get("confidence", 0.0), "confidence", 0.0, 1.0),
        products=products,
        warnings=warnings,
        model=DEFAULT_MODEL,
        surface_material=surface_material,
        surface_roughness=_bounded(
            surface.get("roughness", 0.55), "surface.roughness", 0.02, 1.0,
        ),
        surface_transmittance=_bounded(
            surface.get("transmittance", 0.0), "surface.transmittance", 0.0, 1.0,
        ),
        cast_shadow_mode=cast_shadow_mode,
        direct_light_fraction=_bounded(
            lighting.get("direct_light_fraction", 0.55),
            "lighting.direct_light_fraction", 0.0, 1.0,
        ),
        shadow_confidence=_bounded(
            lighting.get("shadow_confidence", 0.0),
            "lighting.shadow_confidence", 0.0, 1.0,
        ),
        reflection_confidence=_bounded(
            surface.get("reflection_confidence", 0.0),
            "surface.reflection_confidence", 0.0, 1.0,
        ),
        receiver_width_or_diameter_cm=_bounded(
            surface.get("receiver_width_or_diameter_cm", 100.0),
            "surface.receiver_width_or_diameter_cm", 20.0, 500.0,
        ),
        receiver_size_confidence=_bounded(
            surface.get("receiver_size_confidence", 0.0),
            "surface.receiver_size_confidence", 0.0, 1.0,
        ),
    )


def _prompt(product_ids: list[str]) -> str:
    product_section = (
        "The following images are products in this exact order: "
        + ", ".join(product_ids) + "."
        if product_ids else
        "Only a background image is provided. Set products to an empty array."
    )
    product_examples = [
        {
            "product_id": product_id,
            # Deliberately non-uniform schema examples: never anchor the model
            # to a ruler-straight, equal-size catalogue row.
            "center_x": [0.24, 0.55, 0.73][index % 3],
            "bottom_y": [0.63, 0.73, 0.57][index % 3],
            "height_ratio": [0.27, 0.18, 0.31][index % 3],
            "z_order": index + 1,
            "footprint_width": 0.10,
            "source_light_compatibility": 0.8,
            "placement_mode": "upright",
            "physical_type": "rigid_vessel",
            "photographed_pose": "standing_on_base",
            "pose_confidence": 0.9,
            "semantic_category": "identify_the_actual_product_category",
            "estimated_real_height_cm": [26.0, 16.5, 30.0][index % 3],
            "estimated_height_cm_min": [22.0, 15.0, 27.0][index % 3],
            "estimated_height_cm_max": [34.0, 18.0, 34.0][index % 3],
            "footprint_width_cm": [16.0, 7.0, 10.0][index % 3],
            "size_confidence": 0.8,
            "viewpoint_compatibility": 0.85,
            "visual_salience": [0.75, 0.42, 0.68][index % 3],
            "material_mode": ["clear_glass", "opaque", "colored_glass"][index % 3],
            "material_confidence": 0.9,
            "transmission_strength": [0.90, 0.0, 0.55][index % 3],
        }
        for index, product_id in enumerate(product_ids)
    ]
    return f"""
You are a scene-analysis API for deterministic product photography compositing.
You only analyse supplied images. Never generate, redraw, or modify any image.
The first image is the target background. {product_section}
Coordinates are normalized: top-left is (0,0), bottom-right is (1,1); positive image Y points downward.
Return exactly one valid JSON object. Do not output analysis, markdown, prose, or think tags.
support_polygon must enclose exactly one real horizontal tabletop/shelf that can support product bases.
  A transparent or reflective glass top is still a valid receiver. Trace its visible outer glass rim/ellipse,
  not the wall, floor, curtain, furniture or reflection seen through the glass. For a partly occluded round
  top, infer only the conservative support interior enclosed by the physical rim; do not expand to the image
  borders. Depth ordering is mandatory: a pedestal, bracket, screw or table leg visible *below/through* the
  glass is part of the table structure and must not enter protected_regions. A floor plant beside or behind
  the table also must not enter protected_regions merely because its 2-D silhouette overlaps the glass.
  Include an item in protected_regions only when its physical contact/base is visibly on top of the receiver.
  When no product images are supplied, judge and describe the receiver itself; do not route=reject merely
  because you imagine unspecified future products might be too large for it.
  protected_regions must cover only real objects resting on or attached to the receiving surface that must remain visible, such as telephones, cables, cloths, cups, books, and existing tabletop decor.
  Never put the tabletop boundary, desk edge, shelf edge, chair, floor, carpet, wall, cabinet, drawer, or other background/behind-surface furniture in protected_regions. The local geometry engine handles support boundaries separately, and a newly placed foreground object may naturally occlude furniture behind the table.
shadow_direction is the image direction from a contact point toward the darker cast shadow; light_direction is its opposite.
Do not assume that every receiver needs a long cast shadow. Independently classify the receiver material and the light field.
For transparent glass tables, reflections normally dominate on the tabletop; a long opaque-looking shadow often falls through
to a lower floor and must not be painted as a black decal on the glass. Choose cast_shadow_mode=ambient_only when there is no
trustworthy attached directional-shadow evidence, while retaining contact occlusion and recommending reflection.
A product body may cross support_polygon, but its base/contact footprint must remain on the support surface.
    Choose an aesthetically balanced product grouping, not a ruler-straight catalogue row. Use asymmetrical triangular,
    shallow-arc or diagonal depth rhythm when the surface permits it. Balance visual mass and negative space, keep coherent
    edge gaps, and use a primary/secondary/tertiary hierarchy. Treat fixed scene elements near the receiver (for example a
    plant, lamp, chair arm, telephone or other strong mass) as part of the composition: normally put the dominant inserted
    mass toward the opposite side and do not crowd all products against existing clutter. Unless the legal receiver is very
    small, the three-product group should occupy a readable but restrained share of the usable receiver width rather than a
    tiny central knot or three evenly spaced stickers. Judge spacing from visible object edges, not bounding-box centres.
    Do not copy numeric positions from the schema example.
    Classify each supplied product's concrete physical_type and photographed pose before placing it. Return photographed_pose
    as exactly one of unknown, standing_on_base, display_stand, lying_flat, unsupported_or_ambiguous, plus pose_confidence.
    A plate, platter or tray photographed front-facing on a visible easel/stand is display_stand and must remain upright;
    its real visible height is approximately its diameter, not its material thickness. A top-down object actually resting on
    its broad face is lying_flat. Never infer lying_flat merely because the category itself is thin.
    Estimate each product's plausible real-world physical height in centimetres. Use product semantics,
    not the alpha silhouette or the source photograph's crop size. At the same receiver depth, products
    with comparable real heights must have comparable projected heights; genuinely smaller products must
    remain smaller. size_confidence states how reliable that semantic size estimate is.
    Also estimate a plausible real receiver width or round-table diameter. Cross-check every apparent product height using
    product_cm / receiver_cm at the same receiver depth. A 25cm pitcher on a 55cm round side table must not be rendered as
    a tiny 12 percent sticker. Give height ranges rather than pretending a category prior is exact.
    Classify material_mode as opaque_matte, opaque_glossy, metal, wood_paper_fabric,
    opaque (only when the finer class is genuinely uncertain), clear_glass,
    frosted_glass, colored_glass, or mixed_transparent. For glass, distinguish
    intrinsic rims/ribs/labels from low-frequency scenery seen through the object. A neutral rim, handle, highlight or upper
    chamber plus a dark horizontal band aligned with the source room is evidence for clear_glass with transmitted source
    scenery, not colored_glass. Warm/amber room lighting alone is not intrinsic tint evidence for an open clear pitcher,
    jug, carafe or tumbler. Use colored_glass only for an intrinsic tint that remains coherent across the glass body.
    transmission_strength describes how much target-background transmission is physically expected; the renderer, not you,
    computes the final pixels.
    viewpoint_compatibility measures whether the product source camera elevation, visible top/bottom plane,
    and base pose can be reconciled with the background camera using only uniform scale, translation and a
    mild perspective warp. Source RGB and object details are immutable: a visibly lifted, tilted or wrongly
    exposed base cannot be repaired by inventing pixels. If viewpoint_compatibility is below 0.45 for any
    product, or its contact plane cannot lie on the receiver surface, route must be reject.
placement_mode must be upright for a rigid object photographed standing on a base, or surface_flat for a soft/flat object photographed lying on a surface.
A leaning soft object may be upright only when a visible background support can physically hold it; otherwise use surface_flat when the source view can plausibly lie on the support plane, or reject.
For surface_flat, height_ratio is the projected on-plane image height, not an imaginary upright height, and the entire silhouette must fit on the support surface.
If the photographed pose cannot be made physically plausible using only scale, mild perspective and colour grading, route must be reject.
route must be one of strict_composite, restricted_relight, reject.

Return this exact schema with valid values:
{{
  "support_polygon": [[0.0,0.0],[1.0,0.0],[1.0,1.0]],
  "protected_regions": [
    {{"label":"telephone","polygon":[[0.0,0.0],[0.1,0.0],[0.1,0.1]]}}
  ],
  "lighting": {{
    "light_direction": [-0.7,0.7],
    "shadow_direction": [0.7,-0.7],
    "cast_shadow_mode": "directional",
    "direct_light_fraction": 0.6,
    "shadow_confidence": 0.8
  }},
  "surface": {{
    "material_class": "matte",
    "reflectivity": 0.0,
    "roughness": 0.8,
    "transmittance": 0.0,
    "reflection_recommended": false,
    "reflection_confidence": 0.8,
    "receiver_width_or_diameter_cm": 100.0,
    "receiver_size_confidence": 0.6
  }},
  "products": {json.dumps(product_examples, ensure_ascii=False)},
  "route": "strict_composite",
  "confidence": 0.8,
  "warnings": []
}}
The only valid surface.material_class values are: unknown, matte, satin, polished_opaque, mirror, transparent_glass.
""".strip()

def _cache_key(
    background: Path,
    products: list[Path],
    model: str,
    base_url: str = DEFAULT_BASE_URL,
    attempt: int = 0,
) -> str:
    # Cache by the effective endpoint rather than the spelling of its root.
    # Thus ``relay/v1`` and ``relay/v1/`` share a cache, while changing relay
    # hosts can never reuse analysis produced by a different provider.
    provider_endpoint = chat_completions_url(base_url)
    entries = [PROMPT_VERSION, model, f"provider={provider_endpoint}", f"attempt={int(attempt)}"]
    for path in [background, *products]:
        stat = path.stat()
        entries.append(f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}")
    return hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest()[:28]


def request_minimax_advice(
    background: str | Path,
    products: list[dict[str, Any]] | None = None,
    cache_dir: str | Path | None = None,
    *,
    force_refresh: bool = False,
    attempt: int = 0,
    retry_context: str = "",
) -> MiniMaxAdvice:
    settings = get_public_settings()
    api_key = get_secret("minimax_api_key")
    if not api_key:
        raise MiniMaxNotConfigured("MINIMAX_API_KEY is not configured")
    background_path = Path(background).resolve()
    product_items = products or []
    product_ids = [str(item["id"]) for item in product_items]
    product_paths = [
        Path(str(item.get("rgba") or item.get("source"))).resolve()
        for item in product_items
    ]
    model = os.environ.get(
        "MINIMAX_MODEL",
        str(settings.get("minimax_model", DEFAULT_MODEL)),
    )
    base_url = _normalize_base_url(os.environ.get(
        "MINIMAX_BASE_URL",
        str(settings.get("minimax_base_url", DEFAULT_BASE_URL)),
    ))
    cache_path: Path | None = None
    if cache_dir is not None:
        cache_root = Path(cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_path = cache_root / (
            f"{_cache_key(background_path, product_paths, model, base_url, attempt)}.json"
        )
        if cache_path.exists() and not force_refresh:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            advice = validate_advice(cached["advice"], product_ids)
            return replace(
                advice,
                model=str(cached.get("model", model)),
                request_id=cached.get("request_id"),
            )

    prompt = _prompt(product_ids)
    if attempt or retry_context:
        prompt += (
            "\nThis is an independent verification pass. Re-examine support-plane boundaries, "
            "real-world product scale, and source/background viewpoint compatibility. Do not copy a "
            "previous answer. Retry context: " + str(retry_context)[:600]
        )
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    content.append({"type": "text", "text": "BACKGROUND_IMAGE follows."})
    content.append({"type": "image_url", "image_url": {"url": _image_data_url(background_path)}})
    for product_id, path in zip(product_ids, product_paths):
        # A text label immediately before every image prevents multimodal
        # relays from confusing product order when several photos look alike.
        content.append({
            "type": "text",
            "text": f"PRODUCT_IMAGE product_id={product_id} follows.",
        })
        content.append({"type": "image_url", "image_url": {"url": _image_data_url(path)}})
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        # Structured extraction does not need chain-of-thought; a relay may
        # otherwise spend its whole budget inside <think> without final JSON.
        "thinking": {"type": "disabled"},
        "temperature": 0.05 if attempt == 0 else 0.15,
        "top_p": 0.9,
        "max_completion_tokens": 16000,
        "stream": False,
    }
    timeout = float(os.environ.get("MINIMAX_TIMEOUT_SECONDS", "180"))
    try:
        response, _network_route = _post_minimax(
            chat_completions_url(base_url),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            payload=payload,
            timeout=(15, timeout),
            settings=settings,
        )
        response.raise_for_status()
        envelope = response.json()
        text = envelope["choices"][0]["message"]["content"]
    except MiniMaxTransportError:
        raise
    except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
        raise RuntimeError(f"MiniMax scene analysis request failed: {exc}") from exc
    try:
        parsed = extract_json_object(str(text))
    except ContractError as exc:
        finish_reason = ((envelope.get("choices") or [{}])[0]).get("finish_reason")
        raise ContractError(
            "MiniMax did not emit the required JSON object "
            f"(finish_reason={finish_reason}, content_chars={len(str(text))})"
        ) from exc
    advice = validate_advice(parsed, product_ids)
    advice = replace(
        advice,
        model=str(envelope.get("model", model)),
        request_id=str(envelope.get("id")) if envelope.get("id") else None,
    )
    if cache_path is not None:
        cache_path.write_text(json.dumps({
            "prompt_version": PROMPT_VERSION,
            "model": advice.model,
            "provider_endpoint": chat_completions_url(base_url),
            "request_id": advice.request_id,
            "advice": parsed,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    return advice


VISUAL_REVIEW_PROMPT_VERSION = (
    "visual-review-v19-clear-glass-analytic-geometry-calibration"
)


def validate_visual_review(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ContractError("MiniMax visual review must be a JSON object")
    subscores = payload.get("subscores")
    if not isinstance(subscores, dict):
        raise ContractError("MiniMax visual review must contain subscores")
    clean_subscores: dict[str, float] = {}
    for name in VISUAL_SCORE_WEIGHTS:
        clean_subscores[name] = _bounded(
            subscores.get(name), f"subscores.{name}", 0.0, 100.0,
        )
    # The model's top-level score and verdict are deliberately ignored. Only
    # six raw dimensions enter the locally specified weighted calculation.
    score = weighted_visual_score(clean_subscores)
    issues = [str(value)[:400] for value in payload.get("issues", []) if str(value).strip()][:8]
    critical = [str(value)[:400] for value in payload.get("critical_failures", []) if str(value).strip()][:6]
    fixes = [str(value)[:400] for value in payload.get("recommended_adjustments", []) if str(value).strip()][:8]
    verdict = "pass" if score >= 90.0 and not critical else "reject"
    return {
        "visual_score": score,
        "verdict": verdict,
        # One slot is diagnostic evidence only; it can never unlock delivery.
        "batch_unlock": False,
        "subscores": clean_subscores,
        "critical_failures": critical,
        "issues": issues,
        "recommended_adjustments": fixes,
        "score_kind": "locally_weighted_raw_dimension_slot",
    }


def _visual_review_same_coordinate_crop_box(
    alpha: np.ndarray,
    frame_size: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    """Return the one canonical crop window used by target and final evidence.

    Review evidence is only a valid transmission correspondence test when the
    untouched target and the final detail are registered to exactly the same
    pixels.  Keep this geometry in one helper so a future padding adjustment
    cannot silently make the two panels show different parts of the scene.
    """
    frame_width, frame_height = frame_size
    if alpha.shape != (frame_height, frame_width):
        return None
    points = np.argwhere(alpha > 4)
    if not points.size:
        return None
    y0, x0 = points.min(axis=0)
    y1, x1 = points.max(axis=0)
    object_w, object_h = int(x1 - x0 + 1), int(y1 - y0 + 1)
    pad_x = max(48, int(round(object_w * 0.55)))
    pad_top = max(36, int(round(object_h * 0.28)))
    pad_bottom = max(56, int(round(object_h * 0.58)))
    return (
        max(0, int(x0) - pad_x),
        max(0, int(y0) - pad_top),
        min(frame_width, int(x1) + pad_x + 1),
        min(frame_height, int(y1) + pad_bottom + 1),
    )


def _visual_review_crop_boxes(
    final_image: str | Path,
    product_count: int,
) -> dict[str, tuple[int, int, int, int]]:
    """Return registered review windows without letting a neighbour dominate.

    The target and final panels must use the same pixels, but a generous
    context window can include the neighbouring product when two valid objects
    stand near one another.  A reviewer then sees a foreign object in an
    ID-bound panel and may incorrectly call that normal context an overlap.
    Limit only the horizontal *context* at the midpoint between two non-
    overlapping alpha bounding boxes.  The product, base contact and vertical
    reflection room are never clipped.
    """
    final_path = Path(final_image).resolve()
    layers = final_path.parent / "layers"
    with Image.open(final_path) as source:
        frame_size = ImageOps.exif_transpose(source).size
    frame_width, frame_height = frame_size
    alpha_by_id: dict[str, np.ndarray] = {}
    object_bounds: dict[str, tuple[int, int, int, int]] = {}
    for index in range(1, product_count + 1):
        product_id = f"product_{index}"
        alpha = cv2.imread(
            str(layers / f"product_{index}_alpha.png"), cv2.IMREAD_GRAYSCALE,
        )
        if alpha is None or alpha.shape != (frame_height, frame_width):
            continue
        points = np.argwhere(alpha > 4)
        if not points.size:
            continue
        y0, x0 = points.min(axis=0)
        y1, x1 = points.max(axis=0)
        alpha_by_id[product_id] = alpha
        object_bounds[product_id] = (int(x0), int(y0), int(x1) + 1, int(y1) + 1)

    boxes: dict[str, tuple[int, int, int, int]] = {}
    for product_id, alpha in alpha_by_id.items():
        base = _visual_review_same_coordinate_crop_box(alpha, frame_size)
        if base is None:
            continue
        left, top, right, bottom = base
        own_left, _, own_right, _ = object_bounds[product_id]
        for other_id, (other_left, _, other_right, _) in object_bounds.items():
            if other_id == product_id:
                continue
            # Stop at the midpoint only for genuinely separate objects.  If
            # bodies really overlap, evidence must show that instead of hiding
            # it with a crop adjustment.
            if other_right <= own_left:
                left = max(left, (other_right + own_left) // 2)
            elif other_left >= own_right:
                right = min(right, (own_right + other_left + 1) // 2)
        # Never trim into the measured alpha footprint, even at a one-pixel
        # gap or image boundary.
        left = min(left, own_left)
        right = max(right, own_right)
        if right <= left:
            left = max(0, own_left - 1)
            right = min(frame_width, own_right + 1)
        boxes[product_id] = (int(left), int(top), int(right), int(bottom))
    return boxes


def _visual_review_detail_crops(
    final_image: str | Path,
    product_count: int,
) -> dict[str, Path]:
    """Create ID-bound final crops so missing layers can never shift product IDs."""
    final_path = Path(final_image).resolve()
    layers = final_path.parent / "layers"
    if not layers.is_dir():
        return []
    with Image.open(final_path) as source:
        final = ImageOps.exif_transpose(source).convert("RGB")
    output_dir = final_path.parent / "review_crops"
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Path] = {}
    boxes = _visual_review_crop_boxes(final_path, product_count)
    for index in range(1, product_count + 1):
        product_id = f"product_{index}"
        alpha_path = layers / f"product_{index}_alpha.png"
        if not alpha_path.is_file():
            continue
        alpha = cv2.imread(str(alpha_path), cv2.IMREAD_GRAYSCALE)
        if alpha is None:
            continue
        box = boxes.get(product_id)
        if box is None:
            continue
        crop_path = output_dir / f"product_{index}_final_detail.png"
        final.crop(box).save(crop_path, format="PNG", optimize=True)
        results[product_id] = crop_path
    # Keep a tiny deterministic manifest next to the lossless crops so the
    # untouched target crop in the evidence board is guaranteed to use the
    # identical, neighbour-safe pixel rectangle.
    (output_dir / "crop_boxes.json").write_text(
        json.dumps({key: list(value) for key, value in boxes.items()}, indent=2),
        encoding="utf-8",
    )
    return results


def _review_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _flatten_review_image(
    path: str | Path,
    matte: tuple[int, int, int],
) -> Image.Image:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGBA")
    background = Image.new("RGBA", image.size, (*matte, 255))
    return Image.alpha_composite(background, image).convert("RGB")


def _clear_glass_intrinsic_geometry_image(path: str | Path) -> Image.Image:
    """Visualize only measured clear-glass identity, not its old backdrop.

    A raw clear-vessel cutout contains the room photographed through the glass.
    Showing that RGB as the canonical visual reference repeatedly caused the
    reviewer to call an old brown wall an intrinsic amber product colour.  This
    deterministic evidence view uses the exact source alpha contour plus
    source-measured horizontal luminance gradients (vertical ribs) and narrow
    contour detail.  Broad horizontal source-room bands therefore disappear,
    while the spout, handle, body silhouette and fluting remain auditable.
    It is review evidence only and never enters final compositing pixels.
    """
    with Image.open(path) as source:
        rgba = np.asarray(ImageOps.exif_transpose(source).convert("RGBA"))
    rgb = rgba[..., :3].astype(np.float32)
    alpha = rgba[..., 3].astype(np.uint8)
    alpha_fraction = alpha.astype(np.float32) / 255.0
    luma = (
        rgb[..., 0] * 0.2126
        + rgb[..., 1] * 0.7152
        + rgb[..., 2] * 0.0722
    )
    gradient_x = cv2.Sobel(
        luma, cv2.CV_32F, 1, 0, ksize=3,
        borderType=cv2.BORDER_REPLICATE,
    )
    occupied_rows = np.where(np.any(alpha > 8, axis=1))[0]
    occupied_height = (
        int(occupied_rows[-1] - occupied_rows[0] + 1)
        if occupied_rows.size else int(alpha.shape[0])
    )
    persistence_window = int(np.clip(round(occupied_height * 0.10), 15, 201))
    if persistence_window % 2 == 0:
        persistence_window += 1
    # Match the production optical field: vertical ribs persist through much
    # of the vessel, whereas an old wall/table boundary is a local horizontal
    # or diagonal event.  Do not display that source-room band as canonical
    # glass identity in the reviewer evidence.
    gradient_x = cv2.boxFilter(
        gradient_x,
        cv2.CV_32F,
        (1, persistence_window),
        normalize=True,
        borderType=cv2.BORDER_REPLICATE,
    )
    nonzero = np.abs(gradient_x)[alpha > 8]
    scale = float(np.percentile(nonzero, 82.0)) if nonzero.size else 1.0
    signed_ribs = np.tanh(gradient_x / max(6.0, scale * 1.10))

    binary = (alpha > 8).astype(np.uint8)
    contour = cv2.morphologyEx(
        binary, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8),
    ).astype(np.float32)
    # Retain short rim/handle highlights only near the exact alpha boundary;
    # a horizontal old-room table band in the interior cannot enter this term.
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    boundary_zone = np.exp(-distance / 3.2) * alpha_fraction
    gradient_y = np.abs(cv2.Sobel(
        luma, cv2.CV_32F, 0, 1, ksize=3,
        borderType=cv2.BORDER_REPLICATE,
    ))
    y_scale_values = gradient_y[(alpha > 8) & (boundary_zone > 0.20)]
    y_scale = (
        float(np.percentile(y_scale_values, 86.0))
        if y_scale_values.size else 1.0
    )
    narrow_highlight = np.clip(
        gradient_y / max(8.0, y_scale), 0.0, 1.0,
    ) * boundary_zone

    gray = np.full(alpha.shape, 128.0, np.float32)
    gray += alpha_fraction * 14.0
    gray += signed_ribs * alpha_fraction * 48.0
    gray += contour * 74.0
    gray += narrow_highlight * 46.0
    gray = np.clip(gray, 36.0, 246.0).astype(np.uint8)
    return Image.fromarray(np.repeat(gray[..., None], 3, axis=2), mode="RGB")


def _review_panel(
    image: Image.Image,
    label: str,
    size: tuple[int, int],
    *,
    matte: tuple[int, int, int],
) -> Image.Image:
    width, height = size
    label_h = 46
    panel = Image.new("RGB", size, matte)
    fitted = ImageOps.contain(image, (width - 20, height - label_h - 16))
    left = (width - fitted.width) // 2
    top = label_h + (height - label_h - fitted.height) // 2
    panel.paste(fitted, (left, top))
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, width, label_h), fill=(20, 24, 30))
    draw.text((14, 8), label, font=_review_font(25), fill=(255, 255, 255))
    return panel


def _visual_review_material_manifest(
    job_payload: dict[str, Any],
    product_ids: list[str],
) -> list[dict[str, Any]]:
    """Extract the renderer's material contract for an auditable review.

    A clear vessel's source cutout necessarily contains its old room through
    the glass.  That scenery is not product identity, so the reviewer needs a
    product-specific contract rather than one generic sentence that can be
    overlooked while comparing several evidence rows.
    """
    def index_records(value: Any) -> dict[str, dict[str, Any]]:
        indexed: dict[str, dict[str, Any]] = {}
        if isinstance(value, list):
            for raw in value:
                if not isinstance(raw, dict):
                    continue
                product_id = str(raw.get("id", raw.get("product_id", ""))).strip()
                if product_id:
                    indexed[product_id] = raw
        return indexed

    # ``products`` is the authoritative renderer contract.  The semantic
    # advisor metadata deliberately lives in ``ui_placements`` because the
    # low-level compositor product schema historically did not carry category
    # or physical-type fields.  Keep the renderer values authoritative while
    # filling only missing/generic metadata from the ID-matched placement.
    by_id = index_records(job_payload.get("products", []))
    placement_by_id = index_records(job_payload.get("ui_placements", []))

    generic_metadata = {
        "", "unknown", "unspecified", "object", "decor_object",
        "rigid_object", "identify_the_actual_product_category",
    }

    def merged_value(
        rendered: dict[str, Any],
        placement: dict[str, Any],
        field: str,
        default: Any,
    ) -> Any:
        if field in rendered and rendered.get(field) not in (None, ""):
            return rendered[field]
        if field in placement and placement.get(field) not in (None, ""):
            return placement[field]
        return default

    def semantic_value(
        rendered: dict[str, Any],
        placement: dict[str, Any],
        field: str,
    ) -> str:
        candidates = [
            str(rendered.get(field, "")).strip().lower(),
            str(placement.get(field, "")).strip().lower(),
        ]
        for candidate in candidates:
            if candidate not in generic_metadata:
                return candidate
        return next((candidate for candidate in candidates if candidate), "")

    manifest: list[dict[str, Any]] = []
    for product_id in product_ids:
        raw = by_id.get(product_id, {})
        placement = placement_by_id.get(product_id, {})
        mode = canonical_material_mode(str(merged_value(
            raw, placement, "material_mode", "opaque",
        )))
        confidence = float(np.clip(
            merged_value(raw, placement, "material_confidence", 0.0),
            0.0, 1.0,
        ))
        transmission = float(np.clip(
            merged_value(raw, placement, "transmission_strength", 0.0),
            0.0, 1.0,
        ))
        semantic_category = semantic_value(raw, placement, "semantic_category")
        physical_type = semantic_value(raw, placement, "physical_type")
        semantic_tokens = set(re.findall(
            r"[a-z]+", f"{semantic_category}_{physical_type}",
        ))
        clear_vessel = bool(semantic_tokens & {
            "pitcher", "jug", "carafe", "decanter", "tumbler", "goblet",
            "vessel",
        })
        if mode in {"clear_glass", "frosted_glass", "colored_glass"}:
            identity_rule = (
                "target-scene transmission is authorized; broad low-frequency "
                "source-room scenery is non-identity. For clear_glass, dark, "
                "brown or amber source-room regions visible through the vessel "
                "are not product colour and must not be requested back unless "
                "the authoritative mode is colored_glass"
            )
            transmission_scope = (
                "the measured low-frequency interior; keep the original rim, ribs, "
                "handle, labels and high-frequency highlights"
            )
            if clear_vessel:
                transmission_scope += (
                    "; same-coordinate target-backdrop contours and colours visible "
                    "through the vessel are expected transmission, not stains or "
                    "source-scene residue"
                )
        elif mode == "mixed_transparent":
            if "whiskey" in semantic_category or "liquor" in semantic_category:
                identity_rule = (
                    "this is a filled amber-liquid glass bottle: label, cap and the "
                    "dense liquid body remain opaque identity; only clear neck, shoulder "
                    "and thin glass-edge regions may receive target-scene transmission"
                )
                transmission_scope = (
                    "clear neck/shoulder/edge glass only; do not require the target "
                    "background to be visibly transmitted through dense amber liquid or "
                    "the printed label"
                )
            else:
                identity_rule = (
                    "only transparent regions may receive target-scene transmission; "
                    "labels, cap, liquid and opaque detail remain identity"
                )
                transmission_scope = (
                    "only locally clear or smooth transmissive regions, never labels, "
                    "caps or opaque identity-bearing detail"
                )
        else:
            identity_rule = "source RGB, label, pattern and opaque detail remain identity"
            transmission_scope = "none"
        manifest.append({
            "product_id": product_id,
            "material_mode": mode,
            "material_confidence": confidence,
            "transmission_strength": transmission,
            "identity_rule": identity_rule,
            "transmission_scope": transmission_scope,
            "semantic_category": semantic_category,
            "physical_type": physical_type,
        })
    return manifest


def _visual_review_material_contract(manifest: list[dict[str, Any]]) -> str:
    lines = [
        "PER-PRODUCT MATERIAL ROUTE (a renderer hypothesis that must be visually audited):"
    ]
    for item in manifest:
        product_id = str(item["product_id"])
        mode = str(item["material_mode"])
        confidence = float(item["material_confidence"])
        transmission = float(item["transmission_strength"])
        rule = str(item["identity_rule"])
        scope = str(item.get("transmission_scope", "none"))
        semantic_category = str(item.get("semantic_category", "") or "unspecified")
        physical_type = str(item.get("physical_type", "") or "unspecified")
        lines.append(
            f"- {product_id}: material_mode={mode}, confidence={confidence:.2f}, "
            f"target_transmission={transmission:.2f}, semantic_category={semantic_category}, "
            f"physical_type={physical_type}. "
            f"{rule}. Transmission scope: {scope}."
        )
    lines.append(
        "For a clear/frosted/colored glass entry, a broad black, amber, gray or "
        "horizontal source-room band seen through the original vessel is explicitly "
        "NOT missing identity and must not lower any subscore or create a critical "
        "failure when it is replaced by the target receiver/wall."
    )
    lines.append(
        "The material route is not a reason to excuse an unreadable product. "
        "In particular, never infer that a clear_glass product is amber or "
        "smoked merely from source-room scenery. "
        "but explicitly flag a wrong route, vanished rim/handle/base, lost label, "
        "washed-out colour, or missing commercial product presence."
    )
    return "\n".join(lines)


def _visual_review_evidence_boards(
    background: str | Path,
    final_image: str | Path,
    products: list[str | Path],
    detail_paths: dict[str, Path],
    declared_shadow_direction: tuple[float, float] | None,
    material_manifest: list[dict[str, Any]] | None = None,
) -> tuple[Path, Path]:
    """Burn ID labels into lossless boards to prevent relay image mixing.

    Transparent products need one extra piece of *truthful* evidence.  A wall
    / tabletop boundary crossing a clear vessel is not automatically a remnant
    from the source room: it can be the target background seen through the
    vessel.  The detail board therefore carries an untouched, same-coordinate
    target-background crop between the canonical cutout and final crop.  Both
    scene crops use the same crop box and neither carries a coloured coverage
    line or other image overlay.  This lets the reviewer compare a real table
    edge or wall boundary directly, without mistaking an evidence annotation
    for a compositing halo.  In addition
    to the compact all-products board, a high-resolution triptych is written
    for each product.  Sending those focus panels to the reviewer prevents a
    downsampled composite from turning a genuine rib, printed label, or glass
    reflection into a guessed defect.
    """
    output_dir = Path(final_image).resolve().parent / "review_crops"
    output_dir.mkdir(parents=True, exist_ok=True)

    context = Image.new("RGB", (1800, 720), (235, 235, 235))
    bg_panel = _review_panel(
        _flatten_review_image(background, (245, 245, 245)),
        "UNTOUCHED_BACKGROUND",
        (900, 720),
        matte=(245, 245, 245),
    )
    final_panel = _review_panel(
        _flatten_review_image(final_image, (245, 245, 245)),
        "FINAL_COMPOSITE",
        (900, 720),
        matte=(245, 245, 245),
    )
    context.paste(bg_panel, (0, 0))
    context.paste(final_panel, (900, 0))
    if declared_shadow_direction is not None:
        draw = ImageDraw.Draw(context)
        dx, dy = declared_shadow_direction
        origin = (1660, 10)
        tip = (int(origin[0] + dx * 34), int(origin[1] + dy * 34))
        draw.line((origin, tip), fill=(255, 90, 70), width=5)
        draw.ellipse((tip[0] - 4, tip[1] - 4, tip[0] + 4, tip[1] + 4), fill=(255, 90, 70))
        draw.text(
            (1260, 9),
            "DECLARED SHARED SHADOW VECTOR",
            font=_review_font(18),
            fill=(255, 90, 70),
        )
    context_path = output_dir / "context_evidence_board.png"
    context.save(context_path, format="PNG", optimize=True)

    row_h = 650
    panel_w = 720
    detail_board = Image.new(
        "RGB", (panel_w * 3, row_h * len(products)), (128, 128, 128),
    )
    manifest_by_id = {
        str(item.get("product_id")): item
        for item in (material_manifest or [])
        if isinstance(item, dict)
    }
    with Image.open(background) as source:
        untouched_background = ImageOps.exif_transpose(source).convert("RGB")
    layers_dir = Path(final_image).resolve().parent / "layers"
    crop_boxes_path = output_dir / "crop_boxes.json"
    review_crop_boxes: dict[str, tuple[int, int, int, int]] = {}
    try:
        raw_crop_boxes = json.loads(crop_boxes_path.read_text(encoding="utf-8"))
        if isinstance(raw_crop_boxes, dict):
            for product_id, value in raw_crop_boxes.items():
                if isinstance(value, list) and len(value) == 4:
                    review_crop_boxes[str(product_id)] = tuple(
                        int(component) for component in value
                    )
    except (OSError, ValueError, TypeError):
        # Evidence remains valid with the legacy crop helper if an old run
        # lacks the manifest.  New runs always write it before this function.
        review_crop_boxes = {}

    def target_backdrop_panel(
        product_id: str,
        size: tuple[int, int],
    ) -> Image.Image:
        """Render an unmodified target crop registered to FINAL_DETAIL."""
        width, height = size
        alpha_path = layers_dir / f"{product_id}_alpha.png"
        alpha = cv2.imread(str(alpha_path), cv2.IMREAD_GRAYSCALE)
        if alpha is None or alpha.shape != (untouched_background.height, untouched_background.width):
            missing = Image.new("RGB", (width, height), (50, 50, 50))
            ImageDraw.Draw(missing).text(
                (28, height // 2 - 16),
                f"TARGET_BACKDROP product_id={product_id} NOT AVAILABLE",
                font=_review_font(21), fill=(255, 255, 255),
            )
            return missing
        box = review_crop_boxes.get(product_id)
        if box is None:
            box = _visual_review_same_coordinate_crop_box(
                alpha, untouched_background.size,
            )
        if box is None:
            missing = Image.new("RGB", (width, height), (50, 50, 50))
            ImageDraw.Draw(missing).text(
                (28, height // 2 - 16),
                f"TARGET_BACKDROP product_id={product_id} EMPTY COVERAGE",
                font=_review_font(21), fill=(255, 255, 255),
            )
            return missing
        # Do not draw the alpha contour into this crop.  Reviewers previously
        # confused the cyan evidence line with a generated edge halo and could
        # also mistake the target table boundary inside that contour for old
        # source-room RGB.  Registration plus the labels provides the needed
        # correspondence without altering a single evidence pixel.
        crop = untouched_background.crop(box).convert("RGB")
        return _review_panel(
            crop,
            f"TARGET_BACKDROP product_id={product_id} [RAW/NO OVERLAY]",
            (width, height), matte=(245, 245, 245),
        )

    def triptych(
        index: int,
        product: str | Path,
        size: tuple[int, int],
    ) -> Image.Image:
        """Return an ID-bound canonical / target / final evidence strip."""
        width, height = size
        product_id = f"product_{index}"
        material_mode = str(
            manifest_by_id.get(product_id, {}).get("material_mode", "opaque")
        ).upper()
        canonical_mode_label = material_mode
        canonical_image = _flatten_review_image(product, (128, 128, 128))
        if material_mode == "CLEAR_GLASS":
            canonical_mode_label = "CLEAR_GLASS INTRINSIC ALPHA/RIB GEOMETRY"
            canonical_image = _clear_glass_intrinsic_geometry_image(product)
        canonical = _review_panel(
            canonical_image,
            f"CANONICAL_PRODUCT product_id={product_id} [{canonical_mode_label}]",
            (width, height),
            matte=(128, 128, 128),
        )
        detail_path = detail_paths.get(product_id)
        if detail_path is None:
            detail = Image.new("RGB", (width, height), (50, 50, 50))
            ImageDraw.Draw(detail).text(
                (40, height // 2 - 16),
                f"FINAL_DETAIL product_id={product_id} NOT AVAILABLE",
                font=_review_font(28),
                fill=(255, 255, 255),
            )
        else:
            detail = _review_panel(
                _flatten_review_image(detail_path, (245, 245, 245)),
                f"FINAL_DETAIL product_id={product_id} [SAME CROP]",
                (width, height),
                matte=(245, 245, 245),
            )
        strip = Image.new("RGB", (width * 3, height), (128, 128, 128))
        strip.paste(canonical, (0, 0))
        strip.paste(target_backdrop_panel(product_id, (width, height)), (width, 0))
        strip.paste(detail, (width * 2, 0))
        return strip

    for index, product in enumerate(products, 1):
        top = (index - 1) * row_h
        detail_board.paste(triptych(index, product, (panel_w, row_h)), (0, top))
    detail_board_path = output_dir / "id_bound_detail_evidence_board.png"
    detail_board.save(detail_board_path, format="PNG", optimize=True)
    # Product-specific boards are intentionally individual files.  They are
    # appended as separate message images below so image delivery systems cannot
    # silently downsample a tall multi-product board into unreadable fine detail.
    for index, product in enumerate(products, 1):
        focus_path = output_dir / f"product_{index}_material_focus.png"
        triptych(index, product, (1040, 900)).save(
            focus_path, format="PNG", optimize=True,
        )
    return context_path, detail_board_path


def request_minimax_visual_review(
    background: str | Path,
    final_image: str | Path,
    products: list[str | Path],
    *,
    slot_id: int | None = None,
) -> dict[str, Any]:
    settings = get_public_settings()
    api_key = get_secret("minimax_api_key")
    if not api_key:
        raise MiniMaxNotConfigured("MINIMAX_API_KEY is not configured")
    model = os.environ.get("MINIMAX_MODEL", str(settings.get("minimax_model", DEFAULT_MODEL)))
    product_ids = [f"product_{index}" for index in range(1, len(products) + 1)]
    declared_shadow_direction: tuple[float, float] | None = None
    declared_cast_shadow_mode = "directional"
    declared_surface_material = "unknown"
    declared_reflectivity: float | None = None
    declared_roughness: float | None = None
    declared_direct_fraction: float | None = None
    job_payload: dict[str, Any] = {}
    job_path = Path(final_image).resolve().parent / "job.json"
    if job_path.is_file():
        try:
            job_payload = json.loads(job_path.read_text(encoding="utf-8"))
            if not isinstance(job_payload, dict):
                job_payload = {}
            job_lighting = job_payload.get("lighting", {})
            declared_cast_shadow_mode = str(
                job_lighting.get("cast_shadow_mode", "directional")
            )
            declared_surface_material = str(
                job_lighting.get("surface_material", "unknown")
            )
            try:
                declared_reflectivity = float(np.clip(
                    job_lighting.get("surface_reflectivity"), 0.0, 1.0,
                ))
            except (TypeError, ValueError):
                declared_reflectivity = None
            try:
                declared_roughness = float(np.clip(
                    job_lighting.get("reflection_roughness"), 0.0, 1.0,
                ))
            except (TypeError, ValueError):
                declared_roughness = None
            try:
                declared_direct_fraction = float(np.clip(
                    job_lighting.get("direct_light_fraction"), 0.0, 1.0,
                ))
            except (TypeError, ValueError):
                declared_direct_fraction = None
            raw_direction = job_lighting.get(
                "shadow_direction",
            )
            if (
                isinstance(raw_direction, list)
                and len(raw_direction) == 2
                and all(isinstance(value, (int, float)) for value in raw_direction)
            ):
                declared_shadow_direction = (
                    float(raw_direction[0]),
                    float(raw_direction[1]),
                )
            if declared_cast_shadow_mode != "directional":
                declared_shadow_direction = None
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            job_payload = {}
            declared_shadow_direction = None
    material_manifest = _visual_review_material_manifest(job_payload, product_ids)
    material_contract = _visual_review_material_contract(material_manifest)
    placement_by_id = {
        str(item.get("id", item.get("product_id", ""))): item
        for item in job_payload.get("ui_placements", [])
        if isinstance(item, dict)
    }
    geometry_evidence_lines = [
        "DISPLAYED GEOMETRY SANITY EVIDENCE (use for ID/scale verification, not as an automatic visual score):"
    ]
    for product_id in product_ids:
        placement = placement_by_id.get(product_id, {})
        if not placement:
            continue
        try:
            height_ratio = float(placement.get("height_ratio"))
        except (TypeError, ValueError):
            continue
        real_height = placement.get("estimated_real_height_cm")
        geometry_evidence_lines.append(
            f"- {product_id}: displayed_height_ratio={height_ratio:.4f}, "
            f"estimated_real_height_cm={real_height}."
        )
    if len(geometry_evidence_lines) == 1:
        geometry_evidence_lines.append("- no run-local placement evidence available")
    geometry_evidence = "\n".join(geometry_evidence_lines)

    physical_evidence_lines = [
        "DETERMINISTIC EFFECT LOCALIZATION (not a subjective quality score; inspect the named pixels in FINAL_DETAIL):"
    ]
    audit_path = Path(final_image).resolve().parent / "audit.json"
    try:
        audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
        physical_products = audit_payload.get("physical_audit", {}).get("products", [])
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        physical_products = []
    for physical_product in physical_products:
        if not isinstance(physical_product, dict):
            continue
        product_id = str(physical_product.get("product_id", ""))
        if product_id not in product_ids:
            continue
        visibility = physical_product.get("final_pixel_visibility", {})
        contact = visibility.get("contact", {}) if isinstance(visibility, dict) else {}
        reflection = visibility.get("reflection", {}) if isinstance(visibility, dict) else {}
        physical_evidence_lines.append(
            f"- {product_id}: contact_status={contact.get('status', 'unknown')}, "
            f"contact_visible_fraction={contact.get('visible_pixel_fraction', 'unknown')}, "
            f"contact_rgb_delta_p75={contact.get('observed_rgb_delta_p75', 'unknown')}; "
            f"reflection_status={reflection.get('status', 'unknown')}, "
            f"reflection_visible_fraction={reflection.get('visible_pixel_fraction', 'unknown')}, "
            f"reflection_rgb_delta_p75={reflection.get('observed_rgb_delta_p75', 'unknown')}."
        )
    if len(physical_evidence_lines) == 1:
        physical_evidence_lines.append("- no deterministic localization record available")
    physical_evidence_lines.append(
        "A visible status proves only that changed receiver pixels exist and tells you where to look; "
        "it does not prove that the effect is strong or attractive. You may penalize a visibly weak or "
        "implausible effect, but do not report it as absent when the same-ID detail visibly contains it."
    )
    physical_evidence = "\n".join(physical_evidence_lines)
    receiver_evidence = (
        "RECEIVER MATERIAL EVIDENCE: this is a glossy/smoked transparent-glass "
        f"tabletop with reflectivity={declared_reflectivity:.2f} and "
        f"roughness={declared_roughness:.2f}. A visibly attached, vertically "
        "compressed and softly filtered mirror image beneath opaque or amber "
        "products is physically expected. Do not call that reflection artificial "
        "solely because the receiver is transparent glass; audit only whether it "
        "is attached at the contact contour, correctly contained by the tabletop "
        "and plausibly attenuated."
        if (
            declared_surface_material == "transparent_glass"
            and declared_reflectivity is not None
            and declared_roughness is not None
        )
        else
        "RECEIVER MATERIAL EVIDENCE: judge reflection strength from the supplied "
        "background and the declared receiver material; do not assume that every "
        "transparent receiver is matte or that every matte receiver is reflective."
    )
    direction_evidence = (
        f"The job declares cast_shadow_mode={declared_cast_shadow_mode} on a "
        f"surface_material={declared_surface_material} receiver. Absence of a long "
        "directional projection is correct in this mode; score cast_shadows by whether "
        "the renderer avoids an artificial dark decal while retaining believable contact "
        "occlusion. Do not demand a tail or penalize its absence."
        if declared_cast_shadow_mode != "directional"
        else
        "The deterministic job manifest declares one shared image-space cast-shadow "
        f"vector for every product: dx={declared_shadow_direction[0]:.6f}, "
        f"dy={declared_shadow_direction[1]:.6f}. Treat this as a hypothesis to audit "
        "against FINAL_COMPOSITE and each FINAL_DETAIL, not as proof of correctness. "
        "An asymmetric object can shift a shadow centroid toward its spout, handle or "
        "strap even when its projected rays share one vector; compare attached rays, "
        "tail direction and height-proportional reach. Penalize a mismatch when the "
        "visible projection contradicts the declared vector, but do not infer multiple "
        "lights from centroid position alone."
        if declared_shadow_direction is not None
        else
        "No trustworthy shared-shadow vector was found in the run manifest; infer cast "
        "direction strictly from visible attached rays and tails."
    )
    prompt = f"""
You are the final quality gate for deterministic product-photo compositing.
The authoritative product IDs for this review are: {", ".join(product_ids)}.
    CONTEXT_EVIDENCE_BOARD contains UNTOUCHED_BACKGROUND and FINAL_COMPOSITE side by side. ID_BOUND_DETAIL_EVIDENCE_BOARD contains one burned-in labelled row per product: CANONICAL_PRODUCT on the left, TARGET_BACKDROP in the middle, and FINAL_DETAIL on the right with the exact same product_id. Each MATERIAL_FOCUS_BOARD below repeats exactly one such product triptych at higher resolution. TARGET_BACKDROP and FINAL_DETAIL use exactly the same pixel crop window before panel fitting. TARGET_BACKDROP is the raw untouched target crop: it has NO cyan coverage line, NO product outline and NO image overlay. Compare real scene landmarks such as a tabletop edge, wall boundary or receiver highlight directly between the middle and right panels. Product IDs come only from the labels burned into the board. Never infer or renumber an ID from object position or appearance. Never transfer a defect observed on one product to another product.
For opaque, mixed-transparent, frosted or colored products, CANONICAL_PRODUCT is the original RGBA cutout rendered losslessly on a neutral mid-gray evidence matte; the gray matte is not part of the product. For a product whose authoritative mode is CLEAR_GLASS, CANONICAL_PRODUCT is instead a deterministic INTRINSIC ALPHA/RIB GEOMETRY view: it uses the exact original alpha, silhouette and source-measured vertical rib/contour gradients, but deliberately excludes broad low-frequency old-room RGB visible through the glass. Gray inside this evidence view is a diagrammatic carrier, not product opacity or color. Use that panel to audit spout, rim, ribs, handle and silhouette; never infer amber/smoked color from it. Each FINAL_DETAIL is a lossless crop of FINAL_COMPOSITE, supplied only so contact seams, cast shadows and edge quality are not lost when the full 4K frame is downscaled. Use it as evidence for the same product, not as an additional object. If a contact shadow is visible in a detail crop, do not call it absent merely because it is subtle in the full-frame proxy.
Do not penalize removal of scenery or non-target objects outside those canonical cutouts. Judge whether each canonical cutout's identity, edge and physical integration are preserved in the composite.
Deterministic resizing, mild perspective, exposure, white balance and saturation correction are explicitly authorized. Never call a product incomplete merely because it is smaller than its canonical cutout; compare silhouette parts and internal details at their displayed scale. Do not claim that a feature is missing when it is visibly present in the composite.
For a product classified as clear/frosted/colored glass, deterministic optical recomposition is also authorized: the original alpha coverage, rim, ribs, labels and high-frequency identity must remain, but low-frequency scenery seen through the object in the source photo must be replaced by transmission of the target scene. Do not demand preservation of trapped source-room black bands; do penalize lost labels, erased rims, invented texture or implausible opaque glass.
{material_contract}
{geometry_evidence}
{physical_evidence}
Apply that manifest entry-by-entry. A correct clear-glass result must look like the TARGET receiver/wall is visible through it, rather than like a dark source-room sticker. Never award preservation credit, or impose a penalty, for copying source scenery back into a transmissive product.
For a transmissive product, use the same-ID TARGET_BACKDROP panel as the correspondence test. The middle and right panels are registered raw crops, not annotations. A high-contrast wall/tabletop/receiver boundary that is present at the same coordinates in TARGET_BACKDROP and appears through FINAL_DETAIL (allowing a small deterministic lens displacement) is target-scene transmission, not a source-room band, and must not lower a subscore or be called a critical failure. In particular, do not call a target tabletop edge a retained black source band when that edge is visibly present in TARGET_BACKDROP. Only call a dark band residual when it appears exclusively inside the product and does not correspond to the target backdrop after allowing for refraction. Conversely, do not excuse a genuine source-colored stain that fails this correspondence test.
IMPORTANT CLEAR-GLASS CALIBRATION: the CLEAR_GLASS INTRINSIC ALPHA/RIB GEOMETRY panel is an analytical guide, not a brightness target. Target-scene transmission may be much lower contrast than the guide, but the product must still read immediately as the same sellable vessel at the full-frame viewing size. Penalize a vanished or ghost-like rim, spout, handle, fluting or thick base; penalize lost commercial presence even when the target pixels are physically plausible. Also penalize a broad opaque veil, source-coloured residue, or a landmark that fails same-coordinate correspondence. A material declaration never excuses invisibility.
The same-coordinate TARGET_BACKDROP check also applies to grounding and shadows for EVERY material. This target is a transparent smoked-glass table with a real dark circular mounting plate, metal fasteners and dark stand/leg already visible through the receiver. Those pre-existing dark disks, legs and hardware reflections are background pixels, not generated cast shadows, contact blobs or reflection misalignment. Before reporting a hard smear, detached shadow or wall residual, verify that the criticized feature is newly present in FINAL_DETAIL and absent at the same coordinates in TARGET_BACKDROP. A target feature that remains unchanged outside the authorized effect mask must never be charged to the compositor.
On a reflective receiver, reflection strength is material-dependent: an opaque can, amber bottle and clear ribbed pitcher should not have numerically equal reflection opacity. Audit contact-contour attachment, direction, falloff and material plausibility; do not penalize physically weaker clear-glass transmission merely because an adjacent opaque product reflects more strongly.
For edge scoring, compare each FINAL_DETAIL only with the CANONICAL_PRODUCT carrying the same product_id. Pale, white, translucent, frosted or reflective material already present at the same source boundary is intrinsic product appearance and is not by itself a white halo. Report a white/black fringe or hard halo only when the final visibly introduces an extra opaque or bright/dark contour, color spill, jagged residue, or background contamination that is absent from the corresponding canonical boundary. This fairness rule must not excuse a real new fringe, lost alpha detail, cutout hole, source-background remnant, or hard pasted edge.
For CLEAR_GLASS specifically, a narrow bright line exactly following the intrinsic alpha/rim/handle geometry is a physically expected Fresnel/specular cue. Do not call it a halo merely because the target wall is darker. Call a halo only when brightness spills outside the measured alpha or forms a materially wider opaque band than the same-ID intrinsic geometry panel.
    {direction_evidence}
    {receiver_evidence}
Judge only visible realism. Be strict: a raw dimension score of 90 means production-ready and difficult to distinguish from a real photograph.
Return each of the following six subscores as an independent RAW score from 0 to 100. Do not multiply by weights, do not make the six values sum to 100, and do not output an overall score or verdict. The server computes the total locally using these weights: grounding/contact shadows 25%, cast-shadow direction/height/softness 20%, perspective/scale/layout 20%, lighting/color match 15%, cutout edges/identity 10%, reflection/material interaction 10%.
Any obvious floating object, wrong support surface, hard halo, implausible scale, inconsistent shadow direction, or visibly pasted look is a critical failure.
For a matte or weakly reflective support, judge whether the composite correctly avoids an artificial mirror image; do not deduct points merely because no reflection was added.
Return exactly one JSON object and no prose:
{{"subscores":{{"grounding":0,"cast_shadows":0,"geometry_layout":0,"lighting_color":0,"cutout_edges":0,"reflection_material":0}},"critical_failures":[],"issues":[],"recommended_adjustments":[]}}
""".strip()
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    detail_paths = _visual_review_detail_crops(final_image, len(products))
    evidence_mapping: list[dict[str, Any]] = []
    for index, value in enumerate(products, 1):
        product_id = f"product_{index}"
        canonical_label = f"CANONICAL_PRODUCT product_id={product_id}"
        detail_label = f"FINAL_DETAIL product_id={product_id} [SAME CROP]"
        detail_path = detail_paths.get(product_id)
        evidence_mapping.append({
            "product_id": product_id,
            "canonical_label": canonical_label,
            "target_label": (
                f"TARGET_BACKDROP product_id={product_id} [RAW/NO OVERLAY]"
                if detail_path is not None else None
            ),
            "detail_label": detail_label if detail_path is not None else None,
            "detail_available": detail_path is not None,
        })
    context_board, detail_board = _visual_review_evidence_boards(
        background,
        final_image,
        products,
        detail_paths,
        declared_shadow_direction,
        material_manifest,
    )
    for label, path in (
        ("CONTEXT_EVIDENCE_BOARD", context_board),
        ("ID_BOUND_DETAIL_EVIDENCE_BOARD", detail_board),
    ):
        content.append({"type": "text", "text": f"{label} follows."})
        content.append({
            "type": "image_url",
            "image_url": {
                "url": _image_data_url(
                    path,
                    lossless=True,
                    alpha_matte=(245, 245, 245),
                ),
            },
        })
    for index, product_id in enumerate(product_ids, 1):
        focus_path = detail_board.parent / f"product_{index}_material_focus.png"
        if not focus_path.is_file():
            continue
        content.append({
            "type": "text",
            "text": (
                f"MATERIAL_FOCUS_BOARD product_id={product_id} follows. "
                "Use this high-resolution triptych to verify optical transmission, "
                "printed detail, alpha fringe and contact/reflection before claiming "
                "a defect."
            ),
        })
        content.append({
            "type": "image_url",
            "image_url": {
                "url": _image_data_url(
                    focus_path,
                    # A 3-panel focus strip must retain at least roughly one
                    # thousand pixels per product panel.  The ordinary 1400px
                    # proxy cap is appropriate for a whole-scene board but
                    # would reduce a 3120px focus strip to ~466px per panel,
                    # making real glass ribs and label edges impossible to
                    # distinguish from a guessed halo.
                    max_edge=3200,
                    lossless=True,
                    alpha_matte=(245, 245, 245),
                ),
            },
        })
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "thinking": {"type": "disabled"},
        "temperature": 0,
        "top_p": 0.9,
        "max_completion_tokens": 8000,
        "stream": False,
    }
    base_url = os.environ.get("MINIMAX_BASE_URL", str(settings.get("minimax_base_url", DEFAULT_BASE_URL))).rstrip("/")
    timeout = float(os.environ.get("MINIMAX_TIMEOUT_SECONDS", "180"))
    try:
        response, network_route = _post_minimax(
            chat_completions_url(base_url),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            payload=payload,
            timeout=(15, timeout),
            settings=settings,
        )
        response.raise_for_status()
        envelope = response.json()
        text = envelope["choices"][0]["message"]["content"]
        parsed = extract_json_object(str(text))
    except MiniMaxTransportError:
        raise
    except (requests.RequestException, KeyError, IndexError, ValueError, ContractError) as exc:
        raise RuntimeError(f"MiniMax visual review failed: {exc}") from exc
    result = validate_visual_review(parsed)
    result.update({
        "model": str(envelope.get("model", model)),
        "request_id": str(envelope.get("id")) if envelope.get("id") else None,
        "prompt_version": VISUAL_REVIEW_PROMPT_VERSION,
        "raw_model_result": parsed,
        "evidence_policy": (
            "two_lossless_boards_plus_per_product_raw_same_coordinate_"
            "target_backdrop_reference"
        ),
        "evidence_mapping": evidence_mapping,
        "material_manifest": material_manifest,
        "network_route": network_route,
        "declared_shadow_direction": (
            list(declared_shadow_direction)
            if declared_shadow_direction is not None else None
        ),
    })
    if slot_id is not None:
        result["slot_id"] = int(slot_id)
    return result

def _polygon_mask(shape: tuple[int, int], polygon: list[list[float]]) -> np.ndarray:
    height, width = shape
    points = np.asarray(polygon, np.float32).copy()
    points[:, 0] *= max(1, width - 1)
    points[:, 1] *= max(1, height - 1)
    mask = np.zeros((height, width), np.uint8)
    cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 1)
    return mask.astype(bool)


def reconcile_scene_result(
    local: dict[str, Any],
    advice: MiniMaxAdvice,
    image_shape: tuple[int, int],
    cache_dir: str | Path,
    advisor_support_consensus: float | None = None,
) -> dict[str, Any]:
    """Accept only M3 observations that survive deterministic geometry gates."""
    height, width = image_shape
    result = dict(local)
    local_mask = _polygon_mask((height, width), local["support_polygon"])
    advisor_mask = _polygon_mask((height, width), advice.support_polygon)
    union = np.count_nonzero(local_mask | advisor_mask)
    overlap = float(np.count_nonzero(local_mask & advisor_mask) / max(1, union))
    local_confidence = float(local.get("confidence", 0.0))
    # A language/vision adviser often returns a coarse quadrilateral that is
    # semantically a "table" but crosses the actual photographed front or side
    # edge.  It must never replace a high-confidence depth/semantic plane just
    # because the two regions have a moderate IoU.  Adviser geometry is only a
    # recovery path for weak local detection; otherwise it remains evidence
    # for semantics, lighting and protected objects.
    local_agreement_accept = bool(
        local_confidence < 0.80
        and advice.confidence >= 0.78
        and overlap >= 0.68
    )
    independent_cloud_recovery = bool(
        local_confidence < 0.60
        and advice.confidence >= 0.82
        and advisor_support_consensus is not None
        and advisor_support_consensus >= 0.58
        and overlap >= 0.04
    )
    support_accepted = local_agreement_accept or independent_cloud_recovery
    warnings = list(advice.warnings)
    if support_accepted:
        result["support_polygon"] = advice.support_polygon
        result["front_edge_y"] = float(max(point[1] for point in advice.support_polygon))
        support_mask = advisor_mask
        mask_path = Path(cache_dir) / f"minimax_reconciled_{hashlib.sha256(str(advice.request_id).encode()).hexdigest()[:16]}.png"
        save_png(mask_path, support_mask.astype(np.uint8) * 255)
        result["support_mask_path"] = str(mask_path.resolve())
    else:
        support_mask = local_mask
        warnings.append(
            f"MiniMax support candidate retained as advisory only (IoU={overlap:.3f}, confidence={advice.confidence:.3f})"
        )

    protected = list(local.get("protected_polygons", []))
    # A protected region has a very specific depth meaning: it is an existing
    # object *on the receiver plane* that the inserted product must not cover.
    # Vision-language models also tend to list chairs, floors, table edges and
    # cabinetry visible behind/below the tabletop.  Treating those as 2-D body
    # exclusions forces every legal product to the front rim and then clips its
    # cast shadow.  The support polygon already owns table boundaries, while a
    # foreground product may physically occlude furniture behind the plane.
    # Keep only labels that positively identify receiver-plane obstacles.
    receiver_obstacle_terms = {
        "telephone", "phone", "cable", "cord", "wire", "cloth", "rag",
        "towel", "cup", "mug", "book", "paper", "pen", "keyboard",
        "mouse", "lamp", "vase", "bottle", "decor", "decoration",
        "existing_object", "tabletop_object", "desk_object", "on_table",
        "on_desk", "on_shelf",
    }
    non_receiver_terms = {
        "chair", "stool", "sofa", "floor", "carpet", "wall", "cabinet",
        "drawer", "furniture", "background", "table_edge", "desk_edge",
        "shelf_edge", "front_edge", "tabletop_edge", "support_edge",
    }

    def is_receiver_obstacle(label: str) -> bool:
        normalized = re.sub(r"[^a-z0-9_]+", "_", label.lower()).strip("_")
        if any(term in normalized for term in non_receiver_terms):
            return False
        return any(term in normalized for term in receiver_obstacle_terms)

    accepted_labels: list[str] = []
    ignored_labels: list[str] = []
    for polygon, label in zip(advice.protected_polygons, advice.protected_labels):
        if not is_receiver_obstacle(label):
            ignored_labels.append(label)
            continue
        candidate = _polygon_mask((height, width), polygon)
        inside_ratio = float(np.count_nonzero(candidate & support_mask) / max(1, np.count_nonzero(candidate)))
        if inside_ratio >= 0.35:
            protected.append(polygon)
            accepted_labels.append(label)
    result["protected_polygons"] = protected
    if ignored_labels:
        warnings.append(
            "Ignored non-receiver MiniMax protection labels: "
            + ", ".join(ignored_labels[:12])
        )
    if advice.confidence >= 0.62:
        result["shadow_direction"] = advice.shadow_direction
        blended_reflectivity = float(np.clip(
            0.55 * float(local.get("reflectivity", 0.0)) + 0.45 * advice.reflectivity,
            0.0, 1.0,
        ))
        if "transparent-glass" in str(local.get("method", "")):
            # The visible glass rim is direct material evidence.  A VLM may
            # call smoked glass "non-reflective" because the reflection is
            # dark; it must not switch off the physically required reflection.
            blended_reflectivity = max(
                blended_reflectivity, float(local.get("reflectivity", 0.0)),
            )
        result["reflectivity"] = blended_reflectivity
        result["surface_material"] = advice.surface_material
        result["surface_roughness"] = advice.surface_roughness
        result["surface_transmittance"] = advice.surface_transmittance
        result["reflection_confidence"] = advice.reflection_confidence
        result["direct_light_fraction"] = advice.direct_light_fraction
        result["shadow_confidence"] = advice.shadow_confidence
        local_glass = "transparent-glass" in str(local.get("method", ""))
        if local_glass and not (
            advice.cast_shadow_mode == "directional"
            and advice.shadow_confidence >= 0.82
            and advice.direct_light_fraction >= 0.48
        ):
            # The physical rim is stronger evidence than a generic VLM prior:
            # do not paint a dark long shadow on glass without explicit,
            # high-confidence directional evidence.
            result["surface_material"] = "transparent_glass"
            result["cast_shadow_mode"] = "ambient_only"
            result["shadow_opacity"] = 0.0
            result["shadow_length_ratio"] = 0.0
        else:
            result["cast_shadow_mode"] = advice.cast_shadow_mode
            if advice.cast_shadow_mode != "directional":
                result["shadow_opacity"] = 0.0
                result["shadow_length_ratio"] = 0.0
            else:
                result["shadow_opacity"] = float(np.clip(
                    float(local.get("shadow_opacity", 0.23))
                    * (0.45 + 0.75 * advice.direct_light_fraction),
                    0.0,
                    0.55,
                ))
        local_contact = float(local.get("contact_ao_strength", 0.98))
        # An ambient-only transparent receiver still needs an auditable,
        # receiver-side contact seam.  A 1.04 lower bound repeatedly vanished
        # after a 4K image was fitted into the browser and left clear vessels
        # visibly floating.  1.12 strengthens only the alpha-derived near
        # seam (never a long shadow or a generic oval) while remaining below
        # the renderer's dark-puddle regime.
        result["contact_ao_strength"] = float(
            max(local_contact, 1.12)
            if result.get("surface_material") == "transparent_glass"
            else local_contact
        )
        result["reflection_recommended"] = bool(
            advice.reflection_recommended
            or result["reflectivity"] >= 0.54
            or result.get("surface_material") in {"mirror", "transparent_glass"}
        )
    result["method"] = f"{local.get('method', 'local')}+minimax-m3-validated"
    if independent_cloud_recovery:
        result["confidence"] = float(min(
            0.95,
            0.65 * advice.confidence + 0.35 * float(advisor_support_consensus or 0.0),
        ))
    else:
        result["confidence"] = float(min(
            0.99,
            0.70 * float(local.get("confidence", 0.0)) + 0.30 * advice.confidence,
        ))
    result["advisor"] = {
        "status": "used",
        "model": advice.model,
        "request_id": advice.request_id,
        "route": advice.route,
        "confidence": advice.confidence,
        "support_iou": overlap,
        "advisor_support_consensus_iou": advisor_support_consensus,
        "support_accepted": support_accepted,
        "support_geometry_source": "minimax" if support_accepted else "local_depth_semantic",
        "support_recovery_reason": (
            "two_independent_minimax_passes_agree_while_local_is_low_confidence"
            if independent_cloud_recovery else
            "minimax_agrees_with_local_geometry" if local_agreement_accept else
            "local_geometry_retained"
        ),
        "accepted_receiver_obstacle_labels": accepted_labels,
        "ignored_non_receiver_labels": ignored_labels,
        "warnings": warnings,
        "surface_material": advice.surface_material,
        "cast_shadow_mode": advice.cast_shadow_mode,
        "direct_light_fraction": advice.direct_light_fraction,
        "shadow_confidence": advice.shadow_confidence,
        "reflection_confidence": advice.reflection_confidence,
        "receiver_width_or_diameter_cm": advice.receiver_width_or_diameter_cm,
        "receiver_size_confidence": advice.receiver_size_confidence,
    }
    return result


def apply_layout_advice(
    products: list[dict[str, Any]],
    advice: MiniMaxAdvice,
    canvas_size: tuple[int, int] | None = None,
    support_polygon: list[list[float]] | None = None,
) -> list[dict[str, Any]]:
    if advice.route == "reject":
        raise ContractError("MiniMax recommends rejecting this scene/product combination")
    by_id = {item.product_id: item for item in advice.products}
    if set(by_id) != {str(item["id"]) for item in products}:
        raise ContractError("MiniMax layout advice does not cover every product")
    suggestions = [by_id[str(item["id"])] for item in products]
    proposed_heights = np.asarray([item.height_ratio for item in suggestions], np.float64)
    physical_heights = np.asarray([item.estimated_real_height_cm for item in suggestions], np.float64)
    baselines = np.asarray([item.bottom_y for item in suggestions], np.float64)
    proposed_median = float(np.median(proposed_heights))
    physical_median = float(np.median(physical_heights))
    baseline_median = float(np.median(baselines))
    absolute_receiver_span: float | None = None
    scale_polygon: list[list[float]] = []
    if canvas_size is not None and advice.receiver_size_confidence >= 0.40:
        canvas_width, canvas_height = canvas_size
        # The already verified local/consensus receiver is the geometric source
        # of truth.  Never let a convenient model rectangle inflate or shrink
        # the absolute scale ruler used for the final placement.
        scale_polygon = support_polygon or advice.support_polygon
        support_x = [float(point[0]) for point in scale_polygon]
        if support_x and advice.receiver_width_or_diameter_cm > 1.0:
            # Projected receiver width in units of image height.  At the same
            # support depth, real product-height / receiver-width is a usable
            # absolute scale cue even when camera intrinsics are unavailable.
            absolute_receiver_span = (
                (max(support_x) - min(support_x))
                * float(canvas_width) / max(1.0, float(canvas_height))
            )

    def receiver_span_at_product(suggestion: ProductAdvice) -> tuple[float, bool]:
        """Return the receiver ruler projected at the product's own depth.

        The previous absolute-scale cue used the union width of the support
        polygon for every object.  On a perspective trapezoid that union is
        the *near* edge, so products placed toward the far edge were enlarged
        as though they were also near the camera.  For a rectilinear receiving
        plane, intersect its polygon with the product baseline and use that
        local projected width.  Curved/round receivers intentionally retain
        the global diameter: a horizontal chord is not their physical width.
        """
        if absolute_receiver_span is None or not scale_polygon or canvas_size is None:
            return 0.0, False
        points = np.asarray(scale_polygon, np.float32).reshape(-1, 2)
        if len(points) < 3:
            return absolute_receiver_span, False
        perimeter = float(cv2.arcLength(points.reshape(-1, 1, 2), True))
        approximation = cv2.approxPolyDP(
            points.reshape(-1, 1, 2),
            max(1e-5, perimeter * 0.015),
            True,
        ).reshape(-1, 2)
        # A polygon with many stable corners is normally a round/organic
        # receiver contour.  Its full diameter, not a local chord, is the
        # appropriate real-world ruler.
        if len(approximation) > 6:
            return absolute_receiver_span, False

        query_y = float(suggestion.bottom_y)
        intersections: list[float] = []
        for start, end in zip(points, np.roll(points, -1, axis=0)):
            y0, y1 = float(start[1]), float(end[1])
            if not ((y0 <= query_y < y1) or (y1 <= query_y < y0)):
                continue
            fraction = (query_y - y0) / (y1 - y0)
            intersections.append(float(start[0] + fraction * (end[0] - start[0])))
        intersections.sort()
        segments = [
            (intersections[index], intersections[index + 1])
            for index in range(0, len(intersections) - 1, 2)
            if intersections[index + 1] > intersections[index]
        ]
        if not segments:
            return absolute_receiver_span, False
        center_x = float(suggestion.center_x)
        left, right = min(
            segments,
            key=lambda segment: 0.0
            if segment[0] <= center_x <= segment[1]
            else min(abs(center_x - segment[0]), abs(center_x - segment[1])),
        )
        local_x_span = right - left
        global_x_span = float(points[:, 0].max() - points[:, 0].min())
        # Baselines on a segmentation tip/front corner do not provide a stable
        # width measurement.  Keep the global fallback rather than collapsing
        # an otherwise valid product to a tiny scale.
        if local_x_span < global_x_span * 0.40:
            return absolute_receiver_span, False
        canvas_width, canvas_height = canvas_size
        local_span = (
            local_x_span * float(canvas_width) / max(1.0, float(canvas_height))
        )
        return float(local_span), True
    semantic_heights: dict[str, float] = {}
    for suggestion in suggestions:
        # Real size is a relative calibration signal, not an absolute ruler:
        # scene camera intrinsics and table depth remain unknown. A sub-linear
        # exponent prevents an uncertain class estimate from producing a giant
        # prop, while a small baseline factor preserves perspective ordering.
        size_exponent = 0.68 + 0.32 * float(np.clip(
            suggestion.size_confidence, 0.0, 1.0,
        ))
        relative_size = (
            suggestion.estimated_real_height_cm / max(2.0, physical_median)
        ) ** size_exponent
        depth_scale = math.exp((suggestion.bottom_y - baseline_median) * 0.55)
        semantic_target = proposed_median * relative_size * depth_scale
        absolute_scale_confidence = 0.0
        if absolute_receiver_span is not None:
            projected_receiver_span, local_perspective_used = receiver_span_at_product(
                suggestion,
            )
            absolute_target = (
                suggestion.estimated_real_height_cm
                / max(20.0, advice.receiver_width_or_diameter_cm)
                * projected_receiver_span
                * (
                    1.0
                    if local_perspective_used
                    else math.exp((suggestion.bottom_y - baseline_median) * 0.30)
                )
            )
            absolute_scale_confidence = float(np.clip(
                advice.receiver_size_confidence
                * suggestion.size_confidence
                * advice.confidence,
                0.0,
                0.92,
            ))
            # Do not blend the absolute cue into the relative cue and then
            # blend that weakened result a second time into the model proposal.
            # That double attenuation was why a 25 cm pitcher stayed tiny on a
            # confidently identified 70 cm glass table.
            if absolute_scale_confidence >= 0.20:
                semantic_target = absolute_target
        relative_blend = 0.62 * suggestion.size_confidence * advice.confidence
        # The confidence above already multiplies receiver, product-size and
        # scene-advice evidence.  Taking another square-root and capping at
        # 0.82 left a physically calibrated 25 cm / 55 cm pitcher dominated
        # by the model's undersized aesthetic box.  Once all three signals
        # clear the reliability threshold, treat the metric ratio as the
        # primary ruler while still retaining 6%--52% of the visual proposal
        # for category uncertainty and unmodelled camera effects.
        absolute_blend = (
            float(np.clip(
                0.25 + 1.15 * absolute_scale_confidence, 0.0, 0.94,
            ))
            if absolute_scale_confidence >= 0.20 else 0.0
        )
        blend = float(np.clip(max(relative_blend, absolute_blend), 0.0, 0.82))
        blended = math.exp(
            (1.0 - blend) * math.log(max(0.035, suggestion.height_ratio))
            + blend * math.log(max(0.035, semantic_target))
        )
        semantic_heights[suggestion.product_id] = float(np.clip(
            blended,
            max(0.035, suggestion.height_ratio * 0.65),
            min(0.58, suggestion.height_ratio * 1.90),
        ))

        # Printed commercial goods must remain readable at ordinary review
        # size.  This is a bounded floor on projected size, not a request to
        # invent text: every glyph still comes from the original RGBA.  The
        # cap relative to the physical estimate prevents a tiny receiver from
        # turning the package into an implausible hero prop.
        category = str(suggestion.semantic_category).lower()
        if any(token in category for token in ("can", "tin", "beverage")):
            readability_floor = 0.18
            readable_cap = semantic_heights[suggestion.product_id] * 1.28
            semantic_heights[suggestion.product_id] = float(min(
                max(semantic_heights[suggestion.product_id], readability_floor),
                readable_cap,
            ))
        elif any(token in category for token in (
            "bottle", "package", "box", "carton", "book",
        )):
            readability_floor = 0.26
            readable_cap = semantic_heights[suggestion.product_id] * 1.20
            semantic_heights[suggestion.product_id] = float(min(
                max(semantic_heights[suggestion.product_id], readability_floor),
                readable_cap,
            ))
        elif any(token in category for token in (
            "pitcher", "carafe", "jug", "decanter",
        )):
            # A table pitcher is normally at least as tall as an ordinary
            # beverage bottle.  Keep enough projected extent for its handle,
            # rim, ribs and target-scene refraction to remain visible.
            semantic_heights[suggestion.product_id] = float(min(
                max(semantic_heights[suggestion.product_id], 0.30),
                semantic_heights[suggestion.product_id] * 1.32,
            ))

    seeded: list[dict[str, Any]] = []
    for item in products:
        suggestion = by_id[str(item["id"])]
        updated = dict(item)
        updated.update({
            "center_x": suggestion.center_x,
            "baseline_y": suggestion.bottom_y,
            "height_ratio": semantic_heights[suggestion.product_id],
            "z_order": suggestion.z_order,
            "source_light_compatibility": suggestion.source_light_compatibility,
            "placement_mode": suggestion.placement_mode,
            "physical_type": suggestion.physical_type,
            "photographed_pose": suggestion.photographed_pose,
            "pose_confidence": suggestion.pose_confidence,
            "estimated_real_height_cm": suggestion.estimated_real_height_cm,
            "size_confidence": suggestion.size_confidence,
            "viewpoint_compatibility": suggestion.viewpoint_compatibility,
            "semantic_category": suggestion.semantic_category,
            "estimated_height_cm_min": suggestion.estimated_height_cm_min,
            "estimated_height_cm_max": suggestion.estimated_height_cm_max,
            "footprint_width_cm": suggestion.footprint_width_cm,
            "visual_salience": suggestion.visual_salience,
            "material_mode": suggestion.material_mode,
            "material_confidence": suggestion.material_confidence,
            "transmission_strength": suggestion.transmission_strength,
            "receiver_width_or_diameter_cm": advice.receiver_width_or_diameter_cm,
            "receiver_size_confidence": advice.receiver_size_confidence,
            "advisor_height_ratio": suggestion.height_ratio,
            "semantic_scale_applied": semantic_heights[suggestion.product_id],
        })
        seeded.append(updated)
    return seeded
