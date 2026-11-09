from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from pathlib import Path
import re
from typing import Any

import cv2
import numpy as np

from .image_io import load_rgba
from .layout import overlap_ratio
from .render import contact_profile, placement_footprint, polygon_mask


# The automatic solver and the UI must agree on what constitutes a visibly
# staged three-product group.  Keep a small recovery margin above the UI gate:
# exact 0.20 solutions can round below the threshold after preview projection
# or a final legalisation pass.  These constants deliberately live in the
# geometry module so the UI diagnostic can import the same contract instead
# of maintaining an independent magic number.
VISIBLE_STAGING_UI_MIN_RATIO = 0.20
VISIBLE_STAGING_RESTORE_MIN_RATIO = 0.205
VISIBLE_STAGING_RESTORE_TARGET_RATIO = 0.22
FRONT_CLEARANCE_MIN_HEIGHTS = 0.30

# Commercial product groups need to read as one styled arrangement.  Scene
# readability is a soft preference; it must never pull one item far enough
# away that the arrangement becomes three unrelated catalogue cut-outs.  The
# UI uses 0.68 as the visible sparse-layout gate, so the solver keeps a small
# safety margin below it and targets the 0.35--0.55 edge-gap band measured
# against the adjacent displayed product widths.
PRESENTATION_GAP_TARGET_MAX = 0.55
PRESENTATION_GAP_HARD_MAX = 0.62


@dataclass(frozen=True)
class ProductGeometry:
    source_width: int
    source_height: int
    content_width: int
    content_height: int
    content_aspect: float
    contact_width_ratio: float
    visual_mass_fraction: float


def product_geometry(path: str | Path) -> ProductGeometry:
    rgba = load_rgba(path)
    alpha = rgba[..., 3]
    points = np.argwhere(alpha > 4)
    if points.size == 0:
        raise ValueError(f"Empty alpha: {path}")
    y0, x0 = points.min(axis=0)
    y1, x1 = points.max(axis=0)
    contact = contact_profile(alpha) > 4
    contact_x = np.flatnonzero(contact.any(axis=0))
    contact_width = int(contact_x.max() - contact_x.min() + 1) if len(contact_x) else int(x1 - x0 + 1)
    content_w, content_h = int(x1 - x0 + 1), int(y1 - y0 + 1)
    # Use alpha mass quantiles instead of the complete silhouette height when
    # estimating photographic scale.  A thin flower stem, antenna or handle
    # remains part of the preserved product, but it must not make the visual
    # body look miniature.
    row_mass = alpha[y0:y1 + 1, x0:x1 + 1].astype(np.float64).sum(axis=1)
    cumulative = np.cumsum(row_mass)
    if cumulative.size and cumulative[-1] > 0:
        low = int(np.searchsorted(cumulative, cumulative[-1] * 0.05, side="left"))
        high = int(np.searchsorted(cumulative, cumulative[-1] * 0.95, side="left"))
        visual_mass_height = max(1, high - low + 1)
    else:
        visual_mass_height = content_h
    return ProductGeometry(
        source_width=rgba.shape[1],
        source_height=rgba.shape[0],
        content_width=content_w,
        content_height=content_h,
        content_aspect=float(content_w / max(1, content_h)),
        contact_width_ratio=float(contact_width / max(1, rgba.shape[0])),
        visual_mass_fraction=float(visual_mass_height / max(1, content_h)),
    )


def _preview_shape(canvas_size: tuple[int, int], max_width: int = 900) -> tuple[int, int, float]:
    width, height = canvas_size
    scale = min(1.0, max_width / max(1, width))
    return max(1, int(round(width * scale))), max(1, int(round(height * scale))), scale


def normalized_display_width(
    height_ratio: float,
    content_aspect: float,
    canvas_size: tuple[int, int],
) -> float:
    """Convert a Y-normalized product height to X-normalized canvas width."""
    canvas_width, canvas_height = canvas_size
    return float(
        height_ratio
        * content_aspect
        * float(canvas_height) / max(1.0, float(canvas_width))
    )


def _placement_mask(
    rgba: np.ndarray,
    placement: dict[str, Any],
    shape: tuple[int, int],
) -> np.ndarray:
    height, width = shape
    matrix = placement_homography(
        (rgba.shape[1], rgba.shape[0]),
        (width, height),
        placement,
    )
    return cv2.warpPerspective(
        rgba[..., 3], matrix, (width, height),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
    )


def _cached_placement_mask(
    rgba: np.ndarray,
    rgba_path: str,
    placement: dict[str, Any],
    shape: tuple[int, int],
    cache: dict[tuple[Any, ...], np.ndarray] | None,
) -> np.ndarray:
    """Share an exact preview warp between geometry and readability checks.

    The readability search validates a candidate and then scores that same
    candidate immediately.  Without this tiny bounded cache both stages warp
    all three full alpha mattes independently.  Candidate coordinates are part
    of the key, so no result can leak between different placements.
    """
    if cache is None:
        return _placement_mask(rgba, placement, shape)
    key = (
        rgba_path,
        int(shape[0]),
        int(shape[1]),
        float(placement["center_x"]),
        float(placement["baseline_y"]),
        float(placement["height_ratio"]),
        str(placement.get("placement_mode", "upright")),
    )
    cached = cache.get(key)
    if cached is not None:
        return cached
    result = _placement_mask(rgba, placement, shape)
    # A production preview is roughly 900x600.  Keeping every losing search
    # alternative would turn a speed cache into a large memory spike; eighteen
    # entries retain the current three-product candidate plus nearby trials.
    while len(cache) >= 18:
        cache.pop(next(iter(cache)))
    cache[key] = result
    return result


def placement_homography(
    source_size: tuple[int, int],
    canvas_size: tuple[int, int],
    placement: dict[str, Any],
) -> np.ndarray:
    """Build the same deterministic transform for preview and final output."""
    source_w, source_h = source_size
    width, height = canvas_size
    target_h = max(1.0, height * float(placement["height_ratio"]))
    center_x = width * float(placement["center_x"])
    baseline_y = height * float(placement["baseline_y"])
    scale = target_h / max(1, source_h)
    target_w = source_w * scale
    if str(placement.get("placement_mode", "upright")) != "surface_flat":
        x = center_x - target_w / 2
        y = baseline_y - target_h
        return np.array(
            [[scale, 0, x], [0, scale, y], [0, 0, 1]],
            np.float64,
        )

    # A surface-flat source is a photographed top face. Apply a mild,
    # deterministic trapezoid so its far edge recedes with the receiving plane
    # while preserving every original product pixel through one homography.
    source = np.float32([
        [0, 0],
        [source_w - 1, 0],
        [source_w - 1, source_h - 1],
        [0, source_h - 1],
    ])
    target = np.float32([
        [center_x - target_w * 0.46, baseline_y - target_h],
        [center_x + target_w * 0.46, baseline_y - target_h],
        [center_x + target_w * 0.52, baseline_y],
        [center_x - target_w * 0.52, baseline_y],
    ])
    return cv2.getPerspectiveTransform(source, target).astype(np.float64)


def _masks(
    shape: tuple[int, int],
    support_polygon: list[list[float]],
    protected_polygons: list[list[list[float]]],
    support_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if support_mask is None:
        support = polygon_mask(shape, np.asarray(support_polygon, np.float32))
    else:
        exact = np.asarray(support_mask)
        if exact.ndim != 2:
            raise ValueError("support_mask must be a two-dimensional image")
        if exact.shape != shape:
            exact = cv2.resize(
                exact.astype(np.uint8),
                (shape[1], shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        support = exact.astype(bool)
        if not support.any():
            raise ValueError("support_mask is empty")
    protected = np.zeros(shape, bool)
    for polygon in protected_polygons:
        protected |= polygon_mask(shape, np.asarray(polygon, np.float32))
    return support, protected


def _receiver_aware_edge_margin(
    support: np.ndarray,
    requested_margin: int,
) -> int:
    """Cap photographic inset on a genuinely shallow projected receiver.

    A front-facing shelf or coffee-table top may expose only 30--40 planning
    pixels of depth.  Applying the same 7--9 pixel erosion used on a deep,
    oblique tabletop removes nearly half of its usable staging space and turns
    every legal composition into a ruler-straight row.  The physical footprint
    gate still uses the exact support mask; this helper only scales the optional
    photographic breathing inset to the receiver that is visible in the image.
    """
    requested = max(0, int(requested_margin))
    if requested <= 2:
        return requested
    mask = np.asarray(support, bool)
    if mask.ndim != 2 or not mask.any():
        return requested
    column_depths = np.count_nonzero(mask, axis=0)
    column_depths = column_depths[column_depths > 0]
    if column_depths.size < 3:
        return requested
    median_depth = float(np.median(column_depths))
    if median_depth >= mask.shape[0] * 0.14:
        return requested
    shallow_cap = max(2, int(round(median_depth * 0.06)))
    return min(requested, shallow_cap)


def _robust_feature_score(feature: np.ndarray, sample_mask: np.ndarray) -> np.ndarray:
    """Map a non-negative image feature to [0, 1] without scene constants."""
    values = feature[sample_mask]
    values = values[np.isfinite(values)]
    if values.size < 32:
        return np.zeros_like(feature, np.float32)
    floor = float(np.percentile(values, 68.0))
    ceiling = float(np.percentile(values, 97.5))
    if ceiling <= floor + 1e-5:
        return np.zeros_like(feature, np.float32)
    return np.clip((feature - floor) / (ceiling - floor), 0.0, 1.0).astype(
        np.float32,
    )


def background_readability_cost_map(
    background_rgb: np.ndarray,
    shape: tuple[int, int],
    support_mask: np.ndarray,
) -> np.ndarray:
    """Return a small, scene-relative cost map for automatic staging.

    Transparent products transmit the target scene, so a black table bracket,
    clipped specular highlight or dense high-frequency patch can be mistaken
    for contamination inside the cutout.  This map measures those three cues
    relative to the current receiving surface.  It is deliberately *not* an
    obstacle mask: support/protection geometry remains the sole hard gate.
    """
    height, width = shape
    if (
        not isinstance(background_rgb, np.ndarray)
        or background_rgb.ndim != 3
        or background_rgb.shape[2] < 3
        or support_mask.shape != (height, width)
    ):
        return np.zeros((height, width), np.float32)
    resized = cv2.resize(
        background_rgb[..., :3], (width, height), interpolation=cv2.INTER_AREA,
    ).astype(np.float32) / 255.0
    # RGB luminance; colour itself is intentionally ignored because a saturated
    # but even tabletop is not clutter.
    luminance = (
        resized[..., 0] * 0.2126
        + resized[..., 1] * 0.7152
        + resized[..., 2] * 0.0722
    ).astype(np.float32)
    sample = support_mask.astype(bool)
    if np.count_nonzero(sample) < max(64, height * width * 0.003):
        return np.zeros((height, width), np.float32)

    scene_values = luminance[sample]
    scene_median = float(np.median(scene_values))
    scene_spread = max(
        0.055,
        float(np.percentile(scene_values, 82.0) - np.percentile(scene_values, 18.0)),
    )
    sigma = max(3.0, min(height, width) * 0.035)
    local_mean = cv2.GaussianBlur(
        luminance, (0, 0), sigmaX=sigma, sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT101,
    )

    dark_feature = np.maximum(
        np.maximum(0.0, local_mean - luminance),
        np.maximum(0.0, scene_median - luminance - scene_spread * 0.18),
    )
    bright_feature = np.maximum(
        np.maximum(0.0, luminance - local_mean),
        np.maximum(0.0, luminance - scene_median - scene_spread * 0.18),
    )
    # A tiny pre-blur rejects sensor noise while retaining shelf edges, screws,
    # printed patterns and other structures that visually leak through glass.
    detail = cv2.GaussianBlur(
        luminance, (0, 0), sigmaX=0.75, sigmaY=0.75,
        borderType=cv2.BORDER_REFLECT101,
    )
    grad_x = cv2.Sobel(detail, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(detail, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(grad_x, grad_y)
    micro = np.abs(
        detail - cv2.GaussianBlur(
            detail, (0, 0), sigmaX=2.2, sigmaY=2.2,
            borderType=cv2.BORDER_REFLECT101,
        )
    )
    texture_feature = gradient + micro * 2.0

    dark = _robust_feature_score(dark_feature, sample)
    bright = _robust_feature_score(bright_feature, sample)
    texture = _robust_feature_score(texture_feature, sample)
    # A narrow table bracket or reflection rail may occupy less than the
    # robust scorer's upper-percentile support.  In that case its *edges* are
    # detected as texture while the uniformly dark interior incorrectly drops
    # back to zero.  That inversion encouraged products to sit directly over
    # the bracket and penalised the first small move away from it.  Add a
    # scene-relative absolute cue, analogous to the clipped-highlight cue
    # below.  Because it is measured from the current support median (rather
    # than from absolute black), an evenly dark tabletop remains neutral.
    dark_absolute = np.clip(
        (
            scene_median
            - luminance
            - max(0.045, scene_spread * 0.20)
        )
        / max(0.10, scene_spread * 0.85),
        0.0,
        1.0,
    ).astype(np.float32)
    dark = np.maximum(dark, dark_absolute)
    # Clipped highlights remain objectionable even when a glossy tabletop has
    # many other bright pixels, hence the small absolute saturation cue.
    highlight_clip = (
        np.clip((luminance - 0.92) / 0.08, 0.0, 1.0)
        * np.clip((luminance - scene_median - 0.025) / 0.12, 0.0, 1.0)
    )
    bright = np.maximum(bright, highlight_clip.astype(np.float32))
    strongest = np.maximum(np.maximum(dark, bright), texture)
    cost = np.clip(
        0.45 * dark + 0.25 * bright + 0.30 * texture + 0.16 * strongest,
        0.0,
        1.0,
    ).astype(np.float32)
    cost[~sample] = 0.0
    # One-pixel feature spikes should not steer a composition.  The small blur
    # approximates the finite area over which a transmitted feature is noticed.
    cost = cv2.GaussianBlur(cost, (0, 0), sigmaX=1.2, sigmaY=1.2)
    cost[~sample] = 0.0
    return cost


def _material_readability_weight(item: dict[str, Any]) -> float:
    mode = str(item.get("material_mode", "opaque")).strip().lower()
    confidence = float(np.clip(item.get("material_confidence", 0.0), 0.0, 1.0))
    base = {
        "clear_glass": 1.0,
        # A labelled / liquid-filled bottle is mostly opaque identity with
        # transmission confined to neck, shoulder and thin edges.  Treating its
        # whole silhouette like empty clear glass made the layout spread the
        # group apart merely to find a quiet background for an opaque label.
        "mixed_transparent": 0.34,
        "frosted_glass": 0.66,
        "colored_glass": 0.48,
    }.get(mode, 0.08)
    if base <= 0.08:
        return base
    return float(base * (0.35 + 0.65 * confidence))


def _material_readability_band(item: dict[str, Any]) -> tuple[float, float] | None:
    """Return the useful target-structure interval for this material.

    ``background_readability_cost_map`` is scene-relative: zero is a nearly
    featureless receiver while the upper end denotes dark hardware, clipped
    highlights or dense texture.  Empty clear glass needs some target
    structure in order to read as a refractive object, but should not inherit
    the busiest part of the scene.  Pigmented and partly opaque materials need
    progressively less help from the receiver.  Opaque products have no band
    at all, which makes this soft objective a strict no-op for opaque-only
    layouts.
    """
    mode = str(item.get("material_mode", "opaque")).strip().lower()
    return {
        "clear_glass": (0.16, 0.32),
        "frosted_glass": (0.08, 0.34),
        "colored_glass": (0.05, 0.36),
        # A labelled or liquid-filled vessel already carries opaque identity.
        # Do not seek contrast for it; only the contact-risk term below may
        # contribute while another genuinely transparent item is being moved.
        "mixed_transparent": (0.0, 0.36),
    }.get(mode)


def _contact_readability_weight(item: dict[str, Any]) -> float:
    """Weight target-scene interference specifically at an upright base."""
    if str(item.get("placement_mode", "upright")) != "upright":
        return 0.0
    mode = str(item.get("material_mode", "opaque")).strip().lower()
    confidence = float(np.clip(item.get("material_confidence", 0.0), 0.0, 1.0))
    base = {
        "clear_glass": 1.35,
        "frosted_glass": 1.00,
        "colored_glass": 0.85,
        "mixed_transparent": 0.60,
    }.get(mode, 0.12)
    if mode not in {
        "clear_glass", "frosted_glass", "colored_glass", "mixed_transparent",
    }:
        return base
    return float(base * (0.35 + 0.65 * confidence))


def _presentation_halo_weight(item: dict[str, Any]) -> float:
    """Weight visible receiver clutter immediately around a hero product.

    Opaque products hide whatever is directly behind their pixels, but dark
    table hardware, fasteners or clipped highlights that remain visible just
    outside the silhouette still destroy catalogue-style separation.  This is
    deliberately a soft commercial-photography cue rather than a hard support
    constraint.
    """
    if str(item.get("placement_mode", "upright")) != "upright":
        return 0.0
    mode = str(item.get("material_mode", "opaque")).strip().lower()
    base = {
        "metal": 0.62,
        "opaque": 0.58,
        "mixed_transparent": 0.54,
        "colored_glass": 0.48,
        "frosted_glass": 0.44,
        "clear_glass": 0.40,
    }.get(mode, 0.52)
    salience = float(np.clip(item.get("visual_salience", 0.55), 0.0, 1.0))
    return float(base * (0.72 + 0.28 * salience))


def _transparent_boundary_weight(item: dict[str, Any]) -> float:
    """Weight a receiver/backdrop seam crossing a transmitted silhouette.

    A support edge through the middle of an empty glass object is a special
    failure that the ordinary backdrop map cannot see: the map is intentionally
    zero outside the receiving surface.  The line then survives as an
    undistorted, hard split inside the glass and reads like a bad mask.  This
    weight is material-gated, so opaque catalogue goods are never moved merely
    because a tabletop edge passes behind them.
    """
    if str(item.get("placement_mode", "upright")) != "upright":
        return 0.0
    mode = str(item.get("material_mode", "opaque")).strip().lower()
    confidence = float(np.clip(item.get("material_confidence", 0.0), 0.0, 1.0))
    base = {
        "clear_glass": 2.25,
        "frosted_glass": 1.30,
        "colored_glass": 0.82,
        # Only the transmitting neck/edges of a labelled or liquid-filled
        # package are vulnerable; its opaque identity must keep priority.
        "mixed_transparent": 0.34,
    }.get(mode, 0.0)
    return float(base * (0.35 + 0.65 * confidence))


def _transparent_support_boundary_seam_loss(
    visible: np.ndarray,
    support_mask: np.ndarray,
    item: dict[str, Any],
    support_boundary: np.ndarray | None = None,
) -> float:
    """Return [0, 1] risk for a support edge cutting through a clear body.

    This is geometric rather than scene-specific.  It measures only columns
    where the visible product genuinely spans both sides of the support edge,
    then favours a long edge crossing near the middle of the body.  A line that
    merely touches the rim/base, a narrow handle crossing, or an opaque object
    therefore receives little or no penalty.
    """
    if _transparent_boundary_weight(item) <= 0.0:
        return 0.0
    visible = np.asarray(visible, bool)
    support = np.asarray(support_mask, bool)
    if visible.shape != support.shape or np.count_nonzero(visible) < 12:
        return 0.0

    points = np.argwhere(visible)
    top = int(points[:, 0].min())
    bottom = int(points[:, 0].max())
    left = int(points[:, 1].min())
    right = int(points[:, 1].max())
    object_height = max(1, bottom - top + 1)
    object_width = max(1, right - left + 1)

    boundary = support_boundary
    if boundary is None:
        boundary = cv2.morphologyEx(
            support.astype(np.uint8),
            cv2.MORPH_GRADIENT,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        ).astype(bool)
    else:
        boundary = np.asarray(boundary, bool)
        if boundary.shape != support.shape:
            raise ValueError("support_boundary shape does not match support_mask")
    crossing = visible & boundary
    if np.count_nonzero(crossing) < 3:
        return 0.0

    crossing_columns: list[int] = []
    crossing_rows: list[float] = []
    visible_columns = np.flatnonzero(visible[top:bottom + 1].any(axis=0))
    for x in range(left, right + 1):
        column_visible = visible[:, x]
        column_support = support[:, x]
        # The receiver boundary must actually split visible product pixels;
        # this rejects an edge that only grazes the outside of a handle/rim.
        if not (
            np.any(column_visible & column_support)
            and np.any(column_visible & ~column_support)
            and np.any(crossing[:, x])
        ):
            continue
        rows = np.flatnonzero(crossing[:, x])
        crossing_columns.append(x)
        crossing_rows.append(float(np.median(rows)))
    if len(crossing_columns) < 3:
        return 0.0

    column_fraction = len(crossing_columns) / max(1, len(visible_columns))
    span_fraction = (
        max(crossing_columns) - min(crossing_columns) + 1
    ) / object_width
    coverage = float(np.clip(0.58 * column_fraction + 0.42 * span_fraction, 0.0, 1.0))

    normalized_rows = (
        np.asarray(crossing_rows, np.float32) - float(top)
    ) / max(1.0, float(object_height - 1))
    # A tabletop edge through the central 70% is conspicuous.  The previous
    # 0.46 half-width accidentally scored 92% of the body, contradicting this
    # contract and treating a line immediately below the rim like a central
    # cut.  Use the stated 0.35 half-width: rim/base crossings remain available
    # for a physically grounded transparent object, while central cuts still
    # receive the full penalty.
    centrality = np.clip(
        1.0 - np.abs(normalized_rows - 0.50) / 0.35,
        0.0,
        1.0,
    )
    central_score = float(
        0.60 * np.mean(centrality) + 0.40 * np.percentile(centrality, 75.0)
    )
    return float(np.clip(coverage * central_score, 0.0, 1.0))


def _material_readability_loss(
    scene_score: float,
    contact_score: float,
    item: dict[str, Any],
) -> float:
    """Asymmetric band loss for material legibility and base grounding.

    The high-side coefficient is intentionally four times the quiet-side
    coefficient.  It prevents a checker pattern, black mounting plate or
    clipped reflection from being selected merely because it provides more
    contrast than a quiet wall.  Contact risk starts only above 0.34 and is
    measured in a shallow bottom band, so a useful tabletop boundary crossing
    the middle of a glass body is not mistaken for a bad base.
    """
    band = _material_readability_band(item)
    body_loss = 0.0
    if band is not None:
        lower, upper = band
        below = max(0.0, (lower - scene_score) / max(0.05, lower))
        above = max(0.0, (scene_score - upper) / max(0.05, 1.0 - upper))
        body_loss = _material_readability_weight(item) * (below + 4.0 * above)
    contact_excess = max(0.0, (contact_score - 0.34) / 0.66)
    contact_loss = _contact_readability_weight(item) * 3.0 * contact_excess
    return float(body_loss + contact_loss)


def _layout_readability_cost(
    cost_map: np.ndarray,
    support_mask: np.ndarray,
    products: list[dict[str, Any]],
    *,
    rgba_cache: dict[str, np.ndarray] | None = None,
    placement_mask_cache: dict[tuple[Any, ...], np.ndarray] | None = None,
    support_boundary: np.ndarray | None = None,
) -> tuple[float, list[float], list[float]]:
    """Measure target-scene interference under each transmitted silhouette."""
    height, width = cost_map.shape
    weighted_total = 0.0
    total_weight = 0.0
    product_costs: list[float] = []
    boundary_seam_costs: list[float] = []
    rgba_cache = rgba_cache if rgba_cache is not None else {}
    for item in products:
        path = str(item.get("rgba") or item["source"])
        rgba = rgba_cache.get(path)
        if rgba is None:
            rgba = load_rgba(path)
            rgba_cache[path] = rgba
        alpha = _cached_placement_mask(
            rgba,
            path,
            item,
            (height, width),
            placement_mask_cache,
        ).astype(np.float32) / 255.0
        visible = alpha > 0.02
        if np.count_nonzero(visible) < 12:
            product_costs.append(0.0)
            boundary_seam_costs.append(0.0)
            continue
        points = np.argwhere(visible)
        top = int(points[:, 0].min())
        bottom = int(points[:, 0].max())
        left = int(points[:, 1].min())
        right = int(points[:, 1].max())
        vertical = np.clip(
            (np.arange(height, dtype=np.float32) - top) / max(1.0, bottom - top),
            0.0,
            1.0,
        )[:, None]
        # The lower body/contact area is where a transmitted table structure is
        # most often misread as source residue, but the full glass body still
        # participates.
        body_visible = visible & support_mask
        pixel_weight = alpha * (0.62 + 0.38 * vertical) * body_visible
        denominator = float(pixel_weight.sum())
        if denominator > 1e-6:
            mean_cost = float((cost_map * pixel_weight).sum() / denominator)
            tail = float(np.percentile(cost_map[body_visible], 86.0))
            body_cost = 0.68 * mean_cost + 0.32 * tail
        else:
            body_cost = 0.0

        # The receiver immediately below a product remains visible as contact
        # shadow/reflection.  A black table bracket or clipped highlight in
        # this corridor can make even an opaque product appear glued to a long
        # false shadow.  Score a compact, scale-aware corridor for every
        # material; support geometry is still the only hard placement gate.
        object_height = max(1, bottom - top + 1)
        object_width = max(1, right - left + 1)
        corridor_depth = max(4, int(round(object_height * 0.24)))
        corridor_pad = max(2, int(round(object_width * 0.12)))
        corridor_y0 = max(0, bottom - max(1, int(round(object_height * 0.04))))
        corridor_y1 = min(height, bottom + corridor_depth + 1)
        corridor_x0 = max(0, left - corridor_pad)
        corridor_x1 = min(width, right + corridor_pad + 1)
        corridor = np.zeros((height, width), bool)
        corridor[
            corridor_y0:corridor_y1,
            corridor_x0:corridor_x1,
        ] = True
        # Only the receiver that remains visible around the product can
        # visually merge with its contact shadow/reflection.  Including the
        # hidden interior previously made an opaque can move because of a
        # bracket that it completely covered, while under-valuing the bracket
        # fragments still visible at its sides.
        corridor &= support_mask & ~visible
        if np.count_nonzero(corridor) >= 12:
            corridor_values = cost_map[corridor]
            contact_cost = (
                0.58 * float(np.mean(corridor_values))
                + 0.42 * float(np.percentile(corridor_values, 88.0))
            )
        else:
            contact_cost = 0.0

        # A compact lower-body presentation halo is the clean-space analogue
        # of a product photographer moving the hero item a few centimetres so
        # that table fasteners and hard seams do not kiss its outline.  The
        # anisotropic dilation stays local, excludes the product itself and
        # ignores the upper backdrop, so ordinary room texture cannot push an
        # opaque product around the table.
        halo_radius_x = max(3, int(round(object_width * 0.18)))
        halo_radius_y = max(3, int(round(object_height * 0.09)))
        halo_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (halo_radius_x * 2 + 1, halo_radius_y * 2 + 1),
        )
        expanded = cv2.dilate(
            visible.astype(np.uint8), halo_kernel, iterations=1,
        ).astype(bool)
        presentation_halo = expanded & ~visible & support_mask
        lower_start = max(0, top + int(round(object_height * 0.82)))
        presentation_halo[:lower_start] = False
        if np.count_nonzero(presentation_halo) >= 12:
            halo_values = cost_map[presentation_halo]
            presentation_cost = (
                0.48 * float(np.mean(halo_values))
                + 0.52 * float(np.percentile(halo_values, 90.0))
            )
        else:
            presentation_cost = 0.0

        material_weight = (
            _material_readability_weight(item)
            if _material_readability_band(item) is not None
            else 0.0
        )
        contact_weight = _contact_readability_weight(item)
        presentation_weight = _presentation_halo_weight(item)
        boundary_weight = _transparent_boundary_weight(item)
        boundary_seam_cost = _transparent_support_boundary_seam_loss(
            visible, support_mask, item, support_boundary,
        )
        boundary_seam_costs.append(boundary_seam_cost)
        # Do not let an inactive boundary objective dilute the long-standing
        # dark/highlight/texture objective.  The extra normalisation mass exists
        # only while a transmitted silhouette is actually cut by the edge.
        boundary_weight = (
            boundary_weight if boundary_seam_cost > 1e-4 else 0.0
        )
        product_weight = (
            material_weight
            + contact_weight
            + presentation_weight
            + boundary_weight
        )
        product_cost = (
            (
                body_cost * material_weight
                + contact_cost * contact_weight
                + presentation_cost * presentation_weight
                + boundary_seam_cost * boundary_weight
            )
            / max(1e-6, product_weight)
        )
        product_costs.append(product_cost)
        weighted_total += (
            body_cost * material_weight
            + contact_cost * contact_weight
            + presentation_cost * presentation_weight
            + boundary_seam_cost * boundary_weight
        )
        total_weight += product_weight
    if total_weight <= 1e-6:
        return 0.0, product_costs, boundary_seam_costs
    # Opaque bodies contribute no backdrop term, but their receiver corridor
    # remains a real commercial-photography concern.  Normalise the weights so
    # a single metal product can still move off a severe visible table bracket
    # without making ordinary hidden body texture influence its placement.
    weighted_mean = float(weighted_total / max(1e-6, total_weight))
    # Commercial attachment failures are governed by the worst visible base,
    # not by the average material mass.  A clear pitcher contributes a large
    # backdrop-readability weight; averaging alone therefore diluted a metal
    # can sitting on a black table leg from an individual 0.61 risk to a benign
    # group value below 0.20.  Preserve the material-weighted mean, but reserve
    # a bounded objective term for the worst single product.  This remains a
    # soft staging cue: making the maximum dominant here can prevent a useful
    # transparent-product nudge merely because pre-existing table hardware is
    # unavoidable on a compact receiver.
    worst_product = float(max(product_costs, default=0.0))
    return (
        0.65 * weighted_mean + 0.35 * worst_product,
        product_costs,
        boundary_seam_costs,
    )


def _refine_background_readability(
    canvas_size: tuple[int, int],
    support_polygon: list[list[float]],
    protected_polygons: list[list[list[float]]],
    products: list[dict[str, Any]],
    background_rgb: np.ndarray | None,
    shadow_direction: tuple[float, float] | list[float],
    shadow_length_ratio: float,
    shadow_softness_ratio: float,
    support_mask: np.ndarray | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Conservatively nudge transparent products toward a quieter backdrop."""
    empty_metrics = {
        "background_readability_cost": 0.0,
        "background_readability_improvement": 0.0,
        "background_readability_shift_max": 0.0,
        "background_readability_candidates": 0.0,
        "background_readability_trials": 0.0,
        "background_readability_max_product_cost": 0.0,
        "background_readability_foreground_product_cost": 0.0,
        "background_boundary_seam_cost": 0.0,
        "background_boundary_seam_improvement": 0.0,
        "background_readability_depth_shift_max": 0.0,
    }
    if background_rgb is None or not products:
        return products, empty_metrics
    preview_w, preview_h, _ = _preview_shape(canvas_size)
    support, _ = _masks(
        (preview_h, preview_w), support_polygon, protected_polygons,
        support_mask,
    )
    cost_map = background_readability_cost_map(
        background_rgb, (preview_h, preview_w), support,
    )
    rgba_cache: dict[str, np.ndarray] = {}
    placement_mask_cache: dict[tuple[Any, ...], np.ndarray] = {}
    support_boundary = cv2.morphologyEx(
        support.astype(np.uint8),
        cv2.MORPH_GRADIENT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    ).astype(bool)
    baseline_cost, product_costs, boundary_seam_costs = _layout_readability_cost(
        cost_map,
        support,
        products,
        rgba_cache=rgba_cache,
        placement_mask_cache=placement_mask_cache,
        support_boundary=support_boundary,
    )
    if (
        float(cost_map.max()) < 0.08
        and max(boundary_seam_costs, default=0.0) < 0.08
    ):
        return products, empty_metrics
    targets = [
        index for index, item in enumerate(products)
        if (
            (
                _material_readability_weight(item) >= 0.42
                and product_costs[index] >= 0.12
            )
            or (
                _contact_readability_weight(item) >= 0.12
                and product_costs[index] >= 0.075
            )
            or (
                _presentation_halo_weight(item) >= 0.30
                and product_costs[index] >= 0.075
            )
            or boundary_seam_costs[index] >= 0.10
        )
    ]
    if not targets:
        metrics = dict(empty_metrics)
        metrics["background_readability_cost"] = baseline_cost
        metrics["background_readability_max_product_cost"] = float(
            max(product_costs, default=0.0)
        )
        metrics["background_boundary_seam_cost"] = float(
            max(boundary_seam_costs, default=0.0)
        )
        if product_costs:
            foreground_index = max(
                range(len(products)),
                key=lambda index: int(products[index].get("z_order", 0)),
            )
            metrics["background_readability_foreground_product_cost"] = float(
                product_costs[foreground_index]
            )
        return products, metrics

    support_x = np.asarray(support_polygon, np.float32)[:, 0]
    support_span = max(0.10, float(np.ptp(support_x)))
    step = float(np.clip(support_span * 0.040, 0.018, 0.035))
    horizontal_variants: list[tuple[int, float, float, float]] = [
        (index, direction * step * multiplier, 0.0, follow)
        for multiplier in (1.0, 2.0, 2.5)
        # Breadth before distance: every affected product receives the near
        # and medium alternatives before any one product consumes the budget
        # with far excursions.  This is both fairer and measurably faster than
        # the former target-major ordering on a three-product 7K request.
        for index in targets
        for direction in (-1.0, 1.0)
        # Moving a clear item alone can solve backdrop interference while
        # destroying the equal-edge-gap rhythm of an otherwise good group.
        # Also consider a cohesive group nudge: the target receives the full
        # displacement and its companions follow most of it.
        # ``-1`` selects a rank-gradient follow: for a three-item group the
        # immediate neighbour follows half the shift and the far anchor stays
        # put.  This preserves both edge gaps while moving a side item away
        # from a table bracket, unlike an all-or-nothing group translation.
        for follow in ((0.0, 1.0, 0.68, -1.0) if len(products) > 1 else (0.0,))
    ]

    # A support/backdrop seam cannot be escaped by sliding along a horizontal
    # table edge.  Only for products whose transmitted silhouette is actually
    # cut by that edge, add a few conservative receiver-local depth moves.  The
    # maximum retains the same 0.30 displayed-object-height clearance enforced
    # by the authoritative surface diagnostic.  Keeping a second, stricter
    # 0.34 limit here stranded clear products with the receiver edge still
    # crossing their body: the final four percent of usable depth is precisely
    # what resolves that seam while remaining inside the hard placement gate.
    depth_variants: list[tuple[int, float, float, float]] = []
    for index in targets:
        if boundary_seam_costs[index] < 0.10:
            continue
        item = products[index]
        px = int(np.clip(
            round(float(item["center_x"]) * (preview_w - 1)),
            0,
            preview_w - 1,
        ))
        baseline_px = float(item["baseline_y"]) * preview_h
        column = np.flatnonzero(support[:, px])
        if not len(column):
            continue
        breaks = np.flatnonzero(np.diff(column) > 1)
        starts = np.r_[0, breaks + 1]
        ends = np.r_[breaks, len(column) - 1]
        segments = [column[start:end + 1] for start, end in zip(starts, ends)]
        segment = min(
            segments,
            key=lambda values: 0.0
            if float(values[0]) <= baseline_px <= float(values[-1]) + 1.0
            else min(
                abs(baseline_px - float(values[0])),
                abs(baseline_px - float(values[-1])),
            ),
        )
        front_y = float(segment[-1]) / preview_h
        object_height = float(item.get("height_ratio", 0.16))
        deepest_baseline = front_y - max(
            0.015,
            object_height * FRONT_CLEARANCE_MIN_HEIGHTS,
        )
        available_depth = max(
            0.0, deepest_baseline - float(item["baseline_y"]),
        )
        seen_depths: set[float] = set()
        # A quarter-height nudge can leave the receiver edge exactly on the
        # transition between the protected central body and the rim band.  It
        # looks indecisive and is unstable under preview/full-resolution
        # rounding.  Start at 0.35 heights so an accepted move visibly clears
        # the body; the exact front-clearance cap still limits shallow tables.
        for fraction in (0.35, 0.50, 0.65):
            delta_y = min(available_depth, object_height * fraction)
            rounded = round(delta_y, 5)
            if delta_y < 0.012 or rounded in seen_depths:
                continue
            seen_depths.add(rounded)
            depth_variants.append((index, 0.0, delta_y, 0.0))

    # Resolve the hard transmitted support seam before exploring wider lateral
    # presentation nudges.  Fifty-two exact trials cover the complete near/medium
    # stages for all three products plus a deterministic sample of the farthest
    # stage; depth candidates are never starved behind them.
    variants = [*depth_variants, *horizontal_variants]
    candidate_trial_budget = min(52, len(variants))

    x_units_per_y_unit = preview_h / max(1.0, float(preview_w))
    entity_widths = [
        float(item.get("height_ratio", 0.16))
        * product_geometry(item.get("rgba") or item["source"]).content_aspect
        * x_units_per_y_unit
        for item in products
    ]

    def gap_ratio(items: list[dict[str, Any]]) -> float:
        entities = sorted(
            (
                float(item["center_x"]) - entity_widths[index] * 0.5,
                float(item["center_x"]) + entity_widths[index] * 0.5,
            )
            for index, item in enumerate(items)
        )
        gaps = [
            entities[index + 1][0] - entities[index][1]
            for index in range(len(entities) - 1)
        ]
        positive = [value for value in gaps if value > 1e-6]
        if len(positive) < 2:
            return 1.0
        return float(max(positive) / min(positive))

    def gap_to_width(items: list[dict[str, Any]]) -> float:
        entities = sorted(
            (
                float(item["center_x"]) - entity_widths[index] * 0.5,
                float(item["center_x"]) + entity_widths[index] * 0.5,
                entity_widths[index],
            )
            for index, item in enumerate(items)
        )
        positive = [
            entities[index + 1][0] - entities[index][1]
            for index in range(len(entities) - 1)
            if entities[index + 1][0] - entities[index][1] > 1e-6
        ]
        if not positive:
            return 0.0
        median_width = float(np.median([entity[2] for entity in entities]))
        return float(np.median(positive) / max(1e-6, median_width))

    baseline_gap_ratio = gap_ratio(products)
    baseline_gap_to_width = gap_to_width(products)
    # Preserve an already compact group.  If this helper is called on a
    # pre-existing sparse manual layout, it may improve readability without
    # making the spacing worse, but it cannot use that sparse input as license
    # to create an even larger void.
    readability_gap_ceiling = max(
        PRESENTATION_GAP_TARGET_MAX,
        min(PRESENTATION_GAP_HARD_MAX, baseline_gap_to_width),
    )
    ordered_indices = sorted(
        range(len(products)), key=lambda index: float(products[index]["center_x"]),
    )
    rank_by_index = {
        product_index: rank
        for rank, product_index in enumerate(ordered_indices)
    }
    best = products
    best_cost = baseline_cost
    best_product_costs = list(product_costs)
    best_boundary_seam_costs = list(boundary_seam_costs)
    best_objective = baseline_cost
    best_shift = 0.0
    best_depth_shift = 0.0
    accepted_candidates = 0
    candidate_trials = 0
    for target_index, delta_x, delta_y, follow in variants:
        if candidate_trials >= candidate_trial_budget:
            break
        candidate_trials += 1
        proposed = [dict(item) for item in products]
        for index, item in enumerate(proposed):
            if index == target_index:
                applied_shift = delta_x
            elif follow >= 0.0:
                applied_shift = delta_x * follow
            else:
                rank_distance = abs(
                    rank_by_index[index] - rank_by_index[target_index]
                )
                gradient_follow = max(
                    0.0,
                    1.0 - rank_distance / max(1.0, len(products) - 1.0),
                )
                applied_shift = delta_x * gradient_follow
            item["center_x"] = float(np.clip(
                float(item["center_x"]) + applied_shift,
                0.01,
                0.99,
            ))
            if index == target_index:
                item["baseline_y"] = float(np.clip(
                    float(item["baseline_y"]) + delta_y,
                    0.02,
                    0.98,
                ))
        try:
            candidate, repair_messages = legalize_placements(
                canvas_size,
                support_polygon,
                protected_polygons,
                proposed,
                automatic_edge_margin_ratio=0.012,
                shadow_direction=shadow_direction,
                shadow_length_ratio=shadow_length_ratio,
                shadow_softness_ratio=shadow_softness_ratio,
                support_mask=support_mask,
                repair_invalid=False,
                rgba_cache=rgba_cache,
                placement_mask_cache=placement_mask_cache,
            )
        except ValueError:
            continue
        # Background appearance is never allowed to justify a geometric repair;
        # only already-legal small alternatives compete here.
        if repair_messages:
            continue
        accepted_candidates += 1
        (
            candidate_cost,
            candidate_product_costs,
            candidate_boundary_seam_costs,
        ) = _layout_readability_cost(
            cost_map,
            support,
            candidate,
            rgba_cache=rgba_cache,
            placement_mask_cache=placement_mask_cache,
            support_boundary=support_boundary,
        )
        signed_shifts = [
            float(after["center_x"]) - float(before["center_x"])
            for before, after in zip(products, candidate)
        ]
        shifts = [abs(value) for value in signed_shifts]
        group_translation = abs(float(np.mean(signed_shifts)))
        internal_deformation = float(np.mean([
            abs(value - float(np.mean(signed_shifts)))
            for value in signed_shifts
        ]))
        max_shift = float(max(shifts, default=0.0))
        depth_shifts = [
            abs(float(after["baseline_y"]) - float(before["baseline_y"]))
            for before, after in zip(products, candidate)
        ]
        max_depth_shift = float(max(depth_shifts, default=0.0))
        mean_depth_shift = float(np.mean(depth_shifts)) if depth_shifts else 0.0
        candidate_gap_ratio = gap_ratio(candidate)
        candidate_gap_to_width = gap_to_width(candidate)
        if (
            len(candidate) >= 3
            and candidate_gap_to_width > readability_gap_ceiling + 1e-6
        ):
            # Geometry/admissibility remains valid, but this candidate is not
            # a commercially coherent composition.  Reject it before a large
            # background-cost improvement can buy its way past a soft rhythm
            # penalty.
            continue
        rhythm_penalty = (
            max(0.0, candidate_gap_ratio - max(1.85, baseline_gap_ratio)) * 0.08
            + max(0.0, candidate_gap_ratio - baseline_gap_ratio) * 0.08
            + max(
                0.0,
                candidate_gap_to_width - max(
                    PRESENTATION_GAP_TARGET_MAX,
                    min(PRESENTATION_GAP_HARD_MAX, baseline_gap_to_width),
                ),
            ) * 0.25
            + max(
                0.0,
                candidate_gap_to_width - PRESENTATION_GAP_TARGET_MAX,
            ) * 0.35
        )
        objective = (
            candidate_cost
            # A small coherent group translation preserves composition; a
            # one-item excursion destroys it.  Penalise those two motions
            # separately instead of charging every companion shift as though
            # it were independent damage.
            + group_translation * 0.12
            + internal_deformation * 1.20
            + max_shift * 0.10
            # Receiver-local depth is allowed only to remove a hard seam, and
            # is charged more gently than an equivalent horizontal excursion:
            # a shallow staging triangle is desirable, a scattered row is not.
            + mean_depth_shift * 0.28
            + max_depth_shift * 0.12
            + rhythm_penalty
        )
        if (
            baseline_cost - candidate_cost >= 0.040
            and best_objective - objective >= 0.020
        ):
            best = candidate
            best_cost = candidate_cost
            best_product_costs = list(candidate_product_costs)
            best_boundary_seam_costs = list(candidate_boundary_seam_costs)
            best_objective = objective
            best_shift = max_shift
            best_depth_shift = max_depth_shift
            # No later motion can materially beat a near-zero scene cost and a
            # resolved receiver seam; stop before paying for farther variants.
            # The hard placement contract has already passed for ``best``.
            if (
                best_cost <= 0.025
                and max(best_boundary_seam_costs, default=0.0) <= 0.025
                and best_objective <= 0.045
            ):
                break

    foreground_index = max(
        range(len(best)),
        key=lambda index: int(best[index].get("z_order", 0)),
    )
    return best, {
        "background_readability_cost": best_cost,
        "background_readability_improvement": max(0.0, baseline_cost - best_cost),
        "background_readability_shift_max": best_shift,
        "background_readability_depth_shift_max": best_depth_shift,
        "background_readability_candidates": float(accepted_candidates),
        "background_readability_trials": float(candidate_trials),
        "background_readability_max_product_cost": float(
            max(best_product_costs, default=0.0)
        ),
        "background_readability_foreground_product_cost": float(
            best_product_costs[foreground_index]
            if best_product_costs else 0.0
        ),
        "background_boundary_seam_cost": float(
            max(best_boundary_seam_costs, default=0.0)
        ),
        "background_boundary_seam_improvement": max(
            0.0,
            float(max(boundary_seam_costs, default=0.0))
            - float(max(best_boundary_seam_costs, default=0.0)),
        ),
    }


def layout_surface_diagnostics(
    canvas_size: tuple[int, int],
    support_polygon: list[list[float]],
    products: list[dict[str, Any]],
    support_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Measure placement depth in the local tabletop coordinate system.

    Image-space Y is not a depth coordinate on an oblique table: three bases
    can follow a diagonal front edge and still all sit at the same dangerous
    depth.  For each product we find the contiguous support interval beneath
    its centre, then express the baseline between that interval's back and
    front edges.  The same interval supplies a scale-independent front-edge
    clearance measured in displayed object heights.
    """
    preview_w, preview_h, _ = _preview_shape(canvas_size)
    support, _ = _masks(
        (preview_h, preview_w), support_polygon, [], support_mask,
    )
    depths: list[float] = []
    clearances: list[float] = []
    receiver_spans_over_height: list[float] = []
    for item in products:
        px = int(np.clip(round(float(item["center_x"]) * (preview_w - 1)), 0, preview_w - 1))
        baseline_px = float(item["baseline_y"]) * preview_h
        column = np.flatnonzero(support[:, px])
        if not len(column):
            depths.append(1.0)
            clearances.append(-1.0)
            continue

        # Split a potentially concave/disconnected column and use the segment
        # containing the baseline (or its nearest segment when just outside).
        breaks = np.flatnonzero(np.diff(column) > 1)
        starts = np.r_[0, breaks + 1]
        ends = np.r_[breaks, len(column) - 1]
        segments = [(float(column[a]), float(column[b])) for a, b in zip(starts, ends)]

        def segment_distance(segment: tuple[float, float]) -> float:
            back, front = segment
            if back <= baseline_px <= front + 1.0:
                return 0.0
            return min(abs(baseline_px - back), abs(baseline_px - front))

        back, front = min(segments, key=segment_distance)
        span = max(1.0, front - back)
        depth = (baseline_px - back) / span
        object_height = max(1.0, float(item["height_ratio"]) * preview_h)
        clearance = (front - baseline_px) / object_height
        depths.append(float(depth))
        clearances.append(float(clearance))
        receiver_spans_over_height.append(float(span / object_height))

    if not depths:
        return {
            "surface_depth_min": 0.0,
            "surface_depth_max": 0.0,
            "surface_depth_median": 0.0,
            "surface_depth_range": 0.0,
            "front_clearance_min_heights": 0.0,
            "receiver_span_over_height_median": 0.0,
            "front_clearance_required_heights": FRONT_CLEARANCE_MIN_HEIGHTS,
            "automatic_depth_max": 0.76,
        }
    values = np.asarray(depths, np.float64)
    receiver_capacity = float(np.median(
        np.asarray(receiver_spans_over_height, np.float64),
    )) if receiver_spans_over_height else 0.0
    # A near-frontal tabletop can project to less than half an object height.
    # Requiring 0.30 object-heights of empty receiver in that case is not a
    # safety rule; it is geometrically impossible and pushes every base onto
    # the rim during repair.  The exact footprint/inset gate remains hard.
    if receiver_capacity < 0.45:
        clearance_required = 0.025
        automatic_depth_max = 0.90
    elif receiver_capacity < 0.85:
        blend = (receiver_capacity - 0.45) / 0.40
        clearance_required = 0.025 + blend * (
            FRONT_CLEARANCE_MIN_HEIGHTS - 0.025
        )
        automatic_depth_max = 0.90 - blend * (0.90 - 0.76)
    else:
        clearance_required = FRONT_CLEARANCE_MIN_HEIGHTS
        automatic_depth_max = 0.76
    return {
        "surface_depth_min": float(values.min()),
        "surface_depth_max": float(values.max()),
        "surface_depth_median": float(np.median(values)),
        "surface_depth_range": float(np.ptp(values)),
        "front_clearance_min_heights": float(min(clearances)),
        "receiver_span_over_height_median": receiver_capacity,
        "front_clearance_required_heights": float(clearance_required),
        "automatic_depth_max": float(automatic_depth_max),
    }


def _valid_mask(
    alpha: np.ndarray,
    support: np.ndarray,
    protected: np.ndarray,
    accepted: list[np.ndarray],
    placement_mode: str = "upright",
) -> bool:
    body = alpha > 4
    if not body.any() or np.any(body & protected):
        return False
    footprint = placement_footprint(alpha, placement_mode) > 4
    # Preview coordinates are roughly five times smaller than 4K output. A
    # multi-pixel preview dilation therefore permits a much larger overflow
    # than the final compositor's two-pixel tolerance. Require the physical
    # footprint inside the detected support at preview scale so a layout that
    # passes here cannot fail after full-resolution rendering.
    tolerant = support
    if not footprint.any() or np.any(footprint & ~tolerant):
        return False
    # Keep margin below the final compositor gate (0.005). Preview masks are
    # lower resolution, so equality can round upward at full resolution.
    return all(overlap_ratio(alpha, other) <= 0.003 for other in accepted)


def _planned_shadow_receiver_ratios(
    alpha: np.ndarray,
    support: np.ndarray,
    protected: np.ndarray,
    shadow_direction: tuple[float, float] | list[float],
    shadow_length_ratio: float,
    shadow_softness_ratio: float,
    placement_mode: str = "upright",
) -> tuple[float, float]:
    """Estimate how much of an automatic placement's shadow has a receiver.

    This is deliberately a geometry-only planning proxy.  It uses the same
    height-proportional affine projection as the deterministic renderer, but
    does not paint or score appearance.  The hard projection prevents an
    object from being placed where its umbra would leave the tabletop; the
    dilated projection reserves a smaller penumbra safety corridor.  Manual
    placements do not call this gate because their only physical requirement
    is that the measured base remains supported.
    """
    vector = np.asarray(shadow_direction, np.float32)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(vector).all() or norm < 1e-6:
        return 0.0, 0.0
    dx, dy = (float(value) for value in vector / norm)
    binary = alpha >= 32
    occupied = np.argwhere(binary)
    if occupied.size == 0:
        return 0.0, 0.0
    y0, x0 = occupied.min(axis=0)
    y1, x1 = occupied.max(axis=0)
    object_height = max(1, int(y1 - y0 + 1))
    length = float(np.clip(shadow_length_ratio, 0.0, 0.5))
    softness = float(np.clip(shadow_softness_ratio, 0.01, 0.25))

    if placement_mode == "surface_flat":
        short_side = max(1, int(min(y1 - y0 + 1, x1 - x0 + 1)))
        max_distance = max(2, int(round(short_side * 0.045 * length / 0.04)))
        matrix = np.float32([
            [1.0, 0.0, dx * max_distance],
            [0.0, 1.0, dy * max_distance],
        ])
    else:
        max_distance = max(1, int(round(object_height * length / 0.16)))
        height_scale = max_distance / float(object_height)
        # Every source pixel lands on the receiver plane at a distance
        # proportional to its height over the measured base.
        matrix = np.float32([
            [1.0, -dx * height_scale, dx * height_scale * y1],
            [0.0, -dy * height_scale, y1 * (1.0 + dy * height_scale)],
        ])

    hard = cv2.warpAffine(
        binary.astype(np.uint8),
        matrix,
        (alpha.shape[1], alpha.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    ).astype(bool)
    # Include the exact measured contact so the corridor is guaranteed to
    # begin at the object rather than at a detached projected lobe.
    hard |= placement_footprint(alpha, placement_mode) > 4
    if not hard.any():
        return 0.0, 0.0
    penumbra_radius = max(1, int(round(max_distance * softness * 1.15)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (penumbra_radius * 2 + 1, penumbra_radius * 2 + 1),
    )
    soft = cv2.dilate(hard.astype(np.uint8), kernel).astype(bool)
    receiver = support & ~protected
    hard_ratio = float(np.count_nonzero(hard & receiver) / max(1, np.count_nonzero(hard)))
    soft_ratio = float(np.count_nonzero(soft & receiver) / max(1, np.count_nonzero(soft)))
    return hard_ratio, soft_ratio


def legalize_placements(
    canvas_size: tuple[int, int],
    support_polygon: list[list[float]],
    protected_polygons: list[list[list[float]]],
    products: list[dict[str, Any]],
    automatic_edge_margin_ratio: float = 0.0,
    shadow_direction: tuple[float, float] | list[float] | None = None,
    shadow_length_ratio: float = 0.040,
    shadow_softness_ratio: float = 0.15,
    support_mask: np.ndarray | None = None,
    *,
    repair_invalid: bool = True,
    rgba_cache: dict[str, np.ndarray] | None = None,
    placement_mask_cache: dict[tuple[Any, ...], np.ndarray] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Keep a manual layout when valid; minimally repair only invalid items."""
    preview_w, preview_h, _ = _preview_shape(canvas_size)
    support, protected = _masks(
        (preview_h, preview_w), support_polygon, protected_polygons,
        support_mask,
    )
    receiver_support = support.copy()
    receiver_protected = protected.copy()
    # Manual edits obey the exact physical contract: only the footprint must be
    # supported.  Automatic layouts additionally keep a photographic safety
    # margin from table/protected edges.  Keeping this opt-in prevents a user's
    # deliberate overhanging silhouette from being silently moved.
    if automatic_edge_margin_ratio > 0:
        margin = max(
            2,
            int(round(min(preview_w, preview_h) * automatic_edge_margin_ratio)),
        )
        margin = _receiver_aware_edge_margin(receiver_support, margin)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (margin * 2 + 1, margin * 2 + 1),
        )
        inset = cv2.erode(support.astype(np.uint8), kernel).astype(bool)
        if np.count_nonzero(inset) >= preview_w * preview_h * 0.008:
            support = inset
            protected = cv2.dilate(
                protected.astype(np.uint8), kernel,
            ).astype(bool)
    accepted_masks: list[np.ndarray] = []
    repaired: list[dict[str, Any]] = []
    messages: list[str] = []
    rgba_cache = rgba_cache if rgba_cache is not None else {}

    def valid_automatic_shadow(alpha: np.ndarray, mode: str) -> bool:
        # Ambient-only callers pass ``None`` explicitly.  Treating an identity
        # transform as a cast shadow would require the whole upright silhouette
        # to lie inside the tabletop and force products to the front rim.  With
        # no directional projection there is no corridor to reserve; footprint
        # support remains fully authoritative.
        if shadow_direction is None:
            return True
        hard_ratio, soft_ratio = _planned_shadow_receiver_ratios(
            alpha,
            receiver_support,
            receiver_protected,
            shadow_direction,
            shadow_length_ratio,
            shadow_softness_ratio,
            mode,
        )
        # A few antialiased/quantised pixels may cross a polygon edge, but the
        # visible umbra and almost all of the penumbra must have the same real
        # receiving plane.  This prevents hard support-edge truncation before
        # rendering rather than trying to hide it afterwards.
        return hard_ratio >= 0.992 and soft_ratio >= 0.965

    def candidate_is_valid(alpha: np.ndarray, mode: str) -> bool:
        return _valid_mask(
            alpha, support, protected, accepted_masks, mode,
        ) and valid_automatic_shadow(alpha, mode)

    for index, item in enumerate(products):
        rgba_path = str(item.get("rgba") or item["source"])
        rgba = rgba_cache.get(rgba_path)
        if rgba is None:
            rgba = load_rgba(rgba_path)
            rgba_cache[rgba_path] = rgba
        desired = dict(item)
        desired_mask = _cached_placement_mask(
            rgba,
            rgba_path,
            desired,
            (preview_h, preview_w),
            placement_mask_cache,
        )
        desired_mode = str(desired.get("placement_mode", "upright"))
        if candidate_is_valid(desired_mask, desired_mode):
            repaired.append(desired)
            accepted_masks.append(desired_mask)
            continue

        # Readability refinement proposes only small alternatives and rejects
        # every candidate that would need a geometric repair.  Running the full
        # 6-scale x 49-offset search in that caller was therefore pure waste:
        # its repaired result was discarded immediately.  This exact-check mode
        # preserves the same acceptance contract while making the rejection
        # explicit and bounded.
        if not repair_invalid:
            raise ValueError(
                f"Product {index + 1} candidate is invalid without repair"
            )

        offsets = [(0.0, 0.0)]
        for radius in (0.015, 0.03, 0.05, 0.08, 0.12, 0.18):
            offsets.extend([
                (dx, dy)
                for dx, dy in product((-radius, 0.0, radius), repeat=2)
                if dx or dy
            ])
        best: tuple[float, dict[str, Any], np.ndarray] | None = None
        for scale_factor in (1.0, 0.97, 0.93, 0.88, 0.82, 0.74):
            for dx, dy in offsets:
                candidate = dict(desired)
                candidate["center_x"] = float(np.clip(float(desired["center_x"]) + dx, 0.01, 0.99))
                candidate["baseline_y"] = float(np.clip(float(desired["baseline_y"]) + dy, 0.02, 0.98))
                candidate["height_ratio"] = max(0.035, float(desired["height_ratio"]) * scale_factor)
                alpha = _cached_placement_mask(
                    rgba,
                    rgba_path,
                    candidate,
                    (preview_h, preview_w),
                    placement_mask_cache,
                )
                candidate_mode = str(candidate.get("placement_mode", "upright"))
                if not candidate_is_valid(alpha, candidate_mode):
                    continue
                score = dx * dx + dy * dy + (1.0 - scale_factor) ** 2 * 0.28
                if best is None or score < best[0]:
                    best = (score, candidate, alpha)
        if best is None:
            raise ValueError(f"商品 {index + 1} 在当前桌面上找不到安全位置，请使用自动构图")
        _, fixed, fixed_mask = best
        repaired.append(fixed)
        accepted_masks.append(fixed_mask)
        messages.append(f"商品 {index + 1} 已轻微吸附到桌面安全区")
    # Front-most objects are composited last.  Use local support depth rather
    # than image Y so a diagonal table edge cannot invert the layer order.
    def local_depth(index: int) -> float:
        item = repaired[index]
        px = int(np.clip(
            round(float(item["center_x"]) * (preview_w - 1)),
            0,
            preview_w - 1,
        ))
        baseline_px = float(item["baseline_y"]) * preview_h
        column = np.flatnonzero(support[:, px])
        if len(column) < 2:
            return float(item["baseline_y"])
        breaks = np.flatnonzero(np.diff(column) > 1)
        starts = np.r_[0, breaks + 1]
        ends = np.r_[breaks, len(column) - 1]
        segments = [(float(column[a]), float(column[b])) for a, b in zip(starts, ends)]
        back, front = min(
            segments,
            key=lambda segment: 0.0
            if segment[0] <= baseline_px <= segment[1] + 1.0
            else min(abs(baseline_px - segment[0]), abs(baseline_px - segment[1])),
        )
        return float((baseline_px - back) / max(1.0, front - back))

    ordered = sorted(range(len(repaired)), key=local_depth)
    for z, index in enumerate(ordered, 1):
        repaired[index]["z_order"] = z
    return repaired, messages


def refine_seeded_composition(
    canvas_size: tuple[int, int],
    support_polygon: list[list[float]],
    protected_polygons: list[list[list[float]]],
    products: list[dict[str, Any]],
    shadow_direction: tuple[float, float] | list[float] = (0.75, -0.66),
    shadow_length_ratio: float = 0.040,
    shadow_softness_ratio: float = 0.15,
    background_rgb: np.ndarray | None = None,
    support_mask: np.ndarray | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Turn a safe three-item row into a shallow triangular composition.

    Multimodal advisers are good at identifying a tabletop but often emit
    three nearly identical baselines.  That arrangement is legal yet reads as
    a catalogue overlay.  We test a few small, deterministic depth patterns,
    retain only candidates that pass the exact preview geometry gates, and
    choose the one with a visible but restrained depth stagger.  No fixed
    scene coordinates are used: every candidate starts from the adviser seed.
    """
    def with_readable_background(
        items: list[dict[str, Any]], messages: list[str],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        refined, _ = _refine_background_readability(
            canvas_size,
            support_polygon,
            protected_polygons,
            items,
            background_rgb,
            shadow_direction,
            shadow_length_ratio,
            shadow_softness_ratio,
            support_mask,
        )
        return refined, messages

    try:
        safe, repairs = legalize_placements(
            canvas_size,
            support_polygon,
            protected_polygons,
            products,
            automatic_edge_margin_ratio=0.012,
            shadow_direction=shadow_direction,
            shadow_length_ratio=shadow_length_ratio,
            shadow_softness_ratio=shadow_softness_ratio,
            support_mask=support_mask,
        )
    except ValueError:
        # Very narrow shelves may not have room for the preferred safety band.
        # The exact physical footprint gate remains a valid deterministic
        # fallback; it is never weakened below the manual contract.
        try:
            safe, repairs = legalize_placements(
                canvas_size, support_polygon, protected_polygons, products,
                support_mask=support_mask,
            )
        except ValueError:
            # An adviser seed may be physically impossible as a group even
            # though the detected support plane can host the products.  Keep
            # every original item dictionary (and therefore its semantic
            # fields) as the input to the deterministic joint planner instead
            # of leaking the seed failure to the UI as a generic fallback.
            replanned, _ = solve_aesthetic_layout(
                canvas_size,
                support_polygon,
                protected_polygons,
                products,
                shadow_direction=shadow_direction,
                shadow_length_ratio=shadow_length_ratio,
                shadow_softness_ratio=shadow_softness_ratio,
                background_rgb=background_rgb,
                support_mask=support_mask,
            )
            return replanned, [
                "MiniMax 种子无法合法化，已本地联合重排",
            ]
    if len(safe) != 3:
        return with_readable_background(safe, repairs)

    diagnostics = layout_surface_diagnostics(
        canvas_size, support_polygon, safe, support_mask,
    )
    original_by_id = {
        str(item.get("id", index)): item for index, item in enumerate(products)
    }
    repair_count = 0
    max_repair_distance = 0.0
    max_repair_scale = 0.0
    for index, item in enumerate(safe):
        original = original_by_id.get(str(item.get("id", index)), products[index])
        dx = float(item["center_x"]) - float(original["center_x"])
        dy = float(item["baseline_y"]) - float(original["baseline_y"])
        distance = float(np.hypot(dx, dy))
        original_scale = max(1e-6, float(original["height_ratio"]))
        scale_delta = abs(float(item["height_ratio"]) - original_scale) / original_scale
        max_repair_distance = max(max_repair_distance, distance)
        max_repair_scale = max(max_repair_scale, scale_delta)
        if distance > 0.008 or scale_delta > 0.02:
            repair_count += 1

    needs_joint_replan = (
        len(repairs) >= 2
        or repair_count >= 2
        or max_repair_distance > 0.060
        or max_repair_scale > 0.10
        or diagnostics["surface_depth_max"] > 0.82
        or diagnostics["surface_depth_median"] > 0.72
        or diagnostics["front_clearance_min_heights"] < 0.30
    )
    if needs_joint_replan:
        try:
            replanned, _ = solve_aesthetic_layout(
                canvas_size,
                support_polygon,
                protected_polygons,
                safe,
                shadow_direction=shadow_direction,
                shadow_length_ratio=shadow_length_ratio,
                shadow_softness_ratio=shadow_softness_ratio,
                background_rgb=background_rgb,
                support_mask=support_mask,
            )
            return replanned, [
                *repairs,
                "MiniMax 坐标位于桌面前沿或需要大幅修复；已保留场景语义并执行本地联合重排。",
            ]
        except ValueError:
            # Preserve the already legal adviser layout if a genuinely narrow
            # support cannot host the preferred joint composition.
            pass
    baselines = np.asarray([float(item["baseline_y"]) for item in safe])
    preview_w, preview_h, _ = _preview_shape(canvas_size)
    seeded_support, _ = _masks(
        (preview_h, preview_w), support_polygon, protected_polygons,
        support_mask,
    )
    support_rows = np.flatnonzero(seeded_support.any(axis=1))
    support_depth_span = (
        float(np.ptp(support_rows) / max(1, preview_h - 1))
        if len(support_rows) else 0.0
    )
    if support_depth_span < 0.18 or float(np.ptp(baselines)) >= 0.045:
        return with_readable_background(safe, repairs)

    direction = np.asarray(shadow_direction, np.float32)
    retreat_sign = -1.0 if float(direction[1]) >= 0.0 else 1.0
    # Break a ruler-straight row while moving every seed away from the cast
    # direction.  One item stays relatively forward, but none is deliberately
    # pushed toward the edge that must receive its shadow.
    patterns = tuple(
        tuple(retreat_sign * value for value in values)
        for values in (
            (0.050, 0.010, 0.036),
            (0.034, 0.054, 0.008),
            (0.012, 0.052, 0.038),
        )
    )
    best = safe
    best_messages = repairs
    best_score = float(np.ptp(baselines) * 12.0 - len(repairs) * 0.08)
    for pattern in patterns:
        seeded: list[dict[str, Any]] = []
        for item, delta in zip(safe, pattern):
            candidate = dict(item)
            candidate["baseline_y"] = float(np.clip(
                float(item["baseline_y"]) + delta, 0.02, 0.98,
            ))
            # The apparent size change follows the same perspective sign as
            # depth, but stays under six percent to preserve adviser intent.
            candidate["height_ratio"] = float(item["height_ratio"]) * float(
                np.clip(1.0 + delta * 0.72, 0.94, 1.04)
            )
            seeded.append(candidate)
        try:
            legal, messages = legalize_placements(
                canvas_size,
                support_polygon,
                protected_polygons,
                seeded,
                automatic_edge_margin_ratio=0.012,
                shadow_direction=shadow_direction,
                shadow_length_ratio=shadow_length_ratio,
                shadow_softness_ratio=shadow_softness_ratio,
                support_mask=support_mask,
            )
        except ValueError:
            continue
        trial_baselines = np.asarray([
            float(item["baseline_y"]) for item in legal
        ])
        depth_range = float(np.ptp(trial_baselines))
        displacement = float(np.mean(np.abs(trial_baselines - baselines)))
        # Prefer 4.5-8.5% of image height: enough to break the ruler-straight
        # row, not enough to scatter a product set across the entire table.
        depth_fit = 1.0 - min(1.0, abs(depth_range - 0.062) / 0.062)
        score = depth_fit - displacement * 2.4 - len(messages) * 0.10
        if depth_range >= 0.04 and score > best_score:
            best, best_messages, best_score = legal, messages, score
    if best is not safe:
        best_messages = [*best_messages, "已自动形成轻微前后层次，避免三件商品排成僵硬直线"]
    return with_readable_background(best, best_messages)


def compact_sparse_seeded_layout(
    canvas_size: tuple[int, int],
    support_polygon: list[list[float]],
    protected_polygons: list[list[list[float]]],
    products: list[dict[str, Any]],
    shadow_direction: tuple[float, float] | list[float] = (0.75, -0.66),
    shadow_length_ratio: float = 0.040,
    shadow_softness_ratio: float = 0.15,
    support_mask: np.ndarray | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float], list[str]]:
    """Compact a sparse semantic seed without changing identity or scale.

    The multimodal adviser is the most reliable source for product meaning,
    left-to-right story and relative physical size.  A full local replan used
    to throw those facts away merely because the adviser left too much empty
    space.  This bounded repair keeps every height, material, pose and semantic
    field byte-for-byte unchanged.  It only recomputes X positions from the
    *visible alpha widths* and maps each base to the same receiver-local depth
    at its new X coordinate.

    Every proposal must pass the exact footprint, protected-region, overlap and
    shadow-receiver checks with ``repair_invalid=False``.  Failure therefore
    returns the original seed rather than silently shrinking or replacing it.
    """
    empty_metrics = {
        "seed_compaction_applied": 0.0,
        "seed_gap_before_max": 0.0,
        "seed_gap_after_max": 0.0,
        "seed_compaction_shift_max": 0.0,
        "seed_compaction_scale_change_max": 0.0,
        "seed_compaction_trials": 0.0,
    }
    if len(products) != 3:
        return products, empty_metrics, []

    geometries = [
        product_geometry(item.get("rgba") or item["source"])
        for item in products
    ]
    widths = [
        normalized_display_width(
            float(item["height_ratio"]), geometry.content_aspect, canvas_size,
        )
        for item, geometry in zip(products, geometries)
    ]
    order = sorted(
        range(len(products)), key=lambda index: float(products[index]["center_x"]),
    )

    def normalized_gaps(items: list[dict[str, Any]]) -> list[float]:
        values: list[float] = []
        for first, second in zip(order, order[1:]):
            edge_gap = (
                float(items[second]["center_x"]) - widths[second] * 0.5
                - float(items[first]["center_x"]) - widths[first] * 0.5
            )
            mean_width = max(1e-6, (widths[first] + widths[second]) * 0.5)
            values.append(float(edge_gap / mean_width))
        return values

    before_gaps = normalized_gaps(products)
    before_max = float(max(before_gaps, default=0.0))
    metrics = dict(empty_metrics)
    metrics["seed_gap_before_max"] = before_max
    metrics["seed_gap_after_max"] = before_max
    if before_max <= PRESENTATION_GAP_TARGET_MAX:
        return products, metrics, []

    preview_w, preview_h, _ = _preview_shape(canvas_size)
    support, _ = _masks(
        (preview_h, preview_w), support_polygon, protected_polygons,
        support_mask,
    )

    def column_segments(column: np.ndarray) -> list[np.ndarray]:
        if len(column) == 0:
            return []
        breaks = np.flatnonzero(np.diff(column) > 1)
        starts = np.r_[0, breaks + 1]
        ends = np.r_[breaks, len(column) - 1]
        return [column[int(start):int(end) + 1] for start, end in zip(starts, ends)]

    def receiver_depth(item: dict[str, Any]) -> float:
        px = int(np.clip(
            round(float(item["center_x"]) * (preview_w - 1)), 0, preview_w - 1,
        ))
        baseline = float(item["baseline_y"]) * preview_h
        segments = column_segments(np.flatnonzero(support[:, px]))
        if not segments:
            return 0.55
        segment = min(
            segments,
            key=lambda values: 0.0
            if float(values[0]) <= baseline <= float(values[-1]) + 1.0
            else min(abs(baseline - float(values[0])), abs(baseline - float(values[-1]))),
        )
        return float(np.clip(
            (baseline - float(segment[0]))
            / max(1.0, float(segment[-1] - segment[0])),
            0.0,
            1.0,
        ))

    original_depths = [receiver_depth(item) for item in products]

    def baseline_at(
        center_x: float,
        depth: float,
        original_baseline: float,
    ) -> float | None:
        px = int(np.clip(round(center_x * (preview_w - 1)), 0, preview_w - 1))
        segments = [
            values for values in column_segments(np.flatnonzero(support[:, px]))
            if len(values) >= 3
        ]
        if not segments:
            return None
        projected = [
            (float(values[0]) + depth * float(values[-1] - values[0])) / preview_h
            for values in segments
        ]
        # On a support mask with more than one receiver island, stay on the
        # continuation closest to the original table row instead of jumping to
        # a wall, shelf or foreground floor component.
        return float(min(projected, key=lambda value: abs(value - original_baseline)))

    seed_left = min(
        float(products[index]["center_x"]) - widths[index] * 0.5 for index in order
    )
    seed_right = max(
        float(products[index]["center_x"]) + widths[index] * 0.5 for index in order
    )
    seed_midpoint = (seed_left + seed_right) * 0.5
    rgba_cache: dict[str, np.ndarray] = {}
    mask_cache: dict[tuple[Any, ...], np.ndarray] = {}
    best: tuple[float, list[dict[str, Any]], float, float] | None = None
    trials = 0

    # Target a deliberate product-photography breathing band.  Midpoint
    # offsets let an oblique or protected receiver choose the nearest exact
    # legal continuation without changing the intended grouping.
    for gap_ratio in (0.44, 0.50, 0.38, 0.55, 0.34, 0.60):
        pair_gaps = [
            gap_ratio * (widths[first] + widths[second]) * 0.5
            for first, second in zip(order, order[1:])
        ]
        group_width = sum(widths[index] for index in order) + sum(pair_gaps)
        for midpoint_offset in (
            0.0, -0.012, 0.012, -0.025, 0.025, -0.045, 0.045,
            -0.070, 0.070,
        ):
            trials += 1
            cursor = seed_midpoint + midpoint_offset - group_width * 0.5
            candidate = [dict(item) for item in products]
            valid_projection = True
            for slot, product_index in enumerate(order):
                center_x = cursor + widths[product_index] * 0.5
                if not (0.005 < center_x < 0.995):
                    valid_projection = False
                    break
                baseline_y = baseline_at(
                    center_x,
                    original_depths[product_index],
                    float(products[product_index]["baseline_y"]),
                )
                if baseline_y is None:
                    valid_projection = False
                    break
                candidate[product_index]["center_x"] = center_x
                candidate[product_index]["baseline_y"] = baseline_y
                cursor += widths[product_index]
                if slot < len(pair_gaps):
                    cursor += pair_gaps[slot]
            if not valid_projection:
                continue
            try:
                legal, repair_messages = legalize_placements(
                    canvas_size,
                    support_polygon,
                    protected_polygons,
                    candidate,
                    automatic_edge_margin_ratio=0.012,
                    shadow_direction=shadow_direction,
                    shadow_length_ratio=shadow_length_ratio,
                    shadow_softness_ratio=shadow_softness_ratio,
                    support_mask=support_mask,
                    repair_invalid=False,
                    rgba_cache=rgba_cache,
                    placement_mask_cache=mask_cache,
                )
            except ValueError:
                continue
            if repair_messages:
                # Exact-check mode should never emit a repair, but keep the
                # invariant explicit in case legalisation evolves later.
                continue
            after_gaps = normalized_gaps(legal)
            after_max = float(max(after_gaps, default=0.0))
            after_min = float(min(after_gaps, default=0.0))
            if after_min < 0.18 or after_max > PRESENTATION_GAP_HARD_MAX:
                continue
            shifts = [
                abs(float(item["center_x"]) - float(original["center_x"]))
                for item, original in zip(legal, products)
            ]
            baseline_shifts = [
                abs(float(item["baseline_y"]) - float(original["baseline_y"]))
                for item, original in zip(legal, products)
            ]
            gap_balance = float(np.ptp(after_gaps)) if after_gaps else 0.0
            objective = (
                abs(after_max - 0.48) * 0.70
                + gap_balance * 0.45
                + max(shifts, default=0.0) * 0.65
                + max(baseline_shifts, default=0.0) * 0.35
                + abs(midpoint_offset) * 0.25
            )
            if best is None or objective < best[0]:
                best = (
                    objective,
                    legal,
                    after_max,
                    max(shifts, default=0.0),
                )

    metrics["seed_compaction_trials"] = float(trials)
    if best is None:
        return products, metrics, []
    _, compacted, after_max, max_shift = best
    metrics.update({
        "seed_compaction_applied": 1.0,
        "seed_gap_after_max": float(after_max),
        "seed_compaction_shift_max": float(max_shift),
        # This branch is forbidden from changing scale.  Surface this invariant
        # in diagnostics so regressions are visible without pixel inspection.
        "seed_compaction_scale_change_max": 0.0,
    })
    return compacted, metrics, [
        "MiniMax 商品语义、物理比例与左右顺序已保留；仅按真实 Alpha 边缘收紧过大空隙。",
    ]


def _largest_contiguous_column_segment(column: np.ndarray) -> np.ndarray:
    """Return the longest uninterrupted receiver run in a support column.

    Protected objects and concave tabletops can split one image column into
    several valid-looking runs.  Automatic staging must use the same run both
    while solving receiver-local depth and while converting that depth back
    into an image-space baseline; otherwise a correct foreground target can
    silently land in a different, rear support island.
    """
    if len(column) == 0:
        return column
    breaks = np.flatnonzero(np.diff(column) > 1)
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, len(column) - 1]
    start, end = max(
        zip(starts, ends),
        key=lambda pair: int(pair[1]) - int(pair[0]),
    )
    return column[int(start):int(end) + 1]


def _three_product_staging_depths(
    usable: np.ndarray,
    centers: list[float],
    display_heights: list[float],
    initial: list[float],
) -> list[float]:
    """Fit a visible staging triangle to the current receiver columns.

    A fixed normalised receiver depth does not project to a fixed visual
    separation: a shallow tabletop may turn a seemingly healthy 0.15 depth
    range into only a few image pixels.  Solve in image space instead.  The
    low-salience bridge (slot 1) sits about one quarter of a displayed object
    height in front of the anchors, while every base retains at least 0.30
    object heights of front-edge clearance.  The returned values are converted
    back to the receiver-local depth coordinate used by perspective scaling.
    """
    if len(centers) != 3 or len(display_heights) != 3:
        return initial
    height, width = usable.shape
    intervals: list[tuple[float, float]] = []
    for center in centers:
        px = int(np.clip(round(center * (width - 1)), 0, width - 1))
        column = np.flatnonzero(usable[:, px])
        segment = _largest_contiguous_column_segment(column)
        if len(segment) < 3:
            return initial
        # Planning support can be split by protected islands.  The largest
        # contiguous receiver segment is the only defensible automatic stage;
        # legalize_placements remains the authoritative footprint gate.
        intervals.append((
            float(segment[0]) / height,
            float(segment[-1]) / height,
        ))

    heights = np.asarray(display_heights, np.float64)
    median_height = float(np.median(heights))
    if median_height <= 1e-6:
        return initial
    backs = np.asarray([value[0] for value in intervals], np.float64)
    fronts = np.asarray([value[1] for value in intervals], np.float64)
    spans = fronts - backs
    if np.any(spans <= 1e-4):
        return initial
    lower = backs
    receiver_capacity = float(np.median(spans) / median_height)
    # On a normal oblique/overhead receiver the familiar 0.30-height front
    # breathing band and 0.22-height staging triangle remain desirable.  A
    # near-frontal shelf or coffee-table top can itself be only 0.25--0.40
    # product-heights deep in the image, so imposing those same image-space
    # numbers is mathematically impossible.  In that special case the exact
    # inset footprint gate remains hard while presentation depth scales to the
    # receiver that is actually visible.
    if receiver_capacity < 0.45:
        clearance_ratio = 0.025
        preferred_visible_ratio = float(np.clip(
            receiver_capacity * 0.55, 0.12, 0.18,
        ))
        maximum_depth_range = 0.90
    else:
        clearance_ratio = 0.30
        # Preserve the established deep/oblique receiver composition exactly;
        # the adaptive branch is only for genuinely depth-starved surfaces.
        preferred_visible_ratio = 0.26
        maximum_depth_range = 0.36
    required_clearance = heights * clearance_ratio
    upper = fronts - required_clearance
    if np.any(upper <= lower):
        return initial

    legacy_deep_receiver = receiver_capacity >= 0.45
    anchor_offset = (
        0.05 * median_height
        if legacy_deep_receiver
        else 0.04 * median_height
    )
    comfortable = float(np.median(backs + spans * 0.56))
    candidates: list[tuple[float, list[float]]] = []
    if legacy_deep_receiver:
        visible_candidates = [0.26, 0.28, 0.24, 0.22]
    else:
        visible_candidates = []
        for value in (
            preferred_visible_ratio + 0.02,
            preferred_visible_ratio,
            preferred_visible_ratio - 0.02,
        ):
            value = float(np.clip(value, 0.10, 0.24))
            if not any(
                abs(value - existing) < 1.0e-6
                for existing in visible_candidates
            ):
                visible_candidates.append(value)
    # ``visible_ratio`` is the bridge's displacement from the interpolated
    # anchor line, rather than merely from the left item.  Equal restrained
    # anchor offsets make that image-space contract explicit on sloping tops.
    for visible_ratio in visible_candidates:
        if legacy_deep_receiver:
            offsets = np.asarray(
                [0.0, visible_ratio * median_height, anchor_offset],
                np.float64,
            )
        else:
            offsets = np.asarray(
                [
                    anchor_offset,
                    anchor_offset + visible_ratio * median_height,
                    anchor_offset,
                ],
                np.float64,
            )
        reference_low = float(np.max(lower - offsets))
        if legacy_deep_receiver:
            reference_high = float(np.min(upper - offsets))
            depth_ceiling = 0.905
        else:
            # On a depth-starved receiver the bridge offset consumes most of
            # the visible surface.  Choosing the most comfortable reference
            # first and rejecting it afterwards could miss a perfectly valid
            # triangle only one preview row farther back.  Fold the receiver-
            # local depth ceiling into the feasible interval before clipping.
            depth_ceiling = 0.90
            depth_upper = np.minimum(
                upper,
                backs + spans * depth_ceiling,
            )
            reference_high = float(np.min(depth_upper - offsets))
        if reference_low > reference_high:
            continue
        reference = float(np.clip(comfortable, reference_low, reference_high))
        baselines = reference + offsets
        depths = np.clip((baselines - backs) / spans, 0.0, 1.0)
        depth_range = float(np.ptp(depths))
        clearance_margin = float(np.min(
            (fronts - baselines) / np.maximum(1.0e-6, required_clearance),
        ))
        # On an oblique or round receiver, a visually clear 0.26-height
        # triangle can correspond to only ~0.08 of receiver-local depth.
        # Rejecting that solution forced the function back to the coarse
        # [rear, front, rear] preset, which could scatter the three products
        # by nearly a full displayed height in image space.  Visual staging
        # and front clearance are the primary constraints here; retain only a
        # modest independent receiver-depth floor to stop a sloped edge from
        # faking the triangle.
        if not (
            0.08 <= depth_range <= maximum_depth_range
            and clearance_margin >= 0.999
            and float(np.max(depths)) <= depth_ceiling + 1.0e-6
        ):
            continue
        objective = (
            abs(visible_ratio - preferred_visible_ratio)
            + abs(depth_range - min(0.26, maximum_depth_range * 0.72)) * 0.16
            + abs(reference - comfortable) * 0.05
        )
        candidates.append((objective, [float(value) for value in depths]))
    if not candidates:
        return initial
    return min(candidates, key=lambda value: value[0])[1]


def _restore_triangle_after_legalization(
    canvas_size: tuple[int, int],
    support_polygon: list[list[float]],
    protected_polygons: list[list[list[float]]],
    products: list[dict[str, Any]],
    visual_order: list[int],
    shadow_direction: tuple[float, float] | list[float],
    shadow_length_ratio: float,
    shadow_softness_ratio: float,
    support_mask: np.ndarray | None,
) -> list[dict[str, Any]]:
    """Recover a visible foreground bridge after safety legalization.

    A wide handle or irregular base can legitimately pull one rear anchor
    toward the camera.  That safety move must survive, but it can flatten the
    staging triangle produced immediately beforehand.  Recompute the anchor
    line from the already-legal result and move only the centre bridge forward
    enough to recover the intended commercial composition.  The ordinary
    legalizer and front-clearance diagnostics remain authoritative.
    """
    if len(products) != 3 or len(visual_order) != 3:
        return products
    left_index, bridge_index, right_index = visual_order
    left = products[left_index]
    bridge = products[bridge_index]
    right = products[right_index]
    left_x = float(left["center_x"])
    bridge_x = float(bridge["center_x"])
    right_x = float(right["center_x"])
    if not (left_x < bridge_x < right_x) or right_x - left_x <= 1e-5:
        return products

    blend = (bridge_x - left_x) / (right_x - left_x)
    anchor_line = (
        float(left["baseline_y"]) * (1.0 - blend)
        + float(right["baseline_y"]) * blend
    )
    median_height = float(np.median([
        float(item["height_ratio"]) for item in products
    ]))
    if median_height <= 1e-6:
        return products
    current_offset = float(bridge["baseline_y"]) - anchor_line
    target_offset = median_height * VISIBLE_STAGING_RESTORE_TARGET_RATIO
    if current_offset >= median_height * VISIBLE_STAGING_RESTORE_MIN_RATIO:
        return products

    proposed = [dict(item) for item in products]
    proposed[bridge_index]["baseline_y"] = float(
        float(bridge["baseline_y"]) + target_offset - current_offset
    )
    try:
        legal, _ = legalize_placements(
            canvas_size,
            support_polygon,
            protected_polygons,
            proposed,
            automatic_edge_margin_ratio=0.012,
            shadow_direction=shadow_direction,
            shadow_length_ratio=shadow_length_ratio,
            shadow_softness_ratio=shadow_softness_ratio,
            support_mask=support_mask,
        )
    except ValueError:
        return products

    # Do not let recovery of the bridge disturb either already-safe anchor.
    for index in (left_index, right_index):
        if (
            abs(float(legal[index]["center_x"]) - float(products[index]["center_x"]))
            > 0.004
            or abs(
                float(legal[index]["baseline_y"])
                - float(products[index]["baseline_y"])
            ) > 0.004
        ):
            return products
    after_left = legal[left_index]
    after_bridge = legal[bridge_index]
    after_right = legal[right_index]
    after_blend = (
        float(after_bridge["center_x"]) - float(after_left["center_x"])
    ) / max(
        1e-6,
        float(after_right["center_x"]) - float(after_left["center_x"]),
    )
    after_anchor_line = (
        float(after_left["baseline_y"]) * (1.0 - after_blend)
        + float(after_right["baseline_y"]) * after_blend
    )
    after_offset = float(after_bridge["baseline_y"]) - after_anchor_line
    diagnostics = layout_surface_diagnostics(
        canvas_size, support_polygon, legal, support_mask,
    )
    if (
        after_offset < median_height * VISIBLE_STAGING_RESTORE_MIN_RATIO
        or diagnostics["front_clearance_min_heights"]
        < FRONT_CLEARANCE_MIN_HEIGHTS
    ):
        return products
    return legal


def _restore_triangle_after_readability(
    canvas_size: tuple[int, int],
    support_polygon: list[list[float]],
    protected_polygons: list[list[list[float]]],
    before_readability: list[dict[str, Any]],
    products: list[dict[str, Any]],
    visual_order: list[int],
    shadow_direction: tuple[float, float] | list[float],
    shadow_length_ratio: float,
    shadow_softness_ratio: float,
    support_mask: np.ndarray | None,
) -> list[dict[str, Any]]:
    """Recover staging depth without undoing a readability displacement.

    Background readability is deliberately evaluated after ordinary geometry
    legalization.  A clear side anchor can therefore move deeper into the
    receiver to escape a support/backdrop seam *after* the foreground bridge
    was restored, flattening the final three-product group.  Treat every item
    moved by the readability pass as locked, and recover the visible triangle
    with the remaining bridge/anchor baselines.  Every candidate goes through
    the authoritative legalizer and must preserve transparent boundary-seam
    cost, so commercial staging cannot trade away the repair that triggered
    this second pass.
    """
    if (
        len(products) != 3
        or len(before_readability) != 3
        or len(visual_order) != 3
    ):
        return products
    left_index, bridge_index, right_index = visual_order
    left = products[left_index]
    bridge = products[bridge_index]
    right = products[right_index]
    left_x = float(left["center_x"])
    bridge_x = float(bridge["center_x"])
    right_x = float(right["center_x"])
    if not (left_x < bridge_x < right_x) or right_x - left_x <= 1e-5:
        return products

    median_height = float(np.median([
        float(item["height_ratio"]) for item in products
    ]))
    if median_height <= 1e-6:
        return products
    blend = (bridge_x - left_x) / (right_x - left_x)
    anchor_line = (
        float(left["baseline_y"]) * (1.0 - blend)
        + float(right["baseline_y"]) * blend
    )
    current_offset = float(bridge["baseline_y"]) - anchor_line
    if current_offset >= median_height * VISIBLE_STAGING_RESTORE_MIN_RATIO:
        return products

    locked = {
        index
        for index, (before, after) in enumerate(
            zip(before_readability, products)
        )
        if (
            abs(float(after["center_x"]) - float(before["center_x"])) > 0.003
            or abs(
                float(after["baseline_y"])
                - float(before["baseline_y"])
            ) > 0.003
        )
    }
    if len(locked) == len(products):
        return products

    # Positive motion along each degree of freedom increases the bridge's
    # visible offset: the bridge moves toward camera (+Y), anchors move rearward
    # (-Y).  Least-squares allocations minimise total displacement; singleton
    # and pair allocations remain available when receiver geometry blocks one
    # of the otherwise preferable moves.
    freedoms = [
        (bridge_index, 1.0, 1.0),
        (left_index, 1.0 - blend, -1.0),
        (right_index, blend, -1.0),
    ]
    freedoms = [value for value in freedoms if value[0] not in locked]
    if not freedoms:
        return products
    allocations: list[list[tuple[int, float, float]]] = [
        [value] for value in freedoms
    ]
    if len(freedoms) >= 2:
        allocations.extend([
            [freedoms[i], freedoms[j]]
            for i in range(len(freedoms))
            for j in range(i + 1, len(freedoms))
        ])
    if len(freedoms) >= 3:
        allocations.append(list(freedoms))

    preview_w, preview_h, _ = _preview_shape(canvas_size)
    support, protected = _masks(
        (preview_h, preview_w), support_polygon, protected_polygons,
        support_mask,
    )
    rgba_cache: dict[str, np.ndarray] = {}

    def alpha_mask(item: dict[str, Any]) -> np.ndarray:
        path = str(item.get("rgba") or item["source"])
        rgba = rgba_cache.get(path)
        if rgba is None:
            rgba = load_rgba(path)
            rgba_cache[path] = rgba
        return _placement_mask(rgba, item, (preview_h, preview_w))

    def visible_mask(item: dict[str, Any]) -> np.ndarray:
        return alpha_mask(item).astype(np.float32) / 255.0 > 0.02

    # Reject invalid trial baselines directly.  Calling the full legalizer for
    # every losing trial is catastrophically expensive: an invalid trial makes
    # it search hundreds of translation/scale repairs that are then discarded
    # because this function only accepts zero-repair candidates.  This gate is
    # deliberately the same geometry/shadow contract as legalize_placements;
    # the single winning candidate is still sent through that authoritative
    # function once to refresh z-order and guard against future contract drift.
    validation_support = support.copy()
    validation_protected = protected.copy()
    margin = max(
        2,
        int(round(min(preview_w, preview_h) * 0.012)),
    )
    margin_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (margin * 2 + 1, margin * 2 + 1),
    )
    inset = cv2.erode(
        validation_support.astype(np.uint8), margin_kernel,
    ).astype(bool)
    if np.count_nonzero(inset) >= preview_w * preview_h * 0.008:
        validation_support = inset
        validation_protected = cv2.dilate(
            validation_protected.astype(np.uint8), margin_kernel,
        ).astype(bool)

    def legal_trial_masks(
        items: list[dict[str, Any]],
    ) -> list[np.ndarray] | None:
        accepted: list[np.ndarray] = []
        for item in items:
            alpha = alpha_mask(item)
            mode = str(item.get("placement_mode", "upright"))
            if not _valid_mask(
                alpha,
                validation_support,
                validation_protected,
                accepted,
                mode,
            ):
                return None
            hard_ratio, soft_ratio = _planned_shadow_receiver_ratios(
                alpha,
                support,
                protected,
                shadow_direction,
                shadow_length_ratio,
                shadow_softness_ratio,
                mode,
            )
            if hard_ratio < 0.992 or soft_ratio < 0.965:
                return None
            accepted.append(alpha)
        return accepted

    baseline_seams = [
        _transparent_support_boundary_seam_loss(
            visible_mask(item), support, item,
        )
        for item in products
    ]

    best = products
    best_objective = float("inf")
    # Aim for 0.22 and relax only within the explicit recovery margin.  The
    # previous 0.18 fallback was internally accepted but was then rejected by
    # the UI's 0.20 visible-staging gate, making a successful solver result
    # impossible to submit from the real front end.
    for target_ratio in (
        VISIBLE_STAGING_RESTORE_TARGET_RATIO,
        0.21,
        VISIBLE_STAGING_RESTORE_MIN_RATIO,
    ):
        deficit = target_ratio * median_height - current_offset
        if deficit <= 1e-6:
            continue
        for allocation in allocations:
            coefficient_norm = sum(value[1] ** 2 for value in allocation)
            if coefficient_norm <= 1e-8:
                continue
            proposed = [dict(item) for item in products]
            for index, coefficient, direction in allocation:
                displacement = deficit * coefficient / coefficient_norm
                proposed[index]["baseline_y"] = float(
                    float(proposed[index]["baseline_y"])
                    + direction * displacement
                )
            candidate_masks = legal_trial_masks(proposed)
            if candidate_masks is None:
                continue

            after_left = proposed[left_index]
            after_bridge = proposed[bridge_index]
            after_right = proposed[right_index]
            after_span = (
                float(after_right["center_x"])
                - float(after_left["center_x"])
            )
            if after_span <= 1e-5:
                continue
            after_blend = (
                float(after_bridge["center_x"])
                - float(after_left["center_x"])
            ) / after_span
            after_anchor_line = (
                float(after_left["baseline_y"]) * (1.0 - after_blend)
                + float(after_right["baseline_y"]) * after_blend
            )
            after_offset = (
                float(after_bridge["baseline_y"]) - after_anchor_line
            )
            if (
                after_offset
                < median_height * VISIBLE_STAGING_RESTORE_MIN_RATIO
            ):
                continue
            diagnostics = layout_surface_diagnostics(
                canvas_size, support_polygon, proposed, support_mask,
            )
            if (
                diagnostics["front_clearance_min_heights"]
                < FRONT_CLEARANCE_MIN_HEIGHTS
            ):
                continue
            candidate_seams = [
                _transparent_support_boundary_seam_loss(
                    alpha.astype(np.float32) / 255.0 > 0.02,
                    support,
                    item,
                )
                for item, alpha in zip(proposed, candidate_masks)
            ]
            if any(
                after > before + 0.015
                for before, after in zip(baseline_seams, candidate_seams)
            ):
                continue
            vertical_motion = sum(
                abs(float(after["baseline_y"]) - float(before["baseline_y"]))
                for before, after in zip(products, proposed)
            )
            horizontal_motion = sum(
                abs(float(after["center_x"]) - float(before["center_x"]))
                for before, after in zip(products, proposed)
            )
            objective = (
                vertical_motion
                + horizontal_motion * 4.0
                + abs(
                    after_offset / median_height
                    - VISIBLE_STAGING_RESTORE_TARGET_RATIO
                ) * 0.40
            )
            if objective < best_objective:
                best = proposed
                best_objective = objective
    if best is products:
        return products
    try:
        legal, repair_messages = legalize_placements(
            canvas_size,
            support_polygon,
            protected_polygons,
            best,
            automatic_edge_margin_ratio=0.012,
            shadow_direction=shadow_direction,
            shadow_length_ratio=shadow_length_ratio,
            shadow_softness_ratio=shadow_softness_ratio,
            support_mask=support_mask,
        )
    except ValueError:
        return products
    if repair_messages or any(
        abs(float(legal[index]["center_x"])
            - float(products[index]["center_x"])) > 0.004
        or abs(float(legal[index]["baseline_y"])
               - float(products[index]["baseline_y"])) > 0.004
        for index in locked
    ):
        return products
    return legal


def _commercial_readability_scale(
    nominal_heights: list[float],
    geometries: list[ProductGeometry],
    target_mass_height: float,
    *,
    identity_floors: list[float] | None = None,
    physical_evidence: list[float] | None = None,
) -> float:
    """Return one identity-preserving scale for a three-product hero group.

    Physical product/receiver estimates are useful for *relative* scale, but
    the receiver dimension is often an uncertain diameter, visible span or
    catalogue prior.  Blending it as an absolute size can therefore reduce an
    otherwise good three-product composition to thumbnail-sized merchandise.

    Product advertising also has a separate identity-readability requirement.
    A common factor may preserve semantic height ratios when the physical
    evidence is weak, but it must not enlarge an entirely unlabelled group or
    override trustworthy product/receiver dimensions.  Per-product identity
    floors handle the labelled package that actually needs extra pixels.
    """
    if len(nominal_heights) != 3 or len(geometries) != 3:
        return 1.0
    # A three-object arrangement is not, by itself, evidence that every item
    # should be enlarged for advertising readability.  The previous common
    # floor also enlarged unlabelled decor, sculptures and optical vessels,
    # even after trustworthy product/receiver dimensions had already produced
    # a physically coherent scale.  Keep the common commercial boost only for
    # groups that contain an actual identity-readability requirement.
    if identity_floors is not None and not any(
        float(value) > 1e-6 for value in identity_floors
    ):
        return 1.0
    visual_masses = np.asarray([
        max(0.0, float(height))
        * float(np.clip(geometry.visual_mass_fraction, 0.20, 1.0))
        for height, geometry in zip(nominal_heights, geometries)
    ], np.float64)
    if visual_masses.size != 3 or not np.all(np.isfinite(visual_masses)):
        return 1.0
    ordered = np.sort(np.maximum(visual_masses, 1e-6))

    # These floors are relative to the scene-derived single-product mass
    # prior, so they adapt to shallow, oblique and deep receiving surfaces.
    # The smallest item gets the strongest guard because illegible secondary
    # merchandise is the most common failure.  Median and anchor guards keep
    # a group with one naturally tiny item from making the two hero products
    # look timid.
    smallest_floor = target_mass_height * 1.42
    median_floor = target_mass_height * 1.64
    anchor_floor = target_mass_height * 2.10
    requested = max(
        1.0,
        smallest_floor / ordered[0],
        median_floor / ordered[1],
        anchor_floor / ordered[2],
    )

    # Keep the common factor below a commercial emphasis ceiling and below
    # the point where the later 0.42 height clamp would alter only the largest
    # item (which would destroy relative physical size).
    largest_nominal = max(1e-6, max(float(value) for value in nominal_heights))
    ratio_preserving_cap = 0.405 / (largest_nominal * 1.08)
    upper_bound = max(1.0, min(2.00, ratio_preserving_cap))
    scale = float(np.clip(requested, 1.0, upper_bound))

    # When a majority of the group has a jointly trustworthy product-height
    # and receiver-size estimate, that physical evidence is authoritative.
    # Readability may add at most a restrained five percent to the *whole*
    # group; the per-product identity floor below remains available for the
    # one labelled package that actually needs it.  This prevents one small
    # labelled item from making two otherwise unlabelled hero objects 30--60%
    # too large.
    if physical_evidence is not None and physical_evidence:
        evidence = np.asarray([
            float(np.clip(value, 0.0, 1.0)) for value in physical_evidence
        ], np.float64)
        if evidence.size and float(np.median(evidence)) >= 0.35:
            scale = min(scale, 1.05)
    return scale


def _identity_readability_floor(
    item: dict[str, Any],
    target_mass_height: float,
) -> float:
    """Return a display-height floor for text/label-bearing merchandise.

    Pixel identity alone is insufficient when a physically smaller secondary
    product is reduced until its label strokes are only three or four pixels
    high.  Keep a restrained presentation floor for packaged goods while
    leaving unlabelled decor and clear optical vessels on physical scale.
    """
    semantic = " ".join((
        str(item.get("semantic_category", "")),
        str(item.get("physical_type", "")),
    )).lower()
    label_tokens = {
        "can", "bottle", "package", "packaging", "box", "carton",
        "cosmetic", "jar", "tube", "pouch", "branded", "label",
    }
    semantic_tokens = set(re.findall(r"[a-z0-9]+", semantic))
    material = str(item.get("material_mode", "opaque")).lower()
    if material == "clear_glass" and not any(
        token in semantic for token in ("bottle", "label", "branded")
    ):
        return 0.0
    identity_likely = (
        bool(semantic_tokens & label_tokens)
        or str(item.get("rerender_policy", "auto")).lower() == "never"
        or str(item.get("identity_status", "unknown")).lower() == "detected"
    )
    if not identity_likely:
        return 0.0
    # Delivery is 4K or larger: a 13.5--17% display height already leaves
    # roughly 290--370 vertical pixels for the original label.  The previous
    # 23.5% absolute floor was equivalent to more than 500 px at 4K and could
    # make a 9 cm can almost as tall as a 28 cm hero product.  Preserve source
    # RGB/identity at render time; do not solve legibility by violating the
    # scene's physical-size semantics here.
    identity_extra = 0.012 if (
        str(item.get("rerender_policy", "auto")).lower() == "never"
        or str(item.get("identity_status", "unknown")).lower() == "detected"
    ) else 0.0
    return float(np.clip(
        max(0.135, target_mass_height * 1.05) + identity_extra,
        0.135,
        0.175,
    ))


def _coherent_group_repair(
    canvas_size: tuple[int, int],
    support_polygon: list[list[float]],
    protected_polygons: list[list[list[float]]],
    desired: list[dict[str, Any]],
    greedy: list[dict[str, Any]],
    greedy_messages: list[str],
    *,
    shadow_direction: tuple[float, float] | list[float] | None,
    shadow_length_ratio: float,
    shadow_softness_ratio: float,
    support_mask: np.ndarray | None,
    rgba_cache: dict[str, np.ndarray],
    placement_mask_cache: dict[tuple[Any, ...], np.ndarray],
) -> tuple[list[dict[str, Any]], list[str], dict[str, float]]:
    """Repair a three-item group as one composition before per-item fallback.

    The ordinary legalizer is intentionally conservative and processes items
    in source order.  If one large item intersects a protected background
    object, its minimal independent repair can collapse a previously balanced
    edge gap and push the other items toward the receiver rim.  Search a small
    deterministic family of *uniform* group transforms instead: all products
    keep their relative physical scale, horizontal rhythm and semantic order.
    Only when no exact, zero-repair group candidate exists do we retain the
    old per-item result.
    """
    empty = {
        "coherent_group_repair_applied": 0.0,
        "coherent_group_repair_scale": 1.0,
        "coherent_group_repair_dx": 0.0,
        "coherent_group_repair_dy": 0.0,
        "coherent_group_repair_trials": 0.0,
    }
    if len(desired) != 3 or len(greedy) != 3 or not greedy_messages:
        return greedy, greedy_messages, empty

    mean_dx = float(np.mean([
        float(after["center_x"]) - float(before["center_x"])
        for before, after in zip(desired, greedy)
    ]))
    mean_dy = float(np.mean([
        float(after["baseline_y"]) - float(before["baseline_y"])
        for before, after in zip(desired, greedy)
    ]))
    scales = (0.92, 0.88, 0.84, 0.82, 0.80)
    dx_values = tuple(dict.fromkeys(round(
        float(np.clip(mean_dx + delta, -0.16, 0.16)), 6,
    ) for delta in (-0.04, 0.0, 0.04)))
    dy_values = tuple(dict.fromkeys(round(
        float(np.clip(mean_dy + delta, -0.08, 0.08)), 6,
    ) for delta in (-0.02, 0.0, 0.02)))
    best: tuple[
        float, list[dict[str, Any]], float, float, float,
    ] | None = None
    trials = 0
    for scale_factor in scales:
        for dx in dx_values:
            for dy in dy_values:
                trials += 1
                candidate: list[dict[str, Any]] = []
                for item in desired:
                    transformed = dict(item)
                    # Uniformly shrink the product silhouettes, not the
                    # distances between their centres.  Scaling both at once
                    # cancels the extra breathing room we are trying to buy
                    # and recreates the same crowded group at a smaller size.
                    transformed["center_x"] = float(
                        float(item["center_x"]) + dx
                    )
                    transformed["baseline_y"] = float(
                        float(item["baseline_y"]) + dy
                    )
                    transformed["height_ratio"] = float(
                        float(item["height_ratio"]) * scale_factor
                    )
                    candidate.append(transformed)
                try:
                    legal, messages = legalize_placements(
                        canvas_size,
                        support_polygon,
                        protected_polygons,
                        candidate,
                        automatic_edge_margin_ratio=0.012,
                        shadow_direction=shadow_direction,
                        shadow_length_ratio=shadow_length_ratio,
                        shadow_softness_ratio=shadow_softness_ratio,
                        support_mask=support_mask,
                        repair_invalid=False,
                        rgba_cache=rgba_cache,
                        placement_mask_cache=placement_mask_cache,
                    )
                except ValueError:
                    continue
                if messages:
                    continue
                diagnostics = layout_surface_diagnostics(
                    canvas_size, support_polygon, legal, support_mask,
                )
                clearance_required = float(diagnostics.get(
                    "front_clearance_required_heights",
                    FRONT_CLEARANCE_MIN_HEIGHTS,
                ))
                if (
                    diagnostics["surface_depth_max"] > 0.92
                    or diagnostics["front_clearance_min_heights"]
                    < clearance_required - 0.015
                ):
                    continue
                objective = (
                    (1.0 - scale_factor) * 0.55
                    + abs(dx) * 0.22
                    + abs(dy) * 0.30
                    + max(
                        0.0,
                        diagnostics["surface_depth_max"] - 0.82,
                    ) * 3.0
                    + max(
                        0.0,
                        diagnostics["surface_depth_median"] - 0.62,
                    ) * 2.0
                )
                if best is None or objective < best[0]:
                    best = (
                        float(objective), legal, scale_factor, dx, dy,
                    )

    metrics = dict(empty)
    metrics["coherent_group_repair_trials"] = float(trials)
    if best is None:
        return greedy, greedy_messages, metrics
    _, repaired, scale_factor, dx, dy = best
    metrics.update({
        "coherent_group_repair_applied": 1.0,
        "coherent_group_repair_scale": float(scale_factor),
        "coherent_group_repair_dx": float(dx),
        "coherent_group_repair_dy": float(dy),
    })
    return repaired, [
        "单件吸附会破坏构图，已改用保持相对比例与间距的整组联合修复。",
    ], metrics


def solve_aesthetic_layout(
    canvas_size: tuple[int, int],
    support_polygon: list[list[float]],
    protected_polygons: list[list[list[float]]],
    products: list[dict[str, Any]],
    shadow_direction: tuple[float, float] | list[float] = (0.75, -0.66),
    shadow_length_ratio: float = 0.040,
    shadow_softness_ratio: float = 0.15,
    background_rgb: np.ndarray | None = None,
    support_mask: np.ndarray | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Create a deterministic, perspective-aware three-object composition."""
    preview_w, preview_h, _ = _preview_shape(canvas_size)
    support, protected = _masks(
        (preview_h, preview_w), support_polygon, protected_polygons,
        support_mask,
    )
    # The final geometry gate intentionally allows an upright silhouette to
    # extend beyond the tabletop as long as its physical footprint remains on
    # the support plane.  Automatic composition needs a stronger preference:
    # do not put a valid footprint directly on a table edge when there is ample
    # room.  Work on an eroded *planning* surface while keeping the original
    # support mask for the authoritative final gate and for manual layouts.
    edge_margin = max(5, int(round(min(preview_w, preview_h) * 0.014)))
    edge_margin = _receiver_aware_edge_margin(support, edge_margin)
    edge_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (edge_margin * 2 + 1, edge_margin * 2 + 1),
    )
    planning_support = cv2.erode(support.astype(np.uint8), edge_kernel).astype(bool)
    protected_margin = max(7, edge_margin)
    protected_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (protected_margin * 2 + 1, protected_margin * 2 + 1),
    )
    usable = planning_support & ~cv2.dilate(
        protected.astype(np.uint8), protected_kernel,
    ).astype(bool)
    ys, xs = np.nonzero(usable)
    if len(xs) < preview_w * preview_h * 0.01:
        raise ValueError("桌面可用区域太小，无法容纳商品")
    x0, x1 = float(xs.min() / preview_w), float(xs.max() / preview_w)
    y0, y1 = float(ys.min() / preview_h), float(ys.max() / preview_h)
    plane_h = max(0.08, y1 - y0)
    # ``height_ratio`` is normalized by the canvas height while ``center_x``
    # and all horizontal support coordinates are normalized by the canvas
    # width.  Convert a source aspect ratio into X-normalized canvas units
    # before doing group fit, edge-gap or overlap math.  Omitting H/W makes a
    # landscape frame overestimate every product width (1.5x on a 3:2 photo),
    # which unnecessarily shrinks and spreads the entire arrangement.
    x_units_per_y_unit = preview_h / max(1.0, float(preview_w))
    geometries = [product_geometry(item.get("rgba") or item["source"]) for item in products]

    # Product photography usually reads most naturally at roughly 12–18% of
    # the frame height on a medium tabletop.  Larger values quickly resemble
    # oversized props even when they geometrically fit.
    # A front-facing table can expose a very thin top while still holding
    # normal 20–40 cm décor.  Preserve enough visual mass for products to read
    # as physical objects instead of tiny stickers.
    # Equalise the perceived body height, not the full alpha bounding box.
    # Thin stems, aerials and handles remain intact but no longer shrink the
    # main product to sticker scale.  The common group scale still keeps the
    # complete silhouettes within a restrained portion of the table span.
    # The earlier 11–13% body target passed bounding-box geometry yet repeatedly
    # read as small stickers beside real telephones, towels and table legs.
    # Scale the *visual mass* to 12.5–15% and let the existing group-width,
    # protected-region and footprint solvers reduce it only when necessary.
    target_mass_height = float(np.clip(0.115 + plane_h * 0.060, 0.125, 0.15))
    nominal_heights = [
        target_mass_height / max(0.20, geometry.visual_mass_fraction)
        for geometry in geometries
    ]
    semantic_sizes = np.asarray([
        float(item.get("estimated_real_height_cm", 20.0)) for item in products
    ], np.float64)
    semantic_confidences = np.asarray([
        float(item.get("size_confidence", 0.0)) for item in products
    ], np.float64)
    if len(semantic_sizes) and float(np.max(semantic_confidences)) > 0.0:
        size_median = float(np.median(np.clip(semantic_sizes, 2.0, 300.0)))
        for index in range(len(nominal_heights)):
            confidence = float(np.clip(semantic_confidences[index], 0.0, 1.0))
            # A low-confidence category prior remains conservatively
            # compressed, while a high-confidence can/pitcher/bottle estimate
            # should retain its real physical ratio.  A fixed 0.68 exponent
            # made a verified 25 cm pitcher only 1.52x a 14.5 cm can instead
            # of the physically expected 1.72x.
            size_exponent = 0.68 + 0.32 * confidence
            relative = (
                float(np.clip(semantic_sizes[index], 2.0, 300.0))
                / max(2.0, size_median)
            ) ** size_exponent
            nominal_heights[index] *= math.exp(confidence * math.log(max(0.35, relative)))
    receiver_sizes = np.asarray([
        float(item.get("receiver_width_or_diameter_cm", 100.0))
        for item in products
    ], np.float64)
    receiver_confidences = np.asarray([
        float(item.get("receiver_size_confidence", 0.0))
        for item in products
    ], np.float64)
    if len(receiver_sizes) and float(np.max(receiver_confidences)) >= 0.40:
        receiver_cm = float(np.median(np.clip(receiver_sizes, 20.0, 500.0)))
        support_span_height_units = (
            (x1 - x0) * preview_w / max(1.0, float(preview_h))
        )
        for index in range(len(nominal_heights)):
            absolute_target = (
                float(np.clip(semantic_sizes[index], 2.0, 300.0))
                / receiver_cm
                * support_span_height_units
            )
            absolute_evidence = float(np.clip(
                semantic_confidences[index] * receiver_confidences[index],
                0.0, 1.0,
            ))
            # Both inputs are already confidence terms.  Above the joint
            # reliability threshold the real product/receiver ratio is more
            # informative than a generic 12--15% frame-height prior, which
            # otherwise makes normal decor look like miniature stickers.
            absolute_blend = (
                float(np.clip(0.25 + 1.15 * absolute_evidence, 0.0, 0.94))
                if absolute_evidence >= 0.20 else absolute_evidence
            )
            nominal_heights[index] = math.exp(
                (1.0 - absolute_blend)
                * math.log(max(0.035, nominal_heights[index]))
                + absolute_blend * math.log(max(0.035, absolute_target))
            )
    identity_floors = [
        _identity_readability_floor(item, target_mass_height)
        for item in products
    ]
    joint_physical_evidence = [
        float(np.clip(
            float(item.get("size_confidence", 0.0))
            * float(item.get("receiver_size_confidence", 0.0)),
            0.0,
            1.0,
        ))
        for item in products
    ]
    commercial_scale = _commercial_readability_scale(
        nominal_heights,
        geometries,
        target_mass_height,
        identity_floors=identity_floors,
        physical_evidence=joint_physical_evidence,
    )
    nominal_heights = [
        float(height) * commercial_scale for height in nominal_heights
    ]
    total_width = sum(
        height * geometry.content_aspect * x_units_per_y_unit
        for height, geometry in zip(nominal_heights, geometries)
    )
    group_scale = min(1.0, (x1 - x0) * 0.78 / max(0.2, total_width))

    # Use semantic visual weight rather than alpha aspect as the grouping cue.
    # A quiet receiver benefits from a low-salience centre bridge.  A receiver
    # with a conspicuous central mounting plate needs the opposite strategy:
    # assign the widest opaque-enough product to the centre so it cleanly
    # occludes the hardware, leaving the transparent and narrow products on
    # quieter side slots.  This is a scene-aware product-photography decision,
    # not a hard-coded category order.
    indices = list(range(len(products)))
    central_structure_cost = 0.0
    central_occlusion_bridge = 0.0
    if len(indices) == 3:
        salience = [
            float(products[index].get("visual_salience", 0.5))
            for index in indices
        ]
        if background_rgb is not None:
            scene_cost = background_readability_cost_map(
                background_rgb, (preview_h, preview_w), support,
            )
            region = np.zeros_like(support, dtype=bool)
            center = (x0 + x1) * 0.5
            half_width = max(0.028, (x1 - x0) * 0.12)
            rx0 = int(np.clip(round((center - half_width) * preview_w), 0, preview_w))
            rx1 = int(np.clip(round((center + half_width) * preview_w), 0, preview_w))
            ry0 = int(np.clip(round((y0 + plane_h * 0.24) * preview_h), 0, preview_h))
            ry1 = int(np.clip(round((y0 + plane_h * 0.88) * preview_h), 0, preview_h))
            region[ry0:ry1, rx0:rx1] = True
            region &= support
            if np.count_nonzero(region) >= 24:
                values = scene_cost[region]
                central_structure_cost = (
                    0.45 * float(np.mean(values))
                    + 0.55 * float(np.percentile(values, 88.0))
                )
        if central_structure_cost >= 0.34:
            projected_areas = np.asarray([
                max(1e-6, nominal_heights[index] ** 2 * geometries[index].content_aspect)
                for index in indices
            ], np.float64)
            projected_areas /= max(1e-6, float(projected_areas.max()))
            occlusion = {
                "opaque": 1.0,
                "metal": 1.0,
                "mixed_transparent": 0.84,
                "colored_glass": 0.66,
                "frosted_glass": 0.38,
                "clear_glass": 0.08,
            }
            bridge = max(
                indices,
                key=lambda index: (
                    projected_areas[index]
                    * occlusion.get(
                        str(products[index].get("material_mode", "opaque")).lower(),
                        0.90,
                    )
                    + 0.18 * occlusion.get(
                        str(products[index].get("material_mode", "opaque")).lower(),
                        0.90,
                    )
                    + 0.06 * salience[index]
                ),
            )
            central_occlusion_bridge = 1.0
        else:
            flat_candidates = [
                index for index in indices
                if str(products[index].get("placement_mode", "upright"))
                == "surface_flat"
            ]
            if len(flat_candidates) == 1:
                # A single low, horizontally presented object is a natural
                # foreground plinth for two upright anchors.  Treating it as a
                # side column produces the familiar "three unrelated items in
                # a row" failure, especially for bowls, trays and plates.
                # The choice is driven by photographed pose, not by a product
                # name or a case-specific coordinate.
                bridge = flat_candidates[0]
            else:
                bridge = min(
                    indices,
                    key=lambda index: (
                        salience[index], geometries[index].content_aspect,
                    ),
                )
        sides = [index for index in indices if index != bridge]
        sides.sort(key=lambda index: (salience[index], nominal_heights[index]))
        visual_order = [sides[0], bridge, sides[1]]
    else:
        visual_order = indices

    # Build an entity-aware horizontal group.  Equal centre spacing looks
    # uneven whenever one item is much wider than another; equal *edge gaps*
    # create the intended visual rhythm and prevent bounding-box collisions.
    depth_targets = {
        1: [0.66],
        2: [0.60, 0.72],
        # Three-object product photography needs a visible staging triangle,
        # not merely three different receiver-depth numbers.  On a shallow
        # tabletop the earlier 0.17 depth spread projected to only about 13%
        # of an object's displayed height and still read as a ruler-straight
        # catalogue row.  Put the low-salience bridge clearly forward while
        # keeping the two visual anchors at slightly different rear depths.
        # The asymmetric rear pair avoids a mechanical isosceles template and
        # the existing perspective scale below remains tied to the same depth.
        # Keep a clearly readable staging triangle without pushing the front
        # bridge onto the receiver rim.  The previous 0.76 bridge depth left
        # only ~0.22 product-heights of front clearance on compact round
        # tables; pulling the whole group inward also gives reflections enough
        # clean receiver space below each base.
        3: [0.28, 0.70, 0.40],
    }[len(products)]
    display_heights = [0.0] * len(products)
    display_widths = [0.0] * len(products)
    support_span = x1 - x0
    # ``commercial_scale`` protects label/readability and is intentionally
    # capped when physical dimensions are trustworthy.  A separate, common
    # receiver-presentation scale controls how confidently the *whole group*
    # occupies the photographed tabletop.  Keeping these two contracts apart
    # preserves real inter-product size ratios while avoiding the familiar
    # "three miniatures in the middle of a large table" result.
    receiver_presentation_scale = 1.0

    def staged_display_height(product_index: int, perspective_scale: float) -> float:
        placement_mode = str(
            products[product_index].get("placement_mode", "upright")
        )
        # At normal review size, smaller values made short upright goods and
        # top-view trays read as incidental stickers beside a tall hero.  The
        # floors are display-space readability limits; semantic dimensions
        # still determine every relative scale above them.
        minimum_display_height = 0.070 if placement_mode == "surface_flat" else 0.095
        raw_height = float(
            nominal_heights[product_index]
            * group_scale
            * receiver_presentation_scale
            * perspective_scale
        )
        if placement_mode == "surface_flat":
            item = products[product_index]
            footprint_cm = float(item.get("footprint_width_cm", 0.0))
            receiver_cm = float(item.get("receiver_width_or_diameter_cm", 0.0))
            evidence = min(
                float(item.get("size_confidence", 0.0)),
                float(item.get("receiver_size_confidence", 0.0)),
            )
            if footprint_cm > 0.0 and receiver_cm >= 20.0 and evidence >= 0.35:
                # A plate/tray/bowl is perceived primarily by its horizontal
                # footprint.  Scaling it by its small physical *height* makes
                # a wide source crop dominate the whole table.  Project the
                # footprint against the receiver span, allow a restrained
                # commercial enlargement, then convert that width back to the
                # common height-ratio representation used by the renderer.
                target_width = (
                    footprint_cm / receiver_cm * support_span * 1.85
                )
                flat_height_ceiling = (
                    target_width
                    / max(
                        0.05,
                        geometries[product_index].content_aspect
                        * x_units_per_y_unit,
                    )
                    * 1.12
                )
                raw_height = min(
                    raw_height,
                    max(minimum_display_height, flat_height_ceiling),
                )
        height_ratio = float(np.clip(
            raw_height,
            minimum_display_height,
            0.42,
        ))
        readability_floor = identity_floors[product_index]
        if float(products[product_index].get("size_confidence", 0.0)) >= 0.55:
            # A trustworthy real-size prior is a harder contract than a generic
            # label-size preference.  At most a 12% local enlargement may be
            # spent on legibility; source-RGB preservation and 4K output carry
            # the remaining identity burden.
            readability_floor = min(
                readability_floor,
                max(height_ratio, raw_height * 1.12),
            )
        return max(height_ratio, readability_floor)

    for slot, product_index in enumerate(visual_order):
        # The receiver-depth estimate is not a calibrated camera model.  A
        # 20% slope made a deliberately low-salience foreground bridge up to
        # nine percent larger than a rear anchor, enough for a small saturated
        # can to steal the hero role from a physically larger pitcher.  Keep a
        # restrained +/-1% staging cue here; semantic real-world dimensions
        # remain the authoritative relative-scale signal.
        perspective_scale = 0.98 + 0.04 * depth_targets[slot]
        height_ratio = staged_display_height(product_index, perspective_scale)
        display_heights[product_index] = height_ratio
        display_widths[product_index] = (
            height_ratio
            * geometries[product_index].content_aspect
            * x_units_per_y_unit
        )
    visual_width = sum(display_widths[index] for index in visual_order)
    if len(products) == 3 and visual_width > 1.0e-6:
        # Product photography benefits from a legible common scale even when
        # generic real-size priors are conservative.  Target 56% *solid item
        # width*; the proportional edge gaps below bring the complete staged
        # group to roughly 66--70% of the usable receiver span.  This is a
        # uniform enlargement only: relative product dimensions, depth order,
        # alpha, labels and all source pixels remain unchanged.
        target_visual_width = support_span * 0.56
        if visual_width < target_visual_width:
            height_headroom = min(
                (
                    0.42 / max(1.0e-6, display_heights[index])
                    for index in visual_order
                ),
                default=1.0,
            )
            receiver_presentation_scale = float(np.clip(
                target_visual_width / visual_width,
                1.0,
                min(1.22, height_headroom),
            ))
            if receiver_presentation_scale > 1.001:
                for slot, product_index in enumerate(visual_order):
                    perspective_scale = 0.98 + 0.04 * depth_targets[slot]
                    height_ratio = staged_display_height(
                        product_index, perspective_scale,
                    )
                    display_heights[product_index] = height_ratio
                    display_widths[product_index] = (
                        height_ratio
                        * geometries[product_index].content_aspect
                        * x_units_per_y_unit
                    )
                visual_width = sum(
                    display_widths[index] for index in visual_order
                )
    # Negative space should be proportional to the displayed products, not a
    # fixed fraction of a large receiver.  A support-relative 62% span produced
    # catalogue-like rows whose empty gaps were as wide as a product.  Stage a
    # coherent cluster with measured edge gaps, bounded by the current support
    # width so the rule generalises across round, oblique and rectangular tops.
    median_display_width = float(np.median([
        display_widths[index] for index in visual_order
    ]))
    if len(products) == 3:
        desired_gap = float(np.clip(
            # Product presentation needs a visible inter-item breathing band.
            # The earlier 0.34 factor was geometrically legal but, after edge
            # antialiasing and reflection blur, neighbouring packages could
            # still read as one merged cluster.  A restrained 0.38 edge gap
            # preserves a coherent group without turning it into a sparse row.
            median_display_width * 0.38,
            support_span * 0.018,
            support_span * 0.060,
        ))
    elif len(products) == 2:
        desired_gap = float(np.clip(
            median_display_width * 0.52,
            support_span * 0.025,
            support_span * 0.075,
        ))
    else:
        desired_gap = 0.0
    group_target = min(
        support_span * 0.80,
        visual_width + desired_gap * max(0, len(products) - 1),
    )
    group_target = max(visual_width, group_target)
    gap = max(0.018, (group_target - visual_width) / max(1, len(products) - 1))
    group_width = visual_width + gap * max(0, len(products) - 1)
    cursor = (x0 + x1 - group_width) * 0.5
    centers = [0.0] * len(products)
    for product_index in visual_order:
        centers[product_index] = cursor + display_widths[product_index] * 0.5
        cursor += display_widths[product_index] + gap

    if len(products) == 3:
        slot_centers = [centers[index] for index in visual_order]
        slot_heights = [display_heights[index] for index in visual_order]
        depth_targets = _three_product_staging_depths(
            usable,
            slot_centers,
            slot_heights,
            list(depth_targets),
        )
        # Perspective scale and receiver depth are one physical quantity.  If
        # the receiver-column fit changed the staging depth, update the display
        # heights with the same bounded perspective law; semantic size ratios
        # and original product pixels remain untouched.
        for slot, product_index in enumerate(visual_order):
            perspective_scale = 0.98 + 0.04 * depth_targets[slot]
            height_ratio = staged_display_height(
                product_index, perspective_scale,
            )
            display_heights[product_index] = height_ratio
            display_widths[product_index] = (
                height_ratio
                * geometries[product_index].content_aspect
                * x_units_per_y_unit
            )
        # Depth refinement changes perspective scale and therefore product
        # width.  Rebuild the horizontal cluster from those final widths;
        # retaining centres computed from the pre-depth widths created large,
        # uneven holes around a small foreground bridge.
        final_visual_width = sum(
            display_widths[index] for index in visual_order
        )
        final_median_width = float(np.median([
            display_widths[index] for index in visual_order
        ]))
        final_desired_gap = float(np.clip(
            final_median_width * 0.38,
            support_span * 0.018,
            support_span * 0.060,
        ))
        final_group_target = min(
            support_span * 0.80,
            final_visual_width + final_desired_gap * 2,
        )
        final_group_target = max(final_visual_width, final_group_target)
        final_gap = max(
            0.018,
            (final_group_target - final_visual_width) / 2,
        )
        final_group_width = final_visual_width + final_gap * 2
        cursor = (x0 + x1 - final_group_width) * 0.5
        for product_index in visual_order:
            centers[product_index] = (
                cursor + display_widths[product_index] * 0.5
            )
            cursor += display_widths[product_index] + final_gap
        # The support interval changes with X on an oblique/round receiver.
        # Rebuilding the horizontal cluster above can therefore invalidate
        # depth ratios fitted at the earlier centres and flatten (or even
        # reverse) the intended foreground bridge.  Fit once more against the
        # final centres and sizes so the triangle survives the projection.
        depth_targets = _three_product_staging_depths(
            usable,
            [centers[index] for index in visual_order],
            [display_heights[index] for index in visual_order],
            list(depth_targets),
        )

    proposal: list[dict[str, Any] | None] = [None] * len(products)
    for slot, product_index in enumerate(visual_order):
        center_x = centers[product_index]
        px = int(np.clip(round(center_x * (preview_w - 1)), 0, preview_w - 1))
        column_y = np.flatnonzero(usable[:, px])
        segment_y = _largest_contiguous_column_segment(column_y)
        if len(segment_y):
            depth_index = int(round(depth_targets[slot] * (len(segment_y) - 1)))
            baseline_y = float(segment_y[depth_index] / preview_h)
        else:
            baseline_y = y0 + plane_h * depth_targets[slot]
        height_ratio = display_heights[product_index]
        item = dict(products[product_index])
        item.update({
            "center_x": center_x,
            "baseline_y": baseline_y,
            "height_ratio": height_ratio,
            "z_order": slot + 1,
        })
        proposal[product_index] = item

    proposal_items = [item for item in proposal if item is not None]
    rgba_cache: dict[str, np.ndarray] = {}
    placement_mask_cache: dict[tuple[Any, ...], np.ndarray] = {}
    repaired, messages = legalize_placements(
        canvas_size, support_polygon, protected_polygons,
        proposal_items,
        automatic_edge_margin_ratio=0.012,
        shadow_direction=shadow_direction,
        shadow_length_ratio=shadow_length_ratio,
        shadow_softness_ratio=shadow_softness_ratio,
        support_mask=support_mask,
        rgba_cache=rgba_cache,
        placement_mask_cache=placement_mask_cache,
    )
    repaired, messages, coherent_repair_metrics = _coherent_group_repair(
        canvas_size,
        support_polygon,
        protected_polygons,
        proposal_items,
        repaired,
        messages,
        shadow_direction=shadow_direction,
        shadow_length_ratio=shadow_length_ratio,
        shadow_softness_ratio=shadow_softness_ratio,
        support_mask=support_mask,
        rgba_cache=rgba_cache,
        placement_mask_cache=placement_mask_cache,
    )
    repaired = _restore_triangle_after_legalization(
        canvas_size,
        support_polygon,
        protected_polygons,
        repaired,
        visual_order,
        shadow_direction,
        shadow_length_ratio,
        shadow_softness_ratio,
        support_mask,
    )
    before_readability = [dict(item) for item in repaired]
    repaired, readability_metrics = _refine_background_readability(
        canvas_size,
        support_polygon,
        protected_polygons,
        repaired,
        background_rgb,
        shadow_direction,
        shadow_length_ratio,
        shadow_softness_ratio,
        support_mask,
    )
    repaired = _restore_triangle_after_readability(
        canvas_size,
        support_polygon,
        protected_polygons,
        before_readability,
        repaired,
        visual_order,
        shadow_direction,
        shadow_length_ratio,
        shadow_softness_ratio,
        support_mask,
    )
    # legalize receives source-list order; restore IDs are already stable.
    entities = sorted(
        (
            float(item["center_x"]) - float(item["height_ratio"]) * geometry.content_aspect * x_units_per_y_unit * 0.5,
            float(item["center_x"]) + float(item["height_ratio"]) * geometry.content_aspect * x_units_per_y_unit * 0.5,
        )
        for item, geometry in zip(repaired, geometries)
    )
    gaps = np.asarray([
        entities[index + 1][0] - entities[index][1]
        for index in range(len(entities) - 1)
    ], np.float64)
    negative_gap = float(np.maximum(0.0, -gaps).sum()) if gaps.size else 0.0
    positive_gaps = gaps[gaps > 1e-6]
    if positive_gaps.size >= 2:
        gap_ratio = float(positive_gaps.max() / positive_gaps.min())
    else:
        gap_ratio = 1.0
    median_entity_width = float(np.median([
        max(1e-6, right - left) for left, right in entities
    ])) if entities else 1.0
    gap_to_width = (
        float(np.median(positive_gaps) / median_entity_width)
        if positive_gaps.size else 0.0
    )
    diagnostics = layout_surface_diagnostics(
        canvas_size, support_polygon, repaired, support_mask,
    )
    final_visual_masses = np.asarray([
        float(item["height_ratio"]) * geometry.visual_mass_fraction
        for item, geometry in zip(repaired, geometries)
    ], np.float64)
    score = 98.0
    score -= len(messages) * 1.5
    score -= negative_gap * 500.0
    score -= max(0.0, gap_ratio - 2.5) * 3.0
    score -= max(0.0, gap_to_width - PRESENTATION_GAP_TARGET_MAX) * 24.0
    score -= max(0.0, 0.18 - gap_to_width) * 10.0
    score -= max(
        0.0,
        diagnostics["surface_depth_max"]
        - diagnostics.get("automatic_depth_max", 0.76),
    ) * 80.0
    score -= max(0.0, diagnostics["surface_depth_median"] - 0.65) * 50.0
    score -= max(
        0.0,
        diagnostics.get(
            "front_clearance_required_heights",
            FRONT_CLEARANCE_MIN_HEIGHTS,
        ) - diagnostics["front_clearance_min_heights"],
    ) * 25.0
    score -= max(0.0, 0.035 - diagnostics["surface_depth_range"]) * 80.0
    score = float(np.clip(score, 0.0, 100.0))
    return repaired, {
        "layout_score": score,
        "usable_surface_ratio": float(usable.mean()),
        "automatic_repairs": float(len(messages)),
        "commercial_readability_scale": float(commercial_scale * group_scale),
        "receiver_presentation_scale": float(receiver_presentation_scale),
        "receiver_product_width_occupancy": float(
            sum(display_widths[index] for index in visual_order)
            / max(1.0e-6, support_span)
        ),
        "commercial_identity_guard_count": float(sum(
            float(value) > 1e-6 for value in identity_floors
        )),
        "commercial_physical_evidence_median": float(np.median(
            np.asarray(joint_physical_evidence, np.float64)
        )) if joint_physical_evidence else 0.0,
        "commercial_visual_mass_min": float(final_visual_masses.min()),
        "commercial_visual_mass_median": float(np.median(final_visual_masses)),
        "commercial_visual_mass_max": float(final_visual_masses.max()),
        "identity_readability_floor_max": float(max(identity_floors, default=0.0)),
        "entity_gap_ratio": gap_ratio,
        "entity_gap_to_median_width": gap_to_width,
        "central_structure_cost": float(central_structure_cost),
        "central_occlusion_bridge": float(central_occlusion_bridge),
        **coherent_repair_metrics,
        **readability_metrics,
        **diagnostics,
    }
