from __future__ import annotations

"""Deterministic, image-space evidence for the measurable realism gates.

The visual model remains useful for colour, material and overall photographic
judgement, but it is a poor instrument for questions that the renderer can
measure exactly (for example whether a base touches its shadow, whether every
cast shadow follows the declared vector, or whether a footprint lies on the
receiver).  This module records those measurements in ``audit.json`` so the
final gate is both reproducible and bound to the rendered artifact.
"""

from typing import Any, Iterable

import cv2
import numpy as np

from .contracts import Lighting
from .render import (
    RenderedProduct,
    _linear_to_srgb_u8,
    _srgb_u8_to_linear,
    alpha_over,
    contact_profile,
    darken,
)


PHYSICAL_AUDIT_SCHEMA_VERSION = "physical-evidence-v1"
PHYSICAL_SCORE_FLOORS: dict[str, float] = {
    "grounding": 95.0,
    "cast_shadows": 90.0,
    "geometry_layout": 98.0,
    "cutout_edges": 90.0,
}


def _luminance(image: np.ndarray) -> np.ndarray:
    """Return deterministic Rec.709-like image-space luminance.

    Delivered visibility is measured in encoded RGB.  Expected shadow response
    is resolved separately with the compositor's linear-light transfer
    function before being brought back to this same display space.
    """

    rgb = image.astype(np.float32)
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def _percentile(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values.size else 0.0


def _distance_band(mask: np.ndarray, max_distance: float) -> np.ndarray:
    """Return a narrow distance band without transforming the whole 4K frame."""

    occupied = np.argwhere(mask)
    band = np.zeros_like(mask, bool)
    if occupied.size == 0:
        return band
    margin = int(np.ceil(max_distance)) + 2
    y0, x0 = occupied.min(axis=0)
    y1, x1 = occupied.max(axis=0)
    y0 = max(0, int(y0) - margin)
    x0 = max(0, int(x0) - margin)
    y1 = min(mask.shape[0], int(y1) + margin + 1)
    x1 = min(mask.shape[1], int(x1) + margin + 1)
    local = mask[y0:y1, x0:x1]
    distance = cv2.distanceTransform((~local).astype(np.uint8), cv2.DIST_L2, 3)
    band[y0:y1, x0:x1] = distance <= max_distance
    return band


def _final_pixel_visibility(
    layer: RenderedProduct,
    footprint: np.ndarray,
    body: np.ndarray,
    all_bodies: np.ndarray,
    receiver: np.ndarray,
    object_height: int,
    lighting: Lighting,
    background: np.ndarray | None,
    final: np.ndarray | None,
    pixel_context: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Measure whether contact/reflection layers survive into final pixels.

    Layer alpha alone is not evidence of a visible photographic effect.  This
    routine compares the delivered bitmap with the immutable background only
    in receiver-side pixels outside every product body.  It also records when
    the receiver is too dark to provide meaningful shadow contrast, avoiding
    the harmful response of blindly increasing black opacity on a dark table.
    """

    if background is None or final is None:
        return {
            "measured": False,
            "policy": "legacy_call_without_final_pixels",
            "contact": {"status": "not_measured", "score": None},
            "reflection": {"status": "not_measured", "score": None},
            "warnings": [],
            "failures": [],
        }
    if background.shape != final.shape or background.shape[:2] != body.shape:
        raise ValueError("Final-pixel audit inputs must share one image shape")

    if pixel_context is None:
        background_luma = _luminance(background)
        final_luma = _luminance(final)
        actual_rgb_delta = np.max(
            np.abs(final.astype(np.int16) - background.astype(np.int16)), axis=2,
        ).astype(np.float32)
        actual_darkening = np.maximum(background_luma - final_luma, 0.0)
    else:
        background_luma = pixel_context["background_luma"]
        actual_rgb_delta = pixel_context["actual_rgb_delta"]
        actual_darkening = pixel_context["actual_darkening"]
    warnings: list[str] = []
    failures: list[str] = []

    # Contact visibility is evaluated in a narrow exterior band around the
    # measured physical foot.  Broad cast-shadow pixels are deliberately not
    # allowed to stand in for the attached receiver-side seam.
    contact_alpha = layer.contact_shadow.astype(np.float32)
    exterior_candidate = (contact_alpha > 0) & receiver & ~body
    if np.any(footprint):
        max_distance = float(max(4, min(18, round(object_height * 0.08))))
        exterior_candidate &= _distance_band(footprint, max_distance)
    pre_occlusion_count = int(np.count_nonzero(exterior_candidate))
    contact_candidate = exterior_candidate & ~all_bodies
    positive = contact_alpha[contact_candidate]
    if positive.size:
        signal_threshold = max(2.0, _percentile(positive, 45.0))
        contact_zone = contact_candidate & (contact_alpha >= signal_threshold)
    else:
        contact_zone = contact_candidate
    contact_pixels = int(np.count_nonzero(contact_zone))
    has_receiver_counterfactual = bool(
        pixel_context is not None
        and "contact_cast_union" in pixel_context
        and "contact_reflections" in pixel_context
    )
    contact: dict[str, Any] = {
        "zone_pixel_count": contact_pixels,
        "pre_occlusion_zone_pixel_count": pre_occlusion_count,
        "comparison": (
            "final_vs_counterfactual_receiver_before_contact_ao_"
            "outside_all_product_bodies"
            if has_receiver_counterfactual else
            "final_vs_original_background_outside_all_product_bodies"
        ),
    }
    if pre_occlusion_count and contact_pixels < max(4, int(pre_occlusion_count * 0.08)):
        contact.update({
            "status": "occluded_not_measurable",
            "score": None,
            "reason": "another_product_covers_the_receiver_side_contact_zone",
        })
        warnings.append("contact_visibility_occluded_not_measurable")
    elif not contact_pixels:
        contact.update({
            "status": "missing_contact_signal",
            "score": 0.0,
            "reason": "no_contact_layer_pixels_exist_outside_the_product_body",
        })
        failures.append("contact_final_pixel_signal_missing")
    else:
        # Cast loss and planar reflections are composited before contact AO.
        # Comparing the delivered image directly with the immutable background
        # therefore understates an AO seam wherever a reflection brightened the
        # receiver first.  Reconstruct the exact receiver state immediately
        # before AO, but only for this narrow zone so 4K/7K audits do not need
        # another full-frame RGB buffer.
        zone_background_rgb = background[contact_zone]
        if has_receiver_counterfactual:
            zone_background_rgb = darken(
                zone_background_rgb,
                pixel_context["contact_cast_union"][contact_zone],
            )
            for reflection_rgb, reflection_alpha in pixel_context["contact_reflections"]:
                zone_background_rgb = alpha_over(
                    zone_background_rgb,
                    reflection_rgb[contact_zone],
                    reflection_alpha[contact_zone],
                )
        zone_background = _luminance(zone_background_rgb)
        zone_contact_alpha = contact_alpha[contact_zone] / 255.0
        expected_rgb = _linear_to_srgb_u8(
            _srgb_u8_to_linear(zone_background_rgb)
            * (1.0 - zone_contact_alpha[..., None])
        )
        expected_darkening = np.maximum(
            _luminance(zone_background_rgb) - _luminance(expected_rgb),
            0.0,
        )
        if has_receiver_counterfactual:
            zone_final_rgb = final[contact_zone]
            observed_darkening = np.maximum(
                zone_background - _luminance(zone_final_rgb), 0.0,
            )
            observed_delta = np.max(
                np.abs(
                    zone_final_rgb.astype(np.int16)
                    - zone_background_rgb.astype(np.int16)
                ),
                axis=1,
            ).astype(np.float32)
        else:
            observed_darkening = actual_darkening[contact_zone]
            observed_delta = actual_rgb_delta[contact_zone]
        receiver_luma_p50 = _percentile(zone_background, 50.0)
        expected_p75 = _percentile(expected_darkening, 75.0)
        observed_dark_p75 = _percentile(observed_darkening, 75.0)
        observed_delta_p75 = _percentile(observed_delta, 75.0)
        visible_fraction = float(np.mean(
            (observed_delta >= 2.0) & (observed_darkening >= 0.75)
        ))
        realization = float(np.clip(
            observed_dark_p75 / max(0.5, expected_p75), 0.0, 1.5,
        ))
        contact.update({
            "receiver_luminance_p50": round(receiver_luma_p50, 4),
            "expected_darkening_p75": round(expected_p75, 4),
            "observed_darkening_p75": round(observed_dark_p75, 4),
            "observed_rgb_delta_p75": round(observed_delta_p75, 4),
            "visible_pixel_fraction": round(visible_fraction, 6),
            "realization_ratio": round(realization, 6),
        })
        # Below this luminance a black occlusion layer has essentially no
        # display-space headroom.  Treating it as a renderer defect would
        # encourage ever darker, dirtier halos without improving grounding.
        if receiver_luma_p50 < 24.0 and (
            observed_dark_p75 < 0.9 or visible_fraction < 0.18
        ):
            contact.update({
                "status": "dark_receiver_limited",
                # Preserve a passing structural score without claiming that
                # an unobservable effect was perfectly verified.
                "score": 97.0,
                "reason": "receiver_has_insufficient_darkening_headroom_do_not_add_black",
            })
            warnings.append("contact_visibility_limited_by_dark_receiver_no_forced_darkening")
        elif observed_dark_p75 < max(0.9, min(2.5, expected_p75 * 0.22)) or visible_fraction < 0.18:
            score = float(np.clip(60.0 * realization + 40.0 * visible_fraction, 0.0, 88.0))
            contact.update({
                "status": "insufficient_visible_contact",
                "score": round(score, 4),
                "reason": "contact_layer_does_not_create_a_visible_receiver_side_change_in_final",
            })
            failures.append("contact_final_pixel_visibility_insufficient")
        else:
            score = float(np.clip(65.0 + 25.0 * min(1.0, realization) + 10.0 * visible_fraction, 0.0, 100.0))
            contact.update({"status": "visible", "score": round(score, 4)})

    reflection_alpha = layer.reflection_alpha.astype(np.float32)
    reflection_candidate = (reflection_alpha > 0) & receiver & ~body
    reflection_pre_occlusion = int(np.count_nonzero(reflection_candidate))
    reflection_candidate &= ~all_bodies
    positive_reflection = reflection_alpha[reflection_candidate]
    if positive_reflection.size:
        reflection_threshold = max(2.0, _percentile(positive_reflection, 35.0))
        reflection_zone = reflection_candidate & (reflection_alpha >= reflection_threshold)
    else:
        reflection_zone = reflection_candidate
    reflection_pixels = int(np.count_nonzero(reflection_zone))
    reflection: dict[str, Any] = {
        "zone_pixel_count": reflection_pixels,
        "pre_occlusion_zone_pixel_count": reflection_pre_occlusion,
        "comparison": "final_vs_original_background_in_authorized_reflection_zone",
    }
    if not lighting.reflection_enabled:
        reflection.update({
            "status": "not_expected",
            "score": 100.0,
            "reason": "scene_analysis_disabled_tabletop_reflection",
        })
    elif reflection_pre_occlusion and reflection_pixels < max(4, int(reflection_pre_occlusion * 0.08)):
        reflection.update({
            "status": "occluded_not_measurable",
            "score": None,
            "reason": "another_product_covers_the_reflection_zone",
        })
        warnings.append("reflection_visibility_occluded_not_measurable")
    elif not reflection_pixels:
        reflection.update({
            "status": "expected_but_missing_layer",
            "score": 0.0,
            "reason": "reflection_is_enabled_but_no_receiver_side_reflection_pixels_exist",
        })
        failures.append("reflection_final_pixel_signal_missing")
    else:
        factor = reflection_alpha[reflection_zone, None] / 255.0
        expected_rgb = (
            layer.reflection_rgb[reflection_zone].astype(np.float32) * factor
            + background[reflection_zone].astype(np.float32) * (1.0 - factor)
        )
        opportunity = np.max(
            np.abs(expected_rgb - background[reflection_zone].astype(np.float32)), axis=1,
        )
        observed = actual_rgb_delta[reflection_zone]
        opportunity_p75 = _percentile(opportunity, 75.0)
        observed_p75 = _percentile(observed, 75.0)
        visible_fraction = float(np.mean(observed >= 2.0))
        realization = float(np.clip(
            observed_p75 / max(0.5, opportunity_p75), 0.0, 1.5,
        ))
        reflection.update({
            "expected_rgb_delta_p75": round(opportunity_p75, 4),
            "observed_rgb_delta_p75": round(observed_p75, 4),
            "visible_pixel_fraction": round(visible_fraction, 6),
            "realization_ratio": round(realization, 6),
        })
        if opportunity_p75 < 1.25:
            reflection.update({
                "status": "low_contrast_physically_subtle",
                "score": 97.0,
                "reason": "reflection_colour_matches_receiver_so_strong_visibility_is_not_expected",
            })
            warnings.append("reflection_low_contrast_physically_subtle_no_forced_opacity")
        elif observed_p75 < max(1.0, min(3.0, opportunity_p75 * 0.20)) or visible_fraction < 0.15:
            score = float(np.clip(60.0 * realization + 40.0 * visible_fraction, 0.0, 88.0))
            reflection.update({
                "status": "insufficient_visible_reflection",
                "score": round(score, 4),
                "reason": "authorized_reflection_layer_does_not_survive_into_final_pixels",
            })
            failures.append("reflection_final_pixel_visibility_insufficient")
        else:
            score = float(np.clip(65.0 + 25.0 * min(1.0, realization) + 10.0 * visible_fraction, 0.0, 100.0))
            reflection.update({"status": "visible", "score": round(score, 4)})

    return {
        "measured": True,
        "policy": "measure_final_change_only_where_physical_effect_has_receiver_contrast_opportunity",
        "contact": contact,
        "reflection": reflection,
        "warnings": warnings,
        "failures": failures,
    }


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / max(1.0, float(denominator))


def _score_from_cosine(value: float) -> float:
    # Nearest-silhouette correspondence is intentionally conservative and is
    # laterally biased by a broad penumbra.  Cosine 0.70 still proves the same
    # declared quadrant when essentially all shadow energy is forward; 0.90
    # or above is indistinguishable from the declared image-space ray.
    return float(np.clip(80.0 + 20.0 * (value - 0.70) / 0.20, 0.0, 100.0))


def _product_evidence(
    product_id: str,
    layer: RenderedProduct,
    support: np.ndarray,
    protected: np.ndarray,
    lighting: Lighting,
    background: np.ndarray | None = None,
    final: np.ndarray | None = None,
    all_bodies: np.ndarray | None = None,
    pixel_context: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    alpha = layer.alpha
    body = alpha >= 32
    occupied = np.argwhere(body)
    if occupied.size == 0:
        return {
            "product_id": product_id,
            "valid": False,
            "grounding_score": 0.0,
            "cast_shadow_score": 0.0,
            "geometry_score": 0.0,
            "reason": "empty_product_alpha",
        }

    y0, x0 = occupied.min(axis=0)
    y1, x1 = occupied.max(axis=0)
    object_height = max(1, int(y1 - y0 + 1))
    footprint = contact_profile(alpha) > 0
    footprint_count = int(np.count_nonzero(footprint))
    receiver = support & ~protected
    if all_bodies is None:
        all_bodies = body
    supported_count = int(np.count_nonzero(footprint & receiver))
    support_ratio = _ratio(supported_count, footprint_count)
    protected_ratio = _ratio(np.count_nonzero(body & protected), np.count_nonzero(body))

    combined_shadow = np.maximum(layer.contact_shadow, layer.cast_shadow)
    visible_shadow = combined_shadow >= max(2, int(round(float(combined_shadow.max()) * 0.025)))
    # Check each real footprint column for a shadow sample at or immediately
    # below its measured lower contour.  This directly detects the bright slit
    # that makes a deterministic composite look suspended.
    contact_columns = np.flatnonzero(footprint.any(axis=0))
    attached_columns = 0
    gaps: list[float] = []
    if visible_shadow.any():
        distance_to_shadow = cv2.distanceTransform(
            (~visible_shadow).astype(np.uint8), cv2.DIST_L2, 3,
        )
    else:
        distance_to_shadow = np.full(alpha.shape, 1e6, np.float32)
    for x in contact_columns:
        ys = np.flatnonzero(footprint[:, x])
        if not len(ys):
            continue
        bottom = int(ys[-1])
        xa, xb = max(0, int(x) - 2), min(alpha.shape[1], int(x) + 3)
        ya, yb = max(0, bottom - 1), min(alpha.shape[0], bottom + 7)
        if np.any(visible_shadow[ya:yb, xa:xb]):
            attached_columns += 1
        gaps.append(float(distance_to_shadow[bottom, x]))
    attachment_ratio = _ratio(attached_columns, len(contact_columns))
    gap_p95 = float(np.percentile(gaps, 95)) if gaps else 1e6
    grounding_score = 82.0 + 14.0 * attachment_ratio + 4.0 * support_ratio
    if gap_p95 > 2.5:
        grounding_score -= min(35.0, (gap_p95 - 2.5) * 8.0)
    grounding_score = float(np.clip(grounding_score, 0.0, 100.0))

    final_visibility = _final_pixel_visibility(
        layer,
        footprint,
        body,
        all_bodies,
        receiver,
        object_height,
        lighting,
        background,
        final,
        pixel_context,
    )
    contact_visibility_score = final_visibility["contact"].get("score")
    if contact_visibility_score is not None:
        # Structural attachment is an upper bound.  A mathematically present
        # layer cannot receive 100 when the delivered receiver-side pixels do
        # not visibly differ from the immutable background.
        grounding_score = min(grounding_score, float(contact_visibility_score))

    cast = layer.cast_shadow.astype(np.float32)
    cast_threshold = max(2.0, float(cast.max()) * 0.04)
    cast_visible = (cast >= cast_threshold) & ~body
    direction = np.asarray(lighting.direction, np.float64)
    direction /= max(1e-9, float(np.linalg.norm(direction)))
    ys, xs = np.nonzero(cast_visible)
    direction_cosine = -1.0
    forward_energy_ratio = 0.0
    reach_height_ratio = 0.0
    cast_score = 0.0
    if lighting.cast_shadow_mode != "directional":
        # Ambient-only and disabled are physically valid scene intents.  Audit
        # that the renderer respected the absence of a long projection rather
        # than incorrectly requiring every receiver to show one.
        stray_peak = float(cast[~body].max()) if np.any(~body) else 0.0
        stray_share = _ratio(np.count_nonzero(cast_visible), np.count_nonzero(~body))
        cast_score = float(np.clip(
            100.0 - max(0.0, stray_peak - 2.0) * 2.0 - stray_share * 2000.0,
            0.0,
            100.0,
        ))
    elif len(xs):
        weights = cast[ys, xs].astype(np.float64)
        # Measure every visible tail sample relative to its nearest real
        # silhouette pixel.  A footprint-centroid origin is invalid for an
        # asymmetric product (radio, clock, long handle): the product's own
        # body mass can sit left of its foot and falsely reverse the measured
        # vector even though every projected ray goes down-right.
        distance_from_body, labels = cv2.distanceTransformWithLabels(
            (~body).astype(np.uint8),
            cv2.DIST_L2,
            5,
            labelType=cv2.DIST_LABEL_PIXEL,
        )
        max_label = int(labels.max())
        nearest_x = np.zeros(max_label + 1, np.float64)
        nearest_y = np.zeros(max_label + 1, np.float64)
        body_y, body_x = np.nonzero(body)
        body_labels = labels[body_y, body_x]
        nearest_x[body_labels] = body_x
        nearest_y[body_labels] = body_y
        sample_labels = labels[ys, xs]
        vx = xs.astype(np.float64) - nearest_x[sample_labels]
        vy = ys.astype(np.float64) - nearest_y[sample_labels]
        # Symmetric penumbra within a few pixels of the silhouette contains no
        # reliable directional information.  The farther half of the visible
        # tail does, so use it for the vector audit while retaining all energy
        # for the attachment audit above.
        sample_distance = distance_from_body[ys, xs]
        reliable = sample_distance >= max(2.0, float(np.percentile(sample_distance, 50)))
        if np.any(reliable):
            vx = vx[reliable]
            vy = vy[reliable]
            weights = weights[reliable]
        projection = vx * direction[0] + vy * direction[1]
        forward_energy_ratio = float(weights[projection >= -2.0].sum() / max(1.0, weights.sum()))
        centroid = np.asarray([
            float(np.average(vx, weights=weights)),
            float(np.average(vy, weights=weights)),
        ])
        centroid_norm = float(np.linalg.norm(centroid))
        if centroid_norm > 1e-6:
            direction_cosine = float(np.dot(centroid / centroid_norm, direction))
        positive = projection[projection > 0]
        if len(positive):
            reach_height_ratio = float(np.percentile(positive, 97) / object_height)
        direction_score = _score_from_cosine(direction_cosine)
        forward_score = float(np.clip((forward_energy_ratio - 0.72) / 0.25 * 100.0, 0.0, 100.0))
        # The projector is height proportional.  A usable visible tail should
        # cover at least 45% of the declared reach after blur/support clipping.
        declared_reach = max(1e-6, lighting.shadow_length_ratio / 0.16)
        reach_score = float(np.clip(reach_height_ratio / (declared_reach * 0.45) * 100.0, 0.0, 100.0))
        cast_score = 0.35 * direction_score + 0.45 * forward_score + 0.20 * reach_score

    geometry_score = 100.0 * support_ratio
    geometry_score -= min(100.0, protected_ratio * 1000.0)
    geometry_score = float(np.clip(geometry_score, 0.0, 100.0))
    alpha_occupied = alpha > 0
    soft_ratio = _ratio(
        np.count_nonzero((alpha > 0) & (alpha < 250)),
        np.count_nonzero(alpha_occupied),
    )
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (alpha >= 32).astype(np.uint8), 8,
    )
    areas = stats[1:, cv2.CC_STAT_AREA] if component_count > 1 else np.asarray([], np.int32)
    largest_share = _ratio(int(areas.max()) if len(areas) else 0, int(areas.sum()))
    touches_canvas = bool(
        np.any(alpha_occupied[0]) or np.any(alpha_occupied[-1])
        or np.any(alpha_occupied[:, 0]) or np.any(alpha_occupied[:, -1])
    )
    # This score is deliberately structural, not aesthetic: every visible RGB
    # sample still comes from the source RGBA, and the renderer only performs
    # deterministic premultiplied resampling/low-frequency colour adaptation.
    edge_score = 100.0
    edge_score -= max(0.0, 0.995 - largest_share) * 400.0
    edge_score -= max(0.0, soft_ratio - 0.08) * 180.0
    if touches_canvas:
        edge_score -= 35.0
    edge_score = float(np.clip(edge_score, 0.0, 100.0))
    return {
        "product_id": product_id,
        "valid": True,
        "object_bbox": [int(x0), int(y0), int(x1), int(y1)],
        "object_height_px": object_height,
        "footprint_pixel_count": footprint_count,
        "footprint_support_ratio": round(support_ratio, 6),
        "protected_body_overlap_ratio": round(protected_ratio, 8),
        "contact_attachment_ratio": round(attachment_ratio, 6),
        "contact_gap_p95_px": round(gap_p95, 4),
        "shadow_direction_cosine": round(direction_cosine, 6),
        "shadow_forward_energy_ratio": round(forward_energy_ratio, 6),
        "shadow_reach_height_ratio": round(reach_height_ratio, 6),
        "grounding_score": round(grounding_score, 4),
        "cast_shadow_score": round(float(np.clip(cast_score, 0.0, 100.0)), 4),
        "geometry_score": round(geometry_score, 4),
        "alpha_largest_component_share": round(largest_share, 8),
        "alpha_soft_pixel_ratio": round(soft_ratio, 8),
        "alpha_touches_canvas": touches_canvas,
        "edge_identity_policy": (
            "source_alpha_and_high_frequency_locked_"
            "bounded_low_frequency_effects_only"
        ),
        "cutout_edge_score": round(edge_score, 4),
        "final_pixel_visibility": final_visibility,
    }


def audit_physical_composition(
    rendered: Iterable[tuple[Any, RenderedProduct]],
    support: np.ndarray,
    protected: np.ndarray,
    lighting: Lighting,
    background: np.ndarray | None = None,
    final: np.ndarray | None = None,
) -> dict[str, Any]:
    """Return reproducible receiver-space and delivered-pixel evidence.

    ``background`` and ``final`` are optional for source compatibility with
    callers that only need the original structural measurements.  Production
    composition passes both arrays after hard background restoration.
    """

    pairs = list(rendered)
    alphas = [layer.alpha >= 32 for _, layer in pairs]
    all_bodies = np.logical_or.reduce(alphas) if alphas else np.zeros_like(support, bool)
    pixel_context: dict[str, Any] | None = None
    if background is not None and final is not None:
        background_luma = _luminance(background)
        final_luma = _luminance(final)
        cast_union = np.zeros(support.shape, np.uint8)
        contact_reflections: list[tuple[np.ndarray, np.ndarray]] = []
        for _product, layer in pairs:
            cast_union = np.maximum(cast_union, layer.cast_shadow)
            if np.any(layer.reflection_alpha):
                contact_reflections.append(
                    (layer.reflection_rgb, layer.reflection_alpha),
                )
        pixel_context = {
            "background_luma": background_luma,
            "actual_rgb_delta": np.max(
                np.abs(final.astype(np.int16) - background.astype(np.int16)), axis=2,
            ).astype(np.float32),
            "actual_darkening": np.maximum(background_luma - final_luma, 0.0),
            "contact_cast_union": cast_union,
            "contact_reflections": contact_reflections,
        }
    products = [
        _product_evidence(
            product.product_id,
            layer,
            support,
            protected,
            lighting,
            background,
            final,
            all_bodies,
            pixel_context,
        )
        for product, layer in pairs
    ]
    overlap_pixels = 0
    body_pixels = sum(int(np.count_nonzero(alpha)) for alpha in alphas)
    for index, first in enumerate(alphas):
        for second in alphas[index + 1:]:
            overlap_pixels += int(np.count_nonzero(first & second))
    overlap_ratio = _ratio(overlap_pixels, body_pixels)

    grounding = min((float(item["grounding_score"]) for item in products), default=0.0)
    cast_shadows = min((float(item["cast_shadow_score"]) for item in products), default=0.0)
    geometry = min((float(item["geometry_score"]) for item in products), default=0.0)
    cutout_edges = min((float(item["cutout_edge_score"]) for item in products), default=0.0)
    geometry = float(np.clip(geometry - min(100.0, overlap_ratio * 800.0), 0.0, 100.0))
    subscores = {
        "grounding": round(grounding, 4),
        "cast_shadows": round(cast_shadows, 4),
        "geometry_layout": round(geometry, 4),
        "cutout_edges": round(cutout_edges, 4),
    }
    failures = [
        f"{name}_below_{int(PHYSICAL_SCORE_FLOORS[name])}"
        for name, score in subscores.items()
        if score < PHYSICAL_SCORE_FLOORS[name]
    ]
    visibility_warnings = sorted({
        warning
        for item in products
        for warning in item.get("final_pixel_visibility", {}).get("warnings", [])
    })
    visibility_failures = sorted({
        failure
        for item in products
        for failure in item.get("final_pixel_visibility", {}).get("failures", [])
    })
    failures.extend(visibility_failures)
    return {
        "schema_version": PHYSICAL_AUDIT_SCHEMA_VERSION,
        "verifier": "deterministic_receiver_and_final_pixel_measurements",
        "declared_shadow_direction": [round(float(v), 8) for v in lighting.direction],
        "declared_shadow_length_ratio": float(lighting.shadow_length_ratio),
        "declared_cast_shadow_mode": lighting.cast_shadow_mode,
        "subscores": subscores,
        "dimension_floors": dict(PHYSICAL_SCORE_FLOORS),
        "product_overlap_ratio": round(overlap_ratio, 8),
        "products": products,
        "final_pixel_visibility": {
            "measured": background is not None and final is not None,
            "warnings": visibility_warnings,
            "failures": visibility_failures,
        },
        "failures": sorted(set(failures)),
        "verdict": "pass" if not failures else "reject",
    }
