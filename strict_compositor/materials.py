from __future__ import annotations

"""Deterministic target-aware rendering for transmissive products.

An RGBA matte describes geometric coverage; it does not describe the optics of
clear glass.  A photographed clear object contains the source environment in
its RGB (roughly ``S = T * B_source + R``).  Copying ``S`` into a new scene
therefore copies the old room as a black/bright blob.  This module keeps the
measured coverage and high-frequency glass identity while replacing only that
ambiguous low-frequency environment term with pixels from the target scene.

No generative pixels are introduced.  Output pixels are deterministic
combinations of the source cutout, the target background, and fixed filters.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from .material_profiles import (
    TRANSMISSIVE_MATERIAL_MODES,
    canonical_material_mode,
    material_profile,
)
from .optical_matte import (
    OpticalMatte,
    estimate_clear_glass_matte,
    resolve_optical_matte_roi,
)


TRANSMISSIVE_MODES = set(TRANSMISSIVE_MATERIAL_MODES)


@dataclass(frozen=True)
class OpticalReconstruction:
    rgb: np.ndarray
    coverage_alpha: np.ndarray
    shadow_occlusion_alpha: np.ndarray
    reflection_detail_alpha: np.ndarray
    source_environment_replacement_ratio: float
    optical_matte: OpticalMatte | None = None


def _placed_identity_alpha(
    mask: np.ndarray | None,
    alpha: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Validate a canvas-aligned hard identity mask.

    Identity evidence is membership, while ``OpticalMatte`` stores absolute
    opaque coverage.  Weighting membership by the measured product alpha keeps
    antialiased geometry authoritative and makes the resulting mask safe to
    pass directly to ``estimate_clear_glass_matte``.
    """
    if mask is None:
        return None, np.zeros_like(alpha, dtype=bool)
    value = np.asarray(mask)
    if value.shape != alpha.shape:
        raise ValueError("placed_identity_mask must match alpha")
    if not (
        value.dtype == np.bool_
        or np.issubdtype(value.dtype, np.integer)
        or np.issubdtype(value.dtype, np.floating)
    ):
        raise TypeError("placed_identity_mask must be a numeric array")
    if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
        raise ValueError("placed_identity_mask must contain only finite values")
    active = (value > 0) & (alpha > 0)
    absolute = np.where(active, alpha, 0).astype(np.uint8)
    return absolute, active


def _masked_lowpass(
    rgb: np.ndarray,
    alpha: np.ndarray,
    sigma: float,
) -> np.ndarray:
    weight = alpha.astype(np.float32) / 255.0
    blurred_weight = cv2.GaussianBlur(
        weight, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE,
    )
    channels = [
        cv2.GaussianBlur(
            rgb[..., channel].astype(np.float32) * weight,
            (0, 0),
            sigma,
            borderType=cv2.BORDER_REPLICATE,
        )
        for channel in range(3)
    ]
    return np.stack(channels, axis=-1) / np.maximum(
        blurred_weight[..., None], 1.0 / 255.0,
    )


def _masked_horizontal_lowpass(
    rgb: np.ndarray,
    alpha: np.ndarray,
    sigma_x: float,
) -> np.ndarray:
    """Alpha-normalized one-dimensional low pass along image X.

    A photographed clear vessel often contains a broad horizontal shelf or
    floor from the *source* room.  That structure is nearly constant across
    each row, so a horizontal-only filter models it while leaving the changing
    X signal of fluting/ribs available as an identity residual.  A normal
    two-dimensional blur cannot make that distinction: it either retains the
    shelf as a dark band or erases the ribs with it.

    The alpha normalization is important at a glass rim/handle.  Sampling the
    transparent source canvas there would otherwise manufacture a dark halo.
    """
    sigma_x = max(0.55, float(sigma_x))
    radius = max(1, int(np.ceil(sigma_x * 3.0)))
    kernel_x = cv2.getGaussianKernel(radius * 2 + 1, sigma_x).astype(np.float32)
    kernel_y = np.asarray([1.0], np.float32)
    weight = alpha.astype(np.float32) / 255.0
    blurred_weight = cv2.sepFilter2D(
        weight,
        cv2.CV_32F,
        kernel_x,
        kernel_y,
        borderType=cv2.BORDER_REPLICATE,
    )
    channels = [
        cv2.sepFilter2D(
            rgb[..., channel].astype(np.float32) * weight,
            cv2.CV_32F,
            kernel_x,
            kernel_y,
            borderType=cv2.BORDER_REPLICATE,
        )
        for channel in range(3)
    ]
    return np.stack(channels, axis=-1) / np.maximum(
        blurred_weight[..., None], 1.0 / 255.0,
    )


def _object_extent(alpha: np.ndarray) -> int:
    occupied = np.argwhere(alpha > 4)
    if not occupied.size:
        return 1
    y0, x0 = occupied.min(axis=0)
    y1, x1 = occupied.max(axis=0)
    return max(1, int(max(y1 - y0 + 1, x1 - x0 + 1)))


def _clear_glass_refract_target(
    target_low: np.ndarray,
    alpha: np.ndarray,
    boundary_keep: np.ndarray,
    rib_luma: np.ndarray | None = None,
) -> np.ndarray:
    """Apply deterministic cylindrical and rib-aware refraction to target light.

    A clear ribbed vessel should not be an alpha-shaped copy of the wall.  At
    the same time, its photographed source background cannot be reused as a
    fake refraction cue.  This uses only the target image: a bounded lateral
    displacement is strongest inside the vessel, falls to zero at the measured
    contour, and is only a few full-resolution pixels.  It makes a tabletop
    edge seen through glass bend subtly without inventing texture or altering
    any pixel outside the product coverage.
    """
    occupied = np.argwhere(alpha > 4)
    if not occupied.size:
        return target_low.copy()
    y0, x0 = occupied.min(axis=0)
    y1, x1 = occupied.max(axis=0)
    # Handles and spouts are part of the silhouette but not part of the
    # cylindrical optical body.  Using the complete alpha bounding box shifts
    # the lens axis toward a large handle and bends the target wall as if the
    # handle were solid glass.  Select the longest run of columns that carries
    # most of the object's vertical extent; that run is the measured vessel
    # body.  Detached/suspended appendages remain visible through the separate
    # Fresnel/contour residual, but do not steer the body refraction.
    coverage = (alpha > 4).astype(np.float32)
    column_height = coverage.sum(axis=0)
    maximum_column_height = float(column_height.max())
    body_columns = column_height >= max(10.0, maximum_column_height * 0.62)
    runs: list[tuple[int, int]] = []
    run_start: int | None = None
    for column, is_body in enumerate(body_columns.tolist() + [False]):
        if is_body and run_start is None:
            run_start = column
        elif not is_body and run_start is not None:
            runs.append((run_start, column - 1))
            run_start = None
    if runs:
        body_x0, body_x1 = max(runs, key=lambda run: run[1] - run[0])
        if body_x1 - body_x0 < max(6, int((x1 - x0 + 1) * 0.22)):
            body_x0, body_x1 = int(x0), int(x1)
    else:
        body_x0, body_x1 = int(x0), int(x1)
    half_width = max(2.0, float(body_x1 - body_x0 + 1) * 0.5)
    centre_x = (float(body_x0) + float(body_x1)) * 0.5
    # Cap the apparent bend so a narrow tumbler never becomes a fun-house lens.
    amplitude = float(np.clip(half_width * 0.052, 1.4, 8.0))
    yy, xx = np.indices(alpha.shape, dtype=np.float32)
    normalized_x = np.clip((xx - centre_x) / half_width, -1.0, 1.0)
    # 2.6 makes max(abs(u * (1-u^2))) approximately one.  The target sample is
    # displaced inward/outward symmetrically, which preserves the optical axis.
    cylindrical = -2.6 * normalized_x * (1.0 - normalized_x * normalized_x)
    interior = np.clip(1.0 - boundary_keep, 0.0, 1.0)
    body_domain = (
        (xx >= float(body_x0)) & (xx <= float(body_x1))
    ).astype(np.float32)
    interior *= body_domain
    displacement_x = amplitude * cylindrical * interior

    # A fluted vessel is not a single smooth cylindrical lens.  The measured
    # bright/dark vertical rib phases provide geometric evidence for a small
    # local lateral distortion.  Crucially, ``rib_luma`` is formed after a
    # *row-wise horizontal* source low-pass, so a source-room shelf/table that
    # is broad across X contributes approximately zero.  We therefore use the
    # original object's rib geometry to bend only target pixels, rather than
    # copying the source room into the result.  This keeps the transparent body
    # visible against a quiet wall without inventing product texture.
    rib_phase_map: np.ndarray | None = None
    if rib_luma is not None and rib_luma.shape == alpha.shape:
        column_weight = coverage.sum(axis=0)
        valid_columns = (column_weight > 8.0) & body_columns
        if np.any(valid_columns):
            column_signal = (
                (rib_luma.astype(np.float32) * coverage).sum(axis=0)
                / np.maximum(column_weight, 1.0)
            )
            # A light one-dimensional smoothing removes sensor noise while
            # preserving the alternating phases of photographed vertical ribs.
            smooth_sigma = max(0.70, min(1.45, half_width * 0.006))
            column_signal = cv2.GaussianBlur(
                column_signal.reshape(1, -1),
                (0, 0),
                smooth_sigma,
                borderType=cv2.BORDER_REPLICATE,
            )[0]
            absolute_signal = np.abs(column_signal[valid_columns])
            active_signal = absolute_signal[absolute_signal > 0.75]
            # A ribbed vessel may have only a handful of strong columns.  A
            # global percentile would report zero in that case and silently
            # discard precisely the useful geometry.  Isolate non-noise columns
            # first; a flat or noisy source still fails the scale threshold.
            signal_scale = float(np.percentile(active_signal, 65)) if active_signal.size else 0.0
            if signal_scale > 1.25:
                phase = np.tanh(column_signal / max(4.0, signal_scale * 1.35))
                # Keep this cue materially subtle: at final resolution it is a
                # 0.5--2.8px local bend, added to the already bounded broad lens.
                rib_amplitude = float(np.clip(half_width * 0.018, 0.50, 2.80))
                # The outer body band is already represented by the measured
                # Fresnel contour.  Fading rib phase there prevents a broad
                # coloured source matte from being misread as the first flute
                # while leaving the central measured ribs untouched.
                rib_body_fade = np.clip(
                    (1.0 - np.abs(normalized_x)) / 0.22, 0.0, 1.0,
                ) ** 1.5
                rib_phase_map = (
                    phase.reshape(1, -1) * interior * rib_body_fade
                )
                displacement_x += rib_amplitude * rib_phase_map
    max_displacement = amplitude + 2.90
    displacement_x = np.clip(
        displacement_x, -max_displacement, max_displacement,
    )
    map_x = xx + displacement_x
    warped = cv2.remap(
        target_low.astype(np.float32),
        map_x,
        yy,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    if rib_phase_map is not None:
        # In a bright, nearly uniform wall region, pure transmission makes a
        # *ribbed* clear vessel look like a ghost.  Its measured fluting is
        # also evidence for changing surface normals, which in turn reflect a
        # small amount of the current scene, especially the receiver below.
        # Sample only target pixels along a bounded vertical environment path:
        # this is a deterministic Fresnel term, never old source-room RGB.
        # Work only in the measured alpha bounding rectangle.  A 4K canvas
        # can be tens of megapixels while the vessel itself is small; creating
        # a second full-frame RGB environment map here used needless gigabytes
        # and slowed a deterministic composite without changing a pixel
        # outside this box.
        roi = np.s_[int(y0):int(y1) + 1, int(x0):int(x1) + 1]
        roi_y = yy[roi]
        roi_x = xx[roi]
        environment_y = np.clip(
            roi_y + np.maximum(0.0, float(y1) - roi_y) * 0.55,
            0.0,
            float(target_low.shape[0] - 1),
        )
        environment = cv2.remap(
            target_low.astype(np.float32),
            roi_x,
            environment_y.astype(np.float32),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        optical_angle = boundary_keep[roi] * 0.070
        optical_angle += np.abs(rib_phase_map[roi]) * 0.045
        # The central face of a clear vessel has only a small normal-incidence
        # Fresnel contribution.  The term is bounded so it cannot turn a glass
        # body into an invented smoked tint.
        optical_angle += interior[roi] * 0.012
        optical_angle = np.clip(optical_angle, 0.0, 0.105).astype(np.float32)
        # Do not add a synthetic darkening term when this ray sees effectively
        # the same target radiance.  The reflected component is meaningful only
        # where the current receiver/environment differs from transmission.
        environment_contrast = np.mean(
            np.abs(environment - warped[roi]), axis=2,
        )
        optical_angle *= np.clip((environment_contrast - 10.0) / 24.0, 0.0, 1.0)
        warped[roi] = (
            warped[roi] * (1.0 - optical_angle[..., None])
            + environment * optical_angle[..., None]
        )
    if rib_phase_map is not None:
        # Refraction is visible on a quiet wall not only because a ray moves,
        # but because a fluted dielectric locally concentrates / disperses the
        # transmitted scene radiance.  A pure displacement is invisible where
        # the target is nearly uniform, which made the pitcher turn into a
        # ghost at full-frame scale.  This is a small, achromatic, zero-mean
        # transmission modulation driven by the *measured rib phase* and the
        # target RGB alone: no dark source-room colour is reintroduced.  The
        # strength is intentionally within the few-percent dielectric Fresnel
        # regime and fades to zero at the contour.
        positive_phase = np.maximum(rib_phase_map, 0.0)
        transmission_gain = np.clip(
            1.0 + 0.036 * rib_phase_map + 0.008 * positive_phase,
            0.91,
            1.07,
        )
        warped = np.clip(
            warped * transmission_gain[..., None], 0.0, 255.0,
        )
    result = target_low.copy()
    result[alpha > 4] = warped[alpha > 4]
    return result


def _material_defaults(material_mode: str) -> tuple[float, float, float]:
    # default transmission, target blur fraction, high-frequency retention
    return {
        # A clear fluted vessel should bend the target scene rather than blur
        # it into an alpha-shaped haze.  Keep nearly all high-frequency detail
        # and use the measured ribs as the controlled optical field above.
        "clear_glass": (0.88, 0.0016, 0.94),
        "frosted_glass": (0.72, 0.0150, 0.56),
        "colored_glass": (0.58, 0.0045, 0.82),
        "mixed_transparent": (0.46, 0.0040, 0.88),
    }.get(material_mode, (0.0, 0.0, 1.0))


def reinforce_transmissive_readability(
    reconstructed_rgb: np.ndarray,
    source_rgb: np.ndarray,
    target_background: np.ndarray,
    alpha: np.ndarray,
    material_mode: str,
    *,
    visibility_strength: float = 1.0,
) -> np.ndarray:
    """Keep transparent product structure legible without copying its old room.

    A physically plausible target-scene transmission can still disappear in a
    browser-sized product photograph.  This pass adds only achromatic,
    high-frequency cues measured from the original silhouette/ribs and a
    target-adaptive Fresnel outline.  Broad source colour and source-room
    scenery cannot enter because the source signal is row-normalised and
    high-pass filtered before use.
    """
    mode = canonical_material_mode(material_mode)
    profile = material_profile(mode)
    if (
        not np.isfinite(visibility_strength)
        or not 0.0 <= float(visibility_strength) <= 1.0
    ):
        raise ValueError("visibility_strength must be finite and in [0, 1]")
    visibility_strength = float(visibility_strength)
    floor = float(profile.glass_visibility_floor) * (
        0.62 + 0.38 * visibility_strength
    )
    if floor <= 0.0 or mode not in TRANSMISSIVE_MODES:
        return reconstructed_rgb.copy()
    occupied = np.argwhere(alpha > 4)
    if occupied.size == 0:
        return reconstructed_rgb.copy()
    y0, x0 = occupied.min(axis=0)
    y1, x1 = occupied.max(axis=0)
    # Work in a tight ROI; products commonly occupy a small part of a 7K frame.
    pad = 3
    y0 = max(0, int(y0) - pad)
    x0 = max(0, int(x0) - pad)
    y1 = min(alpha.shape[0] - 1, int(y1) + pad)
    x1 = min(alpha.shape[1] - 1, int(x1) + pad)
    region = np.s_[y0:y1 + 1, x0:x1 + 1]
    local_alpha = alpha[region]
    active = local_alpha > 4
    solid = (local_alpha >= 24).astype(np.uint8)
    padded = cv2.copyMakeBorder(solid, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
    short_extent = max(1, min(local_alpha.shape))
    edge_width = float(np.clip(short_extent * 0.009, 1.0, 3.2))
    boundary = np.exp(-np.maximum(distance - 0.5, 0.0) / edge_width)
    boundary *= active.astype(np.float32)

    local_source = source_rgb[region]
    local_target = target_background[region]
    local_result = reconstructed_rgb[region].astype(np.float32).copy()
    # Remove the row-wise source environment; repeated vertical flutes remain.
    row_base = _masked_horizontal_lowpass(
        local_source,
        local_alpha,
        sigma_x=float(np.clip(local_alpha.shape[1] * 0.055, 2.0, 24.0)),
    )
    source_luma = local_source.astype(np.float32) @ np.asarray(
        [0.2126, 0.7152, 0.0722], np.float32,
    )
    row_luma = row_base @ np.asarray([0.2126, 0.7152, 0.0722], np.float32)
    signed_structure = source_luma - row_luma
    prefilter_samples = np.abs(signed_structure[active])
    prefilter_scale = (
        float(np.percentile(prefilter_samples, 82.0))
        if prefilter_samples.size else 1.0
    )
    preview_scale = min(
        1.0,
        1800.0 / max(1.0, float(max(alpha.shape))),
    )
    structure_sigma_x = float(np.clip(
        0.55 / max(1.0e-6, preview_scale), 0.65, 2.40,
    ))
    signed_structure = cv2.GaussianBlur(
        signed_structure, (0, 0), sigmaX=structure_sigma_x, sigmaY=1.25,
        borderType=cv2.BORDER_REPLICATE,
    )
    # Widen sub-display-pixel ribs so positive and negative faces do not
    # average away during a 4K/7K-to-browser fit, then restore only the measured
    # robust amplitude.  The 1.35 ceiling prevents this display adaptation
    # from becoming an unbounded contrast enhancement.
    postfilter_samples = np.abs(signed_structure[active])
    postfilter_scale = (
        float(np.percentile(postfilter_samples, 82.0))
        if postfilter_samples.size else prefilter_scale
    )
    structure_rescale = float(np.clip(
        prefilter_scale / max(1.0e-6, postfilter_scale), 1.0, 1.35,
    ))
    signed_structure *= structure_rescale
    structure_scale_samples = np.abs(signed_structure[active])
    structure_scale = (
        float(np.percentile(structure_scale_samples, 82.0))
        if structure_scale_samples.size else 1.0
    )
    structure = np.tanh(signed_structure / max(4.0, structure_scale * 1.15))
    # A thick glass base can contain much stronger high-frequency energy than
    # the quiet upper ribs.  A single whole-object threshold therefore makes
    # a correctly reconstructed pitcher disappear after display downscaling.
    # Estimate a conservative vertical scale from overlapping horizontal
    # bands.  This is still driven only by measured source structure: it never
    # changes alpha, transmission, or authorises broad source-room colour.
    local_height = local_alpha.shape[0]
    band_height = max(8, int(round(local_height * 0.14)))
    band_step = max(4, band_height // 2)
    band_centres = list(range(0, local_height, band_step))
    if not band_centres or band_centres[-1] != local_height - 1:
        band_centres.append(local_height - 1)
    band_scales: list[float] = []
    half_band = max(1, band_height // 2)
    abs_structure = np.abs(signed_structure)
    for centre in band_centres:
        band_y0 = max(0, int(centre) - half_band)
        band_y1 = min(local_height, int(centre) + half_band + 1)
        band_active = active[band_y0:band_y1]
        band_samples = abs_structure[band_y0:band_y1][band_active]
        band_scales.append(
            float(np.percentile(band_samples, 75.0))
            if band_samples.size else structure_scale
        )
    local_scale_y = np.interp(
        np.arange(local_height, dtype=np.float32),
        np.asarray(band_centres, dtype=np.float32),
        np.asarray(band_scales, dtype=np.float32),
    )
    local_scale_y = cv2.GaussianBlur(
        local_scale_y[:, None], (1, 0),
        sigmaX=0.0, sigmaY=max(1.0, local_height * 0.018),
        borderType=cv2.BORDER_REPLICATE,
    )[:, 0]
    local_scale_y = np.clip(
        local_scale_y,
        max(4.0, structure_scale * 0.55),
        max(4.0, structure_scale),
    )
    local_structure_scale = local_scale_y[:, None]
    # Prevent a broad residual from becoming a smoked veil.  Only local
    # contrast and measured contour topology are authorised.
    structure_support = np.clip(
        (abs_structure - np.maximum(1.0, local_structure_scale * 0.34))
        / np.maximum(3.0, local_structure_scale * 0.54),
        0.0,
        1.0,
    )
    structure_support *= active.astype(np.float32)

    target_luma = local_target.astype(np.float32) @ np.asarray(
        [0.2126, 0.7152, 0.0722], np.float32,
    )
    result_luma = local_result @ np.asarray([0.2126, 0.7152, 0.0722], np.float32)
    edge_samples = np.abs(result_luma - target_luma)[boundary > 0.38]
    current_edge_contrast = (
        float(np.percentile(edge_samples, 60.0)) if edge_samples.size else 0.0
    )
    # Do not let the catalogue readability floor turn into a broad painted
    # outline.  The contour only needs a thin Fresnel cue; the larger profile
    # floor is reserved for measured flutes and the thick base below.
    edge_floor = min(floor, 28.0) if mode == "clear_glass" else floor
    missing = max(0.0, edge_floor - current_edge_contrast)
    # Darken against a bright receiver and brighten against a dark receiver.
    # This keeps a colourless outline readable without painting it white.
    contrast_sign = np.where(target_luma >= 118.0, -1.0, 1.0)
    edge_delta = contrast_sign * boundary * (missing + edge_floor * 0.42)
    # Catalogue glass needs its alternating groove faces to survive a large
    # canvas being fitted into the workbench.  This is still a signed,
    # zero-mean source high-pass: increasing it cannot restore the old room or
    # create a body-wide tint.
    rib_gain = 1.14 if mode == "clear_glass" else 0.82
    rib_delta = structure * structure_support * (floor * rib_gain)
    if mode == "clear_glass":
        # A several-pixel source matte fringe is not a glass flute.  Reserve
        # the contour band for the target-adaptive Fresnel cue and fade the
        # source high-pass there; genuine interior ribs remain untouched.
        rib_delta *= 1.0 - 0.92 * np.power(boundary, 1.35)
        side_release = np.clip(
            (distance - edge_width * 3.15) / max(1.0, edge_width * 1.85),
            0.0,
            1.0,
        )
        side_release = side_release * side_release * (3.0 - 2.0 * side_release)
        rib_delta *= side_release
    # The base carries most of the visual weight of an empty glass vessel.
    # Strengthen its measured high-frequency structure, not its old-room RGB.
    yy = np.linspace(0.0, 1.0, local_alpha.shape[0], dtype=np.float32)[:, None]
    base_weight = np.clip((yy - 0.77) / 0.23, 0.0, 1.0) ** 1.35
    rib_delta *= 1.0 + base_weight * 0.78
    # A transparent catalogue product still needs a weak front-surface volume
    # cue.  Ribs plus a contour alone read as a flat wire drawing on a quiet
    # wall.  Build a low-amplitude cylindrical modulation from target luminance
    # and alpha geometry: bright surroundings darken slightly through thicker
    # paths, dark surroundings receive a faint grazing return.  No source-room
    # colour or low-frequency source pixels enter this term.
    xx = np.linspace(-1.0, 1.0, local_alpha.shape[1], dtype=np.float32)[None, :]
    cylindrical_path = 0.38 + 0.62 * (1.0 - np.abs(xx)) ** 1.55
    body_support = active.astype(np.float32) * np.power(
        np.clip(1.0 - boundary, 0.0, 1.0), 0.62,
    )
    body_density = (
        floor
        * 0.018
        * cylindrical_path
        * body_support
        * (1.0 + base_weight * 0.72)
    )
    body_sign = np.where(target_luma >= 112.0, -1.0, 1.0)
    body_delta = body_sign * body_density
    if mode == "clear_glass":
        # A symmetric final clamp makes a clear vessel look like pale line art
        # on a bright receiver.  Positive flute faces quickly approach white,
        # while the equally important negative faces were previously limited
        # to -16 code values.  The missing dark half of the alternating groove
        # response is especially obvious after a 4K image is fitted into the
        # browser.
        #
        # Allocate contrast per *measured* optical component instead.  Only
        # ``rib_delta`` receives the wider asymmetric range, and it is already
        # gated by signed source high-pass plus ``structure_support``.  The
        # broad body cue remains tiny and the silhouette stays a thin Fresnel
        # edge, so this cannot turn the alpha shape into a smoked or white
        # plate.  The target luminance chooses which signed face carries the
        # most useful contrast: dark grooves on a bright table, bright glints
        # in a dark room, and a balanced response on a mid-tone receiver.
        bright_receiver = np.clip(
            (target_luma - 132.0) / (220.0 - 132.0), 0.0, 1.0,
        )
        bright_receiver = (
            bright_receiver
            * bright_receiver
            * (3.0 - 2.0 * bright_receiver)
        )
        # Wider dark headroom is authorised only for structure that persists
        # down a meaningful part of the object column (a flute) or belongs to
        # the optically thick base.  A photographed wall/table boundary is a
        # short diagonal event in each column and therefore keeps the legacy
        # -16 bound instead of being promoted into product identity.
        active_per_column = np.maximum(
            1.0, np.sum(active.astype(np.float32), axis=0, keepdims=True),
        )
        persistent_fraction = np.sum(
            (structure_support > 0.20).astype(np.float32),
            axis=0,
            keepdims=True,
        ) / active_per_column
        vertical_persistence = np.clip(
            (persistent_fraction - 0.18) / (0.58 - 0.18), 0.0, 1.0,
        )
        vertical_persistence = (
            vertical_persistence
            * vertical_persistence
            * (3.0 - 2.0 * vertical_persistence)
        )
        thick_base_authority = base_weight * structure_support
        expansion_authority = np.maximum(
            vertical_persistence, thick_base_authority,
        )

        negative_rib_limit = -(
            16.0 + 20.0 * bright_receiver * expansion_authority
        )
        # Bright glints were already adequate and are more prone to producing
        # a white veil.  Keep their historical +14 ceiling on every receiver;
        # this iteration deliberately improves the missing negative phase.
        positive_rib_limit = np.full_like(target_luma, 14.0)
        strong_negative = np.clip(
            (-rib_delta - floor * 0.32) / max(4.0, floor * 0.52),
            0.0,
            1.0,
        )
        rib_delta = np.where(
            rib_delta < 0.0,
            rib_delta * (
                1.0
                + 0.32
                * bright_receiver
                * expansion_authority
                * strong_negative
            ),
            rib_delta,
        )
        rib_delta = np.minimum(
            np.maximum(rib_delta, negative_rib_limit),
            positive_rib_limit,
        )
        edge_delta = np.clip(edge_delta, -20.0, 14.0)
        body_delta = np.clip(body_delta, -4.0, 4.0)

        raw_delta = edge_delta + rib_delta + body_delta
        final_negative_limit = negative_rib_limit - 3.0
        final_positive_limit = positive_rib_limit
        delta = np.minimum(
            np.maximum(raw_delta, final_negative_limit),
            final_positive_limit,
        )
    else:
        raw_delta = edge_delta + rib_delta + body_delta
        delta = np.clip(raw_delta, -15.0, 15.0)
    local_result[active] = np.clip(
        local_result[active] + delta[active, None], 0.0, 255.0,
    )
    output = reconstructed_rgb.copy()
    output[region] = np.clip(np.round(local_result), 0, 255).astype(np.uint8)
    return output


def _reconstruct_verified_clear_glass(
    rgb: np.ndarray,
    alpha: np.ndarray,
    target_background: np.ndarray,
    *,
    transmission_strength: float,
    material_confidence: float,
    commercial_visibility: float,
    material_route_reason: str,
    opaque_identity_mask: np.ndarray | None = None,
) -> OpticalReconstruction:
    """Resolve verified clear glass from an auditable ROI optical matte.

    The matte estimator may use source luminance only to locate persistent
    ribs and the measured silhouette.  The reconstructed clear body samples
    its colour exclusively from ``target_background`` in linear sRGB.  This
    prevents a wall, shelf or dark room photographed through the source glass
    from travelling with the product into the target scene.

    Shadow and reflection masks are derived from the same optical contract:
    clear interiors transmit most incident light, while contours, bases and
    measured optical detail contribute stronger but still non-opaque cues.
    """
    confidence = float(np.clip(material_confidence, 0.0, 1.0))
    requested = float(np.clip(transmission_strength, 0.0, 1.0))
    # Zero means "use the material default", not zero transmission.  The
    # former verified path ignored both values after recording them; pass the
    # effective strength into the optical estimator so material semantics now
    # affect absorption, refraction and receiver-light loss continuously.
    effective_transmission = (
        requested
        if requested > 0.0
        else float(_material_defaults("clear_glass")[0])
    )
    # Route confidence answers "is this clear glass?".  Commercial visibility
    # answers "how strongly should an already verified glass object read in a
    # product image?".  They are deliberately independent: using one number
    # for both previously made a correctly classified 0.70-confidence pitcher
    # fall back to legacy source-RGB compositing and, even when forced through
    # the optical route, made it needlessly disappear at review scale.
    commercial_visibility = float(np.clip(commercial_visibility, 0.0, 1.0))
    matte = estimate_clear_glass_matte(
        rgb,
        alpha,
        opaque_identity_mask=opaque_identity_mask,
        confidence=confidence,
        transmission_strength=effective_transmission,
        commercial_visibility=commercial_visibility,
        provenance={
            "material_mode": "clear_glass",
            "material_confidence": confidence,
            "transmission_strength": requested,
            "effective_transmission_strength": effective_transmission,
            "commercial_visibility": commercial_visibility,
            "material_route": "verified_optical_matte",
            "material_route_reason": material_route_reason,
            "integration": "strict_compositor.materials.v2",
            "identity_mask_supplied": bool(
                opaque_identity_mask is not None
                and np.any(opaque_identity_mask)
            ),
        },
    )
    resolved_roi = resolve_optical_matte_roi(matte, target_background)
    output_rgb = rgb.copy()
    x0, y0, x1, y1 = matte.bbox
    active = matte.coverage_alpha > 0.0
    output_roi = output_rgb[y0:y1, x0:x1]
    output_roi[active] = resolved_roi[active]

    shadow_alpha = np.zeros_like(alpha)
    reflection_alpha = np.zeros_like(alpha)
    shadow_loss = np.clip(1.0 - matte.shadow_transmittance, 0.0, 1.0)
    specular_energy = np.mean(matte.specular_rgb, axis=2)
    flow_energy = np.linalg.norm(matte.refraction_flow, axis=2)
    flow_scale = float(np.percentile(flow_energy[active], 90.0)) if np.any(active) else 1.0
    flow_detail = np.clip(flow_energy / max(0.35, flow_scale), 0.0, 1.0)

    # A clear body must not cast the same opaque silhouette as a label or
    # ceramic product.  Its contour/base remains denser, naturally following
    # ``shadow_transmittance`` from the optical matte rather than a uniform
    # ellipse or arbitrary global opacity.
    shadow_roi = np.round(
        255.0 * np.clip(shadow_loss, 0.0, 0.46),
    ).astype(np.uint8)

    # On a glossy receiver the mirror image contains a faint transmitted body
    # plus stronger Fresnel/rib structure.  The 0.34 floor keeps a real glass
    # outline visible after 4K preview downsampling; optical detail raises it
    # locally without turning the whole object opaque.
    reflection_absolute = (
        0.46 * matte.coverage_alpha
        + 0.62 * shadow_loss
        + 2.20 * specular_energy
        + 0.08 * flow_detail * matte.coverage_alpha
    )
    reflection_absolute = np.minimum(
        np.clip(reflection_absolute, 0.0, 0.90),
        matte.coverage_alpha * 0.90,
    )
    reflection_roi = np.round(255.0 * reflection_absolute).astype(np.uint8)

    shadow_alpha[y0:y1, x0:x1] = shadow_roi
    reflection_alpha[y0:y1, x0:x1] = reflection_roi
    occupied = alpha > 4
    shadow_alpha[~occupied] = 0
    reflection_alpha[~occupied] = 0

    # High-confidence colourless glass replaces the ambiguous source room at
    # every clear covered pixel.  Explicit opaque identity regions, if added
    # by a future material classifier, are excluded from this ratio.
    clear_weight = np.clip(
        matte.coverage_alpha - matte.opaque_identity_mask, 0.0, 1.0,
    )
    coverage_weight = float(np.sum(matte.coverage_alpha))
    replacement_ratio = (
        float(np.sum(clear_weight) / coverage_weight)
        if coverage_weight > 1.0e-6
        else 0.0
    )
    # Optical reconstruction is target-correct, but at product-catalog viewing
    # size a colourless vessel may still lose its rim and fluting.  Apply the
    # same bounded, achromatic readability contract used after geometric warp
    # so direct material consumers cannot receive a technically transparent
    # yet visually vanished product.
    output_rgb = reinforce_transmissive_readability(
        output_rgb,
        rgb,
        target_background,
        alpha,
        "clear_glass",
        visibility_strength=commercial_visibility,
    )
    # Readability enhancement is intentionally allowed to alter glass rims and
    # ribs, but never an explicitly protected label, logo or text region.  The
    # strict baseline here is the already-authorised deterministic product
    # transform, not raw pre-placement pixels.
    if opaque_identity_mask is not None:
        identity = (opaque_identity_mask > 0) & (alpha > 0)
        output_rgb[identity] = rgb[identity]
    return OpticalReconstruction(
        output_rgb,
        alpha.copy(),
        shadow_alpha,
        reflection_alpha,
        replacement_ratio,
        matte,
    )


def reconstruct_transmissive_product(
    rgb: np.ndarray,
    alpha: np.ndarray,
    target_background: np.ndarray,
    material_mode: str,
    transmission_strength: float = 0.0,
    material_confidence: float = 1.0,
    placed_identity_mask: np.ndarray | None = None,
    material_route_verified: bool = False,
    commercial_visibility: float | None = None,
) -> OpticalReconstruction:
    """Replace trapped source scenery while preserving glass identity.

    ``alpha`` is returned byte-for-byte.  For opaque material the entire RGB
    and both optical masks are also unchanged.  For transmissive modes only
    the low-frequency interior is target-aware; rims, ribs, text-sized detail,
    highlights and other high-frequency residuals continue to come from the
    original photographed object.
    """

    material_mode = canonical_material_mode(material_mode)
    if rgb.shape != target_background.shape or rgb.shape[:2] != alpha.shape:
        raise ValueError("rgb, alpha and target_background must share a canvas")
    identity_alpha, identity = _placed_identity_alpha(
        placed_identity_mask, alpha,
    )
    if material_mode not in TRANSMISSIVE_MODES:
        return OpticalReconstruction(
            rgb.copy(), alpha.copy(), alpha.copy(), alpha.copy(), 0.0,
        )
    occupied = alpha > 4
    if not np.any(occupied):
        zeros = np.zeros_like(alpha)
        return OpticalReconstruction(
            rgb.copy(), alpha.copy(), zeros, zeros, 0.0,
        )

    # Verified colourless glass uses the explicit ROI optical contract.  Keep
    # the legacy reconstruction below as a conservative fallback for weak
    # material classifications and deliberately low transmission settings.
    # A semantic/material router may explicitly verify the route; otherwise a
    # 0.65 clear-glass confidence is sufficient when transmission evidence is
    # also strong.  Commercial visibility is a separate output-style control
    # and must never decide which physical material implementation is used.
    confidence = float(np.clip(material_confidence, 0.0, 1.0))
    route_verified = bool(material_route_verified)
    visibility = 1.0 if commercial_visibility is None else float(
        commercial_visibility
    )
    if not np.isfinite(visibility) or not 0.0 <= visibility <= 1.0:
        raise ValueError("commercial_visibility must be finite and in [0, 1]")
    verified_by_confidence = confidence >= 0.65
    verified_transmission = (
        float(transmission_strength) <= 0.0
        or float(transmission_strength) >= 0.60
    )
    if (
        material_mode == "clear_glass"
        and (verified_by_confidence or route_verified)
        and verified_transmission
    ):
        route_reason = (
            "explicit_semantic_verification"
            if route_verified
            else "clear_glass_confidence_and_transmission"
        )
        return _reconstruct_verified_clear_glass(
            rgb,
            alpha,
            target_background,
            transmission_strength=transmission_strength,
            material_confidence=material_confidence,
            commercial_visibility=visibility,
            material_route_reason=route_reason,
            opaque_identity_mask=identity_alpha,
        )

    default_strength, target_blur_fraction, high_retention = _material_defaults(
        material_mode,
    )
    requested = float(transmission_strength)
    if requested <= 0.0:
        requested = default_strength
    confidence = float(np.clip(material_confidence, 0.0, 1.0))
    # A weak semantic classification must not erase source pixels.  Once a
    # neutral clear vessel is classified with high confidence, however, the
    # *source room* is not product identity and confidence attenuation must not
    # bring that room back.  In that case honour the requested transmission
    # directly; uncertain and coloured/mixed materials remain conservative.
    if material_mode == "clear_glass" and confidence >= 0.80:
        # Honour the semantic estimate rather than silently forcing virtually
        # perfect transmission.  Target radiance remains authoritative, while
        # the commercial-readability pass below preserves the measured glass
        # structure at normal viewing size.
        strength = float(np.clip(requested, 0.68, 0.92))
    else:
        strength = float(np.clip(
            requested * (0.30 + 0.70 * confidence), 0.0, 0.96,
        ))

    extent = _object_extent(alpha)
    source_sigma = float(np.clip(extent * 0.025, 2.0, 30.0))
    source = rgb.astype(np.float32)
    source_low = _masked_lowpass(rgb, alpha, source_sigma)
    source_high = source - source_low

    blur_sigma = max(0.35, extent * target_blur_fraction)
    target_low = cv2.GaussianBlur(
        target_background.astype(np.float32),
        (0, 0),
        blur_sigma,
        borderType=cv2.BORDER_REPLICATE,
    )

    binary = (alpha >= 32).astype(np.uint8)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    boundary_keep = np.exp(-distance / max(1.15, extent * 0.0065))
    boundary_keep = np.where(occupied, boundary_keep, 0.0).astype(np.float32)

    luma = source @ np.asarray([0.2126, 0.7152, 0.0722], np.float32)
    detail_energy = cv2.GaussianBlur(
        np.mean(np.abs(source_high), axis=2), (0, 0), 0.85,
    )
    detail_norm = np.clip(detail_energy / 34.0, 0.0, 1.0)

    if material_mode in {"clear_glass", "frosted_glass"}:
        optical_candidate = np.ones_like(luma, np.float32)
        tint = np.ones(3, np.float32)
        # A colourless vessel should be dominated by the actual target scene,
        # not darkened uniformly like tinted glass.  The separate measured rim
        # and rib residual below supplies the visible glass identity.
        # Two air/glass interfaces lose several percent even in colourless
        # glass.  A target-exact 0.985 transmission made the body vanish on a
        # quiet wall after 4K preview downsampling.  This restrained neutral
        # attenuation adds real dielectric mass without importing source hue.
        # Keep the body nearly target-exact.  The optical cues below are
        # already sufficient to make a clear vessel legible; a larger neutral
        # density turns a quiet target wall into a milky alpha-shaped veil.
        attenuation = 0.996 if material_mode == "clear_glass" else 0.93
    else:
        # Smooth, sufficiently luminous regions are plausible transmission.
        # Very dark/high-detail regions are likely labels, caps or other
        # opaque identity-bearing parts and remain original source RGB.
        bright_gate = np.clip((luma - 18.0) / 62.0, 0.0, 1.0)
        smooth_gate = 1.0 - np.clip(detail_energy / 46.0, 0.0, 1.0)
        raw_candidate = np.clip(
            bright_gate * (0.38 + 0.62 * smooth_gate), 0.0, 1.0,
        )
        # Mixed products such as a labelled liquor bottle contain both glass
        # and identity-bearing opaque ink/paper.  A soft non-zero tail here
        # used to wash label typography even though its local detail energy
        # correctly said "opaque".  Require positive smooth-transmission
        # evidence before any target RGB is admitted.
        optical_candidate = np.clip(
            (raw_candidate - 0.42) / 0.58, 0.0, 1.0,
        )
        candidates = occupied & (optical_candidate > 0.52)
        if np.count_nonzero(candidates) >= 32:
            sample = source_low[candidates]
            sample_luma = sample @ np.asarray(
                [0.2126, 0.7152, 0.0722], np.float32,
            )
            chromaticity = np.median(
                sample / np.maximum(sample_luma[:, None], 12.0), axis=0,
            )
            normalizer = float(
                chromaticity @ np.asarray([0.2126, 0.7152, 0.0722])
            )
            tint = np.clip(chromaticity / max(0.25, normalizer), 0.62, 1.48)
        else:
            tint = np.ones(3, np.float32)
        attenuation = 0.90 if material_mode == "colored_glass" else 0.94

    # Build the measured rib field before sampling the target.  It is used only
    # as a displacement field for target pixels; source colours never re-enter
    # through this path.  Keeping the array single-channel is important for a
    # full-resolution 4K+ render.
    rib_luma: np.ndarray | None = None
    if material_mode == "clear_glass":
        horizontal_base = _masked_horizontal_lowpass(
            rgb,
            alpha,
            max(1.25, extent * 0.0055),
        )
        # Colourless dielectric structure is carried as a scalar optical
        # field.  Keeping the three source-RGB channels here copied the colour
        # of the room/light used for the product photograph (typically a broad
        # gold or blue fringe) into every rib and silhouette highlight.  The
        # target transmission already provides the new scene chromaticity;
        # only the measured signed luminance variation is intrinsic evidence.
        source_rib_difference = source - horizontal_base
        del horizontal_base
        rib_luma = (
            source_rib_difference[..., 0] * 0.2126
            + source_rib_difference[..., 1] * 0.7152
            + source_rib_difference[..., 2] * 0.0722
        )
        del source_rib_difference
        # A real vertical flute is coherent through a substantial fraction of
        # the vessel height.  A wall/table boundary from the source room is
        # confined to a horizontal or diagonal band; horizontal high-pass
        # alone mistakes its saw-tooth refraction for glass identity.  Retain
        # the vertically persistent X-frequency field and average away those
        # local source-scene bands.  This field controls only target-pixel
        # displacement and neutral rib energy; source RGB still never enters
        # the reconstructed body.
        persistence_window = int(np.clip(round(extent * 0.10), 15, 201))
        if persistence_window % 2 == 0:
            persistence_window += 1
        rib_luma = cv2.boxFilter(
            rib_luma,
            cv2.CV_32F,
            (1, persistence_window),
            normalize=True,
            borderType=cv2.BORDER_REPLICATE,
        )

    transmission_target = (
        _clear_glass_refract_target(
            target_low, alpha, boundary_keep, rib_luma,
        )
        if material_mode == "clear_glass"
        else target_low
    )
    transmitted = transmission_target * tint.reshape(1, 1, 3) * attenuation
    # Clamp the residual so a broad source-scene black band cannot masquerade
    # as intrinsic glass while fine ribs, rims and highlights remain visible.
    negative_limit = -18.0 if material_mode == "clear_glass" else -42.0
    residual = np.clip(source_high, negative_limit, 74.0) * high_retention
    if material_mode == "clear_glass":
        # Do not try to infer glass identity from the sign of a two-dimensional
        # high-pass.  It made the previous implementation suppress the real
        # dark half of a fluted rib along with a source-room shelf, producing a
        # featureless milky cylinder.  A source shelf is row-constant, whereas
        # the pitcher's ribs vary *across X*.  Preserve that X-varying residual
        # and discard only the row-constant source environment.
        assert rib_luma is not None
        rib_detail = np.clip(rib_luma, -34.0, 78.0) * high_retention
        # In photographed clear glass the dark half of a rib is partly a real
        # refractive cue and partly an occluded source-room view.  Preserve it,
        # but at lower energy than the positive specular edge so the vessel
        # cannot collapse into a dark opaque silhouette on a dark receiver.
        rib_detail = np.where(
            rib_detail < 0.0,
            rib_detail * 0.48,
            rib_detail * 0.58,
        )
        # Rims and handles can include short curved / horizontal highlights.
        # Retain those only at the actual alpha contour; signed negative energy
        # receives a tighter boundary power so an interior dark shelf cannot
        # leak back in through a broad halo.
        source_high_luma = (
            source_high[..., 0] * 0.2126
            + source_high[..., 1] * 0.7152
            + source_high[..., 2] * 0.0722
        )
        boundary_positive = np.clip(source_high_luma, 0.0, 54.0)
        boundary_negative = np.clip(source_high_luma, -14.0, 0.0)
        # Fresnel/specular energy from colourless glass follows the target
        # illuminant and is effectively achromatic relative to that scene.  A
        # steeper boundary falloff keeps the measured one-pixel rim/handle
        # highlight but removes the several-pixel coloured matte inherited
        # from the source photograph.  Attenuate the rib field at that same
        # contour so it is not counted twice as both fluting and silhouette.
        contour_core = boundary_keep ** 1.88
        # A broad coloured source matte at a silhouette is also x-varying, so
        # the horizontal rib detector alone can mistake its *inner edge* for a
        # genuine flute.  Suppress that detector only in a narrow
        # distance-to-alpha band; real ribs in the body remain untouched.
        # The physical boundary highlight is reconstructed separately below.
        edge_band_px = float(np.clip(extent * 0.012, 2.5, 12.0))
        edge_matte_zone = np.exp(-distance / edge_band_px).astype(np.float32)
        rib_detail *= np.clip(
            1.0 - 0.98 * np.sqrt(edge_matte_zone), 0.10, 1.0,
        )
        # Coloured spill is even stronger evidence of a photographed source
        # environment: colourless glass cannot intrinsically own a broad gold,
        # blue or green stripe.  Remove that energy farther into the boundary
        # band, while a true narrow rib in the interior is exponentially
        # unaffected.  Only the scalar rib brightness is being attenuated;
        # alpha and geometry never change.
        occupied_y, occupied_x = np.where(occupied)
        short_extent = float(min(
            int(occupied_x.max() - occupied_x.min() + 1),
            int(occupied_y.max() - occupied_y.min() + 1),
        ))
        source_chroma = source.max(axis=2) - source.min(axis=2)
        coloured_spill = np.clip((source_chroma - 18.0) / 54.0, 0.0, 1.0)
        chroma_band_px = float(np.clip(short_extent * 0.035, 5.0, 12.0))
        coloured_spill *= np.exp(-distance / chroma_band_px)
        rib_detail *= 1.0 - np.clip(coloured_spill * 2.70, 0.0, 0.98)

        # ``source_high`` is deliberately blind to a several-pixel constant
        # source matte, including the real brightness measured at the alpha
        # boundary.  Recover only its scalar (achromatic) Fresnel energy and
        # collapse it onto the true alpha contour.  Hue still comes entirely
        # from the target scene, while a bright glass rim/spout/handle remains
        # legible instead of turning into either a gold sticker or a ghost.
        deep_threshold = max(4.0, extent * 0.080)
        deep_interior = occupied & (distance >= deep_threshold)
        reference_samples = luma[deep_interior]
        if reference_samples.size < 32:
            reference_samples = luma[occupied]
        interior_luma_reference = float(np.percentile(reference_samples, 55.0))
        # Estimate the non-glass illumination independently for each row.  A
        # global percentile turns the photographed source room's horizontal
        # wall/table transition into a false Fresnel band.  Interpolated,
        # vertically-smoothed row medians follow that source illumination, so
        # only local rim/handle energy survives as intrinsic glass evidence.
        row_reference = np.full(alpha.shape[0], interior_luma_reference, np.float32)
        valid_rows: list[int] = []
        valid_values: list[float] = []
        for row in range(alpha.shape[0]):
            row_mask = deep_interior[row]
            if int(np.count_nonzero(row_mask)) >= 8:
                valid_rows.append(row)
                valid_values.append(float(np.percentile(luma[row, row_mask], 55.0)))
        if len(valid_rows) >= 2:
            row_reference = np.interp(
                np.arange(alpha.shape[0], dtype=np.float32),
                np.asarray(valid_rows, np.float32),
                np.asarray(valid_values, np.float32),
            ).astype(np.float32)
            row_reference = cv2.GaussianBlur(
                row_reference.reshape(-1, 1),
                (0, 0),
                sigmaX=0.0,
                sigmaY=float(np.clip(extent * 0.010, 1.0, 10.0)),
                borderType=cv2.BORDER_REPLICATE,
            )[:, 0]
        measured_fresnel = (
            np.clip(luma - row_reference[:, None], 0.0, 46.0)
            * (boundary_keep ** 0.88)
            * 0.90
        )
        # A real dielectric has a small Fresnel return at every silhouette
        # edge even when its original photograph has an unusable source-room
        # backdrop.  This is purely alpha-geometry and target-illuminant
        # based; it introduces neither source colour nor invented texture.
        geometric_fresnel = 5.5 * (boundary_keep ** 1.65)
        contour_detail = (
            boundary_positive * contour_core * 0.88
            + boundary_negative * (boundary_keep ** 2.50) * 0.24
            + measured_fresnel
            + geometric_fresnel
        )
        # Broadcast the scalar field only at the final addition.  This keeps
        # source-room hue out of the product while retaining the exact measured
        # rib/rim geometry and is cheaper than a full 4K three-channel copy.
        residual = np.clip(
            (rib_detail + contour_detail)[..., None], -38.0, 74.0,
        )
        del rib_luma
    reconstructed = np.clip(transmitted + residual, 0.0, 255.0)

    if material_mode == "clear_glass":
        # Coverage alpha already preserves the geometric silhouette.  Keeping
        # the complete source low-frequency colour at that silhouette copies a
        # bright/dark source matte into the target as a halo.  Replace its
        # ambiguous low-frequency term even near the boundary while the
        # high-frequency residual above retains the measured rim and handle.
        interior = np.clip(1.0 - 0.045 * boundary_keep, 0.0, 1.0)
    elif material_mode == "frosted_glass":
        interior = np.clip(1.0 - 0.55 * boundary_keep, 0.0, 1.0)
    else:
        interior = np.clip(1.0 - boundary_keep, 0.0, 1.0)
    replacement_cap = 0.94 if material_mode == "clear_glass" else 0.96
    replace_weight = np.clip(
        strength * optical_candidate * interior, 0.0, replacement_cap,
    )
    # Explicit identity evidence is stronger than the material heuristic.  It
    # therefore disables target-scene replacement even on a smooth or bright
    # label that the legacy optical candidate would otherwise misclassify.
    replace_weight[identity] = 0.0
    result = source * (1.0 - replace_weight[..., None]) + reconstructed * replace_weight[..., None]
    output_rgb = rgb.copy()
    output_rgb[occupied] = np.clip(
        np.round(result[occupied]), 0, 255,
    ).astype(np.uint8)

    if material_mode == "clear_glass":
        # A colourless product has weak body opacity but its rim, ribs and
        # base still reflect distinctly on a glossy glass receiver.  A 9%
        # interior alpha was invisible after 4K browser downsampling.
        # Clear glass has little body opacity, but a glass rim/base still gives
        # a recognizable, faint mirror cue on a glossy receiver.  Keep this
        # deliberately below opaque/mixed materials while making the contact
        # and reflected silhouette survive a 4K browser preview.
        base_occlusion, base_reflection = 0.10, 0.48
    elif material_mode == "frosted_glass":
        base_occlusion, base_reflection = 0.15, 0.16
    elif material_mode == "colored_glass":
        base_occlusion, base_reflection = 0.23, 0.22
    else:
        # A labelled bottle is "mixed" only because some glass/liquid regions
        # transmit.  Its dark label and amber body still yield a materially
        # stronger mirror cue than a clear empty pitcher on the same glass top.
        base_occlusion, base_reflection = 0.31, 0.46
    edge_energy = np.clip(boundary_keep * 0.75 + detail_norm * 0.35, 0.0, 1.0)
    occlusion_fraction = np.clip(
        base_occlusion + edge_energy * (0.72 - base_occlusion), 0.0, 1.0,
    )
    reflection_fraction = np.clip(
        base_reflection + edge_energy * (0.90 - base_reflection), 0.0, 1.0,
    )
    shadow_alpha = np.round(alpha.astype(np.float32) * occlusion_fraction).astype(np.uint8)
    reflection_alpha = np.round(alpha.astype(np.float32) * reflection_fraction).astype(np.uint8)
    shadow_alpha[~occupied] = 0
    reflection_alpha[~occupied] = 0
    replacement_ratio = float(np.mean(replace_weight[occupied]))
    output_rgb = reinforce_transmissive_readability(
        output_rgb,
        rgb,
        target_background,
        alpha,
        material_mode,
    )
    output_rgb[identity] = rgb[identity]
    return OpticalReconstruction(
        output_rgb,
        alpha.copy(),
        shadow_alpha,
        reflection_alpha,
        replacement_ratio,
    )
