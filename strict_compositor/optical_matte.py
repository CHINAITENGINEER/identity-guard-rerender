from __future__ import annotations

"""ROI optical mattes for deterministic clear-glass compositing.

An ordinary RGBA cutout cannot faithfully move a transparent product between
rooms: its RGB contains the *source* room seen through the object.  This module
stores transferable optical evidence instead.  The estimator removes broad
source illumination and row-like room boundaries, while retaining measured
vertically persistent fluting and an achromatic contour/Fresnel cue.  The
evaluator then resolves those terms against pixels from the target background
in linear RGB.

The implementation is deliberately deterministic and ROI based.  It does not
generate pixels, does not consult a model, and never converts a complete 4K
target to floating point merely to resolve a small product.
"""

from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType
from typing import Any, Mapping

import cv2
import numpy as np


BBox = tuple[int, int, int, int]
_LUMA = np.asarray([0.2126, 0.7152, 0.0722], np.float32)
_EPSILON = np.float32(1.0e-6)


def _readonly_float32(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be a numeric numpy array")
    result = np.ascontiguousarray(array, dtype=np.float32).copy()
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class OpticalMatte:
    """A validated, immutable optical description for one placed-product ROI.

    ``bbox`` is the half-open target rectangle ``(x0, y0, x1, y1)``.  Every
    array is stored at exactly that ROI size instead of canvas size.

    Array semantics (all float32) are:

    * ``coverage_alpha``: measured geometric coverage in ``[0, 1]``.
    * ``opaque_identity_mask``: absolute opaque coverage, no greater than
      ``coverage_alpha``.  Labels/caps can use this path when explicitly
      supplied; clear material defaults to zero.
    * ``transmission_rgb``: multiplicative linear-RGB transmission.
    * ``refraction_flow``: target sampling displacement in pixels, ``(dx,dy)``.
    * ``specular_rgb``: colourless Fresnel reflectance evidence.  The evaluator
      uses its neutral magnitude to mix a target-scene environment proxy with
      refracted target pixels.  It is not a source-coloured additive halo.
    * ``shadow_transmittance``: fraction of receiver light transmitted through
      the product.  One means no cast-light loss, zero means opaque.
    * ``opaque_identity_rgb``: optional original linear RGB used only where
      ``opaque_identity_mask`` is non-zero.

    The arrays are copied and marked read-only during validation, so later UI
    or rendering code cannot silently mutate a matte after it was audited.
    """

    coverage_alpha: np.ndarray
    opaque_identity_mask: np.ndarray
    transmission_rgb: np.ndarray
    refraction_flow: np.ndarray
    specular_rgb: np.ndarray
    shadow_transmittance: np.ndarray
    bbox: BBox
    confidence: float
    provenance: Mapping[str, Any]
    opaque_identity_rgb: np.ndarray | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.bbox, tuple)
            or len(self.bbox) != 4
            or any(isinstance(v, bool) or not isinstance(v, Integral) for v in self.bbox)
        ):
            raise TypeError("bbox must be four integer values (x0, y0, x1, y1)")
        bbox = tuple(int(v) for v in self.bbox)
        x0, y0, x1, y1 = bbox
        if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
            raise ValueError("bbox must be a non-empty half-open rectangle")
        shape = (y1 - y0, x1 - x0)

        coverage = _readonly_float32(self.coverage_alpha, "coverage_alpha")
        opaque = _readonly_float32(
            self.opaque_identity_mask, "opaque_identity_mask",
        )
        transmission = _readonly_float32(
            self.transmission_rgb, "transmission_rgb",
        )
        flow = _readonly_float32(self.refraction_flow, "refraction_flow")
        specular = _readonly_float32(self.specular_rgb, "specular_rgb")
        shadow = _readonly_float32(
            self.shadow_transmittance, "shadow_transmittance",
        )

        for name, array in (
            ("coverage_alpha", coverage),
            ("opaque_identity_mask", opaque),
            ("shadow_transmittance", shadow),
        ):
            if array.shape != shape:
                raise ValueError(f"{name} must have ROI shape {shape}")
            if float(array.min()) < 0.0 or float(array.max()) > 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        for name, array, channels in (
            ("transmission_rgb", transmission, 3),
            ("refraction_flow", flow, 2),
            ("specular_rgb", specular, 3),
        ):
            if array.shape != (*shape, channels):
                raise ValueError(
                    f"{name} must have ROI shape {(*shape, channels)}",
                )
        if float(transmission.min()) < 0.0 or float(transmission.max()) > 2.0:
            raise ValueError("transmission_rgb must be in [0, 2]")
        if float(specular.min()) < 0.0 or float(specular.max()) > 1.0:
            raise ValueError("specular_rgb must be in [0, 1]")
        if np.any(opaque > coverage + 1.0e-5):
            raise ValueError("opaque_identity_mask cannot exceed coverage_alpha")

        identity: np.ndarray | None
        if self.opaque_identity_rgb is None:
            if np.any(opaque > 1.0e-6):
                raise ValueError(
                    "opaque_identity_rgb is required for a non-empty "
                    "opaque_identity_mask",
                )
            identity = None
        else:
            identity = _readonly_float32(
                self.opaque_identity_rgb, "opaque_identity_rgb",
            )
            if identity.shape != (*shape, 3):
                raise ValueError(
                    f"opaque_identity_rgb must have ROI shape {(*shape, 3)}",
                )
            if float(identity.min()) < 0.0 or float(identity.max()) > 1.0:
                raise ValueError("opaque_identity_rgb must be in [0, 1]")

        confidence = float(self.confidence)
        if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be a finite value in [0, 1]")
        if not isinstance(self.provenance, Mapping):
            raise TypeError("provenance must be a mapping")
        provenance = dict(self.provenance)
        if not provenance:
            raise ValueError("provenance must not be empty")

        object.__setattr__(self, "bbox", bbox)
        object.__setattr__(self, "coverage_alpha", coverage)
        object.__setattr__(self, "opaque_identity_mask", opaque)
        object.__setattr__(self, "transmission_rgb", transmission)
        object.__setattr__(self, "refraction_flow", flow)
        object.__setattr__(self, "specular_rgb", specular)
        object.__setattr__(self, "shadow_transmittance", shadow)
        object.__setattr__(self, "opaque_identity_rgb", identity)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "provenance", MappingProxyType(provenance))

    @property
    def roi_shape(self) -> tuple[int, int]:
        return self.coverage_alpha.shape

    def provenance_dict(self) -> dict[str, Any]:
        """Return a serialization-friendly copy of audit metadata."""
        return dict(self.provenance)


def _validate_rgb_image(rgb: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"{name} must have shape HxWx3")
    if array.dtype == np.uint8:
        return array
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(f"{name} must be uint8 or floating point")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    if float(array.min()) < 0.0 or float(array.max()) > 1.0:
        raise ValueError(f"floating {name} must be normalized to [0, 1]")
    return array


def _validate_alpha(alpha: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(alpha)
    if array.shape != shape:
        raise ValueError("source_alpha must match source_rgb height and width")
    if array.dtype == np.uint8:
        return array
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError("source_alpha must be uint8 or floating point")
    if not np.isfinite(array).all():
        raise ValueError("source_alpha must contain only finite values")
    if float(array.min()) < 0.0 or float(array.max()) > 1.0:
        raise ValueError("floating source_alpha must be normalized to [0, 1]")
    return array


def _normalise_alpha(alpha: np.ndarray) -> np.ndarray:
    if alpha.dtype == np.uint8:
        return alpha.astype(np.float32) / 255.0
    return alpha.astype(np.float32, copy=True)


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    value = rgb.astype(np.float32)
    if rgb.dtype == np.uint8:
        value *= 1.0 / 255.0
    return np.where(
        value <= 0.04045,
        value / 12.92,
        ((value + 0.055) / 1.055) ** 2.4,
    ).astype(np.float32)


def _linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    value = np.clip(linear.astype(np.float32), 0.0, 1.0)
    return np.where(
        value <= 0.0031308,
        value * 12.92,
        1.055 * np.power(value, 1.0 / 2.4) - 0.055,
    ).astype(np.float32)


def _infer_bbox(alpha: np.ndarray) -> BBox:
    threshold = 0 if alpha.dtype == np.uint8 else 1.0 / 255.0
    occupied_rows = np.flatnonzero(np.any(alpha > threshold, axis=1))
    occupied_columns = np.flatnonzero(np.any(alpha > threshold, axis=0))
    if occupied_rows.size == 0 or occupied_columns.size == 0:
        raise ValueError("source_alpha has no non-zero product coverage")
    return (
        int(occupied_columns[0]),
        int(occupied_rows[0]),
        int(occupied_columns[-1] + 1),
        int(occupied_rows[-1] + 1),
    )


def _validate_bbox(bbox: BBox, width: int, height: int) -> BBox:
    if (
        not isinstance(bbox, tuple)
        or len(bbox) != 4
        or any(isinstance(v, bool) or not isinstance(v, Integral) for v in bbox)
    ):
        raise TypeError("bbox must be four integer values (x0, y0, x1, y1)")
    x0, y0, x1, y1 = (int(v) for v in bbox)
    if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
        raise ValueError("bbox must be a non-empty half-open rectangle")
    if x1 > width or y1 > height:
        raise ValueError("bbox lies outside the source image")
    return x0, y0, x1, y1


def _horizontal_masked_lowpass(
    luma: np.ndarray,
    weight: np.ndarray,
    sigma_x: float,
) -> np.ndarray:
    """Row-wise normalized low pass; it follows shelves, not vertical ribs."""
    sigma_x = float(np.clip(sigma_x, 1.5, 64.0))
    radius = max(2, int(np.ceil(3.0 * sigma_x)))
    kernel_x = cv2.getGaussianKernel(radius * 2 + 1, sigma_x).astype(np.float32)
    kernel_y = np.asarray([1.0], np.float32)
    denominator = cv2.sepFilter2D(
        weight,
        cv2.CV_32F,
        kernel_x,
        kernel_y,
        borderType=cv2.BORDER_REPLICATE,
    )
    numerator = cv2.sepFilter2D(
        luma * weight,
        cv2.CV_32F,
        kernel_x,
        kernel_y,
        borderType=cv2.BORDER_REPLICATE,
    )
    return numerator / np.maximum(denominator, _EPSILON)


def _largest_true_run(values: np.ndarray) -> tuple[int, int] | None:
    best: tuple[int, int] | None = None
    start: int | None = None
    for index, active in enumerate(values.tolist() + [False]):
        if active and start is None:
            start = index
        elif not active and start is not None:
            run = (start, index)
            if best is None or run[1] - run[0] > best[1] - best[0]:
                best = run
            start = None
    return best


def _source_environment_boundary_rejection(
    signed_topology: np.ndarray,
    interior: np.ndarray,
    clear_fraction: np.ndarray,
    rib_edge_ramp: np.ndarray,
) -> np.ndarray:
    """Return a soft mask for source-room boundaries masquerading as glass.

    A photographed wall/table or shelf boundary is horizontally coherent but
    occupies only a short vertical interval.  A real flute is the opposite:
    after local centring it persists down a substantial part of a vessel.  The
    detector therefore compares a horizontally aggregated topology-energy map
    with its much broader vertical baseline.  It never inspects target pixels
    and is applied only to source-derived topology terms; target refraction is
    resolved later and cannot be erased by this mask.

    The lowest part of the silhouette is excluded because a thick moulded base
    legitimately contains compact, non-vertical optical structure.  The narrow
    measured silhouette is also excluded through ``interior``/``rib_edge_ramp``.
    """

    signed = np.asarray(signed_topology, dtype=np.float32)
    core = np.clip(
        np.asarray(interior, dtype=np.float32)
        * np.asarray(clear_fraction, dtype=np.float32)
        * np.asarray(rib_edge_ramp, dtype=np.float32),
        0.0,
        1.0,
    )
    if signed.shape != core.shape or signed.ndim != 2:
        raise ValueError("source topology and glass masks must share a 2D shape")
    height, width = signed.shape
    if height < 8 or width < 8 or not np.any(core > 0.08):
        return np.zeros_like(signed, dtype=np.float32)

    yy = (
        (np.arange(height, dtype=np.float32) + 0.5)
        / max(1.0, float(height))
    ).reshape(-1, 1)
    # Keep the bottom twelve percent under the dedicated thick-base model.
    non_base = np.clip((0.90 - yy) / 0.08, 0.0, 1.0)
    core *= non_base
    topology_energy = (
        np.clip((np.abs(signed) - 0.18) / 0.82, 0.0, 1.0)
        * core
    )
    if not np.any(topology_energy > 0.04):
        return np.zeros_like(signed, dtype=np.float32)

    # Horizontal aggregation joins a refracted/saw-tooth room boundary across
    # neighbouring columns.  A narrow persistent flute remains local in X.
    coherent = cv2.GaussianBlur(
        topology_energy,
        (0, 0),
        sigmaX=float(np.clip(width * 0.055, 2.0, 18.0)),
        sigmaY=float(np.clip(height * 0.010, 0.8, 3.5)),
        borderType=cv2.BORDER_REPLICATE,
    )
    vertical_baseline = cv2.GaussianBlur(
        coherent,
        (1, 0),
        sigmaX=0.0,
        sigmaY=float(np.clip(height * 0.105, 4.0, 34.0)),
        borderType=cv2.BORDER_REPLICATE,
    )
    transient_band = np.maximum(coherent - vertical_baseline * 1.08, 0.0)
    samples = transient_band[core > 0.08]
    band_scale = (
        float(np.percentile(samples, 90.0)) if samples.size else 0.0
    )
    if band_scale < 0.006:
        return np.zeros_like(signed, dtype=np.float32)

    coherent_gate = np.clip(
        (transient_band - max(0.003, band_scale * 0.18))
        / max(0.010, band_scale * 0.82),
        0.0,
        1.0,
    )
    local_gate = np.clip(topology_energy * 1.65, 0.0, 1.0)
    rejection = coherent_gate * local_gate * core

    # Explicitly protect columns whose topology is present through much of the
    # clear body.  This final guard makes the result independent of flute pitch
    # and prevents a dense rib family from being mistaken for one broad band.
    body_weight = np.sum(core, axis=0)
    persistent_weight = np.sum(
        (topology_energy > 0.16).astype(np.float32) * core,
        axis=0,
    )
    persistence = persistent_weight / np.maximum(body_weight, _EPSILON)
    persistent_columns = (persistence >= 0.34).astype(np.uint8)[None, :]
    persistent_columns = cv2.dilate(
        persistent_columns,
        np.ones((1, 3), np.uint8),
    )[0].astype(np.float32)
    rejection *= 1.0 - persistent_columns.reshape(1, -1)
    return np.clip(rejection, 0.0, 1.0).astype(np.float32)


def _remap_target_supersampled(
    image: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
) -> np.ndarray:
    """Remap target radiance with a tiny edge-adaptive subpixel footprint.

    Bilinear interpolation evaluates one displaced coordinate.  A rapidly
    varying rib flow can therefore turn a coherent oblique table edge into a
    one-pixel staircase.  Four additional quarter-pixel samples approximate a
    compact pixel footprint, but are blended only where those samples disagree
    with the centre.  Quiet wall texture and already-smooth target regions are
    left effectively unchanged.
    """

    centre = cv2.remap(
        image,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    accumulated = centre * 0.50
    variation = np.zeros(map_x.shape, np.float32)
    offset = 0.30
    for dx, dy in (
        (-offset, -offset),
        (-offset, offset),
        (offset, -offset),
        (offset, offset),
    ):
        sample = cv2.remap(
            image,
            map_x + dx,
            map_y + dy,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        accumulated += sample * 0.125
        variation = np.maximum(
            variation,
            np.mean(np.abs(sample - centre), axis=2),
        )
    edge_gate = np.clip((variation - 0.0025) / 0.035, 0.0, 1.0)
    return (
        centre * (1.0 - edge_gate[..., None])
        + accumulated * edge_gate[..., None]
    ).astype(np.float32)


def _tubular_appendage_optics(
    inside: np.ndarray,
    body_x0: int,
    body_x1: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Infer side-handle optics from measured alpha topology.

    A vessel body is well approximated by the long-column cylindrical lens
    used by :func:`estimate_clear_glass_matte`, but a loop handle is not.  In
    particular, restricting refraction to the longest body-column run leaves
    a perfectly valid handle alpha with zero optical response, so the handle
    reads as a broken wire after preview downsampling.

    This helper finds *enclosed background components* next to the body run.
    The glass between an enclosed hole and the exterior is a measured tubular
    wall.  Distances to those two boundaries provide a local cross-section and
    its normal without a pitcher-specific box, pixel coordinate or source RGB
    cue.  Every returned term is masked by ``inside``.  Consequently genuine
    handle holes remain holes and can never be filled by this inference.
    """
    binary = np.asarray(inside, dtype=np.uint8)
    if binary.ndim != 2:
        raise ValueError("inside must be a two-dimensional mask")
    height, width = binary.shape
    flow = np.zeros((height, width, 2), np.float32)
    blend = np.zeros((height, width), np.float32)
    grazing = np.zeros((height, width), np.float32)
    if height < 8 or width < 8 or body_x1 <= body_x0:
        return flow, blend, grazing, {
            "count": 0,
            "model": "enclosed_alpha_hole_tubular_lens_v1",
            "components": [],
        }

    # Padding the inverse mask with known background creates one authoritative
    # exterior component even when the tight object ROI touches every edge.
    background = (binary == 0).astype(np.uint8)
    padded_background = cv2.copyMakeBorder(
        background, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=1,
    )
    component_count, padded_labels = cv2.connectedComponents(
        padded_background, connectivity=8,
    )
    labels = padded_labels[1:-1, 1:-1]
    exterior_label = int(padded_labels[0, 0])
    exterior = (labels == exterior_label).astype(np.uint8)
    if not np.any(exterior):
        return flow, blend, grazing, {
            "count": 0,
            "model": "enclosed_alpha_hole_tubular_lens_v1",
            "components": [],
        }

    distance_to_exterior = cv2.distanceTransform(
        (1 - exterior).astype(np.uint8), cv2.DIST_L2, 5,
    )
    yy, xx = np.indices((height, width), dtype=np.float32)
    body_centre = 0.5 * float(body_x0 + body_x1 - 1)
    foreground_area = max(1, int(np.count_nonzero(binary)))
    minimum_hole_area = max(24, int(round(height * width * 0.0005)))
    minimum_seed_area = max(18, int(round(foreground_area * 0.0008)))
    components: list[dict[str, Any]] = []

    for label in range(1, int(component_count)):
        if label == exterior_label:
            continue
        hole = labels == label
        hole_area = int(np.count_nonzero(hole))
        if (
            hole_area < minimum_hole_area
            or hole_area > int(round(height * width * 0.42))
        ):
            continue
        hole_y, hole_x = np.where(hole)
        if hole_x.size == 0:
            continue
        hole_x0, hole_x1 = int(hole_x.min()), int(hole_x.max()) + 1
        hole_y0, hole_y1 = int(hole_y.min()), int(hole_y.max()) + 1
        hole_width = hole_x1 - hole_x0
        hole_height = hole_y1 - hole_y0
        if min(hole_width, hole_height) < 5:
            continue

        hole_centre_x = 0.5 * float(hole_x0 + hole_x1 - 1)
        if hole_centre_x >= float(body_x1):
            side = "right"
            side_distance = xx - float(body_x1 - 1)
            outside_body = binary.astype(bool) & (xx >= float(body_x1 - 1))
            joint_x = float(body_x1 - 1)
        elif hole_centre_x <= float(body_x0):
            side = "left"
            side_distance = float(body_x0) - xx
            outside_body = binary.astype(bool) & (xx <= float(body_x0))
            joint_x = float(body_x0)
        else:
            # Openings inside the main vessel body (for example the mouth of
            # a top-down glass) are not side appendages and keep the existing
            # body lens model.
            continue

        distance_to_hole = cv2.distanceTransform(
            (~hole).astype(np.uint8), cv2.DIST_L2, 5,
        )
        local_radius = float(np.clip(
            0.42 * min(hole_width, hole_height),
            max(4.0, min(height, width) * 0.006),
            max(5.0, min(height, width) * 0.18),
        ))
        wall_candidate = (
            binary.astype(bool)
            & (distance_to_hole <= local_radius)
            & (
                distance_to_hole + distance_to_exterior
                <= local_radius * 2.60
            )
        )
        outside_seed = wall_candidate & outside_body
        if int(np.count_nonzero(outside_seed)) < minimum_seed_area:
            continue

        # Smoothly hand the optical field from the cylindrical body to the
        # tubular appendage across a topology-derived joint band.  This avoids
        # the zero-flow seam that previously cut handles at both attachments.
        joint_blend = max(4.0, local_radius * 0.75)
        blend_coordinate = np.clip(
            (side_distance + joint_blend) / (2.0 * joint_blend),
            0.0,
            1.0,
        )
        side_weight = (
            blend_coordinate
            * blend_coordinate
            * (3.0 - 2.0 * blend_coordinate)
        )
        local_weight = wall_candidate.astype(np.float32) * side_weight

        # Signed local cross-section: -1 at the enclosed-hole interface and
        # +1 at the exterior interface.  Its gradient is the alpha-topology
        # surface normal; the odd lens profile bends opposite tube faces in
        # opposite directions while remaining zero at both alpha boundaries.
        distance_sum = distance_to_hole + distance_to_exterior
        cross_section = np.clip(
            (distance_to_hole - distance_to_exterior)
            / np.maximum(distance_sum, _EPSILON),
            -1.0,
            1.0,
        ).astype(np.float32)
        normal_x = cv2.Sobel(
            cross_section, cv2.CV_32F, 1, 0, ksize=3,
        )
        normal_y = cv2.Sobel(
            cross_section, cv2.CV_32F, 0, 1, ksize=3,
        )
        normal_length = np.sqrt(normal_x * normal_x + normal_y * normal_y)
        normal_x /= np.maximum(normal_length, np.float32(1.0e-4))
        normal_y /= np.maximum(normal_length, np.float32(1.0e-4))
        lens_profile = (
            -2.6 * cross_section * (1.0 - cross_section * cross_section)
        )
        wall_thickness = 2.0 * np.minimum(
            distance_to_hole, distance_to_exterior,
        )
        maximum_displacement = float(np.clip(
            min(height, width) * 0.018, 4.0, 18.0,
        ))
        displacement = np.clip(
            wall_thickness * 0.30, 0.35, maximum_displacement,
        )
        local_flow = np.stack(
            [
                normal_x * lens_profile * displacement * local_weight,
                normal_y * lens_profile * displacement * local_weight,
            ],
            axis=2,
        ).astype(np.float32)
        local_grazing = (
            np.power(np.clip(np.abs(lens_profile), 0.0, 1.0), 1.40)
            * local_weight
        ).astype(np.float32)

        selector = local_weight > blend
        flow[selector] = local_flow[selector]
        blend = np.maximum(blend, local_weight)
        grazing = np.maximum(grazing, local_grazing)
        audited_flow = np.linalg.norm(local_flow, axis=2)[local_weight > 0.15]
        components.append({
            "side": side,
            "hole_bbox": [hole_x0, hole_y0, hole_x1, hole_y1],
            "hole_area": hole_area,
            "support_area": int(np.count_nonzero(local_weight > 0.05)),
            "joint_x": joint_x,
            "local_radius": local_radius,
            "flow_p90": (
                float(np.percentile(audited_flow, 90.0))
                if audited_flow.size
                else 0.0
            ),
        })

    # Defence in depth: inferred optical evidence can never leave measured
    # product coverage, even if a future connected-component implementation
    # changes its label ordering.
    measured = binary.astype(np.float32)
    flow *= measured[..., None]
    blend *= measured
    grazing *= measured
    return flow, blend, grazing, {
        "count": len(components),
        "model": "enclosed_alpha_hole_tubular_lens_v1",
        "components": components,
    }


def _extract_opaque_mask(
    mask: np.ndarray | None,
    source_shape: tuple[int, int],
    bbox: BBox,
    coverage: np.ndarray,
) -> np.ndarray:
    if mask is None:
        return np.zeros_like(coverage)
    array = np.asarray(mask)
    x0, y0, x1, y1 = bbox
    if array.shape == source_shape:
        array = array[y0:y1, x0:x1]
    elif array.shape != coverage.shape:
        raise ValueError(
            "opaque_identity_mask must match source image or inferred ROI",
        )
    if array.dtype == np.uint8:
        normalized = array.astype(np.float32) / 255.0
    elif np.issubdtype(array.dtype, np.floating):
        if not np.isfinite(array).all():
            raise ValueError("opaque_identity_mask contains non-finite values")
        if float(array.min()) < 0.0 or float(array.max()) > 1.0:
            raise ValueError("opaque_identity_mask must be in [0, 1]")
        normalized = array.astype(np.float32)
    else:
        raise TypeError("opaque_identity_mask must be uint8 or floating point")
    # Treat the supplied mask as a material-membership weight.  Geometric
    # coverage remains authoritative at anti-aliased product edges.
    return np.minimum(normalized, coverage).astype(np.float32)


def estimate_clear_glass_matte(
    source_rgb: np.ndarray,
    source_alpha: np.ndarray,
    *,
    bbox: BBox | None = None,
    opaque_identity_mask: np.ndarray | None = None,
    confidence: float | None = None,
    transmission_strength: float = 0.97,
    commercial_visibility: float = 1.0,
    provenance: Mapping[str, Any] | None = None,
) -> OpticalMatte:
    """Estimate a transferable clear-glass matte from source RGB and alpha.

    The caller should pass a geometrically aligned cutout, normally after the
    deterministic placement transform.  A supplied ``bbox`` is authoritative;
    otherwise the tight non-zero-alpha rectangle is used.

    Source-room colour is never copied into transmission or specular colour.
    Only achromatic, vertically persistent source contrast contributes to the
    rib modulation.  Broad row-wise darkness is absorbed by the source-room
    estimate and therefore cannot become a black band in the target scene.
    """
    rgb = _validate_rgb_image(source_rgb, "source_rgb")
    alpha = _validate_alpha(source_alpha, rgb.shape[:2])
    if (
        not np.isfinite(transmission_strength)
        or not 0.0 <= float(transmission_strength) <= 1.0
    ):
        raise ValueError("transmission_strength must be finite and in [0, 1]")
    if (
        not np.isfinite(commercial_visibility)
        or not 0.0 <= float(commercial_visibility) <= 1.0
    ):
        raise ValueError("commercial_visibility must be finite and in [0, 1]")
    transmission_strength = float(transmission_strength)
    commercial_visibility = float(commercial_visibility)
    # The verified route starts at 0.60.  Normalising within that interval
    # keeps the former 0.97 estimator behaviour byte-for-byte at its default,
    # while lower semantic transmission now has a continuous, auditable
    # effect instead of being recorded in provenance and otherwise ignored.
    transmission_response = float(np.clip(
        (transmission_strength - 0.60) / 0.37, 0.0, 1.0,
    ))
    visibility_response = 0.62 + 0.38 * commercial_visibility
    image_height, image_width = rgb.shape[:2]
    selected_bbox = (
        _infer_bbox(alpha)
        if bbox is None
        else _validate_bbox(bbox, image_width, image_height)
    )
    x0, y0, x1, y1 = selected_bbox
    coverage = _normalise_alpha(alpha[y0:y1, x0:x1])
    if not np.any(coverage > 1.0 / 255.0):
        raise ValueError("bbox contains no non-zero product coverage")
    source_linear = _srgb_to_linear(rgb[y0:y1, x0:x1])
    height, width = coverage.shape
    inside = (coverage > 1.0 / 255.0).astype(np.uint8)

    # Distance to the measured silhouette gives a stable target-independent
    # contour cue.  Anti-aliased coverage is retained separately and remains
    # the final authority on which target pixels may change.
    # ``bbox`` is normally tight, so an object can touch all four ROI edges.
    # OpenCV's distance transform needs an explicit zero-valued exterior;
    # without this one-pixel frame a fully occupied rectangular ROI reports an
    # effectively infinite distance everywhere and silently loses its Fresnel
    # contour.  Pad only the small ROI and crop back immediately.
    padded_inside = cv2.copyMakeBorder(
        inside, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0,
    )
    distance = cv2.distanceTransform(padded_inside, cv2.DIST_L2, 5)[1:-1, 1:-1]
    # Dielectric Fresnel response is concentrated at a narrow silhouette.
    # A broad distance falloff makes clear glass read as frosted white plastic
    # at normal viewing scale, especially around a thick handle.
    # The final 4K/7K canvas is commonly fitted into a 1200--1900 px browser
    # viewport.  A six-source-pixel dielectric rim then falls below two display
    # pixels and disappears even though the full-resolution calculation is
    # correct.  Keep the measured alpha contour, but let its target-derived
    # Fresnel band survive that deterministic downsample.
    # Keep the higher response from the preview-legibility pass, but confine
    # it to a thinner dielectric interface.  A wide bright band survives the
    # browser downsample by making the whole vessel look frosted; a narrow,
    # stronger band preserves a clear wall while keeping rim/spout topology.
    preview_scale = min(
        1.0,
        1800.0 / max(1.0, float(max(image_height, image_width))),
    )
    edge_width = float(np.clip(
        max(
            min(height, width) * 0.008,
            1.35 / max(1.0e-6, preview_scale),
        ),
        1.35,
        6.0,
    ))
    boundary = np.exp(-np.maximum(distance - 1.0, 0.0) / edge_width)
    boundary *= inside.astype(np.float32)
    interior = np.clip(1.0 - boundary, 0.0, 1.0) * coverage

    source_luma = source_linear @ _LUMA
    row_environment = _horizontal_masked_lowpass(
        source_luma,
        coverage,
        sigma_x=float(np.clip(width * 0.115, 3.0, 64.0)),
    )
    horizontal_residual = (source_luma - row_environment) * coverage

    # A genuine flute/rib is coherent down most of a vessel; a shelf or room
    # reflection occupies only a limited vertical interval.  Weight the column
    # mean by sign persistence so local diagonal bands are rejected as well as
    # perfectly horizontal bands.
    persistence_weight = np.where(interior > 0.08, interior, 0.0)
    column_weight = persistence_weight.sum(axis=0)
    column_mean = (
        (horizontal_residual * persistence_weight).sum(axis=0)
        / np.maximum(column_weight, _EPSILON)
    )
    sign_coherence = np.abs(
        (np.sign(horizontal_residual) * persistence_weight).sum(axis=0)
        / np.maximum(column_weight, _EPSILON)
    )
    column_signal = column_mean * np.clip(sign_coherence * 1.8, 0.0, 1.0)
    valid_columns = column_weight >= max(3.0, float(column_weight.max()) * 0.42)
    # Remove any broad body/source gradient along X.  Only narrow persistent
    # optical structure is transferable between scenes.
    smooth_sigma = float(np.clip(width * 0.030, 1.2, 10.0))
    broad_column_signal = cv2.GaussianBlur(
        column_signal.reshape(1, -1),
        (0, 0),
        sigmaX=smooth_sigma,
        sigmaY=0.0,
        borderType=cv2.BORDER_REPLICATE,
    )[0]
    column_signal = column_signal - broad_column_signal
    column_signal[~valid_columns] = 0.0
    active_amplitudes = np.abs(column_signal[valid_columns])
    active_amplitudes = active_amplitudes[active_amplitudes > 0.0025]
    signal_scale = (
        float(np.percentile(active_amplitudes, 68))
        if active_amplitudes.size
        else 0.0
    )
    if signal_scale >= 0.004:
        rib_phase = np.tanh(column_signal / max(0.012, signal_scale * 1.35))
    else:
        rib_phase = np.zeros(width, np.float32)
    # The first band pass above is applied before ``tanh``.  That nonlinear
    # compression can turn an asymmetric source-room reflection into a broad
    # negative plateau again (a dark vertical band across the reconstructed
    # vessel).  Remove the reintroduced low frequency *after* compression and
    # renormalise only the remaining narrow optical structure.  This preserves
    # genuine repeating flutes while preventing a photographed doorway,
    # cabinet or photographer from becoming transferable glass identity.
    phase_sigma = float(np.clip(width * 0.030, 2.0, 12.0))
    phase_lowpass = cv2.GaussianBlur(
        rib_phase.reshape(1, -1),
        (0, 0),
        sigmaX=phase_sigma,
        sigmaY=0.0,
        borderType=cv2.BORDER_REPLICATE,
    )[0]
    rib_phase = rib_phase - phase_lowpass
    phase_samples = np.abs(rib_phase[valid_columns])
    phase_scale = (
        float(np.percentile(phase_samples, 92))
        if phase_samples.size
        else 0.0
    )
    if phase_scale >= 0.025:
        rib_phase = np.clip(rib_phase / phase_scale, -1.0, 1.0)
    else:
        rib_phase.fill(0.0)
    rib_phase[~valid_columns] = 0.0
    rib_map = rib_phase.reshape(1, -1) * interior
    # A several-pixel coloured source matte at the silhouette is vertically
    # persistent and can otherwise masquerade as a genuine flute.  Collapse
    # the column-derived rib cue in a narrow contour band; the physical rim is
    # reconstructed independently by the achromatic Fresnel term below.
    # ``boundary ** 0.10`` stays close to one for tens of source pixels and
    # therefore erased most legitimate body flutes long after the coloured
    # silhouette matte had ended.  ``rib_edge_ramp`` below already provides
    # the measured narrow exclusion band.  A linear Fresnel fade keeps that
    # safety property without suppressing the product's interior identity.
    rib_map *= np.clip(1.0 - boundary, 0.0, 1.0)
    rib_exclusion = max(5.5, edge_width * 1.20)
    rib_edge_ramp = np.clip(
        (distance - rib_exclusion) / max(2.0, edge_width * 1.4),
        0.0,
        1.0,
    )
    rib_map *= rib_edge_ramp
    # Fluted glass reflects the room primarily where the rib normal changes,
    # not only at the bright half-wave crest.  The derivative of the measured,
    # locally centred rib phase is target-independent surface-normal evidence.
    # Converting it to a sparse grazing-response map lets those narrow faces
    # reflect the *current* room below, producing dark and bright lines instead
    # of a flat beige transparency veil.
    rib_slope = np.abs(np.gradient(rib_phase.astype(np.float32)))
    active_slopes = rib_slope[valid_columns]
    active_slopes = active_slopes[active_slopes > 0.01]
    rib_slope_scale = (
        float(np.percentile(active_slopes, 82))
        if active_slopes.size
        else 0.0
    )
    if rib_slope_scale >= 0.01:
        rib_normal_response = np.power(
            np.clip(rib_slope / rib_slope_scale, 0.0, 1.0),
            1.45,
        ).reshape(1, -1) * interior * rib_edge_ramp
    else:
        rib_normal_response = np.zeros_like(interior)

    opaque = _extract_opaque_mask(
        opaque_identity_mask,
        rgb.shape[:2],
        selected_bbox,
        coverage,
    )
    clear_fraction = np.clip(1.0 - opaque / np.maximum(coverage, _EPSILON), 0.0, 1.0)

    # Neutral Beer-Lambert-like absorption plus a bounded, zero-mean rib cue.
    # No source hue enters this tensor.  This is the main reason a yellow wall
    # or dark source shelf cannot travel with a clear glass product.
    signed_rib_transmission = np.where(
        rib_map >= 0.0,
        0.200 * rib_map,
        0.200 * rib_map,
    )
    signed_rib_transmission *= 0.70 + 0.30 * transmission_response
    absorption = 0.002 + 0.010 * (1.0 - transmission_response)
    transmission_scalar = np.clip(
        1.0 - absorption * interior + signed_rib_transmission,
        0.72,
        1.28,
    )
    transmission = np.repeat(transmission_scalar[..., None], 3, axis=2)
    # ``transmission_rgb`` is conditional on product coverage; geometric
    # anti-aliasing is applied exactly once by the evaluator.  Adding
    # ``1-coverage`` here would make a half-covered edge transmit nearly 2x
    # and then blend that bright value a second time, producing a white halo.
    transmission = (
        transmission * clear_fraction[..., None]
        + (1.0 - clear_fraction)[..., None]
    ).astype(np.float32)
    transmission[coverage <= 0.0] = 1.0

    # The measured silhouette and positive rib response become an achromatic
    # Fresnel/specular residual.  Only magnitude is measured from the source;
    # source RGB ratios are deliberately discarded to prevent coloured matte
    # fringe leakage.
    # The signed rib phase remains in transmission, producing the required
    # alternating bright/dark lensing.  Only its positive phase contributes a
    # micro-specular response; using absolute magnitude here would fill the
    # negative (dark) grooves with light and erase the measured topology.
    positive_rib = np.maximum(rib_map, 0.0)
    source_residual_strength = np.clip(
        np.abs(horizontal_residual) / max(0.035, signal_scale * 2.5),
        0.0,
        1.0,
    )
    # Preserve measured *high-frequency* glass topology without preserving
    # the photographed source room.  A compact Gaussian removes wall/shelf
    # illumination while the absolute achromatic residual keeps thin rim,
    # spout and handle highlights.  It is allowed only on the measured
    # silhouette/rib support and is later coloured from the target
    # illuminant, so a black or amber source environment cannot leak through.
    topology_lowpass = cv2.GaussianBlur(
        source_luma,
        (0, 0),
        sigmaX=2.2,
        sigmaY=2.2,
        borderType=cv2.BORDER_REPLICATE,
    )
    topology_residual = np.abs(source_luma - topology_lowpass) * coverage
    topology_samples = topology_residual[
        (coverage > 0.20) & (boundary > 0.035)
    ]
    topology_scale = (
        float(np.percentile(topology_samples, 88))
        if topology_samples.size
        else 0.0
    )
    if topology_scale >= 0.0035:
        topology_detail = np.clip(
            (topology_residual - 0.0025) / max(0.012, topology_scale),
            0.0,
            1.0,
        )
    else:
        topology_detail = np.zeros_like(coverage)
    # This two-dimensional detail term is deliberately contour-only.  The
    # long interior flutes already have a zero-mean signed representation;
    # letting the absolute high-pass illuminate that core would create a
    # broad frosted fill instead of clear glass.
    # Keep compact spout/handle/rim evidence on the measured interface, but
    # do not let a several-pixel source-room matte remain equally strong four
    # pixels inside that interface.  A slightly steeper falloff preserves the
    # true alpha contour while collapsing coloured source fringe to a thin,
    # neutral target-lit highlight.
    topology_support = np.power(boundary, 0.70)
    topology_detail *= topology_support * clear_fraction
    # Preserve the *polarity* of fine dielectric structure as a neutral
    # transmission perturbation.  An absolute high-pass can only brighten the
    # contour and therefore produces the familiar white cutout halo.  Signed
    # topology yields adjacent bright and dark optical cues like real glass,
    # while the compact high-pass and contour support prevent any broad source
    # room colour or scenery from travelling into the target.
    signed_topology_raw = np.clip(
        (source_luma - topology_lowpass) / max(0.012, topology_scale),
        -1.0,
        1.0,
    )
    signed_topology = signed_topology_raw * topology_support * clear_fraction
    # Preserve only compact, strongly vertical high-frequency optical detail in
    # the clear interior.  This retains original ribs and glints as achromatic
    # gain while rejecting broad horizontal shelves/walls from the source room.
    gradient_x = np.abs(cv2.Sobel(source_luma, cv2.CV_32F, 1, 0, ksize=3))
    gradient_y = np.abs(cv2.Sobel(source_luma, cv2.CV_32F, 0, 1, ksize=3))
    vertical_structure = gradient_x / np.maximum(
        gradient_x + gradient_y,
        0.010,
    )
    detail_gate = np.power(
        np.clip((np.abs(signed_topology_raw) - 0.38) / 0.62, 0.0, 1.0),
        1.35,
    )
    source_boundary_rejection = _source_environment_boundary_rejection(
        signed_topology_raw,
        interior,
        clear_fraction,
        rib_edge_ramp,
    )
    source_topology_keep = 1.0 - source_boundary_rejection
    # The rejection mask has authority only over source-derived optical
    # topology.  Target-scene refraction is sampled later in
    # ``_target_patch_linear`` and remains completely untouched.
    topology_detail *= source_topology_keep
    signed_topology *= source_topology_keep
    interior_topology = (
        signed_topology_raw
        * vertical_structure
        * detail_gate
        * interior
        * rib_edge_ramp
        * clear_fraction
        * source_topology_keep
    )
    # The lower part of a real pitcher contains substantially more glass than
    # its side wall.  Treating that thick base exactly like a thin window makes
    # it disappear on a quiet wall/table.  Retain only compact, achromatic
    # source high-frequency evidence here: the signed high-pass has already
    # discarded the photographed room's low-frequency colour and illumination.
    # A small non-vertical allowance keeps the cellular/glittering detail of a
    # moulded base without admitting a broad source shelf as target imagery.
    normalized_y = (
        (np.arange(height, dtype=np.float32) + 0.5) / max(1.0, float(height))
    ).reshape(-1, 1)
    thick_base_profile = np.power(
        np.clip((normalized_y - 0.76) / 0.24, 0.0, 1.0),
        1.35,
    )
    thick_base_gate = np.power(
        np.clip((np.abs(signed_topology_raw) - 0.20) / 0.80, 0.0, 1.0),
        1.10,
    )
    thick_base_topology = (
        signed_topology_raw
        * thick_base_gate
        * (0.42 + 0.58 * vertical_structure)
        * thick_base_profile
        * interior
        * clear_fraction
        * source_topology_keep
    )
    transmission = np.clip(
        transmission * (
            1.0
            + 0.135 * signed_topology[..., None]
            + 0.245 * interior_topology[..., None]
            + 0.360 * thick_base_topology[..., None]
        ),
        0.72,
        1.28,
    ).astype(np.float32)
    # A colourless object against a quiet, similarly coloured wall can become
    # mathematically correct yet perceptually absent after a 4K image is fitted
    # to a browser canvas.  The silhouette and persistent flutes are measured
    # geometry, so they may carry the stronger grazing-angle dielectric energy
    # required at normal viewing distance.  This remains achromatic and never
    # reuses the photographed source-room colour.
    # A dielectric approaches strong reflectance at grazing incidence.  Keep
    # that energy on the narrow measured interface rather than compensating
    # with a weak, body-wide additive veil in the evaluator.
    fresnel = boundary * (0.900 + 0.150 * source_residual_strength)
    # Repeating flutes are primarily a refractive/transmission phenomenon at
    # normal view.  Feeding every positive flute into the panorama reflection
    # accumulated a broad grey veil whenever the room outside the ROI was
    # darker than the local wall.  Keep only a very small grazing reflection
    # cue here; the signed, locally centred transmission above carries the
    # actual rib identity without a one-sided brightness bias.
    # Only the narrow crest of a measured flute receives a grazing reflection.
    # Using the complete positive half-wave filled roughly half of a ribbed
    # vessel with one-sided grey energy and made it look frosted.  A crest gate
    # preserves sparse, photographic glints while the signed transmission field
    # above remains responsible for the alternating light/dark rib identity.
    rib_crest = np.power(
        np.clip((positive_rib - 0.62) / 0.38, 0.0, 1.0),
        1.65,
    )
    # Only the measured positive crest receives an unresolved glint.  Dark
    # groove faces keep their signed transmission/reflection and are not filled
    # with the same additive energy.
    fresnel += rib_crest * 0.150
    fresnel += rib_normal_response * 0.320
    # ``topology_detail`` is already restricted to the measured contour band.
    # It restores thin spout/handle/rim structure from achromatic high-frequency
    # evidence without copying source-room colour or creating a body-wide veil.
    fresnel += topology_detail * 0.110
    # Optical thickness increases reflection/refraction contrast at the base,
    # but the response still samples the *target* panorama in the evaluator.
    # Only the magnitude of compact source topology controls where it appears.
    fresnel += np.abs(thick_base_topology) * 0.620
    # Fresnel is a *conditional* material response.  Coverage is applied later
    # by the one authoritative alpha-over operation.  Multiplying it here as
    # well squared antialiased coverage and erased precisely the thin glass
    # topology (spout, handle and rim) that should survive at preview scale.
    fresnel *= clear_fraction * visibility_response
    # At a grazing angle even ordinary clear glass approaches strong
    # reflectance.  Keep that energy on the measured narrow interfaces, but do
    # not cap a real rim, handle or flute face at the former 0.50 preview limit.
    # Geometric coverage is still applied once, later, by alpha-over.
    specular = np.repeat(np.clip(fresnel, 0.0, 0.72)[..., None], 3, axis=2)

    # A bounded cylindrical lens plus a sub-pixel phase displacement from the
    # persistent ribs.  Flow fades at the silhouette so ROI seams cannot move.
    column_height = inside.sum(axis=0).astype(np.float32)
    body_columns = column_height >= max(3.0, float(column_height.max()) * 0.62)
    body_run = _largest_true_run(body_columns)
    if body_run is None or body_run[1] - body_run[0] < max(5, int(width * 0.20)):
        body_x0, body_x1 = 0, width
    else:
        body_x0, body_x1 = body_run
    centre_x = (body_x0 + body_x1 - 1) * 0.5
    half_width = max(2.0, (body_x1 - body_x0) * 0.5)
    _, xx = np.indices((height, width), dtype=np.float32)
    normalized_x = np.clip((xx - centre_x) / half_width, -1.0, 1.0)
    body_domain = ((xx >= body_x0) & (xx < body_x1)).astype(np.float32)
    lens = -2.6 * normalized_x * (1.0 - normalized_x * normalized_x)
    # Refraction must survive a 4K-to-browser downsample.  The former 6 px
    # ceiling became barely one display pixel for a large product and made a
    # mathematically refracted vessel read as a flat alpha veil.  Keep the
    # displacement proportional to the measured cylindrical body and bounded
    # below ten percent of its half-width.
    # Keep enough cylindrical displacement for a glass wall to read as a lens
    # rather than a transparent decal.  The amplitude is normalised by the
    # measured body width, so this applies to other clear vessels without a
    # pitcher-specific pixel constant.  The evaluator still suppresses the
    # contrast boost on a genuinely uniform target region.
    lens_amplitude = float(np.clip(half_width * 0.150, 0.95, 40.0))
    flow_response = (
        (0.60 + 0.40 * transmission_response)
        * (0.78 + 0.22 * commercial_visibility)
    )
    flow_x = lens_amplitude * lens * interior * body_domain
    flow_x += float(np.clip(half_width * 0.048, 0.34, 10.5)) * rib_map
    flow_x *= flow_response
    flow_x *= clear_fraction
    flow = np.zeros((height, width, 2), np.float32)
    flow[..., 0] = flow_x

    appendage_flow, appendage_blend, appendage_grazing, appendage_audit = (
        _tubular_appendage_optics(inside, body_x0, body_x1)
    )
    appendage_authority = coverage * clear_fraction
    appendage_blend *= appendage_authority
    appendage_flow *= (
        appendage_authority * flow_response
    )[..., None]
    flow = (
        flow * (1.0 - appendage_blend[..., None])
        + appendage_flow
    ).astype(np.float32)
    # The alpha-derived tubular cross-section also supplies a sparse grazing
    # cue.  It is deliberately bounded and conditional; alpha-over still
    # happens once in the evaluator, and enclosed holes remain exact target.
    fresnel = np.clip(
        fresnel
        + appendage_grazing
        * appendage_authority
        * (0.075 * visibility_response),
        0.0,
        0.72,
    )
    specular = np.repeat(fresnel[..., None], 3, axis=2).astype(np.float32)

    # Clear interiors transmit most of the incident light; silhouette and
    # measured ribs attenuate a little more.  Opaque identity pixels transmit
    # none.  Pixels in holes inside the rectangular ROI remain exactly one.
    conditional_shadow_transmittance = np.clip(
        0.925
        - 0.105 * (1.0 - transmission_response)
        - boundary * 0.24
        - np.abs(rib_map) * 0.075,
        0.58,
        0.95,
    ) * clear_fraction
    shadow_transmittance = (
        1.0 - coverage + coverage * conditional_shadow_transmittance
    ).astype(np.float32)
    shadow_transmittance = np.clip(shadow_transmittance, 0.0, 1.0)

    if confidence is None:
        soft_edge_ratio = float(np.mean((coverage > 0.02) & (coverage < 0.98)))
        area_score = min(1.0, float(np.count_nonzero(inside)) / 4096.0)
        rib_score = min(1.0, signal_scale / 0.035)
        confidence_value = float(np.clip(
            0.72 + 0.09 * area_score + 0.07 * rib_score - 0.16 * soft_edge_ratio,
            0.50,
            0.94,
        ))
    else:
        confidence_value = float(confidence)

    audit: dict[str, Any] = {
        "algorithm": "deterministic_clear_glass_optical_matte",
        "version": 10,
        "colour_space": "linear_srgb",
        "roi_only": True,
        "source_environment_model": "alpha_normalized_row_lowpass",
        "identity_evidence": "vertically_persistent_achromatic_residual",
        "source_rgb_reuse": "opaque_identity_mask_only",
        "fresnel_resolution": "energy_conserving_target_environment_proxy",
        "source_shape": [int(image_height), int(image_width)],
        "bbox": list(selected_bbox),
        "rib_signal_scale": signal_scale,
        "rib_phase_scale": phase_scale,
        "rib_slope_scale": rib_slope_scale,
        "topology_signal_scale": topology_scale,
        "source_environment_boundary_model": (
            "horizontal_coherence_minus_vertical_persistence"
        ),
        "source_environment_boundary_rejection_mean": float(
            np.mean(source_boundary_rejection[coverage > 0.05])
        ) if np.any(coverage > 0.05) else 0.0,
        "source_environment_boundary_rejection_p95": float(
            np.percentile(
                source_boundary_rejection[coverage > 0.05], 95.0,
            )
        ) if np.any(coverage > 0.05) else 0.0,
        "thick_base_model": "signed_compact_highpass_target_reflection",
        "catalogue_base_contrast_limiter": "target_relative_rgb_luma_v2",
        "transmission_strength": transmission_strength,
        "transmission_response": transmission_response,
        "commercial_visibility": commercial_visibility,
        "commercial_visibility_response": visibility_response,
        "tubular_appendages": appendage_audit,
    }
    if provenance is not None:
        if not isinstance(provenance, Mapping):
            raise TypeError("provenance must be a mapping")
        audit["upstream"] = dict(provenance)

    return OpticalMatte(
        coverage_alpha=coverage,
        opaque_identity_mask=opaque,
        transmission_rgb=transmission,
        refraction_flow=flow,
        specular_rgb=specular,
        shadow_transmittance=shadow_transmittance,
        bbox=selected_bbox,
        confidence=confidence_value,
        provenance=audit,
        opaque_identity_rgb=source_linear,
    )


def _target_patch_linear(
    target_rgb: np.ndarray,
    matte: OpticalMatte,
    *,
    target_is_linear: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return base, refracted and cylindrical-reflection target ROIs.

    Clear glass reflects the *target* room as well as transmitting it.  A
    narrow crop around the object is insufficient for that reflection: on a
    quiet wall it collapses to the same colour and the product reads as a
    white outline.  We therefore sample a bounded horizontal environment band
    around the ROI and compress it into the cylindrical body.  This remains
    deterministic, target-derived and ROI-local; the full 4K frame is never
    converted to float.
    """
    target = _validate_rgb_image(target_rgb, "target_rgb")
    if target_is_linear:
        if target.dtype == np.uint8:
            raise TypeError("a linear target must be floating point in [0, 1]")
    x0, y0, x1, y1 = matte.bbox
    height, width = target.shape[:2]
    if x1 > width or y1 > height:
        raise ValueError("matte bbox lies outside target_rgb")

    maximum_flow = np.max(np.abs(matte.refraction_flow), axis=(0, 1))
    roi_height, roi_width = matte.roi_shape
    refraction_pad_x = int(np.ceil(float(maximum_flow[0]))) + 2
    refraction_pad_y = int(np.ceil(float(maximum_flow[1]))) + 2
    # A front-facing cylindrical vessel reflects a much wider room solid angle
    # than its screen-space width.  Restricting this to two body widths made a
    # pitcher against a uniform wall nearly disappear.  Keep only the target
    # row neighbourhood, but allow a broad horizontal field of view.
    environment_pad_x = max(20, width)
    environment_pad_y = int(np.clip(roi_height * 0.08, 8.0, height * 0.08))
    pad_x = max(refraction_pad_x, environment_pad_x)
    pad_y = max(refraction_pad_y, environment_pad_y)
    patch_x0, patch_y0 = max(0, x0 - pad_x), max(0, y0 - pad_y)
    patch_x1, patch_y1 = min(width, x1 + pad_x), min(height, y1 + pad_y)
    target_patch = target[patch_y0:patch_y1, patch_x0:patch_x1]
    patch_linear = (
        target_patch.astype(np.float32, copy=True)
        if target_is_linear
        else _srgb_to_linear(target_patch)
    )
    base_roi = patch_linear[
        y0 - patch_y0:y1 - patch_y0,
        x0 - patch_x0:x1 - patch_x0,
    ]
    yy, xx = np.indices((roi_height, roi_width), dtype=np.float32)
    map_x = xx + float(x0 - patch_x0) + matte.refraction_flow[..., 0]
    map_y = yy + float(y0 - patch_y0) + matte.refraction_flow[..., 1]
    refracted = _remap_target_supersampled(
        patch_linear,
        map_x,
        map_y,
    )

    # Convex cylindrical reflection proxy.  Long, vertically persistent body
    # columns exclude a side handle from pulling the optical axis.  Mapping a
    # wider target band into the vessel produces scene-correlated edge/rib
    # contrast instead of copying any source-room RGB.
    inside = matte.coverage_alpha >= 0.50
    column_height = inside.sum(axis=0).astype(np.float32)
    body_columns = column_height >= max(3.0, float(column_height.max()) * 0.62)
    body_run = _largest_true_run(body_columns)
    if body_run is None or body_run[1] - body_run[0] < max(5, int(roi_width * 0.20)):
        body_x0, body_x1 = 0, roi_width
    else:
        body_x0, body_x1 = body_run
    centre_x = (body_x0 + body_x1 - 1) * 0.5
    half_width = max(2.0, (body_x1 - body_x0) * 0.5)
    normalized_x = np.clip((xx - centre_x) / half_width, -1.0, 1.0)
    # Map the full horizontal target panorama into the visible half-cylinder.
    # The photographed background cannot show the camera-side hemisphere, but
    # its row-wide radiance distribution is a safer proxy than the old room
    # trapped in source RGB.  A full-width *stripe*, not a full-frame float
    # conversion, keeps memory bounded on 4K/7K inputs.
    patch_width = patch_linear.shape[1]
    map_x_reflection = (
        1.0 - (normalized_x + 1.0) * 0.5
    ) * float(max(0, patch_width - 1))
    # Flutes are small cylindrical mirrors.  If every flute samples only the
    # broad vessel normal, a quiet wall turns all of them into the same pale
    # line.  The high-pass part of the audited refraction field carries the
    # measured local rib-normal sign, so steer those narrow faces toward
    # different sectors of the *target* panorama.  No source-room pixel is
    # sampled or copied by this operation.
    flow_x = matte.refraction_flow[..., 0].astype(np.float32)
    flow_low = cv2.GaussianBlur(
        flow_x,
        (0, 0),
        sigmaX=max(2.0, float(roi_width) * 0.018),
        sigmaY=0.75,
        borderType=cv2.BORDER_REPLICATE,
    )
    # A tight matte ROI may contain foreground in every border pixel.  OpenCV
    # then reports an effectively infinite distance because it cannot see any
    # zero-valued exterior, misclassifying the outer dielectric rim as deep
    # interior.  Supply one explicit exterior pixel before measuring depth.
    reflection_inside = cv2.copyMakeBorder(
        inside.astype(np.uint8),
        1,
        1,
        1,
        1,
        cv2.BORDER_CONSTANT,
        value=0,
    )
    reflection_distance = cv2.distanceTransform(
        reflection_inside,
        cv2.DIST_L2,
        3,
    )[1:-1, 1:-1]
    reflection_interior = np.clip(
        (reflection_distance - 1.5) / 3.0,
        0.0,
        1.0,
    )
    micro_normal = (
        (flow_x - flow_low)
        * matte.coverage_alpha
        * reflection_interior
    )
    active_micro = np.abs(micro_normal[inside])
    normal_scale = (
        float(np.percentile(active_micro, 88.0))
        if active_micro.size
        else 0.0
    )
    if normal_scale >= 0.08:
        normal_angle = np.tanh(
            micro_normal / max(0.08, normal_scale * 1.10),
        )
        map_x_reflection += (
            normal_angle * float(max(1, patch_width - 1)) * 0.38
        )
    map_x_reflection = np.clip(
        map_x_reflection,
        0.0,
        float(max(0, patch_width - 1)),
    )
    map_y_reflection = yy + float(y0 - patch_y0)
    reflected = cv2.remap(
        patch_linear,
        map_x_reflection,
        map_y_reflection,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    # A glass interface reflects the target room, but a compressed panorama
    # must not paint a saturated cyan/white outline onto a warm local wall.
    # Preserve target-derived reflected luminance while adapting most of its
    # chromaticity to the same-coordinate illuminant.  A restrained fraction
    # of panorama chroma remains so real coloured surroundings can still
    # appear in the dielectric rim.
    reflected_luma = reflected @ _LUMA
    base_luma = base_roi @ _LUMA
    reflected_neutral = base_roi * (
        (reflected_luma + 0.02) / np.maximum(base_luma + 0.02, _EPSILON)
    )[..., None]
    reflected = np.clip(
        reflected_neutral * 0.55 + reflected * 0.45,
        0.0,
        1.0,
    )
    return base_roi, refracted, reflected


def _resolve_optical_matte_components(
    matte: OpticalMatte,
    target_rgb: np.ndarray,
    *,
    target_is_linear: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return base target and uncomposited resolved product in linear RGB."""
    if not isinstance(matte, OpticalMatte):
        raise TypeError("matte must be an OpticalMatte")
    base, refracted, reflected = _target_patch_linear(
        target_rgb,
        matte,
        target_is_linear=target_is_linear,
    )
    # Treat the neutral matte value as Fresnel reflectance and conserve energy
    # when mixing it with a target-scene reflection proxy.  The previous
    # additive highlight accumulated on bright walls and looked like a pasted
    # white halo; this formulation becomes darker on a bright wall and brighter
    # on a dark tabletop, as real clear-glass interfaces commonly do.
    fresnel = np.clip(
        np.mean(matte.specular_rgb, axis=2, keepdims=True),
        0.0,
        0.72,
    )
    # Refraction is only useful if it remains visible at the final review
    # scale.  Amplify the *difference between two target-scene samples*, never
    # source RGB.  Flat target regions stay mathematically unchanged, while a
    # table rim, wall seam or highlight visibly bends through the vessel.
    flow_energy = np.linalg.norm(matte.refraction_flow, axis=2, keepdims=True)
    active_flow = flow_energy[matte.coverage_alpha[..., None] > 0.05]
    flow_scale = (
        float(np.percentile(active_flow, 90.0))
        if active_flow.size else 1.0
    )
    flow_weight = np.clip(flow_energy / max(0.50, flow_scale), 0.0, 1.0)
    refraction_delta = refracted - base
    # A smooth wall gradient is not an optical edge; globally amplifying its
    # tiny displacement paints a broad dark/bright band.  Boost only when the
    # two target samples contain enough local contrast to be a real rim, seam,
    # highlight or texture feature.
    refraction_evidence = np.clip(
        (np.mean(np.abs(refraction_delta), axis=2, keepdims=True) - 0.012)
        / 0.055,
        0.0,
        1.0,
    )
    raw_refraction_gain = (
        1.0 + (0.52 + 0.68 * flow_weight) * refraction_evidence
    )
    # A coherent receiver edge should be displaced by the glass, not converted
    # into a black, over-sharpened staircase.  Measure gradients in the two
    # target-derived samples and continuously lower only the contrast boost at
    # strong edges.  The refraction flow itself remains unchanged, so a real
    # tabletop boundary still bends through the vessel.
    base_luma_for_edge = base @ _LUMA
    refracted_luma_for_edge = refracted @ _LUMA

    def receiver_gradient(luma: np.ndarray) -> np.ndarray:
        gx = cv2.Sobel(
            luma,
            cv2.CV_32F,
            1,
            0,
            ksize=3,
            scale=0.125,
            borderType=cv2.BORDER_REPLICATE,
        )
        gy = cv2.Sobel(
            luma,
            cv2.CV_32F,
            0,
            1,
            ksize=3,
            scale=0.125,
            borderType=cv2.BORDER_REPLICATE,
        )
        return np.hypot(gx, gy)

    coherent_receiver_edge = cv2.GaussianBlur(
        np.maximum(
            receiver_gradient(base_luma_for_edge),
            receiver_gradient(refracted_luma_for_edge),
        ),
        (0, 0),
        sigmaX=0.85,
        sigmaY=0.85,
        borderType=cv2.BORDER_REPLICATE,
    )
    receiver_edge_gate = np.clip(
        (coherent_receiver_edge - 0.025) / 0.105,
        0.0,
        1.0,
    )[..., None]
    edge_aware_gain_cap = 2.20 - 0.85 * receiver_edge_gate
    refraction_gain = np.minimum(
        raw_refraction_gain,
        edge_aware_gain_cap,
    )
    refracted = np.clip(
        base + refraction_delta * refraction_gain,
        0.0,
        1.0,
    )
    target_peak = np.max(base, axis=2, keepdims=True)
    target_illuminant = np.clip(
        (base + 0.04) / np.maximum(target_peak + 0.04, _EPSILON),
        0.0,
        1.0,
    )
    # A glass interface does not emit light.  Keep only a very small baseline
    # glint plus a sparse nonlinear caustic at the strongest measured grazing
    # interfaces.  The former 0.320*fresnel term added warm radiance over every
    # rib and contour, producing a body-wide beige veil on warm walls.  Most
    # visible energy now comes from the target-environment reflection below.
    caustic_gate = np.power(
        np.clip((fresnel - 0.46) / 0.26, 0.0, 1.0),
        2.0,
    )
    # The linear baseline remains deliberately tiny across rib faces.  Strong
    # energy is reserved for the nonlinear grazing gate, which is spatially
    # confined to the measured outer rim/handle and therefore survives preview
    # resampling without filling the transparent body.
    # Preserve a sparse catalogue-scale glint after a 7K image is fitted to a
    # browser canvas.  The increase is deliberately small and remains gated by
    # the measured Fresnel field; broad clear interiors still receive almost
    # no additive energy and cannot become a white/milky body fill.
    # Raise only the nonlinear grazing-interface glint.  A stronger linear
    # term would brighten the complete body and recreate the flat white veil;
    # this formulation adds catalogue polish to measured rims, handles, rib
    # crests and the thick base while leaving the broad clear interior neutral.
    micro_energy = fresnel * 0.012 + caustic_gate * 0.285
    micro_highlight = micro_energy * (0.30 + 0.70 * target_illuminant)
    transmitted = (
        refracted * matte.transmission_rgb * (1.0 - fresnel)
        + reflected * fresnel
        + micro_highlight
    )
    transmitted = np.clip(transmitted, 0.0, 1.0)

    # Catalogue glass should remain optically tied to the target receiver, but
    # a dark mounting plate sampled by the cylindrical flow can otherwise turn
    # into a high-contrast black triangle inside the thick base.  That is a
    # plausible ray sample and still a poor product photograph.  Constrain only
    # the lowest fifth of the clear silhouette relative to the *target* luma:
    # signed rib transmission, rim/handle Fresnel and compact base highlights
    # remain intact, while extreme black/white connected structures are kept
    # within a restrained commercial-photography contrast envelope.
    coverage = matte.coverage_alpha
    visible_rows = np.flatnonzero(np.any(coverage > 0.04, axis=1))
    if visible_rows.size >= 3:
        top_row = int(visible_rows[0])
        bottom_row = int(visible_rows[-1])
        yy = np.arange(coverage.shape[0], dtype=np.float32)[:, None]
        vertical = np.clip(
            (yy - float(top_row)) / max(1.0, float(bottom_row - top_row)),
            0.0,
            1.0,
        )
        base_gate = np.clip((vertical - 0.69) / 0.17, 0.0, 1.0)
        base_gate = base_gate * base_gate * (3.0 - 2.0 * base_gate)
        clear_coverage = np.clip(
            coverage - matte.opaque_identity_mask,
            0.0,
            1.0,
        )
        base_luma = (base @ _LUMA)[..., None]
        base_region = (base_gate > 0.18) & (clear_coverage > 0.20)
        if np.count_nonzero(base_region) >= 24:
            receiver_values = base_luma[..., 0][base_region]
            receiver_span = float(
                np.percentile(receiver_values, 92.0)
                - np.percentile(receiver_values, 8.0)
            )
        else:
            receiver_span = 0.0
        # A quiet wall or uniformly coloured tabletop cannot create a severe
        # displaced structure, so leave its carefully balanced rib signal
        # untouched.  Activate only where the target receiver itself contains
        # a strong rim, bracket, fastener or highlight transition.
        structure_gate = float(np.clip(
            (receiver_span - 0.055) / 0.145,
            0.0,
            1.0,
        ))
        base_gate = (
            base_gate[..., None]
            * clear_coverage[..., None]
            * structure_gate
        )
        # Bound each channel relative to the receiver as well as luma.  A
        # luma-only clamp removed the black wedge but could leave a blue/cyan
        # metal shard with the same brightness.  The asymmetric RGB envelope
        # preserves small warm highlights while suppressing foreign hardware
        # colour that does not read as clean catalogue glass.
        rgb_delta = transmitted - base
        bounded = np.clip(
            base + np.clip(rgb_delta, -0.018, 0.070),
            0.0,
            1.0,
        )
        bounded_luma = (bounded @ _LUMA)[..., None]
        bounded_luma_delta = np.clip(
            bounded_luma - base_luma,
            -0.018,
            0.065,
        )
        bounded = np.clip(
            bounded + (bounded_luma_delta - (bounded_luma - base_luma)),
            0.0,
            1.0,
        )
        transmitted = (
            transmitted * (1.0 - base_gate)
            + bounded * base_gate
        )
        transmitted = np.clip(transmitted, 0.0, 1.0)

    opaque_fraction = (
        matte.opaque_identity_mask
        / np.maximum(matte.coverage_alpha, _EPSILON)
    )[..., None]
    if matte.opaque_identity_rgb is None:
        resolved = transmitted
    else:
        resolved = (
            transmitted * (1.0 - opaque_fraction)
            + matte.opaque_identity_rgb * opaque_fraction
        )
    return (
        np.ascontiguousarray(base, dtype=np.float32),
        np.ascontiguousarray(np.clip(resolved, 0.0, 1.0), dtype=np.float32),
    )


def resolve_optical_matte_roi_linear(
    matte: OpticalMatte,
    target_rgb: np.ndarray,
    *,
    target_is_linear: bool = False,
) -> np.ndarray:
    """Resolve the uncomposited product colour for one ROI in linear RGB."""
    _, resolved = _resolve_optical_matte_components(
        matte, target_rgb, target_is_linear=target_is_linear,
    )
    return resolved


def resolve_optical_matte_roi(
    matte: OpticalMatte,
    target_rgb: np.ndarray,
    *,
    target_is_linear: bool = False,
) -> np.ndarray:
    """Resolve uncomposited ROI colour using the target's public encoding."""
    target = _validate_rgb_image(target_rgb, "target_rgb")
    resolved = resolve_optical_matte_roi_linear(
        matte, target, target_is_linear=target_is_linear,
    )
    if target_is_linear:
        return resolved
    encoded = _linear_to_srgb(resolved)
    if target.dtype == np.uint8:
        return np.clip(np.round(encoded * 255.0), 0, 255).astype(np.uint8)
    return encoded.astype(target.dtype, copy=False)


def evaluate_optical_matte_roi_linear(
    matte: OpticalMatte,
    target_rgb: np.ndarray,
    *,
    target_is_linear: bool = False,
) -> np.ndarray:
    """Resolve and composite one matte ROI against a target in linear RGB.

    The returned array has ``matte.roi_shape + (3,)`` and is always float32
    linear sRGB.  Only the target patch reachable by ``refraction_flow`` is
    converted, keeping peak memory proportional to the product rather than the
    4K canvas.
    """
    base, resolved = _resolve_optical_matte_components(
        matte, target_rgb, target_is_linear=target_is_linear,
    )
    coverage = matte.coverage_alpha[..., None]
    composed = base * (1.0 - coverage) + resolved * coverage
    # Exact zeros in coverage must remain target-derived even if a future matte
    # contains arbitrary values in unused rectangular corners.
    composed[matte.coverage_alpha <= 0.0] = base[matte.coverage_alpha <= 0.0]
    return np.ascontiguousarray(np.clip(composed, 0.0, 1.0), dtype=np.float32)


def evaluate_optical_matte(
    matte: OpticalMatte,
    target_rgb: np.ndarray,
    *,
    target_is_linear: bool = False,
) -> np.ndarray:
    """Composite a matte into a full target while preserving pixels outside it.

    For ordinary uint8/sRGB input the result is uint8/sRGB.  For normalized
    floating sRGB input the result is normalized floating sRGB.  If
    ``target_is_linear`` is true, input and output are normalized float32
    linear RGB.  In every case, pixels outside non-zero product coverage are
    copied from ``target_rgb`` without a colour-space round trip.
    """
    target = _validate_rgb_image(target_rgb, "target_rgb")
    linear_roi = evaluate_optical_matte_roi_linear(
        matte,
        target,
        target_is_linear=target_is_linear,
    )
    x0, y0, x1, y1 = matte.bbox
    active = matte.coverage_alpha > 0.0
    if target_is_linear:
        output = target.astype(np.float32, copy=True)
        output_roi = output[y0:y1, x0:x1]
        output_roi[active] = linear_roi[active]
        return output

    srgb_roi = _linear_to_srgb(linear_roi)
    if target.dtype == np.uint8:
        encoded_roi = np.clip(np.round(srgb_roi * 255.0), 0, 255).astype(np.uint8)
        output = target.copy()
    else:
        encoded_roi = srgb_roi.astype(target.dtype, copy=False)
        output = target.copy()
    output_roi = output[y0:y1, x0:x1]
    output_roi[active] = encoded_roi[active]
    return output


__all__ = [
    "BBox",
    "OpticalMatte",
    "estimate_clear_glass_matte",
    "evaluate_optical_matte",
    "evaluate_optical_matte_roi_linear",
    "resolve_optical_matte_roi",
    "resolve_optical_matte_roi_linear",
]
