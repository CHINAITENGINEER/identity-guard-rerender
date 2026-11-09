from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .image_io import load_background, save_png


SUPPORT_LABELS = {"table", "desk", "counter", "countertop", "shelf"}
STRUCTURAL_LABELS = {
    "wall", "floor", "ceiling", "windowpane", "pane",
    "table", "desk", "counter", "countertop",
    "shelf", "cabinet", "door", "window", "chair", "seat", "sofa", "bed",
    "bench", "ottoman", "coffee table", "chest of drawers", "bookcase",
    "nightstand", "sideboard",
}
_SEGMENTER: Any | None = None
_SEGMENTER_LOCK = threading.Lock()
_SEGMENTER_INFERENCE_LOCK = threading.Lock()
_DEPTH_ESTIMATOR: Any | None = None
_DEPTH_LOCK = threading.Lock()
_DEPTH_INFERENCE_LOCK = threading.Lock()


def _quiet_model_loading() -> None:
    """Keep Hugging Face model loading independent from the host console.

    The desktop launcher may detach stderr after the web server starts.  tqdm
    then raises OSError while merely trying to redraw a progress bar, which
    used to make scene analysis silently fall back to the unsafe heuristic.
    """
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
    except Exception:
        pass
    try:
        from transformers.utils import logging as transformers_logging

        transformers_logging.disable_progress_bar()
        transformers_logging.set_verbosity_error()
    except Exception:
        pass


@dataclass(frozen=True)
class SceneAnalysis:
    support_polygon: list[list[float]]
    protected_polygons: list[list[list[float]]]
    support_mask_path: str
    front_edge_y: float
    reflectivity: float
    reflection_recommended: bool
    method: str
    confidence: float
    shadow_direction: list[float]
    shadow_length_ratio: float
    shadow_opacity: float
    shadow_softness_ratio: float
    surface_material: str = "unknown"
    surface_roughness: float = 0.55
    surface_transmittance: float = 0.0
    cast_shadow_mode: str = "directional"
    direct_light_fraction: float = 0.55
    shadow_confidence: float = 0.0
    reflection_confidence: float = 0.0
    contact_ao_strength: float = 1.0
    depth_map_path: str | None = None
    normal_map_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _file_key(path: Path) -> str:
    stat = path.stat()
    raw = f"scene-v28-geometry-evidence|{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def _largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 1:
        return np.zeros_like(mask, np.uint8)
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == index).astype(np.uint8)


def _dominant_receiver_instance(mask: np.ndarray) -> np.ndarray:
    """Separate substantial receiver lobes joined only by a narrow bridge.

    Semantic and monocular-depth masks occasionally join two white tabletops
    where their projected edges touch.  ``_clean_surface`` used to close that
    neck before selecting a component, turning the two physical receivers into
    one large placement surface.  A scale-adaptive opening exposes substantial
    surface cores on either side of such a neck.  When (and only when) one
    original connected component contains at least two broad cores, retain the
    best lower-central, area-dominant core and recover its eroded boundary from
    the original mask without flood-filling through the bridge.

    Disconnected masks are deliberately left alone: two pieces of one tabletop
    can be separated by a foreground occluder and the later conservative close
    is responsible for reconnecting those pieces.  Small opened components are
    also ignored so a table leg, bevel or thin decorative tail cannot make this
    routine discard an otherwise valid receiver.
    """

    original = (mask > 0).astype(np.uint8)
    height, width = original.shape
    foreground = int(np.count_nonzero(original))
    if foreground < max(96, int(height * width * 0.003)):
        return original

    original_count, _, _, _ = cv2.connectedComponentsWithStats(original, 8)
    if original_count != 2:
        return original

    # Analysis runs at no more than 1400 px.  About 1.4% of the short side is
    # wide enough to sever a semantic anti-aliased neck, while still far below
    # the breadth of a usable product-bearing receiver core.
    radius = int(np.clip(round(min(height, width) * 0.014), 2, 10))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (radius * 2 + 1, radius * 2 + 1),
    )
    opened = cv2.morphologyEx(original, cv2.MORPH_OPEN, kernel)
    opened_area = int(np.count_nonzero(opened))
    # If opening removes a large share, the input is a genuinely thin or
    # intricate receiver rather than two broad surfaces with a small neck.
    if opened_area < foreground * 0.86:
        return original

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(opened, 8)
    minimum_core_area = max(
        64,
        int(foreground * 0.035),
        int(height * width * 0.0015),
    )
    minimum_core_width = max(12, int(round(width * 0.055)))
    minimum_core_height = max(8, int(round(height * 0.025)))
    candidates: list[tuple[float, int]] = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        component_width = int(stats[index, cv2.CC_STAT_WIDTH])
        component_height = int(stats[index, cv2.CC_STAT_HEIGHT])
        if (
            area < minimum_core_area
            or component_width < minimum_core_width
            or component_height < minimum_core_height
        ):
            continue
        centre_x = float(centroids[index, 0]) / max(1, width - 1)
        centre_y = float(centroids[index, 1]) / max(1, height - 1)
        centrality = float(np.clip(1.0 - abs(centre_x - 0.5) / 0.5, 0.0, 1.0))
        lower_middle = float(np.clip(1.0 - abs(centre_y - 0.60) / 0.60, 0.0, 1.0))
        broadness = float(np.clip(component_width / max(1.0, width * 0.55), 0.0, 1.0))
        score = float(area) * (
            1.0 + 0.28 * centrality + 0.16 * lower_middle + 0.08 * broadness
        )
        candidates.append((score, index))

    if len(candidates) < 2:
        return original

    candidates.sort(reverse=True)
    chosen_index = candidates[0][1]
    chosen_core = (labels == chosen_index).astype(np.uint8)
    # One dilation restores the boundary removed by the opening.  Intersecting
    # with the original mask recovers real receiver pixels, while avoiding a
    # geodesic flood that would simply cross the same narrow bridge again.
    recovery_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (radius * 2 + 3, radius * 2 + 3),
    )
    recovered = original & cv2.dilate(chosen_core, recovery_kernel)
    if int(np.count_nonzero(recovered)) < int(np.count_nonzero(chosen_core)):
        return original
    return recovered.astype(np.uint8)


def _polygon_from_mask(mask: np.ndarray, max_points: int = 18) -> list[list[float]]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No support surface contour was found")
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    epsilon = 0.006 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    while len(approx) > max_points:
        epsilon *= 1.35
        approx = cv2.approxPolyDP(contour, epsilon, True)
    height, width = mask.shape
    points = approx[:, 0, :].astype(np.float64)
    # OpenCV contours may start anywhere.  Normalized coordinates are all the
    # compositor requires, but a stable clockwise order keeps audit JSON clean.
    if cv2.contourArea(approx, oriented=True) < 0:
        points = points[::-1]
    return [[float(x / max(1, width - 1)), float(y / max(1, height - 1))] for x, y in points]


def _semantic_segment(image_rgb: np.ndarray) -> list[dict[str, Any]]:
    global _SEGMENTER
    with _SEGMENTER_LOCK:
        if _SEGMENTER is None:
            _quiet_model_loading()
            from transformers import pipeline
            _SEGMENTER = pipeline(
                "image-segmentation",
                model="nvidia/segformer-b0-finetuned-ade-512-512",
                device=-1,
            )
    from PIL import Image
    # Transformers pipelines are not guaranteed to be re-entrant.  The web
    # UI may fire scene analysis and product refinement at nearly the same
    # time, so serialize inference while still allowing request validation
    # and deterministic post-processing to run concurrently.
    with _SEGMENTER_INFERENCE_LOCK:
        return _SEGMENTER(Image.fromarray(image_rgb))


def _normalize_depth_map(depth: np.ndarray) -> np.ndarray:
    """Return a finite float32 relative-depth map in the closed [0, 1] range."""
    value = np.asarray(depth, dtype=np.float32).squeeze()
    if value.ndim != 2 or not value.size:
        raise ValueError("predicted depth must be a non-empty HxW array")
    finite = np.isfinite(value)
    if not np.any(finite):
        raise ValueError("predicted depth contains no finite samples")
    replacement = float(np.median(value[finite]))
    value = np.where(finite, value, replacement).astype(np.float32, copy=False)
    low, high = np.percentile(value, [1, 99])
    if float(high - low) <= 1e-6:
        return np.zeros_like(value, dtype=np.float32)
    return np.clip((value - low) / float(high - low), 0.0, 1.0).astype(
        np.float32,
        copy=False,
    )


def _unitize_normal_map(normal: np.ndarray) -> np.ndarray:
    """Normalize every HxWx3 vector, using +Z for degenerate samples."""
    value = np.asarray(normal, dtype=np.float32)
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError("normal map must have shape HxWx3")
    value = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    length = np.linalg.norm(value, axis=2, keepdims=True)
    degenerate = length[..., 0] <= 1e-8
    result = value / np.maximum(length, 1e-8)
    if np.any(degenerate):
        result[degenerate] = np.asarray([0.0, 0.0, 1.0], np.float32)
    return result.astype(np.float32, copy=False)


def _horizontal_from_depth(
    image_rgb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate upward-facing planes from a monocular depth point cloud."""
    global _DEPTH_ESTIMATOR
    with _DEPTH_LOCK:
        if _DEPTH_ESTIMATOR is None:
            _quiet_model_loading()
            from transformers import pipeline
            _DEPTH_ESTIMATOR = pipeline(
                "depth-estimation",
                model="depth-anything/Depth-Anything-V2-Small-hf",
                device=-1,
            )
    from PIL import Image

    height, width = image_rgb.shape[:2]
    scale = min(1.0, 448.0 / max(height, width))
    inference = cv2.resize(
        image_rgb,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    ) if scale < 1.0 else image_rgb
    with _DEPTH_INFERENCE_LOCK:
        result = _DEPTH_ESTIMATOR(Image.fromarray(inference))
    predicted = result["predicted_depth"]
    depth = predicted.detach().cpu().numpy() if hasattr(predicted, "detach") else np.asarray(predicted)
    depth = cv2.GaussianBlur(depth.astype(np.float32), (0, 0), 1.15)
    z = _normalize_depth_map(depth)
    h, w = z.shape
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    focal = max(1.0, w * 0.9)
    cloud_x = (xx - w * 0.5) * z / focal
    cloud_y = (yy - h * 0.5) * z / focal
    tangent_x = np.stack(
        [np.gradient(cloud_x, axis=1), np.gradient(cloud_y, axis=1), np.gradient(z, axis=1)],
        axis=2,
    )
    tangent_y = np.stack(
        [np.gradient(cloud_x, axis=0), np.gradient(cloud_y, axis=0), np.gradient(z, axis=0)],
        axis=2,
    )
    normal = _unitize_normal_map(np.cross(tangent_x, tangent_y))
    nx, ny, nz = np.abs(normal[..., 0]), np.abs(normal[..., 1]), np.abs(normal[..., 2])
    horizontal = (ny > 0.36) & (ny > nz * 1.08) & (ny > nx * 1.15)
    horizontal = cv2.morphologyEx(
        horizontal.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8),
    )
    horizontal = cv2.morphologyEx(
        horizontal, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8),
    )
    depth_full = cv2.resize(z, (width, height), interpolation=cv2.INTER_LINEAR)
    normal_full = cv2.resize(normal, (width, height), interpolation=cv2.INTER_LINEAR)
    return (
        cv2.resize(horizontal, (width, height), interpolation=cv2.INTER_NEAREST),
        np.clip(depth_full, 0.0, 1.0).astype(np.float32, copy=False),
        _unitize_normal_map(normal_full),
    )


def _semantic_support(image_rgb: np.ndarray) -> tuple[np.ndarray, list[tuple[str, np.ndarray]]]:
    height, width = image_rgb.shape[:2]
    scale = min(1.0, 896.0 / max(height, width))
    if scale < 1.0:
        inference_rgb = cv2.resize(
            image_rgb,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        inference_rgb = image_rgb
    entries = _semantic_segment(inference_rgb)
    masks: list[tuple[str, np.ndarray]] = []
    support = np.zeros((height, width), np.uint8)
    for entry in entries:
        label = str(entry["label"]).strip().lower()
        raw = np.asarray(entry["mask"].resize((width, height)))
        mask = (raw > 0).astype(np.uint8)
        masks.append((label, mask))
        if label in SUPPORT_LABELS:
            support |= mask
    return support, masks


def _semantic_seed_needs_central_recovery(support: np.ndarray) -> bool:
    """Detect a likely vertical cabinet mislabeled as the only table.

    ADE20K occasionally calls a large tabletop "ceiling" while labeling the
    cabinet below it as "table".  Such a seed is both small and concentrated
    very low in the frame.  It must not be allowed to pull the depth fit away
    from the dominant centre plane.
    """
    ys, _ = np.nonzero(support)
    if not len(ys):
        return True
    return bool(
        float(support.mean()) < 0.08
        and float(np.median(ys) / max(1, support.shape[0] - 1)) > 0.68
    )


def _central_plane_seed(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    seed = np.zeros((height, width), np.uint8)
    seed[
        int(round(height * 0.18)):int(round(height * 0.72)),
        int(round(width * 0.12)):int(round(width * 0.90)),
    ] = 1
    return seed


def _clip_recovered_plane_at_front_edge(
    image_rgb: np.ndarray,
    plane: np.ndarray,
) -> np.ndarray:
    """Stop a recovered centre plane at a strong, broad table front edge."""
    height, width = plane.shape
    ys = np.flatnonzero(plane.any(axis=1))
    if not len(ys):
        return plane
    bottom = int(ys.max())
    search_top = max(int(height * 0.45), int(ys.min()))
    search_bottom = min(bottom, int(height * 0.82))
    if search_bottom <= search_top + 4:
        return plane
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    vertical_gradient = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    scores: list[tuple[float, int]] = []
    for row in range(search_top, search_bottom + 1):
        columns = plane[row].astype(bool)
        if np.count_nonzero(columns) < width * 0.45:
            continue
        values = vertical_gradient[row, columns]
        scores.append((float(values.mean()), row))
    if not scores:
        return plane
    strength, edge_row = max(scores)
    # Only a clearly stronger-than-normal horizontal boundary may clip the
    # plane; textured surfaces must not be truncated by an arbitrary streak.
    reference = float(np.mean(vertical_gradient[plane.astype(bool)]))
    if strength < max(24.0, reference * 2.15):
        return plane
    clipped = plane.copy()
    clipped[edge_row + 1:] = 0
    return _largest_component(clipped)


def _grow_support_plane(horizontal: np.ndarray, furniture: np.ndarray) -> np.ndarray:
    """Recover the full plane from semantic/depth agreement seeds.

    SegFormer often labels the vertical body of a table but leaves a bright,
    textureless tabletop as an interior hole.  Depth normals find that top
    correctly.  Intersecting both predictions therefore retains only a thin
    rim.  Instead, agreement is used as a seed and the complete connected
    normal-plane component is retained.  Unrelated horizontal regions (floor,
    sofa seats, shelves) are not connected to that seed and remain excluded.
    """
    height, width = horizontal.shape
    # First break narrow bridges where two different planes merely touch in
    # the 2-D projection (e.g. a coffee table in front of a sofa seat).
    normal_mask = cv2.morphologyEx(
        horizontal.astype(np.uint8),
        cv2.MORPH_OPEN,
        np.ones((9, 9), np.uint8),
    )
    normal_mask = cv2.morphologyEx(
        normal_mask,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), np.uint8),
    )
    seed_zone = cv2.dilate(furniture.astype(np.uint8), np.ones((15, 15), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(normal_mask, 8)
    candidates: list[tuple[float, np.ndarray]] = []
    frame_area = float(height * width)
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < frame_area * 0.004:
            continue
        component = labels == index
        overlap = int(np.count_nonzero(component & seed_zone.astype(bool)))
        if overlap < max(40, int(area * 0.018)):
            continue
        y0 = int(stats[index, cv2.CC_STAT_TOP])
        component_height = int(stats[index, cv2.CC_STAT_HEIGHT])
        centre_y = (y0 + component_height * 0.5) / max(1, height)
        # Prefer broad, lower-middle support surfaces; the overlap term ties
        # the component to a table/cabinet semantic prediction.
        score = overlap * 3.0 + area * (1.0 + min(0.35, max(0.0, centre_y - 0.25)))
        candidates.append((score, component.astype(np.uint8)))
    if not candidates:
        return furniture & cv2.dilate(normal_mask, np.ones((11, 11), np.uint8))
    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, best = candidates[0]
    # An occluder can split one tabletop into two pieces.  Merge only strongly
    # competitive components in the same vertical band.
    best_ys = np.flatnonzero(best.any(axis=1))
    output = best.copy()
    for score, component in candidates[1:]:
        if score < best_score * 0.34:
            continue
        ys = np.flatnonzero(component.any(axis=1))
        if not len(ys) or not len(best_ys):
            continue
        vertical_gap = max(0, int(ys.min()) - int(best_ys.max()), int(best_ys.min()) - int(ys.max()))
        if vertical_gap <= max(12, int(height * 0.035)):
            output |= component
    agreement = output.astype(bool) & furniture.astype(bool)
    agreement_rows = np.flatnonzero(agreement.any(axis=1))
    if len(agreement_rows):
        agreement_span = int(agreement_rows.max() - agreement_rows.min() + 1)
        # Do not let a connected, farther horizontal plane (typically a sofa
        # seat behind a coffee table) pull the support polygon upward.
        top_limit = int(agreement_rows.min()) - max(
            int(height * 0.075),
            int(agreement_span * 1.9),
        )
        output[:max(0, top_limit)] = 0
    return output


def _fit_support_depth_plane(
    horizontal: np.ndarray,
    furniture: np.ndarray,
    relative_depth: np.ndarray,
) -> np.ndarray:
    """Separate touching horizontal surfaces with a robust depth-plane fit."""
    height, width = horizontal.shape
    seed_zone = cv2.dilate(furniture.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    candidate = horizontal.astype(bool) & seed_zone
    ys, xs = np.nonzero(candidate)
    if len(xs) < max(180, int(height * width * 0.003)):
        return _grow_support_plane(horizontal, furniture)

    # A fixed sampling grid makes analysis repeatable across runs and machines.
    stride = max(1, int(np.ceil(len(xs) / 6500)))
    xs_sample = xs[::stride]
    ys_sample = ys[::stride]
    x = xs_sample.astype(np.float64) / max(1, width - 1)
    y = ys_sample.astype(np.float64) / max(1, height - 1)
    z = relative_depth[ys_sample, xs_sample].astype(np.float64)
    design = np.column_stack([x, y, np.ones_like(x)])
    rng = np.random.default_rng(20260730)
    best: tuple[float, np.ndarray] | None = None
    threshold = 0.045
    for _ in range(128):
        chosen = rng.choice(len(x), 3, replace=False)
        matrix = design[chosen]
        if abs(np.linalg.det(matrix)) < 2e-4:
            continue
        coefficients = np.linalg.solve(matrix, z[chosen])
        residual = np.abs(design @ coefficients - z)
        inliers = residual < threshold
        if int(inliers.sum()) < max(90, int(len(x) * 0.055)):
            continue
        # Foreground/lower planes are more likely to be the user-facing
        # tabletop when a sofa seat or distant shelf touches it in 2-D.
        weights = 1.0 + 0.72 * z[inliers] + 0.26 * y[inliers]
        score = float(weights.sum())
        if best is None or score > best[0]:
            best = (score, inliers)
    if best is None:
        return _grow_support_plane(horizontal, furniture)

    inliers = best[1]
    coefficients, *_ = np.linalg.lstsq(design[inliers], z[inliers], rcond=None)
    for _ in range(2):
        residual = np.abs(design @ coefficients - z)
        inliers = residual < threshold
        coefficients, *_ = np.linalg.lstsq(design[inliers], z[inliers], rcond=None)

    full_y, full_x = np.mgrid[0:height, 0:width]
    predicted = (
        coefficients[0] * full_x / max(1, width - 1)
        + coefficients[1] * full_y / max(1, height - 1)
        + coefficients[2]
    )
    residual_full = np.abs(predicted - relative_depth)
    # Monocular depth is intentionally smooth and may be slightly curved over
    # a textureless glossy top.  A wider reconstruction band recovers the full
    # surface after the stricter RANSAC fit; genuinely separate rear planes
    # remain far outside this band.
    plane = (
        horizontal.astype(bool)
        & (residual_full < 0.14)
    ).astype(np.uint8)
    plane = cv2.morphologyEx(plane, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    plane = cv2.morphologyEx(plane, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    # Keep only plane components that contain fitted semantic agreement.
    seed = np.zeros_like(plane, np.uint8)
    seed[ys_sample[inliers], xs_sample[inliers]] = 1
    seed = cv2.dilate(seed, np.ones((7, 7), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(plane, 8)
    output = np.zeros_like(plane, np.uint8)
    ranked: list[tuple[int, int]] = []
    for index in range(1, count):
        component = labels == index
        overlap = int(np.count_nonzero(component & seed.astype(bool)))
        if overlap:
            ranked.append((overlap, index))
    # One fitted physical plane is the support.  Keeping additional connected
    # components allowed a large floor patch to win later merely because it
    # was bigger than the actual tabletop.
    for _, index in sorted(ranked, reverse=True)[:1]:
        output[labels == index] = 1
    return output if output.any() else _grow_support_plane(horizontal, furniture)


def _classical_support(image_rgb: np.ndarray) -> np.ndarray:
    """Fast fallback for a dominant tabletop around the image centre."""
    height, width = image_rgb.shape[:2]
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    seeds = [
        (int(width * 0.50), int(height * 0.55)),
        (int(width * 0.50), int(height * 0.68)),
        (int(width * 0.35), int(height * 0.58)),
        (int(width * 0.65), int(height * 0.58)),
    ]
    best = np.zeros((height, width), np.uint8)
    for x, y in seeds:
        seed = lab[max(0, y - 8): y + 9, max(0, x - 8): x + 9]
        if seed.size == 0:
            continue
        colour = np.median(seed.reshape(-1, 3), axis=0)
        delta = np.linalg.norm(lab.astype(np.float32) - colour.astype(np.float32), axis=2)
        candidate = (delta < 30).astype(np.uint8)
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((17, 17), np.uint8))
        candidate = _largest_component(candidate)
        if candidate.sum() > best.sum():
            best = candidate
    return best


def _support_geometry_is_suspicious(mask: np.ndarray) -> bool:
    """Reject frame-dominating colour regions before they become tabletops.

    The fast colour fallback used to reward area alone.  In a transparent-glass
    scene the wall visible through the glass therefore became a 97%-confidence
    "table".  A real close-up tabletop may legitimately touch the lower and
    side borders, but a region that also owns much of the top border or spans
    almost the entire frame is not safe receiver geometry.
    """
    candidate = mask.astype(bool)
    if not candidate.any():
        return True
    height, width = candidate.shape
    area_ratio = float(candidate.mean())
    top = float(candidate[0].mean())
    bottom = float(candidate[-1].mean())
    left = float(candidate[:, 0].mean())
    right = float(candidate[:, -1].mean())
    ys, xs = np.nonzero(candidate)
    span_y = float((ys.max() - ys.min() + 1) / max(1, height))
    span_x = float((xs.max() - xs.min() + 1) / max(1, width))
    if area_ratio > 0.55 and (top > 0.20 or (top > 0.05 and bottom > 0.20)):
        return True
    if area_ratio > 0.42 and span_y > 0.94 and span_x > 0.80:
        return True
    if top > 0.45 and bottom > 0.45:
        return True
    if area_ratio > 0.58 and (left > 0.55 or right > 0.55):
        return True
    return False


def _ellipse_support_structure_score(
    gray: np.ndarray,
    blurred: np.ndarray,
    ellipse_mask: np.ndarray,
    centre_x: float,
    major_axis: float,
    minor_axis: float,
) -> float:
    """Measure table-leg/pedestal evidence immediately below an ellipse.

    Round cushions, rugs and chair backs also produce excellent ellipse fits.
    A horizontal glass receiver, however, has a pedestal or legs continuing
    below its front rim.  This weak structural cue is used only to validate the
    special transparent-surface fallback; it never defines final pixels.
    """
    height, width = gray.shape
    rows = np.flatnonzero(ellipse_mask.any(axis=1))
    if not len(rows):
        return 0.0
    bottom = int(rows.max())
    left = max(0, int(round(centre_x - major_axis * 0.18)))
    right = min(width, int(round(centre_x + major_axis * 0.18)))
    top = min(height, bottom + 2)
    lower = min(height, int(round(bottom + max(10.0, minor_axis * 0.65))))
    if right <= left + 3 or lower <= top + 3:
        return 0.0
    strip = gray[top:lower, left:right]
    if strip.size < 64:
        return 0.0
    dark_threshold = float(np.percentile(gray, 30))
    dark_fraction = float(np.mean(strip < dark_threshold))
    gradient_x = np.abs(cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3))
    strong_threshold = float(np.percentile(gradient_x, 85))
    vertical_edge_fraction = float(
        np.mean(gradient_x[top:lower, left:right] > strong_threshold)
    )
    return float(np.clip(max(dark_fraction, vertical_edge_fraction), 0.0, 1.0))


def _elliptical_glass_support(image_rgb: np.ndarray) -> tuple[np.ndarray, float]:
    """Recover a round/oval transparent tabletop from its visible rim.

    Monocular depth and semantic segmentation often see *through* glass.  This
    detector therefore uses only deterministic image evidence: an approximately
    horizontal ellipse, sustained rim gradients, plausible receiver geometry,
    and a pedestal/leg cue below the rim.  It is deliberately conservative and
    is invoked only when the normal support pipeline is missing or suspicious.
    """
    original_height, original_width = image_rgb.shape[:2]
    scale = min(1.0, 1000.0 / max(original_height, original_width))
    if scale < 1.0:
        inference = cv2.resize(
            image_rgb,
            (
                max(1, int(round(original_width * scale))),
                max(1, int(round(original_height * scale))),
            ),
            interpolation=cv2.INTER_AREA,
        )
    else:
        inference = image_rgb
    height, width = inference.shape[:2]
    gray = cv2.cvtColor(inference, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.2)
    median = float(np.median(blurred))
    low = max(18, int(round(median * 0.12)))
    high = max(low + 32, int(round(median * 0.42)))
    edges = cv2.Canny(blurred, low, high)
    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    distance = cv2.distanceTransform(
        (edges == 0).astype(np.uint8), cv2.DIST_L2, 3,
    )
    contours, _ = cv2.findContours(
        edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE,
    )
    candidates: list[tuple[float, np.ndarray]] = []
    frame_area = float(height * width)
    for contour in contours:
        if len(contour) < 45:
            continue
        (centre_x, centre_y), (axis_a, axis_b), angle = cv2.fitEllipse(contour)
        major = float(max(axis_a, axis_b))
        minor = float(min(axis_a, axis_b))
        aspect = minor / max(1.0, major)
        major_angle = float(angle if axis_a >= axis_b else angle + 90.0)
        horizontal_alignment = abs(float(np.cos(np.deg2rad(major_angle))))
        if not (
            width * 0.24 <= major <= width * 0.92
            and 0.20 <= aspect <= 0.68
            and height * 0.40 <= centre_y <= height * 0.93
            and width * -0.08 <= centre_x <= width * 1.08
            and horizontal_alignment >= 0.76
        ):
            continue
        ellipse = ((centre_x, centre_y), (axis_a, axis_b), angle)
        filled = np.zeros((height, width), np.uint8)
        cv2.ellipse(filled, ellipse, 1, thickness=cv2.FILLED)
        visible_area = int(filled.sum())
        theoretical_area = float(np.pi * major * minor * 0.25)
        area_ratio = visible_area / max(1.0, frame_area)
        containment = min(1.0, visible_area / max(1.0, theoretical_area))
        if not (0.028 <= area_ratio <= 0.38 and containment >= 0.55):
            continue
        rim = np.zeros_like(filled)
        cv2.ellipse(rim, ellipse, 1, thickness=2)
        rim_distances = distance[rim.astype(bool)]
        tolerance = float(np.clip(minor * 0.06, 4.0, 12.0))
        rim_evidence = float(np.mean(rim_distances <= tolerance)) if len(rim_distances) else 0.0
        if rim_evidence < 0.48:
            continue
        structure = _ellipse_support_structure_score(
            gray, blurred, filled, centre_x, major, minor,
        )
        if structure < 0.16:
            continue
        centre_prior = float(np.exp(
            -((centre_x / max(1, width) - 0.50) / 0.47) ** 2
            -((centre_y / max(1, height) - 0.70) / 0.38) ** 2
        ))
        aspect_prior = max(0.0, 1.0 - abs(aspect - 0.42) / 0.34)
        size_prior = max(0.0, 1.0 - abs(major / max(1, width) - 0.46) / 0.55)
        score = float(
            0.46 * rim_evidence
            + 0.13 * horizontal_alignment
            + 0.11 * centre_prior
            + 0.07 * aspect_prior
            + 0.05 * size_prior
            + 0.08 * containment
            + 0.10 * min(1.0, structure * 2.2)
        )
        if centre_x / max(1, width) < 0.05 or centre_x / max(1, width) > 0.95:
            score -= 0.10
        candidates.append((score, filled))
    if not candidates:
        return np.zeros((original_height, original_width), np.uint8), 0.0
    score, support = max(candidates, key=lambda item: item[0])
    # High threshold is intentional: this is a rescue path, not a generic
    # "anything round is a table" heuristic.
    if score < 0.78:
        return np.zeros((original_height, original_width), np.uint8), score
    if support.shape != (original_height, original_width):
        support = cv2.resize(
            support,
            (original_width, original_height),
            interpolation=cv2.INTER_NEAREST,
        )
    margin = max(2, int(round(min(original_height, original_width) * 0.004)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (margin * 2 + 1, margin * 2 + 1),
    )
    support = cv2.erode(support.astype(np.uint8), kernel)
    return _largest_component(support), float(np.clip(score, 0.0, 1.0))


def _clean_surface(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    # Instance selection must precede closing.  Closing first thickens a narrow
    # semantic/depth bridge and irreversibly merges two physical tabletops.
    mask = _dominant_receiver_instance(mask)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))
    mask = _largest_component(mask)
    if not mask.any():
        return mask
    # Remove table legs and other narrow semantic tails.  A tabletop produces
    # a broad contiguous row band, while legs rapidly collapse in width.
    widths = np.count_nonzero(mask, axis=1)
    broad = widths >= max(24, int(width * 0.16))
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(broad):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, height - 1))
    if runs:
        run = max(runs, key=lambda item: int(mask[item[0]:item[1] + 1].sum()))
        mask[run[1] + 1:] = 0
    margin = max(3, int(round(min(height, width) * 0.006)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin * 2 + 1, margin * 2 + 1))
    mask = cv2.erode(mask, kernel)
    return _largest_component(mask)


def _protected_from_semantics(
    entries: list[tuple[str, np.ndarray]],
    support: np.ndarray,
    *,
    strict_receiver_evidence: bool = False,
) -> list[list[list[float]]]:
    height, width = support.shape
    support_area = max(1, int(np.count_nonzero(support)))
    polygons: list[tuple[float, list[list[float]]]] = []
    for label, mask in entries:
        if label in STRUCTURAL_LABELS:
            continue
        # An object resting on a table replaces table pixels rather than
        # overlapping them in semantic segmentation.  Detect contact by
        # dilating the object to the support surface, then retain its full mask.
        touches_support = np.any(
            cv2.dilate(mask.astype(np.uint8), np.ones((31, 31), np.uint8))
            & support.astype(np.uint8)
        )
        if not touches_support:
            continue
        candidate = mask.astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
        for index in range(1, count):
            area = int(stats[index, cv2.CC_STAT_AREA])
            if area < height * width * 0.001:
                continue
            component = (labels == index).astype(np.uint8)
            if strict_receiver_evidence:
                # With transparent glass, curtains, walls, plants and the
                # under-table pedestal are visible through/behind the receiver
                # and frequently touch its 2-D rim.  Protect only compact
                # components with substantial *actual* overlap.  MiniMax later
                # supplies semantic receiver-object protection for ambiguous
                # cases; this local gate must not erase the entire glass top.
                overlap = int(np.count_nonzero(component & support.astype(np.uint8)))
                inside_ratio = overlap / max(1, area)
                support_occupancy = overlap / support_area
                if (
                    inside_ratio < 0.20
                    or support_occupancy > 0.32
                    or area > support_area * 0.70
                ):
                    continue
            component = cv2.dilate(component, np.ones((11, 11), np.uint8))
            contour_list, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contour_list:
                continue
            contour = max(contour_list, key=cv2.contourArea)
            approx = cv2.approxPolyDP(contour, 0.012 * cv2.arcLength(contour, True), True)
            points = approx[:, 0, :]
            polygon = [[float(x / max(1, width - 1)), float(y / max(1, height - 1))] for x, y in points]
            if len(polygon) >= 3:
                polygons.append((float(area), polygon))
    return [polygon for _, polygon in sorted(polygons, key=lambda item: item[0], reverse=True)[:12]]


def _protected_from_support_holes(support: np.ndarray) -> list[list[list[float]]]:
    """Treat substantial interior holes as unknown objects on the surface."""
    # Only true contour children are holes.  Filling the external contour and
    # subtracting the support mask also classifies every concave notch and
    # every unrecovered part of a slanted tabletop as a "protected object";
    # that used to block the whole valid placement band.
    contours, hierarchy = cv2.findContours(
        support.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours or hierarchy is None:
        return []
    height, width = support.shape
    output: list[list[list[float]]] = []
    for index, contour in enumerate(contours):
        # CCOMP hierarchy: parent >= 0 means this contour is enclosed by a
        # foreground support contour.
        if int(hierarchy[0, index, 3]) < 0:
            continue
        area = int(abs(cv2.contourArea(contour)))
        if area < height * width * 0.0012:
            continue
        component = np.zeros_like(support, np.uint8)
        cv2.drawContours(component, [contour], -1, 1, thickness=cv2.FILLED)
        component = cv2.dilate(component, np.ones((9, 9), np.uint8))
        contour_list, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        protected_contour = max(contour_list, key=cv2.contourArea)
        approx = cv2.approxPolyDP(
            protected_contour,
            0.012 * cv2.arcLength(protected_contour, True),
            True,
        )
        polygon = [
            [float(x / max(1, width - 1)), float(y / max(1, height - 1))]
            for x, y in approx[:, 0, :]
        ]
        if len(polygon) >= 3:
            output.append(polygon)
    return output[:12]


def _support_interior(support: np.ndarray) -> np.ndarray:
    """Return receiver pixels that are not dominated by the table rim.

    A sharp table edge is useful geometry evidence, but it says nothing about
    the surface BRDF.  Measuring it as part of a Laplacian statistic was the
    main reason that a crisp matte tabletop looked "reflective".  Erode only a
    small, resolution-relative band and fall back to the original mask for a
    genuinely narrow receiver.
    """

    mask = support.astype(np.uint8)
    short_side = max(1, min(mask.shape))
    radius = max(1, int(round(short_side * 0.008)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1),
    )
    interior = cv2.erode(mask, kernel).astype(bool)
    if np.count_nonzero(interior) < max(128, np.count_nonzero(mask) * 0.55):
        return mask.astype(bool)
    return interior


def _detrended_surface_luminance(
    gray: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Remove a first-order illumination ramp without removing reflections."""

    height, width = gray.shape
    ys, xs = np.nonzero(mask)
    values = gray[ys, xs].astype(np.float64)
    stride = max(1, int(np.ceil(len(xs) / 24000)))
    fit_x = xs[::stride]
    fit_y = ys[::stride]
    fit_values = values[::stride]
    design = np.column_stack([
        fit_x / max(1, width - 1),
        fit_y / max(1, height - 1),
        np.ones(len(fit_x), np.float64),
    ])
    # Exclude extreme dark and bright samples from the illumination fit.  A
    # reflected object or compact highlight must remain in the residual rather
    # than tilting the fitted plane toward itself.
    low, high = np.percentile(fit_values, [8, 92])
    stable = (fit_values >= low) & (fit_values <= high)
    if np.count_nonzero(stable) >= 16:
        coefficients, *_ = np.linalg.lstsq(
            design[stable], fit_values[stable], rcond=None,
        )
    else:
        coefficients, *_ = np.linalg.lstsq(design, fit_values, rcond=None)
    yy, xx = np.indices(gray.shape, dtype=np.float32)
    plane = (
        coefficients[0] * xx / max(1, width - 1)
        + coefficients[1] * yy / max(1, height - 1)
        + coefficients[2]
    )
    return gray.astype(np.float32) - plane.astype(np.float32)


def _coherent_specular_evidence(
    image_rgb: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, float]:
    """Measure spatially coherent highlights, rejecting one-pixel texture.

    Random noise, stone grain and printed wood texture can all have enormous
    Laplacian energy.  A photographic specular lobe differs in two useful ways:
    it is brighter than its local neighbourhood and occupies a coherent patch.
    The opening/closing below deliberately removes isolated bright pixels; no
    image content is changed because this mask is analysis evidence only.
    """

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    value = hsv[..., 2].astype(np.float32)
    saturation = hsv[..., 1].astype(np.float32)
    short_side = max(1, min(mask.shape))
    local_sigma = max(1.6, short_side / 110.0)
    local_mean = cv2.GaussianBlur(value, (0, 0), local_sigma)
    positive_residual = np.maximum(value - local_mean, 0.0)
    receiver_values = value[mask]
    bright_floor = max(170.0, float(np.percentile(receiver_values, 82)))
    candidate = (
        mask
        & (value >= bright_floor)
        & (saturation <= 88.0)
        & (positive_residual >= 5.0)
    ).astype(np.uint8)
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    ).astype(bool)
    coherent_share = float(
        np.count_nonzero(candidate) / max(1, np.count_nonzero(mask))
    )
    if np.any(candidate):
        residual_peak = float(np.percentile(positive_residual[candidate], 90))
    else:
        residual_peak = 0.0
    area_evidence = float(np.clip(coherent_share / 0.018, 0.0, 1.0))
    peak_evidence = float(np.clip((residual_peak - 4.0) / 28.0, 0.0, 1.0))
    return area_evidence * peak_evidence, coherent_share


def _reflectivity_score(image_rgb: np.ndarray, support: np.ndarray) -> float:
    """Estimate conservative image evidence for a smooth reflective receiver.

    This is intentionally *not* a material classifier.  Transparent glass has
    independent rim/geometry evidence and is promoted by ``analyze_scene``.
    Here we only decide whether ordinary semantic tabletops contain enough
    coherent specular or low-frequency reflected structure to justify adding a
    mirror layer.  Fine texture is negative evidence, never positive evidence.
    """

    mask = _support_interior(support)
    if np.count_nonzero(mask) < 128:
        return 0.0
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    detrended = _detrended_surface_luminance(gray, mask)

    # Fine-scale variation estimates receiver roughness/printed texture.  Use
    # a robust percentile rather than standard deviation so a real compact
    # highlight is not itself mistaken for a rough material.
    fine = gray - cv2.GaussianBlur(gray, (0, 0), 1.15)
    fine_energy = float(np.percentile(np.abs(fine[mask]), 75))
    smoothness = float(np.clip(1.0 - (fine_energy - 1.5) / 18.0, 0.0, 1.0))

    # Mirror images form broad nonlinear structure after a planar illumination
    # ramp has been removed.  Random texture mostly disappears under this blur.
    structure_sigma = max(2.4, min(mask.shape) / 65.0)
    structure = cv2.GaussianBlur(detrended, (0, 0), structure_sigma)
    structure_energy = float(np.percentile(np.abs(structure[mask]), 82))
    structure_evidence = float(
        np.clip((structure_energy - 2.5) / 28.0, 0.0, 1.0) * smoothness
    )

    highlight_evidence, coherent_share = _coherent_specular_evidence(
        image_rgb, mask,
    )
    # A broad white paint patch or an overexposed matte receiver is not a
    # compact specular lobe.  The local-residual test already rejects most of
    # it; taper the remaining evidence when it covers implausibly much surface.
    if coherent_share > 0.12:
        highlight_evidence *= float(np.clip(
            (0.24 - coherent_share) / 0.12, 0.0, 1.0,
        ))

    score = (
        0.08
        + 0.56 * highlight_evidence
        + 0.20 * structure_evidence
        + 0.12 * highlight_evidence * structure_evidence
    )
    return float(np.clip(score, 0.0, 1.0))


def _lighting_proxy(
    image_rgb: np.ndarray,
    support: np.ndarray,
) -> tuple[list[float], float, float, float]:
    """Estimate a conservative common shadow preset from the support surface."""
    height, width = support.shape
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    smooth = cv2.GaussianBlur(gray, (0, 0), max(7.0, min(height, width) / 34.0))
    interior = cv2.erode(support.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    ys, xs = np.nonzero(interior)
    if len(xs) < 256:
        return [0.76, 0.65], 0.040, 0.23, 0.15
    stride = max(1, int(np.ceil(len(xs) / 12000)))
    xs, ys = xs[::stride], ys[::stride]
    values = smooth[ys, xs]
    low, high = np.percentile(values, [12, 88])
    stable = (values >= low) & (values <= high)
    design = np.column_stack([
        xs[stable] / max(1, width - 1),
        ys[stable] / max(1, height - 1),
        np.ones(int(stable.sum())),
    ])
    coefficients, *_ = np.linalg.lstsq(design, values[stable], rcond=None)
    # Illumination grows toward the light; the cast shadow travels opposite.
    direction = -coefficients[:2].astype(np.float64)
    norm = float(np.linalg.norm(direction))
    if norm < 2.0:
        direction = np.array([0.76, 0.65], np.float64)
    else:
        direction /= norm
    contrast = float(np.std(values) / 255.0)
    # Consumer interiors are commonly lit by a broad ceiling fixture or a
    # large nearby source.  The previous proxy produced a long, hard silhouette
    # even on such scenes; at large manual product scales this visually detached
    # the shadow from the base.  Keep the physical height proportionality in the
    # renderer, but estimate a shorter ray and broader penumbra here.
    # A shadow with an 18% peak becomes effectively invisible after a 4K image
    # is fitted into a browser canvas.  Real broad indoor light still produces
    # a measurable umbra near the base.  These floors preserve the measured
    # contrast trend while keeping the automatic preset photographically
    # visible; explicit user values in the compositor remain untouched.
    length = float(np.clip(0.036 + contrast * 0.06, 0.038, 0.060))
    opacity = float(np.clip(0.22 + contrast * 0.18, 0.22, 0.30))
    softness = float(np.clip(0.19 - contrast * 0.20, 0.12, 0.19))
    return [float(direction[0]), float(direction[1])], length, opacity, softness


def analyze_scene(image_path: str | Path, cache_dir: str | Path, deep: bool = True) -> SceneAnalysis:
    image_path = Path(image_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _file_key(image_path)
    # Cache keys include the analysis contract. Otherwise a renderer update can
    # appear ineffective in the UI because an old lighting proxy is silently
    # reused for the same photograph.
    suffix = "deep_v9" if deep else "fast_v9"
    report_path = cache_dir / f"{key}_{suffix}.json"
    if report_path.exists():
        cached = json.loads(report_path.read_text(encoding="utf-8"))
        cached["reflection_recommended"] = float(cached.get("reflectivity", 0.0)) >= 0.54
        return SceneAnalysis(**cached)

    image_rgb = load_background(image_path)
    full_h, full_w = image_rgb.shape[:2]
    analysis_scale = min(1.0, 1400.0 / max(full_h, full_w))
    if analysis_scale < 1.0:
        image_rgb = cv2.resize(
            image_rgb,
            (
                max(1, int(round(full_w * analysis_scale))),
                max(1, int(round(full_h * analysis_scale))),
            ),
            interpolation=cv2.INTER_AREA,
        )
    entries: list[tuple[str, np.ndarray]] = []
    method = "semantic"
    analysis_error: Exception | None = None
    glass_evidence = 0.0
    relative_depth: np.ndarray | None = None
    relative_normal: np.ndarray | None = None
    # A clearly visible elliptical glass rim is direct receiver evidence.  It
    # is both more trustworthy and dramatically cheaper than running semantic
    # segmentation and monocular depth on the wall seen *through* the table.
    # Probe it before the heavy models; use a deliberately high threshold so a
    # round cushion or rug can never take this shortcut.
    early_glass_support, early_glass_evidence = _elliptical_glass_support(image_rgb)
    early_glass = bool(
        deep
        and early_glass_evidence >= 0.86
        and early_glass_support.any()
        and not _support_geometry_is_suspicious(early_glass_support)
    )
    if early_glass:
        raw_support = early_glass_support
        glass_evidence = float(early_glass_evidence)
        entries = []
        method = "transparent-glass-ellipse-direct-evidence"
    else:
        try:
            if deep:
                semantic_support, entries = _semantic_support(image_rgb)
                depth_geometry = _horizontal_from_depth(image_rgb)
                # Older tests and private integrations may still provide the
                # historical two-array result.  Preserve that call contract;
                # production inference publishes the additional normal map.
                if len(depth_geometry) == 2:
                    horizontal, relative_depth = depth_geometry
                else:
                    horizontal, relative_depth, relative_normal = depth_geometry
                # The ADE20K pipeline returns class masks independently; uncertain
                # pixels may belong to both "table" and "cabinet/sofa".  Blindly
                # subtracting every forbidden semantic mask used to erase a valid
                # tabletop and leave a tiny unrelated island.  The RANSAC depth
                # plane is the geometric arbiter; semantic non-support classes are
                # handled later as protected regions and rejection evidence.
                raw_support = _fit_support_depth_plane(
                    horizontal, semantic_support, relative_depth,
                )

                # Cabinets and shelves are valid support surfaces only when there
                # is no sufficiently broad table/desk/counter plane.
                primary_clean = _clean_surface(raw_support)
                if _semantic_seed_needs_central_recovery(semantic_support):
                    recovered = _clean_surface(
                        _fit_support_depth_plane(
                            horizontal,
                            _central_plane_seed(horizontal.shape),
                            relative_depth,
                        )
                    )
                    recovered_ys, recovered_xs = np.nonzero(recovered)
                    recovered_is_plausible = bool(
                        len(recovered_ys)
                        and float(recovered.mean()) >= max(
                            0.12, float(primary_clean.mean()) * 1.35,
                        )
                        and float(recovered_ys.min() / max(1, recovered.shape[0] - 1)) < 0.30
                        and 0.45 < float(recovered_ys.max() / max(1, recovered.shape[0] - 1)) < 0.86
                        and recovered[
                            int(recovered.shape[0] * 0.50),
                            int(recovered.shape[1] * 0.50),
                        ] > 0
                    )
                    if recovered_is_plausible:
                        raw_support = _clip_recovered_plane_at_front_edge(
                            image_rgb, recovered,
                        )
                        primary_clean = _clean_surface(raw_support)
                        method = "semantic-depth-central-recovery"
                if primary_clean.sum() >= primary_clean.size * 0.015:
                    raw_support = primary_clean
                else:
                    furniture = semantic_support.copy()
                    for label, mask in entries:
                        if label in {"cabinet", "shelf", "counter", "countertop"}:
                            furniture |= mask
                    raw_support = _fit_support_depth_plane(
                        horizontal, furniture, relative_depth,
                    )
                if method != "semantic-depth-central-recovery":
                    method = "semantic-depth-plane"
            else:
                raw_support, entries = _classical_support(image_rgb), []
                method = "classical-preview"
        except Exception as exc:
            analysis_error = exc
            raw_support = np.zeros(image_rgb.shape[:2], np.uint8)
            method = "local-analysis-failed"
    support = _clean_surface(raw_support)
    needs_special_surface_recovery = bool(
        analysis_error is not None
        or support.sum() < support.size * (0.015 if deep else 0.025)
        or _support_geometry_is_suspicious(support)
    )
    if needs_special_surface_recovery:
        glass_support, glass_evidence = _elliptical_glass_support(image_rgb)
        if glass_support.any() and glass_evidence >= 0.78:
            support = glass_support
            method = (
                "transparent-glass-ellipse-depth-fallback"
                if deep else "transparent-glass-ellipse-preview"
            )
            analysis_error = None
        elif analysis_error is not None and deep:
            raise analysis_error
    if deep and entries and support.any():
        upholstered = np.zeros_like(support, np.uint8)
        for label, mask in entries:
            if label in {"sofa", "seat"}:
                upholstered |= mask
        if upholstered.any():
            near_upholstery = cv2.dilate(
                upholstered, np.ones((33, 33), np.uint8),
            ).astype(bool)
            upholstery_overlap = float(
                np.count_nonzero(support.astype(bool) & near_upholstery)
                / max(1, np.count_nonzero(support))
            )
            if upholstery_overlap > 0.55:
                raise ValueError("Detected plane belongs to upholstered seating, not a tabletop")
    if support.sum() < support.size * 0.015 or _support_geometry_is_suspicious(support):
        raise ValueError("No trustworthy tabletop/support surface was found")

    polygon = _polygon_from_mask(support)
    ys = np.flatnonzero(support.any(axis=1))
    front_edge_y = float(ys.max() / max(1, support.shape[0] - 1))
    if method == "classical-preview":
        # A colour-only preview is useful for responsiveness, never as strict
        # evidence.  Area alone must not produce a production confidence score.
        confidence = float(np.clip(0.24 + support.mean() * 0.42, 0.24, 0.48))
    elif method.startswith("transparent-glass-ellipse"):
        ceiling = 0.72 if method.endswith("preview") else 0.88
        confidence = float(np.clip(0.38 + glass_evidence * 0.54, 0.52, ceiling))
    else:
        confidence = float(np.clip(0.42 + support.mean() * 1.35, 0.45, 0.95))
    reflectivity = _reflectivity_score(image_rgb, support)
    if method.startswith("transparent-glass-ellipse"):
        reflectivity = max(0.68, reflectivity)
    shadow_direction, shadow_length, shadow_opacity, shadow_softness = _lighting_proxy(
        image_rgb, support,
    )
    if method.startswith("transparent-glass-ellipse"):
        # A transparent horizontal receiver primarily shows the mirror image
        # of objects.  Long shadows usually continue through the glass to the
        # floor below and cannot be represented as a dark decal on the glass
        # surface.  Preserve contact AO, prefer reflection, and only enable a
        # direction cast when independent semantic evidence explicitly says so.
        surface_material = "transparent_glass"
        surface_roughness = 0.16
        surface_transmittance = 0.78
        cast_shadow_mode = "ambient_only"
        direct_light_fraction = 0.18
        shadow_confidence = 0.30
        reflection_confidence = max(0.82, glass_evidence)
        # Glass still needs a tight receiver-side contact seam.  Suppressing
        # the long cast must not suppress near-field ambient occlusion; doing
        # both made clear vessels appear to hover above the tabletop.
        # The tabletop is transparent so it does not receive a painted long
        # shadow, but it still has a zero-gap contact seam.  This needs enough
        # energy to survive 4K-to-browser downsampling; it is alpha-derived AO,
        # not an extra directional shadow.
        # Glass receivers reveal contact through reflection/transmission as
        # well as occlusion.  Keep the measured alpha-derived seam visible,
        # but do not compensate by painting a heavier dark puddle beneath a
        # clear vessel; that looks like a detached source-room stain.
        contact_ao_strength = 1.10
        shadow_opacity = 0.0
        shadow_length = 0.0
        shadow_softness = 0.20
    elif reflectivity >= 0.74:
        surface_material = "polished_opaque"
        surface_roughness = 0.34
        surface_transmittance = 0.0
        cast_shadow_mode = "directional"
        direct_light_fraction = float(np.clip(shadow_opacity / 0.36, 0.35, 0.85))
        shadow_confidence = 0.52
        reflection_confidence = 0.62
        contact_ao_strength = 0.95
    elif reflectivity >= 0.52:
        surface_material = "satin"
        surface_roughness = 0.62
        surface_transmittance = 0.0
        cast_shadow_mode = "directional"
        direct_light_fraction = float(np.clip(shadow_opacity / 0.34, 0.35, 0.82))
        shadow_confidence = 0.55
        reflection_confidence = 0.48
        contact_ao_strength = 1.0
    else:
        surface_material = "matte"
        surface_roughness = 0.90
        surface_transmittance = 0.0
        cast_shadow_mode = "directional"
        direct_light_fraction = float(np.clip(shadow_opacity / 0.34, 0.35, 0.82))
        shadow_confidence = 0.58
        reflection_confidence = 0.35
        contact_ao_strength = 1.0
    protected = _protected_from_support_holes(support)
    if entries and not method.startswith("transparent-glass-ellipse"):
        protected.extend(_protected_from_semantics(
            entries,
            support,
        ))
    blocked = np.zeros_like(support, np.uint8)
    for protected_polygon in protected:
        points = np.asarray(protected_polygon, np.float32).copy()
        points[:, 0] *= support.shape[1] - 1
        points[:, 1] *= support.shape[0] - 1
        cv2.fillPoly(blocked, [np.round(points).astype(np.int32)], 1)
    usable_support = support.astype(bool) & ~cv2.dilate(
        blocked, np.ones((9, 9), np.uint8),
    ).astype(bool)
    if np.count_nonzero(usable_support) < support.size * 0.012:
        raise ValueError("No usable tabletop remains after protecting existing scene objects")
    mask_path = cache_dir / f"{key}_{suffix}_support.png"
    save_png(mask_path, support.astype(np.uint8) * 255)
    depth_map_path: str | None = None
    normal_map_path: str | None = None
    if relative_depth is not None:
        depth_path = cache_dir / f"{key}_{suffix}_depth.npy"
        depth_evidence = np.asarray(relative_depth, dtype=np.float32)
        if depth_evidence.shape != (full_h, full_w):
            depth_evidence = cv2.resize(
                depth_evidence,
                (full_w, full_h),
                interpolation=cv2.INTER_LINEAR,
            )
        depth_evidence = _normalize_depth_map(depth_evidence)
        np.save(
            depth_path,
            np.ascontiguousarray(depth_evidence, dtype=np.float32),
            allow_pickle=False,
        )
        depth_map_path = str(depth_path.resolve())
    if relative_normal is not None:
        normal_path = cache_dir / f"{key}_{suffix}_normal.npy"
        normal_evidence = np.asarray(relative_normal, dtype=np.float32)
        if normal_evidence.shape[:2] != (full_h, full_w):
            normal_evidence = cv2.resize(
                normal_evidence,
                (full_w, full_h),
                interpolation=cv2.INTER_LINEAR,
            )
        # Interpolation shortens and blends normal vectors.  Normalize only
        # after registration to the original background so every persisted
        # full-resolution sample remains a unit vector.
        normal_evidence = _unitize_normal_map(normal_evidence)
        np.save(
            normal_path,
            np.ascontiguousarray(normal_evidence, dtype=np.float32),
            allow_pickle=False,
        )
        normal_map_path = str(normal_path.resolve())
    analysis = SceneAnalysis(
        support_polygon=polygon,
        protected_polygons=protected,
        support_mask_path=str(mask_path.resolve()),
        front_edge_y=front_edge_y,
        reflectivity=reflectivity,
        reflection_recommended=reflectivity >= 0.54,
        method=method,
        confidence=confidence,
        shadow_direction=shadow_direction,
        shadow_length_ratio=shadow_length,
        shadow_opacity=shadow_opacity,
        shadow_softness_ratio=shadow_softness,
        surface_material=surface_material,
        surface_roughness=surface_roughness,
        surface_transmittance=surface_transmittance,
        cast_shadow_mode=cast_shadow_mode,
        direct_light_fraction=direct_light_fraction,
        shadow_confidence=shadow_confidence,
        reflection_confidence=reflection_confidence,
        contact_ao_strength=contact_ao_strength,
        depth_map_path=depth_map_path,
        normal_map_path=normal_map_path,
    )
    report_path.write_text(json.dumps(analysis.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return analysis
