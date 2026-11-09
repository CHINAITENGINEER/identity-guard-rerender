from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .errors import ContractError
from .image_io import load_background, load_rgba, save_png


@lru_cache(maxsize=4)
def _rembg_session(model: str):
    from rembg import new_session
    # Keep project models off the nearly-full system drive and make model
    # selection reproducible for the local workbench.
    model_home = Path.cwd().resolve() / "ui_workspace" / "model_cache" / "rembg"
    model_home.mkdir(parents=True, exist_ok=True)
    os.environ["U2NET_HOME"] = str(model_home)
    return new_session(model)


@lru_cache(maxsize=1)
def _sam2_box_session():
    """Load the compact promptable segmenter once per UI process."""
    try:
        from transformers import Sam2Model, Sam2Processor
    except ImportError as error:
        raise ContractError(
            "Box-selected cutout requires transformers and torch"
        ) from error
    model_id = "facebook/sam2.1-hiera-tiny"
    # Transformers still performs a network HEAD request for optional files
    # when a model id is used, even if all required SAM2 files are cached.
    # That made box recovery fail whenever Hugging Face was slow although the
    # 155 MB model was already present.  Prefer a complete local snapshot and
    # make the offline contract explicit; only fall back to the model id when
    # no snapshot exists at all.
    cache_homes = [Path.home() / ".cache" / "huggingface"]
    if os.environ.get("HF_HOME"):
        cache_homes.insert(0, Path(os.environ["HF_HOME"]))
    snapshot_candidates = []
    for cache_home in dict.fromkeys(cache_homes):
        snapshot_root = (
            cache_home / "hub" /
            "models--facebook--sam2.1-hiera-tiny" / "snapshots"
        )
        snapshot_candidates.extend(snapshot_root.glob("*"))
    snapshots = sorted(
        (
            path for path in snapshot_candidates
            if (path / "config.json").is_file()
            and (path / "model.safetensors").is_file()
            and (path / "processor_config.json").is_file()
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    model_source = str(snapshots[0]) if snapshots else model_id
    local_only = bool(snapshots)
    processor = Sam2Processor.from_pretrained(
        model_source, local_files_only=local_only,
    )
    model = Sam2Model.from_pretrained(
        model_source, local_files_only=local_only,
    )
    model.eval()
    return processor, model


def _alpha_topology_metrics(alpha: np.ndarray) -> dict[str, Any]:
    """Measure holes that a connected-component score cannot see.

    A badly perforated matte can remain one connected foreground island and
    therefore look structurally perfect to the old audit.  Work on confident
    alpha, flood the exterior background from a one-pixel padded border, then
    count only background islands enclosed by the foreground silhouette.
    These metrics are descriptive by default: handles, baskets and transparent
    vessels contain legitimate holes.  ``alpha_quality`` only turns them into
    a gate when product semantics explicitly describe a solid opaque object.
    """
    stable = np.asarray(alpha) >= 128
    foreground = int(np.count_nonzero(stable))
    if foreground == 0:
        return {
            "internal_hole_count": 0,
            "internal_hole_pixels": 0,
            "internal_hole_fraction": 0.0,
            "largest_internal_hole_fraction": 0.0,
            "row_envelope_gap_fraction": 0.0,
        }

    padded_background = np.pad(
        (~stable).astype(np.uint8), 1, constant_values=1,
    )
    exterior = padded_background.copy()
    cv2.floodFill(exterior, None, (0, 0), 2)
    holes = ((padded_background == 1) & (exterior != 2))[1:-1, 1:-1]
    hole_pixels = int(np.count_nonzero(holes))
    hole_count, _, hole_stats, _ = cv2.connectedComponentsWithStats(
        holes.astype(np.uint8), 8,
    )
    largest_hole = (
        int(hole_stats[1:, cv2.CC_STAT_AREA].max())
        if hole_count > 1 else 0
    )
    enclosed_area = max(1, foreground + hole_pixels)

    rows, columns = np.where(stable)
    envelope_pixels = 0
    missing_pixels = 0
    if len(rows):
        for row_index in range(int(rows.min()), int(rows.max()) + 1):
            columns_here = np.flatnonzero(stable[row_index])
            if columns_here.size < 2:
                continue
            span = int(columns_here[-1] - columns_here[0] + 1)
            envelope_pixels += span
            missing_pixels += span - int(columns_here.size)
    return {
        "internal_hole_count": int(max(0, hole_count - 1)),
        "internal_hole_pixels": hole_pixels,
        "internal_hole_fraction": float(hole_pixels / enclosed_area),
        "largest_internal_hole_fraction": float(largest_hole / enclosed_area),
        "row_envelope_gap_fraction": float(
            missing_pixels / max(1, envelope_pixels)
        ),
    }


def _solid_opaque_topology_expected(subject_label: str) -> bool:
    label = " ".join(str(subject_label).lower().replace("_", " ").split())
    solid_tokens = (
        "figurine", "statue", "sculpture", "solid ornament",
        "ceramic animal", "porcelain animal", "resin animal",
        "toy animal", "decorative animal",
    )
    negative_space_tokens = (
        "transparent", "clear glass", "frosted glass", "wire", "mesh",
        "woven", "basket", "rack", "frame", "handle", "handled",
        "chair", "stool", "lantern", "cage", "fence", "trellis",
        "plant", "flower arrangement",
    )
    return bool(
        any(token in label for token in solid_tokens)
        and not any(token in label for token in negative_space_tokens)
    )


def alpha_quality(
    rgba: np.ndarray,
    subject_label: str = "",
) -> dict[str, Any]:
    alpha = rgba[..., 3]
    binary = alpha > 4
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    areas = sorted((int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, count)), reverse=True)
    total = max(1, int(binary.sum()))
    largest_share = float(areas[0] / total) if areas else 0.0
    touches = bool(binary[0].any() or binary[-1].any() or binary[:, 0].any() or binary[:, -1].any())
    soft = float(np.mean((alpha > 4) & (alpha < 250)))
    foreground_ratio = float(binary.mean())
    height, width = binary.shape
    band_x = max(1, int(round(width * 0.03)))
    band_y = max(1, int(round(height * 0.03)))
    edge_occupancy = {
        "left": float(binary[:, :band_x].mean()),
        "right": float(binary[:, -band_x:].mean()),
        "top": float(binary[:band_y].mean()),
        "bottom": float(binary[-band_y:].mean()),
    }
    opposite_edge_attachment = bool(
        (
            edge_occupancy["left"] > 0.45
            and edge_occupancy["right"] > 0.45
        )
        or (
            edge_occupancy["top"] > 0.45
            and edge_occupancy["bottom"] > 0.45
        )
    )

    # A clean-looking matte can still contain two completely different
    # photographed props.  Reconnect only tiny anti-alias gaps, then detect
    # two large, distant, fully opaque islands.  This catches cases such as a
    # plate plus the vase beside it without treating dust, stems, cables or a
    # narrowly detached handle as a second product.
    stable = (alpha >= 128).astype(np.uint8)
    reconnect_radius = max(1, int(round(min(alpha.shape) * 0.004)))
    reconnect_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (reconnect_radius * 2 + 1, reconnect_radius * 2 + 1),
    )
    stable = cv2.morphologyEx(stable, cv2.MORPH_CLOSE, reconnect_kernel)
    stable_count, _, stable_stats, stable_centroids = (
        cv2.connectedComponentsWithStats(stable, 8)
    )
    stable_indices = sorted(
        range(1, stable_count),
        key=lambda index: int(stable_stats[index, cv2.CC_STAT_AREA]),
        reverse=True,
    )
    stable_total = max(1, int(np.count_nonzero(stable)))
    second_component_share = 0.0
    component_centroid_distance = 0.0
    component_bbox_gap = 0.0
    if len(stable_indices) >= 2:
        first, second = stable_indices[:2]
        second_component_share = float(
            stable_stats[second, cv2.CC_STAT_AREA] / stable_total
        )
        diagonal = max(1.0, float(np.hypot(*alpha.shape)))
        component_centroid_distance = float(
            np.linalg.norm(stable_centroids[first] - stable_centroids[second])
            / diagonal
        )
        first_x, first_y, first_w, first_h = (
            int(value) for value in stable_stats[first, :4]
        )
        second_x, second_y, second_w, second_h = (
            int(value) for value in stable_stats[second, :4]
        )
        gap_x = max(
            0,
            max(first_x, second_x)
            - min(first_x + first_w, second_x + second_w),
        )
        gap_y = max(
            0,
            max(first_y, second_y)
            - min(first_y + first_h, second_y + second_h),
        )
        component_bbox_gap = float(np.hypot(gap_x, gap_y) / diagonal)
    stable_area_ratio = float(
        np.count_nonzero(alpha >= 245)
        / max(1, np.count_nonzero(alpha >= 32))
    )
    multi_object_leakage = bool(
        second_component_share >= 0.15
        and component_centroid_distance >= 0.28
        and component_bbox_gap >= 0.04
        and stable_area_ratio >= 0.85
    )
    topology = _alpha_topology_metrics(alpha)
    solid_topology_expected = _solid_opaque_topology_expected(subject_label)
    opaque_topology_suspect = bool(
        solid_topology_expected
        and (
            (
                topology["internal_hole_fraction"] >= 0.012
                and topology["largest_internal_hole_fraction"] >= 0.003
            )
            or (
                topology["internal_hole_fraction"] >= 0.006
                and topology["row_envelope_gap_fraction"] >= 0.075
            )
        )
    )
    score = 100.0
    if largest_share < 0.94:
        score -= min(26.0, (0.94 - largest_share) * 120)
    if len(areas) > 1 and (total - areas[0]) / total > 0.005:
        score -= min(12.0, 4.0 + (len(areas) - 1) * 2.0)
    if touches:
        score -= 12.0
    if foreground_ratio < 0.015 or foreground_ratio > 0.92:
        score -= 28.0
    if soft > 0.12:
        score -= 12.0
    if opposite_edge_attachment:
        # A foreground object can touch one side of a tight crop.  Broad
        # attachment to two opposite sides is different: it is the signature
        # of a wall/table region being retained while the actual product is
        # cut out as a hole.
        score -= 48.0
    if multi_object_leakage:
        score -= 60.0
    if opaque_topology_suspect:
        score -= min(
            70.0,
            20.0
            + topology["internal_hole_fraction"] * 760.0
            + topology["row_envelope_gap_fraction"] * 180.0,
        )
    rejected = bool(
        opposite_edge_attachment
        or multi_object_leakage
        or opaque_topology_suspect
        or foreground_ratio < 0.006
        or foreground_ratio > 0.94
    )
    verdict = "reject" if rejected else ("pass" if score >= 82 else "review")
    return {
        "score": float(np.clip(score, 0, 100)),
        "foreground_ratio": foreground_ratio,
        "component_count": max(0, count - 1),
        "largest_component_share": largest_share,
        "touches_canvas": touches,
        "soft_alpha_ratio": soft,
        "edge_occupancy": edge_occupancy,
        "opposite_edge_attachment": opposite_edge_attachment,
        "stable_area_ratio": stable_area_ratio,
        "second_component_share": second_component_share,
        "component_centroid_distance": component_centroid_distance,
        "component_bbox_gap": component_bbox_gap,
        "multi_object_leakage": multi_object_leakage,
        **topology,
        "solid_opaque_topology_expected": solid_topology_expected,
        "opaque_topology_suspect": opaque_topology_suspect,
        "verdict": verdict,
        "reason": (
            "一个商品蒙版中保留了两个相距较远的大型不透明主体"
            if multi_object_leakage
            else "solid opaque product contains implausible segmentation holes"
            if opaque_topology_suspect
            else "蒙版疑似前后景反转，已阻止把背景区域当作商品"
            if opposite_edge_attachment
            else ""
        ),
    }


def clean_alpha(rgba: np.ndarray) -> np.ndarray:
    """Remove segmentation dust and soft source shadows.

    Background-removal models often return the photographed floor shadow as a
    broad low-alpha foreground.  Keeping it and then adding a new scene shadow
    creates the characteristic double-shadow seen in early UI results.
    """
    output = rgba.copy()
    alpha = output[..., 3].copy()
    confident = alpha >= 210
    points = np.argwhere(confident)
    if points.size:
        y0, x0 = points.min(axis=0)
        y1, x1 = points.max(axis=0)
        object_h = max(1, int(y1 - y0 + 1))
        bottom = confident[max(int(y0), int(y1 - object_h * 0.12)):int(y1) + 1]
        bottom_x = np.flatnonzero(bottom.any(axis=0))
        if len(bottom_x) >= 4:
            centre = (float(bottom_x.min()) + float(bottom_x.max())) * 0.5
            base_half = max(2.0, (float(bottom_x.max()) - float(bottom_x.min()) + 1) * 0.5)
            yy, xx = np.indices(alpha.shape)
            middle = float(y0) + object_h * 0.48
            progress = np.clip((yy - middle) / max(1.0, float(y1) - middle), 0.0, 1.0)
            allowed_half = base_half * (1.65 - 0.55 * progress)
            outside = (yy >= middle) & (np.abs(xx - centre) > allowed_half) & (alpha > 4)
            inside = (yy >= middle) & (np.abs(xx - centre) <= allowed_half) & (alpha > 4)
            if (
                np.count_nonzero(outside) > np.count_nonzero(alpha > 4) * 0.02
                and np.count_nonzero(inside)
                and float(alpha[outside].mean()) < float(alpha[inside].mean()) * 0.75
            ):
                alpha[outside] = 0
    binary = alpha >= 150
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    if count <= 1:
        return output
    largest = int(stats[1:, cv2.CC_STAT_AREA].max())
    keep = np.zeros_like(binary)
    threshold = max(12, int(largest * 0.0015))
    for index in range(1, count):
        if int(stats[index, cv2.CC_STAT_AREA]) >= threshold:
            keep |= labels == index
    # Reconstruct a narrow anti-aliased matte from the confident foreground.
    # The broad low-alpha source shadow is not allowed back in.
    matte = cv2.GaussianBlur(keep.astype(np.uint8) * 255, (0, 0), 0.58)
    output[..., 3] = matte
    return output


def _semantic_refinement_preservation(
    original_alpha: np.ndarray,
    refined_alpha: np.ndarray,
) -> dict[str, Any]:
    """Audit whether a semantic allow-list preserves the photographed object.

    Scene parsers are deliberately coarse: at their inference resolution a
    dark product stand may be called ``base``/``table`` and a thin circular
    rim may sit just outside the selected semantic class.  Area retention on
    its own does not catch that failure because a large central body can remain
    while nearly every visible edge and the complete support are removed.

    The audit therefore combines three scale-independent signals:

    * total hard-alpha retention;
    * how widely the lost pixels cover the original silhouette boundary; and
    * retention in the bottom support band of the original silhouette.

    A local, one-sided removal (for example an attached photographed shadow)
    is still allowed.  Rejection requires both material area loss and either
    widespread boundary erosion or strong bottom-support destruction.
    """
    if original_alpha.shape != refined_alpha.shape:
        raise ContractError("Semantic refinement alpha shapes do not match")

    original = np.asarray(original_alpha) >= 110
    refined = np.asarray(refined_alpha) >= 110
    original_count = int(np.count_nonzero(original))
    refined_count = int(np.count_nonzero(refined))
    retained = float(refined_count / max(1, original_count))
    lost = original & ~refined

    metrics: dict[str, Any] = {
        "preserves_subject": True,
        "retained_ratio": retained,
        "boundary_loss_ratio": 0.0,
        "boundary_damage_coverage": 0.0,
        "damaged_boundary_bins": 0,
        "valid_boundary_bins": 0,
        "bottom_support_retention": 1.0,
        "bottom_support_share": 0.0,
        "reasons": [],
    }
    if original_count == 0:
        metrics["preserves_subject"] = False
        metrics["reasons"] = ["empty_original_alpha"]
        return metrics

    height, width = original.shape
    points = np.argwhere(original)
    y0, x0 = points.min(axis=0)
    y1, x1 = points.max(axis=0)
    object_height = max(1, int(y1 - y0 + 1))

    # A scale-aware inner boundary band avoids making the decision depend on
    # the source resolution.  It includes thin appendages and internal holes,
    # which are exactly the structures a coarse semantic parser can destroy.
    boundary_radius = int(np.clip(round(min(height, width) * 0.0035), 2, 12))
    kernel_size = boundary_radius * 2 + 1
    eroded = cv2.erode(
        original.astype(np.uint8),
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size),
        ),
    ).astype(bool)
    boundary = original & ~eroded
    boundary_count = int(np.count_nonzero(boundary))
    lost_boundary = boundary & lost
    boundary_loss = float(
        np.count_nonzero(lost_boundary) / max(1, boundary_count)
    )
    metrics["boundary_loss_ratio"] = boundary_loss

    # Measure *coverage* of the boundary damage rather than only its pixel
    # count.  Removing a one-sided shadow occupies few angular sectors;
    # shrinking a plate rim or deleting a stand damages many sectors.
    boundary_points = np.argwhere(boundary)
    if len(boundary_points):
        centre_y, centre_x = points.mean(axis=0)
        angles = np.arctan2(
            boundary_points[:, 0].astype(np.float64) - centre_y,
            boundary_points[:, 1].astype(np.float64) - centre_x,
        )
        bin_count = 24
        angular_bins = np.floor(
            (angles + np.pi) * (bin_count / (2.0 * np.pi)),
        ).astype(np.int32) % bin_count
        lost_at_boundary = lost_boundary[
            boundary_points[:, 0], boundary_points[:, 1]
        ]
        minimum_bin_evidence = max(8, int(round(boundary_count * 0.004)))
        valid_bins = 0
        damaged_bins = 0
        for bin_index in range(bin_count):
            members = angular_bins == bin_index
            member_count = int(np.count_nonzero(members))
            if member_count < minimum_bin_evidence:
                continue
            valid_bins += 1
            if float(np.count_nonzero(lost_at_boundary[members]) / member_count) >= 0.35:
                damaged_bins += 1
        coverage = float(damaged_bins / max(1, valid_bins))
        metrics["boundary_damage_coverage"] = coverage
        metrics["damaged_boundary_bins"] = damaged_bins
        metrics["valid_boundary_bins"] = valid_bins

    # The bottom band is relative to the *object*, not the canvas.  It catches
    # feet, pedestals and stands even when the source crop has a large margin.
    bottom_band_top = max(int(y0), int(round(y1 - object_height * 0.20 + 1)))
    bottom_original = original[bottom_band_top:int(y1) + 1]
    bottom_refined = refined[bottom_band_top:int(y1) + 1]
    bottom_count = int(np.count_nonzero(bottom_original))
    bottom_retained = float(
        np.count_nonzero(bottom_refined & bottom_original) / max(1, bottom_count)
    )
    bottom_share = float(bottom_count / max(1, original_count))
    metrics["bottom_support_retention"] = bottom_retained
    metrics["bottom_support_share"] = bottom_share

    widespread_boundary_erosion = (
        retained < 0.94
        and boundary_loss > 0.42
        and metrics["boundary_damage_coverage"] >= 0.50
    )
    missing_bottom_support = (
        retained < 0.94
        and bottom_share >= 0.02
        and bottom_retained < 0.48
        and boundary_loss > 0.30
        and metrics["boundary_damage_coverage"] >= 0.25
    )
    reasons: list[str] = []
    if widespread_boundary_erosion:
        reasons.append("widespread_subject_boundary_erosion")
    if missing_bottom_support:
        reasons.append("bottom_support_structure_removed")
    metrics["preserves_subject"] = not reasons
    metrics["reasons"] = reasons
    return metrics


def refine_alpha_with_semantics(rgba: np.ndarray) -> tuple[np.ndarray, bool]:
    """Remove model-selected source background/shadows with semantic evidence.

    A dark wall shadow can be as opaque as a black product and fool generic
    matting.  Semantic masks are used only as a deterministic *allow list* for
    the alpha generated by rembg; RGB is never regenerated or replaced.
    """
    from PIL import Image
    from .scene import _semantic_segment

    height, width = rgba.shape[:2]
    scale = min(1.0, 896.0 / max(height, width))
    small_w = max(1, int(round(width * scale)))
    small_h = max(1, int(round(height * scale)))
    inference_rgb = cv2.resize(
        rgba[..., :3], (small_w, small_h), interpolation=cv2.INTER_AREA,
    )
    alpha_small = cv2.resize(
        rgba[..., 3], (small_w, small_h), interpolation=cv2.INTER_AREA,
    )
    foreground = alpha_small >= 110
    if int(foreground.sum()) < small_h * small_w * 0.01:
        return rgba, False

    background_labels = {
        "wall", "floor", "ceiling", "sky", "road", "grass", "water",
        "shelf", "cabinet", "counter", "countertop", "desk", "table",
        "door", "window", "curtain", "blind", "step", "base",
    }
    components: list[tuple[str, np.ndarray, int, float]] = []
    for entry in _semantic_segment(inference_rgb):
        label = str(entry["label"]).strip().lower()
        raw = (np.asarray(entry["mask"].resize((small_w, small_h))) > 0).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
        for index in range(1, count):
            area = int(stats[index, cv2.CC_STAT_AREA])
            if area < 10:
                continue
            component = labels == index
            overlap = float(np.count_nonzero(component & foreground) / max(1, area))
            components.append((label, component, area, overlap))

    selected = np.zeros((small_h, small_w), np.uint8)
    selected_labels: set[str] = set()
    frame_area = float(small_h * small_w)
    foreground_x = np.flatnonzero(foreground.any(axis=0))
    foreground_centre_x = float(np.median(foreground_x)) if len(foreground_x) else small_w * 0.5
    for label, component, area, overlap in components:
        if label in background_labels or overlap < 0.42:
            continue
        component_x = np.flatnonzero(component.any(axis=0))
        if not len(component_x) or abs(float(np.median(component_x)) - foreground_centre_x) > small_w * 0.26:
            continue
        selected[component] = 1
        selected_labels.add(label)
    if not selected.any():
        return rgba, False

    # Bring back small adjacent parts that a scene parser calls "base",
    # "step", or even "table", while broad walls/shelves remain excluded.
    for _ in range(3):
        grown = cv2.dilate(selected, np.ones((19, 19), np.uint8))
        changed = False
        for label, component, area, overlap in components:
            maximum_area = frame_area * (0.008 if label in background_labels else 0.085)
            if overlap < 0.50 or area > maximum_area:
                continue
            if np.any(component & grown.astype(bool)) and np.any(component & ~selected.astype(bool)):
                selected[component] = 1
                selected_labels.add(label)
                changed = True
        if not changed:
            break

    # Semantic masks describe visible classes, not alpha mattes.  They often
    # label the rim, body and inside of a cup as different classes, leaving
    # artificial holes between them.  Fill only the *external* contour of the
    # central semantic object; this preserves its concave silhouette without
    # taking the convex hull or accepting the photographed cast shadow.
    close_width = max(7, int(round(small_w * 0.014)) | 1)
    close_height = max(15, int(round(small_h * 0.042)) | 1)
    joined = cv2.morphologyEx(
        selected,
        cv2.MORPH_CLOSE,
        np.ones((close_height, close_width), np.uint8),
    )
    contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(joined)
    if contours:
        ranked = sorted(contours, key=cv2.contourArea, reverse=True)
        largest_area = max(1.0, cv2.contourArea(ranked[0]))
        for contour in ranked:
            area = cv2.contourArea(contour)
            x, _, width_box, _ = cv2.boundingRect(contour)
            centre_x = x + width_box * 0.5
            if area >= largest_area * 0.025 and abs(centre_x - foreground_centre_x) <= small_w * 0.28:
                cv2.drawContours(filled, [contour], -1, 1, thickness=cv2.FILLED)
    if not filled.any():
        return rgba, False

    # The same physical product is commonly split into classes (for example
    # cup/body/plate/base).  Their union can still contain a false waist that
    # cuts the lower body away.  A tight semantic convex envelope is safe here
    # because it is only an allow-list: the original high-resolution U2Net
    # alpha must also be foreground, so holes/handles are not filled with new
    # pixels.  The envelope mainly rejects floor/table shadow outside the
    # product's semantic extent.
    points_yx = np.argwhere(filled > 0)
    envelope = filled
    articulated = any(
        token in label
        for label in selected_labels
        for token in ("plant", "tree", "flower", "person", "animal")
    )
    if len(points_yx) >= 3 and not articulated:
        hull = cv2.convexHull(points_yx[:, ::-1].astype(np.int32))
        envelope = np.zeros_like(filled)
        cv2.drawContours(envelope, [hull], -1, 1, thickness=cv2.FILLED)

    allowed = cv2.dilate(envelope, np.ones((13, 13), np.uint8))
    allowed = cv2.resize(allowed, (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
    refined = rgba.copy()
    refined[..., 3] = np.where(allowed, rgba[..., 3], 0).astype(np.uint8)
    before = int(np.count_nonzero(rgba[..., 3] >= 110))
    after = int(np.count_nonzero(refined[..., 3] >= 110))
    retained = after / max(1, before)
    if retained < 0.38 or retained > 0.985:
        return rgba, False
    preservation = _semantic_refinement_preservation(
        rgba[..., 3], refined[..., 3],
    )
    if not preservation["preserves_subject"]:
        return rgba, False
    original_points = np.argwhere(rgba[..., 3] >= 110)
    refined_points = np.argwhere(refined[..., 3] >= 110)
    if len(original_points) and len(refined_points):
        original_top, original_bottom = (
            int(original_points[:, 0].min()),
            int(original_points[:, 0].max()),
        )
        refined_top, refined_bottom = (
            int(refined_points[:, 0].min()),
            int(refined_points[:, 0].max()),
        )
        original_span = max(1, original_bottom - original_top + 1)
        refined_span = max(1, refined_bottom - refined_top + 1)
        # Scene parsers often call a black product stand "base" or "table".
        # A semantic allow-list may remove photographed shadows, but it must
        # never collapse the product vertically or discard its entire base.
        if (
            refined_span < original_span * 0.68
            or refined_bottom < original_bottom - original_span * 0.18
        ):
            return rgba, False
    refined = clean_alpha(refined)
    return refined, True


def trim_transparent(rgba: np.ndarray, padding: int = 4) -> np.ndarray:
    alpha = rgba[..., 3]
    rows, columns = np.where(alpha > 4)
    if not len(rows):
        raise ContractError("Cannot trim an empty alpha mask")
    y0, y1 = max(0, rows.min() - padding), min(rgba.shape[0], rows.max() + padding + 1)
    x0, x1 = max(0, columns.min() - padding), min(rgba.shape[1], columns.max() + padding + 1)
    return rgba[y0:y1, x0:x1].copy()


def _select_rembg_candidate(
    candidates: dict[str, np.ndarray],
    subject_label: str = "",
) -> tuple[str, dict[str, Any]]:
    """Choose the more complete plausible mask from independent models.

    Structural alpha quality cannot detect a missing sword handle: a clean
    fragment is still a clean connected component.  Two independent models
    provide completeness evidence.  When one plausible mask contains nearly
    all of another but extends it substantially, the contained mask is treated
    as an incomplete subset rather than a safer cutout.
    """
    if not candidates:
        raise ContractError("No cutout candidate was produced")
    metrics: dict[str, dict[str, Any]] = {}
    masks: dict[str, np.ndarray] = {}
    for model, rgba in candidates.items():
        trimmed = trim_transparent(rgba)
        quality = alpha_quality(trimmed, subject_label=subject_label)
        mask = rgba[..., 3] >= 110
        masks[model] = mask
        metrics[model] = {
            "quality": quality,
            "area": int(mask.sum()),
        }
    viable = [
        model for model in candidates
        if metrics[model]["quality"]["verdict"] != "reject"
    ]
    pool = viable or list(candidates)
    selected = max(
        pool,
        key=lambda model: (
            float(metrics[model]["quality"]["score"]),
            int(metrics[model]["area"]),
        ),
    )

    pair_metrics: list[dict[str, Any]] = []

    def hole_fill_evidence(
        subset_mask: np.ndarray,
        superset_mask: np.ndarray,
    ) -> tuple[float, float]:
        """Detect a larger mask painting over a real handle/strap opening."""
        added = superset_mask & ~subset_mask
        added_area = int(np.count_nonzero(added))
        subset_area = max(1, int(np.count_nonzero(subset_mask)))
        if added_area == 0:
            return 0.0, 0.0
        contours, _ = cv2.findContours(
            subset_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        exterior_fill = np.zeros_like(subset_mask, np.uint8)
        if contours:
            cv2.drawContours(
                exterior_fill, contours, -1, 1, thickness=cv2.FILLED,
            )
        enclosed_void = exterior_fill.astype(bool) & ~subset_mask
        share_inside_void = float(
            np.count_nonzero(added & enclosed_void) / max(1, added_area)
        )
        return share_inside_void, float(added_area / subset_area)

    for first_index, first in enumerate(candidates):
        for second in list(candidates)[first_index + 1:]:
            first_mask, second_mask = masks[first], masks[second]
            intersection = int(np.count_nonzero(first_mask & second_mask))
            union = max(1, int(np.count_nonzero(first_mask | second_mask)))
            first_area = max(1, int(metrics[first]["area"]))
            second_area = max(1, int(metrics[second]["area"]))
            first_in_second = float(intersection / first_area)
            second_in_first = float(intersection / second_area)
            pair = {
                "models": [first, second],
                "iou": float(intersection / union),
                f"{first}_contained_by_{second}": first_in_second,
                f"{second}_contained_by_{first}": second_in_first,
            }
            pair_metrics.append(pair)

            # Prefer a plausible superset when it preserves at least 94% of
            # the other model and does not balloon to a background-sized mask.
            relationships = [
                (first, second, first_in_second, second_area / first_area),
                (second, first, second_in_first, first_area / second_area),
            ]
            for subset, superset, containment, expansion in relationships:
                superset_quality = metrics[superset]["quality"]
                subset_quality = metrics[subset]["quality"]
                hole_fill_share, added_share = hole_fill_evidence(
                    masks[subset], masks[superset],
                )
                pair[f"{superset}_added_inside_{subset}_holes"] = hole_fill_share
                pair[f"{superset}_added_share_over_{subset}"] = added_share
                false_hole_fill = bool(
                    containment >= 0.94
                    and added_share >= 0.018
                    and hole_fill_share >= 0.58
                )
                if (
                    false_hole_fill
                    and float(subset_quality["score"])
                    >= float(superset_quality["score"]) - 6.0
                ):
                    selected = subset
                    continue
                if (
                    containment >= 0.94
                    and 1.02 <= expansion <= 2.80
                    and not false_hole_fill
                    and superset_quality["verdict"] != "reject"
                    and float(superset_quality["score"])
                    >= float(subset_quality["score"]) - 15.0
                ):
                    selected = superset

    selection_reason = "highest_structural_quality"
    promptable_name = "sam2-box-recovery"
    if promptable_name in candidates:
        promptable_mask = masks[promptable_name]
        promptable_quality = metrics[promptable_name]["quality"]
        promptable_area = max(1, int(metrics[promptable_name]["area"]))
        # Net area alone hid the cup-handle failure: SAM2 recovered 4.1% of
        # real handle pixels while disagreeing on 2.8% of boundary pixels, so
        # its net expansion was only 1.3%.  Prefer the promptable witness when
        # it preserves nearly all of a conventional matte and contributes a
        # material amount of new, box-contained structure.
        for base_name, base_mask in masks.items():
            if base_name == promptable_name:
                continue
            base_area = max(1, int(metrics[base_name]["area"]))
            shared = int(np.count_nonzero(base_mask & promptable_mask))
            preserved = float(shared / base_area)
            added = float(
                np.count_nonzero(promptable_mask & ~base_mask) / base_area
            )
            removed = float(
                np.count_nonzero(base_mask & ~promptable_mask) / base_area
            )
            if (
                preserved >= 0.965
                and added >= 0.025
                and removed <= 0.040
                and promptable_quality["verdict"] != "reject"
                and float(promptable_quality["score"])
                >= float(metrics[base_name]["quality"]["score"]) - 12.0
                and 0.94 <= promptable_area / base_area <= 1.35
            ):
                selected = promptable_name
                selection_reason = (
                    "promptable_structural_recovery:"
                    f"base={base_name},preserved={preserved:.4f},"
                    f"added={added:.4f},removed={removed:.4f}"
                )
                break

    selected_quality = dict(metrics[selected]["quality"])
    agreement_candidates: list[tuple[float, float]] = []
    selected_mask = masks[selected]
    selected_area = max(1, int(metrics[selected]["area"]))
    for other, other_mask in masks.items():
        if other == selected:
            continue
        intersection = int(np.count_nonzero(selected_mask & other_mask))
        other_area = max(1, int(metrics[other]["area"]))
        union = max(1, int(np.count_nonzero(selected_mask | other_mask)))
        agreement_candidates.append((
            float(intersection / min(selected_area, other_area)),
            float(intersection / union),
        ))
    if agreement_candidates:
        containment, iou = max(
            agreement_candidates,
            key=lambda pair: pair[0] * 0.72 + pair[1] * 0.28,
        )
        completeness = float(np.clip(
            60.0 + containment * 25.0 + iou * 10.0,
            0.0,
            95.0,
        ))
    else:
        containment, iou, completeness = 0.0, 0.0, 72.0
    structure_score = float(selected_quality["score"])
    selected_quality["structure_score"] = structure_score
    selected_quality["completeness_score"] = completeness
    selected_quality["score"] = float(min(structure_score, completeness))
    selected_quality["score_kind"] = "automatic_confidence_not_ground_truth"
    selected_quality["model_agreement"] = {
        "confirmed_smaller_mask_share": containment,
        "intersection_over_union": iou,
    }
    if selected_quality["verdict"] != "reject":
        selected_quality["verdict"] = (
            "pass" if selected_quality["score"] >= 90.0 else "review"
        )
    return selected, {
        "selected_model": selected,
        "selection_reason": selection_reason,
        "models": list(candidates),
        "candidate_metrics": metrics,
        "pair_metrics": pair_metrics,
        "quality": selected_quality,
    }


def ensemble_rembg_rgba(
    input_path: str | Path,
    output_path: str | Path,
    models: tuple[str, ...] = ("u2net", "isnet-general-use"),
    trim: bool = True,
    promptable_recovery: bool = False,
    subject_label: str = "",
) -> dict[str, Any]:
    """Run independent foreground models and publish the best complete mask.

    In quality mode a box-prompted SAM2 candidate is added as a genuinely
    independent completeness witness.  U2Net and ISNet often agree on the
    same incomplete silhouette (cup handles, straps, thin stems), so their
    agreement alone is not evidence that the product is complete.
    """
    candidates = {
        model: _rembg_candidate(input_path, model)
        for model in models
    }
    promptable_details: dict[str, Any] = {
        "requested": bool(promptable_recovery),
        "status": "not_requested",
    }
    if promptable_recovery:
        scratch = Path(output_path).with_suffix(".sam2-recovery.tmp.png")
        try:
            preliminary_model, _ = _select_rembg_candidate(
                candidates, subject_label=subject_label,
            )
            alpha = candidates[preliminary_model][..., 3]
            rows, columns = np.where(alpha > 4)
            if not len(rows):
                raise ContractError("Cannot prompt SAM2 from an empty matte")
            height, width = alpha.shape
            span = max(
                int(columns.max() - columns.min() + 1),
                int(rows.max() - rows.min() + 1),
            )
            padding = max(12, int(round(span * 0.035)))
            box = (
                max(0, int(columns.min()) - padding),
                max(0, int(rows.min()) - padding),
                min(width, int(columns.max()) + padding + 1),
                min(height, int(rows.max()) + padding + 1),
            )
            promptable_details = sam2_box_rgba(
                input_path, scratch, box, trim=False,
            )
            candidates["sam2-box-recovery"] = load_rgba(scratch)
            promptable_details.update({
                "requested": True,
                "status": "success",
                "box": list(box),
            })
        except Exception as error:
            # Promptable recovery is a quality witness, not a single point of
            # failure.  Preserve its failure visibly so callers cannot mistake
            # the two-model fallback for a three-model completeness check.
            promptable_details = {
                "requested": True,
                "status": "failed",
                "reason": str(error),
            }
        finally:
            scratch.unlink(missing_ok=True)
    selected_model, details = _select_rembg_candidate(
        candidates, subject_label=subject_label,
    )
    details["promptable_recovery"] = promptable_details
    rgba = candidates[selected_model]
    # A promptable mask is already restricted by an object box.  Applying the
    # scene semantic allow-list again can classify a metallic handle as table
    # or wall and delete it—the exact failure the third witness recovered.
    if selected_model == "sam2-box-recovery":
        refined_rgba, refined = rgba, False
    else:
        refined_rgba, refined = refine_alpha_with_semantics(rgba)
    if refined:
        rgba = refined_rgba
        # Re-score after semantic restriction but retain the independent-model
        # completeness evidence.  A refined result cannot regain lost trust.
        refined_structure = alpha_quality(
            trim_transparent(rgba), subject_label=subject_label,
        )
        quality = dict(details["quality"])
        quality["structure_score"] = float(refined_structure["score"])
        quality["score"] = float(min(
            quality["completeness_score"],
            quality["structure_score"],
        ))
        quality["verdict"] = (
            "reject"
            if refined_structure["verdict"] == "reject"
            else ("pass" if quality["score"] >= 90.0 else "review")
        )
        quality["semantic_refinement"] = True
        details["quality"] = quality
    details["semantic_refinement"] = refined
    result = trim_transparent(rgba) if trim else rgba
    save_png(output_path, result)
    return details


def crop_normalized_roi(
    rgb: np.ndarray,
    roi: list[float] | tuple[float, float, float, float],
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop an operator-selected normalized box without resampling source RGB.

    The UI sends ``[x0, y0, x1, y1]`` in orientation-corrected image
    coordinates.  Sorting permits drawing in any direction; clipping handles
    a pointer released just outside the displayed image.  The selected source
    pixels are copied verbatim, so this step cannot redraw product texture.
    """
    if len(roi) != 4:
        raise ContractError("Selection box must contain [x0,y0,x1,y1]")
    values = np.asarray(roi, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ContractError("Selection box contains an invalid coordinate")
    x0n, x1n = sorted((float(values[0]), float(values[2])))
    y0n, y1n = sorted((float(values[1]), float(values[3])))
    x0n, y0n = np.clip([x0n, y0n], 0.0, 1.0)
    x1n, y1n = np.clip([x1n, y1n], 0.0, 1.0)
    height, width = rgb.shape[:2]
    x0 = int(np.floor(x0n * width))
    y0 = int(np.floor(y0n * height))
    x1 = int(np.ceil(x1n * width))
    y1 = int(np.ceil(y1n * height))
    x0, y0 = max(0, min(width - 1, x0)), max(0, min(height - 1, y0))
    x1, y1 = max(x0 + 1, min(width, x1)), max(y0 + 1, min(height, y1))
    if x1 - x0 < 16 or y1 - y0 < 16:
        raise ContractError("Selection box is too small; drag a box around the complete object")
    return rgb[y0:y1, x0:x1].copy(), (x0, y0, x1, y1)


def sam2_box_rgba(
    input_path: str | Path,
    output_path: str | Path,
    box: tuple[int, int, int, int],
    trim: bool = True,
) -> dict[str, Any]:
    """Extract one object with an operator box and preserve its source RGB.

    Unlike a mere crop, the box is passed to SAM2 as an object prompt.  This
    distinguishes a foreground vase from the book directly behind it.  SAM2
    contributes only a mask; every retained RGB pixel comes from the
    orientation-corrected source photograph.
    """
    import torch
    from PIL import Image

    rgb = load_background(input_path)
    height, width = rgb.shape[:2]
    x0, y0, x1, y1 = box
    if (
        x0 < 0 or y0 < 0 or x1 <= x0 + 1 or y1 <= y0 + 1
        or x1 > width or y1 > height
    ):
        raise ContractError("SAM2 selection box must lie inside the input image")
    processor, model = _sam2_box_session()
    inputs = processor(
        images=Image.fromarray(rgb, "RGB"),
        input_boxes=[[[x0, y0, x1, y1]]],
        return_tensors="pt",
    )
    with torch.inference_mode():
        outputs = model(**inputs)
    masks = processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"],
    )[0]
    candidates = masks.reshape(-1, masks.shape[-2], masks.shape[-1]).numpy() > 0
    scores = outputs.iou_scores.detach().cpu().numpy().reshape(-1)
    if len(candidates) == 0:
        raise ContractError("SAM2 did not return a selection mask")

    # Prefer model confidence, but penalize masks that escape far beyond the
    # reviewer box.  A small escape allowance is important for anti-aliased
    # leaves and handles drawn close to the box.
    box_mask = np.zeros((height, width), bool)
    margin = max(3, int(round(max(x1 - x0, y1 - y0) * 0.035)))
    box_mask[max(0, y0 - margin):min(height, y1 + margin),
             max(0, x0 - margin):min(width, x1 + margin)] = True
    objectives: list[float] = []
    candidate_metrics: list[dict[str, Any]] = []
    box_area = float(max(1, (x1 - x0) * (y1 - y0)))
    box_height = float(y1 - y0)
    for index, candidate in enumerate(candidates):
        area = max(1, int(candidate.sum()))
        outside_share = float(np.count_nonzero(candidate & ~box_mask) / area)
        component_count, _, component_stats, _ = cv2.connectedComponentsWithStats(
            candidate.astype(np.uint8), 8,
        )
        if component_count > 1:
            largest_component_index = 1 + int(
                np.argmax(component_stats[1:, cv2.CC_STAT_AREA])
            )
            largest_y = int(
                component_stats[largest_component_index, cv2.CC_STAT_TOP]
            )
            largest_height = int(
                component_stats[largest_component_index, cv2.CC_STAT_HEIGHT]
            )
            largest_area = int(
                component_stats[largest_component_index, cv2.CC_STAT_AREA]
            )
        else:
            largest_y, largest_height, largest_area = height, 0, 0
        spans_box_height = bool(
            largest_height
            and float(largest_y) <= y0 + box_height * 0.28
            and float(largest_y + largest_height - 1) >= y1 - box_height * 0.20
        )
        # SAM's highest-IoU candidate can be a larger "object + photographed
        # shadow/background" mask.  Prefer the most compact candidate that
        # still spans the operator's complete top-to-base selection.  This
        # keeps a vase and its flowers together while rejecting the wall/table
        # halo around a dark plant.
        area_pressure = min(1.5, area / box_area)
        # If every candidate is incomplete the equal penalty has no effect;
        # when at least one full-height candidate exists this decisively
        # prevents a flower-only or lid-only mask from winning on compactness.
        incomplete_penalty = 2.0 if not spans_box_height else 0.0
        objective = (
            float(scores[index])
            - outside_share * 0.45
            - area_pressure * 0.90
            - incomplete_penalty
        )
        objectives.append(objective)
        candidate_metrics.append({
            "index": int(index),
            "predicted_iou": float(scores[index]),
            "objective": float(objective),
            "area_ratio": float(area / box_area),
            "outside_share": outside_share,
            "spans_box_height": spans_box_height,
            "largest_component_share": float(largest_area / area),
        })
    selected_index = int(np.argmax(objectives))
    binary = candidates[selected_index].astype(np.uint8)

    # Box-prompt candidates occasionally contain a few isolated background
    # flecks.  The intended object is the dominant connected component; keep
    # only materially sized pieces so detached product details can survive.
    # Break narrow accidental bridges before ranking components.  The
    # original mask is then recovered inside a small dilation of valid cores,
    # so legitimate fine leaves and handles are retained while isolated book
    # print or shelf flecks disappear.
    core = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )
    core_count, core_labels, core_stats, _ = cv2.connectedComponentsWithStats(core, 8)
    if core_count <= 1:
        raise ContractError("SAM2 produced an empty selection mask")
    largest_core = int(core_stats[1:, cv2.CC_STAT_AREA].max())
    valid_core = np.zeros_like(binary, bool)
    for index in range(1, core_count):
        if int(core_stats[index, cv2.CC_STAT_AREA]) >= max(16, int(largest_core * 0.001)):
            valid_core |= core_labels == index
    allowed = cv2.dilate(
        valid_core.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    ).astype(bool)
    binary = (binary.astype(bool) & allowed).astype(np.uint8)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        raise ContractError("SAM2 produced an empty selection mask")
    largest = int(stats[1:, cv2.CC_STAT_AREA].max())
    retained = np.zeros_like(binary, bool)
    for index in range(1, count):
        if int(stats[index, cv2.CC_STAT_AREA]) >= max(16, int(largest * 0.0025)):
            retained |= labels == index

    # SAM2 can recover a thin handle that the semantic matting models miss,
    # but its full-resolution boundary sometimes contains one- or two-pixel
    # bites.  Close only defects at that microscopic scale.  The bounded
    # kernel is deliberately far smaller than a real handle opening, so the
    # product's negative space remains transparent instead of becoming a
    # filled-in silhouette.
    short_side = min(height, width)
    close_size = int(np.clip(round(short_side * 0.0018), 3, 9))
    if close_size % 2 == 0:
        close_size += 1
    retained = cv2.morphologyEx(
        retained.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_size, close_size),
        ),
    ).astype(bool)
    alpha = cv2.GaussianBlur(
        retained.astype(np.uint8) * 255, (0, 0), 0.62,
    )
    rgba = np.dstack((rgb, alpha))
    result = trim_transparent(rgba, padding=5) if trim else rgba
    save_png(output_path, result)
    return {
        "candidate_count": int(len(candidates)),
        "selected_candidate": selected_index,
        "predicted_iou": float(scores[selected_index]),
        "candidates": candidate_metrics,
        "boundary_close_kernel": close_size,
    }


def solidify_semantic_opaque_panel(
    rgba: np.ndarray,
    subject_label: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fill false internal alpha holes for a verified solid panel product.

    Segmenters often treat printed dark graphics on a pale clock face, book or
    plaque as separate background holes.  For these product classes the entire
    bounded face is physically opaque.  This repair changes alpha only and
    reveals the preserved source RGB beneath it; it never paints or regenerates
    product pixels.

    The route is intentionally narrow.  Transparent, open, handled, woven or
    perforated objects keep their real negative space and are never solidified.
    """

    label = " ".join(str(subject_label).lower().replace("_", "-").split())
    allowed_tokens = (
        "clock",
        "book",
        "closed box",
        "package",
        "packaging",
        "plaque",
        "solid sign",
        "display board",
        "framed print",
        "framed picture",
        "birdhouse",
        "bird house",
    )
    blocked_tokens = (
        "transparent",
        "clear glass",
        "glass frame",
        "empty frame",
        "open frame",
        "wire",
        "mesh",
        "woven",
        "basket",
        "rack",
        "handle",
        "chair",
        "stool",
        "plant",
        "flower",
    )
    details: dict[str, Any] = {
        "schema_version": "semantic-opaque-panel-v1",
        "applied": False,
        "subject_label": str(subject_label),
        "rgb_policy": "preserved_source_rgb_only",
        "repair_expansion_verdict": "not_applicable",
    }
    panel_route = any(token in label for token in allowed_tokens)
    compact_solid_route = _solid_opaque_topology_expected(subject_label)
    if not panel_route and not compact_solid_route:
        details["reason"] = "semantic_class_not_a_solid_panel"
        return rgba, details
    if any(token in label for token in blocked_tokens):
        details["reason"] = "negative_space_or_transparency_must_be_preserved"
        return rgba, details

    source = np.asarray(rgba, dtype=np.uint8)
    if source.ndim != 3 or source.shape[2] != 4:
        details["reason"] = "invalid_rgba_shape"
        return rgba, details
    binary = (source[..., 3] > 4).astype(np.uint8)
    foreground = int(binary.sum())
    if foreground < 128:
        details["reason"] = "empty_or_tiny_alpha"
        return rgba, details

    # A compact opaque figurine can contain a tiny fully enclosed alpha punch
    # where a segmenter mistook a specular highlight for white background.  Do
    # not use a hull here: that would destroy real gaps between limbs or other
    # negative space.  Fill only microscopic enclosed islands, with both a
    # per-hole and total-area cap.  The limits are expressed relative to the
    # confident foreground so they behave consistently from previews to 4K.
    # RGB is never synthesised; the operation merely reveals the original
    # source pixel that already exists below alpha.
    if compact_solid_route and not panel_route:
        stable_binary = (source[..., 3] >= 250).astype(np.uint8)
        stable_foreground = int(stable_binary.sum())
        padded_background = np.pad(
            (stable_binary == 0).astype(np.uint8), 1, constant_values=1,
        )
        exterior = padded_background.copy()
        cv2.floodFill(exterior, None, (0, 0), 2)
        holes = (
            (padded_background == 1) & (exterior != 2)
        )[1:-1, 1:-1].astype(np.uint8)
        hole_count, hole_labels, hole_stats, _ = cv2.connectedComponentsWithStats(
            holes, 8,
        )
        per_hole_cap = max(12, int(round(stable_foreground * 0.0015)))
        total_cap = max(24, int(round(stable_foreground * 0.0030)))
        fill_mask = np.zeros_like(binary, dtype=np.uint8)
        filled_components = 0
        for index in range(1, hole_count):
            area = int(hole_stats[index, cv2.CC_STAT_AREA])
            if area <= per_hole_cap and int(fill_mask.sum()) + area <= total_cap:
                fill_mask[hole_labels == index] = 1
                filled_components += 1
        filled_pixels = int(fill_mask.sum())
        details.update({
            "geometry_model": "bounded_enclosed_microhole_fill",
            "internal_hole_count_before": int(max(0, hole_count - 1)),
            "filled_hole_components": int(filled_components),
            "filled_hole_pixels": filled_pixels,
            "per_hole_cap_pixels": int(per_hole_cap),
            "total_fill_cap_pixels": int(total_cap),
            "stable_foreground_pixels": int(stable_foreground),
            "requires_transparent_audit_border": False,
        })
        if filled_pixels <= 0:
            details["reason"] = "no_bounded_internal_microholes"
            return rgba, details
        repaired = source.copy()
        repaired[..., 3][fill_mask > 0] = 255
        details.update({
            "applied": True,
            "reason": "solid_opaque_internal_microholes_filled",
            "repair_expansion_verdict": "pass",
            "repair_confidence_score": float(np.clip(
                99.0 - 100.0 * filled_pixels / max(1, foreground),
                96.0,
                99.0,
            )),
            "expansion_ratio": float((foreground + filled_pixels) / foreground),
        })
        return repaired, details

    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    component_areas = sorted(
        (int(stats[index, cv2.CC_STAT_AREA]) for index in range(1, count)),
        reverse=True,
    )
    largest_share = float(component_areas[0] / foreground) if component_areas else 0.0
    rows, columns = np.where(binary > 0)
    points = np.column_stack((columns, rows)).astype(np.int32)
    hull = cv2.convexHull(points.reshape(-1, 1, 2))
    filled = np.zeros_like(binary)
    cv2.fillConvexPoly(filled, hull.reshape(-1, 2), 1)
    geometry_model = "convex_hull"

    # Birdhouses are solid painted enclosures, but their roof eaves and body
    # are commonly narrower on different scanlines.  A global convex hull
    # joins the wide eave tips to the lower base and reveals triangular strips
    # of source background.  Fill the observed left/right structural envelope
    # independently on every row instead.  Clamp strong one-sided excursions
    # around the robust centre so a thin hanging chain remains itself without
    # pulling a wedge of background into the opaque body.
    if "birdhouse" in label or "bird house" in label:
        # The dominant connected component already traces the photographed
        # roof, under-eave concavity, cabinet walls and base.  Filling its
        # *external contour* restores the opaque painted face without joining
        # roof tips to the base.  The previous per-row min/max envelope did
        # exactly that and exposed large triangles of the source tabletop.
        component_count, component_labels, component_stats, _ = (
            cv2.connectedComponentsWithStats(binary, 8)
        )
        if component_count > 1:
            largest_index = 1 + int(np.argmax(
                component_stats[1:, cv2.CC_STAT_AREA],
            ))
            dominant = (component_labels == largest_index).astype(np.uint8)
            contours, _ = cv2.findContours(
                dominant, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
            )
            contour_fill = np.zeros_like(binary)
            if contours:
                cv2.drawContours(
                    contour_fill, contours, -1, 1, thickness=cv2.FILLED,
                )
            if int(np.count_nonzero(contour_fill)) > foreground:
                filled = np.maximum(contour_fill, binary)
                geometry_model = "birdhouse_dominant_external_contour"

    # A house-shaped clock has a triangular roof and a long, nearly vertical
    # lower cabinet.  Cloud mattes commonly recover the roof and upper side
    # rails, lose most of the printed face, then recover only part of the
    # bottom rail.  A global convex hull consequently joins the surviving
    # bottom fragment to the right rail with a large diagonal/rounded cut.
    # Infer the two cabinet rails only from trustworthy wide upper rows and
    # extend those rails to the observed bottom.  This is still an alpha-only
    # repair: every revealed colour pixel remains the source photograph.
    if "clock" in label and ("house" in label or "house-shaped" in label):
        row_left = np.full(binary.shape[0], np.nan, dtype=np.float32)
        row_right = np.full(binary.shape[0], np.nan, dtype=np.float32)
        row_span = np.zeros(binary.shape[0], dtype=np.float32)
        for row_index in range(binary.shape[0]):
            columns_here = np.flatnonzero(binary[row_index] > 0)
            if columns_here.size:
                row_left[row_index] = float(columns_here[0])
                row_right[row_index] = float(columns_here[-1])
                row_span[row_index] = float(columns_here[-1] - columns_here[0] + 1)

        max_span = float(np.nanmax(row_span)) if row_span.size else 0.0
        candidate_rows = np.flatnonzero(
            (row_span >= max(8.0, max_span * 0.84))
            & (np.arange(binary.shape[0]) >= int(binary.shape[0] * 0.12))
            & (np.arange(binary.shape[0]) <= int(binary.shape[0] * 0.52))
        )
        if candidate_rows.size >= 12:
            # Use a robust median rather than an extreme pixel, so one stray
            # cloud-alpha whisker cannot enlarge the product silhouette.
            left_rail = int(round(float(np.nanmedian(row_left[candidate_rows]))))
            right_rail = int(round(float(np.nanmedian(row_right[candidate_rows]))))
            # Start the extrapolated cabinet well below the roof/eave.  Using
            # the first wide roof row would create a rectangular shoulder
            # outside the sloping roof; the median wide row is already inside
            # the true vertical body and therefore unions cleanly with it.
            wall_top = int(round(float(np.median(candidate_rows))))
            bottom_row = int(rows.max())
            rail_width = int(right_rail - left_rail + 1)
            if (
                rail_width >= int(binary.shape[1] * 0.55)
                and bottom_row > wall_top + int(binary.shape[0] * 0.30)
            ):
                lower_panel = np.zeros_like(binary)
                cv2.fillConvexPoly(
                    lower_panel,
                    np.asarray(
                        [
                            [left_rail, wall_top],
                            [right_rail, wall_top],
                            [right_rail, bottom_row],
                            [left_rail, bottom_row],
                        ],
                        dtype=np.int32,
                    ),
                    1,
                )
                filled = np.maximum(filled, lower_panel)
                geometry_model = "house_clock_roof_hull_plus_vertical_cabinet"
    filled_area = int(filled.sum())
    canvas_area = int(binary.shape[0] * binary.shape[1])
    expansion_ratio = float(filled_area / max(1, foreground))
    canvas_ratio = float(filled_area / max(1, canvas_area))
    fragmented = bool(
        count - 1 >= 3
        or largest_share < 0.92
        or expansion_ratio >= 1.18
    )
    if not fragmented:
        details.update({
            "reason": "alpha_is_already_structurally_complete",
            "component_count": int(max(0, count - 1)),
            "largest_component_share": largest_share,
            "expansion_ratio": expansion_ratio,
        })
        return rgba, details
    # The real birdhouse regression needs roughly 2.90x recovery because both
    # foreground networks kept the outer rails but lost most of the painted
    # front face.  A generic 1.8x cap would therefore reject the visually
    # correct external-contour reconstruction.  Keep a narrow class-specific
    # allowance while preventing an almost empty matte from turning an entire
    # crop into foreground.
    max_expansion_ratio = (
        3.25 if "birdhouse" in label or "bird house" in label else 4.0
    )
    if (
        filled_area <= foreground
        or expansion_ratio > max_expansion_ratio
        or canvas_ratio < 0.20
        or canvas_ratio > 0.97
    ):
        details.update({
            "reason": "solid_panel_hull_failed_geometry_gate",
            "component_count": int(max(0, count - 1)),
            "largest_component_share": largest_share,
            "expansion_ratio": expansion_ratio,
            "max_expansion_ratio": max_expansion_ratio,
            "repair_expansion_verdict": "reject",
            "canvas_ratio": canvas_ratio,
        })
        return rgba, details

    # Preserve any confident original antialiasing, while the filled interior
    # becomes fully opaque.  Only the newly created outer boundary receives a
    # sub-pixel blur; printed texture and text remain untouched source RGB.
    short_side = min(binary.shape)
    blur_sigma = float(np.clip(short_side * 0.00055, 0.45, 0.85))
    repaired_alpha = cv2.GaussianBlur(
        filled.astype(np.uint8) * 255,
        (0, 0),
        blur_sigma,
    )
    repaired_alpha = np.maximum(repaired_alpha, source[..., 3])
    repaired = source.copy()
    repaired[..., 3] = repaired_alpha
    details.update({
        "applied": True,
        "reason": "fragmented_opaque_panel_alpha_solidified",
        "component_count_before": int(max(0, count - 1)),
        "largest_component_share_before": largest_share,
        "foreground_pixels_before": foreground,
        "foreground_pixels_after": int(np.count_nonzero(repaired_alpha > 4)),
        "expansion_ratio": expansion_ratio,
        "max_expansion_ratio": max_expansion_ratio,
        "repair_expansion_verdict": "pass",
        # This is confidence in an alpha-only semantic repair, not a visual
        # ground-truth score.  It intentionally decreases with inferred area
        # and is later intersected with independent-model completeness.
        "repair_confidence_score": float(np.clip(
            98.0 - max(0.0, expansion_ratio - 1.0) * 2.5,
            90.0,
            98.0,
        )),
        "canvas_ratio": canvas_ratio,
        "boundary_blur_sigma": blur_sigma,
        "geometry_model": geometry_model,
        "requires_transparent_audit_border": True,
    })
    return repaired, details


def remove_attached_bottom_surface_residue(
    rgba: np.ndarray,
    subject_label: str = "",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Remove a small, low-saturation photographed support tail.

    Local foreground models sometimes keep the source tabletop immediately
    below a seated figurine or shared display base.  It remains connected to
    the product, so component filtering cannot remove it.  This conservative
    detector only acts on a late, sustained collapse of the row span whose
    colour saturation also falls sharply relative to the real base above.
    Clear/handled/open products are excluded and RGB is never modified.
    """
    details: dict[str, Any] = {
        "schema_version": "bottom-surface-residue-v1",
        "applied": False,
        "verdict": "pass",
        "subject_label": str(subject_label),
        "rgb_policy": "preserved_source_rgb_only",
    }
    label = " ".join(str(subject_label).lower().replace("_", " ").split())
    blocked = (
        "transparent", "glass", "wire", "cord", "chain", "stem",
        "hanging", "pendant", "tassel",
    )
    if any(token in label for token in blocked):
        details["reason"] = "semantic_negative_space_or_trailing_part"
        return rgba, details
    source = np.asarray(rgba, dtype=np.uint8)
    if source.ndim != 3 or source.shape[2] != 4:
        details["reason"] = "invalid_rgba_shape"
        details["verdict"] = "reject"
        return rgba, details
    binary = source[..., 3] >= 128
    rows, columns = np.where(binary)
    if len(rows) < 128:
        details["reason"] = "empty_or_tiny_alpha"
        details["verdict"] = "reject"
        return rgba, details
    y0, y1 = int(rows.min()), int(rows.max())
    object_height = y1 - y0 + 1
    if object_height < 48:
        details["reason"] = "subject_too_short"
        details["verdict"] = "reject"
        return rgba, details
    spans = np.zeros(object_height, np.float32)
    row_saturation = np.full(object_height, np.nan, np.float32)
    hsv = cv2.cvtColor(source[..., :3], cv2.COLOR_RGB2HSV)
    for relative_y, row_index in enumerate(range(y0, y1 + 1)):
        columns_here = np.flatnonzero(binary[row_index])
        if columns_here.size:
            spans[relative_y] = float(columns_here[-1] - columns_here[0] + 1)
            row_saturation[relative_y] = float(np.median(
                hsv[row_index, columns_here, 1],
            ))
    max_span = float(np.max(spans))
    reference_slice = row_saturation[
        int(object_height * 0.80):max(
            int(object_height * 0.81), int(object_height * 0.91),
        )
    ]
    reference_slice = reference_slice[np.isfinite(reference_slice)]
    reference_saturation = (
        float(np.median(reference_slice)) if reference_slice.size else 0.0
    )
    transition: int | None = None
    search_start = int(object_height * 0.88)
    sustain = max(6, int(round(object_height * 0.018)))
    for relative_y in range(search_start, max(search_start, object_height - sustain)):
        trailing_spans = spans[relative_y:]
        local_saturation = row_saturation[
            relative_y:min(object_height, relative_y + sustain)
        ]
        local_saturation = local_saturation[np.isfinite(local_saturation)]
        if not local_saturation.size:
            continue
        if (
            spans[relative_y] <= max_span * 0.36
            and float(np.mean(trailing_spans <= max_span * 0.48)) >= 0.96
            and reference_saturation >= 55.0
            and float(np.median(local_saturation))
            <= reference_saturation * 0.78
        ):
            transition = relative_y
            break
    if transition is None:
        details["reason"] = "no_sustained_low_saturation_tail"
        return rgba, details
    cut_y = y0 + transition
    tail = binary.copy()
    tail[:cut_y] = False
    foreground = max(1, int(np.count_nonzero(binary)))
    tail_share = float(np.count_nonzero(tail) / foreground)
    tail_height_share = float((y1 - cut_y + 1) / object_height)
    if not (0.002 <= tail_share <= 0.060 and tail_height_share <= 0.12):
        details.update({
            "reason": "tail_failed_conservative_geometry_gate",
            "verdict": "review",
            "tail_pixel_share": tail_share,
            "tail_height_share": tail_height_share,
        })
        return rgba, details
    # The narrow tail is often only the lowest tip of a larger piece of clear
    # adhesive, shelf reflection or photographed support that reaches a little
    # above the row transition.  Use the tail itself as a colour seed and grow
    # only through nearby, low-chroma alpha pixels in the bottom band.  This
    # removes the grey triangular patch behind a figurine without erasing its
    # coloured shoes/base or applying a broad rectangular crop.
    tail_pixels = source[..., :3][tail]
    support_above_cut = np.zeros_like(binary)
    if tail_pixels.size:
        lab = cv2.cvtColor(source[..., :3], cv2.COLOR_RGB2LAB)
        tail_lab = np.median(lab[tail], axis=0).astype(np.float32)
        lab_distance = np.linalg.norm(
            lab.astype(np.float32) - tail_lab[None, None, :],
            axis=2,
        )
        tail_saturation = float(np.median(hsv[..., 1][tail]))
        saturation_limit = float(np.clip(
            max(42.0, tail_saturation * 1.55), 42.0, 84.0,
        ))
        band_top = max(y0, cut_y - int(round(object_height * 0.085)))
        band = np.zeros_like(binary)
        band[band_top:y1 + 1] = True
        support_colour = (
            binary
            & band
            & (hsv[..., 1] <= saturation_limit)
            & (lab_distance <= 34.0)
        )
        seeded = support_colour | tail
        join_radius = max(2, int(round(min(source.shape[:2]) * 0.004)))
        joined = cv2.morphologyEx(
            seeded.astype(np.uint8),
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (join_radius * 2 + 1, join_radius * 2 + 1),
            ),
        )
        component_count, component_labels = cv2.connectedComponents(joined, 8)
        seeded_labels = np.unique(component_labels[tail])
        seeded_labels = seeded_labels[seeded_labels > 0]
        if seeded_labels.size:
            connected_support = np.isin(component_labels, seeded_labels)
            support_above_cut = support_colour & connected_support

    repaired = source.copy()
    remove_mask = tail | support_above_cut
    # Include the anti-aliased fringe of the selected support, but never grow
    # outside the original nonzero alpha or the conservative bottom band.
    if np.any(remove_mask):
        fringe = cv2.dilate(remove_mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        remove_mask |= fringe & (source[..., 3] > 4)
    repaired[..., 3][remove_mask] = 0
    details.update({
        "applied": True,
        "verdict": "pass",
        "reason": "attached_low_saturation_source_surface_removed",
        "cut_row": int(cut_y),
        "cut_row_relative": float(transition / object_height),
        "tail_pixel_share": tail_share,
        "tail_height_share": tail_height_share,
        "reference_saturation": reference_saturation,
        "support_pixels_above_cut": int(np.count_nonzero(
            support_above_cut & (np.indices(binary.shape)[0] < cut_y)
        )),
    })
    return repaired, details


def refine_green_product_shadows(rgba: np.ndarray) -> tuple[np.ndarray, bool]:
    """Remove photographed wall/table shadows around foliage products.

    Promptable segmentation can join a potted plant to its brown cast shadow
    because they touch in the source photograph.  For a confidently green
    upper product, retain green foliage (plus a narrow neighbourhood) and the
    centre-connected horizontal run containing the pot/base on every row.
    The rule is deliberately disabled for non-green products, so cup handles,
    text and arbitrary dark products are unaffected.
    """
    alpha = rgba[..., 3]
    foreground = alpha > 4
    foreground_count = int(foreground.sum())
    if foreground_count < 128:
        return rgba, False
    hsv = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2HSV)
    green = (
        (hsv[..., 0] >= 26)
        & (hsv[..., 0] <= 80)
        & (hsv[..., 1] >= 55)
        & (hsv[..., 2] >= 18)
        & foreground
    )
    green_share = float(green.sum() / foreground_count)
    if green_share < 0.20:
        return rgba, False

    height, width = foreground.shape
    foliage = cv2.dilate(
        green.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    ).astype(bool) & foreground

    # Locate the end of the wide green crown.  Below this transition only the
    # centre-bearing pot/base run is valid; photographed side shadows are not.
    green_per_row = green.sum(axis=1).astype(np.float32)
    window = min(21, max(3, height // 8 * 2 + 1))
    rolling = np.convolve(
        green_per_row, np.ones(window, np.float32) / window, mode="same",
    )
    crown_peak = int(np.argmax(rolling))
    low_threshold = width * 0.08
    consecutive = min(20, max(5, height // 20))
    body_start = height // 2
    search_stop = min(height - consecutive, int(height * 0.72))
    for y in range(crown_peak, max(crown_peak, search_stop)):
        if np.all(rolling[y:y + consecutive] < low_threshold):
            body_start = y
            break

    centre = width // 2
    central_body = np.zeros_like(foreground)
    for y, row in enumerate(foreground):
        if y < body_start:
            continue
        columns = np.flatnonzero(row)
        if not len(columns):
            continue
        cuts = np.flatnonzero(np.diff(columns) > 1)
        starts = np.r_[0, cuts + 1]
        ends = np.r_[cuts, len(columns) - 1]
        runs = [
            (int(columns[start]), int(columns[end]))
            for start, end in zip(starts, ends)
        ]
        left, right = min(
            runs,
            key=lambda run: (
                0
                if run[0] <= centre <= run[1]
                else min(abs(centre - run[0]), abs(centre - run[1]))
            ),
        )
        central_body[y, max(0, left - 3):min(width, right + 4)] = True

    central_body = cv2.morphologyEx(
        central_body.astype(np.uint8),
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    ).astype(bool)
    allowed = foreground & (foliage | central_body)
    retained_share = float(allowed.sum() / foreground_count)
    if retained_share < 0.88 or retained_share > 0.9995:
        return rgba, False
    soft_allow = cv2.GaussianBlur(
        allowed.astype(np.uint8) * 255, (0, 0), 0.55,
    )
    refined = rgba.copy()
    refined[..., 3] = np.minimum(alpha, soft_allow).astype(np.uint8)
    return refined, True


def keep_rectangle(rgba: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    """Remove incidental props outside a reviewer-approved product rectangle."""
    x, y, width, height = rect
    if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > rgba.shape[1] or y + height > rgba.shape[0]:
        raise ContractError("keep rectangle must lie inside the RGBA canvas")
    output = rgba.copy()
    retained = np.zeros(rgba.shape[:2], dtype=bool)
    retained[y:y + height, x:x + width] = True
    output[~retained, 3] = 0
    return output


def remove_dark_foreground(rgba: np.ndarray, threshold: int) -> np.ndarray:
    """Optional reviewer-selected refinement for black display props/stands."""
    if not 0 <= threshold <= 254:
        raise ContractError("dark foreground threshold must be in [0,254]")
    output = rgba.copy()
    luma = (0.2126 * output[..., 0] + 0.7152 * output[..., 1] + 0.0722 * output[..., 2])
    output[luma < threshold, 3] = 0
    return output


def bright_foreground_rgba(input_path: str | Path, output_path: str | Path, rect: tuple[int, int, int, int], threshold: int, trim: bool = True) -> None:
    """Deterministic extraction for a reviewed bright product on a dark set.

    This is useful for products photographed on black display stands: the
    reviewer supplies the product-only rectangle and brightness threshold, so
    the stand cannot leak into the alpha.  It is not a general segmentation
    substitute.
    """
    rgb = load_background(input_path)
    x, y, width, height = rect
    if x < 0 or y < 0 or width <= 1 or height <= 1 or x + width > rgb.shape[1] or y + height > rgb.shape[0]:
        raise ContractError("bright extraction rectangle must lie inside the input image")
    if not 0 <= threshold <= 254:
        raise ContractError("brightness threshold must be in [0,254]")
    luma = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    alpha = np.zeros(rgb.shape[:2], np.uint8)
    alpha[y:y + height, x:x + width] = (luma[y:y + height, x:x + width] >= threshold).astype(np.uint8) * 255
    # Fill small interior breaks but do not expand into the dark support stand.
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 0.55)
    rgba = np.dstack((rgb, alpha))
    if not np.any(alpha > 4):
        raise ContractError("Bright extraction produced an empty alpha mask")
    save_png(output_path, trim_transparent(rgba) if trim else rgba)


def grabcut_rgba(input_path: str | Path, output_path: str | Path, rect: tuple[int, int, int, int], iterations: int = 5, trim: bool = True, keep_rect: tuple[int, int, int, int] | None = None, remove_dark: int | None = None) -> None:
    """Create a deterministic RGBA cutout from a reviewed foreground box.

    The operator-provided rectangle is intentional: automatic whole-image
    segmentation silently deleting product parts is worse than a visible
    validation failure.  Production can replace this with RunningHub's alpha
    output, which uses the same RGBA contract.
    """
    rgb = load_background(input_path)
    height, width = rgb.shape[:2]
    x, y, box_w, box_h = rect
    if x < 0 or y < 0 or box_w <= 1 or box_h <= 1 or x + box_w > width or y + box_h > height:
        raise ContractError("GrabCut rectangle must lie inside the input image")
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    mask = np.zeros((height, width), np.uint8)
    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, mask, rect, background_model, foreground_model, iterations, cv2.GC_INIT_WITH_RECT)
    alpha = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    # A small feather avoids a hard binary seam while retaining deterministic
    # product pixels.  It does not change RGB values.
    alpha = cv2.GaussianBlur(alpha, (0, 0), 0.65)
    rgba = np.dstack((rgb, alpha))
    if not np.any(alpha > 4):
        raise ContractError("Segmentation produced an empty alpha mask")
    if keep_rect:
        rgba = keep_rectangle(rgba, keep_rect)
    if remove_dark is not None:
        rgba = remove_dark_foreground(rgba, remove_dark)
    save_png(output_path, trim_transparent(rgba) if trim else rgba)


def _rembg_candidate(
    input_path: str | Path,
    model: str = "u2net",
) -> np.ndarray:
    """Return a full-canvas model alpha over unchanged source RGB."""
    try:
        from rembg import remove
    except ImportError as error:
        raise ContractError("rembg is not installed; install it or use --method grabcut") from error
    rgb = load_background(input_path)
    # `remove` returns an RGBA estimate.  Its RGB foreground may be altered by
    # the model, so only the generated alpha is accepted into this workflow.
    estimated = np.asarray(remove(rgb, session=_rembg_session(model)))
    if estimated.ndim != 3 or estimated.shape[:2] != rgb.shape[:2] or estimated.shape[2] != 4:
        raise ContractError("Matting model did not return a matching RGBA image")
    alpha = estimated[..., 3]
    if not np.any(alpha > 4):
        raise ContractError("Matting model produced an empty alpha mask")
    rgba = np.dstack((rgb, alpha))
    return clean_alpha(rgba)


def rembg_rgba(input_path: str | Path, output_path: str | Path, model: str = "u2net", trim: bool = True, keep_rect: tuple[int, int, int, int] | None = None, remove_dark: int | None = None) -> None:
    """Use a matting model only to obtain alpha; preserve source RGB exactly."""
    rgba = _rembg_candidate(input_path, model)
    if keep_rect:
        rgba = keep_rectangle(rgba, keep_rect)
    if remove_dark is not None:
        rgba = remove_dark_foreground(rgba, remove_dark)
    save_png(output_path, trim_transparent(rgba) if trim else rgba)
