"""Production-oriented local workbench for deterministic product compositing."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import threading
import time
import webbrowser
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.parse import quote

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field
from PIL import Image, ImageOps

from .contracts import Job
from .errors import ContractError
from .cutout import (
    alpha_quality,
    crop_normalized_roi,
    ensemble_rembg_rgba,
    refine_green_product_shadows,
    refine_alpha_with_semantics,
    remove_attached_bottom_surface_residue,
    rembg_rgba,
    sam2_box_rgba,
    solidify_semantic_opaque_panel,
    trim_transparent,
)
from .image_io import load_background, load_mask, load_rgba, save_png
from .minimax_advisor import (
    MiniMaxAdvice,
    MiniMaxNotConfigured,
    MiniMaxTransportError,
    VISUAL_REVIEW_PROMPT_VERSION,
    apply_layout_advice,
    minimax_status,
    reconcile_scene_result,
    request_minimax_advice,
    request_minimax_subject_roi,
    request_minimax_visual_review,
    test_minimax_connection,
)
from .material_profiles import (
    canonical_material_mode,
    is_opaque,
    material_profile,
)
from .material_router import resolve_material_route
from .pipeline import StrictCompositor
from .hybrid_runninghub import RunningHubHybridProvider, runninghub_hybrid_status
from .learned_effects_runninghub import (
    RunningHubLearnedEffectProvider,
    runninghub_learned_effects_status,
)
from .quality import (
    FINAL_GATE_POLICY_VERSION,
    FINAL_GATE_SCHEMA_VERSION,
    VISUAL_REVIEW_SLOT_COUNT,
    FinalGateError,
    aggregate_visual_reviews,
    build_final_gate,
    build_review_bundle,
    validate_dimension_scores,
    verify_deliverable,
    write_final_gate,
    write_json_atomic,
)
from .placement_solver import (
    VISIBLE_STAGING_UI_MIN_RATIO,
    compact_sparse_seeded_layout,
    legalize_placements,
    placement_homography,
    product_geometry,
    refine_seeded_composition,
    solve_aesthetic_layout,
)
from .runninghub import (
    RunningHubClient,
    RunningHubError,
    run_configured_cutout,
    runninghub_status,
)
from .scene import analyze_scene
from .settings import (
    SettingsError,
    get_public_settings,
    get_secret,
    save_integrations,
)

ROOT = Path.cwd().resolve()
WORK = ROOT / "ui_workspace"
SCENE_CACHE = WORK / "scene_cache"
ADVISOR_CACHE = WORK / "minimax_advice"
UPLOADS = WORK / "uploads"
app = FastAPI(title="IdentityGuard Studio", version="3.78")

# Full-quality extraction is expensive.  Serialize jobs so several browser
# clicks cannot run two matte models concurrently and kill the local server.
_SEGMENT_JOB_LOCK = threading.Lock()
_VISUAL_REVIEW_LOCK = threading.Lock()

# A visual-review slot is immutable once MiniMax has returned a valid JSON
# judgement, including a low one.  Only a transport interruption *before* a
# judgement may be retried.  Keep the retry budget deliberately small: this is
# reliability recovery, not score shopping.
VISUAL_REVIEW_MAX_TRANSPORT_ATTEMPTS = 3
VISUAL_REVIEW_TRANSPORT_RETRY_POLICY = "bounded-exponential-jitter-v1"
VISUAL_REVIEW_SLOT_SCHEMA_VERSION = "visual-review-slot-v2"


def _expand_subject_roi_bbox(
    bbox: list[float] | tuple[float, float, float, float],
    *,
    fallback_used: bool = False,
) -> list[float]:
    """Pad a semantic subject box without letting it become a scene crop.

    Vision-language boxes usually fit the visually dominant body and can miss
    a weak transparent handle, a thin display stand, or the actual contact
    base.  The asymmetric bottom pad is intentional: preserving the support
    hardware matters more than retaining empty source space above the object.
    The local saliency fallback is already generously padded, so it receives
    only a small safety margin.
    """

    values = np.asarray(bbox, dtype=np.float64)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ContractError("Subject ROI must be four finite normalized values")
    x0, x1 = sorted((float(values[0]), float(values[2])))
    y0, y1 = sorted((float(values[1]), float(values[3])))
    if min(x0, y0) < 0.0 or max(x1, y1) > 1.0 or x1 <= x0 or y1 <= y0:
        raise ContractError("Subject ROI must be a non-empty normalized xyxy box")
    width = x1 - x0
    height = y1 - y0
    if width * height < 0.004 or width * height > 0.90:
        raise ContractError("Subject ROI has an implausible area")
    if fallback_used:
        pad_x = max(0.006, width * 0.035)
        pad_top = max(0.006, height * 0.035)
        pad_bottom = max(0.010, height * 0.060)
    else:
        pad_x = max(0.008, width * 0.080)
        pad_top = max(0.008, height * 0.070)
        pad_bottom = max(0.014, height * 0.160)
    return [
        float(np.clip(x0 - pad_x, 0.0, 1.0)),
        float(np.clip(y0 - pad_top, 0.0, 1.0)),
        float(np.clip(x1 + pad_x, 0.0, 1.0)),
        float(np.clip(y1 + pad_bottom, 0.0, 1.0)),
    ]


def _prefer_local_cutout_candidate(
    cloud_quality: dict[str, Any],
    local_quality: dict[str, Any],
) -> bool:
    """Choose the local alpha only when it has a material gate advantage.

    The automatic score is not a visual truth metric.  RunningHub remains the
    quality-first default when both candidates pass the same hard gates; a
    local candidate wins only by clearing a gate that the cloud candidate did
    not, or by a sizeable structural margin.  This prevents a deceptively
    clean but incomplete local fragment from replacing a complete cloud matte.
    """

    verdict_rank = {"reject": 0, "review": 1, "pass": 2}
    cloud_rank = verdict_rank.get(str(cloud_quality.get("verdict", "reject")), 0)
    local_rank = verdict_rank.get(str(local_quality.get("verdict", "reject")), 0)
    if local_rank != cloud_rank:
        return local_rank > cloud_rank
    if bool(cloud_quality.get("multi_object_leakage")) != bool(
        local_quality.get("multi_object_leakage")
    ):
        return not bool(local_quality.get("multi_object_leakage"))
    if bool(cloud_quality.get("opposite_edge_attachment")) != bool(
        local_quality.get("opposite_edge_attachment")
    ):
        return not bool(local_quality.get("opposite_edge_attachment"))
    cloud_score = float(cloud_quality.get("score", 0.0))
    local_score = float(local_quality.get("score", 0.0))
    return local_rank < 2 and local_score >= cloud_score + 4.0


def _semantic_alpha_envelope_gate(
    quality: dict[str, Any],
    rgba: np.ndarray,
    crop_shape: tuple[int, int] | list[int],
    subject_locator: dict[str, Any] | None,
) -> dict[str, Any]:
    """Reject a visibly broad crop/background mask outside a verified subject.

    ``alpha_quality`` intentionally cannot reject a tightly cropped plate,
    book or box merely because it occupies most of its transparent PNG.  In
    the automatic quality route we have stronger evidence: MiniMax has marked
    a *complete* physical product box, and the segmenter received a larger
    safety crop around that box.  A matte whose tight extent consumes the
    safety crop on both axes (or saturates one axis) is therefore retaining
    pixels that the verified subject box explicitly excluded.

    The gate uses alpha-visible geometry only.  RGB underneath zero alpha is
    deliberately ignored because the deterministic RGBA contract preserves
    source RGB there even though those pixels never appear in a composite.
    Thin handles/feet are protected by asymmetric ROI padding and by requiring
    a dense, almost-opaque retained region before this becomes a hard reject.
    """

    result = dict(quality)
    locator = subject_locator or {}
    verified = bool(locator.get("semantic_verified"))
    confidence = float(locator.get("confidence", 0.0) or 0.0)
    raw_bbox = np.asarray(locator.get("bbox", []), dtype=np.float64)
    expanded_bbox = np.asarray(locator.get("expanded_bbox", []), dtype=np.float64)
    height, width = (int(crop_shape[0]), int(crop_shape[1]))
    metrics: dict[str, Any] = {
        "enabled": False,
        "semantic_verified": verified,
        "locator_confidence": confidence,
        "leakage": False,
    }
    if (
        not verified
        or confidence < 0.75
        or raw_bbox.shape != (4,)
        or expanded_bbox.shape != (4,)
        or height <= 0
        or width <= 0
    ):
        result["semantic_alpha_envelope"] = metrics
        result["semantic_envelope_leakage"] = False
        return result

    raw_span = np.maximum(raw_bbox[2:] - raw_bbox[:2], 1.0e-6)
    expanded_span = np.maximum(
        expanded_bbox[2:] - expanded_bbox[:2], 1.0e-6,
    )
    expected_span = np.clip(raw_span / expanded_span, 0.0, 1.0)
    visible = rgba[..., 3] > 4
    visible_y, visible_x = np.nonzero(visible)
    if len(visible_x):
        visible_width = int(visible_x.max() - visible_x.min() + 1)
        visible_height = int(visible_y.max() - visible_y.min() + 1)
    else:
        visible_width = 0
        visible_height = 0
    observed_span = np.asarray([
        visible_width / max(1, width),
        visible_height / max(1, height),
    ], dtype=np.float64)
    excess = observed_span - expected_span
    scale = observed_span / np.maximum(expected_span, 1.0e-6)

    structure = alpha_quality(rgba)
    foreground_ratio = float(structure.get("foreground_ratio", 0.0))
    stable_area_ratio = float(structure.get("stable_area_ratio", 0.0))
    edge_occupancy = structure.get("edge_occupancy") or {}
    left_edge = float(edge_occupancy.get("left", 0.0) or 0.0)
    right_edge = float(edge_occupancy.get("right", 0.0) or 0.0)
    top_edge = float(edge_occupancy.get("top", 0.0) or 0.0)
    bottom_edge = float(edge_occupancy.get("bottom", 0.0) or 0.0)
    dense_visible_region = bool(
        foreground_ratio >= 0.48 and stable_area_ratio >= 0.80
    )
    broad_on_both_axes = bool(excess[0] >= 0.08 and excess[1] >= 0.08)
    saturated_one_axis = bool(
        (
            observed_span[0] >= 0.97
            and excess[0] >= 0.11
            and observed_span[1] >= expected_span[1] - 0.03
        )
        or (
            observed_span[1] >= 0.97
            and excess[1] >= 0.11
            and observed_span[0] >= expected_span[0] - 0.03
        )
    )
    # A scene fragment attached to only one side of the product can evade the
    # two-axis test above.  The house-clock failure is the canonical shape: the
    # verified product box is narrower than the returned matte, while a shelf
    # strip extends the matte downward and remains glued to one vertical edge.
    # Require all four independent signals before rejecting so a legitimate
    # thin handle, foot or decorative tip cannot trigger this route alone.
    asymmetric_axis_overflow = bool(
        (
            excess[1] >= 0.10
            and observed_span[0] <= expected_span[0] - 0.02
            and max(left_edge, right_edge) >= 0.42
        )
        or (
            excess[0] >= 0.10
            and observed_span[1] <= expected_span[1] - 0.02
            and max(top_edge, bottom_edge) >= 0.42
        )
    )
    leakage = bool(
        dense_visible_region
        and (
            broad_on_both_axes
            or saturated_one_axis
            or asymmetric_axis_overflow
        )
    )
    metrics.update({
        "enabled": True,
        "expected_subject_span_in_crop": [
            round(float(expected_span[0]), 6),
            round(float(expected_span[1]), 6),
        ],
        "observed_alpha_extent_in_crop": [
            round(float(observed_span[0]), 6),
            round(float(observed_span[1]), 6),
        ],
        "span_excess": [
            round(float(excess[0]), 6),
            round(float(excess[1]), 6),
        ],
        "span_scale": [
            round(float(scale[0]), 6),
            round(float(scale[1]), 6),
        ],
        "visible_foreground_ratio": round(foreground_ratio, 6),
        "stable_area_ratio": round(stable_area_ratio, 6),
        "dense_visible_region": dense_visible_region,
        "broad_on_both_axes": broad_on_both_axes,
        "saturated_one_axis": saturated_one_axis,
        "asymmetric_axis_overflow": asymmetric_axis_overflow,
        "edge_occupancy": {
            "left": round(left_edge, 6),
            "right": round(right_edge, 6),
            "top": round(top_edge, 6),
            "bottom": round(bottom_edge, 6),
        },
        "leakage": leakage,
    })
    result["semantic_alpha_envelope"] = metrics
    result["semantic_envelope_leakage"] = leakage
    if leakage:
        result["verdict"] = "reject"
        result["score"] = min(float(result.get("score", 0.0)), 35.0)
        result["reason"] = (
            "Verified subject ROI gate rejected a broad visible alpha region "
            "that retained the safety crop/background outside the main product"
        )
    return result


def _visual_review_transport_retry_delay(slot_id: int, failed_attempt: int) -> float:
    """Return a short deterministic exponential backoff for an empty slot.

    The small deterministic jitter prevents concurrent browser jobs from
    hammering the same relay at the same instant while keeping the delay
    reproducible in the audit record. ``failed_attempt`` is one-based.
    """

    base = min(3.0, 0.70 * (2 ** max(0, failed_attempt - 1)))
    digest = hashlib.sha256(
        f"visual-review:{int(slot_id)}:{int(failed_attempt)}".encode("utf-8")
    ).digest()
    jitter = (int.from_bytes(digest[:2], "big") / 65535.0) * 0.22
    return round(min(3.25, base + jitter), 3)


class ProductUI(BaseModel):
    id: str
    source: str
    rgba: str | None = None
    center_x: float = 0.5
    baseline_y: float = 0.65
    height_ratio: float = 0.16
    z_order: int = 1
    placement_mode: str = "upright"
    physical_type: str = "rigid_object"
    photographed_pose: str = "unknown"
    pose_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    semantic_category: str = "decor_object"
    # The cutout stage has already inspected the uncropped source image and
    # can therefore preserve semantic evidence that is easy to lose after the
    # background has been removed (for example a plate's black display easel).
    # These fields are advisory evidence only; they never authorize generated
    # product pixels.
    subject_label: str = ""
    subject_included_parts: list[str] = Field(default_factory=list)
    subject_locator_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    subject_semantic_verified: bool = False
    estimated_real_height_cm: float = 20.0
    estimated_height_cm_min: float = Field(default=15.0, ge=2.0, le=300.0)
    estimated_height_cm_max: float = Field(default=30.0, ge=2.0, le=300.0)
    footprint_width_cm: float = Field(default=12.0, ge=1.0, le=200.0)
    size_confidence: float = 0.0
    viewpoint_compatibility: float = Field(default=0.8, ge=0.0, le=1.0)
    source_light_compatibility: float = Field(default=0.5, ge=0.0, le=1.0)
    visual_salience: float = 0.5
    material_mode: str = "opaque"
    material_confidence: float = 0.0
    transmission_strength: float = 0.0
    receiver_width_or_diameter_cm: float = 100.0
    receiver_size_confidence: float = 0.0
    identity_status: str = "unknown"
    identity_mask: str | None = None
    rerender_policy: str = "auto"


class ComposeUI(BaseModel):
    background: str
    products: list[ProductUI]
    support_polygon: list[list[float]]
    support_mask_path: str | None = None
    # Optional geometry evidence produced by the background-only scene pass.
    # These paths are advisory inputs for learned effects; they never replace
    # the exact support/protected masks or authorize background changes.
    depth_map_path: str | None = None
    normal_map_path: str | None = None
    protected_polygons: list[list[list[float]]] = Field(default_factory=list)
    shadow_direction: list[float] = Field(default_factory=lambda: [0.75, -0.66])
    shadow_length_ratio: float = 0.040
    shadow_opacity: float = 0.23
    shadow_softness_ratio: float = 0.15
    reflection_enabled: bool = False
    reflection_opacity: float = 0.08
    cast_shadow_mode: str = "directional"
    surface_material: str = "unknown"
    surface_reflectivity: float = 0.0
    surface_roughness: float = 0.55
    direct_light_fraction: float = 0.55
    contact_ao_strength: float = 1.0
    auto_repair: bool = True
    # Development can bypass only terminal delivery gates.  Rendering,
    # material routing, 4K output and all diagnostic layers stay identical.
    strict_audit: bool = True
    output_long_edge: int = Field(default=3840, ge=1920, le=7680)
    refinement_pass: int = Field(default=0, ge=0, le=1)
    # A refinement is only allowed to render from an immutable first-pass job.
    # The server, rather than a mutable browser payload, owns the placement,
    # product pixels, material selection and first-pass color gains.
    refinement_base_run: str | None = None
    # Per-compose opt-in.  The global integration switch only declares that
    # this machine is allowed to use the paid workflow; it must never trigger
    # a cloud task merely because a shared RunningHub key was saved.
    learned_effects_enabled: bool = False
    hybrid_rerender_enabled: bool = False
    hybrid_provider: str = "manual_candidate"
    hybrid_candidate: str | None = None
    hybrid_prompt: str = (
        "Re-render only the supplied editable transparent-material regions. "
        "The strict-base image defines exact silhouette, position, scale and camera "
        "geometry; the protect mask defines immutable source pixels. For clear glass, "
        "use the target background and receiver as the transmitted scene: visibly "
        "displace that scene across the vessel width instead of copying it flat. "
        "Vertical ribs must alternate gentle magnification and compression of the "
        "transmitted scene, while the glass outline itself remains unchanged. Concentrate "
        "sparse, high-intensity target-lit glints on the rim, spout, handle joints, handle "
        "outer curve and thick foot; keep broad interior regions transparent rather than "
        "globally bright. Show a coherent thick glass base with localized caustic detail. "
        "Transmit and refract any dark tabletop band through the lower vessel so it bends "
        "continuously at the ribs and never becomes a straight pasted band. Never "
        "fill glass with flat white, milky opacity, fog or a uniform alpha overlay. "
        "Do not erase dark interface contours or turn the product into line art. Keep "
        "reflections restrained, vertical and attached to the real base. Do not "
        "add, remove, restyle or rewrite text, logos, labels, brand colours, closures, "
        "product contents or background objects."
    )


class LayoutUI(BaseModel):
    background: str
    products: list[ProductUI]
    support_polygon: list[list[float]]
    support_mask_path: str | None = None
    depth_map_path: str | None = None
    normal_map_path: str | None = None
    protected_polygons: list[list[list[float]]] = Field(default_factory=list)
    shadow_direction: list[float] = Field(default_factory=lambda: [0.75, -0.66])
    shadow_length_ratio: float = 0.040
    shadow_softness_ratio: float = 0.15
    # The background-only scene pass sees the receiver without three product
    # images competing for attention.  When it has a confident physical-size
    # estimate, carry that one ruler into the multimodal layout pass instead
    # of allowing a second call to silently replace (for example) a 55 cm
    # side table with a 70 cm one and shrink every product.
    receiver_width_or_diameter_cm: float | None = Field(
        default=None, ge=20.0, le=500.0,
    )
    receiver_size_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class VisualReviewUI(BaseModel):
    background: str
    final_path: str
    products: list[str]


def _same_visual_evidence(
    previous_bundle: dict[str, Any], current_bundle: dict[str, Any],
) -> bool:
    """Return true when only non-visual audit evidence changed.

    Five subjective reviews are expensive and stochastic.  Re-running them
    after adding a deterministic audit field would be score shopping, because
    the model sees the exact same pixels.  Reuse is allowed only when model,
    prompt, background, final bitmap, product RGBA files, dimensions and image
    format are byte-identical; the new hard audit is still re-bound and fully
    revalidated by ``build_final_gate``.
    """

    if not isinstance(previous_bundle, dict) or not isinstance(current_bundle, dict):
        return False
    if previous_bundle.get("reviewer") != current_bundle.get("reviewer"):
        return False
    # The model is instructed using run-local material, lighting, placement
    # and support-plane semantics. Identical pixels with a different job are
    # not the same review question and must never inherit the old judgement.
    if previous_bundle.get("job_binding") != current_bundle.get("job_binding"):
        return False
    old_inputs = previous_bundle.get("inputs")
    new_inputs = current_bundle.get("inputs")
    if not isinstance(old_inputs, dict) or not isinstance(new_inputs, dict):
        return False
    keys = (
        "background_sha256", "final_sha256", "product_rgba_sha256",
        "final_dimensions", "final_format",
    )
    return all(old_inputs.get(key) == new_inputs.get(key) for key in keys)


def _visual_review_slot_paths(
    final_path: Path, bundle_sha256: str, slot_id: int,
) -> tuple[Path, Path]:
    """Return authoritative workspace cache and run-local audit mirror paths."""

    filename = f"slot_{int(slot_id)}.json"
    workspace_path = (
        WORK / "visual_review_cache" / str(bundle_sha256) / filename
    )
    run_mirror = (
        final_path.parent
        / "visual_review_slots"
        / str(bundle_sha256)
        / filename
    )
    return workspace_path, run_mirror


def _valid_visual_review_slot(
    record: Any, bundle_sha256: str, slot_id: int,
) -> dict[str, Any] | None:
    if not isinstance(record, dict):
        return None
    candidate = record.get("review")
    if not (
        record.get("schema_version") == VISUAL_REVIEW_SLOT_SCHEMA_VERSION
        and record.get("policy_version") == FINAL_GATE_POLICY_VERSION
        and record.get("bundle_sha256") == bundle_sha256
        and int(record.get("slot_id", -1)) == int(slot_id)
        and isinstance(candidate, dict)
        and int(candidate.get("slot_id", -1)) == int(slot_id)
        and candidate.get("prompt_version") == VISUAL_REVIEW_PROMPT_VERSION
    ):
        return None
    validate_dimension_scores(candidate)
    return dict(candidate)


def _read_visual_review_slot(
    final_path: Path, bundle_sha256: str, slot_id: int,
) -> dict[str, Any] | None:
    """Load one immutable slot and keep both cache and run audit copies."""

    workspace_path, run_mirror = _visual_review_slot_paths(
        final_path, bundle_sha256, slot_id,
    )
    for candidate_path in (workspace_path, run_mirror):
        if not candidate_path.is_file():
            continue
        try:
            record = json.loads(candidate_path.read_text(encoding="utf-8"))
            review = _valid_visual_review_slot(record, bundle_sha256, slot_id)
        except (
            OSError, ValueError, TypeError, json.JSONDecodeError, FinalGateError,
        ):
            review = None
        if review is None:
            continue
        # Promote a valid legacy run-local record into the workspace cache,
        # and always mirror a global hit into the current run for audit.
        if candidate_path != workspace_path or not workspace_path.is_file():
            write_json_atomic(workspace_path, record)
        if candidate_path != run_mirror or not run_mirror.is_file():
            write_json_atomic(run_mirror, record)
        return review
    return None


def _write_visual_review_slot(
    final_path: Path,
    bundle_sha256: str,
    slot_id: int,
    review: dict[str, Any],
) -> None:
    record = {
        "schema_version": VISUAL_REVIEW_SLOT_SCHEMA_VERSION,
        "policy_version": FINAL_GATE_POLICY_VERSION,
        "cache_scope": "workspace_bundle_hash",
        "bundle_sha256": bundle_sha256,
        "slot_id": int(slot_id),
        "review": review,
    }
    workspace_path, run_mirror = _visual_review_slot_paths(
        final_path, bundle_sha256, slot_id,
    )
    # The workspace copy prevents a new run from drawing a second score. The
    # mirror makes every individual run self-contained for later audit.
    write_json_atomic(workspace_path, record)
    write_json_atomic(run_mirror, record)


def local_path(raw: str) -> Path:
    path = Path(raw).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise HTTPException(400, f"文件不存在：{raw}")
    return path


def refinement_base_job(raw: str | None) -> tuple[Path, dict[str, Any]]:
    """Load a first-pass job that a lighting-only refinement may reuse.

    A browser can be stale, edited, or otherwise not be the same state that
    produced the candidate under review.  Refinement therefore never trusts
    browser-supplied geometry/material data: it may reference only a completed
    run below ``ui_workspace/runs`` and uses that run's persisted job payload.
    """
    if not raw:
        raise HTTPException(400, "光影修复缺少首轮候选标识，已拒绝执行。")
    run = Path(raw).expanduser().resolve()
    runs_root = (WORK / "runs").resolve()
    try:
        run.relative_to(runs_root)
    except ValueError as exc:
        raise HTTPException(400, "光影修复只能引用当前工作区内的首轮候选。") from exc
    job_path = run / "job.json"
    if not job_path.is_file():
        raise HTTPException(400, "首轮候选不存在或尚未完成，不能进行光影修复。")
    try:
        data = json.loads(job_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(400, "首轮候选记录不可读取，不能进行光影修复。") from exc
    required = ("background", "support_polygon", "lighting", "products", "ui_placements")
    if not isinstance(data, dict) or any(key not in data for key in required):
        raise HTTPException(400, "首轮候选缺少完整锁定数据，不能进行光影修复。")
    if not isinstance(data["products"], list) or not isinstance(data["ui_placements"], list):
        raise HTTPException(400, "首轮候选商品数据无效，不能进行光影修复。")
    return run, data


def browser_path(raw: str) -> str:
    # File names often contain Chinese characters, spaces and ampersands.  They
    # must be encoded explicitly because this value is used directly in <img>
    # elements as well as in fetch requests.
    value = str(Path(raw).resolve()).replace("\\", "/")
    return "/api/file?path=" + quote(value, safe="/:")


def _product_dict(product: ProductUI) -> dict[str, Any]:
    return product.model_dump() if hasattr(product, "model_dump") else product.dict()


def _layout_support_mask(
    raw_path: str | None,
    canvas_size: tuple[int, int],
) -> np.ndarray | None:
    """Load the exact receiver pixels used by final composition.

    The support polygon remains useful for perspective coordinates and UI
    editing, but it is only an approximation of concave/curved receiver
    boundaries.  When scene analysis produced a mask, layout and compose must
    validate the same physical support pixels.
    """
    if not raw_path:
        return None
    return load_mask(
        local_path(raw_path), canvas_size, allow_resize=True,
    )


LAYOUT_PASS_SCORE = 90.0

# Catalogue groups read as one deliberately styled set when the clear space
# between neighbouring silhouettes is roughly one half of their mean visible
# width.  Perspective, handles and asymmetric cutouts need a generous soft
# band, but a gap above this replan limit no longer reads as breathing room: it
# reads as three unrelated products scattered across a large receiver.
AESTHETIC_GAP_TARGET_MIN = 0.35
AESTHETIC_GAP_TARGET_MAX = 0.55
SPARSE_GAP_REPLAN_RATIO = 0.68


def _normalized_polygon_mask(
    shape: tuple[int, int],
    polygons: list[list[list[float]]],
) -> np.ndarray:
    """Rasterise normalized polygons for inexpensive layout diagnostics."""
    height, width = shape
    mask = np.zeros((height, width), np.uint8)
    for polygon in polygons:
        points = np.asarray(polygon, np.float32)
        if len(points) < 3 or not np.isfinite(points).all():
            continue
        pixels = np.rint(points * np.asarray([width - 1, height - 1])).astype(np.int32)
        cv2.fillPoly(mask, [pixels], 1)
    return mask.astype(bool)


def _column_segment(column: np.ndarray, row: int) -> tuple[int, int] | None:
    """Return the connected support interval closest to a requested row."""
    occupied = np.flatnonzero(column)
    if not len(occupied):
        return None
    split_at = np.flatnonzero(np.diff(occupied) > 1)
    starts = np.r_[occupied[0], occupied[split_at + 1]]
    ends = np.r_[occupied[split_at], occupied[-1]]

    def distance(segment: tuple[int, int]) -> int:
        start, end = segment
        if row < start:
            return start - row
        if row > end:
            return row - end
        return 0

    return min(zip(starts.tolist(), ends.tolist()), key=distance)


def _alpha_visual_geometry(path: str | Path) -> tuple[float, float]:
    """Return robust visible-mass height and content width ratios.

    A thin handle, flower stem, or antenna may define the alpha bounding box but
    should not make the main product body look miniature.  The 5-95% alpha-mass
    interval is stable for such appendages while remaining deterministic.
    """
    alpha = load_rgba(path)[..., 3]
    occupied = np.argwhere(alpha > 4)
    if not occupied.size:
        return 0.0, 0.0
    y0, x0 = occupied.min(axis=0)
    y1, x1 = occupied.max(axis=0)
    content_height = int(y1 - y0 + 1)
    row_mass = alpha.astype(np.float64).sum(axis=1)
    cumulative = np.cumsum(row_mass)
    total = float(cumulative[-1])
    if total <= 0:
        mass_height_ratio = 0.0
    else:
        lower = int(np.searchsorted(cumulative, total * 0.05))
        upper = int(np.searchsorted(cumulative, total * 0.95))
        mass_height_ratio = float((upper - lower + 1) / max(1, content_height))
    content_width = int(x1 - x0 + 1)
    return mass_height_ratio, float(content_width / max(1, content_height))


def normalized_display_width(
    display_height_ratio: float,
    content_aspect: float,
    canvas_size: tuple[int, int],
) -> float:
    """Convert a height-normalised object width into normalised image X.

    ``height_ratio`` is measured against canvas height while ``center_x`` is
    measured against canvas width.  Omitting H/W makes every object appear
    W/H times wider to a landscape layout scorer and causes unnecessary group
    shrinking and excessive whitespace.
    """
    canvas_width, canvas_height = canvas_size
    return float(
        display_height_ratio
        * content_aspect
        * canvas_height
        / max(1, canvas_width)
    )


def _layout_surface_diagnostics(
    canvas_size: tuple[int, int],
    support_polygon: list[list[float]],
    protected_polygons: list[list[list[float]]],
    products: list[dict[str, Any]],
    *,
    support_mask: np.ndarray | None = None,
    advisor_seed: list[dict[str, Any]] | None = None,
    repair_count: int = 0,
) -> dict[str, Any]:
    """Score automatic composition in receiving-surface coordinates.

    Geometric validity is still enforced by the compositor.  This additional
    diagnostic catches legal-but-unphotographic rows: objects following a
    sloped front edge, tiny visual bodies caused by wispy alpha appendages, and
    a model proposal that needed a large one-item repair.
    """
    canvas_width, canvas_height = canvas_size
    preview_width = min(900, max(1, canvas_width))
    preview_height = max(1, int(round(canvas_height * preview_width / max(1, canvas_width))))
    if support_mask is None:
        support = _normalized_polygon_mask(
            (preview_height, preview_width), [support_polygon],
        )
    else:
        exact = np.asarray(support_mask)
        if exact.ndim != 2:
            raise ValueError("support_mask must be a two-dimensional image")
        support = cv2.resize(
            exact.astype(np.uint8),
            (preview_width, preview_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    protected = _normalized_polygon_mask(
        (preview_height, preview_width), protected_polygons,
    )
    usable = support & ~protected

    reasons: list[str] = []
    hard_failures: list[str] = []
    score = 100.0

    def penalize(code: str, amount: float, *, hard: bool = False) -> None:
        nonlocal score
        if code not in reasons:
            reasons.append(code)
        score -= max(0.0, float(amount))
        if hard and code not in hard_failures:
            hard_failures.append(code)

    if not usable.any():
        penalize("surface_geometry_unavailable", 100.0, hard=True)
        return {
            "layout_score": 0.0,
            "layout_verdict": "reject",
            "layout_reasons": reasons,
            "layout_hard_failures": hard_failures,
            "product_surface": [],
        }

    product_surface: list[dict[str, Any]] = []
    visual_boxes: list[tuple[float, float, float]] = []
    mass_heights: list[float] = []
    depths: list[float] = []
    clearance_over_heights: list[float] = []
    receiver_spans_over_height: list[float] = []
    apparent_heights: list[float] = []
    semantic_sizes: list[float] = []
    semantic_size_mins: list[float] = []
    semantic_size_maxs: list[float] = []
    semantic_confidences: list[float] = []
    receiver_sizes: list[float] = []
    receiver_confidences: list[float] = []
    visual_saliencies: list[float] = []
    for item in products:
        center_x = float(item["center_x"])
        baseline_y = float(item["baseline_y"])
        display_height = max(1e-6, float(item["height_ratio"]))
        column_x = int(np.clip(round(center_x * (preview_width - 1)), 0, preview_width - 1))
        baseline_row = int(np.clip(round(baseline_y * (preview_height - 1)), 0, preview_height - 1))
        segment = _column_segment(usable[:, column_x], baseline_row)
        if segment is None:
            depth = 1.0
            front_clearance = 0.0
            receiver_span = 0.0
            penalize("surface_geometry_unavailable", 25.0, hard=True)
        else:
            far_row, front_row = segment
            span = max(1, front_row - far_row)
            receiver_span = float(span / preview_height)
            depth = float(np.clip((baseline_row - far_row) / span, 0.0, 1.0))
            front_clearance = float(max(0, front_row - baseline_row) / preview_height)

        rgba_path = item.get("rgba") or item.get("source")
        mass_ratio, content_aspect = _alpha_visual_geometry(str(rgba_path))
        visual_mass_height = float(display_height * mass_ratio)
        display_width = float(
            display_height * content_aspect * canvas_height / max(1, canvas_width)
        )
        visual_boxes.append((center_x - display_width / 2, center_x + display_width / 2, display_width))
        mass_heights.append(visual_mass_height)
        apparent_heights.append(display_height)
        semantic_size = float(item.get("estimated_real_height_cm", 20.0))
        raw_size_min = float(item.get(
            "estimated_height_cm_min", semantic_size,
        ))
        raw_size_max = float(item.get(
            "estimated_height_cm_max", semantic_size,
        ))
        # MiniMax returns category-scale estimates, not surveyed dimensions.
        # Keep the point estimate inside the interval even if a malformed or
        # rounded response puts it just outside.  The interval is later used
        # to distinguish a *provable* perspective inversion from two equally
        # plausible product-size interpretations.
        semantic_size_min = float(np.clip(
            min(raw_size_min, raw_size_max, semantic_size), 2.0, 300.0,
        ))
        semantic_size_max = float(np.clip(
            max(raw_size_min, raw_size_max, semantic_size),
            semantic_size_min,
            300.0,
        ))
        semantic_sizes.append(semantic_size)
        semantic_size_mins.append(semantic_size_min)
        semantic_size_maxs.append(semantic_size_max)
        semantic_confidences.append(float(item.get("size_confidence", 0.0)))
        receiver_sizes.append(float(item.get("receiver_width_or_diameter_cm", 100.0)))
        receiver_confidences.append(float(item.get("receiver_size_confidence", 0.0)))
        visual_saliencies.append(float(item.get("visual_salience", 0.5)))
        depths.append(depth)
        clearance_ratio = float(front_clearance / display_height)
        clearance_over_heights.append(clearance_ratio)
        receiver_spans_over_height.append(float(
            receiver_span / display_height,
        ))
        expected_aspect = float(
            item.get("footprint_width_cm", 12.0)
            / max(2.0, item.get("estimated_real_height_cm", 20.0))
        )
        observed_to_semantic_aspect = float(
            content_aspect / max(0.05, expected_aspect)
        )
        pose_confidence = float(item.get("pose_confidence", 0.0))
        if (
            str(item.get("placement_mode", "upright")) == "upright"
            and semantic_confidences[-1] >= 0.65
            and (observed_to_semantic_aspect < 0.55 or observed_to_semantic_aspect > 1.90)
        ):
            penalize(
                "semantic_silhouette_aspect_mismatch",
                min(28.0, 12.0 + 8.0 * abs(math.log(max(0.05, observed_to_semantic_aspect)))),
                hard=(
                    pose_confidence >= 0.70
                    and (
                        observed_to_semantic_aspect < 0.32
                        or observed_to_semantic_aspect > 3.20
                    )
                ),
            )
        product_surface.append({
            "id": str(item.get("id", "")),
            "surface_depth": round(depth, 6),
            "front_clearance_ratio": round(front_clearance, 6),
            "front_clearance_over_height": round(clearance_ratio, 6),
            "receiver_span_over_height": round(
                receiver_spans_over_height[-1], 6,
            ),
            "visual_mass_height": round(visual_mass_height, 6),
            "display_width_ratio": round(display_width, 6),
            "apparent_height_ratio": round(display_height, 6),
            "estimated_real_height_cm": round(semantic_sizes[-1], 4),
            "estimated_height_cm_min": round(semantic_size_mins[-1], 4),
            "estimated_height_cm_max": round(semantic_size_maxs[-1], 4),
            "size_confidence": round(semantic_confidences[-1], 4),
            "receiver_width_or_diameter_cm": round(receiver_sizes[-1], 4),
            "receiver_size_confidence": round(receiver_confidences[-1], 4),
            "semantic_width_height_ratio": round(expected_aspect, 6),
            "observed_width_height_ratio": round(content_aspect, 6),
            "observed_to_semantic_aspect": round(observed_to_semantic_aspect, 6),
            "photographed_pose": str(item.get("photographed_pose", "unknown")),
            "pose_confidence": round(pose_confidence, 4),
        })

    depth_median = float(np.median(depths)) if depths else 1.0
    depth_max = float(max(depths, default=1.0))
    depth_min = float(min(depths, default=0.0))
    depth_range = depth_max - depth_min
    receiver_capacity = float(np.median(
        np.asarray(receiver_spans_over_height, np.float64),
    )) if receiver_spans_over_height else 0.0
    if receiver_capacity < 0.45:
        front_clearance_required = 0.025
        visible_staging_required = float(np.clip(
            receiver_capacity * 0.55, 0.12, 0.18,
        ))
        automatic_depth_max = 0.90
        automatic_depth_median_max = 0.78
    elif receiver_capacity < 0.85:
        capacity_blend = (receiver_capacity - 0.45) / 0.40
        front_clearance_required = 0.025 + capacity_blend * (0.30 - 0.025)
        visible_staging_required = 0.18 + capacity_blend * (
            VISIBLE_STAGING_UI_MIN_RATIO - 0.18
        )
        automatic_depth_max = 0.90 - capacity_blend * (0.90 - 0.78)
        automatic_depth_median_max = 0.78 - capacity_blend * (0.78 - 0.68)
    else:
        front_clearance_required = 0.30
        visible_staging_required = VISIBLE_STAGING_UI_MIN_RATIO
        automatic_depth_max = 0.78
        automatic_depth_median_max = 0.68
    baseline_values = [float(item["baseline_y"]) for item in products]
    # Measure the centre bridge against the line joining the two side anchors.
    # Unlike raw image-space Y spread, this removes the tabletop's own slope;
    # unlike receiver-depth range, it still recognises a deliberately visible
    # triangle when safety legalization has pulled one irregular anchor
    # forward.  This is the geometry a human viewer actually reads.
    if len(products) == 3:
        ordered_stage = sorted(
            products, key=lambda item: float(item["center_x"]),
        )
        left_stage, bridge_stage, right_stage = ordered_stage
        stage_span = max(
            1e-6,
            float(right_stage["center_x"]) - float(left_stage["center_x"]),
        )
        stage_blend = (
            float(bridge_stage["center_x"]) - float(left_stage["center_x"])
        ) / stage_span
        stage_anchor_line = (
            float(left_stage["baseline_y"]) * (1.0 - stage_blend)
            + float(right_stage["baseline_y"]) * stage_blend
        )
        visible_staging_ratio = abs(
            float(bridge_stage["baseline_y"]) - stage_anchor_line
        ) / max(
            1e-6,
            float(np.median([
                float(item["height_ratio"]) for item in products
            ])),
        )
    else:
        visible_staging_ratio = float(
            np.ptp(np.asarray(baseline_values, np.float64))
            / max(1e-6, float(np.median(apparent_heights)))
        ) if apparent_heights else 0.0
    minimum_clearance = float(min(clearance_over_heights, default=0.0))
    # Receiver-local depth is only a presentation proxy.  On a round, sloped
    # or island-split support it can call an item "90% forward" even while its
    # base still has more than the required product-height breathing room.
    # The measured front clearance is the physically meaningful gate; use
    # local depth as a warning only when that clearance is also insufficient.
    front_depth_risk = minimum_clearance < front_clearance_required
    if depth_max > automatic_depth_max and front_depth_risk:
        penalize(
            "too_close_to_front_edge",
            min(14.0, (depth_max - automatic_depth_max) * 70.0),
            hard=depth_max > min(0.97, automatic_depth_max + 0.12),
        )
    if depth_median > automatic_depth_median_max and front_depth_risk:
        penalize(
            "group_too_close_to_front_edge",
            min(
                10.0,
                (depth_median - automatic_depth_median_max) * 50.0,
            ),
        )
    if depth_median < 0.22:
        penalize("group_too_close_to_far_edge", min(6.0, (0.22 - depth_median) * 35.0))
    if (
        len(depths) >= 3
        and depth_range < 0.055
        and visible_staging_ratio < 0.18
    ):
        penalize("depth_stagger_too_flat", (0.055 - depth_range) * 50.0)

    if minimum_clearance < front_clearance_required:
        penalize(
            "insufficient_front_clearance",
            min(
                6.0,
                (front_clearance_required - minimum_clearance) * 15.0,
            ),
        )

    mass_median = float(np.median(mass_heights)) if mass_heights else 0.0
    positive_mass = [value for value in mass_heights if value > 1e-6]
    mass_scale_ratio = float(max(positive_mass) / min(positive_mass)) if positive_mass else float("inf")
    # A 25 cm pitcher should visibly outweigh a 14.5 cm can.  Comparing raw
    # visual mass calls that physically correct hierarchy an imbalance.  When
    # semantic size evidence is reliable, compare visual mass *per real cm*;
    # fall back to the raw ratio only for uncertain categories.
    semantic_mass_density = [
        mass_heights[index] / max(2.0, semantic_sizes[index])
        for index in range(len(mass_heights))
        if semantic_confidences[index] >= 0.45
    ]
    semantic_mass_ratio = (
        float(max(semantic_mass_density) / min(semantic_mass_density))
        if len(semantic_mass_density) >= 2
        and min(semantic_mass_density) > 1e-9
        else mass_scale_ratio
    )
    if mass_median < 0.105:
        penalize("visual_mass_too_small", min(8.0, (0.105 - mass_median) * 120.0))
    if semantic_mass_ratio > 1.45:
        penalize(
            "visual_scale_imbalance",
            min(6.0, (semantic_mass_ratio - 1.45) * 8.0),
        )

    normalized_gaps: list[float] = []
    ordered_boxes = sorted(visual_boxes, key=lambda value: value[0])
    for left, right in zip(ordered_boxes, ordered_boxes[1:]):
        mean_width = max(1e-6, (left[2] + right[2]) / 2)
        normalized_gaps.append(float((right[0] - left[1]) / mean_width))
    gap_min = float(min(normalized_gaps, default=0.0))
    gap_max = float(max(normalized_gaps, default=0.0))
    gap_median = float(np.median(normalized_gaps)) if normalized_gaps else 0.0
    positive_gaps = [value for value in normalized_gaps if value > 1e-6]
    gap_ratio = (
        float(max(positive_gaps) / min(positive_gaps))
        if len(positive_gaps) >= 2 else 1.0
    )
    if normalized_gaps and gap_min < 0.18:
        penalize("object_gaps_too_tight", min(8.0, (0.18 - gap_min) * 10.0))
    if gap_ratio > 3.0:
        penalize("uneven_object_gaps", min(6.0, (gap_ratio - 3.0) * 0.8))

    # A valid footprint is necessary but not sufficient for a photographic
    # composition.  Keep composition, semantic scale and perspective as
    # independent dimensions, then use their minimum for the layout gate.  A
    # flat ruler-row or three tiny stickers can therefore never be hidden by a
    # perfect collision score.
    usable_y, usable_x = np.nonzero(usable)
    support_left = float(usable_x.min() / preview_width)
    support_right = float((usable_x.max() + 1) / preview_width)
    support_span_x = max(1e-6, support_right - support_left)
    group_left = float(min((box[0] for box in visual_boxes), default=support_left))
    group_right = float(max((box[1] for box in visual_boxes), default=support_right))
    group_fill_ratio = float((group_right - group_left) / support_span_x)

    composition_score = 100.0
    if gap_max > AESTHETIC_GAP_TARGET_MAX:
        # Keep the 0.55--0.68 interval as tolerance for perspective and highly
        # unequal silhouettes, but make increasingly sparse proposals lose to
        # a compact local candidate.  At >0.68 the explicit replan gate below
        # is authoritative even if every collision/support test still passes.
        composition_score -= min(
            10.0,
            (gap_max - AESTHETIC_GAP_TARGET_MAX) * 18.0,
        )
    if gap_max > SPARSE_GAP_REPLAN_RATIO:
        if "object_gaps_too_sparse" not in reasons:
            reasons.append("object_gaps_too_sparse")
    if (
        len(products) >= 3
        and depth_range < 0.045
        and visible_staging_ratio < 0.18
    ):
        # Below 4.5% receiver-depth separation the three bases read as one
        # artificial catalogue row even when their image Y values differ on a
        # sloping table edge.
        composition_score = min(
            composition_score,
            82.0 + 8.0 * depth_range / 0.045,
        )
        if "flat_catalogue_row" not in reasons:
            reasons.append("flat_catalogue_row")
    if len(products) >= 3 and group_fill_ratio < 0.36:
        composition_score = min(
            composition_score,
            78.0 + 12.0 * group_fill_ratio / 0.36,
        )
        if "group_too_tightly_clustered" not in reasons:
            reasons.append("group_too_tightly_clustered")
    if group_fill_ratio > 0.90:
        composition_score -= min(14.0, (group_fill_ratio - 0.90) * 100.0)
        if "group_spread_too_wide" not in reasons:
            reasons.append("group_spread_too_wide")

    # Equal centre or edge gaps are not aesthetic evidence by themselves.  In
    # combination with equal height and no depth hierarchy they are strong
    # evidence of a mechanical three-column template.
    height_cv = float(
        np.std(apparent_heights) / max(1e-6, np.mean(apparent_heights))
    ) if apparent_heights else 0.0
    gap_cv = float(
        np.std(positive_gaps) / max(1e-6, np.mean(positive_gaps))
    ) if len(positive_gaps) >= 2 else 1.0
    median_display_height_px = float(
        np.median(apparent_heights) * preview_height
    ) if apparent_heights else 1.0
    # ``visible_staging_required`` remains the planning target.  The delivery
    # gate allows a tiny, resolution-aware presentation tolerance: on a 900 px
    # preview a two-to-three-pixel baseline difference is not a meaningful
    # aesthetic distinction.  A hard 0.12 lower floor still rejects an actual
    # ruler row, so this does not turn the target into a bypass.
    visible_staging_tolerance = float(np.clip(
        2.5 / max(1.0, median_display_height_px), 0.010, 0.030,
    ))
    visible_staging_gate = float(max(
        0.12,
        visible_staging_required - visible_staging_tolerance,
    ))
    if (
        len(products) >= 3
        and visible_staging_ratio < visible_staging_gate
    ):
        # Receiver-normalised depth alone can be misleading on a shallow top:
        # a mathematically distinct 0.12--0.16 depth range may project to only
        # a few pixels and still look like three stickers on one ruler line.
        # Require the final bases to form a visible triangle relative to the
        # displayed product size.  The separate depth-range check above still
        # prevents a sloped table edge from faking this evidence.
        composition_score = min(
            composition_score,
            82.0
            + 8.0
            * visible_staging_ratio
            / max(1.0e-6, visible_staging_gate),
        )
        if "visually_flat_staging" not in reasons:
            reasons.append("visually_flat_staging")
    if len(products) >= 3 and visible_staging_ratio > 0.85:
        # Excessive fore/aft separation stops reading as one styled product
        # group and becomes three unrelated longitudinal points.  Keep a broad
        # upper allowance for oblique receivers, but reject the over-correction
        # that can otherwise be introduced while fixing a flat row.
        composition_score = min(
            composition_score,
            86.0 + 6.0 * 0.85 / visible_staging_ratio,
        )
        if "staging_scattered_in_depth" not in reasons:
            reasons.append("staging_scattered_in_depth")
    if (
        len(products) >= 3
        and depth_range < 0.055
        and height_cv < 0.055
        and gap_cv < 0.09
    ):
        composition_score = min(composition_score, 82.0)
        if "mechanical_equal_row" not in reasons:
            reasons.append("mechanical_equal_row")

    # Absolute scale cue: estimated object height divided by estimated receiver
    # width, projected through the measured receiver span.  This is only used
    # when both semantic confidences are meaningful.  It directly catches the
    # previous glass-table failure where 16-25 cm vessels were rendered as tiny
    # ornaments on a 65-75 cm round table.
    semantic_scale_score = 100.0
    semantic_scale_ratios: list[float] = []
    semantic_scale_confidences: list[float] = []
    support_span_height_units = (
        support_span_x * canvas_width / max(1.0, float(canvas_height))
    )
    for index, display_height in enumerate(apparent_heights):
        confidence = float(np.clip(
            semantic_confidences[index] * receiver_confidences[index], 0.0, 1.0,
        ))
        semantic_scale_confidences.append(confidence)
        if confidence < 0.34:
            semantic_scale_ratios.append(1.0)
            continue
        expected = float(np.clip(
            semantic_sizes[index]
            / max(20.0, receiver_sizes[index])
            * support_span_height_units
            * (0.88 + 0.18 * depths[index]),
            0.045,
            0.58,
        ))
        ratio = float(display_height / max(0.035, expected))
        semantic_scale_ratios.append(ratio)

    # Receiver dimensions describe a real table, but a commercial still-life
    # is intentionally enlarged as one coherent group so labels remain
    # readable.  Treat that common enlargement as a presentation choice and
    # audit the *relative* product-size relationships independently.  The old
    # per-item absolute check could call the tall hero "too large" while also
    # calling the two secondary products "too small", an internally
    # contradictory verdict that then pushed the solver toward ugly extremes.
    reliable_ratios = [
        ratio for ratio, confidence in zip(
            semantic_scale_ratios, semantic_scale_confidences,
        )
        if confidence >= 0.34
    ]
    commercial_scale_emphasis = (
        float(np.median(reliable_ratios)) if reliable_ratios else 1.0
    )
    semantic_relative_scale_ratios: list[float] = []
    for ratio, confidence in zip(
        semantic_scale_ratios, semantic_scale_confidences,
    ):
        if confidence < 0.34:
            semantic_relative_scale_ratios.append(1.0)
            continue
        relative = float(ratio / max(0.05, commercial_scale_emphasis))
        semantic_relative_scale_ratios.append(relative)
        log_error = abs(float(np.log(max(0.05, relative))))
        # Category dimensions are only priors (a "cup" can range from an
        # espresso cup to a serving mug).  Keep the score informative, but do
        # not reject an otherwise attractive commercial grouping for a small
        # disagreement with those priors.
        tolerance = float(np.log(1.32))
        candidate = 100.0 - max(0.0, log_error - tolerance) * (
            34.0 + 32.0 * confidence
        )
        semantic_scale_score = min(semantic_scale_score, candidate)
        if relative < 0.64 or relative > 1.56:
            if "semantic_relative_scale_mismatch" not in reasons:
                reasons.append("semantic_relative_scale_mismatch")

    # A common 0.68x--2.25x emphasis covers focal-length uncertainty and the
    # normal enlargement used in merchandise photography.  Outside that band
    # the whole set genuinely reads as miniature or as oversized props.
    if reliable_ratios and commercial_scale_emphasis < 0.68:
        semantic_scale_score = min(
            semantic_scale_score,
            86.0 * commercial_scale_emphasis / 0.68,
        )
        if "semantic_scale_too_small" not in reasons:
            reasons.append("semantic_scale_too_small")
    elif reliable_ratios and commercial_scale_emphasis > 2.25:
        semantic_scale_score = min(
            semantic_scale_score,
            86.0 * 2.25 / commercial_scale_emphasis,
        )
        if "semantic_scale_too_large" not in reasons:
            reasons.append("semantic_scale_too_large")

    # When absolute dimensions are unavailable, retain a conservative visual
    # floor rather than pretending an arbitrarily tiny group is high quality.
    if (
        mass_median < 0.085
        and (not reliable_ratios or commercial_scale_emphasis < 0.90)
    ):
        semantic_scale_score = min(
            semantic_scale_score, 78.0 + 12.0 * mass_median / 0.085,
        )
        if "visual_mass_physically_too_small" not in reasons:
            reasons.append("visual_mass_physically_too_small")

    perspective_score = 100.0
    perspective_violations = 0
    perspective_pairs: list[dict[str, Any]] = []
    for first in range(len(products)):
        for second in range(first + 1, len(products)):
            depth_delta = depths[first] - depths[second]
            if abs(depth_delta) < 0.055:
                continue
            first_scale_interval = (
                apparent_heights[first] / max(2.0, semantic_size_maxs[first]),
                apparent_heights[first] / max(2.0, semantic_size_mins[first]),
            )
            second_scale_interval = (
                apparent_heights[second] / max(2.0, semantic_size_maxs[second]),
                apparent_heights[second] / max(2.0, semantic_size_mins[second]),
            )
            if depth_delta > 0:
                near_index, far_index = first, second
                near_interval, far_interval = (
                    first_scale_interval, second_scale_interval,
                )
            else:
                near_index, far_index = second, first
                near_interval, far_interval = (
                    second_scale_interval, first_scale_interval,
                )
            pair_confidence = min(
                semantic_confidences[first], semantic_confidences[second],
            )
            # Semantic dimensions are category estimates, not millimetre
            # measurements.  A nine-percent inversion threshold was tighter
            # than their uncertainty and rejected visually correct compact
            # triangles.  Reserve this warning for a clearly inverted size
            # cue; the independent relative-scale audit above still catches
            # inconsistent product hierarchy.
            # Fail only when *every* real-height interpretation allowed by the
            # adviser proves the near object substantially smaller in image
            # scale.  Comparing point estimates alone rejected the A03 coffee
            # table composition even though a 12--35 cm figurine and an
            # 8--15 cm mug-with-fruit have overlapping, visually compatible
            # scale intervals.
            proven_inversion = bool(
                near_interval[1] < far_interval[0] * 0.75
            )
            perspective_pairs.append({
                "near_id": str(products[near_index].get("id", "")),
                "far_id": str(products[far_index].get("id", "")),
                "near_scale_interval": [
                    round(float(near_interval[0]), 8),
                    round(float(near_interval[1]), 8),
                ],
                "far_scale_interval": [
                    round(float(far_interval[0]), 8),
                    round(float(far_interval[1]), 8),
                ],
                "pair_confidence": round(float(pair_confidence), 4),
                "proven_inversion": proven_inversion,
            })
            if pair_confidence >= 0.45 and proven_inversion:
                perspective_violations += 1
    if perspective_violations:
        perspective_score = max(72.0, 92.0 - 8.0 * perspective_violations)
        if "perspective_scale_inversion" not in reasons:
            reasons.append("perspective_scale_inversion")

    seed_by_id = {
        str(item.get("id")): item for item in (advisor_seed or [])
    }
    repair_distances: list[float] = []
    repair_scale_changes: list[float] = []
    for item in products:
        seed = seed_by_id.get(str(item.get("id")))
        if not seed:
            continue
        repair_distances.append(float(np.hypot(
            float(item["center_x"]) - float(seed["center_x"]),
            float(item["baseline_y"]) - float(seed["baseline_y"]),
        )))
        seed_height = max(1e-6, float(seed["height_ratio"]))
        repair_scale_changes.append(abs(float(item["height_ratio"]) / seed_height - 1.0))
    repair_max = float(max(repair_distances, default=0.0))
    repair_scale_max = float(max(repair_scale_changes, default=0.0))
    if repair_max > 0.06:
        penalize("advisor_repair_too_large", min(12.0, (repair_max - 0.06) * 60.0))
    if repair_scale_max > 0.10:
        penalize("advisor_scale_repair_too_large", min(8.0, (repair_scale_max - 0.10) * 30.0))
    if repair_count >= 2:
        penalize("multiple_advisor_repairs", min(5.0, float(repair_count)))

    geometry_score = float(np.clip(score, 0.0, 100.0))
    score = float(np.clip(min(
        geometry_score,
        composition_score,
        semantic_scale_score,
        perspective_score,
    ), 0.0, 100.0))
    verdict = "pass" if score >= LAYOUT_PASS_SCORE and not hard_failures else "reject"
    return {
        "layout_score": round(score, 2),
        "layout_verdict": verdict,
        "layout_reasons": reasons,
        "layout_hard_failures": hard_failures,
        "geometry_score": round(geometry_score, 2),
        "composition_score": round(composition_score, 2),
        "semantic_scale_score": round(semantic_scale_score, 2),
        "perspective_score": round(perspective_score, 2),
        "surface_depth_median": round(depth_median, 6),
        "surface_depth_min": round(depth_min, 6),
        "surface_depth_max": round(depth_max, 6),
        "surface_depth_range": round(depth_range, 6),
        "minimum_front_clearance_over_height": round(minimum_clearance, 6),
        "receiver_span_over_height_median": round(receiver_capacity, 6),
        "required_front_clearance_over_height": round(
            front_clearance_required, 6,
        ),
        "required_visible_staging_ratio": round(
            visible_staging_required, 6,
        ),
        "visible_staging_gate_ratio": round(
            visible_staging_gate, 6,
        ),
        "visible_staging_tolerance_ratio": round(
            visible_staging_tolerance, 6,
        ),
        "automatic_depth_max": round(automatic_depth_max, 6),
        "automatic_depth_median_max": round(
            automatic_depth_median_max, 6,
        ),
        "visual_mass_height_median": round(mass_median, 6),
        "visual_mass_scale_ratio": round(mass_scale_ratio, 6),
        "semantic_mass_density_ratio": round(semantic_mass_ratio, 6),
        "minimum_normalized_gap": round(gap_min, 6),
        "maximum_normalized_gap": round(gap_max, 6),
        "median_normalized_gap": round(gap_median, 6),
        "aesthetic_gap_target": [
            AESTHETIC_GAP_TARGET_MIN,
            AESTHETIC_GAP_TARGET_MAX,
        ],
        "sparse_gap_replan_ratio": SPARSE_GAP_REPLAN_RATIO,
        "normalized_gap_ratio": round(gap_ratio, 6),
        "group_fill_ratio": round(group_fill_ratio, 6),
        "apparent_height_cv": round(height_cv, 6),
        "visible_staging_ratio": round(visible_staging_ratio, 6),
        "positive_gap_cv": round(gap_cv, 6),
        "semantic_scale_ratios": [round(value, 6) for value in semantic_scale_ratios],
        "semantic_relative_scale_ratios": [
            round(value, 6) for value in semantic_relative_scale_ratios
        ],
        "commercial_scale_emphasis": round(commercial_scale_emphasis, 6),
        "perspective_scale_violations": int(perspective_violations),
        "perspective_scale_pairs": perspective_pairs,
        "advisor_repair_max": round(repair_max, 6),
        "advisor_scale_repair_max": round(repair_scale_max, 6),
        "automatic_repairs": float(repair_count),
        "product_surface": product_surface,
    }


def _merge_layout_metrics(
    solver_metrics: dict[str, Any],
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(solver_metrics)
    if "layout_score" in merged:
        merged["solver_score"] = float(merged["layout_score"])
    merged.update(diagnostics)
    return merged


def _prefer_replanned_layout(
    seed_metrics: dict[str, Any],
    local_metrics: dict[str, Any],
) -> bool:
    """Prefer an actually admissible layout before comparing soft scores.

    The diagnostic score is useful only inside the same verdict class.  A
    rejected advisor seed can otherwise retain a deceptively high solver
    score and suppress a locally replanned layout that has cleared every hard
    composition gate.  Delivery semantics therefore come first: pass beats
    reject, and the numerical score is used only as the tie-breaker.
    """

    seed_pass = str(seed_metrics.get("layout_verdict", "reject")) == "pass"
    local_pass = str(local_metrics.get("layout_verdict", "reject")) == "pass"
    if local_pass != seed_pass:
        return local_pass
    seed_sparse = float(seed_metrics.get(
        "maximum_normalized_gap", 0.0,
    )) > SPARSE_GAP_REPLAN_RATIO
    local_sparse = float(local_metrics.get(
        "maximum_normalized_gap", 0.0,
    )) > SPARSE_GAP_REPLAN_RATIO
    if seed_sparse != local_sparse:
        # Clearing a visible composition defect is more important than a soft
        # score delta inside the same delivery-verdict class.  This also keeps
        # a higher-scoring but scattered MiniMax seed from blocking the local
        # joint replan that was invoked specifically to repair that spacing.
        return not local_sparse
    seed_semantic = float(seed_metrics.get("semantic_scale_score", 100.0))
    local_semantic = float(local_metrics.get("semantic_scale_score", 100.0))
    if abs(local_semantic - seed_semantic) >= 3.0:
        # A few composition points cannot buy a visibly wrong real-world size
        # relationship.  This is especially important after compacting an
        # adviser seed: the full local planner may clear the same spacing gate
        # only by inflating a labelled secondary product.
        return local_semantic > seed_semantic
    return float(local_metrics.get("layout_score", 0.0)) > float(
        seed_metrics.get("layout_score", 0.0)
    )


def _friendly_error(exc: Exception) -> str:
    text = str(exc)
    mapping = {
        "Product contact footprint lies outside support area": "商品底边没有落在桌面内",
        "Product body overlap exceeds permitted threshold": "商品主体发生重叠",
        "Placement covers a protected region": "商品遮挡了背景中的保护物体",
        "Product body intersects a protected area": "商品遮挡了背景中的保护物体",
    }
    for source, target in mapping.items():
        if source in text:
            return target
    return text


def _advice_attempt(advice: MiniMaxAdvice, attempt: int) -> dict[str, Any]:
    product_reliability = [
        {
            "product_id": item.product_id,
            "estimated_real_height_cm": item.estimated_real_height_cm,
            "size_confidence": item.size_confidence,
            "viewpoint_compatibility": item.viewpoint_compatibility,
            "placement_mode": item.placement_mode,
            "photographed_pose": item.photographed_pose,
            "pose_confidence": item.pose_confidence,
        }
        for item in list(getattr(advice, "products", []))
    ]
    return {
        "attempt": attempt,
        "model": advice.model,
        "request_id": advice.request_id,
        "route": advice.route,
        "confidence": advice.confidence,
        "products": product_reliability,
        "warnings": advice.warnings,
    }


def _advice_reliability(advice: MiniMaxAdvice) -> float:
    products = list(getattr(advice, "products", []))
    if not products:
        return float(advice.confidence)
    size = float(np.mean([item.size_confidence for item in products]))
    viewpoint = float(np.min([item.viewpoint_compatibility for item in products]))
    return float(0.58 * advice.confidence + 0.22 * size + 0.20 * viewpoint)


def _choose_advice(attempts: list[MiniMaxAdvice]) -> MiniMaxAdvice:
    """Choose conservatively without treating repeated opinions as votes."""
    confident_rejects = [
        item for item in attempts if item.route == "reject" and item.confidence >= 0.76
    ]
    if confident_rejects:
        return max(confident_rejects, key=lambda item: item.confidence)
    usable = [item for item in attempts if item.route != "reject"]
    if usable:
        chosen = max(usable, key=_advice_reliability)
    else:
        chosen = max(attempts, key=lambda item: item.confidence)

    # Scene geometry should come from one coherent answer, but product pose is
    # an independent per-object observation.  A retry commonly corrects one
    # difficult item (for example a top-view trinket dish) while having a
    # slightly lower scene confidence overall.  Previously that useful
    # correction was discarded with the whole retry, routing a flat object as
    # upright.  Adopt only a clearly stronger pose observation and leave the
    # selected answer's scale, support and lighting untouched.
    chosen_products = list(getattr(chosen, "products", []))
    if not chosen_products:
        return chosen
    by_attempt = [
        {
            item.product_id: item
            for item in list(getattr(advice, "products", []))
        }
        for advice in usable
    ]
    reconciled: list[Any] = []
    for product in chosen_products:
        candidates = [
            mapping[product.product_id]
            for mapping in by_attempt
            if product.product_id in mapping
        ]
        stronger = max(candidates, key=lambda item: item.pose_confidence)
        if (
            stronger.pose_confidence >= 0.75
            and stronger.pose_confidence >= product.pose_confidence + 0.10
            and stronger.photographed_pose != "unknown"
        ):
            product = replace(
                product,
                placement_mode=stronger.placement_mode,
                photographed_pose=stronger.photographed_pose,
                pose_confidence=stronger.pose_confidence,
            )
        reconciled.append(product)
    return replace(chosen, products=reconciled)


def _consensus_incompatible_viewpoints(
    attempts: list[MiniMaxAdvice],
    chosen: MiniMaxAdvice,
    *,
    threshold: float = 0.45,
) -> list[str]:
    """Return only viewpoint failures supported by every usable observation.

    A retry exists precisely because one multimodal pass can misread a low,
    reclining object as an upright object with an impossible base.  Treating
    the minimum of two low-confidence, conflicting estimates as an oracle made
    placeable surface-flat products fail before deterministic geometry saw
    them.  One observation still fails closed; with independent usable passes,
    a product is rejected here only when all of them remain below the physical
    limit.  A disagreement is not a pass: it merely defers the decision to the
    existing support-footprint and collision solver.
    """

    usable = [item for item in attempts if item.route != "reject"]
    by_attempt = [
        {
            product.product_id: product
            for product in list(getattr(advice, "products", []))
        }
        for advice in usable
    ]
    incompatible: list[str] = []
    for product in list(getattr(chosen, "products", [])):
        observations = [
            float(mapping[product.product_id].viewpoint_compatibility)
            for mapping in by_attempt
            if product.product_id in mapping
        ]
        if not observations:
            observations = [float(product.viewpoint_compatibility)]
        if all(value < threshold for value in observations):
            incompatible.append(product.product_id)
    return incompatible


def _reconcile_advice_with_subject_evidence(
    advice: MiniMaxAdvice,
    source_products: list[dict[str, Any]],
) -> tuple[MiniMaxAdvice, list[dict[str, Any]]]:
    """Resolve one narrow but damaging cross-stage semantic contradiction.

    The uncropped source locator sees contextual evidence that the later
    transparent PNG no longer makes obvious.  A decorative plate photographed
    front-on with a real display easel is an upright display object: using the
    plate's material thickness as its visible height makes it a coin-sized
    sticker.  We only override the layout adviser when the upstream semantic
    locator is verified, confident, and explicitly names both a plate-like
    object and its display support.  A plate without support evidence remains
    free to be classified as lying flat, so this is a pose-evidence gate rather
    than a category-specific fixed coordinate or case lookup.
    """

    source_by_id = {
        str(item.get("id", "")): item for item in source_products
        if str(item.get("id", ""))
    }
    reconciled: list[Any] = []
    corrections: list[dict[str, Any]] = []
    for product in list(getattr(advice, "products", [])):
        source = source_by_id.get(product.product_id, {})
        label = str(source.get("subject_label", "")).strip()
        parts = [
            str(value).strip()
            for value in source.get("subject_included_parts", [])
            if str(value).strip()
        ]
        raw_evidence = " ".join([label, *parts]).lower()
        tokens = set(re.findall(r"[a-z]+", raw_evidence))
        locator_confidence = float(np.clip(
            source.get("subject_locator_confidence", 0.0), 0.0, 1.0,
        ))
        verified = bool(source.get("subject_semantic_verified", False))
        plate_like = bool(tokens & {"plate", "platter", "tray", "charger"}) or any(
            marker in raw_evidence for marker in ("装饰盘", "餐盘", "托盘")
        )
        display_supported = bool(tokens & {"easel"}) or any(
            marker in raw_evidence
            for marker in (
                "display stand", "display support", "support stand",
                "display base", "展示架", "陈列架", "支架",
            )
        )
        explicit_clear_glass = any(
            marker in raw_evidence
            for marker in (
                "clear glass", "transparent glass", "colourless glass",
                "colorless glass", "透明玻璃", "无色玻璃",
            )
        )
        explicitly_coloured_glass = any(
            marker in raw_evidence
            for marker in (
                "amber glass", "colored glass", "coloured glass",
                "smoked glass", "琥珀玻璃", "有色玻璃", "烟灰玻璃",
            )
        )
        neutral_open_vessel = bool(tokens & {
            "pitcher", "jug", "carafe", "decanter", "tumbler",
            "goblet", "glass", "cup",
        })
        if (
            verified and locator_confidence >= 0.80
            and explicit_clear_glass and not explicitly_coloured_glass
            and neutral_open_vessel
        ):
            corrected_material = replace(
                product,
                material_mode="clear_glass",
                material_confidence=max(
                    float(product.material_confidence),
                    max(0.82, locator_confidence * 0.95),
                ),
                transmission_strength=max(
                    float(product.transmission_strength), 0.95,
                ),
            )
            if corrected_material != product:
                corrections.append({
                    "product_id": product.product_id,
                    "reason": "verified_explicit_clear_glass_subject",
                    "subject_label": label,
                    "included_parts": parts,
                    "locator_confidence": round(locator_confidence, 4),
                    "original_material_mode": product.material_mode,
                    "corrected_material_mode": corrected_material.material_mode,
                    "corrected_material_confidence": round(
                        float(corrected_material.material_confidence), 4,
                    ),
                    "corrected_transmission_strength": round(
                        float(corrected_material.transmission_strength), 4,
                    ),
                })
                product = corrected_material
        if not (
            verified and locator_confidence >= 0.72
            and plate_like and display_supported
        ):
            reconciled.append(product)
            continue

        # Prefer a plausible diameter reported by the adviser.  If that same
        # answer also collapsed the footprint to a few centimetres, use a
        # conservative commercial display-plate prior instead of inventing a
        # value from the source crop or from this particular test image.
        diameter_cm = float(product.footprint_width_cm)
        if not 16.0 <= diameter_cm <= 60.0:
            diameter_cm = 28.0
        upright_height_cm = max(
            float(product.estimated_real_height_cm),
            float(np.clip(diameter_cm * 0.86, 18.0, 48.0)),
        )
        corrected = replace(
            product,
            placement_mode="upright",
            physical_type="rigid_display_object",
            photographed_pose="display_stand",
            pose_confidence=max(float(product.pose_confidence), locator_confidence),
            estimated_real_height_cm=upright_height_cm,
            estimated_height_cm_min=max(
                float(product.estimated_height_cm_min),
                upright_height_cm * 0.76,
            ),
            estimated_height_cm_max=max(
                float(product.estimated_height_cm_max),
                upright_height_cm * 1.18,
            ),
            size_confidence=max(
                float(product.size_confidence), locator_confidence * 0.78,
            ),
        )
        corrections.append({
            "product_id": product.product_id,
            "reason": "verified_plate_with_display_support",
            "subject_label": label,
            "included_parts": parts,
            "locator_confidence": round(locator_confidence, 4),
            "original_pose": product.photographed_pose,
            "corrected_pose": corrected.photographed_pose,
            "original_height_cm": round(float(product.estimated_real_height_cm), 4),
            "corrected_height_cm": round(float(corrected.estimated_real_height_cm), 4),
        })
        reconciled.append(corrected)
    return replace(advice, products=reconciled), corrections


def _local_subject_semantic_priors(
    source_products: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fill missing local-fallback semantics from verified subject evidence.

    The subject locator sees the uncropped catalogue photograph and can name a
    product even when MiniMax layout advice is temporarily unavailable.  The
    old fallback discarded that evidence, so every product reverted to the UI
    defaults (20 cm, opaque ``decor_object``) and a candle could become the
    same displayed height as a plate or pitcher.

    These are deliberately broad retail-photography priors, not coordinates
    for a particular scene.  They are applied only when the locator is
    verified and confident, and only to fields that are still at their UI
    defaults or otherwise carry very low confidence.  Explicit/user-provided
    or model-provided size, pose and material evidence remains authoritative.
    """

    default_size = {
        "estimated_real_height_cm": 20.0,
        "estimated_height_cm_min": 15.0,
        "estimated_height_cm_max": 30.0,
        "footprint_width_cm": 12.0,
    }

    def is_default_size(item: dict[str, Any]) -> bool:
        if float(item.get("size_confidence", 0.0)) >= 0.25:
            return False
        return all(
            abs(float(item.get(key, value)) - value) <= 1e-6
            for key, value in default_size.items()
        )

    def has(marker_text: str, *markers: str) -> bool:
        return any(marker in marker_text for marker in markers)

    enriched: list[dict[str, Any]] = []
    corrections: list[dict[str, Any]] = []
    for original in source_products:
        item = dict(original)
        confidence = float(np.clip(
            item.get("subject_locator_confidence", 0.0), 0.0, 1.0,
        ))
        verified = bool(item.get("subject_semantic_verified", False))
        label = str(item.get("subject_label", "")).strip()
        parts = [
            str(value).strip()
            for value in item.get("subject_included_parts", [])
            if str(value).strip()
        ]
        evidence = " ".join([label, *parts]).lower()
        if not (verified and confidence >= 0.72 and evidence):
            enriched.append(item)
            continue

        profile: dict[str, Any] | None = None
        reason = ""
        display_support = has(
            evidence,
            "display stand", "display easel", "display support",
            "support stand", "display base", "easel",
            "展示架", "陈列架", "支架",
        )
        plate_like = has(
            evidence, "plate", "platter", "charger", "decorative dish",
            "装饰盘", "餐盘", "托盘",
        )
        clear_descriptor = has(
            evidence, "clear glass", "transparent glass", "colourless glass",
            "colorless glass", "透明玻璃", "无色玻璃",
        )
        frosted_descriptor = has(
            evidence, "frosted glass", "etched glass", "磨砂玻璃",
        )
        coloured_descriptor = has(
            evidence, "amber glass", "colored glass", "coloured glass",
            "smoked glass", "琥珀玻璃", "有色玻璃", "烟灰玻璃",
        )

        if plate_like and display_support:
            profile = {
                "semantic_category": "decorative_display_plate",
                "physical_type": "rigid_display_object",
                "placement_mode": "upright",
                "photographed_pose": "display_stand",
                "estimated_real_height_cm": 26.0,
                "estimated_height_cm_min": 20.0,
                "estimated_height_cm_max": 36.0,
                "footprint_width_cm": 27.0,
                "material_mode": "opaque_glossy",
                "transmission_strength": 0.0,
            }
            reason = "verified_display_plate_prior"
        elif has(evidence, "candle", "蜡烛", "香薰"):
            # Filled or labelled candle jars read as opaque merchandise even
            # when the outer vessel is coloured glass.  Routing them through
            # full transmission would unnecessarily weaken labels and colour.
            profile = {
                "semantic_category": "scented_candle",
                "physical_type": "rigid_vessel",
                "placement_mode": "upright",
                "photographed_pose": "standing_on_base",
                "estimated_real_height_cm": 10.5,
                "estimated_height_cm_min": 8.0,
                "estimated_height_cm_max": 15.0,
                "footprint_width_cm": 8.2,
                "material_mode": "opaque_glossy",
                "transmission_strength": 0.0,
            }
            reason = "verified_candle_prior"
        elif has(evidence, "pitcher", "jug", "carafe", "decanter", "水壶", "玻璃壶"):
            profile = {
                "semantic_category": "serving_pitcher",
                "physical_type": "rigid_vessel",
                "placement_mode": "upright",
                "photographed_pose": "standing_on_base",
                "estimated_real_height_cm": 24.0,
                "estimated_height_cm_min": 19.0,
                "estimated_height_cm_max": 31.0,
                "footprint_width_cm": 13.5,
                "material_mode": (
                    "clear_glass" if clear_descriptor else
                    "frosted_glass" if frosted_descriptor else
                    "colored_glass" if coloured_descriptor else
                    "opaque_glossy"
                ),
                "transmission_strength": (
                    0.95 if clear_descriptor else
                    0.55 if frosted_descriptor else
                    0.42 if coloured_descriptor else 0.0
                ),
            }
            reason = "verified_pitcher_prior"
        elif has(
            evidence, "tumbler", "drinking glass", "glass cup", "goblet",
            "wine glass", "玻璃杯", "高脚杯",
        ):
            profile = {
                "semantic_category": "drinking_glass",
                "physical_type": "rigid_vessel",
                "placement_mode": "upright",
                "photographed_pose": "standing_on_base",
                "estimated_real_height_cm": 14.5,
                "estimated_height_cm_min": 11.0,
                "estimated_height_cm_max": 19.0,
                "footprint_width_cm": 8.5,
                "material_mode": (
                    "frosted_glass" if frosted_descriptor else
                    "colored_glass" if coloured_descriptor else
                    "clear_glass"
                ),
                "transmission_strength": (
                    0.55 if frosted_descriptor else
                    0.42 if coloured_descriptor else 0.95
                ),
            }
            reason = "verified_drinking_glass_prior"
        elif has(evidence, "bottle", "flask", "酒瓶", "瓶装"):
            profile = {
                "semantic_category": "bottle",
                "physical_type": "rigid_vessel",
                "placement_mode": "upright",
                "photographed_pose": "standing_on_base",
                "estimated_real_height_cm": 27.0,
                "estimated_height_cm_min": 20.0,
                "estimated_height_cm_max": 34.0,
                "footprint_width_cm": 8.5,
                "material_mode": (
                    "mixed_transparent" if (
                        clear_descriptor or coloured_descriptor
                    ) else "opaque_glossy"
                ),
                "transmission_strength": (
                    0.48 if clear_descriptor else
                    0.36 if coloured_descriptor else 0.0
                ),
            }
            reason = "verified_bottle_prior"
        elif has(evidence, "beverage can", "aluminium can", "aluminum can", "易拉罐", "罐装饮料"):
            profile = {
                "semantic_category": "beverage_can",
                "physical_type": "rigid_container",
                "placement_mode": "upright",
                "photographed_pose": "standing_on_base",
                "estimated_real_height_cm": 12.2,
                "estimated_height_cm_min": 10.0,
                "estimated_height_cm_max": 17.0,
                "footprint_width_cm": 6.6,
                "material_mode": "metal",
                "transmission_strength": 0.0,
            }
            reason = "verified_beverage_can_prior"
        elif has(evidence, "mug", "coffee cup", "tea cup", "马克杯", "咖啡杯", "茶杯"):
            profile = {
                "semantic_category": "mug",
                "physical_type": "rigid_vessel",
                "placement_mode": "upright",
                "photographed_pose": "standing_on_base",
                "estimated_real_height_cm": 10.5,
                "estimated_height_cm_min": 8.0,
                "estimated_height_cm_max": 15.0,
                "footprint_width_cm": 10.0,
                "material_mode": "clear_glass" if clear_descriptor else "opaque_glossy",
                "transmission_strength": 0.90 if clear_descriptor else 0.0,
            }
            reason = "verified_mug_prior"
        elif has(evidence, "vase", "花瓶"):
            profile = {
                "semantic_category": "vase",
                "physical_type": "rigid_vessel",
                "placement_mode": "upright",
                "photographed_pose": "standing_on_base",
                "estimated_real_height_cm": 25.0,
                "estimated_height_cm_min": 17.0,
                "estimated_height_cm_max": 38.0,
                "footprint_width_cm": 13.5,
                "material_mode": (
                    "clear_glass" if clear_descriptor else
                    "frosted_glass" if frosted_descriptor else
                    "colored_glass" if coloured_descriptor else
                    "opaque_glossy"
                ),
                "transmission_strength": (
                    0.93 if clear_descriptor else
                    0.55 if frosted_descriptor else
                    0.42 if coloured_descriptor else 0.0
                ),
            }
            reason = "verified_vase_prior"
        elif has(evidence, "bowl", "dish", "碗", "钵"):
            profile = {
                "semantic_category": "bowl",
                "physical_type": "rigid_vessel",
                "estimated_real_height_cm": 8.5,
                "estimated_height_cm_min": 5.0,
                "estimated_height_cm_max": 14.0,
                "footprint_width_cm": 17.0,
                "material_mode": "opaque_glossy",
                "transmission_strength": 0.0,
            }
            reason = "verified_bowl_prior"
        elif has(evidence, "box", "carton", "book", "package", "盒", "纸盒", "书"):
            profile = {
                "semantic_category": "boxed_product",
                "physical_type": "rigid_package",
                "estimated_real_height_cm": 21.0,
                "estimated_height_cm_min": 14.0,
                "estimated_height_cm_max": 34.0,
                "footprint_width_cm": 15.0,
                "material_mode": "wood_paper_fabric",
                "transmission_strength": 0.0,
            }
            reason = "verified_box_prior"
        elif has(evidence, "figurine", "statue", "sculpture", "ornament", "摆件", "雕像", "雕塑"):
            profile = {
                "semantic_category": "figurine",
                "physical_type": "rigid_object",
                "placement_mode": "upright",
                "photographed_pose": "standing_on_base",
                "estimated_real_height_cm": 21.0,
                "estimated_height_cm_min": 12.0,
                "estimated_height_cm_max": 35.0,
                "footprint_width_cm": 10.5,
                "material_mode": "opaque_glossy",
                "transmission_strength": 0.0,
            }
            reason = "verified_figurine_prior"
        elif has(evidence, "potted plant", "planter", "artificial plant", "盆栽", "植物"):
            profile = {
                "semantic_category": "potted_plant",
                "physical_type": "rigid_display_object",
                "placement_mode": "upright",
                "photographed_pose": "standing_on_base",
                "estimated_real_height_cm": 31.0,
                "estimated_height_cm_min": 20.0,
                "estimated_height_cm_max": 50.0,
                "footprint_width_cm": 18.0,
                "material_mode": "opaque_matte",
                "transmission_strength": 0.0,
            }
            reason = "verified_plant_prior"

        if profile is None:
            enriched.append(item)
            continue

        changed: dict[str, dict[str, Any]] = {}
        # Category/type are descriptive and safe to fill only while they are
        # still generic.  They do not alter final product pixels themselves.
        for key, generic_values in {
            "semantic_category": {"", "decor_object"},
            "physical_type": {"", "rigid_object"},
        }.items():
            current = str(item.get(key, ""))
            if current in generic_values and key in profile:
                item[key] = profile[key]
                changed[key] = {"from": current, "to": item[key]}

        if is_default_size(item):
            for key in default_size:
                changed[key] = {"from": item.get(key), "to": profile[key]}
                item[key] = profile[key]
            old_confidence = float(item.get("size_confidence", 0.0))
            item["size_confidence"] = max(old_confidence, confidence * 0.70)
            changed["size_confidence"] = {
                "from": old_confidence, "to": item["size_confidence"],
            }

        pose_confidence = float(item.get("pose_confidence", 0.0))
        if pose_confidence < 0.25:
            for key in ("placement_mode", "photographed_pose"):
                if key in profile and item.get(key) != profile[key]:
                    changed[key] = {"from": item.get(key), "to": profile[key]}
                    item[key] = profile[key]
            if "photographed_pose" in profile:
                item["pose_confidence"] = max(pose_confidence, confidence * 0.74)
                changed["pose_confidence"] = {
                    "from": pose_confidence, "to": item["pose_confidence"],
                }

        material_confidence = float(item.get("material_confidence", 0.0))
        current_material = canonical_material_mode(str(item.get("material_mode", "opaque")))
        if material_confidence < 0.25 and current_material == "opaque":
            item["material_mode"] = profile["material_mode"]
            item["transmission_strength"] = profile["transmission_strength"]
            inferred_material_confidence = confidence * 0.76
            if (
                profile["material_mode"] == "clear_glass"
                and clear_descriptor
                and confidence >= 0.80
            ):
                # The optical renderer intentionally requires >=0.80 before
                # replacing source-room scenery in colourless glass.  An
                # explicit, verified "clear/transparent glass" observation
                # is strong enough to cross that contract; keeping it at
                # locator*0.76 silently routed the same vessel through the old
                # conservative matte and could even break thin handles.
                inferred_material_confidence = max(
                    inferred_material_confidence, 0.82,
                )
            item["material_confidence"] = max(
                material_confidence, inferred_material_confidence,
            )
            changed["material_mode"] = {
                "from": current_material, "to": item["material_mode"],
            }
            changed["transmission_strength"] = {
                "from": original.get("transmission_strength", 0.0),
                "to": item["transmission_strength"],
            }
            changed["material_confidence"] = {
                "from": material_confidence, "to": item["material_confidence"],
            }

        if changed:
            corrections.append({
                "product_id": str(item.get("id", "")),
                "reason": reason,
                "subject_label": label,
                "locator_confidence": round(confidence, 4),
                "changes": changed,
            })
        enriched.append(item)
    return enriched, corrections


def _advisor_support_consensus(attempts: list[MiniMaxAdvice]) -> float | None:
    if len(attempts) < 2:
        return None
    masks: list[np.ndarray] = []
    for advice in attempts:
        points = np.asarray(advice.support_polygon, np.float32)
        points[:, 0] *= 511.0
        points[:, 1] *= 511.0
        mask = np.zeros((512, 512), np.uint8)
        cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 1)
        masks.append(mask.astype(bool))
    scores: list[float] = []
    for first in range(len(masks)):
        for second in range(first + 1, len(masks)):
            union = int(np.count_nonzero(masks[first] | masks[second]))
            scores.append(float(np.count_nonzero(masks[first] & masks[second]) / max(1, union)))
    return max(scores) if scores else None


def _advisor_consensus_polygon(attempts: list[MiniMaxAdvice]) -> list[list[float]] | None:
    candidates = [item for item in attempts if item.route != "reject"]
    if len(candidates) < 2:
        return None
    resolution = 1024
    intersection = np.ones((resolution, resolution), np.uint8)
    for advice in candidates:
        points = np.asarray(advice.support_polygon, np.float32)
        points[:, 0] *= resolution - 1
        points[:, 1] *= resolution - 1
        mask = np.zeros_like(intersection)
        cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 1)
        intersection &= mask
    contours, _ = cv2.findContours(intersection, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < resolution * resolution * 0.015:
        return None
    simplified = cv2.approxPolyDP(contour, resolution * 0.003, True).reshape(-1, 2)
    if len(simplified) < 3:
        return None
    return [
        [float(point[0] / (resolution - 1)), float(point[1] / (resolution - 1))]
        for point in simplified
    ]


def _recover_scene_from_advisor_attempts(
    background: Path,
    attempts: list[MiniMaxAdvice],
    local_error: Exception,
) -> dict[str, Any]:
    """Recover a receiver only from two agreeing independent cloud passes.

    This path is reserved for optically difficult surfaces such as transparent
    glass, where monocular depth can describe the wall visible *through* the
    tabletop.  It intentionally requires stronger agreement than normal local
    reconciliation and then returns deterministic mask pixels for all later
    placement and audit stages.
    """
    if len(attempts) < 2:
        raise RuntimeError(
            "Local receiver geometry failed and two MiniMax verification passes were not available"
        )
    usable = [item for item in attempts[:2] if item.route != "reject"]
    if len(usable) != 2 or min(item.confidence for item in usable) < 0.80:
        raise RuntimeError(
            "Local receiver geometry failed and MiniMax did not provide two confident non-reject results"
        )
    consensus_score = _advisor_support_consensus(usable)
    consensus_polygon = _advisor_consensus_polygon(usable)
    if (
        consensus_score is None
        or consensus_score < 0.62
        or consensus_polygon is None
    ):
        raise RuntimeError(
            "Local receiver geometry failed and the two MiniMax support polygons did not agree"
        )

    image = load_background(background)
    height, width = image.shape[:2]
    points = np.asarray(consensus_polygon, np.float32).copy()
    points[:, 0] *= max(1, width - 1)
    points[:, 1] *= max(1, height - 1)
    support_mask = np.zeros((height, width), np.uint8)
    cv2.fillPoly(support_mask, [np.rint(points).astype(np.int32)], 1)
    area_ratio = float(support_mask.mean())
    ys, xs = np.nonzero(support_mask)
    if not len(ys):
        raise RuntimeError("MiniMax consensus support polygon is empty")
    span_x = float((xs.max() - xs.min() + 1) / max(1, width))
    span_y = float((ys.max() - ys.min() + 1) / max(1, height))
    top_occupancy = float(support_mask[0].mean())
    bottom_occupancy = float(support_mask[-1].mean())
    if not (
        0.015 <= area_ratio <= 0.65
        and span_x >= 0.16
        and span_y >= 0.045
        and top_occupancy < 0.15
        and not (top_occupancy > 0.04 and bottom_occupancy > 0.25)
    ):
        raise RuntimeError(
            "MiniMax consensus support polygon failed deterministic size/boundary validation"
        )

    chosen = replace(_choose_advice(usable), support_polygon=consensus_polygon)
    signature = hashlib.sha256(
        (json.dumps(consensus_polygon, sort_keys=True) + str(chosen.request_id)).encode("utf-8")
    ).hexdigest()[:18]
    mask_path = ADVISOR_CACHE / "reconciled_masks" / f"cloud_recovery_{signature}.png"
    save_png(mask_path, support_mask * 255)
    base = {
        "support_polygon": consensus_polygon,
        "protected_polygons": [],
        "support_mask_path": str(mask_path.resolve()),
        "front_edge_y": float(ys.max() / max(1, height - 1)),
        "reflectivity": float(chosen.reflectivity),
        "reflection_recommended": bool(chosen.reflection_recommended),
        "method": "local-geometry-invalid",
        "confidence": 0.0,
        "shadow_direction": list(chosen.shadow_direction),
        "shadow_length_ratio": (
            0.045 if chosen.cast_shadow_mode == "directional" else 0.0
        ),
        "shadow_opacity": (
            float(np.clip(0.28 * chosen.direct_light_fraction, 0.08, 0.28))
            if chosen.cast_shadow_mode == "directional" else 0.0
        ),
        "shadow_softness_ratio": 0.17,
        "surface_material": chosen.surface_material,
        "surface_roughness": chosen.surface_roughness,
        "surface_transmittance": chosen.surface_transmittance,
        "cast_shadow_mode": chosen.cast_shadow_mode,
        "direct_light_fraction": chosen.direct_light_fraction,
        "shadow_confidence": chosen.shadow_confidence,
        "reflection_confidence": chosen.reflection_confidence,
            "contact_ao_strength": 1.10 if chosen.surface_material == "transparent_glass" else 1.0,
    }
    result = reconcile_scene_result(
        base,
        chosen,
        (height, width),
        ADVISOR_CACHE / "reconciled_masks",
        advisor_support_consensus=consensus_score,
    )
    result["method"] = "minimax-m3-dual-consensus-recovery"
    result["confidence"] = float(min(
        0.92,
        0.58 * min(item.confidence for item in usable) + 0.42 * consensus_score,
    ))
    result["advisor"].update({
        "local_geometry_valid": False,
        "local_failure": _friendly_error(local_error),
        "support_geometry_source": "minimax_dual_consensus",
        "support_recovery_reason": "transparent_or_optically_ambiguous_local_surface",
        "attempts": [_advice_attempt(item, index) for index, item in enumerate(usable)],
        "support_recovery_polygon": "conservative_intersection_of_independent_passes",
    })
    return result


def _advisor_shadow_agreement(attempts: list[MiniMaxAdvice]) -> float | None:
    directions = [
        np.asarray(item.shadow_direction, np.float64)
        for item in attempts if item.route != "reject"
    ]
    if len(directions) < 2:
        return None
    agreements: list[float] = []
    for first in range(len(directions)):
        for second in range(first + 1, len(directions)):
            agreements.append(float(np.dot(directions[first], directions[second])))
    return max(agreements) if agreements else None


def _layout_advice_needs_retry(advice: MiniMaxAdvice) -> bool:
    if advice.route == "reject":
        return advice.confidence < 0.76
    if advice.confidence < 0.72:
        return True
    products = list(getattr(advice, "products", []))
    if not products:
        return False
    return bool(
        min(item.viewpoint_compatibility for item in products) < 0.68
        or float(np.mean([item.size_confidence for item in products])) < 0.62
    )


def _runninghub_retryable(reason: str) -> bool:
    normalized = str(reason).lower()
    permanent = (
        "not_enough_power", "power value", "quota", "insufficient", "unauthorized",
        "forbidden", "api key", "authentication", "workflow", "application id",
        "input node", "disabled", "aspect ratio", "quality gates",
    )
    if any(token in normalized for token in permanent):
        return False
    transient = (
        "timeout", "timed out", "connection", "temporarily", "502", "503", "504",
        "rate limit", "too many requests", "network", "remote disconnected",
    )
    return any(token in normalized for token in transient)


def _color_gain(
    rgba: Any,
    background_rgb: Any,
    placement: dict[str, Any],
    support_polygon: list[list[float]] | None = None,
    adaptation_strength: float = 1.0,
    *,
    surface_material: str = "unknown",
) -> tuple[float, float, float]:
    """Estimate restrained exposure and illuminant adaptation.

    A single scalar made a cool source photograph remain blue on a warm desk.
    Per-channel gains are estimated only from bright, low-chroma pixels, which
    act as an illuminant probe.  Saturated product colours are never treated as
    neutral, and the final white-balance correction is deliberately bounded.

    Opaque receivers can provide a useful local reflectance sample near the
    contact point.  A transparent or mirror receiver cannot: its pixels encode
    transmitted scenery and view-dependent reflections rather than the scene
    illuminant.  For those receivers, use the background around the product's
    full displayed height and deliberately exclude the contact strip.  This is
    still a uniform, low-frequency photographic correction; no product detail
    or alpha pixel is synthesized here.
    """
    import numpy as np

    foreground = rgba[..., :3][rgba[..., 3] >= 200]
    if len(foreground) < 32:
        return (1.0, 1.0, 1.0)
    height, width = background_rgb.shape[:2]
    cx = int(float(placement["center_x"]) * width)
    by = int(float(placement["baseline_y"]) * height)
    displayed_height = max(
        16, int(round(float(placement["height_ratio"]) * height)),
    )
    radius_y = max(8, int(displayed_height * 0.42))
    radius_x = max(12, int(radius_y * rgba.shape[1] / max(1, rgba.shape[0])))
    top, bottom = max(0, by - radius_y), min(height, by + max(4, radius_y // 3))
    left, right = max(0, cx - radius_x), min(width, cx + radius_x)
    surface = background_rgb[top:bottom, left:right]
    if surface.size == 0:
        return (1.0, 1.0, 1.0)
    source_luma = foreground @ np.array([0.2126, 0.7152, 0.0722])
    surface_pixels = surface.reshape(-1, 3)
    if support_polygon and len(support_polygon) >= 3:
        support = np.zeros((height, width), np.uint8)
        points = np.asarray(support_polygon, np.float32).copy()
        points[:, 0] *= width - 1
        points[:, 1] *= height - 1
        cv2.fillPoly(support, [np.rint(points).astype(np.int32)], 1)
        local_support = support[top:bottom, left:right].astype(bool)
        supported_pixels = surface[local_support]
        if len(supported_pixels) >= 32:
            surface_pixels = supported_pixels
    optical_receiver = str(surface_material).strip().lower() in {
        "mirror", "transparent_glass",
    }
    if optical_receiver:
        # Estimate the illumination from the scene the placed product will
        # visually occupy, not from the view-dependent receiver at its base.
        # The expanded horizontal neighbourhood prevents one narrow backdrop
        # feature from becoming a grey card; excluding the lowest 12% avoids
        # sampling the receiver's contact reflection once more.
        aspect = rgba.shape[1] / max(1.0, float(rgba.shape[0]))
        half_environment_width = max(
            16,
            int(round(displayed_height * max(0.32, aspect * 0.72))),
        )
        environment_top = max(0, by - int(round(displayed_height * 1.05)))
        environment_bottom = max(0, min(
            height, by - max(2, int(round(displayed_height * 0.12))),
        ))
        environment_left = max(0, cx - half_environment_width)
        environment_right = min(width, cx + half_environment_width + 1)
        environment = background_rgb[
            environment_top:environment_bottom,
            environment_left:environment_right,
        ]
        if environment.size and environment.reshape(-1, 3).shape[0] >= 32:
            surface_pixels = environment.reshape(-1, 3)

    target_luma = surface_pixels @ np.array([0.2126, 0.7152, 0.0722])
    current = max(1.0, float(np.percentile(source_luma, 65)))
    # Dark or saturated products do not become pale merely because the desk is
    # bright.  Strong exposure expansion was the main reason red clay and wood
    # products looked pasted into the warm conference table.  The restricted
    # interval remains a low-frequency photographic correction, not relight.
    target_level = max(1.0, float(np.percentile(target_luma, 65)))
    # Product albedo and receiver luminance are not exposure references.  The
    # previous ``target * 0.54`` heuristic could force an already-correct can,
    # label or lacquer finish down to 0.80 and erase its gloss.  Preserve the
    # photographed exposure; only a small, evidence-gated colour-temperature
    # adaptation is authorised below.
    exposure = 1.0

    def illuminant(
        pixels: Any,
        *,
        relative_chroma_limit: float = 0.20,
    ) -> Any | None:
        values = pixels.astype(np.float32)
        luma = values @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        chroma = values.max(axis=1) - values.min(axis=1)
        bright = luma >= np.percentile(luma, 55)
        neutral = chroma <= (luma * relative_chroma_limit + 12.0)
        candidates = values[bright & neutral]
        if len(candidates) < 32:
            return None
        estimate = np.median(candidates, axis=0)
        normalizer = float(estimate @ np.array([0.2126, 0.7152, 0.0722]))
        return estimate / max(1.0, normalizer)

    source_light = illuminant(foreground)
    target_light = illuminant(
        surface_pixels,
        # Beige walls and warm ambient light are valid illumination evidence,
        # whereas equally chromatic product pixels must not become a grey card.
        relative_chroma_limit=0.32 if optical_receiver else 0.20,
    )
    white_balance = np.ones(3, np.float32)
    if source_light is not None and target_light is not None:
        raw_balance = np.clip(
            target_light / np.maximum(source_light, 0.25), 0.82, 1.22,
        )
        # Product highlights are not calibrated grey cards.  Treating their
        # arbitrary green/magenta content as a full three-channel illuminant
        # occasionally produced a conspicuous magenta cast on a warm wooden
        # desk.  Project the estimate onto the stable warm/cool axis; exposure
        # remains controlled by the scalar term above and product colours keep
        # their original channel relationships apart from this low-frequency
        # colour-temperature correction.
        log_balance = np.log(np.maximum(raw_balance, 1e-4))
        temperature = float(np.clip(
            (log_balance[0] - log_balance[2]) * 0.5, -0.09, 0.09,
        ))
        white_balance = np.asarray(
            [np.exp(temperature), 1.0, np.exp(-temperature)], np.float32,
        )
        wb_luma = float(white_balance @ np.array([0.2126, 0.7152, 0.0722]))
        white_balance /= max(0.5, wb_luma)
    strength = float(np.clip(adaptation_strength, 0.0, 1.5))
    adapted_white_balance = 1.0 + (white_balance - 1.0) * strength
    gains = np.clip(exposure * adapted_white_balance, 0.95, 1.05)
    return tuple(float(value) for value in gains)


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return Path(__file__).with_name("ui_index.html").read_text(encoding="utf-8")


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "version": "3.78",
        "root": str(ROOT),
        "minimax": minimax_status(),
        "runninghub": runninghub_status(),
        "hybrid_render": runninghub_hybrid_status(),
        "learned_effects": runninghub_learned_effects_status(),
    }


@app.get("/api/integrations")
def integrations() -> dict[str, Any]:
    settings = get_public_settings()
    return {
        **settings,
        "runninghub_learned_status": runninghub_learned_effects_status(),
    }


@app.post("/api/integrations")
def update_integrations(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        settings = save_integrations(payload)
        return {
            **settings,
            "runninghub_learned_status": runninghub_learned_effects_status(),
        }
    except SettingsError as exc:
        raise HTTPException(400, f"云端配置保存失败：{exc}") from exc


@app.post("/api/integrations/test")
def test_integrations(payload: dict[str, Any]) -> dict[str, Any]:
    provider = str(payload.get("provider", ""))
    try:
        if provider == "minimax":
            result = test_minimax_connection()
            settings = get_public_settings()
            detected_base = str(result.get("base_url", ""))
            if detected_base and detected_base != settings["minimax_base_url"]:
                save_integrations({
                    **settings,
                    "minimax_base_url": detected_base,
                })
            return result
        if provider in {"runninghub", "runninghub_learned"}:
            settings = get_public_settings()
            if provider == "runninghub_learned":
                learned_status = runninghub_learned_effects_status()
                if not learned_status["enabled"]:
                    raise RunningHubError(
                        "学习光影配置总开关尚未开启"
                    )
                if not learned_status["secret_configured"]:
                    raise RunningHubError(
                        "学习光影尚未配置 RunningHub 密钥"
                    )
                if not learned_status["workflow_configured"]:
                    raise RunningHubError(
                        "学习光影 V1 尚未配置完整：应用/工作流、无影场景、"
                        "商品 ID 蒙版、几何、法线、提示词和阴影输出节点均为必填"
                    )
                application_id = str(settings["runninghub_learned_app_id"])
                application_mode = str(settings["runninghub_learned_mode"])
            else:
                application_id = str(settings["runninghub_cutout_id"])
                application_mode = str(settings["runninghub_mode"])
            selected_base = str(settings["runninghub_base_url"])
            alternate_base = (
                "https://www.runninghub.ai"
                if selected_base.endswith(".cn")
                else "https://www.runninghub.cn"
            )
            errors: list[str] = []
            for candidate in (selected_base, alternate_base):
                try:
                    client = RunningHubClient(
                        get_secret("runninghub_api_key"),
                        candidate,
                    )
                    # Reading an app definition can succeed for a public app
                    # even when the supplied key belongs to the other region.
                    # A tiny real upload is therefore part of the connection
                    # test; this is the permission required by every cutout.
                    client.test_upload()
                    result = client.test_configuration(
                        application_id,
                        application_mode,
                    )
                    corrected = candidate != selected_base
                    if corrected:
                        settings = save_integrations({
                            **settings,
                            "runninghub_base_url": candidate,
                        })
                    return {
                        **result,
                        "mode": application_mode,
                        "base_url": candidate,
                        "upload_verified": True,
                        "region_auto_corrected": corrected,
                        "provider": provider,
                        "contract": (
                            "scene-shadow-alpha-v1"
                            if provider == "runninghub_learned"
                            else "foreground-alpha"
                        ),
                        "learned_effects": (
                            runninghub_learned_effects_status()
                            if provider == "runninghub_learned"
                            else None
                        ),
                    }
                except Exception as candidate_error:
                    errors.append(f"{candidate}: {_friendly_error(candidate_error)}")
            raise RunningHubError(
                "RunningHub 密钥在中国站和国际站都无法执行真实上传："
                + "；".join(errors)
            )
        raise HTTPException(400, "未知云端服务")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, _friendly_error(exc)) from exc


@app.get("/api/file")
def get_file(path: str):
    return FileResponse(local_path(path))


@app.get("/api/deliver")
def deliver_file(path: str):
    """Download a production artifact only after its authoritative gate passes.

    ``/api/file`` intentionally remains the unrestricted preview route used by
    the existing canvas.  Batch systems and customer delivery must use this
    endpoint, which re-checks the image hash on every request.
    """

    final_path = local_path(path)
    gate_path = final_path.parent / "final_gate.json"
    if not gate_path.is_file():
        raise HTTPException(403, "正式交付门禁尚未完成；当前文件只能预览")
    try:
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        verify_deliverable(final_path, gate)
    except (OSError, ValueError, TypeError, json.JSONDecodeError, FinalGateError) as exc:
        raise HTTPException(403, f"正式交付被拒绝：{exc}") from exc
    return FileResponse(
        final_path,
        media_type="image/png",
        filename=final_path.name,
        headers={
            "X-Final-Gate": "pass",
            "X-Final-SHA256": str(gate["inputs"]["final_sha256"]),
        },
    )


@app.post("/api/upload-image")
async def upload_image(request: Request, filename: str = "image.png") -> dict[str, Any]:
    """Receive a file selected by the browser's native file picker.

    Browsers intentionally hide the source's absolute path.  Keeping a local
    copy gives the deterministic pipeline a normal filesystem path without
    falling back to Tk's dated file dialog.
    """
    allowed = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
    safe_name = Path(filename).name
    suffix = Path(safe_name).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, "请选择 PNG、JPG、WEBP 或 TIFF 图片")
    payload = await request.body()
    if not payload:
        raise HTTPException(400, "所选图片为空")
    if len(payload) > 150 * 1024 * 1024:
        raise HTTPException(413, "图片超过 150MB，无法载入")
    UPLOADS.mkdir(parents=True, exist_ok=True)
    destination = UPLOADS / f"{time.time_ns()}_{safe_name}"
    try:
        destination.write_bytes(payload)
        with Image.open(destination) as source:
            source.verify()
    except Exception as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(400, f"所选文件不是有效图片：{exc}") from exc
    return {
        "path": str(destination),
        "name": safe_name,
        "size": len(payload),
    }


@app.post("/api/browse")
def browse_legacy(payload: dict[str, Any]) -> dict[str, str]:
    """Compatibility endpoint for old cached pages; new UI never calls it."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        filename = filedialog.askopenfilename(
            initialdir=str(ROOT),
            title=str(payload.get("title", "选择图片")),
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.webp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
        return {"path": filename}
    except Exception as exc:
        raise HTTPException(500, f"无法打开文件选择器，请直接粘贴路径：{exc}") from exc


_IMAGE_BROWSER_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}


def _browser_roots() -> list[str]:
    """Return local filesystem roots for the in-workbench image picker."""
    if os.name == "nt":
        try:
            import ctypes

            mask = int(ctypes.windll.kernel32.GetLogicalDrives())
            return [f"{chr(code)}:\\" for code in range(65, 91) if mask & (1 << (code - 65))]
        except Exception:
            pass
    return [str(Path.cwd().anchor or Path.cwd())]


@app.post("/api/image-browser")
def image_browser(payload: dict[str, Any]) -> dict[str, Any]:
    """List local folders and image thumbnails for the custom workbench picker.

    The workbench is bound to localhost and this endpoint deliberately returns
    only directory names plus image files; it replaces the OS picker whose
    thumbnail cache is often disabled on customer Windows machines.
    """
    raw = str(payload.get("path") or ROOT)
    try:
        folder = Path(raw).expanduser()
        if not folder.is_absolute():
            folder = ROOT / folder
        folder = folder.resolve()
    except (OSError, RuntimeError) as exc:
        raise HTTPException(400, f"无法打开此目录：{exc}") from exc
    if not folder.is_dir():
        raise HTTPException(400, "请选择一个存在的文件夹")

    entries: list[dict[str, str]] = []
    try:
        children = sorted(folder.iterdir(), key=lambda item: (not item.is_dir(), item.name.casefold()))
    except OSError as exc:
        raise HTTPException(400, f"无法读取目录：{exc}") from exc
    for child in children[:500]:
        try:
            if child.is_dir():
                entries.append({"kind": "folder", "name": child.name, "path": str(child)})
            elif child.is_file() and child.suffix.lower() in _IMAGE_BROWSER_SUFFIXES:
                entries.append({
                    "kind": "image",
                    "name": child.name,
                    "path": str(child),
                    "preview": browser_path(str(child)),
                })
        except OSError:
            continue
    return {
        "path": str(folder),
        "parent": str(folder.parent) if folder.parent != folder else "",
        "roots": _browser_roots(),
        "entries": entries,
        "truncated": len(children) > 500,
    }


@app.post("/api/info")
def info(payload: dict[str, Any]) -> dict[str, Any]:
    path = local_path(str(payload["path"]))
    with Image.open(path) as source:
        oriented = ImageOps.exif_transpose(source)
        width, height = oriented.size
        has_alpha = "A" in oriented.getbands()
    result: dict[str, Any] = {
        "path": str(path),
        "width": width,
        "height": height,
        "preview": browser_path(str(path)),
    }
    if has_alpha:
        rgba = load_rgba(path)
        geometry = product_geometry(path)
        result.update({
            "content_width": geometry.content_width,
            "content_height": geometry.content_height,
            "content_aspect": geometry.content_aspect,
            "contact_width_ratio": geometry.contact_width_ratio,
            "alpha_quality": alpha_quality(rgba),
        })
    return result


@app.post("/api/analyze-scene")
def analyze_scene_api(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        background = local_path(str(payload["background"]))
        deep = bool(payload.get("deep", True))
        try:
            analysis = analyze_scene(background, SCENE_CACHE, deep=deep)
        except Exception as local_error:
            if not deep or not minimax_status()["configured"]:
                raise
            # Transparent/reflective receivers can defeat both monocular depth
            # and semantic segmentation.  Do not abort before the configured
            # scene adviser has had a chance to identify the physical rim.
            try:
                attempts = [request_minimax_advice(
                    background,
                    products=[],
                    cache_dir=ADVISOR_CACHE / "scene",
                    attempt=0,
                )]
                attempts.append(request_minimax_advice(
                    background,
                    products=[],
                    cache_dir=ADVISOR_CACHE / "scene",
                    attempt=1,
                    retry_context=(
                        "Local depth/semantic receiver analysis failed. Independently trace only "
                        "the physical tabletop rim, including transparent glass. Error: "
                        + _friendly_error(local_error)
                    ),
                ))
                result = _recover_scene_from_advisor_attempts(
                    background, attempts, local_error,
                )
                result["support_mask"] = browser_path(result["support_mask_path"])
                return result
            except Exception as cloud_error:
                raise RuntimeError(
                    f"{_friendly_error(local_error)}; dual MiniMax recovery failed: "
                    f"{_friendly_error(cloud_error)}"
                ) from cloud_error
        result = analysis.to_dict()
        local_shadow_direction = list(result["shadow_direction"])
        if deep and minimax_status()["configured"]:
            try:
                advice_attempts = [request_minimax_advice(
                    background,
                    products=[],
                    cache_dir=ADVISOR_CACHE / "scene",
                    attempt=0,
                )]
                retry_error = ""
                if (
                    float(result.get("confidence", 0.0)) < 0.65
                    or "transparent-glass" in str(result.get("method", ""))
                    or advice_attempts[0].confidence < 0.72
                    or (advice_attempts[0].route == "reject" and advice_attempts[0].confidence < 0.76)
                ):
                    try:
                        advice_attempts.append(request_minimax_advice(
                            background,
                            products=[],
                            cache_dir=ADVISOR_CACHE / "scene",
                            attempt=1,
                            retry_context=(
                                f"Local support confidence={float(result.get('confidence', 0.0)):.3f}; "
                                f"first adviser confidence={advice_attempts[0].confidence:.3f}."
                            ),
                        ))
                    except Exception as retry_exc:
                        retry_error = _friendly_error(retry_exc)
                advice = _choose_advice(advice_attempts)
                if advice.route == "reject":
                    result["advisor"] = {
                        "status": "reject_recommended",
                        "model": advice.model,
                        "request_id": advice.request_id,
                        "route": advice.route,
                        "confidence": advice.confidence,
                        "warnings": advice.warnings,
                        "attempts": [
                            _advice_attempt(item, index)
                            for index, item in enumerate(advice_attempts)
                        ],
                        "retry_error": retry_error,
                    }
                else:
                    support_consensus = _advisor_support_consensus(advice_attempts)
                    consensus_polygon = _advisor_consensus_polygon(advice_attempts)
                    if (
                        float(result.get("confidence", 0.0)) < 0.60
                        and support_consensus is not None
                        and support_consensus >= 0.58
                        and consensus_polygon is not None
                    ):
                        advice = replace(advice, support_polygon=consensus_polygon)
                    image = load_background(background)
                    result = reconcile_scene_result(
                        result,
                        advice,
                        image.shape[:2],
                        ADVISOR_CACHE / "reconciled_masks",
                        advisor_support_consensus=support_consensus,
                    )
                    result["advisor"]["attempts"] = [
                        _advice_attempt(item, index)
                        for index, item in enumerate(advice_attempts)
                    ]
                    result["advisor"]["retry_error"] = retry_error
                    result["advisor"]["support_recovery_polygon"] = (
                        "conservative_intersection_of_independent_passes"
                        if consensus_polygon is not None else "selected_adviser_polygon"
                    )
                    shadow_agreement = _advisor_shadow_agreement(advice_attempts)
                    result["advisor"]["shadow_direction_consensus_cosine"] = shadow_agreement
                    if shadow_agreement is not None and shadow_agreement < 0.50:
                        result["shadow_direction"] = local_shadow_direction
                        result["advisor"]["shadow_direction_source"] = (
                            "local_pixel_evidence_due_minimax_disagreement"
                        )
                        result["advisor"]["warnings"].append(
                            "MiniMax independent passes disagreed on shadow direction; "
                            "local image-gradient evidence retained."
                        )
                    else:
                        result["advisor"]["shadow_direction_source"] = "minimax_consensus"
            except Exception as advisor_error:
                result["advisor"] = {
                    "status": "fallback_local",
                    "reason": _friendly_error(advisor_error),
                }
        else:
            result["advisor"] = {
                "status": "not_configured" if deep else "preview_local",
            }
        result["support_mask"] = browser_path(result["support_mask_path"])
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, f"桌面分析失败：{_friendly_error(exc)}") from exc


@app.post("/api/segment")
def segment(payload: dict[str, Any]) -> dict[str, Any]:
    _SEGMENT_JOB_LOCK.acquire()
    try:
        source = local_path(str(payload["source"]))
        product_id = str(payload["id"])
        roi = payload.get("roi")
        operator_roi = roi is not None
        quality_mode = str(payload.get("quality", "quality"))
        # U2Net plus the deterministic high-confidence matte cleanup retains
        # dark product interiors more reliably than ISNet on black ceramics.
        model = "u2net" if quality_mode == "quality" else "u2net"
        output = WORK / "rgba" / f"{product_id}_{time.time_ns()}.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        selection_rect: tuple[int, int, int, int] | None = None
        prompt_details: dict[str, Any] | None = None
        selection_strategy = "whole_image_original"
        green_shadow_refined = False
        refined = False
        ensemble_details: dict[str, Any] | None = None
        cloud_details: dict[str, Any] | None = None
        cloud_fallback_reason: str | None = None
        segmentation_attempts: list[dict[str, Any]] = []
        candidate_artifacts: list[dict[str, Any]] = []
        subject_locator: dict[str, Any] | None = None
        auto_subject_roi = False
        segmentation_source = source
        chosen_provider = "local"
        used_prompt_fallback = False

        # Quality-mode direct cutout first asks MiniMax for one complete
        # physical product.  This turns the old whole-scene cloud request into
        # a bounded object request and prevents a clean alpha containing two
        # neighbouring products from ever looking like a successful cutout.
        # A local saliency fallback may still provide a prompt box, but it is
        # explicitly marked unverified and the downstream alpha gates remain
        # authoritative.
        if roi is None and quality_mode == "quality":
            try:
                located = request_minimax_subject_roi(
                    source,
                    ADVISOR_CACHE / "subject_roi",
                    allow_fallback=True,
                )
                raw_bbox = [float(value) for value in located.bbox]
                expanded_bbox = _expand_subject_roi_bbox(
                    raw_bbox,
                    fallback_used=located.fallback_used,
                )
                semantic_verified = bool(
                    not located.fallback_used
                    and not located.ambiguous
                    and located.confidence >= 0.55
                )
                fallback_prompt = bool(
                    located.fallback_used
                    and (expanded_bbox[2] - expanded_bbox[0])
                    * (expanded_bbox[3] - expanded_bbox[1]) <= 0.78
                )
                accepted = semantic_verified or fallback_prompt
                subject_locator = {
                    **located.to_dict(),
                    "bbox": raw_bbox,
                    "expanded_bbox": expanded_bbox,
                    "accepted": accepted,
                    "semantic_verified": semantic_verified,
                    "role": "verified_subject_roi" if semantic_verified else "unverified_prompt_roi",
                }
                segmentation_attempts.append({
                    "provider": "minimax_subject_locator"
                    if not located.fallback_used else "local_subject_locator",
                    "status": "success" if accepted else "rejected",
                    "confidence": float(located.confidence),
                    "subject_label": located.subject_label,
                    "fallback_used": bool(located.fallback_used),
                })
                if accepted:
                    roi = expanded_bbox
                    auto_subject_roi = True
            except Exception as locator_error:
                subject_locator = {
                    "accepted": False,
                    "semantic_verified": False,
                    "role": "locator_failed",
                    "reason": _friendly_error(locator_error),
                }
                segmentation_attempts.append({
                    "provider": "minimax_subject_locator",
                    "status": "failed",
                    "reason": _friendly_error(locator_error),
                })
        if roi is not None:
            rgb = load_background(source)
            crop, selection_rect = crop_normalized_roi(rgb, roi)
            selection_source = WORK / "selections" / f"{product_id}_{time.time_ns()}.png"
            save_png(selection_source, crop)
            segmentation_source = selection_source
            # A box is a crop instruction, not permission to swap algorithms:
            # run the same provider chain used by direct cutout on the selected
            # source pixels. Cloud output contributes alpha only.
            if runninghub_status()["enabled"]:
                try:
                    cloud_details = run_configured_cutout(selection_source, output)
                    segmentation_attempts.append({"provider": "runninghub", "status": "success"})
                    if auto_subject_roi:
                        cloud_candidate = (
                            WORK / "rgba" /
                            f"{product_id}_runninghub_candidate_{time.time_ns()}.png"
                        )
                        save_png(cloud_candidate, load_rgba(output))
                        candidate_artifacts.append({
                            "provider": "runninghub",
                            "path": str(cloud_candidate),
                            "preview": browser_path(str(cloud_candidate)),
                            "quality": dict(cloud_details["quality"]),
                        })
                except Exception as cloud_error:
                    cloud_fallback_reason = _friendly_error(cloud_error)
                    segmentation_attempts.append({
                        "provider": "runninghub", "status": "failed",
                        "reason": cloud_fallback_reason,
                    })
            if cloud_details is None:
                ensemble_details = ensemble_rembg_rgba(
                    selection_source, output, trim=True,
                    promptable_recovery=(quality_mode == "quality"),
                    subject_label=str(
                        (subject_locator or {}).get("subject_label", "")
                    ),
                )
                model = str(ensemble_details["selected_model"])
                refined = bool(ensemble_details["semantic_refinement"])
                segmentation_attempts.append({
                    "provider": "local", "status": "success",
                    "score": ensemble_details["quality"]["score"],
                })
                if auto_subject_roi:
                    candidate_artifacts.append({
                        "provider": "local",
                        "path": str(output),
                        "preview": browser_path(str(output)),
                        "quality": dict(ensemble_details["quality"]),
                    })
            else:
                model = str(cloud_details["model"])
                chosen_provider = "runninghub"
            rgba = load_rgba(output)
            crop_height_coverage = float(rgba.shape[0] / max(1, crop.shape[0]))
            prompt_details = {
                "original_crop_height_coverage": crop_height_coverage,
                "fallback_used": False,
                "ensemble": ensemble_details,
                "runninghub": cloud_details,
                "cloud_fallback_reason": cloud_fallback_reason,
                "subject_locator": subject_locator,
            }
            if auto_subject_roi:
                selection_strategy = (
                    "auto_subject_runninghub_alpha_source_rgb"
                    if cloud_details is not None
                    else "auto_subject_local_promptable_pipeline"
                )
            else:
                selection_strategy = (
                    "cropped_runninghub_alpha_source_rgb"
                    if cloud_details is not None
                    else "cropped_original_pipeline"
                )

            # The selection box changes only the input crop. It must never
            # silently swap in a weaker segmentation algorithm, because that
            # made box selection worse than whole-image cutout. An implausible
            # matte remains rejected by the same quality gate and the user can
            # enlarge the crop.
        else:
            if runninghub_status()["enabled"]:
                try:
                    cloud_details = run_configured_cutout(source, output)
                    segmentation_attempts.append({"provider": "runninghub", "status": "success"})
                except Exception as cloud_error:
                    cloud_fallback_reason = _friendly_error(cloud_error)
                    segmentation_attempts.append({
                        "provider": "runninghub", "status": "failed",
                        "reason": cloud_fallback_reason,
                    })
            if cloud_details is None:
                ensemble_details = ensemble_rembg_rgba(
                    source, output, trim=True,
                    promptable_recovery=(quality_mode == "quality"),
                    subject_label=str(
                        (subject_locator or {}).get("subject_label", "")
                    ),
                )
                model = str(ensemble_details["selected_model"])
                refined = bool(ensemble_details["semantic_refinement"])
                segmentation_attempts.append({
                    "provider": "local", "status": "success",
                    "score": ensemble_details["quality"]["score"],
                })
            else:
                model = str(cloud_details["model"])
                chosen_provider = "runninghub"
            rgba = load_rgba(output)
        geometry = product_geometry(output)
        if cloud_details is not None and not used_prompt_fallback:
            quality = dict(cloud_details["quality"])
        elif ensemble_details is not None and not used_prompt_fallback:
            quality = dict(ensemble_details["quality"])
        else:
            quality = alpha_quality(rgba)
            structure_score = float(quality["score"])
            predicted_iou = float(
                (prompt_details or {}).get("predicted_iou", 0.72)
            )
            completeness_score = float(np.clip(predicted_iou * 100.0, 0.0, 89.0))
            quality["structure_score"] = structure_score
            quality["completeness_score"] = completeness_score
            quality["score"] = min(structure_score, completeness_score)
            quality["score_kind"] = "single_prompt_model_confidence"
            if quality["verdict"] != "reject":
                quality["verdict"] = "review"
        if auto_subject_roi and subject_locator is not None:
            quality = _semantic_alpha_envelope_gate(
                quality,
                rgba,
                crop.shape[:2],
                subject_locator,
            )
            # Candidate previews must expose the same hard-gate result as the
            # selected candidate; otherwise the browser can still show a stale
            # 95/96 for a matte that the route has rejected.
            for artifact in candidate_artifacts:
                if artifact.get("provider") == "runninghub":
                    artifact["quality"] = dict(quality)
        if refined and not bool(quality.get("semantic_envelope_leakage")):
            quality["reason"] = "使用语义约束清理了复杂源背景，请检查细边缘"

        # A merely valid cloud matte is not automatically the best matte.  An
        # automatically located subject is always processed by the local
        # ensemble as an independent second candidate; a manual/whole-image
        # request pays that cost only when the cloud confidence is low.  The
        # chooser is gate-first and deliberately keeps RunningHub when both
        # candidates pass equally, because a clean-but-incomplete fragment can
        # otherwise beat a complete matte on a structural score alone.
        compare_local_candidate = bool(
            cloud_details is not None
            and (
                auto_subject_roi
                or float(quality.get("score", 0.0)) < 90.0
            )
        )
        if compare_local_candidate:
            local_alternate = WORK / "rgba" / f"{product_id}_local_alt_{time.time_ns()}.png"
            try:
                local_details = ensemble_rembg_rgba(
                    segmentation_source, local_alternate, trim=True,
                    promptable_recovery=(quality_mode == "quality"),
                    subject_label=str(
                        (subject_locator or {}).get("subject_label", "")
                    ),
                )
                local_quality = dict(local_details["quality"])
                local_rgba = load_rgba(local_alternate)
                if auto_subject_roi and subject_locator is not None:
                    local_quality = _semantic_alpha_envelope_gate(
                        local_quality,
                        local_rgba,
                        crop.shape[:2],
                        subject_locator,
                    )
                comparison_reason = (
                    "auto_subject_dual_candidate"
                    if auto_subject_roi else "cloud_below_90"
                )
                segmentation_attempts.append({
                    "provider": "local", "status": "success",
                    "score": local_quality["score"], "reason": comparison_reason,
                })
                candidate_artifacts.append({
                    "provider": "local",
                    "path": str(local_alternate),
                    "preview": browser_path(str(local_alternate)),
                    "quality": local_quality,
                })
                if _prefer_local_cutout_candidate(quality, local_quality):
                    save_png(output, local_rgba)
                    rgba = load_rgba(output)
                    geometry = product_geometry(output)
                    ensemble_details = local_details
                    quality = local_quality
                    model = str(local_details["selected_model"])
                    refined = bool(local_details["semantic_refinement"])
                    chosen_provider = "local"
                    if auto_subject_roi:
                        selection_strategy = "auto_subject_local_promptable_pipeline"
                    else:
                        selection_strategy = (
                            "cropped_original_pipeline"
                            if selection_rect else "whole_image_original"
                        )
            except Exception as local_compare_error:
                segmentation_attempts.append({
                    "provider": "local", "status": "failed",
                    "reason": _friendly_error(local_compare_error),
                })

        # Retry RunningHub only for a transient transport failure and only when
        # the local fallback still misses the automatic 90-point confidence
        # target. Permanent quota/auth/workflow failures must not be hammered.
        if (
            cloud_details is None
            and cloud_fallback_reason
            and _runninghub_retryable(cloud_fallback_reason)
            and (
                auto_subject_roi
                or float(quality.get("score", 0.0)) < 90.0
            )
        ):
            # The international gateway can reset several consecutive HTTPS
            # connections (WinError 10054).  The user explicitly selected the
            # quality-first cloud route, so make three bounded orchestration
            # retries after the initial failure.  Permanent auth/quota/workflow
            # errors still stop immediately.  A lost task-creation response may
            # leave an orphaned billed task; this trade-off is recorded in the
            # attempt trace and is preferable to silently accepting a polluted
            # local matte in quality-first mode.
            for retry_index in range(1, 4):
                cloud_alternate = (
                    WORK / "rgba" /
                    f"{product_id}_cloud_retry_{retry_index}_{time.time_ns()}.png"
                )
                try:
                    retry_details = run_configured_cutout(
                        segmentation_source, cloud_alternate,
                    )
                    retry_quality = dict(retry_details["quality"])
                    retry_rgba = load_rgba(cloud_alternate)
                    if auto_subject_roi and subject_locator is not None:
                        retry_quality = _semantic_alpha_envelope_gate(
                            retry_quality,
                            retry_rgba,
                            crop.shape[:2],
                            subject_locator,
                        )
                    segmentation_attempts.append({
                        "provider": "runninghub", "status": "success",
                        "score": retry_quality["score"],
                        "reason": "transient_retry",
                        "retry_index": retry_index,
                        "duplicate_task_risk": True,
                    })
                    candidate_artifacts.append({
                        "provider": "runninghub_retry",
                        "path": str(cloud_alternate),
                        "preview": browser_path(str(cloud_alternate)),
                        "quality": retry_quality,
                    })
                    choose_retry = bool(
                        chosen_provider != "runninghub"
                        and not _prefer_local_cutout_candidate(
                            retry_quality,
                            quality,
                        )
                    )
                    if choose_retry:
                        save_png(output, retry_rgba)
                        rgba = load_rgba(output)
                        geometry = product_geometry(output)
                        cloud_details = retry_details
                        quality = retry_quality
                        model = str(retry_details["model"])
                        chosen_provider = "runninghub"
                        cloud_fallback_reason = None
                        selection_strategy = (
                            "auto_subject_runninghub_alpha_source_rgb"
                            if auto_subject_roi
                            else selection_strategy
                        )
                    break
                except Exception as retry_error:
                    retry_reason = _friendly_error(retry_error)
                    cloud_fallback_reason = retry_reason
                    segmentation_attempts.append({
                        "provider": "runninghub", "status": "failed",
                        "reason": retry_reason, "retry": True,
                        "retry_index": retry_index,
                        "duplicate_task_risk": True,
                    })
                    if (
                        retry_index >= 3
                        or not _runninghub_retryable(retry_reason)
                    ):
                        break
                    time.sleep(0.6 * (2 ** (retry_index - 1)))
        subject_label = str(
            (subject_locator or {}).get("subject_label", "")
        )
        bottom_residue_cleanup: dict[str, Any] = {
            "applied": False,
            "verdict": "pass",
            "reason": "no_subject_semantics",
        }
        if auto_subject_roi and subject_label:
            cleaned_rgba, bottom_residue_cleanup = (
                remove_attached_bottom_surface_residue(rgba, subject_label)
            )
            if bool(bottom_residue_cleanup.get("applied")):
                rgba = trim_transparent(cleaned_rgba, padding=5)
                save_png(output, rgba)
                geometry = product_geometry(output)
                segmentation_attempts.append({
                    "provider": "deterministic_bottom_residue_cleanup",
                    "status": "success",
                    "reason": bottom_residue_cleanup.get("reason"),
                })

        # Recompute the final structural gate with product semantics.  Cloud
        # providers and older cached candidates may report a high component
        # score even when a solid figurine is riddled with alpha holes.
        semantic_structure = alpha_quality(
            rgba, subject_label=subject_label,
        )
        prior_completeness = float(quality.get("completeness_score", 100.0))
        quality.update(semantic_structure)
        quality["structure_score"] = float(semantic_structure["score"])
        quality["completeness_score"] = prior_completeness
        quality["score"] = float(min(
            semantic_structure["score"], prior_completeness,
        ))
        if semantic_structure["verdict"] == "reject":
            quality["verdict"] = "reject"
        elif quality["score"] >= 90.0:
            quality["verdict"] = "pass"
        else:
            quality["verdict"] = "review"
        quality["bottom_residue_cleanup"] = bottom_residue_cleanup

        semantic_panel_repair: dict[str, Any] = {
            "applied": False,
            "reason": "not_a_verified_runninghub_panel",
        }
        if (
            auto_subject_roi
            and subject_locator is not None
            and bool(subject_locator.get("semantic_verified"))
        ):
            repaired_rgba, semantic_panel_repair = solidify_semantic_opaque_panel(
                rgba,
                subject_label,
            )
            if bool(semantic_panel_repair.get("applied")):
                rgba = trim_transparent(repaired_rgba, padding=5)
                # ``alpha_quality`` treats broad contact with two opposite
                # canvas edges as a likely inverted/background matte.  That is
                # correct for an arbitrary crop but false for a verified solid
                # panel that has just been tightly trimmed (clock/book/box):
                # its real cabinet naturally spans both left and right edges.
                # Add transparent audit/placement breathing room instead of
                # weakening the global inversion gate.  Visible RGB and alpha
                # are untouched; the border contains alpha=0 only.
                audit_border = 0
                if bool(semantic_panel_repair.get(
                    "requires_transparent_audit_border", True,
                )):
                    audit_border = max(
                        8,
                        int(round(min(rgba.shape[:2]) * 0.04)),
                    )
                    rgba = np.pad(
                        rgba,
                        (
                            (audit_border, audit_border),
                            (audit_border, audit_border),
                            (0, 0),
                        ),
                        mode="constant",
                        constant_values=0,
                    )
                semantic_panel_repair["transparent_audit_border_px"] = audit_border
                save_png(output, rgba)
                geometry = product_geometry(output)
                quality = alpha_quality(rgba, subject_label=subject_label)
                quality["semantic_opaque_panel_repair"] = semantic_panel_repair
                quality["structure_score"] = float(quality.get("score", 0.0))
                # A deterministic fill can restore source RGB, but it is not
                # independent evidence that the inferred area is correct.  Do
                # not erase the ensemble confidence or award an automatic 100.
                repair_confidence = float(
                    semantic_panel_repair.get("repair_confidence_score", 90.0)
                )
                quality["completeness_score"] = float(min(
                    prior_completeness,
                    repair_confidence,
                ))
                quality["score"] = float(min(
                    quality["structure_score"],
                    quality["completeness_score"],
                ))
                quality["verdict"] = (
                    "reject"
                    if str(quality.get("verdict")) == "reject"
                    else ("pass" if quality["score"] >= 90.0 else "review")
                )
                quality["score_kind"] = "semantic_opaque_panel_source_rgb_repair"
                quality["bottom_residue_cleanup"] = bottom_residue_cleanup
                if subject_locator is not None:
                    quality = _semantic_alpha_envelope_gate(
                        quality,
                        rgba,
                        crop.shape[:2],
                        subject_locator,
                    )
                segmentation_attempts.append({
                    "provider": "deterministic_alpha_repair",
                    "status": "success",
                    "reason": semantic_panel_repair.get("reason"),
                    "score": quality.get("score"),
                })
                selection_strategy = (
                    "auto_subject_runninghub_alpha_source_rgb_opaque_panel_repair"
                )
        if prompt_details is not None:
            prompt_details["ensemble"] = ensemble_details
            prompt_details["runninghub"] = cloud_details
            prompt_details["selected_provider"] = chosen_provider
            prompt_details["semantic_opaque_panel_repair"] = semantic_panel_repair
            prompt_details["bottom_residue_cleanup"] = bottom_residue_cleanup
        cutout_ready = bool(
            str(quality.get("verdict", "reject")) == "pass"
            and not bool(quality.get("multi_object_leakage"))
            and not bool(quality.get("opposite_edge_attachment"))
            and not bool(quality.get("semantic_envelope_leakage"))
            and not bool(quality.get("opaque_topology_suspect"))
            and str(bottom_residue_cleanup.get("verdict", "pass")) == "pass"
            and str(
                semantic_panel_repair.get(
                    "repair_expansion_verdict", "not_applicable",
                )
            ) != "reject"
        )
        return {
            "rgba": str(output),
            "preview": browser_path(str(output)),
            "content_aspect": geometry.content_aspect,
            "alpha_quality": quality,
            "model": model,
            "semantic_refinement": refined,
            "selection_mode": (
                "auto_subject_box"
                if auto_subject_roi
                else ("box" if selection_rect else "whole_image")
            ),
            "selection_strategy": selection_strategy,
            "selection_rect": list(selection_rect) if selection_rect else None,
            "prompt_details": prompt_details,
            "green_shadow_refinement": green_shadow_refined,
            "ensemble": ensemble_details,
            "provider": chosen_provider,
            "runninghub": cloud_details,
            "cloud_fallback_reason": cloud_fallback_reason,
            "subject_locator": subject_locator,
            "semantic_opaque_panel_repair": semantic_panel_repair,
            "bottom_residue_cleanup": bottom_residue_cleanup,
            "candidate_artifacts": candidate_artifacts,
            "cutout_ready": cutout_ready,
            "attempts": segmentation_attempts,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, f"抠图失败：{_friendly_error(exc)}") from exc


    finally:
        _SEGMENT_JOB_LOCK.release()


@app.post("/api/create-identity-mask")
def create_identity_mask(payload: dict[str, Any]) -> dict[str, Any]:
    """Create explicit text/logo protection evidence in source-RGBA space.

    The UI intentionally creates a conservative rectangle first.  It is
    intersected with the real product alpha, so the later homography cannot
    turn a label selection into permission to alter the surrounding source
    background.  Fine OCR/polygon masks can replace this artifact later
    without changing the composition contract.
    """
    try:
        rgba_path = local_path(str(payload["rgba"]))
        roi = np.asarray(payload.get("roi"), dtype=np.float32)
        if roi.shape != (4,) or not np.isfinite(roi).all():
            raise ValueError("identity roi must be [x0,y0,x1,y1]")
        x0, y0, x1, y1 = (float(value) for value in roi)
        if min(x0, y0) < 0.0 or max(x1, y1) > 1.0 or x1 <= x0 or y1 <= y0:
            raise ValueError("identity roi must be normalized and non-empty")
        if x1 - x0 < 0.004 or y1 - y0 < 0.004:
            raise ValueError("identity roi is too small")
        rgba = load_rgba(rgba_path)
        height, width = rgba.shape[:2]
        left = int(np.floor(x0 * width))
        top = int(np.floor(y0 * height))
        right = int(np.ceil(x1 * width))
        bottom = int(np.ceil(y1 * height))
        left, right = max(0, left), min(width, right)
        top, bottom = max(0, top), min(height, bottom)
        mask = np.zeros((height, width), np.uint8)
        mask[top:bottom, left:right] = 255
        mask[rgba[..., 3] <= 4] = 0
        if not np.any(mask):
            raise ValueError("selected identity area does not overlap the product")
        output = WORK / "identity_masks" / f"identity_{time.time_ns()}.png"
        save_png(output, mask)
        return {
            "mask": str(output),
            "preview": browser_path(str(output)),
            "pixels": int(np.count_nonzero(mask)),
            "source_dimensions": [width, height],
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, f"identity mask creation failed: {_friendly_error(exc)}") from exc


@app.post("/api/auto-layout")
def auto_layout(payload: LayoutUI) -> dict[str, Any]:
    try:
        background = local_path(payload.background)
        image = load_background(background)
        height, width = image.shape[:2]
        support_mask = _layout_support_mask(
            payload.support_mask_path, (width, height),
        )
        # ``None`` is the explicit geometry contract for ambient-only scenes:
        # there is no directional projection corridor to reserve.  Keeping a
        # unit vector with length zero makes an identity-warped product body
        # look like a shadow and incorrectly forces the whole silhouette into
        # the tabletop polygon.
        layout_shadow_direction = (
            None
            if float(payload.shadow_length_ratio) <= 1.0e-4
            else payload.shadow_direction
        )
        raw_source_products = [
            _product_dict(product) for product in payload.products
        ]
        source_products, local_semantic_priors = (
            _local_subject_semantic_priors(raw_source_products)
        )
        effective_support = payload.support_polygon
        effective_protected = payload.protected_polygons
        advisor_result: dict[str, Any] = {"status": "not_configured"}
        products: list[dict[str, Any]]
        metrics: dict[str, Any]

        def local_candidate(
            candidate_products: list[dict[str, Any]] | None = None,
        ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
            candidate_source = candidate_products or source_products
            candidate, solver_metrics = solve_aesthetic_layout(
                (width, height),
                payload.support_polygon,
                payload.protected_polygons,
                candidate_source,
                shadow_direction=layout_shadow_direction,
                shadow_length_ratio=payload.shadow_length_ratio,
                shadow_softness_ratio=payload.shadow_softness_ratio,
                background_rgb=image,
                support_mask=support_mask,
            )
            diagnostics = _layout_surface_diagnostics(
                (width, height),
                payload.support_polygon,
                payload.protected_polygons,
                candidate,
                support_mask=support_mask,
            )
            return candidate, _merge_layout_metrics(solver_metrics, diagnostics)

        if minimax_status()["configured"]:
            try:
                first_advice = request_minimax_advice(
                    background,
                    products=source_products,
                    cache_dir=ADVISOR_CACHE / "layout",
                    attempt=0,
                )
                first_advice, first_semantic_corrections = (
                    _reconcile_advice_with_subject_evidence(
                        first_advice, source_products,
                    )
                )
                advice_attempts = [first_advice]
                semantic_corrections = list(first_semantic_corrections)
                retry_error = ""
                if _layout_advice_needs_retry(advice_attempts[0]):
                    try:
                        retry_advice = request_minimax_advice(
                            background,
                            products=source_products,
                            cache_dir=ADVISOR_CACHE / "layout",
                            attempt=1,
                            retry_context=(
                                "The first pass was uncertain about physical scale, source pose, "
                                "or viewpoint compatibility. Independently verify all three."
                            ),
                        )
                        retry_advice, retry_semantic_corrections = (
                            _reconcile_advice_with_subject_evidence(
                                retry_advice, source_products,
                            )
                        )
                        advice_attempts.append(retry_advice)
                        semantic_corrections.extend(retry_semantic_corrections)
                    except Exception as retry_exc:
                        retry_error = _friendly_error(retry_exc)
                advice = _choose_advice(advice_attempts)
                scene_receiver_authority = False
                if (
                    payload.receiver_width_or_diameter_cm is not None
                    and payload.receiver_size_confidence >= 0.40
                    and payload.receiver_size_confidence
                    >= float(getattr(
                        advice, "receiver_size_confidence", 0.0,
                    )) - 0.10
                ):
                    advice = replace(
                        advice,
                        receiver_width_or_diameter_cm=float(
                            payload.receiver_width_or_diameter_cm
                        ),
                        receiver_size_confidence=float(
                            payload.receiver_size_confidence
                        ),
                    )
                    scene_receiver_authority = True
                if advice.route == "reject":
                    raise HTTPException(
                        422,
                        "MiniMax scene compatibility analysis recommends reject: "
                        + "; ".join(advice.warnings[:3])
                    )
                incompatible = _consensus_incompatible_viewpoints(
                    advice_attempts, advice,
                )
                if incompatible:
                    raise HTTPException(
                        422,
                        "源商品拍摄视角与当前桌面不兼容，严格保留 RGB 时无法安全修正底座："
                        + "、".join(incompatible),
                    )
                # Scene analysis (and any user correction made on the
                # canvas) is the geometric source of truth.  A second model
                # call must not replace a perspective tabletop polygon with
                # a convenient rectangle merely to make its proposed layout
                # pass.  MiniMax advises semantics, scale and ordering here;
                # the local solver validates those proposals against the
                # already accepted support/protection geometry.
                effective_support = payload.support_polygon
                effective_protected = payload.protected_polygons
                seeded = apply_layout_advice(
                    source_products,
                    advice,
                    canvas_size=(width, height),
                    support_polygon=effective_support,
                )
                seeded_products, messages = refine_seeded_composition(
                    (width, height),
                    effective_support,
                    effective_protected,
                    seeded,
                    shadow_direction=layout_shadow_direction,
                    shadow_length_ratio=payload.shadow_length_ratio,
                    shadow_softness_ratio=payload.shadow_softness_ratio,
                    background_rgb=image,
                    support_mask=support_mask,
                )
                seed_solver_metrics = {
                    "layout_score": float(np.clip(
                        96.0 - len(messages) * 2.0,
                        0.0, 100.0,
                    )),
                    "usable_surface_ratio": 0.0,
                    "automatic_repairs": float(len(messages)),
                    "advisor_used": True,
                    "advisor_confidence": advice.confidence,
                }
                seed_diagnostics = _layout_surface_diagnostics(
                    (width, height),
                    effective_support,
                    effective_protected,
                    seeded_products,
                    advisor_seed=seeded,
                    repair_count=len(messages),
                    support_mask=support_mask,
                )
                seed_metrics = _merge_layout_metrics(
                    seed_solver_metrics, seed_diagnostics,
                )
                products = seeded_products
                metrics = seed_metrics
                compacted_seed = False
                compaction_metrics: dict[str, Any] = {}
                compaction_messages: list[str] = []
                if float(seed_metrics.get(
                    "maximum_normalized_gap", 0.0,
                )) > SPARSE_GAP_REPLAN_RATIO:
                    (
                        compacted_products,
                        compaction_solver_metrics,
                        compaction_messages,
                    ) = compact_sparse_seeded_layout(
                        (width, height),
                        effective_support,
                        effective_protected,
                        seeded_products,
                        shadow_direction=layout_shadow_direction,
                        shadow_length_ratio=payload.shadow_length_ratio,
                        shadow_softness_ratio=payload.shadow_softness_ratio,
                        support_mask=support_mask,
                    )
                    if compaction_solver_metrics.get(
                        "seed_compaction_applied", 0.0,
                    ):
                        compacted_diagnostics = _layout_surface_diagnostics(
                            (width, height),
                            effective_support,
                            effective_protected,
                            compacted_products,
                            advisor_seed=seeded,
                            repair_count=len(messages) + len(compaction_messages),
                            support_mask=support_mask,
                        )
                        compaction_metrics = _merge_layout_metrics(
                            {
                                "layout_score": float(np.clip(
                                    96.0
                                    - len(messages) * 2.0
                                    - len(compaction_messages),
                                    0.0,
                                    100.0,
                                )),
                                "usable_surface_ratio": 0.0,
                                "automatic_repairs": float(
                                    len(messages) + len(compaction_messages)
                                ),
                                **compaction_solver_metrics,
                            },
                            compacted_diagnostics,
                        )
                        # Clearing the defect that triggered this branch is
                        # authoritative when the candidate has no hard geometry
                        # failure and has preserved all three semantic scales.
                        compact_hard_failures = list(compaction_metrics.get(
                            "layout_hard_failures", [],
                        ))
                        if (
                            not compact_hard_failures
                            and float(compaction_metrics.get(
                                "maximum_normalized_gap", 1.0,
                            )) <= SPARSE_GAP_REPLAN_RATIO
                        ):
                            products = compacted_products
                            metrics = compaction_metrics
                            compacted_seed = True
                replan_required = bool(
                    metrics["layout_verdict"] != "pass"
                    or (
                        not compacted_seed
                        and float(metrics.get("advisor_repair_max", 0.0)) > 0.08
                    )
                    or float(metrics.get("advisor_scale_repair_max", 0.0)) > 0.10
                    or float(metrics.get(
                        "maximum_normalized_gap", 0.0,
                    )) > SPARSE_GAP_REPLAN_RATIO
                )
                replanned = False
                if replan_required:
                    local_products, local_metrics = local_candidate(seeded)
                    if _prefer_replanned_layout(metrics, local_metrics):
                        products = local_products
                        metrics = local_metrics
                        metrics.update({
                            "advisor_used": True,
                            "advisor_confidence": advice.confidence,
                            "advisor_seed_score": seed_metrics["layout_score"],
                            "advisor_seed_verdict": seed_metrics["layout_verdict"],
                            "advisor_seed_maximum_normalized_gap": float(
                                seed_metrics.get("maximum_normalized_gap", 0.0)
                            ),
                            "local_joint_replan": True,
                        })
                        replanned = True
                        compacted_seed = False
                if not replanned:
                    metrics["local_joint_replan"] = False
                metrics["semantic_seed_compacted"] = compacted_seed
                if compaction_metrics:
                    metrics["seed_compaction_diagnostics"] = {
                        key: value for key, value in compaction_metrics.items()
                        if key.startswith("seed_")
                    }
                if compaction_messages:
                    messages = [*messages, *compaction_messages]
                advisor_result = {
                    "status": (
                        "seed_replanned_local" if replanned else
                        "seed_compacted" if compacted_seed else
                        "used" if metrics["layout_verdict"] == "pass" else
                        "used_layout_reject"
                    ),
                    "model": advice.model,
                    "request_id": advice.request_id,
                    "route": advice.route,
                    "confidence": advice.confidence,
                    "warnings": advice.warnings,
                    "local_repairs": messages,
                    "semantic_seed_used": True,
                    "receiver_scale_source": (
                        "background_scene_analysis"
                        if scene_receiver_authority else "multimodal_layout_advisor"
                    ),
                    "receiver_width_or_diameter_cm": (
                        float(getattr(
                            advice, "receiver_width_or_diameter_cm", 100.0,
                        ))
                    ),
                    "receiver_size_confidence": float(getattr(
                        advice, "receiver_size_confidence", 0.0,
                    )),
                    "seed_layout_score": seed_metrics["layout_score"],
                    "seed_layout_verdict": seed_metrics["layout_verdict"],
                    "seed_maximum_normalized_gap": float(
                        seed_metrics.get("maximum_normalized_gap", 0.0)
                    ),
                    "attempts": [
                        _advice_attempt(item, index)
                        for index, item in enumerate(advice_attempts)
                    ],
                    "retry_error": retry_error,
                    "local_semantic_priors": local_semantic_priors,
                    "semantic_reconciliations": semantic_corrections,
                    "semantic_scale": [
                        {
                            "product_id": item.product_id,
                            "estimated_real_height_cm": item.estimated_real_height_cm,
                            "size_confidence": item.size_confidence,
                            "photographed_pose": item.photographed_pose,
                            "pose_confidence": item.pose_confidence,
                            "viewpoint_compatibility": item.viewpoint_compatibility,
                        }
                        for item in list(getattr(advice, "products", []))
                    ],
                }
            except MiniMaxNotConfigured:
                products, metrics = local_candidate()
                advisor_result = {
                    "status": "not_configured",
                    "local_semantic_priors": local_semantic_priors,
                }
                metrics["advisor_used"] = False
            except HTTPException:
                raise
            except Exception as advisor_error:
                effective_support = payload.support_polygon
                effective_protected = payload.protected_polygons
                products, metrics = local_candidate()
                advisor_result = {
                    "status": "fallback_local",
                    "reason": _friendly_error(advisor_error),
                    "local_semantic_priors": local_semantic_priors,
                }
                metrics["advisor_used"] = False
        else:
            products, metrics = local_candidate()
            advisor_result = {
                "status": "not_configured",
                "local_semantic_priors": local_semantic_priors,
            }
            metrics["advisor_used"] = False
        return {
            "products": products,
            "metrics": metrics,
            "layout_verdict": metrics["layout_verdict"],
            "layout_reasons": metrics.get("layout_reasons", []),
            "layout_diagnostics": {
                key: value
                for key, value in metrics.items()
                if key in {
                    "surface_depth_median",
                    "surface_depth_min",
                    "surface_depth_max",
                    "surface_depth_range",
                    "minimum_front_clearance_over_height",
                    "visual_mass_height_median",
                    "visual_mass_scale_ratio",
                    "minimum_normalized_gap",
                    "maximum_normalized_gap",
                    "median_normalized_gap",
                    "aesthetic_gap_target",
                    "sparse_gap_replan_ratio",
                    "normalized_gap_ratio",
                    "group_fill_ratio",
                    "apparent_height_cv",
                    "positive_gap_cv",
                    "semantic_scale_ratios",
                    "perspective_scale_violations",
                    "geometry_score",
                    "composition_score",
                    "semantic_scale_score",
                    "perspective_score",
                    "advisor_repair_max",
                    "advisor_scale_repair_max",
                    "product_surface",
                }
            },
            "advisor": advisor_result,
            "support_polygon": effective_support,
            "protected_polygons": effective_protected,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, f"自动构图失败：{_friendly_error(exc)}") from exc


@app.post("/api/repair-layout")
def repair_layout(payload: LayoutUI) -> dict[str, Any]:
    try:
        background = local_path(payload.background)
        image = load_background(background)
        height, width = image.shape[:2]
        support_mask = _layout_support_mask(
            payload.support_mask_path, (width, height),
        )
        products, messages = legalize_placements(
            (width, height),
            payload.support_polygon,
            payload.protected_polygons,
            [_product_dict(product) for product in payload.products],
            support_mask=support_mask,
        )
        return {"products": products, "messages": messages}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, f"摆位修复失败：{_friendly_error(exc)}") from exc


@app.post("/api/visual-review")
def visual_review(payload: VisualReviewUI) -> dict[str, Any]:
    completed_slot_ids: list[int] = []
    bundle: dict[str, Any] | None = None
    final_path: Path | None = None
    transport_attempts: list[dict[str, Any]] = []
    try:
        background = local_path(payload.background)
        final_path = local_path(payload.final_path)
        product_paths = [local_path(value) for value in payload.products]
        audit_path = final_path.parent / "audit.json"
        # The compositor may create a canonical 4K background inside the run
        # directory.  Review and bind against that exact receiver image, not
        # merely the lower-resolution path that the browser originally chose.
        # This closes a background-substitution gap in the delivery audit.
        review_background = background
        job_path = final_path.parent / "job.json"
        if not job_path.is_file():
            raise FinalGateError("Run is missing job.json; visual review is blocked")
        try:
            job_payload = json.loads(job_path.read_text(encoding="utf-8"))
            if not isinstance(job_payload, dict):
                raise FinalGateError("job.json must contain an object")
            job_background = job_payload.get("background")
            if isinstance(job_background, str) and job_background.strip():
                declared_background = Path(job_background)
                if not declared_background.is_absolute():
                    declared_background = job_path.parent / declared_background
                review_background = local_path(str(declared_background))
        except FinalGateError:
            raise
        except (OSError, ValueError, TypeError, json.JSONDecodeError, HTTPException) as exc:
            # A missing, malformed or unresolved manifest cannot be replaced
            # by browser state: doing so would disconnect the visual score
            # from the material and lighting semantics that were reviewed.
            raise FinalGateError("job.json is unreadable or unresolved") from exc
        audit: dict[str, Any] | None = None
        if audit_path.is_file():
            try:
                loaded_audit = json.loads(audit_path.read_text(encoding="utf-8"))
                if isinstance(loaded_audit, dict):
                    audit = loaded_audit
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                # A missing or damaged audit is a hard-gate rejection below;
                # it must never be silently interpreted as a pixel pass.
                audit = None

        reviewer = minimax_status()
        bundle = build_review_bundle(
            review_background,
            final_path,
            product_paths,
            audit,
            audit_path=audit_path if audit_path.is_file() else None,
            job_path=job_path,
            reviewer_model=str(reviewer.get("model", "unknown")),
            prompt_version=VISUAL_REVIEW_PROMPT_VERSION,
        )
        gate_path = final_path.parent / "final_gate.json"
        with _VISUAL_REVIEW_LOCK:
            if gate_path.is_file():
                try:
                    cached = json.loads(gate_path.read_text(encoding="utf-8"))
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    cached = None
                if (
                    isinstance(cached, dict)
                    and cached.get("schema_version") == FINAL_GATE_SCHEMA_VERSION
                    and cached.get("policy_version") == FINAL_GATE_POLICY_VERSION
                    and cached.get("bundle_sha256") == bundle["bundle_sha256"]
                ):
                    cached_review = cached.get("visual_review")
                    if isinstance(cached_review, dict):
                        reconstructed = build_final_gate(bundle, audit, cached_review)
                        if reconstructed == cached:
                            return {
                                **cached,
                                "review_cache_hit": True,
                                "review_slots_reused": VISUAL_REVIEW_SLOT_COUNT,
                            }
                if isinstance(cached, dict):
                    cached_review = cached.get("visual_review")
                    cached_bundle = cached.get("review_bundle")
                    if (
                        isinstance(cached_review, dict)
                        and _same_visual_evidence(cached_bundle, bundle)
                    ):
                        # Same pixels/model/prompt: retain the original five
                        # judgements and bind them to the newly computed hard
                        # audit rather than sampling the model again.
                        reconstructed = build_final_gate(bundle, audit, cached_review)
                        write_json_atomic(
                            final_path.parent / "visual_review.json",
                            aggregate_visual_reviews(cached_review["review_slots"]),
                        )
                        write_final_gate(gate_path, reconstructed)
                        return {
                            **reconstructed,
                            "review_cache_hit": True,
                            "review_slots_reused": VISUAL_REVIEW_SLOT_COUNT,
                            "review_reuse_reason": "identical_visual_evidence_audit_rebound",
                        }

            slot_reviews: list[dict[str, Any]] = []
            reused_slots = 0
            review_request_retries = 0
            for slot_id in range(1, VISUAL_REVIEW_SLOT_COUNT + 1):
                slot_review = _read_visual_review_slot(
                    final_path, str(bundle["bundle_sha256"]), slot_id,
                )
                if slot_review is not None:
                    reused_slots += 1
                if slot_review is None:
                    last_slot_error: Exception | None = None
                    for request_attempt in range(VISUAL_REVIEW_MAX_TRANSPORT_ATTEMPTS):
                        try:
                            candidate_review = request_minimax_visual_review(
                                review_background,
                                final_path,
                                product_paths,
                                slot_id=slot_id,
                            )
                            candidate_review = dict(candidate_review)
                            candidate_review["slot_id"] = slot_id
                            validate_dimension_scores(candidate_review)
                            if candidate_review.get("prompt_version") != VISUAL_REVIEW_PROMPT_VERSION:
                                raise FinalGateError("Visual review prompt version mismatch")
                            slot_review = candidate_review
                            break
                        except MiniMaxTransportError as slot_error:
                            last_slot_error = slot_error
                            attempt_number = request_attempt + 1
                            retry_delay = 0.0
                            if attempt_number < VISUAL_REVIEW_MAX_TRANSPORT_ATTEMPTS:
                                # There is no valid judgement for this slot.
                                # A short backoff gives a flaky local proxy or
                                # relay time to recover, but keeps the same
                                # provider URL and keeps all completed slots.
                                retry_delay = _visual_review_transport_retry_delay(
                                    slot_id, attempt_number,
                                )
                            transport_attempts.append({
                                "slot_id": int(slot_id),
                                "attempt": int(attempt_number),
                                "error_kind": "transport_no_response",
                                "attempted_routes": list(slot_error.attempted_routes),
                                "retry_delay_seconds": retry_delay,
                            })
                            if retry_delay > 0.0:
                                # Retry only when no valid judgement was
                                # produced. A valid low score is never retried,
                                # so this cannot become score shopping.
                                review_request_retries += 1
                                time.sleep(retry_delay)
                                continue
                            raise last_slot_error
                        except (ContractError, FinalGateError) as slot_error:
                            # A syntactically returned provider payload is not
                            # a judgement unless all six required dimensions
                            # validate.  Retrying an empty/invalid slot is
                            # reliability recovery, not score shopping: no
                            # valid score is discarded or written.  Keep the
                            # exact same images, prompt, model, endpoint and
                            # slot id, with the same small fixed attempt cap.
                            last_slot_error = slot_error
                            attempt_number = request_attempt + 1
                            retry_delay = 0.0
                            if attempt_number < VISUAL_REVIEW_MAX_TRANSPORT_ATTEMPTS:
                                retry_delay = _visual_review_transport_retry_delay(
                                    slot_id, attempt_number,
                                )
                            transport_attempts.append({
                                "slot_id": int(slot_id),
                                "attempt": int(attempt_number),
                                "error_kind": "response_without_valid_judgement",
                                "retry_delay_seconds": retry_delay,
                            })
                            if retry_delay > 0.0:
                                review_request_retries += 1
                                time.sleep(retry_delay)
                                continue
                            raise last_slot_error
                    if slot_review is None:
                        raise last_slot_error or FinalGateError("Visual review slot failed")
                    _write_visual_review_slot(
                        final_path,
                        str(bundle["bundle_sha256"]),
                        slot_id,
                        slot_review,
                    )
                slot_reviews.append(slot_review)
                completed_slot_ids.append(slot_id)

            # This always consumes exactly slots 1..5. Scores and critical
            # failures never influence how many requests are executed.
            review = aggregate_visual_reviews(slot_reviews)
            write_json_atomic(final_path.parent / "visual_review.json", review)
            gate = build_final_gate(bundle, audit, review)
            write_final_gate(gate_path, gate)
            try:
                # A previous interrupted click must not leave stale "pending"
                # status after all five immutable review slots have completed.
                (final_path.parent / "visual_review_pending.json").unlink(missing_ok=True)
            except OSError:
                pass
            return {
                **gate,
                "review_cache_hit": reused_slots == VISUAL_REVIEW_SLOT_COUNT,
                "review_slots_reused": reused_slots,
                "review_request_retries": review_request_retries,
            }
    except MiniMaxTransportError as exc:
        # A provider transport failure is neither a low visual score nor a
        # completed review.  Persist only redacted resumable state so the next
        # click can reuse any finished slots and request the missing evidence
        # slot.  No candidate can be delivered while this record exists.
        if final_path is not None and bundle is not None:
            try:
                write_json_atomic(final_path.parent / "visual_review_pending.json", {
                    "schema_version": "visual-review-pending-v1",
                    "state": "pending",
                    "reason": "minimax_transport_error",
                    "bundle_sha256": bundle.get("bundle_sha256"),
                    "completed_slot_ids": sorted(set(completed_slot_ids)),
                    "pending_slot_id": (
                        max(completed_slot_ids, default=0) + 1
                    ),
                    "network_mode": str(
                        get_public_settings().get("minimax_network_mode", "auto")
                    ),
                    "retry_policy": VISUAL_REVIEW_TRANSPORT_RETRY_POLICY,
                    "max_transport_attempts": VISUAL_REVIEW_MAX_TRANSPORT_ATTEMPTS,
                    # Deliberately redacted: only slot, route labels and delay
                    # are retained.  No prompt, image payload, provider body or
                    # API secret enters the resumable record.
                    "failed_attempts": transport_attempts,
                    "updated_unix": int(time.time()),
                })
            except OSError:
                # The HTTP response remains safely blocked even if diagnostics
                # storage itself is unavailable.
                pass
        raise HTTPException(
            status_code=503,
            detail=(
                "视觉评审暂未完成：MiniMax 网络连接暂时不可用。候选图已保留，"
                "但仍不可交付；稍后再次点击合成会只补未完成的评审槽位。"
            ),
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, f"视觉评审失败：{_friendly_error(exc)}") from exc


@app.post("/api/compose")
def compose(payload: ComposeUI) -> dict[str, Any]:
    try:
        # A pass-one compose creates the immutable canonical candidate.  A
        # pass-two request is deliberately *not* a second layout/material
        # request: it renders from that canonical job and may change only the
        # bounded lighting-layer values below.
        base_run: Path | None = None
        base_job: dict[str, Any] | None = None
        if payload.refinement_pass:
            if payload.auto_repair:
                raise HTTPException(
                    400,
                    "光影修复不能重新摆位；请使用首轮锁定候选。",
                )
            base_run, base_job = refinement_base_job(payload.refinement_base_run)
            background_source = local_path(str(base_job["background"]))
        else:
            background_source = local_path(payload.background)
        source_rgb = load_background(background_source)
        source_height, source_width = source_rgb.shape[:2]
        scale_4k = max(1.0, float(payload.output_long_edge) / max(source_height, source_width))
        if scale_4k > 1.0:
            width = int(round(source_width * scale_4k))
            height = int(round(source_height * scale_4k))
            background_rgb = cv2.resize(
                source_rgb, (width, height), interpolation=cv2.INTER_LANCZOS4,
            )
        else:
            background_rgb = source_rgb
            height, width = source_height, source_width
        warnings: list[str] = []
        if base_job is not None:
            # JSON round-trip is a concise deep copy and guarantees that the
            # output job cannot mutate the persisted base descriptor.
            placements = json.loads(json.dumps(base_job["ui_placements"]))
            products = json.loads(json.dumps(base_job["products"]))
            base_lighting = base_job["lighting"]
            support_polygon = base_job["support_polygon"]
            support_mask_path = base_job.get("support_mask")
            # A refinement is locked to the exact first-pass scene evidence.
            # Never accept replacement geometry paths from the mutable second
            # browser request.
            depth_map_path = base_job.get("depth_map")
            normal_map_path = base_job.get("normal_map")
            protected_polygons = base_job.get("protected_polygons", [])
            shadow_direction = base_lighting["shadow_direction"]
            shadow_length_ratio = float(base_lighting["shadow_length_ratio"])
            shadow_opacity = float(base_lighting["shadow_opacity"])
            shadow_softness_ratio = float(base_lighting["shadow_softness_ratio"])
            reflection_enabled = bool(base_lighting["reflection_enabled"])
            reflection_opacity = float(base_lighting["reflection_opacity"])
            cast_shadow_mode = str(base_lighting["cast_shadow_mode"])
            surface_material = str(base_lighting["surface_material"])
            surface_reflectivity = float(base_lighting["surface_reflectivity"])
            surface_roughness = float(base_lighting["reflection_roughness"])
            direct_light_fraction = float(base_lighting["direct_light_fraction"])
            contact_ao_strength = float(base_lighting["contact_ao_strength"])
            output_was_upscaled = bool(base_job.get("upscaled_to_4k", True))
        else:
            placements = [_product_dict(product) for product in payload.products]
            if payload.auto_repair:
                support_mask = _layout_support_mask(
                    payload.support_mask_path, (width, height),
                )
                placements, warnings = legalize_placements(
                    (width, height),
                    payload.support_polygon,
                    payload.protected_polygons,
                    placements,
                    support_mask=support_mask,
                )
            products: list[dict[str, Any]] = []
            for product in placements:
                rgba_path = local_path(product.get("rgba") or product["source"])
                rgba = load_rgba(rgba_path)
                source_h, source_w = rgba.shape[:2]
                homography = placement_homography(
                    (source_w, source_h), (width, height), product,
                )
                material_route = resolve_material_route(
                    str(product.get("material_mode", "opaque")),
                    str(product.get("semantic_category", "")),
                    str(product.get("physical_type", "")),
                )
                material_mode = material_route.resolved_mode
                material_confidence = max(
                    float(product.get("material_confidence", 0.0)),
                    material_route.confidence_floor,
                )
                transmission_strength = max(
                    float(product.get("transmission_strength", 0.0)),
                    material_route.transmission_floor,
                )
                compatibility = float(np.clip(
                    product.get("source_light_compatibility", 0.5), 0.0, 1.0,
                ))
                # A compatible source photograph keeps its original colour and
                # gloss exactly.  Lower confidence permits only a restrained
                # temperature adaptation; exposure is never inferred from desk
                # brightness because product albedo is unknown.
                adaptation_strength = float(np.clip(
                    (0.68 - compatibility) / 0.68, 0.0, 0.55,
                ))
                gain = (
                    (1.0, 1.0, 1.0)
                    if not is_opaque(material_mode) or adaptation_strength <= 0.0
                    else _color_gain(
                        rgba,
                        background_rgb,
                        product,
                        payload.support_polygon,
                        surface_material=payload.surface_material,
                        adaptation_strength=adaptation_strength,
                    )
                )
                gain_limit = float(material_profile(material_mode).colour_gain_limit)
                gain = tuple(float(np.clip(
                    value, 1.0 - gain_limit, 1.0 + gain_limit,
                )) for value in gain)
                products.append({
                    "id": product["id"],
                    "rgba": str(rgba_path),
                    "homography": homography.tolist(),
                    "z_order": int(product.get("z_order", 0)),
                    "rgb_gain": list(gain),
                    "placement_mode": str(product.get("placement_mode", "upright")),
                    "material_mode": material_mode,
                    "material_confidence": material_confidence,
                    "transmission_strength": transmission_strength,
                    "material_route_reason": material_route.reason,
                    "semantic_category": str(product.get("semantic_category", "")),
                    "physical_type": str(product.get("physical_type", "")),
                    "identity_status": str(product.get("identity_status", "unknown")),
                    "identity_mask": (
                        str(local_path(product["identity_mask"]))
                        if product.get("identity_mask") else None
                    ),
                    "rerender_policy": str(product.get("rerender_policy", "auto")),
                })
            support_polygon = payload.support_polygon
            support_mask_path = payload.support_mask_path
            depth_map_path = payload.depth_map_path
            normal_map_path = payload.normal_map_path
            protected_polygons = payload.protected_polygons
            shadow_direction = payload.shadow_direction
            shadow_length_ratio = float(payload.shadow_length_ratio)
            shadow_opacity = float(payload.shadow_opacity)
            shadow_softness_ratio = float(payload.shadow_softness_ratio)
            reflection_enabled = bool(payload.reflection_enabled)
            reflection_opacity = float(payload.reflection_opacity)
            cast_shadow_mode = str(payload.cast_shadow_mode)
            surface_material = str(payload.surface_material)
            surface_reflectivity = float(payload.surface_reflectivity)
            surface_roughness = float(payload.surface_roughness)
            direct_light_fraction = float(payload.direct_light_fraction)
            contact_ao_strength = float(payload.contact_ao_strength)
            output_was_upscaled = bool(scale_4k > 1.0)
        run = WORK / "runs" / f"run_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000:06d}"
        run.mkdir(parents=True, exist_ok=True)
        if scale_4k > 1.0:
            canonical_background = run / "background_canonical_4k.png"
            save_png(canonical_background, background_rgb)
        else:
            canonical_background = background_source
        effective_softness = shadow_softness_ratio
        effective_opacity = (
            shadow_opacity if cast_shadow_mode == "directional" else 0.0
        )
        effective_reflection = reflection_opacity
        tail_strength = 0.38
        if payload.refinement_pass:
            # Keep the object, its RGB gain, transform, light direction and
            # shadow reach identical to the first candidate.  Only the local
            # contact/penumbra and a bounded surface reflection may change.
            effective_softness = max(0.01, effective_softness * 0.82)
            if cast_shadow_mode == "directional":
                effective_opacity = min(0.75, effective_opacity + 0.025)
            if reflection_enabled:
                # A transparent glass receiver may show a reflection, but an
                # ambient-only scene has no evidence for making every retry a
                # stronger mirror.  Earlier code blindly boosted the same
                # reflection on the second candidate, which produced a
                # duplicate-looking bottle/pitcher below the real base.  Keep
                # the analysis-selected strength unchanged in that situation;
                # only a directional-light receiver may receive a bounded
                # deterministic increment.
                if surface_material == "transparent_glass":
                    if cast_shadow_mode == "directional":
                        effective_reflection = min(
                            0.30,
                            max(effective_reflection, 0.18) * 1.04 + 0.003,
                        )
                    else:
                        # The first candidate's glass-table reflection is
                        # scene-derived.  Do not silently weaken it during a
                        # contact-only retry: a low but visible mirror cue is
                        # precisely what differentiates transparent glass from
                        # a matte tabletop.  This retains the analysed amount
                        # within a conservative physical ceiling rather than
                        # increasing it as a score-seeking second opinion.
                        effective_reflection = min(0.30, effective_reflection)
                else:
                    effective_reflection = min(
                        0.25, effective_reflection * 1.12 + 0.008,
                    )
            tail_strength = 0.58
        job_data = {
            "background": str(canonical_background),
            "support_polygon": support_polygon,
            "support_mask": (
                str(local_path(support_mask_path)) if support_mask_path else None
            ),
            "protected_polygons": protected_polygons,
            "lighting": {
                "shadow_direction": shadow_direction,
                "shadow_length_ratio": shadow_length_ratio,
                "shadow_opacity": effective_opacity,
                "shadow_softness_ratio": effective_softness,
                "shadow_tail_strength": tail_strength,
                "reflection_enabled": reflection_enabled,
                "reflection_opacity": effective_reflection,
                "cast_shadow_mode": cast_shadow_mode,
                "surface_material": surface_material,
                "surface_reflectivity": surface_reflectivity,
                "reflection_roughness": surface_roughness,
                "direct_light_fraction": direct_light_fraction,
                "contact_ao_strength": contact_ao_strength,
            },
            "products": products,
            "hybrid_render": (
                json.loads(json.dumps(base_job.get("hybrid_render", {})))
                if base_job is not None else {
                    "enabled": bool(payload.hybrid_rerender_enabled),
                    "provider": str(payload.hybrid_provider),
                    "candidate": (
                        str(local_path(payload.hybrid_candidate))
                        if payload.hybrid_candidate else None
                    ),
                    "prompt": str(payload.hybrid_prompt),
                }
            ),
            "ui_placements": placements,
            "source_dimensions": [source_width, source_height],
            "upscaled_to_4k": output_was_upscaled,
            "refinement_pass": int(payload.refinement_pass),
            "refinement_base_run": str(base_run) if base_run else None,
            "refinement_lock": {
                "placement": bool(base_job is not None),
                "product_pixels": bool(base_job is not None),
                "material": bool(base_job is not None),
                "color_gain": bool(base_job is not None),
                "allowed": (
                    ["contact_ao", "shadow_opacity", "shadow_softness", "reflection_opacity", "shadow_tail"]
                    if base_job is not None else []
                ),
            },
        }
        # Keep legacy manifests byte-for-structure compatible when scene
        # analysis has no geometry evidence.  When evidence exists, resolve it
        # through the same workspace boundary as every other local input and
        # persist it so an immutable refinement can reuse the exact maps.
        if depth_map_path:
            job_data["depth_map"] = str(local_path(depth_map_path))
        if normal_map_path:
            job_data["normal_map"] = str(local_path(normal_map_path))
        (run / "job.json").write_text(
            json.dumps(job_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        hybrid_provider = (
            RunningHubHybridProvider()
            if bool(job_data["hybrid_render"].get("enabled"))
            and str(job_data["hybrid_render"].get("provider")) == "runninghub"
            and not job_data["hybrid_render"].get("candidate")
            else None
        )
        learned_provider = (
            RunningHubLearnedEffectProvider(
                artifact_root=run / "learned_effects",
            )
            if bool(payload.learned_effects_enabled)
            else None
        )
        result = StrictCompositor(
            hybrid_provider=hybrid_provider,
            learned_effect_provider=learned_provider,
        ).compose(
            Job.from_dict(job_data, ROOT),
            run,
            strict_audit=bool(payload.strict_audit),
        )
        audit = json.loads((run / "audit.json").read_text(encoding="utf-8"))
        hybrid_artifacts = {
            name: browser_path(path)
            for name, path in dict(
                audit.get("hybrid_render", {}).get("artifacts", {})
            ).items()
            if isinstance(path, str) and Path(path).exists()
        }
        learned_metadata = dict(
            audit.get("learned_effects", {}).get("batch_metadata", {})
        )
        learned_artifacts: dict[str, str] = {}
        input_manifest = learned_metadata.get("input_manifest")
        if isinstance(input_manifest, str) and Path(input_manifest).exists():
            learned_artifacts["input_manifest"] = browser_path(input_manifest)
        for name, path in dict(
            learned_metadata.get("output_artifacts", {})
        ).items():
            if isinstance(path, str) and Path(path).exists():
                learned_artifacts[str(name)] = browser_path(path)
        return {
            "run": str(run),
            "final": browser_path(str(run / "final.png")),
            "final_path": str(run / "final.png"),
            "mask": browser_path(str(run / "authorized_mask.png")),
            "support_mask": browser_path(str(run / "support_mask.png")),
            "audit": audit,
            "hybrid_artifacts": hybrid_artifacts,
            "learned_artifacts": learned_artifacts,
            "pixel_verdict": str(
                audit.get("pixel_audit", {}).get(
                    "verdict", result.audit.verdict,
                )
            ),
            "physical_verdict": str(
                audit.get("physical_audit", {}).get("verdict", "reject")
            ),
            "delivery_verdict": str(
                audit.get(
                    "verdict",
                    "reject" if payload.strict_audit
                    else "debug_preview_not_audited",
                )
            ),
            "strict_audit": bool(payload.strict_audit),
            "products": placements,
            "warnings": warnings,
            "source_dimensions": [source_width, source_height],
            "output_dimensions": [width, height],
            "upscaled_to_4k": output_was_upscaled,
            "refinement_pass": int(payload.refinement_pass),
            "effective_lighting": job_data["lighting"],
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, f"合成未完成：{_friendly_error(exc)}") from exc


def main() -> None:
    # A hidden Windows launcher can leave stderr connected to an invalid
    # handle.  Third-party progress/logging code must never be able to abort
    # the web service for that reason.
    try:
        sys.stderr.flush()
    except (OSError, ValueError):
        sys.stderr = open(os.devnull, "w", encoding="utf-8")
    url = "http://127.0.0.1:8765"
    webbrowser.open(url)
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="warning")


if __name__ == "__main__":
    main()
