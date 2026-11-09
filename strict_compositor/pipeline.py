from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .contracts import Job
from .hybrid import (
    HybridCandidateProvider,
    HybridRenderRequest,
    HybridRenderError,
    load_registered_candidate,
    recover_hybrid_candidate,
    write_hybrid_inputs,
)
from .identity import (
    build_identity_masks,
    conservative_source_identity_mask,
    warp_identity_mask,
)
from .material_profiles import canonical_material_mode
from .image_io import load_background, load_mask, load_rgba, save_png
from .layout import validate_placement_masks
from .learned_effects import (
    EffectContext,
    LearnedEffectProvider,
    LearnedEffectRequest,
    apply_learned_effect_batch,
    isolate_effect_request,
    load_scene_evidence,
)
from .optical_matte import OpticalMatte
from .physical_audit import audit_physical_composition
from .render import alpha_over, darken, polygon_mask, render_product
from .validate import PixelAudit, audit_background


@dataclass(frozen=True)
class CompositionResult:
    final: np.ndarray
    authorized_mask: np.ndarray
    audit: PixelAudit
    layer_names: tuple[str, ...]


class StrictCompositor:
    def __init__(
        self,
        hybrid_provider: HybridCandidateProvider | None = None,
        learned_effect_provider: LearnedEffectProvider | None = None,
        learned_effect_depth_path: str | Path | None = None,
        learned_effect_normal_path: str | Path | None = None,
    ) -> None:
        self.hybrid_provider = hybrid_provider
        self.learned_effect_provider = learned_effect_provider
        self.learned_effect_depth_path = learned_effect_depth_path
        self.learned_effect_normal_path = learned_effect_normal_path

    def compose(
        self,
        job: Job,
        out_dir: str | Path,
        *,
        strict_audit: bool = True,
    ) -> CompositionResult:
        """Render a scene in delivery or fast-preview mode.

        ``strict_audit=False`` is intentionally a development-only route.  It
        preserves the renderer, full output size, product pixels, layer PNGs
        and optical-matte evidence.  It skips only the two terminal gate
        calculations.  The result is explicitly marked ``skipped_debug`` and
        can never be mistaken for a deliverable.
        """
        out_dir = Path(out_dir)
        background = load_background(job.background)
        height, width = background.shape[:2]
        support = (
            load_mask(job.support_mask, (width, height), allow_resize=True)
            if job.support_mask
            else polygon_mask((height, width), job.support_polygon)
        )
        protected = load_mask(job.protected_mask, (width, height)) if job.protected_mask else np.zeros((height, width), bool)
        for polygon in job.protected_polygons:
            protected |= polygon_mask((height, width), polygon)
        # The original background is deliberately never mutated.
        result = background.copy()
        authorized = np.zeros((height, width), bool)
        rendered: list[tuple[Any, Any]] = []
        identity_bundles: dict[str, Any] = {}
        placed_identity_masks: dict[str, np.ndarray | None] = {}
        for product in sorted(job.products, key=lambda item: item.z_order):
            source_rgba = load_rgba(product.rgba)
            source_identity = None
            identity_source = "none"
            if product.identity_mask is not None:
                # Identity evidence is authored in the exact source-RGBA
                # coordinate system.  Load it before rendering so transparent
                # optical reconstruction itself cannot wash out labels/text.
                source_identity = load_mask(
                    product.identity_mask,
                    (source_rgba.shape[1], source_rgba.shape[0]),
                )
                identity_source = "explicit_mask"
            else:
                source_identity, identity_source = conservative_source_identity_mask(
                    source_rgba,
                    material_mode=product.material_mode,
                    semantic_category=product.semantic_category,
                    physical_type=product.physical_type,
                )
            layer = render_product(
                source_rgba, product.homography, (height, width),
                background, support, protected, job.lighting, product.rgb_gain,
                job.support_polygon, product.placement_mode,
                product.material_mode, product.material_confidence,
                product.transmission_strength,
                source_identity_mask=source_identity,
            )
            placed_identity = None
            if source_identity is not None:
                placed_identity = warp_identity_mask(
                    source_identity,
                    product.homography,
                    (height, width),
                    layer.alpha,
                )
            placed_identity_masks[product.product_id] = placed_identity
            identity_bundles[product.product_id] = build_identity_masks(
                layer.alpha,
                material_mode=product.material_mode,
                semantic_category=product.semantic_category,
                physical_type=product.physical_type,
                identity_status=product.identity_status,
                rerender_policy=product.rerender_policy,
                placed_identity_mask=placed_identity,
            )
            identity_bundles[product.product_id].provenance[
                "source_identity_guard"
            ] = identity_source
            rendered.append((product, layer))
        validate_placement_masks(
            [layer.alpha for _, layer in rendered],
            support,
            protected,
            placement_modes=[
                product.placement_mode for product, _ in rendered
            ],
        )
        # Optional learned effects are a separate, map-only stage.  The
        # provider sees an isolated copy of the complete scene and may return
        # only cast-shadow alpha, low-frequency illumination gain and a small
        # specular residual.  It never receives authority over product alpha,
        # final RGB or the background.  All product predictions are validated
        # and applied atomically; any failure retains the untouched strict
        # tuple below.
        learned_effect_report: dict[str, Any] = {
            "status": "not_requested",
            "provider": (
                getattr(self.learned_effect_provider, "name", None)
                if self.learned_effect_provider is not None
                else None
            ),
            "schema_version": "learned-effects-v1",
        }
        learned_effect_applications: dict[str, Any] = {}
        if self.learned_effect_provider is not None:
            # Constructor paths are an explicit development override.  Normal
            # CLI/UI runs bind the geometry evidence to the immutable Job so
            # analysis, refinement and learned inference all consume the same
            # files instead of silently generating evidence that is never
            # used by the compositor.
            learned_depth_path = (
                self.learned_effect_depth_path
                if self.learned_effect_depth_path is not None
                else job.depth_map
            )
            learned_normal_path = (
                self.learned_effect_normal_path
                if self.learned_effect_normal_path is not None
                else job.normal_map
            )
            evidence_report: Mapping[str, Any] = {
                "schema_version": "learned-scene-evidence-v1",
                "status": "pending_validation",
                "depth": {
                    "configured": bool(learned_depth_path),
                },
                "normal": {
                    "configured": bool(learned_normal_path),
                },
            }
            effect_contexts: list[EffectContext] = []
            for product, layer in rendered:
                placed_identity = placed_identity_masks[product.product_id]
                identity_guard = identity_bundles[
                    product.product_id
                ].structure_guard_mask.copy()
                if placed_identity is not None:
                    identity_guard |= np.asarray(placed_identity) > 0
                effect_contexts.append(EffectContext(
                    product_id=product.product_id,
                    background_rgb=background,
                    support_mask=support,
                    protected_mask=protected,
                    product_rgb=layer.rgb,
                    product_alpha=layer.alpha,
                    identity_guard_mask=identity_guard,
                    deterministic_shadow_alpha=layer.cast_shadow,
                    material_mode=product.material_mode,
                    lighting=job.lighting,
                    metadata={
                        "homography": product.homography.tolist(),
                        "placement_mode": product.placement_mode,
                        "semantic_category": product.semantic_category,
                        "physical_type": product.physical_type,
                        "identity_route": identity_bundles[
                            product.product_id
                        ].route,
                    },
                ))
            try:
                # Geometry is loaded only when a learned provider is active.
                # A malformed optional map invalidates the learned batch, not
                # the deterministic composition surrounding this block.
                scene_evidence = load_scene_evidence(
                    learned_depth_path,
                    learned_normal_path,
                    (height, width),
                )
                evidence_report = {
                    **dict(scene_evidence.report),
                    "status": "validated",
                }
                effect_contexts = [
                    replace(
                        context,
                        depth=scene_evidence.depth,
                        normal=scene_evidence.normal,
                    )
                    for context in effect_contexts
                ]
                effect_request = LearnedEffectRequest(
                    background_rgb=background,
                    support_mask=support,
                    protected_mask=protected,
                    lighting=job.lighting,
                    products=tuple(effect_contexts),
                    depth=scene_evidence.depth,
                    normal=scene_evidence.normal,
                    metadata={
                        "canvas_size": [width, height],
                        "product_count": len(effect_contexts),
                    },
                )
                isolated_request = isolate_effect_request(effect_request)
                batch_prediction = self.learned_effect_provider.predict(
                    isolated_request,
                )
                applications = apply_learned_effect_batch(
                    batch_prediction,
                    effect_request,
                )
                # Two-phase commit: keep the strict render tuple authoritative
                # until every candidate layer, application index and bounded
                # audit value has been constructed successfully.  Provider
                # metadata is untrusted and may raise while being normalised;
                # committing pixels before that point would leave learned RGB
                # behind while the audit incorrectly claimed strict_fallback.
                candidate_rendered = [
                    (
                        product,
                        replace(
                            layer,
                            rgb=application.rgb,
                            cast_shadow=application.shadow_alpha,
                        ),
                    )
                    for (product, layer), application in zip(
                        rendered, applications, strict=True,
                    )
                ]
                candidate_applications = {
                    product.product_id: application
                    for (product, _), application in zip(
                        candidate_rendered, applications, strict=True,
                    )
                }
                candidate_report = {
                    **learned_effect_report,
                    "status": "applied",
                    "scene_evidence": _json_safe_mapping(evidence_report),
                    "batch_metadata": _json_safe_mapping(
                        getattr(batch_prediction, "metadata", {}),
                    ),
                    "products": {
                        product.product_id: _json_safe_mapping(
                            application.report,
                        )
                        for (product, _), application in zip(
                            candidate_rendered, applications, strict=True,
                        )
                    },
                }
                # No operation on the right-hand side can fail here.  Publish
                # the three related pieces of state together only after the
                # complete candidate transaction has passed validation and
                # report normalisation.
                (
                    rendered,
                    learned_effect_applications,
                    learned_effect_report,
                ) = (
                    candidate_rendered,
                    candidate_applications,
                    candidate_report,
                )
            except Exception as exc:
                # Learned effects are an optional enhancement.  Input
                # isolation plus atomic replacement guarantees that even a
                # mutating or malformed provider cannot damage strict layers.
                learned_effect_applications = {}
                learned_effect_report = {
                    **learned_effect_report,
                    "status": "strict_fallback",
                    "reason": str(exc)[:500],
                    "scene_evidence": _json_safe_mapping({
                        **dict(evidence_report),
                        "status": "invalid",
                        "reason": str(exc)[:500],
                    }),
                }
        # Receiver effects are staged independently of product z-order.  Long
        # cast light loss belongs on the bare receiver, planar reflections are
        # then added, and zero-gap contact occlusion modulates both.  The old
        # per-product loop painted reflection over its own AO seam and made the
        # result (and audit) depend on product ordering.
        layers: dict[str, np.ndarray] = {}
        cast_union = np.zeros((height, width), np.uint8)
        contact_union = np.zeros((height, width), np.uint8)
        for product, layer in rendered:
            cast_union = np.maximum(cast_union, layer.cast_shadow)
            contact_union = np.maximum(contact_union, layer.contact_shadow)
            authorized |= layer.cast_shadow > 0
            authorized |= layer.contact_shadow > 0
            layers[f"{product.product_id}_contact_ao"] = layer.contact_shadow
            layers[f"{product.product_id}_cast_shadow"] = layer.cast_shadow
            layers[f"{product.product_id}_contact_bounce"] = layer.bounce_alpha
            application = learned_effect_applications.get(product.product_id)
            if application is not None:
                prediction = application.prediction
                if prediction.shadow_alpha is not None:
                    layers[
                        f"{product.product_id}_learned_shadow_alpha"
                    ] = prediction.shadow_alpha
                if prediction.illumination_gain is not None:
                    gain_preview = np.clip(
                        np.round(
                            (prediction.illumination_gain - 0.82)
                            / (1.18 - 0.82)
                            * 255.0
                        ),
                        0,
                        255,
                    ).astype(np.uint8)
                    layers[
                        f"{product.product_id}_learned_illumination_gain"
                    ] = gain_preview
                if prediction.specular_residual is not None:
                    residual_preview = np.clip(
                        np.round(
                            np.max(
                                np.abs(prediction.specular_residual), axis=2,
                            )
                            / 0.08
                            * 255.0
                        ),
                        0,
                        255,
                    ).astype(np.uint8)
                    layers[
                        f"{product.product_id}_learned_specular_residual"
                    ] = residual_preview

        result = darken(result, cast_union)
        for product, layer in rendered:
            if np.any(layer.reflection_alpha):
                result = alpha_over(result, layer.reflection_rgb, layer.reflection_alpha)
                authorized |= layer.reflection_alpha > 0
                layers[f"{product.product_id}_reflection"] = layer.reflection_alpha
        result = darken(result, contact_union)

        for product, layer in rendered:
            result = alpha_over(result, layer.rgb, layer.alpha)
            authorized |= layer.alpha > 0
            layers[f"{product.product_id}_alpha"] = layer.alpha

        # Freeze the deterministic composition before any model candidate is
        # considered.  Identity recovery always copies from this exact image,
        # never from unwarped source pixels or a model-produced approximation.
        result[~authorized] = background[~authorized]
        strict_base = result.copy()
        rerender_allow = np.zeros((height, width), bool)
        identity_guard = np.zeros((height, width), bool)
        optical_polish = np.zeros((height, width), bool)
        for product, _layer in rendered:
            bundle = identity_bundles[product.product_id]
            rerender_allow |= bundle.rerender_allow_mask
            identity_guard |= bundle.hard_restore_mask
            identity_guard |= bundle.structure_guard_mask
            if (
                canonical_material_mode(product.material_mode) == "clear_glass"
                and bundle.route == "regional_rerender"
            ):
                optical_polish |= bundle.rerender_allow_mask

        hybrid_report: dict[str, Any] = {
            "status": "not_requested",
            "provider": job.hybrid_render.provider,
            "eligible_pixels": int(np.count_nonzero(rerender_allow)),
            "identity_guard_pixels": int(np.count_nonzero(identity_guard)),
            "products": {
                product_id: bundle.provenance
                for product_id, bundle in identity_bundles.items()
            },
        }
        if job.hybrid_render.enabled:
            out_dir.mkdir(parents=True, exist_ok=True)
            strict_path, allow_path, guard_path = write_hybrid_inputs(
                out_dir, strict_base, rerender_allow, identity_guard,
            )
            hybrid_report["artifacts"] = {
                "strict_base": str(strict_path),
                "rerender_mask": str(allow_path),
                "identity_guard": str(guard_path),
            }
            try:
                candidate_path: Path
                provider_name = job.hybrid_render.provider
                provider_task_id: str | None = None
                provider_provenance: dict[str, Any] = {}
                if job.hybrid_render.candidate is not None:
                    candidate_path = job.hybrid_render.candidate
                    provider_name = "manual_candidate"
                elif self.hybrid_provider is not None:
                    provider_result = self.hybrid_provider.render(HybridRenderRequest(
                        strict_base=strict_path,
                        edit_mask=allow_path,
                        protect_mask=guard_path,
                        prompt=job.hybrid_render.prompt,
                        expected_size=(width, height),
                        output_dir=out_dir,
                    ))
                    candidate_path = provider_result.candidate_path
                    provider_name = provider_result.provider
                    provider_task_id = provider_result.task_id
                    provider_provenance = provider_result.provenance or {}
                else:
                    raise HybridRenderError("hybrid candidate provider is not configured")
                candidate, registration = load_registered_candidate(
                    candidate_path, (height, width),
                )
                registered_candidate = out_dir / "hybrid_candidate_registered.png"
                save_png(registered_candidate, candidate)
                result, recovery = recover_hybrid_candidate(
                    strict_base,
                    candidate,
                    rerender_allow,
                    identity_guard,
                    optical_polish_mask=optical_polish,
                )
                hybrid_report = {
                    **hybrid_report,
                    **recovery,
                    "provider": provider_name,
                    "task_id": provider_task_id,
                    "registration": registration,
                    "provider_provenance": provider_provenance,
                    "artifacts": {
                        **hybrid_report.get("artifacts", {}),
                        "candidate_registered": str(registered_candidate),
                    },
                }
            except Exception as exc:
                # Candidate failure must never take the strict output down.
                result = strict_base.copy()
                hybrid_report.update({
                    "status": "strict_fallback",
                    "reason": str(exc)[:500],
                    "outside_changed_pixels": 0,
                    "identity_changed_pixels": 0,
                    "max_outside_delta": 0,
                })

        # Hard restore prevents accidental global colour management/re-encoding
        # changes from ever reaching the final bitmap.
        result[~authorized] = background[~authorized]
        if strict_audit:
            audit = audit_background(background, result, authorized)
            if audit.verdict != "pass":
                raise RuntimeError(f"Background integrity gate failed: {audit}")
            # Final-pixel visibility cannot be audited before all shadows,
            # reflections, products and hard background restoration are complete.
            physical_audit = audit_physical_composition(
                rendered,
                support,
                protected,
                job.lighting,
                background,
                result,
            )
        else:
            audit = PixelAudit(0, 0, "skipped_debug")
            physical_audit = {
                "verdict": "skipped_debug",
                "reason": "development fast mode",
            }

        out_dir.mkdir(parents=True, exist_ok=True)
        save_png(out_dir / "final.png", result)
        save_png(out_dir / "authorized_mask.png", (authorized.astype(np.uint8) * 255))
        save_png(out_dir / "support_mask.png", (support.astype(np.uint8) * 255))
        for name, layer in layers.items():
            save_png(out_dir / "layers" / f"{name}.png", layer)
        product_reports: list[dict[str, Any]] = []
        for product_index, (product, layer) in enumerate(rendered):
            product_report: dict[str, Any] = {
                "id": product.product_id,
                "rgba_sha256": _sha256(product.rgba),
                "homography": product.homography.tolist(),
                "z_order": product.z_order,
                "rgb_gain": product.rgb_gain,
                "placement_mode": product.placement_mode,
                "material_mode": product.material_mode,
                "material_confidence": product.material_confidence,
                "transmission_strength": product.transmission_strength,
                "source_environment_replacement_ratio": (
                    layer.source_environment_replacement_ratio
                ),
                "identity": identity_bundles[product.product_id].provenance,
            }
            optical_record = _write_optical_matte_artifact(
                out_dir,
                product.product_id,
                product_index,
                layer.optical_matte,
            )
            if optical_record is not None:
                product_report["optical_matte"] = optical_record
            product_reports.append(product_report)
        physical_verdict = str(physical_audit.get("verdict", "reject"))
        strict_terminal_pass = bool(
            strict_audit
            and audit.verdict == "pass"
            and physical_verdict == "pass"
        )
        report = {
            # A clean background audit is necessary but cannot overrule a
            # failed grounding/shadow/geometry audit.  Publish one fail-closed
            # terminal verdict so every caller sees the same gate result.
            "verdict": (
                "pending_visual_review"
                if strict_terminal_pass
                else ("reject" if strict_audit else "debug_preview")
            ),
            "batch_unlock": False,
            "audit_mode": "strict" if strict_audit else "debug_fast",
            "diagnostics_exported": True,
            "pixel_audit": audit.to_dict(),
            "physical_audit": physical_audit,
            "background_sha256": _sha256(job.background),
            "output_dimensions": [width, height],
            "output_format": "png",
            "products": product_reports,
            "layers": sorted(layers),
            "hybrid_render": hybrid_report,
            "learned_effects": learned_effect_report,
        }
        (out_dir / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return CompositionResult(result, authorized, audit, tuple(sorted(layers)))


def _json_safe_mapping(value: Any) -> Any:
    """Keep experimental provider provenance serialisable and bounded."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe_mapping(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe_mapping(item) for item in value]
    return repr(value)[:500]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


_OPTICAL_ARRAY_NAMES = (
    "coverage_alpha",
    "opaque_identity_mask",
    "transmission_rgb",
    "refraction_flow",
    "specular_rgb",
    "shadow_transmittance",
)


def _safe_artifact_stem(product_id: str, product_index: int) -> str:
    """Return a stable filename without trusting a user-provided product id."""
    slug = re.sub(r"[^0-9A-Za-z_.-]+", "_", product_id).strip("._")
    if not slug:
        slug = "product"
    return f"{product_index + 1:03d}_{slug[:64]}"


def _array_sha256(name: str, value: np.ndarray) -> str:
    """Hash array identity, not merely its byte payload."""
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_optical_matte_artifact(
    out_dir: Path,
    product_id: str,
    product_index: int,
    optical_matte: OpticalMatte | None,
) -> dict[str, Any] | None:
    """Persist an exact ROI optical contract and return its audit binding.

    Only verified clear-glass reconstruction supplies an optical matte.  The
    archive contains numeric arrays only, so it is safe to load with
    ``allow_pickle=False`` and cannot execute provenance payloads.
    """
    if optical_matte is None:
        return None

    arrays = {
        name: np.ascontiguousarray(getattr(optical_matte, name))
        for name in _OPTICAL_ARRAY_NAMES
    }
    if optical_matte.opaque_identity_rgb is not None:
        arrays["opaque_identity_rgb"] = np.ascontiguousarray(
            optical_matte.opaque_identity_rgb,
        )
    arrays["bbox"] = np.asarray(optical_matte.bbox, dtype=np.int32)
    arrays["confidence"] = np.asarray(
        [optical_matte.confidence], dtype=np.float32,
    )

    artifact_dir = out_dir / "optical_mattes"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_artifact_stem(product_id, product_index)
    npz_path = artifact_dir / f"{stem}.npz"
    np.savez_compressed(npz_path, **arrays)

    array_records = {
        name: {
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "sha256": _array_sha256(name, array),
        }
        for name, array in arrays.items()
    }
    content_digest = hashlib.sha256()
    for name in sorted(array_records):
        content_digest.update(name.encode("utf-8"))
        content_digest.update(b"\0")
        content_digest.update(array_records[name]["sha256"].encode("ascii"))
        content_digest.update(b"\0")

    record: dict[str, Any] = {
        "schema_version": "optical-matte-audit-v1",
        "format": "numpy-npz-numeric-only",
        "path": npz_path.relative_to(out_dir).as_posix(),
        "sha256": _sha256(npz_path),
        "content_sha256": content_digest.hexdigest(),
        "bbox": [int(value) for value in optical_matte.bbox],
        "roi_shape": list(optical_matte.roi_shape),
        "confidence": float(optical_matte.confidence),
        "provenance": _json_safe(optical_matte.provenance_dict()),
        "arrays": array_records,
    }
    metadata_path = artifact_dir / f"{stem}.json"
    metadata_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    record["metadata_path"] = metadata_path.relative_to(out_dir).as_posix()
    record["metadata_sha256"] = _sha256(metadata_path)
    return record
