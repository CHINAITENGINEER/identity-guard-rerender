from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .material_profiles import canonical_material_mode
from .quality import sha256_file


ROUTE_AUDIT_SCHEMA = "render-route-integrity-v1"
CUTOUT_REUSE_SCHEMA = "cutout-reuse-provenance-v1"


def audit_render_route(
    products: Iterable[dict[str, Any]],
    run_audit: dict[str, Any] | None,
) -> dict[str, Any]:
    """Reject a benchmark image when an identity-safe optical route was skipped.

    The deterministic compositor remains a useful preview fallback, but it is
    not visually equivalent to the authorized hybrid route for unlabelled clear
    glass.  Benchmark tooling must therefore expose the route difference rather
    than silently archiving a fallback image beside hybrid reference results.
    """

    normalized: list[dict[str, str]] = []
    required: list[str] = []
    for index, product in enumerate(products, start=1):
        product_id = str(product.get("id") or product.get("product_id") or f"product_{index}")
        material = canonical_material_mode(str(
            product.get("material_mode") or product.get("materialMode") or "opaque"
        ))
        policy = str(
            product.get("rerender_policy") or product.get("rerenderPolicy") or "auto"
        ).strip().lower()
        semantic = str(
            product.get("semantic_category") or product.get("semanticCategory") or ""
        )
        normalized.append({
            "product_id": product_id,
            "material_mode": material,
            "rerender_policy": policy,
            "semantic_category": semantic,
        })
        # Label-free clear glass has no immutable interior RGB.  The local
        # transmission approximation is allowed for interactive preview only;
        # a quality benchmark requires the protected hybrid optical pass.
        if material == "clear_glass" and policy != "never":
            required.append(product_id)
        elif policy == "force":
            required.append(product_id)

    hybrid = dict((run_audit or {}).get("hybrid_render") or {})
    status = str(hybrid.get("status") or "missing")
    effective_pixels = int(hybrid.get("effective_pixels") or 0)
    identity_changed = int(hybrid.get("identity_changed_pixels") or 0)
    outside_changed = int(hybrid.get("outside_changed_pixels") or 0)
    max_outside_delta = int(hybrid.get("max_outside_delta") or 0)

    reasons: list[str] = []
    if required and status != "applied":
        reasons.append("required_identity_safe_hybrid_route_not_applied")
    if required and effective_pixels <= 0:
        reasons.append("hybrid_route_changed_no_authorized_pixels")
    if identity_changed:
        reasons.append("hybrid_route_modified_identity_pixels")
    if outside_changed or max_outside_delta:
        reasons.append("hybrid_route_modified_unauthorized_background")

    verdict = "pass" if not reasons else "reject"
    return {
        "schema_version": ROUTE_AUDIT_SCHEMA,
        "verdict": verdict,
        "expected_route": "identity_safe_hybrid" if required else "strict_or_hybrid",
        "actual_hybrid_status": status,
        "hybrid_required_products": required,
        "effective_pixels": effective_pixels,
        "identity_changed_pixels": identity_changed,
        "outside_changed_pixels": outside_changed,
        "max_outside_delta": max_outside_delta,
        "reasons": reasons,
        "products": normalized,
    }


def reusable_cutout_provenance(
    source_paths: Iterable[str | Path],
    rgba_paths: Iterable[str | Path],
    manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    """Validate that cached mattes are bound to sources and human target review.

    Alpha topology scores cannot decide whether a matte selected the intended
    product.  Reuse is consequently opt-in: every cutout must match both hashes
    and carry an explicit human ``target_verdict=pass``.
    """

    sources = [Path(item) for item in source_paths]
    cutouts = [Path(item) for item in rgba_paths]
    reasons: list[str] = []
    entries = list((manifest or {}).get("products") or [])
    if (manifest or {}).get("schema_version") != CUTOUT_REUSE_SCHEMA:
        reasons.append("missing_or_unsupported_cutout_reuse_manifest")
    if len(sources) != len(cutouts) or len(entries) != len(sources):
        reasons.append("cutout_reuse_manifest_product_count_mismatch")

    checked: list[dict[str, Any]] = []
    for index, (source, rgba) in enumerate(zip(sources, cutouts), start=1):
        entry = entries[index - 1] if index - 1 < len(entries) else {}
        item_reasons: list[str] = []
        if not source.is_file():
            item_reasons.append("source_missing")
        if not rgba.is_file():
            item_reasons.append("rgba_missing")
        if source.is_file() and str(entry.get("source_sha256") or "") != sha256_file(source):
            item_reasons.append("source_hash_mismatch")
        if rgba.is_file() and str(entry.get("rgba_sha256") or "") != sha256_file(rgba):
            item_reasons.append("rgba_hash_mismatch")
        if str(entry.get("target_verdict") or "") != "pass":
            item_reasons.append("human_target_review_not_passed")
        if not str(entry.get("semantic_target") or "").strip():
            item_reasons.append("semantic_target_missing")
        reasons.extend(f"product_{index}:{reason}" for reason in item_reasons)
        checked.append({
            "product_id": f"product_{index}",
            "source": str(source),
            "rgba": str(rgba),
            "semantic_target": str(entry.get("semantic_target") or ""),
            "target_verdict": str(entry.get("target_verdict") or "pending"),
            "reasons": item_reasons,
        })

    return {
        "schema_version": CUTOUT_REUSE_SCHEMA,
        "verdict": "pass" if not reasons else "reject",
        "reasons": reasons,
        "products": checked,
    }


def pending_cutout_reuse_manifest(
    source_paths: Iterable[str | Path],
    rgba_paths: Iterable[str | Path],
) -> dict[str, Any]:
    sources = [Path(item) for item in source_paths]
    cutouts = [Path(item) for item in rgba_paths]
    products: list[dict[str, Any]] = []
    for index, (source, rgba) in enumerate(zip(sources, cutouts), start=1):
        products.append({
            "product_id": f"product_{index}",
            "source": str(source),
            "rgba": str(rgba),
            "source_sha256": sha256_file(source) if source.is_file() else "",
            "rgba_sha256": sha256_file(rgba) if rgba.is_file() else "",
            "semantic_target": "",
            "target_verdict": "pending_human_review",
        })
    return {
        "schema_version": CUTOUT_REUSE_SCHEMA,
        "status": "pending_human_review",
        "products": products,
    }
