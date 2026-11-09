from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable

from PIL import Image


FINAL_GATE_SCHEMA_VERSION = "final-gate-v4"
FINAL_GATE_POLICY_VERSION = "conservative-evidence-consensus-v5"
MIN_DELIVERY_LONG_EDGE = 3840
VISUAL_REVIEW_SLOT_COUNT = 5
MAX_VISUAL_SCORE_IQR = 6.0
VISUAL_SCORE_WEIGHTS: dict[str, float] = {
    "grounding": 0.25,
    "cast_shadows": 0.20,
    "geometry_layout": 0.20,
    "lighting_color": 0.15,
    "cutout_edges": 0.10,
    "reflection_material": 0.10,
}
# A high total must not conceal the exact defects this product is intended to
# eliminate.  In particular, layout/colour cannot compensate for a floating
# object or a damaged product edge.
VISUAL_DIMENSION_FLOORS: dict[str, float] = {
    "grounding": 88.0,
    "cast_shadows": 85.0,
    "geometry_layout": 88.0,
    "cutout_edges": 90.0,
}
PHYSICAL_DIMENSIONS = (
    "grounding", "cast_shadows", "geometry_layout", "cutout_edges",
)
MIN_MODEL_SUBJECTIVE_SCORE = 75.0
MAX_HYBRID_SCORE_IQR = 6.0


class FinalGateError(ValueError):
    """Raised when an artifact is not an authorized delivery."""


@dataclass(frozen=True)
class QualityGate:
    visual_score: float
    pixel_verdict: str
    verdict: str
    batch_unlock: bool

    def to_dict(self) -> dict:
        return asdict(self)


def gate_visual_score(score: float, pixel_verdict: str) -> QualityGate:
    """A score below 90 can never be surfaced as a delivery success."""
    if not 0 <= score <= 100:
        raise ValueError("visual_score must be between 0 and 100")
    passed = pixel_verdict == "pass" and score >= 90
    return QualityGate(score, pixel_verdict, "pass" if passed else "reject", passed)


def validate_dimension_scores(review: dict[str, Any]) -> dict[str, float]:
    """Return all six raw 0-100 dimension scores or reject the slot."""

    source = review.get("subscores")
    if not isinstance(source, dict):
        raise FinalGateError("Visual review must contain raw dimension subscores")
    clean: dict[str, float] = {}
    for name in VISUAL_SCORE_WEIGHTS:
        try:
            value = float(source[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise FinalGateError(f"Missing or invalid raw dimension score: {name}") from exc
        if not math.isfinite(value) or not 0 <= value <= 100:
            raise FinalGateError(f"Raw dimension score {name} must be in [0,100]")
        clean[name] = value
    return clean


def weighted_visual_score(subscores: dict[str, float]) -> float:
    """Calculate the only authoritative visual total from raw dimensions."""

    return round(
        float(sum(subscores[name] * weight for name, weight in VISUAL_SCORE_WEIGHTS.items())),
        4,
    )


def _unique_text(reviews: list[dict[str, Any]], key: str, limit: int) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for review in reviews:
        source = review.get(key, [])
        if not isinstance(source, list):
            continue
        for value in source:
            text = str(value).strip()
            if text and text not in seen:
                seen.add(text)
                values.append(text[:400])
                if len(values) >= limit:
                    return values
    return values


def aggregate_visual_reviews(reviews: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate exactly five predeclared slots without selecting outcomes."""

    slots = [dict(review) for review in reviews]
    if len(slots) != VISUAL_REVIEW_SLOT_COUNT:
        raise FinalGateError(
            f"Exactly {VISUAL_REVIEW_SLOT_COUNT} visual review slots are required"
        )
    slot_ids = [_safe_int(review.get("slot_id"), index) for index, review in enumerate(slots, 1)]
    if sorted(slot_ids) != list(range(1, VISUAL_REVIEW_SLOT_COUNT + 1)):
        raise FinalGateError("Visual review slot ids must be exactly 1 through 5")

    dimensions = [validate_dimension_scores(review) for review in slots]
    dimension_medians = {
        name: float(median(values[name] for values in dimensions))
        for name in VISUAL_SCORE_WEIGHTS
    }
    slot_scores = [weighted_visual_score(values) for values in dimensions]
    ordered_scores = sorted(slot_scores)
    score_q1 = float(ordered_scores[1])
    score_q3 = float(ordered_scores[3])
    score_iqr = round(score_q3 - score_q1, 4)
    visual_score = weighted_visual_score(dimension_medians)

    critical_slot_count = sum(
        bool(review.get("critical_failures")) for review in slots
    )
    critical_unstable = 0 < critical_slot_count < VISUAL_REVIEW_SLOT_COUNT
    critical_consensus = critical_slot_count == VISUAL_REVIEW_SLOT_COUNT
    rejection_reasons: list[str] = []
    if visual_score < 90.0:
        rejection_reasons.append("visual_score_below_90")
    if score_iqr > MAX_VISUAL_SCORE_IQR:
        rejection_reasons.append("total_score_iqr_above_6")
    for name, floor in VISUAL_DIMENSION_FLOORS.items():
        if dimension_medians[name] < floor:
            rejection_reasons.append(f"{name}_median_below_{int(floor)}")
    if critical_unstable:
        rejection_reasons.append("critical_failure_unstable")
    if critical_consensus:
        rejection_reasons.append("critical_failure_consensus")
    passed = not rejection_reasons

    models = sorted({str(review.get("model")) for review in slots if review.get("model")})
    prompts = sorted({str(review.get("prompt_version")) for review in slots if review.get("prompt_version")})
    request_ids = [review.get("request_id") for review in slots]
    critical_failures = _unique_text(slots, "critical_failures", 30)
    return {
        "visual_score": visual_score,
        "verdict": "pass" if passed else "reject",
        "batch_unlock": passed,
        "subscores": dimension_medians,
        "critical_failures": critical_failures,
        "issues": _unique_text(slots, "issues", 30),
        "recommended_adjustments": _unique_text(slots, "recommended_adjustments", 30),
        "score_kind": "locally_weighted_five_slot_dimension_medians",
        "model": models[0] if len(models) == 1 else models,
        "request_id": None,
        "request_ids": request_ids,
        "prompt_version": prompts[0] if len(prompts) == 1 else prompts,
        "review_count": VISUAL_REVIEW_SLOT_COUNT,
        "review_slots": slots,
        "aggregation": {
            "method": "dimension_medians_then_local_weighted_sum",
            "weights": dict(VISUAL_SCORE_WEIGHTS),
            "dimension_floors": dict(VISUAL_DIMENSION_FLOORS),
            "slot_scores": slot_scores,
            "score_q1": score_q1,
            "score_q3": score_q3,
            "total_score_iqr": score_iqr,
            "maximum_total_score_iqr": MAX_VISUAL_SCORE_IQR,
            "critical_slot_count": critical_slot_count,
            "critical_stability": (
                "none"
                if critical_slot_count == 0
                else ("consistent_failure" if critical_consensus else "unstable")
            ),
            "rejection_reasons": rejection_reasons,
        },
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _image_facts(path: Path) -> tuple[list[int], str]:
    with Image.open(path) as image:
        return [int(image.width), int(image.height)], str(image.format or "").upper()


def _job_asset_sha256(job_path: Path, raw_path: Any, field: str) -> str | None:
    """Hash an optional job-referenced mask without retaining its local path."""

    if raw_path in (None, ""):
        return None
    if not isinstance(raw_path, str):
        raise FinalGateError(f"job.json {field} must be a path or null")
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = job_path.parent / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FinalGateError(f"job.json referenced evidence does not exist: {field}")
    return sha256_file(candidate)


def _job_review_semantics(job: dict[str, Any], job_path: Path) -> dict[str, Any]:
    """Return the path-neutral scene semantics shown to the visual reviewer.

    The exact job bytes are bound separately.  This projection makes the
    security-relevant meaning human-auditable while avoiding run-directory
    paths in the semantic record.  Product/source image identity is already
    bound by the independently computed image hashes in the review bundle.
    """

    raw_products = job.get("products")
    if not isinstance(raw_products, list) or not raw_products:
        raise FinalGateError("job.json must contain product semantics")
    products: list[dict[str, Any]] = []
    for index, item in enumerate(raw_products, 1):
        if not isinstance(item, dict):
            raise FinalGateError(f"job.json products[{index}] is invalid")
        products.append({
            key: value
            for key, value in item.items()
            if key not in {"rgba", "source"}
        })

    placements: list[dict[str, Any]] = []
    raw_placements = job.get("ui_placements", [])
    if raw_placements is not None and not isinstance(raw_placements, list):
        raise FinalGateError("job.json ui_placements must be a list")
    for index, item in enumerate(raw_placements or [], 1):
        if not isinstance(item, dict):
            raise FinalGateError(f"job.json ui_placements[{index}] is invalid")
        placements.append({
            key: value
            for key, value in item.items()
            if key not in {"rgba", "source"}
        })

    return {
        "support_polygon": job.get("support_polygon"),
        "protected_polygons": job.get("protected_polygons", []),
        "referenced_evidence_sha256": {
            "support_mask": _job_asset_sha256(
                job_path, job.get("support_mask"), "support_mask",
            ),
            "protected_mask": _job_asset_sha256(
                job_path, job.get("protected_mask"), "protected_mask",
            ),
        },
        "lighting": job.get("lighting"),
        "products": products,
        "ui_placements": placements,
        "source_dimensions": job.get("source_dimensions"),
        "upscaled_to_4k": job.get("upscaled_to_4k"),
        "refinement_pass": job.get("refinement_pass", 0),
        "refinement_lock": job.get("refinement_lock"),
    }


def _load_job_binding(job_path: str | Path) -> dict[str, Any]:
    path = Path(job_path).resolve()
    if not path.is_file():
        raise FinalGateError(f"Review job manifest does not exist: {path}")
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise FinalGateError("job.json is unreadable") from exc
    if not isinstance(loaded, dict):
        raise FinalGateError("job.json must contain an object")
    semantics = _job_review_semantics(loaded, path)
    return {
        "job_sha256": sha256_file(path),
        "semantic_sha256": _canonical_sha256(semantics),
        "semantics": semantics,
    }


def build_review_bundle(
    background: str | Path,
    final_image: str | Path,
    products: Iterable[str | Path],
    audit: dict[str, Any] | None,
    *,
    audit_path: str | Path | None = None,
    job_path: str | Path,
    reviewer_model: str,
    prompt_version: str,
) -> dict[str, Any]:
    """Create an immutable content-addressed description of one review.

    The bundle hash, rather than a request id or wall-clock time, is the cache
    key.  Repeated clicks therefore cannot sample reviews until a preferred
    score appears.  A changed image, audit, model or policy creates a new key.
    """

    background_path = Path(background).resolve()
    final_path = Path(final_image).resolve()
    product_paths = [Path(value).resolve() for value in products]
    for path in [background_path, final_path, *product_paths]:
        if not path.is_file():
            raise FinalGateError(f"Review input does not exist: {path}")

    dimensions, image_format = _image_facts(final_path)
    audit_file = Path(audit_path).resolve() if audit_path is not None else None
    audit_sha256 = sha256_file(audit_file) if audit_file is not None and audit_file.is_file() else None
    job_binding = _load_job_binding(job_path)
    manifest = {
        "schema_version": FINAL_GATE_SCHEMA_VERSION,
        "policy_version": FINAL_GATE_POLICY_VERSION,
        "reviewer": {
            "model": str(reviewer_model),
            "prompt_version": str(prompt_version),
        },
        "inputs": {
            "background_sha256": sha256_file(background_path),
            "final_sha256": sha256_file(final_path),
            "product_rgba_sha256": [sha256_file(path) for path in product_paths],
            "audit_sha256": audit_sha256,
            "final_dimensions": dimensions,
            "final_format": image_format,
        },
        "job_binding": job_binding,
        # The path is intentionally excluded from the content hash. Two exact
        # copies of one artifact are the same review package.
        "audit_present": isinstance(audit, dict),
    }
    return {**manifest, "bundle_sha256": _canonical_sha256(manifest)}


def _pixel_gate(audit: dict[str, Any] | None) -> dict[str, Any]:
    pixel = audit.get("pixel_audit", {}) if isinstance(audit, dict) else {}
    changed = _safe_int(pixel.get("changed_pixels")) if isinstance(pixel, dict) else -1
    maximum = _safe_int(pixel.get("max_delta")) if isinstance(pixel, dict) else -1
    reported = str(pixel.get("verdict", "missing")) if isinstance(pixel, dict) else "missing"
    passed = changed == 0 and maximum == 0 and reported == "pass"
    return {
        "changed_pixels": changed,
        "max_delta": maximum,
        "reported_verdict": reported,
        "verdict": "pass" if passed else "reject",
    }


def _dimension_gate(
    bundle: dict[str, Any],
    audit: dict[str, Any] | None,
    minimum_long_edge: int,
) -> dict[str, Any]:
    actual = list(bundle["inputs"]["final_dimensions"])
    reported_raw = audit.get("output_dimensions", []) if isinstance(audit, dict) else []
    reported = [_safe_int(value) for value in reported_raw] if isinstance(reported_raw, list) else []
    format_name = str(bundle["inputs"].get("final_format", ""))
    passed = (
        len(actual) == 2
        and max(actual) >= int(minimum_long_edge)
        and reported == actual
        and format_name == "PNG"
        and str((audit or {}).get("output_format", "")).lower() == "png"
    )
    return {
        "actual_dimensions": actual,
        "reported_dimensions": reported,
        "minimum_long_edge": int(minimum_long_edge),
        "actual_format": format_name,
        "verdict": "pass" if passed else "reject",
    }


def _input_binding_gate(bundle: dict[str, Any], audit: dict[str, Any] | None) -> dict[str, Any]:
    expected_products: list[str] = []
    if isinstance(audit, dict) and isinstance(audit.get("products"), list):
        expected_products = [
            str(item.get("rgba_sha256", ""))
            for item in audit["products"]
            if isinstance(item, dict)
        ]
    reviewed_products = [str(value) for value in bundle["inputs"]["product_rgba_sha256"]]
    audit_background = str((audit or {}).get("background_sha256", ""))
    reviewed_background = str(bundle["inputs"].get("background_sha256", ""))
    background_matches = bool(audit_background) and audit_background == reviewed_background
    passed = (
        bool(bundle["inputs"].get("audit_sha256"))
        and background_matches
        and bool(expected_products)
        and Counter(expected_products) == Counter(reviewed_products)
    )
    return {
        "audit_present": bool(bundle["inputs"].get("audit_sha256")),
        "audit_background_sha256": audit_background or None,
        "reviewed_background_sha256": reviewed_background or None,
        "background_hashes_match": background_matches,
        "reviewed_product_count": len(reviewed_products),
        "audited_product_count": len(expected_products),
        "product_hashes_match": Counter(expected_products) == Counter(reviewed_products),
        "verdict": "pass" if passed else "reject",
    }


def _physical_gate(audit: dict[str, Any] | None) -> dict[str, Any]:
    source = audit.get("physical_audit") if isinstance(audit, dict) else None
    if not isinstance(source, dict):
        return {
            "schema_version": None,
            "subscores": {},
            "reported_verdict": "missing",
            "failures": ["physical_audit_missing"],
            "verdict": "reject",
        }
    raw = source.get("subscores")
    floors = source.get("dimension_floors")
    clean: dict[str, float] = {}
    failures: list[str] = []
    if not isinstance(raw, dict) or not isinstance(floors, dict):
        failures.append("physical_audit_scores_or_floors_missing")
    else:
        for name in PHYSICAL_DIMENSIONS:
            try:
                score = float(raw[name])
                floor = float(floors[name])
            except (KeyError, TypeError, ValueError) as exc:
                failures.append(f"{name}_invalid")
                continue
            if not math.isfinite(score) or not math.isfinite(floor) or not 0 <= score <= 100:
                failures.append(f"{name}_invalid")
                continue
            clean[name] = score
            if score < floor:
                failures.append(f"{name}_below_{int(floor)}")
    reported = str(source.get("verdict", "missing"))
    if reported != "pass":
        failures.append("physical_audit_reported_reject")
    return {
        "schema_version": source.get("schema_version"),
        "subscores": clean,
        "dimension_floors": dict(floors) if isinstance(floors, dict) else {},
        "reported_verdict": reported,
        "failures": sorted(set(failures)),
        "verdict": "pass" if not failures and len(clean) == len(PHYSICAL_DIMENSIONS) else "reject",
    }


def _evidence_fused_review(
    review_slots: list[dict[str, Any]],
    authoritative_review: dict[str, Any],
    physical: dict[str, Any],
) -> dict[str, Any]:
    """Fuse measurable local evidence with MiniMax's subjective dimensions.

    Deterministic measurements prove that the requested layers exist and obey
    the declared geometry, but they cannot prove that those layers *look*
    attached or photographically plausible.  For the four measurable
    dimensions the local score is therefore an upper bound, not a replacement
    for the visual judgement.  Taking the lower of local evidence and each
    visual slot closes the dangerous case where mathematically connected
    shadows still look detached to people.
    """

    physical_scores = dict(physical.get("subscores", {}))
    fused_slots: list[dict[str, float]] = []
    for slot in review_slots:
        dimensions = validate_dimension_scores(slot)
        for name in PHYSICAL_DIMENSIONS:
            if name in physical_scores:
                dimensions[name] = min(
                    float(dimensions[name]),
                    float(physical_scores[name]),
                )
        fused_slots.append(dimensions)
    medians = {
        name: float(median(values[name] for values in fused_slots))
        for name in VISUAL_SCORE_WEIGHTS
    }
    scores = [weighted_visual_score(values) for values in fused_slots]
    ordered = sorted(scores)
    iqr = round(float(ordered[3] - ordered[1]), 4)
    raw_score = float(authoritative_review["visual_score"])
    model_dimensions = [
        name for name in VISUAL_SCORE_WEIGHTS if name not in PHYSICAL_DIMENSIONS
    ]
    model_weight = sum(VISUAL_SCORE_WEIGHTS[name] for name in model_dimensions)
    model_subjective_score = float(sum(
        float(authoritative_review["subscores"][name]) * VISUAL_SCORE_WEIGHTS[name]
        for name in model_dimensions
    ) / max(1e-9, model_weight))
    critical_slot_count = int(authoritative_review["aggregation"]["critical_slot_count"])
    rejection_reasons: list[str] = []
    visual_score = weighted_visual_score(medians)
    if visual_score < 90.0:
        rejection_reasons.append("evidence_fused_visual_score_below_90")
    if model_subjective_score < MIN_MODEL_SUBJECTIVE_SCORE:
        rejection_reasons.append("model_subjective_score_below_75")
    if iqr > MAX_HYBRID_SCORE_IQR:
        rejection_reasons.append("evidence_fused_score_iqr_above_6")
    # A majority, rather than one inconsistent sample, is required for the
    # model's unstructured critical-failure veto.  Every raw slot remains in
    # the gate record for human audit.
    if critical_slot_count >= 3:
        rejection_reasons.append("critical_failure_majority")
    for name, floor in VISUAL_DIMENSION_FLOORS.items():
        if name in PHYSICAL_DIMENSIONS:
            continue
        if medians[name] < floor:
            rejection_reasons.append(f"{name}_median_below_{int(floor)}")
    return {
        "visual_score": visual_score,
        "subscores": medians,
        "slot_scores": scores,
        "score_q1": float(ordered[1]),
        "score_q3": float(ordered[3]),
        "total_score_iqr": iqr,
        "maximum_total_score_iqr": MAX_HYBRID_SCORE_IQR,
        "raw_model_advisory_score": raw_score,
        "model_subjective_score": round(model_subjective_score, 4),
        "raw_model_critical_slot_count": critical_slot_count,
        "physical_dimensions": list(PHYSICAL_DIMENSIONS),
        "physical_fusion_rule": "minimum_of_visual_slot_and_local_upper_bound",
        "model_dimensions": model_dimensions,
        "rejection_reasons": rejection_reasons,
        "verdict": "pass" if not rejection_reasons else "reject",
    }


def build_final_gate(
    bundle: dict[str, Any],
    audit: dict[str, Any] | None,
    visual_review: dict[str, Any],
    *,
    minimum_long_edge: int = MIN_DELIVERY_LONG_EDGE,
) -> dict[str, Any]:
    """Merge hard audits and visual review into the server-authoritative gate."""

    review_slots = visual_review.get("review_slots")
    if not isinstance(review_slots, list) or len(review_slots) != VISUAL_REVIEW_SLOT_COUNT:
        raise FinalGateError("Five complete visual review slots are required")
    # Re-aggregate the five slots here. Caller/model totals, verdicts, IQR and
    # even previously aggregated medians are never authoritative inputs.
    authoritative_review = aggregate_visual_reviews(review_slots)
    pixel = _pixel_gate(audit)
    dimensions = _dimension_gate(bundle, audit, minimum_long_edge)
    binding = _input_binding_gate(bundle, audit)
    physical = _physical_gate(audit)
    fused = _evidence_fused_review(review_slots, authoritative_review, physical)
    consensus_dimensions = dict(fused["subscores"])
    visual_score = float(fused["visual_score"])
    critical = _unique_text(review_slots, "critical_failures", 30)
    visual_passed = fused["verdict"] == "pass"
    hard_passed = all(
        gate["verdict"] == "pass" for gate in (pixel, dimensions, binding, physical)
    )
    passed = hard_passed and visual_passed

    # Mirror the established review fields at top level so /api/visual-review
    # remains backward compatible while the full authoritative record is also
    # available to delivery and batch systems.
    return {
        "schema_version": FINAL_GATE_SCHEMA_VERSION,
        "policy_version": FINAL_GATE_POLICY_VERSION,
        "bundle_sha256": str(bundle["bundle_sha256"]),
        "review_bundle": dict(bundle),
        "inputs": dict(bundle["inputs"]),
        "job_binding": dict(bundle["job_binding"]),
        "hard_gates": {
            "pixel_integrity": pixel,
            "output_4k_png": dimensions,
            "audit_input_binding": binding,
            "physical_evidence": physical,
        },
        "visual_score": visual_score,
        "pixel_verdict": pixel["verdict"],
        "verdict": "pass" if passed else "reject",
        "delivery_verdict": "pass" if passed else "reject",
        "batch_unlock": passed,
        "subscores": consensus_dimensions,
        "critical_failures": critical,
        "issues": list(authoritative_review.get("issues", [])),
        "recommended_adjustments": list(authoritative_review.get("recommended_adjustments", [])),
        "score_kind": "server_authoritative_evidence_fused_consensus_gate",
        "model": authoritative_review.get("model"),
        "request_id": authoritative_review.get("request_id"),
        "request_ids": list(authoritative_review.get("request_ids", [])),
        "prompt_version": authoritative_review.get("prompt_version"),
        "review_count": VISUAL_REVIEW_SLOT_COUNT,
        "review_slots": list(review_slots),
        "aggregation": dict(fused),
        "model_advisory": authoritative_review,
        "visual_review": authoritative_review,
    }


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    """Atomically persist a gate or resumable review-slot record."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def write_final_gate(path: str | Path, gate: dict[str, Any]) -> None:
    """Atomically publish a completed final gate."""

    write_json_atomic(path, gate)


def verify_deliverable(final_image: str | Path, gate: dict[str, Any]) -> None:
    """Reject stale, edited or non-passing artifacts at download time."""

    final_path = Path(final_image).resolve()
    if str(gate.get("schema_version")) != FINAL_GATE_SCHEMA_VERSION:
        raise FinalGateError("Final gate schema is missing or unsupported")
    if gate.get("verdict") != "pass" or gate.get("batch_unlock") is not True:
        raise FinalGateError("Artifact has not passed the final delivery gate")
    score = float(gate.get("visual_score", -1))
    if not math.isfinite(score) or score < 90.0:
        raise FinalGateError("Artifact visual score is below 90")
    hard_gates = gate.get("hard_gates", {})
    required = (
        "pixel_integrity", "output_4k_png", "audit_input_binding", "physical_evidence",
    )
    if not isinstance(hard_gates, dict) or any(
        not isinstance(hard_gates.get(name), dict)
        or hard_gates[name].get("verdict") != "pass"
        for name in required
    ):
        raise FinalGateError("One or more hard delivery gates did not pass")
    expected_hash = str((gate.get("inputs") or {}).get("final_sha256", ""))
    if not expected_hash or sha256_file(final_path) != expected_hash:
        raise FinalGateError("Final image hash no longer matches final_gate.json")

    bundle = gate.get("review_bundle")
    if not isinstance(bundle, dict):
        raise FinalGateError("Final gate is not bound to an immutable review bundle")
    if (
        bundle.get("schema_version") != gate.get("schema_version")
        or bundle.get("policy_version") != gate.get("policy_version")
        or bundle.get("inputs") != gate.get("inputs")
        or bundle.get("job_binding") != gate.get("job_binding")
    ):
        raise FinalGateError("Final gate fields do not match its review bundle")
    stored_bundle_hash = str(bundle.get("bundle_sha256", ""))
    manifest = {key: value for key, value in bundle.items() if key != "bundle_sha256"}
    if (
        not stored_bundle_hash
        or stored_bundle_hash != str(gate.get("bundle_sha256", ""))
        or _canonical_sha256(manifest) != stored_bundle_hash
    ):
        raise FinalGateError("Final gate review bundle is inconsistent")

    audit_path = final_path.parent / "audit.json"
    expected_audit_hash = str((gate.get("inputs") or {}).get("audit_sha256", ""))
    if (
        not audit_path.is_file()
        or not expected_audit_hash
        or sha256_file(audit_path) != expected_audit_hash
    ):
        raise FinalGateError("audit.json no longer matches the reviewed artifact")
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise FinalGateError("audit.json is unreadable") from exc

    job_path = final_path.parent / "job.json"
    stored_job_binding = bundle.get("job_binding")
    if not isinstance(stored_job_binding, dict):
        raise FinalGateError("Final gate is not bound to job.json semantics")
    live_job_binding = _load_job_binding(job_path)
    if live_job_binding != stored_job_binding:
        raise FinalGateError("job.json no longer matches the reviewed artifact")
    minimum_long_edge = _safe_int(
        hard_gates["output_4k_png"].get("minimum_long_edge"),
        MIN_DELIVERY_LONG_EDGE,
    )
    live_hard_gates = (
        _pixel_gate(audit),
        _dimension_gate(bundle, audit, minimum_long_edge),
        _input_binding_gate(bundle, audit),
        _physical_gate(audit),
    )
    if any(item["verdict"] != "pass" for item in live_hard_gates):
        raise FinalGateError("Live audit revalidation failed")
    visual_review = gate.get("visual_review")
    if not isinstance(visual_review, dict):
        raise FinalGateError("Five-slot visual review evidence is missing")
    reconstructed = build_final_gate(
        bundle,
        audit,
        visual_review,
        minimum_long_edge=minimum_long_edge,
    )
    if reconstructed != gate:
        raise FinalGateError("final_gate.json is inconsistent with its five review slots")
