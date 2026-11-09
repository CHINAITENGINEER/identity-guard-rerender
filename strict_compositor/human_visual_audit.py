from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


class HumanVisualAuditError(ValueError):
    """Raised when a human visual audit is incomplete or bound to another image."""


# These are deliberately categorical rather than numerical.  A weighted score
# can hide a catastrophic defect (for example a beautiful scene with unreadable
# product text).  Human review is a conjunction: every applicable critical
# dimension must pass.
REQUIRED_DIMENSIONS = (
    "identity_and_text",
    "material_readability",
    "relative_scale",
    "semantic_pose_and_support",
    "composition_aesthetics",
    "contact_grounding",
    "lighting_shadow_reflection",
    "edge_and_matte",
)

OPTIONAL_DIMENSIONS = (
    "transparent_optics",
    "receiver_perspective",
)

DIMENSION_VERDICTS = {"pass", "reject", "not_applicable"}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pending_human_visual_audit(
    candidate_path: str | Path,
    *,
    machine_diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    candidate = Path(candidate_path).resolve()
    if not candidate.is_file():
        raise HumanVisualAuditError(f"candidate image does not exist: {candidate}")
    return {
        "schema_version": 1,
        "status": "pending_human_review",
        "verdict": "pending_human_review",
        "candidate_path": str(candidate),
        "candidate_sha256": sha256_file(candidate),
        "review_authority": "human_visual_inspection",
        "machine_scores_are_diagnostic_only": True,
        "machine_diagnostics": dict(machine_diagnostics or {}),
        "dimensions": {
            name: {"verdict": "pending", "observation": ""}
            for name in (*REQUIRED_DIMENSIONS, *OPTIONAL_DIMENSIONS)
        },
    }


def _validate_dimension(name: str, value: Any, *, required: bool) -> str:
    if not isinstance(value, Mapping):
        raise HumanVisualAuditError(f"dimension {name!r} must be an object")
    verdict = str(value.get("verdict", ""))
    if verdict not in DIMENSION_VERDICTS:
        raise HumanVisualAuditError(
            f"dimension {name!r} verdict must be pass, reject, or not_applicable"
        )
    observation = str(value.get("observation", "")).strip()
    if not observation:
        raise HumanVisualAuditError(f"dimension {name!r} needs a visual observation")
    if required and verdict == "not_applicable":
        raise HumanVisualAuditError(f"required dimension {name!r} cannot be not_applicable")
    return verdict


def validate_human_visual_audit(
    payload: Mapping[str, Any],
    *,
    candidate_path: str | Path | None = None,
) -> dict[str, Any]:
    if int(payload.get("schema_version", 0)) != 1:
        raise HumanVisualAuditError("unsupported human visual audit schema")
    if str(payload.get("status", "")) != "reviewed":
        raise HumanVisualAuditError("human visual audit status must be reviewed")
    if str(payload.get("review_authority", "")) != "human_visual_inspection":
        raise HumanVisualAuditError("review authority must be human_visual_inspection")
    reviewer = str(payload.get("reviewer", "")).strip()
    if not reviewer:
        raise HumanVisualAuditError("reviewer is required")

    candidate = Path(candidate_path or str(payload.get("candidate_path", ""))).resolve()
    if not candidate.is_file():
        raise HumanVisualAuditError(f"candidate image does not exist: {candidate}")
    expected_hash = str(payload.get("candidate_sha256", ""))
    actual_hash = sha256_file(candidate)
    if expected_hash != actual_hash:
        raise HumanVisualAuditError("human audit is bound to a different candidate image")

    dimensions = payload.get("dimensions")
    if not isinstance(dimensions, Mapping):
        raise HumanVisualAuditError("dimensions must be an object")
    dimension_verdicts: dict[str, str] = {}
    for name in REQUIRED_DIMENSIONS:
        dimension_verdicts[name] = _validate_dimension(
            name, dimensions.get(name), required=True,
        )
    for name in OPTIONAL_DIMENSIONS:
        if name in dimensions:
            dimension_verdicts[name] = _validate_dimension(
                name, dimensions.get(name), required=False,
            )

    computed_verdict = (
        "pass"
        if all(value == "pass" for value in dimension_verdicts.values() if value != "not_applicable")
        else "reject"
    )
    declared_verdict = str(payload.get("verdict", ""))
    if declared_verdict != computed_verdict:
        raise HumanVisualAuditError(
            f"declared verdict {declared_verdict!r} disagrees with visual dimensions "
            f"({computed_verdict!r})"
        )
    return {
        **dict(payload),
        "candidate_path": str(candidate),
        "candidate_sha256": actual_hash,
        "verdict": computed_verdict,
        "machine_scores_are_diagnostic_only": True,
    }


def write_audit(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)
    return target


def reviewed_at_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
