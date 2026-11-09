"""Build a read-only, leakage-safe benchmark manifest from dataset metadata.

The dataset is produced by another process and may still be changing.  This
tool never scans the classification folders and never opens image pixels.  It
reads the canonical CSV twice and refuses to publish a manifest unless the
bytes remained stable for the complete build.  Outputs must live outside the
dataset directory.

The local corpus contains repeated five-file capture sessions: three product
photographs, one empty receiver/background and one already-composed reference.
Only files 1--4 are automation inputs.  The fifth file is retained as a hash
and image id for optional human review; its path is deliberately omitted so a
batch program cannot accidentally use it as an input or pixel ground truth.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


CANONICAL_INDEX = Path("09_索引与标签") / "all_images_index.csv"
SUMMARY_FILE = Path("执行摘要.json")
REFERENCE_FILENAME = re.compile(r"^5\.(?:jpe?g|png|webp)$", re.IGNORECASE)
INPUT_FILENAME = re.compile(r"^([1-4])\.(?:jpe?g|png|webp)$", re.IGNORECASE)


class DatasetChangedError(RuntimeError):
    """Raised when the producer changed the canonical index during a build."""


@dataclass(frozen=True)
class Snapshot:
    payload: bytes
    sha256: str
    mtime_ns: int


def _snapshot(path: Path) -> Snapshot:
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
        raise DatasetChangedError("canonical CSV changed while it was being read")
    return Snapshot(payload, hashlib.sha256(payload).hexdigest(), after.st_mtime_ns)


def _rows(payload: bytes) -> list[dict[str, str]]:
    text = payload.decode("utf-8-sig")
    return list(csv.DictReader(text.splitlines()))


def _normalise_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _safe_number(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _session_key(row: dict[str, str]) -> str:
    source = Path(row.get("source_path", ""))
    return str(source.parent).casefold()


def _duplicate_key(row: dict[str, str]) -> str:
    group = row.get("duplicate_group_id", "").strip()
    return group or f"sha256:{row.get('sha256', '').strip()}"


def _case_number(session: str) -> tuple[int, str]:
    name = Path(session).name
    match = re.search(r"(\d+)$", name)
    return (int(match.group(1)), name) if match else (10**9, name.casefold())


def _local_cases(rows: Iterable[dict[str, str]]) -> list[dict[str, object]]:
    sessions: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("source_type") == "本地图片":
            sessions[_session_key(row)].append(row)

    cases: list[dict[str, object]] = []
    for session, items in sorted(sessions.items(), key=lambda pair: _case_number(pair[0])):
        numbered: dict[int, dict[str, str]] = {}
        references: list[dict[str, str]] = []
        for row in items:
            filename = row.get("original_filename", "")
            match = INPUT_FILENAME.match(filename)
            if match:
                numbered[int(match.group(1))] = row
            elif REFERENCE_FILENAME.match(filename):
                references.append(row)
        if set(numbered) != {1, 2, 3, 4} or len(references) != 1:
            continue
        reference = references[0]
        products = []
        for index in (1, 2, 3):
            row = numbered[index]
            products.append({
                "image_id": row["image_id"],
                "source_path": row["source_path"],
                "sha256": row["sha256"],
                "main_object": row.get("main_object", ""),
                "material": row.get("material", ""),
                "transparency": row.get("transparency", ""),
                "training_eligible": _normalise_bool(row.get("training_eligible", "")),
            })
        background = numbered[4]
        cases.append({
            "case_id": f"local_session_{len(cases) + 1:02d}",
            "session_name": Path(session).name,
            "products": products,
            "background": {
                "image_id": background["image_id"],
                "source_path": background["source_path"],
                "sha256": background["sha256"],
                "surface_type": background.get("surface_type", ""),
                "surface_reflectivity": background.get("surface_reflectivity", ""),
                "training_eligible": _normalise_bool(background.get("training_eligible", "")),
            },
            "human_reference": {
                "image_id": reference["image_id"],
                "sha256": reference["sha256"],
                "policy": "human_reference_only_do_not_decode_in_automation",
                "is_pixel_ground_truth": False,
            },
            "rights_policy": "internal_evaluation_only_until_local_rights_are_confirmed",
        })
    return cases


def _assign_splits(cases: list[dict[str, object]]) -> None:
    """Keep complete capture sessions together with a deterministic holdout."""
    if len(cases) == 10:
        development = {1, 3, 4, 5, 8, 10}
        validation = {2, 6}
        for index, case in enumerate(cases, start=1):
            case["split"] = (
                "development" if index in development
                else "validation" if index in validation
                else "blind_holdout"
            )
        return
    # For an expanded dataset, do not pretend the historical ten-case mapping
    # still applies.  A stable session hash keeps all files from one shoot in a
    # single split and remains reproducible across machines.
    for case in cases:
        token = hashlib.sha256(str(case["session_name"]).encode("utf-8")).digest()[0]
        case["split"] = (
            "blind_holdout" if token < 26
            else "validation" if token < 77
            else "development"
        )


def _anchor_score(row: dict[str, str], dimension: str) -> float:
    material = row.get("material", "")
    transparency = row.get("transparency", "")
    reflection = row.get("reflection_type", "")
    composition = row.get("composition", "")
    issue = row.get("issue_tags", "")
    role = row.get("reference_role", "")
    score = _safe_number(row.get("synthesis_value", "")) * 3.0
    score += _safe_number(row.get("aesthetic_score", "")) * 2.0
    score += 3.0 if "优秀参考" in role else 0.0
    score -= 2.0 if row.get("quality_status") == "低质量" else 0.0
    if dimension == "transparent_optics":
        score += 8.0 if transparency not in {"", "不透明", "无法判断"} else -8.0
        score += 4.0 if any(word in material for word in ("玻璃", "透明", "液体", "磨砂")) else 0.0
        score += 2.0 if row.get("refraction") not in {"", "无", "无法判断"} else 0.0
    elif dimension == "receiver_reflection":
        score += 8.0 if reflection not in {"", "无倒影"} else -6.0
        score += 4.0 if row.get("surface_reflectivity") in {"中至高反光", "高反光"} else 0.0
    elif dimension == "composition_scale":
        score += 7.0 if any(word in composition for word in ("前后", "高低", "多商品", "留白")) else 0.0
        score += 4.0 if row.get("object_count") not in {"", "单一主体"} else 0.0
    elif dimension == "cutout_edges":
        score += _safe_number(row.get("cutout_difficulty_score", "")) * 2.0
        score += 4.0 if row.get("cutout_edge_complexity") not in {"", "低", "简单"} else 0.0
    elif dimension == "negative_controls":
        score += 7.0 if "问题参考" in role else -5.0
        score += min(6.0, len([part for part in re.split(r"[;；]", issue) if part.strip()]))
    return score


def _reference_anchors(rows: list[dict[str, str]], per_dimension: int = 5) -> dict[str, list[dict[str, object]]]:
    dimensions = (
        "transparent_optics", "receiver_reflection", "composition_scale",
        "cutout_edges", "negative_controls",
    )
    selected_groups: set[str] = set()
    result: dict[str, list[dict[str, object]]] = {}
    for dimension in dimensions:
        candidates = sorted(
            (
                row for row in rows
                if row.get("source_type") == "网络图片"
                and _normalise_bool(row.get("training_eligible", ""))
                and row.get("quality_status") != "低质量"
                and row.get("license_status") not in {"待人工确认", ""}
            ),
            key=lambda row: (-_anchor_score(row, dimension), row.get("image_id", "")),
        )
        picks: list[dict[str, object]] = []
        for row in candidates:
            group = _duplicate_key(row)
            if group in selected_groups:
                continue
            if _anchor_score(row, dimension) < 8.0:
                continue
            picks.append({
                "image_id": row["image_id"],
                "copied_path": row.get("copied_path", ""),
                "sha256": row["sha256"],
                "source_type": row.get("source_type", ""),
                "main_object": row.get("main_object", ""),
                "material": row.get("material", ""),
                "transparency": row.get("transparency", ""),
                "reference_role": row.get("reference_role", ""),
                "reference_reason": row.get("reference_reason", ""),
                "license_status": row.get("license_status", ""),
                "training_eligible": _normalise_bool(row.get("training_eligible", "")),
                "usage": "visual_reference_only_not_pixel_ground_truth",
            })
            selected_groups.add(group)
            if len(picks) >= per_dimension:
                break
        result[dimension] = picks
    return result


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent),
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def build_manifest(dataset_root: Path, output: Path) -> dict[str, object]:
    dataset_root = dataset_root.resolve()
    output = output.resolve()
    try:
        output.relative_to(dataset_root)
    except ValueError:
        pass
    else:
        raise ValueError("benchmark output must be outside the dataset directory")

    index = dataset_root / CANONICAL_INDEX
    first = _snapshot(index)
    rows = _rows(first.payload)
    cases = _local_cases(rows)
    _assign_splits(cases)
    anchors = _reference_anchors(rows)
    summary = {}
    summary_path = dataset_root / SUMMARY_FILE
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    summary_count = summary.get("final_unique_images")
    summary_consistent = summary_count is None or summary_count == len(rows)

    # Re-read after every expensive selection/validation step.  A changing
    # producer must never yield a partly old, partly new benchmark baseline.
    second = _snapshot(index)
    if first.sha256 != second.sha256 or first.mtime_ns != second.mtime_ns:
        raise DatasetChangedError(
            "dataset index changed during benchmark build; retry after producer is idle"
        )

    role_counts = Counter(row.get("reference_role", "") for row in rows)
    payload: dict[str, object] = {
        "schema_version": "strict-compositor-dataset-benchmark-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "root": str(dataset_root),
            "canonical_index": str(index),
            "canonical_index_sha256": first.sha256,
            "canonical_index_mtime_ns": first.mtime_ns,
            "row_count": len(rows),
            "producer_summary_generated_at": summary.get("generated_at"),
            "producer_summary_unique_images": summary_count,
            "producer_summary_consistent": summary_consistent,
            "snapshot_status": (
                "stable_candidate" if summary_consistent
                else "provisional_producer_still_updating"
            ),
            "reference_role_counts": dict(sorted(role_counts.items())),
        },
        "policy": {
            "dataset_is_read_only": True,
            "scan_classification_directories": False,
            "decode_human_reference_composites_in_automation": False,
            "existing_scores_are_visual_ground_truth": False,
            "keep_capture_sessions_and_duplicate_groups_in_one_split": True,
            "training_requires_training_eligible_true_and_separate_rights_review": True,
            "delivery_threshold": 90,
        },
        "strict_composition_cases": cases,
        "visual_reference_anchors": anchors,
    }
    _atomic_json(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        payload = build_manifest(args.dataset_root, args.output)
    except DatasetChangedError as error:
        parser.error(str(error))
    print(json.dumps({
        "output": str(args.output.resolve()),
        "row_count": payload["dataset"]["row_count"],
        "strict_cases": len(payload["strict_composition_cases"]),
        "status": payload["dataset"]["snapshot_status"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
