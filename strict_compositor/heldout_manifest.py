"""Immutable manifest support for the private cross-scene holdout set.

The holdout set is deliberately separated from training, prompt tuning and
threshold tuning.  This module never decodes an image.  It only enumerates the
two declared raw-input directories and reads file bytes to calculate SHA-256.
Generated composites and reference images are therefore outside its read
surface by construction.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "strict-compositor-private-holdout-v1"
PRODUCT_DIRECTORY = "物品图"
BACKGROUND_DIRECTORY = "背景图"
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})
EXPECTED_PRODUCT_COUNT = 30
EXPECTED_BACKGROUND_COUNT = 10


class HoldoutManifestError(RuntimeError):
    """Raised when a frozen holdout no longer matches its source assets."""


class HoldoutLeakageError(HoldoutManifestError):
    """Raised when holdout bytes appear in a training or tuning input."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _raw_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise HoldoutManifestError(f"holdout input directory is missing: {directory}")
    # Deliberately non-recursive: only the declared raw-input directory is in
    # scope.  A generated-output subtree can never be pulled into the manifest.
    return sorted(
        (
            item for item in directory.iterdir()
            if item.is_file() and item.suffix.casefold() in IMAGE_SUFFIXES
        ),
        key=lambda item: item.name.casefold(),
    )


def _asset(source_root: Path, path: Path, role: str, ordinal: int) -> dict[str, Any]:
    stat = path.stat()
    return {
        "asset_id": f"{role}_{ordinal:03d}",
        "role": role,
        "relative_path": path.relative_to(source_root).as_posix(),
        "filename": path.name,
        "size_bytes": stat.st_size,
        "sha256": sha256_file(path),
    }


def build_holdout_manifest(source_root: str | Path) -> dict[str, Any]:
    """Build a deterministic manifest without opening or decoding pixels."""
    root = Path(source_root).resolve()
    products = _raw_images(root / PRODUCT_DIRECTORY)
    backgrounds = _raw_images(root / BACKGROUND_DIRECTORY)
    if len(products) != EXPECTED_PRODUCT_COUNT:
        raise HoldoutManifestError(
            "private holdout must contain exactly 30 raw product images; "
            f"found {len(products)}"
        )
    if len(backgrounds) != EXPECTED_BACKGROUND_COUNT:
        raise HoldoutManifestError(
            "private holdout must contain exactly 10 raw background images; "
            f"found {len(backgrounds)}"
        )

    product_assets = [
        _asset(root, path, "product", index)
        for index, path in enumerate(products, start=1)
    ]
    background_assets = [
        _asset(root, path, "background", index)
        for index, path in enumerate(backgrounds, start=1)
    ]
    cases: list[dict[str, Any]] = []
    for index, background in enumerate(background_assets, start=1):
        start = (index - 1) * 3
        cases.append({
            "case_id": f"set_b_{index:02d}",
            "split": "blind_holdout",
            "background": background,
            "products": product_assets[start : start + 3],
        })

    protected_assets = [
        asset
        for case in cases
        for asset in [case["background"], *case["products"]]
    ]
    dataset_sha256 = _canonical_sha256([
        {
            "relative_path": asset["relative_path"],
            "size_bytes": asset["size_bytes"],
            "sha256": asset["sha256"],
        }
        for asset in protected_assets
    ])
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source_set": "未命名文件夹 3",
        "mapping_rule": (
            "sort raw product filenames case-insensitively, take consecutive "
            "groups of three, and pair them with case-insensitively sorted "
            "raw background filenames"
        ),
        "policy": {
            "evaluation_only": True,
            "allow_training": False,
            "allow_finetuning": False,
            "allow_prompt_tuning": False,
            "allow_parameter_tuning": False,
            "allow_threshold_tuning": False,
            "allow_candidate_selection_across_retries": False,
            "decode_existing_composites": False,
            "generated_assets_in_scope": False,
            "human_visual_review_after_freeze": True,
        },
        "counts": {
            "cases": len(cases),
            "products": len(product_assets),
            "backgrounds": len(background_assets),
        },
        "dataset_sha256": dataset_sha256,
        "cases": cases,
    }
    payload["manifest_sha256"] = _canonical_sha256(payload)
    return payload


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent), text=True,
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


def freeze_holdout_manifest(
    source_root: str | Path, output: str | Path,
) -> dict[str, Any]:
    """Write once; an existing baseline is verified, never silently replaced."""
    output_path = Path(output).resolve()
    current = build_holdout_manifest(source_root)
    if output_path.exists():
        frozen = json.loads(output_path.read_text(encoding="utf-8"))
        validate_holdout_manifest(source_root, frozen)
        if frozen != current:
            raise HoldoutManifestError(
                "existing holdout manifest differs from the deterministic build; "
                "refusing to re-freeze"
            )
        return frozen
    _atomic_json(output_path, current)
    return current


def load_holdout_manifest(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HoldoutManifestError(f"cannot read holdout manifest: {error}") from error
    if not isinstance(payload, dict):
        raise HoldoutManifestError("holdout manifest must be a JSON object")
    return payload


def validate_holdout_manifest(
    source_root: str | Path, manifest: dict[str, Any] | str | Path,
) -> dict[str, Any]:
    """Verify policy, manifest integrity, inventory, mapping and every SHA-256."""
    frozen = load_holdout_manifest(manifest) if isinstance(manifest, (str, Path)) else manifest
    if frozen.get("schema_version") != SCHEMA_VERSION:
        raise HoldoutManifestError("unsupported holdout manifest schema")
    policy = frozen.get("policy") or {}
    forbidden_true = (
        "allow_training", "allow_finetuning", "allow_prompt_tuning",
        "allow_parameter_tuning", "allow_threshold_tuning",
        "allow_candidate_selection_across_retries", "decode_existing_composites",
        "generated_assets_in_scope",
    )
    if not policy.get("evaluation_only") or any(policy.get(key) for key in forbidden_true):
        raise HoldoutManifestError("holdout policy permits training, tuning or generated assets")

    stored_manifest_hash = str(frozen.get("manifest_sha256") or "")
    unsigned = {key: value for key, value in frozen.items() if key != "manifest_sha256"}
    if stored_manifest_hash != _canonical_sha256(unsigned):
        raise HoldoutManifestError("holdout manifest JSON was modified after freezing")

    current = build_holdout_manifest(source_root)
    if frozen != current:
        frozen_cases = frozen.get("cases") or []
        current_cases = current.get("cases") or []
        frozen_assets = {
            asset["relative_path"]: asset
            for case in frozen_cases
            for asset in [case.get("background", {}), *(case.get("products") or [])]
            if asset.get("relative_path")
        }
        current_assets = {
            asset["relative_path"]: asset
            for case in current_cases
            for asset in [case.get("background", {}), *(case.get("products") or [])]
            if asset.get("relative_path")
        }
        added = sorted(set(current_assets) - set(frozen_assets))
        missing = sorted(set(frozen_assets) - set(current_assets))
        changed = sorted(
            path for path in set(frozen_assets) & set(current_assets)
            if (
                frozen_assets[path].get("sha256") != current_assets[path].get("sha256")
                or frozen_assets[path].get("size_bytes") != current_assets[path].get("size_bytes")
            )
        )
        raise HoldoutManifestError(
            "private holdout no longer matches frozen baseline: "
            f"added={added}, missing={missing}, changed={changed}"
        )
    return {
        "verdict": "pass",
        "schema_version": SCHEMA_VERSION,
        "dataset_sha256": current["dataset_sha256"],
        "manifest_sha256": current["manifest_sha256"],
        "counts": current["counts"],
    }


def _candidate_files(paths: Iterable[str | Path]) -> Iterable[Path]:
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            yield from (
                item for item in path.rglob("*")
                if item.is_file() and item.suffix.casefold() in IMAGE_SUFFIXES
            )
        elif path.is_file():
            yield path


def find_holdout_leakage(
    candidate_paths: Iterable[str | Path],
    source_root: str | Path,
    manifest: dict[str, Any] | str | Path,
) -> list[dict[str, str]]:
    """Find same-path or same-byte holdout assets in training/tuning inputs."""
    root = Path(source_root).resolve()
    frozen = load_holdout_manifest(manifest) if isinstance(manifest, (str, Path)) else manifest
    # Never trust a deny-list whose JSON or source bytes have drifted.
    validate_holdout_manifest(root, frozen)
    protected_by_hash: dict[str, str] = {}
    protected_paths: dict[Path, str] = {}
    for case in frozen.get("cases") or []:
        for asset in [case.get("background", {}), *(case.get("products") or [])]:
            relative = str(asset.get("relative_path") or "")
            digest = str(asset.get("sha256") or "")
            if relative and digest:
                protected_paths[(root / Path(relative)).resolve()] = relative
                protected_by_hash[digest] = relative

    collisions: list[dict[str, str]] = []
    seen: set[Path] = set()
    for candidate in _candidate_files(candidate_paths):
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved in protected_paths:
            collisions.append({
                "candidate": str(resolved),
                "holdout_asset": protected_paths[resolved],
                "reason": "same_path",
            })
            continue
        digest = sha256_file(resolved)
        if digest in protected_by_hash:
            collisions.append({
                "candidate": str(resolved),
                "holdout_asset": protected_by_hash[digest],
                "reason": "same_sha256",
            })
    return collisions


def assert_no_holdout_leakage(
    candidate_paths: Iterable[str | Path],
    source_root: str | Path,
    manifest: dict[str, Any] | str | Path,
) -> None:
    collisions = find_holdout_leakage(candidate_paths, source_root, manifest)
    if collisions:
        raise HoldoutLeakageError(
            "holdout assets leaked into training/tuning inputs: "
            + json.dumps(collisions, ensure_ascii=False)
        )


def groups_from_holdout_manifest(
    source_root: str | Path, manifest: dict[str, Any] | str | Path,
) -> list[dict[str, Any]]:
    """Return absolute evaluation inputs only after a full baseline check."""
    root = Path(source_root).resolve()
    frozen = load_holdout_manifest(manifest) if isinstance(manifest, (str, Path)) else manifest
    validate_holdout_manifest(root, frozen)
    return [
        {
            "group_id": str(case["case_id"]),
            "source_set": str(frozen["source_set"]),
            "background": str(root / Path(case["background"]["relative_path"])),
            "products": tuple(
                str(root / Path(asset["relative_path"]))
                for asset in case["products"]
            ),
        }
        for case in frozen["cases"]
    ]
