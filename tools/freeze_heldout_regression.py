"""Freeze, verify or leakage-audit the private cross-scene holdout set."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from strict_compositor.heldout_manifest import (
    HoldoutManifestError,
    find_holdout_leakage,
    freeze_holdout_manifest,
    validate_holdout_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze", help="write the immutable baseline once")
    freeze.add_argument("--source-root", required=True, type=Path)
    freeze.add_argument("--manifest", required=True, type=Path)

    verify = subparsers.add_parser("verify", help="verify files against the baseline")
    verify.add_argument("--source-root", required=True, type=Path)
    verify.add_argument("--manifest", required=True, type=Path)

    audit = subparsers.add_parser("audit", help="reject holdout bytes in tuning inputs")
    audit.add_argument("--source-root", required=True, type=Path)
    audit.add_argument("--manifest", required=True, type=Path)
    audit.add_argument("paths", nargs="+", type=Path)

    args = parser.parse_args()
    try:
        if args.command == "freeze":
            payload = freeze_holdout_manifest(args.source_root, args.manifest)
            result = {
                "verdict": "frozen",
                "manifest": str(args.manifest.resolve()),
                "dataset_sha256": payload["dataset_sha256"],
                "manifest_sha256": payload["manifest_sha256"],
                "counts": payload["counts"],
            }
        elif args.command == "verify":
            result = validate_holdout_manifest(args.source_root, args.manifest)
        else:
            collisions = find_holdout_leakage(
                args.paths, args.source_root, args.manifest,
            )
            result = {
                "verdict": "reject" if collisions else "pass",
                "collisions": collisions,
            }
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 2 if collisions else 0
    except HoldoutManifestError as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
