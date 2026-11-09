from __future__ import annotations

import argparse
import json
from pathlib import Path

from strict_compositor.human_visual_audit import (
    pending_human_visual_audit,
    reviewed_at_now,
    validate_human_visual_audit,
    write_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bind a categorical human visual review to one candidate image.",
    )
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--verdict", required=True, choices=("pass", "reject"))
    args = parser.parse_args()

    observations = json.loads(args.observations.read_text(encoding="utf-8"))
    if not isinstance(observations, dict):
        raise ValueError("observations JSON must be an object keyed by audit dimension")
    payload = pending_human_visual_audit(args.candidate)
    payload.update({
        "status": "reviewed",
        "verdict": args.verdict,
        "reviewer": args.reviewer,
        "reviewed_at": reviewed_at_now(),
        "dimensions": observations,
    })
    validated = validate_human_visual_audit(payload, candidate_path=args.candidate)
    write_audit(args.output, validated)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "candidate_sha256": validated["candidate_sha256"],
        "verdict": validated["verdict"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
