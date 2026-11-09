from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from strict_compositor.human_visual_audit import (
    HumanVisualAuditError,
    OPTIONAL_DIMENSIONS,
    REQUIRED_DIMENSIONS,
    pending_human_visual_audit,
    validate_human_visual_audit,
)


class HumanVisualAuditTests(unittest.TestCase):
    def _candidate(self, root: Path) -> Path:
        path = root / "candidate.png"
        Image.new("RGB", (16, 12), (20, 30, 40)).save(path)
        return path

    def _review(self, candidate: Path, verdict: str = "pass") -> dict:
        review = pending_human_visual_audit(candidate)
        review.update({
            "status": "reviewed",
            "verdict": verdict,
            "reviewer": "codex_visual_review",
            "reviewed_at": "2026-08-07T14:00:00+08:00",
        })
        for name in REQUIRED_DIMENSIONS:
            review["dimensions"][name] = {
                "verdict": verdict,
                "observation": f"human inspected {name}",
            }
        for name in OPTIONAL_DIMENSIONS:
            review["dimensions"][name] = {
                "verdict": "not_applicable",
                "observation": "not present in this case",
            }
        return review

    def test_complete_human_review_passes_without_machine_score(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            candidate = self._candidate(Path(temp))
            validated = validate_human_visual_audit(self._review(candidate))
            self.assertEqual(validated["verdict"], "pass")
            self.assertTrue(validated["machine_scores_are_diagnostic_only"])

    def test_single_visual_failure_forces_reject(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            candidate = self._candidate(Path(temp))
            review = self._review(candidate, verdict="reject")
            for name in REQUIRED_DIMENSIONS:
                review["dimensions"][name]["verdict"] = "pass"
            review["dimensions"]["relative_scale"] = {
                "verdict": "reject",
                "observation": "one product reads as a miniature",
            }
            validated = validate_human_visual_audit(review)
            self.assertEqual(validated["verdict"], "reject")

    def test_rejects_score_only_or_incomplete_review(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            candidate = self._candidate(Path(temp))
            review = pending_human_visual_audit(candidate, machine_diagnostics={"visual_score": 100})
            review.update({
                "status": "reviewed",
                "verdict": "pass",
                "reviewer": "automatic_score",
            })
            with self.assertRaises(HumanVisualAuditError):
                validate_human_visual_audit(review)

    def test_review_is_invalid_after_candidate_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            candidate = self._candidate(Path(temp))
            review = self._review(candidate)
            Image.new("RGB", (16, 12), (99, 88, 77)).save(candidate)
            with self.assertRaisesRegex(HumanVisualAuditError, "different candidate"):
                validate_human_visual_audit(review)


if __name__ == "__main__":
    unittest.main()
