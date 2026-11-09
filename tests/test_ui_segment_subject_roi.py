from __future__ import annotations

import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from strict_compositor import ui
from strict_compositor.image_io import load_background, save_png
from strict_compositor.minimax_advisor import SubjectROI


def _cloud_cutout(source: str | Path, output: str | Path) -> dict:
    """Write a deterministic opaque RGBA candidate for route-contract tests."""

    rgb = load_background(source)
    height, width = rgb.shape[:2]
    alpha = np.zeros((height, width), dtype=np.uint8)
    alpha[
        int(round(height * 0.10)):int(round(height * 0.86)),
        int(round(width * 0.08)):int(round(width * 0.92)),
    ] = 255
    rgba = np.dstack((rgb, alpha))
    save_png(output, rgba)
    return {
        "model": "runninghub-test",
        "quality": {
            "score": 100.0,
            "verdict": "pass",
            "reason": "route fixture",
        },
    }


def _local_cutout(source: str | Path, output: str | Path, **_kwargs) -> dict:
    """Write the independent local candidate without importing model stacks."""

    rgb = load_background(source)
    height, width = rgb.shape[:2]
    alpha = np.zeros((height, width), dtype=np.uint8)
    alpha[
        int(round(height * 0.10)):int(round(height * 0.86)),
        int(round(width * 0.08)):int(round(width * 0.92)),
    ] = 255
    rgba = np.dstack((rgb, alpha))
    save_png(output, rgba)
    return {
        "selected_model": "local-route-test",
        "semantic_refinement": False,
        "quality": {
            "score": 99.0,
            "verdict": "pass",
            "reason": "route fixture",
        },
    }


def _leaky_cutout(source: str | Path, output: str | Path, **_kwargs) -> dict:
    """Retain almost the complete ROI as visible foreground/background."""

    rgb = load_background(source)
    alpha = np.zeros(rgb.shape[:2], dtype=np.uint8)
    alpha[2:-2, 2:-2] = 255
    save_png(output, np.dstack((rgb, alpha)))
    return {
        "model": "leaky-route-test",
        "selected_model": "leaky-route-test",
        "semantic_refinement": False,
        "quality": {
            "score": 96.0,
            "verdict": "pass",
            "reason": "provider wrongly trusted a broad ROI mask",
        },
    }


class SegmentSubjectROIRouteTests(unittest.TestCase):
    """Contract tests for the semantic single-product crop in ``/api/segment``.

    These tests deliberately exercise the route instead of MiniMax itself.  The
    locator transport/JSON contract lives in ``test_minimax_subject_roi.py``;
    this suite verifies when its result may affect source pixels and how that
    decision is exposed to the browser and later audit code.
    """

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "source.png"
        # A non-square image makes normalized-to-pixel rounding observable.
        Image.new("RGB", (1000, 800), (188, 154, 112)).save(self.source)
        self.located = SubjectROI(
            bbox=(0.34, 0.10, 0.61, 0.55),
            subject_label="decorative ceramic plate on display easel",
            confidence=0.92,
            included_parts=("plate body", "border", "display easel"),
            excluded_objects=("telephone", "vase", "book", "reflection"),
            ambiguous=False,
            model="MiniMax-M3",
            request_id="roi-request-route-test",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _route_patches(self, *, minimax_configured: bool = True):
        return (
            patch.object(ui, "WORK", self.root / "workspace"),
            patch.object(ui, "ADVISOR_CACHE", self.root / "advisor-cache"),
            patch.object(
                ui,
                "minimax_status",
                return_value={"configured": minimax_configured},
            ),
            patch.object(ui, "runninghub_status", return_value={"enabled": True}),
            patch.object(ui, "run_configured_cutout", side_effect=_cloud_cutout),
            patch.object(ui, "ensemble_rembg_rgba", side_effect=_local_cutout),
        )

    def _segment_with_locator(
        self,
        payload: dict,
        *,
        minimax_configured: bool = True,
    ):
        with ExitStack() as stack:
            for route_patch in self._route_patches(
                minimax_configured=minimax_configured,
            ):
                stack.enter_context(route_patch)
            locate = stack.enter_context(patch.object(
                ui,
                "request_minimax_subject_roi",
                return_value=self.located,
                create=True,
            ))
            result = ui.segment(payload)
        return result, locate

    def test_quality_mode_auto_roi_adds_asymmetric_safety_padding(self) -> None:
        """A tight semantic box must retain thin handles, feet and the base.

        The production contract is deterministic normalized padding relative
        to the located box: 8% of box width on both sides, 7% of box height at
        the top, and 16% at the bottom.  For this 1000x800 fixture the expected
        crop is [318, 54, 632, 498] after floor/ceil conversion.
        """

        result, locate = self._segment_with_locator({
            "source": str(self.source),
            "id": "product_1",
            "quality": "quality",
        })

        locate.assert_called_once_with(
            self.source.resolve(),
            (self.root / "advisor-cache" / "subject_roi"),
            allow_fallback=True,
        )
        self.assertEqual(result["selection_mode"], "auto_subject_box")
        self.assertEqual(result["selection_rect"], [318, 54, 632, 498])
        self.assertEqual(
            result["selection_strategy"],
            "auto_subject_runninghub_alpha_source_rgb",
        )
        locator = result["subject_locator"]
        self.assertEqual(locator["bbox"], [0.34, 0.10, 0.61, 0.55])
        np.testing.assert_allclose(
            locator["expanded_bbox"],
            [0.3184, 0.0685, 0.6316, 0.622],
            rtol=0.0,
            atol=1e-12,
        )
        self.assertTrue(locator["accepted"])
        self.assertTrue(locator["semantic_verified"])
        self.assertEqual(locator["role"], "verified_subject_roi")
        self.assertEqual(locator["subject_label"], self.located.subject_label)
        self.assertEqual(locator["request_id"], "roi-request-route-test")
        self.assertFalse(locator["fallback_used"])
        self.assertEqual(result["attempts"][0]["provider"], "minimax_subject_locator")
        self.assertEqual(result["attempts"][0]["status"], "success")
        self.assertEqual(
            [item["provider"] for item in result["candidate_artifacts"]],
            ["runninghub", "local"],
        )
        self.assertEqual(result["provider"], "runninghub")
        self.assertTrue(result["cutout_ready"])

    def test_verified_subject_gate_rejects_broad_visible_roi_background(self) -> None:
        """A provider score cannot override a verified subject-envelope leak."""

        with ExitStack() as stack:
            stack.enter_context(patch.object(ui, "WORK", self.root / "workspace"))
            stack.enter_context(patch.object(
                ui, "ADVISOR_CACHE", self.root / "advisor-cache",
            ))
            stack.enter_context(patch.object(
                ui, "runninghub_status", return_value={"enabled": True},
            ))
            stack.enter_context(patch.object(
                ui, "run_configured_cutout", side_effect=_leaky_cutout,
            ))
            stack.enter_context(patch.object(
                ui, "ensemble_rembg_rgba", side_effect=_leaky_cutout,
            ))
            stack.enter_context(patch.object(
                ui,
                "request_minimax_subject_roi",
                return_value=self.located,
                create=True,
            ))
            result = ui.segment({
                "source": str(self.source),
                "id": "product_leaky",
                "quality": "quality",
            })

        quality = result["alpha_quality"]
        self.assertFalse(result["cutout_ready"])
        self.assertEqual(quality["verdict"], "reject")
        self.assertLessEqual(quality["score"], 35.0)
        self.assertTrue(quality["semantic_envelope_leakage"])
        envelope = quality["semantic_alpha_envelope"]
        self.assertTrue(envelope["broad_on_both_axes"])
        self.assertGreater(envelope["span_excess"][0], 0.10)
        self.assertGreater(envelope["span_excess"][1], 0.10)
        self.assertTrue(all(
            artifact["quality"]["semantic_envelope_leakage"]
            for artifact in result["candidate_artifacts"]
        ))

    def test_semantic_gate_ignores_rgb_beneath_zero_alpha(self) -> None:
        """Preserved source RGB in transparent pixels is not visible leakage."""

        crop_shape = (800, 1000)
        rgba = np.zeros((800, 1000, 4), dtype=np.uint8)
        # Simulate a fully preserved wooden source crop beneath alpha.
        rgba[..., :3] = [105, 72, 44]
        rgba[80:680, 90:910, 3] = 255
        locator = {
            **self.located.to_dict(),
            "bbox": [0.34, 0.10, 0.61, 0.55],
            "expanded_bbox": [0.3184, 0.0685, 0.6316, 0.622],
            "semantic_verified": True,
        }

        quality = ui._semantic_alpha_envelope_gate(
            {"score": 96.0, "verdict": "pass"},
            rgba,
            crop_shape,
            locator,
        )

        self.assertFalse(quality["semantic_envelope_leakage"])
        self.assertEqual(quality["verdict"], "pass")

    def test_semantic_gate_rejects_dense_one_axis_background_strip(self) -> None:
        """A side-attached shelf strip cannot hide behind a 90+ alpha score."""

        crop_shape = (1000, 1000)
        rgba = np.zeros((1000, 1000, 4), dtype=np.uint8)
        rgba[:942, :824, :3] = [148, 111, 73]
        rgba[:942, :824, 3] = 255
        locator = {
            **self.located.to_dict(),
            "bbox": [0.069, 0.0865, 0.931, 0.9135],
            "expanded_bbox": [0.0, 0.0, 1.0, 1.0],
            "semantic_verified": True,
        }

        quality = ui._semantic_alpha_envelope_gate(
            {"score": 96.0, "verdict": "pass"},
            rgba,
            crop_shape,
            locator,
        )

        self.assertTrue(quality["semantic_envelope_leakage"])
        self.assertTrue(
            quality["semantic_alpha_envelope"]["asymmetric_axis_overflow"]
        )
        self.assertEqual(quality["verdict"], "reject")
        self.assertLessEqual(quality["score"], 35.0)

    def test_operator_roi_has_priority_and_never_invokes_auto_locator(self) -> None:
        result, locate = self._segment_with_locator({
            "source": str(self.source),
            "id": "product_2",
            "quality": "quality",
            "roi": [0.10, 0.20, 0.40, 0.70],
        })

        locate.assert_not_called()
        self.assertEqual(result["selection_mode"], "box")
        self.assertEqual(result["selection_rect"], [100, 160, 400, 560])
        self.assertEqual(result["selection_strategy"], "cropped_runninghub_alpha_source_rgb")
        self.assertIsNone(result.get("subject_locator"))

    def test_fast_mode_without_operator_roi_does_not_spend_minimax_call(self) -> None:
        result, locate = self._segment_with_locator({
            "source": str(self.source),
            "id": "product_3",
            "quality": "fast",
        })

        locate.assert_not_called()
        self.assertEqual(result["selection_mode"], "whole_image")
        self.assertIsNone(result["selection_rect"])
        self.assertIsNone(result.get("subject_locator"))

if __name__ == "__main__":
    unittest.main()
