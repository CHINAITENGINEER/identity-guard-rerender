from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image, ImageDraw

from strict_compositor.errors import ContractError
from strict_compositor.minimax_advisor import (
    SUBJECT_ROI_PROMPT_VERSION,
    MiniMaxNotConfigured,
    _subject_roi_cache_key,
    request_minimax_subject_roi,
    validate_subject_roi,
)


def valid_subject_payload() -> dict:
    return {
        "subject_bbox": [0.22, 0.08, 0.79, 0.93],
        "subject_label": "ribbed glass pitcher",
        "confidence": 0.94,
        "ambiguous": False,
        "complete_subject": True,
        "included_parts": ["body", "spout", "handle", "base"],
        "excluded_objects": ["tabletop", "plant", "reflection"],
        "warnings": [],
    }


def response_for(payload: dict) -> Mock:
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "id": "roi-request-1",
        "model": "MiniMax-M3",
        "choices": [{"message": {"content": json.dumps(payload)}}],
    }
    return response


class MiniMaxSubjectROITests(unittest.TestCase):
    def test_contract_keeps_complete_attached_parts_and_exclusions(self) -> None:
        result = validate_subject_roi(valid_subject_payload())
        self.assertEqual(result.bbox, (0.22, 0.08, 0.79, 0.93))
        self.assertIn("handle", result.included_parts)
        self.assertIn("reflection", result.excluded_objects)
        self.assertFalse(result.fallback_used)
        self.assertEqual(result.source, "minimax")

    def test_contract_rejects_incomplete_or_malformed_box(self) -> None:
        incomplete = valid_subject_payload()
        incomplete["complete_subject"] = False
        with self.assertRaises(ContractError):
            validate_subject_roi(incomplete)
        reversed_box = valid_subject_payload()
        reversed_box["subject_bbox"] = [0.8, 0.1, 0.2, 0.9]
        with self.assertRaises(ContractError):
            validate_subject_roi(reversed_box)

    def test_request_uses_existing_relay_contract_and_caches_valid_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.png"
            cache = root / "cache"
            Image.new("RGB", (640, 480), "white").save(source)
            response = response_for(valid_subject_payload())
            with (
                patch.dict("os.environ", {}, clear=True),
                patch(
                    "strict_compositor.minimax_advisor.get_secret",
                    return_value="unit-test-key",
                ),
                patch(
                    "strict_compositor.minimax_advisor.get_public_settings",
                    return_value={
                        "minimax_base_url": "https://relay.example/v1",
                        "minimax_model": "MiniMax-M3",
                        "minimax_network_mode": "direct",
                    },
                ),
                patch(
                    "strict_compositor.minimax_advisor._post_minimax",
                    return_value=(response, "direct"),
                ) as post,
            ):
                first = request_minimax_subject_roi(source, cache)
                second = request_minimax_subject_roi(source, cache)

            self.assertEqual(post.call_count, 1)
            self.assertEqual(first, second)
            self.assertEqual(first.request_id, "roi-request-1")
            sent = post.call_args.kwargs["payload"]
            self.assertEqual(sent["temperature"], 0.0)
            content = sent["messages"][0]["content"]
            prompt = content[0]["text"]
            self.assertIn("handle, spout, lid, cap, base", prompt)
            self.assertIn("reflection or shadow is never", prompt)
            cached = list(cache.glob("subject_roi_*.json"))
            self.assertEqual(len(cached), 1)
            cache_payload = json.loads(cached[0].read_text(encoding="utf-8"))
            self.assertEqual(cache_payload["prompt_version"], SUBJECT_ROI_PROMPT_VERSION)

    def test_invalid_model_contract_degrades_but_is_not_cached(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.png"
            cache = root / "cache"
            image = Image.new("RGB", (500, 420), "white")
            ImageDraw.Draw(image).rounded_rectangle((170, 60, 345, 375), fill=(50, 80, 130))
            image.save(source)
            invalid = valid_subject_payload()
            invalid["complete_subject"] = False
            with (
                patch.dict("os.environ", {}, clear=True),
                patch(
                    "strict_compositor.minimax_advisor.get_secret",
                    return_value="unit-test-key",
                ),
                patch(
                    "strict_compositor.minimax_advisor.get_public_settings",
                    return_value={
                        "minimax_base_url": "https://relay.example/v1",
                        "minimax_model": "MiniMax-M3",
                    },
                ),
                patch(
                    "strict_compositor.minimax_advisor._post_minimax",
                    return_value=(response_for(invalid), "direct"),
                ),
            ):
                result = request_minimax_subject_roi(source, cache)
            self.assertTrue(result.fallback_used)
            self.assertTrue(result.ambiguous)
            self.assertLess(result.confidence, 0.45)
            self.assertEqual(list(cache.glob("subject_roi_*.json")), [])

    def test_missing_key_has_explicit_fallback_and_strict_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.png"
            Image.new("RGB", (320, 320), (230, 230, 230)).save(source)
            with (
                patch.dict("os.environ", {}, clear=True),
                patch("strict_compositor.minimax_advisor.get_secret", return_value=""),
                patch(
                    "strict_compositor.minimax_advisor.get_public_settings",
                    return_value={"minimax_model": "MiniMax-M3"},
                ),
            ):
                fallback = request_minimax_subject_roi(source)
                with self.assertRaises(MiniMaxNotConfigured):
                    request_minimax_subject_roi(source, allow_fallback=False)
            self.assertTrue(fallback.fallback_used)
            self.assertEqual(fallback.source, "local_center_saliency")
            self.assertIn("not configured", fallback.warnings[0])

    def test_roi_cache_key_isolated_by_provider(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.png"
            source.write_bytes(b"same-source")
            first = _subject_roi_cache_key(
                source, "MiniMax-M3", "https://relay-a.example/v1",
            )
            equivalent = _subject_roi_cache_key(
                source, "MiniMax-M3", "HTTPS://RELAY-A.EXAMPLE/v1/",
            )
            second = _subject_roi_cache_key(
                source, "MiniMax-M3", "https://relay-b.example/v1",
            )
            self.assertEqual(first, equivalent)
            self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
