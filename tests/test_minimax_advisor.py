from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests
import numpy as np
from PIL import Image, ImageDraw

from strict_compositor import settings
from strict_compositor.errors import ContractError
from strict_compositor.minimax_advisor import (
    DEFAULT_MODEL,
    VISUAL_REVIEW_PROMPT_VERSION,
    MiniMaxAdvice,
    MiniMaxNotConfigured,
    MiniMaxTransportError,
    ProductAdvice,
    _post_minimax,
    _cache_key,
    _image_data_url,
    _visual_review_detail_crops,
    _visual_review_evidence_boards,
    _clear_glass_intrinsic_geometry_image,
    _visual_review_material_contract,
    _visual_review_material_manifest,
    apply_layout_advice,
    extract_json_object,
    reconcile_scene_result,
    request_minimax_advice,
    request_minimax_visual_review,
    validate_advice,
    validate_visual_review,
)


def valid_payload() -> dict:
    return {
        "support_polygon": [[0.08, 0.2], [0.92, 0.2], [0.92, 0.75], [0.08, 0.75]],
        "protected_regions": [
            {
                "label": "telephone",
                "polygon": [[0.1, 0.3], [0.25, 0.3], [0.25, 0.5], [0.1, 0.5]],
            }
        ],
        "lighting": {
            "light_direction": [-0.7, 0.7],
            "shadow_direction": [0.7, -0.7],
        },
        "surface": {
            "reflectivity": 0.18,
            "reflection_recommended": False,
        },
        "products": [
            {
                "product_id": "product_1",
                "center_x": 0.3,
                "bottom_y": 0.62,
                "height_ratio": 0.15,
                "z_order": 1,
                "footprint_width": 0.08,
                "source_light_compatibility": 0.81,
                "placement_mode": "surface_flat",
                "physical_type": "soft_pillow",
            }
        ],
        "route": "strict_composite",
        "confidence": 0.88,
        "warnings": [],
    }


class MiniMaxAdvisorTests(unittest.TestCase):
    def test_visual_review_manifest_merges_id_bound_ui_semantics(self) -> None:
        manifest = _visual_review_material_manifest({
            "products": [
                {
                    "id": "product_1",
                    "material_mode": "clear_glass",
                    "material_confidence": 0.91,
                    "transmission_strength": 0.97,
                    # Historical low-level jobs may contain generic defaults.
                    "semantic_category": "decor_object",
                    "physical_type": "rigid_object",
                },
                {
                    "id": "product_2",
                    "material_mode": "opaque",
                    "material_confidence": 0.96,
                    "transmission_strength": 0.0,
                },
            ],
            # Deliberately reversed: metadata must join by product ID, never index.
            "ui_placements": [
                {
                    "id": "product_2",
                    "semantic_category": "aluminum_beverage_can",
                    "physical_type": "rigid_vessel",
                    "material_mode": "mixed_transparent",
                },
                {
                    "id": "product_1",
                    "semantic_category": "glass_pitcher",
                    "physical_type": "rigid_vessel",
                    "material_mode": "opaque",
                },
            ],
        }, ["product_1", "product_2"])

        self.assertEqual(manifest[0]["material_mode"], "clear_glass")
        self.assertEqual(manifest[0]["material_confidence"], 0.91)
        self.assertEqual(manifest[0]["semantic_category"], "glass_pitcher")
        self.assertEqual(manifest[0]["physical_type"], "rigid_vessel")
        self.assertIn(
            "same-coordinate target-backdrop contours and colours",
            manifest[0]["transmission_scope"],
        )
        self.assertEqual(manifest[1]["material_mode"], "opaque")
        self.assertEqual(
            manifest[1]["semantic_category"], "aluminum_beverage_can",
        )
        contract = _visual_review_material_contract(manifest)
        self.assertIn("semantic_category=glass_pitcher", contract)
        self.assertIn("physical_type=rigid_vessel", contract)
        self.assertIn("never infer that a clear_glass product is amber", contract)

    def test_auto_network_mode_falls_back_to_same_url_without_system_proxy(self) -> None:
        response = Mock()
        direct_session = Mock()
        direct_session.post.return_value = response
        with (
            patch(
                "strict_compositor.minimax_advisor.requests.post",
                side_effect=requests.ConnectionError("proxy closed upstream"),
            ) as system_post,
            patch(
                "strict_compositor.minimax_advisor.requests.Session",
                return_value=direct_session,
            ) as session_factory,
        ):
            received, route = _post_minimax(
                "https://relay.example/v1/chat/completions",
                headers={"Authorization": "Bearer unit-test-key"},
                payload={"model": "MiniMax-M3"},
                timeout=(1.0, 2.0),
                settings={"minimax_network_mode": "auto"},
            )
        self.assertIs(received, response)
        self.assertEqual(route, "direct")
        self.assertEqual(system_post.call_count, 1)
        self.assertEqual(session_factory.call_count, 1)
        self.assertFalse(direct_session.trust_env)
        self.assertEqual(
            direct_session.post.call_args.args[0],
            "https://relay.example/v1/chat/completions",
        )

    def test_http_response_never_triggers_direct_route_or_provider_switch(self) -> None:
        response = Mock(status_code=401)
        with (
            patch(
                "strict_compositor.minimax_advisor.requests.post",
                return_value=response,
            ) as system_post,
            patch("strict_compositor.minimax_advisor.requests.Session") as session_factory,
        ):
            received, route = _post_minimax(
                "https://relay.example/v1/chat/completions",
                headers={"Authorization": "Bearer unit-test-key"},
                payload={"model": "MiniMax-M3"},
                timeout=(1.0, 2.0),
                settings={"minimax_network_mode": "auto"},
            )
        self.assertIs(received, response)
        self.assertEqual(route, "system_proxy")
        self.assertEqual(system_post.call_count, 1)
        session_factory.assert_not_called()

    def test_all_network_routes_fail_as_transport_error(self) -> None:
        direct_session = Mock()
        direct_session.post.side_effect = requests.ConnectionError("direct refused")
        with (
            patch(
                "strict_compositor.minimax_advisor.requests.post",
                side_effect=requests.ConnectionError("proxy refused"),
            ),
            patch(
                "strict_compositor.minimax_advisor.requests.Session",
                return_value=direct_session,
            ),
        ):
            with self.assertRaises(MiniMaxTransportError) as context:
                _post_minimax(
                    "https://relay.example/v1/chat/completions",
                    headers={"Authorization": "Bearer unit-test-key"},
                    payload={"model": "MiniMax-M3"},
                    timeout=(1.0, 2.0),
                    settings={"minimax_network_mode": "auto"},
                )
        self.assertEqual(context.exception.attempted_routes, ("system_proxy", "direct"))

    def test_common_material_aliases_are_normalized_without_dropping_advice(self) -> None:
        payload = valid_payload()
        payload["surface"]["material_class"] = "clear_glass"
        payload["lighting"]["cast_shadow_mode"] = "contact-only"
        payload["products"][0]["material_mode"] = "transparent glass"
        result = validate_advice(payload, ["product_1"])
        self.assertEqual(result.surface_material, "transparent_glass")
        self.assertEqual(result.cast_shadow_mode, "ambient_only")
        self.assertEqual(result.products[0].material_mode, "clear_glass")

    def test_descriptive_compound_material_aliases_remain_constrained(self) -> None:
        payload = valid_payload()
        payload["surface"]["material_class"] = "highly reflective glass tabletop"
        payload["products"][0]["material_mode"] = "clear transparent glass vessel"
        result = validate_advice(payload, ["product_1"])
        self.assertEqual(result.surface_material, "transparent_glass")
        self.assertEqual(result.products[0].material_mode, "clear_glass")

    def test_source_room_band_does_not_turn_neutral_pitcher_into_colored_glass(self) -> None:
        payload = valid_payload()
        product = payload["products"][0]
        product["semantic_category"] = "ribbed_glass_pitcher"
        product["physical_type"] = "clear handled vessel"
        product["material_mode"] = "colored_glass"
        product["transmission_strength"] = 0.32
        result = validate_advice(payload, ["product_1"])
        self.assertEqual(result.products[0].material_mode, "clear_glass")
        self.assertGreaterEqual(result.products[0].transmission_strength, 0.97)

    def test_warm_source_descriptor_does_not_turn_open_clear_pitcher_amber(self) -> None:
        payload = valid_payload()
        product = payload["products"][0]
        # A relay previously emitted this exact category for a colourless
        # pitcher because it saw a warm room through the glass.  ``amber`` by
        # itself is not intrinsic-tint evidence for an open vessel.
        product["semantic_category"] = "amber_glass_pitcher"
        product["physical_type"] = "ribbed handled vessel"
        product["material_mode"] = "colored_glass"
        product["transmission_strength"] = 0.58
        result = validate_advice(payload, ["product_1"])
        self.assertEqual(result.products[0].material_mode, "clear_glass")
        self.assertGreaterEqual(result.products[0].transmission_strength, 0.97)

    def test_generic_tinted_descriptor_does_not_override_open_clear_pitcher(self) -> None:
        payload = valid_payload()
        product = payload["products"][0]
        # This is the failure mode observed in the real frontend run: the
        # model named the warm source appearance "tinted glass pitcher" even
        # though the open vessel itself was colourless.
        product["semantic_category"] = "tinted_glass_pitcher"
        product["physical_type"] = "ribbed handled vessel"
        product["material_mode"] = "colored_glass"
        product["transmission_strength"] = 0.55
        result = validate_advice(payload, ["product_1"])
        self.assertEqual(result.products[0].material_mode, "clear_glass")
        self.assertGreaterEqual(result.products[0].transmission_strength, 0.97)

    def test_verified_clear_pitcher_gets_target_dominant_transmission(self) -> None:
        payload = valid_payload()
        product = payload["products"][0]
        product["semantic_category"] = "clear_glass_pitcher"
        product["physical_type"] = "ribbed handled vessel"
        product["material_mode"] = "clear_glass"
        product["transmission_strength"] = 0.60
        result = validate_advice(payload, ["product_1"])
        self.assertGreaterEqual(result.products[0].transmission_strength, 0.97)

    def test_labelled_liquor_bottle_is_mixed_transparent(self) -> None:
        payload = valid_payload()
        product = payload["products"][0]
        product["semantic_category"] = "whiskey_liquor_bottle"
        product["physical_type"] = "labelled glass bottle with liquid"
        product["material_mode"] = "colored_glass"
        product["transmission_strength"] = 0.25
        result = validate_advice(payload, ["product_1"])
        self.assertEqual(result.products[0].material_mode, "mixed_transparent")
        self.assertGreaterEqual(result.products[0].transmission_strength, 0.42)

    @staticmethod
    def _response(payload: dict, *, score: bool = False) -> Mock:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "id": "request-unit-test",
            "model": "MiniMax-M3",
            "choices": [{
                "message": {
                    "content": json.dumps(payload),
                },
            }],
        }
        return response

    def test_default_model_is_m3_everywhere(self) -> None:
        self.assertEqual(DEFAULT_MODEL, "MiniMax-M3")
        self.assertEqual(settings.PUBLIC_DEFAULTS["minimax_model"], "MiniMax-M3")

    def test_scene_advice_prefers_final_rgba_over_source_photo(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            background = root / "background.jpg"
            source = root / "source.jpg"
            rgba = root / "final_rgba.png"
            for path in (background, source, rgba):
                path.write_bytes(b"unit-test")

            encoded_paths: list[Path] = []

            def encode(path: str | Path, *args, **kwargs) -> str:
                encoded_paths.append(Path(path))
                return f"data:image/test;base64,{Path(path).name}"

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
                    "strict_compositor.minimax_advisor._image_data_url",
                    side_effect=encode,
                ),
                patch(
                    "strict_compositor.minimax_advisor.requests.post",
                    return_value=self._response(valid_payload()),
                ),
            ):
                request_minimax_advice(
                    background,
                    [{"id": "product_1", "source": source, "rgba": rgba}],
                )

            self.assertEqual(encoded_paths, [background.resolve(), rgba.resolve()])
            self.assertNotIn(source.resolve(), encoded_paths)

    def test_cache_key_isolated_by_normalized_provider_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            background = root / "background.png"
            product = root / "product.png"
            background.write_bytes(b"background")
            product.write_bytes(b"product")
            first = _cache_key(
                background,
                [product],
                "MiniMax-M3",
                "HTTPS://RELAY-A.EXAMPLE/v1/",
            )
            equivalent = _cache_key(
                background,
                [product],
                "MiniMax-M3",
                "https://relay-a.example/v1",
            )
            other_relay = _cache_key(
                background,
                [product],
                "MiniMax-M3",
                "https://relay-b.example/v1",
            )
            self.assertEqual(first, equivalent)
            self.assertNotEqual(first, other_relay)

    def test_independent_retry_uses_a_distinct_cache_key(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            background = Path(temporary) / "background.png"
            background.write_bytes(b"background")
            first = _cache_key(background, [], "MiniMax-M3", attempt=0)
            verification = _cache_key(background, [], "MiniMax-M3", attempt=1)
            self.assertNotEqual(first, verification)

    def test_visual_detail_crops_and_evidence_are_lossless_png(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            layers = run / "layers"
            layers.mkdir()
            final = run / "final.png"
            Image.new("RGB", (320, 240), (188, 156, 112)).save(final)
            alpha = Image.new("L", (320, 240), 0)
            ImageDraw.Draw(alpha).rectangle((110, 60, 210, 185), fill=255)
            alpha.save(layers / "product_1_alpha.png")

            crops = _visual_review_detail_crops(final, 1)
            self.assertEqual(len(crops), 1)
            self.assertEqual(set(crops), {"product_1"})
            self.assertEqual(crops["product_1"].suffix.lower(), ".png")
            with Image.open(crops["product_1"]) as crop:
                self.assertEqual(crop.format, "PNG")
            self.assertTrue(
                _image_data_url(crops["product_1"], lossless=True).startswith(
                    "data:image/png;base64,"
                )
            )

    def test_missing_detail_crop_does_not_renumber_later_product(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            layers = run / "layers"
            layers.mkdir()
            final = run / "final.png"
            Image.new("RGB", (320, 240), (188, 156, 112)).save(final)
            alpha = Image.new("L", (320, 240), 0)
            ImageDraw.Draw(alpha).rectangle((110, 60, 210, 185), fill=255)
            alpha.save(layers / "product_2_alpha.png")

            crops = _visual_review_detail_crops(final, 2)

            self.assertEqual(set(crops), {"product_2"})
            self.assertEqual(crops["product_2"].name, "product_2_final_detail.png")

    def test_material_focus_uses_raw_registered_target_without_cyan_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            layers = run / "layers"
            layers.mkdir()
            background = run / "background.png"
            final = run / "final.png"
            product = run / "product.png"

            # A visible receiver boundary gives the two registered panels a
            # scene landmark whose exact pixel correspondence can be tested.
            scene = Image.new("RGB", (320, 240), (226, 214, 184))
            draw = ImageDraw.Draw(scene)
            draw.rectangle((0, 134, 319, 239), fill=(37, 42, 45))
            scene.save(background)
            scene.save(final)
            Image.new("RGBA", (80, 140), (210, 220, 226, 105)).save(product)

            alpha = Image.new("L", (320, 240), 0)
            ImageDraw.Draw(alpha).rectangle((118, 50, 202, 198), fill=255)
            alpha.save(layers / "product_1_alpha.png")
            detail_paths = _visual_review_detail_crops(final, 1)

            _visual_review_evidence_boards(
                background,
                final,
                [product],
                detail_paths,
                None,
                [{"product_id": "product_1", "material_mode": "clear_glass"}],
            )

            focus_path = run / "review_crops" / "product_1_material_focus.png"
            with Image.open(focus_path) as focus_source:
                focus = focus_source.convert("RGB")
                # Panels are 1040px wide. Below the 46px labels, the raw target
                # and final crops must be byte-identical when final==background.
                target_pixels = focus.crop((1040, 46, 2080, 900)).tobytes()
                final_pixels = focus.crop((2080, 46, 3120, 900)).tobytes()
                self.assertEqual(target_pixels, final_pixels)
                pixels = np.asarray(focus)
                self.assertFalse(np.any(np.all(
                    pixels == np.asarray([40, 220, 205], np.uint8), axis=2,
                )))

    def test_clear_glass_identity_evidence_keeps_ribs_not_old_room_band(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "clear.png"
            rgba = np.zeros((180, 140, 4), np.uint8)
            rgba[12:168, 24:116, :3] = [184, 164, 126]
            rgba[12:168, 24:116, 3] = 255
            # Old room/table band: broad and horizontal, therefore non-identity.
            rgba[92:132, 24:116, :3] = [12, 10, 8]
            # Intrinsic fluting: narrow and vertical.
            for x in (42, 58, 76, 94):
                rgba[18:160, x:x + 3, :3] = [240, 242, 245]
            Image.fromarray(rgba, mode="RGBA").save(path)

            evidence = np.asarray(_clear_glass_intrinsic_geometry_image(path))
            clear_upper = float(evidence[58, 68].mean())
            old_band = float(evidence[112, 68].mean())
            rib = float(evidence[58, 58].mean())
            self.assertLess(abs(clear_upper - old_band), 14.0)
            self.assertGreater(abs(rib - clear_upper), 18.0)

    def test_visual_review_binds_lossless_source_and_detail_by_product_id(self) -> None:
        visual_payload = {
            "subscores": {
                "grounding": 90,
                "cast_shadows": 90,
                "geometry_layout": 90,
                "lighting_color": 90,
                "cutout_edges": 90,
                "reflection_material": 90,
            },
            "critical_failures": [],
            "issues": [],
            "recommended_adjustments": [],
        }
        encoded: list[tuple[Path, bool]] = []

        def encode(
            path: str | Path,
            *args,
            lossless: bool = False,
            **kwargs,
        ) -> str:
            encoded.append((Path(path), lossless))
            return "data:image/test;base64,unit-test"

        detail = Path("detail.png").resolve()
        context_board = Path("context-board.png").resolve()
        detail_board = Path("detail-board.png").resolve()
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
                "strict_compositor.minimax_advisor._visual_review_detail_crops",
                return_value={"product_2": detail},
            ),
            patch(
                "strict_compositor.minimax_advisor._visual_review_evidence_boards",
                return_value=(context_board, detail_board),
            ),
            patch(
                "strict_compositor.minimax_advisor._image_data_url",
                side_effect=encode,
            ),
            patch(
                "strict_compositor.minimax_advisor.requests.post",
                return_value=self._response(visual_payload, score=True),
            ) as post,
        ):
            review = request_minimax_visual_review(
                "background.png",
                "final.png",
                ["product-1.png", "product-2.png"],
                slot_id=3,
            )

        self.assertEqual(encoded, [
            (context_board, True),
            (detail_board, True),
        ])
        self.assertEqual(
            [lossless for _, lossless in encoded],
            [True, True],
        )
        request_content = post.call_args.kwargs["json"]["messages"][0]["content"]
        labels = [
            item["text"]
            for item in request_content
            if item.get("type") == "text" and item["text"].endswith(" follows.")
        ]
        self.assertEqual(labels, [
            "CONTEXT_EVIDENCE_BOARD follows.",
            "ID_BOUND_DETAIL_EVIDENCE_BOARD follows.",
        ])
        prompt_text = request_content[0]["text"]
        self.assertIn("Product IDs come only from the labels burned into the board", prompt_text)
        self.assertIn("NO cyan coverage line, NO product outline and NO image overlay", prompt_text)
        self.assertIn("do not call a target tabletop edge a retained black source band", prompt_text)
        self.assertIn("Pale, white, translucent, frosted or reflective material", prompt_text)
        self.assertIn("must not excuse a real new fringe", prompt_text)
        self.assertEqual(post.call_args.kwargs["json"]["temperature"], 0)
        self.assertEqual(review["slot_id"], 3)
        self.assertEqual(review["visual_score"], 90)
        self.assertFalse(review["batch_unlock"])
        self.assertEqual(review["raw_model_result"], visual_payload)
        self.assertEqual(
            review["evidence_policy"],
            "two_lossless_boards_plus_per_product_raw_same_coordinate_target_backdrop_reference",
        )
        self.assertEqual(
            review["prompt_version"],
            VISUAL_REVIEW_PROMPT_VERSION,
        )
        self.assertEqual(
            review["evidence_mapping"],
            [
                {
                    "product_id": "product_1",
                    "canonical_label": "CANONICAL_PRODUCT product_id=product_1",
                    "target_label": None,
                    "detail_label": None,
                    "detail_available": False,
                },
                {
                    "product_id": "product_2",
                    "canonical_label": "CANONICAL_PRODUCT product_id=product_2",
                    "target_label": (
                        "TARGET_BACKDROP product_id=product_2 [RAW/NO OVERLAY]"
                    ),
                    "detail_label": "FINAL_DETAIL product_id=product_2 [SAME CROP]",
                    "detail_available": True,
                },
            ],
        )

    def test_visual_review_ignores_model_total_and_verdict(self) -> None:
        payload = {
            "visual_score": 100,
            "verdict": "pass",
            "subscores": {
                "grounding": 20,
                "cast_shadows": 30,
                "geometry_layout": 40,
                "lighting_color": 50,
                "cutout_edges": 60,
                "reflection_material": 70,
            },
            "critical_failures": [],
            "issues": [],
            "recommended_adjustments": [],
        }
        review = validate_visual_review(payload)
        self.assertAlmostEqual(review["visual_score"], 39.5)
        self.assertEqual(review["verdict"], "reject")
        self.assertFalse(review["batch_unlock"])

    def test_visual_review_exposes_shared_shadow_vector_as_auditable_evidence(self) -> None:
        visual_payload = {
            "subscores": {
                "grounding": 90,
                "cast_shadows": 90,
                "geometry_layout": 90,
                "lighting_color": 90,
                "cutout_edges": 90,
                "reflection_material": 90,
            },
            "critical_failures": [],
            "issues": [],
            "recommended_adjustments": [],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            final = root / "final.png"
            final.write_bytes(b"test")
            (root / "job.json").write_text(json.dumps({
                "lighting": {"shadow_direction": [0.6, 0.8]},
                "products": [{
                    "id": "product_1",
                    "material_mode": "clear_glass",
                    "material_confidence": 0.97,
                    "transmission_strength": 0.94,
                }],
            }), encoding="utf-8")
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
                    "strict_compositor.minimax_advisor._visual_review_detail_crops",
                    return_value={},
                ),
                patch(
                    "strict_compositor.minimax_advisor._visual_review_evidence_boards",
                    return_value=(Path("context.png"), Path("details.png")),
                ),
                patch(
                    "strict_compositor.minimax_advisor._image_data_url",
                    return_value="data:image/test;base64,unit-test",
                ),
                patch(
                    "strict_compositor.minimax_advisor.requests.post",
                    return_value=self._response(visual_payload, score=True),
                ) as post,
            ):
                review = request_minimax_visual_review(
                    "background.png", final, ["product-1.png"],
                )
        prompt_text = post.call_args.kwargs["json"]["messages"][0]["content"][0]["text"]
        self.assertIn("dx=0.600000, dy=0.800000", prompt_text)
        self.assertIn("not as proof of correctness", prompt_text)
        self.assertIn("do not infer multiple lights from centroid position alone", prompt_text)
        self.assertIn("product_1: material_mode=clear_glass", prompt_text)
        self.assertIn("source-room band seen through the original vessel", prompt_text)
        self.assertIn("DISPLAYED GEOMETRY SANITY EVIDENCE", prompt_text)
        self.assertIn("DETERMINISTIC EFFECT LOCALIZATION", prompt_text)
        self.assertIn("INTRINSIC ALPHA/RIB GEOMETRY", prompt_text)
        self.assertIn("dark circular mounting plate", prompt_text)
        self.assertIn("physically expected Fresnel/specular cue", prompt_text)
        self.assertEqual(review["declared_shadow_direction"], [0.6, 0.8])
        self.assertEqual(review["material_manifest"][0]["material_mode"], "clear_glass")


    def test_extracts_json_after_thinking_and_fence(self) -> None:
        value = extract_json_object(
            '<think>private reasoning</think>\n```json\n{"route":"strict_composite"}\n```'
        )
        self.assertEqual(value["route"], "strict_composite")

    def test_strict_validation_rejects_out_of_range_coordinates(self) -> None:
        payload = valid_payload()
        payload["support_polygon"][0][0] = 1.2
        with self.assertRaises(ContractError):
            validate_advice(payload, ["product_1"])

    def test_validation_requires_every_product(self) -> None:
        with self.assertRaises(ContractError):
            validate_advice(valid_payload(), ["product_1", "product_2"])

    def test_invalid_physical_placement_mode_is_rejected(self) -> None:
        payload = valid_payload()
        payload["products"][0]["placement_mode"] = "floating"
        with self.assertRaises(ContractError):
            validate_advice(payload, ["product_1"])

    def test_display_plate_uses_diameter_as_upright_height(self) -> None:
        payload = valid_payload()
        product = payload["products"][0]
        product.update({
            "semantic_category": "decorative_painted_tray",
            "physical_type": "flat_decor",
            "placement_mode": "surface_flat",
            "photographed_pose": "display_stand",
            "pose_confidence": 0.92,
            "estimated_real_height_cm": 3.0,
            "footprint_width_cm": 30.0,
            "size_confidence": 0.85,
        })
        advice = validate_advice(payload, ["product_1"])
        corrected = advice.products[0]
        self.assertEqual(corrected.placement_mode, "upright")
        self.assertEqual(corrected.physical_type, "rigid_display_object")
        self.assertGreaterEqual(corrected.estimated_real_height_cm, 25.8)
        self.assertEqual(corrected.photographed_pose, "display_stand")

    def test_missing_environment_key_never_attempts_network(self) -> None:
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("strict_compositor.minimax_advisor.get_secret", return_value=""),
            patch(
                "strict_compositor.minimax_advisor.get_public_settings",
                return_value={"minimax_base_url": "https://api.minimax.io", "minimax_model": "MiniMax-M3"},
            ),
        ):
            with self.assertRaises(MiniMaxNotConfigured):
                request_minimax_advice("missing.jpg")

    def test_reconcile_accepts_high_overlap_support_and_adds_protection(self) -> None:
        advice = validate_advice(valid_payload(), ["product_1"])
        local = {
            "support_polygon": [[0.07, 0.19], [0.93, 0.19], [0.93, 0.76], [0.07, 0.76]],
            "protected_polygons": [],
            "support_mask_path": "local.png",
            "reflectivity": 0.1,
            "reflection_recommended": False,
            "shadow_direction": [1.0, 0.0],
            "method": "local",
            "confidence": 0.75,
        }
        with tempfile.TemporaryDirectory() as temporary:
            result = reconcile_scene_result(local, advice, (300, 500), Path(temporary))
            self.assertTrue(result["advisor"]["support_accepted"])
            self.assertEqual(result["support_polygon"], advice.support_polygon)
            self.assertEqual(len(result["protected_polygons"]), 1)
            self.assertTrue(Path(result["support_mask_path"]).exists())

    def test_reconcile_ignores_background_furniture_and_table_edges(self) -> None:
        payload = valid_payload()
        payload["protected_regions"].extend([
            {
                "label": "office_chair",
                "polygon": [[0.45, 0.20], [0.75, 0.20], [0.75, 0.70], [0.45, 0.70]],
            },
            {
                "label": "table_edge_front",
                "polygon": [[0.08, 0.68], [0.92, 0.68], [0.92, 0.75], [0.08, 0.75]],
            },
        ])
        advice = validate_advice(payload, ["product_1"])
        local = {
            "support_polygon": [[0.08, 0.2], [0.92, 0.2], [0.92, 0.75], [0.08, 0.75]],
            "protected_polygons": [],
            "support_mask_path": "local.png",
            "reflectivity": 0.1,
            "reflection_recommended": False,
            "shadow_direction": [1.0, 0.0],
            "method": "local",
            "confidence": 0.9,
        }
        with tempfile.TemporaryDirectory() as temporary:
            result = reconcile_scene_result(local, advice, (300, 500), Path(temporary))
        self.assertEqual(len(result["protected_polygons"]), 1)
        self.assertEqual(
            result["advisor"]["accepted_receiver_obstacle_labels"],
            ["telephone"],
        )
        self.assertEqual(
            set(result["advisor"]["ignored_non_receiver_labels"]),
            {"office_chair", "table_edge_front"},
        )

    def test_two_cloud_passes_can_recover_a_low_confidence_local_plane(self) -> None:
        payload = valid_payload()
        payload["confidence"] = 0.86
        advice = validate_advice(payload, ["product_1"])
        local = {
            "support_polygon": [[0.08, 0.62], [0.92, 0.62], [0.92, 0.70], [0.08, 0.70]],
            "protected_polygons": [], "support_mask_path": "local.png",
            "front_edge_y": 0.70, "reflectivity": 0.1,
            "reflection_recommended": False, "shadow_direction": [1.0, 0.0],
            "method": "local", "confidence": 0.43,
        }
        with tempfile.TemporaryDirectory() as temporary:
            result = reconcile_scene_result(
                local, advice, (300, 500), Path(temporary),
                advisor_support_consensus=0.63,
            )
        self.assertTrue(result["advisor"]["support_accepted"])
        self.assertEqual(result["advisor"]["support_geometry_source"], "minimax")
        self.assertEqual(result["front_edge_y"], 0.75)
        self.assertGreater(result["confidence"], 0.70)

    def test_layout_advice_only_seeds_fields_owned_by_the_advisor(self) -> None:
        advice = MiniMaxAdvice(
            support_polygon=[[0.1, 0.2], [0.9, 0.2], [0.9, 0.8], [0.1, 0.8]],
            protected_polygons=[],
            protected_labels=[],
            shadow_direction=[1.0, 0.0],
            light_direction=[-1.0, 0.0],
            reflectivity=0.0,
            reflection_recommended=False,
            route="strict_composite",
            confidence=0.9,
            products=[ProductAdvice("product_1", 0.4, 0.7, 0.2, 1, 0.1, 0.8)],
            warnings=[],
            model="MiniMax-M3",
        )
        products = [{"id": "product_1", "source": "source.jpg", "rgba": "cutout.png"}]
        seeded = apply_layout_advice(products, advice)
        self.assertEqual(seeded[0]["rgba"], "cutout.png")
        self.assertEqual(seeded[0]["center_x"], 0.4)
        self.assertEqual(seeded[0]["baseline_y"], 0.7)
        self.assertEqual(seeded[0]["placement_mode"], "upright")

    def test_semantic_real_size_calibrates_relative_projected_height(self) -> None:
        advice = MiniMaxAdvice(
            support_polygon=[[0.1, 0.2], [0.9, 0.2], [0.9, 0.8], [0.1, 0.8]],
            protected_polygons=[], protected_labels=[], shadow_direction=[1.0, 0.0],
            light_direction=[-1.0, 0.0], reflectivity=0.0,
            reflection_recommended=False, route="strict_composite", confidence=0.95,
            products=[
                ProductAdvice("small", 0.3, 0.6, 0.16, 1, 0.08, 0.8,
                              estimated_real_height_cm=10.0, size_confidence=0.95),
                ProductAdvice("large", 0.7, 0.6, 0.16, 2, 0.08, 0.8,
                              estimated_real_height_cm=40.0, size_confidence=0.95),
            ], warnings=[], model="MiniMax-M3",
        )
        seeded = apply_layout_advice(
            [{"id": "small", "source": "a"}, {"id": "large", "source": "b"}],
            advice,
        )
        self.assertLess(seeded[0]["height_ratio"], seeded[1]["height_ratio"])
        # Both category estimates are high confidence.  Their 4:1 physical
        # ratio must not be flattened by the old fixed 0.68 exponent into an
        # almost equal-looking pair.
        self.assertGreater(
            seeded[1]["height_ratio"] / seeded[0]["height_ratio"],
            1.9,
        )
        self.assertEqual(seeded[1]["estimated_real_height_cm"], 40.0)
        self.assertEqual(seeded[1]["viewpoint_compatibility"], 0.8)

    def test_semantic_scale_uses_local_receiver_width_on_perspective_plane(self) -> None:
        support = [[0.34, 0.20], [0.66, 0.20], [0.94, 0.88], [0.06, 0.88]]
        advice = MiniMaxAdvice(
            support_polygon=support,
            protected_polygons=[], protected_labels=[], shadow_direction=[1.0, 0.0],
            light_direction=[-1.0, 0.0], reflectivity=0.0,
            reflection_recommended=False, route="strict_composite", confidence=0.96,
            receiver_width_or_diameter_cm=100.0,
            receiver_size_confidence=0.95,
            products=[
                ProductAdvice(
                    "far", 0.50, 0.30, 0.20, 1, 0.08, 0.8,
                    estimated_real_height_cm=20.0, size_confidence=0.95,
                ),
                ProductAdvice(
                    "near", 0.50, 0.78, 0.20, 2, 0.08, 0.8,
                    estimated_real_height_cm=20.0, size_confidence=0.95,
                ),
            ], warnings=[], model="MiniMax-M3",
        )
        seeded = apply_layout_advice(
            [{"id": "far", "source": "a"}, {"id": "near", "source": "b"}],
            advice,
            canvas_size=(1200, 800),
            support_polygon=support,
        )

        far_height = float(seeded[0]["height_ratio"])
        near_height = float(seeded[1]["height_ratio"])
        # Same real object, same receiver: the far baseline intersects a
        # narrower projected tabletop span and must therefore render smaller.
        self.assertLess(far_height, near_height * 0.82)

    def test_reliable_absolute_scale_overrides_tiny_model_box(self) -> None:
        support = [
            [0.267, 0.62], [0.642, 0.62],
            [0.642, 0.82], [0.267, 0.82],
        ]
        advice = MiniMaxAdvice(
            support_polygon=support,
            protected_polygons=[], protected_labels=[],
            shadow_direction=[1.0, 0.0], light_direction=[-1.0, 0.0],
            reflectivity=0.68, reflection_recommended=True,
            route="strict_composite", confidence=0.90,
            receiver_width_or_diameter_cm=55.0,
            receiver_size_confidence=0.72,
            products=[ProductAdvice(
                "pitcher", 0.52, 0.72, 0.15, 1, 0.12, 0.8,
                estimated_real_height_cm=25.0, size_confidence=0.80,
            )], warnings=[], model="MiniMax-M3",
        )

        seeded = apply_layout_advice(
            [{"id": "pitcher", "source": "pitcher.png"}], advice,
            canvas_size=(7008, 4672), support_polygon=support,
        )

        # The physical target is roughly (25 / 55) * 0.375 * 1.5 = 0.256
        # frame heights.  Keep a conservative allowance for camera/model
        # uncertainty, but do not let the 0.15 model proposal win.
        self.assertGreater(float(seeded[0]["height_ratio"]), 0.225)
        self.assertLess(float(seeded[0]["height_ratio"]), 0.275)


if __name__ == "__main__":
    unittest.main()
