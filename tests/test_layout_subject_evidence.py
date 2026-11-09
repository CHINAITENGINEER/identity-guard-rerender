from __future__ import annotations

from pathlib import Path

from PIL import Image

import strict_compositor.ui as ui_module
from strict_compositor.minimax_advisor import MiniMaxAdvice, ProductAdvice
from strict_compositor.ui import (
    _local_subject_semantic_priors,
    _reconcile_advice_with_subject_evidence,
)


def _advice(product: ProductAdvice) -> MiniMaxAdvice:
    return MiniMaxAdvice(
        support_polygon=[[0.1, 0.2], [0.9, 0.2], [0.9, 0.8], [0.1, 0.8]],
        protected_polygons=[],
        protected_labels=[],
        shadow_direction=[0.7, 0.7],
        light_direction=[-0.7, -0.7],
        reflectivity=0.4,
        reflection_recommended=True,
        route="strict_composite",
        confidence=0.82,
        products=[product],
        warnings=[],
        model="MiniMax-M3",
    )


def _flat_plate() -> ProductAdvice:
    return ProductAdvice(
        product_id="product_1",
        center_x=0.36,
        bottom_y=0.43,
        height_ratio=0.30,
        z_order=1,
        footprint_width=0.27,
        source_light_compatibility=0.75,
        placement_mode="surface_flat",
        physical_type="rigid_disc",
        photographed_pose="lying_flat",
        pose_confidence=0.95,
        semantic_category="decorative_charger_plate",
        estimated_real_height_cm=3.0,
        estimated_height_cm_min=2.0,
        estimated_height_cm_max=4.0,
        footprint_width_cm=38.0,
        size_confidence=0.70,
        viewpoint_compatibility=0.88,
    )


def test_verified_plate_and_easel_use_diameter_as_visible_height() -> None:
    result, corrections = _reconcile_advice_with_subject_evidence(
        _advice(_flat_plate()),
        [{
            "id": "product_1",
            "subject_label": "decorative floral plate",
            "subject_included_parts": ["plate body", "gold rim", "display easel"],
            "subject_locator_confidence": 0.92,
            "subject_semantic_verified": True,
        }],
    )

    corrected = result.products[0]
    assert corrected.placement_mode == "upright"
    assert corrected.photographed_pose == "display_stand"
    assert corrected.physical_type == "rigid_display_object"
    assert corrected.estimated_real_height_cm >= 32.68
    assert corrected.estimated_height_cm_min >= 24.83
    assert corrected.estimated_height_cm_max >= 38.56
    assert corrected.pose_confidence >= 0.95
    assert corrections[0]["reason"] == "verified_plate_with_display_support"


def test_plate_without_display_support_is_not_forced_upright() -> None:
    original = _flat_plate()
    result, corrections = _reconcile_advice_with_subject_evidence(
        _advice(original),
        [{
            "id": "product_1",
            "subject_label": "decorative floral plate",
            "subject_included_parts": ["plate body", "gold rim"],
            "subject_locator_confidence": 0.94,
            "subject_semantic_verified": True,
        }],
    )

    assert result.products[0] == original
    assert corrections == []


def test_unverified_fallback_box_cannot_override_pose() -> None:
    original = _flat_plate()
    result, corrections = _reconcile_advice_with_subject_evidence(
        _advice(original),
        [{
            "id": "product_1",
            "subject_label": "decorative floral plate",
            "subject_included_parts": ["display easel"],
            "subject_locator_confidence": 0.95,
            "subject_semantic_verified": False,
        }],
    )

    assert result.products[0] == original
    assert corrections == []


def test_verified_explicit_clear_glass_corrects_source_room_colour_bias() -> None:
    biased = ProductAdvice(
        product_id="product_3",
        center_x=0.70,
        bottom_y=0.50,
        height_ratio=0.18,
        z_order=3,
        footprint_width=0.10,
        source_light_compatibility=0.75,
        semantic_category="ribbed_glass_tumbler",
        physical_type="rigid_vessel",
        estimated_real_height_cm=14.0,
        size_confidence=0.74,
        material_mode="colored_glass",
        material_confidence=0.71,
        transmission_strength=0.42,
    )
    result, corrections = _reconcile_advice_with_subject_evidence(
        _advice(biased),
        [{
            "id": "product_3",
            "subject_label": "ribbed clear glass tumbler",
            "subject_included_parts": ["clear body", "thick base"],
            "subject_locator_confidence": 0.85,
            "subject_semantic_verified": True,
        }],
    )

    corrected = result.products[0]
    assert corrected.material_mode == "clear_glass"
    assert corrected.material_confidence >= 0.82
    assert corrected.transmission_strength >= 0.95
    assert corrections[0]["reason"] == "verified_explicit_clear_glass_subject"


def _local_product(
    product_id: str,
    label: str,
    *,
    parts: list[str] | None = None,
    verified: bool = True,
    confidence: float = 0.92,
) -> dict[str, object]:
    return {
        "id": product_id,
        "source": f"{product_id}.png",
        "rgba": f"{product_id}.png",
        "placement_mode": "upright",
        "physical_type": "rigid_object",
        "photographed_pose": "unknown",
        "pose_confidence": 0.0,
        "semantic_category": "decor_object",
        "subject_label": label,
        "subject_included_parts": parts or [],
        "subject_locator_confidence": confidence,
        "subject_semantic_verified": verified,
        "estimated_real_height_cm": 20.0,
        "estimated_height_cm_min": 15.0,
        "estimated_height_cm_max": 30.0,
        "footprint_width_cm": 12.0,
        "size_confidence": 0.0,
        "material_mode": "opaque",
        "material_confidence": 0.0,
        "transmission_strength": 0.0,
    }


def test_verified_subjects_supply_distinct_local_size_and_material_priors() -> None:
    products, corrections = _local_subject_semantic_priors([
        _local_product(
            "product_1",
            "decorative floral plate",
            parts=["plate body", "gold rim", "display easel"],
        ),
        _local_product(
            "product_2",
            "coconut and soy candle in amber glass jar",
        ),
        _local_product(
            "product_3",
            "ribbed clear glass tumbler",
        ),
    ])

    plate, candle, tumbler = products
    assert plate["estimated_real_height_cm"] == 26.0
    assert plate["photographed_pose"] == "display_stand"
    assert plate["physical_type"] == "rigid_display_object"
    assert candle["estimated_real_height_cm"] == 10.5
    assert candle["material_mode"] == "opaque_glossy"
    assert tumbler["estimated_real_height_cm"] == 14.5
    assert tumbler["material_mode"] == "clear_glass"
    assert tumbler["transmission_strength"] == 0.95
    assert float(plate["size_confidence"]) >= 0.60
    assert float(candle["size_confidence"]) >= 0.60
    assert float(tumbler["material_confidence"]) >= 0.82
    assert {item["reason"] for item in corrections} == {
        "verified_display_plate_prior",
        "verified_candle_prior",
        "verified_drinking_glass_prior",
    }


def test_unverified_or_low_confidence_subject_does_not_change_local_defaults() -> None:
    unverified = _local_product(
        "product_1", "clear glass tumbler", verified=False,
    )
    uncertain = _local_product(
        "product_2", "scented candle", confidence=0.61,
    )

    products, corrections = _local_subject_semantic_priors([
        unverified, uncertain,
    ])

    assert products == [unverified, uncertain]
    assert corrections == []


def test_verified_subject_never_overrides_explicit_size_pose_or_material() -> None:
    explicit = _local_product("product_1", "clear glass tumbler")
    explicit.update({
        "placement_mode": "surface_flat",
        "photographed_pose": "lying_flat",
        "pose_confidence": 0.91,
        "estimated_real_height_cm": 33.0,
        "estimated_height_cm_min": 30.0,
        "estimated_height_cm_max": 36.0,
        "footprint_width_cm": 18.0,
        "size_confidence": 0.88,
        "material_mode": "metal",
        "material_confidence": 0.90,
        "transmission_strength": 0.0,
    })

    products, _ = _local_subject_semantic_priors([explicit])
    result = products[0]

    assert result["estimated_real_height_cm"] == 33.0
    assert result["estimated_height_cm_min"] == 30.0
    assert result["estimated_height_cm_max"] == 36.0
    assert result["footprint_width_cm"] == 18.0
    assert result["placement_mode"] == "surface_flat"
    assert result["photographed_pose"] == "lying_flat"
    assert result["material_mode"] == "metal"
    assert result["material_confidence"] == 0.90


def test_auto_layout_transport_fallback_passes_enriched_products_to_solver(
    tmp_path: Path,
    monkeypatch,
) -> None:
    background = tmp_path / "background.png"
    Image.new("RGB", (64, 48), (220, 220, 220)).save(background)
    source = [
        _local_product(
            "product_1",
            "decorative floral plate",
            parts=["display easel"],
        ),
        _local_product(
            "product_2",
            "coconut and soy candle in amber glass jar",
        ),
        _local_product(
            "product_3",
            "ribbed clear glass tumbler",
            confidence=0.85,
        ),
    ]
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        ui_module, "minimax_status", lambda: {"configured": True},
    )

    def fail_advice(*args, **kwargs):
        raise RuntimeError("relay offline")

    monkeypatch.setattr(ui_module, "request_minimax_advice", fail_advice)

    def fake_solver(canvas, support, protected, products, **kwargs):
        captured["products"] = [dict(item) for item in products]
        return [dict(item) for item in products], {"layout_score": 95.0}

    monkeypatch.setattr(ui_module, "solve_aesthetic_layout", fake_solver)
    monkeypatch.setattr(
        ui_module,
        "_layout_surface_diagnostics",
        lambda *args, **kwargs: {
            "layout_verdict": "pass",
            "layout_score": 95.0,
            "layout_reasons": [],
        },
    )

    payload = ui_module.LayoutUI(
        background=str(background),
        products=source,
        support_polygon=[
            [0.05, 0.10], [0.95, 0.10], [0.95, 0.95], [0.05, 0.95],
        ],
    )
    result = ui_module.auto_layout(payload)

    products = captured["products"]
    assert isinstance(products, list)
    plate, candle, tumbler = products
    assert float(plate["estimated_real_height_cm"]) > float(
        tumbler["estimated_real_height_cm"]
    ) > float(candle["estimated_real_height_cm"])
    assert tumbler["material_mode"] == "clear_glass"
    assert float(tumbler["material_confidence"]) >= 0.82
    assert result["advisor"]["status"] == "fallback_local"
    assert result["advisor"]["reason"] == "relay offline"
    assert len(result["advisor"]["local_semantic_priors"]) == 3
    assert result["metrics"]["advisor_used"] is False
