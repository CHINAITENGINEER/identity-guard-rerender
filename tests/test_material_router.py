from __future__ import annotations

import numpy as np

from strict_compositor.effect_calibration import calibrate_reflection_visibility
from strict_compositor.material_router import resolve_material_route


def test_uncertain_ribbed_open_glass_rejects_source_room_colour() -> None:
    route = resolve_material_route(
        "colored_glass", "ribbed_glass_votive_or_planter", "rigid_vessel",
    )
    assert route.resolved_mode == "clear_glass"
    assert route.confidence_floor >= 0.80
    assert route.transmission_floor >= 0.90


def test_stably_coloured_open_glass_keeps_intrinsic_tint() -> None:
    route = resolve_material_route(
        "colored_glass", "blue_ribbed_glass_vase", "rigid_vessel",
    )
    assert route.resolved_mode == "colored_glass"


def test_labelled_liquid_bottle_routes_to_mixed_transparency() -> None:
    route = resolve_material_route(
        "colored_glass", "amber_whiskey_bottle", "rigid_vessel",
    )
    assert route.resolved_mode == "mixed_transparent"


def test_reflection_calibration_uses_visible_contrast_not_fixed_alpha() -> None:
    receiver = np.full((24, 28, 3), 230, np.uint8)
    reflected = np.full_like(receiver, 90)
    alpha = np.zeros((24, 28), np.uint8)
    alpha[5:20, 4:24] = 6
    mask = np.ones_like(alpha, bool)
    calibrated, report = calibrate_reflection_visibility(
        reflected,
        alpha,
        receiver,
        mask,
        surface_reflectivity=0.62,
        roughness=0.18,
        reflection_opacity=0.18,
        material_mode="colored_glass",
    )
    assert report.alpha_scale > 1.0
    assert int(calibrated.max()) > int(alpha.max())
    assert int(calibrated.max()) <= 112


def test_reflection_calibration_never_creates_new_shape() -> None:
    receiver = np.full((20, 20, 3), 180, np.uint8)
    reflected = np.full_like(receiver, 40)
    alpha = np.zeros((20, 20), np.uint8)
    alpha[7:13, 8:12] = 10
    mask = np.ones_like(alpha, bool)
    calibrated, _ = calibrate_reflection_visibility(
        reflected,
        alpha,
        receiver,
        mask,
        surface_reflectivity=0.8,
        roughness=0.08,
        reflection_opacity=0.2,
        material_mode="opaque_glossy",
    )
    assert np.array_equal(calibrated > 0, alpha > 0)
