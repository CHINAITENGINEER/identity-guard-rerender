from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from strict_compositor.contracts import Lighting
from strict_compositor.physical_audit import audit_physical_composition
from strict_compositor.render import RenderedProduct, alpha_over, darken


class FinalPixelVisibilityAuditTests(unittest.TestCase):
    def _fixture(
        self,
        *,
        reflection: bool = False,
        background_value: int = 180,
    ) -> tuple[
        list[tuple[SimpleNamespace, RenderedProduct]],
        np.ndarray,
        np.ndarray,
        Lighting,
        np.ndarray,
    ]:
        height = width = 90
        alpha = np.zeros((height, width), np.uint8)
        alpha[20:55, 30:60] = 255
        rgb = np.zeros((height, width, 3), np.uint8)
        rgb[alpha > 0] = [190, 150, 110]
        contact = np.zeros((height, width), np.uint8)
        contact[53:60, 27:63] = 100
        zeros = np.zeros((height, width), np.uint8)
        reflection_alpha = np.zeros((height, width), np.uint8)
        reflection_rgb = np.zeros_like(rgb)
        if reflection:
            reflection_alpha[58:75, 30:60] = 90
            reflection_rgb[reflection_alpha > 0] = [220, 120, 40]
        layer = RenderedProduct(
            rgb,
            alpha,
            contact,
            zeros,
            reflection_rgb,
            reflection_alpha,
            zeros,
        )
        product = SimpleNamespace(product_id="product_1")
        support = np.ones((height, width), bool)
        protected = np.zeros((height, width), bool)
        lighting = Lighting(
            direction=(1.0, 0.0),
            shadow_length_ratio=0.08,
            reflection_enabled=reflection,
            reflection_opacity=0.30 if reflection else 0.0,
            cast_shadow_mode="ambient_only",
            surface_material="mirror" if reflection else "matte",
            surface_reflectivity=0.8 if reflection else 0.0,
        )
        background = np.full(
            (height, width, 3), background_value, dtype=np.uint8,
        )
        return [(product, layer)], support, protected, lighting, background

    def test_bright_receiver_rejects_layer_that_is_absent_from_final_pixels(self) -> None:
        rendered, support, protected, lighting, background = self._fixture()

        report = audit_physical_composition(
            rendered,
            support,
            protected,
            lighting,
            background,
            background.copy(),
        )

        visibility = report["products"][0]["final_pixel_visibility"]["contact"]
        self.assertEqual(visibility["status"], "insufficient_visible_contact")
        self.assertEqual(visibility["observed_rgb_delta_p75"], 0.0)
        self.assertIn("contact_final_pixel_visibility_insufficient", report["failures"])
        self.assertLess(report["subscores"]["grounding"], 95.0)
        self.assertEqual(report["verdict"], "reject")

    def test_dark_receiver_is_explained_without_requesting_more_black(self) -> None:
        rendered, support, protected, lighting, background = self._fixture(
            background_value=5,
        )

        report = audit_physical_composition(
            rendered,
            support,
            protected,
            lighting,
            background,
            background.copy(),
        )

        visibility = report["products"][0]["final_pixel_visibility"]["contact"]
        self.assertEqual(visibility["status"], "dark_receiver_limited")
        self.assertIn("do_not_add_black", visibility["reason"])
        self.assertEqual(visibility["score"], 97.0)
        self.assertNotIn("contact_final_pixel_visibility_insufficient", report["failures"])
        self.assertEqual(report["verdict"], "pass")
        self.assertLess(report["subscores"]["grounding"], 100.0)

    def test_visible_contact_is_measured_after_final_composition(self) -> None:
        rendered, support, protected, lighting, background = self._fixture()
        layer = rendered[0][1]
        final = darken(background, layer.contact_shadow)
        final = alpha_over(final, layer.rgb, layer.alpha)

        report = audit_physical_composition(
            rendered,
            support,
            protected,
            lighting,
            background,
            final,
        )

        visibility = report["products"][0]["final_pixel_visibility"]["contact"]
        self.assertEqual(visibility["status"], "visible")
        self.assertGreater(visibility["observed_darkening_p75"], 20.0)
        self.assertGreater(visibility["visible_pixel_fraction"], 0.9)
        self.assertGreater(visibility["realization_ratio"], 0.97)
        self.assertLess(visibility["realization_ratio"], 1.03)
        self.assertNotIn("contact_final_pixel_visibility_insufficient", report["failures"])

    def test_contact_uses_reflective_receiver_counterfactual(self) -> None:
        rendered, support, protected, lighting, background = self._fixture(
            reflection=True,
            background_value=100,
        )
        layer = rendered[0][1]
        # Production order is cast -> reflection -> contact -> product.  The
        # reflection makes the receiver brighter than the immutable source, so
        # a background-only comparison would falsely under-report AO delivery.
        receiver_before_contact = alpha_over(
            background,
            layer.reflection_rgb,
            layer.reflection_alpha,
        )
        final = darken(receiver_before_contact, layer.contact_shadow)
        final = alpha_over(final, layer.rgb, layer.alpha)

        report = audit_physical_composition(
            rendered,
            support,
            protected,
            lighting,
            background,
            final,
        )

        visibility = report["products"][0]["final_pixel_visibility"]["contact"]
        self.assertEqual(visibility["status"], "visible")
        self.assertIn("counterfactual_receiver_before_contact_ao", visibility["comparison"])
        self.assertGreater(visibility["realization_ratio"], 0.97)
        self.assertLess(visibility["realization_ratio"], 1.03)
        self.assertGreaterEqual(report["subscores"]["grounding"], 95.0)
        self.assertNotIn("grounding_below_95", report["failures"])

    def test_enabled_reflection_must_survive_into_authorized_final_pixels(self) -> None:
        rendered, support, protected, lighting, background = self._fixture(
            reflection=True,
            background_value=80,
        )
        layer = rendered[0][1]
        # Preserve a valid contact seam but deliberately omit the reflection.
        final = darken(background, layer.contact_shadow)
        final = alpha_over(final, layer.rgb, layer.alpha)

        report = audit_physical_composition(
            rendered,
            support,
            protected,
            lighting,
            background,
            final,
        )

        visibility = report["products"][0]["final_pixel_visibility"]["reflection"]
        self.assertEqual(visibility["status"], "insufficient_visible_reflection")
        self.assertGreater(visibility["expected_rgb_delta_p75"], 10.0)
        self.assertEqual(visibility["observed_rgb_delta_p75"], 0.0)
        self.assertIn("reflection_final_pixel_visibility_insufficient", report["failures"])
        self.assertEqual(report["verdict"], "reject")

    def test_legacy_call_keeps_old_interface_without_inventing_visibility(self) -> None:
        rendered, support, protected, lighting, _ = self._fixture()

        report = audit_physical_composition(
            rendered, support, protected, lighting,
        )

        self.assertFalse(report["final_pixel_visibility"]["measured"])
        self.assertEqual(
            report["products"][0]["final_pixel_visibility"]["contact"]["status"],
            "not_measured",
        )
        self.assertNotIn("contact_final_pixel_visibility_insufficient", report["failures"])


if __name__ == "__main__":
    unittest.main()
