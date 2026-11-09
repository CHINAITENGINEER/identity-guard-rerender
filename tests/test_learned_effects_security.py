from __future__ import annotations

import unittest

import numpy as np

from strict_compositor.contracts import Lighting
from strict_compositor.learned_effects import (
    EffectContext,
    LearnedEffectContractError,
    LearnedEffectPrediction,
    LearnedEffectRequest,
    _shadow_admissible_envelope,
    isolate_effect_request,
    sanitize_effect_prediction,
)


def _context() -> EffectContext:
    height, width = 120, 160
    background = np.full((height, width, 3), 184, np.uint8)
    support = np.ones((height, width), bool)
    protected = np.zeros((height, width), bool)
    protected[82:100, 125:150] = True
    alpha = np.zeros((height, width), np.uint8)
    alpha[34:82, 62:88] = 255
    rgb = np.zeros((height, width, 3), np.uint8)
    rgb[alpha > 0] = (115, 42, 31)
    identity = np.zeros((height, width), bool)
    identity[48:61, 65:85] = True
    baseline = np.zeros((height, width), np.uint8)
    baseline[79:84, 59:96] = 92
    baseline[82:93, 70:105] = 48
    lighting = Lighting(
        direction=(0.78, 0.62),
        shadow_length_ratio=0.14,
        shadow_opacity=0.34,
        shadow_softness_ratio=0.08,
        cast_shadow_mode="directional",
        surface_material="matte",
    )
    return EffectContext(
        product_id="product_1",
        background_rgb=background,
        support_mask=support,
        protected_mask=protected,
        product_rgb=rgb,
        product_alpha=alpha,
        identity_guard_mask=identity,
        deterministic_shadow_alpha=baseline,
        material_mode="opaque_matte",
        lighting=lighting,
        metadata={"nested": {"values": [1, 2]}},
    )


class LearnedEffectSecurityTests(unittest.TestCase):
    def test_uniform_white_shadow_cannot_authorize_the_receiver(self) -> None:
        context = _context()
        malicious = LearnedEffectPrediction(
            shadow_alpha=np.ones(context.product_alpha.shape, np.float32),
        )
        sanitized = sanitize_effect_prediction(malicious, context)
        self.assertIsNotNone(sanitized.shadow_alpha)
        assert sanitized.shadow_alpha is not None
        envelope = _shadow_admissible_envelope(context)
        self.assertFalse(np.any(sanitized.shadow_alpha[~envelope]))
        self.assertLess(np.count_nonzero(envelope), envelope.size)
        self.assertTrue(sanitized.report["shadow_uniform_fill_detected"])
        self.assertGreater(sanitized.report["shadow_outside_envelope_pixels"], 0)

    def test_shadow_density_is_clipped_to_local_envelope_and_local_peak(self) -> None:
        context = _context()
        envelope = _shadow_admissible_envelope(context)
        yy, xx = np.indices(context.product_alpha.shape)
        density = np.clip(1.0 - np.hypot(xx - 86.0, yy - 86.0) / 38.0, 0.0, 1.0)
        sanitized = sanitize_effect_prediction(
            LearnedEffectPrediction(shadow_alpha=density.astype(np.float32)),
            context,
        )
        self.assertIsNotNone(sanitized.shadow_alpha)
        assert sanitized.shadow_alpha is not None
        self.assertFalse(np.any(sanitized.shadow_alpha[~envelope]))
        self.assertLessEqual(int(sanitized.shadow_alpha.max()), 199)
        self.assertGreater(sanitized.report["shadow_clipped_pixels"], 0)
        self.assertGreater(sanitized.report["shadow_accepted_pixels"], 0)
        self.assertLessEqual(sanitized.report["shadow_envelope_fill_ratio"], 1.0)

    def test_detached_shadow_is_rejected(self) -> None:
        context = _context()
        detached = np.zeros(context.product_alpha.shape, np.float32)
        envelope = _shadow_admissible_envelope(context)
        candidates = np.argwhere(envelope & ~(context.deterministic_shadow_alpha > 0))
        self.assertGreater(candidates.shape[0], 0)
        y, x = candidates[0]
        detached[y, x] = 1.0
        with self.assertRaisesRegex(
            LearnedEffectContractError,
            "detached from the deterministic contact projection",
        ):
            sanitize_effect_prediction(
                LearnedEffectPrediction(shadow_alpha=detached),
                context,
            )

    def test_provider_metadata_is_deeply_isolated(self) -> None:
        context = _context()
        request_metadata = {"nested": {"values": [3, 4]}}
        request = LearnedEffectRequest(
            background_rgb=context.background_rgb,
            support_mask=context.support_mask,
            protected_mask=context.protected_mask,
            lighting=context.lighting,
            products=(context,),
            metadata=request_metadata,
        )
        isolated = isolate_effect_request(request)
        request_metadata["nested"]["values"].append(5)
        context.metadata["nested"]["values"].append(6)
        self.assertEqual(isolated.metadata["nested"]["values"], [3, 4])
        self.assertEqual(
            isolated.products[0].metadata["nested"]["values"],
            [1, 2],
        )


if __name__ == "__main__":
    unittest.main()
