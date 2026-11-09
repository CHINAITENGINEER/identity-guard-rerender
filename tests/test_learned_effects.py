from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from strict_compositor.contracts import Job, Lighting
from strict_compositor.image_io import save_png
from strict_compositor.learned_effects import (
    EffectContext,
    LearnedEffectBatchPrediction,
    LearnedEffectContractError,
    LearnedEffectPrediction,
    LearnedEffectRequest,
    apply_learned_effect_batch,
    apply_learned_effect_prediction,
    isolate_effect_request,
    load_scene_evidence,
    sanitize_effect_prediction,
)
from strict_compositor.pipeline import StrictCompositor


def _context() -> EffectContext:
    height, width = 20, 24
    yy, xx = np.indices((height, width))
    background = np.stack(
        [
            70 + xx * 2,
            90 + yy * 2,
            np.full((height, width), 120),
        ],
        axis=2,
    ).clip(0, 255).astype(np.uint8)
    product_rgb = np.full((height, width, 3), [17, 23, 31], np.uint8)
    alpha = np.zeros((height, width), np.uint8)
    alpha[4:16, 6:18] = 255
    # A high-frequency catalogue texture makes accidental blur observable.
    texture = ((xx[4:16, 6:18] + yy[4:16, 6:18]) % 2) * 60
    product_rgb[4:16, 6:18] = np.stack(
        [96 + texture, 105 + texture, 118 + texture], axis=2,
    ).astype(np.uint8)
    support = np.zeros((height, width), bool)
    support[2:19, 2:23] = True
    protected = np.zeros((height, width), bool)
    protected[3:7, 19:22] = True
    identity = np.zeros((height, width), bool)
    identity[8:12, 9:15] = True
    deterministic_shadow = np.zeros((height, width), np.uint8)
    deterministic_shadow[15:19, 7:20] = 70
    lighting = Lighting(
        direction=(1.0, 0.0),
        shadow_length_ratio=0.12,
        cast_shadow_mode="directional",
    )
    return EffectContext(
        product_id="fixture_product",
        background_rgb=background,
        support_mask=support,
        protected_mask=protected,
        product_rgb=product_rgb,
        product_alpha=alpha,
        identity_guard_mask=identity,
        deterministic_shadow_alpha=deterministic_shadow,
        material_mode="mixed_transparent",
        lighting=lighting,
    )


class LearnedEffectsContractTests(unittest.TestCase):
    def test_shadow_is_hard_clipped_to_receiver(self) -> None:
        context = _context()
        clean = sanitize_effect_prediction(
            LearnedEffectPrediction(
                shadow_alpha=np.full(context.product_alpha.shape, 0.72, np.float32),
            ),
            context,
        )
        self.assertIsNotNone(clean.shadow_alpha)
        shadow = clean.shadow_alpha
        assert shadow is not None
        self.assertFalse(np.any(shadow[~context.support_mask]))
        self.assertFalse(np.any(shadow[context.protected_mask]))
        legal = context.support_mask & ~context.protected_mask
        self.assertTrue(np.any(shadow[legal]))
        self.assertGreater(clean.report["shadow_clipped_pixels"], 0)

    def test_full_rgb_gain_is_rejected(self) -> None:
        context = _context()
        with self.assertRaisesRegex(
            LearnedEffectContractError,
            "single-channel|full RGB",
        ):
            sanitize_effect_prediction(
                LearnedEffectPrediction(
                    illumination_gain=np.ones(
                        (*context.product_alpha.shape, 3), np.float32,
                    ),
                ),
                context,
            )

    def test_gain_and_specular_cannot_touch_background_or_identity(self) -> None:
        context = _context()
        gain = np.full(context.product_alpha.shape, 1.14, np.float32)
        residual = np.full(
            (*context.product_alpha.shape, 3), 0.018, np.float32,
        )
        before_alpha = context.product_alpha.copy()
        result = apply_learned_effect_prediction(
            LearnedEffectPrediction(
                illumination_gain=gain,
                specular_residual=residual,
            ),
            context,
        )
        body = context.product_alpha > 0
        np.testing.assert_array_equal(
            result.rgb[~body], context.product_rgb[~body],
        )
        np.testing.assert_array_equal(
            result.rgb[context.identity_guard_mask],
            context.product_rgb[context.identity_guard_mask],
        )
        np.testing.assert_array_equal(context.product_alpha, before_alpha)
        editable = body & ~context.identity_guard_mask
        self.assertTrue(np.any(result.rgb[editable] != context.product_rgb[editable]))
        self.assertEqual(result.report["outside_product_changed_pixels"], 0)
        self.assertEqual(result.report["identity_changed_pixels"], 0)

    def test_low_frequency_gain_does_not_blur_checker_texture(self) -> None:
        context = _context()
        result = apply_learned_effect_prediction(
            LearnedEffectPrediction(
                illumination_gain=np.full(
                    context.product_alpha.shape, 1.10, np.float32,
                ),
            ),
            context,
        )
        editable = (context.product_alpha > 0) & ~context.identity_guard_mask
        original = context.product_rgb[..., 0].astype(np.float32)
        changed = result.rgb[..., 0].astype(np.float32)
        # Alternating source texture must remain strongly alternating; a model
        # is allowed to move its low band, not replace the high-frequency band.
        original_std = float(original[editable].std())
        changed_std = float(changed[editable].std())
        self.assertGreater(changed_std, original_std * 0.88)

    def test_neutral_prediction_is_byte_exact(self) -> None:
        context = _context()
        result = apply_learned_effect_prediction(
            LearnedEffectPrediction(
                illumination_gain=np.ones(
                    context.product_alpha.shape, np.float32,
                ),
                specular_residual=np.zeros(
                    (*context.product_alpha.shape, 3), np.float32,
                ),
            ),
            context,
        )
        np.testing.assert_array_equal(result.rgb, context.product_rgb)
        np.testing.assert_array_equal(
            result.shadow_alpha, context.deterministic_shadow_alpha,
        )

    def test_scene_batch_requires_every_product_exactly_once(self) -> None:
        context = _context()
        request = LearnedEffectRequest(
            background_rgb=context.background_rgb,
            support_mask=context.support_mask,
            protected_mask=context.protected_mask,
            lighting=context.lighting,
            products=(context,),
        )
        with self.assertRaisesRegex(
            LearnedEffectContractError,
            "product ids",
        ):
            apply_learned_effect_batch(
                LearnedEffectBatchPrediction(products={}), request,
            )

    def test_provider_input_isolation_copies_mutable_arrays(self) -> None:
        depth = np.linspace(0.0, 1.0, 20 * 24, dtype=np.float32).reshape(20, 24)
        normal = np.zeros((20, 24, 3), np.float32)
        normal[..., 2] = 1.0
        context = replace(_context(), depth=depth, normal=normal)
        request = LearnedEffectRequest(
            background_rgb=context.background_rgb,
            support_mask=context.support_mask,
            protected_mask=context.protected_mask,
            lighting=context.lighting,
            products=(context,),
            depth=depth,
            normal=normal,
        )
        isolated = isolate_effect_request(request)
        self.assertIsNot(isolated.background_rgb, request.background_rgb)
        self.assertFalse(isolated.background_rgb.flags.writeable)
        self.assertFalse(isolated.products[0].product_rgb.flags.writeable)
        self.assertFalse(isolated.depth.flags.writeable)
        self.assertFalse(isolated.normal.flags.writeable)
        self.assertFalse(isolated.products[0].depth.flags.writeable)
        self.assertFalse(isolated.products[0].normal.flags.writeable)
        with self.assertRaises(ValueError):
            isolated.background_rgb[:] = 0
        with self.assertRaises(ValueError):
            isolated.products[0].product_rgb[:] = 255
        with self.assertRaises(ValueError):
            isolated.products[0].depth[:] = 0
        self.assertTrue(np.any(request.background_rgb))
        self.assertTrue(np.array_equal(
            isolated.products[0].product_rgb,
            request.products[0].product_rgb,
        ))

    def test_scene_evidence_is_strictly_loaded_resized_and_audited(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            depth = np.linspace(
                0.0, 1.0, 6 * 8, dtype=np.float32,
            ).reshape(6, 8)
            normal = np.zeros((6, 8, 3), np.float32)
            normal[..., 0] = 0.6
            normal[..., 2] = 0.8
            np.save(root / "depth.npy", depth)
            np.save(root / "normal.npy", normal)

            loaded = load_scene_evidence(
                root / "depth.npy",
                root / "normal.npy",
                (20, 24),
            )
            self.assertEqual(loaded.depth.shape, (20, 24))
            self.assertEqual(loaded.normal.shape, (20, 24, 3))
            self.assertEqual(loaded.depth.dtype, np.float32)
            self.assertEqual(loaded.normal.dtype, np.float32)
            self.assertFalse(loaded.depth.flags.writeable)
            self.assertFalse(loaded.normal.flags.writeable)
            self.assertGreaterEqual(float(loaded.depth.min()), 0.0)
            self.assertLessEqual(float(loaded.depth.max()), 1.0)
            np.testing.assert_allclose(
                np.linalg.norm(loaded.normal, axis=2),
                1.0,
                atol=1e-5,
            )
            report = loaded.report
            self.assertEqual(report["depth"]["source_shape"], [6, 8])
            self.assertEqual(report["depth"]["canvas_shape"], [20, 24])
            self.assertEqual(report["normal"]["canvas_shape"], [20, 24, 3])
            self.assertTrue(report["depth"]["resized"])
            self.assertEqual(len(report["depth"]["file_sha256"]), 64)
            self.assertEqual(len(report["normal"]["canvas_array_sha256"]), 64)
            self.assertEqual(report["depth"]["source_range"], [0.0, 1.0])

    def test_scene_evidence_rejects_dtype_range_and_nonunit_normals(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bad_depth_dtype = np.zeros((4, 5), np.float64)
            bad_depth_range = np.full((4, 5), 1.2, np.float32)
            bad_depth_shape = np.zeros((4, 5, 1), np.float32)
            bad_depth_nonfinite = np.zeros((4, 5), np.float32)
            bad_depth_nonfinite[1, 2] = np.nan
            bad_normal = np.zeros((4, 5, 3), np.float32)
            np.save(root / "bad_depth_dtype.npy", bad_depth_dtype)
            np.save(root / "bad_depth_range.npy", bad_depth_range)
            np.save(root / "bad_depth_shape.npy", bad_depth_shape)
            np.save(root / "bad_depth_nonfinite.npy", bad_depth_nonfinite)
            np.save(root / "bad_normal.npy", bad_normal)
            with self.assertRaisesRegex(
                LearnedEffectContractError, "dtype float32",
            ):
                load_scene_evidence(root / "bad_depth_dtype.npy", None, (8, 9))
            with self.assertRaisesRegex(
                LearnedEffectContractError, "normalized to",
            ):
                load_scene_evidence(root / "bad_depth_range.npy", None, (8, 9))
            with self.assertRaisesRegex(
                LearnedEffectContractError, "shape HxW",
            ):
                load_scene_evidence(root / "bad_depth_shape.npy", None, (8, 9))
            with self.assertRaisesRegex(
                LearnedEffectContractError, "NaN or infinity",
            ):
                load_scene_evidence(
                    root / "bad_depth_nonfinite.npy", None, (8, 9),
                )
            with self.assertRaisesRegex(
                LearnedEffectContractError, "unit length",
            ):
                load_scene_evidence(None, root / "bad_normal.npy", (8, 9))


class _NeutralProvider:
    name = "neutral_test_provider"

    def predict(self, request: LearnedEffectRequest) -> LearnedEffectBatchPrediction:
        return LearnedEffectBatchPrediction(products={
            context.product_id: LearnedEffectPrediction(
                illumination_gain=np.ones(
                    context.product_alpha.shape, np.float32,
                ),
            )
            for context in request.products
        })


class _MutatingInvalidProvider:
    name = "mutating_invalid_test_provider"

    def predict(self, request: LearnedEffectRequest) -> LearnedEffectBatchPrediction:
        request.background_rgb[:] = 0
        request.products[0].product_rgb[:] = 255
        context = request.products[0]
        return LearnedEffectBatchPrediction(products={
            context.product_id: LearnedEffectPrediction(
                illumination_gain=np.ones(
                    (*context.product_alpha.shape, 3), np.float32,
                ),
            ),
        })


class _EvidenceInspectingProvider:
    name = "evidence_inspecting_provider"

    def __init__(self) -> None:
        self.called = False
        self.observed: dict[str, object] = {}

    def predict(self, request: LearnedEffectRequest) -> LearnedEffectBatchPrediction:
        self.called = True
        context = request.products[0]
        self.observed = {
            "depth_shape": None if request.depth is None else request.depth.shape,
            "normal_shape": None if request.normal is None else request.normal.shape,
            "request_depth_readonly": (
                request.depth is not None and not request.depth.flags.writeable
            ),
            "request_normal_readonly": (
                request.normal is not None and not request.normal.flags.writeable
            ),
            "context_depth_readonly": (
                context.depth is not None and not context.depth.flags.writeable
            ),
            "context_normal_readonly": (
                context.normal is not None and not context.normal.flags.writeable
            ),
            "normal_norm_range": (
                None
                if request.normal is None
                else (
                    float(np.linalg.norm(request.normal, axis=2).min()),
                    float(np.linalg.norm(request.normal, axis=2).max()),
                )
            ),
        }
        return LearnedEffectBatchPrediction(products={
            item.product_id: LearnedEffectPrediction(
                illumination_gain=np.ones(
                    item.product_alpha.shape, np.float32,
                ),
            )
            for item in request.products
        })


class _ExplodingReportMapping(dict):
    """Mapping that is valid provider metadata until audit normalisation."""

    def items(self):
        raise RuntimeError("deliberate report metadata failure")


class _LateReportFailureProvider:
    name = "late_report_failure_provider"

    def __init__(self) -> None:
        self.saw_nonzero_strict_shadow = False

    def predict(self, request: LearnedEffectRequest) -> LearnedEffectBatchPrediction:
        predictions: dict[str, LearnedEffectPrediction] = {}
        for context in request.products:
            strict_shadow = np.asarray(
                context.deterministic_shadow_alpha,
                dtype=np.float32,
            )
            self.saw_nonzero_strict_shadow |= bool(np.any(strict_shadow))
            predictions[context.product_id] = LearnedEffectPrediction(
                # This is deliberately non-neutral: the sanitizer treats the
                # returned map as relative density and scales it by bounded
                # learned authority, so committing it would visibly change the
                # strict directional shadow before metadata serialisation.
                shadow_alpha=strict_shadow / 255.0,
            )
        return LearnedEffectBatchPrediction(
            products=predictions,
            metadata=_ExplodingReportMapping({"provider": self.name}),
        )


class LearnedEffectsPipelineTests(unittest.TestCase):
    def _job(self, root: Path) -> Job:
        background = np.full((72, 96, 3), [80, 100, 120], np.uint8)
        rgba = np.zeros((24, 18, 4), np.uint8)
        rgba[2:22, 3:15, :3] = [110, 70, 45]
        rgba[2:22, 3:15, 3] = 255
        save_png(root / "background.png", background)
        save_png(root / "product.png", rgba)
        return Job.from_dict({
            "background": "background.png",
            "support_polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
            "lighting": {
                "shadow_direction": [1, 0.2],
                "shadow_length_ratio": 0.0,
                "cast_shadow_mode": "ambient_only",
                "reflection_enabled": False,
            },
            "products": [{
                "id": "p",
                "rgba": "product.png",
                "homography": [[1, 0, 38], [0, 1, 35], [0, 0, 1]],
                "material_mode": "opaque_matte",
                "semantic_category": "unlabelled ceramic vase",
            }],
        }, root)

    def test_provider_absent_and_neutral_provider_are_byte_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job = self._job(root)
            baseline = StrictCompositor().compose(
                job, root / "baseline", strict_audit=False,
            )
            neutral = StrictCompositor(
                learned_effect_provider=_NeutralProvider(),
            ).compose(job, root / "neutral", strict_audit=False)
            np.testing.assert_array_equal(neutral.final, baseline.final)
            report = json.loads(
                (root / "neutral" / "audit.json").read_text(encoding="utf-8"),
            )
            self.assertEqual(report["learned_effects"]["status"], "applied")

    def test_invalid_mutating_provider_atomically_falls_back(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job = self._job(root)
            baseline = StrictCompositor().compose(
                job, root / "baseline", strict_audit=False,
            )
            fallback = StrictCompositor(
                learned_effect_provider=_MutatingInvalidProvider(),
            ).compose(job, root / "fallback", strict_audit=False)
            np.testing.assert_array_equal(fallback.final, baseline.final)
            report = json.loads(
                (root / "fallback" / "audit.json").read_text(encoding="utf-8"),
            )
            self.assertEqual(
                report["learned_effects"]["status"], "strict_fallback",
            )

    def test_report_metadata_failure_leaves_no_learned_pixels_or_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job = self._job(root)
            job = replace(
                job,
                lighting=replace(
                    job.lighting,
                    cast_shadow_mode="directional",
                    shadow_length_ratio=0.12,
                    shadow_opacity=0.40,
                ),
            )
            baseline = StrictCompositor().compose(
                job, root / "baseline", strict_audit=False,
            )
            provider = _LateReportFailureProvider()
            fallback = StrictCompositor(
                learned_effect_provider=provider,
            ).compose(job, root / "fallback_report", strict_audit=False)

            self.assertTrue(provider.saw_nonzero_strict_shadow)
            np.testing.assert_array_equal(fallback.final, baseline.final)
            self.assertFalse(
                any("learned_" in name for name in fallback.layer_names),
            )
            report = json.loads(
                (root / "fallback_report" / "audit.json").read_text(
                    encoding="utf-8",
                ),
            )["learned_effects"]
            self.assertEqual(report["status"], "strict_fallback")
            self.assertIn("deliberate report metadata failure", report["reason"])
            self.assertNotIn("products", report)
            self.assertNotIn("batch_metadata", report)

    def test_provider_receives_validated_scene_evidence_and_audit_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job = self._job(root)
            depth = np.linspace(
                0.0, 1.0, 9 * 12, dtype=np.float32,
            ).reshape(9, 12)
            normal = np.zeros((9, 12, 3), np.float32)
            normal[..., 1] = 0.6
            normal[..., 2] = 0.8
            np.save(root / "depth.npy", depth)
            np.save(root / "normal.npy", normal)
            provider = _EvidenceInspectingProvider()
            StrictCompositor(
                learned_effect_provider=provider,
                learned_effect_depth_path=root / "depth.npy",
                learned_effect_normal_path=root / "normal.npy",
            ).compose(job, root / "evidence", strict_audit=False)
            self.assertTrue(provider.called)
            self.assertEqual(provider.observed["depth_shape"], (72, 96))
            self.assertEqual(provider.observed["normal_shape"], (72, 96, 3))
            self.assertTrue(provider.observed["request_depth_readonly"])
            self.assertTrue(provider.observed["request_normal_readonly"])
            self.assertTrue(provider.observed["context_depth_readonly"])
            self.assertTrue(provider.observed["context_normal_readonly"])
            low, high = provider.observed["normal_norm_range"]
            self.assertAlmostEqual(low, 1.0, places=5)
            self.assertAlmostEqual(high, 1.0, places=5)
            report = json.loads(
                (root / "evidence" / "audit.json").read_text(encoding="utf-8"),
            )["learned_effects"]
            self.assertEqual(report["status"], "applied")
            evidence = report["scene_evidence"]
            self.assertEqual(evidence["status"], "validated")
            self.assertEqual(evidence["depth"]["source_shape"], [9, 12])
            self.assertEqual(evidence["depth"]["canvas_shape"], [72, 96])
            self.assertEqual(len(evidence["depth"]["file_sha256"]), 64)
            self.assertEqual(len(evidence["normal"]["canvas_array_sha256"]), 64)

    def test_provider_uses_geometry_evidence_bound_to_job(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job = self._job(root)
            depth = np.linspace(
                0.0, 1.0, 9 * 12, dtype=np.float32,
            ).reshape(9, 12)
            normal = np.zeros((9, 12, 3), np.float32)
            normal[..., 2] = 1.0
            np.save(root / "job_depth.npy", depth)
            np.save(root / "job_normal.npy", normal)
            job = replace(
                job,
                depth_map=root / "job_depth.npy",
                normal_map=root / "job_normal.npy",
            )
            provider = _EvidenceInspectingProvider()

            StrictCompositor(
                learned_effect_provider=provider,
            ).compose(job, root / "job_evidence", strict_audit=False)

            self.assertTrue(provider.called)
            self.assertEqual(provider.observed["depth_shape"], (72, 96))
            self.assertEqual(provider.observed["normal_shape"], (72, 96, 3))
            report = json.loads(
                (root / "job_evidence" / "audit.json").read_text(
                    encoding="utf-8",
                ),
            )["learned_effects"]["scene_evidence"]
            self.assertEqual(report["status"], "validated")
            self.assertEqual(
                Path(report["depth"]["path"]),
                (root / "job_depth.npy").resolve(),
            )

    def test_invalid_scene_evidence_falls_back_before_provider_call(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job = self._job(root)
            np.save(root / "bad_depth.npy", np.zeros((9, 12), np.float64))
            baseline = StrictCompositor().compose(
                job, root / "baseline", strict_audit=False,
            )
            provider = _EvidenceInspectingProvider()
            fallback = StrictCompositor(
                learned_effect_provider=provider,
                learned_effect_depth_path=root / "bad_depth.npy",
            ).compose(job, root / "fallback_evidence", strict_audit=False)
            self.assertFalse(provider.called)
            np.testing.assert_array_equal(fallback.final, baseline.final)
            report = json.loads(
                (root / "fallback_evidence" / "audit.json").read_text(
                    encoding="utf-8",
                ),
            )["learned_effects"]
            self.assertEqual(report["status"], "strict_fallback")
            self.assertEqual(report["scene_evidence"]["status"], "invalid")
            self.assertIn("dtype float32", report["reason"])

    def test_evidence_paths_are_not_loaded_without_a_provider(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job = self._job(root)
            result = StrictCompositor(
                learned_effect_depth_path=root / "missing_depth.npy",
                learned_effect_normal_path=root / "missing_normal.npy",
            ).compose(job, root / "no_provider", strict_audit=False)
            self.assertEqual(result.final.shape, (72, 96, 3))
            report = json.loads(
                (root / "no_provider" / "audit.json").read_text(encoding="utf-8"),
            )["learned_effects"]
            self.assertEqual(report["status"], "not_requested")


if __name__ == "__main__":
    unittest.main()
