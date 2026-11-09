from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from strict_compositor.route_integrity import (
    CUTOUT_REUSE_SCHEMA,
    audit_render_route,
    reusable_cutout_provenance,
)
from strict_compositor.quality import sha256_file


class RenderRouteIntegrityTests(unittest.TestCase):
    def test_clear_glass_benchmark_rejects_silent_strict_fallback(self) -> None:
        report = audit_render_route(
            [{
                "id": "product_1",
                "material_mode": "clear_glass",
                "rerender_policy": "auto",
            }],
            {"hybrid_render": {"status": "not_requested", "eligible_pixels": 500}},
        )
        self.assertEqual(report["verdict"], "reject")
        self.assertIn(
            "required_identity_safe_hybrid_route_not_applied", report["reasons"],
        )

    def test_identity_safe_hybrid_route_passes(self) -> None:
        report = audit_render_route(
            [{
                "id": "product_1",
                "material_mode": "clear_glass",
                "rerender_policy": "auto",
            }],
            {"hybrid_render": {
                "status": "applied",
                "effective_pixels": 500,
                "identity_changed_pixels": 0,
                "outside_changed_pixels": 0,
                "max_outside_delta": 0,
            }},
        )
        self.assertEqual(report["verdict"], "pass")

    def test_opaque_route_does_not_require_hybrid(self) -> None:
        report = audit_render_route(
            [{"id": "product_1", "material_mode": "opaque_glossy"}],
            {"hybrid_render": {"status": "not_requested"}},
        )
        self.assertEqual(report["verdict"], "pass")


class CutoutReuseProvenanceTests(unittest.TestCase):
    def test_old_unbound_cache_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source.jpg"
            rgba = root / "rgba.png"
            source.write_bytes(b"source")
            rgba.write_bytes(b"rgba")
            report = reusable_cutout_provenance([source], [rgba], None)
            self.assertEqual(report["verdict"], "reject")
            self.assertIn(
                "missing_or_unsupported_cutout_reuse_manifest", report["reasons"],
            )

    def test_hash_bound_human_approved_cache_passes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source.jpg"
            rgba = root / "rgba.png"
            source.write_bytes(b"source")
            rgba.write_bytes(b"rgba")
            manifest = {
                "schema_version": CUTOUT_REUSE_SCHEMA,
                "products": [{
                    "source_sha256": sha256_file(source),
                    "rgba_sha256": sha256_file(rgba),
                    "semantic_target": "white ceramic cup with complete heart handle",
                    "target_verdict": "pass",
                }],
            }
            report = reusable_cutout_provenance([source], [rgba], manifest)
            self.assertEqual(report["verdict"], "pass")


if __name__ == "__main__":
    unittest.main()
