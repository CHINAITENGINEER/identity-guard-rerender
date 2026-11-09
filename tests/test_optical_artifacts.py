from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from strict_compositor.contracts import Job
from strict_compositor.image_io import save_png
from strict_compositor.pipeline import StrictCompositor


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


class OpticalMatteArtifactTests(unittest.TestCase):
    def test_verified_clear_glass_writes_exact_roi_audit_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            yy, xx = np.indices((120, 180), dtype=np.uint16)
            background = np.stack(
                (
                    80 + xx // 3,
                    105 + yy // 4,
                    135 + (xx + yy) // 8,
                ),
                axis=-1,
            ).clip(0, 255).astype(np.uint8)

            clear = np.zeros((38, 28, 4), np.uint8)
            clear[3:35, 5:23, :3] = [174, 161, 148]
            clear[3:35, 5:23, 3] = 168
            clear[3:6, 5:23, 3] = 255
            clear[32:35, 5:23, 3] = 255
            clear[5:33, 5:7, 3] = 235
            clear[5:33, 21:23, 3] = 235
            clear[6:32, 10:12, :3] = [210, 208, 204]
            clear[6:32, 10:12, 3] = 210

            opaque = np.zeros((24, 20, 4), np.uint8)
            opaque[2:22, 3:17, :3] = [180, 50, 30]
            opaque[2:22, 3:17, 3] = 255

            save_png(root / "background.png", background)
            save_png(root / "clear.png", clear)
            save_png(root / "opaque.png", opaque)
            job = Job.from_dict(
                {
                    "background": "background.png",
                    "support_polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
                    "lighting": {
                        "shadow_direction": [0.8, 0.45],
                        "shadow_length_ratio": 0.05,
                        "cast_shadow_mode": "ambient_only",
                    },
                    "products": [
                        {
                            "id": "clear/glass:1",
                            "rgba": "clear.png",
                            "homography": [
                                [1, 0, 35], [0, 1, 48], [0, 0, 1],
                            ],
                            "z_order": 0,
                            "material_mode": "clear_glass",
                            "material_confidence": 0.93,
                            "transmission_strength": 0.90,
                        },
                        {
                            "id": "opaque",
                            "rgba": "opaque.png",
                            "homography": [
                                [1, 0, 115], [0, 1, 60], [0, 0, 1],
                            ],
                            "z_order": 1,
                        },
                    ],
                },
                root,
            )

            StrictCompositor().compose(job, root / "out")

            report = json.loads(
                (root / "out" / "audit.json").read_text(encoding="utf-8"),
            )
            clear_report = next(
                item for item in report["products"]
                if item["id"] == "clear/glass:1"
            )
            opaque_report = next(
                item for item in report["products"]
                if item["id"] == "opaque"
            )
            self.assertNotIn("optical_matte", opaque_report)

            optical = clear_report["optical_matte"]
            self.assertEqual(
                optical["schema_version"], "optical-matte-audit-v1",
            )
            self.assertEqual(optical["format"], "numpy-npz-numeric-only")
            self.assertEqual(len(optical["bbox"]), 4)
            x0, y0, x1, y1 = optical["bbox"]
            self.assertEqual(optical["roi_shape"], [y1 - y0, x1 - x0])
            self.assertGreaterEqual(optical["confidence"], 0.90)
            self.assertTrue(optical["provenance"])

            npz_path = root / "out" / optical["path"]
            metadata_path = root / "out" / optical["metadata_path"]
            self.assertTrue(npz_path.is_file())
            self.assertTrue(metadata_path.is_file())
            self.assertEqual(optical["sha256"], _sha256(npz_path))
            self.assertEqual(
                optical["metadata_sha256"], _sha256(metadata_path),
            )
            self.assertNotIn("clear/glass:1", npz_path.name)

            required = {
                "coverage_alpha",
                "opaque_identity_mask",
                "transmission_rgb",
                "refraction_flow",
                "specular_rgb",
                "shadow_transmittance",
                "bbox",
                "confidence",
            }
            with np.load(npz_path, allow_pickle=False) as archive:
                self.assertTrue(required.issubset(archive.files))
                for name in archive.files:
                    self.assertNotEqual(archive[name].dtype.kind, "O")
                roi_shape = (y1 - y0, x1 - x0)
                self.assertEqual(archive["coverage_alpha"].shape, roi_shape)
                self.assertEqual(
                    archive["transmission_rgb"].shape, (*roi_shape, 3),
                )
                self.assertEqual(
                    archive["refraction_flow"].shape, (*roi_shape, 2),
                )
                np.testing.assert_array_equal(
                    archive["bbox"], np.asarray(optical["bbox"], np.int32),
                )
                self.assertEqual(archive["coverage_alpha"].dtype, np.float32)

            sidecar = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(sidecar["bbox"], optical["bbox"])
            self.assertEqual(sidecar["content_sha256"], optical["content_sha256"])
            self.assertEqual(
                set(sidecar["arrays"]), set(optical["arrays"]),
            )


if __name__ == "__main__":
    unittest.main()
