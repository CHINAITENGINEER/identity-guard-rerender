from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from strict_compositor.heldout_manifest import (
    BACKGROUND_DIRECTORY,
    PRODUCT_DIRECTORY,
    HoldoutLeakageError,
    HoldoutManifestError,
    assert_no_holdout_leakage,
    build_holdout_manifest,
    find_holdout_leakage,
    freeze_holdout_manifest,
    groups_from_holdout_manifest,
    validate_holdout_manifest,
)


PRODUCT_NAMES = [
    "DSC09625.JPG", "DSC09626.JPG", "DSC09627.JPG", "DSC09628.JPG",
    "DSC09629.JPG", "DSC09630.JPG", "DSC09631.JPG", "DSC09632.JPG",
    "DSC09633.JPG", "DSC09634.JPG", "DSC09734.JPG", "DSC09735.JPG",
    "DSC09737.JPG", "DSC09738.JPG", "DSC09741.JPG", "DSC09743.JPG",
    "DSC09744.JPG", "DSC09745.JPG", "DSC09746.JPG", "DSC09750.JPG",
    "DSC09782.JPG", "DSC09783.JPG", "DSC09784.JPG", "DSC09785.JPG",
    "DSC09786.JPG", "DSC09787.JPG", "DSC09788.JPG", "DSC09789.JPG",
    "DSC09790.JPG", "DSC09791.JPG",
]
BACKGROUND_NAMES = [
    "DSC09515.JPG", "DSC09516.JPG", "DSC09517.JPG", "DSC09519.JPG",
    "DSC09520.JPG", "DSC09522.JPG", "DSC09523.JPG", "DSC09524.JPG",
    "DSC09525.JPG", "DSC09527.JPG",
]


class HeldoutManifestTests(unittest.TestCase):
    def _source(self, root: Path) -> Path:
        source = root / "未命名文件夹 3"
        products = source / PRODUCT_DIRECTORY
        backgrounds = source / BACKGROUND_DIRECTORY
        products.mkdir(parents=True)
        backgrounds.mkdir(parents=True)
        for index, name in enumerate(PRODUCT_NAMES, start=1):
            (products / name).write_bytes(f"raw-product-{index}".encode())
        for index, name in enumerate(BACKGROUND_NAMES, start=1):
            (backgrounds / name).write_bytes(f"raw-background-{index}".encode())
        # Finder metadata is ignored and never enters a case.
        (source / ".DS_Store").write_bytes(b"metadata")
        (products / ".DS_Store").write_bytes(b"metadata")
        return source

    def test_freezes_exact_ten_case_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = self._source(Path(directory))
            manifest = build_holdout_manifest(source)

            self.assertEqual(
                manifest["counts"], {"cases": 10, "products": 30, "backgrounds": 10},
            )
            self.assertEqual(manifest["cases"][0]["case_id"], "set_b_01")
            self.assertEqual(
                manifest["cases"][0]["background"]["filename"], "DSC09515.JPG",
            )
            self.assertEqual(
                [asset["filename"] for asset in manifest["cases"][0]["products"]],
                ["DSC09625.JPG", "DSC09626.JPG", "DSC09627.JPG"],
            )
            self.assertEqual(
                manifest["cases"][-1]["background"]["filename"], "DSC09527.JPG",
            )
            self.assertEqual(
                [asset["filename"] for asset in manifest["cases"][-1]["products"]],
                ["DSC09789.JPG", "DSC09790.JPG", "DSC09791.JPG"],
            )
            self.assertTrue(all(case["split"] == "blind_holdout" for case in manifest["cases"]))
            self.assertFalse(manifest["policy"]["allow_parameter_tuning"])
            self.assertFalse(manifest["policy"]["decode_existing_composites"])
            self.assertNotIn(str(source.resolve()), json.dumps(manifest, ensure_ascii=False))

    def test_freeze_is_deterministic_and_write_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._source(root)
            output = root / "frozen.json"
            first = freeze_holdout_manifest(source, output)
            second = freeze_holdout_manifest(source, output)
            self.assertEqual(first, second)
            self.assertEqual(validate_holdout_manifest(source, output)["verdict"], "pass")

            (source / PRODUCT_DIRECTORY / PRODUCT_NAMES[0]).write_bytes(b"changed")
            with self.assertRaisesRegex(HoldoutManifestError, "changed"):
                validate_holdout_manifest(source, output)
            with self.assertRaises(HoldoutManifestError):
                freeze_holdout_manifest(source, output)

    def test_rejects_added_or_missing_raw_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._source(root)
            manifest = freeze_holdout_manifest(source, root / "frozen.json")
            (source / PRODUCT_DIRECTORY / "EXTRA.JPG").write_bytes(b"unexpected")
            with self.assertRaisesRegex(HoldoutManifestError, "exactly 30"):
                validate_holdout_manifest(source, manifest)

    def test_detects_same_path_and_copied_byte_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._source(root)
            manifest = build_holdout_manifest(source)
            original = source / PRODUCT_DIRECTORY / PRODUCT_NAMES[0]
            copied = root / "training" / "renamed.jpg"
            copied.parent.mkdir()
            copied.write_bytes(original.read_bytes())

            collisions = find_holdout_leakage([original, copied], source, manifest)
            self.assertEqual({item["reason"] for item in collisions}, {"same_path", "same_sha256"})
            with self.assertRaises(HoldoutLeakageError):
                assert_no_holdout_leakage([copied.parent], source, manifest)

    def test_group_loader_returns_only_frozen_raw_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._source(root)
            manifest = build_holdout_manifest(source)
            groups = groups_from_holdout_manifest(source, manifest)
            self.assertEqual(len(groups), 10)
            self.assertEqual(len(groups[0]["products"]), 3)
            for group in groups:
                self.assertTrue(Path(group["background"]).is_file())
                self.assertTrue(all(Path(path).is_file() for path in group["products"]))
                self.assertNotIn("composite", json.dumps(group).casefold())

    def test_checked_in_private_baseline_matches_when_assets_are_present(self) -> None:
        project = Path(__file__).resolve().parents[1]
        source = project / "未命名文件夹 3"
        frozen = project / "tests" / "fixtures" / "heldout_set_b_manifest.json"
        if not source.is_dir():
            self.skipTest("private holdout assets are intentionally not distributed")
        report = validate_holdout_manifest(source, frozen)
        self.assertEqual(report["verdict"], "pass")
        self.assertEqual(report["counts"], {"cases": 10, "products": 30, "backgrounds": 10})


if __name__ == "__main__":
    unittest.main()
