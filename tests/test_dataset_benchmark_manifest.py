from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from tools.build_dataset_benchmark_manifest import build_manifest


FIELDS = [
    "image_id", "original_filename", "source_type", "source_path",
    "copied_path", "sha256", "duplicate_group_id", "main_object",
    "material", "transparency", "training_eligible", "surface_type",
    "surface_reflectivity", "quality_status", "license_status",
    "reference_role", "reference_reason", "synthesis_value",
    "aesthetic_score", "cutout_difficulty_score", "cutout_edge_complexity",
    "reflection_type", "refraction", "composition", "object_count",
    "issue_tags",
]


class DatasetBenchmarkManifestTests(unittest.TestCase):
    def _dataset(self, root: Path, sessions: int = 1) -> Path:
        dataset = root / "dataset"
        index = dataset / "09_索引与标签" / "all_images_index.csv"
        index.parent.mkdir(parents=True)
        rows: list[dict[str, str]] = []
        for session in range(1, sessions + 1):
            parent = root / "source" / str(session)
            for number in range(1, 6):
                rows.append({
                    "image_id": f"L{session:03d}{number}",
                    "original_filename": f"{number}.JPG" if number < 5 else "5.jpg",
                    "source_type": "本地图片",
                    "source_path": str(parent / (f"{number}.JPG" if number < 5 else "5.jpg")),
                    "copied_path": "unused",
                    "sha256": f"local-{session}-{number}",
                    "main_object": "背景承载面" if number == 4 else "商品",
                    "material": "玻璃" if number == 1 else "陶瓷",
                    "transparency": "高度透明" if number == 1 else "不透明",
                    "training_eligible": "False",
                    "surface_reflectivity": "高反光" if number == 4 else "低反光",
                    "quality_status": "有效",
                    "license_status": "本地授权未提供",
                    "reference_role": "优秀参考" if number in {4, 5} else "普通样本",
                    "reference_reason": "优秀维度：场景化构图",
                    "synthesis_value": "5",
                    "aesthetic_score": "4",
                    "cutout_difficulty_score": "3",
                    "cutout_edge_complexity": "中",
                    "reflection_type": "清晰倒影" if number in {4, 5} else "无倒影",
                    "refraction": "明显" if number == 1 else "无",
                    "composition": "多商品组合;前后错落" if number == 5 else "居中构图",
                    "object_count": "多个主体" if number == 5 else "单一主体",
                    "issue_tags": "",
                })
        # Two anchors share a near-duplicate group.  At most one may appear in
        # the entire reference manifest, even if both rank highly.
        for index_number in range(1, 4):
            rows.append({
                "image_id": f"W{index_number:06d}",
                "original_filename": f"web-{index_number}.jpg",
                "source_type": "网络图片",
                "source_path": f"web-{index_number}.jpg",
                "copied_path": f"copy-{index_number}.jpg",
                "sha256": f"web-{index_number}",
                "duplicate_group_id": "near-one" if index_number < 3 else "",
                "main_object": "玻璃器皿",
                "material": "透明玻璃;镜面金属",
                "transparency": "高度透明",
                "training_eligible": "True",
                "surface_reflectivity": "高反光",
                "quality_status": "有效",
                "license_status": "许可已记录",
                "reference_role": "优秀参考",
                "reference_reason": "优秀维度：透明物体光影",
                "synthesis_value": "5",
                "aesthetic_score": "5",
                "cutout_difficulty_score": "5",
                "cutout_edge_complexity": "高",
                "reflection_type": "镜面倒影",
                "refraction": "明显",
                "composition": "多商品组合;前后错落",
                "object_count": "多个主体",
                "issue_tags": "",
            })
        with index.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        (dataset / "执行摘要.json").write_text(
            json.dumps({"generated_at": "now", "final_unique_images": len(rows)}),
            encoding="utf-8",
        )
        return dataset

    def test_builds_input_only_cases_and_hides_reference_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = self._dataset(root, sessions=10)
            output = root / "out" / "manifest.json"
            manifest = build_manifest(dataset, output)

            self.assertEqual(len(manifest["strict_composition_cases"]), 10)
            case = manifest["strict_composition_cases"][0]
            self.assertEqual(len(case["products"]), 3)
            self.assertIn("source_path", case["background"])
            self.assertNotIn("source_path", case["human_reference"])
            self.assertFalse(case["human_reference"]["is_pixel_ground_truth"])
            self.assertEqual(
                [item["split"] for item in manifest["strict_composition_cases"]],
                [
                    "development", "validation", "development", "development",
                    "development", "validation", "blind_holdout", "development",
                    "blind_holdout", "development",
                ],
            )
            self.assertTrue(output.exists())
            self.assertTrue(manifest["dataset"]["producer_summary_consistent"])

    def test_near_duplicate_anchor_group_is_used_only_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = self._dataset(root)
            manifest = build_manifest(dataset, root / "manifest.json")
            ids = [
                item["image_id"]
                for items in manifest["visual_reference_anchors"].values()
                for item in items
            ]
            self.assertLessEqual(sum(value in {"W000001", "W000002"} for value in ids), 1)

    def test_refuses_to_write_inside_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = self._dataset(root)
            with self.assertRaises(ValueError):
                build_manifest(dataset, dataset / "generated" / "manifest.json")


if __name__ == "__main__":
    unittest.main()
