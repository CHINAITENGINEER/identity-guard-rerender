from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
DEFAULT_RANDOM_INPUT = ROOT / "未命名文件夹 3"


@dataclass(frozen=True)
class BatchJob:
    name: str
    products: tuple[Path, Path, Path]
    background: Path


def natural_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", path.name)]


def image_files(folder: Path) -> list[Path]:
    return sorted(
        (path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=natural_key,
    )


def choose_folder(title: str, initial: Path) -> Path | None:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    selected = filedialog.askdirectory(title=title, initialdir=str(initial))
    root.destroy()
    return Path(selected).resolve() if selected else None


def structured_jobs(folder: Path) -> tuple[list[BatchJob], list[str]]:
    jobs: list[BatchJob] = []
    warnings: list[str] = []
    candidates = [child for child in folder.iterdir() if child.is_dir()]
    # 也允许用户直接选中只有一组图片的文件夹。
    if len(image_files(folder)) >= 4:
        candidates.insert(0, folder)
    for child in sorted(candidates, key=natural_key):
        images = image_files(child)
        if len(images) < 4:
            warnings.append(f"跳过 {child.name}：只有 {len(images)} 张图片，至少需要 4 张")
            continue
        jobs.append(BatchJob(child.name, tuple(images[:3]), images[3]))
        if len(images) > 4:
            warnings.append(f"{child.name}：已忽略第 5 张及其后的 {len(images) - 4} 张图片")
    return jobs, warnings


def find_named_subfolder(folder: Path, preferred: Iterable[str], kind: str) -> Path:
    children = [child for child in folder.iterdir() if child.is_dir()]
    by_name = {child.name.casefold(): child for child in children}
    for name in preferred:
        match = by_name.get(name.casefold())
        if match:
            return match
    keyword = "物" if kind == "product" else "背景"
    matches = [child for child in children if keyword in child.name]
    if len(matches) == 1:
        return matches[0]
    expected = "物品图" if kind == "product" else "背景图"
    raise ValueError(f"{folder} 中没有找到唯一的“{expected}”子文件夹")


def random_jobs(folder: Path, seed: int) -> tuple[list[BatchJob], list[str]]:
    product_folder = find_named_subfolder(folder, ("物品图", "物品", "products", "product"), "product")
    background_folder = find_named_subfolder(folder, ("背景图", "背景", "backgrounds", "background"), "background")
    products = image_files(product_folder)
    backgrounds = image_files(background_folder)
    if not products:
        raise ValueError(f"{product_folder} 中没有图片")
    if not backgrounds:
        raise ValueError(f"{background_folder} 中没有图片")

    rng = random.Random(seed)
    rng.shuffle(products)
    background_cycle = list(backgrounds)
    rng.shuffle(background_cycle)
    jobs: list[BatchJob] = []
    warnings: list[str] = []
    group_count = (len(products) + 2) // 3
    for index in range(group_count):
        group = products[index * 3 : index * 3 + 3]
        if len(group) < 3:
            padding_pool = products[: index * 3] or products
            group.extend(rng.choice(padding_pool) for _ in range(3 - len(group)))
            warnings.append(f"最后一组不足 3 张，已补用 {3 - (len(products) % 3)} 张已遍历物品")
        if index and index % len(background_cycle) == 0:
            rng.shuffle(background_cycle)
        background = background_cycle[index % len(background_cycle)]
        jobs.append(BatchJob(f"random_{index + 1:04d}", tuple(group), background))
    return jobs, warnings


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def product_payload(index: int, source: Path, cutout: dict[str, Any]) -> dict[str, Any]:
    locator = cutout.get("subject_locator") or {}
    label = str(locator.get("subject_label") or "").strip()
    semantic = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_") or "decor_object"
    semantic_verified = bool(locator.get("semantic_verified") and locator.get("accepted") and not locator.get("fallback_used"))
    return {
        "id": f"product_{index + 1}",
        "source": str(source),
        "rgba": str(cutout["rgba"]),
        "center_x": 0.25 + index * 0.25,
        "baseline_y": 0.67,
        "height_ratio": 0.16,
        "z_order": index + 1,
        "placement_mode": "upright",
        "physical_type": "rigid_object",
        "photographed_pose": "unknown",
        "pose_confidence": 0.0,
        "semantic_category": semantic,
        "subject_label": label,
        "subject_included_parts": [str(value) for value in locator.get("included_parts", [])],
        "subject_locator_confidence": float(locator.get("confidence") or 0.0),
        "subject_semantic_verified": semantic_verified,
        "estimated_real_height_cm": 20.0,
        "estimated_height_cm_min": 15.0,
        "estimated_height_cm_max": 30.0,
        "footprint_width_cm": 12.0,
        "size_confidence": 0.0,
        "viewpoint_compatibility": 0.8,
        "source_light_compatibility": 0.5,
        "visual_salience": 0.5,
        "material_mode": "opaque",
        "material_confidence": 0.0,
        "transmission_strength": 0.0,
        "receiver_width_or_diameter_cm": 100.0,
        "receiver_size_confidence": 0.0,
        "identity_status": "unknown",
        "identity_mask": None,
        "rerender_policy": "auto",
    }


def reflection_opacity(scene: dict[str, Any]) -> float:
    reflectivity = float(scene.get("reflectivity") or 0.0)
    if scene.get("surface_material") == "transparent_glass":
        return min(0.38, max(0.24, 0.20 + reflectivity * 0.20))
    return min(0.30, max(0.08, 0.055 + reflectivity * 0.28))


def process_job(job: BatchJob, output_dir: Path, strict_audit: bool) -> dict[str, Any]:
    # 延迟导入：--dry-run 只核对目录，不加载模型。
    from strict_compositor.ui import ComposeUI, LayoutUI, analyze_scene_api, auto_layout, compose, segment

    started = time.time()
    print(f"\n[{job.name}] 1/4 精确分析桌面：{job.background.name}", flush=True)
    scene = analyze_scene_api({"background": str(job.background), "deep": True})

    products: list[dict[str, Any]] = []
    cutouts: list[dict[str, Any]] = []
    for index, source in enumerate(job.products):
        print(f"[{job.name}] 2/4 自动抠图 {index + 1}/3：{source.name}", flush=True)
        cutout = segment({"id": f"{job.name}_product_{index + 1}", "source": str(source), "quality": "quality", "roi": None})
        # 与工作台“一键准备”保持同一准入规则：明确 reject 才阻止；
        # review 仍保留到合成结果中供人工抽查，避免批处理因保守置信度全停。
        if str((cutout.get("alpha_quality") or {}).get("verdict")) == "reject":
            reason = (cutout.get("alpha_quality") or {}).get("reason") or "自动抠图没有通过完整性门禁"
            raise RuntimeError(f"商品 {index + 1} 抠图未通过：{reason}")
        cutouts.append(cutout)
        products.append(product_payload(index, source, cutout))

    layout_data = {
        "background": str(job.background),
        "products": products,
        "support_polygon": scene["support_polygon"],
        "support_mask_path": scene.get("support_mask_path"),
        "depth_map_path": scene.get("depth_map_path"),
        "normal_map_path": scene.get("normal_map_path"),
        "protected_polygons": scene.get("protected_polygons") or [],
        "shadow_direction": scene.get("shadow_direction") or [0.75, -0.66],
        "shadow_length_ratio": float(scene.get("shadow_length_ratio") or 0.04),
        "shadow_softness_ratio": float(scene.get("shadow_softness_ratio") or 0.15),
        "receiver_width_or_diameter_cm": (scene.get("advisor") or {}).get("receiver_width_or_diameter_cm"),
        "receiver_size_confidence": float((scene.get("advisor") or {}).get("receiver_size_confidence") or 0.0),
    }
    print(f"[{job.name}] 3/4 智能摆位", flush=True)
    layout = auto_layout(LayoutUI(**layout_data))
    products = layout["products"]

    compose_data = {
        **layout_data,
        "products": products,
        "support_polygon": layout.get("support_polygon") or scene["support_polygon"],
        "protected_polygons": layout.get("protected_polygons") or scene.get("protected_polygons") or [],
        "shadow_opacity": float(scene.get("shadow_opacity") or 0.23),
        "reflection_enabled": bool(scene.get("reflection_recommended")),
        "reflection_opacity": reflection_opacity(scene),
        "cast_shadow_mode": scene.get("cast_shadow_mode") or "directional",
        "surface_material": scene.get("surface_material") or "unknown",
        "surface_reflectivity": float(scene.get("reflectivity") or 0.0),
        "surface_roughness": float(scene.get("surface_roughness") or 0.55),
        "direct_light_fraction": float(scene.get("direct_light_fraction") or 0.55),
        "contact_ao_strength": float(scene.get("contact_ao_strength") or 1.0),
        "auto_repair": True,
        "strict_audit": strict_audit,
        "output_long_edge": 3840,
        "learned_effects_enabled": False,
        "hybrid_rerender_enabled": False,
        "refinement_pass": 0,
    }
    # LayoutUI 专用字段不是 ComposeUI 字段。
    compose_data.pop("receiver_width_or_diameter_cm", None)
    compose_data.pop("receiver_size_confidence", None)
    print(f"[{job.name}] 4/4 生成 4K 合成", flush=True)
    result = compose(ComposeUI(**compose_data))

    job_output = output_dir / job.name
    job_output.mkdir(parents=True, exist_ok=True)
    final_copy = job_output / "final.png"
    shutil.copy2(result["final_path"], final_copy)
    details = {
        "name": job.name,
        "status": "success",
        "products": [str(path) for path in job.products],
        "background": str(job.background),
        "cutouts": [str(item["rgba"]) for item in cutouts],
        "layout_verdict": layout.get("layout_verdict"),
        "layout_reasons": layout.get("layout_reasons") or [],
        "final": str(final_copy),
        "run": result["run"],
        "output_dimensions": result.get("output_dimensions"),
        "pixel_verdict": result.get("pixel_verdict"),
        "delivery_verdict": result.get("delivery_verdict"),
        "elapsed_seconds": round(time.time() - started, 2),
    }
    atomic_write_json(job_output / "result.json", details)
    return details


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="商品图片自动抠图、智能摆位与 4K 合成")
    parser.add_argument("--mode", choices=("structured", "random"), required=True)
    parser.add_argument("--input", type=Path, help="输入根目录；省略时弹出文件夹选择窗口")
    parser.add_argument("--output", type=Path, help="输出目录；默认在输入目录旁创建带时间戳的文件夹")
    parser.add_argument("--seed", type=int, default=20260814, help="随机组合种子，默认 20260814")
    parser.add_argument("--strict-audit", action="store_true", help="启用正式末端审计（会明显增加耗时和云端调用）")
    parser.add_argument("--dry-run", action="store_true", help="只列出组合，不加载模型、不合成")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    initial = DEFAULT_RANDOM_INPUT if args.mode == "random" else ROOT / "未命名文件夹"
    selected = args.input.resolve() if args.input else choose_folder(
        "选择结构化图片根目录" if args.mode == "structured" else "选择包含“物品图/背景图”的文件夹",
        initial,
    )
    if selected is None:
        print("已取消选择。")
        return 0
    if not selected.is_dir():
        raise SystemExit(f"输入目录不存在：{selected}")

    jobs, warnings = structured_jobs(selected) if args.mode == "structured" else random_jobs(selected, args.seed)
    for warning in warnings:
        print(f"提示：{warning}")
    if not jobs:
        raise SystemExit("没有找到可处理的图片组合。")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = (args.output.resolve() if args.output else selected.parent / f"自动合成_{args.mode}_{stamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "mode": args.mode,
        "input": str(selected),
        "output": str(output_dir),
        "seed": args.seed if args.mode == "random" else None,
        "strict_audit": bool(args.strict_audit),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "warnings": warnings,
        "jobs": [],
    }
    print(f"共发现 {len(jobs)} 组合，输出到：{output_dir}")
    for index, job in enumerate(jobs, start=1):
        print(f"{index:04d}  背景={job.background.name}  物品={', '.join(path.name for path in job.products)}")
    atomic_write_json(output_dir / "batch_manifest.json", manifest)
    if args.dry_run:
        print("\n目录检查完成（dry-run），尚未执行抠图或合成。")
        return 0

    for index, job in enumerate(jobs, start=1):
        print(f"\n========== {index}/{len(jobs)}：{job.name} ==========")
        try:
            record = process_job(job, output_dir, args.strict_audit)
            print(f"完成：{record['final']}（{record['elapsed_seconds']} 秒）")
        except Exception as exc:
            record = {
                "name": job.name,
                "status": "failed",
                "products": [str(path) for path in job.products],
                "background": str(job.background),
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            print(f"失败并继续下一组：{exc}")
        manifest["jobs"].append(record)
        atomic_write_json(output_dir / "batch_manifest.json", manifest)

    successes = sum(item["status"] == "success" for item in manifest["jobs"])
    failures = len(manifest["jobs"]) - successes
    manifest["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    manifest["summary"] = {"total": len(jobs), "success": successes, "failed": failures}
    atomic_write_json(output_dir / "batch_manifest.json", manifest)
    print(f"\n全部处理结束：成功 {successes} 组，失败 {failures} 组。")
    print(f"批次清单：{output_dir / 'batch_manifest.json'}")
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
