from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "_review"
REVIEW.mkdir(exist_ok=True)
E2E_TEMP = ROOT / "ui_workspace" / "e2e_temp"
E2E_TEMP.mkdir(parents=True, exist_ok=True)
# Keep browser profiles and temporary data inside the local workspace.
os.environ["TEMP"] = str(E2E_TEMP)
os.environ["TMP"] = str(E2E_TEMP)

from playwright.sync_api import sync_playwright  # noqa: E402


PLACEMENT_EXPRESSION = (
    "products.map(p => ({id:p.id,center_x:p.center_x,"
    "baseline_y:p.baseline_y,height_ratio:p.height_ratio,z_order:p.z_order}))"
)


def asset_paths() -> tuple[Path, list[tuple[Path, list[float]]]]:
    asset_root = next(
        path
        for path in ROOT.iterdir()
        if path.is_dir() and path.name.endswith(" 3")
    )
    folders = sorted(
        [path for path in asset_root.iterdir() if path.is_dir()],
        key=lambda path: len(list(path.glob("*.JPG"))),
        reverse=True,
    )
    products, backgrounds = folders[0], folders[1]
    return (
        backgrounds / "DSC09516.JPG",
        [
            # Rigid products with viewpoints compatible with the oblique desk.
            # The ROI is deliberately padded; it selects the intended object
            # but never clips the product or asks a weaker segmentation model
            # to repair a tight crop.
            (products / "DSC09634.JPG", [0.33, 0.12, 0.78, 0.91]),
            (products / "DSC09744.JPG", [0.25, 0.16, 0.80, 0.83]),
            (products / "DSC09784.JPG", [0.31, 0.04, 0.73, 0.81]),
        ],
    )


def wait_idle(page, timeout: int = 240_000) -> None:
    page.wait_for_function(
        "() => !document.querySelector('#status').classList.contains('busy') && "
        "document.body.getAttribute('aria-busy') !== 'true'",
        timeout=timeout,
    )


def placements(page) -> list[dict[str, object]]:
    return page.evaluate(PLACEMENT_EXPRESSION)


def write_report(report: dict[str, object]) -> None:
    (REVIEW / "ui_v320_e2e_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def assert_placements_equal(
    expected: list[dict[str, object]],
    actual: list[dict[str, object]],
    tolerance: float = 1e-9,
) -> None:
    if len(expected) != len(actual):
        raise AssertionError(f"placement count changed: {len(expected)} != {len(actual)}")
    for before, after in zip(expected, actual):
        if before["id"] != after["id"]:
            raise AssertionError(f"product order changed: {before['id']} != {after['id']}")
        for field in ("center_x", "baseline_y", "height_ratio"):
            delta = abs(float(before[field]) - float(after[field]))
            if delta > tolerance:
                raise AssertionError(
                    f"{before['id']} {field} was overwritten: "
                    f"{before[field]} -> {after[field]}"
                )
        if int(before["z_order"]) != int(after["z_order"]):
            raise AssertionError(
                f"{before['id']} z_order was overwritten: "
                f"{before['z_order']} -> {after['z_order']}"
            )


def manually_adjust_safe_product(page) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    before = placements(page)
    # Product 2 is the middle/free-space anchor in the deterministic three-
    # product layout. Product 1 can legitimately sit beside a protected chair
    # projection, so a blind screen-space nudge there would test an illegal
    # collision instead of testing persistence of a valid manual transform.
    adjusted_index = 1
    item = page.locator("#overlay .item").nth(adjusted_index)
    item.scroll_into_view_if_needed()
    box = item.bounding_box()
    if not box:
        raise RuntimeError("first product has no display box")
    page.mouse.move(box["x"] + box["width"] * 0.55, box["y"] + box["height"] * 0.55)
    page.mouse.down()
    # Move slightly toward the table interior; the change is deliberately
    # small so the placement remains legal when autoRepair is disabled.
    page.mouse.move(
        box["x"] + box["width"] * 0.55 + 9,
        box["y"] + box["height"] * 0.55 - 4,
        steps=8,
    )
    page.mouse.up()

    handle = page.locator("#overlay .item .handle").nth(adjusted_index)
    handle.scroll_into_view_if_needed()
    handle_box = handle.bounding_box()
    if not handle_box:
        raise RuntimeError("first product resize handle has no display box")
    page.mouse.move(handle_box["x"] + 4, handle_box["y"] + 4)
    page.mouse.down()
    page.mouse.move(handle_box["x"] + 4, handle_box["y"] - 6, steps=8)
    page.mouse.up()
    after = placements(page)
    if abs(float(after[adjusted_index]["center_x"]) - float(before[adjusted_index]["center_x"])) < 1e-4:
        raise AssertionError("manual drag did not change center_x")
    if abs(float(after[adjusted_index]["height_ratio"]) - float(before[adjusted_index]["height_ratio"])) < 1e-4:
        raise AssertionError("manual resize did not change height_ratio")
    return before, after


def main() -> None:
    background, product_items = asset_paths()
    report: dict[str, object] = {
        "background": str(background),
        "products": [str(item[0]) for item in product_items],
        "steps": [],
    }
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=os.environ.get("COMPOSITOR_BROWSER_PATH") or None,
            headless=True,
        )
        page = browser.new_page(viewport={"width": 1720, "height": 1050})
        page_errors: list[str] = []
        page.on("console", lambda message: print("browser:", message.type, message.text))
        page.on("pageerror", lambda error: page_errors.append(str(error)))
        page.goto("http://127.0.0.1:8765", wait_until="networkidle")
        page.wait_for_function(
            "() => document.querySelector('#cloudState').textContent.trim() && "
            "!document.querySelector('#cloudState').textContent.includes('正在读取')"
        )
        page.screenshot(path=str(REVIEW / "ui_v320_initial.png"), full_page=True)
        cloud_state = page.locator("#cloudState").text_content() or ""
        report["steps"].append({"initial_cloud_state": cloud_state})

        page.locator("#background").fill(str(background))
        page.get_by_role("button", name="载入", exact=True).click()
        if not page.locator("#background").is_disabled():
            raise AssertionError("background input stayed enabled during scene analysis")
        page.wait_for_function("() => typeof deepSceneReady !== 'undefined' && deepSceneReady", timeout=300_000)
        wait_idle(page)
        report["steps"].append({
            "scene": page.locator("#sceneQuality").inner_text(),
            "surface": page.locator("#surfaceBadge").inner_text(),
        })
        print("scene ready")

        # A changed path must invalidate the loaded scene immediately. Without
        # this gate the UI could submit a new background with stale geometry.
        page.locator("#background").fill(str(background) + ".changed")
        invalidated = page.evaluate(
            "({backgroundInfo,scene,deepSceneReady,composeDisabled:"
            "document.querySelector('#composeButton').disabled})"
        )
        if invalidated["backgroundInfo"] is not None or invalidated["scene"] is not None:
            raise AssertionError("background path change retained stale scene objects")
        if invalidated["deepSceneReady"] or not invalidated["composeDisabled"]:
            raise AssertionError("background path change did not disable composition")
        report["steps"].append({"background_invalidation": invalidated})
        page.locator("#background").fill(str(background))
        page.get_by_role("button", name="载入", exact=True).click()
        page.wait_for_function(
            "() => typeof deepSceneReady !== 'undefined' && deepSceneReady",
            timeout=300_000,
        )
        wait_idle(page)

        for index, (source, roi) in enumerate(product_items):
            page.locator(f"#source{index}").fill(str(source))
            page.locator(f"#source{index}").dispatch_event("change")
            page.locator("#productCards .card").nth(index).get_by_role(
                "button", name="框选后抠图", exact=True
            ).click()
            page.locator("#roiModal:not([hidden])").wait_for(timeout=60_000)
            image = page.locator("#roiImage")
            image.wait_for(state="visible")
            box = image.bounding_box()
            if not box:
                raise RuntimeError("ROI image has no display box")
            x0, y0, x1, y1 = roi
            page.mouse.move(box["x"] + box["width"] * x0, box["y"] + box["height"] * y0)
            page.mouse.down()
            page.mouse.move(
                box["x"] + box["width"] * x1,
                box["y"] + box["height"] * y1,
                steps=12,
            )
            page.mouse.up()
            page.get_by_role("button", name="使用此框并抠图", exact=True).click()
            if not page.locator("#background").is_disabled():
                raise AssertionError("critical inputs stayed enabled during segmentation")
            page.wait_for_function(
                f"() => Boolean(products[{index}].rgba)",
                timeout=300_000,
            )
            wait_idle(page)
            quality = page.locator("#productCards .quality").nth(index).inner_text()
            report["steps"].append({
                "product": index + 1,
                "quality": quality,
                "rgba": page.evaluate(f"products[{index}].rgba"),
                "alpha_quality": page.evaluate(f"products[{index}].alphaQuality"),
                "cutout_provider": page.evaluate(f"products[{index}].cutoutProvider"),
                "cloud_fallback_reason": page.evaluate(
                    f"products[{index}].cloudFallbackReason"
                ),
                "cutout_model": page.evaluate(f"products[{index}].cutoutModel"),
            })
            print(f"product {index + 1} ready: {quality}")

        page.get_by_role("button", name="智能构图", exact=True).click()
        wait_idle(page, timeout=240_000)
        report["steps"].append({
            "layout_status": page.locator("#status").inner_text(),
            "placements": placements(page),
        })
        layout_status = str(report["steps"][-1]["layout_status"])
        if "失败" in layout_status or "不在" in layout_status:
            raise RuntimeError(f"front-end layout failed: {layout_status}")
        page.screenshot(path=str(REVIEW / "ui_v320_layout.png"), full_page=True)
        print("layout ready")

        _, manual_placements = manually_adjust_safe_product(page)
        page.locator("#autoRepair").uncheck()
        if page.locator("#autoRepair").is_checked():
            raise AssertionError("autoRepair could not be disabled")
        report["steps"].append({
            "manual_placements": manual_placements,
            "auto_repair": False,
        })

        page.get_by_role("button", name="按当前摆位严格合成", exact=True).click()
        if not page.locator("#background").is_disabled():
            raise AssertionError("critical inputs stayed enabled during composition")
        wait_idle(page, timeout=900_000)
        if not page.evaluate("lastComposeResult"):
            raise RuntimeError(
                "front-end compose failed before producing a result: "
                + page.locator("#status").inner_text()
            )
        page.locator("#result img").wait_for(timeout=30_000)
        result_source = page.locator("#result img").get_attribute("src")
        compose_result = page.evaluate("lastComposeResult")
        visual_review = page.evaluate("lastVisualReview")
        if not compose_result:
            raise AssertionError("front-end did not retain compose result")
        if not visual_review:
            raise AssertionError("MiniMax visual review did not complete")
        assert_placements_equal(manual_placements, compose_result["products"])
        assert_placements_equal(manual_placements, placements(page))
        pixel_audit = compose_result["audit"]["pixel_audit"]
        dimensions = compose_result["output_dimensions"]
        report["steps"].append({
            "compose_status": page.locator("#status").inner_text(),
            "result_source": result_source,
            "audit": page.locator("#result .audit").first.inner_text(),
            "compose_result": compose_result,
            "visual_review": visual_review,
            "placements_after_compose": placements(page),
        })
        page.screenshot(path=str(REVIEW / "ui_v320_composite.png"), full_page=True)
        write_report(report)
        compose_status = str(report["steps"][-1]["compose_status"])
        if "失败" in compose_status or "没有开始" in compose_status:
            raise RuntimeError(f"front-end compose failed: {compose_status}")
        if int(pixel_audit["changed_pixels"]) != 0 or int(pixel_audit["max_delta"]) != 0:
            raise AssertionError(f"pixel gate failed: {pixel_audit}")
        if max(int(dimensions[0]), int(dimensions[1])) < 3840:
            raise AssertionError(f"output is not 4K: {dimensions}")
        if float(visual_review["visual_score"]) < 90.0:
            raise AssertionError(
                f"visual score below delivery gate: {visual_review['visual_score']}"
            )
        if visual_review["verdict"] != "pass":
            raise AssertionError(f"visual verdict is not pass: {visual_review['verdict']}")
        if visual_review["batch_unlock"] is not True:
            raise AssertionError("visual review did not unlock batch delivery")
        if page_errors:
            raise AssertionError(f"browser page errors: {page_errors}")
        browser.close()
    write_report(report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
