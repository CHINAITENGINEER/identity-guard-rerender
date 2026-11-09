from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if TYPE_CHECKING:
    from playwright.sync_api import Browser, Page, Playwright

from strict_compositor.heldout_manifest import (
    load_holdout_manifest,
    validate_holdout_manifest,
)
from strict_compositor.human_visual_audit import (
    HumanVisualAuditError,
    pending_human_visual_audit,
    validate_human_visual_audit,
    write_audit,
)
from strict_compositor.quality import (
    FinalGateError,
    sha256_file,
    verify_deliverable,
)
from strict_compositor.route_integrity import (
    audit_render_route,
    pending_cutout_reuse_manifest,
    reusable_cutout_provenance,
)


WORKSPACE = Path(os.environ.get("COMPOSITOR_DATA_ROOT", str(ROOT))).resolve()
SET_A = WORKSPACE / "未命名文件夹"
SET_B = WORKSPACE / "未命名文件夹 3"
SET_B_MANIFEST = WORKSPACE / "tests" / "fixtures" / "heldout_set_b_manifest.json"
FRONTEND_URL = "http://127.0.0.1:8765"
CHROME = Path(os.environ["COMPOSITOR_BROWSER_PATH"]) if os.environ.get("COMPOSITOR_BROWSER_PATH") else None
TEMP_ROOT = WORKSPACE / "ui_workspace" / "runtime_cache" / "tmp"
ACCEPTED_LAYOUT_MODES = frozenset({
    "智能构图完成",
    "本地联合重排完成",
})


@dataclass(frozen=True)
class Group:
    group_id: str
    source_set: str
    background: str
    products: tuple[str, str, str]


def log(message: str) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    try:
        print(f"[{stamp}] {message}", flush=True)
    except (OSError, ValueError):
        # A detached/embedded launcher may close its stdout pipe while the
        # browser job is still healthy.  Logging must never abort a render.
        return


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def build_groups() -> list[Group]:
    groups: list[Group] = []

    # This set is deliberately explicit.  Each group's 5.jpg is a pre-existing
    # composite/reference and is never enumerated, opened, previewed or sent to
    # the frontend.
    for number in range(1, 11):
        folder = SET_A / str(number)
        products = tuple(str(folder / f"{index}.JPG") for index in (1, 2, 3))
        background = str(folder / "4.JPG")
        groups.append(Group(f"set_a_{number:02d}", "未命名文件夹", background, products))

    # Set B was shot as four ordered ten-image batches: one background batch
    # followed by three product batches.  The old manifest grouped the 30
    # products as consecutive triples, which mixed semantic product columns.
    # Keep the manifest as the immutable byte allow-list, but pair the three
    # product batches by matching ordinal so every scene gets one item from
    # each batch.
    frozen = load_holdout_manifest(SET_B_MANIFEST)
    validate_holdout_manifest(SET_B, frozen)
    frozen_assets = {
        asset["filename"]: asset
        for case in frozen["cases"]
        for asset in [case["background"], *case["products"]]
    }
    backgrounds = [
        "DSC09515.JPG", "DSC09516.JPG", "DSC09517.JPG", "DSC09519.JPG",
        "DSC09520.JPG", "DSC09522.JPG", "DSC09523.JPG", "DSC09524.JPG",
        "DSC09525.JPG", "DSC09527.JPG",
    ]
    product_batches = [
        [
            "DSC09625.JPG", "DSC09626.JPG", "DSC09627.JPG", "DSC09628.JPG",
            "DSC09629.JPG", "DSC09630.JPG", "DSC09631.JPG", "DSC09632.JPG",
            "DSC09633.JPG", "DSC09634.JPG",
        ],
        [
            "DSC09734.JPG", "DSC09735.JPG", "DSC09737.JPG", "DSC09738.JPG",
            "DSC09741.JPG", "DSC09743.JPG", "DSC09744.JPG", "DSC09745.JPG",
            "DSC09746.JPG", "DSC09750.JPG",
        ],
        [
            "DSC09782.JPG", "DSC09783.JPG", "DSC09784.JPG", "DSC09785.JPG",
            "DSC09786.JPG", "DSC09787.JPG", "DSC09788.JPG", "DSC09789.JPG",
            "DSC09790.JPG", "DSC09791.JPG",
        ],
    ]
    requested_names = set(backgrounds)
    requested_names.update(name for batch in product_batches for name in batch)
    if requested_names != set(frozen_assets):
        raise RuntimeError("Set B aligned pairing does not match the frozen byte allow-list")
    for ordinal, background_name in enumerate(backgrounds):
        product_names = tuple(batch[ordinal] for batch in product_batches)
        groups.append(Group(
            group_id=f"set_b_{ordinal + 1:02d}",
            source_set="未命名文件夹 3",
            background=str(SET_B / "背景图" / background_name),
            products=tuple(str(SET_B / "物品图" / name) for name in product_names),
        ))

    forbidden = {
        (SET_A / str(number) / "5.jpg").resolve()
        for number in range(1, 11)
    }
    for group in groups:
        for raw_path in (group.background, *group.products):
            path = Path(raw_path).resolve()
            if path in forbidden or path.name.casefold() == "5.jpg":
                raise RuntimeError(f"安全门禁阻止读取既有融合图：{path}")
            if not path.is_file():
                raise FileNotFoundError(path)
    return groups


def page_state(page: Page) -> dict[str, Any]:
    return page.evaluate(
        """
        () => ({
          status: document.querySelector('#status')?.innerText || '',
          status_class: document.querySelector('#status')?.className || '',
          busy: document.body.getAttribute('aria-busy') === 'true',
          deep_scene_ready: Boolean(deepSceneReady),
          scene: scene ? JSON.parse(JSON.stringify(scene)) : null,
          products: products.map(item => JSON.parse(JSON.stringify(item))),
          compose: lastComposeResult ? JSON.parse(JSON.stringify(lastComposeResult)) : null,
          review: lastVisualReview ? JSON.parse(JSON.stringify(lastVisualReview)) : null,
          result_text: document.querySelector('#result')?.innerText || '',
          mode: document.querySelector('#modeBadge')?.innerText || ''
        })
        """
    )


def wait_until(
    page: Page,
    predicate: Callable[[dict[str, Any]], bool],
    *,
    timeout_seconds: int,
    label: str,
    log_interval: int = 25,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_status = ""
    last_log = 0.0
    while time.monotonic() < deadline:
        state = page_state(page)
        now = time.monotonic()
        status = state.get("status", "")
        if status != last_status or now - last_log >= log_interval:
            log(f"{label}: {status or '(no status)'}")
            last_status = status
            last_log = now
        if predicate(state):
            return state
        page.wait_for_timeout(1000)
    raise TimeoutError(f"{label} 超时（{timeout_seconds}s）：{last_status}")


def wait_busy_cycle(page: Page, *, timeout_seconds: int, label: str) -> dict[str, Any]:
    try:
        page.wait_for_function(
            "document.body.getAttribute('aria-busy') === 'true'",
            timeout=5000,
        )
    except Exception:
        # Some cached tasks can finish before the first poll.
        pass
    return wait_until(
        page,
        lambda state: not state["busy"],
        timeout_seconds=timeout_seconds,
        label=label,
    )


def require_accepted_layout(state: dict[str, Any]) -> dict[str, Any]:
    """Fail closed when the frontend leaves an adjustable rejected draft.

    The browser intentionally applies a rejected proposal so a human can drag
    it into place.  An unattended batch has no such human correction step and
    must never treat that draft as permission to compose.
    """
    mode = str(state.get("mode") or "").strip()
    if mode in ACCEPTED_LAYOUT_MODES:
        return state
    status = str(state.get("status") or "").strip()
    raise RuntimeError(
        "智能构图门禁未通过，批处理已停止，拒绝将可调整草案送入严格合成："
        f"mode={mode or '(empty)'}, status={status or '(empty)'}"
    )


def fill_source(page: Page, index: int, source: str) -> None:
    locator = page.locator(f"#source{index}")
    locator.fill(source)
    locator.evaluate("el => el.dispatchEvent(new Event('change', {bubbles: true}))")
    page.wait_for_timeout(150)


def load_reused_rgba(page: Page, index: int, rgba: Path) -> dict[str, Any]:
    """Exercise the real front-end RGBA route for an already audited cutout."""

    fill_source(page, index, str(rgba.resolve()))
    card = page.locator("#productCards .card").nth(index)
    card.get_by_role("button", name="RGBA", exact=True).click()
    state = wait_busy_cycle(
        page, timeout_seconds=180, label=f"商品 {index + 1} 复用已审计 RGBA",
    )
    product = state["products"][index]
    if not product.get("rgba"):
        raise RuntimeError(f"商品 {index + 1} 前端 RGBA 载入失败")
    return state


def load_background(page: Page, path: str) -> dict[str, Any]:
    page.locator("#background").fill(path)
    page.locator("#loadBackgroundButton").click()
    return wait_busy_cycle(page, timeout_seconds=900, label="载入背景/快速桌面")


def roi_retry(page: Page, index: int) -> dict[str, Any]:
    log(f"商品 {index + 1} 启动前端框选重试")
    card = page.locator("#productCards .card").nth(index)
    card.get_by_role("button", name="框选后抠图").click()
    page.locator("#roiModal").wait_for(state="visible", timeout=120_000)
    image = page.locator("#roiImage")
    image.wait_for(state="visible", timeout=120_000)
    box = image.bounding_box()
    if not box:
        raise RuntimeError("框选弹窗未能取得商品原图尺寸")
    inset_x = max(4.0, box["width"] * 0.018)
    inset_y = max(4.0, box["height"] * 0.018)
    page.mouse.move(box["x"] + inset_x, box["y"] + inset_y)
    page.mouse.down()
    page.mouse.move(
        box["x"] + box["width"] - inset_x,
        box["y"] + box["height"] - inset_y,
        steps=18,
    )
    page.mouse.up()
    page.get_by_role("button", name="使用此框并抠图").click()
    return wait_busy_cycle(page, timeout_seconds=1200, label=f"商品 {index + 1} 框选抠图")


def cutout_suspicion_reasons(product: dict[str, Any]) -> list[str]:
    """Return hard reasons a front-end matte must not enter composition.

    The UI's numeric score is diagnostic, not ground truth.  Batch production
    therefore repeats the same semantic/edge invariants used by the service so
    a stale browser state or an older cached alpha cannot be accepted merely
    because it reports 90+.
    """

    reasons: list[str] = []
    if not product.get("rgba"):
        return ["missing_rgba"]
    quality = product.get("alphaQuality") or {}
    if str(quality.get("verdict", "")).lower() == "reject":
        reasons.append("quality_reject")
    for key in (
        "multi_object_leakage",
        "opposite_edge_attachment",
        "semantic_envelope_leakage",
    ):
        if bool(quality.get(key)):
            reasons.append(key)

    semantic = quality.get("semantic_alpha_envelope") or {}
    excess = semantic.get("span_excess") or [0.0, 0.0]
    expected = semantic.get("expected_subject_span_in_crop") or [1.0, 1.0]
    observed = semantic.get("observed_alpha_extent_in_crop") or [0.0, 0.0]
    edges = quality.get("edge_occupancy") or semantic.get("edge_occupancy") or {}
    try:
        dense = (
            float(quality.get("foreground_ratio", 0.0)) >= 0.48
            and float(quality.get("stable_area_ratio", 0.0)) >= 0.80
        )
        asymmetric = dense and (
            (
                float(excess[1]) >= 0.10
                and float(observed[0]) <= float(expected[0]) - 0.02
                and max(float(edges.get("left", 0.0)), float(edges.get("right", 0.0))) >= 0.42
            )
            or (
                float(excess[0]) >= 0.10
                and float(observed[1]) <= float(expected[1]) - 0.02
                and max(float(edges.get("top", 0.0)), float(edges.get("bottom", 0.0))) >= 0.42
            )
        )
    except (IndexError, TypeError, ValueError):
        asymmetric = False
    if asymmetric:
        reasons.append("asymmetric_semantic_background_leakage")
    return sorted(set(reasons))


def direct_cutout_retry(page: Page, index: int) -> dict[str, Any]:
    """Retry through the visible front-end direct-cutout route."""

    log(f"商品 {index + 1} 启动前端直接抠图重试")
    card = page.locator("#productCards .card").nth(index)
    card.get_by_role("button", name="直接抠图", exact=True).click()
    return wait_busy_cycle(
        page, timeout_seconds=1200, label=f"商品 {index + 1} 直接抠图重试",
    )


def ensure_cutouts(page: Page) -> dict[str, Any]:
    state = page_state(page)
    for index, product in enumerate(state["products"]):
        score = float((product.get("alphaQuality") or {}).get("score") or 0)
        reasons = cutout_suspicion_reasons(product)
        if product.get("rgba") and score >= 65 and not reasons:
            continue
        log(
            f"商品 {index + 1} 当前透明图禁止进入合成："
            f"score={score:.1f}, reasons={reasons or ['low_confidence']}"
        )

        # Re-run the semantic direct route first.  This safely retries an
        # already completed RunningHub task/download and preserves the verified
        # MiniMax subject box.  A near-full manual box would erase that evidence
        # and was the reason a background-bearing alpha could bypass the gate.
        for _ in range(2):
            state = direct_cutout_retry(page, index)
            product = state["products"][index]
            score = float((product.get("alphaQuality") or {}).get("score") or 0)
            reasons = cutout_suspicion_reasons(product)
            if product.get("rgba") and score >= 65 and not reasons:
                break
        else:
            raise RuntimeError(
                f"商品 {index + 1} 的前端透明图仍不可信，已阻止进入合成："
                f"score={score:.1f}, reasons={reasons or ['missing_or_low_confidence']}"
            )
    return state


def prepare_group(page: Page) -> dict[str, Any]:
    page.locator("#prepareAllButton").click()
    state = wait_busy_cycle(page, timeout_seconds=2400, label="一键准备/智能摆位")
    state = ensure_cutouts(page)
    if not state.get("deep_scene_ready"):
        raise RuntimeError("精确桌面分析未完成")
    # If a cutout retry happened after prepareAll, layout may not have run.
    if state.get("mode") not in ACCEPTED_LAYOUT_MODES:
        page.locator("#autoLayoutButton").click()
        state = wait_busy_cycle(page, timeout_seconds=900, label="智能构图")
    return require_accepted_layout(state)


def compose_group(page: Page) -> dict[str, Any]:
    page.locator("#composeButton").click()
    state = wait_busy_cycle(page, timeout_seconds=2700, label="严格合成/视觉评审")
    if state.get("compose"):
        return state

    # One deterministic geometry repair is allowed when the physical gate
    # blocks composition; this is not prompt/seed cherry-picking.
    log("首轮未生成候选图，执行一次前端‘仅修复越界’后重试严格合成")
    page.locator("#repairLayoutButton").click()
    wait_busy_cycle(page, timeout_seconds=600, label="仅修复越界")
    page.locator("#composeButton").click()
    return wait_busy_cycle(page, timeout_seconds=2700, label="严格合成重试/视觉评审")


def archive_group(page: Page, group: Group, destination: Path, state: dict[str, Any]) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    cutout_dir = destination / "cutouts"
    cutout_dir.mkdir(parents=True, exist_ok=True)

    archived_cutouts: list[str] = []
    for index, product in enumerate(state.get("products") or [], start=1):
        rgba = Path(product.get("rgba") or "")
        if rgba.is_file():
            target = cutout_dir / f"product_{index}_rgba.png"
            # Reused, hash-bound cutouts are loaded through the real front-end
            # RGBA buttons directly from this archive directory.  Copying a
            # file onto itself fails on Windows (and can race an image reader),
            # even though the render has already completed successfully.
            if rgba.resolve() != target.resolve():
                shutil.copy2(rgba, target)
            archived_cutouts.append(str(target))

    compose = state.get("compose") or {}
    review = state.get("review") or {}
    final_source = Path(compose.get("final_path") or "")
    run_audit_path = final_source.parent / "audit.json"
    run_audit: dict[str, Any] = {}
    if run_audit_path.is_file():
        try:
            run_audit = json.loads(run_audit_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            run_audit = {}
    route_integrity = audit_render_route(state.get("products") or [], run_audit)
    score = float(review.get("visual_score") or 0)
    machine_passed = bool(
        review.get("verdict") == "pass"
        and review.get("batch_unlock") is True
        and review.get("pixel_verdict") == "pass"
        and score >= 90
        and route_integrity["verdict"] == "pass"
    )
    gate_error = ""
    gate: dict[str, Any] | None = None
    if machine_passed:
        gate_path = final_source.parent / "final_gate.json"
        try:
            if not gate_path.is_file():
                raise FinalGateError("final_gate.json is missing")
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
            verify_deliverable(final_source, gate)
        except (OSError, ValueError, TypeError, json.JSONDecodeError, FinalGateError) as error:
            # Browser state is a preview convenience, never delivery evidence.
            # A stale/edited image or incomplete five-slot gate must be archived
            # as a rejected candidate even if page JavaScript still says pass.
            machine_passed = False
            gate_error = str(error)
    final_target = ""
    debug_preview_target = ""
    if final_source.is_file():
        debug_preview = bool(
            run_audit.get("verdict") == "debug_preview"
            or compose.get("delivery_verdict") == "debug_preview_not_audited"
            or compose.get("strict_audit") is not True
        )
        # A layout-rejected/debug render is useful evidence, but calling it a
        # final candidate made visibly broken compositions look like completed
        # work.  Archive it under an unambiguous name and keep `completed`
        # false.  Only a non-debug render may enter human candidate review.
        target_name = (
            "layout_reject_preview_4k.png"
            if debug_preview else "final_candidate_4k.png"
        )
        target = destination / target_name
        shutil.copy2(final_source, target)
        if run_audit_path.is_file():
            shutil.copy2(run_audit_path, destination / "audit.json")
        if machine_passed and gate is not None:
            expected = str(gate["inputs"]["final_sha256"])
            if sha256_file(target) != expected:
                target.unlink(missing_ok=True)
                raise RuntimeError("Archived delivery hash differs from final_gate.json")
            shutil.copy2(final_source.parent / "final_gate.json", destination / "final_gate.json")
        if debug_preview:
            debug_preview_target = str(target)
        else:
            final_target = str(target)

    atomic_json(destination / "render_route_integrity.json", route_integrity)
    archived_sources = [Path(item) for item in group.products]
    archived_rgba = [Path(item) for item in archived_cutouts]
    if len(archived_rgba) == len(archived_sources):
        reuse_manifest_path = destination / "cutout_reuse_manifest.json"
        fresh_manifest = pending_cutout_reuse_manifest(
            archived_sources, archived_rgba,
        )
        preserve_existing = False
        if reuse_manifest_path.is_file():
            try:
                existing_manifest = json.loads(
                    reuse_manifest_path.read_text(encoding="utf-8")
                )
                old_entries = list(existing_manifest.get("products") or [])
                new_entries = list(fresh_manifest.get("products") or [])
                preserve_existing = bool(
                    len(old_entries) == len(new_entries)
                    and all(
                        str(old.get("source_sha256") or "")
                        == str(new.get("source_sha256") or "")
                        and str(old.get("rgba_sha256") or "")
                        == str(new.get("rgba_sha256") or "")
                        for old, new in zip(old_entries, new_entries)
                    )
                )
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                preserve_existing = False
        # Alpha repair or residue cleanup changes the canonical RGBA.  Refresh
        # stale hashes and return the artifact to pending human target review;
        # preserve approvals only when every source/RGBA hash is unchanged.
        if not preserve_existing:
            atomic_json(
                reuse_manifest_path,
                fresh_manifest,
            )

    human_review: dict[str, Any] = {
        "status": "unavailable",
        "verdict": "pending_human_review",
    }
    human_review_error = ""
    human_review_path = destination / "human_visual_audit.json"
    if final_target:
        try:
            if human_review_path.is_file():
                existing_human_review = json.loads(
                    human_review_path.read_text(encoding="utf-8")
                )
                if existing_human_review.get("status") == "reviewed":
                    human_review = validate_human_visual_audit(
                        existing_human_review,
                        candidate_path=final_target,
                    )
                else:
                    human_review = pending_human_visual_audit(
                        final_target,
                        machine_diagnostics={
                            "visual_score": score if review else None,
                            "verdict": review.get("verdict") or "unavailable",
                            "pixel_verdict": review.get("pixel_verdict") or "unavailable",
                            "layout_status": state.get("status", ""),
                        },
                    )
                    write_audit(human_review_path, human_review)
            else:
                human_review = pending_human_visual_audit(
                    final_target,
                    machine_diagnostics={
                        "visual_score": score if review else None,
                        "verdict": review.get("verdict") or "unavailable",
                        "pixel_verdict": review.get("pixel_verdict") or "unavailable",
                        "layout_status": state.get("status", ""),
                    },
                )
                write_audit(human_review_path, human_review)
        except (OSError, ValueError, TypeError, json.JSONDecodeError, HumanVisualAuditError) as error:
            human_review_error = str(error)

    human_passed = human_review.get("verdict") == "pass"
    passed = bool(machine_passed and human_passed)

    try:
        page.locator("#result").scroll_into_view_if_needed()
        page.wait_for_timeout(500)
        page.screenshot(path=str(destination / "frontend_proof.png"), full_page=True)
    except Exception as error:
        log(f"{group.group_id} 前端证明截图失败：{error}")

    report = {
        "group": asdict(group),
        "completed": bool(final_target),
        "passed": passed,
        "machine_passed": machine_passed,
        "render_route_integrity": route_integrity,
        "visual_score": score if review else None,
        "verdict": (
            "pass" if passed else human_review.get("verdict", "pending_human_review")
        ),
        "human_visual_review_required": True,
        "human_visual_review": human_review,
        "human_visual_review_error": human_review_error,
        "machine_scores_are_diagnostic_only": True,
        "batch_unlock": bool(review.get("batch_unlock", False)),
        "delivery_gate_revalidated": bool(machine_passed and gate is not None),
        "delivery_gate_error": gate_error,
        "status": state.get("status", ""),
        "mode": state.get("mode", ""),
        "archived_cutouts": archived_cutouts,
        "final_output": final_target,
        "debug_preview_output": debug_preview_target,
        "run": compose.get("run", ""),
        "compose": compose,
        "visual_review": review,
        "scene_summary": {
            key: (state.get("scene") or {}).get(key)
            for key in (
                "method",
                "confidence",
                "support_polygon",
                "protected_polygons",
                "reflection_recommended",
                "reflectivity",
                "shadow_direction",
                "advisor",
            )
        },
        "products": state.get("products") or [],
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    atomic_json(destination / "group_report.json", report)
    return report


def open_browser(playwright: Playwright) -> tuple[Browser, Page]:
    browser = playwright.chromium.launch(
        headless=True,
        executable_path=str(CHROME) if CHROME else None,
        args=[
            "--no-sandbox",
            "--disable-gpu",
            "--disable-dev-shm-usage",
            "--disk-cache-size=1",
            "--media-cache-size=1",
        ],
    )
    page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1)
    page.set_default_timeout(120_000)
    return browser, page


def init_page(page: Page) -> None:
    page.goto(FRONTEND_URL, wait_until="domcontentloaded", timeout=120_000)
    page.wait_for_function(
        "!document.querySelector('#cloudState').innerText.includes('正在读取配置')",
        timeout=120_000,
    )
    # Batch candidates must exercise the same strict pixel/geometry audit that
    # the operator can enable in the UI. Keep the interactive default intact,
    # but never archive a debug-only preview as a batch deliverable.
    strict_audit = page.locator("#strictAuditEnabled")
    if strict_audit.count() != 1:
        raise RuntimeError("Front-end strict-audit control is missing")
    strict_audit.set_checked(True)
    # The model review is an evaluator only: it does not alter product pixels,
    # lighting layers, layout, or strict audit evidence. Batch exploration is
    # judged by human inspection, so do not spend five independent network
    # calls per group or let one missing slot downgrade a valid strict render.
    ai_review = page.locator("#aiReviewAfterCompose")
    if ai_review.count() != 1:
        raise RuntimeError("Front-end AI-review control is missing")
    ai_review.set_checked(False)


def process_one(
    page: Page,
    group: Group,
    output_root: Path,
    reuse_cutouts_from: Path | None = None,
) -> dict[str, Any]:
    destination = output_root / group.group_id
    report_path = destination / "group_report.json"
    if report_path.is_file():
        try:
            existing = json.loads(report_path.read_text(encoding="utf-8"))
            if (
                existing.get("completed")
                and Path(existing.get("final_output") or "").is_file()
                and (existing.get("render_route_integrity") or {}).get("verdict") == "pass"
            ):
                log(f"{group.group_id} 已完成，断点跳过")
                return existing
        except Exception:
            pass

    log(f"===== {group.group_id} 开始（只使用 3 商品 + 1 背景）=====")
    init_page(page)
    load_background(page, group.background)
    reusable = []
    if reuse_cutouts_from is not None:
        reusable = [
            reuse_cutouts_from / group.group_id / "cutouts" / f"product_{index}_rgba.png"
            for index in (1, 2, 3)
        ]
    if reusable and all(path.is_file() for path in reusable):
        reuse_manifest_path = reuse_cutouts_from / group.group_id / "cutout_reuse_manifest.json"
        reuse_manifest = None
        if reuse_manifest_path.is_file():
            try:
                reuse_manifest = json.loads(reuse_manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                reuse_manifest = None
        provenance = reusable_cutout_provenance(
            group.products, reusable, reuse_manifest,
        )
        atomic_json(destination / "cutout_reuse_validation.json", provenance)
        if provenance["verdict"] == "pass":
            log(f"{group.group_id} reusing hash-bound, human-approved cutouts")
            for index, rgba in enumerate(reusable):
                load_reused_rgba(page, index, rgba)
        else:
            log(
                f"{group.group_id} cached cutouts rejected; rerunning originals: "
                + "; ".join(provenance["reasons"][:4])
            )
            for index, source in enumerate(group.products):
                fill_source(page, index, source)
    else:
        for index, source in enumerate(group.products):
            fill_source(page, index, source)
    prepare_group(page)
    state = compose_group(page)
    report = archive_group(page, group, destination, state)
    log(
        f"===== {group.group_id} 完成："
        f"score={report['visual_score']}, verdict={report['verdict']}, "
        f"output={report['final_output']} ====="
    )
    return report


def main() -> int:
    from playwright.sync_api import sync_playwright

    parser = argparse.ArgumentParser(description="Drive the real frontend for the complete 20-group batch.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--only", action="append", default=[], help="Only run these group IDs")
    parser.add_argument("--start-at", default="", help="Resume selection from this group ID")
    parser.add_argument(
        "--reuse-cutouts-from", type=Path, default=None,
        help="Reuse a previous run's audited cutouts through the real front-end RGBA buttons",
    )
    args = parser.parse_args()

    os.environ["TEMP"] = str(TEMP_ROOT)
    os.environ["TMP"] = str(TEMP_ROOT)
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)

    groups = build_groups()
    if args.only:
        wanted = set(args.only)
        groups = [group for group in groups if group.group_id in wanted]
    if args.start_at:
        ids = [group.group_id for group in groups]
        if args.start_at not in ids:
            raise RuntimeError(f"--start-at 不存在：{args.start_at}")
        groups = groups[ids.index(args.start_at) :]

    input_manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "frontend_url": FRONTEND_URL,
        "policy": {
            "processing": "real browser frontend only",
            "set_a_mapping": "1.JPG, 2.JPG, 3.JPG are products; 4.JPG is background",
            "forbidden_existing_composites": "Every set_a 5.jpg is hard-skipped and never opened",
            "set_b_mapping": (
                "three ordered ten-image product batches paired by ordinal; "
                "all 40 inputs verified against the frozen SHA-256 allow-list"
            ),
            "set_b_frozen_manifest": str(SET_B_MANIFEST),
            "reused_cutouts_from": str(args.reuse_cutouts_from or ""),
        },
        "groups": [asdict(group) for group in groups],
    }
    atomic_json(args.output / "input_manifest.json", input_manifest)

    reports: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    with sync_playwright() as playwright:
        if CHROME is not None and not CHROME.is_file():
            raise FileNotFoundError(CHROME)
        browser, page = open_browser(playwright)
        try:
            for ordinal, group in enumerate(groups, start=1):
                try:
                    reports.append(process_one(
                        page, group, args.output, args.reuse_cutouts_from,
                    ))
                except Exception as error:
                    log(f"{group.group_id} 失败：{error}")
                    traceback.print_exc()
                    failure = {
                        "group_id": group.group_id,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                        "time": datetime.now().isoformat(timespec="seconds"),
                    }
                    failures.append(failure)
                    failed_dir = args.output / group.group_id
                    failed_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        state = page_state(page)
                        atomic_json(failed_dir / "failed_frontend_state.json", state)
                        page.screenshot(path=str(failed_dir / "failed_frontend_proof.png"), full_page=True)
                    except Exception:
                        pass
                summary = {
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                    "requested_groups": len(groups),
                    "processed": ordinal,
                    "completed": sum(bool(item.get("completed")) for item in reports),
                    "passed": sum(bool(item.get("passed")) for item in reports),
                    "rejected": sum(bool(item.get("completed")) and not item.get("passed") for item in reports),
                    "failures": failures,
                    "reports": reports,
                }
                atomic_json(args.output / "batch_manifest.json", summary)
                # Reload after every group to release 4K preview memory.
                try:
                    page.goto("about:blank", wait_until="load", timeout=30_000)
                except Exception:
                    pass
        finally:
            browser.close()

    completed_count = sum(bool(item.get("completed")) for item in reports)
    log(
        f"批处理结束：processed={len(reports)}, "
        f"completed={completed_count}, failures={len(failures)}"
    )
    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(main())
