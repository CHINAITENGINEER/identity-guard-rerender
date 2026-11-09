from __future__ import annotations

import re
import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HTML_PATH = ROOT / "strict_compositor" / "ui_index.html"
E2E_PATH = ROOT / "tools" / "ui_e2e_test.py"


class FrontendReliabilityContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.html = HTML_PATH.read_text(encoding="utf-8")
        cls.e2e = E2E_PATH.read_text(encoding="utf-8")

    def test_inline_javascript_parses(self) -> None:
        node = shutil.which("node")
        if not node:
            self.skipTest("Node.js is required for JavaScript syntax validation")
        scripts = re.findall(r"<script>([\s\S]*?)</script>", self.html)
        self.assertEqual(len(scripts), 1)
        result = subprocess.run(
            [node, "--check", "-"],
            input=scripts[0],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_background_change_invalidates_scene_and_disables_compose(self) -> None:
        self.assertIn('oninput="backgroundPathChanged(this.value)"', self.html)
        self.assertIn("function invalidateBackgroundState", self.html)
        self.assertIn("backgroundInfo=null", self.html)
        self.assertIn("deepSceneReady=false", self.html)
        self.assertIn('$("#composeButton").disabled=!ready', self.html)

    def test_long_operations_share_a_critical_control_lock(self) -> None:
        self.assertIn("function beginCriticalTask()", self.html)
        self.assertIn("function endCriticalTask()", self.html)
        self.assertIn(
            'document.querySelectorAll("aside button,aside input,aside textarea,aside select")',
            self.html,
        )
        self.assertIn("const productRevisions=[0,0,0]", self.html)

    def test_individual_cutout_commits_all_visible_product_paths(self) -> None:
        # Completing product 1 re-renders all three cards.  Product 2/3 paths
        # typed but not yet blurred must therefore enter application state
        # before the first request starts, or the old dataset reappears.
        self.assertIn("function syncVisibleProductSources()", self.html)
        segment = re.search(
            r"function segmentProduct\(index,[\s\S]*?\n\}",
            self.html,
        )
        self.assertIsNotNone(segment)
        self.assertIn("syncVisibleProductSources();", segment.group(0))
        self.assertIn(
            "const requestedSources=products.map((product,index)=>{",
            self.html,
        )
        self.assertIn("assignProductSourceState(index,source);", self.html)

    def test_auto_layout_obeys_backend_surface_gate(self) -> None:
        for required in (
            'const layoutScore=Number(data.metrics?.layout_score??0)',
            'const layoutVerdict=String(data.layout_verdict',
            'const layoutPassed=layoutVerdict==="pass"',
            'data.metrics?.local_joint_replan',
            'data.advisor?.status==="seed_replanned_local"',
            'MiniMax 仅作商品语义与相对关系种子',
            '自动构图未通过',
        ):
            self.assertIn(required, self.html)

    def test_auto_layout_reuses_the_background_receiver_scale(self) -> None:
        ui_source = (ROOT / "strict_compositor" / "ui.py").read_text(
            encoding="utf-8"
        )
        for required in (
            "const sceneReceiver=scene?.advisor||{};",
            "sceneReceiver.receiver_width_or_diameter_cm",
            "sceneReceiver.receiver_size_confidence",
            "receiver_width_or_diameter_cm: float | None",
            '"background_scene_analysis"',
            "scene_receiver_authority = True",
        ):
            self.assertIn(required, self.html + ui_source)

    def test_layout_round_trips_semantic_product_fields_and_exact_support(self) -> None:
        ui_source = (ROOT / "strict_compositor" / "ui.py").read_text(
            encoding="utf-8"
        )
        for required in (
            "estimated_height_cm_min",
            "estimated_height_cm_max",
            "footprint_width_cm",
            "photographed_pose:item.photographed_pose",
            "pose_confidence:Number(item.pose_confidence",
            "viewpoint_compatibility",
            "source_light_compatibility",
            "support_mask_path: str | None = None",
            "support_mask_path:scene.support_mask_path||null",
            "support_mask=support_mask",
        ):
            self.assertIn(required, self.html + ui_source)
        # productPayload removes only browser-only display fields; semantic
        # scale/view/light evidence must remain in the server contract.
        self.assertIn(
            "function productPayload(){return products.map(({preview,sourceAspect,alphaQuality,roi,...item})=>item)}",
            self.html,
        )

    def test_scene_geometry_evidence_round_trips_only_when_available(self) -> None:
        ui_source = (ROOT / "strict_compositor" / "ui.py").read_text(
            encoding="utf-8"
        )
        for required in (
            "depth_map_path: str | None = None",
            "normal_map_path: str | None = None",
            "scene=analysis;syncAdvanced()",
            "function sceneGeometryEvidence()",
            "if(scene?.depth_map_path)evidence.depth_map_path=scene.depth_map_path;",
            "if(scene?.normal_map_path)evidence.normal_map_path=scene.normal_map_path;",
            "...sceneGeometryEvidence(),",
            'depth_map_path = base_job.get("depth_map")',
            'normal_map_path = base_job.get("normal_map")',
            'job_data["depth_map"] = str(local_path(depth_map_path))',
            'job_data["normal_map"] = str(local_path(normal_map_path))',
        ):
            self.assertIn(required, self.html + ui_source)

        # Optional evidence must not add null keys to legacy browser requests
        # or manifests; the fields exist only when scene analysis produced a
        # usable map.
        self.assertNotIn("depth_map_path:scene.depth_map_path||null", self.html)
        self.assertNotIn("normal_map_path:scene.normal_map_path||null", self.html)
        self.assertNotIn('"depth_map": (', ui_source)
        self.assertNotIn('"normal_map": (', ui_source)

    def test_visual_review_discloses_id_mapping_and_unchanged_gate(self) -> None:
        for required in (
            "const evidenceMapping=reviewSlots.find",
            "item.product_id",
            "证据一一映射",
            "浅色/透明材质必须先与同 ID 原图比较",
            "真实缺陷不豁免",
            "90 分与维度下限门禁未降低",
            'review.pixel_verdict==="pass"&&score>=90',
        ):
            self.assertIn(required, self.html)

    def test_visual_review_is_optional_bounded_and_resumable(self) -> None:
        for required in (
            "function requestVisualReviewWithRecovery",
            "attempt<=2",
            "仅补未完成评审槽位",
            'requestVisualReviewWithRecovery({',
            'id="aiReviewAfterCompose"',
            'const runAiReview=$("#aiReviewAfterCompose").checked',
            "未运行 AI 视觉评分",
            "不替代人工肉眼验收",
            "minimaxNetworkMode",
        ):
            self.assertIn(required, self.html)

    def test_preview_material_route_and_delivery_are_not_score_gated(self) -> None:
        for required in (
            "item.dataset.material",
            'product.material_mode==="clear_glass"',
            "drop-shadow",
            "const deliveryReady=Boolean(",
            "下载 4K 人工验收候选 PNG",
            "等待人工验收",
            "changed_pixels||0)===0",
            'data.physical_verdict==="pass"',
            'data.delivery_verdict==="pending_visual_review"',
            'data.audit?.physical_audit?.verdict==="pass"',
        ):
            self.assertIn(required, self.html)

    def test_compose_response_exposes_fail_closed_physical_verdict(self) -> None:
        ui_source = (ROOT / "strict_compositor" / "ui.py").read_text(
            encoding="utf-8",
        )
        pipeline_source = (
            ROOT / "strict_compositor" / "pipeline.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"physical_verdict": str(', ui_source)
        self.assertIn(
            'audit.get("physical_audit", {}).get("verdict", "reject")',
            ui_source,
        )
        self.assertIn('"delivery_verdict": str(', ui_source)
        self.assertIn("strict_terminal_pass = bool(", pipeline_source)
        self.assertIn(
            'else ("reject" if strict_audit else "debug_preview")',
            pipeline_source,
        )

    def test_low_visual_score_repairs_new_pixels_once_instead_of_score_shopping(self) -> None:
        for required in (
            "repairableVisualFailure",
            "refinement_pass:0",
            "refinement_pass:1",
            "refinement_base_run:firstRun",
            "auto_repair:false",
            "const lockedProducts=JSON.parse(JSON.stringify(firstData.products))",
            "首轮锁定的商品像素",
            "第二轮仍不合格即停止，不对同一图片刷分",
        ):
            self.assertIn(required, self.html)

    def test_refinement_is_server_locked_to_the_first_candidate(self) -> None:
        ui_source = (ROOT / "strict_compositor" / "ui.py").read_text(encoding="utf-8")
        for required in (
            "def refinement_base_job",
            "refinement_base_run: str | None = None",
            "if payload.refinement_pass:",
            "if payload.auto_repair:",
            "base_run, base_job = refinement_base_job(payload.refinement_base_run)",
            'placements = json.loads(json.dumps(base_job["ui_placements"]))',
            'products = json.loads(json.dumps(base_job["products"]))',
            '"refinement_lock": {',
        ):
            self.assertIn(required, ui_source)

    def test_refinement_never_replaces_a_higher_score_with_a_lower_one(self) -> None:
        self.assertIn(
            "if(refinedScore<initialScore+0.5)return false;",
            self.html,
        )
        self.assertNotIn(
            "if(refinedCritical<initialCritical)return true;",
            self.html,
        )

    def test_ambient_glass_refinement_preserves_analysed_mirror_strength(self) -> None:
        ui_source = (ROOT / "strict_compositor" / "ui.py").read_text(encoding="utf-8")
        self.assertIn('if surface_material == "transparent_glass"', ui_source)
        self.assertIn('if cast_shadow_mode == "directional"', ui_source)
        self.assertIn('effective_reflection = min(', ui_source)
        self.assertIn('effective_reflection = min(0.30, effective_reflection)', ui_source)
        self.assertNotIn('max(0.24, effective_reflection)', ui_source)
        self.assertIn('duplicate-looking bottle/pitcher', ui_source)
        self.assertIn('max=".42"', self.html)

    def test_polished_opaque_receiver_gets_bounded_preview_visible_reflection(self) -> None:
        self.assertIn(
            'Math.min(.30,Math.max(.08,.055+Number(analysis.reflectivity||0)*.28))',
            self.html,
        )
        self.assertNotIn(
            'Math.min(.24,Math.max(.03,.025+Number(analysis.reflectivity||0)*.18))',
            self.html,
        )

    def test_debug_audit_toggle_does_not_disable_quality_pipeline(self) -> None:
        ui_source = (ROOT / "strict_compositor" / "ui.py").read_text(encoding="utf-8")
        pipeline_source = (
            ROOT / "strict_compositor" / "pipeline.py"
        ).read_text(encoding="utf-8")
        for required in (
            'id="strictAuditEnabled"',
            'const strictAudit=$("#strictAuditEnabled").checked',
            "strict_audit:strictAudit",
            "4K 调试合成结果",
            "分层PNG和光学Matte均已保留",
        ):
            self.assertIn(required, self.html)
        self.assertIn("strict_audit: bool = True", ui_source)
        self.assertIn("strict_audit=bool(payload.strict_audit)", ui_source)
        self.assertIn('PixelAudit(0, 0, "skipped_debug")', pipeline_source)
        self.assertIn('save_png(out_dir / "layers"', pipeline_source)
        self.assertNotIn("preview_max_edge", self.html + ui_source + pipeline_source)

    def test_learned_connection_test_has_dedicated_feedback(self) -> None:
        test_cloud = re.search(
            r"async function testCloud\(provider\)\{([\s\S]*?)\n\}\nfunction setStatus",
            self.html,
        )
        self.assertIsNotNone(test_cloud)
        body = test_cloud.group(1)
        for required in (
            'provider==="runninghub_learned"?"RunningHub 学习阴影 V1"',
            'else if(provider==="runninghub_learned")',
            "RunningHub 学习阴影 V1 密钥、上传权限和专用工作流读取通过",
            "本次只验证配置，没有创建生成任务",
            "要用于图片仍需单独勾选",
            '${providerLabel} 测试失败',
        ):
            self.assertIn(required, body)

    def test_learned_v1_requires_normal_prompt_and_never_submits_v2(self) -> None:
        integration_payload = re.search(
            r"function integrationPayload\(\)\{([\s\S]*?)\n\}",
            self.html,
        )
        self.assertIsNotNone(integration_payload)
        payload_body = integration_payload.group(1)
        for required in (
            "runninghub_learned_normal_node_id",
            "runninghub_learned_prompt_node_id",
            "runninghub_learned_shadow_output_node_id",
            "法线节点（V1 必填）",
            "提示词节点（V1 必填）",
            "V2 未开放（当前不提交）",
        ):
            self.assertIn(required, payload_body + self.html)
        self.assertNotIn("runninghub_learned_gain_output_node_id", payload_body)
        self.assertNotIn("runninghub_learned_specular_output_node_id", payload_body)
        self.assertIn(
            'id="runninghubLearnedGainOutputNode" value="" '
            'placeholder="V2 未开放（当前不提交）" disabled',
            self.html,
        )
        self.assertIn(
            'id="runninghubLearnedSpecularOutputNode" value="" '
            'placeholder="V2 未开放（当前不提交）" disabled',
            self.html,
        )
        self.assertNotIn('<option value="alpha_channel">', self.html)

    def test_learned_configuration_and_per_compose_switches_are_separate(self) -> None:
        for required in (
            'id="runninghubLearnedEnabled"',
            'id="learnedEffectsEnabled"',
            "配置总开关：允许本机调用付费学习阴影 V1 工作流",
            "本次合成启用学习阴影 V1（第二道开关",
            'learned_effects_enabled:$("#learnedEffectsEnabled").checked',
            "保存配置不会自动用于合成",
        ):
            self.assertIn(required, self.html)
        self.assertNotIn('$("#learnedEffectsEnabled").checked=true', self.html)
        self.assertIn(
            "composeToggle.disabled=busy||!learnedCloudStatus.ready",
            self.html,
        )

    def test_learned_health_and_configuration_state_are_visible(self) -> None:
        ui_source = (ROOT / "strict_compositor" / "ui.py").read_text(
            encoding="utf-8"
        )
        for required in (
            '"learned_effects": runninghub_learned_effects_status()',
            '"runninghub_learned_status": runninghub_learned_effects_status()',
            "ready=${learnedCloudStatus.ready}",
            "secret_configured=${learnedCloudStatus.secret_configured}",
            "workflow_configured=${learnedCloudStatus.workflow_configured}",
            'id="learnedEffectsBadge"',
            'id="learnedCloudState"',
        ):
            self.assertIn(required, ui_source + self.html)

    def test_learned_configuration_test_never_starts_a_generation_task(self) -> None:
        ui_source = (ROOT / "strict_compositor" / "ui.py").read_text(
            encoding="utf-8"
        )
        endpoint = re.search(
            r"def test_integrations\(payload: dict\[str, Any\]\)([\s\S]*?)"
            r"\n@app\.get\(\"/api/file\"\)",
            ui_source,
        )
        self.assertIsNotNone(endpoint)
        body = endpoint.group(1)
        self.assertIn("client.test_upload()", body)
        self.assertIn("client.test_configuration(", body)
        self.assertNotIn("run_workflow", body)
        self.assertNotIn("create_task", body)

    def test_e2e_enforces_delivery_and_manual_placement_contracts(self) -> None:
        for required in (
            'visual_review["visual_score"]',
            'visual_review["verdict"] != "pass"',
            'visual_review["batch_unlock"] is not True',
            'pixel_audit["changed_pixels"]',
            'pixel_audit["max_delta"]',
            "assert_placements_equal(manual_placements",
            'page.locator("#autoRepair").uncheck()',
        ):
            self.assertIn(required, self.e2e)


if __name__ == "__main__":
    unittest.main()
