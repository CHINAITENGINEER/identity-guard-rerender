from __future__ import annotations

import json
import io
import shutil
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException
from PIL import Image

from strict_compositor.minimax_advisor import (
    MiniMaxTransportError,
    VISUAL_REVIEW_PROMPT_VERSION,
)
from strict_compositor.cli import quality as cli_quality
from strict_compositor.errors import ContractError
from strict_compositor.quality import (
    FinalGateError,
    aggregate_visual_reviews,
    build_final_gate,
    build_review_bundle,
    sha256_file,
    verify_deliverable,
    write_final_gate,
)
from strict_compositor.ui import VisualReviewUI, deliver_file, visual_review


class FinalGateTests(unittest.TestCase):
    def setUp(self) -> None:
        # Review slots are intentionally workspace-global in production. Keep
        # unit-test slots in an isolated workspace so fixtures cannot become
        # production review evidence or leave cache debris behind.
        self._workspace_temporary = self._temporary_directory()
        self._workspace_patch = patch(
            "strict_compositor.ui.WORK",
            Path(self._workspace_temporary.name) / "isolated_workspace",
        )
        self._workspace_patch.start()

    def tearDown(self) -> None:
        self._workspace_patch.stop()
        self._workspace_temporary.cleanup()

    @staticmethod
    def _temporary_directory() -> tempfile.TemporaryDirectory[str]:
        # The Windows system drive can be intentionally kept very small on the
        # production workstation. Keep these 4K gate fixtures on the workspace
        # drive instead of relying on %TEMP%.
        parent = Path(__file__).resolve().parents[1] / "ui_workspace" / "test_tmp"
        parent.mkdir(parents=True, exist_ok=True)
        return tempfile.TemporaryDirectory(dir=parent)

    def _artifacts(
        self,
        root: Path,
        *,
        dimensions: tuple[int, int] = (3840, 2160),
        pixel_pass: bool = True,
    ) -> tuple[Path, Path, list[Path], Path, dict]:
        background = root / "background.png"
        final = root / "final.png"
        Image.new("RGB", dimensions, (110, 100, 90)).save(background)
        Image.new("RGB", dimensions, (110, 100, 90)).save(final)
        products: list[Path] = []
        for index, color in enumerate(((200, 20, 20, 255), (20, 200, 20, 255), (20, 20, 200, 255)), 1):
            path = root / f"product_{index}.png"
            Image.new("RGBA", (12, 18), color).save(path)
            products.append(path)
        job = {
            "background": str(background),
            "support_polygon": [[0.05, 0.25], [0.95, 0.25], [0.95, 0.95], [0.05, 0.95]],
            "support_mask": None,
            "protected_mask": None,
            "protected_polygons": [],
            "lighting": {
                "shadow_direction": [0.8, -0.6],
                "shadow_length_ratio": 0.05,
                "cast_shadow_mode": "directional",
                "surface_material": "satin",
                "reflection_enabled": False,
            },
            "products": [
                {
                    "id": f"product_{index}",
                    "rgba": str(path),
                    "homography": [[1, 0, index * 20], [0, 1, 100], [0, 0, 1]],
                    "z_order": index,
                    "rgb_gain": [1, 1, 1],
                    "placement_mode": "upright",
                    "material_mode": "opaque",
                    "material_confidence": 0.95,
                    "transmission_strength": 0.0,
                }
                for index, path in enumerate(products, 1)
            ],
            "ui_placements": [
                {
                    "id": f"product_{index}",
                    "center_x": 0.2 + index * 0.2,
                    "baseline_y": 0.7,
                    "height_ratio": 0.16,
                    "z_order": index,
                    "semantic_category": "decor_object",
                }
                for index in range(1, 4)
            ],
            "source_dimensions": list(dimensions),
            "upscaled_to_4k": False,
            "refinement_pass": 0,
        }
        (root / "job.json").write_text(json.dumps(job), encoding="utf-8")
        audit = {
            "verdict": "pending_visual_review",
            "batch_unlock": False,
            "pixel_audit": {
                "changed_pixels": 0 if pixel_pass else 1,
                "max_delta": 0 if pixel_pass else 4,
                "verdict": "pass" if pixel_pass else "reject",
            },
            "physical_audit": {
                "schema_version": "physical-evidence-v1",
                "verdict": "pass",
                "subscores": {
                    "grounding": 99.0,
                    "cast_shadows": 99.0,
                    "geometry_layout": 99.0,
                    "cutout_edges": 99.0,
                },
                "dimension_floors": {
                    "grounding": 95.0,
                    "cast_shadows": 90.0,
                    "geometry_layout": 98.0,
                    "cutout_edges": 90.0,
                },
            },
            "background_sha256": sha256_file(background),
            "output_dimensions": list(dimensions),
            "output_format": "png",
            "products": [
                {"id": f"product_{index}", "rgba_sha256": sha256_file(path)}
                for index, path in enumerate(products, 1)
            ],
        }
        audit_path = root / "audit.json"
        audit_path.write_text(json.dumps(audit), encoding="utf-8")
        return background, final, products, audit_path, audit

    @staticmethod
    def _review(
        score: float = 94.0,
        slot_id: int = 1,
        *,
        critical: bool = False,
    ) -> dict:
        return {
            # Deliberately inconsistent model totals/verdicts are evidence that
            # the server must use only the six raw dimensions.
            "visual_score": 1.0,
            "verdict": "reject",
            "batch_unlock": False,
            "subscores": {
                "grounding": score,
                "cast_shadows": score,
                "geometry_layout": score,
                "lighting_color": score,
                "cutout_edges": score,
                "reflection_material": score,
            },
            "critical_failures": ["floating object"] if critical else [],
            "issues": [],
            "recommended_adjustments": [],
            "model": "MiniMax-M3",
            "request_id": f"review-{slot_id}",
            "prompt_version": "test-prompt",
            "slot_id": slot_id,
        }

    def _consensus(
        self,
        scores: list[float] | None = None,
        *,
        critical_slots: set[int] | None = None,
    ) -> dict:
        scores = scores or [94.0] * 5
        critical_slots = critical_slots or set()
        return aggregate_visual_reviews([
            self._review(score, slot_id, critical=slot_id in critical_slots)
            for slot_id, score in enumerate(scores, 1)
        ])

    def _bundle(
        self,
        background: Path,
        final: Path,
        products: list[Path],
        audit_path: Path,
        audit: dict,
    ) -> dict:
        return build_review_bundle(
            background,
            final,
            products,
            audit,
            audit_path=audit_path,
            job_path=audit_path.parent / "job.json",
            reviewer_model="MiniMax-M3",
            prompt_version="test-prompt",
        )

    def test_server_gate_passes_only_when_visual_pixel_binding_and_4k_pass(self) -> None:
        with self._temporary_directory() as temporary:
            root = Path(temporary)
            background, final, products, audit_path, audit = self._artifacts(root)
            bundle = self._bundle(background, final, products, audit_path, audit)
            gate = build_final_gate(bundle, audit, self._consensus([94] * 5))

            # Local layer measurements are an upper bound.  They must not
            # inflate a valid visual judgement or conceal perceptual defects.
            self.assertEqual(gate["visual_score"], 94.0)
            self.assertEqual(gate["pixel_verdict"], "pass")
            self.assertEqual(gate["verdict"], "pass")
            self.assertTrue(gate["batch_unlock"])
            self.assertEqual(gate["hard_gates"]["output_4k_png"]["verdict"], "pass")

            gate_path = root / "final_gate.json"
            write_final_gate(gate_path, gate)
            verify_deliverable(final, json.loads(gate_path.read_text(encoding="utf-8")))

            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = cli_quality(Namespace(
                    final=str(final), gate=str(gate_path),
                ))
            self.assertEqual(exit_code, 0)
            cli_result = json.loads(output.getvalue())
            self.assertEqual(cli_result["verdict"], "pass")
            self.assertTrue(cli_result["batch_unlock"])

    def test_review_bundle_binds_job_bytes_and_review_semantics(self) -> None:
        with self._temporary_directory() as temporary:
            root = Path(temporary)
            background, final, products, audit_path, audit = self._artifacts(root)
            first = self._bundle(background, final, products, audit_path, audit)

            job_path = root / "job.json"
            job = json.loads(job_path.read_text(encoding="utf-8"))
            job["lighting"]["cast_shadow_mode"] = "ambient_only"
            job["lighting"]["shadow_length_ratio"] = 0.0
            job_path.write_text(json.dumps(job), encoding="utf-8")
            second = self._bundle(background, final, products, audit_path, audit)

            self.assertNotEqual(first["bundle_sha256"], second["bundle_sha256"])
            self.assertNotEqual(
                first["job_binding"]["job_sha256"],
                second["job_binding"]["job_sha256"],
            )
            self.assertNotEqual(
                first["job_binding"]["semantic_sha256"],
                second["job_binding"]["semantic_sha256"],
            )
            self.assertEqual(
                second["job_binding"]["semantics"]["lighting"]["cast_shadow_mode"],
                "ambient_only",
            )

    def test_delivery_rechecks_job_manifest_after_review(self) -> None:
        with self._temporary_directory() as temporary:
            root = Path(temporary)
            background, final, products, audit_path, audit = self._artifacts(root)
            gate = build_final_gate(
                self._bundle(background, final, products, audit_path, audit),
                audit,
                self._consensus([96] * 5),
            )
            write_final_gate(root / "final_gate.json", gate)
            verify_deliverable(final, gate)

            job_path = root / "job.json"
            job = json.loads(job_path.read_text(encoding="utf-8"))
            job["lighting"]["reflection_enabled"] = True
            job_path.write_text(json.dumps(job), encoding="utf-8")
            with self.assertRaisesRegex(
                FinalGateError, "job.json no longer matches",
            ):
                verify_deliverable(final, gate)

    def test_cli_quality_cannot_replace_five_slots_with_manual_claim(self) -> None:
        with self._temporary_directory() as temporary:
            root = Path(temporary)
            _, final, _, _, _ = self._artifacts(root)
            forged_gate = root / "final_gate.json"
            forged_gate.write_text(json.dumps({
                "visual_score": 99,
                "verdict": "pass",
                "batch_unlock": True,
            }), encoding="utf-8")

            with self.assertRaises(ContractError):
                cli_quality(Namespace(
                    final=str(final), gate=str(forged_gate),
                ))

    def test_below_ninety_can_never_unlock(self) -> None:
        with self._temporary_directory() as temporary:
            root = Path(temporary)
            background, final, products, audit_path, audit = self._artifacts(root)
            weak_reviews = [self._review(99, slot_id) for slot_id in range(1, 6)]
            for review in weak_reviews:
                review["subscores"]["lighting_color"] = 60
                review["subscores"]["cutout_edges"] = 99
                review["subscores"]["reflection_material"] = 40
            gate = build_final_gate(
                self._bundle(background, final, products, audit_path, audit),
                audit,
                aggregate_visual_reviews(weak_reviews),
            )
            self.assertEqual(gate["verdict"], "reject")
            self.assertFalse(gate["batch_unlock"])
            with self.assertRaises(FinalGateError):
                verify_deliverable(final, gate)

    def test_pixel_or_4k_failure_can_never_unlock(self) -> None:
        for dimensions, pixel_pass in (((3840, 2160), False), ((1920, 1080), True)):
            with self.subTest(dimensions=dimensions, pixel_pass=pixel_pass):
                with self._temporary_directory() as temporary:
                    root = Path(temporary)
                    background, final, products, audit_path, audit = self._artifacts(
                        root, dimensions=dimensions, pixel_pass=pixel_pass,
                    )
                    gate = build_final_gate(
                        self._bundle(background, final, products, audit_path, audit),
                        audit,
                        self._consensus([99] * 5),
                    )
                    self.assertEqual(gate["verdict"], "reject")
                    self.assertFalse(gate["batch_unlock"])

    def test_reviewed_background_must_match_audited_background(self) -> None:
        with self._temporary_directory() as temporary:
            root = Path(temporary)
            background, final, products, audit_path, audit = self._artifacts(root)
            other_background = root / "other_background.png"
            Image.new("RGB", (3840, 2160), (12, 34, 56)).save(other_background)
            bundle = self._bundle(
                other_background, final, products, audit_path, audit,
            )
            gate = build_final_gate(bundle, audit, self._consensus([99] * 5))

            binding = gate["hard_gates"]["audit_input_binding"]
            self.assertFalse(binding["background_hashes_match"])
            self.assertEqual(binding["verdict"], "reject")
            self.assertEqual(gate["verdict"], "reject")
            self.assertFalse(gate["batch_unlock"])

    def test_delivery_rechecks_final_hash(self) -> None:
        with self._temporary_directory() as temporary:
            root = Path(temporary)
            background, final, products, audit_path, audit = self._artifacts(root)
            gate = build_final_gate(
                self._bundle(background, final, products, audit_path, audit),
                audit,
                self._consensus([96] * 5),
            )
            write_final_gate(root / "final_gate.json", gate)
            deliver_file(str(final))

            Image.new("RGB", (3840, 2160), (0, 0, 0)).save(final)
            with self.assertRaises(HTTPException) as context:
                deliver_file(str(final))
            self.assertEqual(context.exception.status_code, 403)

    def test_consensus_rejects_score_dispersion_above_six(self) -> None:
        review = self._consensus([90, 91, 93, 99, 100])
        self.assertEqual(review["visual_score"], 93)
        self.assertEqual(review["aggregation"]["total_score_iqr"], 8)
        self.assertIn(
            "total_score_iqr_above_6",
            review["aggregation"]["rejection_reasons"],
        )
        self.assertEqual(review["verdict"], "reject")
        self.assertFalse(review["batch_unlock"])

    def test_consensus_rejects_unstable_critical_failure(self) -> None:
        review = self._consensus([95] * 5, critical_slots={2})
        self.assertEqual(review["aggregation"]["critical_slot_count"], 1)
        self.assertEqual(review["aggregation"]["critical_stability"], "unstable")
        self.assertEqual(review["verdict"], "reject")
        self.assertFalse(review["batch_unlock"])

    def test_total_above_ninety_cannot_hide_weak_grounding(self) -> None:
        reviews = []
        for slot_id in range(1, 6):
            review = self._review(96, slot_id)
            review["subscores"]["grounding"] = 80
            reviews.append(review)
        consensus = aggregate_visual_reviews(reviews)

        self.assertGreater(consensus["visual_score"], 90)
        self.assertIn(
            "grounding_median_below_88",
            consensus["aggregation"]["rejection_reasons"],
        )
        self.assertEqual(consensus["verdict"], "reject")
        self.assertFalse(consensus["batch_unlock"])

    def test_local_physical_pass_cannot_override_visual_floating_consensus(self) -> None:
        with self._temporary_directory() as temporary:
            root = Path(temporary)
            background, final, products, audit_path, audit = self._artifacts(root)
            reviews = []
            for slot_id in range(1, 6):
                review = self._review(96, slot_id)
                review["subscores"]["grounding"] = 74
                review["subscores"]["cast_shadows"] = 78
                reviews.append(review)
            gate = build_final_gate(
                self._bundle(background, final, products, audit_path, audit),
                audit,
                aggregate_visual_reviews(reviews),
            )

            self.assertEqual(gate["subscores"]["grounding"], 74)
            self.assertEqual(gate["subscores"]["cast_shadows"], 78)
            self.assertEqual(gate["verdict"], "reject")
            self.assertFalse(gate["batch_unlock"])

    def test_interrupted_consensus_retries_invalid_slot_once(self) -> None:
        with self._temporary_directory() as temporary:
            root = Path(temporary)
            background, final, products, _, _ = self._artifacts(root)
            payload = VisualReviewUI(
                background=str(background),
                final_path=str(final),
                products=[str(path) for path in products],
            )
            attempted: list[int] = []
            failed_once = False

            def provider(*args, slot_id: int, **kwargs):
                nonlocal failed_once
                attempted.append(slot_id)
                if slot_id == 3 and not failed_once:
                    failed_once = True
                    raise MiniMaxTransportError(
                        "simulated transport interruption",
                        attempted_routes=["system_proxy", "direct"],
                    )
                review = self._review(94, slot_id)
                review["prompt_version"] = VISUAL_REVIEW_PROMPT_VERSION
                return review

            with (
                patch("strict_compositor.ui.minimax_status", return_value={"model": "MiniMax-M3"}),
                patch("strict_compositor.ui.request_minimax_visual_review", side_effect=provider),
                patch("strict_compositor.ui.time.sleep") as sleep,
            ):
                completed = visual_review(payload)

            self.assertEqual(attempted, [1, 2, 3, 3, 4, 5])
            self.assertEqual(completed["review_slots_reused"], 0)
            self.assertEqual(completed["review_request_retries"], 1)
            self.assertEqual(completed["review_count"], 5)
            self.assertTrue((root / "final_gate.json").is_file())
            sleep.assert_called_once()

    def test_transport_interruption_is_pending_and_never_becomes_a_pass(self) -> None:
        with self._temporary_directory() as temporary:
            root = Path(temporary)
            background, final, products, _, _ = self._artifacts(root)
            payload = VisualReviewUI(
                background=str(background),
                final_path=str(final),
                products=[str(path) for path in products],
            )

            def provider(*args, **kwargs):
                raise MiniMaxTransportError(
                    "all approved routes unavailable",
                    attempted_routes=["system_proxy", "direct"],
                )

            with (
                patch("strict_compositor.ui.minimax_status", return_value={"model": "MiniMax-M3"}),
                patch("strict_compositor.ui.request_minimax_visual_review", side_effect=provider) as request,
                patch("strict_compositor.ui.time.sleep") as sleep,
            ):
                with self.assertRaises(HTTPException) as context:
                    visual_review(payload)

            self.assertEqual(context.exception.status_code, 503)
            self.assertEqual(request.call_count, 3)
            self.assertEqual(sleep.call_count, 2)
            pending = json.loads((root / "visual_review_pending.json").read_text(encoding="utf-8"))
            self.assertEqual(pending["state"], "pending")
            self.assertEqual(pending["reason"], "minimax_transport_error")
            self.assertEqual(pending["completed_slot_ids"], [])
            self.assertEqual(pending["pending_slot_id"], 1)
            self.assertEqual(pending["retry_policy"], "bounded-exponential-jitter-v1")
            self.assertEqual(pending["max_transport_attempts"], 3)
            self.assertEqual(len(pending["failed_attempts"]), 3)
            self.assertTrue(all(
                entry["error_kind"] == "transport_no_response"
                and entry["attempted_routes"] == ["system_proxy", "direct"]
                for entry in pending["failed_attempts"]
            ))
            self.assertFalse((root / "final_gate.json").exists())

    def test_valid_provider_error_is_not_transport_retried(self) -> None:
        with self._temporary_directory() as temporary:
            root = Path(temporary)
            background, final, products, _, _ = self._artifacts(root)
            payload = VisualReviewUI(
                background=str(background),
                final_path=str(final),
                products=[str(path) for path in products],
            )

            with (
                patch("strict_compositor.ui.minimax_status", return_value={"model": "MiniMax-M3"}),
                patch(
                    "strict_compositor.ui.request_minimax_visual_review",
                    side_effect=RuntimeError("simulated valid provider error"),
                ) as request,
                patch("strict_compositor.ui.time.sleep") as sleep,
            ):
                with self.assertRaises(HTTPException) as context:
                    visual_review(payload)

            self.assertEqual(context.exception.status_code, 400)
            self.assertEqual(request.call_count, 1)
            sleep.assert_not_called()
            self.assertFalse((root / "visual_review_pending.json").exists())

    def test_visual_review_route_caches_one_authoritative_result_per_bundle(self) -> None:
        with self._temporary_directory() as temporary:
            root = Path(temporary)
            background, final, products, _, _ = self._artifacts(root)
            payload = VisualReviewUI(
                background=str(background),
                final_path=str(final),
                products=[str(path) for path in products],
            )
            def provider(*args, slot_id: int, **kwargs):
                review = self._review(93, slot_id)
                review["prompt_version"] = VISUAL_REVIEW_PROMPT_VERSION
                return review

            with (
                patch("strict_compositor.ui.minimax_status", return_value={"model": "MiniMax-M3"}),
                patch("strict_compositor.ui.request_minimax_visual_review", side_effect=provider) as request,
            ):
                first = visual_review(payload)
                second = visual_review(payload)

            self.assertEqual(request.call_count, 5)
            self.assertFalse(first["review_cache_hit"])
            self.assertTrue(second["review_cache_hit"])
            self.assertEqual(first["review_count"], 5)
            self.assertEqual(len(first["review_slots"]), 5)
            self.assertEqual(first["bundle_sha256"], second["bundle_sha256"])
            self.assertTrue((root / "final_gate.json").is_file())

    def test_visual_review_slots_are_reused_across_runs_by_bundle_hash(self) -> None:
        with self._temporary_directory() as temporary:
            root = Path(temporary)
            shared = root / "shared"
            shared.mkdir()
            background, source_final, products, source_audit, _ = self._artifacts(shared)
            run_a = root / "run_a"
            run_b = root / "run_b"
            for run in (run_a, run_b):
                run.mkdir()
                shutil.copy2(source_final, run / "final.png")
                shutil.copy2(source_audit, run / "audit.json")
                shutil.copy2(shared / "job.json", run / "job.json")

            def payload(run: Path) -> VisualReviewUI:
                return VisualReviewUI(
                    background=str(background),
                    final_path=str(run / "final.png"),
                    products=[str(path) for path in products],
                )

            def provider(*args, slot_id: int, **kwargs):
                review = self._review(93, slot_id)
                review["prompt_version"] = VISUAL_REVIEW_PROMPT_VERSION
                return review

            workspace = root / "workspace"
            with (
                patch("strict_compositor.ui.WORK", workspace),
                patch("strict_compositor.ui.minimax_status", return_value={"model": "MiniMax-M3"}),
                patch("strict_compositor.ui.request_minimax_visual_review", side_effect=provider) as request,
            ):
                first = visual_review(payload(run_a))
                second = visual_review(payload(run_b))

            self.assertEqual(request.call_count, 5)
            self.assertFalse(first["review_cache_hit"])
            self.assertTrue(second["review_cache_hit"])
            self.assertEqual(second["review_slots_reused"], 5)
            self.assertEqual(first["bundle_sha256"], second["bundle_sha256"])
            bundle_hash = first["bundle_sha256"]
            for slot_id in range(1, 6):
                self.assertTrue(
                    (workspace / "visual_review_cache" / bundle_hash / f"slot_{slot_id}.json").is_file()
                )
                self.assertTrue(
                    (run_b / "visual_review_slots" / bundle_hash / f"slot_{slot_id}.json").is_file()
                )


if __name__ == "__main__":
    unittest.main()
