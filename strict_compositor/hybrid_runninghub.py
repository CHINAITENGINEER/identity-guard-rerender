from __future__ import annotations

from pathlib import Path
from typing import Any

from .hybrid import HybridRenderRequest, HybridRenderResult
from .runninghub import RunningHubError, configured_client
from .settings import get_public_settings


class RunningHubHybridProvider:
    """RunningHub masked-edit adapter with an independent workflow contract."""

    def __init__(self, timeout: float = 900.0) -> None:
        self.timeout = float(timeout)

    def render(self, request: HybridRenderRequest) -> HybridRenderResult:
        settings = get_public_settings()
        if not bool(settings["runninghub_hybrid_enabled"]):
            raise RunningHubError("RunningHub hybrid rendering is disabled")
        app_id = str(settings["runninghub_hybrid_app_id"])
        source_node = str(settings["runninghub_hybrid_source_node_id"])
        mask_node = str(settings["runninghub_hybrid_mask_node_id"])
        output_node = str(settings["runninghub_hybrid_output_node_id"])
        if not all((app_id, source_node, mask_node, output_node)):
            raise RunningHubError(
                "RunningHub hybrid workflow requires app, source, mask and output node IDs"
            )
        client = configured_client()
        source_name = client.upload(request.strict_base)
        mask_name = client.upload(request.edit_mask)
        nodes: list[dict[str, str]] = [
            {
                "nodeId": source_node,
                "fieldName": str(settings["runninghub_hybrid_source_field_name"] or "image"),
                "fieldValue": source_name,
            },
            {
                "nodeId": mask_node,
                "fieldName": str(settings["runninghub_hybrid_mask_field_name"] or "mask"),
                "fieldValue": mask_name,
            },
        ]
        protect_node = str(settings.get("runninghub_hybrid_protect_node_id") or "")
        if protect_node:
            protect_name = client.upload(request.protect_mask)
            nodes.append({
                "nodeId": protect_node,
                "fieldName": str(
                    settings.get("runninghub_hybrid_protect_field_name") or "mask"
                ),
                "fieldValue": protect_name,
            })
        prompt_node = str(settings["runninghub_hybrid_prompt_node_id"])
        if prompt_node and request.prompt:
            nodes.append({
                "nodeId": prompt_node,
                "fieldName": str(settings["runninghub_hybrid_prompt_field_name"] or "prompt"),
                "fieldValue": request.prompt,
            })
        task_id = client.create_task(
            app_id,
            nodes,
            mode=str(settings["runninghub_hybrid_mode"]),
        )
        result = client.wait(
            task_id,
            output_node_id=output_node,
            timeout=self.timeout,
        )
        suffix = str(result.file_type).lower().lstrip(".")
        if suffix not in {"png", "jpg", "jpeg", "webp", "tif", "tiff"}:
            suffix = "png"
        destination = request.output_dir / f"runninghub_hybrid_{task_id}.{suffix}"
        client.download(result.file_url, destination)
        return HybridRenderResult(
            candidate_path=Path(destination),
            provider="runninghub",
            task_id=task_id,
            provenance={
                "mode": str(settings["runninghub_hybrid_mode"]),
                "application_id": app_id,
                "output_node_id": result.node_id,
                "source_node_id": source_node,
                "mask_node_id": mask_node,
                "protect_node_id": protect_node or None,
            },
        )


def runninghub_hybrid_status() -> dict[str, Any]:
    settings = get_public_settings()
    required = (
        settings["runninghub_hybrid_app_id"],
        settings["runninghub_hybrid_source_node_id"],
        settings["runninghub_hybrid_mask_node_id"],
        settings["runninghub_hybrid_output_node_id"],
    )
    return {
        "enabled": bool(settings["runninghub_hybrid_enabled"]),
        "workflow_configured": all(bool(str(value).strip()) for value in required),
        "mode": settings["runninghub_hybrid_mode"],
        "provider": "runninghub",
    }
