"""
StoryCraft Media Generation Service

ComfyUI Workflow-based media generation service.

Supports:
- Image generation
- Video generation
- Selfhost ComfyUI
- RunningHub backend

Design Goals:
- Strong typing
- Clear execution phases
- Unified error handling
- Extensible backend support
"""

from __future__ import annotations

from typing import Optional, Dict, Any
from loguru import logger

from storycraft.services.comfy_base import (
    ComfyBaseService,
    WorkflowInfo,
)
from storycraft.models.media import MediaResult


# ============================================================
# Media Service
# ============================================================


class MediaService(ComfyBaseService):
    """
    Media generation service (Image + Video).

    Execution Flow:
        1. Resolve workflow
        2. Build parameters
        3. Execute workflow
        4. Normalize result
        5. Return MediaResult
    """

    WORKFLOW_PREFIX = ""  # handled dynamically
    DEFAULT_WORKFLOW = None
    WORKFLOWS_DIR = "workflows"

    # ============================================================
    # Initialization
    # ============================================================

    def __init__(self, config: dict, core=None):
        super().__init__(config, service_name="image", core=core)

    # ============================================================
    # Workflow Discovery Override
    # ============================================================

    def _scan_workflows(self):
        """
        Override to support both image_ and video_ prefixes.
        """
        workflows = super()._scan_workflows()

        # Filter to image_ / video_
        filtered = [
            wf for wf in workflows
            if wf.name.startswith("image_")
            or wf.name.startswith("video_")
        ]

        return filtered

    # ============================================================
    # Public Execution
    # ============================================================

    async def __call__(
        self,
        prompt: str,
        workflow: Optional[str] = None,
        media_type: str = "image",  # "image" or "video"
        comfyui_url: Optional[str] = None,
        runninghub_api_key: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        duration: Optional[float] = None,
        negative_prompt: Optional[str] = None,
        steps: Optional[int] = None,
        seed: Optional[int] = None,
        cfg: Optional[float] = None,
        sampler: Optional[str] = None,
        **params,
    ) -> MediaResult:
        """
        Generate media using ComfyUI workflow.
        """

        workflow_info = self.resolve_workflow(workflow)

        workflow_params = self._build_workflow_params(
            prompt=prompt,
            media_type=media_type,
            width=width,
            height=height,
            duration=duration,
            negative_prompt=negative_prompt,
            steps=steps,
            seed=seed,
            cfg=cfg,
            sampler=sampler,
            extra=params,
        )

        result = await self._execute_workflow(
            workflow_info,
            workflow_params,
            comfyui_url=comfyui_url,
            runninghub_api_key=runninghub_api_key,
        )

        return self._build_media_result(result, media_type)

    # ============================================================
    # Internal Phases
    # ============================================================

    def _build_workflow_params(
        self,
        prompt: str,
        media_type: str,
        width: Optional[int],
        height: Optional[int],
        duration: Optional[float],
        negative_prompt: Optional[str],
        steps: Optional[int],
        seed: Optional[int],
        cfg: Optional[float],
        sampler: Optional[str],
        extra: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build workflow parameter dictionary.
        """

        params: Dict[str, Any] = {"prompt": prompt}

        optional_fields = {
            "width": width,
            "height": height,
            "duration": duration,
            "negative_prompt": negative_prompt,
            "steps": steps,
            "seed": seed,
            "cfg": cfg,
            "sampler": sampler,
        }

        for key, value in optional_fields.items():
            if value is not None:
                params[key] = value

        params.update(extra)

        logger.debug(f"[MediaService] Workflow params: {params}")
        return params

    async def _execute_workflow(
        self,
        workflow: WorkflowInfo,
        workflow_params: Dict[str, Any],
        comfyui_url: Optional[str],
        runninghub_api_key: Optional[str],
    ):
        """
        Execute workflow using ComfyKit.
        """

        try:
            kit = await self.core._get_or_create_comfykit()

            if workflow.source == "runninghub" and workflow.workflow_id:
                workflow_input = workflow.workflow_id
                logger.info(
                    f"[MediaService] Executing RunningHub workflow: {workflow_input}"
                )
            else:
                workflow_input = workflow.path
                logger.info(
                    f"[MediaService] Executing selfhost workflow: {workflow_input}"
                )

            result = await kit.execute(workflow_input, workflow_params)

            if result.status != "completed":
                self._raise_generation_error(result.msg)

            return result

        except Exception as e:
            logger.exception("[MediaService] Workflow execution failed")
            raise

    def _build_media_result(
        self,
        result,
        media_type: str,
    ) -> MediaResult:
        """
        Normalize ExecuteResult → MediaResult.
        """

        if media_type == "video":
            if not result.videos:
                self._raise_generation_error("No video generated")

            video_url = result.videos[0]
            duration = getattr(result, "duration", None)

            logger.info(f"✅ Generated video: {video_url}")

            return MediaResult(
                media_type="video",
                url=video_url,
                duration=duration,
            )

        else:
            if not result.images:
                self._raise_generation_error("No image generated")

            image_url = result.images[0]
            logger.info(f"✅ Generated image: {image_url}")

            return MediaResult(
                media_type="image",
                url=image_url,
            )

    # ============================================================
    # Error Handling
    # ============================================================

    def _raise_generation_error(self, message: Optional[str]):
        error_msg = message or "Media generation failed"
        logger.error(f"[MediaService] {error_msg}")
        raise RuntimeError(error_msg)

  
