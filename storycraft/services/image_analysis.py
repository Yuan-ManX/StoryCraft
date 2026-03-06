"""
Image Analysis Service

Workflow-based image analysis powered by ComfyUI or RunningHub.

This service executes image analysis workflows (e.g., Florence-2, BLIP)
and returns a textual description of the image.
"""

from __future__ import annotations

from typing import Optional, Literal
from pathlib import Path

import aiohttp
from loguru import logger

from storycraft.services.comfy_base import ComfyBaseService
from storycraft.utils.workflow_util import resolve_workflow_path


# ============================================================
# Image Analysis Service
# ============================================================


class ImageAnalysisService(ComfyBaseService):
    """
    Image analysis service based on ComfyUI workflows.

    This service executes vision analysis pipelines such as:
    - Florence-2
    - BLIP
    - other captioning models

    Workflow convention:

        {source}/analyse_image.json

    Examples:
        runninghub/analyse_image.json
        selfhost/analyse_image.json

    Example Usage:

        description = await storycraft.image_analysis("image.jpg")

        description = await storycraft.image_analysis(
            "image.jpg",
            source="selfhost"
        )

        workflows = storycraft.image_analysis.list_workflows()
    """

    WORKFLOW_PREFIX = "analyse_"
    WORKFLOWS_DIR = "workflows"

    # ============================================================
    # Initialization
    # ============================================================

    def __init__(self, config: dict, core=None):
        """
        Initialize image analysis service.

        Args:
            config:
                Global application configuration.

            core:
                StoryCraftCore instance for accessing shared services.
        """
        super().__init__(
            config=config,
            service_name="image_analysis",
            core=core,
        )

    # ============================================================
    # Public API
    # ============================================================

    async def __call__(
        self,
        image_path: str,
        *,
        source: Literal["runninghub", "selfhost"] = "runninghub",
        workflow: Optional[str] = None,
        comfyui_url: Optional[str] = None,
        runninghub_api_key: Optional[str] = None,
        **params,
    ) -> str:
        """
        Analyze an image and return textual description.

        Args:
            image_path:
                Path to image file (local).

            source:
                Workflow source:
                - runninghub (cloud)
                - selfhost (local ComfyUI)

            workflow:
                Explicit workflow path override.

            comfyui_url:
                Optional override for ComfyUI endpoint.

            runninghub_api_key:
                Optional override for RunningHub API key.

            **params:
                Additional workflow parameters.

        Returns:
            Image description text.

        Raises:
            FileNotFoundError
            RuntimeError
        """

        image_path_obj = Path(image_path)

        if not image_path_obj.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # ------------------------------------------------------------
        # Resolve Workflow
        # ------------------------------------------------------------

        workflow = self._resolve_workflow_path(source, workflow)

        workflow_info = self._resolve_workflow(workflow=workflow)

        # ------------------------------------------------------------
        # Prepare Parameters
        # ------------------------------------------------------------

        workflow_params = {
            "image": str(image_path),
            **params,
        }

        logger.debug(f"Workflow parameters: {workflow_params}")

        # ------------------------------------------------------------
        # Execute Workflow
        # ------------------------------------------------------------

        try:

            kit = await self.core._get_or_create_comfykit()

            workflow_input = self._resolve_workflow_input(workflow_info)

            logger.info(f"Executing workflow: {workflow_input}")

            result = await kit.execute(workflow_input, workflow_params)

            if result.status != "completed":
                raise RuntimeError(result.msg or "Image analysis failed")

            description = await self._extract_description(result.outputs)

            logger.info(
                f"Image analysis completed: {description[:80]}..."
            )

            return description

        except Exception as e:
            logger.exception("Image analysis failed")
            raise

    # ============================================================
    # Workflow Resolution
    # ============================================================

    def _resolve_workflow_path(
        self,
        source: str,
        workflow: Optional[str],
    ) -> str:
        """
        Resolve workflow path using naming convention.
        """

        if workflow:
            return workflow

        resolved = resolve_workflow_path("analyse_image", source)

        logger.info(f"Using {source} workflow: {resolved}")

        return resolved

    def _resolve_workflow_input(
        self,
        workflow_info: dict,
    ):
        """
        Determine what should be passed to ComfyKit.

        RunningHub → workflow_id
        Selfhost → workflow path
        """

        if (
            workflow_info.get("source") == "runninghub"
            and "workflow_id" in workflow_info
        ):
            return workflow_info["workflow_id"]

        return workflow_info["path"]

    # ============================================================
    # Result Extraction
    # ============================================================

    async def _extract_description(
        self,
        outputs: dict,
    ) -> str:
        """
        Extract description text from workflow outputs.
        """

        # ------------------------------------------------------------
        # Format 1 — Selfhost Output
        # ------------------------------------------------------------

        description = self._extract_selfhost_text(outputs)

        if description:
            return description

        # ------------------------------------------------------------
        # Format 2 — RunningHub Output
        # ------------------------------------------------------------

        description = await self._extract_runninghub_text(outputs)

        if description:
            return description

        raise RuntimeError("No description found in workflow outputs")

    def _extract_selfhost_text(
        self,
        outputs: dict,
    ) -> Optional[str]:
        """
        Extract caption from self-hosted ComfyUI outputs.

        Example:

            {'6': {'text': ['description']}}
        """

        if not outputs:
            return None

        for node_output in outputs.values():

            if isinstance(node_output, dict) and "text" in node_output:

                text_list = node_output["text"]

                if text_list:
                    return text_list[0]

        return None

    async def _extract_runninghub_text(
        self,
        outputs: dict,
    ) -> Optional[str]:
        """
        Extract caption from RunningHub raw_data.

        Example:

            {
                "raw_data": [
                    {
                        "fileUrl": "...txt",
                        "fileType": "txt"
                    }
                ]
            }
        """

        if not outputs or "raw_data" not in outputs:
            return None

        raw_data = outputs["raw_data"]

        for item in raw_data:

            if item.get("fileType") != "txt":
                continue

            url = item.get("fileUrl")

            if not url:
                continue

            return await self._download_text(url)

        return None

    async def _download_text(self, url: str) -> str:
        """
        Download text content from URL.
        """

        async with aiohttp.ClientSession() as session:

            async with session.get(url) as resp:

                if resp.status != 200:
                    raise RuntimeError(f"Failed to download: {url}")

                text = await resp.text()

                return text.strip()

          
