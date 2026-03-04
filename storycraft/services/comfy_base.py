"""
StoryCraft ComfyUI Base Service

Provides unified workflow discovery and ComfyKit configuration
for all ComfyUI-based services (TTS, Image, etc.)

Design Goals:
- Strong typing
- Explicit workflow resolution contract
- Cached workflow discovery
- Clear config priority rules
- Extensible source abstraction
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from loguru import logger
from comfykit import ComfyKit

from storycraft.utils.os_util import (
    get_resource_path,
    list_resource_files,
    list_resource_dirs,
)

# ============================================================
# Workflow Data Model
# ============================================================


@dataclass(frozen=True)
class WorkflowInfo:
    """
    Structured workflow metadata.
    """

    name: str
    display_name: str
    source: str
    path: str
    key: str
    workflow_id: Optional[str] = None


# ============================================================
# ComfyUI Base Service
# ============================================================


class ComfyBaseService:
    """
    Abstract base service for ComfyUI workflow-based capabilities.

    Subclasses must define:
        WORKFLOW_PREFIX
        DEFAULT_WORKFLOW (optional if config driven)
        WORKFLOWS_DIR
    """

    WORKFLOW_PREFIX: str = ""
    DEFAULT_WORKFLOW: str = ""
    WORKFLOWS_DIR: str = "workflows"

    # ============================================================
    # Initialization
    # ============================================================

    def __init__(
        self,
        config: dict,
        service_name: str,
        core: Optional[Any] = None,
    ):
        self.service_name = service_name
        self.core = core

        comfyui_config = config.get("comfyui", {})
        self.config = comfyui_config.get(service_name, {})
        self.global_config = comfyui_config

        self._workflow_cache: Optional[List[WorkflowInfo]] = None

    # ============================================================
    # Workflow Discovery
    # ============================================================

    def _scan_workflows(self) -> List[WorkflowInfo]:
        """
        Discover all workflow JSON files across resource locations.

        Results are cached after first scan.
        """
        if self._workflow_cache is not None:
            return self._workflow_cache

        workflows: List[WorkflowInfo] = []
        source_dirs = list_resource_dirs(self.WORKFLOWS_DIR)

        if not source_dirs:
            logger.warning("No workflow source directories found")
            self._workflow_cache = []
            return []

        for source in source_dirs:
            workflow_files = list_resource_files(self.WORKFLOWS_DIR, source)

            matching = [
                f
                for f in workflow_files
                if f.startswith(self.WORKFLOW_PREFIX) and f.endswith(".json")
            ]

            for filename in matching:
                try:
                    file_path = Path(
                        get_resource_path(
                            self.WORKFLOWS_DIR,
                            source,
                            filename,
                        )
                    )

                    wf = self._parse_workflow_file(file_path, source)
                    workflows.append(wf)

                    logger.debug(f"[{self.service_name}] Found workflow: {wf.key}")

                except Exception as e:
                    logger.error(
                        f"[{self.service_name}] Failed to parse "
                        f"{source}/{filename}: {e}"
                    )

        workflows.sort(key=lambda w: w.key)
        self._workflow_cache = workflows
        return workflows

    def _parse_workflow_file(
        self,
        file_path: Path,
        source: str,
    ) -> WorkflowInfo:
        """
        Parse workflow JSON and extract metadata.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            content = json.load(f)

        workflow_id = None
        if isinstance(content, dict) and "workflow_id" in content:
            workflow_id = content.get("workflow_id")

        return WorkflowInfo(
            name=file_path.name,
            display_name=f"{file_path.name} - {source.title()}",
            source=source,
            path=str(file_path),
            key=f"{source}/{file_path.name}",
            workflow_id=workflow_id,
        )

    # ============================================================
    # Workflow Resolution
    # ============================================================

    def _get_default_workflow(self) -> str:
        default = self.config.get("default_workflow")

        if not default:
            raise ValueError(
                f"No default workflow configured for '{self.service_name}'. "
                f"Available: {', '.join(self.available) or 'none'}"
            )

        return default

    def resolve_workflow(
        self,
        workflow_key: Optional[str] = None,
    ) -> WorkflowInfo:
        """
        Resolve workflow key into WorkflowInfo.
        """
        if workflow_key is None:
            workflow_key = self._get_default_workflow()

        workflows = self._scan_workflows()

        for wf in workflows:
            if wf.key == workflow_key:
                logger.info(
                    f"[{self.service_name}] Using workflow: {workflow_key}"
                )
                return wf

        available = ", ".join([w.key for w in workflows]) or "none"
        raise ValueError(
            f"Workflow '{workflow_key}' not found. "
            f"Available workflows: {available}"
        )

    # ============================================================
    # ComfyKit Configuration
    # ============================================================

    def build_comfykit_config(
        self,
        comfyui_url: Optional[str] = None,
        runninghub_api_key: Optional[str] = None,
        runninghub_instance_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build ComfyKit configuration with strict priority rules.
        """

        config: Dict[str, Any] = {}

        config["comfyui_url"] = (
            comfyui_url
            or self.global_config.get("comfyui_url")
            or os.getenv("COMFYUI_BASE_URL")
            or "http://127.0.0.1:8188"
        )

        api_key = (
            runninghub_api_key
            or self.global_config.get("runninghub_api_key")
            or os.getenv("RUNNINGHUB_API_KEY")
        )

        if api_key:
            config["runninghub_api_key"] = api_key

        instance_type = (
            runninghub_instance_type
            or self.global_config.get("runninghub_instance_type")
            or os.getenv("RUNNINGHUB_INSTANCE_TYPE")
        )

        if instance_type and instance_type.strip():
            config["runninghub_instance_type"] = instance_type

        logger.debug(f"[{self.service_name}] ComfyKit config: {config}")
        return config

    # ============================================================
    # Public APIs
    # ============================================================

    def list_workflows(self) -> List[WorkflowInfo]:
        return self._scan_workflows()

    @property
    def available(self) -> List[str]:
        return [wf.key for wf in self._scan_workflows()]

    def create_comfykit(self, **kwargs) -> ComfyKit:
        """
        Factory method for creating ComfyKit instance.
        Allows shared instance via core if needed.
        """
        if self.core and hasattr(self.core, "comfykit"):
            return self.core.comfykit

        config = self.build_comfykit_config(**kwargs)
        return ComfyKit(**config)

    # ============================================================
    # Representation
    # ============================================================

    def __repr__(self) -> str:
        try:
            default = self._get_default_workflow()
        except Exception:
            default = None

        return (
            f"<{self.__class__.__name__} "
            f"service='{self.service_name}' "
            f"default={default!r} "
            f"available={len(self.available)}>"
        )
      
