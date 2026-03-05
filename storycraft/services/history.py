"""
History Manager Service

Business-level history management for StoryCraft tasks.

This service provides high-level operations on top of PersistenceService
and remains completely UI-agnostic.
"""

from __future__ import annotations

from typing import Optional, Any
from pathlib import Path
from loguru import logger

from storycraft.services.persistence import PersistenceService


# ============================================================
# History Manager
# ============================================================


class HistoryManager:
    """
    Task history management service.

    Responsibilities:
    - Task listing and filtering
    - Task detail retrieval
    - Task duplication (re-generation)
    - Task deletion
    - Statistics reporting

    This layer contains business logic only and delegates
    storage operations to PersistenceService.
    """

    # ============================================================
    # Initialization
    # ============================================================

    def __init__(self, persistence: PersistenceService):
        """
        Initialize HistoryManager.

        Args:
            persistence:
                Persistence service responsible for data storage.
        """
        self.persistence = persistence

    # ============================================================
    # Query APIs
    # ============================================================

    async def list_tasks(
        self,
        page: int = 1,
        page_size: int = 20,
        status: Optional[str] = None,
        sort_by: str = "created_at",
        sort_order: str = "desc",
    ) -> dict[str, Any]:
        """
        Retrieve paginated task list.

        Args:
            page:
                Page number (1-based).

            page_size:
                Number of items per page.

            status:
                Optional status filter.

            sort_by:
                Sorting field:
                - created_at
                - completed_at
                - title
                - duration

            sort_order:
                Sorting direction:
                - asc
                - desc

        Returns:
            Dictionary containing pagination data:

            {
                "tasks": [...],
                "total": 100,
                "page": 1,
                "page_size": 20,
                "total_pages": 5
            }
        """
        logger.debug(
            f"Listing tasks | page={page} size={page_size} status={status}"
        )

        return await self.persistence.list_tasks_paginated(
            page=page,
            page_size=page_size,
            status=status,
            sort_by=sort_by,
            sort_order=sort_order,
        )

    async def get_task(
        self,
        task_id: str,
    ) -> Optional[dict[str, Any]]:
        """
        Retrieve full task detail.

        Args:
            task_id:
                Task identifier.

        Returns:
            Dictionary containing metadata and storyboard:

            {
                "metadata": {...},
                "storyboard": {...}
            }

            Returns None if task does not exist.
        """
        logger.debug(f"Loading task detail: {task_id}")

        metadata = await self.persistence.load_task_metadata(task_id)
        if not metadata:
            logger.warning(f"Task not found: {task_id}")
            return None

        storyboard = await self.persistence.load_storyboard(task_id)

        return {
            "metadata": metadata,
            "storyboard": storyboard,
        }

    async def get_statistics(self) -> dict[str, Any]:
        """
        Retrieve system-wide task statistics.

        Returns:
            {
                "total_tasks": 100,
                "completed": 95,
                "failed": 5,
                "total_duration": 3600.5,
                "total_size": 1024000000
            }
        """
        logger.debug("Fetching task statistics")

        return await self.persistence.get_statistics()

    # ============================================================
    # Task Operations
    # ============================================================

    async def delete_task(
        self,
        task_id: str,
    ) -> bool:
        """
        Delete a task and all associated files.

        Args:
            task_id:
                Task identifier.

        Returns:
            True if deletion succeeded.
        """
        logger.info(f"Deleting task: {task_id}")

        success = await self.persistence.delete_task(task_id)

        if success:
            logger.info(f"Task deleted: {task_id}")
        else:
            logger.warning(f"Failed to delete task: {task_id}")

        return success

    async def duplicate_task(
        self,
        task_id: str,
    ) -> Optional[dict[str, Any]]:
        """
        Duplicate task parameters for regeneration.

        This feature enables users to quickly regenerate
        videos using previous parameters.

        Args:
            task_id:
                Task identifier.

        Returns:
            Input parameter dictionary:

            {
                "text": "...",
                "mode": "generate",
                "title": "...",
                "n_scenes": 5,
                "tts_inference_mode": "local",
                "tts_voice": "...",
                ...
            }

            Returns None if task does not exist.
        """
        logger.debug(f"Duplicating task parameters: {task_id}")

        metadata = await self.persistence.load_task_metadata(task_id)

        if not metadata:
            logger.warning(f"Task not found for duplication: {task_id}")
            return None

        input_params = metadata.get("input", {})

        logger.info(f"Task parameters duplicated: {task_id}")

        return input_params

    # ============================================================
    # Maintenance
    # ============================================================

    async def rebuild_index(self) -> None:
        """
        Rebuild task index.

        Useful when:
        - manual file edits occurred
        - index corruption
        - system maintenance
        """
        logger.info("Rebuilding task index")

        await self.persistence.rebuild_index()

        logger.info("Task index rebuilt successfully")

    # ============================================================
    # Future Extensions (Phase 3)
    # ============================================================

    async def regenerate_frame(
        self,
        task_id: str,
        frame_index: int,
        **override_params,
    ) -> Optional[str]:
        """
        Regenerate a specific frame.

        Planned functionality:
        - Reload original storyboard
        - Override frame parameters
        - Regenerate media
        - Update storyboard
        - Rebuild final video

        Args:
            task_id:
                Original task ID.

            frame_index:
                Frame index (0-based).

            **override_params:
                Parameters to override
                (e.g., image_prompt, style).

        Returns:
            Path to regenerated frame or None.
        """
        logger.warning(
            "regenerate_frame is not implemented yet (Phase 3 feature)"
        )

        return None

    async def export_task(
        self,
        task_id: str,
        export_path: str,
    ) -> Optional[str]:
        """
        Export task as portable archive.

        Planned export contents:
        - metadata.json
        - storyboard.json
        - generated video
        - frame assets

        Args:
            task_id:
                Task identifier.

            export_path:
                Target export file path (e.g., exports/task.zip).

        Returns:
            Path to exported file or None.
        """
        logger.warning(
            "export_task is not implemented yet (Phase 3 feature)"
        )

        return None

  
