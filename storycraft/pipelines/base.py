from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Callable, Any
from loguru import logger

from storycraft.models.progress import ProgressEvent, ProgressEventType
from storycraft.models.storyboard import VideoGenerationResult


# ============================================================
# Base Pipeline
# ============================================================

class BasePipeline(ABC):
    """
    Abstract base class for video generation pipelines.

    Each pipeline represents a full, independent video generation workflow.

    Principles:
    - Stateless execution (no global state)
    - Explicit progress reporting
    - Unified return protocol (VideoGenerationResult)
    - Compatible with async orchestration
    """

    def __init__(self, storycraft_core: Any):
        """
        Initialize pipeline with core services.

        Args:
            pixelle_video_core:
                Core service container providing:
                - llm
                - tts
                - media
                - video
        """
        self.core = storycraft_core

        # Service shortcuts (convenience)
        self.llm = storycraft_core.llm
        self.tts = storycraft_core.tts
        self.media = storycraft_core.media
        self.video = storycraft_core.video

        # Backward compatibility alias
        self.image = self.media

    # ============================================================
    # Execution Entry
    # ============================================================

    @abstractmethod
    async def __call__(
        self,
        text: str,
        progress_callback: Optional[Callable[[ProgressEvent], None]] = None,
        **kwargs
    ) -> VideoGenerationResult:
        """
        Execute the pipeline.

        Args:
            text:
                Input text (semantics defined by specific pipeline).

            progress_callback:
                Optional callback for structured progress updates.

            **kwargs:
                Pipeline-specific arguments.

        Returns:
            VideoGenerationResult

        Raises:
            Exception:
                Pipeline-specific errors.
        """
        raise NotImplementedError

    # ============================================================
    # Progress Reporting
    # ============================================================

    def _emit_progress(
        self,
        callback: Optional[Callable[[ProgressEvent], None]],
        event: ProgressEvent,
    ) -> None:
        """
        Emit a structured progress event.

        Args:
            callback:
                Optional progress callback.
            event:
                Structured ProgressEvent object.
        """
        if callback:
            callback(event)

        logger.debug(
            f"[{self.__class__.__name__}] "
            f"{event.event_type.value} | "
            f"{event.progress * 100:.1f}%"
        )

    def _report_progress(
        self,
        callback: Optional[Callable[[ProgressEvent], None]],
        event_type: ProgressEventType,
        progress: float,
        **kwargs
    ) -> None:
        """
        Convenience wrapper to build and emit progress events.

        Args:
            callback:
                Progress callback function.

            event_type:
                ProgressEventType enum.

            progress:
                Normalized progress [0.0, 1.0].

            **kwargs:
                Additional ProgressEvent fields.
        """
        event = ProgressEvent(
            event_type=event_type,
            progress=progress,
            **kwargs
        )
        self._emit_progress(callback, event)

    # ============================================================
    # Lifecycle Helpers
    # ============================================================

    def _report_started(
        self,
        callback: Optional[Callable[[ProgressEvent], None]]
    ) -> None:
        self._report_progress(
            callback,
            ProgressEventType.INIT,
            0.0,
            message="Pipeline started"
        )

    def _report_finished(
        self,
        callback: Optional[Callable[[ProgressEvent], None]]
    ) -> None:
        self._report_progress(
            callback,
            ProgressEventType.FINISHED,
            1.0,
            message="Pipeline finished"
        )

    def _report_error(
        self,
        callback: Optional[Callable[[ProgressEvent], None]],
        error: Exception
    ) -> None:
        """
        Standardized error reporting.
        """
        self._report_progress(
            callback,
            ProgressEventType.ERROR,
            1.0,
            message=str(error),
            extra_info=type(error).__name__,
        )
      
