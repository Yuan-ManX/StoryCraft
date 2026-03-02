from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, model_validator


# ============================================================
# Progress Event Type
# ============================================================

class ProgressEventType(str, Enum):
    """
    High-level workflow progress stages.
    """

    INIT = "init"
    PREPROCESS = "preprocess"
    GENERATING_NARRATION = "generating_narration"
    FRAME_STEP = "frame_step"
    CONCATENATING = "concatenating"
    RENDERING = "rendering"
    POSTPROCESS = "postprocess"
    FINISHED = "finished"
    ERROR = "error"


# ============================================================
# Frame Processing Step Type
# ============================================================

class FrameStepType(str, Enum):
    """
    Detailed steps inside each frame generation.
    """

    AUDIO = "audio"
    IMAGE = "image"
    COMPOSE = "compose"
    VIDEO = "video"


# ============================================================
# Progress Event Protocol
# ============================================================

class ProgressEvent(BaseModel):
    """
    Structured workflow progress event.

    Used for:
    - UI real-time rendering
    - Agent execution monitoring
    - Pipeline debugging
    - Distributed task coordination

    Attributes:
        event_type:
            High-level progress stage.

        progress:
            Normalized progress in range [0.0, 1.0].

        frame_current:
            Current frame index (1-based), optional.

        frame_total:
            Total number of frames, optional.

        step:
            Current step index inside one frame.

        action:
            Detailed step action (audio / image / compose / video).

        message:
            Human-readable explanation.

        extra_info:
            Extended metadata for UI or debugging.
    """

    event_type: ProgressEventType = Field(
        description="High-level workflow event type"
    )

    progress: float = Field(
        ge=0.0,
        le=1.0,
        description="Normalized progress value in range [0.0, 1.0]"
    )

    frame_current: Optional[int] = Field(
        default=None,
        ge=1,
        description="Current frame index (1-based)"
    )

    frame_total: Optional[int] = Field(
        default=None,
        ge=1,
        description="Total frame count"
    )

    step: Optional[int] = Field(
        default=None,
        ge=1,
        le=8,
        description="Current step index within frame"
    )

    action: Optional[FrameStepType] = Field(
        default=None,
        description="Current frame processing action"
    )

    message: Optional[str] = Field(
        default=None,
        description="Human-readable event description"
    )

    extra_info: Optional[str] = Field(
        default=None,
        description="Extra debugging or UI hint information"
    )

    # ------------------------------------------------------------
    # Semantic Properties
    # ------------------------------------------------------------

    @property
    def is_frame_event(self) -> bool:
        return self.event_type == ProgressEventType.FRAME_STEP

    @property
    def is_terminal(self) -> bool:
        return self.event_type in {ProgressEventType.FINISHED, ProgressEventType.ERROR}

    # ------------------------------------------------------------
    # Validation Logic
    # ------------------------------------------------------------

    @model_validator(mode="after")
    def _validate_frame_fields(self) -> "ProgressEvent":
        """
        Enforce semantic correctness:
        - frame_current <= frame_total
        - step/action only valid for FRAME_STEP
        """

        if self.frame_current and self.frame_total:
            if self.frame_current > self.frame_total:
                raise ValueError(
                    f"frame_current({self.frame_current}) > frame_total({self.frame_total})"
                )

        if self.event_type != ProgressEventType.FRAME_STEP:
            if self.step is not None or self.action is not None:
                raise ValueError(
                    "step/action fields are only allowed for FRAME_STEP events"
                )

        return self

    # ------------------------------------------------------------
    # Factory Helpers (Builder API)
    # ------------------------------------------------------------

    @classmethod
    def simple(cls, event_type: ProgressEventType, progress: float, message: str = None):
        """Build simple progress event."""
        return cls(
            event_type=event_type,
            progress=progress,
            message=message,
        )

    @classmethod
    def frame_step(
        cls,
        progress: float,
        frame_current: int,
        frame_total: int,
        step: int,
        action: FrameStepType,
        message: str = None,
    ):
        """Build frame-level step progress event."""
        return cls(
            event_type=ProgressEventType.FRAME_STEP,
            progress=progress,
            frame_current=frame_current,
            frame_total=frame_total,
            step=step,
            action=action,
            message=message,
        )

    @classmethod
    def finished(cls, message: str = "Generation finished"):
        """Build finished event."""
        return cls(
            event_type=ProgressEventType.FINISHED,
            progress=1.0,
            message=message,
        )

    @classmethod
    def error(cls, message: str, extra_info: str = None):
        """Build error event."""
        return cls(
            event_type=ProgressEventType.ERROR,
            progress=1.0,
            message=message,
            extra_info=extra_info,
        )
      
