from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, model_validator


# ============================================================
# Media Type Enum
# ============================================================

class MediaType(str, Enum):
    """
    Supported media types for generation outputs.
    """

    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"      # Reserved for future
    MODEL_3D = "3d"      # Reserved for future


# ============================================================
# Base Media Result Protocol
# ============================================================

class MediaResult(BaseModel):
    """
    Unified protocol for workflow media generation results.

    This model standardizes the output of multi-modal generation pipelines
    (image / video / audio / 3D), enabling downstream agents, UI, and orchestration
    layers to consume results in a consistent format.

    Attributes:
        media_type:
            Type of generated media.
            Supported: image, video.
            Reserved: audio, 3d.

        url:
            URL or filesystem path pointing to the generated media artifact.

        duration:
            Duration in seconds.
            Required for video/audio, None for images.

    Examples:
        Image:
            MediaResult(
                media_type=MediaType.IMAGE,
                url="http://example.com/image.png"
            )

        Video:
            MediaResult(
                media_type=MediaType.VIDEO,
                url="http://example.com/video.mp4",
                duration=5.2
            )
    """

    media_type: MediaType = Field(
        description="Type of generated media artifact"
    )

    url: str = Field(
        description="URL or filesystem path to the generated media"
    )

    duration: Optional[float] = Field(
        default=None,
        description="Duration in seconds (required for video/audio, None for images)"
    )

    # ------------------------------------------------------------
    # Semantic Properties
    # ------------------------------------------------------------

    @property
    def is_image(self) -> bool:
        """Return True if the result is an image."""
        return self.media_type == MediaType.IMAGE

    @property
    def is_video(self) -> bool:
        """Return True if the result is a video."""
        return self.media_type == MediaType.VIDEO

    @property
    def is_audio(self) -> bool:
        """Return True if the result is audio (reserved)."""
        return self.media_type == MediaType.AUDIO

    @property
    def is_3d(self) -> bool:
        """Return True if the result is a 3D asset (reserved)."""
        return self.media_type == MediaType.MODEL_3D

    # ------------------------------------------------------------
    # Validation Logic
    # ------------------------------------------------------------

    @model_validator(mode="after")
    def _validate_duration(self) -> "MediaResult":
        """
        Enforce semantic correctness:
        - Video/audio must have duration
        - Image/3D must NOT have duration
        """
        if self.media_type in {MediaType.VIDEO, MediaType.AUDIO}:
            if self.duration is None or self.duration <= 0:
                raise ValueError(
                    f"`duration` must be provided and > 0 for media_type={self.media_type}"
                )
        else:
            if self.duration is not None:
                raise ValueError(
                    f"`duration` must be None for media_type={self.media_type}"
                )
        return self

    # ------------------------------------------------------------
    # Factory Helpers
    # ------------------------------------------------------------

    @classmethod
    def image(cls, url: str) -> "MediaResult":
        """Factory method for image result."""
        return cls(media_type=MediaType.IMAGE, url=url)

    @classmethod
    def video(cls, url: str, duration: float) -> "MediaResult":
        """Factory method for video result."""
        return cls(media_type=MediaType.VIDEO, url=url, duration=duration)

    @classmethod
    def audio(cls, url: str, duration: float) -> "MediaResult":
        """Factory method for audio result (future use)."""
        return cls(media_type=MediaType.AUDIO, url=url, duration=duration)

    @classmethod
    def model_3d(cls, url: str) -> "MediaResult":
        """Factory method for 3D model result (future use)."""
        return cls(media_type=MediaType.MODEL_3D, url=url)
