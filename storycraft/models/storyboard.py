from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field, model_validator


# ============================================================
# Enums
# ============================================================

class MediaType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"


class TTSInferenceMode(str, Enum):
    LOCAL = "local"
    COMFYUI = "comfyui"


# ============================================================
# Storyboard Configuration
# ============================================================

class StoryboardConfig(BaseModel):
    """
    Storyboard generation configuration.
    """

    # Media geometry
    media_width: int = Field(..., description="Media width (px)")
    media_height: int = Field(..., description="Media height (px)")

    # Task isolation
    task_id: Optional[str] = Field(
        default=None,
        description="Task ID for workspace isolation (auto-generated if None)"
    )

    # Storyboard structure
    n_storyboard: int = Field(5, ge=1, description="Number of storyboard frames")

    min_narration_words: int = Field(5, ge=1)
    max_narration_words: int = Field(20, ge=1)

    min_image_prompt_words: int = Field(30, ge=1)
    max_image_prompt_words: int = Field(60, ge=1)

    # Video parameters
    video_fps: int = Field(30, ge=1, description="Video frame rate")

    # Audio parameters
    tts_inference_mode: TTSInferenceMode = Field(
        default=TTSInferenceMode.LOCAL,
        description="TTS inference mode"
    )
    voice_id: Optional[str] = Field(default=None)
    tts_workflow: Optional[str] = Field(default=None)
    tts_speed: Optional[float] = Field(default=None, ge=0.5, le=2.0)
    ref_audio: Optional[str] = Field(default=None)

    # Media workflow
    media_workflow: Optional[str] = Field(default=None)

    # Frame template
    frame_template: str = Field(
        default="1080x1920/default.html",
        description="Template path containing resolution info"
    )
    template_params: Optional[Dict[str, Any]] = Field(default=None)


# ============================================================
# Storyboard Frame
# ============================================================

class StoryboardFrame(BaseModel):
    """
    Single storyboard frame unit.
    """

    index: int = Field(..., ge=0, description="Frame index (0-based)")
    narration: str = Field(..., description="Narration text")
    image_prompt: Optional[str] = Field(
        default=None,
        description="Image generation prompt"
    )

    # Generated resources
    audio_path: Optional[str] = None
    media_type: Optional[MediaType] = None

    image_path: Optional[str] = None
    video_path: Optional[str] = None
    composed_image_path: Optional[str] = None
    video_segment_path: Optional[str] = None

    # Metadata
    duration: float = Field(0.0, ge=0, description="Frame duration in seconds")
    created_at: datetime = Field(default_factory=datetime.now)

    # ------------------------------------------------------------
    # Semantic Properties
    # ------------------------------------------------------------

    @property
    def is_finished(self) -> bool:
        return self.video_segment_path is not None

    @property
    def has_media(self) -> bool:
        return self.media_type is not None

    @property
    def is_image(self) -> bool:
        return self.media_type == MediaType.IMAGE

    @property
    def is_video(self) -> bool:
        return self.media_type == MediaType.VIDEO


# ============================================================
# Content Metadata
# ============================================================

class ContentMetadata(BaseModel):
    """
    High-level content metadata for narration and UI display.
    """

    title: str
    author: Optional[str] = None
    subtitle: Optional[str] = None
    genre: Optional[str] = None
    summary: Optional[str] = None
    publication_year: Optional[str] = None
    cover_url: Optional[str] = None


# ============================================================
# Storyboard Root Object
# ============================================================

class Storyboard(BaseModel):
    """
    Complete storyboard protocol object.
    """

    title: str
    config: StoryboardConfig

    frames: List[StoryboardFrame] = Field(default_factory=list)

    content_metadata: Optional[ContentMetadata] = None

    final_video_path: Optional[str] = None
    total_duration: float = Field(0.0, ge=0)

    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    # ------------------------------------------------------------
    # Semantic Properties
    # ------------------------------------------------------------

    @property
    def is_completed(self) -> bool:
        return all(frame.is_finished for frame in self.frames)

    @property
    def progress(self) -> float:
        if not self.frames:
            return 0.0
        completed = sum(1 for f in self.frames if f.is_finished)
        return completed / len(self.frames)

    @property
    def finished_frames(self) -> List[StoryboardFrame]:
        return [f for f in self.frames if f.is_finished]

    @property
    def pending_frames(self) -> List[StoryboardFrame]:
        return [f for f in self.frames if not f.is_finished]

    # ------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------

    @model_validator(mode="after")
    def _sync_total_duration(self) -> "Storyboard":
        if self.frames:
            self.total_duration = sum(f.duration for f in self.frames)
        return self


# ============================================================
# Video Generation Result
# ============================================================

class VideoGenerationResult(BaseModel):
    """
    Final video generation result protocol.
    """

    video_path: str
    storyboard: Storyboard
    duration: float = Field(..., ge=0)
    file_size: int = Field(..., ge=0)
    created_at: datetime = Field(default_factory=datetime.now)
  
