from __future__ import annotations
from typing import Dict, List, Literal, Any, Optional, Union, Tuple
from pydantic import BaseModel, Field, model_validator


# ============================================================
# 🎬 Core Media Metadata
# ============================================================

class VideoMetadata(BaseModel):
    width: int = Field(..., description="Frame width in pixels")
    height: int = Field(..., description="Frame height in pixels")
    duration_ms: float = Field(..., description="Duration in milliseconds")
    fps: float = Field(..., description="Frames per second")
    has_audio: bool = Field(default=False, description="Whether audio track exists")

    audio_sample_rate_hz: Optional[int] = Field(
        None, gt=0, description="Audio sample rate in Hz"
    )

    @model_validator(mode="after")
    def validate_audio_metadata(self):
        if self.has_audio and self.audio_sample_rate_hz is None:
            raise ValueError("audio_sample_rate_hz required when has_audio=True")
        return self


class ImageMetadata(BaseModel):
    width: int = Field(..., description="Image width in pixels")
    height: int = Field(..., description="Image height in pixels")


# ============================================================
# 🎥 Media & Clip Abstractions
# ============================================================

class MediaAsset(BaseModel):
    """
    Original media asset.
    """
    media_id: str
    path: str
    media_type: Literal["video", "image", "audio"]
    metadata: Union[VideoMetadata, ImageMetadata]
    tags: Optional[List[str]] = Field(default=None, description="Semantic tags")
    extra: Optional[Dict[str, Any]] = None


class ClipSegment(BaseModel):
    """
    Atomic narrative clip unit.
    """
    clip_id: str
    media_id: str
    language: Optional[str] = None
    caption: str = Field(default="", description="Narrative description")
    path: str
    fps: Optional[float] = None
    semantic_tags: Optional[List[str]] = None
    extra: Optional[Dict[str, Any]] = None


# ============================================================
# 🧩 Narrative & Script Layer
# ============================================================

class SubtitleUnit(BaseModel):
    unit_id: str
    index: int = Field(..., ge=0, description="Index within subtitle group")
    text: str = Field(..., description="Subtitle text")


class ShotGroup(BaseModel):
    """
    Visual storytelling group.
    """
    group_id: str
    narrative_intent: str = Field(..., description="Narrative purpose of this group")
    clip_ids: List[str] = Field(..., description="Ordered clip sequence")


class ShotScript(BaseModel):
    group_id: str
    raw_text: str
    subtitle_units: List[SubtitleUnit]


# ============================================================
# 🎙 Audio Narrative Layer
# ============================================================

class VoiceoverTrack(BaseModel):
    group_id: str
    voice_id: str
    path: str
    duration_ms: int = Field(..., gt=0)


class BGMTrack(BaseModel):
    bgm_id: str
    path: str
    duration_ms: int = Field(..., gt=0)
    bpm: float = Field(..., gt=0)
    beats: List[int] = Field(default_factory=list)


# ============================================================
# ⏱ Timeline Composition
# ============================================================

class TimeWindow(BaseModel):
    start_ms: int
    end_ms: int


class AudioMixConfig(BaseModel):
    gain_db: float = Field(default=0.0)
    ducking: Optional[Dict[str, Any]] = None


class ClipTimelineTrack(BaseModel):
    clip_id: str
    source_window: TimeWindow
    timeline_window: TimeWindow


class SubtitleTimelineTrack(BaseModel):
    text: str
    timeline_window: TimeWindow


class VoiceTimelineTrack(BaseModel):
    media_id: str
    timeline_window: TimeWindow


class BGMTimelineTrack(BaseModel):
    bgm_id: str
    timeline_window: TimeWindow
    mix: AudioMixConfig


class TimelineComposition(BaseModel):
    video: List[ClipTimelineTrack] = Field(default_factory=list)
    subtitles: List[SubtitleTimelineTrack] = Field(default_factory=list)
    voiceover: List[VoiceTimelineTrack] = Field(default_factory=list)
    bgm: List[BGMTimelineTrack] = Field(default_factory=list)


# ============================================================
# 🎛 Agent Input DSL Layer
# ============================================================

class AgentBaseInput(BaseModel):
    mode: Literal["auto", "skip", "default"] = "auto"


class SearchMediaInput(AgentBaseInput):
    photo_number: int = 0
    video_number: int = 5
    keyword: str = "scenery"
    orientation: Literal["landscape", "portrait"] = "landscape"
    min_video_duration: int = 1
    max_video_duration: int = 30


class SplitShotsInput(AgentBaseInput):
    min_duration_ms: int = 1000
    max_duration_ms: int = 10000


class GenerateScriptInput(AgentBaseInput):
    user_prompt: str = ""
    custom_script: Optional[Dict[str, Any]] = None


class GenerateVoiceoverInput(AgentBaseInput):
    user_prompt: str = ""


class SelectBGMInput(AgentBaseInput):
    user_prompt: str = ""
    filter_include: Dict[str, List[Union[str, int]]] = {}
    filter_exclude: Dict[str, List[Union[str, int]]] = {}


class PlanTimelineInput(AgentBaseInput):
    sync_with_beats: bool = True


# ============================================================
# 🎞 Final Render Control
# ============================================================

class RenderVideoInput(AgentBaseInput):
    aspect_ratio: Optional[str] = None
    output_max_px: Optional[int] = 1080
    compose_mode: Literal["padding", "crop"] = "padding"
    bg_color: Tuple[int, int, int] = (0, 0, 0)
    crf: int = Field(default=23, ge=10, le=30)

    # subtitle
    font_color: Tuple[int, int, int, int] = (255, 255, 255, 255)
    font_size: int = 40
    margin_bottom: int = 270
    stroke_width: int = 2
    stroke_color: Tuple[int, int, int, int] = (0, 0, 0, 255)

    # audio
    bgm_volume: float = 0.25
    tts_volume: float = 2.0
    include_video_audio: bool = False
