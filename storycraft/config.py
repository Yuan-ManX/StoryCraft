# storycraft/configuration_utils.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Literal, List, Dict
from datetime import timedelta

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    computed_field,
    field_validator,
)


# ============================================================
# 🔧 Path Resolution Utilities
# ============================================================

def _resolve_relative_path_to_config_root(v: Path, info: ValidationInfo) -> Path:
    """
    Resolve relative paths based on config.toml directory (not cwd).

    Requires:
        model_validate(..., context={"config_root": <Path|str>})
    """
    ctx = info.context or {}
    base = ctx.get("config_root")
    if not base:
        return v

    v = v.expanduser()
    if v.is_absolute():
        return v

    root = Path(base).expanduser()
    return (root / v).resolve(strict=False)


def _resolve_paths_recursively(value: Any, info: ValidationInfo) -> Any:
    """
    Recursively resolve Path inside containers.
    """
    if value is None:
        return None

    if isinstance(value, Path):
        return _resolve_relative_path_to_config_root(value, info)

    if isinstance(value, list):
        return [_resolve_paths_recursively(v, info) for v in value]

    if isinstance(value, tuple):
        return tuple(_resolve_paths_recursively(v, info) for v in value)

    if isinstance(value, set):
        return {_resolve_paths_recursively(v, info) for v in value}

    if isinstance(value, dict):
        return {k: _resolve_paths_recursively(v, info) for k, v in value.items()}

    return value


# ============================================================
# 🧠 Base Config Model
# ============================================================

class StoryCraftConfigModel(BaseModel):
    """
    Base class for all StoryCraft configuration models.
    """

    model_config = ConfigDict(extra="forbid")

    @field_validator("*", mode="after")
    @classmethod
    def _resolve_all_paths(cls, v: Any, info: ValidationInfo) -> Any:
        field = cls.model_fields.get(info.field_name)
        extra = (field.json_schema_extra or {}) if field else {}
        if extra.get("resolve_relative") is False:
            return v
        return _resolve_paths_recursively(v, info)


# ============================================================
# 🧪 Developer & Runtime Debug Config
# ============================================================

class DeveloperConfig(StoryCraftConfigModel):
    developer_mode: bool = False
    default_llm: str = "deepseek-chat"
    default_vlm: str = "qwen3-vl-8b-instruct"
    model_registry: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    print_context: bool = False


# ============================================================
# 🎥 Project Workspace Config
# ============================================================

class ProjectConfig(StoryCraftConfigModel):
    media_dir: Path = Field(..., description="Input media workspace (videos / images)")
    bgm_dir: Path = Field(..., description="Background music library directory")
    outputs_dir: Path = Field(..., description="Final cinematic outputs directory")

    @computed_field(return_type=Path)
    @property
    def blobs_dir(self) -> Path:
        """
        Unified artifact storage (frames, clips, embeddings, drafts).
        """
        return self.outputs_dir


# ============================================================
# 🤖 Model Engine Config
# ============================================================

class LLMConfig(StoryCraftConfigModel):
    model: str
    base_url: str
    api_key: str
    timeout: float = 30.0
    temperature: Optional[float] = None
    max_retries: int = 2


class VLMConfig(StoryCraftConfigModel):
    model: str
    base_url: str
    api_key: str
    timeout: float = 20.0
    temperature: Optional[float] = None
    max_retries: int = 2


# ============================================================
# 🧠 Orchestration & Tool Mesh Config (MCP)
# ============================================================

class OrchestrationServerConfig(StoryCraftConfigModel):
    """
    MCP multi-agent orchestration server configuration.
    """

    server_name: str = "storycraft"
    server_cache_dir: str = "./storycraft/.server_cache"

    server_transport: Literal["stdio", "sse", "streamable-http"] = "streamable-http"
    url_scheme: str = "http"
    connect_host: str = "127.0.0.1"
    port: int = Field(ge=1, le=65535)
    path: str = "/mcp"

    json_response: bool = True
    stateless_http: bool = False
    timeout: int = 600

    available_node_pkgs: List[str] = []
    available_nodes: List[str] = []

    @property
    def url(self) -> str:
        return f"{self.url_scheme}://{self.connect_host}:{self.port}{self.path}"


# ============================================================
# 🛠 Skill Pipeline Config
# ============================================================

class SkillsConfig(StoryCraftConfigModel):
    skill_dir: Path = Field(..., description="Cinematic production skill pipelines.")


# ============================================================
# 🌐 Media Acquisition & Understanding
# ============================================================

class PexelsConfig(StoryCraftConfigModel):
    pexels_api_key: str = ""


class SplitShotsConfig(StoryCraftConfigModel):
    transnet_weights: Path = Field(..., description="TransNetV2 model weights")
    transnet_device: str = "cpu"


class UnderstandClipsConfig(StoryCraftConfigModel):
    sample_fps: float = 2.0
    max_frames: int = 64


# ============================================================
# 📝 Script & Narrative Planning
# ============================================================

class ScriptTemplateConfig(StoryCraftConfigModel):
    script_template_dir: Path = Field(..., description="Narrative script template directory")
    script_template_info_path: Path = Field(..., description="Template metadata index")


# ============================================================
# 🗣 Voice & Audio Design
# ============================================================

class GenerateVoiceoverConfig(StoryCraftConfigModel):
    tts_provider_params_path: Path = Field(..., description="TTS provider configuration")
    providers: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


class SelectBGMConfig(StoryCraftConfigModel):
    sample_rate: int = 22050
    hop_length: int = 2048
    frame_length: int = 2048


# ============================================================
# 🎨 Text Rendering & Subtitle Design
# ============================================================

class RecommendTextConfig(StoryCraftConfigModel):
    font_info_path: Path = Field(..., description="Subtitle font configuration")


# ============================================================
# ⏱ Timeline & Editing Rhythm Engine
# ============================================================

class TimelinePlanningConfig(StoryCraftConfigModel):
    beat_type_max: int = 1
    title_duration: int = 5000
    bgm_loop: bool = True
    min_clip_duration: int = 500

    estimate_text_min: int = 1500
    estimate_text_char_per_sec: float = 6.0

    image_default_duration: int = 3000
    group_margin_over_voiceover: int = 1000


class TimelinePlanningProConfig(StoryCraftConfigModel):
    min_single_text_duration: int = 200
    max_text_duration: int = 5000
    img_default_duration: int = 1500

    min_group_margin: int = 1500
    max_group_margin: int = 2000

    min_clip_duration: int = 1000

    tts_margin_mode: Literal["random", "avg", "max", "min"] = "random"
    min_tts_margin: int = 300
    max_tts_margin: int = 400

    text_tts_offset_mode: Literal["random", "avg", "max", "min"] = "random"
    min_text_tts_offset: int = 0
    max_text_tts_offset: int = 0

    long_short_text_duration: int = 3000
    long_text_margin_rate: float = 0.0
    short_text_margin_rate: float = 0.0

    text_duration_mode: Literal["with_tts", "with_clip"] = "with_tts"
    is_text_beats: bool = False


# ============================================================
# 🎬 StoryCraft Master Settings
# ============================================================

class StoryCraftSettings(StoryCraftConfigModel):
    developer: DeveloperConfig
    project: ProjectConfig

    llm: LLMConfig
    vlm: VLMConfig

    orchestration: OrchestrationServerConfig

    skills: SkillsConfig
    search_media: PexelsConfig
    split_shots: SplitShotsConfig
    understand_clips: UnderstandClipsConfig
    script_template: ScriptTemplateConfig
    generate_voiceover: GenerateVoiceoverConfig
    select_bgm: SelectBGMConfig
    recommend_text: RecommendTextConfig
    plan_timeline: TimelinePlanningConfig
    plan_timeline_pro: TimelinePlanningProConfig


# ============================================================
# ⚙ Config Loader
# ============================================================

def load_storycraft_settings(config_path: str | Path) -> StoryCraftSettings:
    path = Path(config_path).expanduser().resolve()
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return StoryCraftSettings.model_validate(
        data,
        context={"config_root": path.parent},
    )


def default_config_path() -> str:
    return os.getenv("STORYCRAFT_CONFIG", "config.toml")
