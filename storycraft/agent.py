from dataclasses import dataclass, field
from datetime import timedelta
from typing import Optional, Any, Dict, Tuple

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from langchain_mcp_adapters.callbacks import Callbacks
from langchain_mcp_adapters.client import MultiServerMCPClient

from storycraft.config import StoryCraftSettings
from storycraft.memory.artifact_store import ArtifactStore
from storycraft.orchestration.node_manager import NodeManager
from storycraft.mcp.hooks import (
    handle_tool_errors,
    on_progress,
    log_tool_request,
)
from storycraft.mcp.sampling import make_sampling_callback
from storycraft.skills.loader import load_skills


# ============================================================
# 🎥 StoryCraft Runtime Context
# ============================================================

@dataclass
class StoryCraftContext:
    """
    Runtime context for StoryCraft AI Video Agent.

    This context represents a full cinematic production session,
    including models, assets, orchestration state, and skill pipelines.
    """

    cfg: StoryCraftSettings
    session_id: str

    # Media workspace
    media_dir: str
    bgm_dir: str
    outputs_dir: str

    # Orchestration
    node_manager: NodeManager

    # Model keys
    chat_model_key: str
    vlm_model_key: str = ""

    # Optional integrations
    pexels_api_key: Optional[str] = None
    tts_config: Optional[dict] = None

    # Runtime model pool (LLM/VLM reuse)
    llm_pool: Dict[Tuple[str, bool], ChatOpenAI] = field(default_factory=dict)

    # Language & narration defaults
    lang: str = "zh"


# ============================================================
# 🎬 StoryCraft Agent Factory
# ============================================================

async def build_storycraft_agent(
    cfg: StoryCraftSettings,
    session_id: str,
    store: ArtifactStore,
    tool_interceptors=None,
    *,
    llm_override: Optional[dict] = None,
    vlm_override: Optional[dict] = None,
):
    """
    Build StoryCraft cinematic AI agent.

    This factory constructs:
      - Core LLM & VLM engines
      - Sampling orchestration
      - MCP multi-server tool interface
      - Skill pipelines
      - LangChain agent runtime

    Returns:
        agent: Cinematic Orchestration Agent
        node_manager: Node orchestration controller
    """

    # --------------------------
    # Helper Utilities
    # --------------------------

    def _resolve(override: Optional[dict], key: str, default: Any) -> Any:
        return (
            override.get(key)
            if isinstance(override, dict) and key in override
            else default
        )

    def _normalize_url(url: str) -> str:
        url = (url or "").strip()
        return url.rstrip("/") if url else url

    # --------------------------
    # 1) Build Language Model
    # --------------------------

    llm_cfg = cfg.llm
    llm = ChatOpenAI(
        model=_resolve(llm_override, "model", llm_cfg.model),
        base_url=_normalize_url(_resolve(llm_override, "base_url", llm_cfg.base_url)),
        api_key=_resolve(llm_override, "api_key", llm_cfg.api_key),
        timeout=_resolve(llm_override, "timeout", llm_cfg.timeout),
        temperature=_resolve(llm_override, "temperature", llm_cfg.temperature),
        streaming=True,
        max_retries=_resolve(llm_override, "max_retries", llm_cfg.max_retries),
        default_headers={
            "api-key": llm_cfg.api_key,
            "Content-Type": "application/json",
        },
    )

    # --------------------------
    # 2) Build Vision-Language Model
    # --------------------------

    vlm_cfg = cfg.vlm
    vlm = ChatOpenAI(
        model=_resolve(vlm_override, "model", vlm_cfg.model),
        base_url=_normalize_url(_resolve(vlm_override, "base_url", vlm_cfg.base_url)),
        api_key=_resolve(vlm_override, "api_key", vlm_cfg.api_key),
        timeout=_resolve(vlm_override, "timeout", vlm_cfg.timeout),
        temperature=_resolve(vlm_override, "temperature", vlm_cfg.temperature),
        max_retries=_resolve(vlm_override, "max_retries", vlm_cfg.max_retries),
        default_headers={
            "api-key": vlm_cfg.api_key,
            "Content-Type": "application/json",
        },
    )

    # --------------------------
    # 3) Sampling Orchestration
    # --------------------------

    sampling_callback = make_sampling_callback(
        llm=llm,
        vlm=vlm,
    )

    # --------------------------
    # 4) MCP Multi-Agent Tool Mesh
    # --------------------------

    mcp_server = cfg.local_mcp_server
    connections = {
        mcp_server.server_name: {
            "transport": mcp_server.server_transport,
            "url": mcp_server.url,
            "timeout": timedelta(seconds=mcp_server.timeout),
            "sse_read_timeout": timedelta(minutes=30),
            "headers": {
                "X-StoryCraft-Session-Id": session_id,
            },
            "session_kwargs": {
                "sampling_callback": sampling_callback,
            },
        },
    }

    mcp_client = MultiServerMCPClient(
        connections=connections,
        tool_interceptors=tool_interceptors,
        callbacks=Callbacks(on_progress=on_progress),
        tool_name_prefix=True,
    )

    tools = await mcp_client.get_tools()

    # --------------------------
    # 5) Skill Pipeline Loader
    # --------------------------

    skills = await load_skills(cfg.skills.skill_dir)

    # --------------------------
    # 6) Node Orchestration Graph
    # --------------------------

    node_manager = NodeManager(tools=tools)

    # --------------------------
    # 7) Cinematic Agent Runtime
    # --------------------------

    agent = create_agent(
        model=llm,
        tools=tools + skills,
        middleware=[log_tool_request, handle_tool_errors],
        store=store,
        context_schema=StoryCraftContext,
    )

    return agent, node_manager
