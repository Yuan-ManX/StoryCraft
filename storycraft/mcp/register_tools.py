from __future__ import annotations

import inspect
import traceback
from dataclasses import asdict
from typing import Annotated, Any, Callable

from pydantic import BaseModel, Field

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession

from storycraft.config import Settings
from storycraft.mcp.sampling_requester import make_llm
from storycraft.memory.agent_memory import ArtifactStore
from storycraft.core.core_nodes.base_node import BaseNode
from storycraft.core.node_state import NodeState
from storycraft.core.node_summary import NodeSummary
from storycraft.skills.skills_io import dump_skills
from storycraft.utils.logging import get_logger
from storycraft.utils.register import NODE_REGISTRY


logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# Tool Wrapper Factory
# -----------------------------------------------------------------------------

def create_tool_wrapper(
    node: BaseNode,
    input_schema: type[BaseModel] | None,
) -> tuple[Callable[..., Any], Any]:
    """
    Convert StoryCraft Node into MCP Tool callable.

    Responsibilities:
      - Context extraction
      - Session lifecycle management
      - Parameter normalization
      - NodeState construction
      - Error isolation
    """
    meta = getattr(node, "meta", None)

    async def wrapper(mcp_ctx: Context, **kwargs) -> dict:
        request = mcp_ctx.request_context.request
        headers = request.headers

        session_id = headers.get("X-StoryCraft-Session-Id", "unknown_session")

        # Session lifecycle hook
        session_manager = mcp_ctx.request_context.lifespan_context
        if hasattr(session_manager, "cleanup_expired_sessions"):
            session_manager.cleanup_expired_sessions(session_id)

        # Merge arguments (FastMCP + raw json)
        req_json = await request.json()
        params = {}
        params.update(kwargs)
        params.update(req_json.get("params", {}).get("arguments", {}))

        # Safety check
        if "artifact_id" not in params:
            raise ValueError("Missing required parameter: artifact_id")

        node_state = NodeState(
            session_id=session_id,
            artifact_id=params["artifact_id"],
            lang=params.get("lang", "zh"),
            node_summary=NodeSummary(),
            llm=make_llm(mcp_ctx),
            mcp_ctx=mcp_ctx,
        )

        try:
            result = await node(node_state, **params)
            return result
        except Exception as e:
            logger.exception(f"[Node] {meta.name if meta else node.__class__.__name__} execution failed")
            raise e

    _inject_signature(wrapper, input_schema, meta)
    return wrapper, meta


# -----------------------------------------------------------------------------
# Signature Injection
# -----------------------------------------------------------------------------

def _inject_signature(
    fn: Callable,
    schema: type[BaseModel] | None,
    meta: Any,
) -> None:
    """
    Inject dynamic function signature & annotations for FastMCP schema generation.
    """
    params = [
        inspect.Parameter(
            "mcp_ctx",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=Context,
        )
    ]

    annotations = {"mcp_ctx": Context}

    if schema:
        for field_name, field in schema.model_fields.items():
            ann = Annotated[field.annotation, field]

            params.append(
                inspect.Parameter(
                    field_name,
                    inspect.Parameter.KEYWORD_ONLY,
                    default=field.default if field.default is not ... else inspect.Parameter.empty,
                    annotation=ann,
                )
            )
            annotations[field_name] = ann

    fn.__name__ = meta.name
    fn.__doc__ = meta.description
    fn.__signature__ = inspect.Signature(params)
    fn.__annotations__ = annotations


# -----------------------------------------------------------------------------
# Tool Registration
# -----------------------------------------------------------------------------

def register(server: FastMCP, cfg: Settings) -> None:
    """
    Register all StoryCraft Nodes as MCP tools.
    """
    logger.info("[MCP] Scanning node packages")

    for pkg in cfg.local_mcp_server.available_node_pkgs:
        NODE_REGISTRY.scan_package(pkg)

    node_classes = [
        NODE_REGISTRY.get(name)
        for name in cfg.local_mcp_server.available_nodes
    ]

    for NodeClass in node_classes:
        node = NodeClass(cfg)
        tool_fn, meta = create_tool_wrapper(node, node.input_schema)

        server.tool(
            name=meta.name,
            description=meta.description,
            meta=asdict(meta),
        )(tool_fn)

        logger.info(f"[MCP] Tool registered: {meta.name}")

    _register_builtin_tools(server)


# -----------------------------------------------------------------------------
# Built-in Tools
# -----------------------------------------------------------------------------

def _register_builtin_tools(server: FastMCP) -> None:
    """
    Register system-level MCP tools.
    """

    @server.tool(
        name="read_node_history",
        description="Retrieve the execution result of any node using artifact_id",
    )
    async def read_node_history(
        mcp_ctx: Context[ServerSession, object],
        query_artifact_id: Annotated[str, Field(description="Artifact ID to query")],
    ) -> dict:

        request = mcp_ctx.request_context.request
        session_id = request.headers.get("X-StoryCraft-Session-Id", "unknown_session")

        store = ArtifactStore(".storycraft/.server_cache", session_id)

        try:
            meta, data = store.load_result(query_artifact_id)
            return {
                "artifact_id": store.generate_artifact_id("read_node_history"),
                "tool_execute_result": {"history": {"meta": meta, "node_data": data}},
                "summary": "History retrieved successfully.",
                "isError": False,
            }
        except Exception as e:
            traceback_info = "".join(traceback.format_exception(e))
            return {
                "artifact_id": store.generate_artifact_id("read_node_history"),
                "tool_execute_result": {},
                "summary": f"History retrieval failed:\n{traceback_info}",
                "isError": True,
            }

    @server.tool(
        name="write_skills",
        description="Persist LLM-generated skill markdown into filesystem.",
    )
    async def write_skills(
        mcp_ctx: Context[ServerSession, object],
        skill_name: Annotated[str, Field(description="Skill file name (no extension)")],
        skill_dir: Annotated[str, Field(description="Storage directory")] = ".storycraft/skills/",
        skill_content: Annotated[str, Field(description="Markdown content")] = "",
    ) -> dict:

        request = mcp_ctx.request_context.request
        session_id = request.headers.get("X-StoryCraft-Session-Id", "unknown_session")

        await dump_skills(
            skill_name=skill_name,
            skill_dir=skill_dir,
            skill_content=skill_content,
        )

        store = ArtifactStore(".storycraft/.server_cache", session_id)

        return {
            "artifact_id": store.generate_artifact_id("write_skills"),
            "tool_execute_result": {},
            "summary": "[Write Skills] Done.",
            "isError": False,
        }

  
