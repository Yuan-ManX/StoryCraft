from __future__ import annotations

import ast
import asyncio
import contextvars
import json
import uuid
from typing import Any, Callable, Optional

from langchain.agents.middleware import wrap_tool_call
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from langchain_mcp_adapters.callbacks import CallbackContext
from langchain_mcp_adapters.interceptors import (
    MCPToolCallRequest,
    MCPToolCallResult,
)

from storycraft.config import Settings


# ============================================================
# Constants
# ============================================================

CUSTOM_MODEL_KEY = "__custom__"

SENSITIVE_KEYS = {
    "api_key", "access_token", "authorization", "token",
    "password", "secret", "x-api-key", "apikey",
}


# ============================================================
# Context variables
# ============================================================

_MCP_LOG_SINK = contextvars.ContextVar("mcp_log_sink", default=None)
_MCP_ACTIVE_TOOL_CALL_ID = contextvars.ContextVar(
    "mcp_active_tool_call_id", default=None
)


# ============================================================
# Context helpers
# ============================================================

def set_mcp_log_sink(
    sink: Optional[Callable[[dict], None]]
):
    return _MCP_LOG_SINK.set(sink)


def reset_mcp_log_sink(token):
    _MCP_LOG_SINK.reset(token)


# ============================================================
# Utility helpers
# ============================================================

def _normalize_url(url: str) -> str:
    url = (url or "").strip()
    return url.rstrip("/") if url else ""


def _mask_secrets(obj: Any) -> Any:
    """
    Recursive desensitization.

    Prevent sensitive information from appearing in:
      - console logs
      - tool traces
      - debug output
      - tool messages
    """
    try:
        if isinstance(obj, dict):
            return {
                k: "***" if str(k).lower() in SENSITIVE_KEYS else _mask_secrets(v)
                for k, v in obj.items()
            }
        if isinstance(obj, list):
            return [_mask_secrets(x) for x in obj]
        if isinstance(obj, tuple):
            return tuple(_mask_secrets(x) for x in obj)
        return obj
    except Exception:
        return "***"


# ============================================================
# LLM factory + cache pool
# ============================================================

def _make_chat_llm(
    cfg: Settings,
    model_name: str,
    *,
    streaming: bool,
) -> ChatOpenAI:
    model_cfg = cfg.developer.chat_models_config.get(model_name, {})

    base_url = _normalize_url(model_cfg.get("base_url", ""))
    api_key = model_cfg.get("api_key")

    return ChatOpenAI(
        model=model_name,
        base_url=base_url,
        api_key=api_key,
        default_headers={
            "api-key": api_key,
            "Content-Type": "application/json",
        },
        timeout=cfg.llm.timeout,
        temperature=model_cfg.get("temperature", cfg.llm.temperature),
        streaming=streaming,
    )


def get_llm(
    cfg: Settings,
    llm_pool: dict[tuple[str, bool], ChatOpenAI],
    *,
    model_name: str,
    streaming: bool,
) -> ChatOpenAI:
    """
    Cached ChatOpenAI factory.

    Key: (model_name, streaming)
    """
    key = (model_name, streaming)
    if key not in llm_pool:
        llm_pool[key] = _make_chat_llm(
            cfg, model_name, streaming=streaming
        )
    return llm_pool[key]


# ============================================================
# Tool interceptors
# ============================================================

@wrap_tool_call
async def log_tool_request(
    request: MCPToolCallRequest,
    handler,
):
    """
    Global MCP tool logging interceptor.
    """

    sink = _MCP_LOG_SINK.get()

    def emit(event: dict):
        if sink:
            sink(event)

    tool_call = request.tool_call
    tool_name = tool_call.get("name", "")

    # Auto inject tool_call_id if missing
    tool_call_id = tool_call.get("id")
    if not tool_call_id:
        tool_call_id = f"mcp_{uuid.uuid4().hex[:8]}"
        tool_call["id"] = tool_call_id

    active_tok = _MCP_ACTIVE_TOOL_CALL_ID.set(tool_call_id)

    extracted_args = _extract_safe_args(
        tool_call.get("args", {}),
        request.runtime.context.node_manager,
    )

    is_error = False
    summary = ""

    try:
        emit({
            "type": "tool_start",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "args": extracted_args,
        })

        print(f"[Tool Start] {tool_name} args={extracted_args}")

        out = await handler(request)

        summary, is_error = _parse_tool_output(out)

        return out

    finally:
        _MCP_ACTIVE_TOOL_CALL_ID.reset(active_tok)

        emit({
            "type": "tool_end",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "is_error": is_error,
            "summary": _mask_secrets(summary),
        })

        state = "ERROR" if is_error else "OK"
        print(f"[Tool End] {tool_name} state={state} summary={summary}\n")


# ============================================================
# Tool error normalization
# ============================================================

@wrap_tool_call
async def handle_tool_errors(
    request: MCPToolCallRequest,
    handler,
):
    """
    Unified exception-to-ToolMessage conversion.
    """

    try:
        out = await handler(request)

        if isinstance(out, Command):
            return out.update["messages"][0]

        if isinstance(out, MCPToolCallResult) and not isinstance(out.content, str):
            return ToolMessage(
                content=out.content[0].get("text", ""),
                tool_call_id=out.tool_call_id,
                name=out.name,
                additional_kwargs={
                    "isError": False,
                    "mcp_raw_text": True,
                },
            )

        return out

    except Exception as e:
        tc = request.tool_call
        safe_args = _mask_secrets(tc.get("args") or {})

        return ToolMessage(
            content=_format_tool_error(
                tc.get("name", ""),
                safe_args,
                e,
            ),
            tool_call_id=tc["id"],
            name=tc.get("name", ""),
            additional_kwargs={
                "isError": True,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "safe_args": safe_args,
            },
        )


# ============================================================
# Progress reporting
# ============================================================

async def on_progress(
    progress: float,
    total: float | None,
    message: str | None,
    context: CallbackContext,
):
    sink = _MCP_LOG_SINK.get()
    if sink:
        sink({
            "type": "tool_progress",
            "tool_call_id": _MCP_ACTIVE_TOOL_CALL_ID.get(),
            "server": context.server_name,
            "name": context.tool_name,
            "progress": progress,
            "total": total,
            "message": message,
        })


# ============================================================
# Streaming callback
# ============================================================

class PrintStreamingTokens(AsyncCallbackHandler):
    """
    Debug streaming output.
    """

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        if token:
            print(token, end="", flush=True)


# ============================================================
# Internal helpers
# ============================================================

def _extract_safe_args(raw_args: dict, node_manager) -> dict:
    exclude = set(node_manager.kind_to_node_ids.keys()) | {
        "inputs", "artifacts_dir", "artifact_id",
        "blobs_dir", "meta_path", "media_dir",
        "bgm_dir", "outputs_dir", "debug_dir",
    }

    args = {
        k: raw_args.get(k, "")
        for k in raw_args
        if k not in exclude
    }
    return _mask_secrets(args)


def _parse_tool_output(out) -> tuple[str, bool]:
    """
    Return (summary, is_error)
    """
    additional = getattr(out, "additional_kwargs", {}) or {}

    if additional.get("isError") is True:
        return str(out.content), True

    if additional.get("mcp_raw_text") is True:
        return str(out.content), False

    try:
        data = ast.literal_eval(out.content)
        return data.get("summary", {}).get("node_summary", ""), data.get("isError", False)
    except Exception:
        c = getattr(out, "content", "")
        return f"skill_ok len={len(c)}", False


def _format_tool_error(
    tool_name: str,
    safe_args: dict,
    exc: Exception,
) -> str:
    return (
        "Tool call failed\n"
        f"Tool name: {tool_name}\n"
        f"Tool params: {safe_args}\n"
        f"Error message: {type(exc).__name__}: {exc}\n\n"
        "Resolution guide:\n"
        "1) Check parameters\n"
        "2) Check missing dependency steps\n"
        "3) Retry if transient\n"
        "4) If unrecoverable — explain to user\n"
    )

