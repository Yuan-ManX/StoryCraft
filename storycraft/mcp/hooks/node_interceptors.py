from __future__ import annotations

import json
import os
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any

from langchain_core.messages import ToolMessage, ToolCall
from langchain_core.tools import ToolException
from langgraph.types import Command
from langchain_mcp_adapters.interceptors import MCPToolCallRequest
from mcp.types import CallToolResult

from storycraft.core.node_manager import NodeManager
from storycraft.memory.file import FileCompressor
from storycraft.utils.logging import get_logger


logger = get_logger(__name__)


# ============================================================
# Media Payload Compressor
# ============================================================

def compress_payload_to_base64(payload: Dict[str, List[Any]]):
    """
    Recursively compress file path payloads into base64 + md5.

    Payload structure:
        {
            "inputs": [
                {"path": "..."},
                ...
            ]
        }
    """
    if not isinstance(payload, dict):
        return payload

    for key, value in payload.items():
        if isinstance(value, list) and all(isinstance(item, dict) for item in value):
            for item in value:
                path = item.get("path")
                if not path:
                    continue
                compress_data = FileCompressor.compress_and_encode(path)
                item.update({
                    "path": path,
                    "base64": compress_data.base64,
                    "md5": compress_data.md5,
                })
        elif isinstance(value, dict):
            compress_payload_to_base64(value)


# ============================================================
# Tool Interceptor (StoryCraft Core)
# ============================================================

class ToolInterceptor:
    """
    StoryCraft Tool Interceptor System

    Responsible for:
      - Auto dependency resolution
      - Media auto injection
      - Artifact lifecycle management
      - DAG execution补全
    """

    # ========================== BEFORE ==========================

    @staticmethod
    async def inject_media_content_before(
        request: MCPToolCallRequest,
        handler,
    ):
        try:
            runtime = request.runtime
            context = runtime.context
            store = runtime.store

            node_id = request.name
            session_id = context.session_id
            lang = context.lang
            artifact_id = store.generate_artifact_id(node_id)
            node_manager: NodeManager = context.node_manager

            input_data = defaultdict(list)

            # ------------------------------------------------
            # Load dependencies
            # ------------------------------------------------
            if node_id == "load_media":
                ToolInterceptor._load_local_media(context, input_data)

            elif node_id in node_manager.id_to_tool:
                await ToolInterceptor._resolve_node_dependencies(
                    request,
                    node_id=node_id,
                    input_data=input_data,
                    node_manager=node_manager,
                    session_id=session_id,
                    store=store,
                )

            else:
                input_data["artifacts_dir"] = store.artifacts_dir

            # ------------------------------------------------
            # Merge args
            # ------------------------------------------------
            new_args = {
                "artifact_id": artifact_id,
                "lang": lang,
                **request.args,
                **input_data,
            }

            return await handler(request.override(args=new_args))

        except Exception as e:
            logger.error("[ToolInterceptor] BEFORE ERROR\n" + "".join(traceback.format_exception(e)))
            raise

    # ========================== AFTER ==========================

    @staticmethod
    async def save_media_content_after(
        request: MCPToolCallRequest,
        handler,
    ):
        try:
            runtime = request.runtime
            context = runtime.context
            store = runtime.store

            tool_call_result: CallToolResult = await handler(request)

            result_json = tool_call_result.model_dump()
            tool_result = json.loads(result_json["content"][0]["text"])

            node_id = request.name
            session_id = context.session_id
            artifact_id = tool_result["artifact_id"]

            if not tool_result["isError"]:
                save_dir = Path(context.media_dir) if node_id == "search_media" else None
                store.save_result(
                    session_id,
                    node_id,
                    tool_result,
                    save_dir,
                )

            tool_call_id = runtime.tool_call_id

            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content={
                                "summary": {
                                    "node_summary": tool_result["summary"],
                                    "tool_excute_result": (
                                        tool_result["tool_excute_result"]
                                        if node_id == "read_node_history"
                                        else {}
                                    ),
                                },
                                "isError": tool_result["isError"],
                            },
                            tool_call_id=tool_call_id,
                        )
                    ],
                    "status": "done",
                }
            )

        except Exception as e:
            logger.error("[ToolInterceptor] AFTER ERROR\n" + "".join(traceback.format_exception(e)))
            raise

    # ========================== INJECTORS ==========================

    @staticmethod
    async def inject_tts_config(request: MCPToolCallRequest, handler):
        try:
            if "voiceover" not in request.name:
                return await handler(request)

            ctx = getattr(request.runtime, "context", None)
            tts_cfg = getattr(ctx, "tts_config", None)

            if not isinstance(tts_cfg, dict):
                return await handler(request)

            provider = str(tts_cfg.get("provider", "")).strip().lower() or "302"
            request.args.setdefault("provider", provider)

            provider_cfg = tts_cfg.get(provider, {})
            if isinstance(provider_cfg, dict):
                for k, v in provider_cfg.items():
                    if v is not None:
                        request.args.setdefault(k, str(v).strip())

        except Exception as e:
            logger.warning(f"[TTS Inject] {e}")

        return await handler(request)

    @staticmethod
    async def inject_pexels_api_key(request: MCPToolCallRequest, handler):
        try:
            if "search_media" not in request.name:
                return await handler(request)

            ctx = getattr(request.runtime, "context", None)
            key = str(getattr(ctx, "pexels_api_key", "")).strip()

            if key:
                request.args["pexels_api_key"] = key

        except Exception as e:
            logger.warning(f"[Pexels Inject] {e}")

        return await handler(request)

    # ========================== INTERNAL ==========================

    @staticmethod
    def _load_local_media(context, input_data):
        media_dir = Path(context.media_dir)
        for file in os.listdir(media_dir):
            path = media_dir / file
            if path.is_dir():
                continue
            compress_data = FileCompressor.compress_and_encode(path)
            input_data["inputs"].append({
                "path": str(path.relative_to(os.getcwd())),
                "base64": compress_data.base64,
                "md5": compress_data.md5,
            })

    @staticmethod
    async def _resolve_node_dependencies(
        request,
        *,
        node_id,
        input_data,
        node_manager,
        session_id,
        store,
    ):
        mode = request.args.get("mode", "auto")
        require_kind = (
            node_manager.id_to_default_require_prior_kind[node_id]
            if mode != "auto"
            else node_manager.id_to_require_prior_kind[node_id]
        )

        collect_result = node_manager.check_excutable(
            session_id, store, require_kind
        )

        if collect_result["excutable"]:
            ToolInterceptor._load_collected_payload(
                collect_result["collected_node"], input_data, store
            )
            return

        missing = collect_result["missing_kind"]

        logger.info(f"[Dependency] `{node_id}` missing {missing}, resolving...")

        await ToolInterceptor._execute_missing_dependencies(
            missing,
            node_manager=node_manager,
            session_id=session_id,
            store=store,
            runtime=request.runtime,
        )

        # Re-check after补齐
        collect_result = node_manager.check_excutable(
            session_id, store, require_kind
        )
        ToolInterceptor._load_collected_payload(
            collect_result["collected_node"], input_data, store
        )

    @staticmethod
    def _load_collected_payload(collected_node, input_data, store):
        for collect_kind, artifact_meta in collected_node.items():
            _, prior_output = store.load_result(artifact_meta.artifact_id)
            compress_payload_to_base64(prior_output["payload"])
            input_data[collect_kind] = prior_output["payload"]

    @staticmethod
    async def _execute_missing_dependencies(
        missing_kinds,
        *,
        node_manager,
        session_id,
        store,
        runtime,
        depth=0,
    ):
        indent = "  " * depth
        for kind in missing_kinds:
            candidates = node_manager.kind_to_node_ids[kind]
            for node_id in candidates:
                try:
                    await ToolInterceptor._execute_node_default(
                        node_id,
                        node_manager=node_manager,
                        session_id=session_id,
                        store=store,
                        runtime=runtime,
                        depth=depth,
                    )
                    logger.info(f"{indent}✓ `{node_id}` satisfied `{kind}`")
                    break
                except ToolException:
                    continue
            else:
                raise ToolException(
                    f"Dependency `{kind}` unsatisfied, all candidates failed: {candidates}"
                )

    @staticmethod
    async def _execute_node_default(
        node_id,
        *,
        node_manager,
        session_id,
        store,
        runtime,
        depth=0,
    ):
        indent = "  " * depth
        logger.info(f"{indent}→ Auto executing `{node_id}`")

        tool = node_manager.get_tool(node_id)

        args = {
            "artifact_id": store.generate_artifact_id(node_id),
            "mode": "default",
        }

        require = node_manager.id_to_default_require_prior_kind[node_id]
        collect = node_manager.check_excutable(session_id, store, require)

        if not collect["excutable"]:
            await ToolInterceptor._execute_missing_dependencies(
                collect["missing_kind"],
                node_manager=node_manager,
                session_id=session_id,
                store=store,
                runtime=runtime,
                depth=depth + 1,
            )

        ToolInterceptor._load_collected_payload(
            collect["collected_node"], args, store
        )

        try:
            return await tool.arun(
                ToolCall(
                    args=args,
                    tool_call_type="default",
                    runtime=runtime,
                )
            )
        except Exception as e:
            raise ToolException(f"`{node_id}` execution failed: {e}")

      
