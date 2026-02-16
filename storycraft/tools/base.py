from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, ClassVar
import os
import json
import traceback

from pydantic import BaseModel, ValidationError

from storycraft.config import Settings
from storycraft.core.node_state import NodeState
from storycraft.memory.file import FileCompressor
from storycraft.utils.logging import get_logger
from storycraft.mcp.sampling_requester import LLMClient


logger = get_logger(__name__)


# ============================================================
# Node Metadata
# ============================================================

@dataclass(slots=True)
class NodeMeta:
    """
    Declarative metadata for workflow nodes.
    """

    name: str
    description: str
    node_id: str
    node_kind: str

    require_prior_kind: List[str] = field(default_factory=list)
    default_require_prior_kind: List[str] = field(default_factory=list)
    next_available_node: List[str] = field(default_factory=list)

    priority: int = 5


# ============================================================
# Node Base Class
# ============================================================

class BaseNode(ABC):
    """
    Abstract execution unit of StoryCraft workflow engine.

    Responsibilities:
        - Input decoding (base64 → server cache)
        - Output encoding (server cache → base64 / client path)
        - Schema validation
        - Lifecycle control
        - Execution dispatch
    """

    meta: NodeMeta

    input_schema: ClassVar[type[BaseModel] | None] = None
    output_schema: ClassVar[type[BaseModel] | None] = None

    def __init__(self, server_cfg: Settings) -> None:
        self.server_cfg = server_cfg
        self.cache_root = (
            Path.cwd() / self.server_cfg.local_mcp_server.server_cache_dir
        ).resolve()

        if not hasattr(self, "meta"):
            raise RuntimeError("Node subclass must define class attribute: meta")

        logger.debug(
            f"[NodeInit] {self.meta.node_id} cache_root={self.cache_root}"
        )

    # ============================================================
    # Context Helpers
    # ============================================================

    def _build_user_context(
        self, node_state: NodeState, params: Dict[str, Any]
    ) -> Dict[str, str]:
        return {
            "session_id": node_state.session_id,
            "artifact_id": node_state.artifact_id,
        }

    # ============================================================
    # Input Pipeline
    # ============================================================

    def _restore_item(
        self,
        node_state: NodeState,
        user_ctx: Dict[str, str],
        item: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Restore base64 → file on server cache.
        """
        result = dict(item)

        base64_data = result.pop("base64", None)
        orig_md5 = result.pop("md5", None)
        orig_path = result.pop("path", None)

        if base64_data and orig_path:
            save_path = (
                self.cache_root
                / user_ctx["session_id"]
                / user_ctx["artifact_id"]
                / os.path.basename(orig_path)
            )

            save_path.parent.mkdir(parents=True, exist_ok=True)
            FileCompressor.decompress_from_string(base64_data, save_path)

            result.update(
                {
                    "path": str(save_path.relative_to(Path.cwd())),
                    "orig_path": orig_path,
                    "orig_md5": orig_md5,
                }
            )

        return result

    def load_inputs_from_client(
        self,
        node_state: NodeState,
        params: Dict[str, Any],
        *,
        save_snapshot: bool = True,
    ) -> Dict[str, Any]:
        """
        Decode payload inputs and persist server cache snapshot.
        """
        user_ctx = self._build_user_context(node_state, params)

        decoded: Dict[str, Any] = {}
        passthrough: Dict[str, Any] = {}

        for key, value in params.items():
            if isinstance(value, list) and all(
                isinstance(v, dict) for v in value
            ):
                decoded[key] = [
                    self._restore_item(node_state, user_ctx, v)
                    for v in value
                ]
            elif isinstance(value, dict):
                decoded[key] = self.load_inputs_from_client(
                    node_state, value, save_snapshot=False
                )
            elif isinstance(value, LLMClient):
                passthrough[key] = value
            else:
                decoded[key] = value

        decoded.update(passthrough)

        if save_snapshot:
            self._persist_input_snapshot(node_state, decoded)

        return decoded

    def _persist_input_snapshot(
        self, node_state: NodeState, payload: Dict[str, Any]
    ):
        snapshot_path = (
            self.cache_root
            / node_state.session_id
            / f"{node_state.artifact_id}.json"
        )
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)

        with snapshot_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    # ============================================================
    # Output Pipeline
    # ============================================================

    def _pack_item(
        self, node_state: NodeState, item: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Convert server-side file → base64 when needed.
        """
        result = dict(item)

        orig_path = result.pop("orig_path", None)
        orig_md5 = result.pop("orig_md5", None)
        server_path = result.pop("path", None)

        if server_path:
            compress_data = FileCompressor.compress_and_encode(server_path)

            if orig_md5 and compress_data.md5 == orig_md5:
                result["path"] = orig_path
            else:
                result.update(
                    {
                        "base64": compress_data.base64,
                        "path": compress_data.filename,
                        "md5": compress_data.md5,
                    }
                )

        return result

    def pack_outputs_to_client(
        self,
        node_state: NodeState,
        outputs: Union[Dict[str, Any], List[str]],
    ) -> Union[Dict[str, Any], List[str]]:
        if not isinstance(outputs, dict):
            return outputs

        packed: Dict[str, Any] = {}

        for key, value in outputs.items():
            if isinstance(value, list) and all(
                isinstance(v, dict) for v in value
            ):
                packed[key] = [
                    self._pack_item(node_state, v) for v in value
                ]
            elif isinstance(value, dict):
                packed[key] = self.pack_outputs_to_client(node_state, value)
            else:
                packed[key] = value

        return packed

    # ============================================================
    # Schema Validation
    # ============================================================

    def _validate_schema(
        self,
        params: Dict[str, Any],
        schema_name: str,
    ):
        schema = getattr(self, schema_name, None)
        if not schema:
            return

        try:
            schema(**params)
        except ValidationError as e:
            raise ValueError(
                f"[{self.meta.node_id}] {schema_name} validation failed: {e}"
            ) from e

    # ============================================================
    # Lifecycle Hooks
    # ============================================================

    def _pre_process(
        self, node_state: NodeState, inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        return inputs

    def _post_process(
        self, node_state: NodeState, outputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        return outputs

    # ============================================================
    # Core Execution
    # ============================================================

    @abstractmethod
    async def process(
        self, node_state: NodeState, inputs: Dict[str, Any]
    ) -> Any:
        ...

    @abstractmethod
    async def default_process(
        self, node_state: NodeState, inputs: Dict[str, Any]
    ) -> Any:
        ...

    async def __call__(
        self, node_state: NodeState, **params
    ) -> Dict[str, Any]:
        try:
            mode = params.get("mode", "auto")

            inputs = self.load_inputs_from_client(node_state, params)
            inputs = self._pre_process(node_state, inputs)

            self._validate_schema(inputs, "input_schema")

            if mode != "auto":
                outputs = await self.default_process(node_state, inputs)
            else:
                outputs = await self.process(node_state, inputs)

            outputs = self._post_process(node_state, outputs)
            self._validate_schema(outputs, "output_schema")

            packed_output = self.pack_outputs_to_client(
                node_state, outputs
            )

            return {
                "artifact_id": node_state.artifact_id,
                "summary": node_state.node_summary.get_summary(
                    node_state.artifact_id
                ),
                "tool_excute_result": packed_output,
                "isError": False,
            }

        except Exception as e:
            return self._handle_exception(node_state, e)

    # ============================================================
    # Error Handling
    # ============================================================

    def _handle_exception(
        self, node_state: NodeState, exc: Exception
    ) -> Dict[str, Any]:
        if self.server_cfg.developer.developer_mode:
            trace = "".join(traceback.format_exception(exc))
            logger.error(trace)
            summary = {
                "error_info": f"[artifact_id={node_state.artifact_id}]\n{trace}"
            }
        else:
            summary = node_state.node_summary.get_summary(
                node_state.artifact_id
            )

        return {
            "artifact_id": node_state.artifact_id,
            "summary": summary,
            "tool_excute_result": {},
            "isError": True,
        }

  
