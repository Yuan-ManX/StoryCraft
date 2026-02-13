from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List, Literal
from datetime import datetime
import uuid

from storycraft.mcp.sampling_requester import SamplingLLMClient
from storycraft.core.summary import NodeSummary

from mcp.server.fastmcp import Context
from mcp.server.session import ServerSession


# ============================================================
# 🎛 Execution Trace & Runtime Telemetry
# ============================================================

@dataclass
class ExecutionTrace:
    """
    Runtime execution trace for debugging, auditing, and observability.
    """
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    node_name: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None

    status: Literal["pending", "running", "success", "failed"] = "pending"
    error: Optional[str] = None

    input_snapshot: Optional[Dict[str, Any]] = None
    output_snapshot: Optional[Dict[str, Any]] = None

    logs: List[str] = field(default_factory=list)

    def mark_running(self):
        self.status = "running"

    def mark_success(self, output: Optional[Dict[str, Any]] = None):
        self.status = "success"
        self.end_time = datetime.utcnow()
        self.output_snapshot = output

    def mark_failed(self, error: str):
        self.status = "failed"
        self.end_time = datetime.utcnow()
        self.error = error


# ============================================================
# 🎛 Runtime Context Kernel
# ============================================================

@dataclass
class NodeRuntimeState:
    """
    StoryCraft Agent Runtime Context Kernel.

    This object carries:
    - execution identity
    - node execution state
    - agent shared memory
    - telemetry & tracing
    - llm + mcp runtime
    """

    # ------------------ Identity ------------------

    session_id: str
    artifact_id: str
    node_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    lang: str = "en"

    # ------------------ Runtime Kernel ------------------

    node_summary: NodeSummary = field(repr=False)
    llm: SamplingLLMClient = field(repr=False)
    mcp_ctx: Context[ServerSession, object] = field(repr=False)

    # ------------------ Runtime Memory ------------------

    shared_memory: Dict[str, Any] = field(default_factory=dict)
    local_memory: Dict[str, Any] = field(default_factory=dict)

    # ------------------ Execution Trace ------------------

    trace: ExecutionTrace = field(default_factory=ExecutionTrace)

    # ------------------ Runtime Flags ------------------

    debug: bool = False
    dry_run: bool = False

    # ============================================================
    # 🧠 Context Utilities
    # ============================================================

    def fork(self, *, node_id: Optional[str] = None) -> "NodeRuntimeState":
        """
        Create a child runtime context for sub-nodes or branches.
        """
        return NodeRuntimeState(
            session_id=self.session_id,
            artifact_id=self.artifact_id,
            node_id=node_id or uuid.uuid4().hex,
            lang=self.lang,
            node_summary=self.node_summary,
            llm=self.llm,
            mcp_ctx=self.mcp_ctx,
            shared_memory=self.shared_memory,
            debug=self.debug,
            dry_run=self.dry_run,
        )

    def log(self, message: str):
        if self.debug:
            print(f"[{self.trace.trace_id}] {message}")
        self.trace.logs.append(message)

    def set_shared(self, key: str, value: Any):
        self.shared_memory[key] = value

    def get_shared(self, key: str, default=None):
        return self.shared_memory.get(key, default)

    def set_local(self, key: str, value: Any):
        self.local_memory[key] = value

    def get_local(self, key: str, default=None):
        return self.local_memory.get(key, default)

    # ============================================================
    # 🎯 Lifecycle Hooks
    # ============================================================

    def start(self, input_snapshot: Optional[Dict[str, Any]] = None):
        self.trace.node_name = self.node_summary.node_name
        self.trace.input_snapshot = input_snapshot
        self.trace.mark_running()

    def finish(self, output_snapshot: Optional[Dict[str, Any]] = None):
        self.trace.mark_success(output_snapshot)

    def fail(self, error: Exception):
        self.trace.mark_failed(str(error))
        raise error
  
