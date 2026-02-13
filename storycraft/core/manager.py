from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

from langchain_core.tools.structured import StructuredTool

from storycraft.memory.artifact_store import ArtifactStore


# ============================================================
# 🎭 Narrative Node Orchestrator
# ============================================================

class NarrativeNodeOrchestrator:
    """
    StoryCraft Narrative Node Orchestration Engine.

    This engine manages:
      - Node registration & lifecycle
      - Dependency graph construction
      - Execution eligibility checking
      - Multi-step cinematic workflow orchestration

    Conceptually, each node represents a cinematic production unit:
        Shot Analysis / Scene Planning / Clip Selection / Audio Design / Editing / Rendering
    """

    def __init__(self, tools: Optional[List[StructuredTool]] = None):
        # Node registries
        self.kind_to_node_ids: Dict[str, List[str]] = defaultdict(list)
        self.id_to_tool: Dict[str, StructuredTool] = {}

        # Execution graph
        self.id_to_next: Dict[str, List[str]] = {}
        self.id_to_priority: Dict[str, int] = {}
        self.id_to_kind: Dict[str, str] = {}

        # Dependency control
        self.id_to_require_prior_kind: Dict[str, List[str]] = {}
        self.id_to_default_require_prior_kind: Dict[str, List[str]] = {}

        # Reverse dependency index
        self.kind_to_dependent_nodes: Dict[str, Set[str]] = defaultdict(set)
        self.kind_to_default_dependent_nodes: Dict[str, Set[str]] = defaultdict(set)

        if tools:
            self._register_tools(tools)

    # ============================================================
    # 🔧 Node Registration & Lifecycle
    # ============================================================

    def _register_tools(self, tools: List[StructuredTool]):
        for tool in tools:
            meta = tool.metadata.get("_meta") if tool.metadata else None
            if meta and meta.get("node_id"):
                self.register_node(tool)

    def register_node(self, tool: StructuredTool) -> bool:
        """
        Register a cinematic node into orchestration graph.

        Each node is defined via StructuredTool + metadata:
            - node_id
            - node_kind
            - priority
            - execution dependencies
        """

        if not tool.metadata:
            return False

        meta = tool.metadata.get("_meta", {})
        node_id = meta.get("node_id")
        if not node_id:
            return False

        if node_id in self.id_to_tool:
            self.remove_node(node_id)

        node_kind = meta.get("node_kind", node_id)
        priority = meta.get("priority", 0)
        next_nodes = meta.get("next_available_node", [])
        require_prior_kind = meta.get("require_prior_kind", [])
        default_require_prior_kind = meta.get("default_require_prior_kind", [])

        # Core registries
        self.id_to_tool[node_id] = tool
        self.id_to_priority[node_id] = priority
        self.id_to_next[node_id] = next_nodes
        self.id_to_kind[node_id] = node_kind
        self.id_to_require_prior_kind[node_id] = require_prior_kind
        self.id_to_default_require_prior_kind[node_id] = default_require_prior_kind

        # Group by narrative kind
        self.kind_to_node_ids[node_kind].append(node_id)
        self._sort_nodes_by_priority(node_kind)

        # Reverse dependency graph
        for kind in require_prior_kind:
            self.kind_to_dependent_nodes[kind].add(node_id)

        for kind in default_require_prior_kind:
            self.kind_to_default_dependent_nodes[kind].add(node_id)

        return True

    def remove_node(self, node_id: str, clean_references: bool = True) -> bool:
        """
        Remove node from orchestration graph.

        Used primarily for dynamic pipeline reconfiguration.
        """

        if node_id not in self.id_to_tool:
            return False

        node_kind = self.id_to_kind[node_id]

        for kind in self.id_to_require_prior_kind.get(node_id, []):
            self.kind_to_dependent_nodes[kind].discard(node_id)

        for kind in self.id_to_default_require_prior_kind.get(node_id, []):
            self.kind_to_default_dependent_nodes[kind].discard(node_id)

        self.id_to_tool.pop(node_id, None)
        self.id_to_priority.pop(node_id, None)
        self.id_to_next.pop(node_id, None)
        self.id_to_kind.pop(node_id, None)
        self.id_to_require_prior_kind.pop(node_id, None)
        self.id_to_default_require_prior_kind.pop(node_id, None)

        if node_id in self.kind_to_node_ids.get(node_kind, []):
            self.kind_to_node_ids[node_kind].remove(node_id)
            if not self.kind_to_node_ids[node_kind]:
                del self.kind_to_node_ids[node_kind]

        if clean_references:
            for nid, next_nodes in self.id_to_next.items():
                if node_id in next_nodes:
                    next_nodes.remove(node_id)

        return True

    # ============================================================
    # 🧠 Execution Graph Logic
    # ============================================================

    def _sort_nodes_by_priority(self, kind: str):
        if kind in self.kind_to_node_ids:
            self.kind_to_node_ids[kind].sort(
                key=lambda nid: self.id_to_priority.get(nid, 0),
                reverse=True
            )

    def get_tool(self, node_id: str) -> Optional[StructuredTool]:
        return self.id_to_tool.get(node_id)

    # ============================================================
    # 🎬 Cinematic Dependency Resolution
    # ============================================================

    def check_execution_ready(
        self,
        session_id: str,
        store: ArtifactStore,
        required_kinds: List[str]
    ) -> Dict[str, Any]:
        """
        Check if required cinematic features are already available.

        Example:
            To execute:
                Timeline Planning
            Requires:
                Shot Segmentation + Clip Understanding + Script Planning

        Returns:
            {
                executable: bool,
                collected_nodes: Dict[kind, artifact],
                missing_kinds: List[str]
            }
        """

        collected = {}

        for kind in required_kinds:
            node_queue = self.kind_to_node_ids.get(kind, [])
            valid_outputs = []

            for node_id in node_queue:
                output = store.get_latest_meta(
                    node_id=node_id,
                    session_id=session_id
                )
                if output is not None:
                    valid_outputs.append(output)

            if valid_outputs:
                latest = max(valid_outputs, key=lambda o: o.created_at)
                collected[kind] = latest

        return {
            "executable": len(collected) == len(required_kinds),
            "collected_nodes": collected,
            "missing_kinds": list(set(required_kinds) - set(collected.keys())),
        }
  
