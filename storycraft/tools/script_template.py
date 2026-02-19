from typing import Any, Dict
from pathlib import Path

from storycraft.tools.base import BaseNode, NodeMeta
from storycraft.core.state import NodeState
from storycraft.utils.recall import StorylineRecall
from storycraft.utils.element_filter import ElementFilter
from storycraft.core.schema import RecommendScriptTemplateInput
from storycraft.utils.register import NODE_REGISTRY


@NODE_REGISTRY.register()
class ScriptTemplateRecommendationNode(BaseNode):
    """
    Script Template Recommendation Node

    This node retrieves, filters, and ranks candidate script templates
    based on user requirements, serving as the style selector for script generation.
    """

    meta = NodeMeta(
        name="script_template",
        description="Recommend suitable script templates (writing styles) for script generation",
        node_id="script_template",
        node_kind="script_template",
        require_prior_kind=[],
        default_require_prior_kind=[],
        next_available_node=["generate_script"],
    )

    input_schema = RecommendScriptTemplateInput

    def __init__(self, server_cfg):
        super().__init__(server_cfg)

        self.element_filter = ElementFilter(
            json_path=self.server_cfg.script_template.script_template_info_path
        )
        self.vectorstore = StorylineRecall.build_vectorstore(
            self.element_filter.library
        )

        self._top_n = int(
            getattr(self.server_cfg.script_template, "top_n", 3)
        )

    async def default_process(
        self,
        node_state: NodeState,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        node_state.node_summary.info_for_user(
            "No script template recommendation, using default style"
        )
        return {"candidates": []}

    async def process(
        self,
        node_state: NodeState,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:

        user_request = inputs.get("user_request", "")
        filter_include = inputs.get("filter_include", {})
        filter_exclude = inputs.get("filter_exclude", {})

        try:
            candidates = await self._recommend_templates(
                node_state=node_state,
                user_request=user_request,
                filter_include=filter_include,
                filter_exclude=filter_exclude,
            )
        except Exception as e:
            node_state.node_summary.add_error(repr(e))
            return {"candidates": []}

        if not candidates:
            node_state.node_summary.info_for_user(
                "No suitable script template found, fallback to default"
            )
            return {"candidates": []}

        node_state.node_summary.info_for_user(
            f"Recommended {len(candidates)} script template(s)"
        )

        return {"candidates": candidates}

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    async def _recommend_templates(
        self,
        *,
        node_state: NodeState,
        user_request: str,
        filter_include: Dict[str, Any],
        filter_exclude: Dict[str, Any],
    ) -> list[dict[str, Any]]:

        script_dir = self._validate_template_dir()

        # Step1: semantic recall
        candidates = StorylineRecall.query_top_n(
            self.vectorstore, query=user_request
        )

        # Step2: tag filtering
        candidates = self.element_filter.filter(
            candidates, filter_include, filter_exclude
        )

        if not candidates:
            raise ValueError("No candidate templates after recall & filtering")

        return candidates[: self._top_n]

    def _validate_template_dir(self) -> Path:
        path: Path = (
            self.server_cfg.script_template.script_template_dir
            .expanduser()
            .resolve()
        )

        if not path.exists():
            raise FileNotFoundError(
                f"script_template_dir not found: {path}"
            )
        if not path.is_dir():
            raise NotADirectoryError(
                f"script_template_dir is not a directory: {path}"
            )

        return path

  
