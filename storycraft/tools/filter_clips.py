from typing import Any, Dict, List

from storycraft.tools.base import BaseNode, NodeMeta
from storycraft.core.node_state import NodeState
from storycraft.core.node_schema import FilterClipsInput
from storycraft.mcp.sampling_requester import LLMClient
from storycraft.utils.prompts import get_prompt
from storycraft.utils.parse import parse_json_dict
from storycraft.utils.register import NODE_REGISTRY


# ============================================================
# Node Definition
# ============================================================

@NODE_REGISTRY.register()
class FilterClipsNode(BaseNode):
    """
    Agent Selection Node:
    Filter candidate video clips based on user semantic requirements.
    """

    meta = NodeMeta(
        name="filter_clips",
        description="Filter clips based on their descriptions according to user requirements.",
        node_id="filter_clips",
        node_kind="filter_clips",
        require_prior_kind=["split_shots", "understand_clips"],
        default_require_prior_kind=["split_shots", "understand_clips"],
        next_available_node=["group_clips", "group_clips_pro"],
    )

    input_schema = FilterClipsInput

    # ============================================================
    # Input Parsing Layer
    # ============================================================

    def _parse_input(
        self, node_state: NodeState, inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        clip_captions = inputs["understand_clips"].get("clip_captions", [])
        clip_info = inputs["split_shots"].get("clips", [])

        duration_lookup = build_duration_lookup(clip_info)
        clip_captions = inject_clip_durations(clip_captions, duration_lookup)

        inputs["clip_captions"] = clip_captions
        inputs["input_clip_ids"] = [
            c.get("clip_id") for c in clip_captions if c.get("clip_id")
        ]

        return inputs

    # ============================================================
    # Default Process (No Filtering)
    # ============================================================

    async def default_process(
        self, node_state: NodeState, inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        node_state.node_summary.info_for_user("Using all clips")

        return {
            "clip_captions": inputs["clip_captions"],
            "selected": inputs["input_clip_ids"],
        }

    # ============================================================
    # Main Process (LLM Semantic Filtering)
    # ============================================================

    async def process(
        self, node_state: NodeState, inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        clip_captions = inputs["clip_captions"]
        input_clip_ids = inputs["input_clip_ids"]
        user_request = inputs.get("user_request") or ""

        if not user_request.strip():
            node_state.node_summary.info_for_user(
                "No explicit user request, using all clips"
            )
            return {
                "clip_captions": clip_captions,
                "selected": input_clip_ids,
            }

        llm: LLMClient = node_state.llm

        system_prompt, user_prompt = self._build_prompts(
            node_state, user_request, clip_captions
        )

        raw = await llm.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,
            top_p=0.9,
            max_tokens=2048,
        )

        selected_ids = self._safe_parse_selection(
            raw, input_clip_ids, node_state
        )

        node_state.node_summary.info_for_user(
            f"Filtered {len(selected_ids)} / {len(input_clip_ids)} clips"
        )

        return {
            "clip_captions": clip_captions,
            "selected": selected_ids,
        }

    # ============================================================
    # Prompt Builder
    # ============================================================

    def _build_prompts(
        self,
        node_state: NodeState,
        user_request: str,
        clip_captions: List[Dict[str, Any]],
    ) -> tuple[str, str]:
        clip_block = build_clips_block(clip_captions)

        system_prompt = get_prompt(
            "filter_clips.system", lang=node_state.lang
        )

        user_prompt = get_prompt(
            "filter_clips.user",
            lang=node_state.lang,
            user_request=user_request,
            clip_captions=clip_block,
        )

        return system_prompt, user_prompt

    # ============================================================
    # LLM Output Parsing Layer
    # ============================================================

    def _safe_parse_selection(
        self,
        raw: str,
        input_clip_ids: List[str],
        node_state: NodeState,
    ) -> List[str]:
        try:
            obj = parse_json_dict(raw)
            return extract_selected_ids(obj, input_clip_ids)
        except Exception:
            node_state.node_summary.info_for_user(
                "Model output parse failed, fallback to all clips"
            )
            return input_clip_ids


# ============================================================
# Helper Functions (Pure Functional Layer)
# ============================================================

def inject_clip_durations(
    clip_captions: List[Dict[str, Any]],
    durations: Dict[str, float],
) -> List[Dict[str, Any]]:
    for clip in clip_captions:
        cid = clip.get("clip_id")
        if cid and cid in durations:
            clip["duration"] = durations[cid]
    return clip_captions


def build_duration_lookup(
    clip_info: List[Dict[str, Any]],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for item in clip_info or []:
        cid = item.get("clip_id")
        if not cid:
            continue
        src = item.get("source_ref") or {}
        dur = src.get("duration", 0) / 1000.0
        out[cid] = dur if dur > 0 else 2.0
    return out


def extract_selected_ids(
    obj: Dict[str, Any],
    input_clip_ids: List[str],
) -> List[str]:
    id_set = set(input_clip_ids)
    results = obj.get("results")

    if not isinstance(results, list):
        raise ValueError('"results" must be list')

    keep_ids: set[str] = set()
    explicit_true = 0

    for item in results:
        if not isinstance(item, dict):
            continue
        cid = item.get("clip_id")
        keep = item.get("keep")

        keep_bool = parse_bool(keep)
        if keep_bool is True:
            explicit_true += 1
            if isinstance(cid, str) and cid in id_set:
                keep_ids.add(cid)

    if explicit_true > 0 and not keep_ids:
        raise ValueError("Model selected items but none valid")

    return [cid for cid in input_clip_ids if cid in keep_ids]


def parse_bool(val: Any) -> bool | None:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        s = val.strip().lower()
        if s in ("true", "yes", "1"):
            return True
        if s in ("false", "no", "0"):
            return False
    return None


def build_clips_block(
    clip_captions: List[Dict[str, Any]],
) -> str:
    blocks: List[str] = []
    for clip in clip_captions:
        blocks.append(
            f"[clip_id={clip.get('clip_id','')}]\n"
            f"caption: {clip.get('caption','')}\n"
        )
    return "\n".join(blocks).strip() + "\n"

