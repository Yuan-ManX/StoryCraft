from typing import Any, Dict, List, Set

from storycraft.tools.base import BaseNode, NodeMeta
from storycraft.core.state import NodeState
from storycraft.core.schema import GroupClipsInput
from storycraft.utils.prompts import get_prompt
from storycraft.utils.parse_json import parse_json_dict
from storycraft.utils.register import NODE_REGISTRY


@NODE_REGISTRY.register()
class GroupClipsNode(BaseNode):
    """
    Group selected clips into semantic segments for script and voiceover generation.
    """

    meta = NodeMeta(
        name="group_clips",
        description="Group clips based on their descriptions according to user requirements.",
        node_id="group_clips",
        node_kind="group_clips",
        require_prior_kind=["filter_clips"],
        default_require_prior_kind=["filter_clips"],
        next_available_node=["generate_script", "generate_script_pro"],
    )

    input_schema = GroupClipsInput

    async def default_process(
        self,
        node_state: NodeState,
        inputs: Dict[str, Any],
    ) -> Any:
        selected = (inputs.get("filter_clips") or {}).get("selected") or []
        groups = _make_single_group_fallback(selected)
        return {"groups": groups}

    async def process(self, node_state: NodeState, inputs: Dict[str, Any], **params) -> Any:

        clip_captions, selected_clips, user_request = self._prepare_inputs(inputs)

        if not selected_clips:
            return {"groups": []}

        try:
            clip_lookup = _build_clip_lookup(clip_captions)
            selected_blocks = [clip_lookup[cid] for cid in selected_clips]

            clip_block = _build_clips_block(selected_blocks)
            raw = await self._call_llm(node_state, clip_block, selected_clips, user_request)

            obj = parse_json_dict(raw)
            groups_raw = _extract_groups_obj(obj)

            groups = _normalize_groups_from_llm(
                groups_raw=groups_raw,
                selected_ids_set=set(selected_clips),
            )

            node_state.node_summary.info_for_user(
                f"Grouping completed: {len(groups)} groups generated"
            )
            return {"groups": groups}

        except Exception as e:
            node_state.node_summary.info_for_user(
                f"Grouping failed: {type(e).__name__}: {e}. Using fallback strategy."
            )
            return {"groups": _make_single_group_fallback(selected_clips)}

    # ---------------------------------------------------------------------
    # Pipeline helpers
    # ---------------------------------------------------------------------

    def _prepare_inputs(self, inputs: Dict[str, Any]):
        fc = inputs.get("filter_clips") or {}
        clip_captions = fc.get("clip_captions") or []
        selected_clips = fc.get("selected") or []
        user_request = inputs.get("user_request") or ""

        return clip_captions, selected_clips, user_request

    async def _call_llm(
        self,
        node_state: NodeState,
        clip_block: List[Dict[str, Any]],
        selected_clips: List[str],
        user_request: str,
    ) -> str:

        system_prompt = get_prompt("group_clips.system", lang=node_state.lang)

        user_prompt = get_prompt(
            "group_clips.user",
            lang=node_state.lang,
            user_request=user_request or "No additional requirements",
            selected_clips=selected_clips,
            clip_captions=clip_block,
            clip_number=len(clip_block),
        )

        return await node_state.llm.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,
            top_p=0.9,
            max_tokens=4096,
            model_preferences=None,
        )


# ---------------------------------------------------------------------
# Parsing & normalization helpers
# ---------------------------------------------------------------------

def _extract_groups_obj(obj: Any) -> list[dict[str, Any]]:
    if isinstance(obj, dict) and isinstance(obj.get("groups"), list):
        return obj["groups"]
    if isinstance(obj, list):
        return obj
    raise ValueError("LLM output missing field `groups`.")


def _normalize_groups_from_llm(
    groups_raw: list[dict[str, Any]],
    selected_ids_set: Set[str],
) -> list[dict[str, Any]]:
    """
    Normalize and validate LLM-generated groups:
    - clip_ids must come from selected set
    - no duplication
    - rewrite group_id
    - guarantee summary existence
    """
    if not groups_raw:
        raise ValueError("groups list is empty")

    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []

    for gi, g in enumerate(groups_raw):
        if not isinstance(g, dict):
            raise ValueError(f"groups[{gi}] is not a dict")

        clip_ids = g.get("clip_ids")
        if not isinstance(clip_ids, list) or not clip_ids:
            raise ValueError(f"groups[{gi}].clip_ids must be a non-empty list")

        cleaned: list[str] = []
        for cid in clip_ids:
            if cid not in selected_ids_set or cid in seen:
                continue
            cleaned.append(cid)
            seen.add(cid)

        if not cleaned:
            raise ValueError(f"groups[{gi}] empty after cleaning")

        summary = g.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            summary = "A group of shots for generating script and voiceover."

        normalized.append(
            {
                "group_id": "",
                "summary": summary.strip(),
                "clip_ids": cleaned,
                "duration": g.get("duration"),
            }
        )

    for i, g in enumerate(normalized, start=1):
        g["group_id"] = f"group_{i:04d}"

    return normalized


# ---------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------

def _build_clip_lookup(clip_captions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {c["clip_id"]: c for c in clip_captions if c.get("clip_id")}


def _make_single_group_fallback(selected_clips: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "group_id": "group_0001",
            "summary": "Aggregate all selected shots in original order.",
            "clip_ids": selected_clips,
        }
    ]


def _build_clips_block(clip_captions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Build structured clip info block for LLM prompt.
    """
    blocks = []
    for clip in clip_captions:
        blocks.append(
            {
                "clip_id": clip.get("clip_id", ""),
                "duration": clip.get("duration", 0.0),
                "caption": clip.get("caption", ""),
            }
        )
    return blocks

