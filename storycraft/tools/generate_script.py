from typing import Any, Dict, List, Tuple
import re

from storycraft.tools.base import BaseNode, NodeMeta
from storycraft.core.state import NodeState
from storycraft.core.schema import GenerateScriptInput
from storycraft.utils.prompts import get_prompt
from storycraft.utils.parse_json import parse_json_dict
from storycraft.utils.register import NODE_REGISTRY


# ============================================================
# Node Definition
# ============================================================

@NODE_REGISTRY.register()
class GenerateScriptNode(BaseNode):
    """
    Script Generation Node

    Generate structured narration scripts based on grouped video clips,
    considering semantic descriptions, duration budgets, and user intent.
    """

    meta = NodeMeta(
        name="generate_script",
        description=(
            "Generate narration scripts or subtitles for grouped video clips. "
            "Supports lyrical, humorous, casual, and custom styles."
        ),
        node_id="generate_script",
        node_kind="generate_script",
        require_prior_kind=["split_shots", "group_clips", "understand_clips"],
        default_require_prior_kind=["split_shots", "group_clips"],
        next_available_node=["generate_voiceover"],
    )

    input_schema = GenerateScriptInput

    # ============================================================
    # Default Process
    # ============================================================

    async def default_process(
        self, node_state: NodeState, inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {"group_scripts": [], "title": ""}

    # ============================================================
    # Main Process
    # ============================================================

    async def process(
        self, node_state: NodeState, inputs: Dict[str, Any]
    ) -> Dict[str, Any]:

        parsed = self._parse_inputs(inputs, node_state)

        if parsed["custom_script"]:
            return self._process_custom_script(parsed, node_state)

        return await self._process_llm_script(parsed, node_state)

    # ============================================================
    # Input Parsing Layer
    # ============================================================

    def _parse_inputs(
        self, inputs: Dict[str, Any], node_state: NodeState
    ) -> Dict[str, Any]:
        clip_info = inputs["split_shots"]["clips"]
        clip_captions = inputs["understand_clips"]["clip_captions"]
        overall = inputs["understand_clips"]["overall"]
        groups = inputs["group_clips"]["groups"]

        duration_lookup = build_duration_lookup(clip_info)
        caption_lookup = build_caption_lookup(clip_captions)

        group_ids = [
            g.get("group_id") for g in groups or [] if g.get("group_id")
        ]

        return {
            "clip_info": clip_info,
            "clip_captions": clip_captions,
            "groups": groups,
            "group_ids": group_ids,
            "group_ids_set": set(group_ids),
            "duration_lookup": duration_lookup,
            "caption_lookup": caption_lookup,
            "overall": overall,
            "user_request": inputs.get("user_request") or "",
            "custom_script": inputs.get("custom_script") or {},
        }

    # ============================================================
    # Custom Script Branch
    # ============================================================

    def _process_custom_script(
        self, parsed: Dict[str, Any], node_state: NodeState
    ) -> Dict[str, Any]:
        custom_script = parsed["custom_script"]

        validate_subtitle_format(custom_script)

        group_scripts: list[dict[str, Any]] = []
        subtitle_index = 1

        for g in custom_script["group_scripts"]:
            units, subtitle_index = make_subtitle_units(
                raw_text=g["raw_text"],
                subtitle_start_index=subtitle_index,
            )
            group_scripts.append(
                {
                    "group_id": g["group_id"],
                    "raw_text": g["raw_text"],
                    "subtitle_units": units,
                }
            )

        return {
            "group_scripts": group_scripts,
            "title": custom_script.get("title", ""),
        }

    # ============================================================
    # LLM Script Branch
    # ============================================================

    async def _process_llm_script(
        self, parsed: Dict[str, Any], node_state: NodeState
    ) -> Dict[str, Any]:

        groups = parsed["groups"]
        group_ids = parsed["group_ids"]

        if not group_ids:
            node_state.node_summary.info_for_user(
                "No valid group found, skip script generation"
            )
            return {"group_scripts": [], "title": ""}

        llm = node_state.llm

        groups_block = build_groups_block_for_script(
            parsed["groups"],
            parsed["duration_lookup"],
            parsed["caption_lookup"],
        )

        system_prompt, user_prompt = self._build_prompts(
            node_state, parsed["user_request"], parsed["overall"], groups_block
        )

        raw = await llm.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,
            top_p=0.9,
            max_tokens=4096,
        )

        group_text_map = self._safe_parse_llm_output(
            raw, group_ids, node_state
        )

        return self._build_group_scripts(
            parsed, group_text_map, node_state
        )

    # ============================================================
    # Prompt Builder
    # ============================================================

    def _build_prompts(
        self,
        node_state: NodeState,
        user_request: str,
        overall: str,
        groups_block: str,
    ) -> tuple[str, str]:

        system_prompt = get_prompt(
            "generate_script.system", lang=node_state.lang
        )

        user_prompt = get_prompt(
            "generate_script.user",
            lang=node_state.lang,
            user_request=user_request or "No requirements",
            overall=overall,
            groups=groups_block,
        )

        return system_prompt, user_prompt

    # ============================================================
    # LLM Output Parsing
    # ============================================================

    def _safe_parse_llm_output(
        self,
        raw: str,
        group_ids: List[str],
        node_state: NodeState,
    ) -> Dict[str, str]:
        try:
            obj = parse_json_dict(raw)
            return extract_group_text_map(obj, group_ids)
        except Exception as e:
            node_state.node_summary.info_for_llm(
                f"script generation parse failed: {type(e).__name__}: {e}"
            )
            return {}

    # ============================================================
    # Output Assembly Layer
    # ============================================================

    def _build_group_scripts(
        self,
        parsed: Dict[str, Any],
        group_text_map: Dict[str, str],
        node_state: NodeState,
    ) -> Dict[str, Any]:

        group_scripts = []
        subtitle_index = 1

        for g in parsed["groups"] or []:
            gid = g.get("group_id")
            if not gid or gid not in parsed["group_ids_set"]:
                continue

            duration = sum(
                parsed["duration_lookup"].get(cid, 0.0)
                for cid in (g.get("clip_ids") or [])
            )
            budget = estimate_script_budget(duration)

            raw_text = (group_text_map.get(gid) or "").strip()
            if not raw_text:
                raise ValueError("LLM returned empty script")

            raw_text = self._apply_budget_trim(
                raw_text, budget, node_state
            )

            units, subtitle_index = make_subtitle_units(
                raw_text, subtitle_index
            )

            group_scripts.append(
                {
                    "group_id": gid,
                    "raw_text": raw_text,
                    "subtitle_units": units,
                }
            )

        return {
            "group_scripts": group_scripts,
            "title": group_text_map.get("title", ""),
        }

    # ============================================================
    # Budget Control
    # ============================================================

    def _apply_budget_trim(
        self,
        raw_text: str,
        budget: Dict[str, Any],
        node_state: NodeState,
    ) -> str:
        max_chars = budget["max_chars"]
        if len(raw_text) > int(max_chars * 2.0):
            node_state.node_summary.info_for_user(
                "Script too long, truncated automatically"
            )
            return raw_text[:max_chars].rstrip()
        return raw_text


# ============================================================
# Helper Functions (Pure Functional Layer)
# ============================================================

_SPLIT_RE = re.compile(r"[，,。！!?？]+")


def split_by_comma(raw_text: str) -> list[str]:
    s = raw_text.strip().replace("\n", "，")
    return [p.strip() for p in _SPLIT_RE.split(s) if p.strip()]


def make_subtitle_units(
    raw_text: str,
    subtitle_start_index: int,
) -> tuple[list[dict[str, Any]], int]:
    parts = split_by_comma(raw_text) or [raw_text.strip()]

    units = []
    cur = subtitle_start_index

    for idx, text in enumerate(parts):
        units.append(
            {
                "unit_id": f"subtitle_{cur:04d}",
                "index_in_group": idx,
                "text": text,
            }
        )
        cur += 1

    return units, cur

