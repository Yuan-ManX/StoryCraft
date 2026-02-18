from typing import Any, Dict, List
import asyncio

from storycraft.tools.base import BaseNode, NodeMeta
from storycraft.core.state import NodeState
from storycraft.core.schema import UnderstandClipsInput
from storycraft.utils.prompts import get_prompt
from storycraft.utils.parse_json import parse_json_dict
from storycraft.utils.register import NODE_REGISTRY


@NODE_REGISTRY.register()
class UnderstandClipsNode(BaseNode):
    """
    Analyze each clip and generate natural language description.
    """

    meta = NodeMeta(
        name="understand_clips",
        description="Analyze clips and generate descriptions. Requires load_media and split_shots output.",
        node_id="understand_clips",
        node_kind="understand_clips",
        require_prior_kind=["load_media", "split_shots"],
        default_require_prior_kind=["load_media", "split_shots"],
        next_available_node=["filter_clips", "filter_clips_pro"],
    )

    input_schema = UnderstandClipsInput

    async def default_process(self, node_state: NodeState, inputs: Dict[str, Any]) -> Any:
        clips = inputs["split_shots"]["clips"]

        clip_captions: list[dict[str, Any]] = [
            {
                "clip_id": clip.get("clip_id"),
                "caption": "no caption",
                "source_ref": {
                    "media_id": clip.get("source_ref", {}).get("media_id", "")
                },
            }
            for clip in clips or []
        ]

        node_state.node_summary.info_for_user(
            f"Skipped description generation for {len(clip_captions)} clips"
        )

        return {
            "clip_captions": clip_captions,
            "overall": "unknown",
        }

    async def process(self, node_state: NodeState, inputs: Dict[str, Any]) -> Any:
        clips = inputs["split_shots"]["clips"]
        media_map = inputs["media"]
        llm = node_state.llm

        system_prompt = get_prompt(
            "understand_clips.system_detail", lang=node_state.lang
        )
        user_prompt = get_prompt(
            "understand_clips.user_detail", lang=node_state.lang
        )

        clip_captions: list[dict[str, Any]] = []

        for clip in clips or []:
            item = await self._process_single_clip(
                clip=clip,
                media_map=media_map,
                llm=llm,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                node_state=node_state,
            )
            clip_captions.append(item)

        overall_summary = await self._build_overall_summary(
            clip_captions, llm, node_state
        )

        node_state.node_summary.info_for_user(
            f"Clip understanding completed: {len(clip_captions)} clips"
        )

        return {
            "clip_captions": clip_captions,
            "overall": overall_summary,
        }

    def _parse_input(self, node_state: NodeState, inputs: Dict[str, Any]):
        media_list = inputs["load_media"]["media"]

        media_map: dict[str, dict[str, Any]] = {}
        for item in media_list or []:
            media_id = item.get("media_id")
            if media_id:
                media_map[str(media_id)] = item

        inputs["media"] = media_map
        return inputs

    async def _process_single_clip(
        self,
        *,
        clip: Dict[str, Any],
        media_map: Dict[str, Any],
        llm,
        system_prompt: str,
        user_prompt: str,
        node_state: NodeState,
    ) -> dict[str, Any]:

        clip_id = str(clip.get("clip_id") or "").strip() or "(unknown)"
        kind = str(clip.get("kind") or "").lower().strip()
        src = clip.get("source_ref") or {}

        out: dict[str, Any] = {"clip_id": clip_id}

        media_id = str(src.get("media_id") or "")
        media_item = media_map.get(media_id)

        if not media_item:
            out["caption"] = f"Error: Media not found for media_id={media_id}"
            return out

        path = str(media_item.get("path") or "").strip()
        if not path:
            out["caption"] = f"Error: Invalid media path for media_id={media_id}"
            return out

        media = self._build_media_input(kind, path, src)
        if not media:
            out["caption"] = f"Error: Unsupported clip kind: {kind}"
            return out

        raw = await self._safe_llm_call(
            llm=llm,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            media=media,
            node_state=node_state,
        )

        if raw is None:
            out["caption"] = "Error: VLM request failed"
            return out

        try:
            obj = parse_json_dict(raw)
            out["caption"] = str(obj.get("caption", "")).strip()
        except Exception:
            out["caption"] = raw.strip() or "Error: Unable to parse model output"

        out["source_ref"] = {"media_id": media_id}
        return out

    async def _safe_llm_call(
        self,
        *,
        llm,
        system_prompt: str,
        user_prompt: str,
        media: list[Any],
        node_state: NodeState,
        max_retries: int = 2,
    ) -> str | None:

        last_exc = None

        for attempt in range(max_retries + 1):
            try:
                return await llm.complete(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    media=media,
                    temperature=0.3,
                    top_p=0.9,
                    max_tokens=2048,
                    model_preferences=None,
                )
            except Exception as e:
                last_exc = e
                await asyncio.sleep(0.3 * (attempt + 1))

        node_state.node_summary.add_error(repr(last_exc))
        return None

    async def _build_overall_summary(
        self,
        clip_captions: list[dict[str, Any]],
        llm,
        node_state: NodeState,
    ) -> str:

        if not clip_captions:
            return ""

        lines = [
            f"- {item.get('clip_id')}: {item.get('caption')}"
            for item in clip_captions
        ]

        system_prompt = get_prompt(
            "understand_clips.system_overall", lang=node_state.lang
        )
        user_prompt = get_prompt(
            "understand_clips.user_overall",
            lang=node_state.lang,
            clips_captions=lines,
        )

        try:
            return await llm.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                media=None,
                temperature=0.3,
                top_p=0.9,
                max_tokens=1024,
                model_preferences=None,
            )
        except Exception as e:
            return f"Error: Summary generation failed: {type(e).__name__}: {e}"

    def _build_media_input(
        self,
        kind: str,
        path: str,
        src: dict[str, Any],
    ) -> list[dict[str, Any]] | None:

        if kind == "image":
            return [{"path": path}]

        if kind == "video":
            in_sec = _safe_float(src.get("start", 0) / 1000.0, 0.0)

            if src.get("end") is not None:
                out_sec = _safe_float(src.get("end", 0) / 1000.0, in_sec)
            else:
                dur = _safe_float(src.get("duration", 0), 0.0)
                out_sec = in_sec + max(0.1, dur)

            if out_sec <= in_sec:
                out_sec = in_sec + 0.1

            return [{
                "path": path,
                "in_sec": in_sec,
                "out_sec": out_sec,
            }]

        return None


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

  
