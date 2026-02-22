from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import accumulate, pairwise
from typing import Any, Dict, List, Tuple, Optional

from storycraft.config import Settings
from storycraft.config import PlanTimelineProConfig
from storycraft.core.state import NodeState
from storycraft.tools.base import BaseNode, NodeMeta
from storycraft.core.schema import PlanTimelineInput
from storycraft.utils.register import NODE_REGISTRY


Milliseconds = int


# ============================================================
# Core Timeline Planner
# ============================================================

class TimelineProPlanner:
    """
    Professional timeline planning engine.

    Responsibilities:
    - Clip duration alignment (TTS / Beats)
    - TTS timeline placement
    - Subtitle timeline placement
    - Beat synchronization
    """

    def __init__(self, cfg: PlanTimelineProConfig):
        self.cfg = cfg

    # ============================================================
    # Main Entry
    # ============================================================

    def plan(
        self,
        node_state: NodeState,
        music: Optional[Dict[str, Any]],
        clip_durations: List[int],
        texts: List[List[str]],
        types: List[str],
        tts_res: Optional[List[Dict[str, Any]]],
        tts_indices_map: Dict[int, int],
        group_indices_map: Dict[int, int],
        title_clip_duration: int,
        is_on_beats: bool,
    ) -> Dict[str, Any]:

        music_offset, new_clip_durations, speeds, time_margins = \
            self._edit_clip_durations(
                node_state, music, clip_durations, texts, types,
                tts_res, tts_indices_map, group_indices_map,
                title_clip_duration, is_on_beats
            )

        tts_res = self._edit_tts_timeline(
            node_state, new_clip_durations, tts_res, tts_indices_map
        )

        text_start_ts, text_durations = self._edit_text_timeline(
            node_state, new_clip_durations, texts, tts_res, tts_indices_map, music
        )

        return {
            "music_offset": music_offset,
            "new_meterial_durations": new_clip_durations,
            "speeds": speeds,
            "time_margins": time_margins,
            "text_start_timestamps": text_start_ts,
            "text_durations": text_durations,
            "tts_res": tts_res,
        }

    # ============================================================
    # Clip Duration Editing
    # ============================================================

    def _edit_clip_durations(
        self,
        node_state: NodeState,
        music: Optional[Dict[str, Any]],
        clip_durations: List[int],
        texts: List[List[str]],
        types: List[str],
        tts_res: Optional[List[Dict[str, Any]]],
        tts_indices_map: Dict[int, int],
        group_indices_map: Dict[int, int],
        title_clip_duration: int,
        is_on_beats: bool,
    ) -> Tuple[int, List[int], List[float], List[int]]:

        cfg = self.cfg

        clip_durations = [
            x if x > 0 else cfg.img_default_duration for x in clip_durations
        ]

        tts_durations = (
            [item["duration"] for item in tts_res]
            if tts_res
            else [
                min(
                    cfg.min_single_text_duration * len("".join(text)),
                    cfg.max_text_duration,
                )
                for text in texts
            ]
        )

        if is_on_beats and music:
            beats = [0] + music.get("beats", [])
            beat_durations = [
                beats[i + 1] - beats[i] for i in range(len(beats) - 1)
            ] + [music["duration"] - beats[-1]]

            music_offset, new_durations = self._edit_by_beats(
                node_state, clip_durations, beat_durations,
                tts_durations, types, tts_indices_map, title_clip_duration
            )
            time_margins = [0] * len(new_durations)

        else:
            if not tts_res:
                node_state.node_summary.add_error(
                    "One of `is_on_beats` or `tts_res` must be provided."
                )
                new_durations = clip_durations
                time_margins = [0] * len(clip_durations)
            else:
                new_durations, time_margins = self._edit_by_tts(
                    node_state, clip_durations, tts_durations,
                    tts_indices_map, group_indices_map
                )
            music_offset = 0

        speeds = [
            1.0 if old >= new or t == "img" else old / new
            for t, old, new in zip(types, clip_durations, new_durations)
        ]

        return music_offset, new_durations, speeds, time_margins

    # ============================================================
    # TTS Based Duration Editing
    # ============================================================

    def _edit_by_tts(
        self,
        node_state: NodeState,
        clip_durations: List[int],
        tts_durations: List[int],
        tts_indices_map: Dict[int, int],
        group_indices_map: Dict[int, int],
    ) -> Tuple[List[int], List[int]]:

        cfg = self.cfg
        paragraph = [0] + list(accumulate(tts_indices_map.values()))

        group_margin = random.randint(cfg.min_group_margin, cfg.max_group_margin)

        time_margins = [
            self._tts_margin() + (group_margin if idx in group_indices_map else 0)
            for idx in range(len(tts_durations))
        ]

        new_durations = []
        for i, tts_dur in enumerate(tts_durations):
            start, end = paragraph[i], paragraph[i + 1]
            clip_num = end - start
            avg = max(
                int((tts_dur + time_margins[i]) / clip_num),
                cfg.min_clip_duration
            )
            new_durations.extend([avg] * clip_num)

        return new_durations, time_margins

    # ============================================================
    # Beat Based Duration Editing
    # ============================================================

    def _edit_by_beats(
        self,
        node_state: NodeState,
        clip_durations: List[int],
        beat_durations: List[int],
        tts_durations: List[int],
        types: List[str],
        tts_indices_map: Dict[int, int],
        title_clip_duration: int,
    ) -> Tuple[int, List[int]]:

        beat_index = 0
        if title_clip_duration:
            acc = 0
            while acc < title_clip_duration and beat_index < len(beat_durations):
                acc += beat_durations[beat_index]
                beat_index += 1
            music_offset = max(0, acc - title_clip_duration)
        else:
            music_offset = 0

        new_durations = []
        for dur in clip_durations:
            acc = 0
            while acc < dur:
                acc += beat_durations[beat_index]
                beat_index = (beat_index + 1) % len(beat_durations)
            new_durations.append(acc)

        return music_offset, new_durations

    # ============================================================
    # TTS Timeline
    # ============================================================

    def _edit_tts_timeline(
        self,
        node_state: NodeState,
        clip_durations: List[int],
        tts_res: Optional[List[Dict[str, Any]]],
        tts_indices_map: Dict[int, int],
    ) -> Optional[List[Dict[str, Any]]]:

        if not tts_res:
            return None

        paragraph = [0] + list(accumulate(tts_indices_map.values()))
        start_ts = [sum(clip_durations[:i]) for i in paragraph[:-1]]

        for item, ts in zip(tts_res, start_ts):
            item["start_timestamp"] = ts

        return tts_res

    # ============================================================
    # Text Timeline
    # ============================================================

    def _edit_text_timeline(
        self,
        node_state: NodeState,
        clip_durations: List[int],
        texts: List[List[str]],
        tts_res: Optional[List[Dict[str, Any]]],
        tts_indices_map: Dict[int, int],
        music: Optional[Dict[str, Any]],
    ) -> Tuple[List[List[int]], List[List[int]]]:

        paragraph = [0] + list(accumulate(tts_indices_map.values()))
        start_ts = [sum(clip_durations[:i]) for i in paragraph[:-1]]
        durations = [
            sum(clip_durations[paragraph[i]: paragraph[i + 1]])
            for i in range(len(paragraph) - 1)
        ]

        text_start, text_duration = [], []

        for txts, st, dur in zip(texts, start_ts, durations):
            total_len = sum(len(x) for x in txts)
            unit_durs = [
                int(len(x) / total_len * dur) for x in txts
            ]
            unit_st = [st + sum(unit_durs[:i]) for i in range(len(unit_durs))]

            text_start.append(unit_st)
            text_duration.append(unit_durs)

        return text_start, text_duration

    # ============================================================
    # Utility
    # ============================================================

    def _tts_margin(self) -> int:
        cfg = self.cfg
        if cfg.tts_margin_mode == "random":
            return random.randint(cfg.min_tts_margin, cfg.max_tts_margin)
        if cfg.tts_margin_mode == "avg":
            return (cfg.min_tts_margin + cfg.max_tts_margin) // 2
        if cfg.tts_margin_mode == "min":
            return cfg.min_tts_margin
        return cfg.max_tts_margin


# ============================================================
# Node Wrapper
# ============================================================

@NODE_REGISTRY.register()
class PlanTimelineProNode(BaseNode):

    meta = NodeMeta(
        name="plan_timeline_pro",
        description="Create professional multi-track timeline",
        node_id="plan_timeline_pro",
        node_kind="plan_timeline",
        require_prior_kind=[
            "split_shots",
            "group_clips",
            "generate_script",
            "tts",
            "music_rec",
        ],
        default_require_prior_kind=[
            "split_shots",
            "group_clips",
            "generate_script",
            "tts",
            "music_rec",
        ],
        next_available_node=["render_video"],
    )

    input_schema = PlanTimelineInput

    def __init__(self, server_cfg: Settings) -> None:
        super().__init__(server_cfg)
        self.planner = TimelineProPlanner(self.server_cfg.plan_timeline_pro)

    async def process(self, node_state: NodeState, inputs: Dict[str, Any]) -> Dict[str, Any]:

        parsed = self._parse_input(node_state, inputs)

        outputs = self.planner.plan(
            node_state=node_state,
            music=parsed["music"],
            clip_durations=parsed["clip_durations"],
            texts=parsed["texts"],
            types=parsed["types"],
            tts_res=parsed["tts_res"],
            tts_indices_map=parsed["text_indices_map"],
            group_indices_map=parsed["text_indices_map"],
            title_clip_duration=parsed["title_clip_duration"],
            is_on_beats=parsed["is_on_beats"],
        )

        return self._combine_outputs(node_state, parsed, outputs)

    # ============================================================
    # Output Assembly
    # ============================================================

    def _combine_outputs(self, node_state, src, out):

        tracks = {"video": [], "subtitles": [], "voiceover": [], "bgm": []}

        timeline = 0
        for clip_id, gid, kind, path, dur, new_dur, rate, size in zip(
            src["clip_ids"],
            src["clip_group_ids"],
            src["types"],
            src["clips"],
            src["clip_durations"],
            out["new_meterial_durations"],
            out["speeds"],
            src["sizes"],
        ):
            tracks["video"].append({
                "clip_id": clip_id,
                "group_id": gid,
                "kind": kind,
                "path": path,
                "source_window": {"start": 0, "end": min(dur, new_dur)},
                "timeline_window": {"start": timeline, "end": timeline + new_dur},
                "playback_rate": rate,
                "size": size,
            })
            timeline += new_dur

        return {"tracks": tracks}

    # ============================================================
    # Input Parsing
    # ============================================================

    def _parse_input(self, node_state, inputs):

        split_shots = inputs["split_shots"]
        group_clips = inputs["group_clips"]
        scripts = inputs["generate_script"]
        music = inputs.get("music_rec", {}).get("bgm")
        tts_res = inputs.get("tts", {}).get("voiceover")
        use_beats = inputs.get("use_beats", False)

        clip_ids, clip_group_ids, clip_durations, clips, types, sizes = [], [], [], [], [], []
        texts, text_indices_map = [], {}

        groups = group_clips["groups"]

        for gi, group in enumerate(groups):
            ids = group["clip_ids"]
            clip_ids.extend(ids)
            clip_group_ids.extend([group["group_id"]] * len(ids))
            text_indices_map[gi] = len(ids)

        for cid in clip_ids:
            idx = int(cid.split("_")[-1]) - 1
            clip = split_shots["clips"][idx]
            clip_durations.append(clip["source_ref"].get("duration", 0))
            clips.append(clip.get("path"))
            types.append(clip.get("kind"))
            sizes.append([
                clip["source_ref"].get("width", 576),
                clip["source_ref"].get("height", 1024),
            ])

        for script in scripts["group_scripts"]:
            texts.append([u["text"] for u in script.get("subtitle_units", [])])

        return {
            "clip_ids": clip_ids,
            "clip_group_ids": clip_group_ids,
            "clip_durations": clip_durations,
            "clips": clips,
            "types": types,
            "sizes": sizes,
            "texts": texts,
            "text_indices_map": text_indices_map,
            "music": music,
            "tts_res": tts_res,
            "is_on_beats": use_beats,
            "title_clip_duration": 0,
        }

  
