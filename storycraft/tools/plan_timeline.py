from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Tuple

from storycraft.config import Settings
from storycraft.config import PlanTimelineConfig
from storycraft.core.state import NodeState
from storycraft.tools.base import BaseNode, NodeMeta
from storycraft.core.schema import PlanTimelineInput
from storycraft.utils.register import NODE_REGISTRY


# ============================================================
# Constants
# ============================================================

Milliseconds = int

DEFAULT_RANDOM_SEED = 42

SECONDS_PER_MINUTE = 60.0
MILLISECONDS_PER_SECOND = 1000.0

SNAP_SAFETY_MAX_STEPS = 10_000
BINARY_SEARCH_ITERATIONS = 50

RATIO_GROWTH_FACTOR = 2.0
RATIO_GROWTH_MAX = 10.0

MIN_SUBTITLE_WEIGHT = 1
CENTER_ALIGN_DIVISOR = 2.0


# ============================================================
# Dataclasses
# ============================================================

@dataclass(frozen=True)
class BeatTrack:
    beat_timestamps_ms: List[Milliseconds]
    beat_durations_ms: List[Milliseconds]
    music_duration_ms: Milliseconds


# ============================================================
# Core Planner
# ============================================================

class TimelinePlanner:
    """
    Pure timeline planning engine.

    Responsible ONLY for:
    - duration modeling
    - beat alignment
    - timeline segmentation
    """

    def __init__(self, config: PlanTimelineConfig, *, random_seed: int = DEFAULT_RANDOM_SEED) -> None:
        self.cfg = config
        self.rng = random.Random(random_seed)

    # ----------------------------------------------------------
    # Entry
    # ----------------------------------------------------------

    def plan(
        self,
        *,
        media: List[Dict[str, Any]],
        clips: List[Dict[str, Any]],
        groups: List[Dict[str, Any]],
        group_scripts: List[Dict[str, Any]],
        voiceovers: List[Dict[str, Any]],
        background_music: Optional[Dict[str, Any]],
        use_beats: bool,
    ) -> Dict[str, Any]:

        media_by_id = self._index_items(media, "media_id")
        clips_by_id = self._index_items(clips, "clip_id")
        scripts_by_gid = self._index_items(group_scripts, "group_id")
        voiceovers_by_gid = self._index_items(voiceovers, "group_id")

        beat_track = self._build_beat_track(background_music, use_beats)

        music_offset, start_beat_idx = self._compute_music_offset(
            beat_track.beat_durations_ms,
            beat_track.music_duration_ms,
            use_beats,
        )

        video_segments, group_states, total_duration, _ = self._build_video_track(
            groups,
            clips_by_id,
            media_by_id,
            scripts_by_gid,
            voiceovers_by_gid,
            background_music,
            beat_track.beat_durations_ms,
            start_beat_idx,
            use_beats,
        )

        return {
            "tracks": {
                "video": video_segments,
                "subtitles": self._build_subtitle_track(groups, group_states),
                "voiceover": self._build_voiceover_track(groups, group_states),
                "bgm": self._build_bgm_track(background_music, total_duration, music_offset),
            }
        }

    # ----------------------------------------------------------
    # Index Helpers
    # ----------------------------------------------------------

    @staticmethod
    def _index_items(items: List[Dict[str, Any]], key: str) -> Dict[str, Dict[str, Any]]:
        return {str(x.get(key)): x for x in items or [] if x.get(key) is not None}

    # ----------------------------------------------------------
    # Beats
    # ----------------------------------------------------------

    def _build_beat_track(self, bgm: Optional[Dict[str, Any]], use_beats: bool) -> BeatTrack:
        if not use_beats or not bgm:
            return BeatTrack([], [], 0)

        duration = int(bgm.get("duration", 0))
        timestamps = self._extract_beat_timestamps(bgm)
        durations = self._timestamps_to_durations(timestamps, duration)

        return BeatTrack(timestamps, durations, duration)

    def _extract_beat_timestamps(self, bgm: Dict[str, Any]) -> List[Milliseconds]:
        beats = bgm.get("beats") or []
        duration = int(bgm.get("duration", 0))

        if beats:
            return [0] + beats if beats[0] != 0 else beats

        bpm = bgm.get("bpm")
        if not bpm:
            return [0]

        interval = int(SECONDS_PER_MINUTE / float(bpm) * MILLISECONDS_PER_SECOND)
        return list(range(0, duration + 1, interval))

    @staticmethod
    def _timestamps_to_durations(ts: List[Milliseconds], total: Milliseconds) -> List[Milliseconds]:
        if len(ts) < 2:
            return []
        d = [b - a for a, b in zip(ts[:-1], ts[1:])]
        if total > ts[-1]:
            d.append(total - ts[-1])
        return d

    def _compute_music_offset(
        self,
        beat_durations: List[Milliseconds],
        music_duration: Milliseconds,
        use_beats: bool,
    ) -> Tuple[Milliseconds, int]:

        if not use_beats or not beat_durations:
            return 0, 0

        title_duration = int(getattr(self.cfg, "title_duration", 0))
        if title_duration <= 0:
            return 0, 0

        acc = 0
        idx = 0
        while acc < title_duration and idx < len(beat_durations):
            acc += beat_durations[idx]
            idx += 1

        return max(0, acc - title_duration), idx % len(beat_durations)

    # ----------------------------------------------------------
    # Video Track
    # ----------------------------------------------------------

    def _build_video_track(
        self,
        groups,
        clips_by_id,
        media_by_id,
        scripts_by_gid,
        voiceovers_by_gid,
        bgm,
        beat_durations,
        beat_idx,
        use_beats,
    ):
        segments = []
        group_states = {}
        cursor = 0

        for group in groups:
            gid = str(group["group_id"])
            clip_ids = [str(x) for x in group.get("clip_ids", [])]
            clips = [clips_by_id[cid] for cid in clip_ids]

            script = scripts_by_gid.get(gid)
            voice = voiceovers_by_gid.get(gid)

            narration_duration = self._estimate_narration_duration(script, voice)

            target_duration = max(
                narration_duration + int(self.cfg.group_margin_over_voiceover),
                len(clips) * int(self.cfg.min_clip_duration),
            )

            if use_beats and bgm:
                durations, beat_idx = self._allocate_using_beats(
                    clips, target_duration, beat_durations, beat_idx
                )
            else:
                durations = self._allocate_without_beats(clips, target_duration)

            group_start = cursor

            for clip, dur in zip(clips, durations):
                seg = self._build_clip_segment(clip, media_by_id, cursor, dur)
                segments.append(seg)
                cursor += dur

            group_states[gid] = {
                "start": group_start,
                "end": cursor,
                "duration": cursor - group_start,
                "voiceover": voice,
                "script": script,
            }

        return segments, group_states, cursor, beat_idx

    # ----------------------------------------------------------
    # Clip Processing
    # ----------------------------------------------------------

    def _build_clip_segment(self, clip, media_by_id, start_ms, dur_ms):
        src_start, src_end, src_dur = self._get_source_window(clip)

        if dur_ms >= src_dur:
            playback = src_dur / dur_ms if dur_ms > 0 else 1.0
            src_s, src_e = src_start, src_end
        else:
            offset = self.rng.randint(0, max(0, src_dur - dur_ms))
            src_s = src_start + offset
            src_e = src_s + dur_ms
            playback = 1.0

        return {
            "clip_id": clip.get("clip_id"),
            "group_id": clip.get("group_id"),
            "kind": clip.get("kind"),
            "path": clip.get("path"),
            "source_path": self._resolve_source_path(clip, media_by_id),
            "source_window": {"start": src_s, "end": src_e},
            "timeline_window": {"start": start_ms, "end": start_ms + dur_ms},
            "playback_rate": playback,
        }

    @staticmethod
    def _resolve_source_path(clip, media_by_id):
        ref = clip.get("source_ref") or {}
        return media_by_id.get(str(ref.get("media_id")), {}).get("path")

    def _get_source_window(self, clip):
        ref = clip.get("source_ref") or {}
        s = int(ref.get("start", 0))
        d = int(ref.get("duration", 0))
        return s, s + d, d

    # ----------------------------------------------------------
    # Duration Modeling
    # ----------------------------------------------------------

    def _estimate_narration_duration(self, script, voice):
        if voice and voice.get("duration", 0) > 0:
            return int(voice["duration"])
        if not script:
            return int(self.cfg.estimate_text_min)

        text = str(script.get("raw_text", "")).strip()
        cps = max(1.0, float(self.cfg.estimate_text_char_per_sec))
        return max(
            int(len(text) / cps * MILLISECONDS_PER_SECOND),
            int(self.cfg.estimate_text_min),
        )

    def _allocate_using_beats(self, clips, target_ms, beats, beat_idx):
        weights = [self._get_source_window(c)[2] for c in clips]
        total = sum(weights)

        targets = [(target_ms * w) // total for w in weights]
        carry = target_ms - sum(targets)
        for i in range(carry):
            targets[i % len(targets)] += 1

        durations = []
        for t in targets:
            acc = 0
            while acc < t:
                acc += beats[beat_idx]
                beat_idx = (beat_idx + 1) % len(beats)
            durations.append(acc)

        return durations, beat_idx

    def _allocate_without_beats(self, clips, target_ms):
        weights = [self._get_source_window(c)[2] for c in clips]
        total = sum(weights)
        ratio = target_ms / max(1, total)
        return [max(int(w * ratio), int(self.cfg.min_clip_duration)) for w in weights]

    # ----------------------------------------------------------
    # Subtitle Track
    # ----------------------------------------------------------

    def _build_subtitle_track(self, groups, states):
        subs = []
        for g in groups:
            gid = str(g["group_id"])
            state = states.get(gid)
            if not state or not state.get("script"):
                continue

            units = state["script"].get("subtitle_units", [])
            if not units:
                continue

            start = int(state["start"])
            end = int(state["end"])
            total = end - start

            weights = [max(len(u.get("text", "")), MIN_SUBTITLE_WEIGHT) for u in units]
            sw = sum(weights)

            cursor = start
            for u, w in zip(units, weights):
                dur = int(total * w / sw)
                subs.append({
                    "group_id": gid,
                    "unit_id": u.get("unit_id"),
                    "text": u.get("text"),
                    "timeline_window": {"start": cursor, "end": cursor + dur}
                })
                cursor += dur

        return subs

    # ----------------------------------------------------------
    # Voiceover Track
    # ----------------------------------------------------------

    def _build_voiceover_track(self, groups, states):
        segments = []
        for g in groups:
            gid = str(g["group_id"])
            state = states.get(gid)
            voice = state.get("voiceover") if state else None
            if not voice:
                continue

            dur = int(voice.get("duration", 0))
            if dur <= 0:
                continue

            gstart, gend = state["start"], state["end"]
            offset = (gend - gstart - dur) / CENTER_ALIGN_DIVISOR
            start = gstart + offset

            segments.append({
                "group_id": gid,
                "voiceover_id": voice.get("voiceover_id"),
                "path": voice.get("path"),
                "timeline_window": {"start": start, "end": start + dur},
            })

        return segments

    # ----------------------------------------------------------
    # BGM Track
    # ----------------------------------------------------------

    def _build_bgm_track(self, bgm, total_ms, offset):
        if not bgm:
            return []

        dur = int(bgm.get("duration", 0))
        if dur <= 0:
            return []

        cursor = 0
        src = offset
        segments = []
        loop = 0

        while cursor < total_ms:
            chunk = min(dur - src, total_ms - cursor)
            segments.append({
                "bgm_id": bgm.get("bgm_id"),
                "path": bgm.get("path"),
                "source_window": {"start": src, "end": src + chunk},
                "loop_idx": loop,
            })
            cursor += chunk
            src += chunk
            if src >= dur:
                src = 0
                loop += 1

        return segments


# ============================================================
# Node Wrapper
# ============================================================

@NODE_REGISTRY.register()
class PlanTimelineNode(BaseNode):
    meta = NodeMeta(
        name="plan_timeline",
        description="Plan full multi-track timeline for video rendering",
        node_id="plan_timeline",
        node_kind="plan_timeline",
        require_prior_kind=["load_media", "split_shots", "group_clips"],
        default_require_prior_kind=["load_media", "split_shots", "group_clips"],
        next_available_node=["render_video"],
    )

    input_schema = PlanTimelineInput

    def __init__(self, server_cfg: Settings) -> None:
        super().__init__(server_cfg)
        self.planner = TimelinePlanner(self.server_cfg.plan_timeline)

    async def process(self, node_state: NodeState, inputs: Dict[str, Any]) -> Dict[str, Any]:
        result = self.planner.plan(
            media=(inputs.get("load_media") or {}).get("media", []),
            clips=(inputs.get("split_shots") or {}).get("clips", []),
            groups=(inputs.get("group_clips") or {}).get("groups", []),
            group_scripts=(inputs.get("generate_script") or {}).get("group_scripts", []),
            voiceovers=(inputs.get("tts") or {}).get("voiceover", []),
            background_music=(inputs.get("music_rec") or {}).get("bgm"),
            use_beats=bool(inputs.get("use_beats", False)),
        )

        node_state.node_summary.info_for_user("时间线构建完成")
        return result

  
