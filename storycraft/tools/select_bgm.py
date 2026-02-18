from typing import Any, Dict
from pathlib import Path
import asyncio
import numpy as np
import librosa

from storycraft.tools.base import BaseNode, NodeMeta
from storycraft.core.state import NodeState
from storycraft.core.schema import SelectBGMInput
from storycraft.utils.element_filter import ElementFilter
from storycraft.utils.recall import StorylineRecall
from storycraft.utils.prompts import get_prompt
from storycraft.utils.parse_json import parse_json_dict
from storycraft.utils.register import NODE_REGISTRY


@NODE_REGISTRY.register()
class SelectBGMNode(BaseNode):
    """
    Select appropriate background music (BGM) based on user request and style constraints.
    """

    meta = NodeMeta(
        name="select_bgm",
        description="Select appropriate background music based on user requirements",
        node_id="select_bgm",
        node_kind="music_rec",
        require_prior_kind=[],
        default_require_prior_kind=[],
        next_available_node=["plan_timeline"],
    )

    input_schema = SelectBGMInput

    def __init__(self, server_cfg):
        super().__init__(server_cfg)
        self.element_filter = ElementFilter(
            json_path=f"{self.server_cfg.project.bgm_dir}/meta.json"
        )
        self.vectorstore = StorylineRecall.build_vectorstore(
            self.element_filter.library
        )

    async def default_process(
        self,
        node_state: NodeState,
        inputs: Dict[str, Any],
    ) -> Any:
        node_state.node_summary.info_for_user("No valid background music selected")
        return {"bgm": {}}

    async def process(
        self,
        node_state: NodeState,
        inputs: Dict[str, Any],
    ) -> Any:

        user_request = inputs.get("user_request", "")
        filter_include = inputs.get("filter_include", {})
        filter_exclude = inputs.get("filter_exclude", {})

        try:
            bgm_info = await self._recommend_bgm(
                node_state=node_state,
                user_request=user_request,
                filter_include=filter_include,
                filter_exclude=filter_exclude,
            )
        except Exception as e:
            node_state.node_summary.add_error(repr(e))
            return {"bgm": {}}

        if not bgm_info:
            node_state.node_summary.info_for_user("No suitable BGM found")
            return {"bgm": {}}

        try:
            metrics = self._analyze_music_metrics(
                bgm_info=bgm_info,
                sr=self.server_cfg.select_bgm.sample_rate,
                hop_length=self.server_cfg.select_bgm.hop_length,
                frame_length=self.server_cfg.select_bgm.frame_length,
            )
        except Exception as e:
            node_state.node_summary.add_error(repr(e))
            return {"bgm": {}}

        if metrics.get("path"):
            node_state.node_summary.info_for_user(
                "Successfully selected background music",
                preview_urls=[metrics["path"]],
            )

        return {"bgm": metrics}

    # ------------------------------------------------------------------
    # Core recommendation pipeline
    # ------------------------------------------------------------------

    async def _recommend_bgm(
        self,
        *,
        node_state: NodeState,
        user_request: str,
        filter_include: Dict[str, Any],
        filter_exclude: Dict[str, Any],
    ) -> dict[str, Any] | None:

        bgm_dir: Path = self.server_cfg.project.bgm_dir.expanduser().resolve()
        self._validate_bgm_dir(bgm_dir)

        # Step1: semantic recall
        candidates = StorylineRecall.query_top_n(
            self.vectorstore, query=user_request
        )

        # Step2: tag filtering
        candidates = self.element_filter.filter(
            candidates, filter_include, filter_exclude
        )

        if not candidates:
            raise FileNotFoundError("No candidate audio found after filtering")

        # Step3: LLM ranking
        selected = await self._llm_select_bgm(
            node_state=node_state,
            candidates=candidates,
            user_request=user_request,
        )

        return selected or candidates[0]

    def _validate_bgm_dir(self, bgm_dir: Path):
        if not bgm_dir.exists():
            raise FileNotFoundError(f"BGM directory not found: {bgm_dir}")
        if not bgm_dir.is_dir():
            raise NotADirectoryError(f"BGM path is not a directory: {bgm_dir}")

    async def _llm_select_bgm(
        self,
        *,
        node_state: NodeState,
        candidates: list[dict[str, Any]],
        user_request: str,
    ) -> dict[str, Any] | None:

        llm = node_state.llm
        system_prompt = get_prompt("select_bgm.system", lang=node_state.lang)
        user_prompt = get_prompt(
            "select_bgm.user",
            lang=node_state.lang,
            candidates=candidates,
            user_request=user_request,
        )

        raw = await self._safe_llm_call(
            llm=llm,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        if not raw:
            return None

        try:
            obj = parse_json_dict(raw)
            if isinstance(obj, dict) and "path" in obj:
                return obj
        except Exception:
            node_state.node_summary.add_error(
                f"Invalid BGM selection output: {raw}"
            )

        return None

    async def _safe_llm_call(
        self,
        *,
        llm,
        system_prompt: str,
        user_prompt: str,
        max_retries: int = 2,
    ) -> str | None:

        last_exc = None

        for attempt in range(max_retries + 1):
            try:
                return await llm.complete(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=0.1,
                    top_p=0.9,
                    max_tokens=2048,
                    model_preferences=None,
                )
            except Exception as e:
                last_exc = e
                await asyncio.sleep(0.3 * (attempt + 1))

        return None

    # ------------------------------------------------------------------
    # Audio analysis
    # ------------------------------------------------------------------

    def _analyze_music_metrics(
        self,
        *,
        bgm_info: Dict[str, Any],
        sr: int,
        hop_length: int,
        frame_length: int,
    ) -> dict[str, Any]:

        path = Path(bgm_info.get("path", ""))
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        y, sample_rate = self._load_audio_mono(path, sr)
        duration = int(librosa.get_duration(y=y, sr=sample_rate) * 1000)

        onset_env = librosa.onset.onset_strength(
            y=y, sr=sample_rate, hop_length=hop_length
        )
        bpm, beat_frames = librosa.beat.beat_track(
            onset_envelope=onset_env,
            sr=sample_rate,
            hop_length=hop_length,
            units="frames",
        )

        beat_times = self._compute_accent_beats(
            y=y,
            sr=sample_rate,
            beat_frames=np.asarray(beat_frames, dtype=int),
            hop_length=hop_length,
        )

        rms = librosa.feature.rms(
            y=y, frame_length=frame_length, hop_length=hop_length
        )[0]

        rms_db = librosa.amplitude_to_db(
            np.maximum(rms, 1e-10), ref=1.0
        )

        return {
            "bgm_id": bgm_info.get("id"),
            "path": str(path),
            "duration": duration,
            "sample_rate": sample_rate,
            "bpm": float(np.atleast_1d(bpm)[0]),
            "beats": beat_times,
            "energy_mean": float(np.mean(rms)),
            "energy_mean_db": float(np.mean(rms_db)),
            "dynamic_range_db": float(
                np.percentile(rms_db, 95.0) - np.percentile(rms_db, 10.0)
            ),
        }

    # ------------------------------------------------------------------
    # DSP utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _load_audio_mono(path: Path, sr: int) -> tuple[np.ndarray, int]:
        try:
            y, sr_out = librosa.load(path, sr=sr, mono=True)
            return y.astype(np.float32, copy=False), int(sr_out)
        except Exception:
            return _ffmpeg_audio_fallback(path, sr)

    @staticmethod
    def _compute_accent_beats(
        y: np.ndarray,
        sr: int,
        beat_frames: np.ndarray,
        hop_length: int,
        top_pct: float = 70.0,
        min_sep_beats: int = 1,
    ) -> list[int]:

        if beat_frames.size == 0:
            return []

        onset_env = librosa.onset.onset_strength(
            y=librosa.effects.percussive(y),
            sr=sr,
            hop_length=hop_length,
        )

        beat_frames = np.clip(
            beat_frames.astype(int), 0, len(onset_env) - 1
        )
        strength = onset_env[beat_frames]

        thr = np.percentile(strength, 100.0 - top_pct)
        cand = np.where(strength >= thr)[0]

        selected = []
        suppressed = np.zeros_like(strength, dtype=bool)

        for idx in cand[np.argsort(-strength[cand])]:
            if suppressed[idx]:
                continue
            selected.append(idx)
            lo = max(0, idx - min_sep_beats)
            hi = min(strength.size, idx + min_sep_beats + 1)
            suppressed[lo:hi] = True

        frames = beat_frames[np.array(sorted(selected), dtype=int)]
        return [
            round(t * 1000)
            for t in librosa.frames_to_time(
                frames, sr=sr, hop_length=hop_length
            )
        ]


def _ffmpeg_audio_fallback(path: Path, sr: int) -> tuple[np.ndarray, int]:
    import subprocess
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(path),
                "-ac", "1",
                "-ar", str(sr),
                "-vn", tmp_path,
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        y, sr_out = librosa.load(tmp_path, sr=sr, mono=True)
        return y.astype(np.float32, copy=False), int(sr_out)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

      
