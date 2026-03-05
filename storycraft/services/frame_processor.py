"""
Frame processor

Process a single storyboard frame through the full pipeline:

TTS → Media Generation → Frame Composition → Video Segment

Key Design:
- TTS drives final video duration
- Structured pipeline steps
- Unified progress reporting
- Clear media type handling
"""

from __future__ import annotations

import os
from typing import Callable, Optional
import httpx
from loguru import logger

from storycraft.models.progress import ProgressEvent
from storycraft.models.storyboard import StoryboardFrame, StoryboardConfig


class FrameProcessor:
    """
    Frame processing pipeline.

    Orchestrates all steps required to convert a storyboard frame
    into a finalized video segment.
    """

    STEP_AUDIO = 1
    STEP_MEDIA = 2
    STEP_COMPOSE = 3
    STEP_VIDEO = 4

    def __init__(self, storycraft_core):
        self.core = storycraft_core

    # ============================================================
    # Entry
    # ============================================================

    async def __call__(
        self,
        frame: StoryboardFrame,
        storyboard,
        config: StoryboardConfig,
        total_frames: int = 1,
        progress_callback: Optional[Callable[[ProgressEvent], None]] = None,
    ) -> StoryboardFrame:

        frame_num = frame.index + 1

        logger.info(f"[FrameProcessor] Processing frame {frame_num}")

        has_existing_media = frame.image_path is not None or frame.video_path is not None
        needs_generation = frame.image_prompt is not None

        try:

            if not frame.audio_path:
                self._progress(progress_callback, frame_num, total_frames, self.STEP_AUDIO, 0.0, "audio")
                await self._step_generate_audio(frame, config)

            else:
                logger.debug(f"Using existing audio: {frame.audio_path}")

            if needs_generation:
                self._progress(progress_callback, frame_num, total_frames, self.STEP_MEDIA, 0.25, "media")
                await self._step_generate_media(frame, config)

            elif has_existing_media:
                logger.debug("Using existing media")

            else:
                frame.image_path = None
                frame.media_type = None
                logger.debug("Media generation skipped")

            compose_progress = 0.50 if (needs_generation or has_existing_media) else 0.33

            self._progress(progress_callback, frame_num, total_frames, self.STEP_COMPOSE, compose_progress, "compose")

            await self._step_compose_frame(frame, storyboard, config)

            video_progress = 0.75 if (needs_generation or has_existing_media) else 0.67

            self._progress(progress_callback, frame_num, total_frames, self.STEP_VIDEO, video_progress, "video")

            await self._step_create_video_segment(frame, config)

            logger.info(f"Frame {frame_num} completed")

            return frame

        except Exception as e:
            logger.exception(f"Frame {frame_num} failed")
            raise

    # ============================================================
    # Progress
    # ============================================================

    def _progress(
        self,
        callback,
        frame_current,
        frame_total,
        step,
        progress,
        action,
    ):
        if callback:
            callback(
                ProgressEvent(
                    event_type="frame_step",
                    progress=progress,
                    frame_current=frame_current,
                    frame_total=frame_total,
                    step=step,
                    action=action,
                )
            )

    # ============================================================
    # Step 1: TTS
    # ============================================================

    async def _step_generate_audio(
        self,
        frame: StoryboardFrame,
        config: StoryboardConfig,
    ):

        logger.debug(f"TTS generation for frame {frame.index}")

        from pixelle_video.utils.os_util import get_task_frame_path

        output_path = get_task_frame_path(config.task_id, frame.index, "audio")

        params = {
            "text": frame.narration,
            "inference_mode": config.tts_inference_mode,
            "output_path": output_path,
            "index": frame.index + 1,
        }

        if config.voice_id:
            params["voice"] = config.voice_id

        if config.tts_speed is not None:
            params["speed"] = config.tts_speed

        if config.tts_workflow:
            params["workflow"] = config.tts_workflow

        if config.ref_audio:
            params["ref_audio"] = config.ref_audio

        audio_path = await self.core.tts(**params)

        frame.audio_path = audio_path

        frame.duration = await self._get_audio_duration(audio_path)

        logger.debug(f"Audio generated ({frame.duration:.2f}s)")

    # ============================================================
    # Step 2: Media
    # ============================================================

    async def _step_generate_media(
        self,
        frame: StoryboardFrame,
        config: StoryboardConfig,
    ):

        logger.debug("Generating media")

        workflow_name = config.media_workflow or ""
        is_video = "video_" in workflow_name.lower()

        media_type = "video" if is_video else "image"

        params = {
            "prompt": frame.image_prompt,
            "workflow": config.media_workflow,
            "media_type": media_type,
            "width": config.media_width,
            "height": config.media_height,
            "index": frame.index + 1,
        }

        if is_video and frame.duration:
            params["duration"] = frame.duration

        result = await self.core.media(**params)

        frame.media_type = result.media_type

        if result.is_image:
            frame.image_path = await self._download_media(result.url, frame, config, "image")

        elif result.is_video:
            frame.video_path = await self._download_media(result.url, frame, config, "video")

            frame.duration = (
                result.duration
                if result.duration
                else await self._get_video_duration(frame.video_path)
            )

        else:
            raise ValueError(f"Unknown media type: {result.media_type}")

    # ============================================================
    # Step 3: Frame Composition
    # ============================================================

    async def _step_compose_frame(
        self,
        frame: StoryboardFrame,
        storyboard,
        config: StoryboardConfig,
    ):

        logger.debug("Composing frame")

        from storycraft.utils.os_util import get_task_frame_path
        from storycraft.utils.template_util import resolve_template_path
        from storycraft.services.html_frame import HTMLFrameGenerator

        output_path = get_task_frame_path(config.task_id, frame.index, "composed")

        template_path = resolve_template_path(config.frame_template)

        generator = HTMLFrameGenerator(template_path)

        media_path = frame.video_path if frame.media_type == "video" else frame.image_path

        ext = {"index": frame.index + 1}

        if config.template_params:
            ext.update(config.template_params)

        composed_path = await generator.generate_frame(
            title=storyboard.title,
            text=frame.narration,
            image=media_path,
            ext=ext,
            output_path=output_path,
        )

        frame.composed_image_path = composed_path

    # ============================================================
    # Step 4: Video Segment
    # ============================================================

    async def _step_create_video_segment(
        self,
        frame: StoryboardFrame,
        config: StoryboardConfig,
    ):

        logger.debug("Creating video segment")

        from storycraft.services.video import VideoService
        from storycraft.utils.os_util import get_task_frame_path

        video_service = VideoService()

        output_path = get_task_frame_path(config.task_id, frame.index, "segment")

        if frame.media_type == "video":

            temp_overlay = get_task_frame_path(config.task_id, frame.index, "overlay") + ".mp4"

            video_service.overlay_image_on_video(
                video=frame.video_path,
                overlay_image=frame.composed_image_path,
                output=temp_overlay,
                scale_mode="contain",
            )

            segment = video_service.merge_audio_video(
                video=temp_overlay,
                audio=frame.audio_path,
                output=output_path,
                replace_audio=True,
            )

            if os.path.exists(temp_overlay):
                os.unlink(temp_overlay)

        else:

            segment = video_service.create_video_from_image(
                image=frame.composed_image_path,
                audio=frame.audio_path,
                output=output_path,
                fps=config.video_fps,
            )

        frame.video_segment_path = segment

    # ============================================================
    # Utilities
    # ============================================================

    async def _get_audio_duration(self, audio_path: str) -> float:

        try:
            import ffmpeg

            probe = ffmpeg.probe(audio_path)

            return float(probe["format"]["duration"])

        except Exception as e:

            logger.warning(f"Audio duration probe failed: {e}")

            size = os.path.getsize(audio_path)

            return max(1.0, size / 2000)

    async def _get_video_duration(self, video_path: str) -> float:

        try:
            import ffmpeg

            probe = ffmpeg.probe(video_path)

            return float(probe["format"]["duration"])

        except Exception as e:

            logger.warning(f"Video duration probe failed: {e}")

            return 1.0

    async def _download_media(
        self,
        url: str,
        frame: StoryboardFrame,
        config: StoryboardConfig,
        media_type: str,
    ) -> str:

        from pixelle_video.utils.os_util import get_task_frame_path

        path = get_task_frame_path(config.task_id, frame.index, media_type)

        timeout = httpx.Timeout(connect=10, read=60)

        async with httpx.AsyncClient(timeout=timeout) as client:

            response = await client.get(url)

            response.raise_for_status()

            with open(path, "wb") as f:
                f.write(response.content)

        return path
      
