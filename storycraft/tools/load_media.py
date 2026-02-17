from typing import Any, Dict, Optional, ClassVar, Type, List
from pathlib import Path
from collections import Counter

from pydantic import BaseModel
from PIL import Image, ImageOps
import av

from storycraft.tools.base import BaseNode, NodeMeta
from storycraft.core.schema import LoadMediaInput
from storycraft.core.state import NodeState
from storycraft.utils.util import get_video_rotation
from storycraft.utils.register import NODE_REGISTRY


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


# ---------------------------------------------------------------------
# Metadata Extractors
# ---------------------------------------------------------------------

def _image_metadata_from_path(path: Path) -> Dict[str, Any]:
    """
    Extract image width / height with EXIF orientation handling.
    """
    with Image.open(path) as img:
        try:
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass
        w, h = img.size

    return {
        "width": int(w),
        "height": int(h),
    }


def _video_metadata_from_path(
    path: Path,
    *,
    round_duration_ndigits: Optional[int] = 3,
) -> Dict[str, Any]:
    """
    Extract video metadata: duration(ms), resolution, fps, audio info.
    """
    container = av.open(str(path))

    try:
        video_stream = next(
            (s for s in container.streams if s.type == "video"),
            None,
        )
        if not video_stream:
            raise ValueError("No video stream found")

        # -------- duration --------
        duration_sec = 0.0
        if container.duration is not None:
            duration_sec = container.duration / 1_000_000
        elif video_stream.duration and video_stream.time_base:
            duration_sec = float(video_stream.duration * video_stream.time_base)

        if round_duration_ndigits is not None:
            duration_sec = round(duration_sec, round_duration_ndigits)

        # -------- resolution & rotation --------
        w = int(video_stream.codec_context.width or 0)
        h = int(video_stream.codec_context.height or 0)

        rotation = get_video_rotation(path)
        if abs(rotation) in (90, 270):
            w, h = h, w

        # -------- fps --------
        fps = 0.0
        if video_stream.average_rate:
            fps = float(video_stream.average_rate)
        elif video_stream.base_rate:
            fps = float(video_stream.base_rate)

        # -------- audio --------
        audio_stream = next(
            (s for s in container.streams if s.type == "audio"),
            None,
        )

        return {
            "duration": int(duration_sec * 1000),
            "width": w,
            "height": h,
            "fps": fps,
            "has_audio": audio_stream is not None,
            "audio_sample_rate_hz": int(audio_stream.rate) if audio_stream and audio_stream.rate else 0,
        }

    finally:
        container.close()


# ---------------------------------------------------------------------
# Node Implementation
# ---------------------------------------------------------------------

@NODE_REGISTRY.register()
class LoadMediaNode(BaseNode):
    """
    Entry node for ingesting and indexing input media assets.
    """

    meta = NodeMeta(
        name="load_media",
        description="Load and index all input media files. Entry point of the entire pipeline.",
        node_id="load_media",
        node_kind="load_media",
        next_available_node=["split_shots", "split_shots_pro"],
    )

    input_schema: ClassVar[Type[BaseModel]] = LoadMediaInput

    async def default_process(self, node_state: NodeState, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return await self.process(node_state, inputs)

    async def process(self, node_state: NodeState, inputs: Dict[str, Any]) -> Dict[str, Any]:
        media_inputs = inputs.get("inputs") or []
        if not isinstance(media_inputs, list) or not media_inputs:
            node_state.node_summary.info_for_user("No input media provided.")
            return {"media": []}

        media_list: List[Dict[str, Any]] = []
        idx = 1

        for item in media_inputs:
            try:
                media = self._analyze_media_item(item, idx, node_state)
                if media:
                    media_list.append(media)
                    idx += 1
            except Exception as e:
                node_state.node_summary.info_for_user(
                    f"Media parse failed: {item.get('orig_path')} | {type(e).__name__}: {e}"
                )

        counter = Counter(m["media_type"] for m in media_list)
        node_state.node_summary.info_for_user(
            f"Media indexing completed: {counter.get('video',0)} video(s), "
            f"{counter.get('image',0)} image(s)"
        )

        return {"media": media_list}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _analyze_media_item(
        self,
        item: Dict[str, Any],
        idx: int,
        node_state: NodeState,
    ) -> Optional[Dict[str, Any]]:

        path = Path(item["path"])
        suffix = path.suffix.lower()

        if suffix in VIDEO_EXTS:
            metadata = _video_metadata_from_path(path)
            media_type = "video"
        elif suffix in IMAGE_EXTS:
            metadata = _image_metadata_from_path(path)
            media_type = "image"
        else:
            node_state.node_summary.info_for_user(
                f"Skipping unsupported media type: {item.get('orig_path')}"
            )
            return None

        media_id = f"media_{idx:04d}"

        node_state.node_summary.info_for_user(
            f"Loaded {media_id}: ({media_type})"
        )

        return {
            "media_id": media_id,
            "path": path,
            "media_type": media_type,
            "metadata": metadata,
            "orig_path": item.get("orig_path"),
            "orig_md5": item.get("orig_md5"),
        }

  
