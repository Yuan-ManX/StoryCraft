from __future__ import annotations

import asyncio
import base64
import math
import os
from io import BytesIO
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

from PIL import Image
from moviepy.video.io.VideoFileClip import VideoFileClip
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from mcp.types import CreateMessageRequestParams, CreateMessageResult, TextContent

from storycraft.utils.logging import get_logger


logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DEFAULT_RESIZE_EDGE = 600
DEFAULT_JPEG_QUALITY = 80
DEFAULT_MIN_FRAMES = 2
DEFAULT_MAX_FRAMES = 6
DEFAULT_FRAMES_PER_SEC = 3.0
GLOBAL_MAX_IMAGE_BLOCKS = 48

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


# -----------------------------------------------------------------------------
# URL & Path Utilities
# -----------------------------------------------------------------------------

def is_data_url(url: str) -> bool:
    return isinstance(url, str) and url.startswith("data:")


def is_http_url(url: str) -> bool:
    return isinstance(url, str) and url.startswith(("http://", "https://"))


def normalize_local_path(url: str) -> str:
    if url.startswith("file://"):
        return urlparse(url).path
    return url


def guess_extension(path_or_url: str) -> str:
    try:
        parsed = urlparse(path_or_url)
        return os.path.splitext(parsed.path if parsed.scheme else path_or_url)[1].lower()
    except Exception:
        return ""


# -----------------------------------------------------------------------------
# Image Processing
# -----------------------------------------------------------------------------

def resize_long_edge(img: Image.Image, long_edge: int) -> Image.Image:
    if long_edge <= 0:
        return img

    w, h = img.size
    scale = long_edge / max(w, h)
    if scale >= 1.0:
        return img

    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return img.resize((nw, nh), Image.LANCZOS)


def pil_to_data_url(img: Image.Image, resize_edge: int, jpeg_quality: int) -> str:
    img = img.convert("RGB")
    img = resize_long_edge(img, resize_edge)

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return f"data:image/jpeg;base64,{b64}"


def image_path_to_data_url(path: str, resize_edge: int, jpeg_quality: int) -> str:
    with Image.open(path) as img:
        return pil_to_data_url(img, resize_edge, jpeg_quality)


# -----------------------------------------------------------------------------
# Video Frame Sampling
# -----------------------------------------------------------------------------

def choose_num_frames(
    duration_sec: float,
    min_frames: int,
    max_frames: int,
    frames_per_sec: float,
) -> int:
    n = int(math.ceil(max(0.0, duration_sec) * frames_per_sec))
    return max(min_frames, min(max_frames, n))


def sample_video_segment(
    video_path: str,
    in_sec: float,
    out_sec: float,
    resize_edge: int,
    jpeg_quality: int,
    min_frames: int,
    max_frames: int,
    frames_per_sec: float,
) -> List[Tuple[float, str]]:
    """
    Sample frames from [in_sec, out_sec].
    Returns list of (relative_time, data_url).
    """
    clip = VideoFileClip(video_path, audio=False)

    try:
        duration = float(clip.duration or 0.0)

        in_sec = max(0.0, min(in_sec, duration))
        out_sec = max(0.0, min(out_sec, duration))

        if out_sec <= in_sec:
            frame = clip.get_frame(in_sec)
            img = Image.fromarray(frame)
            return [(0.0, pil_to_data_url(img, resize_edge, jpeg_quality))]

        seg_dur = out_sec - in_sec
        n = choose_num_frames(seg_dur, min_frames, max_frames, frames_per_sec)

        times = [((i + 0.5) / n) * seg_dur for i in range(n)]

        frames: List[Tuple[float, str]] = []
        for rel_t in times:
            frame = clip.get_frame(in_sec + rel_t)
            img = Image.fromarray(frame)
            frames.append((rel_t, pil_to_data_url(img, resize_edge, jpeg_quality)))

        return frames
    finally:
        clip.close()


# -----------------------------------------------------------------------------
# MCP Message Utilities
# -----------------------------------------------------------------------------

def extract_text_from_mcp(content: Any) -> str:
    if not content:
        return ""

    blocks = content if isinstance(content, list) else [content]
    texts = [
        getattr(b, "text", "")
        for b in blocks
        if getattr(b, "type", None) == "text"
    ]
    return "\n".join(x.strip() for x in texts if x.strip())


def extract_text_from_lc(resp: Any) -> str:
    content = getattr(resp, "content", None)
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        return "\n".join(
            str(b.get("text", "")).strip()
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )

    return str(resp).strip()


# -----------------------------------------------------------------------------
# Media Normalization
# -----------------------------------------------------------------------------

def normalize_media_inputs(media_inputs: List[Any]) -> List[Dict[str, Any]]:
    out = []
    for item in media_inputs or []:
        if isinstance(item, str):
            out.append({"url": item})
        elif isinstance(item, (tuple, list)):
            d = {"url": item[0]}
            if len(item) > 1:
                d["in_sec"] = item[1]
            if len(item) > 2:
                d["out_sec"] = item[2]
            out.append(d)
        elif isinstance(item, dict):
            url = item.get("url") or item.get("path")
            if not url:
                continue
            d = {"url": url}
            d.update({k: item[k] for k in ("in_sec", "out_sec") if k in item})
            out.append(d)
    return out


# -----------------------------------------------------------------------------
# Media Block Builder
# -----------------------------------------------------------------------------

def build_media_blocks(
    media_inputs: List[Any],
    *,
    resize_edge: int,
    jpeg_quality: int,
    min_frames: int,
    max_frames: int,
    frames_per_sec: float,
    global_max_images: int,
) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    img_count = 0

    for idx, item in enumerate(normalize_media_inputs(media_inputs)):
        if img_count >= global_max_images:
            break

        url = normalize_local_path(str(item["url"]))
        ext = guess_extension(url)

        in_sec = item.get("in_sec", 0.0)
        out_sec = item.get("out_sec", 1e12)
        has_segment = "in_sec" in item and "out_sec" in item

        if is_data_url(url):
            blocks.extend([
                {"type": "text", "text": f"Media {idx+1}: inline image"},
                {"type": "image_url", "image_url": {"url": url}},
            ])
            img_count += 1
            continue

        if is_http_url(url):
            if ext in VIDEO_EXTS:
                blocks.append({"type": "text", "text": f"Media {idx+1}: remote video {url}"})
                continue

            blocks.extend([
                {"type": "text", "text": f"Media {idx+1}: {url}"},
                {"type": "image_url", "image_url": {"url": url}},
            ])
            img_count += 1
            continue

        if not os.path.exists(url):
            blocks.append({"type": "text", "text": f"Media {idx+1}: missing file {url}"})
            continue

        if ext in IMAGE_EXTS:
            data_url = image_path_to_data_url(url, resize_edge, jpeg_quality)
            blocks.extend([
                {"type": "text", "text": f"Media {idx+1}: {os.path.basename(url)}"},
                {"type": "image_url", "image_url": {"url": data_url}},
            ])
            img_count += 1
            continue

        if ext in VIDEO_EXTS:
            frames = sample_video_segment(
                url, in_sec, out_sec,
                resize_edge, jpeg_quality,
                min_frames, max_frames, frames_per_sec,
            )
            blocks.append({
                "type": "text",
                "text": f"Media {idx+1}: video {os.path.basename(url)} frames",
            })
            for i, (t, data_url) in enumerate(frames):
                if img_count >= global_max_images:
                    break
                blocks.extend([
                    {"type": "text", "text": f"Frame {i+1}/{len(frames)} (t≈{t:.2f}s)"},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ])
                img_count += 1
            continue

        blocks.append({"type": "text", "text": f"Media {idx+1}: unsupported {url}"})

    return blocks


# -----------------------------------------------------------------------------
# MCP Sampling Callback Factory
# -----------------------------------------------------------------------------

def make_sampling_callback(
    llm,
    vlm,
    *,
    resize_edge: int = DEFAULT_RESIZE_EDGE,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    min_frames: int = DEFAULT_MIN_FRAMES,
    max_frames: int = DEFAULT_MAX_FRAMES,
    frames_per_sec: float = DEFAULT_FRAMES_PER_SEC,
    global_max_images: int = GLOBAL_MAX_IMAGE_BLOCKS,
):
    async def sampling_callback(context, params: CreateMessageRequestParams) -> CreateMessageResult:
        try:
            system_prompt = getattr(params, "systemPrompt", "") or ""
            metadata = getattr(params, "metadata", {}) or {}

            temperature = float(getattr(params, "temperature", 0.6) or 0.6)
            max_tokens = int(getattr(params, "maxTokens", 4096) or 4096)
            top_p = float(metadata.get("top_p", 0.9))

            messages = getattr(params, "messages", []) or []
            media_inputs = metadata.get("media", []) or []

            lc_messages = []
            if system_prompt:
                lc_messages.append(SystemMessage(content=system_prompt))

            media_blocks = []
            if media_inputs:
                media_blocks = await asyncio.to_thread(
                    build_media_blocks,
                    media_inputs,
                    resize_edge=resize_edge,
                    jpeg_quality=jpeg_quality,
                    min_frames=min_frames,
                    max_frames=max_frames,
                    frames_per_sec=frames_per_sec,
                    global_max_images=global_max_images,
                )

            user_indices = [i for i, m in enumerate(messages) if getattr(m, "role", "") == "user"]
            last_user_idx = user_indices[-1] if user_indices else None

            for i, m in enumerate(messages):
                role = getattr(m, "role", "user")
                text = extract_text_from_mcp(getattr(m, "content", None))

                if role == "assistant":
                    lc_messages.append(AIMessage(content=text))
                else:
                    if i == last_user_idx and media_blocks:
                        blocks = [{"type": "text", "text": text}] + media_blocks
                        lc_messages.append(HumanMessage(content=blocks))
                    else:
                        lc_messages.append(HumanMessage(content=text))

            model = vlm if media_inputs else llm
            bound = model.bind(temperature=temperature, max_tokens=max_tokens, top_p=top_p)

            resp = await bound.ainvoke(lc_messages) if hasattr(bound, "ainvoke") else \
                   await asyncio.to_thread(bound.invoke, lc_messages)

            return CreateMessageResult(
                content=TextContent(type="text", text=extract_text_from_lc(resp)),
                model=str(getattr(model, "model", None) or getattr(model, "model_name", None)),
                role="assistant",
                stopReason="endTurn",
            )

        except Exception as e:
            logger.exception("[MCP] sampling callback failed")
            return CreateMessageResult(
                content=TextContent(type="text", text=str(e)),
                model="unknown",
                role="assistant",
                stopReason="error",
            )

    return sampling_callback

