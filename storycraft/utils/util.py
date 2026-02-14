from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Union

from PIL import Image, ExifTags


logger = logging.getLogger(__name__)


PathLike = Union[str, Path]


# -----------------------------------------------------------------------------
# Video Rotation
# -----------------------------------------------------------------------------

def get_video_rotation(path: PathLike) -> int:
    """
    Read video rotation metadata using ffprobe.

    Args:
        path: Video file path

    Returns:
        Rotation degree in {0, 90, 180, 270}
    """
    path = Path(path)

    if not path.exists():
        logger.warning(f"[video-rotation] File not found: {path}")
        return 0

    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream_side_data=rotation",
        "-of", "json",
        str(path),
    ]

    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        info = json.loads(out)
    except Exception as e:
        logger.exception(f"[video-rotation] ffprobe failed: {path}", exc_info=e)
        return 0

    try:
        return int(
            next(
                (
                    sd["rotation"]
                    for sd in info.get("streams", [{}])[0].get("side_data_list", [])
                    if "rotation" in sd
                ),
                0,
            )
        )
    except Exception as e:
        logger.debug(
            f"[video-rotation] Failed parsing rotation: {path} | raw={info}",
            exc_info=e,
        )
        return 0


# -----------------------------------------------------------------------------
# Image Rotation
# -----------------------------------------------------------------------------

def get_image_rotation(path: PathLike) -> int:
    """
    Read image rotation from EXIF Orientation.

    Args:
        path: Image file path

    Returns:
        Rotation degree in {0, 90, 180, 270}
    """
    path = Path(path)

    if not path.exists():
        logger.warning(f"[image-rotation] File not found: {path}")
        return 0

    try:
        with Image.open(path) as img:
            exif = img.getexif()

        if not exif:
            return 0

        orientation_key = _get_exif_orientation_key()
        if not orientation_key:
            return 0

        orientation = exif.get(orientation_key, 1)

        return _orientation_to_angle(orientation)

    except Exception as e:
        logger.debug(f"[image-rotation] Failed reading EXIF: {path}", exc_info=e)
        return 0


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

_ORIENTATION_MAP = {
    1: 0,
    3: 180,
    6: 270,
    8: 90,
}


def _orientation_to_angle(val: int) -> int:
    """Convert EXIF orientation code to degrees."""
    return _ORIENTATION_MAP.get(val, 0)


def _get_exif_orientation_key() -> int | None:
    """Find EXIF orientation key dynamically (safe for PIL versions)."""
    for k, v in ExifTags.TAGS.items():
        if v == "Orientation":
            return k
    return None

