from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Union, Set, Dict

from storycraft.utils.logging import get_logger


logger = get_logger(__name__)


# -------------------------------------------------------------------
# Media Extension Registry
# -------------------------------------------------------------------

IMAGE_EXTS: Set[str] = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
VIDEO_EXTS: Set[str] = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


# -------------------------------------------------------------------
# Scan Result Model
# -------------------------------------------------------------------

@dataclass
class MediaScanResult:
    images: int = 0
    videos: int = 0

    @property
    def total(self) -> int:
        return self.images + self.videos

    def as_dict(self) -> Dict[str, int]:
        return asdict(self)


# -------------------------------------------------------------------
# Scanner Engine
# -------------------------------------------------------------------

class MediaScanner:
    """
    StoryCraft Media Scanner

    Features:
        - Structured statistics output
        - Unified extension registry
        - Safe directory creation
        - Debug tracing
    """

    def __init__(
        self,
        image_exts: Iterable[str] = IMAGE_EXTS,
        video_exts: Iterable[str] = VIDEO_EXTS,
    ):
        self.image_exts = set(e.lower() for e in image_exts)
        self.video_exts = set(e.lower() for e in video_exts)

    def scan(self, media_dir: Union[str, Path]) -> MediaScanResult:
        media_dir = Path(media_dir)
        media_dir.mkdir(parents=True, exist_ok=True)

        logger.debug(f"[MediaScanner] Scanning: {media_dir}")

        result = MediaScanResult()

        for path in media_dir.iterdir():
            if not path.is_file():
                continue

            if path.name.startswith("."):
                continue

            ext = path.suffix.lower()

            if ext in self.image_exts:
                result.images += 1
            elif ext in self.video_exts:
                result.videos += 1

        logger.debug(
            f"[MediaScanner] Result: images={result.images}, "
            f"videos={result.videos}, total={result.total}"
        )

        return result


# -------------------------------------------------------------------
# Functional API
# -------------------------------------------------------------------

_scanner = MediaScanner()


def scan_media_dir(media_dir: Union[str, Path]) -> Dict[str, int]:
    """
    Scan media directory.

    Returns:
        {
            "images": int,
            "videos": int,
            "total": int
        }
    """
    return _scanner.scan(media_dir).as_dict()

