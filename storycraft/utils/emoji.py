import json
import os
import re
from pathlib import Path
from typing import Iterable, List, Optional

import emoji

from storycraft.utils.logging import get_logger


logger = get_logger(__name__)


DEFAULT_EMOJI_JSON = Path("./resource/unicode_emojis.json")


class EmojiFilter:
    """
    Emoji normalization and detection engine.

    Design goals:
    - High performance
    - High recall
    - Safe fallback
    - Deterministic behavior
    """

    # Wide Unicode emoji range
    _EMOJI_REGEX = re.compile(
        "[\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002700-\U000027BF"
        "\U0001F900-\U0001F9FF"
        "\U00002600-\U000026FF"
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F0A0-\U0001F0FF"
        "\U0001F201-\U0001F2FF"
        "\U0001F300-\U0001F3F0"
        "\U00002300-\U000023FF"
        "\U0001F004"
        "\U00002B06"
        "\u200D"
        "]+",
        flags=re.UNICODE,
    )

    def __init__(
        self,
        emoji_json_path: Optional[str | Path] = None,
        *,
        strict: bool = False,
    ):
        """
        Args:
            emoji_json_path: Optional emoji unicode mapping json
            strict: Whether to enforce strict initialization
        """
        self.strict = strict
        self.emoji_json_path = Path(emoji_json_path) if emoji_json_path else DEFAULT_EMOJI_JSON

        self._emoji_table: List[str] = []
        self._emoji_table_re: Optional[re.Pattern] = None

        self._load_emoji_table()

    # ------------------------------------------------------------------ #
    # Initialization
    # ------------------------------------------------------------------ #

    def _load_emoji_table(self) -> None:
        if not self.emoji_json_path.exists():
            if self.strict:
                raise FileNotFoundError(f"Emoji json not found: {self.emoji_json_path}")
            logger.warning(f"[EmojiFilter] Emoji json not found: {self.emoji_json_path}, fallback to regex only.")
            return

        try:
            with self.emoji_json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                raise ValueError("Emoji json must be a list")

            self._emoji_table = sorted(data, key=len, reverse=True)
            self._emoji_table_re = re.compile("|".join(map(re.escape, self._emoji_table)))

            logger.info(f"[EmojiFilter] Loaded {len(self._emoji_table)} emoji patterns")

        except Exception as e:
            if self.strict:
                raise
            logger.exception(f"[EmojiFilter] Failed loading emoji table: {e}")

    # ------------------------------------------------------------------ #
    # Core APIs
    # ------------------------------------------------------------------ #

    def strip(self, text: str) -> str:
        """
        Remove all emoji characters from text.
        """
        if not text:
            return text

        if self._emoji_table_re:
            text = self._emoji_table_re.sub("", text)

        return self._EMOJI_REGEX.sub("", text)

    def contains_only_emoji(self, text: str) -> bool:
        """
        Check whether text contains only emoji characters (ignoring whitespace).
        """
        if not text:
            return False

        text = text.strip().replace(" ", "")
        if not text:
            return False

        if self._emoji_table_re and self._emoji_table_re.fullmatch(text):
            return True

        return all(self.is_emoji(ch) for ch in text)

    @staticmethod
    def is_emoji(ch: str) -> bool:
        """
        Fast emoji detection.
        """
        if not ch:
            return False
        return bool(EmojiFilter._EMOJI_REGEX.match(ch) or emoji.is_emoji(ch))

  
