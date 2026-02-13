import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Iterable

from storycraft.utils.logging import get_logger


logger = get_logger(__name__)

FilterValue = Union[str, List[str]]
FilterDict = Dict[str, FilterValue]
Element = Dict[str, Any]


class ElementFilter:
    """
    Generic element filtering engine for structured libraries
    (music, effects, stickers, props, assets, etc.)

    Design goals:
    - Lightweight
    - Deterministic
    - Easy extension
    - Production safe
    """

    def __init__(
        self,
        library: Optional[List[Element]] = None,
        json_path: Optional[str | Path] = None,
        *,
        strict: bool = True,
    ):
        """
        Args:
            library: In-memory element list
            json_path: JSON file path for element library
            strict: Whether to enforce strict validation
        """
        self.strict = strict
        self.library: List[Element] = []

        self.reload(library=library, json_path=json_path)

    # --------------------------------------------------------------------- #
    # Load & Update
    # --------------------------------------------------------------------- #

    def reload(
        self,
        *,
        json_path: Optional[str | Path] = None,
        library: Optional[List[Element]] = None,
    ) -> None:
        """
        Reload or replace element library.

        Priority:
            library > json_path
        """
        if library is not None:
            self._validate_library(library)
            self.library = library
            logger.debug(f"[ElementFilter] Loaded library: {len(library)} items (in-memory)")
            return

        if json_path is not None:
            self.library = self._load_json(json_path)
            logger.debug(f"[ElementFilter] Loaded library: {len(self.library)} items from {json_path}")
            return

        raise ValueError("Either `library` or `json_path` must be provided")

    def _load_json(self, path: str | Path) -> List[Element]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Library JSON not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        self._validate_library(data)
        return data

    def _validate_library(self, data: Any) -> None:
        if not isinstance(data, list):
            raise ValueError("Element library must be a list of dicts")

        if self.strict:
            for i, item in enumerate(data):
                if not isinstance(item, dict):
                    raise ValueError(f"Invalid element at index {i}: {type(item)}")

    # --------------------------------------------------------------------- #
    # Core Filtering API
    # --------------------------------------------------------------------- #

    def filter(
        self,
        candidates: Optional[Iterable[Element]] = None,
        *,
        include: Optional[FilterDict] = None,
        exclude: Optional[FilterDict] = None,
        fallback_n: int = 10,
    ) -> List[Element]:
        """
        Filter elements by include / exclude conditions.

        Args:
            candidates: Optional candidate pool (default = full library)
            include: Must-match conditions
            exclude: Must-not-match conditions
            fallback_n: Random fallback size if result is empty

        Returns:
            Filtered element list
        """
        pool = list(candidates) if candidates is not None else self.library
        include = include or {}
        exclude = exclude or {}

        results = [
            item for item in pool
            if self._match_include(item, include)
            and not self._match_exclude(item, exclude)
        ]

        if not results and fallback_n > 0:
            logger.warning(
                f"[ElementFilter] Empty result, fallback sampling: {fallback_n}"
            )
            return self._fallback_sample(fallback_n)

        return results

    # --------------------------------------------------------------------- #
    # Matching Logic
    # --------------------------------------------------------------------- #

    @staticmethod
    def _normalize(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v).strip().lower() for v in value]
        return [str(value).strip().lower()]

    def _match_include(self, item: Element, include: FilterDict) -> bool:
        for key, expected in include.items():
            if key not in item:
                return False

            item_values = set(self._normalize(item.get(key)))
            expected_values = set(self._normalize(expected))

            if not (item_values & expected_values):
                return False

        return True

    def _match_exclude(self, item: Element, exclude: FilterDict) -> bool:
        for key, forbidden in exclude.items():
            if key not in item:
                continue

            item_values = set(self._normalize(item.get(key)))
            forbidden_values = set(self._normalize(forbidden))

            if item_values & forbidden_values:
                return True

        return False

    # --------------------------------------------------------------------- #
    # Fallback Strategy
    # --------------------------------------------------------------------- #

    def _fallback_sample(self, n: int) -> List[Element]:
        if not self.library:
            return []

        k = min(n, len(self.library))
        return random.sample(self.library, k)

  
