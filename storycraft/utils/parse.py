from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Iterable

from storycraft.utils.logging import get_logger


logger = get_logger(__name__)


# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------

def try_parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """
    Try to parse a tool-call JSON object from model output.

    Expected format:
        {
            "action": "call_tool",
            "tool": "...",
            "arguments": {...}
        }

    Returns:
        Parsed dict if valid, otherwise None.
    """
    try:
        obj = parse_json_object(text)
    except Exception as e:
        logger.debug(f"[ToolCallParser] JSON parse failed: {e}")
        return None

    if obj.get("action") != "call_tool":
        return None
    if "tool" not in obj:
        return None

    args = obj.get("arguments", {})
    if args is not None and not isinstance(args, dict):
        return None

    return obj


# -------------------------------------------------------------------
# Regex & Constants
# -------------------------------------------------------------------

_CODE_FENCE_RE = re.compile(
    r"```(?:json|jsonc)\s*(.*?)\s*```",
    flags=re.IGNORECASE | re.DOTALL,
)


# -------------------------------------------------------------------
# Core Parser Engine
# -------------------------------------------------------------------

class JSONRobustParser:
    """
    Robust JSON Object Parser for LLM Outputs.

    Features:
        - Supports markdown fenced JSON blocks
        - Extracts JSON objects from noisy text
        - Auto-fixes trailing commas
        - Skips braces inside strings
        - Provides best-effort parsing
    """

    @classmethod
    def parse(cls, text: str) -> Dict[str, Any]:
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")

        candidates = cls._collect_candidates(text)

        last_err: Optional[Exception] = None
        for cand in candidates:
            cleaned = cls._strip_trailing_commas(cand).strip()
            try:
                obj = json.loads(cleaned)
                if isinstance(obj, dict):
                    return obj
            except Exception as e:
                last_err = e
                continue

        raise ValueError("No valid JSON object found") from last_err

    # ---------------- internal helpers ---------------- #

    @classmethod
    def _collect_candidates(cls, text: str) -> Iterable[str]:
        # 1) fenced json blocks
        for block in cls._iter_fenced_json_blocks(text):
            yield from cls._iter_object_candidates(block)

        # 2) full text scan
        yield from cls._iter_object_candidates(text)

    @staticmethod
    def _iter_fenced_json_blocks(text: str) -> Iterable[str]:
        for m in _CODE_FENCE_RE.finditer(text):
            block = m.group(1)
            if block:
                yield block.strip()

    @classmethod
    def _iter_object_candidates(cls, text: str) -> Iterable[str]:
        for idx, ch in enumerate(text):
            if ch == "{":
                obj = cls._extract_balanced_object(text, idx)
                if obj:
                    yield obj

    @staticmethod
    def _extract_balanced_object(text: str, start: int) -> Optional[str]:
        depth = 0
        in_str = False
        escape = False

        for i in range(start, len(text)):
            c = text[i]

            if in_str:
                if escape:
                    escape = False
                elif c == "\\":
                    escape = True
                elif c == '"':
                    in_str = False
                continue

            if c == '"':
                in_str = True
                continue

            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]

        return None

    # ---------------- json cleanup ---------------- #

    @staticmethod
    def _strip_trailing_commas_once(s: str) -> str:
        out = []
        in_str = False
        escape = False
        i, n = 0, len(s)

        while i < n:
            c = s[i]

            if in_str:
                out.append(c)
                if escape:
                    escape = False
                elif c == "\\":
                    escape = True
                elif c == '"':
                    in_str = False
                i += 1
                continue

            if c == '"':
                in_str = True
                out.append(c)
                i += 1
                continue

            if c == ",":
                j = i + 1
                while j < n and s[j] in " \t\r\n":
                    j += 1
                if j < n and s[j] in "}]":
                    i += 1
                    continue

            out.append(c)
            i += 1

        return "".join(out)

    @classmethod
    def _strip_trailing_commas(cls, s: str, max_passes: int = 8) -> str:
        for _ in range(max_passes):
            s2 = cls._strip_trailing_commas_once(s)
            if s2 == s:
                return s2
            s = s2
        return s


# -------------------------------------------------------------------
# Functional Wrapper
# -------------------------------------------------------------------

def parse_json_object(text: str) -> Dict[str, Any]:
    """
    Robust JSON dict parser from LLM outputs.
    """
    return JSONRobustParser.parse(text)

