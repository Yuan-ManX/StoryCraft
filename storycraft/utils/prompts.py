from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Mapping, Optional
from functools import lru_cache

from storycraft.utils.logging import get_logger


logger = get_logger(__name__)


PROMPTS_DIR = Path("prompts/tasks")

# -------------------------------------------------------------------
# Template Cache
# -------------------------------------------------------------------

@dataclass(frozen=True)
class PromptKey:
    task: str
    role: str
    lang: str

    def as_cache_key(self) -> str:
        return f"{self.task}:{self.role}:{self.lang}"


# -------------------------------------------------------------------
# Prompt Engine
# -------------------------------------------------------------------

class PromptBuilder:
    """
    StoryCraft Prompt Engine

    Design:
        - Template-based prompt generation
        - Multi-task / multi-role / multi-language
        - Strong cache + IO-free hot path
        - Safe variable rendering
    """

    _pattern = re.compile(r"{{\s*(\w+)\s*}}")

    def __init__(self, prompts_dir: Path = PROMPTS_DIR):
        self.prompts_dir = Path(prompts_dir)
        self._template_cache: Dict[str, str] = {}

        logger.debug(f"[PromptEngine] Initialized at {self.prompts_dir}")

    # ------------------------------------------------------------------
    # Template Loading
    # ------------------------------------------------------------------

    def _template_path(self, key: PromptKey) -> Path:
        return self.prompts_dir / key.task / key.lang / f"{key.role}.md"

    def _load_template(self, key: PromptKey) -> str:
        cache_key = key.as_cache_key()
        if cache_key in self._template_cache:
            return self._template_cache[cache_key]

        path = self._template_path(key)
        if not path.exists():
            raise FileNotFoundError(f"[Prompt] Template not found: {path}")

        content = path.read_text(encoding="utf-8")
        self._template_cache[cache_key] = content

        logger.debug(f"[Prompt] Loaded template: {path}")
        return content

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def _render_template(
        self,
        template: str,
        variables: Mapping[str, Any],
    ) -> str:
        """
        Safe template rendering:
        - Missing variables will raise KeyError
        - Prevent silent prompt corruption
        """

        def replacer(match: re.Match) -> str:
            key = match.group(1)
            if key not in variables:
                raise KeyError(f"Missing template variable: {key}")
            return str(variables[key])

        return self._pattern.sub(replacer, template)

    def render(
        self,
        task: str,
        role: str,
        lang: str = "zh",
        **variables: Any,
    ) -> str:
        """
        Render single prompt template.
        """
        key = PromptKey(task, role, lang)
        template = self._load_template(key)
        return self._render_template(template, variables)

    # ------------------------------------------------------------------
    # Prompt Pair Builder
    # ------------------------------------------------------------------

    def build(
        self,
        task: str,
        lang: str = "zh",
        system_vars: Optional[Dict[str, Any]] = None,
        user_vars: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """
        Build system + user prompt pair.

        Returns:
            {
                "system": "...",
                "user": "..."
            }
        """
        system_vars = system_vars or {}
        user_vars = user_vars or {}

        return {
            "system": self.render(task, "system", lang, **system_vars),
            "user": self.render(task, "user", lang, **user_vars),
        }


# -------------------------------------------------------------------
# Global Singleton Engine
# -------------------------------------------------------------------

_ENGINE = PromptBuilder()


# -------------------------------------------------------------------
# Functional API
# -------------------------------------------------------------------

@lru_cache(maxsize=256)
def get_prompt(
    name: str,
    lang: str = "zh",
    **kwargs: Any,
) -> str:
    """
    Get single rendered prompt.

    name format: "task.role"

    Example:
        get_prompt("filter_clips.system")
        get_prompt("filter_clips.user", clip_data="...")
    """
    try:
        task, role = name.split(".")
    except ValueError:
        raise ValueError(f"Invalid prompt name: '{name}', expected 'task.role'")

    return _ENGINE.render(task, role, lang, **kwargs)


def build_prompts(
    task: str,
    lang: str = "zh",
    system_vars: Optional[Dict[str, Any]] = None,
    user_vars: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Build full prompt pair.
    """
    return _ENGINE.build(task, lang, system_vars, user_vars)

