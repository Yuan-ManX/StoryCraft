from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


# ============================================================
# Prompt Template
# ============================================================

STYLE_CONVERSION_PROMPT_TEMPLATE = """Convert the following style description into a detailed image generation prompt for Stable Diffusion or FLUX models.

Style Description:
{description}

Constraints:
- Focus on visual elements, colors, lighting, materials, textures, mood, and atmosphere
- Use professional photography and digital art terminology
- Be vivid, precise, and concrete
- Output ONLY the final prompt in English
- Do NOT include explanations or formatting
- Use comma-separated descriptive phrases
- Maximum length: 100 words

Image Generation Prompt:
"""


# ============================================================
# Prompt Config
# ============================================================

@dataclass
class StylePromptConfig:
    """
    Configuration for style prompt generation.
    """

    max_words: int = 100
    language: str = "en"
    enforce_english: bool = True


# ============================================================
# Prompt Builder
# ============================================================

class StylePromptBuilder:
    """
    Prompt builder for style-to-image conversion.

    This class encapsulates all prompt construction logic,
    making it easier to extend, test, and maintain.
    """

    def __init__(self, config: Optional[StylePromptConfig] = None):
        self.config = config or StylePromptConfig()

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------

    def build(self, description: str) -> str:
        """
        Build a structured prompt for style-to-image conversion.

        Args:
            description: User's style description in any language.

        Returns:
            A formatted prompt string for image generation models.

        Raises:
            ValueError: If the description is empty or invalid.
        """
        description = self._sanitize(description)
        self._validate(description)

        return STYLE_CONVERSION_PROMPT_TEMPLATE.format(
            description=description
        )

    # ------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------

    def _sanitize(self, text: str) -> str:
        """
        Normalize user input.

        - Strip whitespace
        - Collapse excessive line breaks
        """
        return " ".join((text or "").strip().split())

    def _validate(self, description: str) -> None:
        """
        Validate input description.
        """
        if not description:
            raise ValueError("Style description must not be empty.")


# ============================================================
# Convenience Function (Public API)
# ============================================================

def build_style_conversion_prompt(
    description: str,
    *,
    config: Optional[StylePromptConfig] = None,
) -> str:
    """
    Build style conversion prompt.

    This is a lightweight wrapper around StylePromptBuilder,
    suitable for direct use in nodes or agents.

    Args:
        description: User's style description in any language.
        config: Optional prompt config.

    Returns:
        A formatted image generation prompt.

    Example:
        >>> build_style_conversion_prompt("Animation")
        "cyberpunk style, neon lights, futuristic cityscape, high contrast lighting..."
    """
    return StylePromptBuilder(config).build(description)
  
