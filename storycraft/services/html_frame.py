"""
StoryCraft HTML Frame Generator

HTML-based frame rendering service.

Design Principles:
- Clear execution phases
- Structured error handling
- Chromium workaround isolation
- Linux compatibility awareness
- Minimal side effects
"""

from __future__ import annotations

import os
import re
import uuid
import shutil
import subprocess

from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from html2image import Html2Image
from loguru import logger
from PIL import Image

from storycraft.utils.template_util import parse_template_size


# ============================================================
# HTML Frame Generator
# ============================================================


class HTMLFrameGenerator:
    """
    HTML → Image frame renderer.

    Supports:
    - Custom HTML templates
    - DSL-style parameters {{param:type=default}}
    - Linux headless rendering
    - Chromium cropping workaround
    """

    CHROMIUM_HEIGHT_OFFSET = 87

    # ============================================================
    # Initialization
    # ============================================================

    def __init__(self, template_path: str):
        self.template_path = template_path
        self.template = self._load_template(template_path)

        self.width, self.height = parse_template_size(template_path)
        self._hti: Optional[Html2Image] = None

        self._check_linux_dependencies()

        logger.debug(
            f"[HTMLFrameGenerator] Loaded template "
            f"{template_path} ({self.width}x{self.height})"
        )

    # ============================================================
    # Template Loading
    # ============================================================

    def _load_template(self, template_path: str) -> str:
        path = Path(template_path)

        if not path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")

        content = path.read_text(encoding="utf-8")

        logger.debug(
            f"[HTMLFrameGenerator] Template size: {len(content)} chars"
        )
        return content

    # ============================================================
    # Linux Dependency Check
    # ============================================================

    def _check_linux_dependencies(self) -> None:
        if os.name != "posix":
            return

        try:
            result = subprocess.run(
                ["fc-list"],
                capture_output=True,
                timeout=2,
            )

            if result.returncode != 0:
                logger.warning(
                    "fontconfig not working properly. "
                    "Install: sudo apt-get install -y fontconfig fonts-liberation fonts-noto-cjk"
                )
            elif not result.stdout:
                logger.warning("No fonts detected by fontconfig.")

        except FileNotFoundError:
            logger.warning("fontconfig not installed.")
        except Exception as e:
            logger.debug(f"Dependency check skipped: {e}")

    # ============================================================
    # Media Size
    # ============================================================

    def get_media_size(self) -> Tuple[int, int]:
        return self.width, self.height

    # ============================================================
    # Parameter DSL
    # ============================================================

    PARAM_PATTERN = r"\{\{([a-zA-Z_][a-zA-Z0-9_]*)(?::([a-z]+))?(?:=([^}]+))?\}\}"

    def parse_template_parameters(self) -> Dict[str, Dict[str, Any]]:
        PRESET = {"title", "text", "image", "index"}

        params: Dict[str, Dict[str, Any]] = {}

        for match in re.finditer(self.PARAM_PATTERN, self.template):
            name, ptype, default = match.groups()
            ptype = ptype or "text"

            if name in PRESET or name in params:
                continue

            if ptype not in {"text", "number", "color", "bool"}:
                ptype = "text"

            params[name] = {
                "type": ptype,
                "default": self._parse_default(ptype, default),
                "label": name,
            }

        return params

    def _parse_default(self, ptype: str, value: Optional[str]) -> Any:
        if value is None:
            return {
                "text": "",
                "number": 0,
                "color": "#000000",
                "bool": False,
            }.get(ptype, "")

        if ptype == "number":
            try:
                return int(value) if "." not in value else float(value)
            except ValueError:
                return 0

        if ptype == "bool":
            return value.lower() in {"true", "1", "yes", "on"}

        if ptype == "color":
            return value if value.startswith("#") else f"#{value}"

        return value

    # ============================================================
    # Render Entry
    # ============================================================

    async def generate_frame(
        self,
        title: str,
        text: str,
        image: str,
        ext: Optional[Dict[str, Any]] = None,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Render HTML → PNG frame.
        """

        image = self._normalize_image_path(image)

        context = {
            "title": title,
            "text": text,
            "image": image,
        }

        if ext:
            context.update(ext)

        html = self._replace_parameters(self.template, context)

        output_path = self._prepare_output_path(output_path)

        await self._render_html(html, output_path)

        self._postprocess_crop(output_path)

        logger.info(f"✅ Frame generated: {output_path}")
        return output_path

    # ============================================================
    # Render Phases
    # ============================================================

    def _normalize_image_path(self, image: str) -> str:
        if image.startswith(("http://", "https://", "file://", "data:")):
            return image

        path = Path(image)
        if not path.is_absolute():
            path = Path.cwd() / image

        return path.as_uri()

    def _prepare_output_path(self, output_path: Optional[str]) -> str:
        if output_path is None:
            filename = f"frame_{uuid.uuid4().hex[:16]}.png"
            return str(Path.cwd() / filename)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        return output_path

    async def _render_html(self, html: str, output_path: str) -> None:
        self._ensure_hti()

        filename = Path(output_path).name

        try:
            self._hti.screenshot(
                html_str=html,
                save_as=filename,
            )

            temp_file = Path.cwd() / filename
            if temp_file.exists():
                shutil.move(str(temp_file), output_path)

        except Exception as e:
            self._raise_render_error(e)

    def _postprocess_crop(self, output_path: str) -> None:
        if not Path(output_path).exists():
            return

        with Image.open(output_path) as img:
            cropped = img.crop(
                (0, 0, self.width, self.height)
            )
            cropped.save(output_path)

    # ============================================================
    # Html2Image
    # ============================================================

    def _ensure_hti(self) -> None:
        if self._hti:
            return

        flags = [
            "--no-sandbox",
            "--disable-gpu",
            "--disable-dev-shm-usage",
            "--hide-scrollbars",
            "--mute-audio",
        ]

        self._hti = Html2Image(
            size=(self.width, self.height + self.CHROMIUM_HEIGHT_OFFSET),
            custom_flags=flags,
        )

    # ============================================================
    # Parameter Replace
    # ============================================================

    def _replace_parameters(
        self,
        html: str,
        values: Dict[str, Any],
    ) -> str:
        def replacer(match):
            name, _, default = match.groups()

            if name in values:
                value = values[name]
                return str(value) if value is not None else ""

            return default or ""

        return re.sub(self.PARAM_PATTERN, replacer, html)

    # ============================================================
    # Error Handling
    # ============================================================

    def _raise_render_error(self, error: Exception):
        logger.error(f"[HTMLFrameGenerator] Render failed: {error}")
        raise RuntimeError(f"HTML rendering failed: {error}")

  
