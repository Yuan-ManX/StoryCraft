from __future__ import annotations

import aiofiles
from pathlib import Path
from typing import List, Dict, Any

from skillkit import SkillManager
from skillkit.integrations.langchain import create_langchain_tools

from storycraft.utils.logging import get_logger


logger = get_logger(__name__)


# ================================
# Skill Loading
# ================================

async def load_skills(skill_dir: str = ".storycraft/skills"):
    """
    Discover and load skills, converting them into LangChain-compatible tools.

    Args:
        skill_dir: Root directory containing skill definitions.

    Returns:
        List of LangChain tools.
    """
    logger.info(f"[SkillLoader] Discovering skills from: {skill_dir}")

    manager = SkillManager(skill_dir=skill_dir)
    await manager.adiscover()

    tools = create_langchain_tools(manager)

    logger.info(f"[SkillLoader] Loaded {len(tools)} skills")
    return tools


# ================================
# Skill Dumping
# ================================

async def dump_skills(
    *,
    skill_name: str,
    skill_dir: str,
    skill_content: str,
    **kwargs,
) -> Dict[str, Any]:
    """
    Persist skill markdown content into project skill directory.

    Path structure:
        <project_root>/<skill_dir>/cutskill_<skill_name>/SKILL.md

    Security:
        - Enforces project root sandbox
        - Prevents path traversal

    Args:
        skill_name: Logical name of the skill.
        skill_dir: Root skill directory.
        skill_content: Markdown content of the skill.

    Returns:
        Standard result dict.
    """
    clean_name = skill_name.strip()
    if not clean_name:
        return _error("skill_name cannot be empty")

    base_path = Path.cwd()
    target_path = base_path / skill_dir / f"cutskill_{clean_name}"
    target_file = target_path / "SKILL.md"

    logger.info(f"[SkillDump] Writing skill: {clean_name} -> {target_file}")

    # ---- Security validation ----
    try:
        resolved_path = target_file.resolve()
        if base_path not in resolved_path.parents:
            return _error(
                f"Security violation: Attempted path escape: {resolved_path}"
            )
    except Exception as e:
        logger.exception("[SkillDump] Path resolution failed")
        return _error(f"Path resolution failed: {str(e)}")

    # ---- Write content ----
    try:
        target_path.mkdir(parents=True, exist_ok=True)

        async with aiofiles.open(
            resolved_path, mode="w", encoding="utf-8"
        ) as f:
            await f.write(skill_content)

        size = len(skill_content.encode("utf-8"))

        logger.info(
            f"[SkillDump] Skill '{clean_name}' written successfully "
            f"({size} bytes)"
        )

        return {
            "status": "success",
            "message": f"Skill '{clean_name}' successfully created.",
            "dir_path": str(target_path),
            "file_path": str(resolved_path),
            "size_bytes": size,
        }

    except PermissionError:
        logger.exception("[SkillDump] Permission denied")
        return _error(f"Permission denied: {target_path}")

    except Exception as e:
        logger.exception("[SkillDump] Write failed")
        return _error(f"Write operation failed: {str(e)}")


# ================================
# Helpers
# ================================

def _error(msg: str) -> Dict[str, Any]:
    return {
        "status": "error",
        "message": msg,
    }

