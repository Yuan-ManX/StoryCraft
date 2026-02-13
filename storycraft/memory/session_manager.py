from __future__ import annotations

import shutil
import uuid
import time
import threading
from pathlib import Path
from typing import Callable, Optional

from storycraft.utils.logging import get_logger
from storycraft.memory.agent_memory import ArtifactStore


logger = get_logger(__name__)


class SessionLifecycleManager:
    """
    StoryCraft Session Runtime Lifecycle Kernel.

    Responsibilities:
        1. Session artifact lifecycle governance
        2. Runtime cache lifecycle governance
        3. Concurrent safe garbage collection
        4. ArtifactStore factory
    """

    def __init__(
        self,
        artifacts_root: str | Path,
        cache_root: str | Path,
        *,
        max_items: int = 256,
        retention_days: int = 3,
        enable_cleanup: bool = False,
    ):
        self.artifacts_root = Path(artifacts_root).expanduser().resolve()
        self.cache_root = Path(cache_root).expanduser().resolve()

        self.max_items = max_items
        self.retention_days = retention_days
        self.enable_cleanup = enable_cleanup

        self.artifacts_root.mkdir(parents=True, exist_ok=True)
        self.cache_root.mkdir(parents=True, exist_ok=True)

        self._gc_lock = threading.Lock()
        self._gc_running = False

        logger.info(
            f"[SessionLifecycle] init | "
            f"artifacts={self.artifacts_root}, "
            f"cache={self.cache_root}, "
            f"max_items={max_items}, retention_days={retention_days}"
        )

    # ============================================================
    # Public API
    # ============================================================

    def get_artifact_store(self, session_id: str) -> ArtifactStore:
        """
        Create artifact store for session and schedule GC.
        """
        if self.enable_cleanup:
            self._trigger_gc_async(session_id)

        return ArtifactStore(self.artifacts_root, session_id)

    def cleanup_expired_sessions(self, current_session_id: Optional[str] = None):
        """
        Global GC entrance.
        """
        if not self.enable_cleanup:
            return

        if not self._gc_lock.acquire(blocking=False):
            return

        try:
            self._gc_running = True
            self._gc_directory(
                self.artifacts_root,
                exclude=current_session_id,
                filter_func=self._is_valid_session_dir,
            )
            self._gc_directory(
                self.cache_root,
                exclude=current_session_id,
                filter_func=self._is_valid_session_dir,
            )
        finally:
            self._gc_running = False
            self._gc_lock.release()

    # ============================================================
    # GC Core
    # ============================================================

    def _trigger_gc_async(self, session_id: str):
        threading.Thread(
            target=self.cleanup_expired_sessions,
            args=(session_id,),
            daemon=True,
            name=f"SessionGC-{session_id[:8]}",
        ).start()

    def _gc_directory(
        self,
        root: Path,
        *,
        exclude: Optional[str] = None,
        filter_func: Optional[Callable[[Path], bool]] = None,
    ):
        """
        Two-phase GC:
            Phase-1: Time-based eviction
            Phase-2: Capacity-based eviction
        """
        if not root.exists():
            return

        now = time.time()
        expiry_cutoff = now - (self.retention_days * 86400)

        valid_items: list[Path] = []
        expired_items: list[Path] = []

        try:
            for p in root.iterdir():
                if filter_func and not filter_func(p):
                    continue

                if exclude and p.name == exclude:
                    continue

                try:
                    mtime = p.stat().st_mtime
                except Exception:
                    continue

                if mtime < expiry_cutoff:
                    expired_items.append(p)
                else:
                    valid_items.append(p)

            # Phase 1: Expiration cleanup
            for p in expired_items:
                logger.info(f"[SessionGC] expired → {p.name}")
                self._safe_remove(p)

            # Phase 2: Capacity enforcement
            if len(valid_items) > self.max_items:
                valid_items.sort(key=lambda x: x.stat().st_mtime)
                overflow = valid_items[: len(valid_items) - self.max_items]

                for p in overflow:
                    logger.info(f"[SessionGC] overflow → {p.name}")
                    self._safe_remove(p)

        except Exception as e:
            logger.error(f"[SessionGC] failure at {root}: {e}", exc_info=True)

    # ============================================================
    # Helpers
    # ============================================================

    def _safe_remove(self, path: Path):
        """
        Robust deletion handling readonly + lock scenarios.
        """
        def onerror(func, p, exc_info):
            import os, stat

            if not os.access(p, os.W_OK):
                os.chmod(p, stat.S_IWUSR)
                func(p)
            else:
                logger.warning(f"[SessionGC] delete failed: {p}")

        if path.is_dir():
            shutil.rmtree(path, onerror=onerror)
        else:
            path.unlink(missing_ok=True)

    @staticmethod
    def _is_valid_session_dir(p: Path) -> bool:
        """
        Only directories with valid UUID4 names are GC targets.
        """
        if not p.is_dir():
            return False

        name = p.name
        if len(name) != 32:
            return False

        try:
            val = uuid.UUID(name)
            return val.hex == name and val.version == 4
        except Exception:
            return False

      
