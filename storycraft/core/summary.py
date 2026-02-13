from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Literal
from datetime import datetime
import logging
import uuid

from storycraft.utils.logging import get_logger


# ============================================================
# 🎛 Log Entry Schema
# ============================================================

LogLevel = Literal["ERROR", "WARNING", "INFO_LLM", "INFO_USER", "DEBUG", "TRACE"]


@dataclass
class LogEntry:
    """
    Atomic structured log entry.
    """
    level: LogLevel
    message: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    artifact_id: Optional[str] = None
    trace_id: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# 🎛 Node Observability Kernel
# ============================================================

@dataclass
class NodeSummary:
    """
    StoryCraft Agent Observability & Feedback Kernel.

    This component provides:
    - Hierarchical structured logging
    - Artifact-level trace mapping
    - LLM optimized context summaries
    - Runtime telemetry aggregation
    - Debug observability & replay support
    """

    # ------------------ Level Definitions ------------------

    ERROR: LogLevel = "ERROR"
    WARNING: LogLevel = "WARNING"
    INFO_LLM: LogLevel = "INFO_LLM"
    INFO_USER: LogLevel = "INFO_USER"
    DEBUG: LogLevel = "DEBUG"
    TRACE: LogLevel = "TRACE"

    LEVELS: Tuple[LogLevel, ...] = (
        ERROR,
        WARNING,
        INFO_LLM,
        INFO_USER,
        DEBUG,
        TRACE,
    )

    # ------------------ Runtime Storage ------------------

    logs: Dict[LogLevel, List[LogEntry]] = field(
        default_factory=lambda: {lvl: [] for lvl in NodeSummary.LEVELS}
    )

    # Artifact → warnings / errors
    artifact_errors: Dict[str, List[str]] = field(default_factory=dict)
    artifact_warnings: Dict[str, List[str]] = field(default_factory=dict)

    # ------------------ Configuration ------------------

    logger_name: str = "NodeSummary"
    auto_console: bool = True

    # Summary levels for final LLM feedback
    summary_levels: List[LogLevel] = field(
        default_factory=lambda: ["ERROR", "WARNING", "INFO_LLM", "INFO_USER"]
    )

    # ------------------ Internal ------------------

    _logger: Optional[logging.Logger] = field(default=None, init=False, repr=False)

    # ============================================================
    # Lifecycle
    # ============================================================

    def __post_init__(self):
        self._logger = get_logger(self.logger_name)

    # ============================================================
    # Core Logging API
    # ============================================================

    def log(
        self,
        level: LogLevel,
        message: str,
        *,
        artifact_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        **extra: Any,
    ):
        entry = LogEntry(
            level=level,
            message=message,
            artifact_id=artifact_id,
            trace_id=trace_id,
            extra=extra,
        )

        self.logs[level].append(entry)

        if artifact_id:
            if level == self.ERROR:
                self.artifact_errors.setdefault(artifact_id, []).append(message)
            elif level == self.WARNING:
                self.artifact_warnings.setdefault(artifact_id, []).append(message)

        if self.auto_console:
            self._console_output(entry)

    # ============================================================
    # Semantic Wrappers
    # ============================================================

    def error(self, message: str, **kwargs):
        self.log(self.ERROR, message, **kwargs)

    def warning(self, message: str, **kwargs):
        self.log(self.WARNING, message, **kwargs)

    def info_llm(self, message: str, **kwargs):
        self.log(self.INFO_LLM, message, **kwargs)

    def info_user(self, message: str, **kwargs):
        self.log(self.INFO_USER, message, **kwargs)

    def debug(self, message: str, **kwargs):
        self.log(self.DEBUG, message, **kwargs)

    def trace(self, message: str, **kwargs):
        self.log(self.TRACE, message, **kwargs)

    # ============================================================
    # Console Adapter
    # ============================================================

    def _console_output(self, entry: LogEntry):
        prefix = f"[ARTIFACT:{entry.artifact_id}] " if entry.artifact_id else ""
        prefix += f"[TRACE:{entry.trace_id}] " if entry.trace_id else ""

        level_map = {
            self.ERROR: logging.ERROR,
            self.WARNING: logging.WARNING,
            self.INFO_LLM: logging.INFO,
            self.INFO_USER: logging.INFO,
            self.DEBUG: logging.DEBUG,
            self.TRACE: logging.DEBUG,
        }

        self._logger.log(level_map[entry.level], f"{prefix}{entry.message}")

    # ============================================================
    # Structured Extraction
    # ============================================================

    def extract(self, level: LogLevel) -> Dict[str, Any]:
        entries = self.logs.get(level, [])
        if not entries:
            return {}

        return {
            "log_lines": "\n".join(self._format(e) for e in entries),
            "extra_data_list": [e.extra for e in entries],
        }

    def _format(self, entry: LogEntry) -> str:
        s = f"[{entry.timestamp}] {entry.message}"
        if entry.artifact_id:
            s += f" [artifact:{entry.artifact_id}]"
        if entry.trace_id:
            s += f" [trace:{entry.trace_id}]"
        return s

    # ============================================================
    # LLM Summary Export
    # ============================================================

    def summary(self, *, artifact_id: Optional[str] = None) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        preview_urls: List[str] = []

        for lvl in self.summary_levels:
            extracted = self.extract(lvl)
            result[lvl] = extracted.get("log_lines", "")

            preview_urls.extend(self._collect_preview_urls(extracted.get("extra_data_list", [])))

        if artifact_id:
            result["artifact_id"] = artifact_id

        result["preview_urls"] = preview_urls
        return result

    def _collect_preview_urls(self, extras: List[Dict[str, Any]]) -> List[str]:
        urls: List[str] = []
        for extra in extras:
            urls.extend(map(str, extra.get("preview_urls", [])))
        return urls

    # ============================================================
    # Maintenance
    # ============================================================

    def clear(self):
        for logs in self.logs.values():
            logs.clear()
        self.artifact_errors.clear()
        self.artifact_warnings.clear()

  
