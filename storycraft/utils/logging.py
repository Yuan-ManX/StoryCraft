#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache, wraps
from pathlib import Path
from typing import Optional, Callable, Dict

import colorlog
from logging.handlers import RotatingFileHandler
from contextlib import contextmanager
from proglog import TqdmProgressBarLogger


# -------------------------------------------------------------------
# Logger Config
# -------------------------------------------------------------------

@dataclass(frozen=True)
class LoggerConfig:
    level: str = "info"
    console: bool = True
    file: bool = False
    log_dir: str = "logs"
    max_bytes: int = 10 * 1024 * 1024
    backup_count: int = 5
    date_format: str = "%Y-%m-%d %H:%M:%S"


DEFAULT_CONFIG = LoggerConfig()

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s (%(filename)s:%(lineno)d)"

LOG_COLOR_MAP = {
    "DEBUG": "cyan",
    "INFO": "green",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "bold_red",
}

LOG_LEVEL_MAP = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

# -------------------------------------------------------------------
# Core Logger Factory
# -------------------------------------------------------------------

@lru_cache(maxsize=256)
def get_logger(
    name: Optional[str] = None,
    config: LoggerConfig = DEFAULT_CONFIG,
) -> logging.Logger:
    """
    StoryCraft unified logger factory.

    Design:
        - Singleton per name
        - Deterministic configuration
        - Zero handler duplication
    """
    if name is None:
        frame = sys._getframe(1)
        name = frame.f_globals.get("__name__", "__main__")

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    level = LOG_LEVEL_MAP.get(config.level.lower(), logging.INFO)
    logger.setLevel(level)
    logger.propagate = False

    # Console handler
    if config.console:
        ch = colorlog.StreamHandler()
        ch.setLevel(level)
        formatter = colorlog.ColoredFormatter(
            f"%(log_color)s{LOG_FORMAT}",
            datefmt=config.date_format,
            log_colors=LOG_COLOR_MAP,
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # File handler
    if config.file:
        log_dir = Path(config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"{datetime.now():%Y-%m-%d}.log"

        fh = RotatingFileHandler(
            log_file,
            maxBytes=config.max_bytes,
            backupCount=config.backup_count,
            encoding="utf-8",
        )
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=config.date_format))
        logger.addHandler(fh)

    return logger

# -------------------------------------------------------------------
# Logging Utilities
# -------------------------------------------------------------------

@contextmanager
def silence_logging():
    """
    Temporarily suppress all logging.
    """
    logging.disable(logging.CRITICAL)
    try:
        yield
    finally:
        logging.disable(logging.NOTSET)


# -------------------------------------------------------------------
# Decorators
# -------------------------------------------------------------------

def log_exception(
    func: Optional[Callable] = None,
    *,
    logger: Optional[logging.Logger] = None,
    level: int = logging.ERROR,
):
    """
    Auto-capture exception and log stacktrace.
    """

    def decorator(fn):
        nonlocal logger
        logger = logger or get_logger(fn.__module__)

        @wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception:
                logger.log(level, f"[Exception] {fn.__name__}", exc_info=True)
                raise

        return wrapper

    return decorator(func) if func else decorator


def log_time(
    func: Optional[Callable] = None,
    *,
    logger: Optional[logging.Logger] = None,
    level: int = logging.DEBUG,
    threshold: Optional[float] = None,
):
    """
    Measure execution time, optional threshold filter.
    """

    def decorator(fn):
        nonlocal logger
        logger = logger or get_logger(fn.__module__)

        @wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = fn(*args, **kwargs)
            elapsed = time.perf_counter() - start

            if threshold is None or elapsed >= threshold:
                logger.log(level, f"[Timing] {fn.__name__}: {elapsed:.4f}s")

            return result

        return wrapper

    return decorator(func) if func else decorator


# -------------------------------------------------------------------
# Progress Logger (MoviePy / Video / Render)
# -------------------------------------------------------------------

class MCPMoviePyLogger(TqdmProgressBarLogger):
    """
    Unified progress logger for MoviePy / Render tasks.
    """

    def __init__(self, report: Callable[[float, float, str], None]):
        super().__init__(logged_bars="all", leave_bars=False, print_messages=True)
        self._report = report
        self._last_ts = 0.0
        self._last_progress = -1.0

    def bars_callback(self, bar, attr, value, old_value=None):
        super().bars_callback(bar, attr, value, old_value)

        if bar not in {"frame_index", "t", "chunk"} or attr != "index":
            return

        state = self.bars.get(bar) or {}
        idx, total = state.get("index"), state.get("total")
        if idx is None or not total:
            return

        progress = max(0.0, min(1.0, float(idx) / float(total)))
        now = time.monotonic()

        if progress < 1.0 and (now - self._last_ts) < 0.15 and (progress - self._last_progress) < 0.002:
            return

        self._last_ts = now
        self._last_progress = progress
        self._report(float(idx), float(total), f"rendering {progress*100:.1f}%")

  
