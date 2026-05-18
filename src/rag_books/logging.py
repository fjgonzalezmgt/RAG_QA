"""Loguru setup for console and file logging."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


_CONFIGURED = False


def setup_logging(project_root: Path, level: str = "INFO") -> None:
    """Configure application logging once.

    Parameters
    ----------
    project_root
        Project folder where the ``logs`` directory will be created.
    level
        Console log level.
    """

    global _CONFIGURED
    if _CONFIGURED:
        return

    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        colorize=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
    )
    logger.add(
        log_dir / "rag_books.log",
        level="DEBUG",
        rotation="5 MB",
        retention="10 days",
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    )

    _CONFIGURED = True
    logger.info("Logging initialized. Log file: {}", log_dir / "rag_books.log")
