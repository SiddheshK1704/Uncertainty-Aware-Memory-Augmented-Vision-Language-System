"""
Logger Utility
==============
Structured logging setup using loguru.
Provides colored console output and optional file logging.
"""

import sys
from pathlib import Path
from loguru import logger


def setup_logger(
    level: str = "INFO",
    log_file: str = None,
    rotation: str = "10 MB",
    retention: str = "7 days",
) -> None:
    """
    Configure loguru logger with console and optional file handler.

    Args:
        level:     Minimum log level ("DEBUG", "INFO", "WARNING", "ERROR")
        log_file:  Path to log file (None = console only)
        rotation:  File rotation size
        retention: How long to keep old log files
    """
    # Remove default handler
    logger.remove()

    # Console handler with colors
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # File handler (optional)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            level=level,
            rotation=rotation,
            retention=retention,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} — {message}",
            encoding="utf-8",
        )
        logger.info(f"Logging to file: {log_file}")


# Default: setup at INFO level on import
setup_logger(level="INFO")