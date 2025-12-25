"""Logging configuration for MSGraph Tool LLM."""

import logging
import sys
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Configure logging for the MSGraph Tool LLM package.

    Args:
        level: Logging level (default: INFO)
        log_file: Optional file path to write logs
        format_string: Custom format string for log messages

    Returns:
        Configured root logger
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format=format_string,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True
    )

    return logging.getLogger("msgraph_tool_llm")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    Args:
        name: Module name for the logger

    Returns:
        Logger instance
    """
    return logging.getLogger(f"msgraph_tool_llm.{name}")
