"""Inference modules for defender-api-tool."""

from typing import Any

__all__ = ["DefenderApiAgent", "MSGraphAgent"]


def __getattr__(name: str) -> Any:
    """Lazy import for torch-dependent modules."""
    if name in ("DefenderApiAgent", "MSGraphAgent"):
        from defender_api_tool.inference.agent import DefenderApiAgent

        return DefenderApiAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
