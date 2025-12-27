"""Training modules for defender-api-tool."""

from typing import Any

__all__ = ["DefenderToolTrainer", "GraphToolTrainer"]


def __getattr__(name: str) -> Any:
    """Lazy import for torch-dependent modules."""
    if name in ("DefenderToolTrainer", "GraphToolTrainer"):
        from defender_api_tool.training.trainer import DefenderToolTrainer

        return DefenderToolTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
