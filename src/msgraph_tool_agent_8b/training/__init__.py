"""Training modules for msgraph-tool-agent-8b."""

from typing import Any

__all__ = ["GraphToolTrainer"]


def __getattr__(name: str) -> Any:
    """Lazy import for torch-dependent modules."""
    if name == "GraphToolTrainer":
        from msgraph_tool_agent_8b.training.trainer import GraphToolTrainer

        return GraphToolTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
