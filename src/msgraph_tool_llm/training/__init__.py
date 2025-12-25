"""Training modules for msgraph-tool-agent-8b."""

__all__ = ["GraphToolTrainer"]


def __getattr__(name: str):
    """Lazy import for torch-dependent modules."""
    if name == "GraphToolTrainer":
        from msgraph_tool_llm.training.trainer import GraphToolTrainer
        return GraphToolTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
