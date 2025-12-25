"""Inference modules for msgraph-tool-agent-8b."""

__all__ = ["MSGraphAgent"]


def __getattr__(name: str):
    """Lazy import for torch-dependent modules."""
    if name == "MSGraphAgent":
        from msgraph_tool_agent_8b.inference.agent import MSGraphAgent

        return MSGraphAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
