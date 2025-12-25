"""Evaluation modules for msgraph-tool-agent-8b."""

from typing import Any

__all__ = ["GraphToolEvaluator", "EvaluationMetrics"]


def __getattr__(name: str) -> Any:
    """Lazy import for torch-dependent modules."""
    if name in ("GraphToolEvaluator", "EvaluationMetrics"):
        from msgraph_tool_agent_8b.evaluation.evaluator import (
            EvaluationMetrics,
            GraphToolEvaluator,
        )

        if name == "GraphToolEvaluator":
            return GraphToolEvaluator
        return EvaluationMetrics
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
