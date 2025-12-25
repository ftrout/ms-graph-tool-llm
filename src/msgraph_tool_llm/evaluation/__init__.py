"""Evaluation modules for msgraph-tool-agent-8b."""

__all__ = ["GraphToolEvaluator", "EvaluationMetrics"]


def __getattr__(name: str):
    """Lazy import for torch-dependent modules."""
    if name in ("GraphToolEvaluator", "EvaluationMetrics"):
        from msgraph_tool_llm.evaluation.evaluator import (
            GraphToolEvaluator,
            EvaluationMetrics,
        )
        if name == "GraphToolEvaluator":
            return GraphToolEvaluator
        return EvaluationMetrics
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
