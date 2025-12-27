"""Evaluation modules for defender-api-tool."""

from typing import Any

__all__ = ["DefenderToolEvaluator", "GraphToolEvaluator", "EvaluationMetrics"]


def __getattr__(name: str) -> Any:
    """Lazy import for torch-dependent modules."""
    if name in ("DefenderToolEvaluator", "GraphToolEvaluator", "EvaluationMetrics"):
        from defender_api_tool.evaluation.evaluator import (
            DefenderToolEvaluator,
            EvaluationMetrics,
        )

        if name == "EvaluationMetrics":
            return EvaluationMetrics
        return DefenderToolEvaluator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
