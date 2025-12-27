"""
defender-api-tool - Enterprise-Grade Security Tool-Calling Agent for Microsoft Defender XDR.

A specialized LLM fine-tuning pipeline that trains models to convert natural language
into precise, schema-validated JSON tool calls for the Microsoft Defender XDR API.
Designed for Security Operations Centers (SOC) and incident response workflows.

Example:
    >>> from defender_api_tool import DefenderApiAgent
    >>> agent = DefenderApiAgent.from_pretrained("fmt0816/securitygraph-agent")
    >>> result = agent.generate("Get all high severity security alerts from the last 24 hours")
    >>> print(result)
    {"name": "security_alerts_list", "arguments": {"$filter": "severity eq 'high'", ...}}
"""

from typing import Any

__version__ = "1.0.0"
__author__ = "ftrout"
__license__ = "MIT"


def __getattr__(name: str) -> Any:
    """Lazy import for optional dependencies."""
    if name == "DefenderApiAgent":
        from defender_api_tool.inference.agent import DefenderApiAgent

        return DefenderApiAgent
    elif name == "DefenderAPIHarvester":
        from defender_api_tool.data.harvester import DefenderAPIHarvester

        return DefenderAPIHarvester
    elif name == "DefenderToolTrainer":
        from defender_api_tool.training.trainer import DefenderToolTrainer

        return DefenderToolTrainer
    elif name == "DefenderToolEvaluator":
        from defender_api_tool.evaluation.evaluator import DefenderToolEvaluator

        return DefenderToolEvaluator
    # Backward compatibility aliases
    elif name == "MSGraphAgent":
        from defender_api_tool.inference.agent import DefenderApiAgent

        return DefenderApiAgent
    elif name == "GraphAPIHarvester":
        from defender_api_tool.data.harvester import DefenderAPIHarvester

        return DefenderAPIHarvester
    elif name == "GraphToolTrainer":
        from defender_api_tool.training.trainer import DefenderToolTrainer

        return DefenderToolTrainer
    elif name == "GraphToolEvaluator":
        from defender_api_tool.evaluation.evaluator import DefenderToolEvaluator

        return DefenderToolEvaluator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # New names
    "DefenderApiAgent",
    "DefenderAPIHarvester",
    "DefenderToolTrainer",
    "DefenderToolEvaluator",
    # Backward compatibility
    "MSGraphAgent",
    "GraphAPIHarvester",
    "GraphToolTrainer",
    "GraphToolEvaluator",
    "__version__",
]
