"""
msgraph-tool-agent-8b - Enterprise-Grade Tool-Calling Agent for Microsoft Graph API.

A specialized LLM fine-tuning pipeline that trains models to convert natural language
into precise, schema-validated JSON tool calls for the Microsoft Graph API.

Example:
    >>> from msgraph_tool_agent_8b import MSGraphAgent
    >>> agent = MSGraphAgent.from_pretrained("ftrout/msgraph-tool-agent-8b")
    >>> result = agent.generate("Send an email to john@contoso.com about the meeting")
    >>> print(result)
    {"name": "me_sendMail", "arguments": {"subject": "...", "toRecipients": [...]}}
"""

__version__ = "1.0.0"
__author__ = "ftrout"
__license__ = "MIT"


def __getattr__(name: str):
    """Lazy import for optional dependencies."""
    if name == "MSGraphAgent":
        from msgraph_tool_agent_8b.inference.agent import MSGraphAgent

        return MSGraphAgent
    elif name == "GraphAPIHarvester":
        from msgraph_tool_agent_8b.data.harvester import GraphAPIHarvester

        return GraphAPIHarvester
    elif name == "GraphToolTrainer":
        from msgraph_tool_agent_8b.training.trainer import GraphToolTrainer

        return GraphToolTrainer
    elif name == "GraphToolEvaluator":
        from msgraph_tool_agent_8b.evaluation.evaluator import GraphToolEvaluator

        return GraphToolEvaluator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MSGraphAgent",
    "GraphAPIHarvester",
    "GraphToolTrainer",
    "GraphToolEvaluator",
    "__version__",
]
