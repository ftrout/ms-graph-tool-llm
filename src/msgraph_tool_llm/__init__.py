"""
MSGraph Tool LLM - Enterprise-Grade Tool-Calling Agent for Microsoft Graph API.

A specialized LLM fine-tuning pipeline that trains models to convert natural language
into precise, schema-validated JSON tool calls for the Microsoft Graph API.

Example:
    >>> from msgraph_tool_llm import MSGraphAgent
    >>> agent = MSGraphAgent.from_pretrained("ftrout/msgraph-tool-agent-7b")
    >>> result = agent.generate("Send an email to john@contoso.com about the meeting")
    >>> print(result)
    {"name": "me_sendMail", "arguments": {"subject": "...", "toRecipients": [...]}}
"""

__version__ = "1.0.0"
__author__ = "ftrout"
__license__ = "MIT"

from msgraph_tool_llm.inference.agent import MSGraphAgent
from msgraph_tool_llm.data.harvester import GraphAPIHarvester
from msgraph_tool_llm.training.trainer import GraphToolTrainer
from msgraph_tool_llm.evaluation.evaluator import GraphToolEvaluator

__all__ = [
    "MSGraphAgent",
    "GraphAPIHarvester",
    "GraphToolTrainer",
    "GraphToolEvaluator",
    "__version__",
]
