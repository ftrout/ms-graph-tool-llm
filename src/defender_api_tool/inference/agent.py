"""
Defender API Agent for inference.

Provides a simple interface for loading and using trained models
for Microsoft Defender XDR API tool calling in security operations.
"""

import json
import os
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from defender_api_tool.utils.logging import get_logger

logger = get_logger("inference.agent")

# System prompt for security operations
SYSTEM_PROMPT = (
    "You are a Security Operations AI Agent for Microsoft Defender XDR. "
    "Given the user request and the available tool definition, generate the "
    "correct JSON tool call for security analysis, incident response, or threat hunting."
)


class DefenderApiAgent:
    """
    Microsoft Defender XDR API tool-calling agent.

    Provides a simple interface for generating security-focused API tool calls
    from natural language instructions for SOC operations.

    Attributes:
        model: The loaded language model
        tokenizer: The model tokenizer
        device: Inference device

    Example:
        >>> agent = DefenderApiAgent.from_pretrained("./defender-api-tool")
        >>> result = agent.generate(
        ...     "Get all high severity security alerts",
        ...     tool=alert_tool_definition
        ... )
        >>> print(result)
        {"name": "security_alerts_list", "arguments": {...}}
    """

    def __init__(self, model: Any, tokenizer: Any, device: str = "cuda"):
        """
        Initialize the agent.

        Args:
            model: Loaded language model
            tokenizer: Model tokenizer
            device: Device for inference
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    @classmethod
    def from_pretrained(
        cls,
        adapter_path: str,
        base_model_id: str = "NousResearch/Hermes-3-Llama-3.1-8B",
        load_in_4bit: bool = False,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "DefenderApiAgent":
        """
        Load a trained agent from a saved adapter.

        Args:
            adapter_path: Path to the LoRA adapter directory
            base_model_id: Base model identifier
            load_in_4bit: Whether to use 4-bit quantization (disabled by default
                for inference to allow adapter merging)
            device_map: Device mapping strategy
            torch_dtype: Torch data type

        Returns:
            Loaded DefenderApiAgent instance

        Raises:
            FileNotFoundError: If adapter path doesn't exist
        """
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(
                f"Adapter not found at {adapter_path}. "
                "Please train the model first or provide correct path."
            )

        logger.info("Loading base model: %s", base_model_id)

        # Load base model in bfloat16 (not quantized) to allow adapter merging
        # 4-bit quantized models cannot have adapters merged into them
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            load_in_4bit=load_in_4bit,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )

        # Load adapter
        logger.info("Loading adapter from: %s", adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path)

        # Merge LoRA adapter into base model and unload adapter wrapper
        # This creates a standalone model with fine-tuned weights
        logger.info("Merging adapter weights into base model...")
        model = model.merge_and_unload()

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Agent loaded successfully on %s", device)

        return cls(model=model, tokenizer=tokenizer, device=device)

    @classmethod
    def from_hub(
        cls,
        repo_id: str,
        base_model_id: str = "NousResearch/Hermes-3-Llama-3.1-8B",
        revision: str = "main",
        **kwargs: Any,
    ) -> "DefenderApiAgent":
        """
        Load a trained agent from Hugging Face Hub.

        Args:
            repo_id: Repository ID on Hugging Face Hub
            base_model_id: Base model identifier
            revision: Git revision to use
            **kwargs: Additional arguments for from_pretrained

        Returns:
            Loaded DefenderApiAgent instance
        """
        from huggingface_hub import snapshot_download

        logger.info("Downloading adapter from Hub: %s", repo_id)
        adapter_path = snapshot_download(repo_id=repo_id, revision=revision)

        return cls.from_pretrained(
            adapter_path=adapter_path, base_model_id=base_model_id, **kwargs
        )

    def generate(
        self,
        instruction: str,
        tool: dict[str, Any] | str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0.1,
        return_dict: bool = True,
    ) -> dict[str, Any] | str:
        """
        Generate a tool call for the given instruction.

        Args:
            instruction: Natural language instruction
            tool: Single tool definition (dict or JSON string)
            tools: List of tool definitions (for multi-tool selection)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 for deterministic)
            return_dict: Whether to return parsed dict or raw string

        Returns:
            Generated tool call as dict or string

        Raises:
            ValueError: If neither tool nor tools is provided
        """
        if tool is None and tools is None:
            raise ValueError("Either 'tool' or 'tools' must be provided")

        # Format tool definition
        if tool is not None:
            tool_str = json.dumps(tool) if isinstance(tool, dict) else tool
        else:
            tool_str = json.dumps(tools)

        # Create messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"User Request: {instruction}\nAvailable Tool: {tool_str}",
            },
        ]

        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only new tokens
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        ).strip()

        # Parse if requested
        if return_dict:
            try:
                parsed: dict[str, Any] = json.loads(response)
                return parsed
            except json.JSONDecodeError:
                logger.warning("Failed to parse response as JSON, returning raw string")
                return str(response)

        return str(response)

    def __call__(
        self,
        instruction: str,
        tool: dict[str, Any] | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | str:
        """
        Shorthand for generate().

        Args:
            instruction: Natural language instruction
            tool: Tool definition
            **kwargs: Additional arguments for generate()

        Returns:
            Generated tool call
        """
        return self.generate(instruction, tool=tool, **kwargs)


# Backward compatibility alias
MSGraphAgent = DefenderApiAgent


# Security-focused tool definitions for Microsoft Defender XDR
COMMON_TOOLS = {
    "list_alerts": {
        "type": "function",
        "function": {
            "name": "security_alerts_list",
            "description": "List security alerts from Microsoft Defender XDR.",
            "parameters": {
                "type": "object",
                "properties": {
                    "$filter": {
                        "type": "string",
                        "description": "OData filter (e.g., severity eq 'high')",
                    },
                    "$select": {
                        "type": "string",
                        "description": "Properties to select",
                    },
                    "$top": {"type": "integer", "description": "Number of results"},
                    "$orderby": {
                        "type": "string",
                        "description": "Order by property",
                    },
                },
            },
        },
    },
    "get_alert": {
        "type": "function",
        "function": {
            "name": "security_alerts_get",
            "description": "Get a specific security alert by ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "alertId": {
                        "type": "string",
                        "description": "The unique alert identifier",
                    },
                },
                "required": ["alertId"],
            },
        },
    },
    "update_alert": {
        "type": "function",
        "function": {
            "name": "security_alerts_update",
            "description": "Update a security alert (assign, change status, add comments).",
            "parameters": {
                "type": "object",
                "properties": {
                    "alertId": {
                        "type": "string",
                        "description": "The unique alert identifier",
                    },
                    "status": {
                        "type": "string",
                        "description": "New status (new, inProgress, resolved)",
                    },
                    "assignedTo": {
                        "type": "string",
                        "description": "Analyst to assign the alert to",
                    },
                    "classification": {
                        "type": "string",
                        "description": "Classification (truePositive, falsePositive, benignPositive)",
                    },
                    "comments": {
                        "type": "string",
                        "description": "Comments to add to the alert",
                    },
                },
                "required": ["alertId"],
            },
        },
    },
    "list_incidents": {
        "type": "function",
        "function": {
            "name": "security_incidents_list",
            "description": "List security incidents from Microsoft Defender XDR.",
            "parameters": {
                "type": "object",
                "properties": {
                    "$filter": {
                        "type": "string",
                        "description": "OData filter (e.g., severity eq 'high')",
                    },
                    "$top": {"type": "integer", "description": "Number of results"},
                    "$orderby": {
                        "type": "string",
                        "description": "Order by property",
                    },
                },
            },
        },
    },
    "get_incident": {
        "type": "function",
        "function": {
            "name": "security_incidents_get",
            "description": "Get a specific security incident with all related alerts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "incidentId": {
                        "type": "string",
                        "description": "The unique incident identifier",
                    },
                    "$expand": {
                        "type": "string",
                        "description": "Related entities to expand (e.g., alerts)",
                    },
                },
                "required": ["incidentId"],
            },
        },
    },
    "run_hunting_query": {
        "type": "function",
        "function": {
            "name": "security_runHuntingQuery",
            "description": "Run an advanced hunting query using Kusto Query Language (KQL).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "KQL query to execute (e.g., DeviceProcessEvents | take 100)",
                    },
                    "timespan": {
                        "type": "string",
                        "description": "Time range for the query (e.g., P7D for 7 days)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    "list_secure_scores": {
        "type": "function",
        "function": {
            "name": "security_secureScores_list",
            "description": "Get the organization's security secure scores.",
            "parameters": {
                "type": "object",
                "properties": {
                    "$top": {"type": "integer", "description": "Number of results"},
                },
            },
        },
    },
    "list_risky_users": {
        "type": "function",
        "function": {
            "name": "riskyUsers_list",
            "description": "List users flagged as risky by Identity Protection.",
            "parameters": {
                "type": "object",
                "properties": {
                    "$filter": {
                        "type": "string",
                        "description": "OData filter (e.g., riskLevel eq 'high')",
                    },
                    "$top": {"type": "integer", "description": "Number of results"},
                },
            },
        },
    },
    "list_sign_ins": {
        "type": "function",
        "function": {
            "name": "signIns_list",
            "description": "Get sign-in logs for security analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "$filter": {
                        "type": "string",
                        "description": "OData filter for sign-ins",
                    },
                    "$top": {"type": "integer", "description": "Number of results"},
                    "$orderby": {
                        "type": "string",
                        "description": "Order by property",
                    },
                },
            },
        },
    },
    "list_ti_indicators": {
        "type": "function",
        "function": {
            "name": "security_tiIndicators_list",
            "description": "List threat intelligence indicators.",
            "parameters": {
                "type": "object",
                "properties": {
                    "$filter": {
                        "type": "string",
                        "description": "OData filter for indicators",
                    },
                    "$top": {"type": "integer", "description": "Number of results"},
                },
            },
        },
    },
}


def main() -> None:
    """Interactive CLI for the Defender API Agent."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Interactive Defender XDR Security Tool Agent"
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default="defender-api-tool",
        help="Path to trained LoRA adapter",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="NousResearch/Hermes-3-Llama-3.1-8B",
        help="Base model identifier",
    )

    args = parser.parse_args()

    from defender_api_tool.utils.logging import setup_logging

    setup_logging()

    # Load agent
    try:
        agent = DefenderApiAgent.from_pretrained(
            adapter_path=args.adapter_path, base_model_id=args.base_model
        )
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    print("\n" + "=" * 60)
    print("DEFENDER XDR SECURITY AGENT - Interactive Mode")
    print("=" * 60)
    print(
        "\nAvailable tools: list_alerts, get_alert, update_alert, "
        "list_incidents, get_incident, run_hunting_query, "
        "list_secure_scores, list_risky_users, list_sign_ins, list_ti_indicators"
    )
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            user_input = input("Security Request > ").strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            if not user_input:
                continue

            # Security-focused tool selection based on keywords
            if any(
                kw in user_input.lower()
                for kw in ["alert", "alerts", "warning", "detection"]
            ):
                if "update" in user_input.lower() or "assign" in user_input.lower():
                    tool = COMMON_TOOLS["update_alert"]
                elif "get" in user_input.lower() or "specific" in user_input.lower():
                    tool = COMMON_TOOLS["get_alert"]
                else:
                    tool = COMMON_TOOLS["list_alerts"]
            elif any(
                kw in user_input.lower()
                for kw in ["incident", "incidents", "breach", "attack"]
            ):
                if "get" in user_input.lower() or "specific" in user_input.lower():
                    tool = COMMON_TOOLS["get_incident"]
                else:
                    tool = COMMON_TOOLS["list_incidents"]
            elif any(
                kw in user_input.lower() for kw in ["hunt", "hunting", "query", "kql"]
            ):
                tool = COMMON_TOOLS["run_hunting_query"]
            elif any(kw in user_input.lower() for kw in ["score", "secure", "posture"]):
                tool = COMMON_TOOLS["list_secure_scores"]
            elif any(
                kw in user_input.lower() for kw in ["risky", "risk", "compromised"]
            ):
                tool = COMMON_TOOLS["list_risky_users"]
            elif any(
                kw in user_input.lower()
                for kw in ["sign", "login", "authentication", "logon"]
            ):
                tool = COMMON_TOOLS["list_sign_ins"]
            elif any(
                kw in user_input.lower()
                for kw in ["threat", "ioc", "indicator", "intelligence"]
            ):
                tool = COMMON_TOOLS["list_ti_indicators"]
            else:
                tool = COMMON_TOOLS["list_alerts"]  # Default to alerts

            result = agent.generate(user_input, tool=tool)

            if isinstance(result, dict):
                print(f"\nAgent > {json.dumps(result, indent=2)}\n")
            else:
                print(f"\nAgent > {result}\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.error("Error: %s", e)


if __name__ == "__main__":
    main()
