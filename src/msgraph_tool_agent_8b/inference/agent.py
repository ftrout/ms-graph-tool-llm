"""
MSGraph Agent for inference.

Provides a simple interface for loading and using trained models
for Microsoft Graph API tool calling.
"""

import json
import os
from typing import Any, Dict, List, Optional, Union

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from msgraph_tool_agent_8b.utils.logging import get_logger

logger = get_logger("inference.agent")

# System prompt for the agent
SYSTEM_PROMPT = (
    "You are an AI Agent for Microsoft Graph. Given the user request and "
    "the available tool definition, generate the correct JSON tool call."
)


class MSGraphAgent:
    """
    Microsoft Graph API tool-calling agent.

    Provides a simple interface for generating Microsoft Graph API tool calls
    from natural language instructions.

    Attributes:
        model: The loaded language model
        tokenizer: The model tokenizer
        device: Inference device

    Example:
        >>> agent = MSGraphAgent.from_pretrained("./msgraph-tool-agent-7b")
        >>> result = agent.generate(
        ...     "Send an email to john@example.com",
        ...     tool=email_tool_definition
        ... )
        >>> print(result)
        {"name": "me_sendMail", "arguments": {...}}
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: str = "cuda"
    ):
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
        torch_dtype: torch.dtype = torch.bfloat16
    ) -> "MSGraphAgent":
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
            Loaded MSGraphAgent instance

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
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_id,
            trust_remote_code=True
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Agent loaded successfully on %s", device)

        return cls(model=model, tokenizer=tokenizer, device=device)

    @classmethod
    def from_hub(
        cls,
        repo_id: str,
        base_model_id: str = "NousResearch/Hermes-3-Llama-3.1-8B",
        revision: str = "main",
        **kwargs
    ) -> "MSGraphAgent":
        """
        Load a trained agent from Hugging Face Hub.

        Args:
            repo_id: Repository ID on Hugging Face Hub
            base_model_id: Base model identifier
            revision: Git revision to use
            **kwargs: Additional arguments for from_pretrained

        Returns:
            Loaded MSGraphAgent instance
        """
        from huggingface_hub import snapshot_download

        logger.info("Downloading adapter from Hub: %s", repo_id)
        adapter_path = snapshot_download(repo_id=repo_id, revision=revision)

        return cls.from_pretrained(
            adapter_path=adapter_path,
            base_model_id=base_model_id,
            **kwargs
        )

    def generate(
        self,
        instruction: str,
        tool: Optional[Union[Dict[str, Any], str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.1,
        return_dict: bool = True
    ) -> Union[Dict[str, Any], str]:
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
                "content": f"User Request: {instruction}\nAvailable Tool: {tool_str}"
            }
        ]

        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.device)

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
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()

        # Parse if requested
        if return_dict:
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                logger.warning("Failed to parse response as JSON, returning raw string")
                return response

        return response

    def __call__(
        self,
        instruction: str,
        tool: Optional[Union[Dict[str, Any], str]] = None,
        **kwargs
    ) -> Union[Dict[str, Any], str]:
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


# Common tool definitions for quick testing
COMMON_TOOLS = {
    "send_mail": {
        "type": "function",
        "function": {
            "name": "me_sendMail",
            "description": "Send a new message.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string", "description": "Email subject"},
                    "body": {"type": "string", "description": "Email body content"},
                    "toRecipients": {"type": "array", "description": "List of recipients"}
                },
                "required": ["subject", "toRecipients"]
            }
        }
    },
    "list_users": {
        "type": "function",
        "function": {
            "name": "users_ListUsers",
            "description": "Retrieve a list of user objects.",
            "parameters": {
                "type": "object",
                "properties": {
                    "$filter": {"type": "string", "description": "OData filter query"},
                    "$select": {"type": "string", "description": "Properties to select"},
                    "$top": {"type": "integer", "description": "Number of results"}
                }
            }
        }
    },
    "list_events": {
        "type": "function",
        "function": {
            "name": "me_events_List",
            "description": "Get the user's calendar events.",
            "parameters": {
                "type": "object",
                "properties": {
                    "$filter": {"type": "string", "description": "Filter events"},
                    "$orderby": {"type": "string", "description": "Order events by"},
                    "$top": {"type": "integer", "description": "Number of events"}
                }
            }
        }
    },
    "get_drive_items": {
        "type": "function",
        "function": {
            "name": "me_drive_root_children",
            "description": "List items in the root of the user's drive.",
            "parameters": {
                "type": "object",
                "properties": {
                    "$filter": {"type": "string", "description": "Filter items"},
                    "$select": {"type": "string", "description": "Properties to select"}
                }
            }
        }
    }
}


def main():
    """Interactive CLI for the MSGraph Agent."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Interactive MSGraph Tool Agent"
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default="msgraph-tool-agent-8b",
        help="Path to trained LoRA adapter"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="NousResearch/Hermes-3-Llama-3.1-8B",
        help="Base model identifier"
    )

    args = parser.parse_args()

    from msgraph_tool_agent_8b.utils.logging import setup_logging
    setup_logging()

    # Load agent
    try:
        agent = MSGraphAgent.from_pretrained(
            adapter_path=args.adapter_path,
            base_model_id=args.base_model
        )
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    print("\n" + "="*60)
    print("MSGRAPH TOOL AGENT - Interactive Mode")
    print("="*60)
    print("\nAvailable tools: send_mail, list_users, list_events, get_drive_items")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            user_input = input("Request > ").strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            if not user_input:
                continue

            # Simple tool selection based on keywords
            if any(kw in user_input.lower() for kw in ["mail", "send", "email"]):
                tool = COMMON_TOOLS["send_mail"]
            elif any(kw in user_input.lower() for kw in ["user", "person", "people"]):
                tool = COMMON_TOOLS["list_users"]
            elif any(kw in user_input.lower() for kw in ["calendar", "event", "meeting"]):
                tool = COMMON_TOOLS["list_events"]
            elif any(kw in user_input.lower() for kw in ["drive", "file", "document"]):
                tool = COMMON_TOOLS["get_drive_items"]
            else:
                tool = COMMON_TOOLS["list_users"]  # Default

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
