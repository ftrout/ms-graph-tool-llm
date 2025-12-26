"""
Gradio demo for MSGraph Tool Agent.

Run with:
    python demo.py                              # Load from local ./msgraph-tool-agent-8b
    python demo.py --adapter-path ./my-adapter  # Load from custom local path
    python demo.py --from-hub username/model    # Load from Hugging Face Hub
    python demo.py --share                      # Create public share link
"""

import argparse
import json
import os
import sys

# Check for gradio before importing
try:
    import gradio as gr
except ImportError:
    print("Gradio is not installed. Install with: pip install 'msgraph-tool-agent-8b[demo]'")
    sys.exit(1)

from msgraph_tool_agent_8b.inference.agent import COMMON_TOOLS, MSGraphAgent
from msgraph_tool_agent_8b.utils.logging import setup_logging

# Global agent instance
agent: MSGraphAgent | None = None


def load_agent(adapter_path: str | None = None, from_hub: str | None = None) -> MSGraphAgent:
    """Load the MSGraph agent from local path or Hugging Face Hub."""
    setup_logging()

    if from_hub:
        print(f"Loading model from Hugging Face Hub: {from_hub}")
        return MSGraphAgent.from_hub(from_hub)
    elif adapter_path:
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(
                f"Adapter not found at '{adapter_path}'. "
                "Train a model first with 'msgraph-train' or specify --from-hub to load from Hub."
            )
        print(f"Loading model from: {adapter_path}")
        return MSGraphAgent.from_pretrained(adapter_path)
    else:
        # Default local path
        default_path = "./msgraph-tool-agent-8b"
        if os.path.exists(default_path):
            print(f"Loading model from: {default_path}")
            return MSGraphAgent.from_pretrained(default_path)
        else:
            raise FileNotFoundError(
                f"No adapter found at default path '{default_path}'. "
                "Use --adapter-path to specify a custom path or --from-hub to load from Hub."
            )


def generate_tool_call(instruction: str, tool_type: str) -> str:
    """Generate a tool call from natural language instruction."""
    global agent

    if agent is None:
        return "Error: Model not loaded. Please restart the demo."

    if not instruction.strip():
        return "Please enter an instruction."

    tool = COMMON_TOOLS.get(tool_type)
    if not tool:
        return f"Unknown tool type: {tool_type}"

    try:
        result = agent.generate(instruction, tool=tool)
        if isinstance(result, dict):
            return json.dumps(result, indent=2)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


# Example prompts for each tool
EXAMPLES = [
    ["Send an email to john@contoso.com about the quarterly report", "send_mail"],
    ["Find all users in the marketing department", "list_users"],
    ["Show me my calendar events for next week", "list_events"],
    ["List all files in my OneDrive root folder", "get_drive_items"],
    ["Email the team about tomorrow's standup meeting", "send_mail"],
    ["Get the top 10 users sorted by display name", "list_users"],
]


def create_demo() -> gr.Interface:
    """Create and return the Gradio interface."""
    return gr.Interface(
        fn=generate_tool_call,
        inputs=[
            gr.Textbox(
                label="Instruction",
                placeholder="Enter a natural language request...",
                lines=2,
            ),
            gr.Dropdown(
                choices=list(COMMON_TOOLS.keys()),
                value="send_mail",
                label="Tool Type",
            ),
        ],
        outputs=gr.Code(label="Generated Tool Call", language="json"),
        title="MSGraph Tool Agent Demo",
        description=(
            "Convert natural language requests into Microsoft Graph API tool calls. "
            "Select a tool type and enter your request in plain English."
        ),
        examples=EXAMPLES,
        theme=gr.themes.Soft(),
        article=(
            "### About\n"
            "This demo uses a fine-tuned language model to generate JSON tool calls "
            "for the Microsoft Graph API. The model was trained using QLoRA on the "
            "Hermes 3 Llama 3.1 8B base model.\n\n"
            "**Note**: Generated tool calls should always be validated before execution."
        ),
    )


def main() -> None:
    """CLI entry point for the Gradio demo."""
    global agent

    parser = argparse.ArgumentParser(
        description="Launch the MSGraph Tool Agent Gradio demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo.py                              # Load from ./msgraph-tool-agent-8b
  python demo.py --adapter-path ./my-model    # Load from custom path
  python demo.py --from-hub ftrout/msgraph-tool-agent-8b  # Load from Hub
  python demo.py --share                      # Create public share link
        """,
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Path to the trained LoRA adapter directory",
    )
    parser.add_argument(
        "--from-hub",
        type=str,
        default=None,
        help="Load adapter from Hugging Face Hub (e.g., 'username/model-name')",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public share link for the demo",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=7860,
        help="Port to run the server on (default: 7860)",
    )
    parser.add_argument(
        "--server-name",
        type=str,
        default="127.0.0.1",
        help="Server name/IP to bind to (default: 127.0.0.1)",
    )

    args = parser.parse_args()

    # Load the agent
    try:
        agent = load_agent(adapter_path=args.adapter_path, from_hub=args.from_hub)
        print("Model loaded successfully!")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)

    # Create and launch demo
    demo = create_demo()
    print(f"\nLaunching demo on http://{args.server_name}:{args.server_port}")

    demo.launch(
        share=args.share,
        server_name=args.server_name,
        server_port=args.server_port,
    )


if __name__ == "__main__":
    main()
