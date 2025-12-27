"""
Gradio demo for Defender API Tool Agent.

Run with:
    python demo.py                              # Load from local ./defender-api-tool
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
    print("Gradio is not installed. Install with: pip install 'defender-api-tool[demo]'")
    sys.exit(1)

from defender_api_tool.inference.agent import COMMON_TOOLS, DefenderApiAgent
from defender_api_tool.utils.logging import setup_logging

# Global agent instance
agent: DefenderApiAgent | None = None


def load_agent(
    adapter_path: str | None = None, from_hub: str | None = None
) -> DefenderApiAgent:
    """Load the Defender API agent from local path or Hugging Face Hub."""
    setup_logging()

    if from_hub:
        print(f"Loading model from Hugging Face Hub: {from_hub}")
        return DefenderApiAgent.from_hub(from_hub)
    elif adapter_path:
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(
                f"Adapter not found at '{adapter_path}'. "
                "Train a model first with 'defender-train' or specify --from-hub to load from Hub."
            )
        print(f"Loading model from: {adapter_path}")
        return DefenderApiAgent.from_pretrained(adapter_path)
    else:
        # Default local path
        default_path = "./defender-api-tool"
        if os.path.exists(default_path):
            print(f"Loading model from: {default_path}")
            return DefenderApiAgent.from_pretrained(default_path)
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
        return "Please enter a security-related instruction."

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


# Security-focused example prompts for each tool
EXAMPLES = [
    ["Get all high severity security alerts from the last 24 hours", "list_alerts"],
    ["Show me the details of alert ID da637551227677560813_-961444813", "get_alert"],
    ["Assign alert to analyst@contoso.com and mark as in progress", "update_alert"],
    ["List all active security incidents", "list_incidents"],
    [
        "Run a hunting query for PowerShell execution events in the last 7 days",
        "run_hunting_query",
    ],
    ["Get the organization's current security score", "list_secure_scores"],
    ["List all risky users flagged by Identity Protection", "list_risky_users"],
    ["Show failed sign-in attempts from the last hour", "list_sign_ins"],
    ["Get threat intelligence indicators for known malware", "list_ti_indicators"],
]


def create_demo() -> gr.Interface:
    """Create and return the Gradio interface."""
    return gr.Interface(
        fn=generate_tool_call,
        inputs=[
            gr.Textbox(
                label="Security Request",
                placeholder="Enter a security operations request in natural language...",
                lines=2,
            ),
            gr.Dropdown(
                choices=list(COMMON_TOOLS.keys()),
                value="list_alerts",
                label="Security Tool",
            ),
        ],
        outputs=gr.Code(label="Generated Tool Call", language="json"),
        title="Defender XDR Security Agent Demo",
        description=(
            "Convert natural language security requests into Microsoft Defender XDR API tool calls. "
            "Select a security tool and enter your request in plain English."
        ),
        examples=EXAMPLES,
        theme=gr.themes.Soft(),
        article=(
            "### About\n"
            "This demo uses a fine-tuned language model to generate JSON tool calls "
            "for the Microsoft Defender XDR API. The model was trained using QLoRA on the "
            "Hermes 3 Llama 3.1 8B base model, specialized for security operations.\n\n"
            "**Use Cases**: Alert triage, incident response, threat hunting, security posture assessment.\n\n"
            "**Note**: Generated tool calls should always be validated before execution in production."
        ),
    )


def main() -> None:
    """CLI entry point for the Gradio demo."""
    global agent

    parser = argparse.ArgumentParser(
        description="Launch the Defender XDR Security Agent Gradio demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo.py                              # Load from ./defender-api-tool
  python demo.py --adapter-path ./my-model    # Load from custom path
  python demo.py --from-hub ftrout/defender-api-tool  # Load from Hub
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
