"""
Gradio demo for MSGraph Tool Agent.

Run with: python demo.py
"""

import json
import gradio as gr

from msgraph_tool_agent_8b.inference.agent import MSGraphAgent, COMMON_TOOLS
from msgraph_tool_agent_8b.utils.logging import setup_logging

setup_logging()

# Load agent globally
print("Loading model... (this may take a minute)")
agent = MSGraphAgent.from_pretrained("./msgraph-tool-agent-8b")
print("Model loaded!")


def generate_tool_call(instruction: str, tool_type: str) -> str:
    """Generate a tool call from natural language instruction."""
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

# Build the Gradio interface
demo = gr.Interface(
    fn=generate_tool_call,
    inputs=[
        gr.Textbox(
            label="Instruction",
            placeholder="Enter a natural language request...",
            lines=2,
        ),
        gr.Dropdown(
            choices=list(COMMON_TOOLS.keys()), value="send_mail", label="Tool Type"
        ),
    ],
    outputs=gr.Code(label="Generated Tool Call", language="json"),
    title="MSGraph Tool Agent Demo",
    description="Convert natural language requests into Microsoft Graph API tool calls.",
    examples=EXAMPLES,
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch(share=False)
