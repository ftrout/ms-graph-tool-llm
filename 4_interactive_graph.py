import torch
import json
import logging
from typing import Dict, Any, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# --- LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_PATH = "ms-graph-v1"

# Mock Tools for Demo
TOOLS = {
    "mail": {
        "name": "me_sendMail",
        "description": "Send a new message.",
        "parameters": {"type": "object", "properties": {"subject": {"type": "string"}, "toRecipients": {"type": "array"}}}
    },
    "calendar": {
        "name": "me_events_List",
        "description": "Retrieve the user's calendar events.",
        "parameters": {"type": "object", "properties": {"$filter": {"type": "string"}, "$orderby": {"type": "string"}}}
    }
}

def load_model() -> Tuple[Optional[Any], Optional[Any]]:
    """
    Loads the fine-tuned MS Graph agent model with LoRA adapter.

    Returns:
        Tuple[Optional[Any], Optional[Any]]: (model, tokenizer) if successful,
            (None, None) if loading fails

    Raises:
        Exception: Model loading errors are caught and logged
    """
    logger.info("Initializing MS Graph Agent...")
    try:
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, load_in_4bit=True, device_map="auto", torch_dtype=torch.bfloat16
        )
        model = PeftModel.from_pretrained(base, ADAPTER_PATH)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        logger.info("Agent initialized successfully")
        return model, tokenizer
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        return None, None

def run_inference(model: Any, tokenizer: Any, user_input: str, tool_def: Dict[str, Any]) -> str:
    """
    Runs inference to generate a tool call for the given user request.

    Args:
        model: Loaded PEFT model
        tokenizer: Model tokenizer
        user_input: Natural language user request
        tool_def: Tool definition dictionary with name, description, and parameters

    Returns:
        str: Generated JSON tool call as a string
    """
    # Detect available device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    messages = [
        {"role": "system", "content": "You are an AI Agent for Microsoft Graph. Given the user request and the available tool definition, generate the correct JSON tool call."},
        {"role": "user", "content": f"User Request: {user_input}\nAvailable Tool: {json.dumps(tool_def)}"}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.1)

    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

if __name__ == "__main__":
    model, tokenizer = load_model()
    if model:
        logger.info("Graph Agent ready for interactive queries")
        print("\n--- GRAPH AGENT READY (Type 'quit' to exit) ---")
        while True:
            user_input = input("\nRequest > ")
            if user_input.lower() in ["quit", "exit"]:
                logger.info("Exiting interactive session")
                break

            # Simple keyword-based routing for demo purposes
            # NOTE: In production, use RAG/embeddings for proper tool selection
            tool_key = "mail" if "mail" in user_input or "send" in user_input else "calendar"
            logger.debug("Selected tool: %s", tool_key)

            response = run_inference(model, tokenizer, user_input, TOOLS[tool_key])
            print(f"Agent > {response}")