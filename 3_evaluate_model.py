import torch
import json
import logging
from typing import Dict, Any
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
ADAPTER_PATH = "ms-graph-v1"  # Matches training output

TEST_TOOL = {
    "type": "function",
    "function": {
        "name": "users_ListUsers",
        "description": "Retrieve a list of user objects.",
        "parameters": {
            "type": "object",
            "properties": {
                "$filter": {"type": "string", "description": "Filter items"},
                "$select": {"type": "string", "description": "Select properties"}
            }
        }
    }
}
TEST_QUERY = "Find the user with email 'admin@contoso.com' and select their id."

def evaluate() -> None:
    """
    Evaluates the fine-tuned MS Graph agent model on a test query.

    Loads the trained LoRA adapter, runs inference on a sample query,
    and displays the generated tool call output.

    Raises:
        Exception: If model loading fails (e.g., adapter not found)
    """
    logger.info("Loading adapter from %s...", ADAPTER_PATH)

    # Detect available device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    try:
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, load_in_4bit=True, device_map="auto", torch_dtype=torch.bfloat16
        )
        model = PeftModel.from_pretrained(base, ADAPTER_PATH)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        logger.info("Model and adapter loaded successfully")
    except Exception as e:
        logger.error("Error loading model. Did you train first? %s", e)
        return

    messages = [
        {"role": "system", "content": "You are an AI Agent for Microsoft Graph. Given the user request and the available tool definition, generate the correct JSON tool call."},
        {"role": "user", "content": f"User Request: {TEST_QUERY}\nAvailable Tool: {json.dumps(TEST_TOOL)}"}
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    logger.info("Generating response for test query: '%s'", TEST_QUERY)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.1)

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    # Validate JSON output
    try:
        parsed = json.loads(response)
        logger.info("✓ Valid JSON output generated")
        print(f"\nAgent Output:\n{json.dumps(parsed, indent=2)}\n")
    except json.JSONDecodeError as e:
        logger.warning("✗ Invalid JSON output: %s", e)
        print(f"\nAgent Output (raw):\n{response}\n")

if __name__ == "__main__":
    evaluate()