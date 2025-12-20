import torch
import json
import logging
import argparse
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

# --- DEFAULT CONFIGURATION ---
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_ADAPTER_PATH = "ms-graph-v1"

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

def evaluate(base_model: str, adapter_path: str, test_query: str) -> None:
    """
    Evaluates the fine-tuned MS Graph agent model on a test query.

    Args:
        base_model: Base model identifier (e.g., "Qwen/Qwen2.5-7B-Instruct")
        adapter_path: Path to the trained LoRA adapter directory
        test_query: Natural language test query to evaluate

    Loads the trained LoRA adapter, runs inference on a sample query,
    and displays the generated tool call output.

    Raises:
        Exception: If model loading fails (e.g., adapter not found)
    """
    logger.info("Loading adapter from %s...", adapter_path)

    # Detect available device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    try:
        base = AutoModelForCausalLM.from_pretrained(
            base_model, load_in_4bit=True, device_map="auto", torch_dtype=torch.bfloat16
        )
        model = PeftModel.from_pretrained(base, adapter_path)
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        logger.info("Model and adapter loaded successfully")
    except Exception as e:
        logger.error("Error loading model. Did you train first? %s", e)
        return

    messages = [
        {"role": "system", "content": "You are an AI Agent for Microsoft Graph. Given the user request and the available tool definition, generate the correct JSON tool call."},
        {"role": "user", "content": f"User Request: {test_query}\nAvailable Tool: {json.dumps(TEST_TOOL)}"}
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    logger.info("Generating response for test query: '%s'", test_query)
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

def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned MS Graph agent model"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help=f"Base model identifier (default: {DEFAULT_BASE_MODEL})"
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=DEFAULT_ADAPTER_PATH,
        help=f"Path to trained LoRA adapter (default: {DEFAULT_ADAPTER_PATH})"
    )
    parser.add_argument(
        "--query",
        type=str,
        default=TEST_QUERY,
        help=f"Test query to evaluate (default: {TEST_QUERY})"
    )

    args = parser.parse_args()

    logger.info("Configuration:")
    logger.info("  Base model: %s", args.base_model)
    logger.info("  Adapter path: %s", args.adapter_path)
    logger.info("  Test query: %s", args.query)

    evaluate(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        test_query=args.query
    )

if __name__ == "__main__":
    main()