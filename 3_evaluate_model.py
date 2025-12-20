import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_PATH = "ms-graph-v1" # Matches training output

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

def evaluate():
    print(f"Loading Adapter from {ADAPTER_PATH}...")
    try:
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, load_in_4bit=True, device_map="auto", torch_dtype=torch.bfloat16
        )
        model = PeftModel.from_pretrained(base, ADAPTER_PATH)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    except Exception as e:
        print(f"Error loading model. Did you train first? {e}")
        return

    messages = [
        {"role": "system", "content": "You are an AI Agent for Microsoft Graph. Given the user request and the available tool definition, generate the correct JSON tool call."},
        {"role": "user", "content": f"User Request: {TEST_QUERY}\nAvailable Tool: {json.dumps(TEST_TOOL)}"}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    print("Generating...")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.1)
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"\nAgent Output: {response}\n")

if __name__ == "__main__":
    evaluate()