import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_PATH = "graph-sentinel-v1"

TEST_TOOL = {
    "type": "function",
    "function": {
        "name": "users_ListUsers",
        "description": "Retrieve a list of user objects.",
        "parameters": {
            "type": "object",
            "properties": {
                "$filter": {"type": "string", "description": "Filter items by property values"},
                "$select": {"type": "string", "description": "Select properties to be returned"}
            }
        }
    }
}
TEST_QUERY = "Find the user with email 'admin@contoso.com' and select their id and display name."

def evaluate():
    print("Loading Qwen 2.5 + Adapter...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, load_in_4bit=True, device_map="auto", torch_dtype=torch.bfloat16
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    messages = [
        {"role": "system", "content": "You are an AI Agent for Microsoft Graph. Given the user request and the available tool definition, generate the correct JSON tool call."},
        {"role": "user", "content": f"User Request: {TEST_QUERY}\nAvailable Tool: {json.dumps(TEST_TOOL)}"}
    ]
    
    # Use apply_chat_template to automatically handle <|im_start|> tokens
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    print("Generating...")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.1)
    
    # Strip the prompt to get only the new tokens
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    print(f"\nUser: {TEST_QUERY}")
    print(f"Agent Output: {response}\n")
    
    try:
        data = json.loads(response)
        print("✅ Valid JSON")
        if "$filter" in data.get("arguments", {}): print("✅ Filter Logic Correct")
        else: print("❌ Filter Logic Missing")
    except:
        print("❌ Invalid JSON Output")

if __name__ == "__main__":
    evaluate()