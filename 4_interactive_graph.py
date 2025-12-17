import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_PATH = "graph-sentinel-v1"

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

def load_model():
    print("Initializing Agent...")
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, load_in_4bit=True, device_map="auto", torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    return model, tokenizer

def run_inference(model, tokenizer, user_input, tool_def):
    messages = [
        {"role": "system", "content": "You are an AI Agent for Microsoft Graph. Generate JSON tool calls."},
        {"role": "user", "content": f"User Request: {user_input}\nAvailable Tool: {json.dumps(tool_def)}"}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.1)
        
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

if __name__ == "__main__":
    model, tokenizer = load_model()
    print("\n--- GRAPH SENTINEL AGENT READY (Type 'quit' to exit) ---")
    while True:
        user_input = input("\nRequest > ")
        if user_input.lower() in ["quit", "exit"]: break
        
        # Simple routing logic for the demo
        tool_key = "mail" if "mail" in user_input or "send" in user_input else "calendar"
        response = run_inference(model, tokenizer, user_input, TOOLS[tool_key])
        print(f"Agent > {response}")