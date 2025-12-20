import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

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

def load_model():
    print("Initializing Agent...")
    try:
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, load_in_4bit=True, device_map="auto", torch_dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(base, ADAPTER_PATH)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        return model, tokenizer
    except Exception as e:
        print(f"Error: {e}")
        return None, None

def run_inference(model, tokenizer, user_input, tool_def):
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
        print("\n--- GRAPH AGENT READY (Type 'quit' to exit) ---")
        while True:
            user_input = input("\nRequest > ")
            if user_input.lower() in ["quit", "exit"]: break
            
            # Simple routing logic for the demo (In production, use RAG/Embeddings here)
            tool_key = "mail" if "mail" in user_input or "send" in user_input else "calendar"
            
            response = run_inference(model, tokenizer, user_input, TOOLS[tool_key])
            print(f"Agent > {response}")