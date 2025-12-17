import requests
import yaml
import json
import os
import re

# --- CONFIGURATION ---
OUTPUT_DIR = "./data_graph"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "graph_tool_dataset.jsonl")
OPENAPI_URL = "https://raw.githubusercontent.com/microsoftgraph/msgraph-metadata/master/openapi/v1.0/openapi.yaml"

TARGET_NAMESPACES = [
    "/me/messages", "/me/events", "/me/drive", "/teams", "/users", "/sites"
]

def download_spec():
    print("Downloading Microsoft Graph OpenAPI Spec (~30MB)...")
    response = requests.get(OPENAPI_URL, stream=True)
    if response.status_code == 200:
        return yaml.safe_load(response.content)
    else:
        raise Exception(f"Failed to download spec: {response.status_code}")

def clean_description(desc):
    if not desc: return "No description provided."
    text = re.sub('<[^<]+?>', '', desc)
    return text.replace('\n', ' ').strip()

def format_tool_definition(path, method, operation):
    tool_name = operation.get('operationId', f"{method}_{path.replace('/', '_')}")
    description = clean_description(operation.get('summary', operation.get('description', '')))
    
    parameters = {"type": "object", "properties": {}, "required": []}
    
    if 'parameters' in operation:
        for param in operation['parameters']:
            if param.get('in') in ['path', 'query']:
                param_name = param['name']
                param_desc = clean_description(param.get('description', ''))
                param_type = param.get('schema', {}).get('type', 'string')
                parameters['properties'][param_name] = {"type": param_type, "description": param_desc}
                if param.get('required', False):
                    parameters['required'].append(param_name)

    if 'requestBody' in operation:
        content = operation['requestBody'].get('content', {})
        if 'application/json' in content:
            parameters['properties']['request_body'] = {
                "type": "object",
                "description": "JSON body required for this action (refer to schema)"
            }

    return {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": description,
            "parameters": parameters
        }
    }

def process_spec(spec):
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    dataset = []
    paths = spec.get('paths', {})
    print(f"Processing {len(paths)} endpoints...")
    
    for path, methods in paths.items():
        if not any(target in path for target in TARGET_NAMESPACES): continue
            
        for method, operation in methods.items():
            if method not in ['get', 'post', 'patch', 'delete']: continue
            
            tool_def = format_tool_definition(path, method, operation)
            summary = clean_description(operation.get('summary', ''))
            if not summary: continue
            
            entry = {
                "instruction": f"I need to {summary.lower()}.",
                "input": json.dumps(tool_def),
                "output": json.dumps({
                    "name": tool_def['function']['name'],
                    "arguments": {"example_arg": "value"}
                })
            }
            dataset.append(entry)
            
    print(f"Generated {len(dataset)} tool-calling samples.")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for entry in dataset:
            f.write(json.dumps(entry) + "\n")

if __name__ == "__main__":
    spec = download_spec()
    process_spec(spec)