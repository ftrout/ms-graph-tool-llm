import requests
import yaml
import json
import os
import re
import random

# --- CONFIGURATION ---
OUTPUT_DIR = "./data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "graph_tool_dataset.jsonl")
OPENAPI_URL = "https://raw.githubusercontent.com/microsoftgraph/msgraph-metadata/master/openapi/v1.0/openapi.yaml"

PROMPT_TEMPLATES = [
    "I need to {action}.",
    "Execute a function to {action}.",
    "Can you {action}?",
]

def download_spec():
    print("Downloading OpenAPI Spec...")
    response = requests.get(OPENAPI_URL, stream=True)
    if response.status_code == 200:
        return yaml.safe_load(response.content)
    else:
        raise Exception(f"Download failed: {response.status_code}")

def clean_text(text):
    if not text: return ""
    text = re.sub('<[^<]+?>', '', text) # Strip HTML
    return re.sub(r'\s+', ' ', text).strip()

def get_properties_recursive(schema):
    """Recursively extracts properties from schema definitions."""
    props = {}
    if 'properties' in schema:
        for k, v in schema['properties'].items():
            props[k] = {
                "type": v.get('type', 'string'),
                "description": clean_text(v.get('description', ''))
            }
    return props

def format_tool(path, method, op):
    clean_path = path.strip("/").replace("/", "_").replace("-", "_").replace("{", "").replace("}", "")
    name = op.get('operationId', f"{method}_{clean_path}")
    desc = clean_text(op.get('summary', op.get('description', '')))
    
    params = {"type": "object", "properties": {}, "required": []}
    
    # Path/Query Params
    for p in op.get('parameters', []):
        if p.get('in') in ['path', 'query']:
            pname = p['name']
            params['properties'][pname] = {
                "type": p.get('schema', {}).get('type', 'string'),
                "description": clean_text(p.get('description', ''))
            }
            if p.get('required'): params['required'].append(pname)
            
    # Body Params (Crucial Fix)
    if 'requestBody' in op:
        content = op['requestBody'].get('content', {})
        if 'application/json' in content:
            schema = content['application/json'].get('schema', {})
            body_props = get_properties_recursive(schema)
            if body_props:
                # Merge body props into top-level for agent simplicity
                for k, v in body_props.items():
                    params['properties'][k] = v
            else:
                params['properties']['request_body'] = {"type": "object", "description": "JSON payload"}

    return {
        "type": "function",
        "function": {"name": name, "description": desc, "parameters": params}
    }

def generate_dummy_args(params):
    args = {}
    for k, v in params['properties'].items():
        if k == "$filter": args[k] = "startswith(displayName, 'A')"
        elif k == "$select": args[k] = "id,displayName"
        elif v['type'] == 'integer': args[k] = 10
        elif v['type'] == 'boolean': args[k] = True
        else: args[k] = f"example_{k}"
    return args

def process_spec(spec):
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    dataset = []
    
    print("Processing endpoints...")
    for path, methods in spec.get('paths', {}).items():
        for method, op in methods.items():
            if method not in ['get', 'post', 'patch', 'put', 'delete']: continue
            try:
                tool = format_tool(path, method, op)
                summary = clean_text(op.get('summary', ''))
                if not summary: continue

                # Generate synthetic data
                args = generate_dummy_args(tool['function']['parameters'])
                
                for tmpl in PROMPT_TEMPLATES:
                    dataset.append({
                        "instruction": tmpl.format(action=summary.lower()),
                        "input": json.dumps(tool),
                        "output": json.dumps({"name": tool['function']['name'], "arguments": args})
                    })
            except: continue
            
    print(f"Generated {len(dataset)} samples.")
    with open(OUTPUT_FILE, 'w') as f:
        for d in dataset: f.write(json.dumps(d) + "\n")

if __name__ == "__main__":
    spec = download_spec()
    process_spec(spec)