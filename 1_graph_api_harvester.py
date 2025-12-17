import requests
import yaml
import json
import os
import re

# --- CONFIGURATION ---
OUTPUT_DIR = "./data_graph"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "graph_tool_dataset.jsonl")
OPENAPI_URL = "https://raw.githubusercontent.com/microsoftgraph/msgraph-metadata/master/openapi/v1.0/openapi.yaml"

def download_spec():
    print("Downloading Microsoft Graph OpenAPI Spec (this is large, ~30MB+)...")
    response = requests.get(OPENAPI_URL, stream=True)
    if response.status_code == 200:
        # Load YAML safely. This might take a moment due to file size.
        return yaml.safe_load(response.content)
    else:
        raise Exception(f"Failed to download spec: {response.status_code}")

def clean_description(desc):
    """Removes HTML tags, newlines, and excessive whitespace."""
    if not desc: return "No description provided."
    # Strip HTML tags
    text = re.sub('<[^<]+?>', '', desc)
    # Collapse whitespace and newlines
    return re.sub(r'\s+', ' ', text).strip()

def format_tool_definition(path, method, operation):
    """
    Converts a Graph API operation into a structured Tool Definition.
    """
    # Construct a unique tool name (e.g., "users_list", "drive_root_children_list")
    # We replace slashes with underscores for a clean function name
    clean_path = path.strip("/").replace("/", "_").replace("-", "_").replace("{", "").replace("}", "")
    tool_name = operation.get('operationId', f"{method}_{clean_path}")
    
    description = clean_description(operation.get('summary', operation.get('description', '')))
    
    parameters = {
        "type": "object",
        "properties": {},
        "required": []
    }
    
    # 1. Path & Query Parameters
    if 'parameters' in operation:
        for param in operation['parameters']:
            if param.get('in') in ['path', 'query']:
                param_name = param['name']
                param_desc = clean_description(param.get('description', ''))
                
                # Simplify types for the LLM
                schema = param.get('schema', {})
                param_type = schema.get('type', 'string')
                
                parameters['properties'][param_name] = {
                    "type": param_type,
                    "description": param_desc
                }
                
                if param.get('required', False):
                    parameters['required'].append(param_name)

    # 2. Body Parameters (Simplified)
    # In a full production system, you would parse the $ref components here.
    # For now, we capture that a body is required.
    if 'requestBody' in operation:
        content = operation['requestBody'].get('content', {})
        if 'application/json' in content:
            parameters['properties']['request_body'] = {
                "type": "object",
                "description": "JSON body payload required for this action. Refer to Microsoft Graph documentation for structure."
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
    total_endpoints = len(paths)
    
    print(f"Processing {total_endpoints} endpoints across ALL namespaces...")
    
    count = 0
    for path, methods in paths.items():
        # --- REMOVED FILTERING LOGIC ---
        # We now process every single path in the spec.
        
        for method, operation in methods.items():
            if method not in ['get', 'post', 'patch', 'delete', 'put']: continue
            
            try:
                # 1. Create Tool Definition
                tool_def = format_tool_definition(path, method, operation)
                
                # 2. Extract Description for Prompt
                summary = clean_description(operation.get('summary', ''))
                
                # Skip endpoints with empty descriptions (bad training data)
                if not summary or len(summary) < 5: 
                    continue
                
                # 3. Create Synthetic Prompt
                user_prompt = f"I need to {summary.lower()}."
                
                entry = {
                    "instruction": user_prompt,
                    "input": json.dumps(tool_def),
                    "output": json.dumps({
                        "name": tool_def['function']['name'],
                        "arguments": {
                            # We leave arguments empty in the template. 
                            # The model learns structure, not specific values.
                            "example_arg": "value" 
                        }
                    })
                }
                dataset.append(entry)
                count += 1
                
                if count % 1000 == 0:
                    print(f"  Generated {count} samples...")
                    
            except Exception as e:
                # Silently skip malformed entries to keep the harvest moving
                continue
            
    print(f"Harvest Complete. Generated {len(dataset)} training samples.")
    print(f"Saving to {OUTPUT_FILE}...")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for entry in dataset:
            f.write(json.dumps(entry) + "\n")

if __name__ == "__main__":
    try:
        spec = download_spec()
        process_spec(spec)
    except KeyboardInterrupt:
        print("\nHarvest interrupted by user.")
    except Exception as e:
        print(f"\nCritical Error: {e}")