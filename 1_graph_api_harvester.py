import requests
import yaml
import json
import os
import re
import random
import logging
from typing import Dict, List, Any, Optional

# --- LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
OUTPUT_DIR = "./data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "graph_tool_dataset.jsonl")
OPENAPI_URL = "https://raw.githubusercontent.com/microsoftgraph/msgraph-metadata/master/openapi/v1.0/openapi.yaml"

PROMPT_TEMPLATES = [
    "I need to {action}.",
    "Execute a function to {action}.",
    "Can you {action}?",
]

def download_spec() -> Dict[str, Any]:
    """
    Downloads the Microsoft Graph OpenAPI specification from GitHub.

    Returns:
        Dict[str, Any]: Parsed YAML specification as a dictionary

    Raises:
        Exception: If download fails or returns non-200 status code
    """
    logger.info("Downloading OpenAPI Spec from %s...", OPENAPI_URL)
    response = requests.get(OPENAPI_URL, stream=True)
    if response.status_code == 200:
        logger.info("Successfully downloaded OpenAPI specification")
        return yaml.safe_load(response.content)
    else:
        raise Exception(f"Download failed with status code: {response.status_code}")

def clean_text(text: Optional[str]) -> str:
    """
    Cleans text by removing HTML tags and normalizing whitespace.

    Args:
        text: Input text string (can be None)

    Returns:
        str: Cleaned text with HTML removed and whitespace normalized
    """
    if not text:
        return ""
    text = re.sub('<[^<]+?>', '', text)  # Strip HTML
    return re.sub(r'\s+', ' ', text).strip()

def get_properties_recursive(schema: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """
    Recursively extracts properties from OpenAPI schema definitions.

    Args:
        schema: OpenAPI schema dictionary containing property definitions

    Returns:
        Dict[str, Dict[str, str]]: Dictionary mapping property names to their
            type and description metadata
    """
    props = {}
    if 'properties' in schema:
        for k, v in schema['properties'].items():
            props[k] = {
                "type": v.get('type', 'string'),
                "description": clean_text(v.get('description', ''))
            }
    return props

def format_tool(path: str, method: str, op: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts an OpenAPI endpoint definition into a tool calling format.

    Args:
        path: API endpoint path (e.g., "/users/{id}")
        method: HTTP method (e.g., "get", "post")
        op: OpenAPI operation object containing endpoint metadata

    Returns:
        Dict[str, Any]: Formatted tool definition with function name, description,
            and parameter schema
    """
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

def generate_dummy_args(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates synthetic example arguments for API tool parameters.

    Args:
        params: Parameter schema dictionary with 'properties' key

    Returns:
        Dict[str, Any]: Dictionary of example argument values for each parameter
    """
    args = {}
    for k, v in params['properties'].items():
        if k == "$filter":
            args[k] = "startswith(displayName, 'A')"
        elif k == "$select":
            args[k] = "id,displayName"
        elif v['type'] == 'integer':
            args[k] = 10
        elif v['type'] == 'boolean':
            args[k] = True
        else:
            args[k] = f"example_{k}"
    return args

def process_spec(spec: Dict[str, Any]) -> None:
    """
    Processes the OpenAPI specification and generates training dataset.

    Args:
        spec: Parsed OpenAPI specification dictionary

    Side Effects:
        - Creates OUTPUT_DIR if it doesn't exist
        - Writes training data to OUTPUT_FILE in JSONL format
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        logger.info("Created output directory: %s", OUTPUT_DIR)

    dataset = []

    logger.info("Processing API endpoints...")
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
            except Exception as e:
                logger.warning("Failed to process %s %s: %s", method.upper(), path, e)
                continue

    logger.info("Generated %d training samples", len(dataset))
    logger.info("Writing dataset to %s...", OUTPUT_FILE)

    with open(OUTPUT_FILE, 'w') as f:
        for d in dataset:
            f.write(json.dumps(d) + "\n")

    logger.info("Dataset successfully written to %s", OUTPUT_FILE)

if __name__ == "__main__":
    spec = download_spec()
    process_spec(spec)